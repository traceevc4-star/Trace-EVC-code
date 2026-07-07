#!/usr/bin/env python3
"""Train the EASE speaker encoder and dump per-utterance embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset

import paths

EMOTIONS = ["angry", "happy", "sad", "surprised", "neutral"]
EMO_ID = {e: i for i, e in enumerate(EMOTIONS)}
DEFAULT_XVEC = paths.EASE_XVECTORS
DEFAULT_EASE_DIR = paths.EASE_DIR
DEFAULT_STATS = paths.EASE_STATS
DEFAULT_CKPT = paths.EASE_CKPT


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.alpha, None


class SpeakerModel(nn.Module):
    def __init__(self, n_spk, n_emo, in_dim=192, dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, dim)
        self.fc = nn.Linear(dim, dim)
        self.fc_embed = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc_embed_1 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, n_spk)
        self.fc4 = nn.Linear(dim, dim)
        self.fc_embed_2 = nn.Linear(dim, dim)
        self.fc5 = nn.Linear(dim, n_emo)

    def forward(self, feat, alpha=1.0):
        feat = self.fc(self.fc_embed(self.fc1(feat)))
        rev = ReverseLayerF.apply(feat, alpha)
        spk = self.fc3(self.fc_embed_1(self.fc2(feat)))
        emo = self.fc5(self.fc_embed_2(self.fc4(rev)))
        return spk, emo, feat


def spk_of(name):
    return name.split("_")[0]


def emo_of(name):
    return name.split("_")[1]


class XVecDS(Dataset):
    def __init__(self, items, cache, spk2id):
        self.items, self.cache, self.spk2id = items, cache, spk2id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        n = self.items[i]
        return self.cache[n].float(), self.spk2id[spk_of(n)], EMO_ID[emo_of(n)]


def collate(b):
    x, s, e = zip(*b)
    return torch.stack(x), torch.tensor(s), torch.tensor(e)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xvec", type=Path, default=DEFAULT_XVEC)
    p.add_argument("--ease-dir", type=Path, default=DEFAULT_EASE_DIR)
    p.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-emo", type=float, default=10.0)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    cache = torch.load(args.xvec, map_location="cpu", weights_only=False)
    names = sorted(cache)
    spk2id = {s: i for i, s in enumerate(sorted({spk_of(n) for n in names}))}
    print(f"x-vectors={len(names)} speakers={len(spk2id)} emotions={len(EMOTIONS)}")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(names), generator=g).tolist()
    n_val = int(len(names) * args.val_frac)
    val_items = [names[i] for i in perm[:n_val]]
    tr_items = [names[i] for i in perm[n_val:]]

    model = SpeakerModel(len(spk2id), len(EMOTIONS)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tl = DataLoader(XVecDS(tr_items, cache, spk2id), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    vl = DataLoader(XVecDS(val_items, cache, spk2id), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    n_steps = args.epochs * len(tl)

    step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, s, e in tl:
            x, s, e = x.to(dev), s.to(dev), e.to(dev)
            p = step / max(n_steps, 1)
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            spk, emo, _ = model(x, alpha)
            loss = F.cross_entropy(spk, s) + args.lambda_emo * F.cross_entropy(emo, e)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
        model.eval()
        sc = ec = n = 0
        with torch.no_grad():
            for x, s, e in vl:
                x, s, e = x.to(dev), s.to(dev), e.to(dev)
                spk, emo, _ = model(x, 0.0)
                sc += (spk.argmax(-1) == s).sum().item()
                ec += (emo.argmax(-1) == e).sum().item()
                n += s.numel()
        if ep % 5 == 0 or ep == 1:
            print(f"ep{ep:02d} val spk_acc={sc/n*100:.1f}% emo_acc={ec/n*100:.1f}% "
                  f"(emo should stay LOW = invariant)", flush=True)

    model.eval()
    args.ease_dir.mkdir(parents=True, exist_ok=True)
    feats = []
    with torch.no_grad():
        for i in range(0, len(names), 1024):
            chunk = names[i:i + 1024]
            x = torch.stack([cache[n].float() for n in chunk]).to(dev)
            _, _, f = model(x, 0.0)
            f = f.cpu().numpy().astype(np.float32)
            for n, v in zip(chunk, f):
                np.save(args.ease_dir / f"{n}.npy", v)
                feats.append(v)
    feats = np.stack(feats)
    np.savez(args.stats, mean=feats.mean(0), std=feats.std(0).clip(min=1e-6))
    torch.save({"model": model.state_dict(), "spk2id": spk2id, "emotions": EMOTIONS,
                "in_dim": 192, "dim": 128}, args.ckpt)
    print(f"saved {len(feats)} EASE embeddings -> {args.ease_dir}")
    print(f"saved stats -> {args.stats}  ckpt -> {args.ckpt}")


if __name__ == "__main__":
    main()
