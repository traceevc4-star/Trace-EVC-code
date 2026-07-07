#!/usr/bin/env python3
"""Train the frozen MEAD intensity head (RankNet over VAD)."""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import paths
from models.emo_compass import IntensityHead
from tasks.emo_compass.data import (
    EMOTIONS_MEAD, attach_vad, load_mead_emo2vec_rows, load_mead_split, speaker_split_rows, vad_stats,
)


def make_pairs(rows):
    """SAME (speaker, emotion, content), different level -> (higher, lower)."""
    groups = defaultdict(dict)
    for r in rows:
        groups[(r["speaker"], r["emotion"], r["content"])][r["level"]] = r
    pairs = []
    for g in groups.values():
        lvls = sorted(g)
        for a in range(len(lvls)):
            for b in range(a + 1, len(lvls)):
                pairs.append((g[lvls[b]], g[lvls[a]]))
    return pairs


class PairDS(Dataset):
    def __init__(self, pairs, mean, std):
        self.pairs, self.mean, self.std = pairs, mean, std

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        hi, lo = self.pairs[i]
        z = lambda e: (e - self.mean) / self.std
        return z(hi["vad"]), z(lo["vad"]), hi["emo_id"]


def collate(batch):
    h, l, e = zip(*batch)
    return torch.stack(h), torch.stack(l), torch.tensor(e, dtype=torch.long)


@torch.no_grad()
def rank_acc(head, pairs, mean, std, device):
    if not pairs:
        return 0.0
    dl = DataLoader(PairDS(pairs, mean, std), batch_size=512, collate_fn=collate)
    ok = tot = 0
    for h, l, e in dl:
        h, l, e = h.to(device), l.to(device), e.to(device)
        ok += (head(h, e) > head(l, e)).sum().item()
        tot += e.numel()
    return ok / tot


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, default=str(paths.MEAD_EMO2VEC))
    p.add_argument("--vad", type=str, default=str(paths.MEAD_VAD))
    p.add_argument("--split-json", type=str, default=str(paths.MEAD_SPLIT))
    p.add_argument("--out", type=str, default=str(paths.MEAD_INTENSITY_HEAD))
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--emo-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    rows = attach_vad(load_mead_emo2vec_rows(args.cache), args.vad)
    _, val_s, test_s = load_mead_split(args.split_json)
    tr, va, te = speaker_split_rows(rows, val_s, test_s)
    mean, std = vad_stats(tr)
    tr_p, va_p, te_p = make_pairs(tr), make_pairs(va), make_pairs(te)
    print(f"rank pairs: train={len(tr_p)} val={len(va_p)} test={len(te_p)}  (input=VAD)")

    head = IntensityHead(3, len(EMOTIONS_MEAD), args.emo_dim, args.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(PairDS(tr_p, mean, std), batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    best, best_state = 0.0, None
    for ep in range(1, args.epochs + 1):
        head.train()
        for h, l, e in loader:
            h, l, e = h.to(device), l.to(device), e.to(device)
            loss = F.binary_cross_entropy_with_logits(head(h, e) - head(l, e), torch.ones(e.numel(), device=device))
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        vacc = rank_acc(head, va_p, mean, std, device)
        if vacc >= best:
            best, best_state = vacc, {k: v.cpu().clone() for k, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"ep{ep:02d} val_rank_acc={vacc*100:.1f}% (best {best*100:.1f}%)", flush=True)

    head.load_state_dict(best_state)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "input_dim": 3, "input": "vad", "n_emo": len(EMOTIONS_MEAD),
                "emo_dim": args.emo_dim, "hidden": args.hidden, "emotions": EMOTIONS_MEAD,
                "mean": mean, "std": std, "val_rank_acc": best}, out)
    print(f"done: best val_rank={best*100:.1f}% -> {out}")


if __name__ == "__main__":
    main()
