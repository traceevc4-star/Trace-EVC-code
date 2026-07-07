#!/usr/bin/env python3
"""Train the frozen MEAD emotion classifier head."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import paths
from models.emo_compass import EmotionClassifierHead
from tasks.emo_compass.data import (
    EMOTIONS_MEAD, emo_stats, load_mead_emo2vec_rows, load_mead_split, speaker_split_rows,
)


class Rows(Dataset):
    def __init__(self, rows, mean, std):
        self.rows, self.mean, self.std = rows, mean, std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return (r["emb"] - self.mean) / self.std, r["emo_id"]


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def run_epoch(loader, head, opt, device, train):
    head.train(train)
    tot = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = head(x)
            loss = F.cross_entropy(logits, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item() * y.numel()
        correct += (logits.argmax(-1) == y).sum().item()
        tot += y.numel()
    return loss_sum / tot, correct / tot


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, default=str(paths.MEAD_EMO2VEC))
    p.add_argument("--split-json", type=str, default=str(paths.MEAD_SPLIT))
    p.add_argument("--out", type=str, default=str(paths.MEAD_EMOTION_HEAD))
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    rows = load_mead_emo2vec_rows(args.cache)
    _, val_s, test_s = load_mead_split(args.split_json)
    tr, va, te = speaker_split_rows(rows, val_s, test_s)
    print(f"rows={len(rows)} train={len(tr)} val={len(va)} test={len(te)}")
    mean, std = emo_stats(tr)

    def dl(rws, shuffle):
        return DataLoader(Rows(rws, mean, std), batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)

    tl, vl, tel = dl(tr, True), dl(va, False), dl(te, False)
    head = EmotionClassifierHead(1024, args.hidden, args.dropout, len(EMOTIONS_MEAD)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best, best_state = 0.0, None
    for ep in range(1, args.epochs + 1):
        _, tra = run_epoch(tl, head, opt, device, True)
        _, vacc = run_epoch(vl, head, None, device, False)
        if vacc >= best:
            best, best_state = vacc, {k: v.cpu().clone() for k, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"ep{ep:02d} train_acc={tra*100:.1f}% val_acc={vacc*100:.1f}% (best {best*100:.1f}%)")

    head.load_state_dict(best_state)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "input_dim": 1024, "hidden": args.hidden, "dropout": args.dropout,
                "emotions": EMOTIONS_MEAD, "mean": mean, "std": std, "val_acc": best}, out)
    print(f"done: best val_acc={best*100:.1f}% -> {out}")


if __name__ == "__main__":
    main()
