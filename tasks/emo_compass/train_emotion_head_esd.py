#!/usr/bin/env python3
"""Train the frozen ESD emotion classifier head."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

import paths
from models.emo_compass import EmotionClassifierHead
from tasks.emo_compass.data import EMOTIONS_ESD, load_esd_index_split


def parse_speaker(wav: str) -> str:
    for part in Path(wav).parts:
        if len(part) == 4 and part.isdigit():
            return part
    raise ValueError(f"Cannot parse ESD speaker from path: {wav}")


def parse_esd_index(wav: str) -> int:
    stem = Path(wav).stem
    try:
        n = int(stem.split("_")[1])
    except Exception as exc:
        raise ValueError(f"Cannot parse ESD utterance index from path: {wav}") from exc
    return ((n - 1) % 350) + 1


def collect_labels(pairs_jsonl: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with Path(pairs_jsonl).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            labels[str(rec["source_wav"])] = rec["source_emotion"]
            labels[str(rec["target_wav"])] = rec["target_emotion"]
    return labels


def load_dataset(pairs_jsonl: Path, cache_path: Path):
    labels = collect_labels(pairs_jsonl)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    label_to_id = {e: i for i, e in enumerate(EMOTIONS_ESD)}
    rows, missing = [], 0
    for wav, emotion in labels.items():
        if wav not in cache:
            missing += 1
            continue
        rows.append({"wav": wav, "speaker": parse_speaker(wav), "index": parse_esd_index(wav),
                     "emotion": emotion, "label": label_to_id[emotion], "embedding": cache[wav].float()})
    if missing:
        print(f"missing embeddings: {missing}")
    if not rows:
        raise RuntimeError("No rows loaded.")
    return rows


def split_by_index(rows, index_split):
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        out[index_split[r["index"]]].append(r)
    return out["train"], out["val"], out["test"]


def tensorize(rows):
    if not rows:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    x = torch.stack([r["embedding"] for r in rows]).float()
    y = torch.tensor([r["label"] for r in rows], dtype=torch.long)
    return x, y


def batches(n, batch_size, shuffle, seed):
    idx = list(range(n))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start: start + batch_size]


@torch.no_grad()
def evaluate(model, x, y, batch_size, device):
    if len(y) == 0:
        return {"loss": float("nan"), "acc": float("nan"), "per_class": {}}
    model.eval()
    preds, losses = [], []
    for idx in batches(len(y), batch_size, False, 0):
        bi = torch.tensor(idx, dtype=torch.long)
        logits = model(x[bi].to(device))
        losses.append(F.cross_entropy(logits, y[bi].to(device), reduction="sum").item())
        preds.append(logits.argmax(-1).cpu())
    pred = torch.cat(preds)
    conf = torch.zeros(len(EMOTIONS_ESD), len(EMOTIONS_ESD), dtype=torch.long)
    for gold, got in zip(y.tolist(), pred.tolist()):
        conf[gold, got] += 1
    per_class = {EMOTIONS_ESD[i]: conf[i, i].item() / max(conf[i].sum().item(), 1) for i in range(len(EMOTIONS_ESD))}
    return {"loss": sum(losses) / len(y), "acc": (pred == y).float().mean().item(), "per_class": per_class}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-jsonl", type=str, default=str(paths.ESD_PAIRS))
    p.add_argument("--cache", type=str, default=str(paths.ESD_EMO2VEC))
    p.add_argument("--index-split-jsonl", type=str, default=str(paths.ESD_INDEX_SPLIT))
    p.add_argument("--out", type=str, default=str(paths.ESD_EMOTION_HEAD))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256, help="0 for a linear head.")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    rows = load_dataset(args.pairs_jsonl, args.cache)
    train_rows, val_rows, test_rows = split_by_index(rows, load_esd_index_split(args.index_split_jsonl))
    for nm, rr in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        print(f"{nm}: n={len(rr)} emotions={dict(sorted(Counter(r['emotion'] for r in rr).items()))}")

    xtr, ytr = tensorize(train_rows)
    xva, yva = tensorize(val_rows)
    xte, yte = tensorize(test_rows)
    mean = xtr.mean(0, keepdim=True)
    std = xtr.std(0, keepdim=True).clamp_min(1e-6)
    xtr = (xtr - mean) / std
    xva = (xva - mean) / std if len(yva) else xva
    xte = (xte - mean) / std if len(yte) else xte

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = EmotionClassifierHead(xtr.shape[1], args.hidden, args.dropout, len(EMOTIONS_ESD)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    best_acc, best_epoch, bad = -1.0, -1, 0
    for epoch in range(args.epochs):
        model.train()
        for idx in batches(len(ytr), args.batch_size, True, args.seed + epoch):
            bi = torch.tensor(idx, dtype=torch.long)
            logits = model(xtr[bi].to(device))
            loss = F.cross_entropy(logits, ytr[bi].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        va = evaluate(model, xva, yva, args.batch_size, device)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"ep{epoch:03d} val_acc={va['acc']*100:.2f}%")
        if va["acc"] > best_acc:
            best_acc, best_epoch, bad = va["acc"], epoch, 0
            torch.save({"model": model.state_dict(), "input_dim": xtr.shape[1], "hidden": args.hidden,
                        "dropout": args.dropout, "emotions": EMOTIONS_ESD, "mean": mean, "std": std,
                        "best_val_acc": best_acc, "best_epoch": best_epoch}, out)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at ep{epoch}, best ep{best_epoch} val_acc={best_acc*100:.2f}%")
                break

    print(f"done: best val_acc={best_acc*100:.2f}% (ep{best_epoch}) -> {out}")


if __name__ == "__main__":
    main()
