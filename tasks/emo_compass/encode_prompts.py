#!/usr/bin/env python3
"""Encode instruction prompts with the frozen E5 text encoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import paths


def load_prompt_records(path: Path, limit: int | None):
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            prompts = rec.get("prompt_variants") or [rec.get("prompt", "")]
            prompts = [p.strip() for p in prompts if str(p).strip()]
            if not prompts:
                continue
            records.append((rec["pair_id"], prompts))
            if limit is not None and len(records) >= limit:
                break
    return records


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts-jsonl", type=str, default=str(paths.ESD_PROMPTS))
    p.add_argument("--model", type=str, default=str(paths.E5_MODEL))
    p.add_argument("--out", type=str, default=str(paths.ESD_PROMPT_EMB))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fp16", action="store_true", help="Store embeddings as float16 to save space.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    prompts_jsonl = Path(args.prompts_jsonl)
    if not prompts_jsonl.exists():
        raise FileNotFoundError(prompts_jsonl)

    from transformers import AutoModel, AutoTokenizer

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    records = load_prompt_records(prompts_jsonl, args.limit)
    print(f"records: {len(records)}")

    flat_texts, spans = [], {}
    for pair_id, prompts in records:
        start = len(flat_texts)
        flat_texts.extend([f"query: {p}" for p in prompts])
        spans[pair_id] = (start, len(flat_texts), prompts)
    print(f"prompt strings: {len(flat_texts)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    encoded = []
    with torch.no_grad():
        for start in range(0, len(flat_texts), args.batch_size):
            texts = flat_texts[start: start + args.batch_size]
            batch = tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            out = model(**batch)
            mask = batch["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            encoded.append(F.normalize(emb, dim=-1).cpu())
            if (start // args.batch_size + 1) % 50 == 0:
                print(f"encoded {min(start + args.batch_size, len(flat_texts))}/{len(flat_texts)}", flush=True)
    all_emb = torch.cat(encoded, dim=0)
    if args.fp16:
        all_emb = all_emb.half()

    pair_emb = {pid: all_emb[start:end].clone() for pid, (start, end, _) in spans.items()}
    pair_prompts = {pid: prompts for pid, (_, _, prompts) in spans.items()}

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pair_emb": pair_emb, "pair_prompts": pair_prompts, "prompt_model": str(args.model),
                "source_jsonl": str(prompts_jsonl), "dim": all_emb.shape[1], "dtype": str(all_emb.dtype)}, out)
    print(f"saved {len(pair_emb)} pair embeddings -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
