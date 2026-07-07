#!/usr/bin/env python3
"""Extract HuBERT km200 content units from MEAD audio."""
from __future__ import annotations

import argparse
import os
import re
import sys
from itertools import groupby
from pathlib import Path

import joblib
import torch
import torch.nn.functional as F

import paths
from utils.audio.vad import trim_long_silences


class LengthRegulator(torch.nn.Module):
    """Inlined from durflex/utils.py (avoids importing the heavy durflex package).
    durations -> mel2unit, e.g. dur=[2,2,3] -> [1,1,2,2,3,3,3]."""

    def forward(self, dur, alpha=1.0):
        dur = torch.round(dur.float() * alpha).long()
        token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
        dur_cumsum = torch.cumsum(dur, 1)
        dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode="constant", value=0)
        pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
        return (token_idx * token_mask.long()).sum(1)

ROOT = paths.MEAD_RAW_AUDIO
EMOTIONS = ["angry", "happy", "sad", "surprised", "neutral"]
OUT = paths.MEAD_PROCESSED
KM200 = paths.HUBERT_KM200
HUBERT = "facebook/hubert-base-ls960"
SR = 16000

_KEY = re.compile(r"/(M\d+|W\d+)/.*?/(angry|happy|sad|surprised|neutral)/level[_ ]?([123])/(\d+)\.wav")


def item_name(path: str):
    m = _KEY.search(path)
    return f"{m.group(1)}_{m.group(2)}_L{m.group(3)}_{m.group(4)}" if m else None


def dedup_seq(seq):
    vals, counts = zip(*[(k.item(), sum(1 for _ in g)) for k, g in groupby(seq)])
    return list(vals), list(counts)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--km200", type=Path, default=KM200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-shards", type=int, default=int(os.environ.get("NUM_SHARDS", "1")))
    p.add_argument("--shard-id", type=int, default=int(os.environ.get("SHARD_ID", "0")))
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    from transformers import HubertModel
    dev = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    km = joblib.load(open(args.km200, "rb")); km.verbose = False
    model = HubertModel.from_pretrained(HUBERT, output_hidden_states=True).to(dev).eval()
    lr = LengthRegulator()
    unit_dir = args.out / "units"

    wavs = []
    for emo in EMOTIONS:
        wavs += sorted(args.root.rglob(f"*/{emo}/level_*/*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    total = len(wavs)
    wavs = wavs[args.shard_id::args.num_shards]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(wavs)}/{total} wavs -> {unit_dir}", flush=True)

    done = err = 0
    for i, w in enumerate(wavs):
        name = item_name(str(w))
        if name is None:
            continue
        spk = name.split("_")[0]
        out_fp = unit_dir / spk / f"{name}.pt"
        if out_fp.exists() and not args.overwrite:
            done += 1; continue
        try:
            wav, _, _ = trim_long_silences(str(w), SR)
            wav = torch.from_numpy(wav).unsqueeze(0).to(dev)
            wav = F.pad(wav, (40, 40), "reflect")
            hidden = model(wav).hidden_states[-1].cpu().squeeze(0).numpy()
            units = torch.IntTensor(km.predict(hidden))
            val, count = dedup_seq(units)
            mel2unit = lr(torch.IntTensor(count).unsqueeze(0)).squeeze(0)
            out_fp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"features": hidden, "units": val, "units_frame": units,
                        "count": count, "mel2unit": mel2unit}, out_fp)
            done += 1
        except Exception as e:
            err += 1
            if err < 10:
                print("ERR", w, e, flush=True)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(wavs)} done={done} err={err}", flush=True)

    print(f"finished: done={done} err={err} -> {unit_dir}")


if __name__ == "__main__":
    main()
