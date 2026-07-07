#!/usr/bin/env python3
"""Extract ECAPA speaker x-vectors from MEAD audio."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import torchaudio

import paths

ROOT = paths.MEAD_RAW_AUDIO
EMOTIONS = ["angry", "happy", "sad", "surprised", "neutral"]
OUT = paths.EASE_XVECTORS

_KEY = re.compile(r"/(M\d+|W\d+)/.*?/(angry|happy|sad|surprised|neutral)/level[_ ]?([123])/(\d+)\.wav")


def item_name(path: str):
    m = _KEY.search(path)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}_L{m.group(3)}_{m.group(4)}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    from speechbrain.pretrained import EncoderClassifier
    dev = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": dev})

    wavs = []
    for emo in EMOTIONS:
        wavs += sorted(args.root.rglob(f"*/{emo}/level_*/*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    print(f"{len(wavs)} MEAD wavs", flush=True)

    cache, err = {}, 0
    for i, w in enumerate(wavs):
        name = item_name(str(w))
        if name is None:
            continue
        try:
            sig, sr = torchaudio.load(str(w))
            if sr != 16000:
                sig = torchaudio.functional.resample(sig, sr, 16000)
            emb = clf.encode_batch(sig.to(dev))[0, 0, :].cpu().float()
            cache[name] = emb
        except Exception as e:
            err += 1
            if err < 10:
                print("ERR", w, e, flush=True)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(wavs)} (err {err})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)
    print(f"saved {len(cache)} x-vectors -> {args.out}")


if __name__ == "__main__":
    main()
