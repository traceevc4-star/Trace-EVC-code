#!/usr/bin/env python3
"""Build the binarized MEAD dataset (mel + units) for training."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

import paths
from utils.audio import wav2spec
from utils.commons.indexed_datasets import IndexedDatasetBuilder

ROOT = paths.MEAD_RAW_AUDIO
EMOTIONS = ["angry", "happy", "sad", "surprised", "neutral"]
SPLIT_JSON = paths.MEAD_SPLIT
OUT = paths.MEAD_BINARY

MEL = dict(fft_size=1024, hop_size=256, win_length=1024, num_mels=80,
           fmin=0, fmax=8000, sample_rate=16000, loud_norm=False, trim_long_sil=True)

_KEY = re.compile(r"/(M\d+|W\d+)/.*?/(angry|happy|sad|surprised|neutral)/level[_ ]?([123])/(\d+)\.wav")


def parse(path: str):
    m = _KEY.search(path)
    if not m:
        return None
    spk, emo, lv, utt = m.group(1), m.group(2), m.group(3), m.group(4)
    return f"{spk}_{emo}_L{lv}_{utt}", spk, emo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--split-json", type=Path, default=SPLIT_JSON)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    d = json.loads(args.split_json.read_text())
    val_s, test_s = set(d["val"]), set(d["test"])

    wavs = []
    for emo in EMOTIONS:
        wavs += sorted(args.root.rglob(f"*/{emo}/level_*/*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]

    parsed = [(str(w),) + parse(str(w)) for w in wavs]
    parsed = [x for x in parsed if x[1] is not None]
    by_split = {"train": [], "valid": [], "test": []}
    for wav_fn, name, spk, emo in parsed:
        pref = "valid" if spk in val_s else ("test" if spk in test_s else "train")
        by_split[pref].append((wav_fn, name, spk, emo))
    print(f"speakers={len({spk for _, _, spk, _ in parsed})}  "
          f"train={len(by_split['train'])} valid={len(by_split['valid'])} test={len(by_split['test'])}")

    args.out.mkdir(parents=True, exist_ok=True)
    for prefix, items in by_split.items():
        builder = IndexedDatasetBuilder(str(args.out / prefix))
        lengths, total_sec, err = [], 0.0, 0
        for wav_fn, name, spk, emo in tqdm(items, desc=prefix):
            try:
                spec = wav2spec(wav_fn, **MEL)
            except Exception as e:
                err += 1
                if err < 5:
                    print("ERR", wav_fn, e)
                continue
            mel = spec["mel"].astype(np.float32)
            wav = spec["wav"]
            builder.add_item({
                "item_name": name, "wav_fn": wav_fn, "mel": mel,
                "spk_id": 0, "emo": emo,
                "sec": len(wav) / MEL["sample_rate"], "len": mel.shape[0],
            })
            lengths.append(mel.shape[0]); total_sec += len(wav) / MEL["sample_rate"]
        builder.finalize()
        np.save(str(args.out / f"{prefix}_lengths.npy"), lengths)
        print(f"| {prefix}: {len(lengths)} items, {total_sec/3600:.2f}h, err={err}")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
