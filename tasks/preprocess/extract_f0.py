#!/usr/bin/env python3
"""YAAPT F0 contours for the binarized MEAD items (run after build_mead_binary.py).

Writes f0_yaapt/<spk>/<item>.npy (raw Hz @50Hz, unvoiced=0) plus f0_spk_stats.npz
(per-speaker voiced log-F0 mean/std) and f0_utt_stats.npz (per-utt z-space stats;
col 0 = z-mean F0). Audio is trimmed with the same trim_long_silences used by the
binarizer so contours stay frame-aligned to the mel/units.
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

import paths

SR = 16000


def yaapt_f0(wav_path: str) -> np.ndarray:
    """ZEST/speech-resynthesis settings: 25ms frames, 20ms hop -> 50 Hz."""
    import amfm_decompy.basic_tools as basic
    import amfm_decompy.pYAAPT as pYAAPT
    from utils.audio.vad import trim_long_silences

    y, _, _ = trim_long_silences(wav_path, SR)
    to_pad = int(25.0 / 1000 * SR) // 2
    y = np.pad(y.astype(np.float64), (to_pad, to_pad), "constant")
    signal = basic.SignalObj(y, SR)
    pitch = pYAAPT.yaapt(signal, frame_length=25.0, frame_space=20.0,
                         nccf_thresh1=0.25, tda_frame_length=25.0)
    return pitch.samp_values.astype(np.float32)


def _extract_one(args):
    item, wav_fn, out_dir = args
    spk = item.split("_")[0]
    out = os.path.join(out_dir, spk, item + ".npy")
    if os.path.exists(out):
        return item, len(np.load(out))
    try:
        f0 = yaapt_f0(wav_fn)
    except Exception as e:
        print(f"| skip {item}: {e}", flush=True)
        return None
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, f0)
    return item, len(f0)


def _iter_items(binary_dir: str):
    from utils.commons.indexed_datasets import IndexedDataset

    for pre in ("train", "valid", "test"):
        prefix = os.path.join(binary_dir, pre)
        if not os.path.exists(prefix + ".idx"):
            continue
        ds = IndexedDataset(prefix)
        for i in range(len(ds)):
            it = ds[i]
            yield it["item_name"], it["wav_fn"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary-dir", default=str(paths.MEAD_BINARY))
    p.add_argument("--out-dir", default=str(paths.MEAD_PROCESSED / "f0_yaapt"))
    p.add_argument("--data-dir", default=str(paths.MEAD_PROCESSED.parent),
                   help="where the *_stats.npz aggregates are written")
    p.add_argument("--nproc", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    items = list(_iter_items(args.binary_dir))
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"no items found under {args.binary_dir} — run build_mead_binary.py first")
    print(f"total items: {len(items)}", flush=True)

    spk_stats = os.path.join(args.data_dir, "f0_spk_stats.npz")
    utt_stats = os.path.join(args.data_dir, "f0_utt_stats.npz")
    os.makedirs(args.data_dir, exist_ok=True)

    t0 = time.time()
    done = []
    work = [(it, wav, args.out_dir) for it, wav in items]
    with Pool(args.nproc) as pool:
        for k, r in enumerate(pool.imap_unordered(_extract_one, work, chunksize=16)):
            if r:
                done.append(r)
            if (k + 1) % 1000 == 0:
                el = time.time() - t0
                print(f"progress {k+1}/{len(items)}  ok={len(done)}  {el:.0f}s  "
                      f"eta={(el / (k + 1)) * (len(items) - k - 1):.0f}s", flush=True)
    print(f"extracted {len(done)}/{len(items)} in {time.time()-t0:.0f}s", flush=True)

    by_spk = defaultdict(list)
    utt_names, utt_raw = [], []
    for item, _ in done:
        spk = item.split("_")[0]
        f0 = np.load(os.path.join(args.out_dir, spk, item + ".npy"))
        v = f0[f0 > 0]
        if len(v) < 5:
            continue
        lv = np.log(v)
        by_spk[spk].append(lv)
        p10, p90 = np.percentile(lv, [10, 90])
        utt_names.append(item)
        utt_raw.append([lv.mean(), lv.std(), p10, p90, p90 - p10, len(v) / len(f0)])

    spk_mean, spk_std = {}, {}
    for spk, chunks in by_spk.items():
        allv = np.concatenate(chunks)
        spk_mean[spk], spk_std[spk] = float(allv.mean()), float(allv.std())
    order = sorted(spk_mean)
    np.savez(spk_stats, speakers=order,
             mean=[spk_mean[s] for s in order], std=[spk_std[s] for s in order])
    print(f"speaker stats ({len(spk_mean)} spks) -> {spk_stats}", flush=True)

    utt_raw = np.array(utt_raw, np.float32)
    Z = utt_raw.copy()
    for i, name in enumerate(utt_names):
        s = name.split("_")[0]
        m, sd = spk_mean[s], spk_std[s] + 1e-8
        Z[i, 0] = (utt_raw[i, 0] - m) / sd
        Z[i, 1] = utt_raw[i, 1] / sd
        Z[i, 2] = (utt_raw[i, 2] - m) / sd
        Z[i, 3] = (utt_raw[i, 3] - m) / sd
        Z[i, 4] = utt_raw[i, 4] / sd
    np.savez(utt_stats, names=utt_names, stats=Z, raw=utt_raw,
             cols=["mean", "std", "p10", "p90", "range", "voiced_frac"])
    print(f"utt stats ({len(utt_names)} utts, 6-dim z-space) -> {utt_stats}", flush=True)
    print("F0_PRECOMPUTE_DONE", flush=True)


if __name__ == "__main__":
    main()
