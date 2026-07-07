#!/usr/bin/env python3
"""Frame-level energy (linear RMS @50Hz) for the binarized MEAD items.

Run both variants: --norm (loudness-normalized, matches the binarizer mel) and
--raw (loudness preserved). Writes energy_rms[_raw]/<spk>/<item>.npy plus
energy[_raw]_spk_stats.npz (per-speaker log-RMS mean/std) and
energy[_raw]_utt_stats.npz (per-utt z-mean log-RMS). Silence floored to 0; audio
trimmed with trim_long_silences to stay aligned with the F0/mel/unit frames.
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
HOP = 320
WIN = 1024


def _extract_one(args):
    item, wav_fn, out_dir, norm = args
    spk = item.split("_")[0]
    out = os.path.join(out_dir, spk, item + ".npy")
    if os.path.exists(out):
        return item, len(np.load(out))
    try:
        import librosa
        from utils.audio.vad import trim_long_silences

        y, _, _ = trim_long_silences(wav_fn, SR, norm=norm)
        rms = librosa.feature.rms(y=y, frame_length=WIN, hop_length=HOP)[0].astype(np.float32)
        db = 20 * np.log10(rms + 1e-12)
        rms[db < db.max() - 60.0] = 0.0
    except Exception as e:
        print(f"| skip {item}: {e}", flush=True)
        return None
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, rms)
    return item, len(rms)


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
    default_norm = os.environ.get("NORM", "1") == "1"
    g = p.add_mutually_exclusive_group()
    g.add_argument("--norm", dest="norm", action="store_true", help="loudness-normalized (default)")
    g.add_argument("--raw", dest="norm", action="store_false", help="true loudness preserved")
    p.set_defaults(norm=default_norm)
    p.add_argument("--binary-dir", default=str(paths.MEAD_BINARY))
    p.add_argument("--out-dir", default=None, help="default: <processed>/energy_rms[_raw]")
    p.add_argument("--data-dir", default=str(paths.MEAD_PROCESSED.parent),
                   help="where the *_stats.npz aggregates are written")
    p.add_argument("--nproc", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    sfx = "" if args.norm else "_raw"
    out_dir = args.out_dir or str(paths.MEAD_PROCESSED / f"energy_rms{sfx}")
    spk_stats = os.path.join(args.data_dir, f"energy{sfx}_spk_stats.npz")
    utt_stats = os.path.join(args.data_dir, f"energy{sfx}_utt_stats.npz")
    os.makedirs(args.data_dir, exist_ok=True)
    print(f"mode = {'NORM' if args.norm else 'RAW'}  ->  {out_dir}", flush=True)

    items = list(_iter_items(args.binary_dir))
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"no items found under {args.binary_dir} — run build_mead_binary.py first")
    print(f"total items: {len(items)}", flush=True)

    t0 = time.time()
    done = []
    work = [(it, wav, out_dir, args.norm) for it, wav in items]
    with Pool(args.nproc) as pool:
        for k, r in enumerate(pool.imap_unordered(_extract_one, work, chunksize=16)):
            if r:
                done.append(r)
            if (k + 1) % 3000 == 0:
                print(f"progress {k+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"extracted {len(done)}/{len(items)} in {time.time()-t0:.0f}s", flush=True)

    by_spk = defaultdict(list)
    utt_names, utt_mean = [], []
    for item, _ in done:
        spk = item.split("_")[0]
        rms = np.load(os.path.join(out_dir, spk, item + ".npy"))
        v = rms > 0
        if v.sum() < 5:
            continue
        lr = np.log(rms[v])
        by_spk[spk].append(lr)
        utt_names.append(item)
        utt_mean.append(lr.mean())

    spk_mean, spk_std = {}, {}
    for spk, chunks in by_spk.items():
        allv = np.concatenate(chunks)
        spk_mean[spk], spk_std[spk] = float(allv.mean()), float(allv.std())
    order = sorted(spk_mean)
    np.savez(spk_stats, speakers=order,
             mean=[spk_mean[s] for s in order], std=[spk_std[s] for s in order])
    print(f"speaker stats ({len(spk_mean)} spks) -> {spk_stats}", flush=True)

    z = np.array([(m - spk_mean[n.split('_')[0]]) / (spk_std[n.split('_')[0]] + 1e-8)
                  for n, m in zip(utt_names, utt_mean)], np.float32)
    np.savez(utt_stats, names=utt_names, z_mean=z)
    print(f"utt stats ({len(utt_names)} utts) -> {utt_stats}", flush=True)
    print("ENERGY_PRECOMPUTE_DONE", flush=True)


if __name__ == "__main__":
    main()
