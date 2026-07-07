#!/usr/bin/env python3
"""Precompute the target emotion-embedding bank for the synthesis stage."""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import types
from pathlib import Path

import torch

if "pathlib._local" not in sys.modules:
    _m = types.ModuleType("pathlib._local")
    for _n in ("Path", "PurePath", "PosixPath", "WindowsPath", "PurePosixPath", "PureWindowsPath"):
        if hasattr(pathlib, _n):
            setattr(_m, _n, getattr(pathlib, _n))
    sys.modules["pathlib._local"] = _m

import paths
from models.emo_compass import RectifiedFlow, TraceAffectDenoiser
from tasks.emo_compass.data import affect_vector, load_mead_records, load_mead_prosody


def items_from_pid(pid: str):
    p = pid.split("_")
    spk, emo, utt, s_lv, t_lv = p[1], p[2], p[3], p[4][1:], p[6][1:]
    return f"{spk}_{emo}_L{s_lv}_{utt}", f"{spk}_{emo}_L{t_lv}_{utt}", int(t_lv)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(paths.MEAD_FLOW_CKPT_DIR / "best_trace_mead.pt"))
    p.add_argument("--prompts-jsonl", type=str, default=str(paths.MEAD_PROMPTS))
    p.add_argument("--prompt-emb", type=str, default=str(paths.MEAD_PROMPT_EMB))
    p.add_argument("--emo-cache", type=str, default=str(paths.MEAD_EMO2VEC))
    p.add_argument("--out", type=str, default=str(paths.DIFFUSED_TARGET_EMB))
    p.add_argument("--sample-steps", type=int, default=1)
    p.add_argument("--num-prompts", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    den = TraceAffectDenoiser(**ckpt["model_config"]).to(device).eval()
    den.load_state_dict(ckpt["denoiser"])
    flow = RectifiedFlow(**ckpt.get("flow_config", {"time_scale": 1000.0})).to(device)
    mean, std = ckpt["affect_mean"], ckpt["affect_std"]
    affect_dim = ckpt["model_config"]["affect_dim"]

    prosody_index = None
    if ckpt.get("prosody_dim", 0) > 0:
        prosody_index = load_mead_prosody(paths.F0_UTT_STATS, paths.ENERGY_UTT_STATS)
    records, _ = load_mead_records(args.prompts_jsonl, args.prompt_emb, args.emo_cache, args.limit,
                                   prosody_index=prosody_index)
    n, K = len(records), max(1, args.num_prompts)
    print(f"records={n} K={K} steps={args.sample_steps} device={device}")

    z_src_all = torch.empty(n, affect_dim)
    real_by_item = {}
    for i, r in enumerate(records):
        zs, zt = affect_vector(r)
        z_src_all[i] = (zs - mean) / std
        si, ti, _ = items_from_pid(r["pair_id"])
        real_by_item.setdefault(si, ((zs - mean) / std).clone())
        real_by_item.setdefault(ti, ((zt - mean) / std).clone())

    g = torch.Generator().manual_seed(args.seed)
    pidx = torch.stack([torch.randint(0, r["prompt_emb"].shape[0], (K,), generator=g) for r in records])

    emb = torch.empty(n, K, affect_dim)
    for k in range(K):
        for s in range(0, n, args.batch_size):
            e = min(n, s + args.batch_size)
            zb = z_src_all[s:e].to(device)
            pb = torch.stack([records[i]["prompt_emb"][pidx[i, k]] for i in range(s, e)]).to(device)
            emb[s:e, k] = flow.sample(den, zb, pb, num_steps=args.sample_steps).cpu()
        print(f"  prompt {k+1}/{K} done", flush=True)

    pairs, by_target_item, by_src_item_tgt = {}, {}, {}
    for i, r in enumerate(records):
        pid = r["pair_id"]
        si, ti, tlv = items_from_pid(pid)
        pairs[pid] = {"emb": emb[i].clone(), "source_item": si, "target_item": ti, "target_level": tlv}
        by_target_item.setdefault(ti, []).append(pid)
        by_src_item_tgt[f"{si}|||{tlv}"] = pid

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_name(out.name + ".tmp")
    torch.save({"pairs": pairs, "by_target_item": by_target_item, "by_src_item_tgt": by_src_item_tgt,
                "real_by_item": real_by_item, "dim": affect_dim, "K": K,
                "sample_steps": args.sample_steps, "standardized": True,
                "affect_mean": mean, "affect_std": std, "ckpt": str(args.ckpt)}, tmp_out)
    os.replace(tmp_out, out)
    print(f"saved pairs={len(pairs)} items={len(real_by_item)} -> {out}")


if __name__ == "__main__":
    main()
