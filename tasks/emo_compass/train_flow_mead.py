#!/usr/bin/env python3
"""Train Emo-Compass on MEAD (intra-emotion intensity)."""
from __future__ import annotations

import argparse
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

import paths
from models.emo_compass import RectifiedFlow, TraceAffectDenoiser
from tasks.emo_compass import common
from tasks.emo_compass.common import is_dist, is_main
from tasks.emo_compass.data import (
    EMOTIONS_MEAD, affect_vector, compute_affect_stats, load_mead_records, load_mead_prosody,
    load_mead_split, split_by_speaker,
)


def step_batch(den, flow, heads, batch, device, args, mean, std):
    emo_head, eh_m, eh_s, int_head, ih_m, ih_s = heads
    z_src = batch["z_src"].to(device)
    z_tgt = batch["z_tgt"].to(device)
    prompt = batch["prompt_emb"].to(device)
    y = batch["target_label"].to(device)
    emo_id = batch["emo_id"].to(device)

    s = torch.rand(z_tgt.shape[0], device=device)
    x_s, v_target = flow.interpolate(z_src, z_tgt, s)
    v_pred = flow.velocity(den, x_s, s, z_src, prompt)
    v_mse = F.mse_loss(v_pred, v_target)
    z_tgt_pred = x_s + (1.0 - s).view(-1, 1) * v_pred
    z_cos = 1.0 - F.cosine_similarity(z_tgt_pred, z_tgt, dim=-1).mean()

    md, sd = mean.to(device), std.to(device)
    raw_pred, raw_tgt = z_tgt_pred * sd + md, z_tgt * sd + md
    _v0, _v1 = args.emo_dim, args.emo_dim + args.vad_dim
    emo_pred = raw_pred[:, : args.emo_dim]
    vad_pred, vad_tgt = raw_pred[:, _v0:_v1], raw_tgt[:, _v0:_v1]

    logits = emo_head((emo_pred - eh_m) / eh_s)
    emo_loss = F.cross_entropy(logits, y)
    emo_acc = (logits.argmax(-1) == y).float().mean()

    s_pred = int_head((vad_pred - ih_m) / ih_s, emo_id)
    s_tgt = int_head((vad_tgt - ih_m) / ih_s, emo_id).detach()
    int_loss = F.mse_loss(s_pred, s_tgt)
    vad_loss = F.mse_loss(z_tgt_pred[:, args.emo_dim:], z_tgt[:, args.emo_dim:])

    loss = (v_mse + args.lambda_z_cos * z_cos + args.lambda_emo * emo_loss
            + args.lambda_int * int_loss + args.lambda_vad * vad_loss)
    return loss, {
        "loss": loss.detach(), "v_mse": v_mse.detach(), "z_cos": z_cos.detach(),
        "emo": emo_loss.detach(), "emo_acc": emo_acc.detach(),
        "intensity": int_loss.detach(), "vad": vad_loss.detach(),
    }


@torch.no_grad()
def evaluate_sampling(den, flow, heads, loader, device, args, mean, std):
    emo_head, eh_m, eh_s, int_head, ih_m, ih_s = heads
    den.eval()
    md, sd = mean.to(device), std.to(device)
    metrics = []
    for batch in loader:
        z_src = batch["z_src"].to(device); z_tgt = batch["z_tgt"].to(device)
        prompt = batch["prompt_emb"].to(device); y = batch["target_label"].to(device)
        emo_id = batch["emo_id"].to(device)
        z0 = flow.sample(den, z_src, prompt, num_steps=max(1, args.sample_steps))
        gen_mse = F.mse_loss(z0, z_tgt)
        raw_pred, raw_tgt, raw_src = z0 * sd + md, z_tgt * sd + md, z_src * sd + md
        ep, et = raw_pred[:, : args.emo_dim], raw_tgt[:, : args.emo_dim]
        _v0, _v1 = args.emo_dim, args.emo_dim + args.vad_dim
        vp, vt, vs = raw_pred[:, _v0:_v1], raw_tgt[:, _v0:_v1], raw_src[:, _v0:_v1]
        emo_cos = F.cosine_similarity(ep, et, -1).mean()
        vad_l2 = torch.linalg.vector_norm(vp - vt, dim=-1).mean()
        logits = emo_head((ep - eh_m) / eh_s)
        emo_acc = (logits.argmax(-1) == y).float().mean()
        s_pred = int_head((vp - ih_m) / ih_s, emo_id)
        s_tgt = int_head((vt - ih_m) / ih_s, emo_id)
        s_src = int_head((vs - ih_m) / ih_s, emo_id)
        int_mse = F.mse_loss(s_pred, s_tgt)
        int_dir = ((s_pred - s_src).sign() == (s_tgt - s_src).sign()).float().mean()
        loss = gen_mse + args.lambda_emo * F.cross_entropy(logits, y) + args.lambda_int * int_mse
        metrics.append({"loss": loss.detach(), "gen_mse": gen_mse.detach(), "emo_cos": emo_cos.detach(),
                        "vad_l2": vad_l2.detach(), "emo_acc": emo_acc.detach(),
                        "int_mse": int_mse.detach(), "int_dir": int_dir.detach()})
    return common.aggregate(metrics)


def train_epoch(den, flow, heads, loader, opt, device, args, mean, std):
    den.train()
    metrics = []
    for batch in loader:
        loss, m = step_batch(den, flow, heads, batch, device, args, mean, std)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(den.parameters(), args.grad_clip); opt.step()
        metrics.append(m)
    return common.aggregate(metrics)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts-jsonl", type=str, default=str(paths.MEAD_PROMPTS))
    p.add_argument("--prompt-emb", type=str, default=str(paths.MEAD_PROMPT_EMB))
    p.add_argument("--emo-cache", type=str, default=str(paths.MEAD_EMO2VEC))
    p.add_argument("--emotion-head", type=str, default=str(paths.MEAD_EMOTION_HEAD))
    p.add_argument("--intensity-head", type=str, default=str(paths.MEAD_INTENSITY_HEAD))
    p.add_argument("--split-json", type=str, default=str(paths.MEAD_SPLIT))
    p.add_argument("--checkpoint-dir", type=str, default=str(paths.MEAD_FLOW_CKPT_DIR))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--nhead", type=int, default=12)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--flow-time-scale", type=float, default=1000.0)
    p.add_argument("--f0-stats", type=str, default=str(paths.F0_UTT_STATS))
    p.add_argument("--energy-stats", type=str, default=str(paths.ENERGY_UTT_STATS))
    p.add_argument("--lambda-z-cos", type=float, default=0.1)
    p.add_argument("--lambda-emo", type=float, default=0.2)
    p.add_argument("--lambda-int", type=float, default=0.5)
    p.add_argument("--lambda-vad", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--sample-steps", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.emo_dim, args.vad_dim, args.prosody_dim = 1024, 3, 2
    random.seed(args.seed); torch.manual_seed(args.seed)
    device, local_rank, rank, world = common.setup_device(args)
    if is_main():
        print(f"distributed={is_dist()} rank={rank} world={world} device={device}")

    prosody_index = load_mead_prosody(args.f0_stats, args.energy_stats)
    records, prompt_dim = load_mead_records(args.prompts_jsonl, args.prompt_emb, args.emo_cache, args.limit,
                                            prosody_index=prosody_index)
    _, val_s, test_s = load_mead_split(args.split_json)
    val_s, test_s = set(val_s), set(test_s)
    train_records = [r for r in records if r["speaker_id"] not in val_s | test_s]
    val_records = split_by_speaker(records, val_s)
    test_records = split_by_speaker(records, test_s)
    if is_main():
        for nm, rr in [("train", train_records), ("val", val_records), ("test", test_records)]:
            print(f"{nm}: n={len(rr)} speakers={sorted(set(r['speaker_id'] for r in rr))}")
    if not (train_records and val_records and test_records):
        raise RuntimeError("Empty split.")

    mean, std = compute_affect_stats(train_records)
    train_loader, val_loader, test_loader, train_sampler = common.build_loaders(
        train_records, val_records, test_records, mean, std, args, rank, world)

    affect_dim = args.emo_dim + args.vad_dim + args.prosody_dim
    den = TraceAffectDenoiser(affect_dim=affect_dim, prompt_dim=prompt_dim, hidden=args.hidden,
                              num_layers=args.num_layers, nhead=args.nhead, dropout=args.dropout).to(device)
    flow = RectifiedFlow(time_scale=args.flow_time_scale).to(device)
    eh, ehm, ehs, head_emos = common.load_emotion_head(args.emotion_head, device)
    assert head_emos == EMOTIONS_MEAD, (head_emos, EMOTIONS_MEAD)
    ih, ihm, ihs = common.load_intensity_head(args.intensity_head, device)
    heads = (eh, ehm, ehs, ih, ihm, ihs)

    n_params = sum(p.numel() for p in den.parameters())
    if is_dist():
        den = DDP(den, device_ids=[local_rank], output_device=local_rank)
    if is_main():
        print(f"affect_dim={affect_dim} prompt_dim={prompt_dim} params={n_params/1e6:.2f}M "
              f"lambdas emo={args.lambda_emo} int={args.lambda_int} vad={args.lambda_vad} zcos={args.lambda_z_cos}")

    opt = AdamW(den.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_trace_mead.pt"

    best, bad = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tr = common.reduce_metrics(train_epoch(den, flow, heads, train_loader, opt, device, args, mean, std), device)
        sched.step()
        stop = False
        if is_main():
            core = den.module if isinstance(den, DDP) else den
            va = evaluate_sampling(core, flow, heads, val_loader, device, args, mean, std)
            print(f"ep{epoch:03d} train loss={tr['loss']:.4f} v_mse={tr['v_mse']:.4f} emo={tr['emo']:.4f}"
                  f"(acc{tr['emo_acc']*100:.0f}) int={tr['intensity']:.4f} vad={tr['vad']:.4f} | "
                  f"val[smp] loss={va['loss']:.4f} gen_mse={va['gen_mse']:.4f} emo_cos={va['emo_cos']:.4f} "
                  f"emo_acc={va['emo_acc']*100:.1f}% int_mse={va['int_mse']:.4f} int_dir={va['int_dir']*100:.1f}%")
            if va["loss"] < best:
                best, bad = va["loss"], 0
                common.save_flow_checkpoint(best_path, core, args, affect_dim, prompt_dim, mean, std, EMOTIONS_MEAD,
                                            extra={"args": {k: str(v) for k, v in vars(args).items()},
                                                   "epoch": epoch, "best_val_loss": best, "val_metrics": va})
                print(f"  saved best -> {best_path}")
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"early stop @ep{epoch}"); stop = True
        if is_dist():
            t = torch.tensor([1 if stop else 0], device=device); dist.broadcast(t, src=0)
            if t.item():
                break
        elif stop:
            break

    if is_main():
        print(f"done: best val_loss={best:.4f} checkpoint={best_path}")
    common.cleanup_dist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
