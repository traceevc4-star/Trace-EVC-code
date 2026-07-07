"""Distributed, DataLoader, head-loading and checkpoint helpers for the flow trainers."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.emo_compass import EmotionClassifierHead, IntensityHead

from .data import TracePromptDataset, collate


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def setup_device(args: argparse.Namespace) -> tuple[torch.device, int, int, int]:
    """Return (device, local_rank, rank, world_size); init NCCL under torchrun."""
    if "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested by torchrun, but CUDA is not available.")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), local_rank, dist.get_rank(), dist.get_world_size()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    return device, 0, 0, 1


def cleanup_dist() -> None:
    if is_dist():
        dist.destroy_process_group()


def reduce_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not is_dist():
        return metrics
    keys = sorted(metrics)
    vals = torch.tensor([metrics[k] for k in keys], dtype=torch.float32, device=device)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals /= dist.get_world_size()
    return {k: vals[i].item() for i, k in enumerate(keys)}


def aggregate(metrics: list[dict]) -> dict[str, float]:
    return {k: torch.stack([m[k].float().cpu() for m in metrics]).mean().item() for k in metrics[0]}


def build_loaders(train_records, val_records, test_records, mean, std, args, rank, world):
    """Return (train_loader, val_loader, test_loader, train_sampler)."""
    pin = None
    train_ds = TracePromptDataset(train_records, mean, std, "train")
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        if is_dist() else None
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, collate_fn=collate,
        drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        TracePromptDataset(val_records, mean, std, "val"), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
    )
    test_loader = DataLoader(
        TracePromptDataset(test_records, mean, std, "test"), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_sampler


def load_emotion_head(path: Path, device: torch.device):
    """Return (head, mean, std, emotions) — a frozen emotion2vec classifier."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    head = EmotionClassifierHead(ckpt["input_dim"], ckpt["hidden"], ckpt["dropout"], len(ckpt["emotions"])).to(device)
    head.load_state_dict(ckpt["model"])
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head, ckpt["mean"].to(device), ckpt["std"].to(device), list(ckpt["emotions"])


def load_intensity_head(path: Path, device: torch.device):
    """Return (head, mean, std) — a frozen VAD RankNet intensity head."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    head = IntensityHead(ckpt["input_dim"], ckpt["n_emo"], ckpt["emo_dim"], ckpt["hidden"]).to(device)
    head.load_state_dict(ckpt["model"])
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head, ckpt["mean"].to(device), ckpt["std"].to(device)


def save_flow_checkpoint(path: Path, core, args, affect_dim, prompt_dim, mean, std, emotions, extra: dict):
    """Write the flow checkpoint with the keys precompute_targets.py expects."""
    payload = {
        "denoiser": core.state_dict(),
        "model_config": {
            "affect_dim": affect_dim, "prompt_dim": prompt_dim, "hidden": args.hidden,
            "num_layers": args.num_layers, "nhead": args.nhead, "dropout": args.dropout,
        },
        "flow_config": {"time_scale": args.flow_time_scale},
        "prediction_type": "flow_velocity",
        "affect_mean": mean, "affect_std": std,
        "emo_dim": args.emo_dim, "vad_dim": args.vad_dim,
        "prosody_dim": getattr(args, "prosody_dim", 0), "emotions": emotions,
    }
    payload.update(extra)
    torch.save(payload, path)
