#!/usr/bin/env python3
"""Emo-Compass inference: source affect + instruction -> target emotion embedding."""
from __future__ import annotations

import argparse
import pathlib
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

if "pathlib._local" not in sys.modules:
    _m = types.ModuleType("pathlib._local")
    for _n in ("Path", "PurePath", "PosixPath", "WindowsPath", "PurePosixPath", "PureWindowsPath"):
        if hasattr(pathlib, _n):
            setattr(_m, _n, getattr(pathlib, _n))
    sys.modules["pathlib._local"] = _m

import paths
from models.emo_compass import RectifiedFlow, TraceAffectDenoiser


def encode_instruction_e5(texts: list[str], model_dir: str, device: torch.device) -> torch.Tensor:
    """Mean-pooled, L2-normalized E5 embeddings for one or more instructions."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device).eval()
    batch = tok([f"query: {t}" for t in texts], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**batch)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    return F.normalize(emb, dim=-1)


def load_flow(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    den = TraceAffectDenoiser(**ckpt["model_config"]).to(device).eval()
    den.load_state_dict(ckpt["denoiser"])
    flow = RectifiedFlow(**ckpt.get("flow_config", {"time_scale": 1000.0})).to(device)
    mean = ckpt["affect_mean"].to(device)
    std = ckpt["affect_std"].to(device)
    return den, flow, mean, std, ckpt


def as_2d(x: torch.Tensor, name: str, dim: int) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[-1] != dim:
        raise ValueError(f"{name} must be [{dim}] or [N,{dim}], got {tuple(x.shape)}")
    return x.float()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(paths.MEAD_FLOW_CKPT_DIR / "best_trace_mead.pt"))
    p.add_argument("--source-emb", type=str, required=True,
                   help="torch .pt with the source affect vector [1027] or [N,1027].")
    p.add_argument("--prompt-emb", type=str, default=None,
                   help="torch .pt with a precomputed E5 instruction embedding [768] or [N,768].")
    p.add_argument("--prompt-text", type=str, default=None, help="Raw instruction text (encoded by E5).")
    p.add_argument("--e5-model", type=str, default=str(paths.E5_MODEL))
    p.add_argument("--out", type=str, default="emo_compass_pred.pt")
    p.add_argument("--steps", type=int, default=50, help="Euler ODE integration steps.")
    p.add_argument("--standardized-out", action="store_true",
                   help="Keep the target in the standardized space (default: un-standardize).")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if (args.prompt_emb is None) == (args.prompt_text is None):
        raise SystemExit("Provide exactly one of --prompt-emb or --prompt-text.")
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    den, flow, mean, std, ckpt = load_flow(args.ckpt, device)
    affect_dim = ckpt["model_config"]["affect_dim"]
    prompt_dim = ckpt["model_config"]["prompt_dim"]

    z_src = as_2d(torch.load(args.source_emb, map_location="cpu", weights_only=False), "source-emb", affect_dim).to(device)
    if args.prompt_emb is not None:
        prompt = as_2d(torch.load(args.prompt_emb, map_location="cpu", weights_only=False), "prompt-emb", prompt_dim).to(device)
    else:
        prompt = encode_instruction_e5([args.prompt_text], args.e5_model, device)
        if prompt.shape[-1] != prompt_dim:
            raise ValueError(f"E5 dim {prompt.shape[-1]} != checkpoint prompt_dim {prompt_dim}")

    n = max(z_src.shape[0], prompt.shape[0])
    if z_src.shape[0] == 1 and n > 1:
        z_src = z_src.expand(n, -1)
    if prompt.shape[0] == 1 and n > 1:
        prompt = prompt.expand(n, -1)
    if z_src.shape[0] != prompt.shape[0]:
        raise ValueError(f"source ({z_src.shape[0]}) and prompt ({prompt.shape[0]}) batch sizes differ")

    z_src_std = (z_src - mean) / std
    z_tgt_std = flow.sample(den, z_src_std, prompt, num_steps=args.steps)
    z_tgt = z_tgt_std if args.standardized_out else z_tgt_std * std + mean

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"z_tgt": z_tgt.cpu(), "standardized": args.standardized_out,
                "emo_dim": ckpt["emo_dim"], "vad_dim": ckpt["vad_dim"], "ckpt": args.ckpt}, out)
    print(f"predicted ẑ_tgt {tuple(z_tgt.shape)} (standardized={args.standardized_out}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
