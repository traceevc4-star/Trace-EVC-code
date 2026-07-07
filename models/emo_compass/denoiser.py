"""Transformer that predicts the affective velocity for the Emo-Compass flow."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding of the (continuous) flow time."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TraceAffectDenoiser(nn.Module):
    """Transformer denoiser D_theta that predicts the affective velocity.

    Tokens fed to the encoder: [current state x_s, source anchor z_src, instruction
    embedding, flow-time embedding]. The output at the state-token position is the
    predicted velocity (Eq. 5).
    """

    def __init__(
        self,
        affect_dim: int = 1027,
        prompt_dim: int = 768,
        hidden: int = 768,
        num_layers: int = 6,
        nhead: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden % nhead != 0:
            raise ValueError(f"hidden={hidden} must be divisible by nhead={nhead}")
        self.affect_dim = affect_dim
        self.prompt_dim = prompt_dim
        self.hidden = hidden

        self.xt_proj = nn.Sequential(nn.Linear(affect_dim, hidden), nn.LayerNorm(hidden))
        self.src_proj = nn.Sequential(nn.Linear(affect_dim, hidden), nn.LayerNorm(hidden))
        self.prompt_proj = nn.Sequential(nn.Linear(prompt_dim, hidden), nn.LayerNorm(hidden))
        self.time_emb = SinusoidalTimeEmbedding(hidden)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, affect_dim))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_src: torch.Tensor,
        prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.stack(
            [
                self.xt_proj(x_t),
                self.src_proj(z_src),
                self.prompt_proj(prompt_emb),
                self.time_proj(self.time_emb(t)),
            ],
            dim=1,
        )
        h = self.encoder(tokens)
        return self.out(h[:, 0])
