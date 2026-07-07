"""Source-anchored rectified flow from the source to the target emotion embedding."""
from __future__ import annotations

import torch
import torch.nn as nn

from .denoiser import TraceAffectDenoiser


class RectifiedFlow(nn.Module):
    """Source-anchored conditional flow-matching bridge (z_src -> z_tgt).

    `time_scale` maps the continuous flow time s in [0, 1] onto the sinusoidal
    time-embedding range the denoiser expects (built for integer diffusion steps).
    """

    def __init__(self, time_scale: float = 1000.0):
        super().__init__()
        self.time_scale = time_scale

    def interpolate(
        self, z_src: torch.Tensor, z_tgt: torch.Tensor, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x_s, v_target) for a batch of flow times s (Eq. 4-5)."""
        ss = s.view(-1, *([1] * (z_src.ndim - 1)))
        x_s = (1.0 - ss) * z_src + ss * z_tgt
        v_target = z_tgt - z_src
        return x_s, v_target

    def velocity(
        self,
        denoiser: TraceAffectDenoiser,
        x_s: torch.Tensor,
        s: torch.Tensor,
        z_src: torch.Tensor,
        prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        return denoiser(x_s, s * self.time_scale, z_src, prompt_emb)

    @torch.no_grad()
    def sample(
        self,
        denoiser: TraceAffectDenoiser,
        z_src: torch.Tensor,
        prompt_emb: torch.Tensor,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Euler ODE integration from x(0)=z_src to x(1)~=z_tgt (Eq. 7)."""
        x = z_src.clone()
        grid = torch.linspace(0.0, 1.0, num_steps + 1, device=z_src.device)
        for i in range(num_steps):
            s = torch.full((z_src.shape[0],), grid[i].item(), device=z_src.device)
            dt = (grid[i + 1] - grid[i]).item()
            v = self.velocity(denoiser, x, s, z_src, prompt_emb)
            x = x + dt * v
        return x
