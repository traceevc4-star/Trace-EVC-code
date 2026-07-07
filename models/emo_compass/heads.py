"""Auxiliary heads: emotion classifier, intensity RankNet, mean-prosody predictor."""
from __future__ import annotations

import torch
import torch.nn as nn


class EmotionClassifierHead(nn.Module):
    """Emotion-category classifier on frozen emotion2vec embeddings."""

    def __init__(self, input_dim: int, hidden: int, dropout: float, num_classes: int = 5):
        super().__init__()
        if hidden <= 0:
            self.net = nn.Linear(input_dim, num_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IntensityHead(nn.Module):
    """Emotion-conditioned scalar intensity, trained as a RankNet over VAD."""

    def __init__(self, input_dim: int = 3, n_emo: int = 4, emo_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.emo = nn.Embedding(n_emo, emo_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + emo_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor, emo_id: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([x, self.emo(emo_id)], dim=-1)).squeeze(-1)
