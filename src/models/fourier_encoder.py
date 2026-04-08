"""Fourier-feature location encoder for latitude/longitude conditioning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FourierLocationEncoderConfig:
    n_freq: int = 32
    hidden_dim: int = 256
    embed_dim: int = 256
    token_dim: int = 128
    n_tokens: int = 4
    dropout: float = 0.1


class FourierLocationEncoder(nn.Module):
    """Encode normalized latitude/longitude pairs with deterministic Fourier features."""

    def __init__(self, config: FourierLocationEncoderConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = 4 * config.n_freq
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.embed_dim),
        )
        self.token_proj = nn.Linear(config.embed_dim, config.n_tokens * config.token_dim)

    def fourier_map(self, latlon_norm: torch.Tensor) -> torch.Tensor:
        feats = []
        for k in range(self.config.n_freq):
            freq = (2.0 ** k) * math.pi
            feats.append(torch.sin(freq * latlon_norm))
            feats.append(torch.cos(freq * latlon_norm))
        return torch.cat(feats, dim=-1)

    def forward(self, latlon_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mapped = self.fourier_map(latlon_norm)
        z_loc = self.mlp(mapped)
        tokens = self.token_proj(z_loc).view(
            latlon_norm.shape[0],
            self.config.n_tokens,
            self.config.token_dim,
        )
        return z_loc, tokens
