"""Reusable residual and attention blocks for the latent diffusion U-Net."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def valid_num_groups(channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(valid_num_groups(channels), channels, eps=1e-6, affine=True)


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        exponent = exponent / max(half_dim - 1, 1)
        emb = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimestepMLP(nn.Module):
    def __init__(self, time_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embed = TimestepEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(timesteps))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SpatialSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = group_norm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x).reshape(batch, channels, height * width).permute(0, 2, 1)
        attended, _ = self.attn(h, h, h, need_weights=False)
        attended = self.out(attended).permute(0, 2, 1).reshape(batch, channels, height, width)
        return x + attended


class CrossAttentionBlock(nn.Module):
    def __init__(self, channels: int, context_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = group_norm(channels)
        self.context_norm = nn.LayerNorm(context_dim)
        self.query_proj = nn.Linear(channels, channels)
        self.key_proj = nn.Linear(context_dim, channels)
        self.value_proj = nn.Linear(context_dim, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x).reshape(batch, channels, height * width).permute(0, 2, 1)
        context = self.context_norm(context_tokens)
        query = self.query_proj(h)
        key = self.key_proj(context)
        value = self.value_proj(context)
        attended, _ = self.attn(query, key, value, need_weights=False)
        attended = self.out_proj(attended).permute(0, 2, 1).reshape(batch, channels, height, width)
        return x + attended


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
