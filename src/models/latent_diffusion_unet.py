"""Modest latent-space diffusion U-Net with coordinate/species conditioning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.models.unet_blocks import (
    CrossAttentionBlock,
    Downsample,
    ResidualBlock,
    SpatialSelfAttention,
    TimestepMLP,
    Upsample,
    group_norm,
)


@dataclass(frozen=True)
class LatentDiffusionUNetConfig:
    latent_channels: int = 4
    base_channels: int = 128
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    time_embed_dim: int = 512
    context_dim: int = 128
    attention_levels: tuple[int, ...] = (1, 2)
    num_heads: int = 4
    dropout: float = 0.0


class LatentDiffusionUNet(nn.Module):
    def __init__(self, config: LatentDiffusionUNetConfig) -> None:
        super().__init__()
        self.config = config
        self.time_mlp = TimestepMLP(config.time_embed_dim, config.time_embed_dim)
        self.conv_in = nn.Conv2d(config.latent_channels, config.base_channels, kernel_size=3, padding=1)

        current_channels = config.base_channels
        self.down_blocks = nn.ModuleList()
        self.skip_channels: list[int] = []
        for level, multiplier in enumerate(config.channel_multipliers):
            out_channels = config.base_channels * multiplier
            level_module = nn.ModuleDict()
            level_module["resblocks"] = nn.ModuleList()
            level_module["self_attn"] = nn.ModuleList()
            level_module["cross_attn"] = nn.ModuleList()
            for _ in range(config.num_res_blocks):
                level_module["resblocks"].append(
                    ResidualBlock(current_channels, out_channels, config.time_embed_dim, dropout=config.dropout)
                )
                use_attention = level in config.attention_levels
                level_module["self_attn"].append(
                    SpatialSelfAttention(out_channels, num_heads=config.num_heads) if use_attention else nn.Identity()
                )
                level_module["cross_attn"].append(
                    CrossAttentionBlock(out_channels, config.context_dim, num_heads=config.num_heads)
                    if use_attention
                    else nn.Identity()
                )
                current_channels = out_channels
                self.skip_channels.append(current_channels)
            level_module["downsample"] = (
                Downsample(current_channels) if level != len(config.channel_multipliers) - 1 else nn.Identity()
            )
            self.down_blocks.append(level_module)

        self.mid_block1 = ResidualBlock(current_channels, current_channels, config.time_embed_dim, dropout=config.dropout)
        self.mid_self_attn = SpatialSelfAttention(current_channels, num_heads=config.num_heads)
        self.mid_cross_attn = CrossAttentionBlock(current_channels, config.context_dim, num_heads=config.num_heads)
        self.mid_block2 = ResidualBlock(current_channels, current_channels, config.time_embed_dim, dropout=config.dropout)

        self.up_blocks = nn.ModuleList()
        reversed_multipliers = list(reversed(config.channel_multipliers))
        skip_channels = list(reversed(self.skip_channels))
        for level, multiplier in enumerate(reversed_multipliers):
            out_channels = config.base_channels * multiplier
            level_module = nn.ModuleDict()
            level_module["resblocks"] = nn.ModuleList()
            level_module["self_attn"] = nn.ModuleList()
            level_module["cross_attn"] = nn.ModuleList()
            for _ in range(config.num_res_blocks):
                skip_ch = skip_channels.pop(0)
                level_module["resblocks"].append(
                    ResidualBlock(current_channels + skip_ch, out_channels, config.time_embed_dim, dropout=config.dropout)
                )
                use_attention = (len(reversed_multipliers) - 1 - level) in config.attention_levels
                level_module["self_attn"].append(
                    SpatialSelfAttention(out_channels, num_heads=config.num_heads) if use_attention else nn.Identity()
                )
                level_module["cross_attn"].append(
                    CrossAttentionBlock(out_channels, config.context_dim, num_heads=config.num_heads)
                    if use_attention
                    else nn.Identity()
                )
                current_channels = out_channels
            level_module["upsample"] = Upsample(current_channels) if level != len(reversed_multipliers) - 1 else nn.Identity()
            self.up_blocks.append(level_module)

        self.norm_out = group_norm(current_channels)
        self.conv_out = nn.Conv2d(current_channels, config.latent_channels, kernel_size=3, padding=1)

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(timesteps)
        h = self.conv_in(latents)
        skips: list[torch.Tensor] = []

        for level in self.down_blocks:
            for resblock, self_attn, cross_attn in zip(level["resblocks"], level["self_attn"], level["cross_attn"]):
                h = resblock(h, time_emb)
                h = self_attn(h)
                h = cross_attn(h, context_tokens) if isinstance(cross_attn, CrossAttentionBlock) else cross_attn(h)
                skips.append(h)
            h = level["downsample"](h)

        h = self.mid_block1(h, time_emb)
        h = self.mid_self_attn(h)
        h = self.mid_cross_attn(h, context_tokens)
        h = self.mid_block2(h, time_emb)

        for level in self.up_blocks:
            for resblock, self_attn, cross_attn in zip(level["resblocks"], level["self_attn"], level["cross_attn"]):
                skip = skips.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = torch.cat((h, skip), dim=1)
                h = resblock(h, time_emb)
                h = self_attn(h)
                h = cross_attn(h, context_tokens) if isinstance(cross_attn, CrossAttentionBlock) else cross_attn(h)
            h = level["upsample"](h)

        return self.conv_out(F.silu(self.norm_out(h)))
