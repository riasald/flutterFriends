"""KL autoencoder for latent butterfly image modeling.

This module is intentionally modeled on the first-stage autoencoder used in the
official CompVis Latent Diffusion repository and the Hugging Face Diffusers
``AutoencoderKL`` implementation, but rewritten as a small, self-contained
plain PyTorch module for this repository.

Primary references:
- https://github.com/CompVis/latent-diffusion
- https://huggingface.co/docs/diffusers/api/models/autoencoderkl
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl.py

The default architecture follows the widely used 256px LDM VAE recipe:
- latent channels: 4
- base channels: 128
- channel multipliers: (1, 2, 4, 4)
- downsampling factor: 8
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _valid_num_groups(channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(
        num_groups=_valid_num_groups(channels),
        num_channels=channels,
        eps=1e-6,
        affine=True,
    )


@dataclass(frozen=True)
class AutoencoderConfig:
    """Architecture and latent-space settings for the KL autoencoder."""

    image_size: int = 256
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 128
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    latent_channels: int = 4
    dropout: float = 0.0
    attention_resolutions: Tuple[int, ...] = ()
    mid_attention: bool = True
    scaling_factor: float = 0.18215
    tanh_out: bool = False

    @property
    def downsample_factor(self) -> int:
        return 2 ** max(len(self.channel_multipliers) - 1, 0)

    @property
    def latent_resolution(self) -> int:
        return self.image_size // self.downsample_factor


class DiagonalGaussianDistribution:
    """Gaussian posterior parameterized by per-location mean and log variance."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, min=-30.0, max=20.0)
        self.deterministic = deterministic

        if deterministic:
            self.std = torch.zeros_like(self.mean)
            self.var = torch.zeros_like(self.mean)
        else:
            self.std = torch.exp(0.5 * self.logvar)
            self.var = torch.exp(self.logvar)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None) -> torch.Tensor:
        if self.deterministic:
            return torch.zeros(
                self.mean.shape[0],
                device=self.mean.device,
                dtype=self.mean.dtype,
            )

        reduction_dims = tuple(range(1, self.mean.ndim))
        if other is None:
            return 0.5 * torch.sum(
                self.mean.pow(2) + self.var - 1.0 - self.logvar,
                dim=reduction_dims,
            )

        if other.deterministic:
            raise ValueError("KL against a deterministic distribution is undefined.")

        return 0.5 * torch.sum(
            ((self.mean - other.mean).pow(2) / other.var)
            + (self.var / other.var)
            - 1.0
            - self.logvar
            + other.logvar,
            dim=reduction_dims,
        )

    def nll(self, sample: torch.Tensor) -> torch.Tensor:
        if self.deterministic:
            return torch.zeros(
                self.mean.shape[0],
                device=self.mean.device,
                dtype=self.mean.dtype,
            )
        log_two_pi = torch.log(torch.tensor(2.0 * torch.pi, device=sample.device, dtype=sample.dtype))
        reduction_dims = tuple(range(1, sample.ndim))
        return 0.5 * torch.sum(
            log_two_pi + self.logvar + (sample - self.mean).pow(2) / self.var,
            dim=reduction_dims,
        )


class ResnetBlock(nn.Module):
    """ResNet block used in the original LDM first-stage autoencoder."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = _group_norm(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(batch, channels, height * width).permute(0, 2, 1)
        k = self.k(h).reshape(batch, channels, height * width)
        attn = torch.bmm(q, k) * (channels ** -0.5)
        attn = torch.softmax(attn, dim=-1)

        v = self.v(h).reshape(batch, channels, height * width)
        h = torch.bmm(v, attn.transpose(1, 2)).reshape(batch, channels, height, width)
        h = self.proj_out(h)
        return x + h


class Downsample(nn.Module):
    """2x spatial downsampling with a learned convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    """2x spatial upsampling followed by a 3x3 convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    """Hierarchical encoder that predicts latent posterior moments."""

    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int,
        base_channels: int,
        channel_multipliers: Sequence[int],
        num_res_blocks: int,
        latent_channels: int,
        dropout: float,
        attention_resolutions: Sequence[int],
        mid_attention: bool,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        current_resolution = image_size
        block_in = base_channels
        self.down = nn.ModuleList()
        for level, multiplier in enumerate(channel_multipliers):
            block_out = base_channels * multiplier
            down = nn.Module()
            down.blocks = nn.ModuleList()
            down.attentions = nn.ModuleList()

            for _ in range(num_res_blocks):
                down.blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                down.attentions.append(
                    AttentionBlock(block_out)
                    if current_resolution in attention_resolutions
                    else nn.Identity()
                )
                block_in = block_out

            if level != len(channel_multipliers) - 1:
                down.downsample = Downsample(block_in)
                current_resolution //= 2
            else:
                down.downsample = nn.Identity()

            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = AttentionBlock(block_in) if mid_attention else nn.Identity()
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout=dropout)

        self.norm_out = _group_norm(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)

        for down in self.down:
            for block, attn in zip(down.blocks, down.attentions):
                h = attn(block(h))
            h = down.downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


class Decoder(nn.Module):
    """Hierarchical decoder that reconstructs images from latent tensors."""

    def __init__(
        self,
        *,
        image_size: int,
        out_channels: int,
        base_channels: int,
        channel_multipliers: Sequence[int],
        num_res_blocks: int,
        latent_channels: int,
        dropout: float,
        attention_resolutions: Sequence[int],
        mid_attention: bool,
        tanh_out: bool,
    ) -> None:
        super().__init__()
        block_in = base_channels * channel_multipliers[-1]
        current_resolution = image_size // (2 ** (len(channel_multipliers) - 1))

        self.conv_in = nn.Conv2d(latent_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = AttentionBlock(block_in) if mid_attention else nn.Identity()
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout=dropout)

        self.up = nn.ModuleList()
        for level, multiplier in enumerate(reversed(tuple(channel_multipliers))):
            block_out = base_channels * multiplier
            up = nn.Module()
            up.blocks = nn.ModuleList()
            up.attentions = nn.ModuleList()

            for _ in range(num_res_blocks + 1):
                up.blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                up.attentions.append(
                    AttentionBlock(block_out)
                    if current_resolution in attention_resolutions
                    else nn.Identity()
                )
                block_in = block_out

            if level != len(channel_multipliers) - 1:
                up.upsample = Upsample(block_in)
                current_resolution *= 2
            else:
                up.upsample = nn.Identity()

            self.up.append(up)

        self.norm_out = _group_norm(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)
        self.tanh_out = tanh_out

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for up in self.up:
            for block, attn in zip(up.blocks, up.attentions):
                h = attn(block(h))
            h = up.upsample(h)

        h = self.conv_out(F.silu(self.norm_out(h)))
        if self.tanh_out:
            h = torch.tanh(h)
        return h


@dataclass
class AutoencoderForwardOutput:
    """Structured forward output with reconstruction and KL posterior."""

    reconstruction: torch.Tensor
    latents: torch.Tensor
    posterior: DiagonalGaussianDistribution


class ButterflyAutoencoder(nn.Module):
    """KL-regularized autoencoder for 256px butterfly images."""

    def __init__(self, config: Optional[AutoencoderConfig] = None, **kwargs: object) -> None:
        super().__init__()
        self.config = config or AutoencoderConfig(**kwargs)
        self._validate_config(self.config)

        self.encoder = Encoder(
            image_size=self.config.image_size,
            in_channels=self.config.in_channels,
            base_channels=self.config.base_channels,
            channel_multipliers=self.config.channel_multipliers,
            num_res_blocks=self.config.num_res_blocks,
            latent_channels=self.config.latent_channels,
            dropout=self.config.dropout,
            attention_resolutions=self.config.attention_resolutions,
            mid_attention=self.config.mid_attention,
        )
        self.decoder = Decoder(
            image_size=self.config.image_size,
            out_channels=self.config.out_channels,
            base_channels=self.config.base_channels,
            channel_multipliers=self.config.channel_multipliers,
            num_res_blocks=self.config.num_res_blocks,
            latent_channels=self.config.latent_channels,
            dropout=self.config.dropout,
            attention_resolutions=self.config.attention_resolutions,
            mid_attention=self.config.mid_attention,
            tanh_out=self.config.tanh_out,
        )

        self.quant_conv = nn.Conv2d(
            2 * self.config.latent_channels,
            2 * self.config.latent_channels,
            kernel_size=1,
        )
        self.post_quant_conv = nn.Conv2d(
            self.config.latent_channels,
            self.config.latent_channels,
            kernel_size=1,
        )
        self.scaling_factor = self.config.scaling_factor

    @staticmethod
    def _validate_config(config: AutoencoderConfig) -> None:
        if config.image_size % config.downsample_factor != 0:
            raise ValueError(
                "image_size must be divisible by the autoencoder downsample factor. "
                f"Got image_size={config.image_size} and factor={config.downsample_factor}."
            )
        if not config.channel_multipliers:
            raise ValueError("channel_multipliers must contain at least one level.")
        if config.latent_channels <= 0:
            raise ValueError("latent_channels must be positive.")

    @property
    def downsample_factor(self) -> int:
        return self.config.downsample_factor

    @property
    def latent_resolution(self) -> int:
        return self.config.latent_resolution

    def encode_distribution(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        moments = self.quant_conv(self.encoder(x))
        return DiagonalGaussianDistribution(moments)

    def encode(
        self,
        x: torch.Tensor,
        *,
        sample_posterior: bool = False,
        return_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        posterior = self.encode_distribution(x)
        latents = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        if return_posterior:
            return latents, posterior
        return latents

    def encode_for_diffusion(
        self,
        x: torch.Tensor,
        *,
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        latents = self.encode(
            x,
            sample_posterior=sample_posterior,
            return_posterior=False,
            generator=generator,
        )
        return latents * self.scaling_factor

    def decode(self, z: torch.Tensor, *, from_scaled_latents: bool = False) -> torch.Tensor:
        if from_scaled_latents:
            z = z / self.scaling_factor
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def decode_from_diffusion_latents(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z, from_scaled_latents=True)

    def forward(
        self,
        x: torch.Tensor,
        *,
        sample_posterior: bool = True,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> AutoencoderForwardOutput | Tuple[torch.Tensor, torch.Tensor, DiagonalGaussianDistribution]:
        posterior = self.encode_distribution(x)
        latents = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        reconstruction = self.decode(latents)

        if return_dict:
            return AutoencoderForwardOutput(
                reconstruction=reconstruction,
                latents=latents,
                posterior=posterior,
            )
        return reconstruction, latents, posterior


def load_autoencoder_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    use_ema: bool = False,
    freeze: bool = False,
) -> ButterflyAutoencoder:
    """Load a trained autoencoder checkpoint for frozen downstream use."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_dict = checkpoint.get("config", {}).get("model")
    if not config_dict:
        raise RuntimeError(f"Checkpoint {checkpoint_path} is missing model config metadata.")
    model = ButterflyAutoencoder(AutoencoderConfig(**config_dict)).to(device)
    state_dict = checkpoint["model_state_dict"]
    if use_ema and checkpoint.get("ema_state_dict"):
        state_dict = checkpoint["ema_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model
