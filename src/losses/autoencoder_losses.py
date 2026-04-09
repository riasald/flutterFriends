"""Loss helpers for KL autoencoder training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AutoencoderLossBreakdown:
    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor
    masked_reconstruction: torch.Tensor
    perceptual: torch.Tensor


def compute_autoencoder_loss(
    *,
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    posterior_kl: torch.Tensor,
    perceptual_loss_fn: Optional[torch.nn.Module] = None,
    mask: Optional[torch.Tensor] = None,
    reconstruction_weight: float = 1.0,
    kl_weight: float = 1e-6,
    masked_reconstruction_weight: float = 0.0,
    perceptual_weight: float = 0.0,
) -> AutoencoderLossBreakdown:
    reconstruction = torch.mean(torch.abs(inputs - reconstructions))
    kl = posterior_kl.mean()

    if mask is not None and masked_reconstruction_weight > 0.0:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(dtype=inputs.dtype)
        masked_reconstruction = torch.mean(torch.abs((inputs - reconstructions) * mask))
    else:
        masked_reconstruction = torch.zeros((), device=inputs.device, dtype=inputs.dtype)

    if perceptual_loss_fn is not None and perceptual_weight > 0.0:
        perceptual = perceptual_loss_fn(inputs, reconstructions)
    else:
        perceptual = torch.zeros((), device=inputs.device, dtype=inputs.dtype)

    total = (
        reconstruction_weight * reconstruction
        + kl_weight * kl
        + masked_reconstruction_weight * masked_reconstruction
        + perceptual_weight * perceptual
    )
    return AutoencoderLossBreakdown(
        total=total,
        reconstruction=reconstruction,
        kl=kl,
        masked_reconstruction=masked_reconstruction,
        perceptual=perceptual,
    )
