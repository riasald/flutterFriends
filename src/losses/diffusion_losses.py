"""Loss helpers for latent diffusion training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def diffusion_mse_loss(
    prediction: torch.Tensor,
    target_noise: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.shape != target_noise.shape:
        raise ValueError(
            f"Diffusion prediction shape {tuple(prediction.shape)} does not match target shape {tuple(target_noise.shape)}."
        )
    per_sample = F.mse_loss(prediction, target_noise, reduction="none").reshape(prediction.shape[0], -1).mean(dim=1)
    if sample_weight is not None:
        weight = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype)
        return (per_sample * weight).mean()
    return per_sample.mean()
