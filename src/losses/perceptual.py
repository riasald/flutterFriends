"""Perceptual reconstruction losses for autoencoder training."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG16_Weights, vgg16


class VGGPerceptualLoss(nn.Module):
    """VGG16 feature-space L1 loss.

    This follows the standard perceptual-loss pattern used in image generation:
    pass both input and reconstruction through a frozen feature extractor and
    compare selected intermediate activations with L1 distance.
    """

    def __init__(
        self,
        *,
        resize_to: int = 224,
        layer_ids: Sequence[int] = (3, 8, 15, 22),
        layer_weights: Sequence[float] | None = None,
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_FEATURES if use_pretrained else None
        backbone = vgg16(weights=weights).features.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone = backbone
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        if layer_weights is None:
            self.layer_weights = tuple(1.0 for _ in self.layer_ids)
        else:
            if len(layer_weights) != len(self.layer_ids):
                raise ValueError("layer_weights must have the same length as layer_ids.")
            self.layer_weights = tuple(float(weight) for weight in layer_weights)
        self.resize_to = int(resize_to)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = (tensor.clamp(-1.0, 1.0) + 1.0) / 2.0
        if self.resize_to > 0 and tensor.shape[-1] != self.resize_to:
            tensor = F.interpolate(
                tensor,
                size=(self.resize_to, self.resize_to),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return (tensor - self.mean) / self.std

    def _extract(self, tensor: torch.Tensor) -> Iterable[torch.Tensor]:
        tensor = self._prepare(tensor)
        for layer_index, layer in enumerate(self.backbone):
            tensor = layer(tensor)
            if layer_index in self.layer_ids:
                yield tensor

    def forward(self, inputs: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_features = list(self._extract(inputs))
        recon_features = list(self._extract(reconstructions))
        loss = torch.zeros((), device=reconstructions.device, dtype=reconstructions.dtype)
        for weight, target, recon in zip(self.layer_weights, target_features, recon_features):
            loss = loss + float(weight) * F.l1_loss(recon, target.detach())
        return loss
