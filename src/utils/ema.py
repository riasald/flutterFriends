"""Minimal EMA helper for model parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


@dataclass
class ExponentialMovingAverage:
    decay: float
    shadow: Dict[str, torch.Tensor]

    @classmethod
    def from_model(cls, model: nn.Module, decay: float) -> "ExponentialMovingAverage":
        shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.state_dict().items()
            if torch.is_floating_point(parameter)
        }
        return cls(decay=decay, shadow=shadow)

    def update(self, model: nn.Module) -> None:
        for name, parameter in model.state_dict().items():
            if name not in self.shadow:
                continue
            shadow_tensor = self.shadow[name]
            if shadow_tensor.device != parameter.device or shadow_tensor.dtype != parameter.dtype:
                shadow_tensor = shadow_tensor.to(device=parameter.device, dtype=parameter.dtype)
                self.shadow[name] = shadow_tensor
            shadow_tensor.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.shadow = {name: tensor.clone() for name, tensor in state_dict.items()}

    def copy_to_model(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        original: Dict[str, torch.Tensor] = {}
        state_dict = model.state_dict()
        for name, ema_value in self.shadow.items():
            if name not in state_dict:
                continue
            tensor = state_dict[name]
            if not torch.is_floating_point(tensor):
                continue
            original[name] = tensor.detach().clone()
            tensor.copy_(ema_value.to(device=tensor.device, dtype=tensor.dtype))
        return original

    def restore(self, model: nn.Module, original: Dict[str, torch.Tensor]) -> None:
        state_dict = model.state_dict()
        for name, value in original.items():
            if name in state_dict:
                state_dict[name].copy_(value.to(device=state_dict[name].device, dtype=state_dict[name].dtype))
