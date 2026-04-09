"""Loss helpers for long-tailed species prior training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _class_weights_from_counts(
    counts: torch.Tensor,
    *,
    strategy: str = "effective_num",
    beta: float = 0.9999,
) -> torch.Tensor:
    counts = counts.to(dtype=torch.float32).clamp_min(1.0)
    strategy_key = strategy.strip().lower()
    if strategy_key in {"", "none", "off"}:
        return torch.ones_like(counts)
    if strategy_key == "inverse_freq":
        weights = counts.reciprocal()
    elif strategy_key == "sqrt_inv":
        weights = counts.rsqrt()
    elif strategy_key == "effective_num":
        beta_tensor = torch.tensor(beta, dtype=torch.float32, device=counts.device)
        weights = (1.0 - beta_tensor) / (1.0 - torch.pow(beta_tensor, counts))
    else:
        raise RuntimeError(f"Unsupported class weighting strategy: {strategy}")
    return weights * (counts.numel() / weights.sum())


def weighting_strategy_for_epoch(
    *,
    strategy: str,
    epoch: int,
    total_epochs: int,
    drw_start_epoch: int = 0,
) -> str:
    """Return the active class-weighting strategy for the current epoch.

    Implements a simple form of deferred re-weighting (DRW): keep class
    weighting off for the early phase of training, then enable the configured
    weighting strategy after ``drw_start_epoch``.
    """

    strategy_key = strategy.strip().lower()
    if strategy_key in {"", "none", "off"}:
        return "none"
    if drw_start_epoch <= 0:
        return strategy
    if epoch < drw_start_epoch:
        return "none"
    return strategy


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss for long-tailed classification."""

    def __init__(
        self,
        class_counts: Sequence[int] | torch.Tensor,
        *,
        max_margin: float = 0.5,
        scale: float = 30.0,
        class_weight_strategy: str = "effective_num",
        class_weight_beta: float = 0.9999,
    ) -> None:
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        margins = counts.clamp_min(1.0).pow(-0.25)
        margins = margins * (max_margin / margins.max())
        class_weights = _class_weights_from_counts(
            counts,
            strategy=class_weight_strategy,
            beta=class_weight_beta,
        )
        self.register_buffer("margins", margins)
        self.register_buffer("class_weights", class_weights)
        self.scale = float(scale)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must be [batch, num_classes]")
        if targets.ndim != 1:
            raise ValueError("targets must be [batch]")
        margins = self.margins[targets]
        adjusted = logits.clone()
        adjusted[torch.arange(logits.shape[0], device=logits.device), targets] -= margins
        return F.cross_entropy(self.scale * adjusted, targets, weight=self.class_weights)


@dataclass(frozen=True)
class PriorMetrics:
    loss: float
    top1: float
    top5: float
    macro_f1: float


def topk_accuracies(logits: torch.Tensor, targets: torch.Tensor, ks: Iterable[int] = (1, 5)) -> dict[int, float]:
    with torch.no_grad():
        max_k = min(max(ks), logits.shape[1])
        _, pred = logits.topk(max_k, dim=1)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        metrics: dict[int, float] = {}
        for k in ks:
            k_eff = min(k, logits.shape[1])
            correct_k = correct[:k_eff].reshape(-1).float().sum()
            metrics[int(k)] = float(correct_k.item() / targets.shape[0])
        return metrics
