"""Condition-token helpers for latent diffusion conditioning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ConditionFuserConfig:
    prior_dim: int = 128
    species_mix_dim: int = 128
    token_dim: int = 128
    n_null_tokens: int = 5
    species_mix_alpha: float = 0.5


class ConditionFuser(nn.Module):
    def __init__(self, config: ConditionFuserConfig) -> None:
        super().__init__()
        self.config = config
        self.prior_to_token = nn.Linear(config.prior_dim, config.token_dim)
        self.species_mix_to_token = nn.Linear(config.species_mix_dim, config.token_dim)
        self.null_tokens = nn.Parameter(torch.zeros(config.n_null_tokens, config.token_dim))
        nn.init.normal_(self.null_tokens, mean=0.0, std=0.02)

    def forward(
        self,
        loc_tokens: torch.Tensor,
        z_prior: torch.Tensor,
        *,
        z_species_mix: torch.Tensor | None = None,
        species_mix_alpha: float | None = None,
    ) -> torch.Tensor:
        prior_token = self.prior_to_token(z_prior).unsqueeze(1)
        if z_species_mix is None:
            hybrid_token = prior_token
        else:
            alpha = self.config.species_mix_alpha if species_mix_alpha is None else float(species_mix_alpha)
            alpha = min(max(alpha, 0.0), 1.0)
            species_token = self.species_mix_to_token(z_species_mix).unsqueeze(1)
            hybrid_token = ((1.0 - alpha) * prior_token) + (alpha * species_token)
        return torch.cat((loc_tokens, hybrid_token), dim=1)

    def make_null_condition(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.null_tokens.unsqueeze(0).expand(batch_size, -1, -1).to(device)
