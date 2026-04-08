"""Species prior head that maps location embeddings to species logits and prior embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.models.fourier_encoder import FourierLocationEncoder, FourierLocationEncoderConfig


@dataclass(frozen=True)
class SpeciesPriorHeadConfig:
    in_dim: int = 256
    hidden_dim: int = 256
    num_species: int = 371
    prior_dim: int = 128
    dropout: float = 0.1
    normalized_classifier: bool = True
    classifier_bias: bool = False


class NormalizedLinear(nn.Module):
    """Cosine-similarity classifier used by margin-based long-tail losses."""

    def __init__(self, in_dim: int, out_dim: int, *, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.normalize(self.weight, dim=1)
        x = F.normalize(x, dim=1)
        logits = F.linear(x, weight, self.bias)
        return logits


class SpeciesPriorHead(nn.Module):
    def __init__(self, config: SpeciesPriorHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = nn.Sequential(
            nn.Linear(config.in_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        if config.normalized_classifier:
            self.classifier = NormalizedLinear(
                config.hidden_dim,
                config.num_species,
                bias=config.classifier_bias,
            )
        else:
            self.classifier = nn.Linear(config.hidden_dim, config.num_species, bias=config.classifier_bias)
        self.prior_proj = nn.Linear(config.hidden_dim, config.prior_dim)

    def forward(self, z_loc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(z_loc)
        logits = self.classifier(hidden)
        z_prior = self.prior_proj(hidden)
        return logits, z_prior

    def species_prototypes(self) -> torch.Tensor:
        """Return classifier-space species prototypes in the hidden feature space."""

        weight = self.classifier.weight
        if isinstance(self.classifier, NormalizedLinear):
            return F.normalize(weight, dim=1)
        return weight

    def species_mixture(
        self,
        logits: torch.Tensor,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a soft species mixture from the prior logits.

        This gives downstream models a locally plausible species blend rather
        than a hard one-hot species identity. The mixed hidden prototype lives
        in the same space used by the classifier and can be projected into the
        learned prior embedding space.
        """

        temperature = max(float(temperature), 1.0e-4)
        scaled_logits = logits / temperature
        if top_k > 0 and top_k < scaled_logits.shape[-1]:
            top_values, top_indices = scaled_logits.topk(top_k, dim=-1)
            masked_logits = torch.full_like(scaled_logits, float("-inf"))
            masked_logits.scatter_(1, top_indices, top_values)
            scaled_logits = masked_logits
        species_probs = torch.softmax(scaled_logits, dim=-1)
        prototypes = self.species_prototypes()
        mixed_hidden = species_probs @ prototypes
        z_species_mix = self.prior_proj(mixed_hidden)
        return species_probs, mixed_hidden, z_species_mix


class LocationSpeciesPriorModel(nn.Module):
    """Convenience wrapper that keeps the location encoder and species head together."""

    def __init__(
        self,
        location_encoder_config: FourierLocationEncoderConfig,
        prior_head_config: SpeciesPriorHeadConfig,
    ) -> None:
        super().__init__()
        self.location_encoder = FourierLocationEncoder(location_encoder_config)
        self.prior_head = SpeciesPriorHead(prior_head_config)

    def forward(
        self,
        latlon_norm: torch.Tensor,
        *,
        mixture_temperature: float = 1.0,
        mixture_top_k: int = 0,
    ) -> dict[str, torch.Tensor]:
        z_loc, loc_tokens = self.location_encoder(latlon_norm)
        logits, z_prior = self.prior_head(z_loc)
        species_probs, species_hidden_mix, z_species_mix = self.prior_head.species_mixture(
            logits,
            temperature=mixture_temperature,
            top_k=mixture_top_k,
        )
        return {
            "logits": logits,
            "z_loc": z_loc,
            "loc_tokens": loc_tokens,
            "z_prior": z_prior,
            "species_probs": species_probs,
            "species_hidden_mix": species_hidden_mix,
            "z_species_mix": z_species_mix,
        }
