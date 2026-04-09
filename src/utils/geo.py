"""Geographic normalization helpers for butterfly coordinate conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class GeoBounds:
    """Inclusive latitude/longitude bounds used for normalization."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self) -> None:
        if self.lat_min >= self.lat_max:
            raise ValueError("lat_min must be smaller than lat_max.")
        if self.lon_min >= self.lon_max:
            raise ValueError("lon_min must be smaller than lon_max.")


DEFAULT_US_BOUNDS = GeoBounds(
    lat_min=18.5,
    lat_max=72.5,
    lon_min=-180.0,
    lon_max=-66.0,
)


def _as_tensor(latlon: torch.Tensor | Sequence[float] | Iterable[Sequence[float]]) -> torch.Tensor:
    if isinstance(latlon, torch.Tensor):
        return latlon
    return torch.as_tensor(latlon, dtype=torch.float32)


def normalize_latlon(
    latlon: torch.Tensor | Sequence[float] | Iterable[Sequence[float]],
    bounds: GeoBounds = DEFAULT_US_BOUNDS,
    *,
    clamp: bool = True,
) -> torch.Tensor:
    """Normalize latitude/longitude pairs into the [0, 1] interval."""

    latlon_tensor = _as_tensor(latlon).to(dtype=torch.float32)
    if latlon_tensor.shape[-1] != 2:
        raise ValueError("latlon must have a final dimension of size 2.")

    lat = latlon_tensor[..., 0]
    lon = latlon_tensor[..., 1]

    if clamp:
        lat = lat.clamp(bounds.lat_min, bounds.lat_max)
        lon = lon.clamp(bounds.lon_min, bounds.lon_max)

    lat_norm = (lat - bounds.lat_min) / (bounds.lat_max - bounds.lat_min)
    lon_norm = (lon - bounds.lon_min) / (bounds.lon_max - bounds.lon_min)
    return torch.stack((lat_norm, lon_norm), dim=-1)


def denormalize_latlon(
    latlon_norm: torch.Tensor | Sequence[float] | Iterable[Sequence[float]],
    bounds: GeoBounds = DEFAULT_US_BOUNDS,
) -> torch.Tensor:
    """Invert ``normalize_latlon`` back to decimal degrees."""

    latlon_tensor = _as_tensor(latlon_norm).to(dtype=torch.float32)
    if latlon_tensor.shape[-1] != 2:
        raise ValueError("latlon_norm must have a final dimension of size 2.")

    lat = latlon_tensor[..., 0] * (bounds.lat_max - bounds.lat_min) + bounds.lat_min
    lon = latlon_tensor[..., 1] * (bounds.lon_max - bounds.lon_min) + bounds.lon_min
    return torch.stack((lat, lon), dim=-1)


def has_valid_latlon(latitude: object, longitude: object) -> bool:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return False
    return torch.isfinite(torch.tensor([lat, lon])).all().item()

