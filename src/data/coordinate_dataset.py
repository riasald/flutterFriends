"""Coordinate-only dataset for location encoder and species prior training."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from src.utils.geo import DEFAULT_US_BOUNDS, GeoBounds, has_valid_latlon, normalize_latlon


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        return [dict(row) for row in csv.DictReader(file_handle)]


@dataclass(frozen=True)
class CoordinateSpeciesDatasetConfig:
    csv_path: str
    geo_bounds: GeoBounds = DEFAULT_US_BOUNDS
    require_coordinates: bool = True


class CoordinateSpeciesDataset(Dataset[Dict[str, Any]]):
    """Map-style dataset with normalized coordinates and species ids only."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        geo_bounds: GeoBounds = DEFAULT_US_BOUNDS,
        require_coordinates: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.geo_bounds = geo_bounds
        self.require_coordinates = require_coordinates
        self.rows = self._filter_rows(read_rows(self.csv_path))
        self.species_ids = [int(row["species_id"]) for row in self.rows]

    def _filter_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for row in rows:
            if not normalize_text(row.get("species")):
                continue
            if normalize_text(row.get("species_id")) == "":
                continue
            if self.require_coordinates and not has_valid_latlon(row.get("latitude"), row.get("longitude")):
                continue
            filtered.append(row)
        return filtered

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        latlon = torch.tensor([lat, lon], dtype=torch.float32)
        latlon_norm = normalize_latlon(latlon, self.geo_bounds, clamp=True)
        return {
            "latlon": latlon,
            "latlon_norm": latlon_norm,
            "species_id": torch.tensor(int(row["species_id"]), dtype=torch.long),
            "species": normalize_text(row.get("species")),
            "record_id": normalize_text(row.get("record_id")),
            "specimen_key": normalize_text(row.get("specimen_key")),
            "source": normalize_text(row.get("source")),
        }
