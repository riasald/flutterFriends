"""Dataset helpers for butterfly image, coordinate, and species training.

Implementation notes:
- Uses ``torch.utils.data.Dataset`` from the official PyTorch data API.
- Uses ``torchvision.transforms.v2`` when available, following the current
  TorchVision recommendation for tensor-first transforms.

Primary references:
- https://pytorch.org/docs/stable/data.html
- https://docs.pytorch.org/vision/stable/transforms.html
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.geo import DEFAULT_US_BOUNDS, GeoBounds, has_valid_latlon, normalize_latlon

try:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import v2 as transforms_v2
except ImportError:  # pragma: no cover
    InterpolationMode = None
    transforms_v2 = None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        return [dict(row) for row in csv.DictReader(file_handle)]


def _resolve_existing_path(
    row: Dict[str, Any],
    columns: Iterable[str],
    *,
    base_dir: Optional[Path] = None,
) -> str:
    search_roots: List[Path] = []
    if base_dir is not None:
        search_roots.append(base_dir)
        if base_dir.parent != base_dir:
            search_roots.append(base_dir.parent)
    for column in columns:
        value = normalize_text(row.get(column))
        if not value:
            continue
        candidate = Path(value)
        if candidate.exists():
            return str(candidate)
        for root in search_roots:
            relative_candidate = (root / candidate).resolve()
            if relative_candidate.exists():
                return str(relative_candidate)
    for column in columns:
        value = normalize_text(row.get(column))
        if value:
            if search_roots:
                candidate = Path(value)
                if not candidate.is_absolute():
                    return str((search_roots[0] / candidate).resolve())
            return value
    return ""


def build_image_transform(
    *,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.nn.Module:
    if transforms_v2 is None or InterpolationMode is None:
        raise RuntimeError("torchvision with transforms.v2 support is required for ButterflyDataset.")

    return transforms_v2.Compose(
        [
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.uint8, scale=True),
            transforms_v2.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_mask_transform(*, image_size: int) -> torch.nn.Module:
    if transforms_v2 is None or InterpolationMode is None:
        raise RuntimeError("torchvision with transforms.v2 support is required for ButterflyDataset.")

    return transforms_v2.Compose(
        [
            transforms_v2.ToImage(),
            transforms_v2.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
            transforms_v2.ToDtype(torch.float32, scale=True),
        ]
    )


@dataclass(frozen=True)
class ButterflyDatasetConfig:
    csv_path: str
    image_size: int = 256
    use_masks: bool = False
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    geo_bounds: GeoBounds = DEFAULT_US_BOUNDS


class ButterflyDataset(Dataset[Dict[str, Any]]):
    """Map-style dataset for butterfly images plus geographic metadata."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        image_size: int = 256,
        use_masks: bool = False,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
        geo_bounds: GeoBounds = DEFAULT_US_BOUNDS,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.use_masks = use_masks
        self.geo_bounds = geo_bounds
        self.rows = self._filter_rows(read_rows(self.csv_path))
        self.csv_dir = self.csv_path.parent
        self.image_transform = build_image_transform(image_size=image_size, mean=mean, std=std)
        self.mask_transform = build_mask_transform(image_size=image_size)

        self.image_column_candidates = (
            "processed_image_absolute_path",
            "processed_image_path",
            "image_path",
            "source_image_path",
            "local_image_path",
        )
        self.mask_column_candidates = (
            "processed_mask_absolute_path",
            "processed_mask_path",
            "mask_path",
        )

        self.species_ids = [int(row["species_id"]) for row in self.rows]

    def _filter_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for row in rows:
            preprocess_status = normalize_text(row.get("preprocess_status"))
            if preprocess_status and preprocess_status != "ok":
                continue
            keep_for_training = normalize_text(row.get("keep_for_training")).lower()
            if keep_for_training and keep_for_training != "true":
                continue
            filtered.append(row)
        return filtered

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, path: str) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def _load_mask(self, path: str, image_size: Tuple[int, int]) -> Image.Image:
        if path and Path(path).exists():
            with Image.open(path) as mask:
                return mask.convert("L")
        return Image.new("L", image_size, color=255)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        image_path = _resolve_existing_path(row, self.image_column_candidates, base_dir=self.csv_dir)
        if not image_path:
            raise FileNotFoundError(f"No usable image path found for row {index} in {self.csv_path}.")

        image = self._load_image(image_path)
        image_tensor = self.image_transform(image)

        if has_valid_latlon(row.get("latitude"), row.get("longitude")):
            latlon = torch.tensor(
                [float(row["latitude"]), float(row["longitude"])],
                dtype=torch.float32,
            )
            latlon_norm = normalize_latlon(latlon, self.geo_bounds, clamp=True)
            has_coordinates = torch.tensor(True, dtype=torch.bool)
        else:
            latlon = torch.zeros(2, dtype=torch.float32)
            latlon_norm = torch.zeros(2, dtype=torch.float32)
            has_coordinates = torch.tensor(False, dtype=torch.bool)

        sample: Dict[str, Any] = {
            "image": image_tensor,
            "latlon": latlon,
            "latlon_norm": latlon_norm,
            "has_coordinates": has_coordinates,
            "species_id": torch.tensor(int(row["species_id"]), dtype=torch.long),
            "image_path": image_path,
            "species": normalize_text(row.get("species")),
            "record_id": normalize_text(row.get("record_id")),
            "source": normalize_text(row.get("source")),
            "specimen_key": normalize_text(row.get("specimen_key")),
        }

        if self.use_masks:
            mask_path = _resolve_existing_path(row, self.mask_column_candidates, base_dir=self.csv_dir)
            mask = self._load_mask(mask_path, image.size)
            mask_tensor = self.mask_transform(mask)
            if mask_tensor.ndim == 3 and mask_tensor.shape[0] > 1:
                mask_tensor = mask_tensor[:1]
            sample["mask"] = mask_tensor
            sample["mask_path"] = mask_path

        return sample
