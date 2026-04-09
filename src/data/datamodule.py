"""Plain-PyTorch datamodule-style loader wrapper for butterfly experiments.

Primary references:
- https://pytorch.org/docs/stable/data.html
- https://docs.pytorch.org/vision/stable/transforms.html
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import ButterflyDataset
from src.utils.config import load_yaml_config
from src.utils.geo import GeoBounds


@dataclass(frozen=True)
class ButterflyDataModuleConfig:
    train_csv: str
    val_csv: str
    test_csv: str
    image_size: int = 256
    use_masks: bool = False
    batch_size: int = 32
    eval_batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last_train: bool = True
    weighted_sampling: bool = True
    image_mean: Sequence[float] = (0.5, 0.5, 0.5)
    image_std: Sequence[float] = (0.5, 0.5, 0.5)
    image_stats_json: str = ""
    prefetch_factor: int = 2
    lat_min: float = 18.5
    lat_max: float = 72.5
    lon_min: float = -180.0
    lon_max: float = -66.0

    @property
    def geo_bounds(self) -> GeoBounds:
        return GeoBounds(
            lat_min=self.lat_min,
            lat_max=self.lat_max,
            lon_min=self.lon_min,
            lon_max=self.lon_max,
        )


class ButterflyDataModule:
    """Small helper object that builds datasets and data loaders."""

    def __init__(self, config: ButterflyDataModuleConfig) -> None:
        self.config = config
        self.train_dataset: Optional[ButterflyDataset] = None
        self.val_dataset: Optional[ButterflyDataset] = None
        self.test_dataset: Optional[ButterflyDataset] = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ButterflyDataModule":
        config_dict = load_yaml_config(config_path)
        return cls(ButterflyDataModuleConfig(**config_dict))

    def setup(self) -> None:
        image_mean = tuple(self.config.image_mean)
        image_std = tuple(self.config.image_std)
        if self.config.image_stats_json and Path(self.config.image_stats_json).exists():
            stats = load_yaml_config(self.config.image_stats_json)
            stats_mean = stats.get("mean")
            stats_std = stats.get("std")
            if stats_mean and stats_std:
                image_mean = tuple(float(value) for value in stats_mean)
                image_std = tuple(float(value) for value in stats_std)

        dataset_kwargs = dict(
            image_size=self.config.image_size,
            use_masks=self.config.use_masks,
            mean=image_mean,
            std=image_std,
            geo_bounds=self.config.geo_bounds,
        )
        self.train_dataset = ButterflyDataset(self.config.train_csv, **dataset_kwargs)
        self.val_dataset = ButterflyDataset(self.config.val_csv, **dataset_kwargs)
        self.test_dataset = ButterflyDataset(self.config.test_csv, **dataset_kwargs)

    def _persistent_workers(self) -> bool:
        return self.config.persistent_workers and self.config.num_workers > 0

    def _prefetch_factor(self) -> Optional[int]:
        if self.config.num_workers <= 0:
            return None
        return max(2, int(self.config.prefetch_factor))

    def _build_weighted_sampler(self) -> Optional[WeightedRandomSampler]:
        if not self.config.weighted_sampling:
            return None
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting a train sampler.")
        if len(self.train_dataset) == 0:
            return None

        class_counts = Counter(self.train_dataset.species_ids)
        sample_weights = torch.tensor(
            [1.0 / class_counts[species_id] for species_id in self.train_dataset.species_ids],
            dtype=torch.double,
        )
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting train_dataloader().")
        sampler = self._build_weighted_sampler()
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self._persistent_workers(),
            prefetch_factor=self._prefetch_factor(),
            drop_last=self.config.drop_last_train,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting val_dataloader().")
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self._persistent_workers(),
            prefetch_factor=self._prefetch_factor(),
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before requesting test_dataloader().")
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self._persistent_workers(),
            prefetch_factor=self._prefetch_factor(),
            drop_last=False,
        )

    def summary(self) -> Dict[str, Any]:
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            raise RuntimeError("Call setup() before requesting summary().")
        return {
            "train_rows": len(self.train_dataset),
            "val_rows": len(self.val_dataset),
            "test_rows": len(self.test_dataset),
            "train_species": len(set(self.train_dataset.species_ids)),
            "val_species": len(set(self.val_dataset.species_ids)),
            "test_species": len(set(self.test_dataset.species_ids)),
        }
