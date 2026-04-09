"""Cached latent/conditioning dataset for faster diffusion training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class CachedDiffusionDataset(Dataset[Dict[str, Any]]):
    """Dataset backed by precomputed latents and condition tokens."""

    def __init__(self, cache_path: str | Path) -> None:
        self.cache_path = Path(cache_path)
        payload = torch.load(self.cache_path, map_location="cpu")
        self.latents = payload["latents"]
        self.cond_tokens = payload["cond_tokens"]
        self.latlon_norm = payload["latlon_norm"]
        self.species_id = payload.get("species_id")
        self.rows = payload.get("rows", [])

        if self.latents.shape[0] != self.cond_tokens.shape[0]:
            raise RuntimeError(f"Cache {self.cache_path} has mismatched latent/condition counts.")
        if self.latlon_norm.shape[0] != self.latents.shape[0]:
            raise RuntimeError(f"Cache {self.cache_path} has mismatched latlon/latent counts.")

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample: Dict[str, Any] = {
            "latent_z0": self.latents[index].to(dtype=torch.float32),
            "cond_tokens": self.cond_tokens[index].to(dtype=torch.float32),
            "latlon_norm": self.latlon_norm[index].to(dtype=torch.float32),
        }
        if self.species_id is not None:
            sample["species_id"] = self.species_id[index].to(dtype=torch.long)
        if self.rows:
            sample["row"] = self.rows[index]
        return sample


@dataclass(frozen=True)
class CachedDiffusionDataModuleConfig:
    train_cache_pt: str
    val_cache_pt: str
    test_cache_pt: str
    batch_size: int = 32
    eval_batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last_train: bool = True
    prefetch_factor: int = 2


class CachedDiffusionDataModule:
    def __init__(self, config: CachedDiffusionDataModuleConfig) -> None:
        self.config = config
        self.train_dataset: Optional[CachedDiffusionDataset] = None
        self.val_dataset: Optional[CachedDiffusionDataset] = None
        self.test_dataset: Optional[CachedDiffusionDataset] = None

    def setup(self) -> None:
        self.train_dataset = CachedDiffusionDataset(self.config.train_cache_pt)
        self.val_dataset = CachedDiffusionDataset(self.config.val_cache_pt)
        self.test_dataset = CachedDiffusionDataset(self.config.test_cache_pt)

    def _persistent_workers(self) -> bool:
        return self.config.persistent_workers and self.config.num_workers > 0

    def _prefetch_factor(self) -> Optional[int]:
        if self.config.num_workers <= 0:
            return None
        return max(2, int(self.config.prefetch_factor))

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting train_dataloader().")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
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
        }
