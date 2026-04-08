#!/usr/bin/env python3
"""Precompute VAE latents and prior condition tokens for diffusion training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ButterflyDataset
from src.models.autoencoder import load_autoencoder_checkpoint
from src.train.train_diffusion import build_condition_tokens, configure_runtime, load_prior_bundle, resolve_device
from src.utils.config import load_yaml_config
from src.utils.geo import GeoBounds


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def maybe_limit_rows(rows: List[Dict[str, Any]], max_rows: int) -> List[Dict[str, Any]]:
    if max_rows <= 0:
        return rows
    return rows[:max_rows]


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_geo_bounds(data_cfg: Dict[str, Any]) -> GeoBounds:
    return GeoBounds(
        lat_min=float(data_cfg.get("lat_min", 18.5)),
        lat_max=float(data_cfg.get("lat_max", 72.5)),
        lon_min=float(data_cfg.get("lon_min", -180.0)),
        lon_max=float(data_cfg.get("lon_max", -66.0)),
    )


def build_image_dataset(csv_path: Path, data_cfg: Dict[str, Any]) -> ButterflyDataset:
    image_mean = tuple((0.5, 0.5, 0.5))
    image_std = tuple((0.5, 0.5, 0.5))
    image_stats_json = normalize_text(data_cfg.get("image_stats_json"))
    if image_stats_json and Path(image_stats_json).exists():
        stats = load_yaml_config(image_stats_json)
        if stats.get("mean") and stats.get("std"):
            image_mean = tuple(float(value) for value in stats["mean"])
            image_std = tuple(float(value) for value in stats["std"])
    return ButterflyDataset(
        csv_path,
        image_size=int(data_cfg.get("image_size", 256)),
        use_masks=False,
        mean=image_mean,
        std=image_std,
        geo_bounds=build_geo_bounds(data_cfg),
    )


def build_loader(dataset: ButterflyDataset, *, batch_size: int, num_workers: int, pin_memory: bool, prefetch_factor: int) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
    return DataLoader(**kwargs)


def rows_for_indices(dataset: ButterflyDataset, indices: Sequence[int]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for index in indices:
        row = dict(dataset.rows[int(index)])
        row["processed_image_absolute_path"] = normalize_text(row.get("processed_image_absolute_path")) or normalize_text(row.get("processed_image_path"))
        selected.append(row)
    return selected


def cache_split(
    *,
    split_name: str,
    csv_path: Path,
    output_dir: Path,
    data_cfg: Dict[str, Any],
    cfg: Dict[str, Any],
    vae,
    prior_model,
    fuser,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    max_rows: int,
) -> Dict[str, Any]:
    raw_rows = maybe_limit_rows(read_rows(csv_path), max_rows)
    limited_csv = output_dir / f"{split_name}_source_rows.csv"
    write_rows(limited_csv, raw_rows)

    dataset = build_image_dataset(limited_csv, data_cfg)
    loader = build_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    conditioning_cfg = dict(cfg.get("conditioning", {}))

    latents_chunks: List[torch.Tensor] = []
    cond_chunks: List[torch.Tensor] = []
    latlon_chunks: List[torch.Tensor] = []
    species_chunks: List[torch.Tensor] = []
    selected_rows: List[Dict[str, Any]] = []
    running_index = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            latlon_norm = batch["latlon_norm"].to(device, non_blocking=True)
            latents = vae.encode_for_diffusion(images, sample_posterior=False).detach().to(dtype).cpu()
            cond_tokens = build_condition_tokens(
                prior_model=prior_model,
                fuser=fuser,
                latlon_norm=latlon_norm,
                conditioning_cfg=conditioning_cfg,
            ).detach().to(dtype).cpu()

            latents_chunks.append(latents)
            cond_chunks.append(cond_tokens)
            latlon_chunks.append(batch["latlon_norm"].detach().cpu().to(torch.float32))
            species_chunks.append(batch["species_id"].detach().cpu().to(torch.long))

            batch_size_actual = int(latents.shape[0])
            selected_rows.extend(rows_for_indices(dataset, range(running_index, running_index + batch_size_actual)))
            running_index += batch_size_actual

    payload = {
        "split": split_name,
        "latents": torch.cat(latents_chunks, dim=0),
        "cond_tokens": torch.cat(cond_chunks, dim=0),
        "latlon_norm": torch.cat(latlon_chunks, dim=0),
        "species_id": torch.cat(species_chunks, dim=0),
        "rows": selected_rows,
        "dtype": str(dtype),
        "source_csv": str(csv_path),
    }
    cache_path = output_dir / f"{split_name}.pt"
    torch.save(payload, cache_path)
    return {
        "cache_path": str(cache_path),
        "rows": int(payload["latents"].shape[0]),
        "latent_shape": list(payload["latents"].shape),
        "cond_shape": list(payload["cond_tokens"].shape),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache diffusion latents and conditioning tokens.")
    parser.add_argument("--config", required=True, help="Path to a diffusion YAML config.")
    parser.add_argument("--output-dir", required=True, help="Directory for cached split tensors.")
    parser.add_argument("--device", default="auto", help="Device string, for example 'cuda' or 'cpu'.")
    parser.add_argument("--dtype", default="float16", help="Cache dtype: float16 or float32.")
    parser.add_argument("--batch-size", type=int, default=64, help="Cache-build batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers for cache build.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Prefetch factor when workers > 0.")
    parser.add_argument("--max-rows-per-split", type=int, default=0, help="Optional smoke-test cap per split.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    device = resolve_device(args.device)
    configure_runtime(dict(cfg.get("training", {})), device)
    data_cfg = load_yaml_config(cfg["data_config"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_key = str(args.dtype).strip().lower()
    dtype = {"float16": torch.float16, "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}[dtype_key]

    vae_locked = load_yaml_config(cfg["vae_locked_config"])
    vae = load_autoencoder_checkpoint(
        vae_locked["checkpoint_path"],
        device=device,
        use_ema=bool(vae_locked.get("use_ema", False)),
        freeze=True,
    )
    prior_model, fuser = load_prior_bundle(cfg, device)

    split_paths = {
        "train": Path(data_cfg["train_csv"]),
        "val": Path(data_cfg["val_csv"]),
        "test": Path(data_cfg["test_csv"]),
    }

    summaries: Dict[str, Any] = {}
    for split_name, csv_path in split_paths.items():
        summaries[split_name] = cache_split(
            split_name=split_name,
            csv_path=csv_path,
            output_dir=output_dir,
            data_cfg=data_cfg,
            cfg=cfg,
            vae=vae,
            prior_model=prior_model,
            fuser=fuser,
            device=device,
            dtype=dtype,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            pin_memory=bool(data_cfg.get("pin_memory", True)),
            prefetch_factor=int(args.prefetch_factor),
            max_rows=int(args.max_rows_per_split),
        )

    manifest = {
        "config": str(Path(args.config).resolve()),
        "output_dir": str(output_dir.resolve()),
        "device": str(device),
        "dtype": str(dtype),
        "summaries": summaries,
    }
    (output_dir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
