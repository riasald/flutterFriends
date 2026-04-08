#!/usr/bin/env python3
"""Evaluate the coordinate-conditioned species prior checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from src.data.coordinate_dataset import CoordinateSpeciesDataset
from src.losses.prior_losses import LDAMLoss
from src.train.train_prior import (
    build_geo_bounds,
    build_loader,
    build_model,
    class_counts_from_dataset,
    evaluate_model,
    resolve_amp_dtype,
    resolve_device,
)
from src.utils.config import load_yaml_config


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the location encoder + species prior.")
    parser.add_argument("--config", required=True, help="Path to prior YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path produced by train_prior.py.")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Evaluation split.")
    parser.add_argument("--device", default="auto", help="Device string, for example 'cuda' or 'cpu'.")
    parser.add_argument("--max-batches", type=int, default=0, help="Optional cap for debugging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    data_cfg = load_yaml_config(cfg["data_config"])
    device = resolve_device(args.device)
    bounds = build_geo_bounds(data_cfg)

    train_dataset = CoordinateSpeciesDataset(data_cfg["train_csv"], geo_bounds=bounds)
    eval_csv = data_cfg[f"{args.split}_csv"]
    eval_dataset = CoordinateSpeciesDataset(eval_csv, geo_bounds=bounds)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    species_to_id = checkpoint.get("species_to_id")
    if not species_to_id:
        species_to_id = json.loads(Path(data_cfg["species_vocab_json"]).read_text(encoding="utf-8"))
    species_to_id = {str(key): int(value) for key, value in species_to_id.items()}
    id_to_species = {value: key for key, value in species_to_id.items()}
    num_species = len(species_to_id)

    model = build_model(cfg, num_species, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    train_class_counts = checkpoint.get("train_class_counts")
    if train_class_counts is None:
        train_class_counts = class_counts_from_dataset(train_dataset, num_species=num_species)
    train_class_counts = torch.as_tensor(train_class_counts, dtype=torch.long)

    dataloader = build_loader(
        eval_dataset,
        batch_size=int(data_cfg.get("eval_batch_size", data_cfg.get("batch_size", 256))),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
    )

    loss_fn = LDAMLoss(
        train_class_counts,
        max_margin=float(cfg["loss"].get("max_margin", 0.5)),
        scale=float(cfg["loss"].get("scale", 30.0)),
        class_weight_strategy=str(cfg["loss"].get("class_weight_strategy", "effective_num")),
        class_weight_beta=float(cfg["loss"].get("class_weight_beta", 0.9999)),
    ).to(device)

    autocast_enabled = device.type == "cuda" and bool(cfg.get("training", {}).get("mixed_precision", True))
    autocast_dtype = resolve_amp_dtype(cfg.get("training", {}).get("amp_dtype", "float16"))
    result = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        loss_fn=loss_fn,
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
        max_batches=int(args.max_batches),
        train_class_counts=train_class_counts,
        id_to_species=id_to_species,
    )

    output_dir = Path(cfg["output_dir"]) / "eval" / args.split
    write_json(
        output_dir / "metrics.json",
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "split": args.split,
            "metrics": result["metrics"],
            "frequency_band_summary": result["frequency_band_summary"],
        },
    )
    write_json(output_dir / "per_species_metrics.json", result["per_species_metrics"])
    write_json(output_dir / "frequency_band_summary.json", result["frequency_band_summary"])
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
