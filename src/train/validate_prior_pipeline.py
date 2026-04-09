#!/usr/bin/env python3
"""Preflight validator for the location encoder + species prior stack."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.data.coordinate_dataset import CoordinateSpeciesDataset
from src.losses.prior_losses import LDAMLoss
from src.train.train_prior import (
    build_geo_bounds,
    build_loader,
    build_model,
    class_counts_from_dataset,
    configure_runtime,
    count_parameters,
    forward_loss,
    load_species_vocab,
    resolve_amp_dtype,
    resolve_device,
)
from src.utils.config import load_yaml_config
from src.utils.geo import has_valid_latlon


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def read_rows(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        rows: List[Dict[str, Any]] = []
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            rows.append(dict(row))
    return rows


def check_split_rows(csv_path: Path, *, max_rows: int, species_to_id: dict[str, int]) -> Dict[str, Any]:
    rows = read_rows(csv_path, max_rows=max_rows)
    missing_coordinates = 0
    missing_species = 0
    unknown_species = 0
    bad_species_ids = 0
    invalid_examples: List[Dict[str, Any]] = []

    valid_species_ids = set(species_to_id.values())
    for row in rows:
        species = normalize_text(row.get("species"))
        species_id_raw = normalize_text(row.get("species_id"))
        species_id: Optional[int]
        try:
            species_id = int(species_id_raw) if species_id_raw != "" else None
        except ValueError:
            species_id = None
        if not has_valid_latlon(row.get("latitude"), row.get("longitude")):
            missing_coordinates += 1
            invalid_examples.append({"reason": "invalid_coordinates", "record_id": normalize_text(row.get("record_id"))})
        if not species:
            missing_species += 1
            invalid_examples.append({"reason": "missing_species", "record_id": normalize_text(row.get("record_id"))})
        elif species not in species_to_id:
            unknown_species += 1
            invalid_examples.append({"reason": "unknown_species", "species": species})
        if species_id is None or species_id not in valid_species_ids:
            bad_species_ids += 1
            invalid_examples.append({"reason": "invalid_species_id", "species_id": species_id_raw})

    return {
        "csv_path": str(csv_path),
        "rows_checked": len(rows),
        "missing_coordinates": int(missing_coordinates),
        "missing_species": int(missing_species),
        "unknown_species": int(unknown_species),
        "invalid_species_ids": int(bad_species_ids),
        "invalid_examples": invalid_examples[:10],
    }


def summarize_dataset(dataset: CoordinateSpeciesDataset) -> Dict[str, Any]:
    species_counts = Counter(int(species_id) for species_id in dataset.species_ids)
    return {
        "rows": len(dataset),
        "species": len(species_counts),
        "class_count_min": int(min(species_counts.values())) if species_counts else 0,
        "class_count_max": int(max(species_counts.values())) if species_counts else 0,
    }


def save_preflight_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the coordinate prior pipeline before long runs.")
    parser.add_argument("--config", required=True, help="Path to the prior YAML config.")
    parser.add_argument("--device", default="auto", help="Device string, for example 'cuda' or 'cpu'.")
    parser.add_argument("--max-path-checks", type=int, default=128, help="Rows per split to verify.")
    parser.add_argument("--max-train-batches", type=int, default=1, help="Train batches to exercise.")
    parser.add_argument("--max-val-batches", type=int, default=1, help="Validation batches to exercise.")
    parser.add_argument("--output-json", default="", help="Optional path for the preflight report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    data_cfg = load_yaml_config(cfg["data_config"])

    train_csv = Path(data_cfg["train_csv"])
    val_csv = Path(data_cfg["val_csv"])
    test_csv = Path(data_cfg["test_csv"])
    species_vocab_json = Path(data_cfg["species_vocab_json"])

    for path in (train_csv, val_csv, test_csv, species_vocab_json):
        if not path.exists():
            raise FileNotFoundError(f"Required prior artifact does not exist: {path}")

    species_to_id, id_to_species = load_species_vocab(species_vocab_json)
    device = resolve_device(args.device)
    training_cfg = dict(cfg.get("training", {}))
    configure_runtime(training_cfg, device)
    bounds = build_geo_bounds(data_cfg)

    report: Dict[str, Any] = {
        "config": str(Path(args.config).resolve()),
        "device": str(device),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "species_vocab_json": str(species_vocab_json),
        "num_species": len(species_to_id),
    }

    report["split_checks"] = {
        "train": check_split_rows(train_csv, max_rows=args.max_path_checks, species_to_id=species_to_id),
        "val": check_split_rows(val_csv, max_rows=args.max_path_checks, species_to_id=species_to_id),
        "test": check_split_rows(test_csv, max_rows=args.max_path_checks, species_to_id=species_to_id),
    }

    train_dataset = CoordinateSpeciesDataset(train_csv, geo_bounds=bounds)
    val_dataset = CoordinateSpeciesDataset(val_csv, geo_bounds=bounds)
    test_dataset = CoordinateSpeciesDataset(test_csv, geo_bounds=bounds)
    report["dataset_summary"] = {
        "train": summarize_dataset(train_dataset),
        "val": summarize_dataset(val_dataset),
        "test": summarize_dataset(test_dataset),
    }

    train_loader = build_loader(
        train_dataset,
        batch_size=int(data_cfg.get("batch_size", 256)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=int(data_cfg.get("eval_batch_size", data_cfg.get("batch_size", 256))),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
    )

    first_train_batch = next(iter(train_loader))
    first_val_batch = next(iter(val_loader))
    report["train_batch"] = {
        "shape": list(first_train_batch["latlon_norm"].shape),
        "dtype": str(first_train_batch["latlon_norm"].dtype),
        "min": float(first_train_batch["latlon_norm"].amin().item()),
        "max": float(first_train_batch["latlon_norm"].amax().item()),
        "has_nan": bool(torch.isnan(first_train_batch["latlon_norm"]).any().item()),
    }
    report["val_batch"] = {
        "shape": list(first_val_batch["latlon_norm"].shape),
        "dtype": str(first_val_batch["latlon_norm"].dtype),
        "min": float(first_val_batch["latlon_norm"].amin().item()),
        "max": float(first_val_batch["latlon_norm"].amax().item()),
        "has_nan": bool(torch.isnan(first_val_batch["latlon_norm"]).any().item()),
    }

    num_species = len(species_to_id)
    model = build_model(cfg, num_species, device)
    report["parameter_counts"] = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optimizer"]["learning_rate"]),
        betas=tuple(cfg["optimizer"].get("betas", (0.9, 0.999))),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
    )
    train_class_counts = class_counts_from_dataset(train_dataset, num_species=num_species)
    loss_fn = LDAMLoss(
        train_class_counts,
        max_margin=float(cfg["loss"].get("max_margin", 0.5)),
        scale=float(cfg["loss"].get("scale", 30.0)),
        class_weight_strategy=str(cfg["loss"].get("class_weight_strategy", "effective_num")),
        class_weight_beta=float(cfg["loss"].get("class_weight_beta", 0.9999)),
    ).to(device)

    autocast_dtype = resolve_amp_dtype(training_cfg.get("amp_dtype", "float16"))
    autocast_enabled = device.type == "cuda" and bool(training_cfg.get("mixed_precision", True))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_loss, train_payload = forward_loss(
        model=model,
        batch=first_train_batch,
        device=device,
        loss_fn=loss_fn,
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
    )
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model.eval()
    with torch.no_grad():
        val_loss, val_payload = forward_loss(
            model=model,
            batch=first_val_batch,
            device=device,
            loss_fn=loss_fn,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
        )

    report["smoke_metrics"] = {
        "train": {
            "loss": float(train_loss.detach().cpu().item()),
            "top1": float(train_payload["top1"]),
            "top5": float(train_payload["top5"]),
        },
        "val": {
            "loss": float(val_loss.detach().cpu().item()),
            "top1": float(val_payload["top1"]),
            "top5": float(val_payload["top5"]),
        },
    }
    report["id_to_species_preview"] = {
        str(class_id): id_to_species[class_id]
        for class_id in sorted(id_to_species)[:10]
    }

    checkpoint_dir = Path(cfg.get("checkpoint_dir", Path(args.config).parent / "preflight"))
    smoke_checkpoint = checkpoint_dir / "preflight_smoke.pt"
    save_preflight_checkpoint(model, smoke_checkpoint)
    report["smoke_checkpoint"] = str(smoke_checkpoint.resolve())
    report["status"] = "ok"

    output_json = Path(args.output_json) if args.output_json else (Path(cfg["output_dir"]) / "logs" / "preflight_report.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
