#!/usr/bin/env python3
"""Preflight validator for the VAE data and training stack."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from src.models.autoencoder import AutoencoderConfig, ButterflyAutoencoder
from src.train.train_autoencoder import (
    build_data_module,
    build_optimizer,
    build_perceptual_loss,
    configure_runtime,
    forward_and_loss,
    maybe_channels_last,
    raw_model,
    resolve_amp_dtype,
    resolve_device,
)
from src.utils.config import load_yaml_config


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


def resolve_image_path(row: Dict[str, Any], *, base_dir: Optional[Path] = None) -> str:
    search_roots: List[Path] = []
    if base_dir is not None:
        search_roots.append(base_dir)
        if base_dir.parent != base_dir:
            search_roots.append(base_dir.parent)
    for column in ("processed_image_path", "image_path", "source_image_path", "local_image_path"):
        value = normalize_text(row.get(column))
        if not value:
            continue
        candidate = Path(value)
        if candidate.exists():
            return str(candidate)
        if not candidate.is_absolute():
            for root in search_roots:
                resolved = (root / candidate).resolve()
                if resolved.exists():
                    return str(resolved)
        return value
    return ""


def check_manifest_paths(csv_path: Path, max_rows: int) -> Dict[str, Any]:
    rows = read_rows(csv_path, max_rows=max_rows)
    missing = []
    ok = 0
    for row in rows:
        image_path = resolve_image_path(row, base_dir=csv_path.parent)
        if image_path and Path(image_path).exists():
            ok += 1
        else:
            missing.append(image_path or "<missing-path-column>")
    return {
        "csv_path": str(csv_path),
        "rows_checked": len(rows),
        "existing_paths": ok,
        "missing_paths": len(missing),
        "missing_examples": missing[:10],
    }


def apply_data_overrides(autoencoder_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(autoencoder_cfg)
    if args.train_csv:
        cfg["train_csv"] = args.train_csv
    if args.val_csv:
        cfg["val_csv"] = args.val_csv
    elif args.train_csv and "val_csv" not in cfg:
        cfg["val_csv"] = args.train_csv
    if args.test_csv:
        cfg["test_csv"] = args.test_csv
    elif args.train_csv and "test_csv" not in cfg:
        cfg["test_csv"] = args.train_csv
    return cfg


def build_data_cfg_preview(autoencoder_cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = load_yaml_config(autoencoder_cfg["data_config"])
    merged = dict(data_cfg)
    for key in ("train_csv", "val_csv", "test_csv"):
        if key in autoencoder_cfg:
            merged[key] = autoencoder_cfg[key]
    return merged


def save_preflight_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": raw_model(model).state_dict()}, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the butterfly VAE pipeline before long training runs.")
    parser.add_argument("--config", required=True, help="Path to the autoencoder YAML config.")
    parser.add_argument("--device", default="auto", help="Device string, for example 'cuda' or 'cpu'.")
    parser.add_argument("--train-csv", default="", help="Optional override for the train CSV.")
    parser.add_argument("--val-csv", default="", help="Optional override for the val CSV.")
    parser.add_argument("--test-csv", default="", help="Optional override for the test CSV.")
    parser.add_argument("--max-path-checks", type=int, default=64, help="Rows per split to verify on disk.")
    parser.add_argument("--max-train-batches", type=int, default=1, help="Train batches to exercise.")
    parser.add_argument("--max-val-batches", type=int, default=1, help="Validation batches to exercise.")
    parser.add_argument("--output-json", default="", help="Optional path for the preflight report.")
    parser.add_argument("--strict-perceptual", action="store_true", help="Fail if perceptual loss init fails.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_data_overrides(load_yaml_config(args.config), args)
    data_cfg_preview = build_data_cfg_preview(cfg)

    train_csv = Path(data_cfg_preview["train_csv"])
    val_csv = Path(data_cfg_preview["val_csv"])
    test_csv = Path(data_cfg_preview["test_csv"])
    image_stats_json = normalize_text(data_cfg_preview.get("image_stats_json"))

    for path in (train_csv, val_csv, test_csv):
        if not path.exists():
            raise FileNotFoundError(f"Required split CSV does not exist: {path}")

    device = resolve_device(args.device)
    training_cfg = dict(cfg.get("training", {}))
    configure_runtime(training_cfg, device)

    report: Dict[str, Any] = {
        "config": str(Path(args.config)),
        "device": str(device),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "image_stats_json": image_stats_json,
        "image_stats_exists": bool(image_stats_json and Path(image_stats_json).exists()),
        "warnings": [],
    }

    if image_stats_json and not Path(image_stats_json).exists():
        report["warnings"].append(f"image_stats_json_missing:{image_stats_json}")

    report["path_checks"] = {
        "train": check_manifest_paths(train_csv, args.max_path_checks),
        "val": check_manifest_paths(val_csv, args.max_path_checks),
        "test": check_manifest_paths(test_csv, args.max_path_checks),
    }

    autoencoder_cfg = AutoencoderConfig(**cfg["model"])
    model = ButterflyAutoencoder(autoencoder_cfg).to(device)
    channels_last = bool(training_cfg.get("channels_last", device.type == "cuda"))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    perceptual_loss_fn = None
    perceptual_ready = True
    try:
        perceptual_loss_fn = build_perceptual_loss(cfg["loss"], device)
    except Exception as exc:  # pragma: no cover - environment dependent
        perceptual_ready = False
        report["warnings"].append(f"perceptual_init_failed:{exc}")
        if args.strict_perceptual:
            raise

    report["perceptual_ready"] = perceptual_ready
    report["perceptual_enabled"] = float(cfg["loss"].get("perceptual_weight", 0.0)) > 0.0

    data_module = build_data_module(cfg)
    summary = data_module.summary()
    report["dataset_summary"] = summary

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    first_train_batch = next(iter(train_loader))
    first_val_batch = next(iter(val_loader))

    report["train_batch"] = {
        "shape": list(first_train_batch["image"].shape),
        "dtype": str(first_train_batch["image"].dtype),
        "min": float(first_train_batch["image"].min().item()),
        "max": float(first_train_batch["image"].max().item()),
        "has_nan": bool(torch.isnan(first_train_batch["image"]).any().item()),
    }
    report["val_batch"] = {
        "shape": list(first_val_batch["image"].shape),
        "dtype": str(first_val_batch["image"].dtype),
        "min": float(first_val_batch["image"].min().item()),
        "max": float(first_val_batch["image"].max().item()),
        "has_nan": bool(torch.isnan(first_val_batch["image"]).any().item()),
    }

    optimizer = build_optimizer(model, cfg["optimizer"], device)
    mixed_precision = bool(training_cfg.get("mixed_precision", device.type == "cuda"))
    autocast_dtype = resolve_amp_dtype(training_cfg.get("amp_dtype", "float16"))
    autocast_enabled = mixed_precision and device.type == "cuda" and autocast_dtype != torch.float32

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_loss, train_metrics, _ = forward_and_loss(
        model=model,
        batch=first_train_batch,
        device=device,
        loss_cfg=cfg["loss"],
        sample_posterior=bool(training_cfg.get("sample_posterior", True)),
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
        channels_last=channels_last,
        perceptual_loss_fn=perceptual_loss_fn,
    )
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model.eval()
    with torch.no_grad():
        _, val_metrics, val_reconstruction = forward_and_loss(
            model=model,
            batch=first_val_batch,
            device=device,
            loss_cfg=cfg["loss"],
            sample_posterior=bool(training_cfg.get("validation_sample_posterior", False)),
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
            channels_last=channels_last,
            perceptual_loss_fn=perceptual_loss_fn,
        )

    report["smoke_metrics"] = {
        "train": train_metrics,
        "val": val_metrics,
        "reconstruction_shape": list(val_reconstruction.shape),
    }

    checkpoint_dir = Path(cfg.get("checkpoint_dir", Path(args.config).parent / "preflight"))
    smoke_checkpoint = checkpoint_dir / "preflight_smoke.pt"
    save_preflight_checkpoint(model, smoke_checkpoint)
    report["smoke_checkpoint"] = str(smoke_checkpoint)
    report["status"] = "ok"

    output_json = Path(args.output_json) if args.output_json else checkpoint_dir / "preflight_report.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
