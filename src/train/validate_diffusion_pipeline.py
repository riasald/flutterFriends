#!/usr/bin/env python3
"""Preflight validator for the latent diffusion training stack."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.data.dataset import _resolve_existing_path, normalize_text
from src.models.autoencoder import load_autoencoder_checkpoint
from src.models.diffusion_schedule import DiffusionSchedule, DiffusionScheduleConfig
from src.models.latent_diffusion_unet import LatentDiffusionUNet, LatentDiffusionUNetConfig
from src.train.train_diffusion import (
    build_datamodule,
    build_optimizer,
    configure_runtime,
    forward_loss,
    load_prior_bundle,
    resolve_amp_dtype,
    resolve_device,
)
from src.utils.config import load_yaml_config


def read_rows(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, Any]] = []
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            rows.append(dict(row))
    return rows


def resolve_image_path(row: Dict[str, Any], *, base_dir: Path) -> str:
    return _resolve_existing_path(
        row,
        (
            "processed_image_absolute_path",
            "processed_image_path",
            "image_path",
            "source_image_path",
            "local_image_path",
        ),
        base_dir=base_dir,
    )


def check_manifest_paths(csv_path: Path, max_rows: int) -> Dict[str, Any]:
    rows = read_rows(csv_path, max_rows=max_rows)
    missing_examples: List[str] = []
    ok = 0
    for row in rows:
        image_path = resolve_image_path(row, base_dir=csv_path.parent)
        if image_path and Path(image_path).exists():
            ok += 1
        else:
            missing_examples.append(image_path or "<missing-path-column>")
    return {
        "csv_path": str(csv_path),
        "rows_checked": len(rows),
        "existing_paths": ok,
        "missing_paths": len(missing_examples),
        "missing_examples": missing_examples[:10],
    }


def save_preflight_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the latent diffusion pipeline before long runs.")
    parser.add_argument("--config", required=True, help="Path to the diffusion YAML config.")
    parser.add_argument("--device", default="auto", help="Device string, for example 'cuda' or 'cpu'.")
    parser.add_argument("--max-path-checks", type=int, default=64, help="Rows per split to verify on disk.")
    parser.add_argument("--max-train-batches", type=int, default=1, help="Train batches to exercise.")
    parser.add_argument("--max-val-batches", type=int, default=1, help="Validation batches to exercise.")
    parser.add_argument("--override-num-workers", type=int, default=-1, help="Optional dataloader worker override for local checks.")
    parser.add_argument("--output-json", default="", help="Optional path for the preflight report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    data_cfg = load_yaml_config(cfg["data_config"])
    if args.override_num_workers >= 0:
        data_cfg["num_workers"] = int(args.override_num_workers)
        if int(args.override_num_workers) == 0:
            data_cfg["persistent_workers"] = False

    using_cached_inputs = {"train_cache_pt", "val_cache_pt", "test_cache_pt"}.issubset(data_cfg.keys())
    train_csv = Path(data_cfg["train_csv"]) if "train_csv" in data_cfg else None
    val_csv = Path(data_cfg["val_csv"]) if "val_csv" in data_cfg else None
    test_csv = Path(data_cfg["test_csv"]) if "test_csv" in data_cfg else None
    image_stats_json = Path(data_cfg["image_stats_json"]) if "image_stats_json" in data_cfg else None
    train_cache_pt = Path(data_cfg["train_cache_pt"]) if "train_cache_pt" in data_cfg else None
    val_cache_pt = Path(data_cfg["val_cache_pt"]) if "val_cache_pt" in data_cfg else None
    test_cache_pt = Path(data_cfg["test_cache_pt"]) if "test_cache_pt" in data_cfg else None
    vae_locked_path = Path(cfg["vae_locked_config"])
    prior_locked_path = Path(cfg["prior_locked_config"])

    required_paths = [vae_locked_path, prior_locked_path]
    if using_cached_inputs:
        required_paths.extend([train_cache_pt, val_cache_pt, test_cache_pt])
    else:
        required_paths.extend([train_csv, val_csv, test_csv, image_stats_json])

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required diffusion artifact does not exist: {path}")

    device = resolve_device(args.device)
    training_cfg = dict(cfg.get("training", {}))
    configure_runtime(training_cfg, device)

    report: Dict[str, Any] = {
        "config": str(Path(args.config).resolve()),
        "device": str(device),
        "using_cached_inputs": using_cached_inputs,
        "vae_locked_config": str(vae_locked_path),
        "prior_locked_config": str(prior_locked_path),
    }

    if using_cached_inputs:
        report["train_cache_pt"] = str(train_cache_pt)
        report["val_cache_pt"] = str(val_cache_pt)
        report["test_cache_pt"] = str(test_cache_pt)
        report["path_checks"] = {
            "train_cache_exists": train_cache_pt.exists(),
            "val_cache_exists": val_cache_pt.exists(),
            "test_cache_exists": test_cache_pt.exists(),
        }
    else:
        report["train_csv"] = str(train_csv)
        report["val_csv"] = str(val_csv)
        report["test_csv"] = str(test_csv)
        report["image_stats_json"] = str(image_stats_json)
        report["path_checks"] = {
            "train": check_manifest_paths(train_csv, args.max_path_checks),
            "val": check_manifest_paths(val_csv, args.max_path_checks),
            "test": check_manifest_paths(test_csv, args.max_path_checks),
        }

    data_module = build_datamodule(data_cfg)
    report["dataset_summary"] = data_module.summary()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    first_train_batch = next(iter(train_loader))
    first_val_batch = next(iter(val_loader))

    if "image" in first_train_batch:
        report["train_batch"] = {
            "image_shape": list(first_train_batch["image"].shape),
            "latlon_shape": list(first_train_batch["latlon_norm"].shape),
            "dtype": str(first_train_batch["image"].dtype),
            "image_min": float(first_train_batch["image"].amin().item()),
            "image_max": float(first_train_batch["image"].amax().item()),
            "has_nan": bool(torch.isnan(first_train_batch["image"]).any().item()),
        }
        report["val_batch"] = {
            "image_shape": list(first_val_batch["image"].shape),
            "latlon_shape": list(first_val_batch["latlon_norm"].shape),
            "dtype": str(first_val_batch["image"].dtype),
            "image_min": float(first_val_batch["image"].amin().item()),
            "image_max": float(first_val_batch["image"].amax().item()),
            "has_nan": bool(torch.isnan(first_val_batch["image"]).any().item()),
        }
    else:
        report["train_batch"] = {
            "latent_shape": list(first_train_batch["latent_z0"].shape),
            "cond_shape": list(first_train_batch["cond_tokens"].shape),
            "latlon_shape": list(first_train_batch["latlon_norm"].shape),
            "dtype": str(first_train_batch["latent_z0"].dtype),
            "latent_min": float(first_train_batch["latent_z0"].amin().item()),
            "latent_max": float(first_train_batch["latent_z0"].amax().item()),
            "has_nan": bool(torch.isnan(first_train_batch["latent_z0"]).any().item()),
        }
        report["val_batch"] = {
            "latent_shape": list(first_val_batch["latent_z0"].shape),
            "cond_shape": list(first_val_batch["cond_tokens"].shape),
            "latlon_shape": list(first_val_batch["latlon_norm"].shape),
            "dtype": str(first_val_batch["latent_z0"].dtype),
            "latent_min": float(first_val_batch["latent_z0"].amin().item()),
            "latent_max": float(first_val_batch["latent_z0"].amax().item()),
            "has_nan": bool(torch.isnan(first_val_batch["latent_z0"]).any().item()),
        }

    vae_locked = load_yaml_config(cfg["vae_locked_config"])
    vae = load_autoencoder_checkpoint(
        vae_locked["checkpoint_path"],
        device=device,
        use_ema=bool(vae_locked.get("use_ema", False)),
        freeze=True,
    )
    prior_model, fuser = load_prior_bundle(cfg, device)
    unet = LatentDiffusionUNet(LatentDiffusionUNetConfig(**cfg["model"])).to(device)
    optimizer = build_optimizer(unet, cfg["optimizer"], device)
    schedule = DiffusionSchedule(DiffusionScheduleConfig(**cfg["diffusion"]), device=device)

    autocast_dtype = resolve_amp_dtype(training_cfg.get("amp_dtype", "float16"))
    autocast_enabled = device.type == "cuda" and bool(training_cfg.get("mixed_precision", True)) and autocast_dtype != torch.float32
    conditioning_cfg = dict(cfg.get("conditioning", {}))

    unet.train()
    optimizer.zero_grad(set_to_none=True)
    train_loss, train_metrics = forward_loss(
        vae=vae,
        prior_model=prior_model,
        fuser=fuser,
        unet=unet,
        schedule=schedule,
        batch=first_train_batch,
        device=device,
        conditioning_cfg=conditioning_cfg,
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
    )
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    unet.eval()
    with torch.no_grad():
        val_loss, val_metrics = forward_loss(
            vae=vae,
            prior_model=prior_model,
            fuser=fuser,
            unet=unet,
            schedule=schedule,
            batch=first_val_batch,
            device=device,
            conditioning_cfg={**conditioning_cfg, "dropout_probability": 0.0},
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
        )

    report["smoke_metrics"] = {
        "train": {
            "loss": float(train_loss.detach().cpu().item()),
            **train_metrics,
        },
        "val": {
            "loss": float(val_loss.detach().cpu().item()),
            **val_metrics,
        },
    }

    checkpoint_dir = Path(cfg.get("checkpoint_dir", Path(args.config).parent / "preflight"))
    smoke_checkpoint = checkpoint_dir / "preflight_smoke.pt"
    save_preflight_checkpoint(unet, smoke_checkpoint)
    report["smoke_checkpoint"] = str(smoke_checkpoint.resolve())
    report["status"] = "ok"

    output_json = Path(args.output_json) if args.output_json else (Path(cfg["output_dir"]) / "logs" / "preflight_report.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
