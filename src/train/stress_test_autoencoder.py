#!/usr/bin/env python3
"""Stronger autoencoder stress tests on a tiny subset of the final dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.data.datamodule import ButterflyDataModule, ButterflyDataModuleConfig
from src.models.autoencoder import AutoencoderConfig, ButterflyAutoencoder
from src.train.train_autoencoder import (
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


def build_data_module(cfg: Dict[str, Any], *, num_workers: Optional[int] = None) -> ButterflyDataModule:
    data_cfg = load_yaml_config(cfg["data_config"])
    merged_cfg = dict(data_cfg)
    merged_cfg["train_csv"] = cfg.get("train_csv", merged_cfg["train_csv"])
    merged_cfg["val_csv"] = cfg.get("val_csv", merged_cfg["val_csv"])
    merged_cfg["test_csv"] = cfg.get("test_csv", merged_cfg.get("test_csv", merged_cfg["val_csv"]))
    if num_workers is not None:
        merged_cfg["num_workers"] = int(num_workers)
        merged_cfg["persistent_workers"] = False
        merged_cfg["pin_memory"] = False
    module = ButterflyDataModule(ButterflyDataModuleConfig(**merged_cfg))
    module.setup()
    return module


def apply_csv_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(cfg)
    if args.train_csv:
        out["train_csv"] = args.train_csv
    if args.val_csv:
        out["val_csv"] = args.val_csv
    if args.test_csv:
        out["test_csv"] = args.test_csv
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test the butterfly autoencoder stack.")
    parser.add_argument("--config", required=True, help="Autoencoder YAML config.")
    parser.add_argument("--device", default="auto", help="Device string, for example 'cuda' or 'cpu'.")
    parser.add_argument("--train-csv", default="", help="Optional train CSV override.")
    parser.add_argument("--val-csv", default="", help="Optional val CSV override.")
    parser.add_argument("--test-csv", default="", help="Optional test CSV override.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for the stress test.")
    parser.add_argument("--steps", type=int, default=20, help="Fixed-batch optimization steps.")
    parser.add_argument("--output-json", default="", help="Optional output report path.")
    parser.add_argument("--disable-perceptual", action="store_true", help="Skip perceptual loss setup.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_csv_overrides(load_yaml_config(args.config), args)
    device = resolve_device(args.device)
    training_cfg = dict(cfg.get("training", {}))
    configure_runtime(training_cfg, device)

    module = build_data_module(cfg, num_workers=args.num_workers)
    train_loader = module.train_dataloader()
    batch = next(iter(train_loader))

    model = ButterflyAutoencoder(AutoencoderConfig(**cfg["model"])).to(device)
    channels_last = bool(training_cfg.get("channels_last", device.type == "cuda"))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = build_optimizer(model, cfg["optimizer"], device)
    autocast_dtype = resolve_amp_dtype(training_cfg.get("amp_dtype", "float16"))
    autocast_enabled = device.type == "cuda" and bool(training_cfg.get("mixed_precision", True))
    perceptual_loss_fn = None
    perceptual_ready = True
    if not args.disable_perceptual:
        try:
            perceptual_loss_fn = build_perceptual_loss(cfg["loss"], device)
        except Exception as exc:  # pragma: no cover - env dependent
            perceptual_ready = False
            perceptual_error = str(exc)
        else:
            perceptual_error = ""
    else:
        perceptual_ready = False
        perceptual_error = "disabled_by_flag"

    train_images = batch["image"].to(device)
    train_images = maybe_channels_last(train_images, channels_last)
    with torch.no_grad():
        posterior_outputs = model(train_images, sample_posterior=True, return_dict=True)
        mode_outputs = model(train_images, sample_posterior=False, return_dict=True)

    losses = []
    grad_norms = []
    nan_detected = False
    model.train()
    for _ in range(max(1, int(args.steps))):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics, _ = forward_and_loss(
            model=model,
            batch=batch,
            device=device,
            loss_cfg=cfg["loss"],
            sample_posterior=True,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
            channels_last=channels_last,
            perceptual_loss_fn=perceptual_loss_fn,
        )
        if not torch.isfinite(loss):
            nan_detected = True
            break
        loss.backward()
        total_grad_sq = torch.zeros((), device=device)
        for parameter in raw_model(model).parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                nan_detected = True
                break
            total_grad_sq = total_grad_sq + parameter.grad.detach().pow(2).sum()
        grad_norms.append(float(torch.sqrt(total_grad_sq).detach().cpu().item()))
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if nan_detected:
            break

    checkpoint_dir = Path(cfg.get("checkpoint_dir", Path(args.config).parent / "stress_test"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "stress_test_roundtrip.pt"
    torch.save({"model_state_dict": raw_model(model).state_dict()}, checkpoint_path)

    reloaded = ButterflyAutoencoder(AutoencoderConfig(**cfg["model"])).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        reference = model(train_images, sample_posterior=False, return_dict=True).reconstruction
        reloaded_recon = reloaded(train_images, sample_posterior=False, return_dict=True).reconstruction
        checkpoint_max_abs_diff = float((reference - reloaded_recon).abs().max().detach().cpu().item())

    report = {
        "config": str(Path(args.config)),
        "device": str(device),
        "dataset_summary": module.summary(),
        "batch_shape": list(batch["image"].shape),
        "posterior_reconstruction_shape": list(posterior_outputs.reconstruction.shape),
        "mode_reconstruction_shape": list(mode_outputs.reconstruction.shape),
        "posterior_latent_shape": list(posterior_outputs.latents.shape),
        "mode_latent_shape": list(mode_outputs.latents.shape),
        "perceptual_ready": perceptual_ready,
        "perceptual_error": perceptual_error,
        "loss_curve": losses,
        "grad_norms": grad_norms,
        "loss_decreased": bool(len(losses) >= 2 and losses[-1] < losses[0]),
        "loss_drop_ratio": float((losses[-1] / losses[0]) if len(losses) >= 2 and losses[0] != 0 else 1.0),
        "nan_detected": nan_detected,
        "checkpoint_roundtrip_max_abs_diff": checkpoint_max_abs_diff,
        "checkpoint_path": str(checkpoint_path),
        "status": "ok" if not nan_detected else "failed_nan",
    }

    output_json = Path(args.output_json) if args.output_json else checkpoint_dir / "stress_test_report.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not nan_detected else 1


if __name__ == "__main__":
    raise SystemExit(main())
