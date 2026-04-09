#!/usr/bin/env python3
"""Evaluate a trained butterfly autoencoder checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torchvision.utils import make_grid, save_image

from src.data.datamodule import ButterflyDataModule, ButterflyDataModuleConfig
from src.losses.autoencoder_losses import compute_autoencoder_loss
from src.losses.perceptual import VGGPerceptualLoss
from src.models.autoencoder import AutoencoderConfig, ButterflyAutoencoder
from src.utils.config import load_yaml_config


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_device(requested: str | None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(name).strip().lower()
    if key not in mapping:
        raise RuntimeError(f"Unsupported AMP dtype: {name}")
    return mapping[key]


def maybe_channels_last(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and tensor.ndim == 4:
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def build_perceptual_loss(loss_cfg: Dict[str, Any], device: torch.device) -> VGGPerceptualLoss | None:
    if float(loss_cfg.get("perceptual_weight", 0.0)) <= 0.0:
        return None
    return VGGPerceptualLoss(
        resize_to=int(loss_cfg.get("perceptual_resize_to", 224)),
        layer_ids=tuple(loss_cfg.get("perceptual_layer_ids", (3, 8, 15, 22))),
        layer_weights=tuple(loss_cfg.get("perceptual_layer_weights", (1.0, 1.0, 1.0, 1.0))),
        use_pretrained=bool(loss_cfg.get("perceptual_use_pretrained", True)),
    ).to(device)


def build_data_module(cfg: Dict[str, Any], split: str) -> ButterflyDataModule:
    data_cfg = load_yaml_config(cfg["data_config"])
    merged_cfg = dict(data_cfg)
    merged_cfg["train_csv"] = cfg.get("train_csv", merged_cfg["train_csv"])
    merged_cfg["val_csv"] = cfg.get("val_csv", merged_cfg["val_csv"])
    merged_cfg["test_csv"] = cfg.get("test_csv", merged_cfg["test_csv"])
    if split == "val":
        merged_cfg["train_csv"] = merged_cfg["val_csv"]
    elif split == "test":
        merged_cfg["train_csv"] = merged_cfg["test_csv"]
    module = ButterflyDataModule(ButterflyDataModuleConfig(**merged_cfg))
    module.setup()
    return module


def save_reconstruction_grid(inputs: torch.Tensor, reconstructions: torch.Tensor, output_path: Path) -> None:
    inputs = inputs.detach().cpu()
    reconstructions = reconstructions.detach().cpu()
    grid = make_grid(((torch.cat((inputs, reconstructions), dim=0)).clamp(-1.0, 1.0) + 1.0) / 2.0, nrow=inputs.shape[0])
    ensure_dir(output_path.parent)
    save_image(grid, output_path)


def save_image_grid(images: torch.Tensor, output_path: Path, *, nrow: int) -> None:
    grid = make_grid((images.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0, nrow=nrow)
    ensure_dir(output_path.parent)
    save_image(grid, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the butterfly autoencoder.")
    parser.add_argument("--config", required=True, help="Path to autoencoder YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path produced by train_autoencoder.py.")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Evaluation split.")
    parser.add_argument("--device", default="auto", help="Device string, for example 'cuda' or 'cpu'.")
    parser.add_argument("--max-batches", type=int, default=0, help="Optional cap for debugging.")
    parser.add_argument("--use-ema", action="store_true", help="Evaluate EMA weights if the checkpoint contains them.")
    parser.add_argument("--num-prior-samples", type=int, default=8, help="How many decoder-only prior samples to save.")
    parser.add_argument("--prior-sample-seed", type=int, default=42, help="Seed for fixed prior samples.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    device = resolve_device(args.device)
    module = build_data_module(cfg, args.split)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = ButterflyAutoencoder(AutoencoderConfig(**cfg["model"])).to(device)
    if bool(cfg["training"].get("channels_last", device.type == "cuda")):
        model = model.to(memory_format=torch.channels_last)
    state_dict = checkpoint["model_state_dict"]
    if args.use_ema and checkpoint.get("ema_state_dict"):
        state_dict = checkpoint["ema_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dataloader = module.val_dataloader() if args.split == "val" else module.test_dataloader()
    loss_cfg = cfg["loss"]
    autocast_enabled = device.type == "cuda" and bool(cfg["training"].get("mixed_precision", True))
    autocast_dtype = resolve_amp_dtype(cfg["training"].get("amp_dtype", "float16"))
    perceptual_loss_fn = build_perceptual_loss(loss_cfg, device)

    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0, "masked_reconstruction": 0.0, "perceptual": 0.0}
    num_batches = 0
    first_inputs = None
    first_recons = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            images = maybe_channels_last(images, bool(cfg["training"].get("channels_last", device.type == "cuda")))
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                outputs = model(
                    images,
                    sample_posterior=bool(cfg["training"].get("validation_sample_posterior", False)),
                    return_dict=True,
                )
                loss_breakdown = compute_autoencoder_loss(
                    inputs=images,
                    reconstructions=outputs.reconstruction,
                    posterior_kl=outputs.posterior.kl(),
                    perceptual_loss_fn=perceptual_loss_fn,
                    mask=mask,
                    reconstruction_weight=float(loss_cfg["reconstruction_weight"]),
                    kl_weight=float(loss_cfg["kl_weight"]),
                    masked_reconstruction_weight=float(loss_cfg.get("masked_reconstruction_weight", 0.0)),
                    perceptual_weight=float(loss_cfg.get("perceptual_weight", 0.0)),
                )

            totals["loss"] += float(loss_breakdown.total.detach().cpu().item())
            totals["reconstruction"] += float(loss_breakdown.reconstruction.detach().cpu().item())
            totals["kl"] += float(loss_breakdown.kl.detach().cpu().item())
            totals["masked_reconstruction"] += float(loss_breakdown.masked_reconstruction.detach().cpu().item())
            totals["perceptual"] += float(loss_breakdown.perceptual.detach().cpu().item())
            num_batches += 1

            if first_inputs is None:
                first_inputs = batch["image"]
                first_recons = outputs.reconstruction.detach()

            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

    if num_batches == 0:
        raise RuntimeError("Evaluation dataloader yielded no batches.")

    metrics = {key: value / num_batches for key, value in totals.items()}
    output_dir = Path(cfg["output_dir"]) / "eval" / args.split
    save_reconstruction_grid(first_inputs, first_recons, output_dir / "reconstruction_grid.png")
    num_prior_samples = max(1, int(args.num_prior_samples))
    prior_latents = torch.randn(
        (
            num_prior_samples,
            model.config.latent_channels,
            model.latent_resolution,
            model.latent_resolution,
        ),
        generator=torch.Generator(device="cpu").manual_seed(int(args.prior_sample_seed)),
        dtype=torch.float32,
    ).to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            prior_samples = model.decode(prior_latents)
    save_image_grid(prior_samples, output_dir / "prior_samples.png", nrow=min(4, num_prior_samples))
    write_json(
        output_dir / "metrics.json",
        {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "metrics": metrics,
            "prior_sample_seed": int(args.prior_sample_seed),
            "num_prior_samples": num_prior_samples,
        },
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
