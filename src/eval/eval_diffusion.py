#!/usr/bin/env python3
"""Evaluate the latent diffusion checkpoint by generating a small preview grid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.models.autoencoder import load_autoencoder_checkpoint
from src.models.conditioning import ConditionFuser, ConditionFuserConfig
from src.models.diffusion_schedule import DiffusionSchedule, DiffusionScheduleConfig
from src.models.fourier_encoder import FourierLocationEncoderConfig
from src.models.latent_diffusion_unet import LatentDiffusionUNet, LatentDiffusionUNetConfig
from src.models.species_prior import LocationSpeciesPriorModel, SpeciesPriorHeadConfig
from src.train.train_diffusion import build_condition_tokens, build_datamodule, configure_runtime, resolve_device
from src.utils.config import load_yaml_config
from src.utils.ema import ExponentialMovingAverage


def load_prior_bundle(cfg, device):
    prior_locked = load_yaml_config(cfg["prior_locked_config"])
    prior_ckpt = torch.load(prior_locked["checkpoint_path"], map_location="cpu")
    prior_artifact_dir = prior_locked.get("artifact_dir") or prior_locked.get("run_metadata", {}).get("artifact_dir")
    if not prior_artifact_dir:
        raise RuntimeError("prior_locked_config is missing artifact_dir metadata.")
    prior_cfg = load_yaml_config(prior_artifact_dir + "/configs/prior_runpod.yaml")
    loc_cfg = FourierLocationEncoderConfig(**prior_cfg["model"]["location_encoder"])
    prior_head_cfg = dict(prior_cfg["model"]["prior_head"])
    prior_head_cfg["num_species"] = len(prior_ckpt["species_to_id"])
    prior_model = LocationSpeciesPriorModel(loc_cfg, SpeciesPriorHeadConfig(**prior_head_cfg)).to(device)
    prior_model.load_state_dict(prior_ckpt["model_state_dict"], strict=False)
    prior_model.eval()
    for parameter in prior_model.parameters():
        parameter.requires_grad_(False)
    fuser = ConditionFuser(
        ConditionFuserConfig(
            prior_dim=int(prior_head_cfg["prior_dim"]),
            species_mix_dim=int(prior_head_cfg["prior_dim"]),
            token_dim=int(loc_cfg.token_dim),
            n_null_tokens=int(loc_cfg.n_tokens + 1),
            species_mix_alpha=float(cfg.get("conditioning", {}).get("species_mix_alpha", 0.5)),
        )
    ).to(device)
    return prior_model, fuser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate latent diffusion by sampling a preview grid.")
    parser.add_argument("--config", required=True, help="Path to diffusion YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Diffusion checkpoint to evaluate.")
    parser.add_argument("--device", default="auto", help="Device string, for example 'cuda' or 'cpu'.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples for the preview grid.")
    parser.add_argument("--guidance-scale", type=float, default=3.0, help="Classifier-free guidance scale.")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA diffusion weights when present in the checkpoint.")
    return parser.parse_args()


def repeat_rows_to_length(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    if tensor.shape[0] >= target_length:
        return tensor[:target_length]
    repeats = math.ceil(target_length / tensor.shape[0])
    repeat_dims = [repeats] + [1] * (tensor.ndim - 1)
    return tensor.repeat(*repeat_dims)[:target_length]


def sample_images_for_latlon(
    *,
    vae,
    prior_model: LocationSpeciesPriorModel,
    fuser: ConditionFuser,
    unet: torch.nn.Module,
    schedule: DiffusionSchedule,
    latlon_norm: torch.Tensor,
    conditioning_cfg,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        cond_tokens = build_condition_tokens(
            prior_model=prior_model,
            fuser=fuser,
            latlon_norm=latlon_norm,
            conditioning_cfg=conditioning_cfg,
        )
        null_tokens = fuser.make_null_condition(latlon_norm.shape[0], device)
        latents = schedule.sample_loop(
            unet,
            shape=torch.Size((latlon_norm.shape[0], vae.config.latent_channels, vae.latent_resolution, vae.latent_resolution)),
            context_tokens=cond_tokens,
            guidance_scale=guidance_scale,
            null_context=null_tokens,
            prediction_type=str(schedule.config.prediction_type),
            device=device,
        )
        return vae.decode_from_diffusion_latents(latents).cpu().clamp(-1.0, 1.0)


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    device = resolve_device(args.device)
    configure_runtime(dict(cfg.get("training", {})), device)
    data_cfg = load_yaml_config(cfg["data_config"])
    data_module = build_datamodule(data_cfg)
    batch = next(iter(data_module.val_dataloader()))

    vae_locked = load_yaml_config(cfg["vae_locked_config"])
    vae = load_autoencoder_checkpoint(vae_locked["checkpoint_path"], device=device, use_ema=bool(vae_locked.get("use_ema", False)), freeze=True)
    prior_model, fuser = load_prior_bundle(cfg, device)

    unet = LatentDiffusionUNet(LatentDiffusionUNetConfig(**cfg["model"])).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    unet.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if args.use_ema and checkpoint.get("ema_state_dict"):
        ema = ExponentialMovingAverage.from_model(unet, decay=0.0)
        ema.load_state_dict(checkpoint["ema_state_dict"])
        ema.copy_to_model(unet)
    if checkpoint.get("fuser_state_dict"):
        fuser.load_state_dict(checkpoint["fuser_state_dict"], strict=False)
    unet.eval()
    schedule = DiffusionSchedule(DiffusionScheduleConfig(**cfg["diffusion"]), device=device)

    conditioning_cfg = dict(cfg.get("conditioning", {}))
    same_location_latlon = batch["latlon_norm"][:1].to(device).expand(args.num_samples, -1)
    multi_location_latlon = repeat_rows_to_length(batch["latlon_norm"].to(device), args.num_samples)
    same_location_images = sample_images_for_latlon(
        vae=vae,
        prior_model=prior_model,
        fuser=fuser,
        unet=unet,
        schedule=schedule,
        latlon_norm=same_location_latlon,
        conditioning_cfg=conditioning_cfg,
        guidance_scale=args.guidance_scale,
        device=device,
    )
    multi_location_images = sample_images_for_latlon(
        vae=vae,
        prior_model=prior_model,
        fuser=fuser,
        unet=unet,
        schedule=schedule,
        latlon_norm=multi_location_latlon,
        conditioning_cfg=conditioning_cfg,
        guidance_scale=args.guidance_scale,
        device=device,
    )

    output_dir = Path(cfg["output_dir"]) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    from torchvision.utils import save_image

    grid_path = output_dir / "preview_grid.png"
    same_location_grid_path = output_dir / "preview_same_location.png"
    multi_location_grid_path = output_dir / "preview_multi_location.png"
    save_image((same_location_images + 1.0) / 2.0, grid_path, nrow=min(args.num_samples, 4))
    save_image((same_location_images + 1.0) / 2.0, same_location_grid_path, nrow=min(args.num_samples, 4))
    save_image((multi_location_images + 1.0) / 2.0, multi_location_grid_path, nrow=min(args.num_samples, 4))
    metrics = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "guidance_scale": args.guidance_scale,
        "num_samples": args.num_samples,
        "used_ema": bool(args.use_ema and checkpoint.get("ema_state_dict")),
        "grid_path": str(grid_path.resolve()),
        "same_location_grid_path": str(same_location_grid_path.resolve()),
        "multi_location_grid_path": str(multi_location_grid_path.resolve()),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
