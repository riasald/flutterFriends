#!/usr/bin/env python3
"""Train a latent diffusion model conditioned on location and prior embeddings."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.data.cached_diffusion_dataset import CachedDiffusionDataModule, CachedDiffusionDataModuleConfig
from src.data.datamodule import ButterflyDataModule, ButterflyDataModuleConfig
from src.losses.diffusion_losses import diffusion_mse_loss
from src.models.autoencoder import load_autoencoder_checkpoint
from src.models.conditioning import ConditionFuser, ConditionFuserConfig
from src.models.diffusion_schedule import DiffusionSchedule, DiffusionScheduleConfig
from src.models.fourier_encoder import FourierLocationEncoderConfig
from src.models.latent_diffusion_unet import LatentDiffusionUNet, LatentDiffusionUNetConfig
from src.models.species_prior import LocationSpeciesPriorModel, SpeciesPriorHeadConfig
from src.utils.config import load_yaml_config
from src.utils.ema import ExponentialMovingAverage


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


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


def configure_runtime(training_cfg: Dict[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return
    if "set_float32_matmul_precision" in training_cfg:
        torch.set_float32_matmul_precision(str(training_cfg["set_float32_matmul_precision"]))
    allow_tf32 = bool(training_cfg.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = bool(training_cfg.get("cudnn_benchmark", True))


def maybe_compile_model(model: torch.nn.Module, training_cfg: Dict[str, Any]) -> torch.nn.Module:
    if not bool(training_cfg.get("compile", False)):
        return model
    if hasattr(torch, "compile"):
        return torch.compile(model, mode=training_cfg.get("compile_mode", "default"))
    return model


def raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def build_datamodule(data_cfg: Dict[str, Any]) -> ButterflyDataModule | CachedDiffusionDataModule:
    if {"train_cache_pt", "val_cache_pt", "test_cache_pt"}.issubset(data_cfg.keys()):
        module = CachedDiffusionDataModule(CachedDiffusionDataModuleConfig(**data_cfg))
        module.setup()
        return module
    module = ButterflyDataModule(ButterflyDataModuleConfig(**data_cfg))
    module.setup()
    return module


def load_prior_bundle(cfg: Dict[str, Any], device: torch.device) -> tuple[LocationSpeciesPriorModel, ConditionFuser]:
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


def build_condition_tokens(
    *,
    prior_model: LocationSpeciesPriorModel,
    fuser: ConditionFuser,
    latlon_norm: torch.Tensor,
    conditioning_cfg: Dict[str, Any],
) -> torch.Tensor:
    prior_outputs = prior_model(
        latlon_norm,
        mixture_temperature=float(conditioning_cfg.get("prior_temperature", 1.0)),
        mixture_top_k=int(conditioning_cfg.get("prior_top_k", 0)),
    )
    return fuser(
        prior_outputs["loc_tokens"],
        prior_outputs["z_prior"],
        z_species_mix=prior_outputs.get("z_species_mix"),
        species_mix_alpha=float(conditioning_cfg.get("species_mix_alpha", 0.5)),
    )


def build_optimizer(model: torch.nn.Module, optimizer_cfg: Dict[str, Any], device: torch.device) -> torch.optim.Optimizer:
    kwargs: Dict[str, Any] = dict(
        lr=float(optimizer_cfg["learning_rate"]),
        betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )
    if bool(optimizer_cfg.get("fused", device.type == "cuda")) and device.type == "cuda":
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(model.parameters(), **kwargs)


def repeat_rows_to_length(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    if tensor.shape[0] >= target_length:
        return tensor[:target_length]
    repeats = math.ceil(target_length / tensor.shape[0])
    repeat_dims = [repeats] + [1] * (tensor.ndim - 1)
    return tensor.repeat(*repeat_dims)[:target_length]


def prune_periodic_checkpoints(checkpoint_dir: Path, keep: int) -> None:
    periodic = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if keep <= 0:
        for path in periodic:
            path.unlink(missing_ok=True)
        return
    for path in periodic[:-keep]:
        path.unlink(missing_ok=True)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    training_cfg: Dict[str, Any],
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_name = str(training_cfg.get("lr_scheduler", "constant")).strip().lower()
    if scheduler_name in {"", "none", "constant"}:
        return None
    if scheduler_name != "cosine":
        raise RuntimeError(f"Unsupported diffusion lr_scheduler: {scheduler_name}")
    warmup_steps = int(training_cfg.get("warmup_steps", 0))
    epochs = int(training_cfg["epochs"])
    total_steps = max(steps_per_epoch * epochs, 1)
    min_lr_scale = float(training_cfg.get("min_lr_scale", 0.1))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress_denom = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / progress_denom, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def compute_loss_weight(
    *,
    schedule: DiffusionSchedule,
    timesteps: torch.Tensor,
    prediction_type: str,
    loss_weighting: str,
    min_snr_gamma: float,
) -> torch.Tensor | None:
    if loss_weighting in {"", "none", "uniform"}:
        return None
    if loss_weighting != "min_snr":
        raise RuntimeError(f"Unsupported diffusion loss_weighting: {loss_weighting}")
    snr = schedule.compute_snr(timesteps).clamp_min(1e-8)
    clipped = torch.minimum(snr, torch.full_like(snr, float(min_snr_gamma)))
    if prediction_type == "epsilon":
        return clipped / snr
    if prediction_type == "v_prediction":
        return clipped / (snr + 1.0)
    raise RuntimeError(f"Unsupported diffusion prediction_type for min_snr: {prediction_type}")


def forward_loss(
    *,
    vae,
    prior_model: LocationSpeciesPriorModel,
    fuser: ConditionFuser,
    unet: torch.nn.Module,
    schedule: DiffusionSchedule,
    batch: Dict[str, Any],
    device: torch.device,
    conditioning_cfg: Dict[str, Any],
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> tuple[torch.Tensor, Dict[str, float]]:
    latlon_norm = batch["latlon_norm"].to(device, non_blocking=True)
    if not torch.isfinite(latlon_norm).all():
        raise RuntimeError("Encountered non-finite lat/lon tensor in diffusion batch.")

    if "latent_z0" in batch and "cond_tokens" in batch:
        z0 = batch["latent_z0"].to(device, non_blocking=True)
        cond_tokens = batch["cond_tokens"].to(device, non_blocking=True)
    else:
        images = batch["image"].to(device, non_blocking=True)
        if not torch.isfinite(images).all():
            raise RuntimeError("Encountered non-finite image tensor in diffusion batch.")
        with torch.no_grad():
            z0 = vae.encode_for_diffusion(images, sample_posterior=False)
            cond_tokens = build_condition_tokens(
                prior_model=prior_model,
                fuser=fuser,
                latlon_norm=latlon_norm,
                conditioning_cfg=conditioning_cfg,
            )

    batch_size = z0.shape[0]
    timesteps = schedule.sample_timesteps(batch_size, device=device)
    noise = torch.randn_like(z0)
    zt = schedule.q_sample(z0, timesteps, noise)
    diffusion_cfg = schedule.config
    prediction_type = str(diffusion_cfg.prediction_type)
    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "v_prediction":
        target = schedule.compute_velocity(z0, noise, timesteps)
    else:
        raise RuntimeError(f"Unsupported diffusion prediction_type: {prediction_type}")

    condition_dropout = float(conditioning_cfg.get("dropout_probability", 0.1))
    if condition_dropout > 0.0:
        drop_mask = (torch.rand(batch_size, device=device) < condition_dropout).view(batch_size, 1, 1)
        null_tokens = fuser.make_null_condition(batch_size, device)
        cond_tokens = torch.where(drop_mask, null_tokens, cond_tokens)

    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
        model_output = unet(zt, timesteps, cond_tokens)
        loss = diffusion_mse_loss(
            model_output,
            target,
            sample_weight=compute_loss_weight(
                schedule=schedule,
                timesteps=timesteps,
                prediction_type=prediction_type,
                loss_weighting=str(diffusion_cfg.loss_weighting),
                min_snr_gamma=float(diffusion_cfg.min_snr_gamma),
            ),
        )

    if not torch.isfinite(loss):
        raise RuntimeError(f"Diffusion loss became non-finite: {float(loss.detach().cpu().item())}")
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "latent_std": float(z0.detach().std().cpu().item()),
        "noise_std": float(noise.detach().std().cpu().item()),
        "target_std": float(target.detach().std().cpu().item()),
    }


def save_sample_grid(
    *,
    path: Path,
    vae,
    prior_model: LocationSpeciesPriorModel,
    fuser: ConditionFuser,
    unet: torch.nn.Module,
    schedule: DiffusionSchedule,
    batch: Dict[str, Any],
    device: torch.device,
    guidance_scale: float,
    num_samples: int,
    conditioning_cfg: Dict[str, Any],
) -> None:
    from torchvision.utils import save_image

    with torch.no_grad():
        if "cond_tokens" in batch:
            cond_tokens_all = batch["cond_tokens"].to(device)
            same_location_tokens = cond_tokens_all[:1].expand(num_samples, -1, -1)
            multi_location_tokens = repeat_rows_to_length(cond_tokens_all, num_samples)
        else:
            latlon_all = batch["latlon_norm"].to(device)
            same_location_latlon = latlon_all[:1].expand(num_samples, -1)
            multi_location_latlon = repeat_rows_to_length(latlon_all, num_samples)
            same_location_tokens = build_condition_tokens(
                prior_model=prior_model,
                fuser=fuser,
                latlon_norm=same_location_latlon,
                conditioning_cfg=conditioning_cfg,
            )
            multi_location_tokens = build_condition_tokens(
                prior_model=prior_model,
                fuser=fuser,
                latlon_norm=multi_location_latlon,
                conditioning_cfg=conditioning_cfg,
            )
        null_tokens = fuser.make_null_condition(num_samples, device)

        def decode(tokens: torch.Tensor) -> torch.Tensor:
            latents = schedule.sample_loop(
                unet,
                shape=torch.Size((num_samples, vae.config.latent_channels, vae.latent_resolution, vae.latent_resolution)),
                context_tokens=tokens,
                guidance_scale=guidance_scale,
                null_context=null_tokens,
                prediction_type=str(schedule.config.prediction_type),
                device=device,
            )
            return vae.decode_from_diffusion_latents(latents).cpu().clamp(-1.0, 1.0)

        multi_images = decode(multi_location_tokens)
        same_images = decode(same_location_tokens)
        save_image((multi_images + 1.0) / 2.0, path, nrow=min(num_samples, 4))
        save_image((same_images + 1.0) / 2.0, path.with_name(f"{path.stem}_same{path.suffix}"), nrow=min(num_samples, 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the latent diffusion model.")
    parser.add_argument("--config", required=True, help="Path to diffusion YAML config.")
    parser.add_argument("--device", default="auto", help="Device string, for example 'cuda' or 'cpu'.")
    parser.add_argument("--resume", default="", help="Optional checkpoint path to resume from.")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Optional cap for smoke tests.")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Optional cap for smoke tests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    training_cfg = dict(cfg.get("training", {}))
    set_seed(int(cfg["seed"]), deterministic=bool(training_cfg.get("deterministic", False)))

    output_dir = Path(cfg["output_dir"])
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    logs_dir = output_dir / "logs"
    sample_dir = output_dir / "samples"
    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)
    ensure_dir(logs_dir)
    ensure_dir(sample_dir)

    device = resolve_device(args.device)
    configure_runtime(training_cfg, device)
    data_cfg = load_yaml_config(cfg["data_config"])
    using_cached_inputs = {"train_cache_pt", "val_cache_pt", "test_cache_pt"}.issubset(data_cfg.keys())
    data_module = build_datamodule(data_cfg)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

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
    train_model = maybe_compile_model(unet, training_cfg)
    ema_decay = float(training_cfg.get("ema_decay", 0.0))
    ema = ExponentialMovingAverage.from_model(raw_model(unet), ema_decay) if ema_decay > 0.0 else None
    scheduler = build_lr_scheduler(optimizer, training_cfg=training_cfg, steps_per_epoch=len(train_loader))

    autocast_dtype = resolve_amp_dtype(training_cfg.get("amp_dtype", "float16"))
    autocast_enabled = device.type == "cuda" and bool(training_cfg.get("mixed_precision", True)) and autocast_dtype != torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled and autocast_dtype == torch.float16)
    conditioning_cfg = dict(cfg.get("conditioning", {}))
    guidance_scale = float(cfg["inference"].get("guidance_scale", 3.0))
    epochs = int(training_cfg["epochs"])
    best_val_loss = float("inf")
    history: list[Dict[str, Any]] = []
    global_step = 0
    start_epoch = 1

    write_json(
        logs_dir / "run_metadata.json",
        {
            "created_at_utc": utc_now_iso(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "config_path": str(Path(args.config).resolve()),
            "using_cached_inputs": using_cached_inputs,
            "dataset_summary": data_module.summary(),
        },
    )

    resume_path = args.resume or str(training_cfg.get("resume_from", "")).strip()
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
        raw_model(unet).load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("fuser_state_dict"):
            fuser.load_state_dict(checkpoint["fuser_state_dict"], strict=False)
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if ema is not None and checkpoint.get("ema_state_dict") is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        history = checkpoint.get("history", [])
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    try:
        for epoch in range(start_epoch, epochs + 1):
            train_model.train()
            train_total = 0.0
            train_batches = 0
            optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(train_loader, start=1):
                loss, metrics = forward_loss(
                    vae=vae,
                    prior_model=prior_model,
                    fuser=fuser,
                    unet=train_model,
                    schedule=schedule,
                    batch=batch,
                    device=device,
                    conditioning_cfg=conditioning_cfg,
                    autocast_enabled=autocast_enabled,
                    autocast_dtype=autocast_dtype,
                )
                scaler.scale(loss).backward()
                grad_clip_norm = float(training_cfg.get("grad_clip_norm", 0.0))
                if grad_clip_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(raw_model(unet).parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(raw_model(unet))
                global_step += 1
                train_total += metrics["loss"]
                train_batches += 1
                if global_step % int(training_cfg.get("log_every", 10)) == 0:
                    current_lr = float(optimizer.param_groups[0]["lr"])
                    print(
                        f"epoch={epoch} step={global_step} loss={metrics['loss']:.4f} "
                        f"latent_std={metrics['latent_std']:.4f} noise_std={metrics['noise_std']:.4f} "
                        f"target_std={metrics['target_std']:.4f} lr={current_lr:.6f}"
                    )
                if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                    break

            train_metrics = {"loss": float(train_total / max(train_batches, 1))}

            train_model.eval()
            val_total = 0.0
            val_batches = 0
            first_val_batch: Optional[Dict[str, Any]] = None
            ema_original = ema.copy_to_model(raw_model(unet)) if ema is not None else None
            try:
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader, start=1):
                        if first_val_batch is None:
                            first_val_batch = batch
                        loss, _ = forward_loss(
                            vae=vae,
                            prior_model=prior_model,
                            fuser=fuser,
                            unet=train_model,
                            schedule=schedule,
                            batch=batch,
                            device=device,
                            conditioning_cfg={**conditioning_cfg, "dropout_probability": 0.0},
                            autocast_enabled=autocast_enabled,
                            autocast_dtype=autocast_dtype,
                        )
                        val_total += float(loss.detach().cpu().item())
                        val_batches += 1
                        if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                            break
            finally:
                if ema is not None and ema_original is not None:
                    ema.restore(raw_model(unet), ema_original)

            val_metrics = {"loss": float(val_total / max(val_batches, 1)), "ema_enabled": ema is not None}
            epoch_record = {
                "epoch": epoch,
                "completed_at_utc": utc_now_iso(),
                "train": train_metrics,
                "val": val_metrics,
                "global_step": global_step,
            }
            history.append(epoch_record)
            write_json(output_dir / "history.json", {"history": history})
            append_jsonl(logs_dir / "epoch_metrics.jsonl", epoch_record)

            checkpoint_payload = {
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "model_state_dict": raw_model(unet).state_dict(),
                "fuser_state_dict": fuser.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "ema_state_dict": ema.state_dict() if ema is not None else None,
                "config": cfg,
                "history": history,
            }
            torch.save(checkpoint_payload, checkpoint_dir / "last.pt")

            if val_metrics["loss"] <= best_val_loss:
                best_val_loss = val_metrics["loss"]
                checkpoint_payload["best_val_loss"] = best_val_loss
                torch.save(checkpoint_payload, checkpoint_dir / "best.pt")
                if first_val_batch is not None:
                    ema_original = ema.copy_to_model(raw_model(unet)) if ema is not None else None
                    try:
                        save_sample_grid(
                            path=sample_dir / f"epoch_{epoch:03d}.png",
                            vae=vae,
                            prior_model=prior_model,
                            fuser=fuser,
                            unet=train_model,
                            schedule=schedule,
                            batch=first_val_batch,
                            device=device,
                            guidance_scale=guidance_scale,
                            num_samples=int(cfg["inference"].get("num_preview_samples", 4)),
                            conditioning_cfg=conditioning_cfg,
                        )
                    finally:
                        if ema is not None and ema_original is not None:
                            ema.restore(raw_model(unet), ema_original)

            save_every_epochs = int(training_cfg.get("save_every_epochs", 0))
            max_periodic_checkpoints = int(training_cfg.get("max_periodic_checkpoints", 2))
            if save_every_epochs > 0 and max_periodic_checkpoints != 0 and epoch % save_every_epochs == 0:
                torch.save(checkpoint_payload, checkpoint_dir / f"epoch_{epoch:03d}.pt")
                prune_periodic_checkpoints(checkpoint_dir, keep=max_periodic_checkpoints)

            save_samples_every_epochs = int(training_cfg.get("save_samples_every_epochs", 0))
            if first_val_batch is not None and save_samples_every_epochs > 0 and epoch % save_samples_every_epochs == 0:
                ema_original = ema.copy_to_model(raw_model(unet)) if ema is not None else None
                try:
                    save_sample_grid(
                        path=sample_dir / f"epoch_{epoch:03d}_periodic.png",
                        vae=vae,
                        prior_model=prior_model,
                        fuser=fuser,
                        unet=train_model,
                        schedule=schedule,
                        batch=first_val_batch,
                        device=device,
                        guidance_scale=guidance_scale,
                        num_samples=int(cfg["inference"].get("num_preview_samples", 4)),
                        conditioning_cfg=conditioning_cfg,
                    )
                finally:
                    if ema is not None and ema_original is not None:
                        ema.restore(raw_model(unet), ema_original)

            write_json(
                logs_dir / "run_status.json",
                {
                    "status": "running",
                    "updated_at_utc": utc_now_iso(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                },
            )
            print(f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f}")
    except Exception as error:
        write_json(
            logs_dir / "failure_report.json",
            {"status": "failed", "error_type": type(error).__name__, "error_message": str(error)},
        )
        (logs_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise

    write_json(output_dir / "final_metrics.json", {"best_val_loss": best_val_loss, "history": history})
    write_json(
        logs_dir / "run_status.json",
        {"status": "completed", "updated_at_utc": utc_now_iso(), "best_val_loss": best_val_loss},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
