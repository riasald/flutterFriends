#!/usr/bin/env python3
"""Train the butterfly KL autoencoder with a plain PyTorch loop."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from torchvision.utils import make_grid, save_image

from src.data.datamodule import ButterflyDataModule, ButterflyDataModuleConfig
from src.losses.autoencoder_losses import compute_autoencoder_loss
from src.losses.perceptual import VGGPerceptualLoss
from src.models.autoencoder import AutoencoderConfig, ButterflyAutoencoder
from src.utils.config import load_yaml_config
from src.utils.ema import ExponentialMovingAverage


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


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


def build_data_module(autoencoder_cfg: Dict[str, Any]) -> ButterflyDataModule:
    data_cfg = load_yaml_config(autoencoder_cfg["data_config"])
    merged_cfg = dict(data_cfg)
    merged_cfg["train_csv"] = autoencoder_cfg.get("train_csv", merged_cfg["train_csv"])
    merged_cfg["val_csv"] = autoencoder_cfg.get("val_csv", merged_cfg["val_csv"])
    merged_cfg["test_csv"] = autoencoder_cfg.get("test_csv", merged_cfg.get("test_csv", merged_cfg["val_csv"]))
    module = ButterflyDataModule(ButterflyDataModuleConfig(**merged_cfg))
    module.setup()
    return module


def save_reconstruction_grid(
    *,
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    output_path: Path,
) -> None:
    inputs = inputs.detach().cpu()
    reconstructions = reconstructions.detach().cpu()
    stacked = torch.cat((inputs, reconstructions), dim=0)
    stacked = stacked.clamp(-1.0, 1.0)
    grid = make_grid((stacked + 1.0) / 2.0, nrow=inputs.shape[0])
    ensure_dir(output_path.parent)
    save_image(grid, output_path)


def save_image_grid(images: torch.Tensor, output_path: Path, *, nrow: int) -> None:
    grid = make_grid((images.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0, nrow=nrow)
    ensure_dir(output_path.parent)
    save_image(grid, output_path)


def configure_runtime(training_cfg: Dict[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return
    if "set_float32_matmul_precision" in training_cfg:
        torch.set_float32_matmul_precision(str(training_cfg["set_float32_matmul_precision"]))
    allow_tf32 = bool(training_cfg.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = bool(training_cfg.get("cudnn_benchmark", True))


def maybe_channels_last(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and tensor.ndim == 4:
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def build_optimizer(model: torch.nn.Module, optimizer_cfg: Dict[str, Any], device: torch.device) -> torch.optim.Optimizer:
    kwargs: Dict[str, Any] = dict(
        lr=float(optimizer_cfg["learning_rate"]),
        betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )
    fused_requested = bool(optimizer_cfg.get("fused", device.type == "cuda"))
    if fused_requested and device.type == "cuda":
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(model.parameters(), **kwargs)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: Dict[str, Any],
    *,
    total_optimizer_steps: int,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    name = str(scheduler_cfg.get("name", "none")).lower()
    if name in {"", "none", "off"}:
        return None
    if name != "cosine":
        raise RuntimeError(f"Unsupported scheduler: {name}")

    warmup_steps = max(0, int(scheduler_cfg.get("warmup_steps", 0)))
    min_lr_scale = float(scheduler_cfg.get("min_lr_scale", 0.1))
    total_steps = max(total_optimizer_steps, 1)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_perceptual_loss(loss_cfg: Dict[str, Any], device: torch.device) -> Optional[VGGPerceptualLoss]:
    if float(loss_cfg.get("perceptual_weight", 0.0)) <= 0.0:
        return None
    module = VGGPerceptualLoss(
        resize_to=int(loss_cfg.get("perceptual_resize_to", 224)),
        layer_ids=tuple(loss_cfg.get("perceptual_layer_ids", (3, 8, 15, 22))),
        layer_weights=tuple(loss_cfg.get("perceptual_layer_weights", (1.0, 1.0, 1.0, 1.0))),
        use_pretrained=bool(loss_cfg.get("perceptual_use_pretrained", True)),
    )
    return module.to(device)


def maybe_compile_model(model: torch.nn.Module, training_cfg: Dict[str, Any]) -> torch.nn.Module:
    if not bool(training_cfg.get("compile", False)):
        return model
    compile_kwargs = {
        "mode": training_cfg.get("compile_mode", "default"),
    }
    if hasattr(torch, "compile"):
        return torch.compile(model, **compile_kwargs)
    return model


def raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    base_model = raw_model(model)
    total = sum(parameter.numel() for parameter in base_model.parameters())
    trainable = sum(parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
    }


def build_run_metadata(
    *,
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    data_module: ButterflyDataModule,
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
    channels_last: bool,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": {
            "requested": args.device,
            "resolved": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cudnn_available": bool(torch.backends.cudnn.is_available()),
        },
        "config_path": str(Path(args.config).resolve()),
        "output_dir": str(Path(cfg["output_dir"]).resolve()),
        "checkpoint_dir": str(Path(cfg["checkpoint_dir"]).resolve()),
        "data_config": str(cfg["data_config"]),
        "dataset_summary": data_module.summary(),
        "dataloader": {
            "train_batches_per_epoch": len(train_loader),
            "val_batches": len(val_loader),
            "batch_size": int(data_module.config.batch_size),
            "eval_batch_size": int(data_module.config.eval_batch_size),
            "num_workers": int(data_module.config.num_workers),
            "pin_memory": bool(data_module.config.pin_memory),
            "persistent_workers": bool(data_module.config.persistent_workers),
            "prefetch_factor": int(data_module.config.prefetch_factor),
            "drop_last_train": bool(data_module.config.drop_last_train),
        },
        "runtime": {
            "autocast_enabled": bool(autocast_enabled),
            "autocast_dtype": str(autocast_dtype),
            "channels_last": bool(channels_last),
        },
        "model": {
            "config": cfg["model"],
            "parameter_counts": count_parameters(model),
        },
    }
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        metadata["device"]["name"] = torch.cuda.get_device_name(device_index)
        metadata["device"]["total_memory_bytes"] = int(props.total_memory)
        metadata["device"]["capability"] = f"{props.major}.{props.minor}"
    return metadata


def write_failure_artifacts(
    *,
    logs_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    error: BaseException,
    run_state: Dict[str, Any],
) -> None:
    ensure_dir(logs_dir)
    traceback_text = traceback.format_exc()
    (logs_dir / "traceback.txt").write_text(traceback_text, encoding="utf-8")
    failure_payload = {
        "status": "failed",
        "failed_at_utc": utc_now_iso(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "config_path": str(Path(args.config).resolve()),
        "output_dir": str(output_dir.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "run_state": run_state,
        "training_config": cfg.get("training", {}),
    }
    write_json(logs_dir / "failure_report.json", failure_payload)
    write_json(
        logs_dir / "run_status.json",
        {
            "status": "failed",
            "updated_at_utc": utc_now_iso(),
            "run_state": run_state,
        },
    )


@contextmanager
def maybe_ema_scope(
    model: torch.nn.Module,
    ema: Optional[ExponentialMovingAverage],
    enabled: bool,
) -> Iterator[None]:
    if ema is None or not enabled:
        yield
        return
    original = ema.copy_to_model(raw_model(model))
    try:
        yield
    finally:
        ema.restore(raw_model(model), original)


def forward_and_loss(
    *,
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    loss_cfg: Dict[str, Any],
    sample_posterior: bool,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
    channels_last: bool,
    perceptual_loss_fn: Optional[torch.nn.Module],
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    images = maybe_channels_last(images, channels_last)
    mask = batch.get("mask")
    if mask is not None:
        mask = mask.to(device, non_blocking=True)

    with torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
        enabled=autocast_enabled,
    ):
        outputs = model(images, sample_posterior=sample_posterior, return_dict=True)
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

    metrics = {
        "loss": float(loss_breakdown.total.detach().cpu().item()),
        "reconstruction": float(loss_breakdown.reconstruction.detach().cpu().item()),
        "kl": float(loss_breakdown.kl.detach().cpu().item()),
        "masked_reconstruction": float(loss_breakdown.masked_reconstruction.detach().cpu().item()),
        "perceptual": float(loss_breakdown.perceptual.detach().cpu().item()),
    }
    return loss_breakdown.total, metrics, outputs.reconstruction.detach()


def run_validation(
    *,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_cfg: Dict[str, Any],
    sample_posterior: bool,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
    channels_last: bool,
    perceptual_loss_fn: Optional[torch.nn.Module],
    max_batches: int,
) -> Dict[str, Any]:
    model.eval()
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0, "masked_reconstruction": 0.0, "perceptual": 0.0}
    num_batches = 0
    first_images = None
    first_recons = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            loss, metrics, reconstructions = forward_and_loss(
                model=model,
                batch=batch,
                device=device,
                loss_cfg=loss_cfg,
                sample_posterior=sample_posterior,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
                channels_last=channels_last,
                perceptual_loss_fn=perceptual_loss_fn,
            )
            _ = loss
            for key, value in metrics.items():
                totals[key] += value
            num_batches += 1
            if first_images is None:
                first_images = batch["image"]
                first_recons = reconstructions
            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break

    if num_batches == 0:
        raise RuntimeError("Validation dataloader yielded no batches.")

    averaged = {key: value / num_batches for key, value in totals.items()}
    return {
        "metrics": averaged,
        "images": first_images,
        "reconstructions": first_recons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the butterfly autoencoder.")
    parser.add_argument("--config", required=True, help="Path to autoencoder YAML config.")
    parser.add_argument("--device", default="auto", help="Device string, for example 'cuda' or 'cpu'.")
    parser.add_argument("--resume", default="", help="Optional checkpoint path to resume from.")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Optional cap for debugging.")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Optional cap for debugging.")
    return parser.parse_args()


def run_training(args: argparse.Namespace, cfg: Dict[str, Any]) -> int:
    training_cfg = dict(cfg.get("training", {}))
    device = resolve_device(args.device)
    configure_runtime(training_cfg, device)
    data_module = build_data_module(cfg)

    model = ButterflyAutoencoder(AutoencoderConfig(**cfg["model"])).to(device)
    channels_last = bool(training_cfg.get("channels_last", device.type == "cuda"))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = build_optimizer(model, cfg["optimizer"], device)
    perceptual_loss_fn = build_perceptual_loss(cfg["loss"], device)
    autocast_dtype = resolve_amp_dtype(training_cfg.get("amp_dtype", "float16"))
    autocast_enabled = device.type == "cuda" and bool(training_cfg.get("mixed_precision", True))
    use_grad_scaler = autocast_enabled and autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    train_model = maybe_compile_model(model, training_cfg)

    output_dir = Path(cfg["output_dir"])
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    logs_dir = output_dir / "logs"
    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)
    ensure_dir(logs_dir)

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    max_train_batches = int(args.max_train_batches)
    max_val_batches = int(args.max_val_batches)
    train_batches_per_epoch = max_train_batches if max_train_batches > 0 else len(train_loader)
    grad_accum_steps = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    epochs = int(training_cfg["epochs"])
    optimizer_steps_per_epoch = math.ceil(train_batches_per_epoch / grad_accum_steps)
    total_optimizer_steps = epochs * max(1, optimizer_steps_per_epoch)
    scheduler = build_scheduler(
        optimizer,
        cfg.get("scheduler", {}),
        total_optimizer_steps=total_optimizer_steps,
    )

    ema_decay = float(training_cfg.get("ema_decay", 0.0))
    ema = ExponentialMovingAverage.from_model(model, ema_decay) if ema_decay > 0.0 else None

    history = []
    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1
    run_state: Dict[str, Any] = {
        "phase": "initializing",
        "epoch": 0,
        "global_step": 0,
        "best_val_loss": None,
        "last_checkpoint": None,
        "last_epoch_metrics": None,
    }
    setattr(args, "_run_state", run_state)
    run_metadata = build_run_metadata(
        args=args,
        cfg=cfg,
        data_module=data_module,
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
        channels_last=channels_last,
    )
    write_json(logs_dir / "run_metadata.json", run_metadata)
    write_json(
        logs_dir / "run_status.json",
        {
            "status": "running",
            "updated_at_utc": utc_now_iso(),
            "run_state": run_state,
        },
    )

    resume_path = args.resume or str(training_cfg.get("resume_from", "")).strip()
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
        raw_model(model).load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if ema is not None and checkpoint.get("ema_state_dict") is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        history = checkpoint.get("history", [])
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        run_state["last_checkpoint"] = str(Path(resume_path).resolve())
        run_state["best_val_loss"] = best_val_loss

    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 0.0))
    sample_posterior = bool(training_cfg.get("sample_posterior", True))
    validation_sample_posterior = bool(training_cfg.get("validation_sample_posterior", False))
    evaluate_with_ema = bool(training_cfg.get("evaluate_with_ema", ema is not None))
    quicklook_samples = max(1, int(training_cfg.get("quicklook_num_samples", 8)))
    quicklook_seed = int(training_cfg.get("quicklook_seed", cfg["seed"]))
    base_model = raw_model(model)
    fixed_prior_latents = torch.randn(
        (
            quicklook_samples,
            base_model.config.latent_channels,
            base_model.latent_resolution,
            base_model.latent_resolution,
        ),
        generator=torch.Generator(device="cpu").manual_seed(quicklook_seed),
        dtype=torch.float32,
    ).to(device)

    for epoch in range(start_epoch, epochs + 1):
        train_model.train()
        run_state["phase"] = "train"
        run_state["epoch"] = epoch
        run_state["global_step"] = global_step
        optimizer.zero_grad(set_to_none=True)
        running = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0, "masked_reconstruction": 0.0, "perceptual": 0.0}
        num_batches = 0
        optimizer_steps_this_epoch = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            loss, metrics, _ = forward_and_loss(
                model=train_model,
                batch=batch,
                device=device,
                loss_cfg=cfg["loss"],
                sample_posterior=sample_posterior,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
                channels_last=channels_last,
                perceptual_loss_fn=perceptual_loss_fn,
            )
            scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()

            for key, value in metrics.items():
                running[key] += value
            num_batches += 1

            should_step = (batch_idx % grad_accum_steps == 0) or (batch_idx == train_batches_per_epoch)
            if should_step:
                if grad_clip_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(raw_model(model).parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                optimizer_steps_this_epoch += 1
                run_state["global_step"] = global_step
                if ema is not None:
                    ema.update(raw_model(model))

                if optimizer_steps_this_epoch % int(training_cfg.get("log_every", 50)) == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"epoch={epoch} step={global_step} "
                        f"loss={metrics['loss']:.4f} recon={metrics['reconstruction']:.4f} "
                        f"kl={metrics['kl']:.4f} perceptual={metrics['perceptual']:.4f} lr={lr:.6f}"
                    )

            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break

        if num_batches == 0:
            raise RuntimeError("Train dataloader yielded no batches.")

        train_metrics = {key: value / num_batches for key, value in running.items()}
        run_state["phase"] = "validate"
        with maybe_ema_scope(train_model, ema, evaluate_with_ema):
            val_result = run_validation(
                model=train_model,
                dataloader=val_loader,
                device=device,
                loss_cfg=cfg["loss"],
                sample_posterior=validation_sample_posterior,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
                channels_last=channels_last,
                perceptual_loss_fn=perceptual_loss_fn,
                max_batches=max_val_batches,
            )
            with torch.no_grad():
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    prior_samples = raw_model(train_model).decode(fixed_prior_latents)
        val_metrics = val_result["metrics"]

        epoch_record = {
            "epoch": epoch,
            "completed_at_utc": utc_now_iso(),
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "global_step": global_step,
            "optimizer_steps_this_epoch": optimizer_steps_this_epoch,
            "used_ema_for_validation": evaluate_with_ema,
        }
        history.append(epoch_record)
        write_json(output_dir / "history.json", {"history": history})
        append_jsonl(logs_dir / "epoch_metrics.jsonl", epoch_record)

        save_reconstruction_grid(
            inputs=val_result["images"],
            reconstructions=val_result["reconstructions"],
            output_path=output_dir / "reconstructions" / f"epoch_{epoch:03d}.png",
        )
        save_reconstruction_grid(
            inputs=val_result["images"],
            reconstructions=val_result["reconstructions"],
            output_path=output_dir / "quicklook" / "latest_reconstruction.png",
        )
        save_image_grid(
            prior_samples,
            output_path=output_dir / "prior_samples" / f"epoch_{epoch:03d}.png",
            nrow=min(4, quicklook_samples),
        )
        save_image_grid(
            prior_samples,
            output_path=output_dir / "quicklook" / "latest_prior_samples.png",
            nrow=min(4, quicklook_samples),
        )

        checkpoint_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "model_state_dict": raw_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "config": cfg,
            "history": history,
        }
        torch.save(checkpoint_payload, checkpoint_dir / "last.pt")
        run_state["last_checkpoint"] = str((checkpoint_dir / "last.pt").resolve())

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_payload["best_val_loss"] = best_val_loss
            torch.save(checkpoint_payload, checkpoint_dir / "best.pt")
            save_reconstruction_grid(
                inputs=val_result["images"],
                reconstructions=val_result["reconstructions"],
                output_path=output_dir / "quicklook" / "best_reconstruction.png",
            )
            save_image_grid(
                prior_samples,
                output_path=output_dir / "quicklook" / "best_prior_samples.png",
                nrow=min(4, quicklook_samples),
            )
            run_state["last_checkpoint"] = str((checkpoint_dir / "best.pt").resolve())

        run_state["phase"] = "epoch_complete"
        run_state["best_val_loss"] = best_val_loss
        run_state["last_epoch_metrics"] = epoch_record
        write_json(
            logs_dir / "run_status.json",
            {
                "status": "running",
                "updated_at_utc": utc_now_iso(),
                "run_state": run_state,
            },
        )
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_recon={val_metrics['reconstruction']:.4f} "
            f"val_perceptual={val_metrics['perceptual']:.4f}"
        )

    write_json(
        output_dir / "final_metrics.json",
        {
            "best_val_loss": best_val_loss,
            "history": history,
        },
    )
    run_state["phase"] = "complete"
    write_json(
        logs_dir / "run_status.json",
        {
            "status": "completed",
            "updated_at_utc": utc_now_iso(),
            "run_state": run_state,
            "best_val_loss": best_val_loss,
        },
    )
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    training_cfg = dict(cfg.get("training", {}))
    set_seed(int(cfg["seed"]), deterministic=bool(training_cfg.get("deterministic", False)))
    output_dir = Path(cfg["output_dir"])
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    logs_dir = output_dir / "logs"
    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)
    ensure_dir(logs_dir)
    try:
        return run_training(args, cfg)
    except Exception as error:
        write_failure_artifacts(
            logs_dir=logs_dir,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            args=args,
            cfg=cfg,
            error=error,
            run_state=getattr(
                args,
                "_run_state",
                {
                    "phase": "exception",
                    "note": "See traceback.txt for the full stack trace.",
                },
            ),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
