#!/usr/bin/env python3
"""Train the coordinate-conditioned butterfly species prior."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.coordinate_dataset import CoordinateSpeciesDataset
from src.losses.prior_losses import LDAMLoss, topk_accuracies, weighting_strategy_for_epoch
from src.models.fourier_encoder import FourierLocationEncoderConfig
from src.models.species_prior import LocationSpeciesPriorModel, SpeciesPriorHeadConfig
from src.utils.config import load_yaml_config
from src.utils.geo import GeoBounds


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
    (logs_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(
        logs_dir / "failure_report.json",
        {
            "status": "failed",
            "failed_at_utc": utc_now_iso(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "config_path": str(Path(args.config).resolve()),
            "output_dir": str(output_dir.resolve()),
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "run_state": run_state,
            "training_config": cfg.get("training", {}),
            "loss_config": cfg.get("loss", {}),
        },
    )
    write_json(
        logs_dir / "run_status.json",
        {
            "status": "failed",
            "updated_at_utc": utc_now_iso(),
            "run_state": run_state,
        },
    )


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


def build_geo_bounds(data_cfg: Dict[str, Any]) -> GeoBounds:
    return GeoBounds(
        lat_min=float(data_cfg.get("lat_min", 18.5)),
        lat_max=float(data_cfg.get("lat_max", 72.5)),
        lon_min=float(data_cfg.get("lon_min", -180.0)),
        lon_max=float(data_cfg.get("lon_max", -66.0)),
    )


def build_datasets(data_cfg: Dict[str, Any]) -> dict[str, CoordinateSpeciesDataset]:
    bounds = build_geo_bounds(data_cfg)
    return {
        "train": CoordinateSpeciesDataset(data_cfg["train_csv"], geo_bounds=bounds),
        "val": CoordinateSpeciesDataset(data_cfg["val_csv"], geo_bounds=bounds),
        "test": CoordinateSpeciesDataset(data_cfg["test_csv"], geo_bounds=bounds),
    }


def build_loader(
    dataset: CoordinateSpeciesDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
    return DataLoader(**kwargs)


def build_weighted_sampler(dataset: CoordinateSpeciesDataset) -> WeightedRandomSampler | None:
    if len(dataset) == 0:
        return None
    class_counts = Counter(dataset.species_ids)
    sample_weights = torch.tensor(
        [1.0 / class_counts[int(species_id)] for species_id in dataset.species_ids],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def load_species_vocab(path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    vocab = json.loads(Path(path).read_text(encoding="utf-8"))
    species_to_id = {str(key): int(value) for key, value in vocab.items()}
    id_to_species = {value: key for key, value in species_to_id.items()}
    return species_to_id, id_to_species


def class_counts_from_dataset(dataset: CoordinateSpeciesDataset, *, num_species: int) -> torch.Tensor:
    counts = torch.zeros(num_species, dtype=torch.long)
    for species_id in dataset.species_ids:
        counts[int(species_id)] += 1
    return counts


def build_model(cfg: Dict[str, Any], num_species: int, device: torch.device) -> torch.nn.Module:
    loc_cfg = FourierLocationEncoderConfig(**cfg["model"]["location_encoder"])
    prior_cfg_dict = dict(cfg["model"]["prior_head"])
    prior_cfg_dict["num_species"] = num_species
    prior_cfg = SpeciesPriorHeadConfig(**prior_cfg_dict)
    model = LocationSpeciesPriorModel(loc_cfg, prior_cfg).to(device)
    return model


def build_loss_fn(
    *,
    class_counts: torch.Tensor,
    loss_cfg: Dict[str, Any],
    epoch: int,
    total_epochs: int,
    device: torch.device,
) -> LDAMLoss:
    active_weighting = weighting_strategy_for_epoch(
        strategy=str(loss_cfg.get("class_weight_strategy", "effective_num")),
        epoch=epoch,
        total_epochs=total_epochs,
        drw_start_epoch=int(loss_cfg.get("drw_start_epoch", 0)),
    )
    return LDAMLoss(
        class_counts,
        max_margin=float(loss_cfg.get("max_margin", 0.5)),
        scale=float(loss_cfg.get("scale", 30.0)),
        class_weight_strategy=active_weighting,
        class_weight_beta=float(loss_cfg.get("class_weight_beta", 0.9999)),
    ).to(device)


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


def forward_loss(
    *,
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    loss_fn: torch.nn.Module,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, Any]]:
    latlon_norm = batch["latlon_norm"].to(device, non_blocking=True)
    targets = batch["species_id"].to(device, non_blocking=True)
    if not torch.isfinite(latlon_norm).all():
        raise RuntimeError("Encountered non-finite normalized coordinates in the prior batch.")
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
        outputs = model(latlon_norm)
        logits = outputs["logits"]
        loss = loss_fn(logits, targets)
    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "Prior model produced non-finite logits. "
            f"logits_min={float(logits.detach().amin().cpu().item()):.6f} "
            f"logits_max={float(logits.detach().amax().cpu().item()):.6f}"
        )
    if not torch.isfinite(loss):
        raise RuntimeError(
            "Prior loss became non-finite. "
            f"loss={float(loss.detach().cpu().item())} "
            f"logits_min={float(logits.detach().amin().cpu().item()):.6f} "
            f"logits_max={float(logits.detach().amax().cpu().item()):.6f}"
        )
    topk = topk_accuracies(logits.detach(), targets.detach(), ks=(1, 5))
    preds = logits.detach().argmax(dim=1)
    return loss, {
        "logits": logits.detach(),
        "preds": preds,
        "targets": targets.detach(),
        "top1": topk[1],
        "top5": topk[5],
    }


def frequency_band(count: int) -> str:
    if count < 10:
        return "rare"
    if count < 30:
        return "tail"
    if count < 100:
        return "medium"
    return "common"


def summarize_frequency_bands(
    *,
    train_class_counts: torch.Tensor,
    per_species_recall: dict[int, float],
    eval_true_counts: Counter[int],
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {"rare": [], "tail": [], "medium": [], "common": []}
    grouped_rows: Counter[str] = Counter()
    for species_id, count in enumerate(train_class_counts.tolist()):
        if species_id not in per_species_recall:
            continue
        band = frequency_band(int(count))
        grouped[band].append(float(per_species_recall[species_id]))
        grouped_rows[band] += int(eval_true_counts.get(species_id, 0))
    return {
        band: {
            "species": len(recalls),
            "rows": int(grouped_rows[band]),
            "mean_recall": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        }
        for band, recalls in grouped.items()
    }


def evaluate_model(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
    max_batches: int,
    train_class_counts: torch.Tensor,
    id_to_species: dict[int, str],
) -> Dict[str, Any]:
    model.eval()
    loss_total = 0.0
    top1_total = 0.0
    top5_total = 0.0
    example_count = 0
    preds_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            loss, payload = forward_loss(
                model=model,
                batch=batch,
                device=device,
                loss_fn=loss_fn,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
            )
            batch_size = int(payload["targets"].shape[0])
            loss_total += float(loss.detach().cpu().item()) * batch_size
            top1_total += float(payload["top1"]) * batch_size
            top5_total += float(payload["top5"]) * batch_size
            example_count += batch_size
            preds_all.append(payload["preds"].cpu())
            targets_all.append(payload["targets"].cpu())
            if max_batches > 0 and batch_idx >= max_batches:
                break

    if example_count == 0:
        raise RuntimeError("Evaluation dataloader yielded no batches.")

    preds = torch.cat(preds_all).numpy()
    targets = torch.cat(targets_all).numpy()
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    true_counts = Counter(int(value) for value in targets.tolist())
    correct_counts = Counter()
    for target, pred in zip(targets.tolist(), preds.tolist()):
        if int(target) == int(pred):
            correct_counts[int(target)] += 1
    per_species_recall = {
        species_id: float(correct_counts.get(species_id, 0) / count)
        for species_id, count in true_counts.items()
    }
    per_species_metrics = {
        id_to_species.get(species_id, str(species_id)): {
            "species_id": int(species_id),
            "eval_rows": int(true_counts[species_id]),
            "recall": float(per_species_recall[species_id]),
            "train_rows": int(train_class_counts[species_id].item()),
            "frequency_band": frequency_band(int(train_class_counts[species_id].item())),
        }
        for species_id in sorted(per_species_recall)
    }
    metrics = {
        "loss": float(loss_total / example_count),
        "top1": float(top1_total / example_count),
        "top5": float(top5_total / example_count),
        "macro_f1": macro_f1,
    }
    return {
        "metrics": metrics,
        "per_species_metrics": per_species_metrics,
        "frequency_band_summary": summarize_frequency_bands(
            train_class_counts=train_class_counts,
            per_species_recall=per_species_recall,
            eval_true_counts=true_counts,
        ),
    }


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    base_model = raw_model(model)
    total = sum(parameter.numel() for parameter in base_model.parameters())
    trainable = sum(parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the location encoder + species prior.")
    parser.add_argument("--config", required=True, help="Path to prior YAML config.")
    parser.add_argument("--device", default="auto", help="Device string, for example 'cuda' or 'cpu'.")
    parser.add_argument("--resume", default="", help="Optional checkpoint path to resume from.")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Optional cap for smoke tests.")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Optional cap for smoke tests.")
    return parser.parse_args()


def run_training(args: argparse.Namespace, cfg: Dict[str, Any]) -> int:
    training_cfg = dict(cfg.get("training", {}))
    data_cfg = load_yaml_config(cfg["data_config"])
    output_dir = Path(cfg["output_dir"])
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    logs_dir = output_dir / "logs"
    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)
    ensure_dir(logs_dir)

    device = resolve_device(args.device)
    configure_runtime(training_cfg, device)
    datasets = build_datasets(data_cfg)
    species_to_id, id_to_species = load_species_vocab(data_cfg["species_vocab_json"])
    num_species = len(species_to_id)
    train_class_counts = class_counts_from_dataset(datasets["train"], num_species=num_species)

    weighted_sampling = bool(training_cfg.get("weighted_sampling", False))
    train_sampler = build_weighted_sampler(datasets["train"]) if weighted_sampling else None
    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(data_cfg.get("batch_size", 256)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)) and int(data_cfg.get("num_workers", 0)) > 0,
        prefetch_factor=(max(2, int(data_cfg.get("prefetch_factor", 2))) if int(data_cfg.get("num_workers", 0)) > 0 else None),
    )
    val_loader = build_loader(
        datasets["val"],
        batch_size=int(data_cfg.get("eval_batch_size", data_cfg.get("batch_size", 256))),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
    )

    model = build_model(cfg, num_species, device)
    optimizer = build_optimizer(model, cfg["optimizer"], device)
    epochs = int(training_cfg["epochs"])
    max_train_batches = int(args.max_train_batches)
    max_val_batches = int(args.max_val_batches)
    train_batches_per_epoch = max_train_batches if max_train_batches > 0 else len(train_loader)
    total_optimizer_steps = epochs * max(1, train_batches_per_epoch)
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}), total_optimizer_steps=total_optimizer_steps)
    autocast_dtype = resolve_amp_dtype(training_cfg.get("amp_dtype", "float16"))
    autocast_enabled = device.type == "cuda" and bool(training_cfg.get("mixed_precision", True))
    use_grad_scaler = autocast_enabled and autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    train_model = maybe_compile_model(model, training_cfg)

    history: list[dict[str, Any]] = []
    best_val_macro_f1 = float("-inf")
    global_step = 0
    start_epoch = 1
    run_state: Dict[str, Any] = {"phase": "initializing", "epoch": 0, "global_step": 0}
    setattr(args, "_run_state", run_state)

    write_json(
        logs_dir / "run_metadata.json",
        {
            "created_at_utc": utc_now_iso(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "config_path": str(Path(args.config).resolve()),
            "dataset_rows": {split: len(dataset) for split, dataset in datasets.items()},
            "num_species": num_species,
            "weighted_sampling": weighted_sampling,
            "class_count_range": {
                "min": int(train_class_counts[train_class_counts > 0].min().item()),
                "max": int(train_class_counts.max().item()),
            },
            "parameter_counts": count_parameters(model),
        },
    )
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
        history = checkpoint.get("history", [])
        best_val_macro_f1 = float(checkpoint.get("best_val_macro_f1", best_val_macro_f1))
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 0.0))

    for epoch in range(start_epoch, epochs + 1):
        loss_fn = build_loss_fn(
            class_counts=train_class_counts,
            loss_cfg=cfg["loss"],
            epoch=epoch,
            total_epochs=epochs,
            device=device,
        )
        train_model.train()
        run_state.update({"phase": "train", "epoch": epoch, "global_step": global_step})
        train_loss_total = 0.0
        train_top1_total = 0.0
        train_top5_total = 0.0
        example_count = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader, start=1):
            loss, payload = forward_loss(
                model=train_model,
                batch=batch,
                device=device,
                loss_fn=loss_fn,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
            )
            scaler.scale(loss).backward()
            if grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model(model).parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            run_state["global_step"] = global_step

            batch_size = int(payload["targets"].shape[0])
            train_loss_total += float(loss.detach().cpu().item()) * batch_size
            train_top1_total += float(payload["top1"]) * batch_size
            train_top5_total += float(payload["top5"]) * batch_size
            example_count += batch_size

            if global_step % int(training_cfg.get("log_every", 50)) == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={float(loss.detach().cpu().item()):.4f} "
                    f"top1={float(payload['top1']):.4f} top5={float(payload['top5']):.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}"
                )

            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break

        if example_count == 0:
            raise RuntimeError("Train dataloader yielded no batches.")

        train_metrics = {
            "loss": float(train_loss_total / example_count),
            "top1": float(train_top1_total / example_count),
            "top5": float(train_top5_total / example_count),
        }
        run_state["phase"] = "validate"
        val_result = evaluate_model(
            model=train_model,
            dataloader=val_loader,
            device=device,
            loss_fn=loss_fn,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
            max_batches=max_val_batches,
            train_class_counts=train_class_counts,
            id_to_species=id_to_species,
        )

        epoch_record = {
            "epoch": epoch,
            "completed_at_utc": utc_now_iso(),
            "train": train_metrics,
            "val": val_result["metrics"],
            "val_frequency_bands": val_result["frequency_band_summary"],
            "lr": optimizer.param_groups[0]["lr"],
            "global_step": global_step,
            "active_class_weight_strategy": weighting_strategy_for_epoch(
                strategy=str(cfg["loss"].get("class_weight_strategy", "effective_num")),
                epoch=epoch,
                total_epochs=epochs,
                drw_start_epoch=int(cfg["loss"].get("drw_start_epoch", 0)),
            ),
        }
        history.append(epoch_record)
        write_json(output_dir / "history.json", {"history": history})
        append_jsonl(logs_dir / "epoch_metrics.jsonl", epoch_record)
        write_json(output_dir / "val_per_species_metrics.json", val_result["per_species_metrics"])
        write_json(output_dir / "val_frequency_band_summary.json", val_result["frequency_band_summary"])

        checkpoint_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_macro_f1": best_val_macro_f1,
            "model_state_dict": raw_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "config": cfg,
            "history": history,
            "train_class_counts": train_class_counts,
            "species_to_id": species_to_id,
        }
        torch.save(checkpoint_payload, checkpoint_dir / "last.pt")
        run_state["last_checkpoint"] = str((checkpoint_dir / "last.pt").resolve())

        val_macro_f1 = float(val_result["metrics"]["macro_f1"])
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            checkpoint_payload["best_val_macro_f1"] = best_val_macro_f1
            torch.save(checkpoint_payload, checkpoint_dir / "best.pt")
            write_json(output_dir / "best_val_per_species_metrics.json", val_result["per_species_metrics"])
            write_json(output_dir / "best_val_frequency_band_summary.json", val_result["frequency_band_summary"])
            run_state["last_checkpoint"] = str((checkpoint_dir / "best.pt").resolve())

        run_state.update(
            {
                "phase": "epoch_complete",
                "best_val_macro_f1": best_val_macro_f1,
                "last_epoch_metrics": epoch_record,
            }
        )
        write_json(
            logs_dir / "run_status.json",
            {
                "status": "running",
                "updated_at_utc": utc_now_iso(),
                "run_state": run_state,
            },
        )

        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} train_top1={train_metrics['top1']:.4f} "
            f"val_loss={val_result['metrics']['loss']:.4f} val_top1={val_result['metrics']['top1']:.4f} "
            f"val_top5={val_result['metrics']['top5']:.4f} val_macro_f1={val_result['metrics']['macro_f1']:.4f}"
        )

    write_json(
        output_dir / "final_metrics.json",
        {
            "best_val_macro_f1": best_val_macro_f1,
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
            "best_val_macro_f1": best_val_macro_f1,
        },
    )
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    set_seed(int(cfg["seed"]), deterministic=bool(cfg.get("training", {}).get("deterministic", False)))
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
