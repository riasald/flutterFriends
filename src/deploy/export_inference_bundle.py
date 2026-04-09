#!/usr/bin/env python3
"""Create a slim, inference-only bundle for hosting the butterfly generator."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a production-ready inference bundle.")
    parser.add_argument("--project-root", default=".", help="Repository root.")
    parser.add_argument("--output-dir", default="dist", help="Directory for the staging folder and zip.")
    parser.add_argument("--bundle-name", default="butterfly_geo_gen_inference_v1", help="Output bundle name.")
    parser.add_argument(
        "--diffusion-checkpoint",
        default="artifacts/diffusion_nymphalidae_us_v1_2026-04-05/butterfly_geo_gen_diffusion/checkpoints/runpod/diffusion_quality/best.pt",
        help="Training diffusion checkpoint to slim.",
    )
    parser.add_argument("--diffusion-config", default="configs/diffusion_runpod_quality.yaml", help="Diffusion YAML config.")
    parser.add_argument(
        "--diffusion-artifact-name",
        default="diffusion_nymphalidae_us_v1_2026-04-05",
        help="Production diffusion artifact folder name.",
    )
    parser.add_argument(
        "--vae-checkpoint",
        default="artifacts/vae_nymphalidae_us_v1_2026-04-04/checkpoints/best.pt",
        help="Training VAE checkpoint to slim.",
    )
    parser.add_argument(
        "--vae-artifact-name",
        default="vae_nymphalidae_us_v1_2026-04-04",
        help="Production VAE artifact folder name.",
    )
    parser.add_argument(
        "--prior-checkpoint",
        default="artifacts/prior_nymphalidae_us_v1_2026-04-04/checkpoints/best.pt",
        help="Training prior checkpoint to slim.",
    )
    parser.add_argument(
        "--prior-artifact-name",
        default="prior_nymphalidae_us_v1_2026-04-04",
        help="Production prior artifact folder name.",
    )
    parser.add_argument(
        "--species-vocab-json",
        default="D:/butterfly_data/nymphalidae_merged/species_to_id_final.json",
        help="Optional species vocab JSON to include for human-readable species names.",
    )
    parser.add_argument("--prefer-ema", action=argparse.BooleanOptionalAction, default=True, help="Export EMA weights when present.")
    parser.add_argument("--zip", action=argparse.BooleanOptionalAction, default=True, help="Create a zip archive.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_if_exists(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, *, ignore: Iterable[str] = ()) -> None:
    if not src.exists():
        return
    ignore_set = set(ignore)

    def ignore_func(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in ignore_set or name == "__pycache__" or name.endswith(".pyc")}

    shutil.copytree(src, dst, ignore=ignore_func, dirs_exist_ok=True)


def checkpoint_tensor_bytes(payload: Dict[str, Any]) -> int:
    total = 0

    def visit(value: Any) -> None:
        nonlocal total
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(payload)
    return total


def cpu_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def slim_checkpoint(
    *,
    source_path: Path,
    destination_path: Path,
    prefer_ema: bool,
    extra_keys: Iterable[str],
) -> Dict[str, Any]:
    checkpoint = torch.load(source_path, map_location="cpu")
    weight_key = "ema_state_dict" if prefer_ema and checkpoint.get("ema_state_dict") else "model_state_dict"
    if weight_key not in checkpoint:
        raise RuntimeError(f"{source_path} does not contain {weight_key}.")

    payload: Dict[str, Any] = {
        "format_version": "butterfly-inference-slim-v1",
        "source_checkpoint": str(source_path),
        "source_weight_key": weight_key,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_state_dict": cpu_state_dict(checkpoint[weight_key]),
    }
    for key in ("config", "epoch", "global_step", "best_val_loss", "best_val_macro_f1"):
        if key in checkpoint:
            payload[key] = checkpoint[key]
    for key in extra_keys:
        if key in checkpoint and checkpoint[key] is not None:
            value = checkpoint[key]
            if isinstance(value, dict) and all(torch.is_tensor(item) for item in value.values()):
                value = cpu_state_dict(value)
            payload[key] = value

    ensure_dir(destination_path.parent)
    torch.save(payload, destination_path)
    return {
        "source": str(source_path),
        "destination": str(destination_path),
        "source_bytes": source_path.stat().st_size,
        "destination_bytes": destination_path.stat().st_size,
        "source_weight_key": weight_key,
        "tensor_bytes": checkpoint_tensor_bytes(payload),
    }


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_yaml(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def zip_directory(source_dir: Path, zip_path: Path) -> None:
    remove_if_exists(zip_path)
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in source_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir.parent))


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = (project_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    staging_root = output_dir / f"{args.bundle_name}_staging"
    bundle_root = staging_root / args.bundle_name
    repo_root = bundle_root / "butterfly_geo_gen"
    data_root = bundle_root / "butterfly_data"
    zip_path = output_dir / f"{args.bundle_name}.zip"

    remove_if_exists(staging_root)
    remove_if_exists(zip_path)
    ensure_dir(repo_root)
    ensure_dir(data_root)

    for directory in ("configs", "src", "scripts"):
        copy_tree(project_root / directory, repo_root / directory)
    for filename in ("README.md", "requirements.txt", "requirements-runpod.txt", "pyproject.toml", ".gitignore"):
        copy_file(project_root / filename, repo_root / filename)

    diffusion_ckpt = (project_root / args.diffusion_checkpoint).resolve()
    vae_ckpt = (project_root / args.vae_checkpoint).resolve()
    prior_ckpt = (project_root / args.prior_checkpoint).resolve()

    diffusion_artifact = repo_root / "artifacts" / args.diffusion_artifact_name
    vae_artifact = repo_root / "artifacts" / args.vae_artifact_name
    prior_artifact = repo_root / "artifacts" / args.prior_artifact_name

    checkpoint_reports = {
        "diffusion": slim_checkpoint(
            source_path=diffusion_ckpt,
            destination_path=diffusion_artifact / "checkpoints" / "production.pt",
            prefer_ema=args.prefer_ema,
            extra_keys=("fuser_state_dict",),
        ),
        "vae": slim_checkpoint(
            source_path=vae_ckpt,
            destination_path=vae_artifact / "checkpoints" / "production.pt",
            prefer_ema=args.prefer_ema,
            extra_keys=(),
        ),
        "prior": slim_checkpoint(
            source_path=prior_ckpt,
            destination_path=prior_artifact / "checkpoints" / "production.pt",
            prefer_ema=False,
            extra_keys=("species_to_id", "train_class_counts"),
        ),
    }

    for src_artifact, dst_artifact in (
        (project_root / "artifacts" / args.vae_artifact_name, vae_artifact),
        (project_root / "artifacts" / args.prior_artifact_name, prior_artifact),
    ):
        for relative in ("README.md", "manifest.json"):
            copy_file(src_artifact / relative, dst_artifact / relative)
        copy_tree(src_artifact / "configs", dst_artifact / "configs")

    species_vocab = Path(args.species_vocab_json)
    if species_vocab.exists():
        copy_file(species_vocab, data_root / "nymphalidae_merged" / "species_to_id_final.json")

    diffusion_config_src = project_root / args.diffusion_config
    diffusion_config_name = "diffusion_production.yaml"
    diffusion_cfg = read_yaml(diffusion_config_src)
    diffusion_cfg["vae_locked_config"] = "${PROJECT_ROOT}/configs/vae_locked.yaml"
    diffusion_cfg["prior_locked_config"] = "${PROJECT_ROOT}/configs/prior_locked.yaml"
    diffusion_cfg["output_dir"] = "${PROJECT_ROOT}/outputs/production/diffusion"
    diffusion_cfg["checkpoint_dir"] = "${PROJECT_ROOT}/checkpoints/production/diffusion"
    diffusion_cfg.setdefault("training", {})["compile"] = False
    write_yaml(repo_root / "configs" / diffusion_config_name, diffusion_cfg)

    write_yaml(
        repo_root / "configs" / "vae_locked.yaml",
        {
            "name": "vae_production",
            "checkpoint_path": f"${{PROJECT_ROOT}}/artifacts/{args.vae_artifact_name}/checkpoints/production.pt",
            "use_ema": False,
            "freeze": True,
            "run_metadata": {
                "stage": "first_stage_autoencoder",
                "artifact_dir": f"${{PROJECT_ROOT}}/artifacts/{args.vae_artifact_name}",
                "source_checkpoint": str(vae_ckpt),
                "production_slim": True,
            },
        },
    )
    write_yaml(
        repo_root / "configs" / "prior_locked.yaml",
        {
            "name": "prior_production",
            "checkpoint_path": f"${{PROJECT_ROOT}}/artifacts/{args.prior_artifact_name}/checkpoints/production.pt",
            "freeze": True,
            "artifact_dir": f"${{PROJECT_ROOT}}/artifacts/{args.prior_artifact_name}",
            "run_metadata": {
                "stage": "coordinate_species_prior",
                "artifact_dir": f"${{PROJECT_ROOT}}/artifacts/{args.prior_artifact_name}",
                "source_checkpoint": str(prior_ckpt),
                "production_slim": True,
            },
        },
    )
    write_yaml(
        repo_root / "configs" / "diffusion_locked.yaml",
        {
            "name": "diffusion_production",
            "checkpoint_path": f"${{PROJECT_ROOT}}/artifacts/{args.diffusion_artifact_name}/checkpoints/production.pt",
            "use_ema": False,
            "config_path": f"${{PROJECT_ROOT}}/configs/{diffusion_config_name}",
            "run_metadata": {
                "stage": "coordinate_conditioned_latent_diffusion",
                "artifact_dir": f"${{PROJECT_ROOT}}/artifacts/{args.diffusion_artifact_name}",
                "source_checkpoint": str(diffusion_ckpt),
                "source_weight_key": checkpoint_reports["diffusion"]["source_weight_key"],
                "production_slim": True,
            },
            "inference_preset": {
                "guidance_scales": [1.9, 2.3, 2.7],
                "prior_temperatures": [1.0, 1.2, 1.35],
                "anchor_top_species": 6,
                "coordinate_jitter_deg": 0.35,
                "num_candidates": 16,
                "num_outputs": 8,
                "samples_per_batch": 2,
                "postprocess": True,
            },
        },
    )

    quickstart = """# Butterfly Geo Generator Inference Bundle

This bundle is inference-only. It contains slim production checkpoints, the API
server, and the final generation command. It does not contain training datasets
or optimizer state.

## Linux / GPU Host

```bash
cd /workspace/butterfly_geo_gen
pip install -e .
export PROJECT_ROOT=/workspace/butterfly_geo_gen
export BUTTERFLY_DATA_ROOT=/workspace/butterfly_data
export GENERATION_DEVICE=cuda
export GENERATION_API_PORT=8000
bash scripts/runpod_serve_generation_api.sh
```

## Local Windows Smoke Test

```powershell
powershell -ExecutionPolicy Bypass -File scripts\\run_generation_api.ps1 -Device cuda
```

## API Flow

1. `GET /health`
2. `POST /generate` with JSON like `{\"lat\":29.6516,\"lon\":-82.3248,\"num_candidates\":8,\"num_outputs\":4}`
3. Poll `GET /jobs/{job_id}` and display `selected_grid_url` when complete.
"""
    (bundle_root / "README_INFERENCE.md").write_text(quickstart, encoding="utf-8")

    manifest = {
        "bundle_name": args.bundle_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "production_repo_dir": str(repo_root),
        "production_data_dir": str(data_root),
        "checkpoints": checkpoint_reports,
        "zip_path": str(zip_path),
    }
    (bundle_root / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.zip:
        zip_directory(bundle_root, zip_path)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

