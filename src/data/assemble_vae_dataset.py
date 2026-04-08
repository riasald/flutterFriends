#!/usr/bin/env python3
"""Assemble a portable VAE-ready dataset from multiple preprocessing outputs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

from src.data.build_splits import assign_species_groups, build_group_maps, summarize_split
from src.data.filter_preprocessed_for_vae import (
    group_key_for_row,
    is_preprocess_ok,
    is_rectangular_false_positive,
    rank_score,
)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        return [dict(row) for row in csv.DictReader(file_handle)]


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clean_internal_fields(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        cleaned.append({key: value for key, value in row.items() if not key.startswith("_")})
    return cleaned


def expand_inputs(values: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for value in values:
        matches = [Path(item) for item in sorted(glob.glob(value))]
        if not matches:
            matches = [Path(value)]
        for path in matches:
            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    return paths


def dedupe_rows(rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    duplicate_count = 0
    for row in rows:
        key = (
            normalize_text(row.get("source_image_path"))
            or normalize_text(row.get("image_url"))
            or normalize_text(row.get("image_path"))
            or normalize_text(row.get("processed_image_path"))
            or normalize_text(row.get("record_id")) + "|" + normalize_text(row.get("media_id"))
            or normalize_text(row.get("_source_manifest_path")) + "|" + normalize_text(row.get("source_row_index"))
        )
        if not key:
            key = str(len(deduped) + duplicate_count)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        deduped.append(dict(row))
    return deduped, duplicate_count


def resolve_existing_path(value: str, *, base_dir: Path) -> str:
    candidate = Path(value)
    if candidate.exists():
        return str(candidate)
    if not candidate.is_absolute():
        relative_candidate = (base_dir / candidate).resolve()
        if relative_candidate.exists():
            return str(relative_candidate)
    return ""


def resolve_processed_source_path(row: Dict[str, Any], *, base_dir: Path) -> str:
    for column in ("processed_image_path", "image_path"):
        value = normalize_text(row.get(column))
        if not value:
            continue
        resolved = resolve_existing_path(value, base_dir=base_dir)
        if resolved:
            return resolved
    return ""


def filter_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    max_images_per_specimen: int,
    min_mask_quality: float,
    min_coverage_ratio: float,
    min_symmetry_score: float,
    drop_recovered_output: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    filtered_candidates: List[Dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()

    for row in rows:
        if not is_preprocess_ok(row):
            rejection_counts[f"preprocess:{normalize_text(row.get('preprocess_status')) or 'other'}"] += 1
            continue
        if is_rectangular_false_positive(row):
            rejection_counts["rectangular_false_positive"] += 1
            continue
        crop_method = normalize_text(row.get("crop_method"))
        if drop_recovered_output and crop_method == "recovered_from_existing_output":
            rejection_counts["recovered_output"] += 1
            continue
        mask_quality_text = normalize_text(row.get("mask_quality_score"))
        if mask_quality_text and parse_float(mask_quality_text, 0.0) < min_mask_quality:
            rejection_counts["mask_quality_below_threshold"] += 1
            continue
        coverage_text = normalize_text(row.get("coverage_ratio"))
        if coverage_text and parse_float(coverage_text, 0.0) < min_coverage_ratio:
            rejection_counts["coverage_below_threshold"] += 1
            continue
        symmetry_text = normalize_text(row.get("symmetry_score"))
        if symmetry_text and parse_float(symmetry_text, 0.0) < min_symmetry_score:
            rejection_counts["symmetry_below_threshold"] += 1
            continue
        filtered_candidates.append(dict(row))

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in filtered_candidates:
        grouped[group_key_for_row(row)].append(row)

    kept_rows: List[Dict[str, Any]] = []
    duplicate_drop_count = 0
    for group_key, group_rows in grouped.items():
        ranked_rows = sorted(group_rows, key=rank_score, reverse=True)
        for index, row in enumerate(ranked_rows):
            item = dict(row)
            item["vae_rank_score"] = f"{rank_score(item):.4f}"
            item["vae_group_key"] = group_key
            item["vae_group_size"] = str(len(group_rows))
            item["vae_group_rank"] = str(index + 1)
            keep = index < max(1, max_images_per_specimen)
            item["vae_keep"] = str(keep)
            if keep:
                kept_rows.append(item)
            else:
                duplicate_drop_count += 1

    kept_rows.sort(key=lambda row: int(normalize_text(row.get("source_row_index")) or "0"))
    summary = {
        "preprocess_ok_rows": len(filtered_candidates) + rejection_counts["rectangular_false_positive"],
        "rectangular_false_positive_rejections": rejection_counts["rectangular_false_positive"],
        "duplicate_drop_count": duplicate_drop_count,
        "kept_rows": len(kept_rows),
        "unique_groups": len(grouped),
        "rejection_counts": dict(rejection_counts),
    }
    return kept_rows, summary


def ensure_unique_destination(destination: Path, *, source_row_index: str) -> Path:
    if not destination.exists():
        return destination
    suffix = destination.suffix
    stem = destination.stem
    parent = destination.parent
    candidate = parent / f"{stem}_{source_row_index or 'dup'}{suffix}"
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        candidate = parent / f"{stem}_{source_row_index or 'dup'}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def link_or_copy_file(source: Path, destination: Path, *, link_mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"

    if link_mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy_fallback"

    shutil.copy2(source, destination)
    return "copy"


def materialize_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    output_dir: Path,
    link_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows_list = list(rows)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    materialized_rows: List[Dict[str, Any]] = []
    mode_counts: Counter[str] = Counter()
    missing_source = 0
    renamed_collisions = 0

    for row in rows_list:
        source_csv_path = normalize_text(row.get("_source_manifest_path"))
        base_dir = Path(source_csv_path).parent if source_csv_path else Path.cwd()
        source_path = resolve_processed_source_path(row, base_dir=base_dir)
        if not source_path:
            missing_source += 1
            continue

        source = Path(source_path)
        destination = images_dir / source.name
        final_destination = ensure_unique_destination(
            destination,
            source_row_index=normalize_text(row.get("source_row_index")),
        )
        if final_destination != destination:
            renamed_collisions += 1

        mode_used = link_or_copy_file(source, final_destination, link_mode=link_mode)
        mode_counts[mode_used] += 1

        item = dict(row)
        item["original_processed_image_path"] = normalize_text(row.get("processed_image_path"))
        relative_path = final_destination.relative_to(output_dir).as_posix()
        item["processed_image_path"] = relative_path
        item["processed_image_absolute_path"] = str(final_destination)
        item["image_path"] = relative_path
        materialized_rows.append(item)

    summary = {
        "input_rows": len(rows_list),
        "materialized_rows": len(materialized_rows),
        "missing_source_rows": missing_source,
        "collision_renames": renamed_collisions,
        "link_mode": link_mode,
        "link_mode_counts": dict(mode_counts),
        "images_dir": str(images_dir),
    }
    return materialized_rows, summary


def compute_image_stats(rows: Iterable[Dict[str, Any]], *, dataset_root: Path) -> Dict[str, Any]:
    rows_list = list(rows)
    if not rows_list:
        raise RuntimeError("Cannot compute image stats with zero rows.")

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for row in rows_list:
        image_path = resolve_existing_path(normalize_text(row.get("processed_image_path")), base_dir=dataset_root)
        if not image_path:
            raise FileNotFoundError(f"Missing image for stats: {row.get('processed_image_path')}")
        with Image.open(image_path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        pixels = array.reshape(-1, 3)
        channel_sum += pixels.sum(axis=0)
        channel_sq_sum += np.square(pixels).sum(axis=0)
        total_pixels += pixels.shape[0]

    mean = channel_sum / total_pixels
    variance = np.maximum(channel_sq_sum / total_pixels - np.square(mean), 0.0)
    std = np.sqrt(variance)
    return {
        "images_used": len(rows_list),
        "total_pixels": int(total_pixels),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble a VAE-ready dataset from multiple preprocessing outputs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input CSV paths or globs.")
    parser.add_argument("--output-dir", required=True, help="Output dataset root.")
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to materialize final training images.",
    )
    parser.add_argument(
        "--max-images-per-specimen",
        type=int,
        default=1,
        help="Maximum number of images to keep per specimen group.",
    )
    parser.add_argument(
        "--min-mask-quality",
        type=float,
        default=0.0,
        help="Reject rows with mask_quality_score below this threshold when present.",
    )
    parser.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=0.0,
        help="Reject rows with coverage_ratio below this threshold when present.",
    )
    parser.add_argument(
        "--min-symmetry-score",
        type=float,
        default=0.0,
        help="Reject rows with symmetry_score below this threshold when present.",
    )
    parser.add_argument(
        "--drop-recovered-output",
        action="store_true",
        help="Reject rows carried over from older recovered preprocessing outputs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split assignment.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction.")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Test split fraction.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stats").mkdir(parents=True, exist_ok=True)
    (output_dir / "splits").mkdir(parents=True, exist_ok=True)

    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        raise RuntimeError("No input manifests were found.")

    merged_rows: List[Dict[str, Any]] = []
    input_counts: Dict[str, int] = {}
    for path in input_paths:
        rows = read_rows(path)
        input_counts[str(path)] = len(rows)
        for row in rows:
            item = dict(row)
            item["_source_manifest_path"] = str(path)
            merged_rows.append(item)

    deduped_rows, duplicate_count = dedupe_rows(merged_rows)
    deduped_rows.sort(key=lambda row: int(normalize_text(row.get("source_row_index")) or "0"))
    raw_csv = output_dir / "processed_metadata_raw.csv"
    write_rows(raw_csv, clean_internal_fields(deduped_rows))

    filtered_rows, filter_summary = filter_rows(
        deduped_rows,
        max_images_per_specimen=args.max_images_per_specimen,
        min_mask_quality=float(args.min_mask_quality),
        min_coverage_ratio=float(args.min_coverage_ratio),
        min_symmetry_score=float(args.min_symmetry_score),
        drop_recovered_output=bool(args.drop_recovered_output),
    )
    filtered_source_csv = output_dir / "processed_metadata_filtered_source.csv"
    write_rows(filtered_source_csv, clean_internal_fields(filtered_rows))

    materialized_rows, materialize_summary = materialize_rows(
        filtered_rows,
        output_dir=output_dir,
        link_mode=args.link_mode,
    )
    final_csv = output_dir / "processed_metadata.csv"
    write_rows(final_csv, clean_internal_fields(materialized_rows))

    rows_by_group, groups_by_species = build_group_maps(materialized_rows)
    assignments = assign_species_groups(
        groups_by_species,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    split_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for group_key, group_rows in rows_by_group.items():
        split_name = assignments.get(group_key, "train")
        for row in group_rows:
            item = {key: value for key, value in row.items() if not key.startswith("_")}
            item["split"] = split_name
            split_rows[split_name].append(item)

    for split_name in ("train", "val", "test"):
        split_rows[split_name].sort(key=lambda row: int(normalize_text(row.get("source_row_index")) or "0"))
        write_rows(output_dir / "splits" / f"{split_name}.csv", split_rows[split_name])

    split_summary = {
        "train": summarize_split(split_rows["train"]),
        "val": summarize_split(split_rows["val"]),
        "test": summarize_split(split_rows["test"]),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
    }
    write_json(output_dir / "splits" / "split_summary.json", split_summary)

    image_stats = compute_image_stats(split_rows["train"], dataset_root=output_dir)
    image_stats["input_csv"] = str(output_dir / "splits" / "train.csv")
    write_json(output_dir / "stats" / "train_image_stats.json", image_stats)

    assembly_summary = {
        "inputs": input_counts,
        "input_rows_before_dedupe": len(merged_rows),
        "duplicate_rows_dropped": duplicate_count,
        "rows_after_dedupe": len(deduped_rows),
        "filter_summary": filter_summary,
        "filter_config": {
            "max_images_per_specimen": int(args.max_images_per_specimen),
            "min_mask_quality": float(args.min_mask_quality),
            "min_coverage_ratio": float(args.min_coverage_ratio),
            "min_symmetry_score": float(args.min_symmetry_score),
            "drop_recovered_output": bool(args.drop_recovered_output),
        },
        "materialize_summary": materialize_summary,
        "final_rows": len(materialized_rows),
        "split_summary": split_summary,
        "image_stats_json": str(output_dir / "stats" / "train_image_stats.json"),
        "final_csv": str(final_csv),
    }
    write_json(output_dir / "assembly_summary.json", assembly_summary)
    print(json.dumps(assembly_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
