#!/usr/bin/env python3
"""Filter merged preprocessed images down to safer VAE training examples."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
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
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def group_key_for_row(row: Dict[str, Any]) -> str:
    return (
        normalize_text(row.get("specimen_key"))
        or normalize_text(row.get("record_id"))
        or normalize_text(row.get("image_url"))
        or normalize_text(row.get("image_path"))
    )


def is_preprocess_ok(row: Dict[str, Any]) -> bool:
    return (
        normalize_text(row.get("preprocess_status")) == "ok"
        and normalize_text(row.get("keep_for_training")).lower() == "true"
    )


def is_rectangular_false_positive(row: Dict[str, Any]) -> bool:
    polygon_vertices = parse_int(row.get("polygon_vertices"))
    mask_solidity = parse_float(row.get("mask_solidity"))
    fill_ratio = parse_float(row.get("fill_ratio"))
    view_hint = normalize_text(row.get("view_hint")).lower()
    return (
        view_hint in {"", "unknown"}
        and polygon_vertices > 0
        and polygon_vertices <= 6
        and mask_solidity >= 0.90
        and fill_ratio >= 0.80
    )


def rank_score(row: Dict[str, Any]) -> float:
    quality = parse_float(row.get("mask_quality_score"))
    saturation = min(parse_float(row.get("mask_mean_saturation")), 140.0) / 140.0
    coverage = parse_float(row.get("coverage_ratio"))
    polygon_vertices = min(parse_int(row.get("polygon_vertices")), 12) / 12.0
    fill_ratio = parse_float(row.get("fill_ratio"))
    view_hint = normalize_text(row.get("view_hint")).lower()

    score = quality
    score += 0.35 if view_hint == "dorsal" else 0.0
    score -= 0.10 if view_hint == "unknown" else 0.0
    score += 0.20 * saturation
    score += 0.15 * polygon_vertices
    score += 0.10 * min(max(coverage, 0.0), 0.35) / 0.35
    score -= 0.15 if fill_ratio >= 0.88 else 0.0
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter merged preprocessed butterfly images for VAE training.")
    parser.add_argument("--input-csv", required=True, help="Merged processed metadata CSV.")
    parser.add_argument("--output-csv", required=True, help="Filtered CSV for VAE training.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON output.")
    parser.add_argument(
        "--max-images-per-specimen",
        type=int,
        default=1,
        help="Keep at most this many images per specimen/record group.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(Path(args.input_csv))

    filtered_candidates: List[Dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for row in rows:
        if not is_preprocess_ok(row):
            rejection_counts[f"preprocess:{normalize_text(row.get('preprocess_status')) or 'other'}"] += 1
            continue
        if is_rectangular_false_positive(row):
            rejection_counts["rectangular_false_positive"] += 1
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
            row = dict(row)
            row["vae_rank_score"] = f"{rank_score(row):.4f}"
            row["vae_group_key"] = group_key
            row["vae_group_size"] = str(len(group_rows))
            row["vae_group_rank"] = str(index + 1)
            keep = index < max(1, args.max_images_per_specimen)
            row["vae_keep"] = str(keep)
            if keep:
                kept_rows.append(row)
            else:
                duplicate_drop_count += 1

    kept_rows.sort(key=lambda row: int(normalize_text(row.get("source_row_index")) or "0"))
    write_rows(Path(args.output_csv), kept_rows)

    summary = {
        "input_csv": args.input_csv,
        "output_csv": args.output_csv,
        "input_rows": len(rows),
        "preprocess_ok_rows": len(filtered_candidates) + rejection_counts["rectangular_false_positive"],
        "rectangular_false_positive_rejections": rejection_counts["rectangular_false_positive"],
        "duplicate_drop_count": duplicate_drop_count,
        "kept_rows": len(kept_rows),
        "unique_groups": len(grouped),
        "rejection_counts": dict(rejection_counts),
    }

    summary_path = Path(args.summary_json) if args.summary_json else Path(args.output_csv).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
