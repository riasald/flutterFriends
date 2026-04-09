#!/usr/bin/env python3
"""Audit the final assembled VAE dataset for integrity and obvious image issues."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from PIL import Image, ImageDraw, ImageOps


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
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(row: Dict[str, Any], column: str, *, dataset_root: Path) -> Path:
    value = normalize_text(row.get(column))
    if not value:
        return Path()
    candidate = Path(value)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        direct = (dataset_root / candidate).resolve()
        if direct.exists():
            return direct
        via_parent = (dataset_root.parent / candidate).resolve()
        if via_parent.exists():
            return via_parent
    return candidate


def sha1_for_path(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_rows(rows: List[Dict[str, Any]], *, sample_size: int, seed: int) -> List[Dict[str, Any]]:
    if not rows:
        return []
    rng = random.Random(seed)
    if len(rows) <= sample_size:
        return list(rows)
    return rng.sample(rows, sample_size)


def build_contact_sheet(
    rows: Sequence[Dict[str, Any]],
    *,
    dataset_root: Path,
    output_path: Path,
    title: str,
) -> None:
    if not rows:
        return

    thumb_width = 192
    thumb_height = 192
    padding = 12
    label_height = 64
    title_height = 36
    columns = 2
    rows_count = len(rows)
    canvas = Image.new(
        "RGB",
        (padding + columns * (thumb_width + padding), title_height + padding + rows_count * (thumb_height + label_height + padding)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 10), title, fill="black")

    current_y = title_height + padding
    for row in rows:
        source_path = resolve_path(row, "source_image_path", dataset_root=dataset_root)
        processed_path = resolve_path(row, "processed_image_path", dataset_root=dataset_root)

        with Image.open(source_path) as image:
            source = image.convert("RGB")
        with Image.open(processed_path) as image:
            processed = image.convert("RGB")

        source_thumb = ImageOps.contain(source, (thumb_width, thumb_height), Image.Resampling.LANCZOS)
        processed_thumb = ImageOps.contain(processed, (thumb_width, thumb_height), Image.Resampling.LANCZOS)
        source_panel = Image.new("RGB", (thumb_width, thumb_height), "white")
        processed_panel = Image.new("RGB", (thumb_width, thumb_height), "white")
        source_panel.paste(source_thumb, ((thumb_width - source_thumb.width) // 2, (thumb_height - source_thumb.height) // 2))
        processed_panel.paste(processed_thumb, ((thumb_width - processed_thumb.width) // 2, (thumb_height - processed_thumb.height) // 2))

        left_x = padding
        right_x = padding + thumb_width + padding
        canvas.paste(source_panel, (left_x, current_y))
        canvas.paste(processed_panel, (right_x, current_y))
        draw.rectangle([left_x, current_y, left_x + thumb_width, current_y + thumb_height], outline="black", width=2)
        draw.rectangle([right_x, current_y, right_x + thumb_width, current_y + thumb_height], outline="black", width=2)

        species = normalize_text(row.get("species"))[:28]
        crop_method = normalize_text(row.get("crop_method"))[:28]
        label_lines = [
            f"{species}",
            f"view={normalize_text(row.get('view_hint')) or 'unknown'} crop={crop_method}",
            f"q={parse_float(row.get('mask_quality_score')):.3f} cov={parse_float(row.get('coverage_ratio')):.3f}",
        ]
        for line_index, line in enumerate(label_lines):
            draw.text((left_x, current_y + thumb_height + 4 + line_index * 16), line, fill="black")

        current_y += thumb_height + label_height + padding

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the final assembled butterfly VAE dataset.")
    parser.add_argument("--input-csv", required=True, help="Final processed_metadata.csv path.")
    parser.add_argument("--output-dir", required=True, help="Directory for audit outputs.")
    parser.add_argument("--sample-size", type=int, default=24, help="Sample size per audit bucket.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--compute-sha1", action="store_true", help="Compute exact duplicate hashes across the full dataset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    dataset_root = input_csv.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_csv)
    counts = Counter()
    crop_counts = Counter()
    view_counts = Counter()
    source_counts = Counter()
    size_counts = Counter()
    mode_counts = Counter()
    corrupt_examples: List[Dict[str, Any]] = []

    recovered_rows: List[Dict[str, Any]] = []
    low_quality_rows: List[Dict[str, Any]] = []
    low_coverage_rows: List[Dict[str, Any]] = []
    low_symmetry_rows: List[Dict[str, Any]] = []
    random_pool = list(rows)

    hash_to_path: Dict[str, str] = {}
    duplicate_examples: List[Dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        counts["rows"] += 1
        crop_method = normalize_text(row.get("crop_method"))
        view_hint = normalize_text(row.get("view_hint")) or "unknown"
        crop_counts[crop_method] += 1
        view_counts[view_hint] += 1
        source_counts[normalize_text(row.get("source"))] += 1

        if crop_method == "recovered_from_existing_output":
            recovered_rows.append(row)
        if parse_float(row.get("mask_quality_score"), 1.0) < 0.65:
            low_quality_rows.append(row)
        if parse_float(row.get("coverage_ratio"), 1.0) < 0.03:
            low_coverage_rows.append(row)
        if normalize_text(row.get("symmetry_score")) and parse_float(row.get("symmetry_score"), 1.0) < 0.80:
            low_symmetry_rows.append(row)

        processed_path = resolve_path(row, "processed_image_path", dataset_root=dataset_root)
        try:
            with Image.open(processed_path) as image:
                size_counts[image.size] += 1
                mode_counts[image.mode] += 1
            if args.compute_sha1:
                digest = sha1_for_path(processed_path)
                if digest in hash_to_path and len(duplicate_examples) < 20:
                    duplicate_examples.append(
                        {
                            "first": hash_to_path[digest],
                            "duplicate": str(processed_path),
                            "sha1": digest,
                        }
                    )
                else:
                    hash_to_path.setdefault(digest, str(processed_path))
        except Exception as exc:  # pragma: no cover - audit only
            corrupt_examples.append(
                {
                    "row_index": index,
                    "processed_image_path": str(processed_path),
                    "error": str(exc),
                }
            )

    audit_buckets = {
        "random_overall": sample_rows(random_pool, sample_size=args.sample_size, seed=args.seed),
        "recovered_rows": sample_rows(recovered_rows, sample_size=args.sample_size, seed=args.seed + 1),
        "low_quality_rows": sample_rows(low_quality_rows, sample_size=args.sample_size, seed=args.seed + 2),
        "low_coverage_rows": sample_rows(low_coverage_rows, sample_size=args.sample_size, seed=args.seed + 3),
        "low_symmetry_rows": sample_rows(low_symmetry_rows, sample_size=args.sample_size, seed=args.seed + 4),
    }

    for bucket_name, bucket_rows in audit_buckets.items():
        if not bucket_rows:
            continue
        write_rows(output_dir / f"{bucket_name}.csv", bucket_rows)
        build_contact_sheet(
            bucket_rows,
            dataset_root=dataset_root,
            output_path=output_dir / f"{bucket_name}.png",
            title=bucket_name.replace("_", " "),
        )

    summary = {
        "input_csv": str(input_csv),
        "rows": len(rows),
        "crop_counts_top": crop_counts.most_common(20),
        "view_counts": dict(view_counts),
        "source_counts": dict(source_counts),
        "recovered_rows": len(recovered_rows),
        "low_quality_rows_lt_065": len(low_quality_rows),
        "low_coverage_rows_lt_003": len(low_coverage_rows),
        "low_symmetry_rows_lt_080": len(low_symmetry_rows),
        "corrupt_images": len(corrupt_examples),
        "corrupt_examples": corrupt_examples[:20],
        "size_counts": {str(key): value for key, value in size_counts.items()},
        "mode_counts": dict(mode_counts),
        "sha1_duplicates_found": len(duplicate_examples),
        "sha1_duplicate_examples": duplicate_examples,
        "audit_outputs": {key: str(output_dir / f"{key}.png") for key in audit_buckets if audit_buckets[key]},
    }

    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
