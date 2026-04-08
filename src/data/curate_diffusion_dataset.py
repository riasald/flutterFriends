#!/usr/bin/env python3
"""Filter an existing diffusion dataset down to cleaner butterfly crops for higher-quality retraining."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write for {path}.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_optional_float(value: Any) -> float | None:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def passes_quality_filters(row: Dict[str, Any], args: argparse.Namespace, rejection_counts: Counter[str]) -> bool:
    mask_quality = parse_optional_float(row.get("mask_quality_score"))
    if mask_quality is None or mask_quality < float(args.min_mask_quality_score):
        rejection_counts["mask_quality_score"] += 1
        return False

    symmetry = parse_optional_float(row.get("symmetry_score"))
    if symmetry is None or symmetry < float(args.min_symmetry_score):
        rejection_counts["symmetry_score"] += 1
        return False

    fill_ratio = parse_optional_float(row.get("fill_ratio"))
    if fill_ratio is None or fill_ratio < float(args.min_fill_ratio):
        rejection_counts["fill_ratio"] += 1
        return False

    coverage_ratio = parse_optional_float(row.get("coverage_ratio"))
    if coverage_ratio is None or coverage_ratio < float(args.min_coverage_ratio):
        rejection_counts["coverage_ratio_low"] += 1
        return False
    if coverage_ratio > float(args.max_coverage_ratio):
        rejection_counts["coverage_ratio_high"] += 1
        return False

    horizontal_ratio = parse_optional_float(row.get("horizontal_ratio"))
    if horizontal_ratio is None or horizontal_ratio < float(args.min_horizontal_ratio):
        rejection_counts["horizontal_ratio"] += 1
        return False

    wing_balance_ratio = parse_optional_float(row.get("wing_balance_ratio"))
    if wing_balance_ratio is None or wing_balance_ratio < float(args.min_wing_balance_ratio):
        rejection_counts["wing_balance_ratio"] += 1
        return False

    if args.drop_border_touch:
        touches_border = normalize_text(row.get("mask_touches_border")).lower()
        if touches_border in {"true", "1", "yes"}:
            rejection_counts["mask_touches_border"] += 1
            return False

    allowed_view_statuses = {status.strip() for status in args.allowed_view_statuses if status.strip()}
    if allowed_view_statuses:
        view_status = normalize_text(row.get("view_status"))
        if view_status and view_status not in allowed_view_statuses:
            rejection_counts["view_status"] += 1
            return False

    return True


def curate_split(rows: Iterable[Dict[str, Any]], args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    kept: List[Dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for row in rows:
        if passes_quality_filters(row, args, rejection_counts):
            kept.append(dict(row))
    return kept, dict(rejection_counts)


def build_summary_payload(
    *,
    input_dir: Path,
    output_dir: Path,
    split_rows: Dict[str, List[Dict[str, Any]]],
    split_rejections: Dict[str, Dict[str, int]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    split_summaries: Dict[str, Any] = {}
    for split_name, rows in split_rows.items():
        species = {normalize_text(row.get("species")) for row in rows if normalize_text(row.get("species"))}
        split_summaries[split_name] = {
            "rows": len(rows),
            "species": len(species),
            "rejections": split_rejections[split_name],
        }
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "thresholds": {
            "min_mask_quality_score": float(args.min_mask_quality_score),
            "min_symmetry_score": float(args.min_symmetry_score),
            "min_fill_ratio": float(args.min_fill_ratio),
            "min_coverage_ratio": float(args.min_coverage_ratio),
            "max_coverage_ratio": float(args.max_coverage_ratio),
            "min_horizontal_ratio": float(args.min_horizontal_ratio),
            "min_wing_balance_ratio": float(args.min_wing_balance_ratio),
            "drop_border_touch": bool(args.drop_border_touch),
            "allowed_view_statuses": list(args.allowed_view_statuses),
        },
        "split_summaries": split_summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate a cleaner diffusion dataset from an existing strict diffusion manifest.")
    parser.add_argument("--input-dir", required=True, help="Directory containing train/val/test CSVs.")
    parser.add_argument("--output-dir", required=True, help="Destination directory for curated CSVs.")
    parser.add_argument("--image-stats-json", default="", help="Optional image stats JSON to copy into the curated dataset.")
    parser.add_argument("--min-mask-quality-score", type=float, default=0.75)
    parser.add_argument("--min-symmetry-score", type=float, default=0.84)
    parser.add_argument("--min-fill-ratio", type=float, default=0.56)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.045)
    parser.add_argument("--max-coverage-ratio", type=float, default=0.43)
    parser.add_argument("--min-horizontal-ratio", type=float, default=1.12)
    parser.add_argument("--min-wing-balance-ratio", type=float, default=0.92)
    parser.add_argument("--drop-border-touch", action="store_true", default=False)
    parser.add_argument(
        "--allowed-view-statuses",
        nargs="*",
        default=["accepted", "metadata_dorsal"],
        help="Allowed view_status values. Empty list disables the filter.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    split_rejections: Dict[str, Dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        rows = read_rows(input_dir / f"{split_name}.csv")
        curated_rows, rejection_counts = curate_split(rows, args)
        write_rows(output_dir / f"{split_name}.csv", curated_rows)
        split_rows[split_name] = curated_rows
        split_rejections[split_name] = rejection_counts

    if args.image_stats_json:
        stats_src = Path(args.image_stats_json)
        if not stats_src.exists():
            raise FileNotFoundError(f"image-stats-json does not exist: {stats_src}")
        stats_dir = output_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        (stats_dir / "train_image_stats.json").write_text(stats_src.read_text(encoding="utf-8"), encoding="utf-8")

    summary_payload = build_summary_payload(
        input_dir=input_dir,
        output_dir=output_dir,
        split_rows=split_rows,
        split_rejections=split_rejections,
        args=args,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
