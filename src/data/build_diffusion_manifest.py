#!/usr/bin/env python3
"""Build strict coordinate-conditioned diffusion manifests using cleaned VAE-ready images."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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


def processed_index(processed_rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in processed_rows:
        specimen_key = normalize_text(row.get("specimen_key"))
        source = normalize_text(row.get("source"))
        record_id = normalize_text(row.get("record_id"))
        image_url = normalize_text(row.get("image_url"))
        for key in (("specimen_key", specimen_key, ""), ("record_id", source, record_id), ("image_url", image_url, "")):
            if key[1]:
                index[key] = row
    return index


def find_processed_row(strict_row: Dict[str, Any], processed_lookup: Dict[Tuple[str, str, str], Dict[str, Any]]) -> Dict[str, Any] | None:
    specimen_key = normalize_text(strict_row.get("specimen_key"))
    source = normalize_text(strict_row.get("source"))
    record_id = normalize_text(strict_row.get("record_id"))
    image_url = normalize_text(strict_row.get("image_url"))
    for candidate in (("specimen_key", specimen_key, ""), ("record_id", source, record_id), ("image_url", image_url, "")):
        if candidate[1] and candidate in processed_lookup:
            return processed_lookup[candidate]
    return None


def merge_strict_with_processed(
    strict_rows: Iterable[Dict[str, Any]],
    processed_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Counter[str]]:
    merged_rows: List[Dict[str, Any]] = []
    summary: Counter[str] = Counter()
    for strict_row in strict_rows:
        processed_row = find_processed_row(strict_row, processed_lookup)
        if processed_row is None:
            summary["missing_processed_match"] += 1
            continue
        processed_abs = normalize_text(processed_row.get("processed_image_absolute_path"))
        if not processed_abs:
            summary["missing_processed_image_path"] += 1
            continue
        if normalize_text(processed_row.get("keep_for_training")).lower() not in {"true", "1", "yes"}:
            summary["processed_keep_false"] += 1
            continue
        if normalize_text(processed_row.get("preprocess_status")) != "ok":
            summary["processed_preprocess_not_ok"] += 1
            continue

        merged = dict(strict_row)
        merged["source_image_path"] = normalize_text(strict_row.get("image_path"))
        merged["processed_image_path"] = normalize_text(processed_row.get("processed_image_path")) or processed_abs
        merged["processed_image_absolute_path"] = processed_abs
        merged["processed_mask_path"] = normalize_text(processed_row.get("processed_mask_path"))
        merged["preprocess_status"] = normalize_text(processed_row.get("preprocess_status"))
        merged["keep_for_training"] = normalize_text(processed_row.get("keep_for_training"))
        merged["crop_method"] = normalize_text(processed_row.get("crop_method"))
        merged["view_hint"] = normalize_text(processed_row.get("view_hint"))
        merged["view_status"] = normalize_text(processed_row.get("view_status"))
        merged["symmetry_score"] = normalize_text(processed_row.get("symmetry_score"))
        merged["wing_balance_ratio"] = normalize_text(processed_row.get("wing_balance_ratio"))
        merged["horizontal_ratio"] = normalize_text(processed_row.get("horizontal_ratio"))
        merged["fill_ratio"] = normalize_text(processed_row.get("fill_ratio"))
        merged["coverage_ratio"] = normalize_text(processed_row.get("coverage_ratio"))
        merged["mask_quality_score"] = normalize_text(processed_row.get("mask_quality_score"))
        merged["mask_touches_border"] = normalize_text(processed_row.get("mask_touches_border"))
        merged_rows.append(merged)
        summary["matched_rows"] += 1
    return merged_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a strict coordinate-conditioned diffusion manifest.")
    parser.add_argument("--processed-manifest", required=True, help="Processed VAE-ready metadata CSV.")
    parser.add_argument("--strict-train-csv", required=True, help="Strict train split CSV with coordinates.")
    parser.add_argument("--strict-val-csv", required=True, help="Strict val split CSV with coordinates.")
    parser.add_argument("--strict-test-csv", required=True, help="Strict test split CSV with coordinates.")
    parser.add_argument("--output-dir", required=True, help="Destination folder for merged diffusion manifests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    processed_manifest = Path(args.processed_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_rows = read_rows(processed_manifest)
    processed_lookup = processed_index(processed_rows)

    split_summaries: Dict[str, Any] = {}
    totals = Counter()

    for split_name, split_path_str in (("train", args.strict_train_csv), ("val", args.strict_val_csv), ("test", args.strict_test_csv)):
        strict_rows = read_rows(Path(split_path_str))
        merged_rows, summary = merge_strict_with_processed(strict_rows, processed_lookup)
        if merged_rows:
            write_rows(output_dir / f"{split_name}.csv", merged_rows)
        split_summaries[split_name] = {
            "strict_rows": len(strict_rows),
            "merged_rows": len(merged_rows),
            "summary": dict(summary),
            "species": len({normalize_text(row.get("species")) for row in merged_rows}),
        }
        totals.update(summary)

    summary_payload = {
        "processed_manifest": str(processed_manifest),
        "output_dir": str(output_dir),
        "split_summaries": split_summaries,
        "totals": dict(totals),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
