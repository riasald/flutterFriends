#!/usr/bin/env python3
"""Recover accepted preprocessing rows from shard image outputs and build a remaining manifest."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


INDEX_SUFFIX_RE = re.compile(r"_(\d{6,})$")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


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


def parse_source_index_from_image(path: Path) -> int:
    match = INDEX_SUFFIX_RE.search(path.stem)
    if not match:
        raise RuntimeError(f"Could not parse source row index from {path}")
    return int(match.group(1))


def build_recovered_row(source_row: Dict[str, Any], processed_path: Path, source_index: int) -> Dict[str, Any]:
    row = dict(source_row)
    row["source_image_path"] = normalize_text(source_row.get("image_path")) or normalize_text(source_row.get("local_image_path"))
    row["processed_image_path"] = str(processed_path)
    row["processed_mask_path"] = ""
    row["preprocess_status"] = "ok"
    row["crop_method"] = "recovered_from_existing_output"
    row["crop_bbox"] = ""
    row["view_hint"] = normalize_text(source_row.get("view_hint")) or "unknown"
    row["view_source"] = normalize_text(source_row.get("view_source")) or "none"
    row["view_status"] = "accepted"
    row["rejection_reason"] = ""
    row["keep_for_training"] = "True"
    row["source_row_index"] = str(source_index)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover accepted preprocessing outputs and build a remaining CSV.")
    parser.add_argument("--input-csv", required=True, help="Original source CSV used for preprocessing.")
    parser.add_argument("--images-glob", required=True, help="Glob for processed shard image files, e.g. shard_*/images/*.jpg")
    parser.add_argument("--recovered-csv", required=True, help="Output CSV for rows already processed successfully.")
    parser.add_argument("--remaining-csv", required=True, help="Output CSV containing only rows that still need preprocessing.")
    parser.add_argument("--summary-json", required=True, help="Summary JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = read_rows(Path(args.input_csv))
    image_paths = [Path(path) for path in sorted(glob.glob(args.images_glob))]
    if not image_paths:
        raise RuntimeError(f"No processed images matched {args.images_glob}")

    recovered_rows: List[Dict[str, Any]] = []
    recovered_indices: Set[int] = set()
    duplicate_indices: List[int] = []
    missing_source_indices: List[int] = []

    for image_path in image_paths:
        source_index = parse_source_index_from_image(image_path)
        if source_index in recovered_indices:
            duplicate_indices.append(source_index)
            continue
        if source_index < 1 or source_index > len(source_rows):
            missing_source_indices.append(source_index)
            continue
        recovered_indices.add(source_index)
        recovered_rows.append(build_recovered_row(source_rows[source_index - 1], image_path, source_index))

    remaining_rows = [
        dict(row)
        for index, row in enumerate(source_rows, start=1)
        if index not in recovered_indices
    ]

    recovered_rows.sort(key=lambda row: int(normalize_text(row.get("source_row_index")) or "0"))
    write_rows(Path(args.recovered_csv), recovered_rows)
    write_rows(Path(args.remaining_csv), remaining_rows)

    summary = {
        "input_csv": args.input_csv,
        "images_glob": args.images_glob,
        "recovered_csv": args.recovered_csv,
        "remaining_csv": args.remaining_csv,
        "total_source_rows": len(source_rows),
        "recovered_rows": len(recovered_rows),
        "remaining_rows": len(remaining_rows),
        "duplicate_index_count": len(duplicate_indices),
        "duplicate_index_examples": duplicate_indices[:20],
        "missing_source_index_count": len(missing_source_indices),
        "missing_source_index_examples": missing_source_indices[:20],
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
