#!/usr/bin/env python3
"""Assemble a self-contained diffusion dataset with portable image paths."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def resolve_processed_image_path(row: Dict[str, Any]) -> Path:
    for column in ("processed_image_absolute_path", "processed_image_path"):
        value = normalize_text(row.get(column))
        if not value:
            continue
        candidate = Path(value)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Row is missing a usable processed image path: "
        f"record_id={normalize_text(row.get('record_id'))} "
        f"specimen_key={normalize_text(row.get('specimen_key'))}"
    )


def choose_destination_name(
    source_path: Path,
    *,
    used_names: Dict[str, Path],
) -> str:
    base_name = source_path.name
    stem = source_path.stem
    suffix = source_path.suffix
    if base_name not in used_names:
        used_names[base_name] = source_path
        return base_name
    if used_names[base_name] == source_path:
        return base_name
    index = 1
    while True:
        candidate = f"{stem}_{index:03d}{suffix}"
        if candidate not in used_names:
            used_names[candidate] = source_path
            return candidate
        if used_names[candidate] == source_path:
            return candidate
        index += 1


def rewrite_row(row: Dict[str, Any], *, relative_image_path: str, source_path: Path) -> Dict[str, Any]:
    updated = dict(row)
    updated["original_processed_image_absolute_path"] = str(source_path)
    updated["processed_image_absolute_path"] = ""
    updated["processed_image_path"] = relative_image_path.replace("\\", "/")
    updated["original_image_path"] = normalize_text(row.get("image_path"))
    updated["original_source_image_path"] = normalize_text(row.get("source_image_path"))
    updated["image_path"] = ""
    updated["source_image_path"] = ""
    return updated


def filesystem_path(path: Path) -> str:
    resolved = path.resolve()
    as_str = str(resolved)
    if os.name == "nt" and not as_str.startswith("\\\\?\\"):
        return "\\\\?\\" + as_str
    return as_str


def assemble_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    image_stats_json: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    stats_dir = output_dir / "stats"
    images_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if not image_stats_json.exists():
        raise FileNotFoundError(f"Image stats file does not exist: {image_stats_json}")
    shutil.copy2(image_stats_json, stats_dir / "train_image_stats.json")

    used_names: Dict[str, Path] = {}
    source_to_relative: Dict[Path, str] = {}
    copy_summary: Counter[str] = Counter()
    total_bytes = 0
    split_summaries: Dict[str, Any] = {}

    for split_name in ("train", "val", "test"):
        split_path = input_dir / f"{split_name}.csv"
        rows = read_rows(split_path)
        rewritten_rows: List[Dict[str, Any]] = []
        unique_species = set()
        for row in rows:
            source_path = resolve_processed_image_path(row).resolve()
            if source_path not in source_to_relative:
                dest_name = choose_destination_name(source_path, used_names=used_names)
                dest_path = images_dir / dest_name
                shutil.copy2(filesystem_path(source_path), filesystem_path(dest_path))
                source_to_relative[source_path] = f"images/{dest_name}"
                copy_summary["copied_images"] += 1
                total_bytes += os.path.getsize(filesystem_path(dest_path))
            rewritten_rows.append(
                rewrite_row(
                    row,
                    relative_image_path=source_to_relative[source_path],
                    source_path=source_path,
                )
            )
            unique_species.add(normalize_text(row.get("species")))
        write_rows(output_dir / f"{split_name}.csv", rewritten_rows)
        split_summaries[split_name] = {
            "rows": len(rewritten_rows),
            "species": len({species for species in unique_species if species}),
        }

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "image_stats_json": str((stats_dir / "train_image_stats.json").resolve()),
        "split_summaries": split_summaries,
        "unique_images": int(copy_summary["copied_images"]),
        "total_bytes": int(total_bytes),
        "total_gib": float(total_bytes / (1024**3)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble a self-contained diffusion dataset for RunPod.")
    parser.add_argument("--input-dir", required=True, help="Directory containing train/val/test diffusion CSVs.")
    parser.add_argument("--image-stats-json", required=True, help="Image stats JSON to copy alongside the dataset.")
    parser.add_argument("--output-dir", required=True, help="Destination folder for the portable diffusion dataset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = assemble_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        image_stats_json=Path(args.image_stats_json),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
