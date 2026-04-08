#!/usr/bin/env python3
"""Compute per-channel mean and standard deviation for processed training images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def read_rows(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        rows = []
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            rows.append(dict(row))
    return rows


def resolve_image_path(row: Dict[str, Any], *, base_dir: Optional[Path] = None) -> str:
    search_roots: List[Path] = []
    if base_dir is not None:
        search_roots.append(base_dir)
        if base_dir.parent != base_dir:
            search_roots.append(base_dir.parent)
    for column in ("processed_image_path", "image_path", "source_image_path", "local_image_path"):
        path = normalize_text(row.get(column))
        if not path:
            continue
        candidate = Path(path)
        if candidate.exists():
            return str(candidate)
        if not candidate.is_absolute():
            for root in search_roots:
                relative_candidate = (root / candidate).resolve()
                if relative_candidate.exists():
                    return str(relative_candidate)
    return ""


def valid_rows(rows: Iterable[Dict[str, Any]], *, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    output = []
    for row in rows:
        if normalize_text(row.get("preprocess_status")) not in {"", "ok"}:
            continue
        keep = normalize_text(row.get("keep_for_training"))
        if keep and keep.lower() != "true":
            continue
        path = resolve_image_path(row, base_dir=base_dir)
        if path:
            item = dict(row)
            item["_resolved_image_path"] = path
            output.append(item)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-channel image statistics from a CSV manifest.")
    parser.add_argument("--input-csv", required=True, help="Processed or cleaned metadata CSV.")
    parser.add_argument("--output-json", required=True, help="Path for mean/std JSON output.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick estimation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    rows = valid_rows(read_rows(input_csv, max_rows=args.max_images), base_dir=input_csv.parent)
    if not rows:
        raise RuntimeError("No valid image rows were found for statistics computation.")

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for row in rows:
        with Image.open(row["_resolved_image_path"]) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        pixels = array.reshape(-1, 3)
        channel_sum += pixels.sum(axis=0)
        channel_sq_sum += np.square(pixels).sum(axis=0)
        total_pixels += pixels.shape[0]

    mean = channel_sum / total_pixels
    variance = np.maximum(channel_sq_sum / total_pixels - np.square(mean), 0.0)
    std = np.sqrt(variance)

    payload = {
        "input_csv": str(Path(args.input_csv)),
        "images_used": len(rows),
        "total_pixels": int(total_pixels),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
