#!/usr/bin/env python3
"""Create a random before/after QA sheet for preprocessed butterfly images."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageOps, ImageDraw


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample random preprocessed butterflies into a QA sheet.")
    parser.add_argument("--source-csv", required=True, help="Source manifest used to preprocess the images.")
    parser.add_argument("--processed-dir", default=None, help="Directory containing an images/ subfolder.")
    parser.add_argument("--processed-csv", default=None, help="Optional processed manifest to sample from directly.")
    parser.add_argument("--output-dir", required=True, help="Directory for the QA sheet and sampled row CSV.")
    parser.add_argument("--sample-size", type=int, default=12, help="Number of samples to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible QA samples.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_csv = Path(args.source_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_index: Dict[int, Dict[str, Any]] = {}
    with source_csv.open("r", newline="", encoding="utf-8") as file_handle:
        rows_by_index = {
            index: dict(row)
            for index, row in enumerate(csv.DictReader(file_handle), start=1)
        }

    if args.processed_csv:
        processed_rows: List[Dict[str, Any]] = []
        with Path(args.processed_csv).open("r", newline="", encoding="utf-8") as file_handle:
            for row in csv.DictReader(file_handle):
                if normalize_text(row.get("preprocess_status")) != "ok":
                    continue
                processed_path = normalize_text(row.get("processed_image_path"))
                if processed_path and Path(processed_path).exists():
                    processed_rows.append(dict(row))
        if not processed_rows:
            raise RuntimeError(f"No ok processed rows found in {args.processed_csv}")
        random.seed(args.seed)
        sampled_items = random.sample(processed_rows, min(args.sample_size, len(processed_rows)))
    else:
        if not args.processed_dir:
            raise RuntimeError("Either --processed-dir or --processed-csv is required.")
        processed_dir = Path(args.processed_dir)
        images_dir = processed_dir / "images"
        processed_files = sorted(images_dir.glob("*.jpg"))
        if not processed_files:
            raise RuntimeError(f"No processed images found in {images_dir}")

        random.seed(args.seed)
        sample_files = random.sample(processed_files, min(args.sample_size, len(processed_files)))
        sampled_items = []
        for processed_path in sample_files:
            source_index = int(processed_path.stem.rsplit("_", 1)[-1])
            sampled_items.append(
                {
                    "source_row_index": str(source_index),
                    "processed_image_path": str(processed_path),
                }
            )

    thumb_width = 256
    thumb_height = 256
    padding = 14
    label_height = 54
    canvas = Image.new(
        "RGB",
        (padding + 2 * (thumb_width + padding), padding + len(sampled_items) * (thumb_height + label_height + padding)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    sample_rows: List[Dict[str, Any]] = []

    current_y = padding
    for item in sampled_items:
        index = int(normalize_text(item.get("source_row_index")) or "0")
        processed_path = Path(normalize_text(item.get("processed_image_path")))
        row = rows_by_index[index]
        source_path = Path(normalize_text(row.get("image_path")) or normalize_text(row.get("local_image_path")))

        with Image.open(source_path) as image:
            original = image.convert("RGB")
        with Image.open(processed_path) as image:
            processed = image.convert("RGB")

        original_thumb = ImageOps.contain(original, (thumb_width, thumb_height), Image.Resampling.LANCZOS)
        processed_thumb = ImageOps.contain(processed, (thumb_width, thumb_height), Image.Resampling.LANCZOS)
        original_panel = Image.new("RGB", (thumb_width, thumb_height), "white")
        processed_panel = Image.new("RGB", (thumb_width, thumb_height), "white")
        original_panel.paste(original_thumb, ((thumb_width - original_thumb.width) // 2, (thumb_height - original_thumb.height) // 2))
        processed_panel.paste(processed_thumb, ((thumb_width - processed_thumb.width) // 2, (thumb_height - processed_thumb.height) // 2))

        left_x = padding
        right_x = padding + thumb_width + padding
        canvas.paste(original_panel, (left_x, current_y))
        canvas.paste(processed_panel, (right_x, current_y))
        draw.rectangle([left_x, current_y, left_x + thumb_width, current_y + thumb_height], outline="black", width=2)
        draw.rectangle([right_x, current_y, right_x + thumb_width, current_y + thumb_height], outline="black", width=2)
        draw.text((left_x, current_y + thumb_height + 6), f"Original | idx {index}", fill="black")
        draw.text((right_x, current_y + thumb_height + 6), f"Processed | {normalize_text(row.get('species'))[:36]}", fill="black")

        sample_rows.append(
            {
                "source_row_index": index,
                "species": normalize_text(row.get("species")),
                "record_id": normalize_text(row.get("record_id")),
                "source": normalize_text(row.get("source")),
                "source_image_path": str(source_path),
                "processed_image_path": str(processed_path),
            }
        )
        current_y += thumb_height + label_height + padding

    sheet_path = output_dir / "random_sample_sheet.png"
    rows_path = output_dir / "random_sample_rows.csv"
    canvas.save(sheet_path)
    with rows_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    print(
        {
            "sample_count": len(sample_rows),
            "sheet_path": str(sheet_path),
            "rows_path": str(rows_path),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
