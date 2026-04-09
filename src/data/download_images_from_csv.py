#!/usr/bin/env python3
"""
Download butterfly images from an existing metadata CSV.

This is the handoff step after metadata retrieval completes. It avoids
re-querying GBIF and iDigBio and simply downloads the image URLs already
captured in a CSV manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from src.data.fetch_butterfly_gbif_idigbio import download_images, ensure_dir, write_csv, write_json


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        return [dict(row) for row in reader]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download images from an existing butterfly metadata CSV.")
    parser.add_argument("--input-csv", required=True, help="CSV file with image_url and metadata columns.")
    parser.add_argument("--output-dir", required=True, help="Directory to write downloaded images into.")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path for the updated CSV with local_image_path and download status columns.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path for the download summary JSON.",
    )
    parser.add_argument(
        "--user-agent",
        default="butterfly-geo-gen/0.1 (contact: samir.saldanha@outlook.com)",
        help="User-Agent header used for image requests.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=180,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=12,
        help="Number of parallel download workers.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Write progress and an updated CSV checkpoint every N completed downloads.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_csv = Path(args.output_csv) if args.output_csv else input_csv
    summary_json = Path(args.summary_json) if args.summary_json else output_dir / "download_summary.json"
    progress_json = output_dir / "download_progress.json"

    ensure_dir(output_dir)
    rows = read_csv_rows(input_csv)

    write_json(
        progress_json,
        {
            "stage": "starting_downloads",
            "input_csv": str(input_csv),
            "output_dir": str(output_dir),
            "total_rows": len(rows),
        },
    )

    last_checkpoint = 0

    def progress_callback(completed: int, total: int, downloaded: int, existing: int, failed: int) -> None:
        nonlocal last_checkpoint
        should_checkpoint = (completed - last_checkpoint) >= max(1, args.checkpoint_every) or completed == total
        if should_checkpoint:
            write_csv(output_csv, rows)
            last_checkpoint = completed
        write_json(
            progress_json,
            {
                "stage": "downloading_images",
                "input_csv": str(input_csv),
                "output_dir": str(output_dir),
                "output_csv": str(output_csv),
                "completed": completed,
                "total": total,
                "downloaded": downloaded,
                "skipped_existing": existing,
                "failed": failed,
            },
        )

    summary = download_images(
        rows,
        output_dir,
        user_agent=args.user_agent,
        request_timeout=args.request_timeout,
        download_workers=args.download_workers,
        progress_callback=progress_callback,
    )

    write_csv(output_csv, rows)
    write_json(
        summary_json,
        {
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "output_dir": str(output_dir),
            "request_timeout": args.request_timeout,
            "download_workers": args.download_workers,
            "summary": summary,
        },
    )
    write_json(
        progress_json,
        {
            "stage": "complete",
            "input_csv": str(input_csv),
            "output_dir": str(output_dir),
            "output_csv": str(output_csv),
            "summary_json": str(summary_json),
            **summary,
        },
    )

    print("Done.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
