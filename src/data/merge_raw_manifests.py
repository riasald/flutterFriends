#!/usr/bin/env python3
"""Merge multiple raw manifest CSVs and deduplicate by exact image URL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from src.data.fetch_butterfly_gbif_idigbio import dedupe_rows, write_csv


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        return [dict(row) for row in reader]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge raw metadata manifests.")
    parser.add_argument("--output-csv", required=True, help="Merged output CSV.")
    parser.add_argument("--summary-json", default=None, help="Optional summary JSON output path.")
    parser.add_argument("inputs", nargs="+", help="Input CSV paths to merge.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = [Path(value) for value in args.inputs]
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json) if args.summary_json else output_csv.with_suffix(".summary.json")

    merged_rows: List[Dict[str, Any]] = []
    input_counts = {}
    for path in input_paths:
        rows = read_rows(path)
        input_counts[str(path)] = len(rows)
        merged_rows.extend(rows)

    deduped_rows = dedupe_rows(merged_rows)
    write_csv(output_csv, deduped_rows)
    summary_json.write_text(
        json.dumps(
            {
                "inputs": input_counts,
                "merged_rows_before_dedupe": len(merged_rows),
                "merged_rows_after_dedupe": len(deduped_rows),
                "output_csv": str(output_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Done.")
    print(json.dumps({"after_dedupe": len(deduped_rows), "output_csv": str(output_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
