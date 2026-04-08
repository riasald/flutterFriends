#!/usr/bin/env python3
"""Merge shard-level preprocessing CSVs into a single manifest."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows_list = list(rows)
    ok_rows = [row for row in rows_list if row.get("preprocess_status") == "ok"]
    rejected_rows = [row for row in rows_list if row.get("preprocess_status") == "rejected_view"]
    return {
        "total_rows": len(rows_list),
        "processed_ok": len(ok_rows),
        "rejected_view": len(rejected_rows),
        "missing_input": sum(row.get("preprocess_status") == "missing_input" for row in rows_list),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded preprocessing CSVs into one manifest.")
    parser.add_argument("--input-glob", required=True, help="Glob for shard processed_metadata.csv files.")
    parser.add_argument("--output-csv", required=True, help="Path for merged processed CSV.")
    parser.add_argument("--summary-json", default=None, help="Optional summary JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not paths:
        raise RuntimeError(f"No shard CSVs matched: {args.input_glob}")

    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(read_rows(path))

    rows.sort(key=lambda row: int(normalize_text(row.get("source_row_index")) or "0"))
    output_csv = Path(args.output_csv)
    write_rows(output_csv, rows)

    summary = {
        "input_glob": args.input_glob,
        "output_csv": str(output_csv),
        "input_files": [str(path) for path in paths],
        "stats": summarize(rows),
    }
    summary_json = Path(args.summary_json) if args.summary_json else output_csv.with_suffix(".summary.json")
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
