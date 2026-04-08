#!/usr/bin/env python3
"""Validate that sharded preprocessing completed before downstream training prep."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check shard preprocessing progress files before merging.")
    parser.add_argument("--input-glob", required=True, help="Glob for shard progress.json files.")
    parser.add_argument("--expected-shards", type=int, default=0, help="Optional expected shard count.")
    parser.add_argument("--output-json", default="", help="Optional path for a summary JSON report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    progress_paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not progress_paths:
        raise RuntimeError(f"No shard progress files matched: {args.input_glob}")

    if args.expected_shards and len(progress_paths) != args.expected_shards:
        raise RuntimeError(
            f"Expected {args.expected_shards} shard progress files, found {len(progress_paths)} for {args.input_glob}"
        )

    shard_reports: List[Dict[str, Any]] = []
    incomplete: List[str] = []
    missing_outputs: List[str] = []

    for path in progress_paths:
        payload = load_json(path)
        shard_dir = path.parent
        processed_csv = shard_dir / "processed_metadata.csv"
        summary_json = shard_dir / "preprocess_summary.json"
        stage = str(payload.get("stage", "")).strip().lower()
        report = {
            "shard": shard_dir.name,
            "progress_json": str(path),
            "stage": stage,
            "processed_csv": str(processed_csv),
            "summary_json": str(summary_json),
            "processed_csv_exists": processed_csv.exists(),
            "summary_json_exists": summary_json.exists(),
        }
        shard_reports.append(report)

        if stage != "complete":
            incomplete.append(shard_dir.name)
        if not processed_csv.exists() or not summary_json.exists():
            missing_outputs.append(shard_dir.name)

    summary = {
        "input_glob": args.input_glob,
        "shard_count": len(progress_paths),
        "all_complete": not incomplete,
        "all_outputs_present": not missing_outputs,
        "incomplete_shards": incomplete,
        "missing_output_shards": missing_outputs,
        "shards": shard_reports,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if incomplete or missing_outputs:
        raise RuntimeError("Shard preprocessing is incomplete; refusing to continue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
