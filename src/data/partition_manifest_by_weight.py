#!/usr/bin/env python3
"""Partition a CSV manifest into weighted subsets for heterogeneous preprocessing."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        return [dict(row) for row in csv.DictReader(file_handle)]


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_output_spec(spec: str) -> Tuple[Path, float]:
    if ":" not in spec:
        raise ValueError(f"Output spec must be PATH:WEIGHT, got {spec!r}")
    path_str, weight_str = spec.rsplit(":", 1)
    weight = float(weight_str)
    if weight <= 0:
        raise ValueError(f"Weight must be > 0 for {spec!r}")
    return Path(path_str), weight


def compute_counts(total_rows: int, weights: Sequence[float]) -> List[int]:
    total_weight = sum(weights)
    raw_counts = [total_rows * weight / total_weight for weight in weights]
    counts = [int(value) for value in raw_counts]
    remaining = total_rows - sum(counts)

    fractional = sorted(
        ((raw - base, index) for index, (raw, base) in enumerate(zip(raw_counts, counts))),
        reverse=True,
    )
    for _, index in fractional[:remaining]:
        counts[index] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Partition a manifest into weighted CSV subsets.")
    parser.add_argument("--input-csv", required=True, help="Input CSV to partition.")
    parser.add_argument(
        "--output",
        action="append",
        required=True,
        help="Output spec as PATH:WEIGHT. Repeat for each target subset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for deterministic partitioning.")
    parser.add_argument("--summary-json", required=True, help="Summary JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    rows = read_rows(input_csv)
    outputs = [parse_output_spec(spec) for spec in args.output]
    if not outputs:
        raise RuntimeError("At least one --output must be provided.")

    rng = random.Random(args.seed)
    shuffled_rows = list(rows)
    rng.shuffle(shuffled_rows)

    weights = [weight for _, weight in outputs]
    counts = compute_counts(len(shuffled_rows), weights)

    offset = 0
    summary_outputs: List[Dict[str, Any]] = []
    for (path, weight), count in zip(outputs, counts):
        subset = shuffled_rows[offset:offset + count]
        offset += count
        write_rows(path, subset)
        summary_outputs.append(
            {
                "path": str(path),
                "weight": weight,
                "rows": len(subset),
            }
        )

    summary = {
        "input_csv": str(input_csv),
        "seed": args.seed,
        "input_rows": len(rows),
        "outputs": summary_outputs,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
