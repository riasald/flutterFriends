#!/usr/bin/env python3
"""Build train/val/test CSV splits from cleaned butterfly metadata.

The split logic keeps all rows from the same specimen together to prevent
multi-image leakage across train and evaluation splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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

    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def group_key_for_row(row: Dict[str, Any]) -> str:
    return (
        normalize_text(row.get("specimen_key"))
        or normalize_text(row.get("record_id"))
        or normalize_text(row.get("image_url"))
        or normalize_text(row.get("image_path"))
    )


def split_counts(num_groups: int, val_fraction: float, test_fraction: float) -> Tuple[int, int, int]:
    if num_groups <= 0:
        return 0, 0, 0
    if num_groups == 1:
        return 1, 0, 0
    if num_groups == 2:
        return 1, 1, 0
    if num_groups == 3:
        return 1, 1, 1

    val_count = max(1, round(num_groups * val_fraction))
    test_count = max(1, round(num_groups * test_fraction))
    train_count = num_groups - val_count - test_count

    if train_count <= 0:
        overflow = 1 - train_count
        if test_count >= val_count and test_count > 1:
            test_count -= overflow
        else:
            val_count = max(1, val_count - overflow)
        train_count = num_groups - val_count - test_count

    if train_count <= 0:
        train_count = max(1, num_groups - 2)
        remainder = num_groups - train_count
        val_count = max(1, remainder // 2)
        test_count = remainder - val_count

    return train_count, val_count, test_count


def assign_species_groups(
    groups_by_species: Dict[str, List[str]],
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> Dict[str, str]:
    rng = random.Random(seed)
    assignments: Dict[str, str] = {}

    for species in sorted(groups_by_species):
        keys = list(groups_by_species[species])
        rng.shuffle(keys)
        train_count, val_count, test_count = split_counts(len(keys), val_fraction, test_fraction)

        ordered = (
            [("train", train_count), ("val", val_count), ("test", test_count)]
        )
        cursor = 0
        for split_name, count in ordered:
            for key in keys[cursor:cursor + count]:
                assignments[key] = split_name
            cursor += count

        for key in keys[cursor:]:
            assignments[key] = "train"

    return assignments


def build_group_maps(rows: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[str]]]:
    rows_by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    species_by_group: Dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        key = group_key_for_row(row)
        if not key:
            raise RuntimeError("Encountered a row without any usable group key.")
        rows_by_group[key].append(row)
        species_by_group[key][normalize_text(row.get("species"))] += 1

    groups_by_species: Dict[str, List[str]] = defaultdict(list)
    for key, species_counter in species_by_group.items():
        species, _ = species_counter.most_common(1)[0]
        groups_by_species[species].append(key)

    return rows_by_group, groups_by_species


def summarize_split(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    species_counter = Counter(normalize_text(row.get("species")) for row in rows)
    group_counter = Counter(group_key_for_row(row) for row in rows)
    return {
        "rows": len(rows),
        "groups": len(group_counter),
        "unique_species": len(species_counter),
        "top_species": species_counter.most_common(20),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val/test splits for butterfly metadata.")
    parser.add_argument("--input-csv", required=True, help="Cleaned metadata CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for split CSV files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic shuffling.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of groups to place in validation.")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Fraction of groups to place in test.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_csv)
    rows_by_group, groups_by_species = build_group_maps(rows)
    assignments = assign_species_groups(
        groups_by_species,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    split_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for group_key, group_rows in rows_by_group.items():
        split_name = assignments.get(group_key, "train")
        for row in group_rows:
            row_with_split = dict(row)
            row_with_split["split"] = split_name
            split_rows[split_name].append(row_with_split)

    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"
    summary_json = output_dir / "split_summary.json"

    write_rows(train_csv, split_rows["train"])
    write_rows(val_csv, split_rows["val"])
    write_rows(test_csv, split_rows["test"])
    write_json(
        summary_json,
        {
            "input_csv": str(input_csv),
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "train": summarize_split(split_rows["train"]),
            "val": summarize_split(split_rows["val"]),
            "test": summarize_split(split_rows["test"]),
        },
    )

    print("Done.")
    print(
        json.dumps(
            {
                "train_rows": len(split_rows["train"]),
                "val_rows": len(split_rows["val"]),
                "test_rows": len(split_rows["test"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
