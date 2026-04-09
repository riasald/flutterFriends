#!/usr/bin/env python3
"""
Clean and normalize butterfly metadata after the fetch step.

This script is the bridge between raw fetch manifests and later preprocessing
or training code. It fixes species labels, enforces the U.S. geographic scope,
flags duplicate records, and emits cleaned plus rejected CSVs with a summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.data.schema import RAW_REQUIRED_COLUMNS

NON_SPECIES_TOKENS = {
    "aff",
    "aff.",
    "cf",
    "cf.",
    "complex",
    "group",
    "indet",
    "nr",
    "nr.",
    "sp",
    "sp.",
    "spp",
    "spp.",
    "unknown",
}
US_STATE_NAMES = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
    "district of columbia",
}
US_REGIONS = [
    ("continental_us", 24.0, 49.8, -125.5, -66.0),
    ("alaska", 51.0, 72.5, -180.0, -129.0),
    ("alaska_dateline", 51.0, 72.5, 170.0, 180.0),
    ("hawaii", 18.5, 22.8, -161.0, -154.0),
]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def ascii_text(value: Any) -> str:
    text = normalize_text(value)
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def title_token(token: str) -> str:
    if not token:
        return ""
    return token[0].upper() + token[1:].lower()


def tokenize_alpha(text: str) -> List[str]:
    return re.findall(r"[A-Za-z-]+", ascii_text(text))


def parse_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def parse_month(event_date: str) -> str:
    match = re.search(r"(\d{4})-(\d{2})", normalize_text(event_date))
    if not match:
        return ""
    return match.group(2)


def normalize_country(value: Any) -> str:
    text = normalize_text(value)
    lower = text.lower()
    if "united states" in lower or lower in {"usa", "us", "u.s.a.", "u.s."}:
        return "United States"
    return text


def normalize_state(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    return " ".join(part.capitalize() for part in text.split())


def parse_binomial_candidate(
    candidate: str,
    genus_hint: str = "",
    *,
    require_lowercase_second_token: bool = False,
) -> Optional[str]:
    raw_parts = [part for part in normalize_text(candidate).split() if part]
    if len(raw_parts) < 2:
        return None
    if require_lowercase_second_token and not raw_parts[1][:1].islower():
        return None

    tokens = tokenize_alpha(candidate)
    if len(tokens) < 2:
        return None

    genus = title_token(tokens[0])
    species = tokens[1].lower()
    if species in NON_SPECIES_TOKENS:
        return None
    if genus_hint and genus.lower() != genus_hint.lower():
        return None
    return f"{genus} {species}"


def canonical_species_from_row(row: Dict[str, Any]) -> Tuple[str, str]:
    genus_hint = title_token(tokenize_alpha(row.get("genus", ""))[0]) if tokenize_alpha(row.get("genus", "")) else ""
    raw_species = normalize_text(row.get("raw_species"))
    species_field = normalize_text(row.get("species"))
    scientific_name = normalize_text(row.get("scientific_name"))

    raw_tokens = tokenize_alpha(raw_species)
    if len(raw_tokens) >= 2:
        parsed = parse_binomial_candidate(raw_species, genus_hint)
        if parsed:
            return parsed, "raw_species_binomial"
    elif len(raw_tokens) == 1 and genus_hint:
        epithet = raw_tokens[0].lower()
        if epithet not in NON_SPECIES_TOKENS:
            return f"{genus_hint} {epithet}", "raw_species_epithet"

    species_tokens = tokenize_alpha(species_field)
    if len(species_tokens) >= 2:
        parsed = parse_binomial_candidate(species_field, genus_hint)
        if parsed:
            return parsed, "species_field_binomial"
    elif len(species_tokens) == 1 and genus_hint:
        epithet = species_tokens[0].lower()
        if epithet not in NON_SPECIES_TOKENS:
            return f"{genus_hint} {epithet}", "species_field_epithet"

    parsed = parse_binomial_candidate(
        scientific_name,
        genus_hint,
        require_lowercase_second_token=True,
    )
    if parsed:
        return parsed, "scientific_name_binomial"

    return "", "missing_species"


def classify_us_region(latitude: float, longitude: float) -> str:
    for region_name, lat_min, lat_max, lon_min, lon_max in US_REGIONS:
        if lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max:
            return region_name
    return ""


def row_passes_us_scope(row: Dict[str, Any], latitude: float, longitude: float) -> Tuple[bool, str]:
    region = classify_us_region(latitude, longitude)
    if region:
        return True, region

    country = normalize_country(row.get("country"))
    state_name = normalize_state(row.get("state_province")).lower()
    if country == "United States" and state_name in US_STATE_NAMES:
        return False, "us_metadata_but_out_of_bounds"
    return False, "non_us_coordinate"


def validate_required_columns(fieldnames: Iterable[str]) -> None:
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in set(fieldnames)]
    if missing:
        raise RuntimeError(f"Missing required columns: {', '.join(missing)}")


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        validate_required_columns(reader.fieldnames or [])
        return [dict(row) for row in reader]


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clean_rows(
    rows: List[Dict[str, Any]],
    *,
    require_downloaded: bool,
    allow_missing_coordinates: bool,
    filter_us_scope: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    seen_urls = set()
    record_image_counts = Counter(
        f"{normalize_text(row.get('source'))}:{normalize_text(row.get('record_id'))}"
        for row in rows
        if normalize_text(row.get("record_id"))
    )

    cleaned_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    rejection_counts: Dict[str, int] = defaultdict(int)

    for row in rows:
        base_row = dict(row)
        image_url = normalize_text(row.get("image_url"))
        specimen_key = f"{normalize_text(row.get('source'))}:{normalize_text(row.get('record_id'))}"

        if not image_url:
            base_row["reject_reason"] = "missing_image_url"
            rejected_rows.append(base_row)
            rejection_counts["missing_image_url"] += 1
            continue

        if image_url in seen_urls:
            base_row["reject_reason"] = "duplicate_image_url"
            rejected_rows.append(base_row)
            rejection_counts["duplicate_image_url"] += 1
            continue
        seen_urls.add(image_url)

        family = normalize_text(row.get("family"))
        if family and family.lower() != "nymphalidae":
            base_row["reject_reason"] = "non_nymphalidae_family"
            rejected_rows.append(base_row)
            rejection_counts["non_nymphalidae_family"] += 1
            continue

        latitude = parse_float(row.get("latitude"))
        longitude = parse_float(row.get("longitude"))
        has_coordinates = latitude is not None and longitude is not None
        region_or_reason = ""

        if not has_coordinates and not allow_missing_coordinates:
            base_row["reject_reason"] = "missing_coordinate"
            rejected_rows.append(base_row)
            rejection_counts["missing_coordinate"] += 1
            continue

        if has_coordinates and filter_us_scope:
            in_scope, region_or_reason = row_passes_us_scope(row, latitude, longitude)
            if not in_scope:
                base_row["reject_reason"] = region_or_reason
                rejected_rows.append(base_row)
                rejection_counts[region_or_reason] += 1
                continue
        elif has_coordinates:
            region_or_reason = classify_us_region(latitude, longitude) or "outside_us_scope"
        else:
            region_or_reason = "missing_coordinate"

        canonical_species, species_source = canonical_species_from_row(row)
        if not canonical_species:
            base_row["reject_reason"] = "missing_species_label"
            rejected_rows.append(base_row)
            rejection_counts["missing_species_label"] += 1
            continue

        image_path = normalize_text(row.get("image_path")) or normalize_text(row.get("local_image_path"))
        download_status = normalize_text(row.get("download_status"))
        if require_downloaded:
            if download_status not in {"downloaded", "exists"}:
                base_row["reject_reason"] = "image_not_downloaded"
                rejected_rows.append(base_row)
                rejection_counts["image_not_downloaded"] += 1
                continue
            if not image_path or not Path(image_path).exists():
                base_row["reject_reason"] = "missing_local_image"
                rejected_rows.append(base_row)
                rejection_counts["missing_local_image"] += 1
                continue

        cleaned_rows.append(
            {
                "image_path": image_path,
                "source": normalize_text(row.get("source")),
                "record_id": normalize_text(row.get("record_id")),
                "media_id": normalize_text(row.get("media_id")),
                "specimen_key": specimen_key,
                "record_image_count": record_image_counts.get(specimen_key, 1),
                "family": "Nymphalidae",
                "genus": title_token(tokenize_alpha(row.get("genus", ""))[0]) if tokenize_alpha(row.get("genus", "")) else canonical_species.split()[0],
                "species": canonical_species,
                "species_source": species_source,
                "scientific_name": normalize_text(row.get("scientific_name")),
                "latitude": latitude if latitude is not None else "",
                "longitude": longitude if longitude is not None else "",
                "has_coordinates": has_coordinates,
                "us_region": region_or_reason,
                "country": normalize_country(row.get("country")),
                "state_province": normalize_state(row.get("state_province")),
                "locality": normalize_text(row.get("locality")),
                "event_date": normalize_text(row.get("event_date")),
                "month": parse_month(row.get("event_date", "")),
                "basis_of_record": normalize_text(row.get("basis_of_record")),
                "license": normalize_text(row.get("license")),
                "rights_holder": normalize_text(row.get("rights_holder")),
                "institution_code": normalize_text(row.get("institution_code")),
                "catalog_number": normalize_text(row.get("catalog_number")),
                "image_url": image_url,
                "image_format": normalize_text(row.get("image_format")),
                "record_url": normalize_text(row.get("record_url")),
                "download_status": download_status,
                "download_error": normalize_text(row.get("download_error")),
                "raw_species": normalize_text(row.get("raw_species")),
            }
        )

    return cleaned_rows, rejected_rows, dict(rejection_counts)


def assign_species_ids(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    species_counts = Counter(row["species"] for row in rows)
    ordered_species = sorted(species_counts, key=lambda value: (-species_counts[value], value))
    species_to_id = {species: index for index, species in enumerate(ordered_species)}
    for row in rows:
        row["species_id"] = species_to_id[row["species"]]
    return species_to_id


def summarize_rows(cleaned_rows: List[Dict[str, Any]], rejected_rows: List[Dict[str, Any]], rejection_counts: Dict[str, int]) -> Dict[str, Any]:
    species_counts = Counter(row["species"] for row in cleaned_rows)
    region_counts = Counter(row["us_region"] for row in cleaned_rows)
    source_counts = Counter(row["source"] for row in cleaned_rows)

    return {
        "cleaned_rows": len(cleaned_rows),
        "rejected_rows": len(rejected_rows),
        "rejection_counts": rejection_counts,
        "unique_species": len(species_counts),
        "top_species": species_counts.most_common(25),
        "rows_by_region": dict(region_counts),
        "rows_by_source": dict(source_counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and normalize fetched butterfly metadata.")
    parser.add_argument("--input-csv", required=True, help="Raw or combined metadata CSV.")
    parser.add_argument("--output-csv", required=True, help="Path for cleaned metadata CSV.")
    parser.add_argument(
        "--species-json",
        default=None,
        help="Optional output path for species_to_id JSON.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional output path for a cleaning summary JSON.",
    )
    parser.add_argument(
        "--rejected-csv",
        default=None,
        help="Optional output path for rejected rows.",
    )
    parser.add_argument(
        "--require-downloaded",
        action="store_true",
        help="Keep only rows whose local image path exists and download status is successful.",
    )
    parser.add_argument(
        "--allow-missing-coordinates",
        action="store_true",
        help="Keep rows without coordinates for image-only training manifests.",
    )
    parser.set_defaults(filter_us_scope=True)
    parser.add_argument(
        "--filter-us-scope",
        dest="filter_us_scope",
        action="store_true",
        help="Keep only rows whose coordinates fall in the U.S. scope. Enabled by default.",
    )
    parser.add_argument(
        "--no-us-scope-filter",
        dest="filter_us_scope",
        action="store_false",
        help="Do not filter rows to U.S. coordinates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_dir = output_csv.parent
    species_json = Path(args.species_json) if args.species_json else output_dir / "species_to_id.json"
    summary_json = Path(args.summary_json) if args.summary_json else output_dir / "cleaning_summary.json"
    rejected_csv = Path(args.rejected_csv) if args.rejected_csv else output_dir / "rejected_metadata.csv"

    rows = read_rows(input_csv)
    cleaned_rows, rejected_rows, rejection_counts = clean_rows(
        rows,
        require_downloaded=args.require_downloaded,
        allow_missing_coordinates=args.allow_missing_coordinates,
        filter_us_scope=args.filter_us_scope,
    )
    species_to_id = assign_species_ids(cleaned_rows)

    write_rows(output_csv, cleaned_rows)
    write_rows(rejected_csv, rejected_rows)
    write_json(species_json, species_to_id)
    write_json(
        summary_json,
        {
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "rejected_csv": str(rejected_csv),
            "species_json": str(species_json),
            "require_downloaded": args.require_downloaded,
            "allow_missing_coordinates": args.allow_missing_coordinates,
            "filter_us_scope": args.filter_us_scope,
            "summary": summarize_rows(cleaned_rows, rejected_rows, rejection_counts),
        },
    )

    print("Done.")
    print(
        json.dumps(
            {
                "cleaned_rows": len(cleaned_rows),
                "rejected_rows": len(rejected_rows),
                "unique_species": len(species_to_id),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
