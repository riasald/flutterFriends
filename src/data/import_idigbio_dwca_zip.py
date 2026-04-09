#!/usr/bin/env python3
"""
Import an iDigBio DwC-A bulk download zip into the project's raw manifest schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from src.data.fetch_butterfly_gbif_idigbio import (
    IMAGE_SUFFIXES,
    build_row,
    dedupe_rows,
    normalize_image_url,
    normalize_text,
    parse_idigbio_geopoint,
    write_csv,
)


def is_image_like(media_type: str, image_format: str, image_url: str) -> bool:
    media_type = normalize_text(media_type).lower()
    image_format = normalize_text(image_format).lower()
    image_url = normalize_image_url(image_url).lower()

    if media_type in {"image", "stillimage"}:
        return True
    if image_format.startswith("image/"):
        return True
    if image_format in {"jpeg", "jpg", "png", "tiff", "tif", "webp", "gif", "bmp"}:
        return True
    return image_url.endswith(tuple(IMAGE_SUFFIXES))


def read_csv_from_zip(archive: zipfile.ZipFile, member_name: str) -> csv.DictReader:
    file_handle = archive.open(member_name)
    text = (line.decode("utf-8-sig", errors="replace") for line in file_handle)
    return csv.DictReader(text)


def load_occurrences(archive: zipfile.ZipFile) -> Dict[str, Dict[str, Any]]:
    occurrences: Dict[str, Dict[str, Any]] = {}
    reader = read_csv_from_zip(archive, "occurrence.csv")
    for row in reader:
        core_id = normalize_text(row.get("coreid"))
        if not core_id:
            continue
        latitude, longitude = parse_idigbio_geopoint(row.get("idigbio:geoPoint"))
        occurrences[core_id] = {
            "record_id": core_id,
            "institution_code": normalize_text(row.get("dwc:institutionCode")),
            "catalog_number": normalize_text(row.get("dwc:catalogNumber")),
            "family": normalize_text(row.get("dwc:family")),
            "genus": normalize_text(row.get("dwc:genus")),
            "species": normalize_text(row.get("gbif:canonicalName") or row.get("dwc:scientificName")),
            "raw_species": normalize_text(row.get("dwc:specificEpithet")),
            "scientific_name": normalize_text(row.get("dwc:scientificName")),
            "latitude": latitude,
            "longitude": longitude,
            "country": normalize_text(row.get("dwc:country")),
            "state_province": normalize_text(row.get("dwc:stateProvince")),
            "locality": normalize_text(row.get("dwc:locality") or row.get("dwc:verbatimLocality")),
            "event_date": normalize_text(row.get("idigbio:eventDate") or row.get("dwc:eventDate")),
            "basis_of_record": normalize_text(row.get("dwc:basisOfRecord")),
            "record_url": f"https://search.idigbio.org/v2/view/records/{core_id}",
        }
    return occurrences


def import_rows(zip_path: Path) -> List[Dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as archive:
        occurrences = load_occurrences(archive)
        media_reader = read_csv_from_zip(archive, "multimedia.csv")
        rows: List[Dict[str, Any]] = []

        for media_row in media_reader:
            core_id = normalize_text(media_row.get("coreid"))
            occurrence = occurrences.get(core_id)
            if not occurrence:
                continue

            image_url = normalize_image_url(media_row.get("ac:accessURI"))
            if not image_url:
                continue
            image_format = normalize_text(media_row.get("dcterms:format"))
            media_type = normalize_text(media_row.get("dc:type") or media_row.get("idigbio:mediaType"))
            if not is_image_like(media_type, image_format, image_url):
                continue

            rows.append(
                build_row(
                    source="idigbio_dwca",
                    record_id=occurrence["record_id"],
                    media_id=normalize_text(media_row.get("idigbio:uuid") or media_row.get("coreid")),
                    institution_code=occurrence["institution_code"],
                    catalog_number=occurrence["catalog_number"],
                    family=occurrence["family"],
                    genus=occurrence["genus"],
                    species=occurrence["species"] or occurrence["raw_species"],
                    scientific_name=occurrence["scientific_name"],
                    latitude=occurrence["latitude"],
                    longitude=occurrence["longitude"],
                    country=occurrence["country"],
                    state_province=occurrence["state_province"],
                    locality=occurrence["locality"],
                    event_date=occurrence["event_date"],
                    basis_of_record=occurrence["basis_of_record"],
                    license_value=normalize_text(
                        media_row.get("xmpRights:WebStatement") or media_row.get("dcterms:rights")
                    ),
                    rights_holder="",
                    image_url=image_url,
                    image_format=image_format,
                    publisher="",
                    dataset_key="",
                    record_url=occurrence["record_url"],
                )
            )

    return dedupe_rows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import iDigBio DwC-A zip into raw manifest CSV.")
    parser.add_argument("--zip-path", required=True, help="Path to the iDigBio bulk download zip.")
    parser.add_argument("--output-csv", required=True, help="Output CSV in raw manifest format.")
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip_path)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json) if args.summary_json else output_csv.with_suffix(".summary.json")

    rows = import_rows(zip_path)
    write_csv(output_csv, rows)
    summary_json.write_text(
        json.dumps(
            {
                "zip_path": str(zip_path),
                "output_csv": str(output_csv),
                "rows": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Done.")
    print(json.dumps({"rows": len(rows), "output_csv": str(output_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
