#!/usr/bin/env python3
"""
Fetch U.S. Nymphalidae specimen/image metadata from GBIF and iDigBio.

What it does:
- GBIF: resolves the Nymphalidae taxon key, then pulls U.S. occurrences with
  coordinates and still images.
- iDigBio: pulls records and media for U.S. Nymphalidae with coordinates and
  images, then joins them on record UUID.
- Normalizes species labels toward species-level binomials where possible.
- Writes source CSVs, a combined deduplicated CSV, and a summary JSON.
- Downloads image files by default and records local paths.

This script intentionally stays lightweight and uses only requests plus the
standard library so it can run easily on local machines and cloud boxes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import mimetypes
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests

GBIF_API = "https://api.gbif.org/v1"
IDIGBIO_SEARCH_API = "https://search.idigbio.org/v2"

DEFAULT_CONTACT_EMAIL = "samir.saldanha@outlook.com"
DEFAULT_USER_AGENT = f"butterfly-geo-gen/0.1 (contact: {DEFAULT_CONTACT_EMAIL})"
DEFAULT_GBIF_BASIS_OF_RECORD = "PRESERVED_SPECIMEN"
DEFAULT_REQUEST_TIMEOUT = 180
DEFAULT_REQUEST_RETRIES = 8
DEFAULT_GBIF_PAGE_SIZE = 100
DEFAULT_IDIGBIO_PAGE_SIZE = 5000
DEFAULT_DOWNLOAD_WORKERS = 12
DEFAULT_IDIGBIO_BASIS_OF_RECORD = "preservedspecimen"
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
}
THREAD_LOCAL = threading.local()
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


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
    retries: int = DEFAULT_REQUEST_RETRIES,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    last_status_detail: Optional[str] = None

    for attempt in range(retries):
        response: Optional[requests.Response] = None
        try:
            response = session.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                timeout=timeout,
                headers={"Accept": "application/json"},
            )

            if response.status_code in RETRY_STATUS_CODES:
                last_status_detail = f"HTTP {response.status_code} for {url}"
                wait_seconds = min(2**attempt, 20)
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries - 1:
                break
            wait_seconds = min(2**attempt, 20)
            time.sleep(wait_seconds)
        finally:
            if response is not None:
                response.close()

    detail = str(last_error) if last_error is not None else last_status_detail or "unknown error"
    raise RuntimeError(f"Request failed: {url}\n{detail}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or "file"


def guess_extension(url: str, content_type: str = "") -> str:
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_SUFFIXES:
        return ext
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return guessed
    return ".jpg"


def normalize_image_url(url: Any) -> str:
    return normalize_text(url)


def normalize_species_binomial(
    scientific_name: Any,
    genus: Any,
    species_value: Any,
) -> Tuple[str, bool]:
    genus_text = normalize_text(genus)
    species_text = normalize_text(species_value)
    candidates: List[Tuple[str, str]] = []

    species_tokens = [part for part in re.split(r"\s+", species_text) if part]
    if len(species_tokens) >= 2:
        candidates.append(("species_value_binomial", species_text))
    elif genus_text and species_text:
        candidates.append(("genus_plus_species", f"{genus_text} {species_text}"))

    candidates.append(("scientific_name", normalize_text(scientific_name)))

    for source_name, candidate in candidates:
        raw_parts = [part for part in re.split(r"\s+", candidate) if part]
        parts = list(raw_parts)
        if len(parts) < 2:
            continue

        genus_part = re.sub(r"[^A-Za-z-]", "", parts[0])
        species_part = re.sub(r"[^A-Za-z-]", "", parts[1]).lower()
        if not genus_part or not species_part:
            continue
        if species_part in NON_SPECIES_TOKENS:
            continue
        if not re.fullmatch(r"[A-Za-z-]+", genus_part):
            continue
        if not re.fullmatch(r"[a-z-]+", species_part):
            continue
        if source_name == "scientific_name":
            second_token = raw_parts[1]
            if not second_token[:1].islower():
                continue
        if source_name == "species_value_binomial" and genus_text and genus_part.lower() != genus_text.lower():
            continue

        normalized = f"{genus_part[0].upper()}{genus_part[1:].lower()} {species_part}"
        return normalized, True

    return "", False


def build_row(
    *,
    source: str,
    record_id: Any,
    media_id: Any,
    institution_code: Any,
    catalog_number: Any,
    family: Any,
    genus: Any,
    species: Any,
    scientific_name: Any,
    latitude: Any,
    longitude: Any,
    country: Any,
    state_province: Any,
    locality: Any,
    event_date: Any,
    basis_of_record: Any,
    license_value: Any,
    rights_holder: Any,
    image_url: Any,
    image_format: Any,
    publisher: Any,
    dataset_key: Any,
    record_url: Any,
) -> Dict[str, Any]:
    genus_text = normalize_text(genus)
    raw_species = normalize_text(species)
    scientific_name_text = normalize_text(scientific_name)
    canonical_species, is_species_level = normalize_species_binomial(
        scientific_name_text,
        genus_text,
        raw_species,
    )

    return {
        "source": normalize_text(source),
        "record_id": normalize_text(record_id),
        "media_id": normalize_text(media_id),
        "institution_code": normalize_text(institution_code),
        "catalog_number": normalize_text(catalog_number),
        "family": normalize_text(family),
        "genus": genus_text,
        "species": canonical_species or raw_species,
        "raw_species": raw_species,
        "scientific_name": scientific_name_text,
        "is_species_level": is_species_level,
        "latitude": latitude,
        "longitude": longitude,
        "country": normalize_text(country),
        "state_province": normalize_text(state_province),
        "locality": normalize_text(locality),
        "event_date": normalize_text(event_date),
        "basis_of_record": normalize_text(basis_of_record),
        "license": normalize_text(license_value),
        "rights_holder": normalize_text(rights_holder),
        "image_url": normalize_image_url(image_url),
        "image_format": normalize_text(image_format),
        "publisher": normalize_text(publisher),
        "dataset_key": normalize_text(dataset_key),
        "record_url": normalize_text(record_url),
        "local_image_path": "",
        "download_status": "pending",
        "download_error": "",
    }


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    species_counter = Counter(
        row["species"]
        for row in rows
        if normalize_text(row.get("species")) and row.get("is_species_level")
    )
    source_counter = Counter(row.get("source", "") for row in rows)

    latitudes = [row["latitude"] for row in rows if isinstance(row.get("latitude"), (int, float))]
    longitudes = [row["longitude"] for row in rows if isinstance(row.get("longitude"), (int, float))]

    return {
        "rows": len(rows),
        "rows_with_species_level_label": sum(1 for row in rows if row.get("is_species_level")),
        "unique_species": len(species_counter),
        "top_species": species_counter.most_common(20),
        "by_source": dict(source_counter),
        "latitude_range": [min(latitudes), max(latitudes)] if latitudes else [],
        "longitude_range": [min(longitudes), max(longitudes)] if longitudes else [],
    }


def page_limit(page_size: int, max_records: Optional[int], offset: int) -> int:
    if max_records is None:
        return page_size
    return min(page_size, max(max_records - offset, 0))


def should_continue(offset: int, max_records: Optional[int]) -> bool:
    return max_records is None or offset < max_records


def resolve_gbif_taxon_key(session: requests.Session, name: str = "Nymphalidae") -> int:
    data = request_json(
        session,
        "GET",
        f"{GBIF_API}/species/match",
        params={"name": name, "rank": "family"},
    )
    usage_key = data.get("usageKey")
    if not usage_key:
        raise RuntimeError(f"Could not resolve GBIF taxon key for {name!r}")
    return int(usage_key)


def gbif_rows(
    session: requests.Session,
    max_records: Optional[int],
    basis_of_record: Optional[str],
    *,
    page_size: int,
    request_timeout: int,
    request_retries: int,
    progress_callback: Optional[Callable[[str, int, int, Optional[int]], None]] = None,
) -> List[Dict[str, Any]]:
    taxon_key = resolve_gbif_taxon_key(session)
    rows: List[Dict[str, Any]] = []
    offset = 0
    limit = page_size
    total_estimate: Optional[int] = None

    while should_continue(offset, max_records):
        current_limit = page_limit(limit, max_records, offset)
        if current_limit <= 0:
            break

        params: Dict[str, Any] = {
            "taxonKey": taxon_key,
            "country": "US",
            "hasCoordinate": "true",
            "mediaType": "StillImage",
            "limit": current_limit,
            "offset": offset,
        }
        if basis_of_record:
            params["basisOfRecord"] = basis_of_record

        page = request_json(
            session,
            "GET",
            f"{GBIF_API}/occurrence/search",
            params=params,
            timeout=request_timeout,
            retries=request_retries,
        )

        results = page.get("results", [])
        if not results:
            break
        if total_estimate is None:
            raw_total = page.get("count")
            if isinstance(raw_total, int):
                total_estimate = raw_total

        for occurrence in results:
            media_items = as_list(occurrence.get("media"))
            image_media = []
            for media in media_items:
                identifier = normalize_image_url(media.get("identifier"))
                media_type = normalize_text(media.get("type"))
                image_format = normalize_text(media.get("format"))
                if not identifier:
                    continue
                if (
                    media_type == "StillImage"
                    or image_format.startswith("image/")
                    or identifier.lower().endswith(tuple(IMAGE_SUFFIXES))
                ):
                    image_media.append(media)

            if not image_media:
                continue

            for index, media in enumerate(image_media, start=1):
                rows.append(
                    build_row(
                        source="gbif",
                        record_id=occurrence.get("key"),
                        media_id=media.get("identifier") or f"gbif-media-{occurrence.get('key')}-{index}",
                        institution_code=occurrence.get("institutionCode"),
                        catalog_number=occurrence.get("catalogNumber"),
                        family=occurrence.get("family"),
                        genus=occurrence.get("genus"),
                        species=occurrence.get("species"),
                        scientific_name=first_nonempty(
                            occurrence.get("acceptedScientificName"),
                            occurrence.get("scientificName"),
                        ),
                        latitude=occurrence.get("decimalLatitude"),
                        longitude=occurrence.get("decimalLongitude"),
                        country=occurrence.get("country"),
                        state_province=occurrence.get("stateProvince"),
                        locality=occurrence.get("locality"),
                        event_date=occurrence.get("eventDate"),
                        basis_of_record=occurrence.get("basisOfRecord"),
                        license_value=first_nonempty(media.get("license"), occurrence.get("license")),
                        rights_holder=first_nonempty(
                            media.get("rightsHolder"),
                            occurrence.get("rightsHolder"),
                            media.get("creator"),
                        ),
                        image_url=media.get("identifier"),
                        image_format=media.get("format"),
                        publisher=occurrence.get("publisher"),
                        dataset_key=occurrence.get("datasetKey"),
                        record_url=f"https://www.gbif.org/occurrence/{occurrence.get('key')}",
                    )
                )

        offset += len(results)
        if progress_callback is not None:
            progress_callback("fetching_gbif", offset, len(rows), total_estimate)
        if offset % max(500, limit * 5) == 0:
            print(f"GBIF progress: scanned {offset} occurrences, kept {len(rows)} image rows...")
        if page.get("endOfRecords"):
            break
        time.sleep(0.2)

    return rows


def idigbio_post_search(
    session: requests.Session,
    endpoint: str,
    body: Dict[str, Any],
    *,
    request_timeout: int,
    request_retries: int,
) -> Dict[str, Any]:
    return request_json(
        session,
        "POST",
        f"{IDIGBIO_SEARCH_API}/{endpoint.lstrip('/')}",
        json_body=body,
        timeout=request_timeout,
        retries=request_retries,
    )


def parse_idigbio_geopoint(value: Any) -> Tuple[Optional[float], Optional[float]]:
    if isinstance(value, dict):
        lat = value.get("lat")
        lon = value.get("lon")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return float(lat), float(lon)
    if isinstance(value, list) and len(value) >= 2:
        lon, lat = value[0], value[1]
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return float(lat), float(lon)
    if isinstance(value, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(numbers) >= 2:
            first = float(numbers[0])
            second = float(numbers[1])
            if -90 <= first <= 90 and -180 <= second <= 180:
                return first, second
            if -180 <= first <= 180 and -90 <= second <= 90:
                return second, first
    return None, None


def build_idigbio_query(
    *,
    include_country_filter: bool,
    require_geopoint: bool,
    basis_of_record: Optional[str],
) -> Dict[str, Any]:
    query: Dict[str, Any] = {
        "family": "nymphalidae",
        "hasImage": "true",
    }
    if include_country_filter:
        query["country"] = "united states"
    if require_geopoint:
        query["geopoint"] = {"type": "exists"}
    if basis_of_record:
        query["basisofrecord"] = basis_of_record
    return query


def idigbio_record_rows(
    session: requests.Session,
    max_records: Optional[int],
    *,
    page_size: int,
    request_timeout: int,
    request_retries: int,
    include_country_filter: bool,
    require_geopoint: bool,
    basis_of_record: Optional[str],
    progress_callback: Optional[Callable[[str, int, int, Optional[int]], None]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    total_estimate: Optional[int] = None

    query = build_idigbio_query(
        include_country_filter=include_country_filter,
        require_geopoint=require_geopoint,
        basis_of_record=basis_of_record,
    )

    fields = [
        "uuid",
        "institutioncode",
        "catalognumber",
        "family",
        "genus",
        "specificepithet",
        "scientificname",
        "country",
        "stateprovince",
        "locality",
        "geopoint",
        "datecollected",
        "eventdate",
        "basisofrecord",
    ]

    while should_continue(offset, max_records):
        current_limit = page_limit(page_size, max_records, offset)
        if current_limit <= 0:
            break

        body = {
            "rq": query,
            "fields": fields,
            "offset": offset,
            "limit": current_limit,
            "sort": ["uuid"],
        }
        page = idigbio_post_search(
            session,
            "search/records",
            body,
            request_timeout=request_timeout,
            request_retries=request_retries,
        )
        items = page.get("items", [])
        if not items:
            break
        if total_estimate is None:
            raw_total = first_nonempty(
                page.get("itemCount"),
                page.get("itemcount"),
                page.get("count"),
                page.get("totalItems"),
            )
            if isinstance(raw_total, int):
                total_estimate = raw_total

        for item in items:
            index_terms = item.get("indexTerms", {}) or {}
            data_terms = item.get("data", {}) or {}

            geopoint = first_nonempty(index_terms.get("geopoint"), data_terms.get("geopoint"))
            latitude, longitude = parse_idigbio_geopoint(geopoint)

            rows.append(
                {
                    "record_uuid": normalize_text(
                        first_nonempty(index_terms.get("uuid"), data_terms.get("idigbio:uuid"))
                    ),
                    "institution_code": normalize_text(index_terms.get("institutioncode")),
                    "catalog_number": normalize_text(index_terms.get("catalognumber")),
                    "family": normalize_text(index_terms.get("family")),
                    "genus": normalize_text(index_terms.get("genus")),
                    "species": normalize_text(index_terms.get("specificepithet")),
                    "scientific_name": normalize_text(index_terms.get("scientificname")),
                    "latitude": latitude,
                    "longitude": longitude,
                    "country": normalize_text(index_terms.get("country")),
                    "state_province": normalize_text(index_terms.get("stateprovince")),
                    "locality": normalize_text(index_terms.get("locality")),
                    "event_date": normalize_text(
                        first_nonempty(index_terms.get("datecollected"), index_terms.get("eventdate"))
                    ),
                    "basis_of_record": normalize_text(index_terms.get("basisofrecord")),
                }
            )

        offset += len(items)
        if progress_callback is not None:
            progress_callback("fetching_idigbio_records", offset, len(rows), total_estimate)
        if offset % max(5000, page_size) == 0:
            print(f"iDigBio record progress: scanned {offset} records, kept {len(rows)} metadata rows...")
        if len(items) < current_limit:
            break
        time.sleep(0.2)

    return rows


def idigbio_media_rows(
    session: requests.Session,
    max_records: Optional[int],
    *,
    page_size: int,
    request_timeout: int,
    request_retries: int,
    include_country_filter: bool,
    require_geopoint: bool,
    basis_of_record: Optional[str],
    progress_callback: Optional[Callable[[str, int, int, Optional[int]], None]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    total_estimate: Optional[int] = None

    record_query = build_idigbio_query(
        include_country_filter=include_country_filter,
        require_geopoint=require_geopoint,
        basis_of_record=basis_of_record,
    )
    media_query = {"data.ac:accessURI": {"type": "exists"}}

    fields = [
        "uuid",
        "accessuri",
        "rights",
        "webstatement",
        "records",
        "format",
        "datemodified",
    ]

    while should_continue(offset, max_records):
        current_limit = page_limit(page_size, max_records, offset)
        if current_limit <= 0:
            break

        body = {
            "rq": record_query,
            "mq": media_query,
            "fields": fields,
            "offset": offset,
            "limit": current_limit,
            "sort": ["uuid"],
        }
        page = idigbio_post_search(
            session,
            "search/media",
            body,
            request_timeout=request_timeout,
            request_retries=request_retries,
        )
        items = page.get("items", [])
        if not items:
            break
        if total_estimate is None:
            raw_total = first_nonempty(
                page.get("itemCount"),
                page.get("itemcount"),
                page.get("count"),
                page.get("totalItems"),
            )
            if isinstance(raw_total, int):
                total_estimate = raw_total

        for item in items:
            index_terms = item.get("indexTerms", {}) or {}
            records = [normalize_text(record_uuid) for record_uuid in as_list(index_terms.get("records"))]
            media_uuid = normalize_text(index_terms.get("uuid"))
            access_uri = normalize_image_url(index_terms.get("accessuri"))
            if not media_uuid or not access_uri:
                continue

            media_row = {
                "media_uuid": media_uuid,
                "image_url": access_uri,
                "image_format": normalize_text(index_terms.get("format")),
                "license": normalize_text(
                    first_nonempty(index_terms.get("webstatement"), index_terms.get("rights"))
                ),
            }

            if not records:
                rows.append({"record_uuid": "", **media_row})
                continue

            for record_uuid in records:
                rows.append({"record_uuid": record_uuid, **media_row})

        offset += len(items)
        if progress_callback is not None:
            progress_callback("fetching_idigbio_media", offset, len(rows), total_estimate)
        if offset % max(5000, page_size) == 0:
            print(f"iDigBio media progress: scanned {offset} media rows, kept {len(rows)} image rows...")
        if len(items) < current_limit:
            break
        time.sleep(0.2)

    return rows


def idigbio_rows(
    session: requests.Session,
    max_records: Optional[int],
    *,
    page_size: int,
    request_timeout: int,
    request_retries: int,
    include_country_filter: bool,
    require_geopoint: bool,
    basis_of_record: Optional[str],
    progress_callback: Optional[Callable[[str, int, int, Optional[int]], None]] = None,
) -> List[Dict[str, Any]]:
    records = idigbio_record_rows(
        session,
        max_records,
        page_size=page_size,
        request_timeout=request_timeout,
        request_retries=request_retries,
        include_country_filter=include_country_filter,
        require_geopoint=require_geopoint,
        basis_of_record=basis_of_record,
        progress_callback=progress_callback,
    )
    media = idigbio_media_rows(
        session,
        max_records,
        page_size=page_size,
        request_timeout=request_timeout,
        request_retries=request_retries,
        include_country_filter=include_country_filter,
        require_geopoint=require_geopoint,
        basis_of_record=basis_of_record,
        progress_callback=progress_callback,
    )

    records_by_uuid = {row["record_uuid"]: row for row in records if row.get("record_uuid")}
    joined: List[Dict[str, Any]] = []

    for media_row in media:
        record_uuid = media_row.get("record_uuid", "")
        record = records_by_uuid.get(record_uuid)
        if not record:
            continue

        joined.append(
            build_row(
                source="idigbio",
                record_id=record_uuid,
                media_id=media_row.get("media_uuid", ""),
                institution_code=record.get("institution_code", ""),
                catalog_number=record.get("catalog_number", ""),
                family=record.get("family", ""),
                genus=record.get("genus", ""),
                species=record.get("species", ""),
                scientific_name=record.get("scientific_name", ""),
                latitude=record.get("latitude"),
                longitude=record.get("longitude"),
                country=record.get("country", ""),
                state_province=record.get("state_province", ""),
                locality=record.get("locality", ""),
                event_date=record.get("event_date", ""),
                basis_of_record=record.get("basis_of_record", ""),
                license_value=media_row.get("license", ""),
                rights_holder="",
                image_url=media_row.get("image_url", ""),
                image_format=media_row.get("image_format", ""),
                publisher="",
                dataset_key="",
                record_url=f"https://search.idigbio.org/v2/view/records/{record_uuid}",
            )
        )

    return joined


def dedupe_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_urls = set()
    deduped_rows: List[Dict[str, Any]] = []

    for row in rows:
        image_url = normalize_image_url(row.get("image_url"))
        if not image_url or image_url in seen_urls:
            continue
        seen_urls.add(image_url)
        deduped_rows.append(row)

    return deduped_rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_download_path(outdir: Path, row: Dict[str, Any], image_url: str) -> Path:
    source_dir = outdir / row["source"]
    ensure_dir(source_dir)

    url_hash = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:12]
    extension = guess_extension(image_url, row.get("image_format", ""))
    stem = safe_filename(f"{row['source']}_{row['record_id']}_{url_hash}")
    return source_dir / f"{stem}{extension}"


def get_thread_session(user_agent: str) -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    session_user_agent = getattr(THREAD_LOCAL, "user_agent", None)
    if session is None or session_user_agent != user_agent:
        session = make_session(user_agent)
        THREAD_LOCAL.session = session
        THREAD_LOCAL.user_agent = user_agent
    return session


def download_single_image(
    index: int,
    row: Dict[str, Any],
    outdir: Path,
    user_agent: str,
    request_timeout: int,
) -> Dict[str, Any]:
    image_url = normalize_image_url(row.get("image_url"))
    if not image_url:
        return {
            "index": index,
            "local_image_path": "",
            "download_status": "missing_url",
            "download_error": "Missing image_url",
        }

    filepath = build_download_path(outdir, row, image_url)
    if filepath.exists():
        return {
            "index": index,
            "local_image_path": str(filepath),
            "download_status": "exists",
            "download_error": "",
        }

    session = get_thread_session(user_agent)
    response: Optional[requests.Response] = None

    try:
        response = session.get(image_url, timeout=request_timeout, stream=True)
        if response.status_code in RETRY_STATUS_CODES:
            time.sleep(2)
            response.close()
            response = session.get(image_url, timeout=request_timeout, stream=True)

        response.raise_for_status()
        final_path = filepath.with_suffix(
            guess_extension(image_url, response.headers.get("Content-Type", ""))
        )
        ensure_dir(final_path.parent)
        with final_path.open("wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    file_handle.write(chunk)

        return {
            "index": index,
            "local_image_path": str(final_path),
            "download_status": "downloaded",
            "download_error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "index": index,
            "local_image_path": str(filepath),
            "download_status": "failed",
            "download_error": str(exc),
        }
    finally:
        if response is not None:
            response.close()


def download_images(
    rows: List[Dict[str, Any]],
    outdir: Path,
    *,
    user_agent: str,
    request_timeout: int,
    download_workers: int,
    progress_callback: Optional[Callable[[int, int, int, int, int], None]] = None,
) -> Dict[str, int]:
    ensure_dir(outdir)

    downloaded = 0
    skipped_existing = 0
    failed = 0

    def task(item: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        index, row = item
        return download_single_image(
            index,
            row,
            outdir,
            user_agent,
            request_timeout,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=download_workers) as executor:
        for completed_count, result in enumerate(executor.map(task, enumerate(rows)), start=1):
            row = rows[result["index"]]
            row["local_image_path"] = result["local_image_path"]
            row["download_status"] = result["download_status"]
            row["download_error"] = result["download_error"]

            status = result["download_status"]
            if status == "downloaded":
                downloaded += 1
            elif status == "exists":
                skipped_existing += 1
            else:
                failed += 1
                if status == "failed":
                    print(
                        f"[warn] failed to download {row.get('image_url', '')}: {result['download_error']}",
                        file=sys.stderr,
                    )

            if completed_count % 500 == 0:
                if progress_callback is not None:
                    progress_callback(completed_count, len(rows), downloaded, skipped_existing, failed)
                print(
                    "Download progress: "
                    f"{completed_count}/{len(rows)} completed, "
                    f"{downloaded} downloaded, {skipped_existing} existing, {failed} failed..."
                )

        if progress_callback is not None:
            progress_callback(len(rows), len(rows), downloaded, skipped_existing, failed)

    return {
        "downloaded": downloaded,
        "skipped_existing": skipped_existing,
        "failed": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch U.S. Nymphalidae image metadata from GBIF and iDigBio.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for metadata, images, and summary files.",
    )
    parser.add_argument(
        "--gbif-max-records",
        type=int,
        default=None,
        help="Max GBIF occurrence records to scan. Defaults to exhausting the API.",
    )
    parser.add_argument(
        "--idigbio-max-records",
        type=int,
        default=None,
        help="Max iDigBio records/media to scan. Defaults to exhausting the API.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent header sent to upstream APIs.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=DEFAULT_REQUEST_RETRIES,
        help="How many times to retry transient API failures.",
    )
    parser.add_argument(
        "--gbif-page-size",
        type=int,
        default=DEFAULT_GBIF_PAGE_SIZE,
        help="GBIF page size per occurrence search request.",
    )
    parser.add_argument(
        "--idigbio-page-size",
        type=int,
        default=DEFAULT_IDIGBIO_PAGE_SIZE,
        help="iDigBio page size per records/media request.",
    )
    parser.add_argument(
        "--idigbio-basis-of-record",
        default=DEFAULT_IDIGBIO_BASIS_OF_RECORD,
        help=(
            "Optional iDigBio basisofrecord filter. Defaults to preservedspecimen "
            "to keep specimen-oriented dorsal-view data."
        ),
    )
    parser.set_defaults(idigbio_country_filter=False)
    parser.add_argument(
        "--idigbio-country-filter",
        dest="idigbio_country_filter",
        action="store_true",
        help="Apply a United States country filter directly in the iDigBio API query.",
    )
    parser.add_argument(
        "--idigbio-no-country-filter",
        dest="idigbio_country_filter",
        action="store_false",
        help="Do not apply a country filter in the iDigBio API query. Enabled by default.",
    )
    parser.set_defaults(idigbio_require_geopoint=False)
    parser.add_argument(
        "--idigbio-require-geopoint",
        dest="idigbio_require_geopoint",
        action="store_true",
        help="Require coordinates inside the iDigBio API query.",
    )
    parser.add_argument(
        "--idigbio-no-geopoint-filter",
        dest="idigbio_require_geopoint",
        action="store_false",
        help="Do not require coordinates in the iDigBio API query. Enabled by default.",
    )
    parser.add_argument(
        "--gbif-basis-of-record",
        default=DEFAULT_GBIF_BASIS_OF_RECORD,
        help=(
            "GBIF basisOfRecord filter. Defaults to PRESERVED_SPECIMEN to keep "
            "the dataset closer to specimen-style dorsal views."
        ),
    )
    parser.set_defaults(download_images=True)
    parser.add_argument(
        "--download-images",
        dest="download_images",
        action="store_true",
        help="Download images as part of the fetch step. Enabled by default.",
    )
    parser.add_argument(
        "--skip-image-downloads",
        dest="download_images",
        action="store_false",
        help="Fetch metadata only and skip image downloads.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=DEFAULT_DOWNLOAD_WORKERS,
        help="Number of parallel workers for image downloads.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    progress_json = outdir / "progress.json"

    progress_state: Dict[str, Any] = {
        "stage": "starting",
        "outdir": str(outdir),
        "query": {
            "family": "Nymphalidae",
            "includes_alaska_and_hawaii": True,
            "gbif_basis_of_record": args.gbif_basis_of_record,
            "idigbio_basis_of_record": args.idigbio_basis_of_record,
            "idigbio_country_filter": args.idigbio_country_filter,
            "idigbio_require_geopoint": args.idigbio_require_geopoint,
            "download_images": args.download_images,
        },
    }
    write_json(progress_json, progress_state)

    def update_progress(
        stage: str,
        scanned: Optional[int] = None,
        kept: Optional[int] = None,
        total_estimate: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        progress_state["stage"] = stage
        if scanned is not None:
            progress_state["scanned"] = scanned
        if kept is not None:
            progress_state["kept_rows"] = kept
        if total_estimate is not None:
            progress_state["source_total_estimate"] = total_estimate
        if extra:
            progress_state.update(extra)
        write_json(progress_json, progress_state)

    session = make_session(args.user_agent)

    print("Fetching GBIF rows...")
    update_progress("fetching_gbif", scanned=0, kept=0)
    gbif = dedupe_rows(
        gbif_rows(
            session,
            args.gbif_max_records,
            args.gbif_basis_of_record,
            page_size=args.gbif_page_size,
            request_timeout=args.request_timeout,
            request_retries=args.request_retries,
            progress_callback=update_progress,
        )
    )
    print(f"GBIF image rows after URL dedupe: {len(gbif)}")

    gbif_csv = outdir / "gbif_nymphalidae_us.csv"
    write_csv(gbif_csv, gbif)
    update_progress("gbif_complete", kept=len(gbif), extra={"gbif_csv": str(gbif_csv)})

    print("Fetching iDigBio rows...")
    update_progress("fetching_idigbio", scanned=0, kept=0)
    idigbio = dedupe_rows(
        idigbio_rows(
            session,
            args.idigbio_max_records,
            page_size=args.idigbio_page_size,
            request_timeout=args.request_timeout,
            request_retries=args.request_retries,
            include_country_filter=args.idigbio_country_filter,
            require_geopoint=args.idigbio_require_geopoint,
            basis_of_record=args.idigbio_basis_of_record,
            progress_callback=update_progress,
        )
    )
    print(f"iDigBio image rows after URL dedupe: {len(idigbio)}")

    idigbio_csv = outdir / "idigbio_nymphalidae_us.csv"
    write_csv(idigbio_csv, idigbio)
    update_progress("idigbio_complete", kept=len(idigbio), extra={"idigbio_csv": str(idigbio_csv)})

    combined = dedupe_rows(gbif + idigbio)
    print(f"Combined image rows after URL dedupe: {len(combined)}")

    combined_csv = outdir / "combined_nymphalidae_us.csv"
    write_csv(combined_csv, combined)
    update_progress("metadata_complete", kept=len(combined), extra={"combined_csv": str(combined_csv)})

    download_summary = {"downloaded": 0, "skipped_existing": 0, "failed": 0}
    if args.download_images:
        print("Downloading images...")
        update_progress("downloading_images", scanned=0, kept=len(combined))
        download_summary = download_images(
            combined,
            outdir / "images",
            user_agent=args.user_agent,
            request_timeout=args.request_timeout,
            download_workers=args.download_workers,
            progress_callback=lambda completed, total, downloaded, existing, failed: update_progress(
                "downloading_images",
                scanned=completed,
                kept=len(combined),
                extra={
                    "download_total": total,
                    "downloaded": downloaded,
                    "skipped_existing": existing,
                    "failed_downloads": failed,
                },
            ),
        )
        write_csv(combined_csv, combined)
        update_progress("downloads_complete", kept=len(combined), extra=download_summary)

    summary_json = outdir / "summary.json"

    summary = {
        "query": {
            "family": "Nymphalidae",
            "includes_alaska_and_hawaii": True,
            "gbif_basis_of_record": args.gbif_basis_of_record,
            "download_images": args.download_images,
            "request_timeout": args.request_timeout,
            "request_retries": args.request_retries,
            "gbif_page_size": args.gbif_page_size,
            "idigbio_page_size": args.idigbio_page_size,
            "idigbio_basis_of_record": args.idigbio_basis_of_record,
            "idigbio_country_filter": args.idigbio_country_filter,
            "idigbio_require_geopoint": args.idigbio_require_geopoint,
            "download_workers": args.download_workers,
        },
        "outputs": {
            "gbif_csv": str(gbif_csv),
            "idigbio_csv": str(idigbio_csv),
            "combined_csv": str(combined_csv),
            "images_dir": str(outdir / "images"),
        },
        "gbif": summarize_rows(gbif),
        "idigbio": summarize_rows(idigbio),
        "combined": summarize_rows(combined),
        "downloads": download_summary,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    update_progress("complete", kept=len(combined), extra={"summary_json": str(summary_json), **download_summary})

    print("Done.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
