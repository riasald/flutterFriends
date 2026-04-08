#!/usr/bin/env python3
"""
Request and optionally monitor an iDigBio bulk download job.

Why this exists:
- iDigBio's official API documentation recommends the Download API for large
  result sets that exceed what is practical through the Search API.
- Our broadened preserved-specimen Nymphalidae query appears to exceed 100k
  results, so this path is more appropriate than paged interactive search.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from src.data.fetch_butterfly_gbif_idigbio import (
    DEFAULT_CONTACT_EMAIL,
    DEFAULT_IDIGBIO_BASIS_OF_RECORD,
    DEFAULT_REQUEST_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_USER_AGENT,
    RETRY_STATUS_CODES,
    build_idigbio_query,
    ensure_dir,
    write_json,
)

IDIGBIO_DOWNLOAD_API = "https://api.idigbio.org/v2/download/"


def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
    return session


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: int,
    retries: int,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    last_status: Optional[str] = None

    for attempt in range(retries):
        response: Optional[requests.Response] = None
        try:
            response = session.request(method=method, url=url, params=params, timeout=timeout)
            if response.status_code in RETRY_STATUS_CODES:
                last_status = f"HTTP {response.status_code} for {url}"
                time.sleep(min(2**attempt, 20))
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(min(2**attempt, 20))
        finally:
            if response is not None:
                response.close()

    detail = str(last_error) if last_error is not None else last_status or "unknown error"
    raise RuntimeError(f"Request failed: {url}\n{detail}")


def download_file(session: requests.Session, url: str, path: Path, timeout: int) -> None:
    ensure_dir(path.parent)
    response: Optional[requests.Response] = None
    try:
        response = session.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        with path.open("wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    file_handle.write(chunk)
    finally:
        if response is not None:
            response.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Request a bulk iDigBio download job.")
    parser.add_argument("--outdir", required=True, help="Output directory for job metadata and optional zip download.")
    parser.add_argument(
        "--status-url",
        default=None,
        help="Existing iDigBio download status URL. If provided, no new job is requested.",
    )
    parser.add_argument(
        "--email",
        default=DEFAULT_CONTACT_EMAIL,
        help="Email address for the iDigBio download job.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent header for API requests.",
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
        help="Retry count for transient failures.",
    )
    parser.add_argument(
        "--basis-of-record",
        default=DEFAULT_IDIGBIO_BASIS_OF_RECORD,
        help="iDigBio basisofrecord filter. Defaults to preservedspecimen.",
    )
    parser.set_defaults(country_filter=False)
    parser.add_argument(
        "--country-filter",
        dest="country_filter",
        action="store_true",
        help="Apply a United States country filter inside the iDigBio query.",
    )
    parser.add_argument(
        "--no-country-filter",
        dest="country_filter",
        action="store_false",
        help="Do not apply a country filter. Enabled by default.",
    )
    parser.set_defaults(require_geopoint=False)
    parser.add_argument(
        "--require-geopoint",
        dest="require_geopoint",
        action="store_true",
        help="Require coordinates in the download query.",
    )
    parser.add_argument(
        "--no-geopoint-filter",
        dest="require_geopoint",
        action="store_false",
        help="Do not require coordinates in the download query. Enabled by default.",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Poll the job status until it finishes or fails.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=300,
        help="Polling interval in seconds when --poll is used.",
    )
    parser.add_argument(
        "--download-zip",
        action="store_true",
        help="Download the DwC-A zip automatically once the job succeeds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    session = make_session(args.user_agent)
    request_json_path = outdir / "request.json"
    status_json_path = outdir / "status.json"
    if args.status_url:
        status_url = args.status_url
        if not request_json_path.exists():
            write_json(
                request_json_path,
                {
                    "download_api": IDIGBIO_DOWNLOAD_API,
                    "status_url": status_url,
                    "resumed": True,
                },
            )
        status = request_with_retry(
            session,
            "GET",
            status_url,
            timeout=args.request_timeout,
            retries=args.request_retries,
        )
        write_json(status_json_path, status)
        print(json.dumps(status, indent=2))
    else:
        query = build_idigbio_query(
            include_country_filter=args.country_filter,
            require_geopoint=args.require_geopoint,
            basis_of_record=args.basis_of_record,
        )
        request_params = {
            "rq": json.dumps(query, separators=(",", ":")),
            "email": args.email,
        }

        print("Requesting iDigBio bulk download job...")
        job = request_with_retry(
            session,
            "GET",
            IDIGBIO_DOWNLOAD_API,
            params=request_params,
            timeout=args.request_timeout,
            retries=args.request_retries,
        )
        write_json(
            request_json_path,
            {
                "download_api": IDIGBIO_DOWNLOAD_API,
                "email": args.email,
                "query": query,
                "request_params": request_params,
            },
        )
        write_json(status_json_path, job)
        status_url = job.get("status_url")
        if not status_url:
            raise RuntimeError("iDigBio download job did not return a status_url.")
        print(json.dumps(job, indent=2))

    if not args.poll:
        return 0

    while True:
        time.sleep(args.poll_interval)
        status = request_with_retry(
            session,
            "GET",
            status_url,
            timeout=args.request_timeout,
            retries=args.request_retries,
        )
        write_json(status_json_path, status)
        print(json.dumps(status, indent=2))

        task_status = str(status.get("task_status") or "").upper()
        if task_status == "SUCCESS":
            if args.download_zip and status.get("download_url"):
                zip_path = outdir / "idigbio_download.zip"
                print(f"Downloading archive to {zip_path} ...")
                download_file(session, status["download_url"], zip_path, args.request_timeout)
            return 0
        if task_status in {"FAILURE", "REVOKED"}:
            raise RuntimeError(f"iDigBio download job ended with task_status={task_status}")


if __name__ == "__main__":
    raise SystemExit(main())
