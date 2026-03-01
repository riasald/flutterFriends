import csv
import os
import time
import json
import hashlib
import threading
from io import BytesIO
from urllib.parse import urlparse

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------
# INPUT / OUTPUT PATHS
# -----------------------
IN_CSV = "../data/raw_metadata/nymphalidae_us_raw_2020_2024_humanobs.csv"

IMAGES_RAW_DIR = "../data/images_raw"
IMAGES_LICENSED_DIR = "../data/images_licensed"

OUT_LICENSED_CSV = "../data/filtered_metadata/metadata_licensed.csv"
OUT_QUALITY_CSV = "../data/filtered_metadata/metadata_quality.csv"

# Progress / resume state
STATE_DIR = "../data/filtered_metadata/_state"
PROGRESS_JSONL = os.path.join(STATE_DIR, "progress.jsonl")  # append-only log
PROGRESS_INDEX = os.path.join(STATE_DIR, "processed_keys.txt")  # fast load set

# -----------------------
# FILTER SETTINGS
# -----------------------
ALLOWED_LICENSE_KEYWORDS = [
    "cc0",
    "cc-by",
    "cc by",
    "cc-by-sa",
    "cc by-sa",
    "creativecommons.org/licenses/by",
    "creativecommons.org/publicdomain/zero",
]

MIN_SIZE = 256   # minimum width and height
REQUEST_TIMEOUT = (10, 60)

# Concurrency (network-bound → threads help a lot)
MAX_WORKERS = 16

# Politeness / throttling
SLEEP_BETWEEN_REQUESTS = 0.05  # per request, per worker (small but helps)
GLOBAL_RPS_LIMIT = 20          # approximate overall cap; adjust lower if you get 429s

# Known total rows (from your log). Optional exact counter included below.
TOTAL_ROWS_HINT = 75000

# -----------------------
# HTTP session w/ retries
# -----------------------
def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "ButterflyImageDownloader/1.0 (student project)"})
    return session

# -----------------------
# Helpers
# -----------------------
def ensure_dirs():
    os.makedirs(IMAGES_RAW_DIR, exist_ok=True)
    os.makedirs(IMAGES_LICENSED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_LICENSED_CSV), exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)

def normalize_license(s: str | None) -> str:
    return (s or "").strip().lower()

def license_allowed(license_text: str) -> bool:
    lt = normalize_license(license_text)
    if not lt:
        return False
    return any(k in lt for k in ALLOWED_LICENSE_KEYWORDS)

def guess_extension_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        if path.endswith(ext):
            return ext
    return ".jpg"

def stable_name(occ_key: str, image_url: str) -> str:
    h = hashlib.md5(image_url.encode("utf-8")).hexdigest()[:10]
    return f"{occ_key}_{h}"

def save_bytes(path: str, b: bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b)
    os.replace(tmp, path)  # atomic move

def check_image_quality(img_bytes: bytes) -> tuple[bool, int, int]:
    """Returns (ok, width, height)."""
    try:
        with Image.open(BytesIO(img_bytes)) as im:
            im = im.convert("RGB")
            w, h = im.size
            return (w >= MIN_SIZE and h >= MIN_SIZE, w, h)
    except Exception:
        return (False, 0, 0)

# -----------------------
# Simple global rate limiter (token bucket-ish)
# -----------------------
class RateLimiter:
    def __init__(self, rps: float):
        self.min_interval = 1.0 / max(rps, 1e-6)
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            dt = now - self.last
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)
            self.last = time.time()

rate_limiter = RateLimiter(GLOBAL_RPS_LIMIT)

def download_image_bytes(session: requests.Session, url: str) -> bytes | None:
    # global throttle + small worker sleep
    rate_limiter.wait()
    if SLEEP_BETWEEN_REQUESTS:
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        return None

    # If you keep hitting 429, lower GLOBAL_RPS_LIMIT and/or MAX_WORKERS
    if r.status_code == 429:
        # server asked us to slow down
        time.sleep(10)
        return None

    if r.status_code != 200:
        return None

    return r.content

# -----------------------
# Resume state
# -----------------------
def load_processed_set() -> set[str]:
    processed = set()
    if os.path.exists(PROGRESS_INDEX):
        with open(PROGRESS_INDEX, "r", encoding="utf-8") as f:
            for line in f:
                k = line.strip()
                if k:
                    processed.add(k)
    return processed

_index_lock = threading.Lock()

def append_processed_key(key: str):
    # Append to processed_keys.txt (fast to load on next run)
    with _index_lock:
        with open(PROGRESS_INDEX, "a", encoding="utf-8") as f:
            f.write(key + "\n")

_jsonl_lock = threading.Lock()

def append_progress_record(record: dict):
    # Append JSONL progress log (debug/audit)
    with _jsonl_lock:
        with open(PROGRESS_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# -----------------------
# Output CSV writers (append-only, resumable)
# -----------------------
def open_csv_append(path: str, fieldnames: list[str]):
    exists = os.path.exists(path)
    f = open(path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
        f.flush()
    return f, writer

_out_lock = threading.Lock()

def write_row(writer: csv.DictWriter, row: dict):
    with _out_lock:
        writer.writerow(row)

# -----------------------
# Worker: process a single metadata row
# -----------------------
def process_one(row: dict, session: requests.Session):
    occ_key = str(row.get("occurrence_key", "")).strip()
    if not occ_key:
        return ("skip", occ_key, None)

    image_url = (row.get("image_url") or "").strip()
    if not image_url:
        return ("skip", occ_key, None)

    license_text = row.get("license") or ""
    if not license_allowed(license_text):
        return ("skip", occ_key, None)

    species = (row.get("species") or "").strip()
    lat = row.get("lat") or ""
    lon = row.get("lon") or ""

    ext = guess_extension_from_url(image_url)
    base = stable_name(occ_key, image_url)
    raw_path = os.path.join(IMAGES_RAW_DIR, base + ext)
    licensed_path = os.path.join(IMAGES_LICENSED_DIR, base + ext)

    # If already have quality output image, we can consider it done quickly
    # (still log licensed row if output CSVs were deleted—your call; here we skip re-writing)
    if os.path.exists(licensed_path):
        return ("done_existing", occ_key, {
            "occurrence_key": occ_key,
            "species": species,
            "lat": lat,
            "lon": lon,
            "image_url": image_url,
            "license": license_text,
            "image_path_raw": raw_path,
            "width": None,
            "height": None,
            "image_path": licensed_path,
        })

    img_bytes = None

    # Download only if raw file not already present
    if os.path.exists(raw_path):
        try:
            with open(raw_path, "rb") as f:
                img_bytes = f.read()
        except Exception:
            img_bytes = None
    else:
        img_bytes = download_image_bytes(session, image_url)
        if not img_bytes:
            return ("fail_download", occ_key, None)
        # Save raw immediately
        try:
            save_bytes(raw_path, img_bytes)
        except Exception:
            return ("fail_save_raw", occ_key, None)

    # Quality check from bytes (no extra disk read if just downloaded)
    ok, w, h = check_image_quality(img_bytes)
    if not ok:
        # remove raw if too small/corrupt
        try:
            if os.path.exists(raw_path):
                os.remove(raw_path)
        except OSError:
            pass
        return ("fail_quality", occ_key, {"width": w, "height": h})

    # Save licensed copy (atomic)
    try:
        if not os.path.exists(licensed_path):
            save_bytes(licensed_path, img_bytes)
    except Exception:
        return ("fail_save_licensed", occ_key, None)

    return ("ok", occ_key, {
        "occurrence_key": occ_key,
        "species": species,
        "lat": lat,
        "lon": lon,
        "image_url": image_url,
        "license": license_text,
        "image_path_raw": raw_path,
        "width": w,
        "height": h,
        "image_path": licensed_path,
    })

# -----------------------
# Optional: exact row count (1 pass). If you already know it's 75000, skip.
# -----------------------
def count_rows_quick(path: str) -> int:
    # counts data rows excluding header
    with open(path, "r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in f) - 1

# -----------------------
# Main
# -----------------------
def main():
    ensure_dirs()
    session = make_session()

    # Resume set
    processed = load_processed_set()

    # Output writers (append)
    licensed_fields = ["occurrence_key", "species", "lat", "lon", "image_url", "license", "image_path_raw"]
    quality_fields = ["occurrence_key", "species", "lat", "lon", "image_url", "license", "width", "height", "image_path"]

    f_lic, w_lic = open_csv_append(OUT_LICENSED_CSV, licensed_fields)
    f_q, w_q = open_csv_append(OUT_QUALITY_CSV, quality_fields)

    # Total rows
    total_rows = TOTAL_ROWS_HINT
    print(f"Resuming: {len(processed)} rows already processed (from {PROGRESS_INDEX})")
    print(f"Total rows (hint): {total_rows} from earlier log")

    # Stream input + submit work in chunks so we don't queue 75k futures at once
    submitted = 0
    seen = 0
    kept_quality = 0
    kept_licensed = 0
    failed = 0
    start = time.time()

    def submit_batch(executor, batch):
        futures = []
        for r in batch:
            futures.append(executor.submit(process_one, r, session))
        return futures

    BATCH_SIZE = 1000
    INFLIGHT_MAX = MAX_WORKERS * 200  # cap queued tasks

    inflight = []

    try:
        with open(IN_CSV, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                batch = []
                for row in reader:
                    seen += 1

                    occ_key = str(row.get("occurrence_key", "")).strip()
                    if occ_key and occ_key in processed:
                        # already done on a prior run
                        continue

                    batch.append(row)

                    if len(batch) >= BATCH_SIZE:
                        inflight.extend(submit_batch(ex, batch))
                        submitted += len(batch)
                        batch = []

                    # Keep inflight bounded
                    if len(inflight) >= INFLIGHT_MAX:
                        # drain some completed futures
                        new_inflight = []
                        for fut in as_completed(inflight, timeout=None):
                            status, key, payload = fut.result()
                            if key:
                                processed.add(key)
                                append_processed_key(key)

                            # Write outputs if success
                            if status in ("ok", "done_existing") and payload:
                                # licensed row always (license-approved)
                                write_row(w_lic, {
                                    "occurrence_key": payload["occurrence_key"],
                                    "species": payload["species"],
                                    "lat": payload["lat"],
                                    "lon": payload["lon"],
                                    "image_url": payload["image_url"],
                                    "license": payload["license"],
                                    "image_path_raw": payload["image_path_raw"],
                                })
                                kept_licensed += 1

                                # quality row only when we actually know dimensions
                                if payload.get("width") is not None:
                                    write_row(w_q, {
                                        "occurrence_key": payload["occurrence_key"],
                                        "species": payload["species"],
                                        "lat": payload["lat"],
                                        "lon": payload["lon"],
                                        "image_url": payload["image_url"],
                                        "license": payload["license"],
                                        "width": payload["width"],
                                        "height": payload["height"],
                                        "image_path": payload["image_path"],
                                    })
                                    kept_quality += 1
                            elif status.startswith("fail"):
                                failed += 1

                            append_progress_record({
                                "status": status,
                                "occurrence_key": key,
                                "seen_row": seen,
                                "ts": time.time(),
                            })

                            # stop draining once we've reduced inflight enough
                            # (we can't easily break as_completed, so we rebuild)
                        # after draining everything, reset inflight
                        inflight = new_inflight

                    # progress print based on file position (seen out of total)
                    if seen % 500 == 0:
                        elapsed = time.time() - start
                        rate = seen / elapsed if elapsed > 0 else 0
                        print(
                            f"Seen {seen}/{total_rows} rows | "
                            f"submitted={submitted} | "
                            f"licensed={kept_licensed} | quality={kept_quality} | failed={failed} | "
                            f"{rate:.1f} rows/s"
                        )

                # submit remainder batch
                if batch:
                    inflight.extend(submit_batch(ex, batch))
                    submitted += len(batch)

                # drain all remaining inflight futures
                for fut in as_completed(inflight):
                    status, key, payload = fut.result()
                    if key:
                        processed.add(key)
                        append_processed_key(key)

                    if status in ("ok", "done_existing") and payload:
                        write_row(w_lic, {
                            "occurrence_key": payload["occurrence_key"],
                            "species": payload["species"],
                            "lat": payload["lat"],
                            "lon": payload["lon"],
                            "image_url": payload["image_url"],
                            "license": payload["license"],
                            "image_path_raw": payload["image_path_raw"],
                        })
                        kept_licensed += 1

                        if payload.get("width") is not None:
                            write_row(w_q, {
                                "occurrence_key": payload["occurrence_key"],
                                "species": payload["species"],
                                "lat": payload["lat"],
                                "lon": payload["lon"],
                                "image_url": payload["image_url"],
                                "license": payload["license"],
                                "width": payload["width"],
                                "height": payload["height"],
                                "image_path": payload["image_path"],
                            })
                            kept_quality += 1
                    elif status.startswith("fail"):
                        failed += 1

                    append_progress_record({
                        "status": status,
                        "occurrence_key": key,
                        "seen_row": seen,
                        "ts": time.time(),
                    })

    finally:
        # flush and close outputs
        try:
            f_lic.flush()
            f_q.flush()
        except Exception:
            pass
        f_lic.close()
        f_q.close()

    elapsed = time.time() - start
    print("\nDone!")
    print(f"Seen rows: {seen}/{total_rows}")
    print(f"Submitted (not previously processed): {submitted}")
    print(f"License-approved rows appended: {kept_licensed} -> {OUT_LICENSED_CSV}")
    print(f"Quality-approved rows appended:  {kept_quality} -> {OUT_QUALITY_CSV}")
    print(f"Failed rows this run: {failed}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Images saved to: {IMAGES_RAW_DIR} and {IMAGES_LICENSED_DIR}")
    print(f"Resume state: {PROGRESS_INDEX} and {PROGRESS_JSONL}")

if __name__ == "__main__":
    main()