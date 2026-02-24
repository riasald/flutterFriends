import csv
import os
import time
import hashlib
import requests
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------
# INPUT / OUTPUT PATHS
# -----------------------
IN_CSV = "../data/raw_metadata/nymphalidae_us_raw_2020_2024_humanobs.csv"

IMAGES_RAW_DIR = "../data/images_raw"
IMAGES_LICENSED_DIR = "../data/images_licensed"

OUT_LICENSED_CSV = "../data/filtered_metadata/metadata_licensed.csv"
OUT_QUALITY_CSV = "../data/filtered_metadata/metadata_quality.csv"

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
SLEEP_BETWEEN = 0.2  # be polite (helps avoid blocks)

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
    adapter = HTTPAdapter(max_retries=retry)
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
    return ".jpg"  # default fallback

def stable_name(occ_key: str, image_url: str, idx: int) -> str:
    # Use a hash of the URL so filenames are stable even if idx changes
    h = hashlib.md5(image_url.encode("utf-8")).hexdigest()[:10]
    return f"{occ_key}_{idx}_{h}"

def download_image_bytes(session: requests.Session, url: str) -> bytes | None:
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        return None

    if r.status_code == 429:
        time.sleep(10)
        return None

    if r.status_code != 200:
        return None

    return r.content

def check_image_quality(img_bytes: bytes) -> tuple[bool, int, int]:
    """
    Returns (ok, width, height).
    ok = False if cannot open OR too small.
    """
    try:
        with Image.open(BytesIO(img_bytes)) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w < MIN_SIZE or h < MIN_SIZE:
                return (False, w, h)
            return (True, w, h)
    except Exception:
        return (False, 0, 0)

def save_bytes(path: str, b: bytes):
    with open(path, "wb") as f:
        f.write(b)

# -----------------------
# Main
# -----------------------
def main():
    ensure_dirs()
    session = make_session()

    # Read your metadata rows
    with open(IN_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    licensed_rows = []
    quality_rows = []

    print(f"Loaded {len(rows)} metadata rows from {IN_CSV}")

    for i, row in enumerate(rows, start=1):
        occ_key = str(row.get("occurrence_key", "")).strip()
        species = (row.get("species") or "").strip()
        lat = row.get("lat") or ""
        lon = row.get("lon") or ""
        image_url = (row.get("image_url") or "").strip()
        license_text = row.get("license") or ""

        if not occ_key or not image_url:
            continue

        # ---- Step 2.2: license filter first (saves time)
        if not license_allowed(license_text):
            continue

        # figure out file name
        ext = guess_extension_from_url(image_url)
        base = stable_name(occ_key, image_url, idx=1)  # idx=1 is fine since we hash URL
        raw_path = os.path.join(IMAGES_RAW_DIR, base + ext)

        # ---- Step 2.1: download (skip if already downloaded)
        if not os.path.exists(raw_path):
            img_bytes = download_image_bytes(session, image_url)
            if not img_bytes:
                continue
            save_bytes(raw_path, img_bytes)
            time.sleep(SLEEP_BETWEEN)

        # record licensed metadata row (raw download)
        licensed_rows.append({
            "occurrence_key": occ_key,
            "species": species,
            "lat": lat,
            "lon": lon,
            "image_url": image_url,
            "license": license_text,
            "image_path_raw": raw_path,
        })

        # ---- Step 2.3: quality checks (open + size)
        try:
            with open(raw_path, "rb") as fimg:
                b = fimg.read()
        except Exception:
            continue

        ok, w, h = check_image_quality(b)
        if not ok:
            # discard too small/corrupted (remove raw file)
            try:
                os.remove(raw_path)
            except OSError:
                pass
            continue

        # move/copy to licensed folder (we'll COPY to keep raw intact)
        licensed_path = os.path.join(IMAGES_LICENSED_DIR, base + ext)
        if not os.path.exists(licensed_path):
            save_bytes(licensed_path, b)

        quality_rows.append({
            "occurrence_key": occ_key,
            "species": species,
            "lat": lat,
            "lon": lon,
            "image_url": image_url,
            "license": license_text,
            "width": w,
            "height": h,
            "image_path": licensed_path,
        })

        if i % 500 == 0:
            print(f"Processed {i}/{len(rows)} rows... kept quality={len(quality_rows)}")

    # Write metadata_licensed.csv (license-approved downloads)
    with open(OUT_LICENSED_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["occurrence_key", "species", "lat", "lon", "image_url", "license", "image_path_raw"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(licensed_rows)

    # Write metadata_quality.csv (license + openable + >=256px)
    with open(OUT_QUALITY_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["occurrence_key", "species", "lat", "lon", "image_url", "license", "width", "height", "image_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(quality_rows)

    print("Done!")
    print(f"License-approved rows written: {len(licensed_rows)} -> {OUT_LICENSED_CSV}")
    print(f"Quality-approved rows written:  {len(quality_rows)} -> {OUT_QUALITY_CSV}")
    print(f"Images saved to: {IMAGES_RAW_DIR} and {IMAGES_LICENSED_DIR}")

if __name__ == "__main__":
    main()