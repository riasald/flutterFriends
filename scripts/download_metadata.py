import csv
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------
# CONFIG (edit these)
# -----------------------
TAXON_KEY = 7017
OUT_CSV = "../data/raw_metadata/nymphalidae_us_raw_2020_2024_humanobs.csv"

BASE_URL = "https://api.gbif.org/v1/occurrence/search"
LIMIT = 300

# Stop after writing this many CSV rows (each row = one image URL)
MAX_ROWS = 75_000

# If the script crashes, it will resume from the last saved offset.
PROGRESS_FILE = "../data/raw_metadata/progress_offset.txt"

# -----------------------
# HTTP session w/ retries
# -----------------------
def make_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=8,                # total retries
        backoff_factor=1.5,     # wait longer each retry: 1.5s, 3s, 6s...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({"User-Agent": "NymphalidaeDatasetBuilder/1.0 (student project)"})
    return session

# -----------------------
# Progress helpers
# -----------------------
def load_offset() -> int:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            s = f.read().strip()
            if s.isdigit():
                return int(s)
    return 0

def save_offset(offset: int) -> None:
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        f.write(str(offset))

def ensure_outdir():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

def count_existing_rows(csv_path: str) -> int:
    """Counts data rows in an existing CSV (excludes header)."""
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        # subtract header if present
        return max(0, sum(1 for _ in f) - 1)

# -----------------------
# Main fetch loop
# -----------------------
def main():
    ensure_outdir()
    session = make_session()

    # Resume-safe row count
    existing_rows = count_existing_rows(OUT_CSV)
    if existing_rows >= MAX_ROWS:
        print(f"CSV already has {existing_rows} rows (>= {MAX_ROWS}). Nothing to do.")
        return

    # If CSV doesn't exist, write header. If it exists, append.
    write_header = not os.path.exists(OUT_CSV)
    mode = "a" if os.path.exists(OUT_CSV) else "w"

    offset = load_offset()
    total_rows_written = existing_rows

    print(f"Starting at offset={offset}")
    print(f"Already have {existing_rows} rows in CSV. Will stop at {MAX_ROWS} rows total.")

    # MATCHES THIS QUERY:
    # https://api.gbif.org/v1/occurrence/search?taxonKey=7017&country=US&mediaType=StillImage&hasCoordinate=true&basisOfRecord=HUMAN_OBSERVATION&year=2020,2024&limit=1
    params = {
        "taxonKey": TAXON_KEY,
        "country": "US",# occurence country is US (for geographic modeling purposes)
        "mediaType": "StillImage", # only occurences with an image
        "hasCoordinate": "true", # has lat and long recorded
        "basisOfRecord": "HUMAN_OBSERVATION", # high-res photos, but may have messy backgrounds and not always looking down
        "year": "2020,2024",
        "limit": LIMIT, # page size
        "offset": offset
    }

    with open(OUT_CSV, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["occurrence_key", "species", "lat", "lon", "image_url", "license"])

        while total_rows_written < MAX_ROWS:
            print(f"Fetching offset: {params['offset']} (rows so far: {total_rows_written}/{MAX_ROWS})")

            try:
                r = session.get(BASE_URL, params=params, timeout=(10, 60))
            except requests.RequestException as e:
                print(f"Request failed at offset {params['offset']}: {e}")
                print("Sleeping 10s then retrying same offset...")
                time.sleep(10)
                continue

            if r.status_code == 429:
                print("Hit rate limit (429). Sleeping 30s then retrying same offset...")
                time.sleep(30)
                continue

            if r.status_code != 200:
                print(f"Non-200 status {r.status_code}. Sleeping 10s then retrying same offset...")
                time.sleep(10)
                continue

            try:
                data = r.json()
            except ValueError:
                print("Could not parse JSON. Sleeping 10s then retrying same offset...")
                time.sleep(10)
                continue

            results = data.get("results", [])
            if not results:
                print("No more results returned. Done.")
                break

            rows_written_this_page = 0

            for record in results:
                if total_rows_written >= MAX_ROWS:
                    break

                lat = record.get("decimalLatitude")
                lon = record.get("decimalLongitude")
                species = record.get("species")
                occurrence_key = record.get("key")

                media_list = record.get("media", []) or []
                for media in media_list:
                    if total_rows_written >= MAX_ROWS:
                        break

                    image_url = media.get("identifier")
                    license_ = media.get("license")

                    if image_url:
                        writer.writerow([occurrence_key, species, lat, lon, image_url, license_])
                        rows_written_this_page += 1
                        total_rows_written += 1

            f.flush()

            # Advance pagination and save progress after a successful page write
            params["offset"] += LIMIT
            save_offset(params["offset"])

            print(f"Wrote {rows_written_this_page} rows this page. Total: {total_rows_written}. Next offset={params['offset']}")

            time.sleep(1)

    if total_rows_written >= MAX_ROWS:
        print(f"Reached MAX_ROWS={MAX_ROWS}. Stopping. CSV saved at: {OUT_CSV}")
    else:
        print(f"Finished early (ran out of records). Total rows: {total_rows_written}. CSV saved at: {OUT_CSV}")

if __name__ == "__main__":
    main()