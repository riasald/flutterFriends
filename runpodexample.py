import requests
# super basic python example call to runpod. better version below.
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}

data = {
  "input": {
    "lat": 29.6516,
    "lon": -82.3248,
    "num_candidates": 1,
    "num_outputs": 1,
    "samples_per_batch": 1
  }
}


response = requests.post('https://api.runpod.ai/v2/m8q4fur9v0n4cu/run', headers=headers, json=data)



# actaul code example just need to add the api key to make this work probably.

import base64
import json
import time
from pathlib import Path

import requests

API_KEY = "YOUR_RUNPOD_API_KEY"
ENDPOINT_ID = "m8q4fur9v0n4cu"

RUN_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "input": {
        "lat": 29.6516,
        "lon": -82.3248,
        "num_candidates": 4,
        "num_outputs": 4,
        "samples_per_batch": 1,
    }
}

run_resp = requests.post(RUN_URL, headers=headers, json=payload, timeout=60)
run_resp.raise_for_status()
run_data = run_resp.json()

job_id = run_data["id"]
print("submitted job:", job_id)

while True:
    status_resp = requests.get(f"{STATUS_URL}/{job_id}", headers=headers, timeout=60)
    status_resp.raise_for_status()
    status_data = status_resp.json()

    status = status_data.get("status")
    print("status:", status)

    if status == "COMPLETED":
        output = status_data["output"]

        image_bytes = base64.b64decode(output["image_base64"])
        out_path = Path("serverless_result.png")
        out_path.write_bytes(image_bytes)
        print("saved image to:", out_path.resolve())

        report = output["report"]
        print("top species:")
        for item in report["top_species"][:5]:
            print(f"  {item['rank']}. {item['species']} ({item['probability']:.4f})")

        print("\nselected candidates:")
        for item in report["selected"]:
            print(
                f"  rank={item['rank']} quality={item['quality_score']:.3f} "
                f"brightness={item['foreground_luminance']:.3f}"
            )
        break

    if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
        print(json.dumps(status_data, indent=2))
        raise RuntimeError(f"job ended with status {status}")

    time.sleep(5)
