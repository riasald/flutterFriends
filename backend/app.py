import base64
import json
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "m8q4fur9v0n4cu")
PORT = int(os.getenv("PORT", "8000"))

if not RUNPOD_API_KEY:
    raise RuntimeError("RUNPOD_API_KEY is missing from backend/.env")

RUN_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status"

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ButterflyRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    num_candidates: int = 4
    num_outputs: int = 4
    samples_per_batch: int = 1

def is_within_us(lat: float, lon: float) -> bool:
    return 24.5 <= lat <= 49.5 and -125 <= lon <= -66.5

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate-butterfly")
def generate_butterfly(req: ButterflyRequest) -> dict[str, Any]:
    if not is_within_us(req.lat, req.lon):
        raise HTTPException(
            status_code=400,
            detail="Coordinates must be within the United States."
        )
     
    payload = {
        "input": {
            "lat": req.lat,
            "lon": req.lon,
            "num_candidates": req.num_candidates,
            "num_outputs": req.num_outputs,
            "samples_per_batch": req.samples_per_batch,
        }
    }

    try:
        run_resp = requests.post(RUN_URL, headers=HEADERS, json=payload, timeout=60)
        run_resp.raise_for_status()
    except requests.RequestException as e:
        detail = getattr(e.response, "text", str(e))
        raise HTTPException(status_code=502, detail=f"Failed to submit RunPod job: {detail}")

    run_data = run_resp.json()
    job_id = run_data.get("id")

    if not job_id:
        raise HTTPException(status_code=502, detail=f"RunPod response missing job id: {run_data}")

    max_attempts = 60

    for _ in range(max_attempts):
        try:
            status_resp = requests.get(f"{STATUS_URL}/{job_id}", headers=HEADERS, timeout=60)
            status_resp.raise_for_status()
        except requests.RequestException as e:
            detail = getattr(e.response, "text", str(e))
            raise HTTPException(status_code=502, detail=f"Failed to poll RunPod status: {detail}")

        status_data = status_resp.json()
        status = status_data.get("status")

        if status == "COMPLETED":
            output = status_data.get("output", {})
            image_base64 = output.get("image_base64")

            if not image_base64:
                raise HTTPException(status_code=502, detail=f"RunPod completed without image_base64: {status_data}")

            report = output.get("report", {})
            image_url = f"data:image/png;base64,{image_base64}"

            return {
                "job_id": job_id,
                "status": status,
                "image_url": image_url,
                "report": report,
            }

        if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            raise HTTPException(
                status_code=502,
                detail=f"RunPod job ended with status {status}: {json.dumps(status_data)}",
            )

        time.sleep(5)

    raise HTTPException(status_code=504, detail="Timed out waiting for RunPod job to complete")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)