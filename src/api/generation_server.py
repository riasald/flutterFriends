#!/usr/bin/env python3
"""Small job-based HTTP API for location-conditioned butterfly generation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.utils.config import load_yaml_config


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def project_root() -> Path:
    return Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()


def output_root() -> Path:
    return Path(os.environ.get("GENERATION_API_OUTPUT_ROOT", project_root() / "outputs" / "api_generation")).resolve()


def load_generation_defaults() -> Dict[str, Any]:
    root = project_root()
    locked_path = Path(os.environ.get("DIFFUSION_LOCKED_CONFIG", root / "configs" / "diffusion_locked.yaml"))
    locked = load_yaml_config(locked_path)
    preset = dict(locked.get("inference_preset", {}))
    return {
        "locked_config": str(locked_path),
        "config": os.environ.get("GENERATION_CONFIG", locked["config_path"]),
        "checkpoint": os.environ.get("GENERATION_CHECKPOINT", locked["checkpoint_path"]),
        "use_ema": os.environ.get("GENERATION_USE_EMA", str(locked.get("use_ema", True))).lower() not in {"0", "false", "no"},
        "num_candidates": int(os.environ.get("GENERATION_NUM_CANDIDATES", preset.get("num_candidates", 16))),
        "num_outputs": int(os.environ.get("GENERATION_NUM_OUTPUTS", preset.get("num_outputs", 8))),
        "samples_per_batch": int(os.environ.get("GENERATION_SAMPLES_PER_BATCH", preset.get("samples_per_batch", 4))),
        "guidance_scales": [float(v) for v in os.environ.get("GENERATION_GUIDANCE_SCALES", "1.9 2.3 2.7").split()],
        "prior_temperatures": [float(v) for v in os.environ.get("GENERATION_PRIOR_TEMPERATURES", "1.0 1.2 1.35").split()],
        "anchor_top_species": int(os.environ.get("GENERATION_ANCHOR_TOP_SPECIES", 6)),
        "coordinate_jitter_deg": float(os.environ.get("GENERATION_COORDINATE_JITTER_DEG", 0.35)),
        "diversity_weight": float(os.environ.get("GENERATION_DIVERSITY_WEIGHT", 0.16)),
        "min_foreground_luminance": float(os.environ.get("GENERATION_MIN_FOREGROUND_LUMINANCE", 0.10)),
        "display_scale": float(os.environ.get("GENERATION_DISPLAY_SCALE", 2.0)),
        "display_sharpen": float(os.environ.get("GENERATION_DISPLAY_SHARPEN", 0.05)),
        "device": os.environ.get("GENERATION_DEVICE", os.environ.get("DEVICE", "cuda")),
    }


class GenerateRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    seed: int = 42
    num_candidates: int | None = Field(default=None, ge=1, le=256)
    num_outputs: int | None = Field(default=None, ge=1, le=32)
    guidance_scales: List[float] | None = None
    prior_temperatures: List[float] | None = None
    anchor_top_species: int | None = Field(default=None, ge=0, le=32)
    coordinate_jitter_deg: float | None = Field(default=None, ge=0.0, le=5.0)
    min_foreground_luminance: float | None = Field(default=None, ge=0.0, le=1.0)
    save_all_candidates: bool = False


class JobRecord(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at_utc: str
    updated_at_utc: str
    request: Dict[str, Any]
    output_dir: str
    selected_grid_url: str | None = None
    report_url: str | None = None
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str = ""


DEFAULTS = load_generation_defaults()
OUTPUT_ROOT = output_root()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
JOBS: Dict[str, JobRecord] = {}
JOBS_LOCK = Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("GENERATION_API_MAX_WORKERS", "1")))

app = FastAPI(title="Butterfly Geo Generator API", version="0.1.0")

cors_origins = os.environ.get("GENERATION_API_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if cors_origins == "*" else [origin.strip() for origin in cors_origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=str(OUTPUT_ROOT)), name="files")


def update_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        payload = model_to_dict(job)
        payload.update(updates)
        payload["updated_at_utc"] = utc_now_iso()
        JOBS[job_id] = JobRecord(**payload)


def tail(text: str, max_chars: int = 4000) -> str:
    return text[-max_chars:] if len(text) > max_chars else text


def request_value(request: GenerateRequest, name: str) -> Any:
    value = getattr(request, name)
    return DEFAULTS[name] if value is None else value


def build_command(job_id: str, request: GenerateRequest, output_dir: Path) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "src.inference.generate_butterflies",
        "--config",
        str(DEFAULTS["config"]),
        "--checkpoint",
        str(DEFAULTS["checkpoint"]),
        "--lat",
        str(request.lat),
        "--lon",
        str(request.lon),
        "--device",
        str(DEFAULTS["device"]),
        "--seed",
        str(request.seed),
        "--num-candidates",
        str(request_value(request, "num_candidates")),
        "--num-outputs",
        str(request_value(request, "num_outputs")),
        "--samples-per-batch",
        str(DEFAULTS["samples_per_batch"]),
        "--guidance-scales",
        *[str(value) for value in (request.guidance_scales or DEFAULTS["guidance_scales"])],
        "--prior-temperatures",
        *[str(value) for value in (request.prior_temperatures or DEFAULTS["prior_temperatures"])],
        "--anchor-top-species",
        str(request_value(request, "anchor_top_species")),
        "--coordinate-jitter-deg",
        str(request_value(request, "coordinate_jitter_deg")),
        "--diversity-weight",
        str(DEFAULTS["diversity_weight"]),
        "--min-foreground-luminance",
        str(request_value(request, "min_foreground_luminance")),
        "--display-scale",
        str(DEFAULTS["display_scale"]),
        "--display-sharpen",
        str(DEFAULTS["display_sharpen"]),
        "--output-dir",
        str(output_dir),
    ]
    if DEFAULTS["use_ema"]:
        command.append("--use-ema")
    if request.save_all_candidates:
        command.append("--save-all-candidates")
    return command


def run_generation_job(job_id: str, request: GenerateRequest) -> None:
    output_dir = OUTPUT_ROOT / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(job_id, request, output_dir)
    (output_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    update_job(job_id, status="running")
    try:
        completed = subprocess.run(
            command,
            cwd=str(project_root()),
            env={**os.environ, "PROJECT_ROOT": str(project_root())},
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_tail = tail(completed.stdout)
        stderr_tail = tail(completed.stderr)
        status = "completed" if completed.returncode == 0 else "failed"
        update_job(
            job_id,
            status=status,
            returncode=int(completed.returncode),
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            selected_grid_url=f"/files/{job_id}/selected_grid.png" if (output_dir / "selected_grid.png").exists() else None,
            report_url=f"/files/{job_id}/generation_report.json" if (output_dir / "generation_report.json").exists() else None,
            error="" if completed.returncode == 0 else stderr_tail or stdout_tail,
        )
    except Exception as exc:  # noqa: BLE001
        update_job(job_id, status="failed", error=str(exc))


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "defaults": DEFAULTS, "output_root": str(OUTPUT_ROOT)}


@app.post("/generate", response_model=JobRecord)
def generate(request: GenerateRequest) -> JobRecord:
    job_id = uuid.uuid4().hex
    output_dir = OUTPUT_ROOT / job_id
    record = JobRecord(
        job_id=job_id,
        status="queued",
        created_at_utc=utc_now_iso(),
        updated_at_utc=utc_now_iso(),
        request=model_to_dict(request),
        output_dir=str(output_dir),
    )
    with JOBS_LOCK:
        JOBS[job_id] = record
    future: Future[None] = EXECUTOR.submit(run_generation_job, job_id, request)
    future.add_done_callback(lambda _: None)
    return record


@app.get("/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail=f"Unknown generation job: {job_id}")
        return JOBS[job_id]


@app.get("/jobs", response_model=List[JobRecord])
def list_jobs() -> List[JobRecord]:
    with JOBS_LOCK:
        return list(JOBS.values())
