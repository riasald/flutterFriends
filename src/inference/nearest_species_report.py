#!/usr/bin/env python3
"""Report the top-k prior species predictions for a given coordinate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from src.train.train_prior import build_geo_bounds, build_model, load_species_vocab, resolve_device
from src.utils.config import load_yaml_config
from src.utils.geo import normalize_latlon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect prior species predictions for a latitude/longitude pair.")
    parser.add_argument("--config", required=True, help="Path to the prior YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path produced by train_prior.py.")
    parser.add_argument("--lat", required=True, type=float, help="Latitude in decimal degrees.")
    parser.add_argument("--lon", required=True, type=float, help="Longitude in decimal degrees.")
    parser.add_argument("--top-k", default=10, type=int, help="How many species to report.")
    parser.add_argument("--device", default="auto", help="Device string, for example 'cuda' or 'cpu'.")
    parser.add_argument("--output-json", default="", help="Optional output path for the report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    data_cfg = load_yaml_config(cfg["data_config"])
    device = resolve_device(args.device)

    species_to_id, id_to_species = load_species_vocab(data_cfg["species_vocab_json"])
    model = build_model(cfg, len(species_to_id), device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    bounds = build_geo_bounds(data_cfg)
    latlon = torch.tensor([[float(args.lat), float(args.lon)]], dtype=torch.float32)
    latlon_norm = normalize_latlon(latlon, bounds, clamp=True).to(device)

    with torch.no_grad():
        outputs = model(latlon_norm)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        top_k = min(int(args.top_k), probs.shape[-1])
        top_probs, top_indices = probs.topk(top_k, dim=-1)

    predictions = []
    for rank, (species_id, probability) in enumerate(
        zip(top_indices[0].detach().cpu().tolist(), top_probs[0].detach().cpu().tolist()),
        start=1,
    ):
        predictions.append(
            {
                "rank": rank,
                "species_id": int(species_id),
                "species": id_to_species.get(int(species_id), str(species_id)),
                "probability": float(probability),
            }
        )

    report: Dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "latitude": float(args.lat),
        "longitude": float(args.lon),
        "normalized_latlon": [float(value) for value in latlon_norm[0].detach().cpu().tolist()],
        "top_k": top_k,
        "predictions": predictions,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
