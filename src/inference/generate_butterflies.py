#!/usr/bin/env python3
"""Generate location-conditioned butterflies with quality filtering and reranking."""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.nn import functional as F

from src.eval.eval_diffusion import load_prior_bundle
from src.models.autoencoder import load_autoencoder_checkpoint
from src.models.diffusion_schedule import DiffusionSchedule, DiffusionScheduleConfig
from src.models.latent_diffusion_unet import LatentDiffusionUNet, LatentDiffusionUNetConfig
from src.train.train_diffusion import configure_runtime, resolve_device
from src.utils.config import load_yaml_config
from src.utils.ema import ExponentialMovingAverage
from src.utils.geo import GeoBounds, normalize_latlon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate filtered butterfly samples for a coordinate.")
    parser.add_argument("--config", required=True, help="Path to diffusion YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Diffusion checkpoint path.")
    parser.add_argument("--lat", required=True, type=float, help="Latitude in decimal degrees.")
    parser.add_argument("--lon", required=True, type=float, help="Longitude in decimal degrees.")
    parser.add_argument("--device", default="auto", help="Device string, for example 'cuda' or 'cpu'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation.")
    parser.add_argument("--num-candidates", type=int, default=48, help="How many raw candidates to sample.")
    parser.add_argument("--num-outputs", type=int, default=8, help="How many final butterflies to keep.")
    parser.add_argument("--samples-per-batch", type=int, default=8, help="Per-batch sampling size.")
    parser.add_argument(
        "--guidance-scales",
        nargs="+",
        type=float,
        default=[3.5, 4.0, 4.5],
        help="Guidance scales to mix across candidate generation.",
    )
    parser.add_argument(
        "--prior-temperatures",
        nargs="+",
        type=float,
        default=[],
        help="Optional prior temperatures to mix across candidate generation.",
    )
    parser.add_argument("--use-ema", action="store_true", help="Use EMA diffusion weights when present.")
    parser.add_argument(
        "--anchor-top-species",
        type=int,
        default=4,
        help="How many top local prior species to use as anchored conditioning variants.",
    )
    parser.add_argument(
        "--coordinate-jitter-deg",
        type=float,
        default=0.0,
        help="Uniform jitter in decimal degrees for sampling nearby coordinates.",
    )
    parser.add_argument("--diversity-weight", type=float, default=0.12, help="Penalty for picking near-duplicate candidates.")
    parser.add_argument("--min-quality-score", type=float, default=0.35, help="Minimum quality for preferred candidates.")
    parser.add_argument(
        "--min-foreground-luminance",
        type=float,
        default=0.10,
        help="Preferred minimum mean butterfly luminance for final selection before diversity reranking.",
    )
    parser.add_argument("--top-k-species", type=int, default=10, help="How many local prior species to report.")
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    parser.add_argument("--disable-postprocess", action="store_true", help="Disable background cleanup and mild tone adjustment.")
    parser.add_argument("--display-scale", type=float, default=2.0, help="Upscale factor for saved display outputs.")
    parser.add_argument("--display-sharpen", type=float, default=0.06, help="Unsharp amount for saved display outputs.")
    parser.add_argument("--save-all-candidates", action="store_true", help="Save every sampled candidate image.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_species_vocab(path: str | Path) -> tuple[Dict[str, int], Dict[int, str]]:
    vocab = json.loads(Path(path).read_text(encoding="utf-8"))
    species_to_id = {str(key): int(value) for key, value in vocab.items()}
    id_to_species = {value: key for key, value in species_to_id.items()}
    return species_to_id, id_to_species


def load_species_vocab_with_fallback(
    path: str | Path,
    *,
    prior_model,
) -> tuple[Dict[str, int], Dict[int, str]]:
    vocab_path = Path(path)
    if vocab_path.exists():
        return load_species_vocab(vocab_path)
    num_species = int(prior_model.prior_head.config.num_species)
    species_to_id = {f"species_{index}": index for index in range(num_species)}
    id_to_species = {index: f"species_{index}" for index in range(num_species)}
    return species_to_id, id_to_species


def build_geo_bounds(data_cfg: Dict[str, Any]) -> GeoBounds:
    return GeoBounds(
        lat_min=float(data_cfg.get("lat_min", 18.5)),
        lat_max=float(data_cfg.get("lat_max", 72.5)),
        lon_min=float(data_cfg.get("lon_min", -180.0)),
        lon_max=float(data_cfg.get("lon_max", -66.0)),
    )


def split_counts(total: int, buckets: int) -> List[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if i < remainder else 0) for i in range(buckets)]


def sample_latlon_batch(
    *,
    lat: float,
    lon: float,
    bounds: GeoBounds,
    batch_size: int,
    jitter_deg: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    latlon = torch.tensor([[lat, lon]], dtype=torch.float32).repeat(batch_size, 1)
    if jitter_deg > 0.0:
        latlon = latlon + torch.empty_like(latlon).uniform_(-jitter_deg, jitter_deg)
    latlon[:, 0] = latlon[:, 0].clamp(bounds.lat_min, bounds.lat_max)
    latlon[:, 1] = latlon[:, 1].clamp(bounds.lon_min, bounds.lon_max)
    return latlon, normalize_latlon(latlon, bounds, clamp=True).to(device)


def build_condition_tokens_for_generation(
    *,
    prior_model,
    fuser,
    latlon_norm: torch.Tensor,
    mixture_temperature: float,
    mixture_top_k: int,
    species_mix_alpha: float,
    species_anchor_id: int | None = None,
) -> torch.Tensor:
    prior_outputs = prior_model(
        latlon_norm,
        mixture_temperature=float(mixture_temperature),
        mixture_top_k=int(mixture_top_k),
    )
    z_species_mix = prior_outputs.get("z_species_mix")
    effective_species_mix_alpha = float(species_mix_alpha)
    if species_anchor_id is not None:
        prototypes = prior_model.prior_head.species_prototypes().to(latlon_norm.device)
        anchor_indices = torch.full((latlon_norm.shape[0],), int(species_anchor_id), device=latlon_norm.device, dtype=torch.long)
        anchored_hidden = prototypes.index_select(0, anchor_indices)
        z_species_mix = prior_model.prior_head.prior_proj(anchored_hidden)
        effective_species_mix_alpha = max(effective_species_mix_alpha, 0.85)
    return fuser(
        prior_outputs["loc_tokens"],
        prior_outputs["z_prior"],
        z_species_mix=z_species_mix,
        species_mix_alpha=effective_species_mix_alpha,
    )


def build_generation_variants(
    *,
    guidance_scales: List[float],
    prior_temperatures: List[float],
    top_species: List[Dict[str, Any]],
    anchor_top_species: int,
) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    anchor_species = top_species[: max(int(anchor_top_species), 0)]
    for guidance_scale in guidance_scales:
        for prior_temperature in prior_temperatures:
            variants.append(
                {
                    "guidance_scale": float(guidance_scale),
                    "prior_temperature": float(prior_temperature),
                    "species_anchor_id": None,
                    "species_anchor": "mixture",
                }
            )
            for prediction in anchor_species:
                variants.append(
                    {
                        "guidance_scale": float(guidance_scale),
                        "prior_temperature": float(prior_temperature),
                        "species_anchor_id": int(prediction["species_id"]),
                        "species_anchor": str(prediction["species"]),
                    }
                )
    return variants


def _border_pixels(image_01: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            image_01[:, 0, :].T,
            image_01[:, -1, :].T,
            image_01[:, :, 0].T,
            image_01[:, :, -1].T,
        ),
        dim=0,
    )


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    pooled = F.max_pool2d(mask.float().unsqueeze(0).unsqueeze(0), kernel_size=kernel, stride=1, padding=radius)
    return pooled[0, 0] > 0.5


def _erode_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    pooled = 1.0 - F.max_pool2d(1.0 - mask.float().unsqueeze(0).unsqueeze(0), kernel_size=kernel, stride=1, padding=radius)
    return pooled[0, 0] > 0.5


def infer_foreground_mask(images_01: torch.Tensor) -> torch.Tensor:
    masks: List[torch.Tensor] = []
    for image_01 in images_01:
        border = _border_pixels(image_01)
        bg_rgb = border.median(dim=0).values
        border_luminance = 0.299 * border[:, 0] + 0.587 * border[:, 1] + 0.114 * border[:, 2]
        bg_luminance = float(border_luminance.median().item())
        border_saturation = border.amax(dim=1) - border.amin(dim=1)
        border_distance = torch.linalg.vector_norm(border - bg_rgb.unsqueeze(0), dim=1)

        image_hw = image_01.permute(1, 2, 0)
        luminance = 0.299 * image_01[0] + 0.587 * image_01[1] + 0.114 * image_01[2]
        saturation = image_01.amax(dim=0) - image_01.amin(dim=0)
        color_distance = torch.linalg.vector_norm(image_hw - bg_rgb.view(1, 1, 3), dim=-1)
        luminance_delta = (bg_luminance - luminance).clamp_min(0.0)

        distance_threshold = max(float(torch.quantile(border_distance, 0.98).item()) + 0.04, 0.06)
        saturation_threshold = max(float(torch.quantile(border_saturation, 0.98).item()) + 0.02, 0.05)
        luminance_threshold = max(float(torch.quantile((bg_luminance - border_luminance).clamp_min(0.0), 0.98).item()) + 0.04, 0.08)

        mask = (
            (color_distance > distance_threshold)
            | (luminance_delta > luminance_threshold)
            | ((saturation > saturation_threshold) & (luminance_delta > luminance_threshold * 0.5))
        )
        if float(mask.float().mean().item()) > 0.8:
            mask = (color_distance > max(distance_threshold * 1.5, 0.10)) | (luminance_delta > max(luminance_threshold * 1.35, 0.12))

        mask_f = mask.float().unsqueeze(0).unsqueeze(0)
        mask_f = F.max_pool2d(mask_f, kernel_size=3, stride=1, padding=1)
        mask_f = 1.0 - F.max_pool2d(1.0 - mask_f, kernel_size=5, stride=1, padding=2)
        masks.append(mask_f[0, 0] > 0.5)
    return torch.stack(masks, dim=0)


def cleanup_border_halo(image_01: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    cleaned = image_01.clone()
    dilated = _dilate_mask(mask, radius=1)
    outer_ring = dilated & ~mask
    if bool(outer_ring.any()):
        halo = cleaned[:, outer_ring]
        white = torch.ones_like(halo)
        cleaned[:, outer_ring] = torch.lerp(halo, white, 0.92).clamp(0.0, 1.0)
    return cleaned


def connected_component_stats(mask: torch.Tensor) -> tuple[int, float]:
    mask_cpu = mask.to(dtype=torch.bool, device="cpu")
    height, width = mask_cpu.shape
    visited = torch.zeros((height, width), dtype=torch.bool)
    components: List[int] = []
    for row in range(height):
        for col in range(width):
            if not bool(mask_cpu[row, col]) or bool(visited[row, col]):
                continue
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            size = 0
            while queue:
                y, x = queue.popleft()
                size += 1
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and bool(mask_cpu[ny, nx]) and not bool(visited[ny, nx]):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            components.append(size)
    if not components:
        return 0, 0.0
    total = float(sum(components))
    return len(components), float(max(components) / max(total, 1.0))


def gaussian_score(value: float, target: float, sigma: float) -> float:
    sigma = max(sigma, 1.0e-6)
    return float(math.exp(-((value - target) ** 2) / (2.0 * sigma * sigma)))


def score_candidate(image: torch.Tensor) -> Dict[str, float]:
    image_01 = ((image + 1.0) / 2.0).clamp(0.0, 1.0).unsqueeze(0)
    mask = infer_foreground_mask(image_01)[0]
    coverage = float(mask.float().mean().item())
    if coverage < 0.01:
        return {
            "quality_score": 0.0,
            "coverage": coverage,
            "symmetry_score": 0.0,
            "border_touch": 1.0,
            "fill_ratio": 0.0,
            "aspect_ratio": 0.0,
            "largest_component_ratio": 0.0,
            "component_count": 0.0,
            "foreground_luminance": 1.0,
            "foreground_std": 0.0,
            "center_distance": 1.0,
        }

    coords = torch.nonzero(mask, as_tuple=False)
    y0 = int(coords[:, 0].min().item())
    y1 = int(coords[:, 0].max().item()) + 1
    x0 = int(coords[:, 1].min().item())
    x1 = int(coords[:, 1].max().item()) + 1
    bbox_h = max(y1 - y0, 1)
    bbox_w = max(x1 - x0, 1)
    bbox_area = float(bbox_h * bbox_w)
    fill_ratio = float(mask[y0:y1, x0:x1].float().mean().item())
    aspect_ratio = float(bbox_w / max(bbox_h, 1))

    border_pixels = torch.cat([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    border_touch = float(border_pixels.float().mean().item())

    mirrored_mask = torch.flip(mask, dims=[1])
    symmetry_score = float(1.0 - (mask.float() - mirrored_mask.float()).abs().mean().item())

    image_crop = image_01[0]
    mirrored_image = torch.flip(image_crop, dims=[2])
    overlap = (mask & mirrored_mask).float()
    overlap_sum = float(overlap.sum().item())
    if overlap_sum > 0.0:
        rgb_symmetry = float(
            1.0
            - ((image_crop - mirrored_image).abs().mean(dim=0) * overlap).sum().item() / max(overlap_sum, 1.0)
        )
    else:
        rgb_symmetry = 0.0

    component_count, largest_component_ratio = connected_component_stats(mask)
    centroid_y = float(coords[:, 0].float().mean().item() / mask.shape[0])
    centroid_x = float(coords[:, 1].float().mean().item() / mask.shape[1])
    center_distance = float(math.sqrt((centroid_y - 0.5) ** 2 + (centroid_x - 0.5) ** 2))

    foreground_pixels = image_crop[:, mask]
    if foreground_pixels.numel() > 0:
        foreground_luminance = float(
            (0.299 * foreground_pixels[0] + 0.587 * foreground_pixels[1] + 0.114 * foreground_pixels[2]).mean().item()
        )
    else:
        foreground_luminance = 1.0
    foreground_std = float(foreground_pixels.std().item()) if foreground_pixels.numel() > 1 else 0.0

    coverage_score = gaussian_score(coverage, target=0.18, sigma=0.11)
    aspect_score = gaussian_score(math.log(max(aspect_ratio, 1.0e-6)), target=math.log(1.25), sigma=0.55)
    fill_score = gaussian_score(fill_ratio, target=0.48, sigma=0.18)
    center_score = gaussian_score(center_distance, target=0.0, sigma=0.18)
    component_score = max(0.0, 1.0 - max(component_count - 2, 0) * 0.18) * largest_component_ratio
    border_score = max(0.0, 1.0 - border_touch / 0.08)
    luminance_score = gaussian_score(foreground_luminance, target=0.24, sigma=0.08)
    if foreground_luminance < 0.07:
        luminance_score *= 0.2
    elif foreground_luminance < 0.10:
        luminance_score *= 0.45
    elif foreground_luminance < 0.13:
        luminance_score *= 0.72
    texture_score = gaussian_score(foreground_std, target=0.16, sigma=0.08)
    symmetry_blend = 0.6 * symmetry_score + 0.4 * max(rgb_symmetry, 0.0)

    quality_score = (
        0.15 * coverage_score
        + 0.16 * symmetry_blend
        + 0.12 * border_score
        + 0.10 * fill_score
        + 0.08 * aspect_score
        + 0.12 * component_score
        + 0.07 * center_score
        + 0.16 * luminance_score
        + 0.06 * texture_score
    )

    return {
        "quality_score": float(max(0.0, min(1.0, quality_score))),
        "coverage": coverage,
        "symmetry_score": symmetry_score,
        "rgb_symmetry": rgb_symmetry,
        "border_touch": border_touch,
        "fill_ratio": fill_ratio,
        "aspect_ratio": aspect_ratio,
        "largest_component_ratio": largest_component_ratio,
        "component_count": float(component_count),
        "foreground_luminance": foreground_luminance,
        "foreground_std": foreground_std,
        "center_distance": center_distance,
        "bbox_area_ratio": bbox_area / float(mask.numel()),
    }


def postprocess_candidate_image(image: torch.Tensor) -> torch.Tensor:
    image_01 = ((image + 1.0) / 2.0).clamp(0.0, 1.0)
    mask = infer_foreground_mask(image_01.unsqueeze(0))[0]
    processed = image_01.clone()
    processed[:, ~mask] = 1.0
    if bool(mask.any()):
        processed = cleanup_border_halo(processed, mask)
    return processed.mul(2.0).sub(1.0)


def postprocess_candidate_batch(images: torch.Tensor) -> torch.Tensor:
    return torch.stack([postprocess_candidate_image(image) for image in images], dim=0)


def upscale_and_sharpen_images(images_01: torch.Tensor, *, scale: float, sharpen: float) -> torch.Tensor:
    if scale > 1.0:
        images_01 = F.interpolate(images_01, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    if sharpen > 0.0:
        padded = F.pad(images_01, (1, 1, 1, 1), mode="reflect")
        blurred = F.avg_pool2d(padded, kernel_size=3, stride=1)
        images_01 = (images_01 + float(sharpen) * (images_01 - blurred)).clamp(0.0, 1.0)
    return images_01


def polish_display_image(image: torch.Tensor, *, scale: float, sharpen: float) -> torch.Tensor:
    image_01 = ((image + 1.0) / 2.0).clamp(0.0, 1.0).unsqueeze(0)
    image_01 = upscale_and_sharpen_images(image_01, scale=scale, sharpen=sharpen)[0]
    mask = infer_foreground_mask(image_01.unsqueeze(0))[0]
    polished = image_01.clone()
    polished[:, ~mask] = 1.0
    if bool(mask.any()):
        polished = cleanup_border_halo(polished, mask)
    return polished.mul(2.0).sub(1.0)


def polish_display_batch(images: torch.Tensor, *, scale: float, sharpen: float) -> torch.Tensor:
    return torch.stack([polish_display_image(image, scale=scale, sharpen=sharpen) for image in images], dim=0)


def build_candidate_embeddings(vae, images: torch.Tensor, device: torch.device, batch_size: int = 16) -> torch.Tensor:
    latents: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size].to(device)
            z = vae.encode_for_diffusion(batch, sample_posterior=False).flatten(start_dim=1)
            latents.append(z.cpu())
    return F.normalize(torch.cat(latents, dim=0), dim=1)


def greedy_diverse_selection(
    *,
    quality_scores: List[float],
    luminance_scores: List[float],
    embeddings: torch.Tensor,
    num_outputs: int,
    diversity_weight: float,
    min_quality_score: float,
    min_foreground_luminance: float,
) -> List[int]:
    sorted_indices = sorted(range(len(quality_scores)), key=lambda idx: quality_scores[idx], reverse=True)
    preferred = [
        idx
        for idx in sorted_indices
        if quality_scores[idx] >= min_quality_score and luminance_scores[idx] >= min_foreground_luminance
    ]
    remaining = preferred if len(preferred) >= num_outputs else sorted_indices
    selected: List[int] = []
    while remaining and len(selected) < num_outputs:
        best_idx = None
        best_objective = float("-inf")
        for idx in remaining:
            luminance_penalty = 0.0
            if luminance_scores[idx] < min_foreground_luminance:
                luminance_penalty = 0.08 * (min_foreground_luminance - luminance_scores[idx]) / max(
                    min_foreground_luminance, 1.0e-6
                )
            if not selected:
                objective = quality_scores[idx] - luminance_penalty
            else:
                similarities = torch.mv(embeddings[selected], embeddings[idx])
                objective = quality_scores[idx] - diversity_weight * float(similarities.max().item()) - luminance_penalty
            if objective > best_objective:
                best_objective = objective
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining = [idx for idx in remaining if idx != best_idx]
    return selected


def top_species_predictions(
    *,
    prior_model,
    id_to_species: Dict[int, str],
    latlon_norm: torch.Tensor,
    mixture_temperature: float,
    mixture_top_k: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    with torch.no_grad():
        outputs = prior_model(
            latlon_norm,
            mixture_temperature=mixture_temperature,
            mixture_top_k=mixture_top_k,
        )
        probs = outputs["species_probs"][0]
        top_probs, top_indices = probs.topk(min(top_k, probs.shape[0]), dim=-1)
    predictions: List[Dict[str, Any]] = []
    for rank, (species_id, probability) in enumerate(zip(top_indices.tolist(), top_probs.tolist()), start=1):
        predictions.append(
            {
                "rank": rank,
                "species_id": int(species_id),
                "species": id_to_species.get(int(species_id), str(species_id)),
                "probability": float(probability),
            }
        )
    return predictions


def save_outputs(
    *,
    output_dir: Path,
    selected_images: torch.Tensor,
    candidate_images: torch.Tensor,
    selected_records: List[Dict[str, Any]],
    all_records: List[Dict[str, Any]],
    top_species: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    from torchvision.utils import save_image

    ensure_dir(output_dir)
    selected_dir = output_dir / "selected"
    ensure_dir(selected_dir)
    grid_path = output_dir / "selected_grid.png"
    selected_display = polish_display_batch(selected_images, scale=float(args.display_scale), sharpen=float(args.display_sharpen))
    save_image((selected_display + 1.0) / 2.0, grid_path, nrow=min(selected_display.shape[0], 4))
    for rank, image in enumerate(selected_display, start=1):
        save_image((image + 1.0) / 2.0, selected_dir / f"selected_{rank:02d}.png")

    if args.save_all_candidates:
        candidates_dir = output_dir / "all_candidates"
        ensure_dir(candidates_dir)
        all_grid_path = output_dir / "all_candidates_grid.png"
        candidate_display = polish_display_batch(candidate_images, scale=float(args.display_scale), sharpen=float(args.display_sharpen))
        save_image((candidate_display + 1.0) / 2.0, all_grid_path, nrow=min(candidate_display.shape[0], 6))
        for idx, image in enumerate(candidate_display, start=1):
            save_image((image + 1.0) / 2.0, candidates_dir / f"candidate_{idx:03d}.png")

    report = {
        "latitude": float(args.lat),
        "longitude": float(args.lon),
        "num_candidates": int(args.num_candidates),
        "num_outputs": int(args.num_outputs),
        "guidance_scales": [float(value) for value in args.guidance_scales],
        "prior_temperatures": [float(value) for value in args.prior_temperatures] if args.prior_temperatures else [],
        "anchor_top_species": int(args.anchor_top_species),
        "coordinate_jitter_deg": float(args.coordinate_jitter_deg),
        "min_foreground_luminance": float(args.min_foreground_luminance),
        "postprocessed": not args.disable_postprocess,
        "display_scale": float(args.display_scale),
        "display_sharpen": float(args.display_sharpen),
        "selected_grid_path": str(grid_path.resolve()),
        "top_species": top_species,
        "selected": selected_records,
        "all_candidates": all_records,
    }
    (output_dir / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    cfg = load_yaml_config(args.config)
    device = resolve_device(args.device)
    configure_runtime(dict(cfg.get("training", {})), device)

    output_dir = Path(args.output_dir) if args.output_dir else (Path(cfg["output_dir"]) / "generation")
    ensure_dir(output_dir)

    vae_locked = load_yaml_config(cfg["vae_locked_config"])
    vae = load_autoencoder_checkpoint(
        vae_locked["checkpoint_path"],
        device=device,
        use_ema=bool(vae_locked.get("use_ema", False)),
        freeze=True,
    )
    prior_model, fuser = load_prior_bundle(cfg, device)

    prior_locked = load_yaml_config(cfg["prior_locked_config"])
    prior_artifact_dir = prior_locked.get("artifact_dir") or prior_locked.get("run_metadata", {}).get("artifact_dir")
    if not prior_artifact_dir:
        raise RuntimeError("prior_locked_config is missing artifact_dir metadata.")
    prior_cfg = load_yaml_config(Path(prior_artifact_dir) / "configs" / "prior_runpod.yaml")
    prior_data_cfg = load_yaml_config(prior_cfg["data_config"])
    _, id_to_species = load_species_vocab_with_fallback(
        prior_data_cfg["species_vocab_json"],
        prior_model=prior_model,
    )
    bounds = build_geo_bounds(prior_data_cfg)

    latlon = torch.tensor([[float(args.lat), float(args.lon)]], dtype=torch.float32)
    latlon_norm = normalize_latlon(latlon, bounds, clamp=True).to(device)

    unet = LatentDiffusionUNet(LatentDiffusionUNetConfig(**cfg["model"])).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    unet.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if args.use_ema and checkpoint.get("ema_state_dict"):
        ema = ExponentialMovingAverage.from_model(unet, decay=0.0)
        ema.load_state_dict(checkpoint["ema_state_dict"])
        ema.copy_to_model(unet)
    if checkpoint.get("fuser_state_dict"):
        fuser.load_state_dict(checkpoint["fuser_state_dict"], strict=False)
    unet.eval()
    schedule = DiffusionSchedule(DiffusionScheduleConfig(**cfg["diffusion"]), device=device)

    conditioning_cfg = dict(cfg.get("conditioning", {}))
    guidance_scales = [float(value) for value in args.guidance_scales]
    prior_temperatures = [float(value) for value in args.prior_temperatures] or [float(conditioning_cfg.get("prior_temperature", 1.0))]
    top_species = top_species_predictions(
        prior_model=prior_model,
        id_to_species=id_to_species,
        latlon_norm=latlon_norm,
        mixture_temperature=float(conditioning_cfg.get("prior_temperature", 1.0)),
        mixture_top_k=int(conditioning_cfg.get("prior_top_k", 0)),
        top_k=int(args.top_k_species),
    )
    variants = build_generation_variants(
        guidance_scales=guidance_scales,
        prior_temperatures=prior_temperatures,
        top_species=top_species,
        anchor_top_species=int(args.anchor_top_species),
    )

    images_batches: List[torch.Tensor] = []
    records: List[Dict[str, Any]] = []
    per_variant_counts = split_counts(int(args.num_candidates), len(variants))

    candidate_index = 0
    for variant, total_count in zip(variants, per_variant_counts):
        if total_count <= 0:
            continue
        remaining = total_count
        while remaining > 0:
            batch_size = min(int(args.samples_per_batch), remaining)
            remaining -= batch_size
            raw_latlon, latlon_norm_batch = sample_latlon_batch(
                lat=float(args.lat),
                lon=float(args.lon),
                bounds=bounds,
                batch_size=batch_size,
                jitter_deg=float(args.coordinate_jitter_deg),
                device=device,
            )
            cond_tokens = build_condition_tokens_for_generation(
                prior_model=prior_model,
                fuser=fuser,
                latlon_norm=latlon_norm_batch,
                mixture_temperature=float(variant["prior_temperature"]),
                mixture_top_k=int(conditioning_cfg.get("prior_top_k", 0)),
                species_mix_alpha=float(conditioning_cfg.get("species_mix_alpha", 0.5)),
                species_anchor_id=variant["species_anchor_id"],
            )
            null_tokens = fuser.make_null_condition(batch_size, device)
            with torch.no_grad():
                latents = schedule.sample_loop(
                    unet,
                    shape=torch.Size((batch_size, vae.config.latent_channels, vae.latent_resolution, vae.latent_resolution)),
                    context_tokens=cond_tokens,
                    guidance_scale=float(variant["guidance_scale"]),
                    null_context=null_tokens,
                    prediction_type=str(schedule.config.prediction_type),
                    device=device,
                )
                batch_images = vae.decode_from_diffusion_latents(latents).cpu().clamp(-1.0, 1.0)
            if not args.disable_postprocess:
                batch_images = postprocess_candidate_batch(batch_images)
            for sample_index, image in enumerate(batch_images):
                candidate_index += 1
                score = score_candidate(image)
                score["candidate_index"] = candidate_index
                score["guidance_scale"] = float(variant["guidance_scale"])
                score["prior_temperature"] = float(variant["prior_temperature"])
                score["species_anchor_id"] = variant["species_anchor_id"]
                score["species_anchor"] = variant["species_anchor"]
                score["lat"] = float(raw_latlon[sample_index, 0].item())
                score["lon"] = float(raw_latlon[sample_index, 1].item())
                records.append(score)
            images_batches.append(batch_images)

    candidate_images = torch.cat(images_batches, dim=0)
    embeddings = build_candidate_embeddings(vae, candidate_images, device)
    selected_indices = greedy_diverse_selection(
        quality_scores=[record["quality_score"] for record in records],
        luminance_scores=[record["foreground_luminance"] for record in records],
        embeddings=embeddings,
        num_outputs=int(args.num_outputs),
        diversity_weight=float(args.diversity_weight),
        min_quality_score=float(args.min_quality_score),
        min_foreground_luminance=float(args.min_foreground_luminance),
    )

    selected_images = candidate_images[selected_indices]
    selected_records = []
    for rank, idx in enumerate(selected_indices, start=1):
        record = dict(records[idx])
        record["rank"] = rank
        selected_records.append(record)

    save_outputs(
        output_dir=output_dir,
        selected_images=selected_images,
        candidate_images=candidate_images,
        selected_records=selected_records,
        all_records=records,
        top_species=top_species,
        args=args,
    )

    summary = {
        "output_dir": str(output_dir.resolve()),
        "selected_grid": str((output_dir / "selected_grid.png").resolve()),
        "report_json": str((output_dir / "generation_report.json").resolve()),
        "postprocessed": not args.disable_postprocess,
        "top_species": top_species,
        "selected": selected_records,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
