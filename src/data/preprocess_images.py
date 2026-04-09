#!/usr/bin/env python3
"""
Preprocess downloaded butterfly images into cleaner, model-ready crops.

Current v1 behavior:
- reads a CSV produced by the fetch step
- loads each downloaded image from `local_image_path`
- detects the main centered butterfly-like foreground component
- crops around that region with margin and resizes to a square output
- optionally whitens the background outside the detected foreground mask
- writes processed images, optional masks, an updated CSV, and a summary JSON

This is intentionally heuristic-driven so we can clean a large dataset before
introducing heavier segmentation models.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency check
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:  # pragma: no cover - runtime dependency check
    np = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover - runtime dependency check
    Image = None  # type: ignore[assignment]

try:
    from rembg import new_session as rembg_new_session
    from rembg import remove as rembg_remove
except ImportError:  # pragma: no cover - optional runtime dependency
    rembg_new_session = None  # type: ignore[assignment]
    rembg_remove = None  # type: ignore[assignment]


_REMBG_SESSION_CACHE: Dict[str, Any] = {}


def ensure_runtime_dependencies() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if np is None:
        missing.append("numpy")
    if Image is None:
        missing.append("Pillow")
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(f"Missing runtime dependencies: {missing_str}")


def ensure_segmentation_backend_available(segmentation_backend: str) -> None:
    if segmentation_backend == "rembg" and (rembg_remove is None or rembg_new_session is None):
        raise RuntimeError(
            "The 'rembg' segmentation backend requires the optional 'rembg' package. "
            "Install it with CPU or GPU support before using --segmentation-backend rembg."
        )


def configure_cv_runtime(cv_threads: int) -> None:
    if cv2 is None:
        return
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass


def configure_rembg_runtime() -> None:
    if rembg_new_session is None or rembg_remove is None:
        return
    if os.environ.get("U2NET_HOME"):
        ensure_dir(Path(os.environ["U2NET_HOME"]))
        return
    cache_dir = (Path.cwd() / ".cache" / "u2net").resolve()
    ensure_dir(cache_dir)
    os.environ["U2NET_HOME"] = str(cache_dir)


def preload_onnxruntime_gpu_dlls() -> None:
    torch_lib_dir: Optional[str] = None
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return

    try:
        import torch  # type: ignore

        _ = torch.__version__
        torch_lib_dir = str((Path(torch.__file__).resolve().parent / "lib").resolve())
    except Exception:
        pass

    if torch_lib_dir:
        current_path = os.environ.get("PATH", "")
        path_entries = current_path.split(os.pathsep) if current_path else []
        if torch_lib_dir not in path_entries:
            os.environ["PATH"] = torch_lib_dir + os.pathsep + current_path if current_path else torch_lib_dir

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass


def resolve_rembg_providers(provider_name: str) -> Optional[List[str]]:
    name = normalize_text(provider_name).lower()
    if not name or name == "auto":
        return None
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return None

    available = set(ort.get_available_providers())
    provider_map = {
        "cpu": ["CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    }
    requested = provider_map.get(name)
    if requested is None:
        return None
    if requested[0] not in available:
        return ["CPUExecutionProvider"]
    return requested
    try:
        if cv_threads >= 0:
            cv2.setNumThreads(int(cv_threads))
    except Exception:
        pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


VIEW_HINT_PATTERNS = {
    "dorsal": [
        re.compile(pattern)
        for pattern in (
            r"(?<![a-z])dorsal(?![a-z])",
            r"(?<![a-z])upperside(?![a-z])",
            r"(?<![a-z])upper_side(?![a-z])",
            r"(?<![a-z])upper-side(?![a-z])",
            r"(?<![a-z])top(?![a-z])",
            r"[_-]d(?:[_-]|$)",
        )
    ],
    "ventral": [
        re.compile(pattern)
        for pattern in (
            r"(?<![a-z])ventral(?![a-z])",
            r"(?<![a-z])underside(?![a-z])",
            r"(?<![a-z])under_side(?![a-z])",
            r"(?<![a-z])under-side(?![a-z])",
            r"(?<![a-z])bottom(?![a-z])",
            r"[_-]v(?:[_-]|$)",
        )
    ],
    "lateral": [
        re.compile(pattern)
        for pattern in (
            r"(?<![a-z])lateral(?![a-z])",
            r"(?<![a-z])side(?![a-z])",
            r"(?<![a-z])profile(?![a-z])",
        )
    ],
}


def read_csv_rows(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        rows = []
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            rows.append(dict(row))
    return rows


def write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_image_rgb(
    path: Path,
    *,
    decode_max_side: int = 0,
) -> np.ndarray:
    imread_flag = cv2.IMREAD_COLOR
    suffix = path.suffix.lower()
    if decode_max_side > 0 and suffix in {".jpg", ".jpeg"}:
        try:
            with Image.open(path) as image:
                source_width, source_height = image.size
            source_max_side = max(source_width, source_height)
            if source_max_side >= decode_max_side * 8:
                imread_flag = cv2.IMREAD_REDUCED_COLOR_8
            elif source_max_side >= decode_max_side * 4:
                imread_flag = cv2.IMREAD_REDUCED_COLOR_4
            elif source_max_side >= decode_max_side * 2:
                imread_flag = cv2.IMREAD_REDUCED_COLOR_2
        except Exception:
            imread_flag = cv2.IMREAD_COLOR

    image_bgr = cv2.imread(str(path), imread_flag)
    if image_bgr is None:
        with Image.open(path) as image:
            return np.array(image.convert("RGB"))
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(path: Path, image: np.ndarray, *, jpeg_quality: int = 95) -> None:
    ensure_dir(path.parent)
    path_str = str(path)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        success = cv2.imwrite(path_str, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    elif suffix == ".png":
        success = cv2.imwrite(path_str, image_bgr)
    else:
        success = cv2.imwrite(path_str, image_bgr)
    if not success:
        Image.fromarray(image).save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    ensure_dir(path.parent)
    success = cv2.imwrite(str(path), mask)
    if not success:
        Image.fromarray(mask).save(path)


def save_preview(path: Path, original: np.ndarray, processed: np.ndarray) -> None:
    ensure_dir(path.parent)
    preview = Image.new("RGB", (original.shape[1] + processed.shape[1], max(original.shape[0], processed.shape[0])), "white")
    preview.paste(Image.fromarray(original), (0, 0))
    preview.paste(Image.fromarray(processed), (original.shape[1], 0))
    preview.save(path)


def infer_view_hint(row: Dict[str, Any]) -> Tuple[str, str]:
    text_fields = [
        row.get("image_url"),
        row.get("media_id"),
        row.get("record_url"),
        row.get("image_path"),
        row.get("source_image_path"),
        row.get("catalog_number"),
    ]
    haystack = " ".join(unquote(normalize_text(value)).lower() for value in text_fields if normalize_text(value))
    if not haystack:
        return "unknown", "none"

    matched_labels = [
        label
        for label, patterns in VIEW_HINT_PATTERNS.items()
        if any(pattern.search(haystack) for pattern in patterns)
    ]
    if len(matched_labels) == 1:
        return matched_labels[0], "metadata"
    return "unknown", "metadata_ambiguous" if matched_labels else "none"


def sample_border_pixels(image: np.ndarray, border_fraction: float = 0.06) -> np.ndarray:
    height, width = image.shape[:2]
    border = max(4, int(min(height, width) * border_fraction))
    top = image[:border, :, :]
    bottom = image[-border:, :, :]
    left = image[:, :border, :]
    right = image[:, -border:, :]
    return np.concatenate(
        [
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ],
        axis=0,
    )


def sample_top_corner_pixels(image: np.ndarray, corner_fraction: float = 0.08) -> np.ndarray:
    height, width = image.shape[:2]
    corner_h = max(8, int(height * corner_fraction))
    corner_w = max(8, int(width * corner_fraction))
    top_left = image[:corner_h, :corner_w, :]
    top_right = image[:corner_h, -corner_w:, :]
    return np.concatenate(
        [
            top_left.reshape(-1, 3),
            top_right.reshape(-1, 3),
        ],
        axis=0,
    )


def estimate_background_color(image: np.ndarray) -> np.ndarray:
    border_pixels = sample_border_pixels(image).astype(np.uint8)
    hsv = cv2.cvtColor(border_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    low_saturation = hsv[:, 1] <= np.percentile(hsv[:, 1], 55)
    bright_pixels = hsv[:, 2] >= np.percentile(hsv[:, 2], 60)
    candidate_pixels = border_pixels[low_saturation & bright_pixels]
    if candidate_pixels.shape[0] < 32:
        candidate_pixels = sample_top_corner_pixels(image)
    return np.median(candidate_pixels, axis=0).astype(np.uint8)


def build_foreground_mask(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    background_color = estimate_background_color(image).astype(np.float32)
    image_float = image.astype(np.float32)
    color_distance = np.linalg.norm(image_float - background_color, axis=2)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    saturation_threshold = max(42.0, float(np.percentile(saturation, 78)))
    contrast_threshold = max(28.0, float(np.percentile(color_distance, 70)))
    value_floor = max(18.0, float(np.percentile(value, 8)))
    value_ceiling = min(245.0, float(np.percentile(value, 95)))

    mask = (
        (saturation > saturation_threshold)
        | (
            (color_distance > contrast_threshold)
            & (value > value_floor)
            & (value < value_ceiling)
        )
    ).astype(np.uint8) * 255

    close_kernel = max(5, int(min(height, width) * 0.015))
    if close_kernel % 2 == 0:
        close_kernel += 1
    open_kernel = max(3, int(min(height, width) * 0.01))
    if open_kernel % 2 == 0:
        open_kernel += 1

    close_struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    open_struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_struct, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_struct, iterations=1)

    return mask


def resize_working_image(
    image: np.ndarray,
    max_side: int,
) -> Tuple[np.ndarray, float]:
    if max_side <= 0:
        return image, 1.0
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return image, 1.0
    resized = cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, float(scale)


def get_rembg_session(model_name: str, *, provider_name: str = "auto") -> Any:
    cache_key = f"{model_name}:{normalize_text(provider_name).lower() or 'auto'}"
    cached = _REMBG_SESSION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if rembg_new_session is None:
        raise RuntimeError("rembg is not installed.")
    providers = resolve_rembg_providers(provider_name)
    try:
        if providers is None:
            session = rembg_new_session(model_name)
        else:
            session = rembg_new_session(model_name, providers=providers)
    except Exception as exc:
        message = str(exc).lower()
        if providers is None and ("cuda" in message or "provider" in message or "executionprovider" in message):
            session = rembg_new_session(model_name, providers=["CPUExecutionProvider"])
        else:
            raise
    _REMBG_SESSION_CACHE[cache_key] = session
    return session


def normalize_mask_array(mask_output: Any) -> np.ndarray:
    if isinstance(mask_output, bytes):
        with Image.open(io.BytesIO(mask_output)) as mask_image:
            array = np.array(mask_image.convert("L"))
    elif Image is not None and isinstance(mask_output, Image.Image):
        array = np.array(mask_output.convert("L"))
    else:
        array = np.asarray(mask_output)
        if array.ndim == 3 and array.shape[2] == 4:
            array = array[:, :, 3]
        elif array.ndim == 3:
            array = array.max(axis=2)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.where(array > 0, 255, 0).astype(np.uint8)


def build_rembg_mask(
    image: np.ndarray,
    *,
    model_name: str,
    provider_name: str = "auto",
) -> np.ndarray:
    session = get_rembg_session(model_name, provider_name=provider_name)
    mask_output = rembg_remove(image, session=session, only_mask=True)
    mask = normalize_mask_array(mask_output)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def component_statistics(
    mask: np.ndarray,
    image: np.ndarray,
) -> List[Dict[str, Any]]:
    height, width = mask.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    stats_out: List[Dict[str, Any]] = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label_index in range(1, num_labels):
        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        w = int(stats[label_index, cv2.CC_STAT_WIDTH])
        h = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        label_mask = labels == label_index
        fill_ratio = area / max(w * h, 1)
        centroid_x, centroid_y = centroids[label_index]
        shape_metrics = contour_shape_metrics(label_mask.astype(np.uint8) * 255)
        stats_out.append(
            {
                "label": label_index,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
                "fill_ratio": fill_ratio,
                "aspect_ratio": max(w / max(h, 1), h / max(w, 1)),
                "centroid_x": float(centroid_x),
                "centroid_y": float(centroid_y),
                "touches_border": x <= 1 or y <= 1 or (x + w) >= width - 1 or (y + h) >= height - 1,
                "mean_saturation": float(hsv[:, :, 1][label_mask].mean()),
                "mean_value": float(hsv[:, :, 2][label_mask].mean()),
                "mean_gray": float(gray[label_mask].mean()),
                "polygon_vertices": int(shape_metrics["polygon_vertices"]),
                "solidity": float(shape_metrics["solidity"]),
                "label_mask": label_mask,
            }
        )
    return stats_out


def specimen_candidate_components(
    mask: np.ndarray,
    image: np.ndarray,
) -> List[Dict[str, Any]]:
    height, width = mask.shape[:2]
    image_area = height * width
    candidates: List[Dict[str, Any]] = []
    for component in component_statistics(mask, image):
        area = component["area"]
        horizontal_ratio = component["w"] / max(component["h"], 1)
        rectangular_like = (
            int(component.get("polygon_vertices", 0)) <= 6
            and float(component.get("solidity", 0.0)) >= 0.92
            and float(component["fill_ratio"]) >= 0.78
        )
        if area < image_area * 0.015:
            continue
        if component["mean_saturation"] < 35.0:
            continue
        if horizontal_ratio < 0.85 or horizontal_ratio > 5.5:
            continue
        if component["fill_ratio"] < 0.08:
            continue
        if rectangular_like:
            continue
        candidates.append(component)
    return candidates


def mask_symmetry_score(mask: np.ndarray) -> float:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return 0.0
    x0, y0, x1, y1 = bbox
    cropped = (mask[y0:y1, x0:x1] > 0)
    if cropped.size == 0 or cropped.shape[1] < 2:
        return 0.0

    width = cropped.shape[1]
    half = width // 2
    if half == 0:
        return 0.0
    left = cropped[:, :half]
    right = cropped[:, width - half:]
    right = np.fliplr(right)
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(left, right).sum()
    return float(intersection / union)


def mask_left_right_balance(mask: np.ndarray) -> float:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return 0.0
    x0, y0, x1, y1 = bbox
    cropped = (mask[y0:y1, x0:x1] > 0)
    if cropped.size == 0 or cropped.shape[1] < 2:
        return 0.0
    width = cropped.shape[1]
    half = width // 2
    if half == 0:
        return 0.0
    left_area = float(cropped[:, :half].sum())
    right_area = float(cropped[:, width - half:].sum())
    larger = max(left_area, right_area, 1.0)
    smaller = min(left_area, right_area)
    return float(smaller / larger)


def contour_shape_metrics(mask: np.ndarray) -> Dict[str, Any]:
    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "polygon_vertices": 0,
            "solidity": 0.0,
        }

    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    if contour_area <= 0.0:
        return {
            "polygon_vertices": 0,
            "solidity": 0.0,
        }

    perimeter = float(cv2.arcLength(contour, True))
    epsilon = max(2.0, perimeter * 0.02)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = contour_area / max(hull_area, 1.0)
    return {
        "polygon_vertices": int(len(approx)),
        "solidity": float(solidity),
    }


def score_selected_mask(
    main_mask: np.ndarray,
    image: np.ndarray,
) -> Dict[str, Any]:
    bbox = bbox_from_mask(main_mask)
    if bbox is None:
        return {
            "score": 0.0,
            "symmetry_score": 0.0,
            "wing_balance_ratio": 0.0,
            "horizontal_ratio": 0.0,
            "fill_ratio": 0.0,
            "coverage_ratio": 0.0,
            "mean_saturation": 0.0,
            "touches_border": False,
            "centroid_x_ratio": 0.5,
            "centroid_y_ratio": 0.5,
            "polygon_vertices": 0,
            "solidity": 0.0,
        }

    x0, y0, x1, y1 = bbox
    height, width = image.shape[:2]
    mask_pixels = main_mask > 0
    area = float(mask_pixels.sum())
    bbox_area = max((x1 - x0) * (y1 - y0), 1)
    fill_ratio = area / bbox_area
    coverage_ratio = area / max(height * width, 1)
    horizontal_ratio = (x1 - x0) / max((y1 - y0), 1)
    symmetry_score = mask_symmetry_score(main_mask)
    wing_balance_ratio = mask_left_right_balance(main_mask)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mean_saturation = float(hsv[:, :, 1][mask_pixels].mean()) if np.any(mask_pixels) else 0.0
    low_saturation_fraction = (
        float((hsv[:, :, 1][mask_pixels] < 28).mean())
        if np.any(mask_pixels)
        else 1.0
    )
    ys, xs = np.where(mask_pixels)
    centroid_x_ratio = float(xs.mean() / max(width, 1)) if xs.size else 0.5
    centroid_y_ratio = float(ys.mean() / max(height, 1)) if ys.size else 0.5
    touches_border = bbox_touches_border(bbox, width, height, margin=max(2, int(min(width, height) * 0.01)))
    shape_metrics = contour_shape_metrics(main_mask)
    polygon_vertices = int(shape_metrics["polygon_vertices"])
    solidity = float(shape_metrics["solidity"])

    area_score = 1.0 if 0.02 <= coverage_ratio <= 0.40 else 0.68 if 0.01 <= coverage_ratio <= 0.55 else 0.18
    aspect_score = 1.0 if 1.05 <= horizontal_ratio <= 4.2 else 0.68 if 0.85 <= horizontal_ratio <= 5.2 else 0.18
    fill_score = 1.0 if 0.10 <= fill_ratio <= 0.86 else 0.65
    saturation_score = 0.40 + min(mean_saturation, 150.0) / 150.0
    low_saturation_penalty = max(0.18, 1.0 - low_saturation_fraction * 0.9)
    symmetry_factor = 0.45 + symmetry_score
    balance_factor = 0.35 + min(wing_balance_ratio, 1.0)
    center_x_score = max(0.25, 1.15 - abs(centroid_x_ratio - 0.50) * 1.6)
    vertical_score = max(0.45, 1.08 - abs(centroid_y_ratio - 0.43) * 1.5)
    border_penalty = 0.72 if touches_border else 1.0
    coverage_weight = 0.45 + min(math.sqrt(max(coverage_ratio, 0.0)), 0.75)
    rectangle_penalty = 0.25 if polygon_vertices <= 6 and solidity >= 0.92 and fill_ratio >= 0.78 else 1.0

    score = (
        coverage_weight
        * area_score
        * aspect_score
        * fill_score
        * saturation_score
        * low_saturation_penalty
        * symmetry_factor
        * balance_factor
        * center_x_score
        * vertical_score
        * border_penalty
        * rectangle_penalty
    )
    return {
        "score": float(score),
        "symmetry_score": float(symmetry_score),
        "wing_balance_ratio": float(wing_balance_ratio),
        "horizontal_ratio": float(horizontal_ratio),
        "fill_ratio": float(fill_ratio),
        "coverage_ratio": float(coverage_ratio),
        "mean_saturation": float(mean_saturation),
        "low_saturation_fraction": float(low_saturation_fraction),
        "touches_border": bool(touches_border),
        "centroid_x_ratio": float(centroid_x_ratio),
        "centroid_y_ratio": float(centroid_y_ratio),
        "polygon_vertices": polygon_vertices,
        "solidity": solidity,
    }


def is_strong_specimen_mask(mask_metrics: Dict[str, Any]) -> bool:
    return (
        mask_metrics["coverage_ratio"] >= 0.02
        and mask_metrics["coverage_ratio"] <= 0.42
        and mask_metrics["horizontal_ratio"] >= 1.0
        and mask_metrics["horizontal_ratio"] <= 4.8
        and mask_metrics["symmetry_score"] >= 0.42
        and mask_metrics["wing_balance_ratio"] >= 0.28
        and mask_metrics["mean_saturation"] >= 45.0
        and mask_metrics["low_saturation_fraction"] <= 0.32
        and not bool(mask_metrics["touches_border"])
    )


def assess_training_readiness(
    *,
    main_mask: np.ndarray,
    image: np.ndarray,
    view_hint: str,
    require_dorsal: bool,
    reject_unknown_view: bool,
    candidate_count: int,
    mask_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bbox = bbox_from_mask(main_mask)
    if bbox is None:
        return {
            "keep_for_training": False,
            "view_status": "missing_mask",
            "rejection_reason": "missing_mask",
            "symmetry_score": 0.0,
            "horizontal_ratio": 0.0,
            "fill_ratio": 0.0,
            "coverage_ratio": 0.0,
            "mean_saturation": 0.0,
            "low_saturation_fraction": 1.0,
            "touches_border": False,
            "candidate_count": candidate_count,
        }

    metrics = mask_metrics or score_selected_mask(main_mask, image)
    symmetry_score = float(metrics["symmetry_score"])
    wing_balance_ratio = float(metrics["wing_balance_ratio"])
    horizontal_ratio = float(metrics["horizontal_ratio"])
    fill_ratio = float(metrics["fill_ratio"])
    coverage_ratio = float(metrics["coverage_ratio"])
    mean_saturation = float(metrics["mean_saturation"])
    low_saturation_fraction = float(metrics["low_saturation_fraction"])
    touches_border = bool(metrics["touches_border"])
    polygon_vertices = int(metrics.get("polygon_vertices", 0))
    solidity = float(metrics.get("solidity", 0.0))

    rejection_reason = ""
    if require_dorsal and view_hint in {"ventral", "lateral"}:
        rejection_reason = f"metadata_view_{view_hint}"
    elif require_dorsal and reject_unknown_view and view_hint == "unknown":
        rejection_reason = "unknown_view"
    elif view_hint == "unknown" and coverage_ratio < 0.08 and horizontal_ratio < 1.25:
        rejection_reason = "compact_unknown_view"
    elif horizontal_ratio < 1.0:
        rejection_reason = "not_spread_horizontal"
    elif horizontal_ratio > 5.2:
        rejection_reason = "too_wide_for_specimen"
    elif symmetry_score < (0.28 if view_hint == "dorsal" else 0.38):
        rejection_reason = "low_bilateral_symmetry"
    elif wing_balance_ratio < (0.20 if view_hint == "dorsal" else 0.26):
        rejection_reason = "one_sided_specimen"
    elif coverage_ratio < 0.02:
        rejection_reason = "specimen_too_small"
    elif view_hint == "unknown" and polygon_vertices <= 6 and solidity >= 0.92 and fill_ratio >= 0.78:
        rejection_reason = "rectangular_non_specimen"
    elif mean_saturation < 24.0:
        rejection_reason = "low_color_saturation"
    elif fill_ratio < 0.08:
        rejection_reason = "fragmented_mask"
    elif touches_border and low_saturation_fraction > 0.20:
        rejection_reason = "border_connected_artifact"
    elif touches_border and coverage_ratio > 0.48:
        rejection_reason = "touches_image_border"

    keep_for_training = rejection_reason == ""
    view_status = "accepted"
    if view_hint in {"dorsal", "ventral", "lateral"}:
        view_status = f"metadata_{view_hint}"
    if rejection_reason:
        view_status = f"rejected_{rejection_reason}"

    return {
        "keep_for_training": keep_for_training,
        "view_status": view_status,
        "rejection_reason": rejection_reason,
        "symmetry_score": symmetry_score,
        "wing_balance_ratio": float(wing_balance_ratio),
        "horizontal_ratio": float(horizontal_ratio),
        "fill_ratio": float(fill_ratio),
        "coverage_ratio": float(coverage_ratio),
        "mean_saturation": float(mean_saturation),
        "low_saturation_fraction": float(low_saturation_fraction),
        "touches_border": bool(touches_border),
        "candidate_count": int(candidate_count),
        "polygon_vertices": polygon_vertices,
        "solidity": solidity,
    }


def remove_border_artifacts(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    image_area = height * width
    cleaned = np.zeros_like(mask)
    min_area = max(300, int(height * width * 0.002))

    for component in component_statistics(mask, image):
        area = component["area"]
        if area < min_area:
            continue

        x = component["x"]
        y = component["y"]
        w = component["w"]
        h = component["h"]
        fill_ratio = component["fill_ratio"]
        aspect_ratio = component["aspect_ratio"]
        mean_saturation = component["mean_saturation"]
        mean_gray = component["mean_gray"]

        touches_border = component["touches_border"]
        wide_and_short = w > width * 0.22 and h < height * 0.12
        low_strip = y > height * 0.75 and h < height * 0.22
        ruler_like = wide_and_short and fill_ratio > 0.18 and mean_saturation < 55.0
        tick_like = h > height * 0.03 and w < width * 0.015 and mean_saturation < 55.0
        dark_label_block = low_strip and fill_ratio > 0.25 and mean_gray < 120.0
        oversized_sheet = area > image_area * 0.7 and fill_ratio > 0.65
        overly_rectangular = aspect_ratio > 5.5 and fill_ratio > 0.3

        if oversized_sheet or dark_label_block or ruler_like or tick_like or overly_rectangular:
            continue
        if touches_border and (wide_and_short or low_strip):
            continue

        cleaned[component["label_mask"]] = 255

    return cleaned


def reconnect_specimen_regions(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    kernel = max(5, int(min(height, width) * 0.02))
    if kernel % 2 == 0:
        kernel += 1
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struct, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, struct, iterations=1)
    return mask


def select_main_component(mask: np.ndarray, image: np.ndarray) -> Tuple[np.ndarray, str]:
    height, width = mask.shape[:2]
    image_area = height * width
    min_area = max(400, int(image_area * 0.003))

    components = component_statistics(mask, image)
    if not components:
        return np.zeros_like(mask), "fallback_center_crop"

    center_x = width / 2.0
    half_width = max(width / 2.0, 1.0)

    best_mask = None
    best_score = -1.0

    for component in components:
        area = component["area"]
        if area < min_area:
            continue

        x = component["x"]
        y = component["y"]
        w = component["w"]
        h = component["h"]
        fill_ratio = component["fill_ratio"]
        aspect_ratio = component["aspect_ratio"]
        centroid_x = component["centroid_x"]
        centroid_y = component["centroid_y"]
        mean_saturation = component["mean_saturation"]
        touches_border = component["touches_border"]
        polygon_vertices = int(component.get("polygon_vertices", 0))
        solidity = float(component.get("solidity", 0.0))
        rectangular_like = polygon_vertices <= 6 and solidity >= 0.92 and fill_ratio >= 0.78

        if area > image_area * 0.7 and fill_ratio > 0.65:
            continue
        if rectangular_like:
            continue

        horizontal_ratio = w / max(h, 1)
        center_x_score = max(0.2, 1.2 - abs(centroid_x - center_x) / half_width)
        vertical_score = max(0.35, 1.22 - 0.55 * (centroid_y / max(height, 1)))
        aspect_score = 1.0 if 1.12 <= horizontal_ratio <= 4.5 else 0.7 if 0.9 <= horizontal_ratio <= 5.5 else 0.25
        fill_score = 1.0 if 0.12 <= fill_ratio <= 0.82 else 0.55
        saturation_score = 0.6 + min(mean_saturation, 120.0) / 120.0
        border_penalty = 0.75 if touches_border else 1.0

        score = area * center_x_score * vertical_score * aspect_score * fill_score * saturation_score * border_penalty
        if score > best_score:
            best_score = score
            best_mask = component["label_mask"]

    if best_mask is None:
        return np.zeros_like(mask), "fallback_center_crop"

    main_mask = best_mask.astype(np.uint8) * 255

    dilate_kernel = max(5, int(min(height, width) * 0.025))
    if dilate_kernel % 2 == 0:
        dilate_kernel += 1
    dilate_struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    main_mask = cv2.dilate(main_mask, dilate_struct, iterations=1)

    return main_mask, "foreground_component"


def refine_mask_with_grabcut(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return mask

    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    if width < 20 or height < 20:
        return mask

    grabcut_mask = np.full(mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    grabcut_mask[mask > 0] = cv2.GC_PR_FGD
    inset = max(3, int(min(mask.shape[:2]) * 0.01))
    grabcut_mask[:inset, :] = cv2.GC_BGD
    grabcut_mask[-inset:, :] = cv2.GC_BGD
    grabcut_mask[:, :inset] = cv2.GC_BGD
    grabcut_mask[:, -inset:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            image,
            grabcut_mask,
            None,
            bgd_model,
            fgd_model,
            2,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return mask

    refined = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    post_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, post_kernel, iterations=1)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, post_kernel, iterations=1)
    return refined


def prune_attached_artifacts(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return mask

    x0, y0, x1, y1 = bbox
    min_side = min(x1 - x0, y1 - y0)
    if min_side < 80:
        return mask

    def odd_kernel_size(value: int) -> int:
        return value if value % 2 == 1 else value + 1

    kernel_sizes = sorted(
        {
            odd_kernel_size(max(5, int(min_side * 0.015))),
            odd_kernel_size(max(7, int(min_side * 0.025))),
            odd_kernel_size(max(9, int(min_side * 0.040))),
        }
    )
    best_mask = mask
    best_metrics = score_selected_mask(mask, image)

    for kernel_size in kernel_sizes:
        struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(mask, struct, iterations=1)
        if not np.any(eroded > 0):
            continue
        eroded_main_mask, _ = select_main_component(eroded, image)
        if not np.any(eroded_main_mask > 0):
            continue
        restore_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(5, kernel_size - 2), max(5, kernel_size - 2)),
        )
        restored = cv2.dilate(eroded_main_mask, restore_kernel, iterations=1)
        restored = cv2.bitwise_and(mask, restored)
        post_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        restored = cv2.morphologyEx(restored, cv2.MORPH_CLOSE, post_kernel, iterations=1)
        if not np.any(restored > 0):
            continue
        restored_metrics = score_selected_mask(restored, image)
        improved = (
            restored_metrics["score"] > best_metrics["score"] + 0.04
            or (
                restored_metrics["low_saturation_fraction"] < best_metrics["low_saturation_fraction"] - 0.12
                and restored_metrics["symmetry_score"] >= best_metrics["symmetry_score"] - 0.05
            )
            or (
                not restored_metrics["touches_border"]
                and best_metrics["touches_border"]
                and restored_metrics["mean_saturation"] >= best_metrics["mean_saturation"] - 8.0
            )
        )
        if improved:
            best_mask = restored
            best_metrics = restored_metrics

    return best_mask


def refine_from_color_core(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    mask_pixels = mask > 0
    if not np.any(mask_pixels):
        return mask

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    mask_saturation = saturation[mask_pixels]
    mask_value = value[mask_pixels]
    if mask_saturation.size < 64:
        return mask

    saturation_threshold = max(24.0, min(72.0, float(np.percentile(mask_saturation, 60)) * 0.75))
    dark_value_threshold = float(np.percentile(mask_value, 35))
    seed = (
        mask_pixels
        & (
            (saturation >= saturation_threshold)
            | ((value <= dark_value_threshold) & (saturation >= saturation_threshold * 0.45))
        )
    ).astype(np.uint8) * 255
    if not np.any(seed > 0):
        return mask

    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    seed_main_mask, _ = select_main_component(seed, image)
    if not np.any(seed_main_mask > 0):
        return mask

    refined = refine_mask_with_grabcut(image, seed_main_mask)
    refined = remove_border_artifacts(refined, image)
    refined = prune_attached_artifacts(refined, image)
    refined_main_mask, _ = select_main_component(refined, image)
    if not np.any(refined_main_mask > 0):
        return mask

    original_metrics = score_selected_mask(mask, image)
    refined_metrics = score_selected_mask(refined_main_mask, image)
    improved = (
        refined_metrics["score"] > original_metrics["score"] + 0.03
        or refined_metrics["low_saturation_fraction"] < original_metrics["low_saturation_fraction"] - 0.12
        or (
            not refined_metrics["touches_border"]
            and original_metrics["touches_border"]
            and refined_metrics["mean_saturation"] >= original_metrics["mean_saturation"] - 10.0
        )
    )
    return refined_main_mask if improved else mask


def build_centered_grabcut_mask(
    image: np.ndarray,
    *,
    max_side: int = 960,
    iterations: int = 1,
    aggressive_top_crop: bool = False,
) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        resized = cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image

    resized_height, resized_width = resized.shape[:2]
    if aggressive_top_crop:
        rect = (
            max(0, int(resized_width * 0.17)),
            max(0, int(resized_height * 0.24)),
            max(32, int(resized_width * 0.66)),
            max(32, int(resized_height * 0.50)),
        )
    else:
        rect = (
            max(0, int(resized_width * 0.12)),
            max(0, int(resized_height * 0.14)),
            max(32, int(resized_width * 0.76)),
            max(32, int(resized_height * 0.68)),
        )

    grabcut_mask = np.zeros((resized_height, resized_width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            resized,
            grabcut_mask,
            rect,
            bgd_model,
            fgd_model,
            iterations,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return np.zeros((height, width), dtype=np.uint8)

    mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    if scale < 1.0:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def bbox_area_ratio(bbox: Tuple[int, int, int, int], width: int, height: int) -> float:
    x0, y0, x1, y1 = bbox
    return float(((x1 - x0) * (y1 - y0)) / max(width * height, 1))


def bbox_touches_border(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    margin: int = 2,
) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 <= margin or y0 <= margin or x1 >= width - margin or y1 >= height - margin


def centered_grabcut_bbox(
    image: np.ndarray,
    *,
    max_side: int = 1024,
    iterations: int = 1,
    aggressive_top_crop: bool = False,
) -> Optional[Tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        resized = cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image

    resized_height, resized_width = resized.shape[:2]
    rect_width_fraction = 0.64 if aggressive_top_crop else 0.72
    rect_x_fraction = 0.18 if aggressive_top_crop else 0.14
    rect_width = max(32, int(resized_width * rect_width_fraction))
    rect_height_fraction = 0.50 if aggressive_top_crop else 0.58
    rect_y_fraction = 0.30 if aggressive_top_crop else 0.22
    rect_height = max(32, int(resized_height * rect_height_fraction))
    rect_x = max(0, int(resized_width * rect_x_fraction))
    rect_y = max(0, int(resized_height * rect_y_fraction))
    rect = (rect_x, rect_y, rect_width, rect_height)

    grabcut_mask = np.zeros((resized_height, resized_width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            resized,
            grabcut_mask,
            rect,
            bgd_model,
            fgd_model,
            iterations,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return None

    foreground_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    bbox = bbox_from_mask(foreground_mask)
    if bbox is None:
        return None

    x0, y0, x1, y1 = bbox
    if scale < 1.0:
        return (
            int(x0 / scale),
            int(y0 / scale),
            int(x1 / scale),
            int(y1 / scale),
        )
    return bbox


def select_best_specimen_mask(
    image: np.ndarray,
    *,
    use_grabcut_refine: bool,
    segmentation_backend: str,
    rembg_model: str,
    rembg_provider: str,
) -> Tuple[np.ndarray, str, int, Dict[str, Any]]:
    def evaluate_candidate(candidate_mask: np.ndarray, method_prefix: str) -> Optional[Tuple[np.ndarray, str, Dict[str, Any], int]]:
        if not np.any(candidate_mask > 0):
            return None
        cleaned_mask = remove_border_artifacts(candidate_mask, image)
        cleaned_mask = reconnect_specimen_regions(cleaned_mask)
        if not np.any(cleaned_mask > 0):
            return None
        candidate_count = len(specimen_candidate_components(cleaned_mask, image))
        selected_mask, component_method = select_main_component(cleaned_mask, image)
        if not np.any(selected_mask > 0):
            return None
        selected_mask = prune_attached_artifacts(selected_mask, image)
        initial_metrics = score_selected_mask(selected_mask, image)
        if initial_metrics["low_saturation_fraction"] > 0.22 or initial_metrics["touches_border"]:
            selected_mask = refine_from_color_core(selected_mask, image)
        method = f"{method_prefix}_{component_method}"
        if use_grabcut_refine:
            refined_mask = refine_mask_with_grabcut(image, selected_mask)
            refined_mask = remove_border_artifacts(refined_mask, image)
            refined_mask = prune_attached_artifacts(refined_mask, image)
            refined_selected_mask, _ = select_main_component(refined_mask, image)
            if np.any(refined_selected_mask > 0):
                selected_mask = refined_selected_mask
                selected_mask = prune_attached_artifacts(selected_mask, image)
                method = f"{method}_grabcut"
        return selected_mask, method, score_selected_mask(selected_mask, image), candidate_count

    candidate_count = 0
    best_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    best_method = "fallback_center_crop"
    best_metrics = score_selected_mask(best_mask, image)
    best_score = -1.0

    if segmentation_backend in {"auto", "rembg"} and rembg_remove is not None and rembg_new_session is not None:
        rembg_mask = build_rembg_mask(image, model_name=rembg_model, provider_name=rembg_provider)
        rembg_candidate = evaluate_candidate(rembg_mask, f"rembg_{rembg_model}")
        if rembg_candidate is not None:
            selected_mask, method, metrics, component_count = rembg_candidate
            candidate_count = max(candidate_count, component_count)
            if is_strong_specimen_mask(metrics):
                return selected_mask, method, candidate_count, metrics
            best_mask = selected_mask
            best_method = method
            best_metrics = metrics
            best_score = metrics["score"]

    if segmentation_backend == "rembg":
        return best_mask, best_method, candidate_count, best_metrics

    heuristic_mask = build_foreground_mask(image)
    heuristic_candidate = evaluate_candidate(heuristic_mask, "heuristic_mask")
    if heuristic_candidate is not None:
        selected_mask, method, metrics, component_count = heuristic_candidate
        candidate_count = max(candidate_count, component_count)
        if is_strong_specimen_mask(metrics):
            return selected_mask, method, candidate_count, metrics
        best_mask = selected_mask
        best_method = method
        best_metrics = metrics
        best_score = metrics["score"]

    centered_mask = build_centered_grabcut_mask(image, aggressive_top_crop=False)
    upper_centered_mask = build_centered_grabcut_mask(image, aggressive_top_crop=True)
    candidate_masks: List[Tuple[np.ndarray, str]] = [
        (centered_mask, "centered_grabcut_mask"),
        (upper_centered_mask, "upper_centered_grabcut_mask"),
    ]
    if heuristic_candidate is not None and np.any(centered_mask > 0):
        candidate_masks.append((cv2.bitwise_or(heuristic_mask, centered_mask), "heuristic_plus_centered_mask"))
    if heuristic_candidate is not None and np.any(upper_centered_mask > 0):
        candidate_masks.append((cv2.bitwise_or(heuristic_mask, upper_centered_mask), "heuristic_plus_upper_centered_mask"))

    for candidate_mask, method_prefix in candidate_masks:
        candidate = evaluate_candidate(candidate_mask, method_prefix)
        if candidate is None:
            continue
        selected_mask, method, metrics, component_count = candidate
        candidate_count = max(candidate_count, component_count)
        if metrics["score"] > best_score:
            best_mask = selected_mask
            best_method = method
            best_metrics = metrics
            best_score = metrics["score"]

    return best_mask, best_method, candidate_count, best_metrics


def refine_crop_bbox(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Tuple[Tuple[int, int, int, int], str]:
    height, width = image.shape[:2]
    area_ratio = bbox_area_ratio(bbox, width, height)
    touches_border = bbox_touches_border(bbox, width, height)

    if not touches_border and area_ratio <= 0.68:
        return bbox, ""

    refined_bbox = centered_grabcut_bbox(image, aggressive_top_crop=touches_border)
    if refined_bbox is None:
        return bbox, ""

    refined_area_ratio = bbox_area_ratio(refined_bbox, width, height)
    refined_touches_border = bbox_touches_border(refined_bbox, width, height)
    refined_width = refined_bbox[2] - refined_bbox[0]
    refined_height = refined_bbox[3] - refined_bbox[1]
    refined_horizontal_ratio = refined_width / max(refined_height, 1)

    if refined_area_ratio < 0.03 or refined_horizontal_ratio < 0.75:
        return bbox, ""

    improved = False
    if touches_border and not refined_touches_border:
        improved = True
    elif refined_area_ratio < area_ratio * 0.82:
        improved = True
    elif refined_bbox[1] > bbox[1] + int(height * 0.05):
        improved = True
    elif refined_bbox[3] < bbox[3] - int(height * 0.05):
        improved = True

    if improved:
        return refined_bbox, "centered_grabcut_bbox"
    return bbox, ""


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    margin_fraction_x: float,
    margin_fraction_top: Optional[float] = None,
    margin_fraction_bottom: Optional[float] = None,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    box_width = x1 - x0
    box_height = y1 - y0
    margin_top_fraction = margin_fraction_x if margin_fraction_top is None else margin_fraction_top
    margin_bottom_fraction = margin_fraction_x if margin_fraction_bottom is None else margin_fraction_bottom
    margin_x = int(round(box_width * margin_fraction_x))
    margin_top = int(round(box_height * margin_top_fraction))
    margin_bottom = int(round(box_height * margin_bottom_fraction))

    left = max(0, x0 - margin_x)
    top = max(0, y0 - margin_top)
    right = min(width, x1 + margin_x)
    bottom = min(height, y1 + margin_bottom)
    return left, top, right, bottom


def fallback_center_bbox(width: int, height: int) -> Tuple[int, int, int, int]:
    side = int(min(width, height) * 0.92)
    left = (width - side) // 2
    top = (height - side) // 2
    return left, top, left + side, top + side


def crop_and_resize(
    image: np.ndarray,
    mask: np.ndarray,
    image_size: int,
    clean_background: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    crop = image.copy()
    crop_mask = mask.copy()

    if clean_background and np.any(crop_mask > 0):
        background_color = np.array([255, 255, 255], dtype=np.uint8)
        clean = crop.copy()
        clean[crop_mask == 0] = background_color
        crop = clean

    height, width = crop.shape[:2]
    side = max(height, width)
    square_image = np.full((side, side, 3), 255, dtype=np.uint8)
    square_mask = np.zeros((side, side), dtype=np.uint8)
    offset_y = (side - height) // 2
    offset_x = (side - width) // 2
    square_image[offset_y:offset_y + height, offset_x:offset_x + width] = crop
    square_mask[offset_y:offset_y + height, offset_x:offset_x + width] = crop_mask

    resized_image = cv2.resize(square_image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(square_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask


def build_output_name(row: Dict[str, Any], index: int) -> str:
    source = normalize_text(row.get("source")) or "unknown"
    record_id = normalize_text(row.get("record_id")) or f"row_{index:06d}"
    return f"{source}_{record_id}_{index:06d}"


def preprocess_row(
    row: Dict[str, Any],
    index: int,
    *,
    images_dir: Path,
    masks_dir: Path,
    previews_dir: Path,
    image_size: int,
    margin_fraction: float,
    clean_background: bool,
    save_masks_flag: bool,
    save_preview_flag: bool,
    grabcut_refine: bool,
    segmentation_backend: str,
    rembg_model: str,
    rembg_provider: str,
    work_max_side: int,
    jpeg_quality: int,
    require_dorsal: bool,
    reject_unknown_view: bool,
    allow_fallback_crops: bool,
) -> Dict[str, Any]:
    row = dict(row)
    resolved_image_path = normalize_text(row.get("image_path")) or normalize_text(row.get("local_image_path"))
    image_path = Path(resolved_image_path)
    row["source_image_path"] = resolved_image_path
    row["processed_image_path"] = ""
    row["processed_mask_path"] = ""
    row["preprocess_status"] = ""
    row["crop_method"] = ""
    row["crop_bbox"] = ""
    row["view_hint"] = ""
    row["view_source"] = ""
    row["view_status"] = ""
    row["rejection_reason"] = ""
    row["keep_for_training"] = ""
    row["symmetry_score"] = ""
    row["wing_balance_ratio"] = ""
    row["horizontal_ratio"] = ""
    row["fill_ratio"] = ""
    row["coverage_ratio"] = ""
    row["candidate_count"] = ""
    row["clean_background"] = ""
    row["mask_quality_score"] = ""
    row["mask_mean_saturation"] = ""
    row["mask_touches_border"] = ""
    row["polygon_vertices"] = ""
    row["mask_solidity"] = ""
    row["working_scale"] = ""
    row["working_image_size"] = ""
    row["source_row_index"] = str(index)

    if not image_path.exists():
        row["preprocess_status"] = "missing_input"
        return row

    view_hint, view_source = infer_view_hint(row)
    row["view_hint"] = view_hint
    row["view_source"] = view_source

    image = load_image_rgb(image_path, decode_max_side=max(work_max_side, image_size))
    working_image, working_scale = resize_working_image(image, work_max_side)
    height, width = working_image.shape[:2]

    main_mask, crop_method, candidate_count, mask_metrics = select_best_specimen_mask(
        working_image,
        use_grabcut_refine=grabcut_refine,
        segmentation_backend=segmentation_backend,
        rembg_model=rembg_model,
        rembg_provider=rembg_provider,
    )
    bbox = bbox_from_mask(main_mask)

    if bbox is None:
        bbox = fallback_center_bbox(width, height)
        crop_method = "fallback_center_crop"
        main_mask = np.zeros((height, width), dtype=np.uint8)
        main_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255
        mask_metrics = score_selected_mask(main_mask, working_image)

    readiness = assess_training_readiness(
        main_mask=main_mask,
        image=working_image,
        view_hint=view_hint,
        require_dorsal=require_dorsal,
        reject_unknown_view=reject_unknown_view,
        candidate_count=candidate_count,
        mask_metrics=mask_metrics,
    )
    row["view_status"] = readiness["view_status"]
    row["rejection_reason"] = readiness["rejection_reason"]
    row["keep_for_training"] = str(readiness["keep_for_training"])
    row["symmetry_score"] = f"{readiness['symmetry_score']:.4f}"
    row["wing_balance_ratio"] = f"{readiness['wing_balance_ratio']:.4f}"
    row["horizontal_ratio"] = f"{readiness['horizontal_ratio']:.4f}"
    row["fill_ratio"] = f"{readiness['fill_ratio']:.4f}"
    row["coverage_ratio"] = f"{readiness['coverage_ratio']:.4f}"
    row["mask_quality_score"] = f"{mask_metrics['score']:.4f}"
    row["mask_mean_saturation"] = f"{readiness['mean_saturation']:.2f}"
    row["mask_touches_border"] = str(readiness["touches_border"])
    row["polygon_vertices"] = str(readiness["polygon_vertices"])
    row["mask_solidity"] = f"{readiness['solidity']:.4f}"
    row["candidate_count"] = str(readiness["candidate_count"])

    if crop_method == "fallback_center_crop" and not allow_fallback_crops:
        row["preprocess_status"] = "rejected_view"
        row["view_status"] = "rejected_fallback_crop"
        row["rejection_reason"] = "fallback_crop"
        row["keep_for_training"] = "False"
        row["crop_method"] = crop_method
        return row

    if not readiness["keep_for_training"]:
        row["preprocess_status"] = "rejected_view"
        row["crop_method"] = crop_method
        return row

    crop_bbox = expand_bbox(
        bbox,
        width,
        height,
        margin_fraction_x=margin_fraction,
        margin_fraction_top=margin_fraction,
        margin_fraction_bottom=margin_fraction,
    )
    left, top, right, bottom = crop_bbox

    cropped_image = working_image[top:bottom, left:right, :]
    cropped_mask = main_mask[top:bottom, left:right]
    processed_image, processed_mask = crop_and_resize(
        cropped_image,
        cropped_mask,
        image_size=image_size,
        clean_background=clean_background,
    )

    stem = build_output_name(row, index)
    processed_image_path = images_dir / f"{stem}.jpg"
    save_image_rgb(processed_image_path, processed_image, jpeg_quality=jpeg_quality)

    processed_mask_path = masks_dir / f"{stem}.png"
    if save_masks_flag:
        save_mask(processed_mask_path, processed_mask)
    if save_preview_flag:
        preview_path = previews_dir / f"{stem}.jpg"
        preview_before = cv2.resize(cropped_image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        save_preview(preview_path, preview_before, processed_image)

    row["processed_image_path"] = str(processed_image_path)
    row["processed_mask_path"] = str(processed_mask_path) if save_masks_flag else ""
    row["preprocess_status"] = "ok"
    row["crop_method"] = crop_method
    row["crop_bbox"] = json.dumps([left, top, right, bottom])
    row["clean_background"] = str(clean_background)
    row["working_scale"] = f"{working_scale:.6f}"
    row["working_image_size"] = json.dumps([width, height])
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop and clean downloaded butterfly images.")
    parser.add_argument("--input-csv", required=True, help="Combined CSV from the fetch step.")
    parser.add_argument("--output-dir", required=True, help="Directory for processed outputs.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Final square image size. Defaults to 256.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for processed .jpg outputs.",
    )
    parser.add_argument(
        "--work-max-side",
        type=int,
        default=960,
        help="Resize large source images so their longest side is at most this value before segmentation and cropping.",
    )
    parser.add_argument(
        "--margin-fraction",
        type=float,
        default=0.08,
        help="Margin around the detected butterfly bounding box.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for development runs.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the input rows across N shards for parallel preprocessing.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to process when num-shards > 1.",
    )
    parser.set_defaults(clean_background=True)
    parser.add_argument(
        "--clean-background",
        dest="clean_background",
        action="store_true",
        help="Whiten background outside the detected butterfly mask. Enabled by default.",
    )
    parser.add_argument(
        "--keep-background",
        dest="clean_background",
        action="store_false",
        help="Keep the cropped background instead of whitening it.",
    )
    parser.set_defaults(save_masks=False)
    parser.add_argument(
        "--save-masks",
        dest="save_masks",
        action="store_true",
        help="Write binary masks for inspection.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional override for the processed CSV path.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Write progress checkpoints every N images.",
    )
    parser.add_argument(
        "--csv-checkpoint-every",
        type=int,
        default=1000,
        help="Rewrite the shard CSV checkpoint every N images. Final CSV is always written at the end.",
    )
    parser.set_defaults(save_previews=False)
    parser.add_argument(
        "--save-previews",
        dest="save_previews",
        action="store_true",
        help="Save before-and-after preview images for QA.",
    )
    parser.set_defaults(grabcut_refine=False)
    parser.add_argument(
        "--grabcut-refine",
        dest="grabcut_refine",
        action="store_true",
        help="Use GrabCut refinement for tougher backgrounds and label artifacts.",
    )
    parser.add_argument(
        "--segmentation-backend",
        choices=["auto", "heuristic", "rembg"],
        default="auto",
        help="Segmentation backend: auto prefers rembg when installed, otherwise heuristic.",
    )
    parser.add_argument(
        "--rembg-model",
        default="birefnet-general-lite",
        help="Rembg model to use when the rembg backend is active.",
    )
    parser.add_argument(
        "--rembg-provider",
        choices=["auto", "cuda", "directml", "cpu"],
        default="auto",
        help="Execution provider preference for rembg ONNX inference.",
    )
    parser.set_defaults(require_dorsal=True)
    parser.add_argument(
        "--require-dorsal",
        dest="require_dorsal",
        action="store_true",
        help="Reject images explicitly marked as ventral or lateral. Enabled by default.",
    )
    parser.add_argument(
        "--allow-any-view",
        dest="require_dorsal",
        action="store_false",
        help="Keep dorsal, ventral, or unknown-view images.",
    )
    parser.set_defaults(reject_unknown_view=False)
    parser.add_argument(
        "--reject-unknown-view",
        dest="reject_unknown_view",
        action="store_true",
        help="Reject images when the view cannot be inferred from metadata or shape heuristics.",
    )
    parser.set_defaults(allow_fallback_crops=False)
    parser.add_argument(
        "--allow-fallback-crops",
        dest="allow_fallback_crops",
        action="store_true",
        help="Keep center-crop fallback images instead of rejecting them. Disabled by default.",
    )
    parser.add_argument(
        "--cv-threads",
        type=int,
        default=1,
        help="OpenCV thread count per process. Use 1 when running many shard processes in parallel.",
    )
    return parser.parse_args()


def summarize_processed_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows_list = list(rows)
    total = len(rows_list)
    ok_rows = [row for row in rows_list if row.get("preprocess_status") == "ok"]
    fallback_rows = [row for row in ok_rows if row.get("crop_method") == "fallback_center_crop"]
    missing_rows = [row for row in rows_list if row.get("preprocess_status") == "missing_input"]
    rejected_rows = [row for row in rows_list if row.get("preprocess_status") == "rejected_view"]
    rejection_counts: Dict[str, int] = {}
    for row in rejected_rows:
        reason = normalize_text(row.get("rejection_reason")) or "unknown"
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    return {
        "total_rows": total,
        "processed_ok": len(ok_rows),
        "fallback_center_crop": len(fallback_rows),
        "missing_input": len(missing_rows),
        "rejected_view": len(rejected_rows),
        "rejection_counts": rejection_counts,
        "clean_background_enabled": any(row.get("clean_background") == "True" for row in ok_rows),
    }


def main() -> int:
    args = parse_args()
    ensure_runtime_dependencies()
    ensure_segmentation_backend_available(args.segmentation_backend)
    configure_cv_runtime(args.cv_threads)
    configure_rembg_runtime()
    if args.segmentation_backend in {"auto", "rembg"}:
        preload_onnxruntime_gpu_dlls()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    previews_dir = output_dir / "previews"
    output_csv = Path(args.output_csv) if args.output_csv else output_dir / "processed_metadata.csv"
    summary_json = output_dir / "preprocess_summary.json"
    progress_json = output_dir / "progress.json"

    ensure_dir(output_dir)
    ensure_dir(images_dir)
    if args.save_masks:
        ensure_dir(masks_dir)
    if args.save_previews:
        ensure_dir(previews_dir)

    rows = read_csv_rows(input_csv, max_rows=args.max_images)
    if args.num_shards < 1:
        raise RuntimeError("--num-shards must be at least 1.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise RuntimeError("--shard-index must be in [0, num_shards).")

    indexed_rows = list(enumerate(rows, start=1))
    selected_rows = [
        (source_index, row)
        for source_index, row in indexed_rows
        if (source_index - 1) % args.num_shards == args.shard_index
    ]
    processed_rows = []
    write_json(
        progress_json,
        {
            "stage": "starting",
            "input_csv": str(input_csv),
            "output_dir": str(output_dir),
            "total_input_rows": len(rows),
            "total_rows": len(selected_rows),
            "processed": 0,
            "rembg_provider": args.rembg_provider,
            "cv_threads": args.cv_threads,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
    )

    for processed_index, (source_index, row) in enumerate(selected_rows, start=1):
        if processed_index % 100 == 0:
            print(f"Processed {processed_index}/{len(selected_rows)} images in shard {args.shard_index}...")
        processed_rows.append(
            preprocess_row(
                row,
                source_index,
                images_dir=images_dir,
                masks_dir=masks_dir,
                previews_dir=previews_dir,
                image_size=args.image_size,
                margin_fraction=args.margin_fraction,
                clean_background=args.clean_background,
                save_masks_flag=args.save_masks,
                save_preview_flag=args.save_previews,
                grabcut_refine=args.grabcut_refine,
                segmentation_backend=args.segmentation_backend,
                rembg_model=args.rembg_model,
                rembg_provider=args.rembg_provider,
                work_max_side=args.work_max_side,
                jpeg_quality=args.jpeg_quality,
                require_dorsal=args.require_dorsal,
                reject_unknown_view=args.reject_unknown_view,
                allow_fallback_crops=args.allow_fallback_crops,
            )
        )
        should_write_progress = processed_index % args.checkpoint_every == 0 or processed_index == len(selected_rows)
        should_write_csv = (
            args.csv_checkpoint_every > 0
            and (processed_index % args.csv_checkpoint_every == 0 or processed_index == len(selected_rows))
        )
        if should_write_csv:
            write_csv_rows(output_csv, processed_rows)
        if should_write_progress:
            write_json(
                progress_json,
                {
                    "stage": "processing",
                    "input_csv": str(input_csv),
                    "output_dir": str(output_dir),
                    "total_input_rows": len(rows),
                    "total_rows": len(selected_rows),
                    "processed": processed_index,
                    "clean_background": args.clean_background,
                    "save_masks": args.save_masks,
                    "save_previews": args.save_previews,
                    "grabcut_refine": args.grabcut_refine,
                    "segmentation_backend": args.segmentation_backend,
                    "rembg_model": args.rembg_model,
                    "rembg_provider": args.rembg_provider,
                    "work_max_side": args.work_max_side,
                    "jpeg_quality": args.jpeg_quality,
                    "cv_threads": args.cv_threads,
                    "require_dorsal": args.require_dorsal,
                    "reject_unknown_view": args.reject_unknown_view,
                    "allow_fallback_crops": args.allow_fallback_crops,
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                    "stats": summarize_processed_rows(processed_rows),
                },
            )

    write_csv_rows(output_csv, processed_rows)

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "output_dir": str(output_dir),
        "image_size": args.image_size,
        "margin_fraction": args.margin_fraction,
        "clean_background": args.clean_background,
        "save_masks": args.save_masks,
        "save_previews": args.save_previews,
        "grabcut_refine": args.grabcut_refine,
        "segmentation_backend": args.segmentation_backend,
        "rembg_model": args.rembg_model,
        "rembg_provider": args.rembg_provider,
        "work_max_side": args.work_max_side,
        "jpeg_quality": args.jpeg_quality,
        "cv_threads": args.cv_threads,
        "require_dorsal": args.require_dorsal,
        "reject_unknown_view": args.reject_unknown_view,
        "allow_fallback_crops": args.allow_fallback_crops,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "stats": summarize_processed_rows(processed_rows),
    }
    write_json(summary_json, summary)
    write_json(
        progress_json,
        {
            "stage": "complete",
            "input_csv": str(input_csv),
            "output_dir": str(output_dir),
            "output_csv": str(output_csv),
            "summary_json": str(summary_json),
            "stats": summary["stats"],
        },
    )

    print("Done.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
