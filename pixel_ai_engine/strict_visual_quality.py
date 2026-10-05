"""Conservative target-aware visual quality metrics for pixel-art sprites.

High-frequency noise must never be interpreted as facial or anatomical detail.
All structural scores compare the generated sprite with the real dataset target.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, convolve, gaussian_filter, label


def _rgba(image: Any) -> np.ndarray:
    if hasattr(image, "convert"):
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError("Expected an RGB or RGBA image")
    if array.shape[2] == 3:
        alpha = np.full(array.shape[:2] + (1,), 255, dtype=np.uint8)
        array = np.concatenate([array, alpha], axis=2)
    return array.astype(np.uint8, copy=False)


def _bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.logical_or(left, right).sum())
    if not union:
        return 100.0
    return float(np.logical_and(left, right).sum()) / union * 100.0


def _edge_map(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    luminance = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    dx = np.abs(np.diff(luminance, axis=1, prepend=luminance[:, :1]))
    dy = np.abs(np.diff(luminance, axis=0, prepend=luminance[:1, :]))
    return ((dx + dy) >= 30.0) & binary_dilation(mask, structure=np.ones((3, 3), dtype=bool))


def _region_score(
    generated_rgb: np.ndarray,
    target_rgb: np.ndarray,
    generated_mask: np.ndarray,
    target_mask: np.ndarray,
    region: np.ndarray,
) -> Dict[str, float]:
    gm = generated_mask & region
    tm = target_mask & region
    union = gm | tm
    if not np.any(tm):
        return {"score": 0.0, "mask_iou": 0.0, "color": 0.0, "edge": 0.0}

    mask_iou = _iou(gm, tm)
    generated_blur = gaussian_filter(generated_rgb.astype(np.float32), sigma=(0.65, 0.65, 0.0))
    target_blur = gaussian_filter(target_rgb.astype(np.float32), sigma=(0.65, 0.65, 0.0))
    blurred_distance = np.mean(np.abs(generated_blur[union] - target_blur[union]))
    raw_distance = np.mean(np.abs(generated_rgb[union].astype(np.float32) - target_rgb[union].astype(np.float32)))
    blurred_score = max(0.0, 100.0 - float(blurred_distance) / 1.8)
    raw_score = max(0.0, 100.0 - float(raw_distance) / 1.2)
    color_score = raw_score * 0.75 + blurred_score * 0.25

    generated_edges = _edge_map(generated_rgb, gm) & region
    target_edges = _edge_map(target_rgb, tm) & region
    generated_count = int(generated_edges.sum())
    target_count = int(target_edges.sum())
    if target_count == 0:
        edge_score = 100.0 if generated_count == 0 else 0.0
    else:
        tolerant_target = binary_dilation(target_edges, structure=np.ones((3, 3), dtype=bool))
        tolerant_generated = binary_dilation(generated_edges, structure=np.ones((3, 3), dtype=bool))
        overlap = int((generated_edges & tolerant_target).sum()) + int((target_edges & tolerant_generated).sum())
        edge_score = min(100.0, overlap / max(1, generated_count + target_count) * 100.0)
        density_balance = min(generated_count, target_count) / max(1, generated_count, target_count)
        edge_score *= density_balance

    score = mask_iou * 0.30 + color_score * 0.45 + edge_score * 0.25
    return {
        "score": round(max(0.0, min(100.0, score)), 4),
        "mask_iou": round(mask_iou, 4),
        "color": round(color_score, 4),
        "edge": round(edge_score, 4),
    }


def _visual_noise_score(
    generated_rgb: np.ndarray,
    target_rgb: np.ndarray,
    generated_mask: np.ndarray,
    target_mask: np.ndarray,
) -> Dict[str, float]:
    structure = np.ones((3, 3), dtype=int)
    generated_labels, generated_components = label(generated_mask, structure=structure)
    _, target_components = label(target_mask, structure=structure)
    sizes = np.bincount(generated_labels.ravel())[1:] if generated_components else np.array([], dtype=int)
    speckle_pixels = int(sizes[sizes <= 4].sum()) if len(sizes) else 0
    speckle_ratio = speckle_pixels / max(1, int(generated_mask.sum()))

    generated_edges = _edge_map(generated_rgb, generated_mask)
    target_edges = _edge_map(target_rgb, target_mask)
    generated_edge_density = float(generated_edges.sum()) / max(1, int(generated_mask.sum()))
    target_edge_density = float(target_edges.sum()) / max(1, int(target_mask.sum()))
    edge_excess = max(0.0, generated_edge_density / max(0.01, target_edge_density) - 1.0)

    quantized_generated = generated_rgb[generated_mask] // 16
    quantized_target = target_rgb[target_mask] // 16
    generated_colors = len(np.unique(quantized_generated, axis=0)) if len(quantized_generated) else 0
    target_colors = len(np.unique(quantized_target, axis=0)) if len(quantized_target) else 0
    color_excess = max(0.0, generated_colors / max(1, target_colors) - 1.0)

    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    generated_lum = generated_rgb[..., 0] * 0.299 + generated_rgb[..., 1] * 0.587 + generated_rgb[..., 2] * 0.114
    target_lum = target_rgb[..., 0] * 0.299 + target_rgb[..., 1] * 0.587 + target_rgb[..., 2] * 0.114
    generated_hf = float(np.mean(np.abs(convolve(generated_lum, kernel))[generated_mask])) if np.any(generated_mask) else 0.0
    target_hf = float(np.mean(np.abs(convolve(target_lum, kernel))[target_mask])) if np.any(target_mask) else 0.0
    frequency_excess = max(0.0, generated_hf / max(1.0, target_hf) - 1.0)

    extra_components = max(0, int(generated_components) - int(target_components))
    penalty = (
        min(45.0, speckle_ratio * 500.0)
        + min(25.0, edge_excess * 25.0)
        + min(20.0, color_excess * 20.0)
        + min(20.0, frequency_excess * 20.0)
        + min(20.0, extra_components * 4.0)
    )
    return {
        "score": round(max(0.0, 100.0 - penalty), 4),
        "component_count": int(generated_components),
        "target_component_count": int(target_components),
        "speckle_ratio": round(speckle_ratio, 6),
        "edge_excess": round(edge_excess, 6),
        "color_excess": round(color_excess, 6),
        "frequency_excess": round(frequency_excess, 6),
    }


def compute_strict_visual_metrics(generated: Any, target: Any) -> Dict[str, Any]:
    """Score structure against a real target and penalize visual noise."""
    generated_array = _rgba(generated)
    target_array = _rgba(target)
    if generated_array.shape != target_array.shape:
        raise ValueError("Generated and target images must share a canvas")

    generated_mask = generated_array[..., 3] > 30
    target_mask = target_array[..., 3] > 30
    target_box = _bbox(target_mask)
    if target_box is None or not np.any(generated_mask):
        return {
            "strict_global": 0.0,
            "strict_anatomy": 0.0,
            "strict_face": 0.0,
            "strict_silhouette": 0.0,
            "strict_visual_noise": 0.0,
            "strict_audit_available": False,
        }

    x0, y0, x1, y1 = target_box
    height = max(1, y1 - y0)
    full_region = np.ones(target_mask.shape, dtype=bool)
    face_region = np.zeros(target_mask.shape, dtype=bool)
    body_region = np.zeros(target_mask.shape, dtype=bool)
    face_region[max(0, y0 + int(height * 0.08)):min(target_mask.shape[0], y0 + int(height * 0.42)), x0:x1] = True
    body_region[max(0, y0 + int(height * 0.30)):min(target_mask.shape[0], y0 + int(height * 0.82)), x0:x1] = True

    generated_rgb = generated_array[..., :3]
    target_rgb = target_array[..., :3]
    whole = _region_score(generated_rgb, target_rgb, generated_mask, target_mask, full_region)
    face = _region_score(generated_rgb, target_rgb, generated_mask, target_mask, face_region)
    body = _region_score(generated_rgb, target_rgb, generated_mask, target_mask, body_region)
    noise = _visual_noise_score(generated_rgb, target_rgb, generated_mask, target_mask)
    silhouette = _iou(generated_mask, target_mask)
    global_score = (
        whole["score"] * 0.25
        + body["score"] * 0.25
        + face["score"] * 0.25
        + silhouette * 0.05
        + noise["score"] * 0.20
    )
    return {
        "strict_global": round(max(0.0, min(100.0, global_score)), 4),
        "strict_anatomy": body["score"],
        "strict_face": face["score"],
        "strict_silhouette": round(silhouette, 4),
        "strict_visual_noise": noise["score"],
        "strict_whole_structure": whole["score"],
        "strict_face_details": face,
        "strict_body_details": body,
        "strict_noise_details": noise,
        "strict_audit_available": True,
    }


def apply_strict_visual_metrics(metrics: Dict[str, Any], generated: Any, target: Any) -> Dict[str, Any]:
    """Merge strict scores conservatively; strict metrics can lower, never inflate."""
    strict = compute_strict_visual_metrics(generated, target)
    merged = dict(metrics)
    merged.update(strict)
    if not strict.get("strict_audit_available"):
        return merged

    def conservative(key: str, strict_key: str) -> None:
        strict_value = float(strict[strict_key])
        current = merged.get(key)
        try:
            merged[key] = round(min(float(current), strict_value), 4)
        except (TypeError, ValueError):
            merged[key] = round(strict_value, 4)

    conservative("score_total", "strict_global")
    conservative("cuerpo_precision", "strict_anatomy")
    conservative("gestos_ojos", "strict_face")
    conservative("detalle_facial", "strict_face")
    conservative("silueta_iou_real", "strict_silhouette")
    conservative("silhouette_iou", "strict_silhouette")
    conservative("ruido_huerfano", "strict_visual_noise")
    conservative("micro_detalles", "strict_visual_noise")
    return merged


__all__ = ["apply_strict_visual_metrics", "compute_strict_visual_metrics"]
