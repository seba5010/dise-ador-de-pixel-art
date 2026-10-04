"""Pure anatomical measurements and an optional differentiable silhouette loss."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


ENABLE_SILHOUETTE_LOSS = os.environ.get("PIXEL_AI_ENABLE_SILHOUETTE_LOSS", "0").strip().lower() not in {
    "0", "false", "no", "off"
}


def _rgba_array(image: Any) -> np.ndarray:
    if hasattr(image, "convert"):
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("Expected an HxWxC image")
    if array.shape[2] == 4:
        return array.astype(np.uint8, copy=False)
    if array.shape[2] == 3:
        alpha = np.full(array.shape[:2] + (1,), 255, dtype=np.uint8)
        return np.concatenate([array.astype(np.uint8, copy=False), alpha], axis=2)
    raise ValueError("Expected RGB or RGBA channels")


def foreground_mask(image: Any, *, alpha_threshold: int = 16) -> np.ndarray:
    return _rgba_array(image)[..., 3] >= int(alpha_threshold)


def _bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def compute_anatomical_metrics(generated: Any, target: Any) -> Dict[str, float]:
    generated_mask = foreground_mask(generated)
    target_mask = foreground_mask(target)
    if generated_mask.shape != target_mask.shape:
        raise ValueError("Generated and target masks must share a canvas")
    height, width = generated_mask.shape
    generated_box = _bbox(generated_mask)
    target_box = _bbox(target_mask)
    if generated_box is None or target_box is None:
        return {
            "body_height_ratio": 0.0,
            "body_width_ratio": 0.0,
            "center_offset_x": 1.0,
            "foot_anchor_error": 1.0,
            "silhouette_iou": 0.0,
            "pose_alignment": 0.0,
            "body_proportion_error": 1.0,
            "cuerpo_precision": 0.0,
            "silueta_iou_real": 0.0,
        }

    gx0, gy0, gx1, gy1 = generated_box
    tx0, ty0, tx1, ty1 = target_box
    generated_height = (gy1 - gy0) / max(1, height)
    generated_width = (gx1 - gx0) / max(1, width)
    target_height = (ty1 - ty0) / max(1, height)
    target_width = (tx1 - tx0) / max(1, width)
    center_offset = abs(((gx0 + gx1) / 2.0) - ((tx0 + tx1) / 2.0)) / max(1, width)
    foot_error = abs(gy1 - ty1) / max(1, height)
    intersection = int(np.logical_and(generated_mask, target_mask).sum())
    union = int(np.logical_or(generated_mask, target_mask).sum())
    iou = intersection / union if union else 1.0
    proportion_error = (abs(generated_height - target_height) + abs(generated_width - target_width)) / 2.0
    pose_alignment = max(0.0, 1.0 - (center_offset + foot_error) / 2.0)
    anatomy = max(0.0, 1.0 - (proportion_error + center_offset + foot_error) / 3.0)
    return {
        "body_height_ratio": round(generated_height, 6),
        "body_width_ratio": round(generated_width, 6),
        "target_body_height_ratio": round(target_height, 6),
        "target_body_width_ratio": round(target_width, 6),
        "center_offset_x": round(center_offset, 6),
        "foot_anchor_error": round(foot_error, 6),
        "silhouette_iou": round(iou * 100.0, 4),
        "pose_alignment": round(pose_alignment * 100.0, 4),
        "body_proportion_error": round(proportion_error, 6),
        "cuerpo_precision": round(anatomy * 100.0, 4),
        "silueta_iou_real": round(iou * 100.0, 4),
    }


def differentiable_silhouette_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft IoU loss over alpha channels; inputs use the project's -1..1 range."""
    if prediction.ndim != 4 or target.ndim != 4 or prediction.shape[1] < 4 or target.shape[1] < 4:
        raise ValueError("Expected NCHW tensors with an alpha channel")
    pred_alpha = ((prediction[:, 3:4] + 1.0) * 0.5).clamp(0.0, 1.0)
    target_alpha = ((target[:, 3:4] + 1.0) * 0.5).clamp(0.0, 1.0)
    dimensions = tuple(range(1, pred_alpha.ndim))
    intersection = (pred_alpha * target_alpha).sum(dim=dimensions)
    union = (pred_alpha + target_alpha - pred_alpha * target_alpha).sum(dim=dimensions)
    return (1.0 - (intersection + eps) / (union + eps)).mean()


__all__ = [
    "ENABLE_SILHOUETTE_LOSS",
    "compute_anatomical_metrics",
    "differentiable_silhouette_loss",
    "foreground_mask",
]
