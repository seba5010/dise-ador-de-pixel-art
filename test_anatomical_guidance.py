import numpy as np
import torch
from PIL import Image

from pixel_ai_engine.anatomical_guidance import (
    ENABLE_SILHOUETTE_LOSS,
    compute_anatomical_metrics,
    differentiable_silhouette_loss,
)
from pixel_ai_engine.quality_guidance import build_quality_vector


def _sprite(box, size=(16, 16)):
    data = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    x0, y0, x1, y1 = box
    data[y0:y1, x0:x1, :3] = 180
    data[y0:y1, x0:x1, 3] = 255
    return Image.fromarray(data, "RGBA")


def test_identical_masks_produce_perfect_anatomical_scores():
    sprite = _sprite((4, 2, 12, 15))
    metrics = compute_anatomical_metrics(sprite, sprite)

    assert metrics["body_height_ratio"] == 13 / 16
    assert metrics["body_width_ratio"] == 0.5
    assert metrics["center_offset_x"] == 0.0
    assert metrics["foot_anchor_error"] == 0.0
    assert metrics["silhouette_iou"] == 100.0
    assert metrics["pose_alignment"] == 100.0
    assert metrics["body_proportion_error"] == 0.0


def test_shift_and_scale_are_measured_without_mutating_inputs():
    generated = _sprite((6, 4, 13, 14))
    target = _sprite((4, 2, 12, 15))
    before = np.asarray(generated).copy()
    metrics = compute_anatomical_metrics(generated, target)

    assert metrics["center_offset_x"] > 0.0
    assert metrics["foot_anchor_error"] > 0.0
    assert metrics["body_proportion_error"] > 0.0
    assert 0.0 < metrics["silhouette_iou"] < 100.0
    assert np.array_equal(np.asarray(generated), before)


def test_anatomical_metrics_feed_quality_vector():
    metrics = compute_anatomical_metrics(_sprite((5, 3, 12, 14)), _sprite((4, 2, 12, 15)))
    vector = build_quality_vector(metrics).to_dict()

    assert vector["anatomy"] == metrics["cuerpo_precision"]
    assert vector["silhouette"] == metrics["silueta_iou_real"]
    assert vector["pose"] == metrics["pose_alignment"]


def test_silhouette_loss_is_finite_differentiable_and_disabled_by_default():
    prediction = torch.zeros(2, 4, 8, 8, requires_grad=True)
    target = torch.zeros(2, 4, 8, 8)
    target[:, 3, 2:6, 2:6] = 1.0
    loss = differentiable_silhouette_loss(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert ENABLE_SILHOUETTE_LOSS is False
