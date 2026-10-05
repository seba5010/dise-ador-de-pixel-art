import numpy as np
from PIL import Image, ImageDraw

from pixel_ai_engine.quality_gate import QualityGate
from pixel_ai_engine.strict_visual_quality import (
    apply_strict_visual_metrics,
    compute_strict_visual_metrics,
)


def _character():
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((9, 3, 22, 14), fill=(210, 150, 105, 255), outline=(35, 25, 25, 255))
    draw.point((13, 9), fill=(10, 10, 10, 255))
    draw.point((18, 9), fill=(10, 10, 10, 255))
    draw.rectangle((10, 14, 21, 25), fill=(225, 225, 225, 255), outline=(30, 30, 30, 255))
    draw.rectangle((8, 16, 9, 23), fill=(210, 150, 105, 255))
    draw.rectangle((22, 16, 23, 23), fill=(210, 150, 105, 255))
    draw.rectangle((11, 26, 14, 29), fill=(35, 35, 40, 255))
    draw.rectangle((17, 26, 20, 29), fill=(35, 35, 40, 255))
    return image


def test_identical_target_receives_high_strict_scores():
    target = _character()

    metrics = compute_strict_visual_metrics(target, target)

    assert metrics["strict_global"] >= 99.0
    assert metrics["strict_face"] >= 99.0
    assert metrics["strict_anatomy"] >= 99.0
    assert metrics["strict_visual_noise"] >= 99.0


def test_same_silhouette_with_random_pixel_noise_cannot_score_perfect_face_or_body():
    target = _character()
    noisy = np.asarray(target).copy()
    mask = noisy[..., 3] > 30
    rng = np.random.default_rng(42)
    noisy[mask, :3] = rng.integers(0, 256, size=(int(mask.sum()), 3), dtype=np.uint8)
    generated = Image.fromarray(noisy, mode="RGBA")

    metrics = compute_strict_visual_metrics(generated, target)

    assert metrics["strict_silhouette"] == 100.0
    assert metrics["strict_face"] < 75.0
    assert metrics["strict_anatomy"] < 75.0
    assert metrics["strict_visual_noise"] < 80.0
    assert metrics["strict_global"] < 70.0


def test_strict_merge_can_only_lower_optimistic_legacy_metrics():
    target = _character()
    generated = target.copy()
    ImageDraw.Draw(generated).rectangle((9, 3, 22, 14), fill=(255, 0, 255, 255))
    legacy = {
        "score_total": 100.0,
        "cuerpo_precision": 100.0,
        "gestos_ojos": 100.0,
        "detalle_facial": 100.0,
        "ruido_huerfano": 100.0,
        "micro_detalles": 100.0,
    }

    merged = apply_strict_visual_metrics(legacy, generated, target)

    assert merged["score_total"] < 100.0
    assert merged["gestos_ojos"] < 75.0
    assert merged["detalle_facial"] == merged["strict_face"]


def test_quality_gate_blocks_distorted_face_even_with_matching_silhouette():
    target = _character()
    noisy = np.asarray(target).copy()
    face = (noisy[..., 3] > 30) & (np.indices(noisy.shape[:2])[0] < 15)
    rng = np.random.default_rng(7)
    noisy[face, :3] = rng.integers(0, 256, size=(int(face.sum()), 3), dtype=np.uint8)
    generated = Image.fromarray(noisy, mode="RGBA")

    audit = QualityGate.evaluate_single_frame(generated, target, frame_idx=0)

    assert audit["aprobado"] is False
    assert "FACE_STRUCTURE_FAIL" in audit["issues"]
    assert audit["quality"]["face"] < 75.0
