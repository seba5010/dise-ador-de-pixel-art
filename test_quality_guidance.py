import json

import pytest

from pixel_ai_engine.quality_guidance import (
    QualityVector,
    build_quality_vector,
    normalize_quality_score,
)


def test_quality_score_normalization_is_finite_and_bounded():
    assert normalize_quality_score(0.82) == 82.0
    assert normalize_quality_score(0.0) == 0.0
    assert normalize_quality_score(120.0) == 100.0
    assert normalize_quality_score(-4.0) == 0.0
    assert normalize_quality_score(float("nan")) is None
    assert normalize_quality_score(float("inf")) is None
    assert normalize_quality_score(True) is None
    assert normalize_quality_score("invalid") is None


def test_quality_vector_maps_quality_gate_aliases_without_inventing_missing_scores():
    vector = build_quality_vector(
        {
            "score_total": 76.8,
            "fidelidad_paleta": 91,
            "alineacion_molde": 79,
            "micro_textura": 68,
            "pureza_alfa": 99,
            "score_gestos_ojos": 61,
            "score_ropa_delantal": 94,
            "score_pelo_gorro": 88,
            "score_tatuajes_brazos": 74,
            "score_objetos_utensilios": 69.5,
            "score_zapatos_pies": 96,
        }
    )

    payload = vector.to_dict()
    assert payload["global"] == 76.8
    assert payload["silhouette"] == 79.0
    assert payload["pose"] == 79.0
    assert payload["face"] == 61.0
    assert payload["clothing"] == 94.0
    assert payload["palette"] == 91.0
    assert payload["alpha"] == 99.0
    assert payload["micro_detail"] == 68.0
    assert payload["anatomy"] is None
    assert payload["outline"] is None


def test_quality_vector_maps_enhancer_and_critical_metrics():
    vector = build_quality_vector(
        {
            "cuerpo_precision": 82,
            "silueta_iou_real": 79,
            "gestos_ojos": 61,
            "pelo_gorro": 88,
            "ropa_delantal": 94,
            "tatuajes_brazos": 74,
            "objetos_utensilios": 69.5,
            "zapatos_pies": 96,
            "fidelidad_paleta": 92,
            "riqueza_paleta": 90,
            "pureza_alfa": 99,
            "micro_detalles": 68,
            "nitidez_bordes": 80,
            "pureza_bordes": 76,
            "definicion_tinta": 74,
        },
        {"g_loss": 8.4, "d_loss": 0.7, "l1_loss": 0.08, "edge_loss": 0.04, "lr": 1.5e-4},
    )

    # The fallback global score averages only categories that were actually observed.
    assert vector.global_score == pytest.approx(81.5139, abs=0.001)
    assert vector.anatomy == 82.0
    assert vector.palette == 91.0
    assert vector.outline == pytest.approx(76.6667, abs=0.001)
    assert vector.training_stability == 100.0


def test_missing_is_distinct_from_measured_zero_and_payload_is_json_safe():
    vector = build_quality_vector({"score_total": 0, "pureza_alfa": 0, "defectos_cuerpo": 400})

    assert vector.global_score == 0.0
    assert vector.alpha == 0.0
    assert vector.face is None
    assert "face" not in vector.to_dict(include_unavailable=False)
    json.dumps(vector.to_dict(), allow_nan=False)


def test_quality_vector_round_trip_preserves_unavailable_categories():
    original = QualityVector(global_score=75.0, face=42.0, alpha=99.0)
    restored = QualityVector.from_dict(original.to_dict())
    assert restored == original


def test_training_stability_penalizes_non_finite_metrics_and_skipped_amp_steps():
    vector = build_quality_vector(
        {"score_total": 80},
        {
            "g_loss": float("nan"),
            "d_loss": 0.7,
            "l1_loss": 0.08,
            "edge_loss": 0.04,
            "lr": 1.5e-4,
            "skipped_amp_steps": 5,
        },
    )
    assert vector.training_stability == 65.0
