import json

import pytest

from pixel_ai_engine.quality_guidance import (
    FrameQualityTracker,
    GuidanceInterventionPolicy,
    HardExampleMiningPolicy,
    LossMultiplierController,
    QualityGuidanceConfig,
    QualityGuidanceController,
    QualityTrendAnalyzer,
    QualityVector,
    compare_sampling_ab,
    compare_loss_ab,
    build_quality_vector,
    diagnose_quality_bottleneck,
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


def test_bottleneck_diagnosis_does_not_hide_weak_face_behind_global_score():
    diagnosis = diagnose_quality_bottleneck(
        QualityVector(
            global_score=92,
            anatomy=86,
            silhouette=84,
            face=42,
            clothing=98,
            palette=99,
            alpha=100,
        )
    )

    assert diagnosis["primary_problem"] == "face"
    assert diagnosis["severity"] == "critical"
    assert diagnosis["critical_breaches"] == []  # 42 is poor, but above the configurable floor of 40.
    assert diagnosis["confidence"] >= 0.6


@pytest.mark.parametrize(
    ("face_score", "expected"),
    [(68, "low"), (62, "medium"), (50, "high"), (30, "critical")],
)
def test_severity_levels_are_driven_by_category_deficit(face_score, expected):
    diagnosis = diagnose_quality_bottleneck(
        QualityVector(face=face_score),
        thresholds={"face": 70},
        critical_floors={"face": 35},
    )
    assert diagnosis["severity"] == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([60, 65, 72], "IMPROVING"),
        ([72, 72.5, 72], "PLATEAU"),
        ([72, 67, 63], "REGRESSION"),
        ([70, 78, 68, 77], "OSCILLATION"),
        ([72, 60, 51], "COLLAPSE"),
    ],
)
def test_trend_analyzer_detects_required_states(values, expected):
    history = [QualityVector(global_score=80, face=value) for value in values]
    trend = QualityTrendAnalyzer().analyze(history)
    assert trend["by_category"]["face"] == expected


def test_regression_escalates_severity_even_when_latest_score_is_acceptable():
    trend = QualityTrendAnalyzer().analyze(
        [QualityVector(face=78), QualityVector(face=74), QualityVector(face=70)]
    )
    diagnosis = diagnose_quality_bottleneck(QualityVector(face=70), trend=trend)
    assert diagnosis["primary_problem"] is None
    assert diagnosis["severity"] == "medium"


def test_observational_controller_recommends_but_never_modifies_training():
    controller = QualityGuidanceController()
    decision = controller.evaluate(
        {
            "score_total": 76.8,
            "cuerpo_precision": 82,
            "silueta_iou_real": 79,
            "gestos_ojos": 61,
            "ropa_delantal": 94,
            "fidelidad_paleta": 91,
            "pureza_alfa": 99,
            "micro_detalles": 68,
        },
        epoch=40,
    )

    assert decision["mode"] == "observational"
    assert decision["primary_problem"] == "face"
    assert decision["severity"] == "medium"
    assert decision["recommended_action"] == "REINFORCE"
    assert decision["action"] == "CONTINUE"
    assert decision["training_modified"] is False


def test_controller_state_round_trip_is_backward_compatible():
    controller = QualityGuidanceController(config=QualityGuidanceConfig(targets={"face": 74}))
    controller.evaluate({"score_total": 80, "gestos_ojos": 65}, epoch=10)
    serialized = json.loads(json.dumps(controller.export_state(), allow_nan=False))

    restored = QualityGuidanceController(state=serialized)
    assert restored.export_state()["quality_history"] == serialized["quality_history"]
    assert restored.export_state()["decision_history"] == serialized["decision_history"]
    assert restored.config.targets["face"] == 74.0

    empty = QualityGuidanceController(state={"legacy_checkpoint": True})
    assert empty.export_state()["last_action"] == "CONTINUE"


def test_config_forces_observational_mode_during_increment_one():
    config = QualityGuidanceConfig(mode="active", targets={"face": 75})
    assert config.mode == "observational"
    assert config.targets["face"] == 75.0


def test_frame_quality_tracker_uses_stable_keys_and_persists_ema():
    tracker = FrameQualityTracker(ema_decay=0.5)
    tracker.update("alex_rbchef", 7, 80, epoch=1, metrics={"color": 82, "silhouette": 78})
    record = tracker.update("alex_rbchef", 7, 60, epoch=2, metrics={"color": 62}, confidence=0.9)

    assert record is not None
    assert record["quality"] == 70.0
    assert record["observations"] == 2
    assert record["confidence"] == 0.9
    assert "alex_rbchef::frame_007" in tracker.export()

    restored = FrameQualityTracker(json.loads(json.dumps(tracker.export())))
    assert restored.export() == tracker.export()


def test_frame_quality_tracker_rejects_invalid_measurements():
    tracker = FrameQualityTracker()
    assert tracker.update("a", 1, float("nan"), epoch=1) is None
    assert tracker.update("a", 1, 50, epoch=1, confidence=0) is None
    assert tracker.export() == {}


def test_hard_example_weights_are_bounded_and_require_reliable_observations():
    samples = [
        {"char_id": "a", "frame_idx": 0},
        {"char_id": "a", "frame_idx": 1},
        {"char_id": "a", "frame_idx": 2},
        {"char_id": "a", "frame_idx": 3},
    ]
    records = {
        "a::frame_000": {"quality": 92, "observations": 3, "confidence": 1.0},
        "a::frame_001": {"quality": 65, "observations": 3, "confidence": 1.0},
        "a::frame_002": {"quality": 40, "observations": 3, "confidence": 1.0},
        "a::frame_003": {"quality": 20, "observations": 1, "confidence": 1.0},
    }
    plan = HardExampleMiningPolicy().build_plan(samples, records, enabled=True)

    assert plan.active is True
    assert plan.weights[0] == 1.0
    assert 1.5 <= plan.weights[1] <= 2.0
    assert plan.weights[2] == 2.0
    assert plan.weights[3] == 1.0
    assert max(plan.weights) <= 2.0
    assert min(plan.weights) >= 1.0


def test_sampling_falls_back_to_uniform_until_metrics_are_reliable():
    samples = [{"char_id": "a", "frame_idx": index} for index in range(3)]
    records = {
        f"a::frame_{index:03d}": {"quality": 20, "observations": 1, "confidence": 1.0}
        for index in range(3)
    }
    plan = HardExampleMiningPolicy().build_plan(samples, records, enabled=True)
    assert plan.active is False
    assert plan.reason == "insufficient_reliable_frame_metrics"
    assert plan.weights == (1.0, 1.0, 1.0)


def test_sampling_ab_report_is_deterministic_and_does_not_claim_quality_gain():
    samples = [{"char_id": "a", "frame_idx": index} for index in range(2)]
    records = {
        "a::frame_000": {"quality": 90, "observations": 2, "confidence": 1.0},
        "a::frame_001": {"quality": 30, "observations": 2, "confidence": 1.0},
    }
    plan = HardExampleMiningPolicy().build_plan(samples, records, enabled=True)
    report = compare_sampling_ab(plan)

    assert report["baseline_max_probability"] == 0.5
    assert report["experiment_max_probability"] > 0.5
    assert report["weight_ratio"] <= 2.0
    assert report["quality_improvement_claimed"] is False


def test_frame_quality_persists_inside_backward_compatible_guidance_state():
    tracker = FrameQualityTracker()
    tracker.update("a", 0, 55, epoch=1)
    tracker.update("a", 0, 50, epoch=2)
    controller = QualityGuidanceController(
        state={
            "frame_quality": tracker.export(),
            "sampling_weights": {"a::frame_000": 1.75},
        }
    )
    state = controller.export_state()
    assert state["frame_quality"]["a::frame_000"]["observations"] == 2
    assert state["sampling_weights"]["a::frame_000"] == 1.75

    legacy = QualityGuidanceController(state={}).export_state()
    assert legacy["frame_quality"] == {}
    assert legacy["sampling_weights"] == {}


def test_adaptive_loss_changes_exactly_one_bounded_multiplier():
    controller = LossMultiplierController(step=0.1)
    plan = controller.propose(
        {"color": 1.2, "alpha": 1.0, "edge": 1.0, "adversarial": 1.0},
        {"recommended_action": "ADJUST_WEIGHTS", "primary_problem": "palette"},
        enabled=True,
    )

    assert plan["active"] is True
    assert plan["target"] == "color"
    assert plan["multipliers"]["color"] == 1.25
    assert plan["changed_count"] == 1
    assert 0.75 <= min(plan["multipliers"].values())
    assert max(plan["multipliers"].values()) <= 1.25


def test_adaptive_loss_is_exactly_neutral_when_disabled_or_not_requested():
    controller = LossMultiplierController()
    disabled = controller.propose(
        {"edge": 1.25},
        {"recommended_action": "ADJUST_WEIGHTS", "primary_problem": "outline"},
        enabled=False,
    )
    unrelated = controller.propose(
        {"edge": 1.25},
        {"recommended_action": "REINFORCE", "primary_problem": "face"},
        enabled=True,
    )

    assert set(disabled["multipliers"].values()) == {1.0}
    assert set(unrelated["multipliers"].values()) == {1.0}
    assert disabled["changed_count"] == unrelated["changed_count"] == 0


def test_loss_ab_report_preserves_baseline_and_makes_no_quality_claim():
    plan = LossMultiplierController().propose(
        {},
        {"recommended_action": "ADJUST_WEIGHTS", "primary_problem": "alpha"},
        enabled=True,
    )
    report = compare_loss_ab(plan)

    assert report["changed_count"] == 1
    assert report["baseline_weights"]["alpha"] == 2.5
    assert report["experiment_weights"]["alpha"] == 2.75
    assert report["quality_improvement_claimed"] is False


def test_intervention_policy_requires_sustained_collapse_for_rollback():
    policy = GuidanceInterventionPolicy(cooldown_epochs=5, max_consecutive=3)
    critical_only = policy.decide(
        10,
        {"recommended_action": "ROLLBACK", "trend": {"status": "REGRESSION"}},
    )
    sustained = policy.decide(
        10,
        {"recommended_action": "ROLLBACK", "trend": {"status": "COLLAPSE"}},
    )

    assert critical_only["authorized_action"] == "CONTINUE"
    assert critical_only["reason"] == "rollback_requires_sustained_collapse"
    assert sustained["authorized_action"] == "ROLLBACK"


def test_intervention_policy_enforces_cooldown_and_maximum():
    policy = GuidanceInterventionPolicy(cooldown_epochs=5, max_consecutive=3)
    guidance = {"recommended_action": "REINFORCE", "trend": {"status": "REGRESSION"}}
    first = policy.decide(10, guidance)
    cooldown = policy.decide(12, guidance, first["state"])
    second = policy.decide(15, guidance, first["state"])
    third = policy.decide(20, guidance, second["state"])
    exhausted = policy.decide(25, guidance, third["state"])

    assert first["authorized_action"] == "ADJUST_SAMPLING"
    assert cooldown["reason"] == "cooldown_active"
    assert second["authorized_action"] == third["authorized_action"] == "ADJUST_SAMPLING"
    assert exhausted["authorized_action"] == "CONTINUE"
    assert exhausted["reason"] == "intervention_budget_exhausted"
