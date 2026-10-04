"""
Quality Guidance Controller

Este módulo encapsula la lógica de diagnóstico y decisión de calidad del entrenamiento
supervisado. Mantiene el orquestador principal en `train_supervised.py` y separa la
semántica de análisis visual y la acción correctiva en un componente modular y testeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

QUALITY_DIMENSIONS: Tuple[str, ...] = (
    "global_score",
    "anatomy",
    "silhouette",
    "pose",
    "face",
    "hair",
    "clothing",
    "arms_hands",
    "feet",
    "props",
    "palette",
    "alpha",
    "micro_detail",
    "outline",
    "training_stability",
)


def _environment_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


ENABLE_QUALITY_GUIDANCE = _environment_flag("PIXEL_AI_ENABLE_QUALITY_GUIDANCE", True)
ENABLE_SMART_SAMPLING = _environment_flag("PIXEL_AI_ENABLE_SMART_SAMPLING", True)
ENABLE_ADAPTIVE_LOSS = _environment_flag("PIXEL_AI_ENABLE_ADAPTIVE_LOSS", True)
ENABLE_QUALITY_CHECKPOINT = False


def _finite_number(value: Any) -> Optional[float]:
    """Return a finite float without accepting booleans as quality scores."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_quality_score(value: Any) -> Optional[float]:
    """Normalize a quality score to the inclusive 0..100 range.

    Existing auditors publish percentages. Fractional signals are also accepted for
    future IoU-style producers: values strictly between 0 and 1 are interpreted as
    ratios. Missing and non-finite values remain unavailable instead of becoming 0.
    """
    number = _finite_number(value)
    if number is None:
        return None
    if 0.0 < number < 1.0:
        number *= 100.0
    return round(min(100.0, max(0.0, number)), 4)


def _first_score(metrics: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key in metrics:
            score = normalize_quality_score(metrics.get(key))
            if score is not None:
                return score
    return None


def _mean_available(*values: Optional[float]) -> Optional[float]:
    available = [float(value) for value in values if value is not None]
    if not available:
        return None
    return round(sum(available) / len(available), 4)


@dataclass(frozen=True)
class QualityVector:
    """Unified, JSON-safe view of visual and training quality.

    ``None`` means that a category was not observed. It is deliberately distinct
    from ``0.0``, which is a valid measured collapse.
    """

    global_score: Optional[float] = None
    anatomy: Optional[float] = None
    silhouette: Optional[float] = None
    pose: Optional[float] = None
    face: Optional[float] = None
    hair: Optional[float] = None
    clothing: Optional[float] = None
    arms_hands: Optional[float] = None
    feet: Optional[float] = None
    props: Optional[float] = None
    palette: Optional[float] = None
    alpha: Optional[float] = None
    micro_detail: Optional[float] = None
    outline: Optional[float] = None
    training_stability: Optional[float] = None

    def to_dict(self, *, include_unavailable: bool = True) -> Dict[str, Optional[float]]:
        result: Dict[str, Optional[float]] = {
            "global": self.global_score,
            **{
                item.name: getattr(self, item.name)
                for item in fields(self)
                if item.name != "global_score"
            },
        }
        if include_unavailable:
            return result
        return {key: value for key, value in result.items() if value is not None}

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "QualityVector":
        source = payload if isinstance(payload, Mapping) else {}
        kwargs = {
            "global_score": normalize_quality_score(source.get("global", source.get("global_score"))),
        }
        for name in QUALITY_DIMENSIONS:
            if name == "global_score":
                continue
            kwargs[name] = normalize_quality_score(source.get(name))
        return cls(**kwargs)


def frame_quality_key(char_id: Any, frame_idx: Any) -> str:
    """Stable identifier shared by cache samples, metrics and sampler weights."""
    try:
        normalized_index = int(frame_idx)
    except (TypeError, ValueError):
        normalized_index = -1
    return f"{str(char_id)}::frame_{normalized_index:03d}"


class FrameQualityTracker:
    """Accumulate trustworthy per-frame metrics using a bounded moving average."""

    def __init__(
        self,
        records: Optional[Mapping[str, Any]] = None,
        *,
        ema_decay: float = 0.7,
        max_records: int = 10000,
    ):
        self.ema_decay = min(0.95, max(0.0, float(ema_decay)))
        self.max_records = max(1, int(max_records))
        self.records: Dict[str, Dict[str, Any]] = {}
        if isinstance(records, Mapping):
            for key, value in records.items():
                if isinstance(value, Mapping):
                    score = normalize_quality_score(value.get("quality"))
                    observations = _finite_number(value.get("observations"))
                    if score is None or observations is None or observations < 1:
                        continue
                    raw_confidence = _finite_number(value.get("confidence", 1.0))
                    confidence = 1.0 if raw_confidence is None else raw_confidence
                    if confidence > 1.0:
                        confidence /= 100.0
                    key_text = str(key)
                    key_char, _, key_frame = key_text.rpartition("::frame_")
                    stored_frame = _finite_number(value.get("frame_idx"))
                    if stored_frame is None:
                        stored_frame = _finite_number(key_frame)
                    self.records[str(key)] = {
                        "char_id": str(value.get("char_id", key_char or "unknown")),
                        "frame_idx": int(stored_frame) if stored_frame is not None else -1,
                        "quality": score,
                        "observations": int(observations),
                        "confidence": round(min(1.0, max(0.0, confidence)), 4),
                        "last_epoch": int(_finite_number(value.get("last_epoch")) or 0),
                        "metrics": dict(value.get("metrics", {})) if isinstance(value.get("metrics"), Mapping) else {},
                    }

    def update(
        self,
        char_id: Any,
        frame_idx: Any,
        quality: Any,
        *,
        epoch: int,
        metrics: Optional[Mapping[str, Any]] = None,
        confidence: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        score = normalize_quality_score(quality)
        finite_confidence = _finite_number(confidence)
        if score is None or finite_confidence is None or finite_confidence <= 0.0:
            return None
        key = frame_quality_key(char_id, frame_idx)
        previous = self.records.get(key)
        observations = int(previous.get("observations", 0)) + 1 if previous else 1
        if previous:
            score = round(
                float(previous["quality"]) * self.ema_decay + score * (1.0 - self.ema_decay),
                4,
            )
        confidence_value = min(1.0, max(0.0, float(finite_confidence)))
        record = {
            "char_id": str(char_id),
            "frame_idx": int(frame_idx),
            "quality": score,
            "observations": observations,
            "confidence": round(confidence_value, 4),
            "last_epoch": int(epoch),
            "metrics": {
                key: round(value, 4)
                for key, raw in (metrics or {}).items()
                if (value := _finite_number(raw)) is not None
            },
        }
        self.records[key] = record
        if len(self.records) > self.max_records:
            oldest = sorted(
                self.records,
                key=lambda item: (self.records[item].get("last_epoch", 0), item),
            )[: len(self.records) - self.max_records]
            for old_key in oldest:
                self.records.pop(old_key, None)
        return dict(record)

    def update_many(self, measurements: Iterable[Mapping[str, Any]], *, epoch: int) -> None:
        for measurement in measurements:
            if not isinstance(measurement, Mapping):
                continue
            self.update(
                measurement.get("char_id", "unknown"),
                measurement.get("frame_idx", -1),
                measurement.get("quality"),
                epoch=epoch,
                metrics=measurement.get("metrics") if isinstance(measurement.get("metrics"), Mapping) else None,
                confidence=float(_finite_number(measurement.get("confidence")) or 1.0),
            )

    def export(self) -> Dict[str, Dict[str, Any]]:
        return {key: dict(value) for key, value in sorted(self.records.items())}


@dataclass(frozen=True)
class SamplingPlan:
    active: bool
    reason: str
    weights: Tuple[float, ...]
    sample_keys: Tuple[str, ...]
    eligible_frames: int
    hard_frames: int
    min_weight: float
    max_weight: float
    effective_sample_size: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "reason": self.reason,
            "weights": list(self.weights),
            "sample_keys": list(self.sample_keys),
            "eligible_frames": self.eligible_frames,
            "hard_frames": self.hard_frames,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "effective_sample_size": self.effective_sample_size,
        }


class HardExampleMiningPolicy:
    """Convert reliable frame scores into conservative 1.0..2.0 weights."""

    def __init__(
        self,
        *,
        min_observations: int = 2,
        min_confidence: float = 0.75,
        max_weight: float = 2.0,
        hard_threshold: float = 75.0,
        very_hard_threshold: float = 50.0,
    ):
        self.min_observations = max(1, int(min_observations))
        self.min_confidence = min(1.0, max(0.0, float(min_confidence)))
        self.max_weight = min(2.0, max(1.0, float(max_weight)))
        self.hard_threshold = normalize_quality_score(hard_threshold) or 75.0
        self.very_hard_threshold = normalize_quality_score(very_hard_threshold) or 50.0

    def _weight_for(self, record: Optional[Mapping[str, Any]]) -> Tuple[float, bool]:
        if not isinstance(record, Mapping):
            return 1.0, False
        score = normalize_quality_score(record.get("quality"))
        observations = _finite_number(record.get("observations"))
        confidence = _finite_number(record.get("confidence"))
        if (
            score is None
            or observations is None
            or observations < self.min_observations
            or confidence is None
            or confidence < self.min_confidence
        ):
            return 1.0, False
        if score <= self.very_hard_threshold:
            return self.max_weight, True
        if score <= self.hard_threshold:
            severity = (self.hard_threshold - score) / max(1.0, self.hard_threshold - self.very_hard_threshold)
            weight = min(self.max_weight, 1.5 + severity * (self.max_weight - 1.5))
            return round(weight, 4), True
        return 1.0, True

    def build_plan(
        self,
        samples: Iterable[Mapping[str, Any]],
        frame_quality: Optional[Mapping[str, Any]],
        *,
        enabled: bool,
        recommendation: str = "REINFORCE",
    ) -> SamplingPlan:
        records = frame_quality if isinstance(frame_quality, Mapping) else {}
        keys: List[str] = []
        weights: List[float] = []
        eligible = 0
        hard = 0
        for sample in samples:
            key = frame_quality_key(sample.get("char_id", "unknown"), sample.get("frame_idx", -1))
            weight, is_eligible = self._weight_for(records.get(key))
            keys.append(key)
            weights.append(weight)
            eligible += int(is_eligible)
            hard += int(is_eligible and weight > 1.0)

        intervention_allowed = recommendation in {"REINFORCE", "ADJUST_SAMPLING"}
        active = bool(enabled and intervention_allowed and eligible > 0 and hard > 0)
        if not enabled:
            reason = "feature_disabled"
        elif not intervention_allowed:
            reason = "guidance_did_not_request_sampling"
        elif eligible == 0:
            reason = "insufficient_reliable_frame_metrics"
        elif hard == 0:
            reason = "no_hard_frames_detected"
        else:
            reason = "hard_example_sampling_active"
        if not active:
            weights = [1.0 for _ in weights]

        total = sum(weights)
        squared = sum(weight * weight for weight in weights)
        effective_size = (total * total / squared) if squared > 0.0 else 0.0
        return SamplingPlan(
            active=active,
            reason=reason,
            weights=tuple(weights),
            sample_keys=tuple(keys),
            eligible_frames=eligible,
            hard_frames=hard,
            min_weight=round(min(weights), 4) if weights else 1.0,
            max_weight=round(max(weights), 4) if weights else 1.0,
            effective_sample_size=round(effective_size, 4),
        )


def compare_sampling_ab(plan: SamplingPlan) -> Dict[str, Any]:
    """Deterministic distribution comparison; it does not claim model improvement."""
    count = len(plan.weights)
    if count == 0:
        return {"sample_count": 0, "baseline_max_probability": 0.0, "experiment_max_probability": 0.0}
    total = sum(plan.weights)
    baseline_probability = 1.0 / count
    experiment_probabilities = [weight / total for weight in plan.weights]
    return {
        "sample_count": count,
        "active": plan.active,
        "baseline_max_probability": round(baseline_probability, 8),
        "experiment_max_probability": round(max(experiment_probabilities), 8),
        "effective_sample_size": plan.effective_sample_size,
        "hard_frames": plan.hard_frames,
        "weight_ratio": round(plan.max_weight / max(plan.min_weight, 1e-8), 4),
        "quality_improvement_claimed": False,
    }


BASE_LOSS_WEIGHTS: Dict[str, float] = {
    "color": 5.0,
    "alpha": 2.5,
    "edge": 1.5,
    "adversarial": 0.05,
}


class LossMultiplierController:
    """Select one bounded, attributable loss correction for the next epoch."""

    PROBLEM_TO_LOSS = {
        "palette": "color",
        "alpha": "alpha",
        "micro_detail": "edge",
        "outline": "edge",
    }

    def __init__(self, *, step: float = 0.1, minimum: float = 0.75, maximum: float = 1.25):
        self.step = min(0.25, max(0.01, float(step)))
        self.minimum = min(1.0, max(0.5, float(minimum)))
        self.maximum = max(1.0, min(1.5, float(maximum)))

    @staticmethod
    def neutral() -> Dict[str, float]:
        return {key: 1.0 for key in BASE_LOSS_WEIGHTS}

    def propose(
        self,
        current: Optional[Mapping[str, Any]],
        guidance: Optional[Mapping[str, Any]],
        *,
        enabled: bool,
    ) -> Dict[str, Any]:
        multipliers = self.neutral()
        recommendation = str((guidance or {}).get("recommended_action", "CONTINUE"))
        problem = str((guidance or {}).get("primary_problem", ""))
        target = self.PROBLEM_TO_LOSS.get(problem)
        active = bool(enabled and recommendation == "ADJUST_WEIGHTS" and target)
        if active and target is not None:
            previous = _finite_number((current or {}).get(target)) or 1.0
            multipliers[target] = round(min(self.maximum, max(self.minimum, previous + self.step)), 4)
        reason = "adaptive_loss_active" if active else (
            "feature_disabled" if not enabled else "guidance_did_not_request_loss_adjustment"
        )
        return {
            "active": active,
            "reason": reason,
            "target": target if active else None,
            "multipliers": multipliers,
            "effective_weights": {
                key: round(BASE_LOSS_WEIGHTS[key] * multipliers[key], 6)
                for key in BASE_LOSS_WEIGHTS
            },
            "changed_count": sum(value != 1.0 for value in multipliers.values()),
        }


def compare_loss_ab(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Describe baseline/effective loss weights without claiming model improvement."""
    effective = plan.get("effective_weights", {}) if isinstance(plan, Mapping) else {}
    return {
        "baseline_weights": dict(BASE_LOSS_WEIGHTS),
        "experiment_weights": {
            key: float(effective.get(key, value)) for key, value in BASE_LOSS_WEIGHTS.items()
        },
        "changed_count": int(plan.get("changed_count", 0)) if isinstance(plan, Mapping) else 0,
        "quality_improvement_claimed": False,
    }


def _training_stability(training_metrics: Mapping[str, Any]) -> Optional[float]:
    watched = ("g_loss", "d_loss", "l1_loss", "edge_loss", "lr")
    present = [key for key in watched if key in training_metrics]
    if not present and "skipped_amp_steps" not in training_metrics:
        return None

    invalid_count = sum(_finite_number(training_metrics.get(key)) is None for key in present)
    skipped = _finite_number(training_metrics.get("skipped_amp_steps")) or 0.0
    penalty = invalid_count * 25.0 + min(50.0, max(0.0, skipped) * 2.0)
    return round(max(0.0, 100.0 - penalty), 4)


def build_quality_vector(
    metrics: Optional[Mapping[str, Any]],
    training_metrics: Optional[Mapping[str, Any]] = None,
) -> QualityVector:
    """Build a unified vector from QualityGate, Enhancer and training payloads."""
    visual: Mapping[str, Any] = metrics if isinstance(metrics, Mapping) else {}
    training: Mapping[str, Any] = training_metrics if isinstance(training_metrics, Mapping) else {}

    nested_guide = visual.get("quality_guide")
    guide: Mapping[str, Any] = nested_guide if isinstance(nested_guide, Mapping) else {}

    anatomy = _first_score(visual, "cuerpo_precision", "anatomy", "anatomia")
    silhouette = _first_score(visual, "silueta_iou_real", "alineacion_molde", "silhouette")
    pose = _first_score(visual, "pose_alignment", "pose", "alineacion_molde")
    face = _first_score(visual, "score_gestos_ojos", "gestos_ojos", "detalle_facial", "face")
    if face is None:
        face = _first_score(guide, "face_score")
    hair = _first_score(visual, "score_pelo_gorro", "pelo_gorro", "hair")
    clothing = _first_score(visual, "score_ropa_delantal", "ropa_delantal", "clothing")
    if clothing is None:
        clothing = _first_score(guide, "clothes_score")
    arms_hands = _first_score(
        visual,
        "score_tatuajes_brazos",
        "tatuajes_brazos",
        "preservacion_tatuajes",
        "arms_hands",
    )
    feet = _first_score(visual, "score_zapatos_pies", "zapatos_pies", "feet")
    props = _first_score(visual, "score_objetos_utensilios", "objetos_utensilios", "props")
    if props is None:
        props = _first_score(guide, "accessory_score")
    palette = _mean_available(
        _first_score(visual, "fidelidad_paleta", "palette"),
        _first_score(visual, "riqueza_paleta"),
    )
    alpha = _first_score(visual, "pureza_alfa", "alpha")
    micro_detail = _mean_available(
        _first_score(visual, "micro_textura", "micro_detail"),
        _first_score(visual, "micro_detalles"),
    )
    outline = _mean_available(
        _first_score(visual, "nitidez_bordes", "outline"),
        _first_score(visual, "pureza_bordes"),
        _first_score(visual, "definicion_tinta"),
    )

    training_stability = _first_score(visual, "training_stability")
    if training_stability is None:
        training_stability = _training_stability(training)

    global_score = _first_score(visual, "score_total", "score_critico", "global", "global_score")
    if global_score is None:
        global_score = _first_score(guide, "guide_score")
    if global_score is None:
        global_score = _mean_available(
            anatomy,
            silhouette,
            pose,
            face,
            hair,
            clothing,
            arms_hands,
            feet,
            props,
            palette,
            alpha,
            micro_detail,
            outline,
        )

    return QualityVector(
        global_score=global_score,
        anatomy=anatomy,
        silhouette=silhouette,
        pose=pose,
        face=face,
        hair=hair,
        clothing=clothing,
        arms_hands=arms_hands,
        feet=feet,
        props=props,
        palette=palette,
        alpha=alpha,
        micro_detail=micro_detail,
        outline=outline,
        training_stability=training_stability,
    )


DEFAULT_TARGETS: Dict[str, float] = {
    "global": 80.0,
    "anatomy": 75.0,
    "silhouette": 75.0,
    "pose": 75.0,
    "face": 70.0,
    "hair": 75.0,
    "clothing": 80.0,
    "arms_hands": 70.0,
    "feet": 80.0,
    "props": 70.0,
    "palette": 85.0,
    "alpha": 95.0,
    "micro_detail": 70.0,
    "outline": 75.0,
    "training_stability": 90.0,
}

DEFAULT_CRITICAL_FLOORS: Dict[str, float] = {
    "global": 35.0,
    "anatomy": 45.0,
    "silhouette": 45.0,
    "face": 40.0,
    "palette": 50.0,
    "alpha": 75.0,
    "training_stability": 50.0,
}

GUIDANCE_ACTIONS: Tuple[str, ...] = (
    "CONTINUE",
    "REINFORCE",
    "ADJUST_WEIGHTS",
    "ADJUST_SAMPLING",
    "REDUCE_LR",
    "FREEZE",
    "ROLLBACK",
    "STOP",
)


@dataclass
class QualityGuidanceConfig:
    enabled: bool = ENABLE_QUALITY_GUIDANCE
    mode: str = "observational"
    targets: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TARGETS))
    critical_floors: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CRITICAL_FLOORS))
    baseline_points: int = 5
    baseline_margin: float = 5.0
    trend_window: int = 5
    min_trend_points: int = 3
    improvement_delta: float = 3.0
    regression_delta: float = 3.0
    collapse_delta: float = 15.0
    plateau_range: float = 2.0
    max_history: int = 200

    def __post_init__(self) -> None:
        # Increment 1 is intentionally incapable of applying interventions.
        self.mode = "observational"
        self.baseline_points = max(2, int(self.baseline_points))
        self.trend_window = max(3, int(self.trend_window))
        self.min_trend_points = max(3, int(self.min_trend_points))
        self.max_history = max(10, int(self.max_history))
        self.targets = _normalized_thresholds(self.targets, DEFAULT_TARGETS)
        self.critical_floors = _normalized_thresholds(self.critical_floors, DEFAULT_CRITICAL_FLOORS)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "targets": dict(self.targets),
            "critical_floors": dict(self.critical_floors),
            "baseline_points": self.baseline_points,
            "baseline_margin": self.baseline_margin,
            "trend_window": self.trend_window,
            "min_trend_points": self.min_trend_points,
            "improvement_delta": self.improvement_delta,
            "regression_delta": self.regression_delta,
            "collapse_delta": self.collapse_delta,
            "plateau_range": self.plateau_range,
            "max_history": self.max_history,
        }

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "QualityGuidanceConfig":
        data = dict(payload) if isinstance(payload, Mapping) else {}
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})


def _normalized_thresholds(
    values: Optional[Mapping[str, Any]], defaults: Mapping[str, float]
) -> Dict[str, float]:
    result = dict(defaults)
    if isinstance(values, Mapping):
        for key, raw_value in values.items():
            score = normalize_quality_score(raw_value)
            if score is not None:
                result[str(key)] = score
    return result


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _coerce_quality_vector(value: Any) -> Optional[QualityVector]:
    if isinstance(value, QualityVector):
        return value
    if not isinstance(value, Mapping):
        return None
    if isinstance(value.get("quality_vector"), Mapping):
        return QualityVector.from_dict(value["quality_vector"])
    guidance = value.get("guidance", value.get("quality_guidance"))
    if isinstance(guidance, Mapping) and isinstance(guidance.get("quality_vector"), Mapping):
        return QualityVector.from_dict(guidance["quality_vector"])
    if isinstance(value.get("quality"), Mapping):
        return build_quality_vector(value["quality"])
    canonical_keys = {"global", "global_score", *QUALITY_DIMENSIONS}
    if canonical_keys.intersection(value.keys()):
        return QualityVector.from_dict(value)
    return build_quality_vector(value)


class QualityTrendAnalyzer:
    """Classify multi-audit trajectories without acting on training."""

    def __init__(self, config: Optional[QualityGuidanceConfig] = None):
        self.config = config or QualityGuidanceConfig()

    def _classify(self, values: List[float]) -> str:
        if len(values) < self.config.min_trend_points:
            return "INSUFFICIENT_DATA"
        recent = values[-self.config.trend_window :]
        delta = recent[-1] - recent[0]
        value_range = max(recent) - min(recent)
        changes = [right - left for left, right in zip(recent, recent[1:])]
        signs = [1 if change > 0.5 else -1 if change < -0.5 else 0 for change in changes]
        nonzero = [sign for sign in signs if sign]
        sign_changes = sum(left != right for left, right in zip(nonzero, nonzero[1:]))

        if recent[-1] <= 20.0 or delta <= -self.config.collapse_delta:
            return "COLLAPSE"
        if sign_changes >= 2 and value_range >= max(4.0, self.config.plateau_range * 2.0):
            return "OSCILLATION"
        if delta <= -self.config.regression_delta:
            return "REGRESSION"
        if delta >= self.config.improvement_delta:
            return "IMPROVING"
        if value_range <= self.config.plateau_range:
            return "PLATEAU"
        return "STABLE"

    def analyze(self, history: Iterable[Any]) -> Dict[str, Any]:
        vectors = [vector for item in history if (vector := _coerce_quality_vector(item)) is not None]
        categories = ("global",) + tuple(name for name in QUALITY_DIMENSIONS if name != "global_score")
        by_category: Dict[str, str] = {}
        deltas: Dict[str, Optional[float]] = {}
        counts: Dict[str, int] = {}

        for category in categories:
            attribute = "global_score" if category == "global" else category
            values = [getattr(vector, attribute) for vector in vectors]
            observed = [float(value) for value in values if value is not None]
            by_category[category] = self._classify(observed)
            counts[category] = len(observed)
            deltas[category] = round(observed[-1] - observed[0], 4) if len(observed) >= 2 else None

        meaningful = [status for status in by_category.values() if status != "INSUFFICIENT_DATA"]
        priority = ("COLLAPSE", "REGRESSION", "OSCILLATION", "PLATEAU", "IMPROVING", "STABLE")
        overall = next((status for status in priority if status in meaningful), "INSUFFICIENT_DATA")
        return {
            "status": overall,
            "by_category": by_category,
            "deltas": deltas,
            "observations": counts,
            "regression_categories": [
                category
                for category, status in by_category.items()
                if status in {"REGRESSION", "COLLAPSE"}
            ],
            "window": self.config.trend_window,
        }


def _severity_from_deficit(deficit: float) -> str:
    if deficit >= 25.0:
        return "critical"
    if deficit >= 15.0:
        return "high"
    if deficit >= 5.0:
        return "medium"
    return "low"


def diagnose_quality_bottleneck(
    quality: Any,
    *,
    thresholds: Optional[Mapping[str, Any]] = None,
    critical_floors: Optional[Mapping[str, Any]] = None,
    trend: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Find the worst observed category without treating unavailable data as zero."""
    vector = _coerce_quality_vector(quality) or QualityVector()
    target_map = _normalized_thresholds(thresholds, DEFAULT_TARGETS)
    floor_map = _normalized_thresholds(critical_floors, DEFAULT_CRITICAL_FLOORS)
    payload = vector.to_dict()
    deficits: Dict[str, float] = {}
    observed = 0

    for category, target in target_map.items():
        value = payload.get(category)
        if value is None:
            continue
        observed += 1
        deficit = round(float(target) - float(value), 4)
        if deficit > 0.0:
            deficits[category] = deficit

    ranked = sorted(deficits, key=lambda name: (-deficits[name], name))
    primary = ranked[0] if ranked else None
    secondary = ranked[1:3]
    severity = _severity_from_deficit(deficits.get(primary, 0.0)) if primary else "low"

    critical_breaches = [
        category
        for category, floor in floor_map.items()
        if payload.get(category) is not None and float(payload[category]) < float(floor)
    ]
    trend_status = str((trend or {}).get("status", "INSUFFICIENT_DATA"))
    if critical_breaches or trend_status == "COLLAPSE":
        severity = "critical"
        if primary is None and critical_breaches:
            primary = critical_breaches[0]
    elif trend_status == "REGRESSION":
        severity = {"low": "medium", "medium": "high", "high": "critical", "critical": "critical"}[severity]

    coverage = observed / max(1, len(target_map))
    separation = 1.0
    if len(ranked) > 1 and deficits[ranked[0]] > 0:
        separation = max(0.0, min(1.0, (deficits[ranked[0]] - deficits[ranked[1]]) / deficits[ranked[0]]))
    confidence = round(min(0.99, 0.45 + coverage * 0.4 + separation * 0.14), 2) if primary else round(min(0.95, 0.4 + coverage * 0.5), 2)

    return {
        "primary_problem": primary,
        "secondary_problems": secondary,
        "severity": severity,
        "confidence": confidence,
        "deficits": deficits,
        "critical_breaches": critical_breaches,
        "observed_categories": observed,
    }


class QualityGuidanceController:
    """Produce explainable recommendations while preserving observational safety."""

    STATE_VERSION = 1
    SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}

    def __init__(
        self,
        config: Optional[QualityGuidanceConfig] = None,
        target_assimilation: Optional[float] = None,
        max_reinforcement_rounds: int = 3,
        state: Optional[Mapping[str, Any]] = None,
    ):
        self.config = config or QualityGuidanceConfig()
        if target_assimilation is not None:
            target = normalize_quality_score(target_assimilation)
            if target is not None:
                self.config.targets["global"] = target
        self.target_assimilation = self.config.targets["global"]
        self.max_reinforcement_rounds = max(0, int(max_reinforcement_rounds))
        self.quality_history: List[Dict[str, Any]] = []
        self.decision_history: List[Dict[str, Any]] = []
        self.frame_quality: Dict[str, Dict[str, Any]] = {}
        self.sampling_weights: Dict[str, float] = {}
        self.sampling_plan: Dict[str, Any] = {}
        self.loss_multipliers: Dict[str, float] = {
            "color": 1.0,
            "alpha": 1.0,
            "edge": 1.0,
            "adversarial": 1.0,
        }
        self.intervention_count = 0
        self.history = self.decision_history  # Backward-compatible public alias.
        if state:
            self.load_state(state)

    def normalize_metrics(self, metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
        return build_quality_vector(metrics).to_dict(include_unavailable=False)  # type: ignore[return-value]

    def _effective_thresholds(self) -> Dict[str, float]:
        thresholds = dict(self.config.targets)
        baseline = self.quality_history[: self.config.baseline_points]
        if len(baseline) < self.config.baseline_points:
            return thresholds
        for category, configured_target in list(thresholds.items()):
            values = [
                item.get("quality_vector", {}).get(category)
                for item in baseline
                if isinstance(item.get("quality_vector"), Mapping)
            ]
            observed = [float(value) for value in values if _finite_number(value) is not None]
            if len(observed) < self.config.baseline_points:
                continue
            relative_target = _median(observed) - self.config.baseline_margin
            floor = self.config.critical_floors.get(category, 0.0)
            thresholds[category] = round(max(floor, min(configured_target, relative_target)), 4)
        return thresholds

    def _recommend_action(self, diagnosis: Mapping[str, Any], trend: Mapping[str, Any]) -> str:
        problem = diagnosis.get("primary_problem")
        severity = diagnosis.get("severity", "low")
        if not problem:
            return "CONTINUE"
        if trend.get("status") == "COLLAPSE" or severity == "critical":
            return "ROLLBACK"
        if problem == "training_stability":
            return "REDUCE_LR" if severity in {"medium", "high"} else "CONTINUE"
        if problem in {"palette", "alpha", "micro_detail", "outline"} and severity == "high":
            return "ADJUST_WEIGHTS"
        if severity in {"medium", "high"} or trend.get("status") == "REGRESSION":
            return "REINFORCE"
        return "CONTINUE"

    def _recommendation_text(self, diagnosis: Mapping[str, Any], recommendation: str) -> str:
        problem = diagnosis.get("primary_problem")
        if problem is None:
            return "Las señales observadas no muestran un cuello de botella dominante; continuar y acumular tendencia."
        if recommendation == "ROLLBACK":
            return f"Se recomienda revisar el último snapshot saludable por degradación crítica en {problem}."
        if recommendation == "ADJUST_WEIGHTS":
            return f"Se recomienda evaluar un ajuste acotado y atribuible de loss para {problem} en un incremento futuro."
        if recommendation == "REDUCE_LR":
            return "Se recomienda revisar estabilidad y una reducción acotada de LR en el incremento de Recovery."
        if recommendation == "REINFORCE":
            return f"Se recomienda refuerzo dirigido a {problem}, sujeto a feature flag y validación A/B futura."
        return f"Mantener observación de {problem}; el déficit actual no justifica una intervención."

    def evaluate(
        self,
        metrics: Optional[Dict[str, Any]],
        trend: Optional[Dict[str, Any]] = None,
        *,
        training_metrics: Optional[Mapping[str, Any]] = None,
        history: Optional[Iterable[Any]] = None,
        epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        vector = build_quality_vector(metrics, training_metrics)
        historical = list(history) if history is not None else list(self.quality_history)
        trend_report = dict(trend) if isinstance(trend, Mapping) else QualityTrendAnalyzer(self.config).analyze([*historical, vector])
        thresholds = self._effective_thresholds()
        diagnosis = diagnose_quality_bottleneck(
            vector,
            thresholds=thresholds,
            critical_floors=self.config.critical_floors,
            trend=trend_report,
        )
        recommended_action = self._recommend_action(diagnosis, trend_report)
        quality_payload = vector.to_dict()
        decision = {
            "enabled": bool(self.config.enabled),
            "mode": self.config.mode,
            "action": "CONTINUE",
            "recommended_action": recommended_action,
            "problem": diagnosis["primary_problem"],
            "primary_problem": diagnosis["primary_problem"],
            "secondary_problems": diagnosis["secondary_problems"],
            "severity": diagnosis["severity"],
            "confidence": diagnosis["confidence"],
            "quality_score": quality_payload.get("global"),
            "quality_vector": quality_payload,
            "trend": trend_report,
            "diagnosis": diagnosis,
            "effective_thresholds": thresholds,
            "recommendation": self._recommendation_text(diagnosis, recommended_action),
            "training_modified": False,
            "reinforcement_rounds_remaining": self.max_reinforcement_rounds,
        }
        if epoch is not None:
            decision["epoch"] = int(epoch)

        record = {"epoch": int(epoch) if epoch is not None else None, "quality_vector": quality_payload}
        if epoch is not None:
            self.quality_history = [item for item in self.quality_history if item.get("epoch") != int(epoch)]
            self.decision_history = [item for item in self.decision_history if item.get("epoch") != int(epoch)]
            self.history = self.decision_history
        self.quality_history.append(record)
        self.decision_history.append(dict(decision))
        self.quality_history = self.quality_history[-self.config.max_history :]
        self.decision_history = self.decision_history[-self.config.max_history :]
        self.history = self.decision_history
        return decision

    def decide_action(self, metrics: Optional[Dict[str, Any]], trend: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.evaluate(metrics=metrics, trend=trend)

    def build_recommendation(self, metrics: Optional[Dict[str, Any]], trend: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.evaluate(metrics=metrics, trend=trend)

    def summarize(self, metrics: Optional[Dict[str, Any]], trend: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.evaluate(metrics=metrics, trend=trend)

    def export_state(self) -> Dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "config": self.config.to_dict(),
            "last_action": self.decision_history[-1].get("action") if self.decision_history else "CONTINUE",
            "last_problem": self.decision_history[-1].get("primary_problem") if self.decision_history else None,
            "intervention_count": self.intervention_count,
            "loss_multipliers": dict(self.loss_multipliers),
            "sampling_weights": dict(self.sampling_weights),
            "sampling_plan": dict(self.sampling_plan),
            "frame_quality": {key: dict(value) for key, value in self.frame_quality.items()},
            "quality_history": list(self.quality_history),
            "decision_history": list(self.decision_history),
        }

    def load_state(self, state: Optional[Mapping[str, Any]]) -> None:
        if not isinstance(state, Mapping):
            return
        if isinstance(state.get("config"), Mapping):
            try:
                self.config = QualityGuidanceConfig.from_dict(state["config"])
                self.target_assimilation = self.config.targets["global"]
            except (TypeError, ValueError):
                # Corrupt/legacy configuration must not prevent checkpoint resume.
                pass
        quality_history = state.get("quality_history", [])
        decision_history = state.get("decision_history", [])
        self.quality_history = [dict(item) for item in quality_history if isinstance(item, Mapping)][-self.config.max_history :]
        self.decision_history = [dict(item) for item in decision_history if isinstance(item, Mapping)][-self.config.max_history :]
        frame_quality = state.get("frame_quality", {})
        self.frame_quality = FrameQualityTracker(frame_quality).export() if isinstance(frame_quality, Mapping) else {}
        sampling_weights = state.get("sampling_weights", {})
        self.sampling_weights = {
            str(key): min(2.0, max(1.0, float(value)))
            for key, raw in sampling_weights.items()
            if (value := _finite_number(raw)) is not None
        } if isinstance(sampling_weights, Mapping) else {}
        self.sampling_plan = dict(state.get("sampling_plan", {})) if isinstance(state.get("sampling_plan"), Mapping) else {}
        loss_multipliers = state.get("loss_multipliers", {})
        if isinstance(loss_multipliers, Mapping):
            for key in self.loss_multipliers:
                value = _finite_number(loss_multipliers.get(key))
                if value is not None:
                    self.loss_multipliers[key] = min(1.25, max(0.75, float(value)))
        self.intervention_count = max(0, int(_finite_number(state.get("intervention_count")) or 0))
        self.history = self.decision_history


__all__ = [
    "BASE_LOSS_WEIGHTS",
    "DEFAULT_CRITICAL_FLOORS",
    "DEFAULT_TARGETS",
    "GUIDANCE_ACTIONS",
    "FrameQualityTracker",
    "HardExampleMiningPolicy",
    "LossMultiplierController",
    "ENABLE_ADAPTIVE_LOSS",
    "ENABLE_QUALITY_CHECKPOINT",
    "ENABLE_QUALITY_GUIDANCE",
    "ENABLE_SMART_SAMPLING",
    "QUALITY_DIMENSIONS",
    "QualityGuidanceConfig",
    "QualityGuidanceController",
    "QualityTrendAnalyzer",
    "QualityVector",
    "SamplingPlan",
    "build_quality_vector",
    "diagnose_quality_bottleneck",
    "compare_sampling_ab",
    "compare_loss_ab",
    "frame_quality_key",
    "normalize_quality_score",
]
