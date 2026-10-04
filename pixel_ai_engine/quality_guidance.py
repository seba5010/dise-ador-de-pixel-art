"""
Quality Guidance Controller

Este módulo encapsula la lógica de diagnóstico y decisión de calidad del entrenamiento
supervisado. Mantiene el orquestador principal en `train_supervised.py` y separa la
semántica de análisis visual y la acción correctiva en un componente modular y testeable.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .enhancer import PixelArtEnhancer


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


class QualityGuidanceController:
    """Diagnostica fallas visuales y decide la siguiente acción segura."""

    SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}

    def __init__(self, target_assimilation: float = 99.5, max_reinforcement_rounds: int = 3):
        self.target_assimilation = float(target_assimilation)
        self.max_reinforcement_rounds = int(max_reinforcement_rounds)
        self.history: List[Dict[str, Any]] = []

    def normalize_metrics(self, metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
        if not isinstance(metrics, dict):
            return {}

        normalized: Dict[str, float] = {}
        for key in [
            "score_total",
            "fidelidad_paleta",
            "alineacion_molde",
            "micro_textura",
            "pureza_alfa",
            "cuerpo_precision",
            "defectos_cuerpo",
            "total_px_cuerpo",
            "colores_ia",
            "colores_original",
            "silueta_iou_real",
            "preservacion_tatuajes",
            "micro_detalles",
            "pelo_gorro",
            "gestos_ojos",
            "ropa_delantal",
            "tatuajes_brazos",
            "objetos_utensilios",
            "zapatos_pies",
        ]:
            value = metrics.get(key)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number != number or number in (float("inf"), float("-inf")):
                continue
            normalized[key] = number

        if "score_total" not in normalized and normalized:
            normalized["score_total"] = sum(normalized.values()) / len(normalized)

        return normalized

    def _metric_value(self, metrics: Dict[str, float], *keys: str, default: float = 0.0) -> float:
        for key in keys:
            value = metrics.get(key)
            if value is not None:
                return float(value)
        return float(default)

    def _compute_root_causes(self, metrics: Dict[str, float]) -> List[str]:
        causes: List[str] = []
        if self._metric_value(metrics, "fidelidad_paleta", "colores_ia", "colores_original") < 85.0:
            causes.append("paleta desalineada")
        if self._metric_value(metrics, "cuerpo_precision", "silueta_iou_real") < 85.0:
            causes.append("anatomía y silueta inestables")
        if self._metric_value(metrics, "pureza_alfa", "micro_textura") < 85.0:
            causes.append("alfa o micro-textura con artefactos")
        if self._metric_value(metrics, "gestos_ojos", "pelo_gorro") < 85.0:
            causes.append("detalle facial o capilar frágil")
        if self._metric_value(metrics, "ropa_delantal", "objetos_utensilios", "zapatos_pies") < 85.0:
            causes.append("ropa, accesorios o pies desalineados")
        if not causes:
            causes.append("sin causa principal detectada")
        return causes

    def _severity_for(self, score_total: float, root_causes: List[str]) -> str:
        if score_total < 50.0 or "paleta desalineada" in root_causes and score_total < 60.0:
            return "critical"
        if score_total < 75.0 or len(root_causes) >= 3:
            return "high"
        if score_total < 90.0:
            return "medium"
        return "low"

    def _action_for(self, score_total: float, root_causes: List[str], severity: str) -> str:
        if score_total < 50.0:
            return "rollback_to_last_healthy_snapshot"
        if "paleta desalineada" in root_causes:
            return "recalibrate_palette_and_reinforce"
        if "anatomía y silueta inestables" in root_causes:
            return "reinforce_anatomy_with_targeted_preview"
        if "alfa o micro-textura con artefactos" in root_causes:
            return "clean_alpha_and_micro_details"
        if severity in {"high", "medium"}:
            return "continue_with_guarded_reinforcement"
        return "continue_training"

    def build_quality_guide(self, metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        guide = PixelArtEnhancer.build_quality_guide(metrics or {})
        if not isinstance(guide, dict):
            return {
                "guide_score": 0.0,
                "body_score": 0.0,
                "face_score": 0.0,
                "clothes_score": 0.0,
                "accessory_score": 0.0,
                "dominant_signal": "body",
                "alerts": ["Sin métricas de calidad disponibles."],
            }
        return guide

    def evaluate(self, metrics: Optional[Dict[str, Any]], trend: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        normalized = self.normalize_metrics(metrics)
        guide = self.build_quality_guide(normalized)
        score_total = float(self._metric_value(normalized, "score_total", default=guide.get("guide_score", 0.0)))
        root_causes = self._compute_root_causes(normalized)
        severity = self._severity_for(score_total, root_causes)
        action = self._action_for(score_total, root_causes, severity)
        recommendation = self._recommendation_text(score_total, severity, action, root_causes)

        decision = {
            "severity": severity,
            "score_total": round(score_total, 2),
            "target_assimilation": round(self.target_assimilation, 2),
            "action": action,
            "root_causes": root_causes,
            "recommendation": recommendation,
            "quality_guide": guide,
            "trend": trend or {},
            "reinforcement_rounds_remaining": max(0, self.max_reinforcement_rounds),
        }

        self.history.append(decision)
        return decision

    def decide_action(self, metrics: Optional[Dict[str, Any]], trend: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.evaluate(metrics=metrics, trend=trend)

    def build_recommendation(self, metrics: Optional[Dict[str, Any]], trend: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.evaluate(metrics=metrics, trend=trend)

    def summarize(self, metrics: Optional[Dict[str, Any]], trend: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.evaluate(metrics=metrics, trend=trend)

    def _recommendation_text(self, score_total: float, severity: str, action: str, root_causes: List[str]) -> str:
        if score_total >= self.target_assimilation:
            return "La calidad actual supera el umbral objetivo; continuar el ciclo actual con vigilancia normal."

        cause_text = ", ".join(root_causes)
        if action == "rollback_to_last_healthy_snapshot":
            return (
                f"Colapso de calidad severo ({score_total:.1f} / {self.target_assimilation:.1f}). "
                f"Retroceder al último snapshot saludable y reforzar el siguiente tramo con un objetivo centrado en {cause_text}."
            )
        if action == "recalibrate_palette_and_reinforce":
            return (
                f"Se detectó pérdida crítica de paleta ({cause_text}). "
                f"Ajustar la remap de colores y volver a reforzar con una vista previa focalizada sobre identidad cromática."
            )
        if action == "reinforce_anatomy_with_targeted_preview":
            return (
                f"El fallo principal apunta a anatomía ({cause_text}). "
                f"Reforzar poses y silueta con preview orientado a cuerpo y detalle facial antes de continuar."
            )
        if action == "clean_alpha_and_micro_details":
            return (
                f"Los artefactos de alfa o micro-textura son relevantes ({cause_text}). "
                f"Limpiar el canal alfa y reforzar detalles antes de continuar el siguiente ciclo."
            )
        if severity == "high":
            return (
                f"La calidad está por debajo del umbral objetivo ({score_total:.1f} < {self.target_assimilation:.1f}) "
                f"y requiere refuerzo moderado: {cause_text}."
            )
        return (
            f"La calidad requiere seguimiento cercano ({score_total:.1f} < {self.target_assimilation:.1f}). "
            f"Mantener vigilancia y evaluar {cause_text} antes del siguiente tramo."
        )


__all__ = [
    "QUALITY_DIMENSIONS",
    "QualityGuidanceController",
    "QualityVector",
    "build_quality_vector",
    "normalize_quality_score",
]
