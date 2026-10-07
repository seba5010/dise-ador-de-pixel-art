import os
import sys
import time
import json
import math

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
from typing import Any, Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import argparse

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pixel_ai_engine.config import (
    MODEL_RESOLUTION,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    DEVICE,
    USE_AMP
)
from pixel_ai_engine.models import PixelArtUNetGenerator, PixelArtPatchDiscriminator
from pixel_ai_engine.dataset import place_in_cell
from pixel_ai_engine.enhancer import PixelArtEnhancer
from pixel_ai_engine.phase3_critical_enhancer import Phase3CriticalReviewer
from pixel_ai_engine.anatomical_guidance import (
    ENABLE_SILHOUETTE_LOSS,
    compute_anatomical_metrics,
    differentiable_silhouette_loss,
)
from pixel_ai_engine.strict_visual_quality import apply_strict_visual_metrics
from pixel_ai_engine.quality_guidance import (
    BASE_LOSS_WEIGHTS,
    ENABLE_ADAPTIVE_LOSS,
    ENABLE_QUALITY_CHECKPOINT,
    ENABLE_QUALITY_GUIDANCE,
    ENABLE_SMART_SAMPLING,
    FrameQualityTracker,
    GuidanceInterventionPolicy,
    HardExampleMiningPolicy,
    LossMultiplierController,
    QualityGuidanceController,
    SamplingPlan,
    assess_quality_checkpoint,
    compare_sampling_ab,
    compare_loss_ab,
)
from pixel_ai_engine.palette_remap import (
    extract_character_palette,
    remap_image_to_palette,
    clean_orphan_pixels,
    despeckle_chromatic_noise,
)
from pixel_ai_engine.training_recovery import (
    activate_recovery_status,
    choose_recovery_snapshot,
    copy_checkpoint_atomic,
    detect_training_instability,
    find_session_recovery_checkpoint,
    finite_positive,
    generate_session_id,
    get_session_dir,
    init_session,
    load_session_manifest,
    load_status_file,
    record_session_checkpoint,
    retain_checkpoint_atomic,
    write_status_file,
)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR = CHECKPOINT_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR = CHECKPOINT_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SAMPLES_DIR = PROJECT_ROOT / "training_samples"
TRAIN_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR = TRAIN_SAMPLES_DIR / "audit_history"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

STATUS_FILE = PROJECT_ROOT / "training_status.json"
STOP_FLAG_FILE = PROJECT_ROOT / "stop_training.flag"
PAUSE_FLAG_FILE = PROJECT_ROOT / "pause_training.flag"
RECOVERY_CHECKPOINT_FILE = CHECKPOINT_DIR / "recovery_checkpoint.pt"

QUALITY_AUDIT_SAMPLES_PER_EPOCH = 10
QUALITY_AUDIT_WINDOW_EPOCHS = 14
QUALITY_AUDIT_MIN_FRAMES_PER_CHARACTER = 2
QUALITY_TRACKER_MIN_COVERAGE = 1.0

CACHE_PATH = PROJECT_ROOT / "dataset_supervisado" / "supervised_cache_8x12.pt"


def _disabled_guidance() -> Dict[str, Any]:
    return {
        "enabled": False,
        "mode": "observational",
        "action": "CONTINUE",
        "recommended_action": "CONTINUE",
        "problem": None,
        "primary_problem": None,
        "secondary_problems": [],
        "severity": "low",
        "quality_vector": {},
        "trend": {"status": "DISABLED"},
        "training_modified": False,
    }


def _evaluate_observational_guidance(
    quality: Dict[str, Any],
    *,
    epoch: int,
    training_metrics: Dict[str, Any],
    previous_state: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    runtime_metrics = dict(training_metrics)
    previous = previous_state if isinstance(previous_state, dict) else {}
    cumulative_skips = max(0, int(runtime_metrics.get("skipped_amp_steps", 0) or 0))
    previous_cumulative = max(0, int(previous.get("last_skipped_amp_steps", 0) or 0))
    # AMP counters restart with the process, so a lower value means a new run.
    current_delta = cumulative_skips if cumulative_skips < previous_cumulative else cumulative_skips - previous_cumulative
    recent_skips = [
        max(0, int(value))
        for value in previous.get("amp_skip_history", [])
        if isinstance(value, (int, float))
    ][-4:]
    recent_skips.append(current_delta)
    runtime_metrics["skipped_amp_steps_window"] = sum(recent_skips)

    controller = QualityGuidanceController(state=previous_state)
    decision = controller.evaluate(
        quality,
        training_metrics=runtime_metrics,
        epoch=epoch,
    )
    state = controller.export_state()
    state["last_skipped_amp_steps"] = cumulative_skips
    state["amp_skip_history"] = recent_skips
    state["amp_skips_in_window"] = sum(recent_skips)
    for key in ("dataset_character_ids", "dataset_sample_count"):
        if key in previous:
            state[key] = previous[key]
    return decision, state


def _dataset_sample_descriptors(dataset: Any) -> list[Dict[str, Any]]:
    samples = getattr(dataset, "samples", None)
    if isinstance(samples, list):
        return [
            {
                "char_id": sample.get("char_id", "unknown"),
                "frame_idx": sample.get("frame_idx", index),
            }
            for index, sample in enumerate(samples)
            if isinstance(sample, dict)
        ]
    return [{"char_id": "unknown", "frame_idx": index} for index in range(len(dataset))]


def _build_sampling_plan(
    dataset: Any,
    frame_quality: Optional[Dict[str, Any]],
    guidance: Optional[Dict[str, Any]],
) -> SamplingPlan:
    recommendation = "CONTINUE"
    if isinstance(guidance, dict):
        recommendation = str(guidance.get("authorized_action", guidance.get("recommended_action", "CONTINUE")))
    from pixel_ai_engine.hard_examples import load_manual_sampling_weights

    return HardExampleMiningPolicy().build_plan(
        _dataset_sample_descriptors(dataset),
        frame_quality,
        enabled=bool(ENABLE_QUALITY_GUIDANCE and ENABLE_SMART_SAMPLING),
        recommendation=recommendation,
        manual_weights=load_manual_sampling_weights(),
    )


def _build_training_dataloader(dataset: Dataset, batch_size: int, plan: SamplingPlan) -> DataLoader:
    if plan.active and len(plan.weights) == len(dataset):
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(plan.weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def _attach_sampling_plan(status_data: Dict[str, Any], plan: SamplingPlan) -> Dict[str, Any]:
    data = dict(status_data)
    plan_payload = plan.to_dict()
    summary = {key: value for key, value in plan_payload.items() if key not in {"weights", "sample_keys"}}
    summary["ab"] = compare_sampling_ab(plan)
    weights = {
        key: weight
        for key, weight in zip(plan.sample_keys, plan.weights)
        if weight > 1.0
    }
    guidance_state = dict(data.get("guidance_state", {}))
    guidance_state["sampling_weights"] = weights
    guidance_state["sampling_plan"] = summary
    data["guidance_state"] = guidance_state
    data["sampling"] = summary

    guidance = data.get("guidance")
    if isinstance(guidance, dict):
        guidance = dict(guidance)
        guidance["sampling_change"] = summary
        if plan.active:
            guidance["action"] = "ADJUST_SAMPLING"
            guidance["training_modified"] = True
        data["guidance"] = guidance
        data["quality_guidance"] = guidance
        for entry in data.get("history", []):
            if isinstance(entry, dict) and entry.get("epoch") == data.get("epoch"):
                entry["guidance"] = guidance
                entry["quality_guidance"] = guidance
        decisions = guidance_state.get("decision_history")
        if isinstance(decisions, list) and decisions:
            decisions[-1] = dict(decisions[-1])
            decisions[-1]["sampling_change"] = summary
    return data


def _build_loss_plan(guidance_state: Optional[Dict[str, Any]], guidance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    state = guidance_state if isinstance(guidance_state, dict) else {}
    effective_guidance = dict(guidance) if isinstance(guidance, dict) else {}
    if "authorized_action" in effective_guidance:
        effective_guidance["recommended_action"] = effective_guidance["authorized_action"]
    return LossMultiplierController().propose(
        state.get("loss_multipliers", {}),
        effective_guidance,
        enabled=bool(ENABLE_QUALITY_GUIDANCE and ENABLE_ADAPTIVE_LOSS),
    )


def _attach_loss_plan(status_data: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(status_data)
    payload = dict(plan)
    payload["ab"] = compare_loss_ab(payload)
    guidance_state = dict(data.get("guidance_state", {}))
    guidance_state["loss_multipliers"] = dict(payload.get("multipliers", {}))
    guidance_state["loss_plan"] = payload
    data["guidance_state"] = guidance_state
    data["adaptive_loss"] = payload
    guidance = data.get("guidance")
    if isinstance(guidance, dict):
        guidance = dict(guidance)
        guidance["loss_change"] = payload
        if payload.get("active"):
            guidance["action"] = "ADJUST_WEIGHTS"
            guidance["training_modified"] = True
        data["guidance"] = guidance
        data["quality_guidance"] = guidance
        for entry in data.get("history", []):
            if isinstance(entry, dict) and entry.get("epoch") == data.get("epoch"):
                entry["guidance"] = guidance
                entry["quality_guidance"] = guidance
        decisions = guidance_state.get("decision_history")
        if isinstance(decisions, list) and decisions:
            decisions[-1] = dict(decisions[-1])
            decisions[-1]["loss_change"] = payload
    return data


def _apply_intervention_policy(status_data: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(status_data)
    guidance = data.get("guidance")
    state = data.get("guidance_state", {})
    if not isinstance(guidance, dict) or not isinstance(state, dict):
        return data
    policy_decision = GuidanceInterventionPolicy().decide(
        int(data.get("epoch", 0)),
        guidance,
        state.get("intervention_state"),
    )
    guidance = dict(guidance)
    guidance["authorized_action"] = policy_decision["authorized_action"]
    guidance["intervention_policy"] = {
        key: value for key, value in policy_decision.items() if key != "state"
    }
    state = dict(state)
    state["intervention_state"] = policy_decision["state"]
    state["intervention_count"] = policy_decision["state"]["total_interventions"]
    data["guidance"] = guidance
    data["quality_guidance"] = guidance
    data["guidance_state"] = state
    for entry in data.get("history", []):
        if isinstance(entry, dict) and entry.get("epoch") == data.get("epoch"):
            entry["guidance"] = guidance
            entry["quality_guidance"] = guidance
    return data


def _apply_authorized_lr_reduction(
    status_data: Dict[str, Any],
    optimizer_g: Any,
    optimizer_d: Any,
    scheduler_g: Any = None,
    scheduler_d: Any = None,
    *,
    factor: float = 0.5,
    minimum_lr: float = 1e-6,
) -> tuple[Dict[str, Any], Optional[float]]:
    """Apply a QC-authorized LR reduction and make it survive scheduler steps."""
    data = dict(status_data)
    guidance = data.get("guidance")
    if not isinstance(guidance, dict) or guidance.get("authorized_action") != "REDUCE_LR":
        return data, None

    factor = min(1.0, max(0.05, float(factor)))
    minimum_lr = max(0.0, float(minimum_lr))

    def reduce_optimizer(optimizer: Any) -> list[float]:
        values: list[float] = []
        for group in getattr(optimizer, "param_groups", []):
            old_lr = float(group.get("lr", minimum_lr))
            new_lr = max(minimum_lr, old_lr * factor)
            group["lr"] = new_lr
            values.append(new_lr)
        return values

    generator_lrs = reduce_optimizer(optimizer_g)
    discriminator_lrs = reduce_optimizer(optimizer_d)
    for scheduler, values in ((scheduler_g, generator_lrs), (scheduler_d, discriminator_lrs)):
        if scheduler is None or not values:
            continue
        if hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = [max(minimum_lr, float(value) * factor) for value in scheduler.base_lrs]
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = list(values)

    guidance = dict(guidance)
    guidance["action"] = "REDUCE_LR"
    guidance["training_modified"] = True
    guidance["lr_change"] = {
        "factor": factor,
        "generator_lr": generator_lrs[0] if generator_lrs else None,
        "discriminator_lr": discriminator_lrs[0] if discriminator_lrs else None,
    }
    data["guidance"] = guidance
    data["quality_guidance"] = guidance
    data["lr"] = generator_lrs[0] if generator_lrs else data.get("lr")
    for entry in data.get("history", []):
        if isinstance(entry, dict) and entry.get("epoch") == data.get("epoch"):
            entry["guidance"] = guidance
            entry["quality_guidance"] = guidance
    return data, generator_lrs[0] if generator_lrs else None


def _quality_is_critical(status_data: Dict[str, Any]) -> bool:
    guidance = status_data.get("guidance")
    return isinstance(guidance, dict) and str(guidance.get("severity", "")).lower() == "critical"


def _approved_checkpoint_path() -> Optional[Path]:
    for candidate in (
        CHECKPOINT_DIR / "best_quality_generator.pt",
        CHECKPOINT_DIR / "best_generator.pt",
    ):
        if candidate.is_file():
            return candidate
    return None


def _attach_model_approval(status_data: Dict[str, Any], *, final_critical: bool) -> Dict[str, Any]:
    """Separate the resumable latest checkpoint from the model served to generation."""
    data = dict(status_data)
    selected = _approved_checkpoint_path()
    selected_name = None
    if selected is not None:
        try:
            selected_name = str(selected.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            selected_name = str(selected)
    data["model_approval"] = {
        "approved": not final_critical,
        "latest_rejected": bool(final_critical),
        "selected_checkpoint": selected_name,
        "reason": "final_quality_critical" if final_critical else "quality_gate_passed",
    }
    if final_critical:
        data["status"] = "COMPLETADO_CON_ALERTA_QC"
    return data


def _guidance_state_from_checkpoint(
    status_data: Optional[Dict[str, Any]], checkpoint: Any, *, prefer_checkpoint: bool = False
) -> Optional[Dict[str, Any]]:
    status_state = (status_data or {}).get("guidance_state")
    checkpoint_state = checkpoint.get("guidance_state") if isinstance(checkpoint, dict) else None
    candidates = (checkpoint_state, status_state) if prefer_checkpoint else (status_state, checkpoint_state)
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return None


def _quality_summary_from_history(history: Any) -> Dict[str, Optional[float]]:
    summary_keys = (
        "global", "anatomy", "silhouette", "pose", "face", "hair", "clothing",
        "arms_hands", "feet", "props", "palette", "alpha", "micro_detail", "outline",
    )
    summary: Dict[str, Optional[float]] = {f"best_{key}": None for key in summary_keys}
    if not isinstance(history, list):
        return summary
    for entry in history:
        if not isinstance(entry, dict):
            continue
        guidance = entry.get("guidance", entry.get("quality_guidance"))
        vector = guidance.get("quality_vector") if isinstance(guidance, dict) else None
        if not isinstance(vector, dict):
            continue
        for key in summary_keys:
            value = vector.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                summary_key = f"best_{key}"
                previous = summary[summary_key]
                summary[summary_key] = round(max(float(value), previous if previous is not None else float("-inf")), 4)
    return summary


def _print_quality_guidance(guidance: Any) -> None:
    if not isinstance(guidance, dict) or not guidance.get("enabled"):
        return
    vector = guidance.get("quality_vector") if isinstance(guidance.get("quality_vector"), dict) else {}

    def metric(name: str) -> str:
        value = vector.get(name)
        return f"{float(value):.1f}%" if isinstance(value, (int, float)) else "N/D"

    primary = guidance.get("primary_problem") or "ninguno"
    secondary = ", ".join(guidance.get("secondary_problems") or []) or "ninguno"
    mode = "AJUSTANDO" if guidance.get("training_modified") else "OBSERVANDO"
    authorized_action = guidance.get("authorized_action", guidance.get("recommended_action", "CONTINUE"))
    print("\nQUALITY GUIDANCE", flush=True)
    print(
        f"  Global: {metric('global')} | Anatomía: {metric('anatomy')} | "
        f"Silueta: {metric('silhouette')} | Rostro: {metric('face')}",
        flush=True,
    )
    print(
        f"  Ropa: {metric('clothing')} | Paleta: {metric('palette')} | "
        f"Alfa: {metric('alpha')} | Microdetalle: {metric('micro_detail')}",
        flush=True,
    )
    print(
        f"  Problema principal: {primary} | Secundarios: {secondary} | "
        f"Severidad: {str(guidance.get('severity', 'low')).upper()}",
        flush=True,
    )
    print(
        f"  Acción recomendada: {guidance.get('recommended_action', 'CONTINUE')} | "
        f"Acción autorizada: {authorized_action} | MODO ACTUAL: {mode}",
        flush=True,
    )
    coverage = guidance.get("audit_coverage")
    if isinstance(coverage, dict):
        print(
            f"  Cobertura QC: {int(coverage.get('characters_covered', 0))}/"
            f"{int(coverage.get('characters_total', 0))} personajes | "
            f"mín. {int(coverage.get('minimum_frames_per_character', 0))} frames/personaje | "
            f"tracker {float(coverage.get('tracker_frame_coverage', 0.0)) * 100.0:.1f}% | "
            f"rollback {'HABILITADO' if coverage.get('ready_for_rollback') else 'EN ESPERA'}",
            flush=True,
        )
    certification = guidance.get("full_certification")
    if isinstance(certification, dict) and certification.get("due"):
        print(
            f"  Certificación completa: {'APROBADA' if certification.get('passed') else 'NO APROBADA'}",
            flush=True,
        )


def _active_recovery_checkpoint(status_data: Dict[str, Any]) -> Optional[Path]:
    recovery = status_data.get("recovery")
    if not isinstance(recovery, dict) or not recovery.get("active"):
        return None
    raw_path = recovery.get("checkpoint")
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(CHECKPOINT_DIR.resolve())
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _last_quality_healthy_checkpoint(
    history: Any,
    failed_epoch: int,
    session_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Return the exact latest epoch accepted by QC, prioritizing current session."""
    if session_dir is not None and session_dir.exists():
        session_plan = find_session_recovery_checkpoint(session_dir, failed_epoch)
        if session_plan is not None:
            try:
                ckpt = torch.load(session_plan["path"], map_location="cpu")
                if _checkpoint_is_complete(ckpt):
                    session_plan["checkpoint"] = ckpt
                    return session_plan
            except Exception:
                pass

    checkpoint_file = CHECKPOINT_DIR / "last_quality_healthy_checkpoint.pt"
    if not checkpoint_file.is_file():
        return None
    try:
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
    except Exception:
        return None
    if not _checkpoint_is_complete(checkpoint):
        return None
    source_epoch = int(checkpoint.get("epoch", 0))
    if source_epoch <= 0 or source_epoch >= int(failed_epoch):
        return None
    matching_entry = next(
        (
            entry for entry in reversed(history if isinstance(history, list) else [])
            if isinstance(entry, dict) and int(entry.get("epoch", 0)) == source_epoch
        ),
        None,
    )
    if not isinstance(matching_entry, dict):
        # Do not restore a checkpoint retained by a previous training era.
        return None
    guidance = matching_entry.get("guidance", matching_entry.get("quality_guidance"))
    if isinstance(guidance, dict) and str(guidance.get("severity", "low")).lower() == "critical":
        return None
    return {
        "epoch": source_epoch,
        "path": checkpoint_file,
        "selection": "latest_qc_healthy_epoch",
        "checkpoint": checkpoint,
    }


def _checkpoint_is_complete(checkpoint: Any) -> bool:
    return (
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("generator"), dict)
        and isinstance(checkpoint.get("discriminator"), dict)
        and int(checkpoint.get("epoch", 0)) >= 0
        and finite_positive(checkpoint.get("loss")) is not None
    )


def _best_checkpoint_is_consistent(checkpoint: Any) -> bool:
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("generator"), dict):
        return False
    loss = finite_positive(checkpoint.get("loss"))
    best_loss = finite_positive(checkpoint.get("best_loss", checkpoint.get("loss")))
    if loss is None or best_loss is None:
        return False
    tolerance = max(1e-5, best_loss * 0.01)
    return loss <= best_loss + tolerance


def _repair_best_checkpoint(source_checkpoint: Dict[str, Any], source_epoch: int, source_loss: float) -> Optional[Path]:
    best_path = CHECKPOINT_DIR / "best_generator.pt"
    current_best = None
    if best_path.exists():
        try:
            current_best = torch.load(best_path, map_location="cpu")
        except Exception:
            current_best = None
    if _best_checkpoint_is_consistent(current_best):
        return None

    backup_path = None
    if best_path.exists():
        quarantine_dir = CHECKPOINT_DIR / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        old_epoch = current_best.get("epoch", "unknown") if isinstance(current_best, dict) else "unreadable"
        backup_path = quarantine_dir / f"best_generator_epoch_{old_epoch}_{time.strftime('%Y%m%d_%H%M%S')}.pt"
        copy_checkpoint_atomic(best_path, backup_path)

    repaired_state = {
        "epoch": int(source_epoch),
        "generator": source_checkpoint["generator"],
        "loss": float(source_loss),
        "best_loss": float(source_loss),
        "recovered_from": source_checkpoint.get("recovery_origin", "automatic_recovery"),
    }
    temporary_file = best_path.with_suffix(best_path.suffix + ".tmp")
    torch.save(repaired_state, temporary_file)
    os.replace(temporary_file, best_path)
    return backup_path


def _replace_latest_with_recovery(source_epoch: int, source_loss: float) -> Optional[Path]:
    latest_path = CHECKPOINT_DIR / "latest_checkpoint.pt"
    latest_checkpoint = None
    if latest_path.exists():
        try:
            latest_checkpoint = torch.load(latest_path, map_location="cpu")
        except Exception:
            latest_checkpoint = None
    latest_epoch = int(latest_checkpoint.get("epoch", -1)) if isinstance(latest_checkpoint, dict) else -1
    latest_loss = finite_positive(latest_checkpoint.get("loss")) if isinstance(latest_checkpoint, dict) else None
    if latest_epoch == int(source_epoch) and latest_loss is not None and abs(latest_loss - source_loss) <= 1e-8:
        return None

    backup_path = None
    if latest_path.exists():
        quarantine_dir = CHECKPOINT_DIR / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        backup_path = quarantine_dir / f"latest_checkpoint_epoch_{latest_epoch}_{time.strftime('%Y%m%d_%H%M%S')}.pt"
        copy_checkpoint_atomic(latest_path, backup_path)
    copy_checkpoint_atomic(RECOVERY_CHECKPOINT_FILE, latest_path)
    return backup_path


def _materialize_recovery_route(
    status_data: Dict[str, Any],
    reason: Dict[str, Any],
    status_name: str,
    total_epochs: Optional[int] = None,
    rejected_metrics: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    failed_epoch = int(reason.get("failed_epoch", status_data.get("epoch", 0)))
    session_dir = None
    session_dir_raw = status_data.get("session_dir")
    if session_dir_raw:
        cand = Path(session_dir_raw)
        if not cand.is_absolute():
            cand = PROJECT_ROOT / cand
        if cand.exists():
            session_dir = cand
    elif status_data.get("session_id"):
        cand = get_session_dir(status_data["session_id"], CHECKPOINT_DIR)
        if cand.exists():
            session_dir = cand

    is_start_mode = str(status_data.get("mode", "")).lower() == "start"
    allow_snapshots_fallback = not is_start_mode

    recovery_plan = _last_quality_healthy_checkpoint(status_data.get("history", []), failed_epoch, session_dir=session_dir)
    if recovery_plan is None:
        recovery_plan = choose_recovery_snapshot(
            status_data.get("history", []),
            SNAPSHOTS_DIR,
            failed_epoch,
            session_dir=session_dir,
            allow_snapshots_fallback=allow_snapshots_fallback,
        )
    if recovery_plan is None:
        return None

    source_file = Path(recovery_plan["path"])
    source_checkpoint = recovery_plan.get("checkpoint")
    if not isinstance(source_checkpoint, dict):
        source_checkpoint = torch.load(source_file, map_location="cpu")
    if not _checkpoint_is_complete(source_checkpoint):
        raise RuntimeError(f"El snapshot de recuperacion no contiene un estado completo: {source_file}")
    source_epoch = int(source_checkpoint.get("epoch", recovery_plan["epoch"]))
    source_loss = finite_positive(source_checkpoint.get("loss"))
    if source_loss is None:
        raise RuntimeError(f"El snapshot de recuperacion no contiene una perdida valida: {source_file}")

    source_optimizer_lr = None
    opt_state = source_checkpoint.get("opt_g")
    if isinstance(opt_state, dict):
        param_groups = opt_state.get("param_groups", [])
        if param_groups:
            source_optimizer_lr = finite_positive(param_groups[0].get("lr"))
    preferred_lr = min(8e-5, (source_optimizer_lr or 1.5e-4) * 0.5)

    source_checkpoint["best_loss"] = source_loss
    source_checkpoint["recovery_origin"] = str(source_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
    temporary_recovery = RECOVERY_CHECKPOINT_FILE.with_suffix(RECOVERY_CHECKPOINT_FILE.suffix + ".tmp")
    torch.save(source_checkpoint, temporary_recovery)
    os.replace(temporary_recovery, RECOVERY_CHECKPOINT_FILE)

    relative_recovery_file = RECOVERY_CHECKPOINT_FILE.relative_to(PROJECT_ROOT)
    repaired_status = activate_recovery_status(
        status_data=status_data,
        source_epoch=source_epoch,
        source_file=relative_recovery_file,
        source_loss=source_loss,
        failed_epoch=failed_epoch,
        reason={**reason, "selection": recovery_plan.get("selection")},
        total_epochs=max(source_epoch, int(total_epochs or status_data.get("total_epochs", failed_epoch))),
        preferred_lr=preferred_lr,
        rejected_metrics=rejected_metrics or {
            "g_loss": reason.get("g_loss"),
            "l1_loss": reason.get("l1_loss"),
            "quality": status_data.get("quality"),
        },
        status_name=status_name,
    )
    if "session_id" in status_data:
        repaired_status["session_id"] = status_data.get("session_id")
    if "session_dir" in status_data:
        repaired_status["session_dir"] = status_data.get("session_dir")
    if "session_start_epoch" in status_data:
        repaired_status["session_start_epoch"] = status_data.get("session_start_epoch")
    write_status_file(STATUS_FILE, repaired_status)
    backup_path = _repair_best_checkpoint(source_checkpoint, source_epoch, source_loss)
    latest_backup_path = _replace_latest_with_recovery(source_epoch, source_loss)
    return {
        "checkpoint": RECOVERY_CHECKPOINT_FILE,
        "source_epoch": source_epoch,
        "source_loss": source_loss,
        "preferred_lr": preferred_lr,
        "reason": reason,
        "backup": backup_path,
        "latest_backup": latest_backup_path,
        "already_active": False,
    }


def repair_current_training_state(status_name: str = "RECUPERACION_LISTA") -> Optional[Dict[str, Any]]:
    status_data = load_status_file(STATUS_FILE)
    active_checkpoint = _active_recovery_checkpoint(status_data)
    if active_checkpoint is not None:
        recovery = status_data.get("recovery", {})
        source_epoch = int(recovery.get("source_epoch", status_data.get("epoch", 0)))
        source_loss = finite_positive(status_data.get("g_loss"))
        latest_backup_path = None
        if source_loss is not None:
            latest_backup_path = _replace_latest_with_recovery(source_epoch, source_loss)
        return {
            "checkpoint": active_checkpoint,
            "source_epoch": source_epoch,
            "source_loss": source_loss,
            "preferred_lr": finite_positive(recovery.get("preferred_lr")),
            "reason": recovery.get("reason", {}),
            "latest_backup": latest_backup_path,
            "already_active": True,
        }

    reason = detect_training_instability(status_data.get("history", []), quality=status_data.get("quality"))
    if reason is None and status_data.get("status") == "PAUSADO_QC":
        reason = {
            "code": "quality_control_paused",
            "message": status_data.get("error_details", {}).get("error", "Pausado por Control de Calidad."),
            "failed_epoch": int(status_data.get("epoch", 100)),
            "g_loss": status_data.get("g_loss"),
            "l1_loss": status_data.get("l1_loss"),
            "quality": status_data.get("quality"),
        }
    if reason is None:
        return None
    return _materialize_recovery_route(status_data, reason, status_name)

# -------------------------------------------------------------
# 1. PÉRDIDA DE BORDES ACELERADA EN GPU CON KORNIA (DE FORGE)
# -------------------------------------------------------------
try:
    import kornia
    class KorniaSobelLoss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, pred, target):
            p_rgb = pred[:, :3]
            t_rgb = target[:, :3]
            p_grad = kornia.filters.sobel(p_rgb)
            t_grad = kornia.filters.sobel(t_rgb)
            return nn.functional.smooth_l1_loss(p_grad, t_grad)
    HAS_KORNIA = True
except Exception:
    HAS_KORNIA = False

class FallbackSobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, pred, target):
        loss = 0.0
        for c in range(min(3, pred.shape[1])):
            p_c = pred[:, c:c+1]
            t_c = target[:, c:c+1]
            p_gx = nn.functional.conv2d(p_c, self.kx, padding=1)
            p_gy = nn.functional.conv2d(p_c, self.ky, padding=1)
            t_gx = nn.functional.conv2d(t_c, self.kx, padding=1)
            t_gy = nn.functional.conv2d(t_c, self.ky, padding=1)
            p_edge = torch.sqrt(torch.clamp(p_gx**2 + p_gy**2, min=1e-5))
            t_edge = torch.sqrt(torch.clamp(t_gx**2 + t_gy**2, min=1e-5))
            loss += nn.functional.smooth_l1_loss(p_edge, t_edge)
        return loss / 3.0

# -------------------------------------------------------------
# 2. DATASET CON AUMENTO INTELIGENTE DE DATOS (ALBUMENTATIONS)
# -------------------------------------------------------------
try:
    import albumentations as A
    try:
        # Desactivar explícitamente tono y saturación para conservar la identidad exacta del personaje
        AUG_PIPELINE = A.Compose([
            A.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.0, hue=0.0, p=0.4),
        ])
    except Exception:
        AUG_PIPELINE = A.Compose([
            A.ColorJitter(brightness=(0.97, 1.03), contrast=(0.97, 1.03), saturation=(1.0, 1.0), hue=(0.0, 0.0), p=0.4),
        ])
    HAS_ALBUMENTATIONS = True
except Exception:
    HAS_ALBUMENTATIONS = False

class SupervisedTensorDataset(Dataset):
    def __init__(self, cache_path, augment: bool = True):
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache no encontrada: {cache_path}. Ejecuta prepare_supervised_dataset.py primero.")
        self.samples = torch.load(cache_path)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        front = s["front_tensor"] # (3, H, W)
        f_idx = s["frame_idx"]
        target = s["target_tensor"] # (4, H, W)

        # Si el aumento esta activo, aplicar sutil variacion al frontal para robustez
        if self.augment and HAS_ALBUMENTATIONS and torch.rand(1).item() < 0.35:
            f_np = ((front.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            aug_np = AUG_PIPELINE(image=f_np)["image"]
            aug_t = torch.from_numpy(aug_np.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
            front = aug_t

        return front, f_idx, target, s["char_id"]


def compute_frame_quality_batch(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    frame_indices: Any,
    char_ids: Any,
) -> list[Dict[str, Any]]:
    """Compute detached per-sample scores; these values never enter ``total_g``."""
    if predictions.ndim != 4 or targets.ndim != 4 or predictions.shape[0] != targets.shape[0]:
        return []
    measurements: list[Dict[str, Any]] = []
    with torch.no_grad():
        predicted = predictions.detach().float().clamp(-1.0, 1.0)
        expected = targets.detach().float().clamp(-1.0, 1.0)
        batch_size = predicted.shape[0]
        for index in range(batch_size):
            color_error = torch.mean(torch.abs(predicted[index, :3] - expected[index, :3])) / 2.0
            color_score = float((1.0 - color_error).clamp(0.0, 1.0).item() * 100.0)
            if predicted.shape[1] >= 4 and expected.shape[1] >= 4:
                alpha_error = torch.mean(torch.abs(predicted[index, 3] - expected[index, 3])) / 2.0
                alpha_score = float((1.0 - alpha_error).clamp(0.0, 1.0).item() * 100.0)
                pred_mask = predicted[index, 3] > 0.0
                target_mask = expected[index, 3] > 0.0
                intersection = torch.logical_and(pred_mask, target_mask).sum().float()
                union = torch.logical_or(pred_mask, target_mask).sum().float()
                silhouette = float((intersection / union.clamp_min(1.0)).item() * 100.0)
                target_pixels = int(target_mask.sum().item())
            else:
                alpha_score = 100.0
                silhouette = color_score
                target_pixels = int(expected[index, :3].abs().sum().item() > 0)
            quality = color_score * 0.50 + silhouette * 0.30 + alpha_score * 0.20
            frame_value = frame_indices[index]
            frame_idx = int(frame_value.item()) if hasattr(frame_value, "item") else int(frame_value)
            char_id = char_ids[index] if isinstance(char_ids, (list, tuple)) else char_ids[index]
            measurements.append({
                "char_id": str(char_id),
                "frame_idx": frame_idx,
                "quality": round(quality, 4),
                "confidence": 1.0 if target_pixels > 0 else 0.25,
                "metrics": {
                    "color": round(color_score, 4),
                    "alpha": round(alpha_score, 4),
                    "silhouette": round(silhouette, 4),
                },
            })
    return measurements

def get_hardware_telemetry():
    telemetry = {
        "gpu_temp": 55,
        "cpu_temp": None,
        "vram_gb": 0.88,
        "vram_total_gb": 4.0,
        "vram_pct": 22.0,
        "gpu_util": 0
    }
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, text=True, timeout=1.5
        )
        parts = [x.strip() for x in res.stdout.strip().split(",")]
        if len(parts) >= 3:
            telemetry["gpu_temp"] = int(parts[0])
            used_mb = int(parts[1])
            tot_mb = int(parts[2])
            telemetry["vram_gb"] = round(used_mb / 1024.0, 2)
            telemetry["vram_total_gb"] = round(tot_mb / 1024.0, 1)
            telemetry["vram_pct"] = round((used_mb / max(1, tot_mb)) * 100, 1)
            if len(parts) > 3 and parts[3].isdigit():
                telemetry["gpu_util"] = int(parts[3])
    except Exception:
        pass

    try:
        import subprocess
        cmd = ['powershell', '-NoProfile', '-Command', '(Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction SilentlyContinue).CurrentTemperature']
        out = subprocess.check_output(cmd, text=True, timeout=2.0).strip()
        digits = [int(x.strip()) for x in out.split() if x.strip().isdigit()]
        if digits:
            c_temp = round(digits[0] / 10.0 - 273.15)
            if 20 <= c_temp <= 115:
                telemetry["cpu_temp"] = c_temp
    except Exception:
        pass

    return telemetry

def get_gpu_temperature():
    return get_hardware_telemetry()["gpu_temp"]

def manage_adaptive_thermal_throttle(temp_gpu_limit: int = 84,
                                     temp_cpu_limit: int = 90,
                                     temp_gpu_cooldown: int = 74,
                                     temp_cpu_cooldown: int = 80):
    """
    Termostato Inteligente Dual para Laptops (Protección Integral de GPU y CPU):
    - Monitorea simultáneamente la GPU (NVIDIA) y la CPU (WMI ACPI).
    - Si la GPU >= limit o CPU >= limit: PAUSA el entrenamiento hasta que bajen a cooldown.
    - Micro-pausa preventiva de 3s si GPU >= 80°C o CPU >= 85°C para disipar calor de heatpipes.
    - Pausa base de 0.5s en cada época para desestresar los VRMs.
    """
    if not torch.cuda.is_available():
        return None, None

    hw = get_hardware_telemetry()
    g_temp = hw.get("gpu_temp")
    c_temp = hw.get("cpu_temp")

    gpu_over = (g_temp is not None and g_temp >= temp_gpu_limit)
    cpu_over = (c_temp is not None and c_temp >= temp_cpu_limit)

    if gpu_over or cpu_over:
        motivo = []
        if gpu_over: motivo.append(f"GPU {g_temp}°C >= {temp_gpu_limit}°C")
        if cpu_over: motivo.append(f"CPU {c_temp}°C >= {temp_cpu_limit}°C")
        print(f"\n  [🔥 TERMOSTATO ACTIVO: {', '.join(motivo)}] Pausando entrenamiento para enfriar...", flush=True)

        while True:
            time.sleep(3.0)
            hw = get_hardware_telemetry()
            g_temp = hw.get("gpu_temp")
            c_temp = hw.get("cpu_temp")
            g_ok = (g_temp is None or g_temp <= temp_gpu_cooldown)
            c_ok = (c_temp is None or c_temp <= temp_cpu_cooldown)
            print(f"     -> Enfriando... GPU: {g_temp or '?'}°C (meta <= {temp_gpu_cooldown}°C) | CPU: {c_temp or '?'}°C (meta <= {temp_cpu_cooldown}°C)...", end="\r", flush=True)
            if g_ok and c_ok:
                break
        print(f"\n  [❄️ TEMPERATURAS SEGURAS ALCANZADAS: GPU {g_temp}°C | CPU {c_temp}°C] Reanudando...\n", flush=True)
    elif (g_temp is not None and g_temp >= 80) or (c_temp is not None and c_temp >= 85):
        time.sleep(3.0)
    else:
        time.sleep(0.5)

    return g_temp, c_temp

def get_available_snapshots():
    snaps = []
    status_data = load_status_file(STATUS_FILE)
    max_valid_epoch = None
    if status_data.get("recovery_events"):
        max_valid_epoch = int(status_data.get("epoch", 0))
    seen_epochs = set()
    session_dir_raw = status_data.get("session_dir")
    if session_dir_raw:
        sdir = Path(session_dir_raw)
        if not sdir.is_absolute():
            sdir = PROJECT_ROOT / sdir
        if sdir.exists():
            for p in sorted(sdir.glob("epoch_*.pt")):
                try:
                    ep = int(p.stem.replace("epoch_", ""))
                    if max_valid_epoch is not None and ep > max_valid_epoch:
                        continue
                    seen_epochs.add(ep)
                    img_url = f"/training_samples/audit_history/preview_epoch_{ep:03d}.png"
                    snaps.append({"epoch": ep, "file": f"{sdir.name}/{p.name}", "preview_url": img_url, "session_id": sdir.name})
                except Exception:
                    pass
    if SNAPSHOTS_DIR.exists():
        for p in sorted(SNAPSHOTS_DIR.glob("checkpoint_epoch_*.pt")):
            name = p.stem
            try:
                ep = int(name.replace("checkpoint_epoch_", ""))
                if max_valid_epoch is not None and ep > max_valid_epoch:
                    continue
                if ep in seen_epochs:
                    continue
                img_url = f"/training_samples/audit_history/preview_epoch_{ep:03d}.png"
                snaps.append({"epoch": ep, "file": p.name, "preview_url": img_url})
            except Exception:
                pass
    snaps.sort(key=lambda x: x["epoch"])
    return snaps

def update_status(epoch, total_epochs, status_str, g_loss, d_loss, l1_val, edge_val, start_time, lr_val=1.5e-4, error_details=None, skipped_amp=0, epoch_duration=None, quality=None, guidance_runtime=None, reevaluate_guidance=True, session_id=None, session_dir=None, session_start_epoch=None):
    elapsed = round(time.time() - start_time, 1)
    history = []
    past_eras = []
    best_loss = None
    initial_loss = None
    last_error = None
    prev_quality = None
    prev_guidance = None
    guidance_state: Dict[str, Any] = {}
    recovery = None
    recovery_events = []
    dataset_layout = {}
    previous_total_frames = 1008
    best_quality_score = None
    quality_checkpoint = None

    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                prev = json.load(f)
                history = prev.get("history", [])
                past_eras = prev.get("past_eras", [])
                raw_best = prev.get("best_loss")
                raw_init = prev.get("initial_loss")
                last_error = prev.get("error_details")
                prev_quality = prev.get("quality")
                prev_guidance = prev.get("guidance", prev.get("quality_guidance"))
                if isinstance(prev.get("guidance_state"), dict):
                    guidance_state = prev["guidance_state"]
                recovery = prev.get("recovery")
                recovery_events = prev.get("recovery_events", [])
                dataset_layout = prev.get("dataset_layout", {})
                previous_total_frames = prev.get("total_frames", previous_total_frames)
                best_quality_score = prev.get("best_quality_score")
                quality_checkpoint = prev.get("quality_checkpoint")
                if session_id is None:
                    session_id = prev.get("session_id")
                if session_dir is None:
                    session_dir = prev.get("session_dir")
                if session_start_epoch is None:
                    session_start_epoch = prev.get("session_start_epoch")
                # Solo aceptar numeros finitos estrictamente positivos (evita 0.0 heredado)
                if raw_best is not None and isinstance(raw_best, (int, float)) and math.isfinite(float(raw_best)) and float(raw_best) > 0.0:
                    best_loss = float(raw_best)
                if raw_init is not None and isinstance(raw_init, (int, float)) and math.isfinite(float(raw_init)) and float(raw_init) > 0.0:
                    initial_loss = float(raw_init)
        except Exception:
            history = []
            past_eras = []
            guidance_state = {}

    # Recuperar de historial valido si aun no estan inicializados o eran 0.0 heredados
    valid_losses = [h["loss"] for h in history if isinstance(h.get("loss"), (int, float)) and math.isfinite(h["loss"]) and h["loss"] > 0.0]
    if valid_losses:
        if initial_loss is None:
            initial_loss = valid_losses[0]
        if best_loss is None:
            best_loss = min(valid_losses)

    if error_details is not None:
        last_error = error_details

    if (
        status_str == "ENTRENANDO"
        and isinstance(recovery, dict)
        and recovery.get("active")
        and int(epoch) > int(recovery.get("source_epoch", -1))
    ):
        recovery = dict(recovery)
        recovery["active"] = False
        recovery["completed_at_epoch"] = int(epoch)
        recovery["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        last_error = None

    def safe_num(v):
        if v is None:
            return None
        try:
            f = float(v)
            return round(f, 4) if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    f_loss = safe_num(g_loss)
    f_d = safe_num(d_loss)
    f_l1 = safe_num(l1_val)
    f_edge = safe_num(edge_val)
    try:
        f_lr = float(lr_val) if (lr_val is not None and math.isfinite(float(lr_val))) else 1.5e-4
    except Exception:
        f_lr = 1.5e-4

    # Comprobar finitud de metricas: si alguna metrica solicitada no es finita o hay NaN, forzar ERROR_NAN
    has_nan = (
        (g_loss is not None and f_loss is None) or
        (d_loss is not None and f_d is None) or
        (l1_val is not None and f_l1 is None) or
        (edge_val is not None and f_edge is None)
    )

    if has_nan or (status_str == "COMPLETADO" and f_loss is None):
        status_str = "ERROR_NAN"
        print(f"[!] ADVERTENCIA CRITICA: Gradiente o perdida no finita detectada: g_loss={g_loss}, d_loss={d_loss}")

    hw = get_hardware_telemetry()

    if f_loss is not None and f_loss > 0.0 and math.isfinite(f_loss):
        if initial_loss is None or not math.isfinite(initial_loss) or initial_loss <= 0.0:
            initial_loss = f_loss
        if best_loss is None or not math.isfinite(best_loss) or best_loss <= 0.0 or f_loss < best_loss:
            best_loss = f_loss

        history = [h for h in history if h.get("epoch") != int(epoch)]
        history.append({
            "epoch": int(epoch),
            "loss": f_loss,
            "g_loss": f_loss,
            "d_loss": f_d if f_d is not None else 0.0,
            "l1_loss": f_l1 if f_l1 is not None else 0.0,
            "edge_loss": f_edge if f_edge is not None else 0.0,
            "lr": f_lr,
            "gpu_temp": hw["gpu_temp"],
            "elapsed_sec": elapsed
        })
        history.sort(key=lambda x: x["epoch"])

    loss_reduction_pct = 0.0
    if initial_loss is not None and f_loss is not None and initial_loss > 0.0 and math.isfinite(initial_loss) and math.isfinite(f_loss):
        loss_reduction_pct = round(((initial_loss - f_loss) / initial_loss) * 100.0, 1)

    sec_per_epoch = epoch_duration if (epoch_duration is not None and epoch_duration > 0.0) else (elapsed / max(1, epoch) if epoch > 0 else 0)
    rem_epochs = max(0, total_epochs - epoch)
    eta_sec = round(sec_per_epoch * rem_epochs)
    fps = round(1008.0 / max(0.1, sec_per_epoch), 2) if sec_per_epoch > 0 else 0.0

    resolved_quality = quality if isinstance(quality, dict) and quality else prev_quality
    if isinstance(guidance_runtime, dict):
        guidance_state = {**guidance_state, **guidance_runtime}
    guidance_quality = resolved_quality
    if reevaluate_guidance and isinstance(quality, dict) and quality:
        guidance_quality = _rolling_dataset_quality(
            history,
            quality,
            epoch=int(epoch),
            frame_quality=guidance_state.get("frame_quality", {}),
            expected_character_ids=guidance_state.get("dataset_character_ids", []),
            expected_samples=guidance_state.get("dataset_sample_count", previous_total_frames),
        )
    quality_guidance = prev_guidance if isinstance(prev_guidance, dict) else None
    if not ENABLE_QUALITY_GUIDANCE:
        quality_guidance = _disabled_guidance()
    elif reevaluate_guidance and isinstance(quality, dict) and quality:
        try:
            quality_guidance, guidance_state = _evaluate_observational_guidance(
                guidance_quality,
                epoch=int(epoch),
                training_metrics={
                    "g_loss": f_loss,
                    "d_loss": f_d,
                    "l1_loss": f_l1,
                    "edge_loss": f_edge,
                    "lr": f_lr,
                    "epoch": int(epoch),
                    "skipped_amp_steps": int(skipped_amp),
                },
                previous_state=guidance_state,
            )
            certification = quality_guidance.get("full_certification")
            if isinstance(certification, dict):
                expected_ids = guidance_state.get("dataset_character_ids") or []
                epoch_characters = int(certification.get("epoch_characters_audited") or 0)
                certification["passed"] = bool(
                    certification.get("due")
                    and certification.get("coverage_complete")
                    and epoch_characters >= max(1, len(expected_ids))
                    and str(quality_guidance.get("severity", "low")).lower() != "critical"
                )
        except Exception as guidance_error:
            # A broken audit must not corrupt the training state.
            quality_guidance = dict(quality_guidance or _disabled_guidance())
            quality_guidance.update({
                "enabled": True,
                "mode": "enforcing",
                "error": str(guidance_error),
                "training_modified": False,
            })

    if quality_guidance is None:
        quality_guidance = _disabled_guidance()

    # Couple the audit and its decision to the epoch record for trend/resume support.
    if f_loss is not None and f_loss > 0.0:
        for history_entry in history:
            if history_entry.get("epoch") == int(epoch):
                if isinstance(resolved_quality, dict):
                    history_entry["quality"] = resolved_quality
                history_entry["guidance"] = quality_guidance
                history_entry["quality_guidance"] = quality_guidance
                break

    status_data = {
        "epoch": int(epoch),
        "total_epochs": int(total_epochs),
        "phase": "Supervisado Pix2Pix + Kornia GPU + 8bit AdamW",
        "phase_id": 3,
        "status": status_str,
        "g_loss": f_loss,
        "d_loss": f_d,
        "l1_loss": f_l1,
        "edge_loss": f_edge,
        "best_loss": round(best_loss, 4) if (best_loss is not None and math.isfinite(best_loss) and best_loss > 0.0) else None,
        "initial_loss": round(initial_loss, 4) if (initial_loss is not None and math.isfinite(initial_loss) and initial_loss > 0.0) else None,
        "loss_reduction_pct": loss_reduction_pct,
        "lr": f_lr,
        "gpu_temp": hw["gpu_temp"],
        "cpu_temp": hw["cpu_temp"],
        "vram_gb": hw["vram_gb"],
        "vram_total_gb": hw["vram_total_gb"],
        "vram_pct": hw["vram_pct"],
        "gpu_util": hw["gpu_util"],
        "epoch_sec": round(sec_per_epoch, 1),
        "elapsed_sec": elapsed,
        "eta_sec": eta_sec,
        "fps": fps,
        "timestamp": time.strftime("%H:%M:%S"),
        "total_frames": int(dataset_layout.get("sample_count", previous_total_frames)) if isinstance(dataset_layout, dict) else int(previous_total_frames),
        "skipped_amp_steps": int(skipped_amp),
        "error_details": last_error,
        "snapshots": get_available_snapshots(),
        "past_eras": past_eras,
        "history": history,
        "quality": resolved_quality,
        "guidance": quality_guidance,
        "quality_guidance": quality_guidance,
        "guidance_state": guidance_state,
        "recovery": recovery,
        "recovery_events": recovery_events,
        "dataset_layout": dataset_layout,
        "best_quality_score": best_quality_score,
        "quality_checkpoint": quality_checkpoint,
        "session_id": session_id,
        "session_dir": session_dir,
        "session_start_epoch": session_start_epoch,
    }
    write_status_file(STATUS_FILE, status_data)
    return status_data

def save_checkpoint(file_path: Path, epoch: int, loss: float, best_loss: float,
                    generator: nn.Module, discriminator: nn.Module,
                    opt_g: Any = None, opt_d: Any = None, scaler_g: Any = None, scaler_d: Any = None,
                    sched_g: Any = None, sched_d: Any = None, is_best_model: bool = False,
                    guidance_state: Optional[Dict[str, Any]] = None,
                    session_id: Optional[str] = None, session_epoch: Optional[int] = None,
                    source_checkpoint: Optional[str] = None, qc_result: Optional[Dict[str, Any]] = None):
    """Guarda un checkpoint completo con formato unificado y preservacion estricta de metricas y estados."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    f_loss = float(loss) if (loss is not None and math.isfinite(float(loss))) else None
    f_best = float(best_loss) if (best_loss is not None and math.isfinite(float(best_loss))) else 999.0
    if guidance_state is None:
        stored_state = load_status_file(STATUS_FILE).get("guidance_state")
        guidance_state = stored_state if isinstance(stored_state, dict) else {}
    if is_best_model or file_path.name == "best_generator.pt":
        # Modelo ligero de inferencia (< 65MB, compatible con limites de GitHub y warm-start)
        state = {
            "epoch": int(epoch),
            "generator": generator.state_dict(),
            "loss": f_loss,
            "best_loss": f_best,
            "guidance_state": guidance_state,
            "session_id": session_id,
            "session_epoch": session_epoch,
        }
        temporary = file_path.with_suffix(file_path.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, file_path)
        return
    state = {
        "epoch": int(epoch),
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "loss": f_loss,
        "best_loss": f_best,
        "opt_g": opt_g.state_dict() if opt_g is not None else None,
        "opt_d": opt_d.state_dict() if opt_d is not None else None,
        "scaler_g": scaler_g.state_dict() if (scaler_g is not None and hasattr(scaler_g, "state_dict")) else None,
        "scaler_d": scaler_d.state_dict() if (scaler_d is not None and hasattr(scaler_d, "state_dict")) else None,
        "scheduler_g": sched_g.state_dict() if (sched_g is not None and hasattr(sched_g, "state_dict")) else None,
        "scheduler_d": sched_d.state_dict() if (sched_d is not None and hasattr(sched_d, "state_dict")) else None,
        "rng_state": torch.get_rng_state().cpu(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "guidance_state": guidance_state,
        "session_id": session_id,
        "session_epoch": session_epoch,
        "source_checkpoint": source_checkpoint,
        "qc_result": qc_result,
    }
    temporary = file_path.with_suffix(file_path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, file_path)


def _maybe_save_quality_checkpoint(
    status_data: Dict[str, Any],
    *,
    epoch: int,
    loss: float,
    best_loss: float,
    generator: nn.Module,
) -> Dict[str, Any]:
    data = dict(status_data)
    guidance = data.get("guidance") or {}
    vector = guidance.get("quality_vector", {})
    coverage = guidance.get("audit_coverage") if isinstance(guidance, dict) else None
    coverage_ready = bool(isinstance(coverage, dict) and coverage.get("ready_for_rollback"))
    assessment = assess_quality_checkpoint(vector)
    assessment["coverage_ready"] = coverage_ready
    previous = data.get("best_quality_score")
    previous_score = float(previous) if isinstance(previous, (int, float)) and math.isfinite(float(previous)) else None
    improved = bool(
        ENABLE_QUALITY_CHECKPOINT
        and assessment["eligible"]
        and coverage_ready
        and (previous_score is None or assessment["score"] > previous_score)
    )
    checkpoint_info = {**assessment, "enabled": bool(ENABLE_QUALITY_CHECKPOINT), "improved": improved}
    if improved:
        destination = CHECKPOINT_DIR / "best_quality_generator.pt"
        state = {
            "epoch": int(epoch),
            "generator": generator.state_dict(),
            "loss": float(loss),
            "best_loss": float(best_loss),
            "quality_score": assessment["score"],
            "quality_vector": dict(vector),
            "guidance_state": data.get("guidance_state", {}),
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, destination)
        data["best_quality_score"] = assessment["score"]
        checkpoint_info.update({"file": destination.name, "epoch": int(epoch)})
    elif isinstance(data.get("quality_checkpoint"), dict):
        prior_checkpoint = data["quality_checkpoint"]
        checkpoint_info.update({key: prior_checkpoint[key] for key in ("file", "epoch") if key in prior_checkpoint})
    data["quality_checkpoint"] = checkpoint_info
    certification = guidance.get("full_certification") if isinstance(guidance, dict) else None
    if isinstance(certification, dict):
        data["quality_certification"] = dict(certification)
    return data

def _tensor_to_preview_image(tensor, has_alpha=False):
    channels = 4 if has_alpha else 3
    array = ((tensor[:channels].permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, "RGBA" if has_alpha else "RGB")


def _fit_preview_cell(image):
    return place_in_cell(image, cell_w=128, cell_h=128)


def _generate_full_sheet_preview(generator, front, template_manager, character_palette):
    cells = []
    batch_size = 4
    total_frames = 96

    for start in range(0, total_frames, batch_size):
        frame_indices = range(start, min(start + batch_size, total_frames))
        poses = torch.stack([template_manager.get_frame_tensor(frame_idx) for frame_idx in frame_indices]).to(DEVICE)
        fronts = front.unsqueeze(0).expand(poses.shape[0], -1, -1, -1).to(DEVICE)
        conditions = torch.cat([fronts, poses], dim=1)
        with autocast(enabled=USE_AMP):
            predictions = generator(conditions)

        for prediction in predictions:
            raw_image = _tensor_to_preview_image(prediction, has_alpha=True)
            remapped = remap_image_to_palette(
                raw_image,
                character_palette,
                tolerance=35.0,
                binarize_alpha=True,
            )
            remapped = despeckle_chromatic_noise(remapped)
            cleaned = clean_orphan_pixels(remapped, min_connected_size=3, binarize=True)
            cells.append(_fit_preview_cell(cleaned))

    sheet = Image.new("RGBA", (128 * 8, 128 * 12), (16, 18, 26, 255))
    for frame_idx, cell in enumerate(cells):
        column = frame_idx % 8
        row = frame_idx // 8
        sheet.paste(cell, (column * 128, row * 128), cell)
    sheet.save(TRAIN_SAMPLES_DIR / "latest_preview.png")


def _generate_all_frame_comparison(generator, samples, template_manager, output_dir=None, epoch_label=None, batch_size=4, rows_per_chunk=32):
    output_dir = Path(output_dir or TRAIN_SAMPLES_DIR / "live_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mostrar una sola referencia por personaje/variante para evitar duplicar
    # todos los frames de la misma variante en la comparativa de auditoría.
    representative_by_char: dict[str, dict] = {}
    for sample in samples:
        char_id = str(sample["char_id"])
        current = representative_by_char.get(char_id)
        if current is None or int(sample["frame_idx"]) < int(current["frame_idx"]):
            representative_by_char[char_id] = sample

    ordered_samples = sorted(
        representative_by_char.values(),
        key=lambda sample: (str(sample["char_id"]), int(sample["frame_idx"]))
    )
    if not ordered_samples:
        return None

    columns = ["Referencia", "Pose", "IA cruda", "IA remapeada", "Ground truth"]
    chunk_rows = []
    chunk_paths = []
    generation = time.time_ns()
    epoch = int(epoch_label or 0)
    total_width = 128 * 5 + 35
    row_height = 150

    def save_chunk():
        chunk_index = len(chunk_paths)
        image = Image.new("RGBA", (total_width, row_height * len(chunk_rows)), (11, 13, 19, 255))
        draw = ImageDraw.Draw(image)
        for row_index, (label, cells) in enumerate(chunk_rows):
            y = row_index * row_height
            safe_label = label.encode("ascii", "replace").decode("ascii")
            draw.text((6, y + 2), safe_label, fill=(220, 226, 236, 255))
            image.paste(cells[0], (5, y + 20), cells[0])
            image.paste(cells[1], (138, y + 20), cells[1])
            image.paste(cells[2], (271, y + 20), cells[2])
            image.paste(cells[3], (404, y + 20), cells[3])
            image.paste(cells[4], (537, y + 20), cells[4])
        filename = f"comparison_epoch_{epoch:03d}_{generation}_part_{chunk_index:03d}.png"
        image.save(output_dir / filename)
        chunk_paths.append(filename)
        chunk_rows.clear()

    generator.eval()
    with torch.no_grad():
        for start in range(0, len(ordered_samples), batch_size):
            batch = ordered_samples[start:start + batch_size]
            poses = torch.stack([template_manager.get_frame_tensor(int(sample["frame_idx"])) for sample in batch])
            fronts = torch.stack([sample["front_tensor"] for sample in batch])
            conditions = torch.cat([fronts, poses], dim=1).to(DEVICE)
            with autocast(enabled=USE_AMP):
                predictions = generator(conditions)

            for sample, pose, prediction in zip(batch, poses, predictions):
                front_image = _tensor_to_preview_image(sample["front_tensor"])
                front_cell = _fit_preview_cell(front_image)
                pose_cell = _fit_preview_cell(_tensor_to_preview_image(pose))
                raw_cell = _fit_preview_cell(_tensor_to_preview_image(prediction, has_alpha=True))
                palette = extract_character_palette(front_cell, include_props=True)
                remapped = remap_image_to_palette(raw_cell, palette, tolerance=35.0, binarize_alpha=True)
                remapped = despeckle_chromatic_noise(remapped)
                remapped = clean_orphan_pixels(remapped, min_connected_size=3, binarize=True)
                target_cell = _fit_preview_cell(_tensor_to_preview_image(sample["target_tensor"], has_alpha=True))
                label = f"{sample['char_id']} / frame {int(sample['frame_idx']):03d}"
                chunk_rows.append((label, (front_cell, pose_cell, raw_cell, remapped, target_cell)))
                if len(chunk_rows) >= rows_per_chunk:
                    save_chunk()

    if chunk_rows:
        save_chunk()

    manifest = {
        "epoch": epoch,
        "generation": generation,
        "row_count": len(ordered_samples),
        "complete": True,
        "columns": columns,
        "chunks": chunk_paths,
        "rows": [
            {"char_id": str(sample["char_id"]), "frame_idx": int(sample["frame_idx"])}
            for sample in ordered_samples
        ],
    }
    manifest_path = output_dir / "manifest.json"
    temporary_path = output_dir / f"manifest_{generation}.tmp"
    temporary_path.write_text(json.dumps(manifest, ensure_ascii=True), encoding="utf-8")
    os.replace(temporary_path, manifest_path)

    current_paths = set(chunk_paths)
    for stale_path in output_dir.glob("comparison_epoch_*_part_*.png"):
        if stale_path.name not in current_paths:
            try:
                stale_path.unlink()
            except OSError:
                pass
    return manifest


def _audit_training_prediction(generated, target, palette):
    """Compare one raw prediction with its exact dataset target."""
    qc = PixelArtEnhancer.analyze_quality(generated, palette=palette, target_img=target)
    anatomy_metrics = compute_anatomical_metrics(generated, target)
    anatomy_metrics["anatomy_geometry"] = anatomy_metrics.pop("cuerpo_precision", None)
    qc.update(anatomy_metrics)
    qc.update(Phase3CriticalReviewer.audit_frame(generated))
    return apply_strict_visual_metrics(qc, generated, target)


def _aggregate_training_quality(audits, *, include_sample_metrics=True):
    """Use a conservative percentile and retain mean/worst evidence."""
    valid = [item for item in audits if isinstance(item.get("metrics"), dict) and item["metrics"]]
    if not valid:
        return {}
    metrics_list = [item["metrics"] for item in valid]
    conservative_keys = {
        "score_total", "cuerpo_precision", "silueta_iou_real", "silhouette_iou", "pose_alignment",
        "nitidez_bordes", "pureza_alfa", "ruido_huerfano", "fidelidad_paleta", "micro_detalles",
        "preservacion_tatuajes", "pelo_gorro", "gestos_ojos", "ropa_delantal", "tatuajes_brazos",
        "objetos_utensilios", "zapatos_pies", "score_critico", "pureza_bordes", "definicion_tinta",
        "profundidad_sombra", "riqueza_paleta", "detalle_facial", "anatomy_geometry",
        "strict_global", "strict_anatomy", "strict_face", "strict_silhouette", "strict_visual_noise",
        "strict_whole_structure",
    }
    count_keys = {"defectos_cuerpo", "total_px_cuerpo"}
    aggregated = {}
    for key in set().union(*(metrics.keys() for metrics in metrics_list)):
        values = [
            float(metrics[key]) for metrics in metrics_list
            if isinstance(metrics.get(key), (int, float))
            and not isinstance(metrics.get(key), bool)
            and math.isfinite(float(metrics[key]))
        ]
        if not values:
            continue
        if key in conservative_keys:
            aggregated[key] = round(float(np.percentile(values, 25)), 4)
        elif key in count_keys:
            aggregated[key] = int(sum(values))
        else:
            aggregated[key] = round(float(np.mean(values)), 4)

    global_scores = [float(item["metrics"]["score_total"]) for item in valid if item["metrics"].get("score_total") is not None]
    aggregated.update({
        "comparison_source": "dataset_targets",
        "comparison_method": "percentil_25_conservador",
        "evaluated_sample_count": len(valid),
        "quality_average": round(float(np.mean(global_scores)), 4) if global_scores else None,
        "quality_worst": round(float(min(global_scores)), 4) if global_scores else None,
        "evaluated_samples": [
            {
                "character_id": item["character_id"],
                "frame_idx": int(item["frame_idx"]),
                "score_total": item["metrics"].get("score_total"),
                "face": item["metrics"].get("strict_face"),
                "anatomy": item["metrics"].get("strict_anatomy"),
                "silhouette": item["metrics"].get("strict_silhouette"),
                "visual_noise": item["metrics"].get("strict_visual_noise"),
            }
            for item in valid
        ],
    })
    if include_sample_metrics:
        aggregated["evaluated_sample_metrics"] = [
            {
                "character_id": str(item["character_id"]),
                "frame_idx": int(item["frame_idx"]),
                "metrics": dict(item["metrics"]),
            }
            for item in valid
        ]
    aggregated["quality_guide"] = PixelArtEnhancer.build_quality_guide(aggregated)
    return aggregated


def _frame_tracker_summary(frame_quality, *, expected_samples, expected_characters):
    records = frame_quality if isinstance(frame_quality, dict) else {}
    valid = [record for record in records.values() if isinstance(record, dict)]
    characters = {str(record.get("char_id", "unknown")) for record in valid}
    qualities = [
        float(record["quality"])
        for record in valid
        if isinstance(record.get("quality"), (int, float)) and math.isfinite(float(record["quality"]))
    ]

    def metric_percentile(metric):
        values = [
            float(record["metrics"][metric])
            for record in valid
            if isinstance(record.get("metrics"), dict)
            and isinstance(record["metrics"].get(metric), (int, float))
            and math.isfinite(float(record["metrics"][metric]))
        ]
        return round(float(np.percentile(values, 25)), 4) if values else None

    sample_total = max(0, int(expected_samples or 0))
    character_total = max(0, int(expected_characters or 0))
    return {
        "observed_frames": len(valid),
        "expected_frames": sample_total,
        "frame_coverage": round(min(1.0, len(valid) / max(1, sample_total)), 4),
        "observed_characters": len(characters),
        "expected_characters": character_total,
        "character_coverage": round(min(1.0, len(characters) / max(1, character_total)), 4),
        "quality_p25": round(float(np.percentile(qualities, 25)), 4) if qualities else None,
        "quality_average": round(float(np.mean(qualities)), 4) if qualities else None,
        "quality_worst": round(float(min(qualities)), 4) if qualities else None,
        "color_p25": metric_percentile("color"),
        "alpha_p25": metric_percentile("alpha"),
        "silhouette_p25": metric_percentile("silhouette"),
    }


def _rolling_dataset_quality(
    history,
    current_quality,
    *,
    epoch,
    frame_quality,
    expected_character_ids,
    expected_samples,
):
    """Aggregate stratified deep audits and the all-frame training tracker."""
    first_epoch = max(1, int(epoch) - QUALITY_AUDIT_WINDOW_EPOCHS + 1)
    audit_records = []
    for entry in history if isinstance(history, list) else []:
        if not isinstance(entry, dict) or int(entry.get("epoch", 0)) < first_epoch:
            continue
        prior_quality = entry.get("quality")
        if isinstance(prior_quality, dict):
            audit_records.extend(prior_quality.get("evaluated_sample_metrics", []))
    if isinstance(current_quality, dict):
        audit_records.extend(current_quality.get("evaluated_sample_metrics", []))

    latest_by_sample = {}
    for record in audit_records:
        if not isinstance(record, dict) or not isinstance(record.get("metrics"), dict):
            continue
        key = (str(record.get("character_id", "unknown")), int(record.get("frame_idx", -1)))
        latest_by_sample[key] = {
            "character_id": key[0],
            "frame_idx": key[1],
            "metrics": dict(record["metrics"]),
        }
    unique_records = list(latest_by_sample.values())
    rolling = _aggregate_training_quality(unique_records, include_sample_metrics=False)
    expected_ids = {str(value) for value in expected_character_ids or []}
    frames_by_character = {character_id: set() for character_id in expected_ids}
    for character_id, frame_idx in latest_by_sample:
        if character_id in frames_by_character:
            frames_by_character[character_id].add(frame_idx)
        elif not expected_ids:
            frames_by_character.setdefault(character_id, set()).add(frame_idx)
    covered_characters = sum(bool(frames) for frames in frames_by_character.values())
    expected_character_count = len(expected_ids) or len(frames_by_character)
    min_frames = min((len(frames) for frames in frames_by_character.values()), default=0)
    tracker = _frame_tracker_summary(
        frame_quality,
        expected_samples=expected_samples,
        expected_characters=expected_character_count,
    )
    character_coverage = round(covered_characters / max(1, expected_character_count), 4)
    ready = bool(
        bool(expected_ids)
        and expected_character_count > 0
        and character_coverage >= 1.0
        and min_frames >= QUALITY_AUDIT_MIN_FRAMES_PER_CHARACTER
        and tracker["frame_coverage"] >= QUALITY_TRACKER_MIN_COVERAGE
        and int(tracker["expected_frames"] or 0) > 0
    )
    if not rolling:
        fallback = dict(current_quality or {})
        fallback["audit_coverage"] = {
            "window_start_epoch": first_epoch,
            "window_end_epoch": int(epoch),
            "window_epochs": QUALITY_AUDIT_WINDOW_EPOCHS,
            "deep_samples": len(unique_records),
            "characters_covered": covered_characters,
            "characters_total": expected_character_count,
            "character_coverage": character_coverage,
            "minimum_frames_per_character": min_frames,
            "required_frames_per_character": QUALITY_AUDIT_MIN_FRAMES_PER_CHARACTER,
            "tracker_frames_observed": tracker["observed_frames"],
            "tracker_frames_total": tracker["expected_frames"],
            "tracker_frame_coverage": tracker["frame_coverage"],
            "ready_for_rollback": False,
        }
        fallback["full_certification"] = {
            "due": int(epoch) % 10 == 0,
            "coverage_complete": False,
            "characters_certified": covered_characters,
            "frames_tracked": tracker["observed_frames"],
        }
        return fallback
    coverage = {
        "window_start_epoch": first_epoch,
        "window_end_epoch": int(epoch),
        "window_epochs": QUALITY_AUDIT_WINDOW_EPOCHS,
        "deep_samples": len(unique_records),
        "characters_covered": covered_characters,
        "characters_total": expected_character_count,
        "character_coverage": character_coverage,
        "minimum_frames_per_character": min_frames,
        "required_frames_per_character": QUALITY_AUDIT_MIN_FRAMES_PER_CHARACTER,
        "tracker_frames_observed": tracker["observed_frames"],
        "tracker_frames_total": tracker["expected_frames"],
        "tracker_frame_coverage": tracker["frame_coverage"],
        "ready_for_rollback": ready,
    }

    # The broad tracker may lower a deep-audit metric but never inflate it.
    if tracker["quality_p25"] is not None and isinstance(rolling.get("score_total"), (int, float)):
        rolling["score_total"] = round(min(float(rolling["score_total"]), tracker["quality_p25"]), 4)
    for quality_key, tracker_key in (
        ("silueta_iou_real", "silhouette_p25"),
        ("pureza_alfa", "alpha_p25"),
    ):
        tracker_value = tracker.get(tracker_key)
        if tracker_value is not None and isinstance(rolling.get(quality_key), (int, float)):
            rolling[quality_key] = round(min(float(rolling[quality_key]), float(tracker_value)), 4)

    rolling.update({
        "comparison_source": "rolling_dataset_targets_plus_frame_tracker",
        "comparison_method": "percentil_25_ventana_estratificada",
        "audit_coverage": coverage,
        "frame_tracker_summary": tracker,
        "full_certification": {
            "due": int(epoch) % 10 == 0,
            "coverage_complete": ready,
            "characters_certified": covered_characters,
            "frames_tracked": tracker["observed_frames"],
            "epoch_characters_audited": len({
                str(record.get("character_id"))
                for record in (current_quality or {}).get("evaluated_sample_metrics", [])
                if isinstance(record, dict)
            }) if isinstance(current_quality, dict) else 0,
        },
    })
    rolling["quality_guide"] = PixelArtEnhancer.build_quality_guide(rolling)
    return rolling


def _select_quality_sample_indices(samples, epoch_label=None, limit=4):
    """Rotate characters and frames so repeated audits cover the full dataset."""
    grouped = {}
    for index, sample in enumerate(samples):
        grouped.setdefault(str(sample.get("char_id", "unknown")), []).append(index)
    if not grouped:
        return []
    character_ids = list(grouped)
    cursor = max(0, int(epoch_label or 1) - 1)
    selected = []
    for offset in range(min(int(limit), len(character_ids))):
        character_id = character_ids[(cursor * int(limit) + offset) % len(character_ids)]
        character_samples = grouped[character_id]
        selected.append(character_samples[cursor % len(character_samples)])
    return selected


def generate_preview(generator, dataset, epoch_label=None, *, full_certification=False):
    generator.eval()
    with torch.no_grad():
        character_count = len({str(sample.get("char_id", "unknown")) for sample in dataset.samples})
        audit_limit = character_count if full_certification else QUALITY_AUDIT_SAMPLES_PER_EPOCH
        sample_indices = _select_quality_sample_indices(
            dataset.samples,
            epoch_label=epoch_label,
            limit=audit_limit,
        )
        from pixel_ai_engine.dataset import TemplateManager
        tmpl_path = PROJECT_ROOT / "dataset_moldes" / "plantilla de los spritesheets.png"
        if not tmpl_path.exists():
            tmpl_path = PROJECT_ROOT / "plantilla de los spritesheets.png"
        tm = TemplateManager(tmpl_path, MODEL_RESOLUTION, 12, 8, 1024, 1536)

        cards = []
        quality_audits = []

        for s_idx in sample_indices:
            # Evaluar siempre sobre la muestra fija sin aumentos de datos aleatorios
            s = dataset.samples[s_idx]
            front = s["front_tensor"]
            f_idx = s["frame_idx"]
            target = s["target_tensor"]
            char_id = s["char_id"]
            pose = tm.get_frame_tensor(f_idx)

            cond = torch.cat([front, pose], dim=0).unsqueeze(0).to(DEVICE)
            with autocast(enabled=USE_AMP):
                out = generator(cond).squeeze(0)

            # 1. Frontal Chibi (Referencia real original sin aumentos)
            front_image = _tensor_to_preview_image(front)
            f_pil = _fit_preview_cell(front_image)

            # 2. Pose (Molde geométrico)
            pose_image = _tensor_to_preview_image(pose)
            p_pil = _fit_preview_cell(pose_image)

            # 3. Prediccion IA Cruda (Sin retoques - para auditoría transparente de fallos)
            raw_prediction = _tensor_to_preview_image(out, has_alpha=True)
            raw_pred_pil = _fit_preview_cell(raw_prediction)
            
            # 4. Predicción IA con Remapeo de Paleta + Filtro Morfológico Anti-Hollín + Alfa Puro
            char_palette = extract_character_palette(f_pil, include_props=True)
            snapped_pil = remap_image_to_palette(raw_pred_pil, char_palette, tolerance=35.0, binarize_alpha=True)
            snapped_pil = despeckle_chromatic_noise(snapped_pil)
            pred_pil = clean_orphan_pixels(snapped_pil, min_connected_size=3, binarize=True)

            # 5. Ground Truth Real
            target_image = _tensor_to_preview_image(target, has_alpha=True)
            tgt_pil = _fit_preview_cell(target_image)

            try:
                quality_audits.append({
                    "character_id": char_id,
                    "frame_idx": f_idx,
                    "metrics": _audit_training_prediction(raw_pred_pil, tgt_pil, char_palette),
                })
            except Exception as audit_error:
                print(f"[QC] Comparación omitida para {char_id}/frame_{int(f_idx):03d}: {audit_error}", flush=True)

            # Tira comparativa con 5 paneles: Frontal | Pose | IA Cruda | IA Remapeada | Ground Truth
            strip = Image.new("RGBA", (128 * 5 + 35, 128 + 20), (16, 18, 26, 255))
            strip.paste(f_pil, (5, 10))
            strip.paste(p_pil, (128 + 10, 10))
            strip.paste(raw_pred_pil, (256 + 15, 10))
            strip.paste(pred_pil, (384 + 20, 10))
            strip.paste(tgt_pil, (512 + 25, 10))
            cards.append(strip)

        total_w = cards[0].width
        total_h = sum(c.height for c in cards) + 30
        comp_img = Image.new("RGBA", (total_w, total_h), (11, 13, 19, 255))
        y = 10
        for c in cards:
            comp_img.paste(c, (0, y))
            y += c.height + 5

        comp_img.save(TRAIN_SAMPLES_DIR / "latest_detail_comparison.png")
        if full_certification and epoch_label is not None:
            comp_img.save(AUDIT_DIR / f"preview_epoch_{epoch_label:03d}.png")

        sheet_front = dataset.samples[sample_indices[0]]["front_tensor"]
        sheet_palette = extract_character_palette(_fit_preview_cell(_tensor_to_preview_image(sheet_front)), include_props=True)
        _generate_full_sheet_preview(generator, sheet_front, tm, sheet_palette)
        if full_certification:
            certified_samples = [dataset.samples[index] for index in sample_indices]
            _generate_all_frame_comparison(generator, certified_samples, tm, epoch_label=epoch_label)

        # Every percentage comes from prediction versus exact dataset target.
        return _aggregate_training_quality(quality_audits)

def train_supervised_model(epochs: int = 150, batch_size: int = 4, lr: float = 1.5e-4, mode: str = "resume", respawn_epoch: int = None):
    # Limpiar banderas anteriores
    if STOP_FLAG_FILE.exists():
        try: STOP_FLAG_FILE.unlink()
        except Exception: pass
    if PAUSE_FLAG_FILE.exists():
        try: PAUSE_FLAG_FILE.unlink()
        except Exception: pass

    automatic_recovery = None
    if mode == "resume" and respawn_epoch is None:
        try:
            automatic_recovery = repair_current_training_state(status_name="RECUPERANDO")
        except Exception as recovery_error:
            print(f"[!] No se pudo preparar la ruta automatica de recuperacion: {recovery_error}")

    status_at_start = load_status_file(STATUS_FILE)
    if mode == "resume" and str(status_at_start.get("status", "")).upper() == "DATASET_ACTUALIZADO":
        raise RuntimeError(
            "El dataset cambió de escala. Usa modo START para abrir una era nueva compatible; "
            "RESUME conservaría métricas y optimizadores de la escala anterior."
        )

    dataset = SupervisedTensorDataset(CACHE_PATH, augment=True)
    dataset_character_ids = sorted({
        str(item.get("char_id", "unknown"))
        for item in _dataset_sample_descriptors(dataset)
    })
    starting_guidance_state = status_at_start.get("guidance_state", {}) if isinstance(status_at_start.get("guidance_state"), dict) else {}
    frame_quality_tracker = FrameQualityTracker(starting_guidance_state.get("frame_quality", {}))
    current_sampling_plan = _build_sampling_plan(
        dataset,
        frame_quality_tracker.export(),
        status_at_start.get("guidance") if isinstance(status_at_start.get("guidance"), dict) else None,
    )
    current_loss_plan = _build_loss_plan(
        starting_guidance_state,
        status_at_start.get("guidance") if isinstance(status_at_start.get("guidance"), dict) else None,
    )

    from pixel_ai_engine.dataset import TemplateManager
    tmpl_path = PROJECT_ROOT / "dataset_moldes" / "plantilla de los spritesheets.png"
    if not tmpl_path.exists():
        tmpl_path = PROJECT_ROOT / "plantilla de los spritesheets.png"
    tmpl_mgr = TemplateManager(tmpl_path, MODEL_RESOLUTION, 12, 8, 1024, 1536)

    generator = PixelArtUNetGenerator().to(DEVICE)
    discriminator = PixelArtPatchDiscriminator().to(DEVICE)

    start_epoch = 1
    best_loss = None
    loaded_ckpt = None

    # Logica de Respawn o Reanudacion
    if respawn_epoch is not None and respawn_epoch > 0:
        respawn_candidate = None
        if SESSIONS_DIR.exists():
            for sdir in sorted(SESSIONS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if sdir.is_dir() and (sdir / f"epoch_{respawn_epoch:03d}.pt").exists():
                    respawn_candidate = sdir / f"epoch_{respawn_epoch:03d}.pt"
                    break
        if respawn_candidate is None:
            respawn_candidate = SNAPSHOTS_DIR / f"checkpoint_epoch_{respawn_epoch:03d}.pt"
            if not respawn_candidate.exists():
                respawn_candidate = SNAPSHOTS_DIR / f"checkpoint_epoch_{respawn_epoch}.pt"
        if respawn_candidate and respawn_candidate.exists():
            loaded_ckpt = torch.load(respawn_candidate, map_location=DEVICE)
            generator.load_state_dict(loaded_ckpt["generator"])
            if "discriminator" in loaded_ckpt:
                discriminator.load_state_dict(loaded_ckpt["discriminator"])
            start_epoch = respawn_epoch + 1
            best_loss = loaded_ckpt.get("best_loss", loaded_ckpt.get("loss", None))
            print(f"\n[RESPAWN] == RESPAWN EXITOSO: Regresando al punto de guardado EPOCA {respawn_epoch} ({respawn_candidate.name}) ==\n")
        else:
            print(f"[!] Aviso: No se encontro snapshot para epoca {respawn_epoch}. Iniciando estandar.")
    elif mode == "resume":
        ckpt_candidate = automatic_recovery.get("checkpoint") if automatic_recovery else None
        if ckpt_candidate is None:
            ckpt_candidate = _active_recovery_checkpoint(status_at_start)
        if ckpt_candidate is None:
            ckpt_candidate = CHECKPOINT_DIR / "latest_checkpoint.pt"
        if not ckpt_candidate.exists():
            ckpt_candidate = CHECKPOINT_DIR / "best_generator.pt"
        if ckpt_candidate.exists():
            try:
                loaded_ckpt = torch.load(ckpt_candidate, map_location=DEVICE)
                if isinstance(loaded_ckpt, dict) and "generator" in loaded_ckpt:
                    generator.load_state_dict(loaded_ckpt["generator"])
                    if "discriminator" in loaded_ckpt:
                        discriminator.load_state_dict(loaded_ckpt["discriminator"])
                    start_epoch = loaded_ckpt.get("epoch", 0) + 1
                    if automatic_recovery:
                        best_loss = loaded_ckpt.get("loss", loaded_ckpt.get("best_loss", None))
                        print(
                            f"[RECUPERACION] Ruta segura activa: {Path(ckpt_candidate).name} "
                            f"-> reanudando desde epoca {start_epoch}"
                        )
                    else:
                        best_loss = loaded_ckpt.get("best_loss", None)
                        print(f"[OK] Reanudando entrenamiento desde epoca {start_epoch} ({Path(ckpt_candidate).name})")
            except Exception as e:
                print(f"[!] Error al reanudar checkpoint: {e}")
    elif mode == "start":
        # Restablecer historial de entrenamiento en STATUS_FILE para nuevo experimento,
        # pero archivando la era anterior en past_eras para comparar lineas del tiempo
        if STATUS_FILE.exists():
            try:
                with open(STATUS_FILE, "r", encoding="utf-8") as f:
                    sdata = json.load(f)
                old_hist = sdata.get("history", [])
                past_eras = sdata.get("past_eras", [])
                if old_hist and len(old_hist) > 0:
                    era_idx = len(past_eras) + 1
                    era_quality_summary = _quality_summary_from_history(old_hist)
                    past_eras.append({
                        "era": era_idx,
                        "name": f"Era {era_idx}",
                        "timestamp": sdata.get("timestamp", time.strftime("%H:%M:%S")),
                        "epochs": sdata.get("epoch", len(old_hist)),
                        "initial_loss": sdata.get("initial_loss"),
                        "best_loss": sdata.get("best_loss"),
                        "history": old_hist,
                        "quality_summary": era_quality_summary,
                        **era_quality_summary,
                    })
                    print(f"[HISTORIAL] Era {era_idx} archivada con {len(old_hist)} épocas para comparativa en la línea del tiempo.")
                sdata["past_eras"] = past_eras
                sdata["history"] = []
                sdata["initial_loss"] = None
                sdata["best_loss"] = None
                sdata["loss_reduction_pct"] = 0.0
                sdata["guidance_state"] = {}
                sdata["guidance"] = _disabled_guidance()
                sdata["quality_guidance"] = sdata["guidance"]
                sdata["epoch"] = 0
                sdata["status"] = "INICIANDO"
                write_status_file(STATUS_FILE, sdata)
            except Exception as e:
                print(f"[!] Aviso al archivar era previa: {e}")

        # Warm start: Si existe best_generator o base_generator, iniciar desde pesos entrenados
        warm_ckpt = _active_recovery_checkpoint(status_at_start)
        if warm_ckpt is None:
            quality_warm_start = CHECKPOINT_DIR / "best_quality_generator.pt"
            warm_ckpt = quality_warm_start if quality_warm_start.exists() else CHECKPOINT_DIR / "best_generator.pt"
        if not warm_ckpt.exists():
            warm_ckpt = CHECKPOINT_DIR / "base_generator_16x4.pt"
        if warm_ckpt.exists():
            try:
                ckpt = torch.load(warm_ckpt, map_location=DEVICE)
                gen_state = ckpt.get("generator", ckpt) if isinstance(ckpt, dict) else ckpt
                generator.load_state_dict(gen_state, strict=False)
                print(f"[OK] Warm Start: Inicializando generador desde pesos existentes: {warm_ckpt.name}")
            except Exception as e:
                print(f"[!] Aviso: No se pudo cargar warm start: {e}. Iniciando desde inicializacion normal.")
        best_loss = 999.0
        print("[NUEVA ERA] La métrica best_loss se reinicia; los pesos previos se usan sólo como base visual.")

    # -------------------------------------------------------------
    # GESTION AISLADA DE SESION (Checkpoints por carpeta de sesion)
    # -------------------------------------------------------------
    current_session_id = None
    current_session_dir = None
    session_start_epoch = start_epoch
    if mode == "start":
        current_session_id = generate_session_id()
        current_session_dir = get_session_dir(current_session_id, CHECKPOINT_DIR)
        warm_ckpt_str = None
        if warm_ckpt and hasattr(warm_ckpt, "exists") and warm_ckpt.exists():
            try:
                warm_ckpt_str = str(warm_ckpt.relative_to(PROJECT_ROOT)).replace("\\", "/")
            except ValueError:
                warm_ckpt_str = str(warm_ckpt).replace("\\", "/")
        init_session(
            current_session_dir,
            current_session_id,
            start_epoch=1,
            target_epochs=epochs,
            mode="start",
            source_checkpoint=warm_ckpt_str,
        )
        print(f"[SESIÓN] Nueva sesión iniciada: {current_session_id} en {current_session_dir.name}")
    elif mode == "resume":
        prev_session_id = status_at_start.get("session_id")
        if (
            prev_session_id
            and get_session_dir(prev_session_id, CHECKPOINT_DIR).exists()
            and respawn_epoch is None
            and not automatic_recovery
        ):
            current_session_id = prev_session_id
            current_session_dir = get_session_dir(current_session_id, CHECKPOINT_DIR)
            session_start_epoch = int(status_at_start.get("session_start_epoch", start_epoch))
        else:
            current_session_id = generate_session_id()
            current_session_dir = get_session_dir(current_session_id, CHECKPOINT_DIR)
            session_start_epoch = start_epoch
            source_ckpt_str = None
            if ckpt_candidate and hasattr(ckpt_candidate, "exists") and ckpt_candidate.exists():
                try:
                    source_ckpt_str = str(ckpt_candidate.relative_to(PROJECT_ROOT)).replace("\\", "/")
                except ValueError:
                    source_ckpt_str = str(ckpt_candidate).replace("\\", "/")
            elif ckpt_candidate:
                source_ckpt_str = str(ckpt_candidate).replace("\\", "/")
            init_session(
                current_session_dir,
                current_session_id,
                start_epoch=start_epoch,
                target_epochs=epochs,
                mode="resume",
                source_checkpoint=source_ckpt_str,
            )
        print(f"[SESIÓN] Sesión activa: {current_session_id} (Época inicial: {start_epoch})")

    # Old checkpoints remain valid. If a modern checkpoint carries observational
    # history and the status file does not, restore only that metadata.
    if mode == "resume" and isinstance(loaded_ckpt, dict):
        restored_guidance_state = _guidance_state_from_checkpoint(
            status_at_start,
            loaded_ckpt,
            prefer_checkpoint=respawn_epoch is not None,
        )
        if restored_guidance_state is not None and status_at_start.get("guidance_state") != restored_guidance_state:
            status_at_start = dict(status_at_start)
            status_at_start["guidance_state"] = restored_guidance_state
            persisted_status = load_status_file(STATUS_FILE)
            persisted_status["guidance_state"] = restored_guidance_state
            write_status_file(STATUS_FILE, persisted_status)

    # Rebuild runtime controllers after checkpoint metadata restoration.  This
    # keeps status-only, modern checkpoint and legacy checkpoint resumes
    # behaviorally equivalent.
    restored_runtime_state = (
        status_at_start.get("guidance_state", {})
        if isinstance(status_at_start.get("guidance_state"), dict)
        else {}
    )
    frame_quality_tracker = FrameQualityTracker(restored_runtime_state.get("frame_quality", {}))
    current_sampling_plan = _build_sampling_plan(
        dataset,
        frame_quality_tracker.export(),
        status_at_start.get("guidance") if isinstance(status_at_start.get("guidance"), dict) else None,
    )
    current_loss_plan = _build_loss_plan(
        restored_runtime_state,
        status_at_start.get("guidance") if isinstance(status_at_start.get("guidance"), dict) else None,
    )

    # Recuperar best_loss de best_generator.pt para archivos antiguos compatibles
    if mode != "start" and (best_loss is None or (isinstance(best_loss, (int, float)) and best_loss >= 990.0)):
        best_ckpt_file = CHECKPOINT_DIR / "best_generator.pt"
        if best_ckpt_file.exists():
            try:
                b_data = torch.load(best_ckpt_file, map_location="cpu")
                if isinstance(b_data, dict):
                    cand_best = b_data.get("best_loss", b_data.get("loss", 999.0))
                    if cand_best is not None and math.isfinite(cand_best) and float(cand_best) > 0.0:
                        best_loss = float(cand_best)
                        print(f"[OK] best_loss historico recuperado de best_generator.pt: {best_loss:.4f}")
            except Exception:
                pass
    if best_loss is None or not math.isfinite(best_loss) or best_loss <= 0.0:
        best_loss = 999.0

    total_target_epochs = (start_epoch - 1) + epochs if mode == "resume" and respawn_epoch is None else (start_epoch - 1 + epochs if respawn_epoch else epochs)
    effective_lr = float(lr)
    if automatic_recovery:
        effective_lr = min(effective_lr, float(automatic_recovery.get("preferred_lr") or effective_lr))
        recovery_status = load_status_file(STATUS_FILE)
        recovery_status["total_epochs"] = int(total_target_epochs)
        recovery_status["status"] = "RECUPERANDO"
        recovery_status["lr"] = effective_lr
        recovery_status["timestamp"] = time.strftime("%H:%M:%S")
        write_status_file(STATUS_FILE, recovery_status)

    print("=" * 70)
    print("  ENTRENAMIENTO SUPERVISADO CON KORNIA GPU + 8-BIT ADAMW + RESPAWN")
    print(f"  Modo: {mode.upper()} | Epocas: {start_epoch} a {total_target_epochs} | Batch: {batch_size} | LR: {effective_lr}")
    print(f"  Muestras: {len(dataset)} pares Ground-Truth | Dispositivo: {DEVICE} | Mejor Loss Inicial: {best_loss:.4f}")
    print("=" * 70)

    # Optimizadores 8-bit AdamW de Forge (solo admiten tensores CUDA).
    try:
        if DEVICE.type != "cuda":
            raise RuntimeError("BitsAndBytes requiere CUDA")
        import bitsandbytes as bnb
        opt_g = bnb.optim.AdamW8bit(generator.parameters(), lr=effective_lr, betas=(0.5, 0.999), weight_decay=1e-4)
        opt_d = bnb.optim.AdamW8bit(discriminator.parameters(), lr=effective_lr * 0.5, betas=(0.5, 0.999), weight_decay=1e-4)
        print("[OK] Optimizador BitsAndBytes 8-bit AdamW activo (VRAM: ~1.5 GB)")
    except Exception as e:
        opt_g = torch.optim.Adam(generator.parameters(), lr=effective_lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=effective_lr * 0.5, betas=(0.5, 0.999))
        print(f"[!] Optimizador estandar PyTorch Adam activo: {e}")

    scaler_g = GradScaler(enabled=USE_AMP, init_scale=2048.0)
    scaler_d = GradScaler(enabled=USE_AMP, init_scale=2048.0)

    # Restaurar estados de optimizadores, escaladores y RNG si estan disponibles en el checkpoint
    if loaded_ckpt is not None:
        if "opt_g" in loaded_ckpt and loaded_ckpt["opt_g"] is not None:
            try:
                opt_g.load_state_dict(loaded_ckpt["opt_g"])
                print("[OK] Estado de optimizador opt_g restaurado con exito.")
            except Exception as e:
                print(f"[!] Aviso al restaurar estado de opt_g: {e}")
        if "opt_d" in loaded_ckpt and loaded_ckpt["opt_d"] is not None:
            try:
                opt_d.load_state_dict(loaded_ckpt["opt_d"])
                print("[OK] Estado de optimizador opt_d restaurado con exito.")
            except Exception as e:
                print(f"[!] Aviso al restaurar estado de opt_d: {e}")
        if automatic_recovery:
            for param_group in opt_g.param_groups:
                param_group["lr"] = effective_lr
            for param_group in opt_d.param_groups:
                param_group["lr"] = effective_lr * 0.5
            print(f"[RECUPERACION] Learning rate reducido y aplicado: G={effective_lr:.8f} | D={effective_lr * 0.5:.8f}")
        if "scaler_g" in loaded_ckpt and loaded_ckpt["scaler_g"] is not None and scaler_g is not None:
            try: scaler_g.load_state_dict(loaded_ckpt["scaler_g"])
            except Exception: pass
        if "scaler_d" in loaded_ckpt and loaded_ckpt["scaler_d"] is not None and scaler_d is not None:
            try: scaler_d.load_state_dict(loaded_ckpt["scaler_d"])
            except Exception: pass
        if "rng_state" in loaded_ckpt and loaded_ckpt["rng_state"] is not None:
            try:
                rng = loaded_ckpt["rng_state"]
                if hasattr(rng, "cpu"):
                    rng = rng.cpu()
                if isinstance(rng, torch.Tensor) and rng.dtype != torch.uint8:
                    rng = rng.to(torch.uint8)
                torch.set_rng_state(rng)
                print("[OK] Estado CPU RNG restaurado.")
            except Exception as e:
                print(f"[*] Aviso al restaurar CPU RNG: {e}")
        if "cuda_rng_state" in loaded_ckpt and loaded_ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
            try:
                cuda_rng = loaded_ckpt["cuda_rng_state"]
                if isinstance(cuda_rng, list):
                    cuda_rng_cpu = [r.cpu() if hasattr(r, "cpu") else r for r in cuda_rng]
                    torch.cuda.set_rng_state_all(cuda_rng_cpu)
                elif hasattr(cuda_rng, "cpu"):
                    torch.cuda.set_rng_state(cuda_rng.cpu())
                print("[OK] Estado CUDA RNG restaurado.")
            except Exception as e:
                print(f"[*] Aviso al restaurar CUDA RNG: {e}")

    criterion_l1 = nn.SmoothL1Loss()
    criterion_edge = KorniaSobelLoss().to(DEVICE) if HAS_KORNIA else FallbackSobelLoss().to(DEVICE)
    if HAS_KORNIA:
        print("[OK] Perdida de bordes Kornia Sobel acelerada en GPU activa")
    criterion_bce = nn.BCEWithLogitsLoss()

    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=max(1, total_target_epochs - start_epoch + 1), eta_min=1e-6)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=max(1, total_target_epochs - start_epoch + 1), eta_min=1e-6)

    # Restaurar schedulers si estan disponibles en checkpoint
    if loaded_ckpt is not None and not automatic_recovery:
        if "scheduler_g" in loaded_ckpt and loaded_ckpt["scheduler_g"] is not None:
            try:
                scheduler_g.load_state_dict(loaded_ckpt["scheduler_g"])
                print("[OK] Estado de scheduler_g restaurado con exito.")
            except Exception as e:
                print(f"[*] Aviso al restaurar scheduler_g: {e}")
        if "scheduler_d" in loaded_ckpt and loaded_ckpt["scheduler_d"] is not None:
            try:
                scheduler_d.load_state_dict(loaded_ckpt["scheduler_d"])
                print("[OK] Estado de scheduler_d restaurado con exito.")
            except Exception as e:
                print(f"[*] Aviso al restaurar scheduler_d: {e}")
    elif automatic_recovery:
        print("[RECUPERACION] Schedulers reiniciados desde el checkpoint sano para evitar repetir la trayectoria degradada.")

    start_time = time.time()
    previous_history = status_at_start.get("history", [])
    previous_metrics = previous_history[-1] if previous_history else {}
    avg_g = float(finite_positive(previous_metrics.get("g_loss", previous_metrics.get("loss"))) or finite_positive(loaded_ckpt.get("loss") if isinstance(loaded_ckpt, dict) else None) or 0.0)
    avg_d = float(finite_positive(previous_metrics.get("d_loss")) or 0.0)
    avg_l1 = float(finite_positive(previous_metrics.get("l1_loss")) or 0.0)
    avg_edge = float(finite_positive(previous_metrics.get("edge_loss")) or 0.0)
    skipped_amp_g = 0
    skipped_amp_d = 0
    pause_requested = False
    last_completed_epoch = start_epoch - 1

    for epoch in range(start_epoch, total_target_epochs + 1):
        epoch_start_time = time.time()
        previous_epoch_metrics = (avg_g, avg_d, avg_l1, avg_edge)
        current_lr = float(opt_g.param_groups[0].get("lr", effective_lr))
        dataloader = _build_training_dataloader(dataset, batch_size, current_sampling_plan)
        if PAUSE_FLAG_FILE.exists():
            print("\n[PAUSA] Senal de pausa previa a la epoca recibida. Guardando checkpoint...")
            try: PAUSE_FLAG_FILE.unlink()
            except Exception: pass
            save_checkpoint(CHECKPOINT_DIR / 'latest_checkpoint.pt', last_completed_epoch, avg_g, best_loss, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d, scheduler_g, scheduler_d)
            update_status(last_completed_epoch, total_target_epochs, "PAUSADO", avg_g, avg_d, avg_l1, avg_edge, start_time, current_lr, skipped_amp=skipped_amp_g + skipped_amp_d)
            return

        if STOP_FLAG_FILE.exists():
            print("\n[STOP] Senal de detencion recibida antes de iniciar la siguiente epoca. Guardando frontera segura...")
            try: STOP_FLAG_FILE.unlink()
            except Exception: pass
            save_checkpoint(CHECKPOINT_DIR / 'latest_checkpoint.pt', last_completed_epoch, avg_g, best_loss, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d, scheduler_g, scheduler_d)
            update_status(last_completed_epoch, total_target_epochs, "DETENIDO", avg_g, avg_d, avg_l1, avg_edge, start_time, current_lr, skipped_amp=skipped_amp_g + skipped_amp_d)
            return

        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_l1 = 0.0
        epoch_edge = 0.0
        loss_multipliers = current_loss_plan.get("multipliers", LossMultiplierController.neutral())

        for batch_i, (fronts, f_indices, targets, char_ids) in enumerate(dataloader):
            # Pequeño desahogo cooperativo para que el sistema operativo, VS Code y el navegador no se congelen
            if batch_i % 10 == 0:
                time.sleep(0.002)

            if PAUSE_FLAG_FILE.exists():
                print(f"\n[PAUSA] Senal de pausa en lote {batch_i}. Completando epoca {epoch} para pausar en frontera limpia...", flush=True)
                try: PAUSE_FLAG_FILE.unlink()
                except Exception: pass
                pause_requested = True

            if STOP_FLAG_FILE.exists():
                print(f"\n[STOP] Senal de detencion en lote {batch_i}. Se conserva intacto el checkpoint de la epoca {last_completed_epoch}...", flush=True)
                try: STOP_FLAG_FILE.unlink()
                except Exception: pass
                prev_g, prev_d, prev_l1, prev_edge = previous_epoch_metrics
                update_status(last_completed_epoch, total_target_epochs, 'DETENIDO', prev_g, prev_d, prev_l1, prev_edge, start_time, current_lr, skipped_amp=skipped_amp_g + skipped_amp_d)
                return

            fronts = fronts.to(DEVICE)
            targets = targets.to(DEVICE)
            poses = torch.stack([tmpl_mgr.get_frame_tensor(idx) for idx in f_indices]).to(DEVICE)
            cond = torch.cat([fronts, poses], dim=1)

            # Generador
            opt_g.zero_grad()
            with autocast(enabled=USE_AMP):
                preds = generator(cond)
                l1_color = criterion_l1(preds[:, :3], targets[:, :3]) * BASE_LOSS_WEIGHTS["color"] * loss_multipliers["color"]
                l1_alpha = criterion_l1(preds[:, 3:], targets[:, 3:]) * BASE_LOSS_WEIGHTS["alpha"] * loss_multipliers["alpha"]
                edge_loss = criterion_edge(preds[:, :3], targets[:, :3]) * BASE_LOSS_WEIGHTS["edge"] * loss_multipliers["edge"]

                # Discriminador: orden canonico (condition, target)
                d_fake = discriminator(cond, preds)
                adv_loss = criterion_bce(d_fake.float().clamp(-30.0, 30.0), torch.ones_like(d_fake).float()) * BASE_LOSS_WEIGHTS["adversarial"] * loss_multipliers["adversarial"]
                silhouette_loss = (
                    differentiable_silhouette_loss(preds, targets) * 0.5
                    if ENABLE_SILHOUETTE_LOSS
                    else preds.new_zeros(())
                )
                total_g = l1_color + l1_alpha + edge_loss + adv_loss + silhouette_loss

            # 1. Comprobacion estricta de finitud de componentes ANTES de backward y acumulacion
            g_components = [
                ("l1_color", l1_color.item()),
                ("l1_alpha", l1_alpha.item()),
                ("edge_loss", edge_loss.item()),
                ("adv_loss", adv_loss.item()),
                ("silhouette_loss", silhouette_loss.item()),
                ("total_g", total_g.item())
            ]
            for c_name, c_val in g_components:
                if not math.isfinite(c_val):
                    diag = f"Perdida no finita/NaN en componente Generador '{c_name}'={c_val} (Epoca {epoch}, Lote {batch_i})"
                    print(f"\n[!] ERROR_NAN: {diag}")
                    update_status(epoch, total_target_epochs, "ERROR_NAN", None, None, None, None, start_time, current_lr,
                                  error_details={"epoch": epoch, "batch": batch_i, "metric": c_name, "error": diag},
                                  skipped_amp=skipped_amp_g + skipped_amp_d)
                    return

            frame_quality_tracker.update_many(
                compute_frame_quality_batch(preds, targets, f_indices, char_ids),
                epoch=epoch,
            )

            scaler_g.scale(total_g).backward()
            scaler_g.unscale_(opt_g)
            g_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            scale_g_before = scaler_g.get_scale()
            scaler_g.step(opt_g)
            scaler_g.update()
            scale_g_after = scaler_g.get_scale()
            if scale_g_after < scale_g_before or (hasattr(g_norm, "item") and not math.isfinite(g_norm.item())):
                skipped_amp_g += 1

            # Discriminador (Aprende mas lento para no aplastar al generador)
            opt_d.zero_grad()
            with autocast(enabled=USE_AMP):
                d_real = discriminator(cond, targets)
                d_fake_det = discriminator(cond, preds.detach())
                loss_d_real = criterion_bce(d_real.float().clamp(-30.0, 30.0), torch.ones_like(d_real).float())
                loss_d_fake = criterion_bce(d_fake_det.float().clamp(-30.0, 30.0), torch.zeros_like(d_fake_det).float())
                total_d = (loss_d_real + loss_d_fake) * 0.5

            d_components = [
                ("loss_d_real", loss_d_real.item()),
                ("loss_d_fake", loss_d_fake.item()),
                ("total_d", total_d.item())
            ]
            for d_name, d_val in d_components:
                if not math.isfinite(d_val):
                    diag = f"Perdida no finita/NaN en componente Discriminador '{d_name}'={d_val} (Epoca {epoch}, Lote {batch_i})"
                    print(f"\n[!] ERROR_NAN: {diag}")
                    update_status(epoch, total_target_epochs, "ERROR_NAN", None, None, None, None, start_time, current_lr,
                                  error_details={"epoch": epoch, "batch": batch_i, "metric": d_name, "error": diag},
                                  skipped_amp=skipped_amp_g + skipped_amp_d)
                    return

            scaler_d.scale(total_d).backward()
            scaler_d.unscale_(opt_d)
            d_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
            scale_d_before = scaler_d.get_scale()
            scaler_d.step(opt_d)
            scaler_d.update()
            scale_d_after = scaler_d.get_scale()
            if scale_d_after < scale_d_before or (hasattr(d_norm, "item") and not math.isfinite(d_norm.item())):
                skipped_amp_d += 1

            epoch_g_loss += total_g.item()
            epoch_d_loss += total_d.item()
            epoch_l1 += l1_color.item()
            epoch_edge += edge_loss.item()

        n_batches = max(1, len(dataloader))
        avg_g = epoch_g_loss / n_batches
        avg_d = epoch_d_loss / n_batches
        avg_l1 = epoch_l1 / n_batches
        avg_edge = epoch_edge / n_batches

        # Verificar finitud estricta en cada epoca
        if not math.isfinite(avg_g) or not math.isfinite(avg_d):
            diag = f"Epoca {epoch} produjo perdidas promedio no finitas (G: {avg_g}, D: {avg_d})"
            print(f"\n[!] ERROR CRITICO: {diag}. Deteniendo por seguridad.")
            update_status(epoch, total_target_epochs, "ERROR_NAN", None, None, None, None, start_time, current_lr,
                          error_details={"epoch": epoch, "batch": "resumen_epoca", "metric": "avg_g", "error": diag},
                          skipped_amp=skipped_amp_g + skipped_amp_d)
            return

        amp_msg = f" | Pasos AMP Omitidos: {skipped_amp_g + skipped_amp_d}" if (skipped_amp_g + skipped_amp_d) > 0 else ""
        print(f"Epoca [{epoch:03d}/{total_target_epochs:03d}] - G_Loss: {avg_g:.4f} | Color_L1: {avg_l1:.4f} | Borde: {avg_edge:.4f} | D_Loss: {avg_d:.4f}{amp_msg}")

        # Muestra visual y auditoría clínica del cuerpo
        save_audit = (epoch % 10 == 0)
        last_qc = generate_preview(
            generator,
            dataset,
            epoch_label=epoch,
            full_certification=save_audit,
        )
        if last_qc and "score_total" in last_qc:
            c_prec = last_qc.get("cuerpo_precision", last_qc["score_total"])
            def_px = last_qc.get("defectos_cuerpo", 0)
            tot_px = last_qc.get("total_px_cuerpo", 0)
            c_ia = last_qc.get("colores_ia", 0)
            c_tgt = last_qc.get("colores_original", 0)
            print(f"  -> Calidad Anatómica: {last_qc['score_total']}% | Cuerpo: {c_prec}% ({def_px} defectos de {tot_px} px) | Colores: {c_ia} (Meta: {c_tgt})", flush=True)

        # Validar estabilidad antes de publicar o guardar pesos de la epoca actual.
        epoch_duration = round(time.time() - epoch_start_time, 1)
        status_before_epoch = load_status_file(STATUS_FILE)
        instability = detect_training_instability(
            status_before_epoch.get("history", []),
            current_epoch=epoch,
            current_g_loss=avg_g,
            current_l1_loss=avg_l1,
            quality=last_qc,
        )
        if instability is not None:
            recovery_route = _materialize_recovery_route(
                status_data=status_before_epoch,
                reason=instability,
                status_name="RECUPERACION_LISTA",
                total_epochs=total_target_epochs,
                rejected_metrics={
                    "epoch": epoch,
                    "g_loss": avg_g,
                    "d_loss": avg_d,
                    "l1_loss": avg_l1,
                    "edge_loss": avg_edge,
                    "quality": last_qc,
                },
            )
            if recovery_route is not None:
                print(
                    f"\n[SALVAGUARDA ANTI-COLAPSO] Epoca {epoch} rechazada. "
                    f"Ruta segura preparada desde epoca {recovery_route['source_epoch']} "
                    f"({Path(recovery_route['checkpoint']).name}). Pulsa Reanudar para continuar con LR reducido.",
                    flush=True,
                )
            else:
                prev_g, prev_d, prev_l1, prev_edge = previous_epoch_metrics
                update_status(
                    last_completed_epoch,
                    total_target_epochs,
                    "COLAPSO_DETECTADO",
                    prev_g,
                    prev_d,
                    prev_l1,
                    prev_edge,
                    start_time,
                    current_lr,
                    error_details={
                        "epoch": epoch,
                        "metric": instability.get("code"),
                        "error": instability.get("message"),
                    },
                    skipped_amp=skipped_amp_g + skipped_amp_d,
                )
                print(f"\n[SALVAGUARDA ANTI-COLAPSO] Epoca {epoch} rechazada, pero no hay snapshot de recuperacion.", flush=True)
            return

        # La epoca es sana: actualizar estado y avanzar schedulers antes de serializarlos.
        published_status = update_status(
            epoch,
            total_target_epochs,
            "ENTRENANDO",
            avg_g,
            avg_d,
            avg_l1,
            avg_edge,
            start_time,
            current_lr,
            skipped_amp=skipped_amp_g + skipped_amp_d,
            epoch_duration=epoch_duration,
            quality=last_qc,
            guidance_runtime={
                "frame_quality": frame_quality_tracker.export(),
                "loss_multipliers": dict(loss_multipliers),
                "loss_plan": dict(current_loss_plan),
                "dataset_character_ids": dataset_character_ids,
                "dataset_sample_count": len(dataset),
            },
        )
        published_status = _apply_intervention_policy(published_status)
        authorized_action = published_status.get("guidance", {}).get("authorized_action")
        if authorized_action == "STOP":
            guidance_recovery = _materialize_recovery_route(
                status_data=published_status,
                reason={
                    "code": "repeated_quality_collapse",
                    "message": "Segunda degradación crítica confirmada; rebobinando a checkpoint saludable para reanudar.",
                    "failed_epoch": epoch,
                    "g_loss": avg_g,
                    "l1_loss": avg_l1,
                    "quality": last_qc,
                },
                status_name="RECUPERACION_LISTA",
                total_epochs=total_target_epochs,
                rejected_metrics={
                    "epoch": epoch,
                    "g_loss": avg_g,
                    "d_loss": avg_d,
                    "l1_loss": avg_l1,
                    "edge_loss": avg_edge,
                    "quality": last_qc,
                },
            )
            if guidance_recovery is not None:
                intervention_state = published_status.get("guidance_state", {}).get("intervention_state")
                if isinstance(intervention_state, dict):
                    intervention_state["critical_streak"] = 0
                    intervention_state["consecutive_interventions"] = 0
                    intervention_state["rollback_count"] = 0
                print(
                    f"\n[CONTROL DE CALIDAD] Segunda degradación crítica confirmada: checkpoint degradado rechazado. "
                    f"Rebobinando a época saludable {guidance_recovery['source_epoch']} "
                    f"y reanudando automáticamente con LR={guidance_recovery['preferred_lr']:.8f}...",
                    flush=True,
                )
                return
            else:
                published_status["status"] = "PAUSADO_QC"
                published_status["error_details"] = {
                    "epoch": epoch,
                    "metric": "repeated_quality_collapse",
                    "error": "Control de Calidad pausó el entrenamiento tras una recaída crítica posterior al rollback (sin snapshot disponible).",
                }
                published_status = _attach_model_approval(published_status, final_critical=True)
                write_status_file(STATUS_FILE, published_status)
                print(
                    "\n[CONTROL DE CALIDAD] Segunda degradación crítica confirmada. "
                    "Entrenamiento pausado; no se encontró snapshot saludable para rollback.",
                    flush=True,
                )
                return
        if authorized_action == "ROLLBACK":
            # The existing TrainingRecovery machinery remains the sole owner of
            # snapshot selection, atomic copying, LR reduction and history repair.
            write_status_file(STATUS_FILE, published_status)
            guidance_recovery = _materialize_recovery_route(
                status_data=published_status,
                reason={
                    "code": "guidance_sustained_quality_collapse",
                    "message": "Guidance confirmó una degradación visual sostenida.",
                    "failed_epoch": epoch,
                    "g_loss": avg_g,
                    "l1_loss": avg_l1,
                    "quality": last_qc,
                },
                status_name="RECUPERACION_LISTA",
                total_epochs=total_target_epochs,
                rejected_metrics={
                    "epoch": epoch,
                    "g_loss": avg_g,
                    "d_loss": avg_d,
                    "l1_loss": avg_l1,
                    "edge_loss": avg_edge,
                    "quality": last_qc,
                },
            )
            if guidance_recovery is not None:
                print(
                    f"\n[GUIDANCE + RECOVERY] Colapso sostenido rechazado; "
                    f"rollback preparado desde época {guidance_recovery['source_epoch']}.",
                    flush=True,
                )
                return
            published_status["guidance"]["authorized_action"] = "CONTINUE"
            published_status["guidance"]["intervention_policy"]["reason"] = "rollback_snapshot_unavailable"
            intervention_state = published_status.get("guidance_state", {}).get("intervention_state")
            if isinstance(intervention_state, dict):
                intervention_state["rollback_count"] = max(0, int(intervention_state.get("rollback_count", 1)) - 1)
                intervention_state["total_interventions"] = max(0, int(intervention_state.get("total_interventions", 1)) - 1)
                intervention_state["critical_streak"] = 0
                published_status["guidance_state"]["intervention_count"] = intervention_state["total_interventions"]
        published_status, reduced_lr = _apply_authorized_lr_reduction(
            published_status,
            opt_g,
            opt_d,
            scheduler_g,
            scheduler_d,
        )
        if reduced_lr is not None:
            current_lr = reduced_lr
            print(
                f"\n[CONTROL DE CALIDAD] Learning rate reducido a {reduced_lr:.8f} por inestabilidad.",
                flush=True,
            )
        next_sampling_plan = _build_sampling_plan(
            dataset,
            frame_quality_tracker.export(),
            published_status.get("guidance"),
        )
        published_status = _attach_sampling_plan(published_status, next_sampling_plan)
        next_loss_plan = _build_loss_plan(
            published_status.get("guidance_state"),
            published_status.get("guidance"),
        )
        published_status = _attach_loss_plan(published_status, next_loss_plan)
        published_status = _maybe_save_quality_checkpoint(
            published_status,
            epoch=epoch,
            loss=avg_g,
            best_loss=best_loss,
            generator=generator,
        )
        write_status_file(STATUS_FILE, published_status)
        current_sampling_plan = next_sampling_plan
        current_loss_plan = next_loss_plan
        _print_quality_guidance(published_status.get("guidance"))
        scheduler_g.step()
        scheduler_d.step()

        # -------------------------------------------------------------
        # SISTEMA DE RESPAWN: Guardado cada 10 epocas con estado completo
        # -------------------------------------------------------------
        if epoch % 10 == 0:
            snapshot_file = SNAPSHOTS_DIR / f"checkpoint_epoch_{epoch:03d}.pt"
            save_checkpoint(snapshot_file, epoch, avg_g, best_loss, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d, scheduler_g, scheduler_d)
            gen_only_file = SNAPSHOTS_DIR / f"generator_epoch_{epoch:03d}.pt"
            torch.save(generator.state_dict(), gen_only_file)
            print(f"[RESPAWN SNAPSHOT] Punto de restauracion guardado: Epoca {epoch:03d} (Loss: {avg_g:.4f}, Best: {best_loss:.4f})")

        # -------------------------------------------------------------
        # CHECKPOINT DE SESION: Guardado aislado por sesion y rotacion
        # -------------------------------------------------------------
        session_epoch = epoch - session_start_epoch + 1
        is_epoch_healthy = not _quality_is_critical(published_status)
        qc_status_str = "APPROVED" if is_epoch_healthy else "REJECTED"
        session_ckpt_file = None
        if current_session_dir is not None:
            session_ckpt_file = current_session_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                session_ckpt_file,
                epoch,
                avg_g,
                best_loss,
                generator,
                discriminator,
                opt_g,
                opt_d,
                scaler_g,
                scaler_d,
                scheduler_g,
                scheduler_d,
                is_best_model=False,
                guidance_state=published_status.get("guidance_state"),
                session_id=current_session_id,
                session_epoch=session_epoch,
                qc_result={"status": qc_status_str, "is_healthy": is_epoch_healthy},
            )
            record_session_checkpoint(
                current_session_dir,
                epoch=epoch,
                session_epoch=session_epoch,
                checkpoint_filename=session_ckpt_file.name,
                metrics={"g_loss": avg_g, "l1_loss": avg_l1, "d_loss": avg_d, "edge_loss": avg_edge, "lr": current_lr},
                qc_result={"status": qc_status_str, "is_healthy": is_epoch_healthy},
                is_healthy=is_epoch_healthy,
                max_kept_healthy=10,
            )

        # Checkpoints de mejor rendimiento y ultimo (preservando siempre best_loss)
        if avg_g < best_loss and math.isfinite(avg_g) and avg_g > 0.0 and not _quality_is_critical(published_status):
            best_loss = avg_g
            save_checkpoint(CHECKPOINT_DIR / "best_generator.pt", epoch, avg_g, best_loss, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d, scheduler_g, scheduler_d)
            print(f"[*] ¡Nuevo mejor modelo registrado! G_Loss: {best_loss:.4f}")

        if math.isfinite(avg_g):
            latest_checkpoint = CHECKPOINT_DIR / "latest_checkpoint.pt"
            save_checkpoint(
                latest_checkpoint,
                epoch,
                avg_g,
                best_loss,
                generator,
                discriminator,
                opt_g,
                opt_d,
                scaler_g,
                scaler_d,
                scheduler_g,
                scheduler_d,
                session_id=current_session_id,
                session_epoch=session_epoch,
            )
            if not _quality_is_critical(published_status):
                retain_checkpoint_atomic(
                    latest_checkpoint,
                    CHECKPOINT_DIR / "last_quality_healthy_checkpoint.pt",
                )
                published_status["last_quality_healthy_epoch"] = int(epoch)
                if session_ckpt_file is not None:
                    try:
                        published_status["last_quality_healthy_session_epoch"] = str(session_ckpt_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
                    except ValueError:
                        published_status["last_quality_healthy_session_epoch"] = str(session_ckpt_file).replace("\\", "/")
                write_status_file(STATUS_FILE, published_status)
            last_completed_epoch = epoch

        # Termostato Inteligente Adaptativo en Tiempo Real (Protección Dual: GPU <= 84°C, CPU <= 90°C)
        manage_adaptive_thermal_throttle()

        # Si se solicito pausa limpia durante los lotes de esta epoca, pausar ahora en frontera limpia
        if pause_requested:
            print(f"\n[PAUSA] Epoca {epoch} finalizada al 100%. Pausando entrenamiento en frontera limpia.", flush=True)
            if PAUSE_FLAG_FILE.exists():
                try: PAUSE_FLAG_FILE.unlink()
                except Exception: pass
            update_status(epoch, total_target_epochs, "PAUSADO", avg_g, avg_d, avg_l1, avg_edge, start_time, current_lr, skipped_amp=skipped_amp_g + skipped_amp_d, epoch_duration=epoch_duration, reevaluate_guidance=False)
            return

    if math.isfinite(avg_g) and math.isfinite(avg_d) and avg_g > 0.0:
        if PAUSE_FLAG_FILE.exists():
            try: PAUSE_FLAG_FILE.unlink()
            except Exception: pass
        if STOP_FLAG_FILE.exists():
            try: STOP_FLAG_FILE.unlink()
            except Exception: pass
        final_lr = float(opt_g.param_groups[0].get("lr", effective_lr))
        final_status = update_status(
            last_completed_epoch,
            total_target_epochs,
            "COMPLETADO",
            avg_g,
            avg_d,
            avg_l1,
            avg_edge,
            start_time,
            final_lr,
            skipped_amp=skipped_amp_g + skipped_amp_d,
            reevaluate_guidance=False,
        )
        final_critical = _quality_is_critical(final_status)
        final_status = _attach_model_approval(final_status, final_critical=final_critical)
        write_status_file(STATUS_FILE, final_status)
        if final_critical:
            print(
                "\n[CONTROL DE CALIDAD] Ciclo finalizado, pero el último checkpoint fue rechazado. "
                "Se conservará el mejor checkpoint aprobado para generación.",
                flush=True,
            )
        else:
            print("\n[OK] Ciclo de entrenamiento finalizado y aprobado por Control de Calidad.", flush=True)
    else:
        diag = "Ciclo finalizado con perdidas invalidas o no finitas."
        update_status(last_completed_epoch, total_target_epochs, "ERROR_NAN", None, None, None, None, start_time, effective_lr,
                      error_details={"epoch": last_completed_epoch, "batch": "final", "metric": "avg_g", "error": diag},
                      skipped_amp=skipped_amp_g + skipped_amp_d)
        print(f"\n[!] {diag}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--mode", type=str, default="resume", choices=["start", "resume"])
    parser.add_argument("--respawn_epoch", type=int, default=None, help="Epoca exacta a la cual rebobinar (Respawn)")
    parser.add_argument("--repair_state", action="store_true", help="Prepara una ruta segura de recuperacion sin iniciar entrenamiento")
    args = parser.parse_args()
    if args.repair_state:
        result = repair_current_training_state()
        if result is None:
            print("[OK] No se detecto una degradacion sostenida que requiera reparacion.")
        else:
            print(
                f"[OK] Recuperacion preparada: epoca {result['source_epoch']} | "
                f"checkpoint={result['checkpoint']} | LR={result['preferred_lr']}"
            )
        raise SystemExit(0)
    train_supervised_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mode=args.mode,
        respawn_epoch=args.respawn_epoch
    )
