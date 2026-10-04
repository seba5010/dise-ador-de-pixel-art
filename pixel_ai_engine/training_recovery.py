import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def finite_positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _finite_nonnegative(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def load_status_file(status_file: Path) -> Dict[str, Any]:
    if not status_file.exists():
        return {}
    try:
        with open(status_file, "r", encoding="utf-8") as status_handle:
            data = json.load(status_handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_status_file(status_file: Path, data: Dict[str, Any]) -> None:
    status_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = status_file.with_suffix(status_file.suffix + ".tmp")
    with open(temporary_file, "w", encoding="utf-8") as status_handle:
        json.dump(data, status_handle, indent=2, allow_nan=False)
    os.replace(temporary_file, status_file)


def copy_checkpoint_atomic(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary_file)
    os.replace(temporary_file, destination)


def _history_points(history: Iterable[Dict[str, Any]]) -> List[Dict[str, float]]:
    points: List[Dict[str, float]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        epoch = finite_positive(entry.get("epoch"))
        g_loss = finite_positive(entry.get("g_loss", entry.get("loss")))
        l1_loss = finite_positive(entry.get("l1_loss"))
        if epoch is None or g_loss is None or l1_loss is None:
            continue
        points.append({"epoch": int(epoch), "g_loss": g_loss, "l1_loss": l1_loss})
    points.sort(key=lambda point: point["epoch"])
    return points


def detect_training_instability(
    history: Iterable[Dict[str, Any]],
    current_epoch: Optional[int] = None,
    current_g_loss: Any = None,
    current_l1_loss: Any = None,
    quality: Optional[Dict[str, Any]] = None,
    recent_window: int = 3,
    warmup_epochs: int = 10,
) -> Optional[Dict[str, Any]]:
    points = _history_points(history)
    current_g = finite_positive(current_g_loss)
    current_l1 = finite_positive(current_l1_loss)
    if current_epoch is not None and current_g is not None and current_l1 is not None:
        points = [point for point in points if point["epoch"] != int(current_epoch)]
        points.append({"epoch": int(current_epoch), "g_loss": current_g, "l1_loss": current_l1})
        points.sort(key=lambda point: point["epoch"])

    if not points:
        return None

    latest = points[-1]
    in_warmup = latest["epoch"] <= warmup_epochs
    if latest["g_loss"] > 75.0 or (not in_warmup and latest["l1_loss"] > 0.20):
        return {
            "code": "absolute_loss_spike",
            "message": "La perdida supero el limite absoluto de seguridad.",
            "failed_epoch": latest["epoch"],
            "g_loss": round(latest["g_loss"], 6),
            "l1_loss": round(latest["l1_loss"], 6),
        }

    if in_warmup:
        return None

    if isinstance(quality, dict):
        colors_ai = _finite_nonnegative(quality.get("colores_ia"))
        colors_target = finite_positive(quality.get("colores_original"))
        body_precision = _finite_nonnegative(quality.get("cuerpo_precision"))
        score_total = _finite_nonnegative(quality.get("score_total"))
        palette_collapsed = (
            colors_ai is not None
            and colors_target is not None
            and colors_ai <= max(4.0, colors_target * 0.05)
        )
        anatomy_collapsed = (
            (body_precision is not None and body_precision < 20.0)
            or (score_total is not None and score_total < 20.0)
        )
        if palette_collapsed or anatomy_collapsed:
            return {
                "code": "quality_collapse",
                "message": "La auditoria visual detecto colapso de paleta o anatomia.",
                "failed_epoch": latest["epoch"],
                "g_loss": round(latest["g_loss"], 6),
                "l1_loss": round(latest["l1_loss"], 6),
                "quality": quality,
            }

    if latest["epoch"] <= 20 or len(points) < max(6, recent_window):
        return None

    baseline = points[: min(20, len(points))]
    baseline_g = statistics.median(point["g_loss"] for point in baseline)
    baseline_l1 = statistics.median(point["l1_loss"] for point in baseline)
    g_limit = max(baseline_g * 3.5, baseline_g + 0.10)
    l1_limit = max(baseline_l1 * 3.0, baseline_l1 + 0.015)
    recent = points[-recent_window:]

    if all(point["g_loss"] > g_limit and point["l1_loss"] > l1_limit for point in recent):
        return {
            "code": "sustained_reconstruction_drift",
            "message": "La perdida de reconstruccion y la perdida total se degradaron de forma sostenida.",
            "failed_epoch": latest["epoch"],
            "g_loss": round(latest["g_loss"], 6),
            "l1_loss": round(latest["l1_loss"], 6),
            "baseline_g_loss": round(baseline_g, 6),
            "baseline_l1_loss": round(baseline_l1, 6),
            "g_limit": round(g_limit, 6),
            "l1_limit": round(l1_limit, 6),
            "recent_window": recent_window,
        }

    return None


def _snapshot_epoch(snapshot_file: Path) -> Optional[int]:
    name = snapshot_file.stem
    prefix = "checkpoint_epoch_"
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix):])
    except ValueError:
        return None


def choose_recovery_snapshot(
    history: Iterable[Dict[str, Any]],
    snapshots_dir: Path,
    failed_epoch: int,
) -> Optional[Dict[str, Any]]:
    points = [point for point in _history_points(history) if point["epoch"] < int(failed_epoch)]
    snapshots = []
    if snapshots_dir.exists():
        for snapshot_file in snapshots_dir.glob("checkpoint_epoch_*.pt"):
            epoch = _snapshot_epoch(snapshot_file)
            if epoch is not None and epoch < int(failed_epoch):
                snapshots.append((epoch, snapshot_file))
    snapshots.sort(key=lambda item: item[0])
    if not snapshots:
        return None

    if not points:
        epoch, snapshot_file = snapshots[-1]
        return {"epoch": epoch, "path": snapshot_file, "selection": "latest_available"}

    baseline = points[: min(20, len(points))]
    baseline_g = statistics.median(point["g_loss"] for point in baseline)
    baseline_l1 = statistics.median(point["l1_loss"] for point in baseline)
    healthy_g_limit = max(baseline_g * 1.5, baseline_g + 0.02)
    healthy_l1_limit = max(baseline_l1 * 2.0, baseline_l1 + 0.0075)
    healthy_epochs = [
        point["epoch"]
        for point in points
        if point["g_loss"] <= healthy_g_limit and point["l1_loss"] <= healthy_l1_limit
    ]
    safe_epoch = max(healthy_epochs) if healthy_epochs else baseline[-1]["epoch"]
    eligible = [item for item in snapshots if item[0] <= safe_epoch]
    epoch, snapshot_file = eligible[-1] if eligible else snapshots[0]
    return {
        "epoch": epoch,
        "path": snapshot_file,
        "selection": "last_healthy_snapshot",
        "safe_epoch": safe_epoch,
        "healthy_g_limit": round(healthy_g_limit, 6),
        "healthy_l1_limit": round(healthy_l1_limit, 6),
    }


def activate_recovery_status(
    status_data: Dict[str, Any],
    source_epoch: int,
    source_file: Path,
    source_loss: float,
    failed_epoch: int,
    reason: Dict[str, Any],
    total_epochs: int,
    preferred_lr: float,
    rejected_metrics: Optional[Dict[str, Any]] = None,
    status_name: str = "RECUPERACION_LISTA",
) -> Dict[str, Any]:
    data = dict(status_data or {})
    old_history = list(data.get("history", []))
    kept_history = [
        entry for entry in old_history
        if isinstance(entry, dict) and int(entry.get("epoch", 0)) <= int(source_epoch)
    ]
    discarded_history = [
        entry for entry in old_history
        if isinstance(entry, dict) and int(entry.get("epoch", 0)) > int(source_epoch)
    ]
    recovery_events = list(data.get("recovery_events", []))
    event_id = f"{int(source_epoch)}:{int(failed_epoch)}"
    if discarded_history and not any(event.get("id") == event_id for event in recovery_events if isinstance(event, dict)):
        recovery_events.append({
            "id": event_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_epoch": int(source_epoch),
            "failed_epoch": int(failed_epoch),
            "reason": reason,
            "history": discarded_history,
        })

    last_healthy = kept_history[-1] if kept_history else {}
    initial_loss = None
    if kept_history:
        initial_loss = finite_positive(kept_history[0].get("g_loss", kept_history[0].get("loss")))
    source_loss = finite_positive(source_loss) or 999.0
    loss_reduction = 0.0
    if initial_loss is not None:
        loss_reduction = round(((initial_loss - source_loss) / initial_loss) * 100.0, 1)

    recovery = {
        "active": True,
        "source_epoch": int(source_epoch),
        "failed_epoch": int(failed_epoch),
        "checkpoint": str(source_file).replace("\\", "/"),
        "preferred_lr": float(preferred_lr),
        "reason": reason,
        "rejected_metrics": rejected_metrics or {},
        "activated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    data.update({
        "epoch": int(source_epoch),
        "total_epochs": int(total_epochs),
        "status": status_name,
        "g_loss": round(source_loss, 4),
        "d_loss": last_healthy.get("d_loss"),
        "l1_loss": last_healthy.get("l1_loss"),
        "edge_loss": last_healthy.get("edge_loss"),
        "best_loss": round(source_loss, 4),
        "initial_loss": round(initial_loss, 4) if initial_loss is not None else round(source_loss, 4),
        "loss_reduction_pct": loss_reduction,
        "lr": float(preferred_lr),
        "eta_sec": None,
        "timestamp": time.strftime("%H:%M:%S"),
        "history": kept_history,
        "recovery": recovery,
        "recovery_events": recovery_events,
        "error_details": {
            "epoch": int(failed_epoch),
            "metric": reason.get("code", "training_instability"),
            "error": reason.get("message", "Inestabilidad de entrenamiento detectada."),
        },
        "quality": None,
    })
    return data
