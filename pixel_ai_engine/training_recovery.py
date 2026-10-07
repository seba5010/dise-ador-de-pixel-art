import json
import math
import os
import shutil
import statistics
import threading
import time
from uuid import uuid4
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


def _unique_temporary_path(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.{uuid4().hex}.tmp"
    )


def _replace_with_retry(temporary_file: Path, destination: Path, attempts: int = 12) -> None:
    """Replace atomically, tolerating short-lived Windows reader/antivirus locks."""
    for attempt in range(attempts):
        try:
            os.replace(temporary_file, destination)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(0.25, 0.02 * (2 ** attempt)))


def write_status_file(status_file: Path, data: Dict[str, Any]) -> None:
    status_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = _unique_temporary_path(status_file)
    try:
        with open(temporary_file, "w", encoding="utf-8") as status_handle:
            json.dump(data, status_handle, indent=2, allow_nan=False)
            status_handle.flush()
            os.fsync(status_handle.fileno())
        _replace_with_retry(temporary_file, status_file)
    finally:
        if temporary_file.exists():
            try:
                temporary_file.unlink()
            except OSError:
                pass


def copy_checkpoint_atomic(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = _unique_temporary_path(destination)
    try:
        shutil.copy2(source, temporary_file)
        _replace_with_retry(temporary_file, destination)
    finally:
        if temporary_file.exists():
            try:
                temporary_file.unlink()
            except OSError:
                pass


def retain_checkpoint_atomic(source: Path, destination: Path) -> None:
    """Retain an independent checkpoint copy atomically, never sharing hardlinks."""
    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = _unique_temporary_path(destination)
    try:
        shutil.copy2(source, temporary_file)
        _replace_with_retry(temporary_file, destination)
    finally:
        if temporary_file.exists():
            try:
                temporary_file.unlink()
            except OSError:
                pass


def generate_session_id(prefix: str = "session_") -> str:
    """Generate a unique timestamped session ID."""
    return f"{prefix}{time.strftime('%Y%m%d_%H%M%S')}"


def get_session_dir(session_id: str, base_dir: Optional[Path] = None) -> Path:
    """Resolve session directory under base checkpoints dir."""
    root = Path(base_dir) if base_dir is not None else Path("checkpoints")
    return root / "sessions" / session_id


def load_session_manifest(session_dir: Path) -> Dict[str, Any]:
    """Safely load session.json manifest."""
    manifest_file = session_dir / "session.json"
    if not manifest_file.exists():
        return {}
    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_session_manifest(session_dir: Path, data: Dict[str, Any]) -> None:
    """Atomically save session.json manifest."""
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = session_dir / "session.json"
    temporary_file = _unique_temporary_path(manifest_file)
    try:
        with open(temporary_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(temporary_file, manifest_file)
    finally:
        if temporary_file.exists():
            try:
                temporary_file.unlink()
            except OSError:
                pass


def init_session(
    session_dir: Path,
    session_id: str,
    start_epoch: int,
    target_epochs: int,
    mode: str = "resume",
    source_checkpoint: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Initialize a new session manifest in session_dir."""
    session_dir.mkdir(parents=True, exist_ok=True)
    existing = load_session_manifest(session_dir)
    if existing and existing.get("session_id") == session_id:
        return existing
    manifest = {
        "session_id": session_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start_epoch": int(start_epoch),
        "target_epochs": int(target_epochs),
        "mode": str(mode),
        "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
        "latest_epoch": None,
        "latest_healthy_epoch": None,
        "checkpoints": [],
        "metadata": metadata or {},
    }
    save_session_manifest(session_dir, manifest)
    return manifest


def record_session_checkpoint(
    session_dir: Path,
    epoch: int,
    session_epoch: int,
    checkpoint_filename: str,
    metrics: Optional[Dict[str, Any]] = None,
    qc_result: Optional[Dict[str, Any]] = None,
    is_healthy: bool = True,
    max_kept_healthy: int = 10,
    max_kept_rejected: int = 2,
) -> Dict[str, Any]:
    """
    Record an epoch checkpoint in session.json and enforce ring buffer retention:
    Keep only the last max_kept_healthy (default 10) approved checkpoints of this session,
    automatically pruning older checkpoints to save disk space while preserving history.
    """
    manifest = load_session_manifest(session_dir)
    if not manifest:
        manifest = {
            "session_id": session_dir.name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_epoch": int(epoch),
            "target_epochs": int(epoch),
            "mode": "unknown",
            "source_checkpoint": None,
            "checkpoints": [],
            "metadata": {},
        }

    metrics = metrics or {}
    qc_result = qc_result or {}
    checkpoints = list(manifest.get("checkpoints", []))

    entry_index = next((i for i, c in enumerate(checkpoints) if int(c.get("epoch", -1)) == int(epoch)), None)
    entry_data = {
        "epoch": int(epoch),
        "session_epoch": int(session_epoch),
        "filename": checkpoint_filename,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "g_loss": metrics.get("g_loss"),
        "l1_loss": metrics.get("l1_loss"),
        "d_loss": metrics.get("d_loss"),
        "edge_loss": metrics.get("edge_loss"),
        "lr": metrics.get("lr"),
        "qc_status": qc_result.get("status", "APPROVED" if is_healthy else "REJECTED"),
        "qc_severity": qc_result.get("severity", "low" if is_healthy else "critical"),
        "is_healthy": bool(is_healthy),
        "pruned": False,
    }
    if entry_index is not None:
        checkpoints[entry_index] = entry_data
    else:
        checkpoints.append(entry_data)

    checkpoints.sort(key=lambda c: int(c.get("epoch", 0)))

    # Ring buffer cleanup: keep only last max_kept_healthy approved/healthy files
    healthy_unpruned = [c for c in checkpoints if c.get("is_healthy") and not c.get("pruned")]
    if len(healthy_unpruned) > max_kept_healthy:
        to_prune = healthy_unpruned[:-max_kept_healthy]
        for c in to_prune:
            target = session_dir / c["filename"]
            if target.is_file():
                try:
                    target.unlink()
                except OSError:
                    pass
            c["pruned"] = True

    # Ring buffer cleanup: keep only last max_kept_rejected rejected files
    rejected_unpruned = [c for c in checkpoints if not c.get("is_healthy") and not c.get("pruned")]
    if len(rejected_unpruned) > max_kept_rejected:
        to_prune = rejected_unpruned[:-max_kept_rejected]
        for c in to_prune:
            target = session_dir / c["filename"]
            if target.is_file():
                try:
                    target.unlink()
                except OSError:
                    pass
            c["pruned"] = True

    manifest["checkpoints"] = checkpoints
    manifest["latest_epoch"] = int(epoch)
    healthy_remaining = [c for c in checkpoints if c.get("is_healthy") and not c.get("pruned")]
    if healthy_remaining:
        manifest["latest_healthy_epoch"] = int(healthy_remaining[-1]["epoch"])
    elif is_healthy:
        manifest["latest_healthy_epoch"] = int(epoch)

    save_session_manifest(session_dir, manifest)
    return manifest


def find_session_recovery_checkpoint(
    session_dir: Path,
    failed_epoch: int,
) -> Optional[Dict[str, Any]]:
    """
    Find the highest approved, healthy, unpruned checkpoint within the given session
    strictly before failed_epoch.
    """
    manifest = load_session_manifest(session_dir)
    if not manifest:
        candidates = []
        for ckpt in session_dir.glob("epoch_*.pt"):
            try:
                ep = int(ckpt.stem.replace("epoch_", ""))
                if ep < int(failed_epoch):
                    candidates.append((ep, ckpt))
            except ValueError:
                pass
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        best_ep, best_path = candidates[-1]
        return {
            "epoch": best_ep,
            "path": best_path,
            "selection": "session_healthy_checkpoint",
            "session_id": session_dir.name,
        }

    checkpoints = manifest.get("checkpoints", [])
    eligible = [
        c for c in checkpoints
        if int(c.get("epoch", 0)) < int(failed_epoch)
        and c.get("is_healthy")
        and not c.get("pruned")
        and (session_dir / c.get("filename", "")).is_file()
    ]
    if not eligible:
        return None
    eligible.sort(key=lambda c: int(c.get("epoch", 0)))
    selected = eligible[-1]
    return {
        "epoch": int(selected["epoch"]),
        "path": session_dir / selected["filename"],
        "selection": "session_healthy_checkpoint",
        "session_id": manifest.get("session_id", session_dir.name),
        "metrics": selected,
    }


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
    session_dir: Optional[Path] = None,
    allow_snapshots_fallback: bool = True,
) -> Optional[Dict[str, Any]]:
    # Priorizar checkpoints saludables de la sesion actual si esta disponible
    if session_dir is not None and session_dir.exists():
        session_recovery = find_session_recovery_checkpoint(session_dir, failed_epoch)
        if session_recovery is not None:
            return session_recovery
        if not allow_snapshots_fallback:
            return None

    history_entries = [entry for entry in history if isinstance(entry, dict)]
    points = [point for point in _history_points(history_entries) if point["epoch"] < int(failed_epoch)]
    quality_health: Dict[int, bool] = {}
    for entry in history_entries:
        epoch_value = finite_positive(entry.get("epoch"))
        if epoch_value is None:
            continue
        guidance = entry.get("guidance", entry.get("quality_guidance"))
        if not isinstance(guidance, dict):
            continue
        severity = str(guidance.get("severity", "low")).lower()
        recommendation = str(guidance.get("recommended_action", "CONTINUE")).upper()
        quality_health[int(epoch_value)] = severity != "critical" and recommendation != "ROLLBACK"
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
        and quality_health.get(point["epoch"], True)
    ]
    safe_epoch = max(healthy_epochs) if healthy_epochs else baseline[-1]["epoch"]
    eligible = [item for item in snapshots if item[0] <= safe_epoch]
    epoch, snapshot_file = eligible[-1] if eligible else snapshots[0]
    return {
        "epoch": epoch,
        "path": snapshot_file,
        "selection": "last_healthy_quality_snapshot" if quality_health else "last_healthy_snapshot",
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
    guidance_state = dict(data.get("guidance_state", {}))
    if "intervention_state" in guidance_state and isinstance(guidance_state["intervention_state"], dict):
        intervention_state = dict(guidance_state["intervention_state"])
        intervention_state["critical_streak"] = 0
        intervention_state["consecutive_interventions"] = 0
        intervention_state["rollback_count"] = 0
        guidance_state["intervention_state"] = intervention_state
    guidance_state["intervention_count"] = 0
    data["guidance_state"] = guidance_state
    return data
