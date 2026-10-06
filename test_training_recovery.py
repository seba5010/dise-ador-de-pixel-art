import json
from pathlib import Path

import pixel_ai_engine.training_recovery as recovery_mod

from pixel_ai_engine.training_recovery import (
    activate_recovery_status,
    choose_recovery_snapshot,
    detect_training_instability,
    retain_checkpoint_atomic,
    write_status_file,
)


def _degrading_history():
    history = []
    for epoch in range(1, 87):
        progress = (epoch - 1) / 85.0
        history.append({
            "epoch": epoch,
            "g_loss": 0.05 + 0.2515 * progress,
            "l1_loss": 0.007 + 0.0238 * progress,
            "d_loss": 0.69 - 0.52 * progress,
            "edge_loss": 0.0006 + 0.001 * progress,
        })
    return history


def test_detects_sustained_drift_instead_of_single_spike():
    history = _degrading_history()
    reason = detect_training_instability(history)
    assert reason is not None
    assert reason["code"] == "sustained_reconstruction_drift"
    assert reason["failed_epoch"] == 86

    healthy_history = history[:20]
    healthy_history.append({"epoch": 21, "g_loss": 0.30, "l1_loss": 0.04})
    assert detect_training_instability(healthy_history) is None

    quality_reason = detect_training_instability(
        history[:20],
        quality={"colores_ia": 0, "colores_original": 427, "cuerpo_precision": 0, "score_total": 0},
    )
    assert quality_reason is not None
    assert quality_reason["code"] == "quality_collapse"


def test_initial_training_warmup_is_not_rejected_without_snapshot():
    reason = detect_training_instability(
        [],
        current_epoch=1,
        current_g_loss=0.6911,
        current_l1_loss=0.3840,
        quality={
            "colores_ia": 5434,
            "colores_original": 1283,
            "cuerpo_precision": 58.5,
            "score_total": 48.4,
        },
    )
    assert reason is None

    emergency_reason = detect_training_instability(
        [],
        current_epoch=1,
        current_g_loss=100.0,
        current_l1_loss=0.3840,
    )
    assert emergency_reason is not None
    assert emergency_reason["code"] == "absolute_loss_spike"

    post_warmup_reason = detect_training_instability(
        [],
        current_epoch=11,
        current_g_loss=0.6911,
        current_l1_loss=0.3840,
    )
    assert post_warmup_reason is not None
    assert post_warmup_reason["code"] == "absolute_loss_spike"


def test_selects_last_snapshot_inside_healthy_window(tmp_path: Path):
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    for epoch in range(10, 90, 10):
        (snapshots_dir / f"checkpoint_epoch_{epoch:03d}.pt").touch()

    plan = choose_recovery_snapshot(_degrading_history(), snapshots_dir, failed_epoch=86)
    assert plan is not None
    assert plan["epoch"] == 20
    assert plan["path"].name == "checkpoint_epoch_020.pt"


def test_recovery_status_preserves_rejected_history():
    history = _degrading_history()
    status = {
        "epoch": 86,
        "total_epochs": 135,
        "status": "PAUSADO",
        "history": history,
        "past_eras": [{"era": 1}],
    }
    repaired = activate_recovery_status(
        status_data=status,
        source_epoch=20,
        source_file=Path("checkpoints/recovery_checkpoint.pt"),
        source_loss=history[19]["g_loss"],
        failed_epoch=86,
        reason={"code": "sustained_reconstruction_drift", "message": "drift"},
        total_epochs=135,
        preferred_lr=7.5e-5,
    )

    assert repaired["epoch"] == 20
    assert repaired["status"] == "RECUPERACION_LISTA"
    assert len(repaired["history"]) == 20
    assert len(repaired["recovery_events"]) == 1
    assert len(repaired["recovery_events"][0]["history"]) == 66
    assert repaired["past_eras"] == [{"era": 1}]
    assert repaired["recovery"]["active"] is True


def test_status_write_retries_transient_windows_replace_lock(tmp_path, monkeypatch):
    status_file = tmp_path / "training_status.json"
    real_replace = recovery_mod.os.replace
    attempts = {"count": 0}

    def flaky_replace(source, destination):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError(5, "temporary Windows lock", str(destination))
        return real_replace(source, destination)

    monkeypatch.setattr(recovery_mod.os, "replace", flaky_replace)
    monkeypatch.setattr(recovery_mod.time, "sleep", lambda _seconds: None)

    write_status_file(status_file, {"status": "ENTRENANDO", "epoch": 28})

    assert attempts["count"] == 3
    assert json.loads(status_file.read_text(encoding="utf-8"))["epoch"] == 28
    assert not list(tmp_path.glob(".training_status.json.*.tmp"))


def test_retain_checkpoint_keeps_previous_epoch_when_latest_is_replaced(tmp_path):
    latest = tmp_path / "latest.pt"
    healthy = tmp_path / "healthy.pt"
    latest.write_bytes(b"epoch-15")

    retain_checkpoint_atomic(latest, healthy)
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"epoch-16")
    recovery_mod.os.replace(replacement, latest)

    assert latest.read_bytes() == b"epoch-16"
    assert healthy.read_bytes() == b"epoch-15"


def test_qc_auto_resume_command_preserves_target_and_batch(monkeypatch):
    import sprite_studio

    monkeypatch.setattr(sprite_studio, "AUTO_RESUME_QC_RECOVERY", True)
    monkeypatch.setattr(sprite_studio, "AUTO_RESUME_QC_MAX_ATTEMPTS", 1)
    status = {
        "epoch": 20,
        "total_epochs": 500,
        "status": "RECUPERACION_LISTA",
        "recovery": {"active": True, "source_epoch": 20, "auto_resume_attempts": 0},
    }
    command = [
        "python.exe", "train_supervised.py", "--epochs", "500",
        "--batch_size", "6", "--lr", "0.0001", "--mode", "start",
    ]

    resumed = sprite_studio._build_qc_auto_resume_command(command, status)

    assert resumed is not None
    assert resumed[resumed.index("--epochs") + 1] == "480"
    assert resumed[resumed.index("--batch_size") + 1] == "6"
    assert resumed[resumed.index("--mode") + 1] == "resume"
    assert resumed[resumed.index("--lr") + 1] == "0.0001"

    status["recovery"]["auto_resume_attempts"] = 1
    assert sprite_studio._build_qc_auto_resume_command(command, status) is None


def test_qc_auto_resume_marks_status_and_launches_once(tmp_path, monkeypatch):
    import sprite_studio

    checkpoint = tmp_path / "checkpoints" / "recovery_checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    status = {
        "epoch": 20,
        "total_epochs": 50,
        "status": "RECUPERACION_LISTA",
        "recovery": {
            "active": True,
            "source_epoch": 20,
            "failed_epoch": 41,
            "checkpoint": "checkpoints/recovery_checkpoint.pt",
        },
    }
    written = []
    launched = []

    class FakeProcess:
        def poll(self):
            return None

    fake_process = FakeProcess()
    monkeypatch.setattr(sprite_studio, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sprite_studio, "TRAINING_LOG_FILE", tmp_path / "training_logs" / "training.log")
    monkeypatch.setattr(sprite_studio, "AUTO_RESUME_QC_RECOVERY", True)
    monkeypatch.setattr(sprite_studio, "AUTO_RESUME_QC_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(sprite_studio, "GLOBAL_TRAINING_PROC", None)
    monkeypatch.setattr(sprite_studio, "load_status_file", lambda _path: dict(status))
    monkeypatch.setattr(sprite_studio, "get_active_recovery_checkpoint", lambda: checkpoint)
    monkeypatch.setattr(sprite_studio, "write_status_file", lambda _path, data: written.append(dict(data)))
    monkeypatch.setattr(
        sprite_studio,
        "launch_training_process",
        lambda command, env: launched.append((list(command), dict(env))) or fake_process,
    )

    result = sprite_studio._maybe_auto_resume_qc_recovery(
        ["python.exe", "train_supervised.py", "--epochs", "50", "--batch_size", "4", "--mode", "start"],
        {"PYTHONUNBUFFERED": "1"},
        return_code=0,
    )

    assert result is fake_process
    assert len(launched) == 1
    assert launched[0][0][launched[0][0].index("--epochs") + 1] == "30"
    assert written[-1]["status"] == "RECUPERANDO"
    assert written[-1]["recovery"]["auto_resume_attempts"] == 1
    assert written[-1]["auto_resume"]["failed_epoch"] == 41
