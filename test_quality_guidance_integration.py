import json
import time

import torch

from pixel_ai_engine import train_supervised


def _telemetry():
    return {
        "gpu_temp": None,
        "cpu_temp": None,
        "vram_gb": 0.0,
        "vram_total_gb": 0.0,
        "vram_pct": 0,
        "gpu_util": 0,
    }


def _quality(face):
    return {
        "score_total": 76.8,
        "cuerpo_precision": 82.0,
        "silueta_iou_real": 79.0,
        "gestos_ojos": float(face),
        "ropa_delantal": 94.0,
        "fidelidad_paleta": 91.0,
        "pureza_alfa": 99.0,
        "micro_detalles": 68.0,
    }


def _configure_status_test(monkeypatch, tmp_path):
    status_file = tmp_path / "training_status.json"
    monkeypatch.setattr(train_supervised, "STATUS_FILE", status_file)
    monkeypatch.setattr(train_supervised, "get_hardware_telemetry", _telemetry)
    monkeypatch.setattr(train_supervised, "get_available_snapshots", lambda: [])
    monkeypatch.setattr(train_supervised, "ENABLE_QUALITY_GUIDANCE", True)
    return status_file


def _publish(epoch, face):
    return train_supervised.update_status(
        epoch,
        10,
        "ENTRENANDO",
        8.4 - epoch * 0.1,
        0.7,
        0.08,
        0.04,
        time.time() - 1,
        lr_val=1.5e-4,
        skipped_amp=0,
        epoch_duration=1.0,
        quality=_quality(face),
    )


def test_status_persists_observational_guidance_and_epoch_history(monkeypatch, tmp_path):
    status_file = _configure_status_test(monkeypatch, tmp_path)

    status = _publish(1, 71)
    status = _publish(2, 68)
    status = _publish(3, 63)
    persisted = json.loads(status_file.read_text(encoding="utf-8"))

    assert status == persisted
    assert persisted["guidance"]["mode"] == "observational"
    assert persisted["guidance"]["action"] == "CONTINUE"
    assert persisted["guidance"]["recommended_action"] == "REINFORCE"
    assert persisted["guidance"]["training_modified"] is False
    assert persisted["guidance"]["trend"]["by_category"]["face"] == "REGRESSION"
    assert persisted["quality_guidance"] == persisted["guidance"]
    assert len(persisted["guidance_state"]["quality_history"]) == 3
    assert all("quality" in entry and "guidance" in entry for entry in persisted["history"])
    assert persisted["history"][-1]["guidance"]["primary_problem"] == "face"


def test_pause_style_update_preserves_guidance_and_resume_history(monkeypatch, tmp_path):
    _configure_status_test(monkeypatch, tmp_path)
    training_status = _publish(1, 65)
    state_before = training_status["guidance_state"]

    paused = train_supervised.update_status(
        1,
        10,
        "PAUSADO",
        training_status["g_loss"],
        training_status["d_loss"],
        training_status["l1_loss"],
        training_status["edge_loss"],
        time.time() - 1,
        lr_val=training_status["lr"],
        quality=None,
    )

    assert paused["status"] == "PAUSADO"
    assert paused["guidance"] == training_status["guidance"]
    assert paused["guidance_state"] == state_before
    assert len(paused["history"]) == 1

    resumed = _publish(2, 66)
    assert len(resumed["guidance_state"]["quality_history"]) == 2
    assert [item["epoch"] for item in resumed["history"]] == [1, 2]


def test_feature_flag_disables_guidance_without_changing_training_metrics(monkeypatch, tmp_path):
    _configure_status_test(monkeypatch, tmp_path)
    monkeypatch.setattr(train_supervised, "ENABLE_QUALITY_GUIDANCE", False)

    status = _publish(1, 40)
    assert status["guidance"]["enabled"] is False
    assert status["guidance"]["recommended_action"] == "CONTINUE"
    assert status["guidance"]["training_modified"] is False
    assert status["g_loss"] == 8.3
    assert status["lr"] == 1.5e-4


def test_checkpoint_persists_guidance_state_and_old_checkpoint_is_compatible(monkeypatch, tmp_path):
    status_file = _configure_status_test(monkeypatch, tmp_path)
    published = _publish(1, 65)
    checkpoint_file = tmp_path / "checkpoint.pt"
    generator = torch.nn.Linear(2, 2)
    discriminator = torch.nn.Linear(2, 1)

    train_supervised.save_checkpoint(
        checkpoint_file,
        epoch=1,
        loss=8.3,
        best_loss=8.3,
        generator=generator,
        discriminator=discriminator,
    )
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

    assert checkpoint["guidance_state"] == published["guidance_state"]
    assert train_supervised._guidance_state_from_checkpoint({}, checkpoint) == published["guidance_state"]
    assert train_supervised._guidance_state_from_checkpoint({}, {"epoch": 1}) is None

    status_state = {"quality_history": [{"epoch": 9}]}
    assert train_supervised._guidance_state_from_checkpoint(
        {"guidance_state": status_state}, checkpoint
    ) == status_state
    assert train_supervised._guidance_state_from_checkpoint(
        {"guidance_state": status_state}, checkpoint, prefer_checkpoint=True
    ) == checkpoint["guidance_state"]
    assert status_file.is_file()


def test_past_era_quality_summary_tracks_best_observed_categories():
    history = [
        {"guidance": {"quality_vector": {"global": 70, "face": 55, "alpha": 99}}},
        {"guidance": {"quality_vector": {"global": 75, "face": 62, "alpha": 98}}},
    ]
    summary = train_supervised._quality_summary_from_history(history)

    assert summary["best_global"] == 75.0
    assert summary["best_face"] == 62.0
    assert summary["best_alpha"] == 99.0
    assert summary["best_palette"] is None


def test_monitor_reads_new_guidance_contract():
    project_root = train_supervised.PROJECT_ROOT
    monitor = (project_root / "monitor.html").read_text(encoding="utf-8")
    studio = (project_root / "sprite_studio.html").read_text(encoding="utf-8")

    assert "d.guidance || d.quality_guidance" in monitor
    assert "OBSERVACIONAL" in monitor
    assert "info.guidance || info.quality_guidance" in studio
