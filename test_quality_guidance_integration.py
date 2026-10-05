import json
import time

import torch
from torch.utils.data import WeightedRandomSampler

from pixel_ai_engine import train_supervised
from pixel_ai_engine.quality_guidance import SamplingPlan


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
    assert persisted["guidance"]["recommended_action"] == "ROLLBACK"
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
        {"guidance": {"quality_vector": {"global": 70, "face": 55, "alpha": 99, "pose": 73, "outline": 68}}},
        {"guidance": {"quality_vector": {"global": 75, "face": 62, "alpha": 98, "pose": 78, "outline": 72}}},
    ]
    summary = train_supervised._quality_summary_from_history(history)

    assert summary["best_global"] == 75.0
    assert summary["best_face"] == 62.0
    assert summary["best_alpha"] == 99.0
    assert summary["best_pose"] == 78.0
    assert summary["best_outline"] == 72.0
    assert summary["best_palette"] is None


def test_frame_quality_batch_is_per_sample_and_detached_from_training_graph():
    targets = torch.zeros(2, 4, 4, 4)
    targets[:, 3, 1:3, 1:3] = 1.0
    predictions = targets.clone().requires_grad_(True)
    predictions.data[1, :3] = 1.0
    predictions.data[1, 3] = -1.0

    measurements = train_supervised.compute_frame_quality_batch(
        predictions,
        targets,
        torch.tensor([3, 9]),
        ["hero_a", "hero_b"],
    )

    assert [item["frame_idx"] for item in measurements] == [3, 9]
    assert [item["char_id"] for item in measurements] == ["hero_a", "hero_b"]
    assert measurements[0]["quality"] == 100.0
    assert measurements[1]["quality"] < measurements[0]["quality"]
    assert predictions.grad is None


def test_dataset_target_audits_use_conservative_multi_character_score():
    audits = [
        {
            "character_id": f"hero_{index}",
            "frame_idx": index,
            "metrics": {
                "score_total": score,
                "strict_face": score - 5,
                "strict_anatomy": score - 3,
                "strict_silhouette": 95,
                "strict_visual_noise": score - 8,
                "cuerpo_precision": score - 3,
            },
        }
        for index, score in enumerate((40.0, 60.0, 80.0, 100.0))
    ]

    quality = train_supervised._aggregate_training_quality(audits)

    assert quality["comparison_source"] == "dataset_targets"
    assert quality["comparison_method"] == "percentil_25_conservador"
    assert quality["evaluated_sample_count"] == 4
    assert quality["score_total"] == 55.0
    assert quality["quality_average"] == 70.0
    assert quality["quality_worst"] == 40.0
    assert [sample["character_id"] for sample in quality["evaluated_samples"]] == [
        "hero_0", "hero_1", "hero_2", "hero_3"
    ]


def test_quality_sample_selection_rotates_characters_and_frames():
    samples = [
        {"char_id": character, "frame_idx": frame}
        for character in ("a", "b", "c", "d", "e")
        for frame in range(2)
    ]

    epoch_one = train_supervised._select_quality_sample_indices(samples, epoch_label=1, limit=4)
    epoch_two = train_supervised._select_quality_sample_indices(samples, epoch_label=2, limit=4)

    assert len(epoch_one) == len(epoch_two) == 4
    assert {samples[index]["char_id"] for index in epoch_one} != {
        samples[index]["char_id"] for index in epoch_two
    }
    assert all(samples[index]["frame_idx"] == 0 for index in epoch_one)
    assert all(samples[index]["frame_idx"] == 1 for index in epoch_two)


def test_active_sampling_plan_uses_weighted_sampler_and_is_persisted():
    dataset = torch.utils.data.TensorDataset(torch.arange(3))
    plan = SamplingPlan(
        active=True,
        reason="hard_example_sampling_active",
        weights=(2.0, 1.0, 1.5),
        sample_keys=("hero:0", "hero:1", "hero:2"),
        eligible_frames=3,
        hard_frames=2,
        min_weight=1.0,
        max_weight=2.0,
        effective_sample_size=2.7931,
    )

    loader = train_supervised._build_training_dataloader(dataset, 1, plan)
    status = train_supervised._attach_sampling_plan(
        {"epoch": 4, "history": [], "guidance_state": {}, "guidance": {"action": "CONTINUE"}},
        plan,
    )

    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert status["guidance"]["action"] == "ADJUST_SAMPLING"
    assert status["guidance"]["training_modified"] is True
    assert status["guidance_state"]["sampling_weights"] == {"hero:0": 2.0, "hero:2": 1.5}
    assert status["sampling"]["max_weight"] == 2.0
    assert status["sampling"]["ab"]["quality_improvement_claimed"] is False


def test_inactive_sampling_plan_keeps_random_sampler():
    dataset = torch.utils.data.TensorDataset(torch.arange(2))
    plan = SamplingPlan(
        active=False,
        reason="feature_disabled",
        weights=(1.0, 1.0),
        sample_keys=("hero:0", "hero:1"),
        eligible_frames=0,
        hard_frames=0,
        min_weight=1.0,
        max_weight=1.0,
        effective_sample_size=2.0,
    )

    loader = train_supervised._build_training_dataloader(dataset, 1, plan)

    assert not isinstance(loader.sampler, WeightedRandomSampler)


def test_guidance_runtime_frame_state_reaches_checkpoint(monkeypatch, tmp_path):
    _configure_status_test(monkeypatch, tmp_path)
    frame_state = {
        "hero:2": {
            "char_id": "hero",
            "frame_idx": 2,
            "quality": 42.0,
            "confidence": 1.0,
            "observations": 3,
            "last_epoch": 5,
        }
    }
    train_supervised.update_status(
        1,
        2,
        "ENTRENANDO",
        1.0,
        0.5,
        0.2,
        0.1,
        time.time(),
        quality=_quality(60),
        guidance_runtime={"frame_quality": frame_state},
    )
    checkpoint_file = tmp_path / "runtime-checkpoint.pt"
    train_supervised.save_checkpoint(
        checkpoint_file,
        1,
        1.0,
        1.0,
        torch.nn.Linear(1, 1),
        torch.nn.Linear(1, 1),
    )

    payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    restored = payload["guidance_state"]["frame_quality"]["hero:2"]
    assert restored["quality"] == 42.0
    assert restored["confidence"] == 1.0
    assert restored["observations"] == 3
    assert restored["last_epoch"] == 5


def test_adaptive_loss_plan_is_persisted_and_attributable(monkeypatch):
    monkeypatch.setattr(train_supervised, "ENABLE_QUALITY_GUIDANCE", True)
    monkeypatch.setattr(train_supervised, "ENABLE_ADAPTIVE_LOSS", True)
    plan = train_supervised._build_loss_plan(
        {"loss_multipliers": {"alpha": 1.0}},
        {"recommended_action": "ADJUST_WEIGHTS", "primary_problem": "alpha"},
    )
    status = train_supervised._attach_loss_plan(
        {"epoch": 2, "history": [], "guidance_state": {}, "guidance": {"action": "CONTINUE"}},
        plan,
    )

    assert plan["changed_count"] == 1
    assert status["guidance"]["action"] == "ADJUST_WEIGHTS"
    assert status["guidance"]["training_modified"] is True
    assert status["guidance_state"]["loss_multipliers"]["alpha"] == 1.1
    assert status["adaptive_loss"]["ab"]["quality_improvement_claimed"] is False


def test_adaptive_loss_feature_flag_restores_exact_base_weights(monkeypatch):
    monkeypatch.setattr(train_supervised, "ENABLE_ADAPTIVE_LOSS", False)
    plan = train_supervised._build_loss_plan(
        {"loss_multipliers": {"color": 1.25, "edge": 1.25}},
        {"recommended_action": "ADJUST_WEIGHTS", "primary_problem": "palette"},
    )

    assert plan["active"] is False
    assert plan["effective_weights"] == {
        "color": 5.0,
        "alpha": 2.5,
        "edge": 1.5,
        "adversarial": 0.05,
    }


def test_intervention_policy_state_is_attached_for_resume():
    status = train_supervised._apply_intervention_policy(
        {
            "epoch": 11,
            "history": [{"epoch": 11}],
            "guidance_state": {},
            "guidance": {
                "recommended_action": "ROLLBACK",
                "trend": {"status": "COLLAPSE"},
            },
        }
    )

    assert status["guidance"]["authorized_action"] == "ROLLBACK"
    assert status["guidance_state"]["intervention_count"] == 1
    assert status["guidance_state"]["intervention_state"]["last_intervention_epoch"] == 11
    assert status["history"][0]["guidance"]["authorized_action"] == "ROLLBACK"


def test_policy_authorization_prevents_simultaneous_sampling_and_loss(monkeypatch):
    monkeypatch.setattr(train_supervised, "ENABLE_SMART_SAMPLING", True)
    monkeypatch.setattr(train_supervised, "ENABLE_ADAPTIVE_LOSS", True)

    sampling_guidance = {
        "recommended_action": "REINFORCE",
        "authorized_action": "ADJUST_SAMPLING",
        "primary_problem": "face",
    }
    loss_guidance = {
        "recommended_action": "ADJUST_WEIGHTS",
        "authorized_action": "ADJUST_WEIGHTS",
        "primary_problem": "alpha",
    }
    loss_during_sampling = train_supervised._build_loss_plan({}, sampling_guidance)
    loss_plan = train_supervised._build_loss_plan({}, loss_guidance)

    assert loss_during_sampling["active"] is False
    assert loss_plan["active"] is True


def test_monitor_reads_new_guidance_contract():
    project_root = train_supervised.PROJECT_ROOT
    monitor = (project_root / "monitor.html").read_text(encoding="utf-8")
    studio = (project_root / "sprite_studio.html").read_text(encoding="utf-8")

    assert "d.guidance || d.quality_guidance" in monitor
    assert "OBSERVACIONAL" in monitor
    assert "anatomía ${vector.anatomy" in monitor
    assert 'id="qc-live-state-text"' in monitor
    assert 'id="qc-comparison-source"' in monitor
    assert "targets reales" in monitor
    assert "AJUSTANDO" in monitor
    assert "updateQualityHero(d, guidance)" in monitor
    assert "info.guidance || info.quality_guidance" in studio
    assert 'id="monQualityCard"' in studio
    assert 'id="monQualityScore"' in studio
    assert 'id="monQualityState"' in studio
    assert 'id="monQualityComparison"' in studio
    assert "updateTrainingQualityPanel(info)" in studio
    assert "quality.evaluated_sample_count" in studio


def test_terminal_quality_status_reports_when_training_is_adjusting(capsys):
    train_supervised._print_quality_guidance({
        "enabled": True,
        "training_modified": True,
        "quality_vector": {"global": 72.5},
        "recommended_action": "REINFORCE",
        "authorized_action": "ADJUST_SAMPLING",
    })

    output = capsys.readouterr().out
    assert "Global: 72.5%" in output
    assert "Acción autorizada: ADJUST_SAMPLING" in output
    assert "MODO ACTUAL: AJUSTANDO" in output


def test_cpu_smoke_training_completes_and_resumes_with_guidance(monkeypatch, tmp_path):
    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return (
                torch.zeros(3, 8, 8),
                0,
                torch.zeros(4, 8, 8),
                "tiny",
            )

    class TinyGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Conv2d(6, 4, kernel_size=1)

        def forward(self, value):
            return torch.tanh(self.layer(value))

    class TinyDiscriminator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Conv2d(10, 1, kernel_size=1)

        def forward(self, condition, target):
            return self.layer(torch.cat([condition, target], dim=1))

    class TinyTemplateManager:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_frame_tensor(self, _frame_index):
            return torch.zeros(3, 8, 8)

    import pixel_ai_engine.dataset as dataset_module

    checkpoint_dir = tmp_path / "checkpoints"
    snapshots_dir = checkpoint_dir / "snapshots"
    checkpoint_dir.mkdir()
    snapshots_dir.mkdir()
    monkeypatch.setattr(train_supervised, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(train_supervised, "SNAPSHOTS_DIR", snapshots_dir)
    monkeypatch.setattr(train_supervised, "STATUS_FILE", tmp_path / "training_status.json")
    monkeypatch.setattr(train_supervised, "STOP_FLAG_FILE", tmp_path / "stop.flag")
    monkeypatch.setattr(train_supervised, "PAUSE_FLAG_FILE", tmp_path / "pause.flag")
    monkeypatch.setattr(train_supervised, "CACHE_PATH", tmp_path / "cache.pt")
    monkeypatch.setattr(train_supervised, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(train_supervised, "USE_AMP", False)
    monkeypatch.setattr(train_supervised, "HAS_KORNIA", False)
    monkeypatch.setattr(train_supervised, "SupervisedTensorDataset", TinyDataset)
    monkeypatch.setattr(train_supervised, "PixelArtUNetGenerator", TinyGenerator)
    monkeypatch.setattr(train_supervised, "PixelArtPatchDiscriminator", TinyDiscriminator)
    monkeypatch.setattr(dataset_module, "TemplateManager", TinyTemplateManager)
    monkeypatch.setattr(train_supervised, "get_hardware_telemetry", _telemetry)
    monkeypatch.setattr(train_supervised, "get_available_snapshots", lambda: [])
    monkeypatch.setattr(train_supervised, "generate_preview", lambda *_args, **_kwargs: _quality(65))
    monkeypatch.setattr(train_supervised, "manage_adaptive_thermal_throttle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_supervised, "detect_training_instability", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_supervised, "repair_current_training_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_supervised, "ENABLE_QUALITY_GUIDANCE", True)

    train_supervised.train_supervised_model(epochs=1, batch_size=1, lr=1e-4, mode="start")
    first = json.loads(train_supervised.STATUS_FILE.read_text(encoding="utf-8"))
    assert first["status"] == "COMPLETADO"
    assert first["epoch"] == 1
    assert first["guidance"]["action"] == "CONTINUE"
    assert (checkpoint_dir / "latest_checkpoint.pt").is_file()

    train_supervised.train_supervised_model(epochs=1, batch_size=1, lr=1e-4, mode="resume")
    resumed = json.loads(train_supervised.STATUS_FILE.read_text(encoding="utf-8"))
    assert resumed["status"] == "COMPLETADO"
    assert resumed["epoch"] == 2
    assert [entry["epoch"] for entry in resumed["history"]] == [1, 2]
    assert len(resumed["guidance_state"]["quality_history"]) == 2

    class PauseDuringBatch:
        def __init__(self):
            self.checks = 0

        def exists(self):
            self.checks += 1
            return self.checks == 3

        def unlink(self):
            pass

    monkeypatch.setattr(train_supervised, "PAUSE_FLAG_FILE", PauseDuringBatch())
    train_supervised.train_supervised_model(epochs=1, batch_size=1, lr=1e-4, mode="resume")
    paused = json.loads(train_supervised.STATUS_FILE.read_text(encoding="utf-8"))
    assert paused["status"] == "PAUSADO"
    assert paused["epoch"] == 3
    assert paused["guidance_state"]["quality_history"][-1]["epoch"] == 3

    monkeypatch.setattr(train_supervised, "PAUSE_FLAG_FILE", tmp_path / "pause-after-resume.flag")
    train_supervised.train_supervised_model(epochs=1, batch_size=1, lr=1e-4, mode="resume")
    after_pause = json.loads(train_supervised.STATUS_FILE.read_text(encoding="utf-8"))
    assert after_pause["status"] == "COMPLETADO"
    assert after_pause["epoch"] == 4
    assert [entry["epoch"] for entry in after_pause["history"]] == [1, 2, 3, 4]
