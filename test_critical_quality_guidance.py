import numpy as np
import inspect
import torch
from PIL import Image

from pixel_ai_engine import train_supervised
from pixel_ai_engine.phase3_critical_enhancer import Phase3CriticalReviewer
from pixel_ai_engine.quality_guidance import assess_quality_checkpoint, build_quality_vector


def _critical_sprite():
    data = np.zeros((16, 16, 4), dtype=np.uint8)
    data[2:15, 4:12, :3] = (80, 120, 170)
    data[2:15, 4:12, 3] = 255
    data[2:15, 4, :3] = 20
    data[2:15, 11, :3] = 20
    return Image.fromarray(data, "RGBA")


def test_critical_audit_is_pure_and_feeds_quality_vector():
    sprite = _critical_sprite()
    before = np.asarray(sprite).copy()
    metrics = Phase3CriticalReviewer.audit_frame(sprite)
    vector = build_quality_vector(metrics).to_dict()

    assert np.array_equal(np.asarray(sprite), before)
    assert vector["global"] == metrics["score_critico"]
    assert vector["face"] == metrics["detalle_facial"]
    assert vector["outline"] is not None
    assert vector["palette"] == metrics["riqueza_paleta"]


def test_training_preview_never_calls_destructive_critical_elevation():
    source = inspect.getsource(train_supervised.generate_preview)
    audit_source = inspect.getsource(train_supervised._audit_training_prediction)
    assert "Phase3CriticalReviewer.audit_frame" in audit_source
    assert "elevate_frame(" not in source + audit_source


def test_quality_checkpoint_criterion_requires_coverage_and_critical_floors():
    healthy = assess_quality_checkpoint(
        {"global": 82, "anatomy": 80, "silhouette": 78, "face": 75, "palette": 88, "alpha": 99}
    )
    collapsed = assess_quality_checkpoint(
        {"global": 82, "anatomy": 80, "silhouette": 78, "face": 75, "palette": 20, "alpha": 99}
    )
    sparse = assess_quality_checkpoint({"global": 90, "alpha": 99})

    assert healthy["eligible"] is True
    assert collapsed["eligible"] is False
    assert collapsed["critical_breaches"] == ["palette"]
    assert sparse["eligible"] is False


def test_best_quality_checkpoint_is_independent_and_only_improves(monkeypatch, tmp_path):
    monkeypatch.setattr(train_supervised, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(train_supervised, "ENABLE_QUALITY_CHECKPOINT", True)
    generator = torch.nn.Linear(2, 2)
    high_vector = {
        "global": 85, "anatomy": 82, "silhouette": 80, "face": 75,
        "palette": 90, "alpha": 99, "micro_detail": 74, "outline": 78,
    }
    first = train_supervised._maybe_save_quality_checkpoint(
        {"guidance": {"quality_vector": high_vector}, "guidance_state": {}},
        epoch=7,
        loss=2.0,
        best_loss=1.5,
        generator=generator,
    )
    quality_file = tmp_path / "best_quality_generator.pt"
    payload = torch.load(quality_file, map_location="cpu", weights_only=False)
    second = train_supervised._maybe_save_quality_checkpoint(
        {**first, "guidance": {"quality_vector": {**high_vector, "global": 80}}},
        epoch=8,
        loss=1.8,
        best_loss=1.5,
        generator=generator,
    )

    assert quality_file.is_file()
    assert not (tmp_path / "best_generator.pt").exists()
    assert payload["epoch"] == 7
    assert payload["quality_score"] == first["best_quality_score"]
    assert second["best_quality_score"] == first["best_quality_score"]
    assert second["quality_checkpoint"]["improved"] is False


def test_quality_checkpoint_feature_flag_disables_write(monkeypatch, tmp_path):
    monkeypatch.setattr(train_supervised, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(train_supervised, "ENABLE_QUALITY_CHECKPOINT", False)
    status = train_supervised._maybe_save_quality_checkpoint(
        {
            "guidance": {"quality_vector": {
                "global": 90, "anatomy": 90, "silhouette": 90, "face": 90,
                "palette": 90, "alpha": 99,
            }},
            "guidance_state": {},
        },
        epoch=1,
        loss=1.0,
        best_loss=1.0,
        generator=torch.nn.Linear(1, 1),
    )

    assert status["quality_checkpoint"]["enabled"] is False
    assert status["quality_checkpoint"]["improved"] is False
    assert not (tmp_path / "best_quality_generator.pt").exists()
