import json

from PIL import Image, ImageDraw
import pytest

from pixel_ai_engine.frame_quality_review import FrameQualityReviewManager
from pixel_ai_engine.frame_regeneration import FrameRegenerationManager, rank_regeneration_candidates


def _sprite(color):
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((4, 2, 11, 14), fill=(*color, 255))
    return image


def _tree(tmp_path):
    run = tmp_path / "output" / "run_001"
    frame = run / "enhanced_frames" / "frame_003.png"
    frame.parent.mkdir(parents=True)
    _sprite((10, 10, 10)).save(frame)
    (run / "metadata.json").write_text(json.dumps({
        "character": "New Character",
        "variant": "rnormal",
        "format": "8x12",
        "front_image": "personajes/new/new_rnormal.png",
    }), encoding="utf-8")
    variant = tmp_path / "dataset_frames_individuales" / "NEW CHARACTER" / "new_ropa_normal"
    variant.mkdir(parents=True)
    target = variant / "frame_004.png"
    _sprite((255, 255, 255)).save(target)
    _sprite((200, 200, 200)).save(variant / "00_frontal_identidad.png")
    (variant / "manifest.json").write_text(json.dumps({
        "character": "New Character",
        "variant": "rnormal",
        "frames": [{"slot": 4, "file": target.name}],
    }), encoding="utf-8")
    manager = FrameQualityReviewManager(
        tmp_path / "reviews.jsonl",
        project_root=tmp_path,
        output_root=tmp_path / "output",
        dataset_frames_root=tmp_path / "dataset_frames_individuales",
        supervised_root=tmp_path / "dataset_supervisado",
        evaluator=lambda *args, **kwargs: {
            "audit_available": True,
            "aprobado": False,
            "score_total": 58.0,
            "quality": {"global": 58.0, "face": 40.0, "anatomy": 90.0, "alpha": 100.0},
            "issues": ["face"],
            "severity": "high",
            "diagnosis": {"primary_problem": "face"},
        },
    )
    review = manager.evaluate_frame("run_001", 3)
    return manager, review, frame


def test_candidate_ranking_rejects_critical_regression_and_selects_real_improvement():
    original = {
        "quality": {"global": 58, "face": 40, "props": 55, "anatomy": 90, "alpha": 100, "palette": 90},
        "diagnosis": {"primary_problem": "face"},
    }
    result = rank_regeneration_candidates(original, [
        {"candidate_id": "candidate_bad", "quality": {"global": 76, "face": 75, "props": 70, "anatomy": 55, "alpha": 100, "palette": 90}},
        {"candidate_id": "candidate_good", "quality": {"global": 86, "face": 82, "props": 70, "anatomy": 89, "alpha": 100, "palette": 91}},
    ])

    assert result["best_candidate"] == "candidate_good"
    bad = next(item for item in result["candidates"] if item["candidate_id"] == "candidate_bad")
    assert bad["safe"] is False
    assert "REGRESSION_DETECTED" in bad["violations"]


def test_candidate_ranking_requires_global_improvement_for_non_metric_diagnosis():
    original = {
        "score_total": 70.0,
        "quality": {"global": 70.0, "alpha": 100.0, "anatomy": 80.0},
        "diagnosis": {"primary_problem": "training_stability"},
    }
    candidates = [{
        "candidate_id": "candidate_01",
        "score_total": 69.0,
        "quality": {"global": 69.0, "alpha": 100.0, "anatomy": 80.0},
        "issues": [],
    }]

    ranked = rank_regeneration_candidates(original, candidates)

    assert ranked["primary_problem"] == "global"
    assert ranked["best_candidate"] is None
    assert "dominant_problem_not_improved" in ranked["candidates"][0]["violations"]


class _Generator:
    def generate(self, review, num_candidates):
        return [_sprite((120, 0, 0)), _sprite((0, 200, 0))][:num_candidates]


def _candidate_audit(image, *_args, **_kwargs):
    pixel = image.convert("RGBA").getpixel((5, 5))
    if pixel[1] > pixel[0]:
        quality = {"global": 82.0, "face": 80.0, "anatomy": 90.0, "alpha": 100.0, "palette": 90.0}
    else:
        quality = {"global": 62.0, "face": 50.0, "anatomy": 90.0, "alpha": 100.0, "palette": 90.0}
    return {
        "audit_available": True,
        "score_total": quality["global"],
        "quality": quality,
        "issues": [],
        "severity": "low",
        "diagnosis": {"primary_problem": None},
    }


def test_regeneration_candidates_are_isolated_and_best_is_applied_with_backup(tmp_path):
    manager, review, active_frame = _tree(tmp_path)
    service = FrameRegenerationManager(manager, generator=_Generator(), evaluator=_candidate_audit)
    before = active_frame.read_bytes()

    regenerated = service.regenerate(review["review_id"], num_candidates=2, actor="qa")

    assert regenerated["status"] == "REGENERATED"
    assert regenerated["regeneration"]["ranking"]["best_candidate"] == "candidate_02"
    assert active_frame.read_bytes() == before
    candidate_dir = tmp_path / regenerated["regeneration"]["candidate_dir"]
    assert (candidate_dir / "candidate_01.png").is_file()
    assert (candidate_dir / "candidate_02.png").is_file()

    applied = service.apply_best(review["review_id"], actor="qa")

    assert applied["status"] == "REEVALUATED"
    assert active_frame.read_bytes() != before
    assert (candidate_dir / "frame_003_original.png").read_bytes() == before
    actions = [event["action"] for event in applied["history"]]
    assert actions[-2:] == ["candidate_applied", "reevaluated"]
    with pytest.raises(ValueError, match="regeneration_not_ready"):
        service.apply_best(review["review_id"], actor="qa")
    with pytest.raises(ValueError, match="regeneration_not_discardable"):
        service.discard(review["review_id"], actor="qa")


def test_discard_regeneration_preserves_original_frame(tmp_path):
    manager, review, active_frame = _tree(tmp_path)
    service = FrameRegenerationManager(manager, generator=_Generator(), evaluator=_candidate_audit)
    before = active_frame.read_bytes()
    service.regenerate(review["review_id"], num_candidates=1)

    discarded = service.discard(review["review_id"], actor="qa")

    assert discarded["regeneration"]["state"] == "DISCARDED"
    assert discarded["status"] == "NEEDS_REVIEW"
    assert active_frame.read_bytes() == before
