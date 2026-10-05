import json
import threading
import urllib.error
import urllib.request

import pytest
from PIL import Image

from pixel_ai_engine.frame_quality_review import FrameQualityReviewManager, FrameReviewStatus
from pixel_ai_engine.quality_gate import QualityGate


def _png(path, alpha=255):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (16, 16), (120, 80, 40, alpha)).save(path)


def _fixture_tree(tmp_path, *, include_target=True):
    output = tmp_path / "output"
    run = output / "run_001"
    generated = run / "enhanced_frames" / "frame_003.png"
    _png(generated)
    (run / "metadata.json").write_text(
        json.dumps({
            "character": "Ada Lovelace",
            "variant": "rnormal",
            "front_image": "personajes/ada_lovelace/ada_lovelace_rnormal.png",
            "format": "8x12",
        }),
        encoding="utf-8",
    )

    dataset = tmp_path / "dataset_frames_individuales"
    variant_dir = dataset / "ADA LOVELACE" / "ada_lovelace_ropa_normal"
    variant_dir.mkdir(parents=True)
    frame_file = variant_dir / "frame_004_r01_c04.png"
    if include_target:
        _png(frame_file)
    _png(variant_dir / "00_frontal_identidad.png")
    (variant_dir / "manifest.json").write_text(
        json.dumps({
            "character": "Ada Lovelace",
            "variant": "rnormal",
            "variant_name": "ropa_normal",
            "frames": [{"slot": 4, "row": 1, "column": 4, "file": frame_file.name}],
        }),
        encoding="utf-8",
    )
    supervised = tmp_path / "dataset_supervisado"
    supervised.mkdir()
    return output, dataset, supervised, generated, frame_file


def _audit(_generated, target, reference_front=None, frame_idx=None, metadata=None):
    available = target is not None and target.is_file()
    return {
        "audit_available": available,
        "aprobado": available,
        "score_total": 88.0 if available else None,
        "quality": {"global": 88.0, "face": 81.0, "alpha": 100.0} if available else {"global": None},
        "issues": [],
        "severity": "low",
        "diagnosis": {"primary_problem": None},
        "identity_confidence": None,
    }


def _manager(tmp_path, *, include_target=True, evaluator=_audit):
    output, dataset, supervised, generated, target = _fixture_tree(tmp_path, include_target=include_target)
    manager = FrameQualityReviewManager(
        tmp_path / "frame_review_queue.jsonl",
        project_root=tmp_path,
        output_root=output,
        dataset_frames_root=dataset,
        supervised_root=supervised,
        evaluator=evaluator,
    )
    return manager, generated, target


def test_review_status_vocabulary_rejects_arbitrary_values():
    assert FrameQualityReviewManager.validate_status(FrameReviewStatus.APPROVED) == "APPROVED"
    with pytest.raises(ValueError, match="Unsupported"):
        FrameQualityReviewManager.validate_status("LOOKS_FINE")


def test_evaluate_persists_exact_target_identity_and_history(tmp_path):
    manager, generated, target = _manager(tmp_path)
    review = manager.evaluate_frame("run_001", 3, epoch=12, actor="tester")

    assert review["status"] == "OK"
    assert review["character_id"] == "Ada Lovelace"
    assert review["variant"] == "rnormal"
    assert review["frame_idx"] == 3
    assert review["row"] == 0 and review["column"] == 3
    assert review["generated_frame_path"] == generated.relative_to(tmp_path).as_posix()
    assert review["target_frame_path"] == target.relative_to(tmp_path).as_posix()
    assert review["history"][-1]["action"] == "evaluated"

    resumed = FrameQualityReviewManager(
        manager.queue_path,
        project_root=tmp_path,
        output_root=manager.output_root,
        dataset_frames_root=manager.dataset_frames_root,
        supervised_root=manager.supervised_root,
        evaluator=_audit,
    )
    assert resumed.get_review(review["review_id"]) == review
    assert resumed.summary()["total"] == 1
    assert resumed.summary()["pending"] == 1


def test_reevaluate_updates_metrics_without_modifying_images(tmp_path):
    calls = {"count": 0}

    def evaluator(*args, **kwargs):
        calls["count"] += 1
        result = _audit(*args, **kwargs)
        result["score_total"] = 70.0 + calls["count"]
        result["quality"]["global"] = result["score_total"]
        result["aprobado"] = False
        result["issues"] = ["face"]
        result["severity"] = "medium"
        return result

    manager, generated, target = _manager(tmp_path, evaluator=evaluator)
    generated_before = generated.read_bytes()
    target_before = target.read_bytes()
    created = manager.evaluate_frame("run_001", 3)
    reviewed = manager.reevaluate_frame(created["review_id"], actor="human")

    assert reviewed["status"] == "REEVALUATED"
    assert reviewed["score_total"] == 72.0
    assert reviewed["times_reevaluated"] == 1
    assert [event["action"] for event in reviewed["history"]] == ["evaluated", "reevaluated"]
    assert generated.read_bytes() == generated_before
    assert target.read_bytes() == target_before


def test_approve_and_reject_are_persisted_with_actor(tmp_path):
    manager, _, _ = _manager(tmp_path)
    review = manager.evaluate_frame("run_001", 3)
    approved = manager.approve_frame(review["review_id"], actor="qa-user", reason="verified")
    rejected = manager.reject_frame(review["review_id"], actor="qa-user", reason="visual defect")

    assert approved["status"] == "APPROVED"
    assert approved["approved_by"] == "qa-user"
    assert rejected["status"] == "REJECTED"
    assert rejected["times_failed"] == 1
    assert rejected["history"][-1]["action"] == "rejected"
    assert manager.summary()["rejected"] == 1


def test_missing_target_is_error_and_cannot_be_approved(tmp_path):
    manager, _, _ = _manager(tmp_path, include_target=False)
    review = manager.evaluate_frame("run_001", 3)

    assert review["status"] == "ERROR"
    assert review["audit_available"] is False
    assert review["score_total"] is None
    with pytest.raises(ValueError, match="missing_target"):
        manager.approve_frame(review["review_id"])


def test_path_traversal_and_unknown_run_are_rejected(tmp_path):
    manager, _, _ = _manager(tmp_path)
    with pytest.raises(ValueError, match="Invalid run_id"):
        manager.resolve_run_frame("../outside", 3)
    with pytest.raises(FileNotFoundError, match="Run not found"):
        manager.resolve_run_frame("unknown", 3)


def test_evaluate_is_idempotent_per_run_and_frame(tmp_path):
    manager, _, _ = _manager(tmp_path)
    first = manager.evaluate_frame("run_001", 3)
    second = manager.evaluate_frame("run_001", 3)

    assert first["review_id"] == second["review_id"]
    assert len(manager.list_reviews()) == 1
    assert len(second["history"]) == 2


def test_quality_gate_uses_real_target_metrics_and_rejects_fractional_alpha(tmp_path):
    generated = Image.new("RGBA", (24, 24), (0, 0, 0, 0))
    target = Image.new("RGBA", (24, 24), (0, 0, 0, 0))
    for image in (generated, target):
        for x in range(6, 18):
            for y in range(4, 21):
                image.putpixel((x, y), (80, 120, 160, 255))
    generated.putpixel((10, 10), (80, 120, 160, 128))
    generated_path = tmp_path / "generated.png"
    target_path = tmp_path / "target.png"
    generated.save(generated_path)
    target.save(target_path)

    result = QualityGate.evaluate_single_frame(generated_path, target_path, frame_idx=2)

    assert result["audit_available"] is True
    assert result["quality"]["silhouette"] is not None
    assert "ALPHA_FAIL" in result["issues"]
    assert result["aprobado"] is False
    assert result["raw_metrics"]["silueta_iou_real"] > 99.0


def test_quality_gate_never_approves_without_target(tmp_path):
    generated = tmp_path / "generated.png"
    _png(generated)

    result = QualityGate.evaluate_single_frame(generated, None)

    assert result["audit_available"] is False
    assert result["aprobado"] is False
    assert result["score_total"] is None
    assert result["diagnosis"]["reason"] == "target_not_found"


def _json_request(url, *, method="GET", payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def test_frame_review_rest_endpoints_validate_and_persist(tmp_path, monkeypatch):
    import sprite_studio
    from pixel_ai_engine.hard_examples import HardExampleQueue
    from pixel_ai_engine.interactive_qc import InteractiveQualityControlService

    manager, _, _ = _manager(tmp_path)
    hard_examples = HardExampleQueue(manager, tmp_path / "hard_examples.jsonl")
    service = InteractiveQualityControlService(
        manager,
        hard_examples=hard_examples,
        batch_path=tmp_path / "qc_batch_jobs.json",
    )
    monkeypatch.setattr(sprite_studio, "FRAME_REVIEW_MANAGER", manager)
    monkeypatch.setattr(sprite_studio, "HARD_EXAMPLE_QUEUE", hard_examples)
    monkeypatch.setattr(sprite_studio, "QC_SERVICE", service)
    server = sprite_studio.ThreadingHTTPServer(("127.0.0.1", 0), sprite_studio.SpriteStudioHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, evaluated = _json_request(
            base + "/api/qc/frame/evaluate",
            method="POST",
            payload={"run_id": "run_001", "frame_idx": 3, "user": "qa"},
        )
        assert status == 200
        review_id = evaluated["review"]["review_id"]

        status, queue = _json_request(base + "/api/qc/review-queue?run_id=run_001")
        assert status == 200
        assert queue["count"] == 1
        assert queue["capabilities"]["regeneration"] is True

        status, record = _json_request(base + f"/api/qc/frame/{review_id}")
        assert status == 200
        assert record["target_frame_path"].endswith("frame_004_r01_c04.png")

        status, approved = _json_request(
            base + f"/api/qc/frame/{review_id}/approve",
            method="POST",
            payload={"user": "qa", "reason": "visual check"},
        )
        assert status == 200
        assert approved["review"]["status"] == "APPROVED"

        status, rejected = _json_request(
            base + f"/api/qc/frame/{review_id}/reject",
            method="POST",
            payload={"user": "qa", "reason": "changed decision"},
        )
        assert status == 200
        assert rejected["review"]["status"] == "REJECTED"

        status, error = _json_request(
            base + "/api/qc/frame/evaluate",
            method="POST",
            payload={"frame_idx": 3},
        )
        assert status == 400
        assert "obligatorios" in error["error"]

        status, error = _json_request(base + f"/api/qc/frame/{review_id}/approve")
        assert status == 400
        assert "inválido" in error["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
