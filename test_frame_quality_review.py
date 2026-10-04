import json
from pathlib import Path

import pytest

from pixel_ai_engine.quality_gate import QualityGate
from pixel_ai_engine.frame_quality_review import FrameQualityReviewManager, VALID_REVIEW_STATUSES


def test_quality_gate_evaluate_single_frame_requires_reference_data(tmp_path):
    generated = tmp_path / "generated.png"
    generated.write_bytes(b"fake")

    result = QualityGate.evaluate_single_frame(generated, None, metadata={"character_id": "ALEX", "frame_idx": 5})

    assert result["audit_available"] is False
    assert result["aprobado"] is False
    assert result["score_total"] is None
    assert result["issues"] == []


def test_frame_quality_review_manager_registers_and_reevaluates_frame(tmp_path):
    queue_path = tmp_path / "frame_review_queue.jsonl"
    manager = FrameQualityReviewManager(base_dir=tmp_path, queue_path=queue_path)

    generated = tmp_path / "generated.png"
    target = tmp_path / "target.png"
    generated.write_bytes(b"generated")
    target.write_bytes(b"target")

    review = manager.register_frame(
        character_id="ALEX",
        variant="chef_white",
        frame_idx=12,
        generated_frame_path=str(generated),
        target_frame_path=str(target),
        metadata={"row": 1, "column": 2, "epoch": 5},
    )

    assert review["status"] == "NEEDS_REVIEW"
    assert review["review_id"]
    assert queue_path.exists()

    reevaluated = manager.reevaluate_frame(
        review["review_id"],
        quality={
            "global": 58.0,
            "anatomy": 78.0,
            "silhouette": 70.0,
            "face": 49.0,
            "clothing": 92.0,
            "props": 55.0,
            "palette": 92.0,
            "alpha": 100.0,
            "micro_detail": 63.0,
        },
        issues=["face", "props", "micro_detail"],
    )

    assert reevaluated["status"] == "REEVALUATED"
    assert reevaluated["quality"]["global"] == 58.0
    assert reevaluated["times_reevaluated"] == 1
    assert len(reevaluated["history"]) >= 2


def test_frame_quality_review_manager_approve_and_reject_are_persisted(tmp_path):
    manager = FrameQualityReviewManager(base_dir=tmp_path)
    review = manager.register_frame(
        character_id="ALEX",
        variant="chef_white",
        frame_idx=3,
        generated_frame_path=str(tmp_path / "a.png"),
        target_frame_path=str(tmp_path / "b.png"),
    )

    approved = manager.approve_frame(review["review_id"], user="qa")
    assert approved["status"] == "APPROVED"

    other = manager.register_frame(
        character_id="AMARO",
        variant="urban",
        frame_idx=9,
        generated_frame_path=str(tmp_path / "c.png"),
        target_frame_path=str(tmp_path / "d.png"),
    )

    rejected = manager.reject_frame(other["review_id"], reason="bad face")
    assert rejected["status"] == "REJECTED"
    assert rejected["user_action"] == "REJECTED"

    pending = manager.get_queue(status="APPROVED")
    assert len(pending) == 1
    assert pending[0]["review_id"] == approved["review_id"]


def test_review_statuses_are_restricted_to_valid_vocab(tmp_path):
    assert set(VALID_REVIEW_STATUSES) >= {"OK", "NEEDS_REVIEW", "REEVALUATED", "APPROVED", "REJECTED"}
    assert "INVALID_STATUS" not in VALID_REVIEW_STATUSES


def test_manager_resolves_safe_paths_and_keeps_history(tmp_path):
    manager = FrameQualityReviewManager(base_dir=tmp_path)
    safe_target = tmp_path / "safe_target.png"
    safe_target.write_bytes(b"target")

    review = manager.register_frame(
        character_id="CONNY",
        variant="variant",
        frame_idx=21,
        generated_frame_path=str(tmp_path / "generated.png"),
        target_frame_path=str(safe_target),
    )

    assert review["target_frame_path"] == str(safe_target)
    assert review["history"][-1]["action"] == "registered"
    assert manager.resolve_target_path("CONNY", "variant", 21) == str(safe_target)

    escape_path = tmp_path / ".." / "outside.png"
    assert manager._ensure_safe_path(str(escape_path)) is None
