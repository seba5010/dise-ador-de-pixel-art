from pixel_ai_engine.hard_examples import HardExampleQueue, load_manual_sampling_weights
from pixel_ai_engine.quality_guidance import HardExampleMiningPolicy
from test_frame_regeneration import _tree


def test_manual_reinforcement_deduplicates_and_never_uses_generated_as_target(tmp_path):
    manager, review, _ = _tree(tmp_path)
    manager.reject_frame(review["review_id"], actor="qa", reason="face")
    queue = HardExampleQueue(manager, tmp_path / "hard_examples.jsonl")

    first = queue.enqueue(review["review_id"], actor="qa")
    second = queue.enqueue(review["review_id"], actor="qa")

    assert len(queue.list()) == 1
    assert first["target_verified"] is True
    assert first["target_frame_path"] != first["generated_frame_path"]
    assert second["times_failed"] == 2
    assert 1.0 < second["priority"] <= 2.0
    updated_review = manager.get_review(review["review_id"])
    assert updated_review["status"] == "SENT_TO_REINFORCEMENT"


def test_reinforcement_is_blocked_when_target_is_missing(tmp_path):
    manager, review, _ = _tree(tmp_path)
    target = tmp_path / review["target_frame_path"]
    target.unlink()
    queue = HardExampleQueue(manager, tmp_path / "hard_examples.jsonl")

    try:
        queue.enqueue(review["review_id"])
    except ValueError as error:
        assert str(error) == "target_not_found"
    else:
        raise AssertionError("reinforcement must be blocked without a target")
    assert queue.list() == []


def test_priority_decays_and_resolves_after_quality_improves(tmp_path):
    manager, review, _ = _tree(tmp_path)
    queue = HardExampleQueue(manager, tmp_path / "hard_examples.jsonl")
    queued = queue.enqueue(review["review_id"])
    improved = dict(manager.get_review(review["review_id"]))
    improved["score_total"] = 85.0
    improved["status"] = "APPROVED"

    resolved = queue.record_outcome(improved)

    assert resolved["priority"] == 1.0
    assert resolved["active"] is False
    assert queue.summary()["resolved"] == 1
    assert queue.manual_weights() == {}


def test_unchanged_first_outcome_does_not_decay_from_zero_baseline(tmp_path):
    manager, review, _ = _tree(tmp_path)
    queue = HardExampleQueue(manager, tmp_path / "hard_examples.jsonl")
    queued = queue.enqueue(review["review_id"])

    unchanged = queue.record_outcome(manager.get_review(review["review_id"]))

    assert unchanged["priority"] == queued["priority"]


def test_manual_feedback_overrides_automatic_sampling_conservatively(tmp_path):
    path = tmp_path / "hard_examples.jsonl"
    path.write_text(
        '{"active":true,"target_verified":true,"character_id":"New Character",'
        '"variant":"rnormal","frame_idx":3,"priority":1.7}\n',
        encoding="utf-8",
    )
    manual = load_manual_sampling_weights(path)
    plan = HardExampleMiningPolicy().build_plan(
        [{"char_id": "new_character_rnormal", "frame_idx": 3}],
        {},
        enabled=True,
        recommendation="CONTINUE",
        manual_weights=manual,
    )

    assert plan.active is True
    assert plan.reason == "manual_hard_examples_active"
    assert plan.weights == (1.7,)
    assert plan.max_weight <= 2.0
