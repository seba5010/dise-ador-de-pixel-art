import json
import socket
import threading
import time

from pixel_ai_engine.hard_examples import HardExampleQueue
from pixel_ai_engine.interactive_qc import InteractiveQualityControlService
from test_frame_regeneration import _tree
from test_frame_quality_review import _json_request, _manager


class _FakeRegeneration:
    def __init__(self, manager, delay=0.0):
        self.manager = manager
        self.delay = delay
        self.calls = []

    def regenerate(self, review_id, *, num_candidates=3, actor=None):
        self.calls.append((review_id, num_candidates, actor))
        if self.delay:
            time.sleep(self.delay)
        return self.manager.get_review(review_id)

    def apply_best(self, review_id, *, candidate_id=None, actor=None):
        return self.manager.get_review(review_id)

    def discard(self, review_id, *, actor=None):
        return self.manager.get_review(review_id)


def _service(tmp_path, delay=0.0):
    manager, review, _ = _tree(tmp_path)
    hard = HardExampleQueue(manager, tmp_path / "hard_examples.jsonl")
    regeneration = _FakeRegeneration(manager, delay=delay)
    service = InteractiveQualityControlService(
        manager,
        regeneration_manager=regeneration,
        hard_examples=hard,
        batch_path=tmp_path / "batch_jobs.json",
    )
    return service, manager, review, regeneration


def test_batch_reevaluate_and_reinforcement_are_persisted(tmp_path):
    service, manager, review, _ = _service(tmp_path)

    reevaluation = service.start_batch("REEVALUATE_ALL", background=False, actor="qa")
    reinforcement = service.start_batch(
        "REINFORCE_SELECTED",
        review_ids=[review["review_id"]],
        background=False,
        actor="qa",
    )

    assert service.get_batch(reevaluation["job_id"])["status"] == "COMPLETED"
    assert service.get_batch(reinforcement["job_id"])["status"] == "COMPLETED"
    assert service.hard_examples.summary()["active"] == 1
    assert manager.get_review(review["review_id"])["status"] == "SENT_TO_REINFORCEMENT"


def test_batch_regeneration_only_selects_defective_frames(tmp_path):
    service, _, review, regeneration = _service(tmp_path)

    job = service.start_batch("REGENERATE_DEFECTIVE", background=False, num_candidates=2)

    persisted = service.get_batch(job["job_id"])
    assert persisted["status"] == "COMPLETED"
    assert persisted["review_ids"] == [review["review_id"]]
    assert regeneration.calls[0][1] == 2


def test_batch_can_be_cancelled_and_resume_marks_stale_jobs_interrupted(tmp_path):
    service, manager, review, regeneration = _service(tmp_path, delay=0.15)
    job = service.start_batch(
        "REGENERATE_DEFECTIVE",
        review_ids=[review["review_id"]],
        background=True,
    )
    service.cancel_batch(job["job_id"])
    deadline = time.time() + 2.0
    while time.time() < deadline and service.get_batch(job["job_id"])["status"] in {"RUNNING", "CANCELLING", "PENDING"}:
        time.sleep(0.02)
    assert service.get_batch(job["job_id"])["status"] == "CANCELLED"

    path = tmp_path / "batch_jobs.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["stale"] = {"job_id": "stale", "status": "RUNNING", "created_at": "x"}
    path.write_text(json.dumps(data), encoding="utf-8")
    resumed = InteractiveQualityControlService(
        manager,
        regeneration_manager=regeneration,
        hard_examples=service.hard_examples,
        batch_path=path,
    )
    assert resumed.get_batch("stale")["status"] == "INTERRUPTED"


def test_loop_metrics_report_regeneration_and_recurrent_failures(tmp_path):
    service, manager, review, _ = _service(tmp_path)
    manager.reject_frame(review["review_id"])
    manager.reject_frame(review["review_id"])

    metrics = service.metrics()

    assert metrics["frames_rejected"] == 1
    assert metrics["recurrent_failures"] == 1


def test_extended_qc_endpoints_and_malformed_requests(tmp_path, monkeypatch):
    import sprite_studio

    manager, _, _ = _manager(tmp_path)
    regeneration = _FakeRegeneration(manager)
    hard = HardExampleQueue(manager, tmp_path / "hard_examples.jsonl")
    service = InteractiveQualityControlService(
        manager,
        regeneration_manager=regeneration,
        hard_examples=hard,
        batch_path=tmp_path / "batch_jobs.json",
    )
    monkeypatch.setattr(sprite_studio, "FRAME_REVIEW_MANAGER", manager)
    monkeypatch.setattr(sprite_studio, "HARD_EXAMPLE_QUEUE", hard)
    monkeypatch.setattr(sprite_studio, "QC_SERVICE", service)
    server = sprite_studio.ThreadingHTTPServer(("127.0.0.1", 0), sprite_studio.SpriteStudioHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, evaluated = _json_request(
            base + "/api/qc/frame/evaluate",
            method="POST",
            payload={"run_id": "run_001", "frame_idx": 3},
        )
        assert status == 200
        review_id = evaluated["review"]["review_id"]

        status, regeneration_job = _json_request(
            base + f"/api/qc/frame/{review_id}/regenerate",
            method="POST",
            payload={"num_candidates": 2},
        )
        assert status == 202
        job_id = regeneration_job["job"]["job_id"]
        deadline = time.time() + 2
        while time.time() < deadline:
            _, job = _json_request(base + f"/api/qc/batch/{job_id}")
            if job["status"] not in {"PENDING", "RUNNING"}:
                break
            time.sleep(0.02)
        assert job["status"] == "COMPLETED"

        status, hard_result = _json_request(
            base + f"/api/qc/frame/{review_id}/reinforce",
            method="POST",
            payload={"user": "qa"},
        )
        assert status == 200
        assert hard_result["hard_example"]["target_verified"] is True

        status, confirmation = _json_request(
            base + "/api/qc/batch/start",
            method="POST",
            payload={"action": "REINFORCE_SELECTED", "review_ids": [review_id]},
        )
        assert status == 409
        assert confirmation["error"] == "confirmation_required"

        with socket.create_connection(server.server_address, timeout=2) as connection:
            connection.sendall(b"BAD\r\n\r\n")
            connection.recv(1024)
        status, queue = _json_request(base + "/api/qc/review-queue")
        assert status == 200
        assert queue["capabilities"]["batch_actions"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
