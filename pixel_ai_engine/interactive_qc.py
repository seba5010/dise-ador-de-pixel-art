"""Application service for interactive QC, feedback and cancellable batch jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, Iterable, Mapping, Optional
from uuid import uuid4

from .frame_quality_review import (
    ENABLE_BATCH_QC_ACTIONS,
    ENABLE_FRAME_REGENERATION,
    ENABLE_MANUAL_REINFORCEMENT,
    FrameQualityReviewManager,
)
from .frame_regeneration import FrameRegenerationManager
from .hard_examples import HardExampleQueue


BATCH_ACTIONS = {"REEVALUATE_ALL", "REGENERATE_DEFECTIVE", "REGENERATE_SELECTED", "REINFORCE_SELECTED"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class InteractiveQualityControlService:
    def __init__(
        self,
        review_manager: FrameQualityReviewManager,
        *,
        regeneration_manager: Optional[FrameRegenerationManager] = None,
        hard_examples: Optional[HardExampleQueue] = None,
        batch_path: Optional[str | Path] = None,
    ):
        self.reviews = review_manager
        self.regeneration = regeneration_manager or FrameRegenerationManager(review_manager)
        self.hard_examples = hard_examples or HardExampleQueue(review_manager)
        self.batch_path = Path(batch_path or (review_manager.project_root / "qc_batch_jobs.json")).resolve()
        self._lock = threading.RLock()
        self._cancel: Dict[str, threading.Event] = {}
        self._recover_interrupted_jobs()

    def reevaluate(self, review_id: str, *, actor: Optional[str] = None) -> Dict[str, Any]:
        review = self.reviews.reevaluate_frame(review_id, actor=actor)
        self.hard_examples.record_outcome(review)
        return review

    def regenerate(self, review_id: str, *, num_candidates: int = 3, actor: Optional[str] = None) -> Dict[str, Any]:
        if not ENABLE_FRAME_REGENERATION:
            raise RuntimeError("frame_regeneration_disabled")
        return self.regeneration.regenerate(review_id, num_candidates=num_candidates, actor=actor)

    def apply_best(self, review_id: str, *, candidate_id: Optional[str] = None, actor: Optional[str] = None) -> Dict[str, Any]:
        review = self.regeneration.apply_best(review_id, candidate_id=candidate_id, actor=actor)
        self.hard_examples.record_outcome(review)
        return review

    def discard_regeneration(self, review_id: str, *, actor: Optional[str] = None) -> Dict[str, Any]:
        return self.regeneration.discard(review_id, actor=actor)

    def reinforce(self, review_id: str, *, actor: Optional[str] = None) -> Dict[str, Any]:
        if not ENABLE_MANUAL_REINFORCEMENT:
            raise RuntimeError("manual_reinforcement_disabled")
        return self.hard_examples.enqueue(review_id, actor=actor)

    def summary(self, *, run_id: Optional[str] = None) -> Dict[str, Any]:
        summary = self.reviews.summary(run_id=run_id)
        summary["hard_examples"] = self.hard_examples.summary()
        summary["metrics"] = self.metrics(run_id=run_id)
        return summary

    def metrics(self, *, run_id: Optional[str] = None) -> Dict[str, Any]:
        records = self.reviews.list_reviews(run_id=run_id)
        regeneration_deltas = []
        approval_seconds = []
        for record in records:
            regeneration = record.get("regeneration") if isinstance(record.get("regeneration"), Mapping) else {}
            ranking = regeneration.get("ranking") if isinstance(regeneration.get("ranking"), Mapping) else {}
            best_id = ranking.get("best_candidate")
            best = next((item for item in ranking.get("candidates", []) if item.get("candidate_id") == best_id), None)
            if isinstance(best, Mapping) and best.get("global_delta") is not None:
                regeneration_deltas.append(float(best["global_delta"]))
            if record.get("approved_at") and record.get("created_at"):
                try:
                    approved = datetime.fromisoformat(str(record["approved_at"]).replace("Z", "+00:00"))
                    created = datetime.fromisoformat(str(record["created_at"]).replace("Z", "+00:00"))
                    approval_seconds.append(max(0.0, (approved - created).total_seconds()))
                except ValueError:
                    pass
        improved = sum(delta > 0 for delta in regeneration_deltas)
        return {
            "frames_rejected": sum(record.get("status") == "REJECTED" for record in records),
            "frames_regenerated": sum(int(record.get("times_regenerated", 0)) > 0 for record in records),
            "regenerations_measured": len(regeneration_deltas),
            "regenerations_improved": improved,
            "regeneration_improvement_pct": round(improved / len(regeneration_deltas) * 100.0, 2) if regeneration_deltas else None,
            "frames_sent_to_reinforcement": sum(bool(record.get("sent_to_reinforcement")) for record in records),
            "recurrent_failures": sum(int(record.get("times_failed", 0)) >= 2 for record in records),
            "mean_seconds_to_approval": round(sum(approval_seconds) / len(approval_seconds), 2) if approval_seconds else None,
        }

    def _read_jobs(self) -> Dict[str, Dict[str, Any]]:
        if not self.batch_path.is_file():
            return {}
        try:
            data = json.loads(self.batch_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(key): dict(value) for key, value in data.items() if isinstance(value, Mapping)} if isinstance(data, Mapping) else {}

    def _write_jobs(self, jobs: Mapping[str, Mapping[str, Any]]) -> None:
        self.batch_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.batch_path.with_suffix(self.batch_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(jobs, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.batch_path)

    def _recover_interrupted_jobs(self) -> None:
        with self._lock:
            jobs = self._read_jobs()
            changed = False
            for job in jobs.values():
                if job.get("status") in {"RUNNING", "CANCELLING"}:
                    job["status"] = "INTERRUPTED"
                    job["finished_at"] = _now()
                    changed = True
            if changed:
                self._write_jobs(jobs)

    def get_batch(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._read_jobs().get(str(job_id))
            return dict(job) if job else None

    def list_batches(self) -> list[Dict[str, Any]]:
        with self._lock:
            return sorted(self._read_jobs().values(), key=lambda job: str(job.get("created_at", "")), reverse=True)

    def _eligible_ids(self, action: str, review_ids: Optional[Iterable[str]]) -> list[str]:
        explicit = [str(item) for item in (review_ids or [])]
        if action == "REINFORCE_SELECTED":
            if not explicit:
                raise ValueError("review_ids_required")
            candidates = [self.reviews.get_review(review_id) for review_id in explicit]
        else:
            candidates = [self.reviews.get_review(review_id) for review_id in explicit] if explicit else self.reviews.list_reviews()
        eligible = []
        for record in candidates:
            if not record:
                continue
            if action == "REEVALUATE_ALL":
                if record.get("status") not in {"APPROVED", "SUPERSEDED"}:
                    eligible.append(str(record["review_id"]))
            elif action == "REGENERATE_SELECTED":
                if record.get("target_frame_path") and record.get("status") != "APPROVED":
                    eligible.append(str(record["review_id"]))
            elif action == "REGENERATE_DEFECTIVE":
                score = record.get("score_total")
                low_score = score is None or float(score) < 80.0
                severe = str(record.get("severity", "")).lower() in {"high", "critical"}
                if record.get("target_frame_path") and record.get("status") != "APPROVED" and (low_score or severe):
                    eligible.append(str(record["review_id"]))
            else:
                if record.get("target_frame_path") and record.get("status") != "APPROVED":
                    eligible.append(str(record["review_id"]))
        return list(dict.fromkeys(eligible))

    def start_batch(
        self,
        action: str,
        *,
        review_ids: Optional[Iterable[str]] = None,
        actor: Optional[str] = None,
        num_candidates: int = 3,
        background: bool = True,
    ) -> Dict[str, Any]:
        if not ENABLE_BATCH_QC_ACTIONS and str(action).upper() != "REGENERATE_SELECTED":
            raise RuntimeError("batch_qc_actions_disabled")
        action = str(action).upper()
        if action not in BATCH_ACTIONS:
            raise ValueError("unsupported_batch_action")
        ids = self._eligible_ids(action, review_ids)
        job_id = uuid4().hex
        now = _now()
        job = {
            "job_id": job_id,
            "action": action,
            "status": "PENDING",
            "review_ids": ids,
            "total": len(ids),
            "completed": 0,
            "failed": 0,
            "progress": 0.0,
            "errors": [],
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "actor": actor,
        }
        with self._lock:
            jobs = self._read_jobs()
            jobs[job_id] = job
            self._write_jobs(jobs)
            self._cancel[job_id] = threading.Event()
        if background:
            threading.Thread(
                target=self._run_batch,
                args=(job_id, num_candidates),
                daemon=True,
                name=f"qc-batch-{job_id[:8]}",
            ).start()
        else:
            self._run_batch(job_id, num_candidates)
        return dict(self.get_batch(job_id) or job)

    def _update_job(self, job_id: str, **updates: Any) -> Dict[str, Any]:
        with self._lock:
            jobs = self._read_jobs()
            if job_id not in jobs:
                raise KeyError(job_id)
            jobs[job_id].update(updates)
            self._write_jobs(jobs)
            return dict(jobs[job_id])

    def _run_batch(self, job_id: str, num_candidates: int) -> None:
        job = self._update_job(job_id, status="RUNNING", started_at=_now())
        cancel = self._cancel.setdefault(job_id, threading.Event())
        total = int(job.get("total", 0))
        completed = failed = 0
        errors = []
        for review_id in job.get("review_ids", []):
            if cancel.is_set():
                self._update_job(job_id, status="CANCELLED", finished_at=_now())
                return
            try:
                if job["action"] == "REEVALUATE_ALL":
                    self.reevaluate(review_id, actor=job.get("actor"))
                elif job["action"] in {"REGENERATE_DEFECTIVE", "REGENERATE_SELECTED"}:
                    self.regenerate(review_id, num_candidates=num_candidates, actor=job.get("actor"))
                else:
                    self.reinforce(review_id, actor=job.get("actor"))
                completed += 1
            except Exception as error:
                failed += 1
                errors.append({"review_id": review_id, "error": str(error)})
            self._update_job(
                job_id,
                completed=completed,
                failed=failed,
                errors=errors[-100:],
                progress=round((completed + failed) / max(1, total) * 100.0, 2),
            )
        if cancel.is_set():
            self._update_job(job_id, status="CANCELLED", finished_at=_now())
            return
        self._update_job(
            job_id,
            status="COMPLETED" if failed == 0 else "COMPLETED_WITH_ERRORS",
            progress=100.0,
            finished_at=_now(),
        )

    def cancel_batch(self, job_id: str) -> Dict[str, Any]:
        job = self.get_batch(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.get("status") not in {"PENDING", "RUNNING", "CANCELLING"}:
            return job
        self._cancel.setdefault(job_id, threading.Event()).set()
        return self._update_job(job_id, status="CANCELLING")


__all__ = ["BATCH_ACTIONS", "InteractiveQualityControlService"]
