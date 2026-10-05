"""Persistent manual hard-example feedback backed by verified dataset targets."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, Iterable, Mapping, Optional
from uuid import uuid4

from .config import PROJECT_ROOT
from .frame_quality_review import FrameQualityReviewManager
from .quality_guidance import frame_quality_key


HARD_EXAMPLES_FILE = PROJECT_ROOT / "hard_examples.jsonl"
MAX_HARD_EXAMPLE_PRIORITY = 2.0
SEVERITY_PRIORITY = {"low": 1.2, "medium": 1.4, "high": 1.7, "critical": 2.0}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _slug(value: Any) -> str:
    text = re.sub(r"\s+", "_", str(value or "").strip().casefold().replace("-", "_"))
    return re.sub(r"[^a-z0-9_]+", "", text).strip("_")


def _bounded_priority(severity: str, failures: int) -> float:
    repeat_priority = 1.3 if failures <= 1 else 1.5 if failures == 2 else 1.7 if failures == 3 else 2.0
    return round(min(MAX_HARD_EXAMPLE_PRIORITY, max(SEVERITY_PRIORITY.get(str(severity).lower(), 1.4), repeat_priority)), 2)


class HardExampleQueue:
    """Materialized JSONL queue; generated frames are evidence, never targets."""

    def __init__(
        self,
        review_manager: FrameQualityReviewManager,
        path: Optional[str | Path] = None,
    ):
        self.review_manager = review_manager
        self.path = Path(path or (review_manager.project_root / "hard_examples.jsonl")).resolve()
        self._lock = threading.RLock()

    @staticmethod
    def key(character_id: Any, variant: Any, frame_idx: Any) -> str:
        return f"{_slug(character_id)}::{_slug(variant)}::frame_{int(frame_idx):03d}"

    def _read(self) -> Dict[str, Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        if not self.path.is_file():
            return records
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid hard-example JSONL at line {line_number}") from error
                if isinstance(item, dict) and item.get("key"):
                    records[str(item["key"])] = item
        return records

    def _write(self, records: Iterable[Mapping[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in sorted(records, key=lambda item: str(item.get("key", ""))):
                handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def list(self, *, active_only: bool = False) -> list[Dict[str, Any]]:
        with self._lock:
            records = list(self._read().values())
        if active_only:
            records = [record for record in records if record.get("active", True)]
        return sorted(records, key=lambda item: str(item.get("last_failed_at", item.get("added_at", ""))), reverse=True)

    def summary(self) -> Dict[str, Any]:
        records = self.list()
        active = [record for record in records if record.get("active", True)]
        return {
            "total": len(records),
            "active": len(active),
            "resolved": len(records) - len(active),
            "recurrent": sum(int(record.get("times_failed", 0)) >= 2 for record in active),
            "max_priority": max((float(record.get("priority", 1.0)) for record in active), default=1.0),
        }

    def _validated_target(self, review: Mapping[str, Any]) -> Path:
        run = self.review_manager.resolve_run_frame(str(review["run_id"]), int(review["frame_idx"]))
        assets = self.review_manager.resolve_dataset_assets(run["metadata"], int(review["frame_idx"]))
        expected = assets.get("target")
        if expected is None or not Path(expected).is_file():
            raise ValueError("target_not_found")
        recorded_raw = review.get("target_frame_path")
        if not recorded_raw:
            raise ValueError("target_not_found")
        recorded = Path(str(recorded_raw))
        if not recorded.is_absolute():
            recorded = self.review_manager.project_root / recorded
        if recorded.resolve() != Path(expected).resolve():
            raise ValueError("target_identity_mismatch")
        if _slug(review.get("character_id")) != _slug(assets.get("character_id")):
            raise ValueError("target_character_mismatch")
        if _slug(review.get("variant")) != _slug(assets.get("variant")):
            raise ValueError("target_variant_mismatch")
        return Path(expected).resolve()

    def enqueue(self, review_id: str, *, actor: Optional[str] = None, reason: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        review = self.review_manager.get_review(review_id)
        if review is None:
            raise KeyError(review_id)
        target = self._validated_target(review)
        key = self.key(review.get("character_id"), review.get("variant"), review.get("frame_idx"))
        now = _now()
        with self._lock:
            records = self._read()
            previous = records.get(key)
            failures = max(
                int(review.get("times_failed", 0) or 0),
                int(previous.get("times_failed", 0) or 0) + 1 if previous else 1,
            )
            priority = _bounded_priority(str(review.get("severity") or "medium"), failures)
            history = list(previous.get("history", [])) if previous else []
            history.append({
                "action": "manual_reinforcement_requested",
                "time": now,
                "actor": actor,
                "priority": priority,
                "score": review.get("score_total"),
            })
            record = {
                "hard_example_id": previous.get("hard_example_id") if previous else uuid4().hex,
                "key": key,
                "review_id": review_id,
                "run_id": review.get("run_id"),
                "character_id": review.get("character_id"),
                "variant": review.get("variant"),
                "frame_idx": int(review.get("frame_idx")),
                "reason": list(reason or review.get("issues") or []),
                "severity": str(review.get("severity") or "medium"),
                "times_failed": failures,
                "priority": priority,
                "last_score": review.get("score_total"),
                "generated_frame_path": review.get("generated_frame_path"),
                "target_frame_path": target.relative_to(self.review_manager.project_root).as_posix(),
                "target_verified": True,
                "active": True,
                "status": "ACTIVE",
                "added_at": previous.get("added_at") if previous else now,
                "last_failed_at": now,
                "history": history,
            }
            records[key] = record
            self._write(records.values())
        self.review_manager.mark_for_reinforcement(review_id, hard_example=record, actor=actor)
        return dict(record)

    def record_outcome(self, review: Mapping[str, Any], *, approval_threshold: float = 85.0) -> Optional[Dict[str, Any]]:
        key = self.key(review.get("character_id"), review.get("variant"), review.get("frame_idx"))
        try:
            score = float(review.get("score_total"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(score):
            return None
        with self._lock:
            records = self._read()
            if key not in records:
                return None
            record = dict(records[key])
            previous = float(record.get("priority", 1.0))
            improved = score > float(record.get("last_score", 0.0) or 0.0)
            priority = max(1.0, round(previous - 0.3, 2)) if improved else previous
            resolved = score >= float(approval_threshold) or review.get("status") == "APPROVED"
            if resolved:
                priority = 1.0
            record.update({
                "priority": priority,
                "last_score": round(score, 4),
                "last_evaluated_at": _now(),
                "active": not resolved,
                "status": "RESOLVED" if resolved else "ACTIVE",
            })
            history = list(record.get("history", []))
            history.append({
                "action": "quality_outcome",
                "time": _now(),
                "score": round(score, 4),
                "priority_before": previous,
                "priority_after": priority,
                "resolved": resolved,
            })
            record["history"] = history
            records[key] = record
            self._write(records.values())
            return record

    def manual_weights(self) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for record in self.list(active_only=True):
            char_id = f"{_slug(record.get('character_id'))}_{_slug(record.get('variant'))}".strip("_")
            key = frame_quality_key(char_id, record.get("frame_idx"))
            weights[key] = min(MAX_HARD_EXAMPLE_PRIORITY, max(1.0, float(record.get("priority", 1.0))))
        return weights


def load_manual_sampling_weights(path: Optional[str | Path] = None) -> Dict[str, float]:
    """Read verified active weights without requiring a review manager at train time."""
    target = Path(path or HARD_EXAMPLES_FILE)
    if not target.is_file():
        return {}
    weights: Dict[str, float] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or not record.get("active", True) or not record.get("target_verified"):
            continue
        char_id = f"{_slug(record.get('character_id'))}_{_slug(record.get('variant'))}".strip("_")
        key = frame_quality_key(char_id, record.get("frame_idx", -1))
        try:
            priority = float(record.get("priority", 1.0))
        except (TypeError, ValueError):
            continue
        weights[key] = min(MAX_HARD_EXAMPLE_PRIORITY, max(1.0, priority))
    return weights


__all__ = [
    "HARD_EXAMPLES_FILE",
    "HardExampleQueue",
    "MAX_HARD_EXAMPLE_PRIORITY",
    "load_manual_sampling_weights",
]
