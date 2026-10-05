"""Persistent, safe per-frame quality review state.

Generated frames are evidence only.  This module never writes to dataset roots and
never uses generated images as training targets.
"""

from __future__ import annotations

from enum import Enum
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, Iterable, Mapping, Optional
from uuid import uuid4
from datetime import datetime, timezone

from .config import OUTPUT_DIR, PROJECT_ROOT
from .frame_map import get_frame_semantic_info


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


ENABLE_FRAME_REVIEW = _env_flag("PIXEL_AI_ENABLE_FRAME_REVIEW", True)
ENABLE_FRAME_REGENERATION = _env_flag("PIXEL_AI_ENABLE_FRAME_REGENERATION", False)
ENABLE_MANUAL_REINFORCEMENT = _env_flag("PIXEL_AI_ENABLE_MANUAL_REINFORCEMENT", False)
ENABLE_BATCH_QC_ACTIONS = _env_flag("PIXEL_AI_ENABLE_BATCH_QC_ACTIONS", False)

FRAME_REVIEW_QUEUE_FILE = PROJECT_ROOT / "frame_review_queue.jsonl"
DATASET_FRAMES_ROOT = PROJECT_ROOT / "dataset_frames_individuales"
SUPERVISED_DATASET_ROOT = PROJECT_ROOT / "dataset_supervisado"


class FrameReviewStatus(str, Enum):
    OK = "OK"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REEVALUATED = "REEVALUATED"
    REGENERATED = "REGENERATED"
    SENT_TO_REINFORCEMENT = "SENT_TO_REINFORCEMENT"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"
    ERROR = "ERROR"


VALID_REVIEW_STATUSES = tuple(item.value for item in FrameReviewStatus)
TERMINAL_REVIEW_STATUSES = {FrameReviewStatus.APPROVED.value, FrameReviewStatus.REJECTED.value}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. -]{0,159}$")
_VARIANT_ALIASES = {
    "ropa_blanca_chef": "rbchef",
    "ropa_negra_chef": "rnchef",
    "ropa_normal": "rnormal",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    text = re.sub(r"\s+", "_", text)
    return re.sub(r"[^a-z0-9_]+", "", text).strip("_")


def _variant_code(value: Any) -> str:
    normalized = _slug(value)
    return _VARIANT_ALIASES.get(normalized, normalized)


def _finite_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(min(100.0, max(0.0, number)), 4) if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class FrameQualityReviewManager:
    """Materialize review records in JSONL with atomic writes and trace history."""

    def __init__(
        self,
        queue_path: Optional[Path] = None,
        *,
        base_dir: Optional[Path] = None,
        project_root: Optional[Path] = None,
        output_root: Optional[Path] = None,
        dataset_frames_root: Optional[Path] = None,
        supervised_root: Optional[Path] = None,
        evaluator: Any = None,
    ):
        root = Path(project_root or base_dir or PROJECT_ROOT).resolve()
        self.project_root = root
        self.base_dir = root  # Compatibility with the first review-queue API.
        self.queue_path = Path(queue_path or (root / "frame_review_queue.jsonl")).resolve()
        self.output_root = Path(output_root or (root / "output")).resolve()
        self.dataset_frames_root = Path(dataset_frames_root or (root / "dataset_frames_individuales")).resolve()
        self.supervised_root = Path(supervised_root or (root / "dataset_supervisado")).resolve()
        self._evaluator = evaluator
        self._lock = threading.RLock()

    @staticmethod
    def validate_status(status: Any) -> str:
        value = status.value if isinstance(status, FrameReviewStatus) else str(status)
        if value not in {item.value for item in FrameReviewStatus}:
            raise ValueError(f"Unsupported frame review status: {value}")
        return value

    @staticmethod
    def _validate_identifier(value: Any, label: str) -> str:
        text = str(value or "").strip()
        if not text or not _SAFE_ID.fullmatch(text) or ".." in text or "/" in text or "\\" in text:
            raise ValueError(f"Invalid {label}")
        return text

    @staticmethod
    def _ensure_within(path: Path, root: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("Resolved path escapes its allowed root") from error
        return resolved

    def _relative(self, path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.project_root).as_posix()
        except ValueError:
            return resolved.as_posix()

    def _ensure_safe_path(self, candidate: Optional[str | Path]) -> Optional[str]:
        """Return an absolute in-project path for legacy callers, otherwise None."""
        if candidate is None:
            return None
        try:
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            return str(self._ensure_within(path, self.project_root))
        except (TypeError, ValueError, OSError):
            return None

    def _read_records(self) -> Dict[str, Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        if not self.queue_path.exists():
            return records
        with self.queue_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid review JSONL at line {line_number}") from error
                if isinstance(record, dict) and record.get("review_id"):
                    self._validate_identifier(record["review_id"], "review_id")
                    record["status"] = self.validate_status(record.get("status"))
                    records[str(record["review_id"])] = record
        return records

    def _write_records(self, records: Iterable[Mapping[str, Any]]) -> None:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.queue_path.with_suffix(self.queue_path.suffix + ".tmp")
        ordered = sorted(records, key=lambda item: (str(item.get("created_at", "")), str(item.get("review_id", ""))))
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in ordered:
                handle.write(json.dumps(_json_safe(record), ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.queue_path)

    def _mutate(self, review_id: str, callback: Any) -> Dict[str, Any]:
        with self._lock:
            records = self._read_records()
            if review_id not in records:
                raise KeyError(review_id)
            updated = callback(dict(records[review_id]))
            updated["status"] = self.validate_status(updated["status"])
            updated["updated_at"] = _now()
            records[review_id] = updated
            self._write_records(records.values())
            return dict(updated)

    def get_review(self, review_id: str) -> Optional[Dict[str, Any]]:
        review_id = self._validate_identifier(review_id, "review_id")
        with self._lock:
            record = self._read_records().get(review_id)
            return dict(record) if record else None

    def get_record(self, review_id: str) -> Optional[Dict[str, Any]]:
        return self.get_review(review_id)

    def list_reviews(
        self,
        *,
        status: Optional[str] = None,
        run_id: Optional[str] = None,
        include_terminal: bool = True,
    ) -> list[Dict[str, Any]]:
        if status is not None:
            status = self.validate_status(status)
        if run_id is not None:
            run_id = self._validate_identifier(run_id, "run_id")
        with self._lock:
            records = list(self._read_records().values())
        if status is not None:
            records = [record for record in records if record.get("status") == status]
        if run_id is not None:
            records = [record for record in records if record.get("run_id") == run_id]
        if not include_terminal:
            records = [record for record in records if record.get("status") not in TERMINAL_REVIEW_STATUSES]
        return sorted(records, key=lambda item: str(item.get("updated_at", "")), reverse=True)

    def get_queue(self, *, status: Optional[str] = None) -> list[Dict[str, Any]]:
        return self.list_reviews(status=status)

    def list_queue(self, *, status: Optional[str] = None) -> list[Dict[str, Any]]:
        return self.list_reviews(status=status)

    def summary(self, *, run_id: Optional[str] = None) -> Dict[str, Any]:
        records = self.list_reviews(run_id=run_id)
        by_status = {status.value: 0 for status in FrameReviewStatus}
        for record in records:
            by_status[str(record.get("status"))] = by_status.get(str(record.get("status")), 0) + 1
        return {
            "total": len(records),
            "pending": sum(by_status.get(status, 0) for status in ("OK", "NEEDS_REVIEW", "REEVALUATED", "ERROR")),
            "approved": by_status["APPROVED"],
            "rejected": by_status["REJECTED"],
            "regenerated": by_status["REGENERATED"],
            "sent_to_reinforcement": by_status["SENT_TO_REINFORCEMENT"],
            "by_status": by_status,
        }

    def _load_run_metadata(self, run_dir: Path) -> Dict[str, Any]:
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            return {}
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def resolve_run_frame(self, run_id: str, frame_idx: int) -> Dict[str, Any]:
        run_id = self._validate_identifier(run_id, "run_id")
        frame_idx = int(frame_idx)
        if frame_idx < 0 or frame_idx > 4095:
            raise ValueError("Invalid frame_idx")
        run_dir = self._ensure_within(self.output_root / run_id, self.output_root)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run not found: {run_id}")
        metadata = self._load_run_metadata(run_dir)
        generated = run_dir / "enhanced_frames" / f"frame_{frame_idx:03d}.png"
        if not generated.is_file():
            generated = run_dir / "raw_frames" / f"frame_{frame_idx:03d}.png"
        if not generated.is_file():
            raise FileNotFoundError(f"Generated frame not found: {frame_idx}")
        return {"run_dir": run_dir, "generated": generated.resolve(), "metadata": metadata}

    def _identity_from_metadata(self, metadata: Mapping[str, Any]) -> tuple[str, str]:
        character = str(metadata.get("character_id") or metadata.get("character") or "").strip()
        variant = str(metadata.get("variant") or "").strip()
        front = Path(str(metadata.get("front_image") or ""))
        stem = _slug(front.stem)
        if not variant:
            for alias, code in _VARIANT_ALIASES.items():
                if stem.endswith("_" + code) or stem.endswith("_" + alias):
                    variant = code
                    break
        if not character and stem:
            suffixes = tuple("_" + item for item in set(_VARIANT_ALIASES.values()) | set(_VARIANT_ALIASES.keys()))
            character = stem
            for suffix in suffixes:
                if character.endswith(suffix):
                    character = character[: -len(suffix)]
                    break
        return character, _variant_code(variant)

    def resolve_dataset_assets(self, metadata: Mapping[str, Any], frame_idx: int) -> Dict[str, Any]:
        character, variant = self._identity_from_metadata(metadata)
        character_slug = _slug(character)
        if not character_slug or not variant:
            return {"character_id": character or None, "variant": variant or None, "target": None, "reference": None}

        matching_manifest = None
        if self.dataset_frames_root.is_dir():
            for manifest_path in self.dataset_frames_root.rglob("manifest.json"):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if _slug(manifest.get("character")) == character_slug and _variant_code(manifest.get("variant")) == variant:
                    matching_manifest = (manifest_path, manifest)
                    break

        target = None
        reference = None
        resolved_character = character
        if matching_manifest is not None:
            manifest_path, manifest = matching_manifest
            resolved_character = str(manifest.get("character") or character)
            frame_entry = next(
                (
                    entry for entry in manifest.get("frames", [])
                    if isinstance(entry, Mapping) and int(entry.get("slot", -1)) == int(frame_idx) + 1
                ),
                None,
            )
            if frame_entry:
                candidate = self._ensure_within(manifest_path.parent / str(frame_entry.get("file", "")), self.dataset_frames_root)
                if candidate.is_file():
                    target = candidate
            identity_candidate = manifest_path.parent / "00_frontal_identidad.png"
            if identity_candidate.is_file():
                reference = self._ensure_within(identity_candidate, self.dataset_frames_root)

        if target is None:
            candidate = self.supervised_root / "frames_png" / f"{character_slug}_{variant}_frame_{int(frame_idx):03d}.png"
            candidate = self._ensure_within(candidate, self.supervised_root)
            if candidate.is_file():
                target = candidate
        if reference is None:
            candidate = self.supervised_root / "reference_fronts" / f"{character_slug}_{variant}_chibi_front.png"
            candidate = self._ensure_within(candidate, self.supervised_root)
            if candidate.is_file():
                reference = candidate

        return {
            "character_id": resolved_character,
            "variant": variant,
            "target": target,
            "reference": reference,
        }

    def _evaluate(self, generated: Path, target: Optional[Path], reference: Optional[Path], frame_idx: int, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        evaluator = self._evaluator
        if evaluator is None:
            from .quality_gate import QualityGate
            evaluator = QualityGate.evaluate_single_frame
        return evaluator(
            generated,
            target,
            reference_front=reference,
            frame_idx=frame_idx,
            metadata=dict(metadata),
        )

    @staticmethod
    def _history_event(action: str, record: Mapping[str, Any], **details: Any) -> Dict[str, Any]:
        return {
            "action": action,
            "time": _now(),
            "run_id": record.get("run_id"),
            "character_id": record.get("character_id"),
            "variant": record.get("variant"),
            "frame_idx": record.get("frame_idx"),
            "status": record.get("status"),
            "generated_frame_path": record.get("generated_frame_path"),
            "target_frame_path": record.get("target_frame_path"),
            **_json_safe(details),
        }

    def evaluate_frame(self, run_id: str, frame_idx: int, *, epoch: Optional[int] = None, actor: Optional[str] = None) -> Dict[str, Any]:
        if not ENABLE_FRAME_REVIEW:
            raise RuntimeError("Frame review is disabled")
        run = self.resolve_run_frame(run_id, frame_idx)
        assets = self.resolve_dataset_assets(run["metadata"], frame_idx)
        audit = self._evaluate(run["generated"], assets["target"], assets["reference"], frame_idx, run["metadata"])
        format_type = str(run["metadata"].get("format") or "8x12")
        columns = 8 if format_type == "8x12" else 16
        pose = get_frame_semantic_info(int(frame_idx), format_type)
        quality = dict(audit.get("quality") or {})
        global_score = _finite_score(audit.get("score_total", quality.get("global")))
        status = FrameReviewStatus.OK.value if audit.get("aprobado") else FrameReviewStatus.NEEDS_REVIEW.value
        if not audit.get("audit_available"):
            status = FrameReviewStatus.ERROR.value
        timestamp = _now()

        with self._lock:
            records = self._read_records()
            existing = next(
                (
                    record for record in records.values()
                    if record.get("run_id") == run_id and int(record.get("frame_idx", -1)) == int(frame_idx)
                ),
                None,
            )
            review_id = str(existing.get("review_id")) if existing else uuid4().hex
            created_at = str(existing.get("created_at")) if existing else timestamp
            history = list(existing.get("history", [])) if existing else []
            record = {
                "review_id": review_id,
                "run_id": run_id,
                "character_id": assets["character_id"],
                "variant": assets["variant"],
                "frame_idx": int(frame_idx),
                "row": int(frame_idx) // columns,
                "column": int(frame_idx) % columns,
                "pose": pose,
                "epoch": int(epoch) if epoch is not None else None,
                "status": status,
                "generated_frame_path": self._relative(run["generated"]),
                "target_frame_path": self._relative(assets["target"]),
                "reference_front_path": self._relative(assets["reference"]),
                "quality": quality,
                "score_total": global_score,
                "issues": list(audit.get("issues") or []),
                "severity": str(audit.get("severity") or "low"),
                "diagnosis": dict(audit.get("diagnosis") or {}),
                "audit_available": bool(audit.get("audit_available")),
                "identity_confidence": audit.get("identity_confidence"),
                "user_action": None,
                "times_failed": int(existing.get("times_failed", 0)) if existing else 0,
                "times_regenerated": int(existing.get("times_regenerated", 0)) if existing else 0,
                "times_reevaluated": int(existing.get("times_reevaluated", 0)) if existing else 0,
                "sent_to_reinforcement": bool(existing.get("sent_to_reinforcement", False)) if existing else False,
                "created_at": created_at,
                "updated_at": timestamp,
                "history": history,
            }
            record["history"].append(self._history_event(
                "evaluated",
                record,
                actor=actor,
                score=global_score,
                issues=record["issues"],
                audit_available=record["audit_available"],
            ))
            records[review_id] = record
            self._write_records(records.values())
            return dict(record)

    def register_frame(
        self,
        *,
        character_id: str,
        variant: str,
        frame_idx: int,
        generated_frame_path: str | Path,
        target_frame_path: Optional[str | Path] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register an explicit frame pair without evaluating or changing either file."""
        metadata = dict(metadata or {})
        generated = self._ensure_safe_path(generated_frame_path)
        target = self._ensure_safe_path(target_frame_path)
        if generated is None:
            raise ValueError("generated_frame_path escapes the project root")
        if target_frame_path is not None and target is None:
            raise ValueError("target_frame_path escapes the project root")
        now = _now()
        record = {
            "review_id": uuid4().hex,
            "run_id": str(metadata.get("run_id") or f"manual_{uuid4().hex[:12]}"),
            "character_id": str(character_id),
            "variant": str(variant),
            "frame_idx": int(frame_idx),
            "row": metadata.get("row"),
            "column": metadata.get("column"),
            "pose": metadata.get("pose"),
            "epoch": metadata.get("epoch"),
            "status": FrameReviewStatus.NEEDS_REVIEW.value,
            "generated_frame_path": generated,
            "target_frame_path": target,
            "reference_front_path": self._ensure_safe_path(metadata.get("reference_front_path")),
            "quality": {},
            "score_total": None,
            "issues": [],
            "severity": "unknown",
            "diagnosis": {},
            "audit_available": bool(target and Path(target).is_file()),
            "identity_confidence": None,
            "user_action": None,
            "times_failed": 0,
            "times_regenerated": 0,
            "times_reevaluated": 0,
            "sent_to_reinforcement": False,
            "created_at": now,
            "updated_at": now,
            "history": [],
        }
        record["history"].append(self._history_event("registered", record))
        with self._lock:
            records = self._read_records()
            records[record["review_id"]] = record
            self._write_records(records.values())
        return dict(record)

    def reevaluate_frame(
        self,
        review_id: str,
        *,
        actor: Optional[str] = None,
        quality: Optional[Mapping[str, Any]] = None,
        issues: Optional[list[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        review = self.get_review(review_id)
        if review is None:
            raise KeyError(review_id)
        if quality is not None:
            # Kept for compatibility with the original UI. New endpoints always
            # invoke the evaluator so client-provided metrics are not trusted.
            clean_quality = {str(key): _finite_score(value) for key, value in quality.items()}
            audit = {
                "audit_available": bool(review.get("target_frame_path")),
                "quality": clean_quality,
                "score_total": clean_quality.get("global"),
                "issues": list(issues or []),
                "severity": str((metadata or {}).get("severity") or "unknown"),
                "diagnosis": dict(metadata or {}),
            }
        else:
            run_id = str(review.get("run_id") or "")
            try:
                run = self.resolve_run_frame(run_id, int(review["frame_idx"]))
                generated = run["generated"]
                run_metadata = run["metadata"]
            except (FileNotFoundError, ValueError):
                generated_raw = review.get("generated_frame_path")
                generated = Path(str(generated_raw))
                if not generated.is_absolute():
                    generated = self.project_root / generated
                generated = self._ensure_within(generated, self.project_root)
                run_metadata = dict(metadata or {})
            target_raw = review.get("target_frame_path")
            reference_raw = review.get("reference_front_path")
            target = Path(str(target_raw)) if target_raw else None
            reference = Path(str(reference_raw)) if reference_raw else None
            if target is not None and not target.is_absolute():
                target = self.project_root / target
            if reference is not None and not reference.is_absolute():
                reference = self.project_root / reference
            audit = self._evaluate(generated, target, reference, int(review["frame_idx"]), run_metadata)

        def update(record: Dict[str, Any]) -> Dict[str, Any]:
            previous_quality = dict(record.get("quality") or {})
            quality = dict(audit.get("quality") or {})
            score = _finite_score(audit.get("score_total", quality.get("global")))
            record.update({
                "status": FrameReviewStatus.REEVALUATED.value if audit.get("audit_available") else FrameReviewStatus.ERROR.value,
                "quality": quality,
                "score_total": score,
                "issues": list(audit.get("issues") or []),
                "severity": str(audit.get("severity") or "low"),
                "diagnosis": dict(audit.get("diagnosis") or {}),
                "audit_available": bool(audit.get("audit_available")),
                "identity_confidence": audit.get("identity_confidence"),
                "times_reevaluated": int(record.get("times_reevaluated", 0)) + 1,
                "user_action": "REEVALUATE",
            })
            history = list(record.get("history", []))
            history.append(self._history_event(
                "reevaluated",
                record,
                actor=actor,
                previous_quality=previous_quality,
                new_quality=quality,
                score=score,
            ))
            record["history"] = history
            return record

        return self._mutate(review_id, update)

    def approve_frame(
        self,
        review_id: str,
        *,
        actor: Optional[str] = None,
        reason: Optional[str] = None,
        user: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        actor = actor or user
        reason = reason or comment
        def update(record: Dict[str, Any]) -> Dict[str, Any]:
            if not record.get("audit_available") or not record.get("target_frame_path"):
                raise ValueError("approval_blocked_missing_target")
            if "ALPHA_FAIL" in set(record.get("issues") or []):
                raise ValueError("approval_blocked_alpha_fail")
            record["status"] = FrameReviewStatus.APPROVED.value
            record["user_action"] = "APPROVED"
            record["approved_by"] = actor
            record["approved_at"] = _now()
            history = list(record.get("history", []))
            history.append(self._history_event("approved", record, actor=actor, reason=reason, score=record.get("score_total")))
            record["history"] = history
            return record

        return self._mutate(self._validate_identifier(review_id, "review_id"), update)

    def reject_frame(
        self,
        review_id: str,
        *,
        actor: Optional[str] = None,
        reason: Optional[str] = None,
        user: Optional[str] = None,
    ) -> Dict[str, Any]:
        actor = actor or user
        def update(record: Dict[str, Any]) -> Dict[str, Any]:
            record["status"] = FrameReviewStatus.REJECTED.value
            record["user_action"] = "REJECTED"
            record["rejected_by"] = actor
            record["rejected_at"] = _now()
            record["times_failed"] = int(record.get("times_failed", 0)) + 1
            history = list(record.get("history", []))
            history.append(self._history_event("rejected", record, actor=actor, reason=reason, score=record.get("score_total")))
            record["history"] = history
            return record

        return self._mutate(self._validate_identifier(review_id, "review_id"), update)


__all__ = [
    "ENABLE_BATCH_QC_ACTIONS",
    "ENABLE_FRAME_REGENERATION",
    "ENABLE_FRAME_REVIEW",
    "ENABLE_MANUAL_REINFORCEMENT",
    "FRAME_REVIEW_QUEUE_FILE",
    "FrameQualityReviewManager",
    "FrameReviewStatus",
    "VALID_REVIEW_STATUSES",
]
