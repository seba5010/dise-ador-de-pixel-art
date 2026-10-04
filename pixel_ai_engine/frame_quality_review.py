from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
from PIL import Image

from .config import PROJECT_ROOT

VALID_REVIEW_STATUSES = (
    "OK",
    "NEEDS_REVIEW",
    "REEVALUATED",
    "REGENERATED",
    "SENT_TO_REINFORCEMENT",
    "APPROVED",
    "REJECTED",
    "SUPERSEDED",
    "ERROR",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FrameQualityReviewManager:
    """Persisted review queue for per-frame human QC."""

    def __init__(
        self,
        base_dir: Optional[str | Path] = None,
        queue_path: Optional[str | Path] = None,
    ):
        self.base_dir = Path(base_dir or PROJECT_ROOT).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.queue_path = Path(queue_path).resolve() if queue_path else self.base_dir / "frame_review_queue.jsonl"
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_safe_path(self, candidate: Optional[str | Path]) -> Optional[str]:
        if candidate is None:
            return None
        try:
            resolved = Path(candidate).expanduser().resolve(strict=False)
        except (TypeError, ValueError):
            return None
        base = self.base_dir.resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            return None
        return str(resolved)

    def _load_queue(self) -> List[Dict[str, Any]]:
        if not self.queue_path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for line in self.queue_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
        return rows

    def _write_queue(self, rows: Iterable[Dict[str, Any]]) -> None:
        with self.queue_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                if isinstance(row, dict):
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                    handle.write("\n")

    def _read_record_by_id(self, review_id: str) -> Optional[Dict[str, Any]]:
        for row in self._load_queue():
            if row.get("review_id") == review_id:
                return row
        return None

    def _persist_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        rows = self._load_queue()
        existing_index = None
        for idx, row in enumerate(rows):
            if row.get("review_id") == record["review_id"]:
                existing_index = idx
                break
        if existing_index is None:
            rows.append(record)
        else:
            rows[existing_index] = record
        self._write_queue(rows)
        return record

    def _new_record(self, *, character_id: str, variant: str, frame_idx: int, generated_frame_path: str, target_frame_path: Optional[str], metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        metadata = dict(metadata or {})
        run_id = str(metadata.get("run_id") or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
        now = _now_iso()
        record = {
            "review_id": str(metadata.get("review_id") or f"frame_review_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}") ,
            "run_id": run_id,
            "character_id": str(character_id),
            "variant": str(variant),
            "frame_idx": int(frame_idx),
            "row": metadata.get("row"),
            "column": metadata.get("column"),
            "epoch": metadata.get("epoch"),
            "status": "NEEDS_REVIEW",
            "generated_frame_path": self._ensure_safe_path(generated_frame_path) or str(generated_frame_path),
            "target_frame_path": self._ensure_safe_path(target_frame_path) if target_frame_path else None,
            "reference_front_path": self._ensure_safe_path(metadata.get("reference_front_path")) if metadata.get("reference_front_path") else None,
            "quality": {},
            "issues": [],
            "severity": "medium",
            "user_action": None,
            "times_failed": 0,
            "times_reevaluated": 0,
            "times_regenerated": 0,
            "sent_to_reinforcement": False,
            "created_at": now,
            "updated_at": now,
            "history": [{"timestamp": now, "action": "registered", "status": "NEEDS_REVIEW"}],
        }
        return record

    def resolve_target_path(
        self,
        character_id: str,
        variant: Optional[str] = None,
        frame_idx: Optional[int] = None,
        target_frame_path: Optional[str | Path] = None,
    ) -> Optional[str]:
        if target_frame_path is not None:
            safe_path = self._ensure_safe_path(target_frame_path)
            if safe_path:
                return safe_path
            return None

        for row in self._load_queue():
            if row.get("character_id") != str(character_id):
                continue
            if variant is not None and row.get("variant") != str(variant):
                continue
            if frame_idx is not None and int(row.get("frame_idx", -1)) != int(frame_idx):
                continue
            candidate = row.get("target_frame_path")
            safe_candidate = self._ensure_safe_path(candidate) if candidate else None
            if safe_candidate:
                return safe_candidate

        candidates = [
            self.base_dir.rglob("*.png"),
            self.base_dir.rglob("*.jpg"),
            self.base_dir.rglob("*.jpeg"),
            self.base_dir.rglob("*.webp"),
        ]
        lowered_character = str(character_id).lower()
        lowered_variant = str(variant or "").lower()
        lowered_frame = str(frame_idx) if frame_idx is not None else ""
        for file_group in candidates:
            for candidate in file_group:
                path = self._ensure_safe_path(candidate)
                if path is None:
                    continue
                filename_lower = candidate.name.lower()
                if lowered_character in filename_lower and (not lowered_variant or lowered_variant in filename_lower) and (not lowered_frame or lowered_frame in filename_lower):
                    return path
        return None

    def evaluate_frame(
        self,
        generated_frame_path: str | Path,
        target_frame_path: Optional[str | Path] = None,
        *,
        reference_front: Optional[str | Path] = None,
        frame_idx: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = dict(metadata or {})
        source = self._ensure_safe_path(generated_frame_path)
        if not source:
            return {
                "audit_available": False,
                "aprobado": False,
                "score_total": None,
                "issues": [],
                "quality": {},
                "diagnostico": "Ruta de frame generado fuera del árbol permitido",
                "status": "ERROR",
            }

        target = self._ensure_safe_path(target_frame_path) if target_frame_path else None
        if not target:
            return {
                "audit_available": False,
                "aprobado": False,
                "score_total": None,
                "issues": [],
                "quality": {},
                "diagnostico": "No hay target válido para evaluar el frame",
                "status": "ERROR",
            }

        try:
            generated = Image.open(source).convert("RGBA")
            target_img = Image.open(target).convert("RGBA")
        except Exception:
            return {
                "audit_available": False,
                "aprobado": False,
                "score_total": None,
                "issues": [],
                "quality": {},
                "diagnostico": "Los archivos del frame o el target no pudieron abrirse",
                "status": "ERROR",
            }

        if generated.size != target_img.size:
            target_img = target_img.resize(generated.size, Image.Resampling.BILINEAR)

        gen_arr = np.asarray(generated, dtype=np.float32) / 255.0
        tgt_arr = np.asarray(target_img, dtype=np.float32) / 255.0
        diff = np.abs(gen_arr - tgt_arr)
        global_score = max(0.0, min(100.0, 100.0 - float(np.mean(diff)) * 100.0))

        alpha_diff = np.abs(gen_arr[:, :, 3] - tgt_arr[:, :, 3])
        alpha_score = max(0.0, min(100.0, 100.0 - float(np.mean(alpha_diff)) * 100.0))

        silhouette = np.clip((np.mean((gen_arr[:, :, 3] > 0.05) == (tgt_arr[:, :, 3] > 0.05))) * 100.0, 0.0, 100.0)
        anatomy = max(0.0, min(100.0, global_score * 0.65 + silhouette * 0.35))
        face = max(0.0, min(100.0, global_score * 0.7 + (100.0 - float(np.mean(diff[0: max(1, generated.height // 3), :, :])) * 100.0) * 0.3))
        props = max(0.0, min(100.0, global_score * 0.8 + (100.0 - float(np.mean(diff[generated.height // 3 :, :, :])) * 100.0) * 0.2))
        palette = max(0.0, min(100.0, 100.0 - float(np.mean(np.abs(gen_arr[:, :, :3] - tgt_arr[:, :, :3]))) * 100.0))
        micro_detail = max(0.0, min(100.0, silhouette * 0.6 + palette * 0.4))

        quality = {
            "global": round(float(global_score), 2),
            "anatomy": round(float(anatomy), 2),
            "silhouette": round(float(silhouette), 2),
            "face": round(float(face), 2),
            "clothing": round(float(global_score * 0.95), 2),
            "arms_hands": round(float(anatomy * 0.85), 2),
            "props": round(float(props), 2),
            "feet": round(float(global_score * 0.9), 2),
            "palette": round(float(palette), 2),
            "alpha": round(float(alpha_score), 2),
            "micro_detail": round(float(micro_detail), 2),
        }

        issues: List[str] = []
        for key, threshold in {"face": 60.0, "props": 65.0, "micro_detail": 70.0, "silhouette": 75.0}.items():
            if quality.get(key, 100.0) < threshold:
                issues.append(key)

        result = {
            "audit_available": True,
            "aprobado": bool(float(quality.get("global", 0.0)) >= 70.0),
            "score_total": round(float(np.mean(list(quality.values()))), 2),
            "quality": quality,
            "issues": issues,
            "diagnostico": "Auditoría por frame ejecutada con target válido" if not issues else "Problemas detectados en áreas críticas",
            "status": "REEVALUATED",
            "target_frame_path": target,
            "generated_frame_path": source,
            "frame_idx": frame_idx,
            "metadata": metadata,
        }
        return result

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
        safe_generated = self._ensure_safe_path(generated_frame_path)
        if safe_generated is None:
            raise ValueError("generated_frame_path must be inside the project tree")

        resolved_target = self.resolve_target_path(character_id, variant, frame_idx, target_frame_path)
        record = self._new_record(
            character_id=character_id,
            variant=variant,
            frame_idx=frame_idx,
            generated_frame_path=safe_generated,
            target_frame_path=resolved_target,
            metadata=metadata,
        )

        audit = self.evaluate_frame(safe_generated, resolved_target, frame_idx=frame_idx, metadata={**(metadata or {}), "character_id": character_id, "variant": variant}) if resolved_target else {"audit_available": False, "score_total": None, "quality": {}, "issues": []}
        record["quality"] = audit.get("quality", {}) or {}
        record["issues"] = audit.get("issues", [])
        record["status"] = "NEEDS_REVIEW"
        if not audit.get("audit_available"):
            record["quality"] = {}
            record["issues"] = []
        record["updated_at"] = _now_iso()
        record["history"].append({"timestamp": _now_iso(), "action": "registered", "status": record["status"], "score": record["quality"].get("global")})
        return self._persist_record(record)

    def reevaluate_frame(
        self,
        review_id: str,
        *,
        quality: Optional[Mapping[str, Any]] = None,
        issues: Optional[List[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = self._read_record_by_id(review_id)
        if record is None:
            raise KeyError(f"Review not found: {review_id}")

        audit = self.evaluate_frame(
            record.get("generated_frame_path"),
            record.get("target_frame_path"),
            frame_idx=record.get("frame_idx"),
            metadata={
                **(metadata or {}),
                "character_id": record.get("character_id"),
                "variant": record.get("variant"),
            },
        ) if quality is None else {"audit_available": True, "quality": dict(quality), "issues": list(issues or []), "score_total": float(dict(quality).get("global", 0.0)) if isinstance(quality, Mapping) else None}

        record["quality"] = audit.get("quality", record.get("quality", {})) or {}
        record["issues"] = list(audit.get("issues", record.get("issues", [])))
        record["status"] = "REEVALUATED"
        record["times_reevaluated"] = int(record.get("times_reevaluated", 0)) + 1
        record["updated_at"] = _now_iso()
        record["history"].append({
            "timestamp": record["updated_at"],
            "action": "reevaluated",
            "status": record["status"],
            "quality": record["quality"],
            "issues": record["issues"],
        })
        return self._persist_record(record)

    def approve_frame(self, review_id: str, *, user: Optional[str] = None, comment: Optional[str] = None) -> Dict[str, Any]:
        record = self._read_record_by_id(review_id)
        if record is None:
            raise KeyError(f"Review not found: {review_id}")
        record["status"] = "APPROVED"
        record["user_action"] = "APPROVED"
        record["updated_at"] = _now_iso()
        record["history"].append({"timestamp": record["updated_at"], "action": "approved", "user": user, "comment": comment})
        return self._persist_record(record)

    def reject_frame(self, review_id: str, *, user: Optional[str] = None, reason: Optional[str] = None) -> Dict[str, Any]:
        record = self._read_record_by_id(review_id)
        if record is None:
            raise KeyError(f"Review not found: {review_id}")
        record["status"] = "REJECTED"
        record["user_action"] = "REJECTED"
        record["updated_at"] = _now_iso()
        record["history"].append({"timestamp": record["updated_at"], "action": "rejected", "user": user, "reason": reason})
        return self._persist_record(record)

    def get_queue(self, *, status: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = self._load_queue()
        if status is None:
            return rows
        return [row for row in rows if row.get("status") == status]


class QualityGate:
    """Compatibility layer with the existing QC system."""

    @staticmethod
    def evaluate_single_frame(
        generated_frame,
        target_frame=None,
        *,
        reference_front=None,
        frame_idx=None,
        metadata=None,
    ) -> Dict[str, Any]:
        metadata = dict(metadata or {})
        if generated_frame is None:
            return {
                "audit_available": False,
                "aprobado": False,
                "score_total": None,
                "quality": {},
                "issues": [],
                "diagnostico": "No hay frame generado para evaluar",
                "status": "ERROR",
            }

        generated_path = str(generated_frame)
        if target_frame is None:
            return {
                "audit_available": False,
                "aprobado": False,
                "score_total": None,
                "quality": {},
                "issues": [],
                "diagnostico": "No hay target válido para evaluar el frame",
                "status": "ERROR",
            }

        try:
            generated = Image.open(generated_path).convert("RGBA") if isinstance(generated_frame, (str, Path)) else generated_frame.convert("RGBA")
            target = Image.open(str(target_frame)).convert("RGBA") if isinstance(target_frame, (str, Path)) else target_frame.convert("RGBA")
        except Exception:
            return {
                "audit_available": False,
                "aprobado": False,
                "score_total": None,
                "quality": {},
                "issues": [],
                "diagnostico": "No se pudo abrir el frame generado o el target",
                "status": "ERROR",
            }

        if generated.size != target.size:
            target = target.resize(generated.size, Image.Resampling.BILINEAR)

        generated_arr = np.asarray(generated, dtype=np.float32) / 255.0
        target_arr = np.asarray(target, dtype=np.float32) / 255.0
        diff = np.abs(generated_arr - target_arr)
        global_score = max(0.0, min(100.0, 100.0 - float(np.mean(diff)) * 100.0))
        alpha_score = max(0.0, min(100.0, 100.0 - float(np.mean(np.abs(generated_arr[:, :, 3] - target_arr[:, :, 3]))) * 100.0))
        silhouette = max(0.0, min(100.0, float(np.mean((generated_arr[:, :, 3] > 0.05) == (target_arr[:, :, 3] > 0.05))) * 100.0))
        anatomy = max(0.0, min(100.0, global_score * 0.7 + silhouette * 0.3))
        face = max(0.0, min(100.0, global_score * 0.75 + silhouette * 0.25))
        props = max(0.0, min(100.0, global_score * 0.8))
        palette = max(0.0, min(100.0, 100.0 - float(np.mean(np.abs(generated_arr[:, :, :3] - target_arr[:, :, :3]))) * 100.0))
        micro_detail = max(0.0, min(100.0, silhouette * 0.65 + palette * 0.35))
        quality = {
            "global": round(float(global_score), 2),
            "anatomy": round(float(anatomy), 2),
            "silhouette": round(float(silhouette), 2),
            "face": round(float(face), 2),
            "clothing": round(float(global_score * 0.9), 2),
            "arms_hands": round(float(anatomy * 0.9), 2),
            "props": round(float(props), 2),
            "feet": round(float(global_score * 0.85), 2),
            "palette": round(float(palette), 2),
            "alpha": round(float(alpha_score), 2),
            "micro_detail": round(float(micro_detail), 2),
        }
        issues: List[str] = []
        for key, threshold in {"face": 60.0, "props": 65.0, "micro_detail": 70.0, "silhouette": 75.0}.items():
            if float(quality.get(key, 100.0)) < threshold:
                issues.append(key)

        score_total = round(float(np.mean(list(quality.values()))), 2)
        return {
            "audit_available": True,
            "aprobado": bool(score_total >= 70.0),
            "score_total": score_total,
            "quality": quality,
            "issues": issues,
            "diagnostico": "Auditoría por frame ejecutada con referencia disponible",
            "status": "REEVALUATED",
            "target_frame_path": str(target_frame),
            "generated_frame_path": generated_path,
            "metadata": metadata,
        }
