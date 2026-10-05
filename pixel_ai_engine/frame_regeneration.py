"""Safe single-frame regeneration and candidate ranking.

Candidates are output artifacts only.  They never become dataset targets.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Protocol, Sequence
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from .config import CHECKPOINT_DIR, DEVICE, MODEL_RESOLUTION, PROJECT_ROOT, USE_AMP, get_phase_config
from .dataset import TemplateManager, adapt_front_to_chibi, get_cell_coordinates, place_in_cell
from .enhancer import PixelArtEnhancer
from .frame_quality_review import FrameQualityReviewManager
from .models import PixelArtUNetGenerator


MAX_ALLOWED_REGRESSION = 5.0
CRITICAL_FLOORS: Dict[str, float] = {
    "anatomy": 55.0,
    "silhouette": 55.0,
    "face": 45.0,
    "props": 45.0,
    "alpha": 95.0,
    "palette": 60.0,
}
CRITICAL_ISSUES = {
    "ALPHA_FAIL",
    "BORDER_TOUCH_TOP",
    "BORDER_TOUCH_BOTTOM",
    "BORDER_TOUCH_LEFT",
    "BORDER_TOUCH_RIGHT",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _score(payload: Mapping[str, Any], name: str) -> Optional[float]:
    quality = payload.get("quality") if isinstance(payload.get("quality"), Mapping) else payload
    value = quality.get(name) if isinstance(quality, Mapping) else None
    if value is None and name == "global":
        value = payload.get("score_total")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return min(100.0, max(0.0, number)) if np.isfinite(number) else None


def rank_regeneration_candidates(
    original: Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
    *,
    primary_problem: Optional[str] = None,
    critical_floors: Optional[Mapping[str, float]] = None,
    max_allowed_regression: float = MAX_ALLOWED_REGRESSION,
) -> Dict[str, Any]:
    """Rank only candidates that improve the dominant issue without regressions."""
    floors = dict(CRITICAL_FLOORS)
    if critical_floors:
        floors.update({str(key): float(value) for key, value in critical_floors.items()})
    dominant = str(primary_problem or (original.get("diagnosis") or {}).get("primary_problem") or "global")
    # Some run-level diagnoses (for example training stability) have no
    # comparable per-frame metric. Fall back to the global frame score.
    if _score(original, dominant) is None:
        dominant = "global"
    tolerance = max(0.0, float(max_allowed_regression))
    original_quality = original.get("quality") if isinstance(original.get("quality"), Mapping) else original
    ranked = []

    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        quality = candidate.get("quality") if isinstance(candidate.get("quality"), Mapping) else {}
        violations = []
        regressions: Dict[str, float] = {}
        if candidate.get("audit_available") is False or _score(candidate, "global") is None:
            violations.append("audit_unavailable")
        for category, floor in floors.items():
            value = _score(candidate, category)
            if value is not None and value < floor:
                violations.append(f"{category}_below_floor")
        if CRITICAL_ISSUES.intersection(set(candidate.get("issues") or [])):
            violations.append("critical_image_issue")

        for category, old_value in original_quality.items() if isinstance(original_quality, Mapping) else []:
            try:
                before = float(old_value)
            except (TypeError, ValueError):
                continue
            after = _score(candidate, str(category))
            if after is not None and before - after > tolerance:
                regressions[str(category)] = round(before - after, 4)
        if regressions:
            violations.append("REGRESSION_DETECTED")

        before_dominant = _score(original, dominant)
        after_dominant = _score(candidate, dominant)
        dominant_delta = (
            round(after_dominant - before_dominant, 4)
            if before_dominant is not None and after_dominant is not None
            else None
        )
        if dominant_delta is not None and dominant_delta <= 0.0:
            violations.append("dominant_problem_not_improved")

        global_delta = None
        old_global = _score(original, "global")
        new_global = _score(candidate, "global")
        if old_global is not None and new_global is not None:
            global_delta = round(new_global - old_global, 4)
        utility = (new_global or 0.0) + max(0.0, dominant_delta or 0.0) * 1.5
        candidate.update({
            "safe": not violations,
            "violations": list(dict.fromkeys(violations)),
            "regressions": regressions,
            "dominant_problem": dominant,
            "dominant_delta": dominant_delta,
            "global_delta": global_delta,
            "ranking_score": round(utility, 4),
            "quality": dict(quality),
        })
        ranked.append(candidate)

    ranked.sort(key=lambda item: (bool(item["safe"]), float(item["ranking_score"])), reverse=True)
    safe = [item for item in ranked if item["safe"]]
    best = safe[0] if safe else None
    if best is None:
        reason = "Ningún candidato superó pisos críticos y tolerancia de regresión"
    else:
        deltas = []
        if best.get("dominant_delta") is not None:
            deltas.append(f"{dominant} {best['dominant_delta']:+.1f}")
        if best.get("global_delta") is not None:
            deltas.append(f"global {best['global_delta']:+.1f}")
        reason = "Mejor candidato seguro: " + ", ".join(deltas or ["sin regresiones críticas"])
    return {
        "best_candidate": best.get("candidate_id") if best else None,
        "reason": reason,
        "primary_problem": dominant,
        "max_allowed_regression": tolerance,
        "regression_detected": best is None and any("REGRESSION_DETECTED" in item["violations"] for item in ranked),
        "candidates": ranked,
    }


class CandidateGenerator(Protocol):
    def generate(self, review: Mapping[str, Any], num_candidates: int) -> Sequence[Image.Image]: ...


class TorchFrameCandidateGenerator:
    """Generate several alternatives while loading the selected checkpoint once."""

    def __init__(self, review_manager: FrameQualityReviewManager):
        self.review_manager = review_manager

    def _safe_existing_path(self, raw: Any, roots: Sequence[Path]) -> Optional[Path]:
        if not raw:
            return None
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute():
            candidate = self.review_manager.project_root / candidate
        resolved = candidate.resolve()
        if not resolved.is_file():
            return None
        for root in roots:
            try:
                resolved.relative_to(root.resolve())
                return resolved
            except ValueError:
                continue
        return None

    def _checkpoint(self, metadata: Mapping[str, Any]) -> Path:
        roots = [CHECKPOINT_DIR, PROJECT_ROOT / "checkpoints"]
        configured = self._safe_existing_path(metadata.get("checkpoint"), roots)
        if configured:
            return configured
        preferred = []
        for pattern in ("best_generator*.pt", "latest_checkpoint*.pt", "generator_epoch_*.pt"):
            preferred.extend(CHECKPOINT_DIR.rglob(pattern))
        existing = sorted((path for path in preferred if path.is_file()), key=lambda path: path.stat().st_mtime, reverse=True)
        if not existing:
            raise FileNotFoundError("No compatible generator checkpoint was found")
        return existing[0]

    def generate(self, review: Mapping[str, Any], num_candidates: int) -> Sequence[Image.Image]:
        import torch
        from torch.cuda.amp import autocast

        run_id = str(review["run_id"])
        frame_idx = int(review["frame_idx"])
        run = self.review_manager.resolve_run_frame(run_id, frame_idx)
        metadata = run["metadata"]
        roots = [PROJECT_ROOT / "personajes", self.review_manager.dataset_frames_root, self.review_manager.supervised_root]
        front_path = self._safe_existing_path(metadata.get("front_image"), roots)
        if front_path is None and review.get("reference_front_path"):
            front_path = self._safe_existing_path(review.get("reference_front_path"), roots)
        if front_path is None:
            raise FileNotFoundError("Identity reference was not found")

        format_type = str(metadata.get("format") or "8x12")
        phase_cfg = get_phase_config("2" if format_type == "8x12" else "1")
        front = adapt_front_to_chibi(Image.open(front_path), target_size=MODEL_RESOLUTION)
        palette = PixelArtEnhancer.extract_palette(front, max_colors=40)
        front_array = np.asarray(front, dtype=np.float32).copy()
        mask = front_array[..., 3] > 20
        rgb = front_array[..., :3]
        rgb[~mask] = 0.0
        front_tensor = torch.from_numpy(rgb / 127.5 - 1.0).permute(2, 0, 1).float().to(DEVICE)
        templates = TemplateManager(
            phase_cfg["template_path"], MODEL_RESOLUTION,
            phase_cfg["grid_rows"], phase_cfg["grid_cols"],
            phase_cfg["canvas_w"], phase_cfg["canvas_h"],
        )
        pose = templates.get_frame_tensor(frame_idx).to(DEVICE)
        base_condition = torch.cat([front_tensor, pose], dim=0).unsqueeze(0)
        generator = PixelArtUNetGenerator().to(DEVICE)
        checkpoint = torch.load(self._checkpoint(metadata), map_location=DEVICE)
        generator.load_state_dict(checkpoint.get("generator", checkpoint) if isinstance(checkpoint, dict) else checkpoint)
        generator.eval()

        outputs = []
        with torch.no_grad():
            for index in range(max(1, min(8, int(num_candidates)))):
                torch.manual_seed(1009 + frame_idx * 97 + index)
                condition = base_condition.clone()
                if index:
                    noise = torch.randn_like(condition[:, :3]) * min(0.035, 0.008 * index)
                    condition[:, :3] = (condition[:, :3] + noise).clamp(-1.0, 1.0)
                with autocast(enabled=USE_AMP):
                    tensor = generator(condition).squeeze(0)
                array = tensor.detach().cpu().permute(1, 2, 0).numpy()
                array = np.clip((array + 1.0) * 127.5, 0, 255).astype(np.uint8)
                raw = Image.fromarray(array, mode="RGBA")
                outputs.append(PixelArtEnhancer.enhance_frame(
                    raw, palette=palette, snap_palette=True,
                    remove_noise=True, binarize=True, sharpen_tattoos=True,
                ))
        return outputs


class FrameRegenerationManager:
    def __init__(
        self,
        review_manager: FrameQualityReviewManager,
        *,
        generator: Optional[CandidateGenerator] = None,
        evaluator: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        self.review_manager = review_manager
        self.generator = generator or TorchFrameCandidateGenerator(review_manager)
        self.evaluator = evaluator

    def _evaluate(self, image: Image.Image, target: Path, reference: Optional[Path], review: Mapping[str, Any]) -> Dict[str, Any]:
        if self.evaluator is None:
            from .quality_gate import QualityGate
            evaluator = QualityGate.evaluate_single_frame
        else:
            evaluator = self.evaluator
        return evaluator(image, target, reference_front=reference, frame_idx=int(review["frame_idx"]), metadata=dict(review))

    def regenerate(self, review_id: str, *, num_candidates: int = 3, actor: Optional[str] = None) -> Dict[str, Any]:
        review = self.review_manager.get_review(review_id)
        if review is None:
            raise KeyError(review_id)
        if not review.get("target_frame_path"):
            raise ValueError("regeneration_blocked_missing_target")
        target = self.review_manager.project_root / str(review["target_frame_path"])
        reference = self.review_manager.project_root / str(review["reference_front_path"]) if review.get("reference_front_path") else None
        target = target.resolve()
        if not target.is_file():
            raise ValueError("regeneration_blocked_missing_target")
        run = self.review_manager.resolve_run_frame(str(review["run_id"]), int(review["frame_idx"]))
        generation_id = f"regen_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
        candidate_dir = run["run_dir"] / "regeneration" / f"frame_{int(review['frame_idx']):03d}" / generation_id
        candidate_dir.mkdir(parents=True, exist_ok=False)
        images = list(self.generator.generate(review, max(1, min(8, int(num_candidates)))))
        candidates = []
        for index, image in enumerate(images, start=1):
            candidate_id = f"candidate_{index:02d}"
            path = candidate_dir / f"{candidate_id}.png"
            image.convert("RGBA").save(path, format="PNG")
            audit = self._evaluate(image, target, reference if reference and reference.is_file() else None, review)
            candidates.append({
                "candidate_id": candidate_id,
                "path": path.relative_to(self.review_manager.project_root).as_posix(),
                "score_total": audit.get("score_total"),
                "quality": dict(audit.get("quality") or {}),
                "issues": list(audit.get("issues") or []),
                "severity": audit.get("severity"),
                "diagnosis": dict(audit.get("diagnosis") or {}),
                "created_at": _now(),
                "state": "READY",
            })
        ranking = rank_regeneration_candidates(
            review,
            candidates,
            primary_problem=(review.get("diagnosis") or {}).get("primary_problem"),
        )
        return self.review_manager.record_regeneration(
            review_id,
            generation_id=generation_id,
            candidate_dir=candidate_dir.relative_to(self.review_manager.project_root).as_posix(),
            ranking=ranking,
            actor=actor,
        )

    def _atomic_copy(self, source: Path, destination: Path) -> None:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)

    def _patch_sheets(self, run_dir: Path, frame_idx: int, replacement: Image.Image, format_type: str, backup_dir: Path) -> None:
        cfg = get_phase_config("2" if format_type == "8x12" else "1")
        row, column = divmod(frame_idx, int(cfg["grid_cols"]))
        x0, y0, x1, y1 = get_cell_coordinates(
            int(cfg["canvas_w"]), int(cfg["canvas_h"]), row, column,
            int(cfg["grid_rows"]), int(cfg["grid_cols"]),
        )
        placed = place_in_cell(replacement, cell_w=x1 - x0, cell_h=y1 - y0)
        for name, background in (("spritesheet_clean.png", (0, 0, 0, 0)), ("spritesheet_grid.png", (220, 224, 232, 255))):
            path = run_dir / name
            if not path.is_file():
                continue
            sheet_backup = backup_dir / f"{path.stem}_before_apply.png"
            if not sheet_backup.exists():
                shutil.copy2(path, sheet_backup)
            sheet = Image.open(path).convert("RGBA")
            draw = ImageDraw.Draw(sheet)
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=background)
            if name == "spritesheet_grid.png":
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(180, 185, 195, 255), width=1)
            sheet.alpha_composite(placed, (x0, y0))
            temporary = path.with_suffix(".png.tmp")
            sheet.save(temporary, format="PNG")
            os.replace(temporary, path)

    def apply_best(self, review_id: str, *, candidate_id: Optional[str] = None, actor: Optional[str] = None) -> Dict[str, Any]:
        review = self.review_manager.get_review(review_id)
        if review is None:
            raise KeyError(review_id)
        regeneration = dict(review.get("regeneration") or {})
        if regeneration.get("state") != "READY":
            raise ValueError("regeneration_not_ready")
        ranking = dict(regeneration.get("ranking") or {})
        selected = str(candidate_id or ranking.get("best_candidate") or "")
        if not selected:
            raise ValueError("no_safe_candidate")
        candidate = next((item for item in ranking.get("candidates", []) if item.get("candidate_id") == selected), None)
        if not candidate or not candidate.get("safe"):
            raise ValueError("candidate_not_safe")
        source = (self.review_manager.project_root / str(candidate["path"])).resolve()
        run = self.review_manager.resolve_run_frame(str(review["run_id"]), int(review["frame_idx"]))
        active = run["generated"]
        candidate_dir = (self.review_manager.project_root / str(regeneration["candidate_dir"])).resolve()
        expected_root = (run["run_dir"] / "regeneration" / f"frame_{int(review['frame_idx']):03d}").resolve()
        if not candidate_dir.is_relative_to(expected_root):
            raise ValueError("invalid_candidate_directory")
        if not source.is_file() or not source.is_relative_to(candidate_dir):
            raise ValueError("invalid_candidate_path")
        backup = candidate_dir / f"frame_{int(review['frame_idx']):03d}_original.png"
        if not backup.exists():
            shutil.copy2(active, backup)
        self._atomic_copy(source, active)
        replacement = Image.open(source).convert("RGBA")
        self._patch_sheets(run["run_dir"], int(review["frame_idx"]), replacement, str(run["metadata"].get("format") or "8x12"), candidate_dir)
        self.review_manager.record_candidate_applied(
            review_id,
            candidate_id=selected,
            original_path=backup.relative_to(self.review_manager.project_root).as_posix(),
            replacement_path=source.relative_to(self.review_manager.project_root).as_posix(),
            actor=actor,
        )
        return self.review_manager.reevaluate_frame(review_id, actor=actor)

    def discard(self, review_id: str, *, actor: Optional[str] = None) -> Dict[str, Any]:
        return self.review_manager.discard_regeneration(review_id, actor=actor)


__all__ = [
    "CRITICAL_FLOORS",
    "FrameRegenerationManager",
    "MAX_ALLOWED_REGRESSION",
    "TorchFrameCandidateGenerator",
    "rank_regeneration_candidates",
]
