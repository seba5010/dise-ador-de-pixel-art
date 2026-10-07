"""
Módulo de Auditoría Periódica y Auto-Inyección de Ejemplos Difíciles
Ejecutado cada 10 épocas post-guardado de respawn. Audita a todos los personajes
(29 monos) en GPU, genera el reporte versionado en 'reportes/', actualiza el
reporte canónico e inyecta los frames con desviaciones anatómicas o de contorno
directamente a hard_examples.jsonl para que la época 11 (21, 31, 41...) los priorice
automáticamente en el WeightedRandomSampler del DataLoader.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pixel_ai_engine.config import DEVICE, PROJECT_ROOT, USE_AMP
from pixel_ai_engine.dataset import TemplateManager
from pixel_ai_engine.hard_examples import HARD_EXAMPLES_FILE, _slug
from audit_imaginary_frames import (
    audit_single_character,
    build_markdown_master_report,
    find_character_front,
)


def inject_weak_frames_to_hard_examples(
    characters_results: List[Dict[str, Any]],
    threshold_anatomy: float = 75.0,
    threshold_outline: float = 75.0,
    max_stray_limbs: float = 0.03,
    priority: float = 2.0,
    dataset_character_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Lee hard_examples.jsonl, identifica frames débiles de la auditoría y los inyecta
    atómicamente con prioridad alta para que el WeightedRandomSampler los priorice.
    """
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    existing_records: Dict[str, Dict[str, Any]] = {}
    if HARD_EXAMPLES_FILE.is_file():
        for line in HARD_EXAMPLES_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict) and rec.get("key"):
                    existing_records[rec["key"]] = rec
            except json.JSONDecodeError:
                pass

    injected_count = 0
    newly_added = 0
    updated_count = 0
    affected_chars = set()

    # Mapear nombres base de personajes a sus variantes en el dataset
    ds_variants_by_base: Dict[str, List[str]] = {}
    if dataset_character_ids:
        for ds_id in dataset_character_ids:
            parts = ds_id.split("_")
            base = parts[0].lower()
            variant = "_".join(parts[1:]) if len(parts) > 1 else "rnormal"
            ds_variants_by_base.setdefault(base, []).append(variant)

    for char_res in characters_results:
        cname_raw = char_res["character"]
        cname = _slug(cname_raw)
        frames = char_res.get("frames", [])

        # Variantes para este personaje
        variants = ds_variants_by_base.get(cname, [])
        if not variants:
            # Fallback a variantes comunes
            variants = ["rnormal"]
            if cname in ["alex", "amaro", "andrea", "bastian", "belial", "carlos", "conny", "dafne", "dana", "duvan", "erin", "juan", "mario", "zack"]:
                variants.extend(["rbchef", "rnchef"])
        if "rnormal" not in variants:
            variants.append("rnormal")

        for f in frames:
            f_idx = f["frame_idx"]
            raw_anat = f.get("raw_anatomy", 100.0)
            raw_out = f.get("raw_outline", 100.0)
            raw_stray = f.get("raw_stray_limbs", 0.0)

            is_weak = (
                raw_anat < threshold_anatomy or
                raw_out < threshold_outline or
                raw_stray > max_stray_limbs
            )
            if not is_weak:
                continue

            reasons = []
            if raw_anat < threshold_anatomy:
                reasons.append("anatomy_deficit")
            if raw_out < threshold_outline:
                reasons.append("outline_deficit")
            if raw_stray > max_stray_limbs:
                reasons.append("stray_limbs_excess")

            for variant in variants:
                key = f"{cname}::{variant}::frame_{f_idx:03d}"
                prev = existing_records.get(key)

                record = {
                    "active": True,
                    "added_at": prev.get("added_at", now_iso) if prev else now_iso,
                    "character_id": cname,
                    "variant": variant,
                    "frame_idx": f_idx,
                    "key": key,
                    "last_failed_at": now_iso,
                    "last_score": round(raw_anat, 2),
                    "priority": priority,
                    "reason": reasons,
                    "severity": "high" if raw_anat < 65.0 else "medium",
                    "status": "ACTIVE",
                    "target_verified": True,
                    "times_failed": (prev.get("times_failed", 0) + 1) if prev else 1,
                    "source": "periodic_surgical_audit",
                }
                if prev:
                    updated_count += 1
                else:
                    newly_added += 1

                existing_records[key] = record
                injected_count += 1
                affected_chars.add(cname)

    # Escritura atómica a disco
    temp_file = HARD_EXAMPLES_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8", newline="\n") as f:
        for rec in sorted(existing_records.values(), key=lambda r: str(r.get("key"))):
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp_file, HARD_EXAMPLES_FILE)

    return {
        "total_injected": injected_count,
        "newly_added": newly_added,
        "updated_count": updated_count,
        "characters_affected": sorted(list(affected_chars)),
    }


def run_periodic_audit_and_injection(
    generator: torch.nn.Module,
    template_manager: TemplateManager,
    epoch: int,
    session_id: str,
    device: torch.device,
    reports_dir: Optional[Path] = None,
    canonical_output: Optional[Path] = None,
    batch_size: int = 16,
    inject_hard_examples: bool = True,
    threshold_anatomy: float = 75.0,
    threshold_outline: float = 75.0,
    max_stray_limbs: float = 0.03,
    priority: float = 2.0,
    era_name: str = "Era 2 (Fase 2: Transferencia 8x12 - 96 Frames / Acciones Complejas)",
    dataset_character_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Ejecuta la auditoría quirúrgica sobre los 29 personajes, genera reportes y
    auto-inyecta los frames débiles a hard_examples.jsonl.
    """
    rep_dir = reports_dir or (PROJECT_ROOT / "reportes")
    rep_dir.mkdir(parents=True, exist_ok=True)
    can_out = canonical_output or (PROJECT_ROOT / "REPORTE_CALIDAD_FRAMES_IMAGINARIOS.md")

    pdir = PROJECT_ROOT / "personajes"
    char_dirs = sorted([d for d in pdir.iterdir() if d.is_dir()])
    char_tasks = []
    for cd in char_dirs:
        fp = find_character_front(cd)
        if fp:
            char_tasks.append((cd.name, fp))

    generator_was_training = generator.training
    generator.eval()

    now_dt = datetime.now()
    timestamp_tag = now_dt.strftime("%Y%m%d_%H%M%S")
    timestamp_human = now_dt.strftime("%d/%m/%Y %H:%M:%S")

    t_start = datetime.now()
    characters_results = []

    with torch.no_grad():
        for idx, (cname, front_p) in enumerate(char_tasks, 1):
            char_res = audit_single_character(
                generator=generator,
                template_manager=template_manager,
                front_path=front_p,
                device=device,
                batch_size=batch_size,
            )
            characters_results.append(char_res)

    if generator_was_training:
        generator.train()

    t_dur = (datetime.now() - t_start).total_seconds()
    device_str = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"

    # Construir reporte Markdown
    report_md = build_markdown_master_report(
        characters_results=characters_results,
        checkpoint_name=f"checkpoint_epoch_{epoch:03d}.pt",
        epoch=epoch,
        session_id=session_id,
        era_name=era_name,
        timestamp_str=timestamp_human,
        device_name=device_str,
    )

    # 1. Guardar archivos versionados en reportes/
    era_slug = "era_2"
    file_path_md = rep_dir / f"reporte_calidad_{era_slug}_epoca_{epoch}_{timestamp_tag}.md"
    file_path_json = rep_dir / f"reporte_calidad_{era_slug}_epoca_{epoch}_{timestamp_tag}.json"

    with open(file_path_md, "w", encoding="utf-8") as f:
        f.write(report_md)

    with open(file_path_json, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "era": era_name,
                "epoch": epoch,
                "session_id": session_id,
                "checkpoint": f"checkpoint_epoch_{epoch:03d}.pt",
                "timestamp": timestamp_human,
                "timestamp_tag": timestamp_tag,
                "device": device_str,
                "total_characters": len(characters_results),
                "total_frames": len(characters_results) * 96,
                "duration_seconds": round(t_dur, 2),
            },
            "characters_summaries": [c["summary"] for c in characters_results],
            "characters_frames": {c["character"]: c["frames"] for c in characters_results},
        }, f, indent=2)

    # 2. Actualizar reporte canónico
    with open(can_out, "w", encoding="utf-8") as f:
        f.write(report_md)

    # 3. Auto-inyección a hard_examples.jsonl
    injected_summary = {"total_injected": 0, "newly_added": 0, "updated_count": 0}
    if inject_hard_examples:
        injected_summary = inject_weak_frames_to_hard_examples(
            characters_results=characters_results,
            threshold_anatomy=threshold_anatomy,
            threshold_outline=threshold_outline,
            max_stray_limbs=max_stray_limbs,
            priority=priority,
            dataset_character_ids=dataset_character_ids,
        )

    return {
        "epoch": epoch,
        "duration_seconds": round(t_dur, 2),
        "report_md": str(file_path_md),
        "report_json": str(file_path_json),
        "injected_count": injected_summary["total_injected"],
        "newly_added": injected_summary["newly_added"],
        "updated_count": injected_summary["updated_count"],
        "characters_affected": injected_summary.get("characters_affected", []),
    }
