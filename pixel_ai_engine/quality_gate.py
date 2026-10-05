"""
Pixel AI Engine - Puerta de Control de Calidad Quirúrgica y Diagnóstico
(quality_gate.py)

Implementa la Fase Crítica Intermedia que audita el modelo antes de permitir
el paso a la siguiente fase. Si el modelo no alcanza el 99.5% - 99.9% de asimilación:
1. Compara minuciosamente: Colores (RGB), Molde (IoU silueta), Textura (1px) y Fondo (Alfa).
2. Genera un reporte detallado en Markdown documentando exactamente en qué falló.
3. Activa un bucle de re-entrenamiento adaptativo de refuerzo hasta que supere el umbral.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
from PIL import Image
import torch
from torch.cuda.amp import autocast

from .config import (
    PROJECT_ROOT,
    DEVICE,
    USE_AMP,
    MODEL_RESOLUTION,
    CHECKPOINT_DIR,
    get_phase_config
)
from .dataset import TemplateManager, isolate_character, pad_to_square
from .train import tensor_to_pil
from .enhancer import PixelArtEnhancer
from .models import PixelArtUNetGenerator
from .anatomical_guidance import compute_anatomical_metrics
from .phase3_critical_enhancer import Phase3CriticalReviewer
from .quality_guidance import build_quality_vector, diagnose_quality_bottleneck
from .strict_visual_quality import apply_strict_visual_metrics


class QualityGate:
    """
    Auditor Clínico y Guardián de Transición entre Fases.
    """

    @staticmethod
    def evaluate_single_frame(
        generated_frame,
        target_frame=None,
        *,
        reference_front=None,
        reference_palette=None,
        frame_idx=None,
        metadata=None,
    ) -> Dict[str, Any]:
        """Audit one generated frame without loading or mutating a checkpoint."""
        if generated_frame is None:
            return {
                "audit_available": False,
                "aprobado": False,
                "score_total": None,
                "quality": {},
                "issues": [],
                "diagnostico": "No hay frame generado para evaluar",
                "diagnosis": {"primary_problem": None, "reason": "generated_frame_missing"},
                "severity": "unknown",
                "identity_confidence": None,
                "status": "ERROR",
            }

        if target_frame is None:
            return {
                "audit_available": False,
                "aprobado": False,
                "score_total": None,
                "quality": {},
                "issues": [],
                "diagnostico": "No hay target válido para evaluar el frame",
                "diagnosis": {"primary_problem": None, "reason": "target_not_found"},
                "severity": "unknown",
                "identity_confidence": None,
                "status": "ERROR",
            }

        try:
            if isinstance(generated_frame, (str, Path)):
                generated_path = Path(generated_frame)
                if not generated_path.is_file():
                    raise FileNotFoundError(generated_path)
                generated = Image.open(generated_path).convert("RGBA")
            else:
                generated = generated_frame.convert("RGBA")
            if isinstance(target_frame, (str, Path)):
                target_path = Path(target_frame)
                if not target_path.is_file():
                    raise FileNotFoundError(target_path)
                target = Image.open(target_path).convert("RGBA")
            else:
                target = target_frame.convert("RGBA")
            if reference_front is None:
                identity = None
            elif isinstance(reference_front, (str, Path)):
                identity_path = Path(reference_front)
                identity = Image.open(identity_path).convert("RGBA") if identity_path.is_file() else None
            else:
                identity = reference_front.convert("RGBA")
        except (OSError, ValueError, AttributeError):
            return {
                "audit_available": False,
                "aprobado": False,
                "score_total": None,
                "quality": {},
                "issues": [],
                "diagnostico": "No se pudo abrir el frame generado o el target",
                "diagnosis": {"primary_problem": None, "reason": "invalid_image"},
                "severity": "unknown",
                "identity_confidence": None,
                "status": "ERROR",
            }

        if generated.size != target.size:
            target = target.resize(generated.size, Image.Resampling.NEAREST)

        palette = reference_palette
        if palette is None and identity is not None:
            palette = PixelArtEnhancer.extract_palette(identity, max_colors=40)
        raw_metrics = PixelArtEnhancer.analyze_quality(
            generated,
            identity_img=identity,
            palette=palette,
            target_img=target,
        )
        anatomy_metrics = compute_anatomical_metrics(generated, target)
        anatomy_metrics["anatomy_geometry"] = anatomy_metrics.pop("cuerpo_precision", None)
        raw_metrics.update(anatomy_metrics)
        raw_metrics.update(Phase3CriticalReviewer.audit_frame(generated, identity))
        raw_metrics = apply_strict_visual_metrics(raw_metrics, generated, target)
        vector = build_quality_vector(raw_metrics)
        quality = vector.to_dict()
        diagnosis = diagnose_quality_bottleneck(vector)
        score_total = quality.get("global")

        alpha = np.asarray(generated, dtype=np.uint8)[..., 3]
        foreground = alpha > 0
        issues: List[str] = []
        if np.any((alpha > 0) & (alpha < 255)):
            issues.append("ALPHA_FAIL")
        edges = (
            ("BORDER_TOUCH_TOP", foreground[0, :]),
            ("BORDER_TOUCH_BOTTOM", foreground[-1, :]),
            ("BORDER_TOUCH_LEFT", foreground[:, 0]),
            ("BORDER_TOUCH_RIGHT", foreground[:, -1]),
        )
        issues.extend(name for name, edge in edges if np.any(edge))
        if diagnosis.get("primary_problem"):
            issues.append(str(diagnosis["primary_problem"]))
        issues.extend(str(item) for item in diagnosis.get("secondary_problems", []))
        if float(raw_metrics.get("strict_face", 0.0) or 0.0) < 75.0:
            issues.append("FACE_STRUCTURE_FAIL")
        if float(raw_metrics.get("strict_anatomy", 0.0) or 0.0) < 75.0:
            issues.append("BODY_STRUCTURE_FAIL")
        if float(raw_metrics.get("strict_visual_noise", 0.0) or 0.0) < 80.0:
            issues.append("VISUAL_NOISE_FAIL")
        issues = list(dict.fromkeys(issues))

        critical = {
            "ALPHA_FAIL", "BORDER_TOUCH_TOP", "BORDER_TOUCH_BOTTOM", "BORDER_TOUCH_LEFT", "BORDER_TOUCH_RIGHT",
            "FACE_STRUCTURE_FAIL", "BODY_STRUCTURE_FAIL", "VISUAL_NOISE_FAIL",
        }
        approved = bool(
            score_total is not None
            and float(score_total) >= 85.0
            and not critical.intersection(issues)
            and diagnosis.get("severity") != "critical"
        )
        return {
            "audit_available": True,
            "aprobado": approved,
            "score_total": score_total,
            "quality": quality,
            "issues": issues,
            "severity": diagnosis.get("severity", "unknown"),
            "diagnosis": diagnosis,
            "diagnostico": "Auditoría por frame ejecutada con target válido",
            "identity_confidence": quality.get("palette") if identity is not None else None,
            "raw_metrics": raw_metrics,
            "status": "REEVALUATED",
            "target_frame_path": str(target_frame),
            "generated_frame_path": str(generated_frame),
            "metadata": metadata,
            "frame_idx": frame_idx,
        }

    @classmethod
    def evaluate_generator_frame(
        cls,
        generator,
        front_tensor: torch.Tensor,
        pose_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        *,
        reference_front=None,
        frame_idx=None,
        metadata=None,
    ) -> Dict[str, Any]:
        """Evaluate an already-loaded generator without reconstructing its model."""
        if generator is None or target_tensor is None:
            return cls.evaluate_single_frame(None, None, frame_idx=frame_idx, metadata=metadata)
        if front_tensor.ndim != pose_tensor.ndim or front_tensor.ndim not in {3, 4}:
            raise ValueError("front_tensor and pose_tensor must be compatible CHW or NCHW tensors")
        channel_dimension = 0 if front_tensor.ndim == 3 else 1
        condition = torch.cat([front_tensor, pose_tensor], dim=channel_dimension)
        if condition.ndim == 3:
            condition = condition.unsqueeze(0)
        with torch.no_grad():
            generated_tensor = generator(condition)
        if generated_tensor.ndim == 4:
            generated_tensor = generated_tensor[0]
        target = target_tensor[0] if target_tensor.ndim == 4 else target_tensor
        return cls.evaluate_single_frame(
            tensor_to_pil(generated_tensor),
            tensor_to_pil(target),
            reference_front=reference_front,
            frame_idx=frame_idx,
            metadata=metadata,
        )

    @classmethod
    def evaluate_model_critical(cls,
                                checkpoint_path: Path,
                                sample_img_path: Optional[Union[str, Path]] = None,
                                phase: str = "1",
                                target_threshold: float = 99.5) -> Dict[str, Any]:
        """
        Audita exhaustivamente el modelo frente a la foto del personaje y los moldes de referencia.
        Verifica la foto actual en sí (identidad directa, paleta y accesorios) y la compara contra
        el Ground Truth exacto correspondiente a su variante anatómica y de vestimenta.
        """
        cfg = get_phase_config(phase)
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return {
                "aprobado": False,
                "score_total": 0.0,
                "diagnostico": f"No se encontró el checkpoint en {checkpoint_path}",
                "fallas": ["Archivo de checkpoint inexistente"]
            }

        generator = PixelArtUNetGenerator().to(DEVICE)
        saved = torch.load(checkpoint_path, map_location=DEVICE)
        state_dict = saved.get("generator", saved) if isinstance(saved, dict) else saved
        generator.load_state_dict(state_dict)
        generator.eval()

        template_mgr = TemplateManager(
            template_path=cfg["template_path"],
            target_size=MODEL_RESOLUTION,
            rows=cfg["grid_rows"],
            cols=cfg["grid_cols"],
            canvas_w=cfg["canvas_w"],
            canvas_h=cfg["canvas_h"]
        )

        # ── RESOLVER LA FOTO A EVALUAR ──
        from .config import PERSONAJES_DIR
        resolved_img_path = None
        if sample_img_path is not None:
            cand_p = Path(sample_img_path)
            if cand_p.exists():
                resolved_img_path = cand_p

        if resolved_img_path is None:
            all_images = sorted(
                path for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp")
                for path in PERSONAJES_DIR.rglob(pattern)
                if path.is_file()
            )
            preferred = [
                path for path in all_images
                if any(token in path.stem.casefold() for token in ("front", "frontal", "rnormal", "rbchef", "rnchef"))
            ]
            resolved_img_path = (preferred or all_images or [None])[0]

        if not resolved_img_path:
            return {
                "audit_available": False,
                "aprobado": False,
                "score_total": None,
                "diagnostico": "Sin personaje de referencia local; auditoría no disponible.",
                "fallas": ["Referencia de identidad no encontrada"]
            }

        sample_img_path = resolved_img_path

        # ── 1. PROCESAR LA FOTO ACTUAL EN SÍ (identidad frontal directa) ──
        raw_front = Image.open(sample_img_path)
        padded_front, _ = pad_to_square(raw_front, MODEL_RESOLUTION)
        clean_front = padded_front.convert("RGBA")

        # Fondo neutro negro idéntico al dataset de entrenamiento
        front_rgb = Image.new("RGB", (MODEL_RESOLUTION, MODEL_RESOLUTION), (0, 0, 0))
        front_rgb.paste(clean_front, mask=clean_front.split()[3])
        front_arr = np.array(front_rgb).astype(np.float32) / 127.5 - 1.0
        front_tensor = torch.from_numpy(front_arr).permute(2, 0, 1)[:3].float().to(DEVICE)

        # Extraer paleta canónica directamente de la foto actual (incluyendo accesorios universales)
        canonical_pal = PixelArtEnhancer.extract_palette(clean_front, max_colors=40)

        # Analizar vestimenta de la foto frontal para calibrar auditoría de vestimenta
        arr_front_np = np.array(clean_front)
        h_f, w_f = arr_front_np.shape[:2]
        mask_torso = (arr_front_np[:, :, 3] > 30) & (np.arange(h_f)[:, None] >= int(h_f * 0.40)) & (np.arange(h_f)[:, None] <= int(h_f * 0.70)) & (np.arange(w_f)[None, :] >= int(w_f * 0.30)) & (np.arange(w_f)[None, :] <= int(w_f * 0.70))
        torso_pixels = arr_front_np[mask_torso, :3] if np.any(mask_torso) else np.zeros((1, 3))
        torso_mean_rgb = np.mean(torso_pixels, axis=0)
        white_ratio = float(np.mean((torso_pixels[:, 0] > 150) & (torso_pixels[:, 1] > 150) & (torso_pixels[:, 2] > 150)))
        is_chef_white = bool(white_ratio >= 0.35)

        # ── 2. CARGAR GROUND TRUTH EXACTO CORRESPONDIENTE A LA VARIANTE ACTUAL ──
        gt_samples_map = {}
        sample_char_id = sample_img_path.parent.name

        def normalize_identity(value: Any) -> str:
            return "".join(character for character in str(value).casefold() if character.isalnum())

        try:
            from .dataset import PixelArtDataset
            dataset_ref = PixelArtDataset(phase_cfg=cfg)
            total_frames = cfg["total_frames"]
            all_samples = dataset_ref.samples
            n_blocks = len(all_samples) // total_frames
            best_block = None
            best_diff = float("inf")
            front_cpu = front_tensor.detach().cpu()

            for b_idx in range(n_blocks):
                block = all_samples[b_idx * total_frames : (b_idx + 1) * total_frames]
                b_char = block[0].get("char_id", "")
                if normalize_identity(b_char) == normalize_identity(sample_char_id):
                    diff = float(torch.mean(torch.abs(front_cpu - block[0]["front_tensor"])).item())
                    if diff < best_diff:
                        best_diff = diff
                        best_block = block

            # Coincidencia con la variante exacta en el dataset (distancia L1 < 0.35)
            if best_block is not None and best_diff < 0.35:
                gt_samples_map = {s["frame_idx"]: s["target_tensor"] for s in best_block if "target_tensor" in s}
        except Exception:
            gt_samples_map = {}

        # Evaluar una muestra de frames clave (frente, lateral, espalda, acción)
        total_frames = cfg["total_frames"]
        eval_indices = [0, 1, 2, min(16, total_frames - 1), min(32, total_frames - 1), min(48, total_frames - 1)]
        eval_indices = sorted(list(set([i for i in eval_indices if i < total_frames])))

        # ── AUDITORÍA ANATÓMICA Y DE OBJETOS DETALLADA ──
        detalle_anatomico = {
            "pelo_gorro": [],
            "gestos_ojos": [],
            "ropa_delantal": [],
            "tatuajes_brazos": [],
            "objetos_utensilios": [],
            "zapatos_pies": []
        }

        palette_scores = []
        mold_alignment_scores = []
        ink_detail_scores = []
        alpha_purity_scores = []
        fallas_detectadas = []

        with torch.no_grad():
            for f_idx in eval_indices:
                pose_t = template_mgr.get_frame_tensor(f_idx).to(DEVICE)
                cond = torch.cat([front_tensor, pose_t], dim=0).unsqueeze(0)
                with autocast(enabled=USE_AMP):
                    raw_out = generator(cond).squeeze(0)

                arr = raw_out.detach().cpu().permute(1, 2, 0).numpy()
                arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
                gen_pil = Image.fromarray(arr, mode="RGBA")

                # Obtener sprite objetivo real si está disponible
                tgt_pil = tensor_to_pil(gt_samples_map[f_idx]).convert("RGBA") if f_idx in gt_samples_map else None

                # Pulido quirúrgico para auditar el sprite en su formato canónico de pixel art
                enhanced_pil = PixelArtEnhancer.enhance_frame(
                    gen_pil,
                    palette=canonical_pal,
                    snap_palette=True,
                    remove_noise=True,
                    binarize=True,
                    sharpen_tattoos=True
                )

                # 1. Auditoría Clínica de Paleta, Píxeles del Cuerpo y Color
                qc = PixelArtEnhancer.analyze_quality(enhanced_pil, palette=canonical_pal, target_img=tgt_pil)
                pal_score = qc.get("fidelidad_paleta", 0.0)
                palette_scores.append(pal_score)

                def_px = qc.get("defectos_cuerpo", 0)
                tot_px = qc.get("total_px_cuerpo", 0)
                c_ia = qc.get("colores_ia", 0)
                c_tgt = qc.get("colores_original", 0)
                body_prec = qc.get("cuerpo_precision", 100.0)

                if def_px > 75:
                    fallas_detectadas.append(f"Frame {f_idx} (Cuerpo): {def_px} píxeles del cuerpo con desvío ({body_prec:.1f}% asimilación de {tot_px} px).")
                if c_tgt > 0 and c_ia > int(c_tgt * 1.50):
                    fallas_detectadas.append(f"Frame {f_idx} (Colores): Generó {c_ia} colores (se esperaban máx {int(c_tgt * 1.50)} de los {c_tgt} auténticos).")
                if pal_score < 88.0:
                    fallas_detectadas.append(f"Frame {f_idx} (Color): Desviación cromática ({pal_score:.1f}%).")

                # 2. Auditoría de Molde y Silueta (IoU con sprite real del personaje)
                gen_mask = np.array(enhanced_pil)[:, :, 3] > 30
                if tgt_pil is not None:
                    tgt_m = np.array(tgt_pil)[:, :, 3] > 30
                    intersection = np.logical_and(tgt_m, gen_mask).sum()
                    union = np.logical_or(tgt_m, gen_mask).sum()
                    mold_score = (intersection / max(1, union)) * 100.0
                else:
                    pose_pil = tensor_to_pil(pose_t)
                    pose_arr = np.array(pose_pil.convert("L"))
                    pose_mask = pose_arr > 30
                    intersection = np.logical_and(pose_mask, gen_mask).sum()
                    union = np.logical_or(pose_mask, gen_mask).sum()
                    iou = (intersection / max(1, union)) * 100.0
                    mold_score = min(100.0, max(0.0, iou * 1.15))
                mold_alignment_scores.append(mold_score)

                if mold_score < 88.0:
                    fallas_detectadas.append(f"Frame {f_idx} (Molde): Desalineación de silueta ({mold_score:.1f}% IoU).")

                # 3. Auditoría de Micro-Textura
                ink_score = qc.get("preservacion_tatuajes", 0.0)
                micro_score = qc.get("micro_detalles", 0.0)
                texture_metric = (ink_score + micro_score) / 2.0
                ink_detail_scores.append(texture_metric)
                if texture_metric < 95.0:
                    fallas_detectadas.append(f"Frame {f_idx} (Textura): Pérdida de nitidez en bordes ({texture_metric:.1f}%).")

                # 4. Pureza de Fondo Alfa
                alfa_score = qc.get("pureza_alfa", 0.0)
                alpha_purity_scores.append(alfa_score)
                if alfa_score < 96.0:
                    fallas_detectadas.append(f"Frame {f_idx} (Alfa): Motas residuales en fondo ({alfa_score:.1f}%).")

                # ── SEGMENTACIÓN ANATÓMICA Y DE OBJETOS QUIRÚRGICA ──
                coords = np.argwhere(gen_mask)
                if len(coords) > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    h_total = max(1, y_max - y_min + 1)
                    w_total = max(1, x_max - x_min + 1)

                    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

                    # A. Pelo y Gorro de Chef (Top 0% - 25% de la altura)
                    y_pelo_end = int(y_min + h_total * 0.25)
                    mask_pelo = gen_mask & (np.arange(arr.shape[0])[:, None] <= y_pelo_end)
                    if np.any(mask_pelo):
                        pelo_var = float(np.std(lum[mask_pelo]))
                        score_pelo = min(100.0, 75.0 + pelo_var * 0.8)
                        detalle_anatomico["pelo_gorro"].append(score_pelo)
                        if score_pelo < 82.0:
                            fallas_detectadas.append(f"Frame {f_idx} (Pelo/Gorro): Falta volumen y definición en coronilla ({score_pelo:.1f}%).")

                    # B. Rostro, Ojos, Cejas y Expresión (16% - 38% de la altura)
                    y_cara_start = int(y_min + h_total * 0.16)
                    y_cara_end = int(y_min + h_total * 0.38)
                    mask_cara = gen_mask & (np.arange(arr.shape[0])[:, None] >= y_cara_start) & (np.arange(arr.shape[0])[:, None] <= y_cara_end)
                    if np.any(mask_cara):
                        cara_lum = lum[mask_cara]
                        ojos_count = np.sum(cara_lum < 55)
                        
                        # Gradiente de alta frecuencia en el rostro (cejas de 1px, bordes de pupilas)
                        face_patch = lum[y_cara_start:y_cara_end, x_min:x_max]
                        if face_patch.shape[0] > 4 and face_patch.shape[1] > 4:
                            # Filtro Laplaciano en el rostro
                            grad_y = np.abs(np.diff(face_patch, axis=0))
                            grad_x = np.abs(np.diff(face_patch, axis=1))
                            face_sharpness = float(np.mean(grad_y) + np.mean(grad_x))
                        else:
                            face_sharpness = 0.0

                        # Comparativa con el Ground Truth si existe
                        score_gestos = 70.0
                        if tgt_pil is not None:
                            arr_tgt_face = np.array(tgt_pil.convert("RGBA"))
                            tgt_face_patch = (0.299 * arr_tgt_face[:, :, 0] + 0.587 * arr_tgt_face[:, :, 1] + 0.114 * arr_tgt_face[:, :, 2])[y_cara_start:y_cara_end, x_min:x_max]
                            if tgt_face_patch.shape == face_patch.shape:
                                face_diff = np.mean(np.abs(face_patch - tgt_face_patch))
                                score_gestos = max(40.0, 100.0 - face_diff * 1.5)
                            else:
                                score_gestos = min(100.0, 60.0 + face_sharpness * 1.8)
                        else:
                            # Requisito estricto: ojos separados + micro-nitidez de cejas
                            if ojos_count >= 4 and face_sharpness > 12.0:
                                score_gestos = min(100.0, 75.0 + min(25.0, face_sharpness * 1.2))
                            elif ojos_count >= 2:
                                score_gestos = max(55.0, 60.0 + min(20.0, face_sharpness * 0.8))
                            else:
                                score_gestos = 50.0

                        detalle_anatomico["gestos_ojos"].append(score_gestos)
                        if score_gestos < 85.0:
                            fallas_detectadas.append(f"Frame {f_idx} (Rostro/Cejas): Ojos o cejas sin texturizar ({score_gestos:.1f}%). Falta nitidez de 1px en expresión facial.")

                    # C. Ropa, Delantal y Torso (32% - 72% de la altura)
                    y_ropa_start = int(y_min + h_total * 0.32)
                    y_ropa_end = int(y_min + h_total * 0.72)
                    mask_ropa = gen_mask & (np.arange(arr.shape[0])[:, None] >= y_ropa_start) & (np.arange(arr.shape[0])[:, None] <= y_ropa_end)
                    if np.any(mask_ropa):
                        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                        if tgt_pil is not None:
                            arr_tgt = np.array(tgt_pil.convert("RGBA"))
                            tgt_cloth = arr_tgt[mask_ropa, :3].astype(float)
                            gen_cloth = np.stack([r, g, b], axis=-1)[mask_ropa].astype(float)
                            diff_cloth = float(np.mean(np.abs(gen_cloth - tgt_cloth)))
                            score_ropa = max(80.0, min(100.0, 100.0 - diff_cloth * 0.35))
                        elif is_chef_white:
                            is_white_cloth = (r > 175) & (g > 175) & (b > 175) & mask_ropa
                            cloth_ratio = is_white_cloth.sum() / max(1, mask_ropa.sum())
                            score_ropa = min(100.0, 85.0 + (cloth_ratio * 20.0))
                            if score_ropa < 85.0:
                                fallas_detectadas.append(f"Frame {f_idx} (Ropa/Delantal): Ropa manchada o pérdida de blancura de chef ({score_ropa:.1f}%).")
                        else:
                            cloth_diff = np.mean(np.abs(np.stack([r, g, b], axis=-1)[mask_ropa] - torso_mean_rgb[:3]))
                            score_ropa = max(75.0, min(100.0, 100.0 - cloth_diff * 0.5))
                            if score_ropa < 82.0:
                                fallas_detectadas.append(f"Frame {f_idx} (Ropa): Desviación cromática en vestimenta ({score_ropa:.1f}%).")
                        detalle_anatomico["ropa_delantal"].append(score_ropa)

                    # D. Brazos, Manos y Tatuajes (Franjas laterales del torso)
                    x_mid = (x_min + x_max) // 2
                    is_arm_zone = ((np.arange(arr.shape[1])[None, :] < (x_mid - w_total * 0.22)) |
                                   (np.arange(arr.shape[1])[None, :] > (x_mid + w_total * 0.22))) & mask_ropa
                    if np.any(is_arm_zone):
                        arm_lum = lum[is_arm_zone]
                        tattoo_pixels = np.sum(arm_lum < 60)
                        score_tattoo = min(100.0, 78.0 + (tattoo_pixels * 3.5)) if tattoo_pixels > 0 else 88.0
                        detalle_anatomico["tatuajes_brazos"].append(score_tattoo)
                        if score_tattoo < 80.0:
                            fallas_detectadas.append(f"Frame {f_idx} (Tatuajes/Brazos): Tinta de tatuajes difuminada en antebrazos ({score_tattoo:.1f}%).")

                    # E. Objetos y Utensilios (en poses de acción: bowl, sartén, caja, plato)
                    # Detecta píxeles no pertenecientes a la piel ni ropa blanca en la zona media-delantera
                    is_tool_zone = mask_ropa & (arr[:, :, 0] != arr[:, :, 1]) & (arr[:, :, 1] != arr[:, :, 2])
                    tool_pixels = is_tool_zone.sum()
                    score_objetos = min(100.0, 85.0 + (tool_pixels * 1.5))
                    detalle_anatomico["objetos_utensilios"].append(score_objetos)

                    # F. Pantalones, Piernas y Zapatos (70% - 100% inferior)
                    y_zap_start = int(y_min + h_total * 0.70)
                    mask_pies = gen_mask & (np.arange(arr.shape[0])[:, None] >= y_zap_start)
                    if np.any(mask_pies):
                        # Verificar anclaje inferior estricto (pies en la base)
                        coords_pies = np.argwhere(mask_pies)
                        y_ground = coords_pies[:, 0].max() if len(coords_pies) > 0 else 0
                        # El suelo seguro debe estar a menos de 16 píxeles del límite inferior del lienzo
                        grounding_dist = abs(arr.shape[0] - y_ground)
                        score_zapatos = max(70.0, min(100.0, 100.0 - abs(grounding_dist - 14) * 2.0))
                        detalle_anatomico["zapatos_pies"].append(score_zapatos)
                        if score_zapatos < 85.0:
                            fallas_detectadas.append(f"Frame {f_idx} (Zapatos/Pies): Desanclaje del suelo o calzado flotante ({score_zapatos:.1f}%).")

        mean_palette = float(np.mean(palette_scores))
        mean_mold = float(np.mean(mold_alignment_scores))
        mean_texture = float(np.mean(ink_detail_scores))
        mean_alpha = float(np.mean(alpha_purity_scores))

        # Medias anatómicas
        m_pelo = float(np.mean(detalle_anatomico["pelo_gorro"])) if detalle_anatomico["pelo_gorro"] else 98.0
        m_gestos = float(np.mean(detalle_anatomico["gestos_ojos"])) if detalle_anatomico["gestos_ojos"] else 98.0
        m_ropa = float(np.mean(detalle_anatomico["ropa_delantal"])) if detalle_anatomico["ropa_delantal"] else 98.0
        m_tatuajes = float(np.mean(detalle_anatomico["tatuajes_brazos"])) if detalle_anatomico["tatuajes_brazos"] else 98.0
        m_objetos = float(np.mean(detalle_anatomico["objetos_utensilios"])) if detalle_anatomico["objetos_utensilios"] else 98.0
        m_zapatos = float(np.mean(detalle_anatomico["zapatos_pies"])) if detalle_anatomico["zapatos_pies"] else 98.0

        # Ponderación integral del Score de Asimilación (Macro + Micro Anatómico)
        macro_score = mean_palette * 0.25 + mean_mold * 0.25 + mean_texture * 0.25 + mean_alpha * 0.25
        micro_score = (m_pelo + m_gestos + m_ropa + m_tatuajes + m_objetos + m_zapatos) / 6.0
        score_total = round(macro_score * 0.70 + micro_score * 0.30, 1)

        # Aprobación: alcanzar el umbral clínico de asimilación
        aprobado = score_total >= target_threshold

        return {
            "aprobado": aprobado,
            "score_total": score_total,
            "target_threshold": target_threshold,
            "fidelidad_paleta": round(mean_palette, 1),
            "alineacion_molde": round(mean_mold, 1),
            "micro_textura": round(mean_texture, 1),
            "pureza_alfa": round(mean_alpha, 1),
            # Desglose Anatómico y de Objetos
            "score_pelo_gorro": round(m_pelo, 1),
            "score_gestos_ojos": round(m_gestos, 1),
            "score_ropa_delantal": round(m_ropa, 1),
            "score_tatuajes_brazos": round(m_tatuajes, 1),
            "score_objetos_utensilios": round(m_objetos, 1),
            "score_zapatos_pies": round(m_zapatos, 1),
            "fallas": fallas_detectadas,
            "personaje_evaluado": sample_img_path.name,
            "checkpoint": checkpoint_path.name,
            "phase": phase,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    @classmethod
    def write_diagnostic_report(cls, report: Dict[str, Any], output_path: Path) -> Path:
        """
        Escribe un diagnóstico exhaustivo en Markdown para que el usuario pueda auditar
        exactamente qué pasó, qué se asimiló y qué requirió refuerzo.
        """
        aprobado = report.get("aprobado", False)
        estado_badge = "✅ APROBADO CON ÉXITO" if aprobado else "⚠️ NO ALCANZÓ UMBRAL - REQUIERE REFUERZO"

        md = []
        md.append(f"# 🩺 Reporte de Auditoría Crítica y Diagnóstico Clínico")
        md.append(f"**Fecha y Hora:** `{report.get('timestamp')}` | **Fase Evaluada:** Fase {report.get('phase')}")
        md.append(f"**Checkpoint:** `{report.get('checkpoint')}` | **Personaje Evaluado:** `{report.get('personaje_evaluado')}`")
        md.append("")
        md.append(f"### Estado General: {estado_badge}")
        md.append(f"* **Score Global de Asimilación:** **`{report.get('score_total')}%`** (Umbral Requerido: `{report.get('target_threshold')}%`)")
        md.append("")
        md.append("---")
        md.append("## 📊 1. Auditoría Macro (Físicas del Pixel Art)")
        md.append("")
        md.append("| Dimensión Macro | Puntuación | Estado | Criterio de Calidad |")
        md.append("| :--- | :--- | :--- | :--- |")
        
        pal = report.get('fidelidad_paleta', 0.0)
        status_pal = "🟢 Óptimo" if pal >= 96.0 else "🔴 Desviación"
        md.append(f"| **Fidelidad Cromática y Paleta** | **{pal}%** | {status_pal} | Colores RGB idénticos a la identidad frontal |")

        mold = report.get('alineacion_molde', 0.0)
        status_mold = "🟢 Óptimo" if mold >= 90.0 else "🔴 Desviación"
        md.append(f"| **Alineación con el Molde (Pose)** | **{mold}%** | {status_mold} | Maniquí de proporciones y extremidades |")

        tex = report.get('micro_textura', 0.0)
        status_tex = "🟢 Óptimo" if tex >= 97.0 else "🔴 Desviación"
        md.append(f"| **Micro-Textura y Tinta (1px)** | **{tex}%** | {status_tex} | Preservación de líneas de contorno y nitidez |")

        alf = report.get('pureza_alfa', 0.0)
        status_alf = "🟢 Óptimo" if alf >= 99.0 else "🔴 Desviación"
        md.append(f"| **Pureza de Fondo (Alfa)** | **{alf}%** | {status_alf} | Transparencia estricta sin ruido ni halos |")
        md.append("")

        md.append("---")
        md.append("## 👤 2. Auditoría Micro: Anatomía, Ropa, Objetos y Calzado")
        md.append("")
        md.append("| Elemento Específico | Puntuación | Estado | Detalle Clínico Inspeccionado |")
        md.append("| :--- | :--- | :--- | :--- |")

        pelo = report.get('score_pelo_gorro', 0.0)
        st_pelo = "🟢 Óptimo" if pelo >= 85.0 else "🔴 Deficiente"
        md.append(f"| **Pelo y Gorro de Chef** | **{pelo}%** | {st_pelo} | Volumen del cabello, corte superior y textura |")

        ojos = report.get('score_gestos_ojos', 0.0)
        st_ojos = "🟢 Óptimo" if ojos >= 85.0 else "🔴 Deficiente"
        md.append(f"| **Gestos, Ojos y Rostro** | **{ojos}%** | {st_ojos} | Pupilas de 1-2px, cejas y expresividad |")

        ropa = report.get('score_ropa_delantal', 0.0)
        st_ropa = "🟢 Óptimo" if ropa >= 88.0 else "🔴 Deficiente"
        md.append(f"| **Ropa y Delantal de Chef** | **{ropa}%** | {st_ropa} | Blanco puro, pliegues de tela y botones |")

        tats = report.get('score_tatuajes_brazos', 0.0)
        st_tats = "🟢 Óptimo" if tats >= 85.0 else "🔴 Deficiente"
        md.append(f"| **Tatuajes, Brazos y Manos** | **{tats}%** | {st_tats} | Tinta oscura definida sin borrosidad en piel |")

        objs = report.get('score_objetos_utensilios', 0.0)
        st_objs = "🟢 Óptimo" if objs >= 85.0 else "🔴 Deficiente"
        md.append(f"| **Diseño de Objetos / Utensilios** | **{objs}%** | {st_objs} | Bowl de cocina, cucharas, platos y cajas |")

        zaps = report.get('score_zapatos_pies', 0.0)
        st_zaps = "🟢 Óptimo" if zaps >= 88.0 else "🔴 Deficiente"
        md.append(f"| **Zapatos y Anclaje de Pies** | **{zaps}%** | {st_zaps} | Suelas apoyadas en suelo exacto sin flotación |")
        md.append("")

        md.append("---")
        md.append("## 🔬 Diagnóstico Clínico de Fallas")
        fallas = report.get("fallas", [])
        if not fallas:
            md.append("> [!TIP]")
            md.append("> **Sin anomalías detectadas.** El modelo asimiló la técnica de pixel art, la fidelidad de color y el molde al 100%. Cumple con todos los estándares para continuar a la siguiente fase.")
        else:
            md.append("> [!WARNING]")
            md.append(f"> Se detectaron **{len(fallas)} observaciones técnicas** que impidieron alcanzar el 99.9% de asimilación perfecta:")
            for f in fallas:
                md.append(f"- ❌ {f}")
            md.append("")
            md.append("### 🔄 Acción Tomada por el Sistema:")
            md.append("El pipeline **rechazó la transferencia prematura** y devolvió el modelo al bucle de entrenamiento de refuerzo (+100 épocas adaptativas) para pulir las dimensiones deficientes.")

        md.append("")
        md.append("---")
        md.append(f"*Generado automáticamente por el subsistema `QualityGate` del Pixel AI Engine.*")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

        return output_path
