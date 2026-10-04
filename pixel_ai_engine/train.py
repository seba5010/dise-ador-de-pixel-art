"""
Pixel AI Engine - Training Script (train.py)
Entrenamiento local con aceleración CUDA, soporte para 4 GB de VRAM (NVIDIA RTX 3050 Ti),
precisión mixta float16 (AMP), optimizador Adam con Cosine LR Decay y guardado de checkpoints.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from typing import Optional, Dict, Any, Tuple
from .config import (
    PROJECT_ROOT,
    CHECKPOINT_DIR,
    SAMPLES_DIR,
    DEVICE,
    USE_AMP,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE_G,
    LEARNING_RATE_D,
    BETA1,
    BETA2,
    WEIGHT_DECAY,
    SAVE_EVERY_EPOCHS,
    SAMPLE_EVERY_EPOCHS,
    GRID_ROWS,
    GRID_COLS,
    TOTAL_FRAMES,
    CELL_WIDTH,
    CELL_HEIGHT,
    get_phase_config
)
from .dataset import PixelArtDataset, get_dataloader, TemplateManager, pad_to_square, unpad_from_square, place_in_cell
from .models import PixelArtUNetGenerator, PixelArtPatchDiscriminator, PixelArtLoss
from .enhancer import PixelArtEnhancer


def get_gpu_temperature() -> Optional[int]:
    """Lee la temperatura en tiempo real de la GPU NVIDIA en °C."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            encoding="utf-8",
            timeout=1
        )
        return int(out.strip())
    except Exception:
        return None


def get_cpu_temperature() -> Optional[int]:
    """Lee la temperatura de la CPU / Placa Madre en °C vía WMI."""
    try:
        cmd = ["powershell", "-NoProfile", "-Command", "(Get-WmiObject MSAcpi_ThermalZoneTemperature -Namespace root/wmi -ErrorAction SilentlyContinue).CurrentTemperature"]
        out = subprocess.check_output(cmd, encoding="utf-8", timeout=1.5).strip()
        if out:
            temps = [int(l.strip()) for l in out.splitlines() if l.strip().isdigit()]
            if temps:
                return int(round((max(temps) - 2732) / 10.0))
    except Exception:
        pass
    return None


def manage_adaptive_thermal_throttle(temp_gpu_limit: int = 80,
                                     temp_cpu_limit: int = 85,
                                     temp_gpu_cooldown: int = 68,
                                     temp_cpu_cooldown: int = 76) -> Tuple[Optional[int], Optional[int]]:
    """
    Termostato Inteligente Dual para Laptops (Protección Integral de GPU, CPU y Placa Madre):
    - Monitorea simultáneamente la GPU (NVIDIA) y la CPU/Placa (WMI).
    - Si la GPU >= 80°C o la CPU >= 85°C: PAUSA TOTAL del entrenamiento hasta que ambos
      componentes bajen a niveles seguros (GPU <= 68°C y CPU <= 76°C).
    - Micro-respirador preventivo: si GPU >= 75°C o CPU >= 80°C, toma 4 segundos de pausa
      al final de cada época para permitir que los ventiladores disipen el calor de los heatpipes.
    - Pausa base de 1 segundo en cada época para desestresar los VRMs de la placa.
    """
    if not torch.cuda.is_available():
        return None, None

    torch.cuda.empty_cache()
    g_temp = get_gpu_temperature()
    c_temp = get_cpu_temperature()

    gpu_over = (g_temp is not None and g_temp >= temp_gpu_limit)
    cpu_over = (c_temp is not None and c_temp >= temp_cpu_limit)

    # 1. Alerta Crítica Dual: Pausa total hasta enfriar ambos procesadores
    if gpu_over or cpu_over:
        motivo = []
        if gpu_over: motivo.append(f"GPU {g_temp}°C >= {temp_gpu_limit}°C")
        if cpu_over: motivo.append(f"CPU {c_temp}°C >= {temp_cpu_limit}°C")
        print(f"\n  [🔥 PROTECCIÓN TÉRMICA INTEGRAL: {', '.join(motivo)}] Pausando entrenamiento para enfriar...", flush=True)
        
        while True:
            time.sleep(3.0)
            g_cur = get_gpu_temperature()
            c_cur = get_cpu_temperature()
            g_temp = g_cur if g_cur is not None else g_temp
            c_temp = c_cur if c_cur is not None else c_temp
            
            g_ok = (g_temp is None or g_temp <= temp_gpu_cooldown)
            c_ok = (c_temp is None or c_temp <= temp_cpu_cooldown)
            
            print(f"     -> Enfriando... GPU: {g_temp or '?'}°C (meta <= {temp_gpu_cooldown}°C) | CPU: {c_temp or '?'}°C (meta <= {temp_cpu_cooldown}°C)...", end="\r", flush=True)
            if g_ok and c_ok:
                break
        print(f"\n  [❄️ TEMPERATURAS SEGURAS ALCANZADAS: GPU {g_temp}°C | CPU {c_temp}°C] Reanudando...\n", flush=True)

    # 2. Respirador Preventivo si alguno se acerca a zona tibia
    elif (g_temp is not None and g_temp >= 75) or (c_temp is not None and c_temp >= 80):
        print(f"     [🌡️ Micro-pausa preventiva de 4s para disipar calor: GPU {g_temp or '?'}°C | CPU {c_temp or '?'}°C]", end="\r", flush=True)
        time.sleep(4.0)
    else:
        time.sleep(1.0)

    return g_temp, c_temp


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convierte un tensor RGB [-1, 1] o RGBA [-1, 1] a PIL Image.
    """
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
    if arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    return Image.fromarray(arr, mode="RGBA")


def generate_sample_preview(generator: nn.Module,
                            template_mgr: TemplateManager,
                            sample_front: torch.Tensor,
                            epoch: int,
                            output_path: Path,
                            phase_cfg: Optional[Dict] = None,
                            sample_target: Optional[torch.Tensor] = None,
                            dataset: Optional[Any] = None) -> Dict[str, float]:
    """
    Genera una vista previa visual adaptada al formato de la fase activa
    (16x4 para Fase 1, o 8x12 para Fase 2) y calcula métricas de calidad quirúrgica.
    Además, genera una comparativa dual de alta resolución (latest_detail_comparison.png)
    con VISTA FRONTAL (Frame 0) y VISTA PERFIL LATERAL (Caminata de perfil & zoom HD tatuajes)
    para verificar que los micro-detalles y tatuajes superen la calidad de las muestras originales.
    """
    cols = phase_cfg["grid_cols"] if phase_cfg else GRID_COLS
    rows = phase_cfg["grid_rows"] if phase_cfg else GRID_ROWS
    total_frames = phase_cfg["total_frames"] if phase_cfg else TOTAL_FRAMES
    cell_w = phase_cfg["cell_w"] if phase_cfg else CELL_WIDTH
    cell_h = phase_cfg["cell_h"] if phase_cfg else CELL_HEIGHT
    
    # Extraer imagen frontal PIL y paleta canónica UNA SOLA VEZ para toda la cuadrícula
    front_pil = tensor_to_pil(sample_front)
    canonical_pal = PixelArtEnhancer.extract_palette(front_pil, max_colors=40)
    
    generator.eval()
    preview_cells = []
    first_enhanced_pil = None
    
    with torch.no_grad():
        for frame_idx in range(total_frames):
            pose_tensor = template_mgr.get_frame_tensor(frame_idx).to(DEVICE)
            cond = torch.cat([sample_front, pose_tensor], dim=0).unsqueeze(0)  # (1, 6, H, W)
            
            with autocast(enabled=USE_AMP):
                pred_rgba = generator(cond).squeeze(0)  # (4, H, W)
                
            # Convertir tensor a PIL RGBA
            arr = pred_rgba.detach().cpu().permute(1, 2, 0).numpy()
            arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
            pred_pil = Image.fromarray(arr, mode="RGBA")
            
            # Aplicar pulido quirúrgico (borde nítido, eliminación de ruido, refuerzo de tatuajes)
            enhanced_pil = PixelArtEnhancer.enhance_frame(
                pred_pil,
                palette=canonical_pal,
                snap_palette=True,
                remove_noise=True,
                binarize=True,
                sharpen_tattoos=True
            )
            
            if frame_idx == 0:
                first_enhanced_pil = enhanced_pil.copy()
            
            # Extraer figura y reposicionar en celda de preview
            cell = place_in_cell(enhanced_pil, cell_w=cell_w, cell_h=cell_h)
            preview_cells.append(cell)
            
    # Ensamblar cuadrícula con fondo neutro claro
    grid_img = Image.new("RGBA", (cell_w * cols, cell_h * rows), (215, 218, 226, 255))
    
    for idx, cell in enumerate(preview_cells):
        r = idx // cols
        c = idx % cols
        grid_img.paste(cell, (c * cell_w, r * cell_h), cell)
        
    try:
        grid_img.save(output_path)
    except Exception as e:
        print(f"[Aviso] No se pudo guardar {output_path.name}: {e}")
        
    try:
        # Guardar copia en latest_preview.png para el monitor web en vivo
        latest_path = output_path.parent / "latest_preview.png"
        grid_img.save(latest_path)
    except Exception:
        pass

    # ── GENERAR COMPARATIVA QUIRÚRGICA DE DETALLE (latest_detail_comparison.png) ──
    # Arquitectura Dual: Fila 1 (Frontal) + Fila 2 (Perfil Lateral con Zoom HD Tatuajes)
    p4_raw = None
    p3 = None
    try:
        w_panel = 256
        h_panel = 256
        header_h = 32
        footer_h = 28
        total_w = w_panel * 5
        total_h = header_h * 2 + h_panel * 2 + footer_h

        comp_img = Image.new("RGBA", (total_w, total_h), (15, 18, 25, 255))
        draw = ImageDraw.Draw(comp_img)

        try:
            font_title = ImageFont.truetype("arial.ttf", 15)
            font_sub = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font_title = None
            font_sub = None

        # ------------------------------------------------------------------
        # [FILA 1] VISTA FRONTAL (Frame 0)
        # ------------------------------------------------------------------
        p1_front = front_pil.convert("RGBA").resize((w_panel, h_panel), Image.Resampling.NEAREST)
        pose_t0 = template_mgr.get_frame_tensor(0)
        p2_front = tensor_to_pil(pose_t0).convert("RGBA").resize((w_panel, h_panel), Image.Resampling.NEAREST)
        
        if sample_target is not None:
            p3_front = tensor_to_pil(sample_target).convert("RGBA").resize((w_panel, h_panel), Image.Resampling.NEAREST)
        else:
            p3_front = Image.new("RGBA", (w_panel, h_panel), (25, 30, 42, 255))
        p3 = p3_front
            
        with torch.no_grad():
            cond0 = torch.cat([sample_front, pose_t0.to(DEVICE)], dim=0).unsqueeze(0)
            with autocast(enabled=USE_AMP):
                raw_pred0 = generator(cond0).squeeze(0)
            p4_front = tensor_to_pil(raw_pred0).convert("RGBA").resize((w_panel, h_panel), Image.Resampling.NEAREST)
            
        p5_front = first_enhanced_pil if first_enhanced_pil else PixelArtEnhancer.enhance_frame(
            p4_front, palette=canonical_pal, snap_palette=True, remove_noise=True, binarize=True, sharpen_tattoos=True
        )
        p5_front = p5_front.convert("RGBA").resize((w_panel, h_panel), Image.Resampling.NEAREST)
        p4_raw = p4_front

        # ------------------------------------------------------------------
        # [FILA 2] VISTA PERFIL LATERAL (Frame de caminata de lado)
        # ------------------------------------------------------------------
        profile_idx = 16 if cols == 4 else 48
        if profile_idx >= total_frames:
            profile_idx = min(4 * cols, total_frames - 1)
        pose_prof = template_mgr.get_frame_tensor(profile_idx)
        p2_prof = tensor_to_pil(pose_prof).convert("RGBA").resize((w_panel, h_panel), Image.Resampling.NEAREST)

        # Zoom quirúrgico de alta definición en tatuajes y torso de Alex
        try:
            p1_prof = front_pil.crop((50, 45, 205, 175)).convert("RGBA").resize((w_panel, h_panel), Image.Resampling.NEAREST)
        except Exception:
            p1_prof = p1_front.copy()

        # Buscar target ground truth del frame de perfil lateral
        target_prof_tensor = None
        if dataset is not None and hasattr(dataset, "samples") and len(dataset.samples) > 0:
            first_char = dataset.samples[0].get("char_id")
            for s in dataset.samples:
                if s.get("char_id") == first_char and s.get("frame_idx") == profile_idx:
                    target_prof_tensor = s.get("target_tensor")
                    break

        if target_prof_tensor is not None:
            p3_prof = tensor_to_pil(target_prof_tensor).convert("RGBA").resize((w_panel, h_panel), Image.Resampling.NEAREST)
        else:
            p3_prof = Image.new("RGBA", (w_panel, h_panel), (25, 30, 42, 255))

        with torch.no_grad():
            cond_prof = torch.cat([sample_front, pose_prof.to(DEVICE)], dim=0).unsqueeze(0)
            with autocast(enabled=USE_AMP):
                raw_pred_prof = generator(cond_prof).squeeze(0)
            p4_prof = tensor_to_pil(raw_pred_prof).convert("RGBA").resize((w_panel, h_panel), Image.Resampling.NEAREST)

        p5_prof = PixelArtEnhancer.enhance_frame(
            p4_prof, palette=canonical_pal, snap_palette=True, remove_noise=True, binarize=True, sharpen_tattoos=True
        )
        p5_prof = p5_prof.convert("RGBA").resize((w_panel, h_panel), Image.Resampling.NEAREST)

        # ------------------------------------------------------------------
        # ENSAMBLAR CANVAS Y ROTULAR SECCIONES
        # ------------------------------------------------------------------
        y_f1 = header_h
        comp_img.paste(p1_front, (0, y_f1), p1_front)
        comp_img.paste(p2_front, (w_panel, y_f1), p2_front)
        comp_img.paste(p3_front, (w_panel * 2, y_f1), p3_front)
        comp_img.paste(p4_front, (w_panel * 3, y_f1), p4_front)
        comp_img.paste(p5_front, (w_panel * 4, y_f1), p5_front)

        y_f2 = header_h * 2 + h_panel
        comp_img.paste(p1_prof, (0, y_f2), p1_prof)
        comp_img.paste(p2_prof, (w_panel, y_f2), p2_prof)
        comp_img.paste(p3_prof, (w_panel * 2, y_f2), p3_prof)
        comp_img.paste(p4_prof, (w_panel * 3, y_f2), p4_prof)
        comp_img.paste(p5_prof, (w_panel * 4, y_f2), p5_prof)

        ACCENT_BLUE = (100, 160, 255, 255)
        ACCENT_GREEN = (90, 220, 140, 255)
        TEXT_DIM = (180, 190, 210, 255)
        LINE_COLOR = (42, 48, 62, 255)

        # Header Fila 1
        draw.rectangle([(0, 0), (total_w, header_h)], fill=(22, 27, 38, 255))
        draw.text((16, 7), "[1] AUDITORIA FRONTAL (Frame 00 - Reposo Frente a Frente)", fill=ACCENT_BLUE, font=font_title)

        # Header Fila 2
        y_h2 = header_h + h_panel
        draw.rectangle([(0, y_h2), (total_w, y_h2 + header_h)], fill=(22, 27, 38, 255))
        draw.text((16, y_h2 + 7), f"[2] AUDITORIA PERFIL LATERAL (Frame {profile_idx:02d} - Caminata Lateral, Silueta y Tatuajes)", fill=ACCENT_GREEN, font=font_title)

        # Footer
        y_foot = total_h - footer_h
        draw.rectangle([(0, y_foot), (total_w, total_h)], fill=(22, 27, 38, 255))
        col_names = ["1. Referencia HD (Zoom)", "2. Maniqui Pose 3D", "3. Sprite Objetivo", "4. Red Neuronal Cruda", "5. Pulido Quirurgico 1:1"]
        for i, name in enumerate(col_names):
            draw.text((i * w_panel + 16, y_foot + 6), name, fill=TEXT_DIM, font=font_sub)

        for i in range(1, 5):
            x = i * w_panel
            draw.line([(x, y_f1), (x, y_f1 + h_panel)], fill=LINE_COLOR, width=1)
            draw.line([(x, y_f2), (x, y_f2 + h_panel)], fill=LINE_COLOR, width=1)

        comp_path = output_path.parent / "latest_detail_comparison.png"
        comp_img.save(comp_path)
    except Exception as e:
        print(f"[Aviso] No se pudo generar latest_detail_comparison.png: {e}")
        
    # Auditoría quirúrgica directa sobre la red neuronal pura (SIN MAQUILLAJE ni filtros cosméticos)
    eval_target = p4_raw if p4_raw is not None else (first_enhanced_pil if first_enhanced_pil else preview_cells[0])
    target_gt_img = p3 if ('p3' in locals() and p3 is not None and sample_target is not None) else None
    qc = PixelArtEnhancer.analyze_quality(eval_target, palette=canonical_pal, target_img=target_gt_img)
    if first_enhanced_pil is not None:
        qc_enh = PixelArtEnhancer.analyze_quality(first_enhanced_pil, palette=canonical_pal, target_img=target_gt_img)
        qc["enhanced_total"] = qc_enh["score_total"]
    qc["quality_guide"] = PixelArtEnhancer.build_quality_guide(qc)
    generator.train()
    return qc


def train(epochs: int = EPOCHS,
          batch_size: int = BATCH_SIZE,
          resume: bool = False,
          lr_g: float = LEARNING_RATE_G,
          lr_d: float = LEARNING_RATE_D,
          infinite: bool = False,
          phase: str = "2",
          transfer_from: Optional[str] = None):
    """
    Ciclo de entrenamiento principal con soporte de 2 Fases (Transfer Learning).
    Fase 1: Plantilla Chica (16x4 = 64 frames) -> aprende rostros, ropa y caminatas.
    Fase 2: Plantilla Grande (8x12 = 96 frames) -> transfiere pesos y aprende poses detalladas.
    """
    phase_cfg = get_phase_config(phase)
    
    print("=" * 70)
    print(f"  PIXEL ART AI ENGINE - {phase_cfg['phase_name'].upper()}")
    print(f"  Plantilla: {phase_cfg['template_path'].name} ({phase_cfg['grid_cols']} cols x {phase_cfg['grid_rows']} filas = {phase_cfg['total_frames']} frames)")
    print(f"  Dispositivo: {DEVICE}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU: {gpu_name} ({vram_gb:.2f} GB VRAM detectada)")
        print(f"  Aceleración AMP (Mixed Precision FP16): {'ACTIVA' if USE_AMP else 'DESACTIVADA'}")
    print("=" * 70)

    # 1. Dataset y DataLoaders
    dataloader = get_dataloader(batch_size=batch_size, phase_cfg=phase_cfg)
    dataset = dataloader.dataset
    template_mgr = dataset.template_mgr
    
    # 2. Inicializar Modelos
    generator = PixelArtUNetGenerator().to(DEVICE)
    discriminator = PixelArtPatchDiscriminator().to(DEVICE)
    
    # 3. Transfer Learning (Transferencia de Fase 1 a Fase 2)
    if not resume:
        tf_path = None
        if transfer_from and Path(transfer_from).exists():
            tf_path = Path(transfer_from)
        elif str(phase_cfg["phase"]) == "2" and (CHECKPOINT_DIR / "base_generator_16x4.pt").exists():
            tf_path = CHECKPOINT_DIR / "base_generator_16x4.pt"
            
        if tf_path:
            print(f"\n[TRANSFER LEARNING] Cargando experiencia de Fase 1 desde: {tf_path.name}")
            saved = torch.load(tf_path, map_location=DEVICE)
            state_dict = saved.get("generator", saved) if isinstance(saved, dict) else saved
            generator.load_state_dict(state_dict)
            print("[TRANSFER LEARNING] ¡Pesos base transferidos exitosamente a la Fase 2!\n")
    
    # 4. Criterios y Optimizadores
    criterion = PixelArtLoss().to(DEVICE)
    
    optimizer_g = Adam(generator.parameters(), lr=lr_g, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    optimizer_d = Adam(discriminator.parameters(), lr=lr_d, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=epochs, eta_min=2e-5)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=epochs, eta_min=1e-5)
    
    scaler = GradScaler(enabled=USE_AMP)
    
    start_epoch = 1
    best_loss = float("inf")
    
    # Modo infinito: correr hasta Ctrl+C
    infinite_mode = infinite or (epochs == 0)
    total_epochs = 999_999 if infinite_mode else epochs
    
    # Archivo de estado para el monitor web en vivo
    status_path = PROJECT_ROOT / "training_status.json"
    training_start_time = time.time()
    
    # Checkpoints dinámicos según la fase
    latest_ckpt_path = phase_cfg["ckpt_latest"]
    best_gen_path    = phase_cfg["ckpt_best"]
    
    # Reanudar desde checkpoint si se solicita
    if resume and latest_ckpt_path.exists():
        print(f"[Checkpoint] Reanudando entrenamiento desde: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=DEVICE)
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        optimizer_d.load_state_dict(ckpt["optimizer_d"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        
        # Reajustar Learning Rate activo para que el modelo no quede congelado en 1e-6
        for param_group in optimizer_g.param_groups:
            param_group['lr'] = lr_g
        for param_group in optimizer_d.param_groups:
            param_group['lr'] = lr_d
            
        # Si epochs indicado es menor o igual a start_epoch, interpretarlo como épocas adicionales (+epochs)
        if total_epochs < start_epoch:
            total_epochs = start_epoch + epochs - 1
            
        remaining_epochs = max(10, total_epochs - start_epoch + 1)
        scheduler_g = CosineAnnealingLR(optimizer_g, T_max=remaining_epochs, eta_min=2e-5)
        scheduler_d = CosineAnnealingLR(optimizer_d, T_max=remaining_epochs, eta_min=1e-5)
        print(f"[Checkpoint] Reanudado en época {start_epoch} (meta: época {total_epochs}) con LR reactivado a {lr_g:.2e}.")
        
    # Muestra de referencia para previews periódicas (primer personaje del dataset)
    sample_front = dataset.samples[0]["front_tensor"].to(DEVICE)
    sample_target = dataset.samples[0]["target_tensor"].to(DEVICE) if "target_tensor" in dataset.samples[0] else None
    last_qc = {
        "score_total": 0.0,
        "pureza_alfa": 0.0,
        "nitidez_bordes": 0.0,
        "ruido_huerfano": 0.0,
        "fidelidad_paleta": 0.0,
        "micro_detalles": 0.0,
        "preservacion_tatuajes": 0.0
    }
    
    print("\nIniciando ciclo de entrenamiento...")
    if infinite_mode:
        print("  [MODO INFINITO] El entrenamiento corre hasta que presiones Ctrl+C.")
        print(f"  Checkpoints se guardan cada {SAVE_EVERY_EPOCHS} épocas.")
        print(f"  Previews se generan cada {SAMPLE_EVERY_EPOCHS} épocas.")
        print("  Monitor en vivo: abrir monitor.html en el navegador.\n")
    
    try:
        for epoch in range(start_epoch, total_epochs + 1):
            epoch_start_time = time.time()
            generator.train()
            discriminator.train()
            
            running_g_loss = 0.0
            running_d_loss = 0.0
            running_l1 = 0.0
            running_edge = 0.0
            
            for step, (condition, target, _) in enumerate(dataloader):
                condition = condition.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)
                
                # ----------------------------------------------------------------
                # (A) ENTRENAR DISCRIMINADOR
                # ----------------------------------------------------------------
                optimizer_d.zero_grad()
                
                with autocast(enabled=USE_AMP):
                    real_pred = discriminator(condition, target)
                    with torch.no_grad():
                        fake_sprites = generator(condition)
                        
                    fake_pred = discriminator(condition, fake_sprites.detach())
                    d_loss, d_metrics = criterion.discriminator_loss(real_pred, fake_pred)
                    
                scaler.scale(d_loss).backward()
                scaler.unscale_(optimizer_d)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
                scaler.step(optimizer_d)
                
                # ----------------------------------------------------------------
                # (B) ENTRENAR GENERADOR
                # ----------------------------------------------------------------
                optimizer_g.zero_grad()
                
                with autocast(enabled=USE_AMP):
                    fake_sprites = generator(condition)
                    fake_pred_disc = discriminator(condition, fake_sprites)
                    # Pérdida con identidad frontal para afinar y mejorar calidad
                    g_loss, g_metrics = criterion.generator_loss(
                        fake_pred_disc,
                        fake_sprites,
                        target,
                        front_identity=condition[:, :3]
                    )
                    
                scaler.scale(g_loss).backward()
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
                scaler.step(optimizer_g)
                scaler.update()
                
                running_g_loss += g_metrics["g_total"]
                running_d_loss += d_metrics["d_total"]
                running_l1 += g_metrics["g_l1"]
                running_edge += g_metrics["g_edge"]
                
            scheduler_g.step()
            scheduler_d.step()
            
            # Estadísticas de la época
            num_batches = len(dataloader)
            avg_g_loss = running_g_loss / num_batches
            avg_d_loss = running_d_loss / num_batches
            avg_l1 = running_l1 / num_batches
            avg_edge = running_edge / num_batches
            epoch_sec = time.time() - epoch_start_time
            current_lr = scheduler_g.get_last_lr()[0]
            
            # Termostato Inteligente Adaptativo en Tiempo Real (Protección Dual: GPU <= 80°C, CPU <= 85°C)
            gpu_temp, cpu_temp = manage_adaptive_thermal_throttle(temp_gpu_limit=80, temp_cpu_limit=85, temp_gpu_cooldown=68, temp_cpu_cooldown=76)
            temp_parts = []
            if gpu_temp is not None: temp_parts.append(f"GPU: {gpu_temp}°C")
            if cpu_temp is not None: temp_parts.append(f"CPU: {cpu_temp}°C")
            temp_str = f" | {' | '.join(temp_parts)}" if temp_parts else ""
                
            print(f"Época [{epoch:03d}/{('INF' if infinite_mode else f'{total_epochs:03d}')}] ({phase_cfg['phase_name']}) - {epoch_sec:.1f}s | G_Loss: {avg_g_loss:.2f} (L1: {avg_l1:.3f}, Edge: {avg_edge:.3f}) | D_Loss: {avg_d_loss:.3f} | LR: {current_lr:.2e}{temp_str}", flush=True)

            # Salvaguarda inteligente anti-colapso: si la pérdida se dispara, restaurar y frenar
            if epoch > 50 and (avg_l1 > 0.20 or avg_g_loss > 75.0):
                print(f"\n[SALVAGUARDA ANTI-COLAPSO] Inestabilidad prevenida en época {epoch}. Los mejores pesos están protegidos intactos en disco.")
                if best_gen_path.exists():
                    generator.load_state_dict(torch.load(best_gen_path, map_location=DEVICE))
                break

            # Generar vista previa visual adaptada a la fase y calcular QC quirúrgico
            if epoch % SAMPLE_EVERY_EPOCHS == 0 or epoch == 1:
                preview_file = SAMPLES_DIR / f"preview_epoch_{epoch:03d}.png"
                last_qc = generate_sample_preview(generator, template_mgr, sample_front, epoch, preview_file, phase_cfg=phase_cfg, sample_target=sample_target, dataset=dataset)
                c_prec = last_qc.get("cuerpo_precision", last_qc["score_total"])
                def_px = last_qc.get("defectos_cuerpo", 0)
                tot_px = last_qc.get("total_px_cuerpo", 0)
                c_ia = last_qc.get("colores_ia", 0)
                c_tgt = last_qc.get("colores_original", 0)
                print(f"  -> Vista previa: {preview_file.name} | Calidad Pura: {last_qc['score_total']}% | Cuerpo: {c_prec}% ({def_px} defectos de {tot_px} px) | Colores: {c_ia} (Objetivo: {c_tgt}) [Con Filtros: {last_qc.get('enhanced_total', 0)}%]", flush=True)
            
            # Escribir estado en JSON para el monitor web en tiempo real
            elapsed_total = time.time() - training_start_time
            if isinstance(last_qc, dict) and "quality_guide" not in last_qc:
                try:
                    last_qc["quality_guide"] = PixelArtEnhancer.build_quality_guide(last_qc)
                except Exception:
                    pass

            status = {
                "epoch": epoch,
                "total_epochs": "\u221e" if infinite_mode else total_epochs,
                "infinite_mode": infinite_mode,
                "phase": phase_cfg["phase_name"],
                "phase_id": 1 if phase_cfg["phase"] == "1" else 2,
                "total_frames": phase_cfg["total_frames"],
                "template_name": phase_cfg["template_path"].name,
                "status": "ENTRENANDO",
                "g_loss": round(avg_g_loss, 4),
                "d_loss": round(avg_d_loss, 4),
                "l1_loss": round(avg_l1, 4),
                "edge_loss": round(avg_edge, 4),
                "lr": current_lr,
                "epoch_sec": round(epoch_sec, 1),
                "elapsed_sec": round(elapsed_total, 0),
                "best_loss": round(best_loss, 4) if best_loss != float("inf") else None,
                "gpu_temp": gpu_temp,
                "cpu_temp": cpu_temp,
                "timestamp": time.strftime("%H:%M:%S"),
                "quality": last_qc
            }
            try:
                with open(status_path, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
                
            # Guardar Checkpoints
            if epoch % SAVE_EVERY_EPOCHS == 0:
                checkpoint_data = {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "best_loss": best_loss,
                    "phase": phase_cfg["phase"]
                }
                torch.save(checkpoint_data, latest_ckpt_path)
                
                # Guardar mejor generador para inferencia directa o transfer
                if avg_g_loss < best_loss:
                    best_loss = avg_g_loss
                    torch.save(generator.state_dict(), best_gen_path)
                    print(f"  [*] Nuevo mejor modelo guardado en: {best_gen_path.name} (Loss: {best_loss:.2f})", flush=True)

    except KeyboardInterrupt:
        print("\n\n[Ctrl+C detectado] Guardando checkpoint de emergencia...")
        checkpoint_data = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "best_loss": best_loss,
            "phase": phase_cfg["phase"]
        }
        torch.save(checkpoint_data, latest_ckpt_path)
        if avg_g_loss < best_loss:
            torch.save(generator.state_dict(), best_gen_path)
        try:
            status["status"] = "DETENIDO"
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        print(f"[OK] Checkpoint guardado en época {epoch}. Puedes reanudar con --resume.")

    # Al finalizar normalmente
    if not best_gen_path.exists():
        torch.save(generator.state_dict(), best_gen_path)
    try:
        status["status"] = "COMPLETADO"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    print("\n" + "=" * 70)
    print(f"  ENTRENAMIENTO DE {phase_cfg['phase_name'].upper()} COMPLETADO")
    print(f"  Pesos guardados en: {CHECKPOINT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar Generador de Pixel Art Spritesheets")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Número de épocas. Usa 0 para modo infinito (hasta Ctrl+C).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Tamaño del batch")
    parser.add_argument("--resume", action="store_true", help="Reanudar desde último checkpoint de la fase")
    parser.add_argument("--infinite", action="store_true", help="Modo infinito: entrena sin límite hasta Ctrl+C")
    parser.add_argument("--phase", type=str, default="2", choices=["1", "2", "16x4", "8x12"], help="Fase: 1 (base 16x4) o 2 (transferencia 8x12)")
    parser.add_argument("--transfer", type=str, default="", help="Ruta al checkpoint de Fase 1 para transferir pesos a Fase 2")
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=args.resume,
        infinite=args.infinite,
        phase=args.phase,
        transfer_from=args.transfer if args.transfer else None
    )
