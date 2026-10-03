"""
Pixel AI Engine - Inference & Spritesheet Generator (generate_character_sheet.py)
Toma como entrada una imagen frontal de un nuevo personaje (ej: mauricio_rnormal.png),
carga el modelo generador entrenado y predice automáticamente los 64 frames organizados
en la grilla fija de 16 filas x 4 columnas (724x2172), listo para exportar a Unity.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import torch
from torch.cuda.amp import autocast

from .config import (
    PROJECT_ROOT,
    PERSONAJES_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    GRID_ROWS,
    GRID_COLS,
    TOTAL_FRAMES,
    MODEL_RESOLUTION,
    DEVICE,
    USE_AMP,
    TEMPLATE_PATH,
    ROW_ANIMATION_NAMES,
    get_phase_config
)
from .dataset import (
    TemplateManager,
    isolate_character,
    pad_to_square,
    unpad_from_square,
    place_in_cell,
    get_cell_coordinates
)
from .models import PixelArtUNetGenerator


def resolve_character_input(input_arg: str) -> Path:
    """
    Resuelve inteligentemente la ruta de un personaje aceptando:
      - Rutas directas: 'personajes/mauricio/mauricio_rnormal.png'
      - Solo nombre del personaje: 'mauricio' -> busca 'personajes/mauricio/mauricio_rnormal.png'
      - Subcarpetas: 'andres_arica' -> busca 'personajes/andres_arica/andres_rnormal.png'
    """
    p = Path(input_arg)
    if p.exists() and p.is_file():
        return p
        
    # Buscar en carpeta personajes/
    candidates = [
        PROJECT_ROOT / p,
        PERSONAJES_DIR / p,
        PERSONAJES_DIR / p / f"{p.name}_rnormal.png",
        PERSONAJES_DIR / p / "rnormal.png",
        PROJECT_ROOT / p / f"{p.name}_rnormal.png",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
            
    # Si p es un directorio dentro de personajes, buscar el primer PNG frontal
    if (PERSONAJES_DIR / p).is_dir():
        pngs = list((PERSONAJES_DIR / p).glob("*.png"))
        rnormals = [f for f in pngs if "rnormal" in f.name.lower() and "movimiento" not in f.name.lower()]
        if rnormals:
            return rnormals[0]
        non_movements = [f for f in pngs if "movimiento" not in f.name.lower()]
        if non_movements:
            return non_movements[0]

    raise FileNotFoundError(f"No se pudo encontrar la imagen del personaje para '{input_arg}'. Rutas buscadas:\n" + "\n".join(f"  - {c}" for c in candidates))


def clean_pixel_art_alpha(img_rgba: Image.Image, alpha_threshold: int = 100) -> Image.Image:
    """
    Binariza o limpia el canal alfa para asegurar bordes de pixel art sólidos sin halos difusos.
    """
    arr = np.array(img_rgba)
    alpha = arr[:, :, 3]
    # Pixels con opacidad por debajo del umbral se hacen completamente transparentes (0)
    arr[alpha < alpha_threshold, 3] = 0
    # Pixels con opacidad alta se solidifican (255)
    arr[alpha >= alpha_threshold, 3] = 255
    return Image.fromarray(arr, mode="RGBA")


def generate_spritesheet(input_image_path: Path,
                         checkpoint_path: Path,
                         output_image_path: Optional[Path] = None,
                         template_path: Optional[Path] = None,
                         alpha_threshold: int = 60,
                         phase: Optional[str] = "2",
                         export_unity_meta: bool = False) -> Path:
    
    input_image_path = resolve_character_input(str(input_image_path))
        
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint del generador en: {checkpoint_path}")

    # Determinar fase activa (si no se especifica, auto-detectar por nombre de checkpoint o plantilla)
    if phase is None:
        ckpt_str = str(checkpoint_path).lower()
        tmpl_str = str(template_path).lower() if template_path else ""
        if "16x4" in ckpt_str or "16x4" in tmpl_str:
            target_phase = "1"
        else:
            # Por defecto Fase 2 (Molde definitivo 8x12 con 96 poses)
            target_phase = "2"
    else:
        target_phase = str(phase)

    phase_cfg = get_phase_config(target_phase)
    cols = phase_cfg["grid_cols"]
    rows = phase_cfg["grid_rows"]
    total_frames = phase_cfg["total_frames"]
    canvas_w = phase_cfg["canvas_w"]
    canvas_h = phase_cfg["canvas_h"]
    row_names = phase_cfg["row_names"]
    tmpl_path = template_path or phase_cfg["template_path"]
        
    print("=" * 70)
    print(f"  PIXEL ART AI ENGINE - GENERADOR DE HOJAS DE SPRITES ({cols}x{rows} - {total_frames} frames)")
    print(f"  Modo: {phase_cfg['phase_name']}")
    print(f"  Entrada frontal: {input_image_path.name}")
    print(f"  Checkpoint: {checkpoint_path.name}")
    print(f"  Plantilla: {tmpl_path.name}")
    print(f"  Dispositivo: {DEVICE}")
    print("=" * 70)

    # 1. Cargar y preparar imagen de entrada frontal (Input X, 3 canales RGB)
    raw_front = Image.open(input_image_path)
    padded_front, _ = pad_to_square(raw_front, MODEL_RESOLUTION)
    clean_front = padded_front.convert("RGBA")
    
    front_rgb = Image.new("RGB", (MODEL_RESOLUTION, MODEL_RESOLUTION), (0, 0, 0))
    front_rgb.paste(clean_front, mask=clean_front.split()[3])
    front_arr = np.array(front_rgb).astype(np.float32) / 127.5 - 1.0
    front_tensor = torch.from_numpy(front_arr).permute(2, 0, 1)[:3].float().to(DEVICE)  # (3, H, W)

    # 2. Cargar plantilla de movimientos correspondiente a la fase
    template_mgr = TemplateManager(template_path=tmpl_path, target_size=MODEL_RESOLUTION,
                                   rows=rows, cols=cols, canvas_w=canvas_w, canvas_h=canvas_h)
    print(f"[Plantilla] Cargada plantilla de movimientos ({cols}x{rows}) desde: {tmpl_path.name}")

    # 3. Cargar el modelo generador
    generator = PixelArtUNetGenerator().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    if isinstance(checkpoint, dict) and "generator" in checkpoint:
        generator.load_state_dict(checkpoint["generator"])
    else:
        generator.load_state_dict(checkpoint)
        
    generator.eval()
    print("[Modelo] Generador cargado exitosamente.")

    # 4. Inferencia frame por frame
    print(f"\nGenerando {total_frames} frames condicionados...")
    generated_frames = []
    
    with torch.no_grad():
        for frame_idx in range(total_frames):
            pose_tensor = template_mgr.get_frame_tensor(frame_idx).to(DEVICE)
            condition = torch.cat([front_tensor, pose_tensor], dim=0).unsqueeze(0)  # (1, 6, H, W)
            
            with autocast(enabled=USE_AMP):
                output_tensor = generator(condition).squeeze(0)  # (4, H, W)
                
            arr = output_tensor.detach().cpu().permute(1, 2, 0).numpy()
            arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
            frame_rgba = Image.fromarray(arr, mode="RGBA")
            
            # Limpieza de alfa para pixel art nítido
            frame_clean = clean_pixel_art_alpha(frame_rgba, alpha_threshold=alpha_threshold)
            generated_frames.append(frame_clean)
            
            row = frame_idx // cols
            col = frame_idx % cols
            anim_name = row_names[row] if row < len(row_names) else f"fila_{row+1}"
            print(f"  Frame {frame_idx + 1:02d}/{total_frames} generado: Fila {row + 1:02d} Col {col} ({anim_name})", end="\r")

    print(f"\n\nEnsamblando la hoja de sprites completa ({canvas_w} x {canvas_h})...")
    # 5. Crear el lienzo canónico transparente
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    
    for frame_idx, frame_img in enumerate(generated_frames):
        row = frame_idx // cols
        col = frame_idx % cols
        
        # Coordenadas exactas en el canvas final
        x0, y0, x1, y1 = get_cell_coordinates(canvas_w, canvas_h, row, col, rows, cols)
        target_w = x1 - x0
        target_h = y1 - y0
        
        cell_sprite = place_in_cell(frame_img, cell_w=target_w, cell_h=target_h)
        canvas.paste(cell_sprite, (x0, y0), cell_sprite)

    # 6. Guardar archivo final
    if output_image_path is None:
        char_name = input_image_path.stem.replace("_rnormal", "").replace("rnormal", "character")
        output_image_path = OUTPUT_DIR / f"{char_name}_spritesheet_{cols}x{rows}.png"
    else:
        output_image_path = Path(output_image_path)
        
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_image_path, format="PNG")
    
    # Guardar versión con fondo claro para visualización nítida e inmediata en Windows
    preview_path = output_image_path.with_name(f"{output_image_path.stem}_vista_previa.png")
    bg_preview = Image.new("RGBA", canvas.size, (230, 233, 240, 255))
    bg_preview.paste(canvas, (0, 0), canvas)
    bg_preview.convert("RGB").save(preview_path, format="PNG")
    
    print("=" * 70)
    print(f"  HOJA GENERADA EXITOSAMENTE: {output_image_path}")
    print(f"  Dimensiones: {canvas.size[0]} x {canvas.size[1]} px | Formato: RGBA (Transparente para Unity/Godot)")
    print(f"  Vista Previa Clara: {preview_path.name}")
    print(f"  Grilla: {rows} filas x {cols} columnas ({total_frames} frames)")
    print("=" * 70)
    
    return output_image_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar Spritesheet completo de 96 frames (8x12) o 64 frames (16x4) a partir de un personaje frontal")
    parser.add_argument("--input", "-i", type=str, required=True, help="Ruta a la imagen frontal del personaje (ej: belial/belial_rnormal.png o mauricio_rnormal.png)")
    parser.add_argument("--checkpoint", "-c", type=str, default=str(CHECKPOINT_DIR / "best_generator.pt"), help="Ruta al checkpoint del generador (.pt)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Ruta de salida para la hoja de sprites generada")
    parser.add_argument("--template", "-t", type=str, default=None, help="Ruta a plantilla personalizada (opcional)")
    parser.add_argument("--phase", "-p", type=str, default="2", choices=["1", "2"], help="Fase / Formato: '2' para 8x12 (96 frames, por defecto), '1' para 16x4 (64 frames)")
    parser.add_argument("--alpha-threshold", type=int, default=60, help="Umbral de recorte alfa para pixel art (0-255)")
    args = parser.parse_args()

    # Si no se encuentra best_generator.pt, intentar latest_checkpoint.pt o base_generator_16x4.pt
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        fallback_ckpt = CHECKPOINT_DIR / "latest_checkpoint.pt"
        if not fallback_ckpt.exists():
            fallback_ckpt = CHECKPOINT_DIR / "base_generator_16x4.pt"
        if fallback_ckpt.exists():
            ckpt_path = fallback_ckpt

    generate_spritesheet(
        input_image_path=Path(args.input),
        checkpoint_path=ckpt_path,
        output_image_path=Path(args.output) if args.output else None,
        template_path=Path(args.template) if args.template else None,
        alpha_threshold=args.alpha_threshold,
        phase=args.phase
    )
