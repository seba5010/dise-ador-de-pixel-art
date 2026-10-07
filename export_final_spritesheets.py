"""
export_final_spritesheets.py
Pipeline de entrega final automatizada (Post-Época 90).
Toma el mejor checkpoint verificado (best_quality_generator.pt o best_generator.pt),
recorre todos los personajes del directorio `personajes/`, genera sus hojas 8x12 (96 poses),
aplica el candado de silueta y empaqueta las imágenes PNG transparentes listas para motor de juegos.
"""

import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
import numpy as np

from pixel_ai_engine.config import (
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    PERSONAJES_DIR,
    PHASE2_TEMPLATE_PATH as TEMPLATE_PATH_8x12,
    DEVICE,
    USE_AMP,
    MODEL_RESOLUTION,
)
from pixel_ai_engine.generate_character_sheet import (
    clean_pixel_art_alpha,
    resolve_character_input,
)
from pixel_ai_engine.dataset import (
    TemplateManager,
    pad_to_square,
    place_in_cell,
    get_cell_coordinates,
)
from pixel_ai_engine.models import PixelArtUNetGenerator
from pixel_ai_engine.phase3_critical_enhancer import PixelArtEnhancer
from pixel_ai_engine.palette_remap import (
    extract_character_palette,
    remap_image_to_palette,
    despeckle_chromatic_noise,
    clean_orphan_pixels,
)
from pixel_ai_engine.head_rigging import HeadRiggingManager
from pixel_ai_engine.pixel_art_fixer import snap_and_fix_pixel_art
from pixel_ai_engine.fractional_downscale import fractional_silhouette_downscale
from pixel_ai_engine.cell_pipeline import SingleCellCoordinator


def get_best_available_checkpoint() -> Path:
    candidates = [
        CHECKPOINT_DIR / "snapshots" / "generator_epoch_090.pt",
        CHECKPOINT_DIR / "snapshots" / "checkpoint_epoch_090.pt",
        CHECKPOINT_DIR / "latest_checkpoint.pt",
        CHECKPOINT_DIR / "best_generator.pt",
        CHECKPOINT_DIR / "best_quality_generator.pt",
        CHECKPOINT_DIR / "recovery_checkpoint.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No se encontró ningún checkpoint en " + str(CHECKPOINT_DIR))


def find_all_characters() -> List[Path]:
    """Encuentra todas las imágenes frontales representativas de personajes/."""
    characters = []
    if not PERSONAJES_DIR.exists():
        return characters

    for char_dir in sorted(PERSONAJES_DIR.iterdir()):
        if char_dir.is_dir():
            # Buscar *_rnormal.png o cualquier PNG principal
            rnormals = list(char_dir.glob("*_rnormal.png"))
            if rnormals:
                characters.append(rnormals[0])
            else:
                pngs = [p for p in char_dir.glob("*.png") if "movimiento" not in p.name.lower()]
                if pngs:
                    characters.append(pngs[0])
    return characters


def export_character_spritesheet_8x12(
    generator: torch.nn.Module,
    template_mgr: TemplateManager,
    input_image_path: Path,
    output_dir: Path,
    alpha_threshold: int = 60,
) -> Path:
    char_name = input_image_path.parent.name
    output_path = output_dir / f"{char_name}_spritesheet_8x12.png"
    preview_path = output_dir / f"{char_name}_vista_previa.png"

    raw_front = Image.open(input_image_path)
    padded_front, _ = pad_to_square(raw_front, MODEL_RESOLUTION)
    clean_front = padded_front.convert("RGBA")

    front_rgb = Image.new("RGB", (MODEL_RESOLUTION, MODEL_RESOLUTION), (0, 0, 0))
    front_rgb.paste(clean_front, mask=clean_front.split()[3])
    front_arr = np.array(front_rgb).astype(np.float32) / 127.5 - 1.0
    front_tensor = torch.from_numpy(front_arr).permute(2, 0, 1)[:3].float().to(DEVICE)

    # 1. Extraer paleta canónica y cabeza canónica de alta fidelidad (Técnicas 3 y 9)
    char_palette = extract_character_palette(clean_front, include_props=True)
    canonical_head, _ = HeadRiggingManager.extract_canonical_head(clean_front)

    rows = 12
    cols = 8
    total_frames = 96
    canvas_w = 1024
    canvas_h = 1536

    generated_frames = []
    with torch.no_grad():
        for frame_idx in range(total_frames):
            row_idx = frame_idx // cols
            pose_tensor = template_mgr.get_frame_tensor(frame_idx).to(DEVICE)
            condition = torch.cat([front_tensor, pose_tensor], dim=0).unsqueeze(0)

            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=USE_AMP):
                output_tensor = generator(condition).squeeze(0)

            arr = output_tensor.detach().cpu().permute(1, 2, 0).numpy()
            arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
            frame_rgba = Image.fromarray(arr, mode="RGBA")

            # Candado de silueta morfológico contra el molde de la pose
            pose_img = template_mgr.get_frame_padded_pil(frame_idx)
            clipped = PixelArtEnhancer.clip_stray_limbs_against_template(frame_rgba, pose_img, margin_px=6)

            # Anclaje Modular de Cabeza e Identidad Facial (Técnica 9: Paper Doll)
            with_head = HeadRiggingManager.composite_head_onto_frame(
                clipped, canonical_head, pose_img, direction_row=row_idx
            )

            # Limpieza de alfa
            frame_clean = clean_pixel_art_alpha(with_head, alpha_threshold=alpha_threshold)

            # Remapeo estricto a Paleta del Personaje (Técnica 3: Cero colores inventados)
            if char_palette is not None and len(char_palette) > 0:
                frame_clean = remap_image_to_palette(frame_clean, char_palette)

            # Despeckle Cromático y Limpieza de Hollín (Técnica 8)
            frame_clean = despeckle_chromatic_noise(frame_clean)
            frame_clean = clean_orphan_pixels(frame_clean)

            # Snapping de Cuadrícula y Recuperación de Contorno de 1px (Técnica 8)
            frame_clean = snap_and_fix_pixel_art(frame_clean, enforce_dark_outline=True)

            generated_frames.append(frame_clean)

    # Ensamblado en cuadrícula canónica (Técnica 5)
    coordinator = SingleCellCoordinator(
        rows=rows, cols=cols, cell_w=128, cell_h=128, canvas_w=canvas_w, canvas_h=canvas_h
    )
    canvas = coordinator.assemble_cells_into_sheet(generated_frames)

    canvas.save(output_path, format="PNG")

    bg_preview = Image.new("RGBA", canvas.size, (230, 233, 240, 255))
    bg_preview.paste(canvas, (0, 0), canvas)
    bg_preview.convert("RGB").save(preview_path, format="PNG")

    return output_path


def export_all_characters(
    checkpoint_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_characters: Optional[int] = None,
    char_filter: Optional[str] = None,
) -> Path:
    ckpt_file = checkpoint_path or get_best_available_checkpoint()
    out_dir = output_dir or (OUTPUT_DIR / "spritesheets_finales_era2")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PIPELINE DE ENTREGA FINAL - GENERADOR COMPLETO 8x12 (12 TÉCNICAS)")
    print(f"  Checkpoint fuente: {ckpt_file.name}")
    print(f"  Directorio destino: {out_dir}")
    print(f"  Dispositivo: {DEVICE}")
    print("=" * 70)

    # Cargar plantilla y generador
    template_mgr = TemplateManager(
        template_path=TEMPLATE_PATH_8x12,
        target_size=MODEL_RESOLUTION,
        rows=12,
        cols=8,
        canvas_w=1024,
        canvas_h=1536,
    )

    generator = PixelArtUNetGenerator().to(DEVICE)
    checkpoint = torch.load(ckpt_file, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "generator" in checkpoint:
        generator.load_state_dict(checkpoint["generator"])
    else:
        generator.load_state_dict(checkpoint)
    generator.eval()

    character_files = find_all_characters()
    if char_filter:
        character_files = [cf for cf in character_files if char_filter.lower() in cf.parent.name.lower()]

    if max_characters is not None:
        character_files = character_files[:max_characters]

    print(f"Exportando {len(character_files)} personajes...")
    exported: List[Path] = []
    t0 = time.time()

    for idx, char_file in enumerate(character_files, 1):
        print(f"[{idx:02d}/{len(character_files):02d}] Procesando {char_file.parent.name}...", flush=True)
        out_file = export_character_spritesheet_8x12(
            generator, template_mgr, char_file, out_dir
        )
        exported.append(out_file)

    elapsed = round(time.time() - t0, 1)

    # Escribir reporte de entrega
    report_path = PROJECT_ROOT / "reportes" / "entrega_final_era2.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Reporte de Entrega Final de Spritesheets (Era 2 - 8x12 - 12 Técnicas)\n\n")
        f.write(f"- 📅 **Fecha:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 💾 **Checkpoint Evaluado:** `{ckpt_file.name}`\n")
        f.write(f"- 🎮 **Personajes Exportados:** `{len(exported)}`\n")
        f.write(f"- ⏱️ **Tiempo Total:** `{elapsed}s`\n")
        f.write(f"- 📁 **Carpeta de Salida:** `{out_dir}`\n\n")
        f.write("## Spritesheets Generados\n\n")
        f.write("| # | Personaje | Archivo PNG (Transparente) | Vista Previa |\n")
        f.write("| :-: | :--- | :--- | :--- |\n")
        for i, path in enumerate(exported, 1):
            name = path.stem.replace("_spritesheet_8x12", "")
            prev = path.with_name(f"{name}_vista_previa.png")
            f.write(f"| {i:02d} | `{name}` | `{path.name}` | `{prev.name}` |\n")

    print("\n" + "=" * 70)
    print(f"  ENTREGA COMPLETADA EXITOSAMENTE EN {elapsed}s")
    print(f"  Total spritesheets: {len(exported)}")
    print(f"  Reporte guardado en: {report_path}")
    print("=" * 70)
    return report_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exportar spritesheets con arquitectura de 12 técnicas")
    parser.add_argument("--char", type=str, default=None, help="Personaje específico (ej: mauricio)")
    parser.add_argument("--max", type=int, default=None, help="Límite máximo de personajes")
    parser.add_argument("--output", type=str, default=None, help="Directorio de salida personalizado")
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else None
    export_all_characters(char_filter=args.char, max_characters=args.max, output_dir=out_p)
