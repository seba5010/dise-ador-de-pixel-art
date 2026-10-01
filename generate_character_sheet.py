"""
Entrypoint principal para generar hojas de spritesheets completas (16x4) a partir de una imagen frontal
"""
import sys
from pathlib import Path
import argparse

# Añadir directorio actual al path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pixel_ai_engine.generate_character_sheet import generate_spritesheet
from pixel_ai_engine.config import CHECKPOINT_DIR, TOTAL_FRAMES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Generar Spritesheet completo de {TOTAL_FRAMES} frames a partir de un personaje frontal")
    parser.add_argument("--input", "-i", type=str, required=True, help="Ruta a la imagen frontal del personaje (ej: belial/belial_rnormal.png o mauricio_rnormal.png)")
    parser.add_argument("--checkpoint", "-c", type=str, default=str(CHECKPOINT_DIR / "best_generator.pt"), help="Ruta al checkpoint del generador (.pt)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Ruta de salida para la hoja de sprites generada")
    parser.add_argument("--template", "-t", type=str, default=None, help="Ruta a plantilla personalizada (opcional)")
    parser.add_argument("--phase", "-p", type=str, default=None, choices=["1", "2"], help="Fase / Formato: '2' para 8x12 (96 frames, por defecto), '1' para 16x4 (64 frames)")
    parser.add_argument("--alpha-threshold", type=int, default=60, help="Umbral de recorte alfa para pixel art (0-255)")
    args = parser.parse_args()

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
