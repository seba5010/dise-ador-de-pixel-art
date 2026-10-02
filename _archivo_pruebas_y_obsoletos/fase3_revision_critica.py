"""
FASE 3: SCRIPT PRINCIPAL DE REVISIÓN CRÍTICA Y ELEVACIÓN DE DISEÑO
Pixel AI Engine - Spritesheet Quality Elevation

Toma una hoja de spritesheet generada en Fase 2 (96 poses en 8x12), ejecuta una auditoría
quirúrgica y aplica la capa de elevación para superar los diseños originales:
- Agrega profundidad 3D y micro-sombras a la ropa y accesorios.
- Refuerza contornos de 1px (anti-fusión de extremidades).
- Binariza y limpia 100% el fondo alfa.
- Genera comparativa visual de alta resolución.
"""

import sys
import argparse
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pixel_ai_engine.phase3_critical_enhancer import Phase3CriticalReviewer
from pixel_ai_engine.config import OUTPUT_DIR, PERSONAJES_DIR, SAMPLES_DIR
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Fase 3: Revisión Crítica y Elevación de Calidad de Spritesheets")
    parser.add_argument("--sheet", "-s", type=str, default=None,
                        help="Ruta a la hoja de spritesheet generada o 'latest' para la más reciente")
    parser.add_argument("--latest", "-l", action="store_true",
                        help="Revisar automáticamente la época más reciente generada por el entrenamiento")
    parser.add_argument("--identity", "-i", type=str, default=None,
                        help="Ruta a la imagen frontal de identidad del personaje")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Ruta de guardado para la hoja elevada")
    parser.add_argument("--phase", "-p", type=str, default=None, choices=["1", "2"],
                        help="Fase del formato ('1' para 16x4 o '2' para 8x12; si no se indica, se autodetecta)")
    args = parser.parse_args()

    # 1. Resolver spritesheet de entrada
    sheet_path = None
    use_latest = args.latest or (args.sheet in [None, "latest", "reciente", "actual"])

    if not use_latest and args.sheet:
        p = Path(args.sheet)
        if p.exists():
            sheet_path = p
        elif (PROJECT_ROOT / p).exists():
            sheet_path = PROJECT_ROOT / p

    if sheet_path is None:
        # Buscar en training_samples la época más reciente
        candidates = sorted(SAMPLES_DIR.glob("preview_epoch_*.png"), key=lambda f: f.stat().st_mtime)
        valid = [f for f in candidates if "FASE3" not in f.name and "vista_previa" not in f.name]
        if valid:
            sheet_path = valid[-1]
            print(f"[Auto-Detección] Época más reciente detectada: {sheet_path.name}")
        else:
            # Fallback a generated_spritesheets
            candidates_out = sorted(OUTPUT_DIR.glob("*.png"), key=lambda f: f.stat().st_mtime)
            valid_out = [f for f in candidates_out if "vista_previa" not in f.name and "FASE3" not in f.name]
            if valid_out:
                sheet_path = valid_out[-1]
                print(f"[Auto-Detección] Hoja de spritesheet detectada: {sheet_path.name}")

    if not sheet_path or not sheet_path.exists():
        print(f"[Error] No se encontró ninguna hoja de sprites para procesar.")
        print(f"Por favor especifica la ruta con --sheet ruta/al/archivo.png")
        sys.exit(1)

    # 2. Auto-detección de fase si no se especificó
    phase = args.phase
    if phase is None:
        try:
            with Image.open(sheet_path) as im:
                w, h = im.size
                if w <= 750:
                    phase = "1"
                else:
                    phase = "2"
        except Exception:
            phase = "1"
        print(f"[Auto-Detección] Cuadrícula detectada: Fase {phase} ({'8x12 - 96 poses' if phase == '2' else '16x4 - 64 poses'})")

    # 3. Auto-detección de identidad frontal si no se especificó
    id_path = Path(args.identity) if args.identity else None
    if id_path is None or not id_path.exists():
        # Buscar candidatos por defecto
        id_candidates = [
            PERSONAJES_DIR / "conny" / "conny_rbchef.png",
            PERSONAJES_DIR / "conny" / "conny_rnormal.png",
            PERSONAJES_DIR / "alex" / "alex_rbchef.png"
        ]
        for cand in id_candidates:
            if cand.exists():
                id_path = cand
                print(f"[Auto-Detección] Identidad frontal: {id_path.name}")
                break

    print("=" * 75)
    print("      🔍 INICIANDO FASE 3: AUDITORÍA CRÍTICA Y ELEVACIÓN DE DISEÑO")
    print(f"      Hoja a procesar: {sheet_path.name}")
    print(f"      Formato: Fase {phase} ({'8x12 - 96 poses' if phase == '2' else '16x4 - 64 poses'})")
    if id_path:
        print(f"      Identidad frontal: {id_path.name}")
    print("=" * 75)

    # 2. Ejecutar elevación
    out_path = Path(args.output) if args.output else None

    elevated_path, report = Phase3CriticalReviewer.elevate_spritesheet(
        sheet_path=sheet_path,
        output_path=out_path,
        identity_path=id_path,
        phase=phase
    )

    print("Resumen de Auditoría y Mejora:")
    print(f"  • Puntuación Fase 2 (Antes):  {report['calidad_promedio_fase2']}%")
    print(f"  • Puntuación Fase 3 (Elevada): {report['calidad_promedio_fase3_elevada']}%")
    print(f"  • Incremento de Calidad:      +{report['mejora_porcentual']}%")
    print(f"  • Archivo PNG Transparente:   {elevated_path}")
    print(f"  • Vista Previa Clara:         {report['archivo_preview']}")


if __name__ == "__main__":
    main()
