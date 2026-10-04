"""
prepare_supervised_dataset.py
================================
Lee los frames ya cortados desde `dataset_frames_individuales/` y construye
el cache de entrenamiento supervisado (supervised_cache_8x12.pt).

Estructura esperada de dataset_frames_individuales/:
  <PERSONAJE_MAYUS>/
    <char_variante>/
      manifest.json             -> metadatos del personaje y variante
      00_frontal_identidad.png  -> imagen frontal de referencia
      frame_001_r01_c01.png     -> frame individual
      frame_002_r01_c02.png
      ...

El manifest.json contiene:
  - character    -> nombre del personaje (ej. "alex")
  - variant      -> clave de variante (ej. "rnormal", "rbchef", "rnchef")
  - variant_name -> nombre descriptivo (ej. "ropa_normal")
  - frames[]     -> lista de frames con row, col, frame_index, etc.

NOTA: Auto-descubrimiento completo - incluye TODOS los personajes en
      dataset_frames_individuales/ sin lista manual.
"""

import os
import sys
import json
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pixel_ai_engine.config import MODEL_RESOLUTION, DEVICE
from pixel_ai_engine.dataset import (
    CANONICAL_FRONT_REFERENCE_HEIGHT,
    CANONICAL_SPRITE_HEIGHT,
    CANONICAL_SPRITE_MAX_WIDTH,
    TRAINING_FRAME_HEIGHT,
    TRAINING_FRAME_MAX_WIDTH,
    TemplateManager,
    adapt_front_to_chibi,
    pad_target_frame_canonical,
    get_cell_coordinates,
    get_8x12_index_from_16x4
)
from pixel_ai_engine.frame_map import get_frame_semantic_info

DATASET_OUT = PROJECT_ROOT / "dataset_supervisado"
FRAMES_PNG_DIR = DATASET_OUT / "frames_png"
CAPTIONS_TXT_DIR = DATASET_OUT / "captions_txt"
FRONTS_DIR = DATASET_OUT / "reference_fronts"
STATUS_PATH = PROJECT_ROOT / "training_status.json"

# Directorio raiz del nuevo dataset de frames individuales
FRAMES_ROOT = PROJECT_ROOT / "dataset_frames_individuales"

FRAMES_PNG_DIR.mkdir(parents=True, exist_ok=True)
CAPTIONS_TXT_DIR.mkdir(parents=True, exist_ok=True)
FRONTS_DIR.mkdir(parents=True, exist_ok=True)

# Mapeo de clave variante a descripcion legible para captions
VARIANT_DESCRIPTIONS = {
    "rnormal":          "ropa normal casual",
    "rbchef":           "uniforme de chef blanco con delantal",
    "rnchef":           "uniforme de chef negro con delantal",
    "rbnormal":         "ropa normal casual",
    "ropa_normal":      "ropa normal casual",
    "ropa_blanca_chef": "uniforme de chef blanco con delantal",
    "ropa_negra_chef":  "uniforme de chef negro con delantal",
}


def get_variant_desc(variant_key: str, variant_name: str = "") -> str:
    """Devuelve descripcion legible dada la clave o nombre de la variante."""
    for k, v in VARIANT_DESCRIPTIONS.items():
        if k in variant_key.lower() or k in variant_name.lower():
            return v
    return "ropa casual"


def mark_dataset_layout_updated(total_samples: int) -> None:
    status_data = {}
    if STATUS_PATH.exists():
        try:
            with open(STATUS_PATH, "r", encoding="utf-8") as status_file:
                status_data = json.load(status_file)
        except (OSError, json.JSONDecodeError):
            status_data = {}

    recovery = status_data.get("recovery")
    if isinstance(recovery, dict):
        recovery["active"] = False
        recovery["invalidated_by_dataset_layout"] = True

    # Resetear estado de entrenamiento a epoca 0
    status_data["status"] = "DATASET_ACTUALIZADO"
    status_data["epoch"] = 0
    status_data["total_epochs"] = 0
    status_data["best_loss"] = None
    status_data["dataset_layout"] = {
        "version": 5,
        "source": "dataset_frames_individuales",
        "front_reference_height": CANONICAL_FRONT_REFERENCE_HEIGHT,
        "training_frame_height": TRAINING_FRAME_HEIGHT,
        "training_frame_max_width": TRAINING_FRAME_MAX_WIDTH,
        "canonical_sprite_height": CANONICAL_SPRITE_HEIGHT,
        "canonical_sprite_max_width": CANONICAL_SPRITE_MAX_WIDTH,
        "cell_width": 128,
        "cell_height": 128,
        "sample_count": int(total_samples),
        "requires_new_era": True,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    status_data["dataset_message"] = (
        "Dataset completo con TODOS los personajes regenerado desde "
        "dataset_frames_individuales. Inicia nueva era desde epoca 0."
    )
    status_data["timestamp"] = time.strftime("%H:%M:%S")

    temp_path = STATUS_PATH.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as status_file:
        json.dump(status_data, status_file, indent=2, ensure_ascii=False)
    os.replace(temp_path, STATUS_PATH)


def has_trainable_content(image: Image.Image, minimum_pixels: int = 40) -> bool:
    alpha = np.array(image.convert("RGBA"))[:, :, 3]
    return int(np.count_nonzero(alpha > 20)) >= minimum_pixels


def remove_stale_empty_sample(png_filename: str, txt_filename: str) -> None:
    for path in (FRAMES_PNG_DIR / png_filename, CAPTIONS_TXT_DIR / txt_filename):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def discover_variants() -> list:
    """
    Recorre dataset_frames_individuales/ y devuelve lista de dicts con info
    de cada variante encontrada via manifest.json (auto-descubrimiento completo).
    """
    variants = []
    if not FRAMES_ROOT.exists():
        print(f"[ERROR] No existe: {FRAMES_ROOT}")
        return variants

    for char_dir in sorted(FRAMES_ROOT.iterdir()):
        if not char_dir.is_dir():
            continue
        if char_dir.name.startswith("00_"):
            continue  # Saltar carpeta de moldes de poses

        for variant_dir in sorted(char_dir.iterdir()):
            if not variant_dir.is_dir():
                continue

            manifest_path = variant_dir / "manifest.json"
            if not manifest_path.exists():
                print(f"  [!] Sin manifest.json: {variant_dir.relative_to(PROJECT_ROOT)}")
                continue

            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                print(f"  [!] Error manifest {manifest_path.name}: {e}")
                continue

            char_name    = manifest.get("character", char_dir.name.lower())
            variant_key  = manifest.get("variant", variant_dir.name)
            variant_name = manifest.get("variant_name", "")
            variant_desc = get_variant_desc(variant_key, variant_name)

            variants.append({
                "char_name":    char_name,
                "variant_key":  variant_key,
                "variant_desc": variant_desc,
                "variant_dir":  variant_dir,
                "manifest":     manifest,
            })

    return variants


def build_supervised_dataset():
    print("=" * 70)
    print("  EXTRACCION Y ALINEACION DE FRAMES - DATASET COMPLETO")
    print(f"  Fuente: dataset_frames_individuales/")
    print(f"  Destino: {DATASET_OUT.relative_to(PROJECT_ROOT)}")
    print("=" * 70)

    variants = discover_variants()
    if not variants:
        print("[ERROR] No se encontraron variantes validas.")
        return []

    chars_found = sorted(set(v["char_name"] for v in variants))
    print(f"\n  Personajes descubiertos ({len(chars_found)}): {', '.join(chars_found)}")
    print(f"  Total variantes a procesar: {len(variants)}\n")

    dataset_samples = []
    total_cut_frames = 0
    skipped_variants = 0

    for v in variants:
        char_name    = v["char_name"]
        variant_key  = v["variant_key"]
        variant_desc = v["variant_desc"]
        variant_dir  = v["variant_dir"]

        # Sanitizar nombre para uso en nombres de archivo (espacios -> guion bajo)
        char_id_safe = char_name.replace(" ", "_").replace("-", "_")
        char_var_id  = f"{char_id_safe}_{variant_key}"
        print(f"[*] Procesando: {char_var_id.upper()}")

        # 1. Imagen frontal de referencia
        frontal_path = variant_dir / "00_frontal_identidad.png"
        if not frontal_path.exists():
            print(f"    [!] Sin 00_frontal_identidad.png, saltando variante.")
            skipped_variants += 1
            continue

        try:
            raw_front   = Image.open(frontal_path).convert("RGBA")
            chibi_front = adapt_front_to_chibi(raw_front, target_size=MODEL_RESOLUTION)
        except Exception as e:
            print(f"    [!] Error procesando frontal: {e}")
            skipped_variants += 1
            continue

        chibi_front_path = FRONTS_DIR / f"{char_var_id}_chibi_front.png"
        chibi_front.save(chibi_front_path)

        arr_f  = np.array(chibi_front).astype(np.float32)
        mask_f = arr_f[:, :, 3] > 20
        rgb_f  = arr_f[:, :, :3]
        rgb_f[~mask_f] = 0.0
        front_tensor = torch.from_numpy(rgb_f / 127.5 - 1.0).permute(2, 0, 1).float()

        # 2. Procesar cada frame individual (frame_NNN_rRR_cCC.png)
        frame_files = sorted(variant_dir.glob("frame_*.png"))
        if not frame_files:
            print(f"    [!] Sin frames PNG en {variant_dir.name}. Saltando.")
            skipped_variants += 1
            continue

        char_frames = 0
        for frame_path in frame_files:
            # Nombre: frame_001_r01_c01.png -> row=1, col=1
            stem  = frame_path.stem
            parts = stem.split("_")
            try:
                row_num = int(parts[2][1:])  # r01 -> 1
                col_num = int(parts[3][1:])  # c01 -> 1
            except (IndexError, ValueError):
                continue

            # Indice 0-basado en grilla 8x12
            f_idx = (row_num - 1) * 8 + (col_num - 1)

            png_filename = f"{char_var_id}_frame_{f_idx:03d}.png"
            txt_filename = f"{char_var_id}_frame_{f_idx:03d}.txt"

            try:
                cell = Image.open(frame_path).convert("RGBA")
            except Exception:
                continue

            padded_tgt = pad_target_frame_canonical(cell, target_size=MODEL_RESOLUTION)

            if not has_trainable_content(padded_tgt):
                remove_stale_empty_sample(png_filename, txt_filename)
                continue

            padded_tgt.save(FRAMES_PNG_DIR / png_filename)

            try:
                info = get_frame_semantic_info(f_idx, "8x12")
                caption = (
                    f"pixel art of {char_name}, {variant_desc}, "
                    f"facing {info['direction']}, {info['action_desc']}, "
                    f"{info['sub_phase']}, 16-bit retro game style, "
                    f"clean transparent background, masterwork pixel art, crisp outline"
                )
            except Exception:
                caption = (
                    f"pixel art of {char_name}, {variant_desc}, "
                    f"frame {f_idx}, 16-bit retro game style, "
                    f"clean transparent background, masterwork pixel art"
                )

            with open(CAPTIONS_TXT_DIR / txt_filename, "w", encoding="utf-8") as f_txt:
                f_txt.write(caption)

            arr_t  = np.array(padded_tgt).astype(np.float32)
            mask_t = arr_t[:, :, 3] > 20
            rgb_t  = arr_t[:, :, :3]
            rgb_t[~mask_t] = 0.0
            norm_rgb_t   = rgb_t / 127.5 - 1.0
            norm_alpha_t = np.where(mask_t, 1.0, -1.0).astype(np.float32)[:, :, np.newaxis]
            rgba_t = np.concatenate([norm_rgb_t, norm_alpha_t], axis=-1)
            target_tensor = torch.from_numpy(rgba_t).permute(2, 0, 1).float()

            dataset_samples.append({
                "front_tensor":  front_tensor,
                "frame_idx":     f_idx,
                "target_tensor": target_tensor,
                "char_id":       char_var_id,
                "png_name":      png_filename,
            })
            char_frames += 1
            total_cut_frames += 1

        print(f"    -> {char_frames} frames procesados")

    print("\n" + "=" * 70)
    print(f"  RESUMEN:")
    print(f"    Variantes procesadas: {len(variants) - skipped_variants} / {len(variants)}")
    print(f"    Total muestras de entrenamiento: {total_cut_frames}")
    if skipped_variants:
        print(f"    Variantes saltadas (sin frontal/frames): {skipped_variants}")
    print("\n  Guardando cache optimizada para PyTorch...")
    cache_path = DATASET_OUT / "supervised_cache_8x12.pt"
    torch.save(dataset_samples, cache_path)
    size_mb = round(cache_path.stat().st_size / (1024 * 1024), 1)
    print(f"  [OK] Cache guardada: {cache_path.relative_to(PROJECT_ROOT)} ({size_mb} MB)")
    mark_dataset_layout_updated(len(dataset_samples))
    print("  [OK] training_status.json reseteado a epoca 0.")
    print("       Listo para nuevo entrenamiento con TODOS los personajes.")
    print("=" * 70)
    return dataset_samples


if __name__ == "__main__":
    build_supervised_dataset()