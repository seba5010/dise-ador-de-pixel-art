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

FRAMES_PNG_DIR.mkdir(parents=True, exist_ok=True)
CAPTIONS_TXT_DIR.mkdir(parents=True, exist_ok=True)
FRONTS_DIR.mkdir(parents=True, exist_ok=True)


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

    status_data["status"] = "DATASET_ACTUALIZADO"
    status_data["dataset_layout"] = {
        "version": 4,
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
        "Sprites ampliados y cache regenerada. Inicia una nueva era; no reanudes la métrica anterior."
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

# Plantilla de poses canonica 8x12
TEMPLATE_PATH = PROJECT_ROOT / "dataset_moldes" / "plantilla de los spritesheets.png"
if not TEMPLATE_PATH.exists():
    TEMPLATE_PATH = PROJECT_ROOT / "plantilla de los spritesheets.png"

tmpl_mgr = TemplateManager(
    template_path=TEMPLATE_PATH,
    target_size=MODEL_RESOLUTION,
    rows=12,
    cols=8,
    canvas_w=1024,
    canvas_h=1536
)

CHARACTERS = ["alex", "amaro", "belial", "conny", "dana"]
VARIANTS = [
    ("rnormal", "ropa normal casual"),
    ("rbchef", "uniforme de chef blanco con delantal"),
    ("rnchef", "uniforme de chef negro con delantal")
]

def build_supervised_dataset():
    print("=" * 70)
    print("  EXTRACCION Y ALINEACION QUIRURGICA DE FRAMES GROUND-TRUTH")
    print(f"  Personajes: {', '.join(CHARACTERS)}")
    print(f"  Destino: {DATASET_OUT}")
    print("=" * 70)

    dataset_samples = []
    total_cut_frames = 0

    p_dir = PROJECT_ROOT / "personajes"
    if not p_dir.exists():
        p_dir = PROJECT_ROOT.parent / "personajes"

    for char_name in CHARACTERS:
        cdir = p_dir / char_name
        if not cdir.exists():
            print(f"[!] No existe directorio para {char_name}: {cdir}")
            continue

        for var_key, var_desc in VARIANTS:
            # Buscar hoja de sprites correspondiente
            possible_sheet_names = [
                f"movimientos_{var_key}.png",
                f"movimiento_{var_key}.png"
            ]
            sheet_path = next((cdir / s for s in possible_sheet_names if (cdir / s).exists()), None)
            
            # Buscar imagen frontal de referencia correspondiente
            possible_front_names = [
                f"{char_name}_{var_key}.png",
                f"{char_name}_{var_key}_frente.png"
            ]
            front_path = next((cdir / f for f in possible_front_names if (cdir / f).exists()), None)

            if not sheet_path or not front_path:
                continue

            print(f"\n[*] Procesando: {char_name.upper()} ({var_key})")
            print(f"    Frontal: {front_path.name}")
            print(f"    Hoja:    {sheet_path.name}")

            # 1. Acondicionar imagen frontal con Adaptador Chibi
            raw_front = Image.open(front_path).convert("RGBA")
            chibi_front = adapt_front_to_chibi(raw_front, target_size=MODEL_RESOLUTION)
            chibi_front_path = FRONTS_DIR / f"{char_name}_{var_key}_chibi_front.png"
            chibi_front.save(chibi_front_path)

            # Normalizar tensor frontal [-1, 1]
            arr_f = np.array(chibi_front).astype(np.float32)
            mask_f = arr_f[:, :, 3] > 20
            rgb_f = arr_f[:, :, :3]
            rgb_f[~mask_f] = 0.0
            front_tensor = torch.from_numpy(rgb_f / 127.5 - 1.0).permute(2, 0, 1).float()

            # 2. Cargar hoja de movimientos
            sheet_img = Image.open(sheet_path).convert("RGBA")
            sw, sh = sheet_img.size
            ratio = sw / max(1, sh)
            is_8x12 = ratio >= 0.45

            char_var_id = f"{char_name}_{var_key}"

            if is_8x12:
                # 8x12: 96 frames
                for r in range(12):
                    for c in range(8):
                        f_idx = r * 8 + c
                        x0, y0, x1, y1 = get_cell_coordinates(sw, sh, r, c, 12, 8)
                        cell = sheet_img.crop((x0, y0, x1, y1))
                        padded_tgt = pad_target_frame_canonical(cell, target_size=MODEL_RESOLUTION)

                        png_filename = f"{char_var_id}_frame_{f_idx:03d}.png"
                        txt_filename = f"{char_var_id}_frame_{f_idx:03d}.txt"
                        if not has_trainable_content(padded_tgt):
                            remove_stale_empty_sample(png_filename, txt_filename)
                            continue

                        # Guardar PNG limpio
                        padded_tgt.save(FRAMES_PNG_DIR / png_filename)

                        # Generar caption descriptivo para entrenamiento LoRA
                        info = get_frame_semantic_info(f_idx, "8x12")
                        caption = (
                            f"pixel art of {char_name}, {var_desc}, facing {info['direction']}, "
                            f"{info['action_desc']}, {info['sub_phase']}, 16-bit retro game style, "
                            f"clean transparent background, masterwork pixel art, crisp outline"
                        )
                        with open(CAPTIONS_TXT_DIR / txt_filename, "w", encoding="utf-8") as f_txt:
                            f_txt.write(caption)

                        # Crear tensor objetivo [-1, 1] con canal alfa
                        arr_t = np.array(padded_tgt).astype(np.float32)
                        mask_t = arr_t[:, :, 3] > 20
                        rgb_t = arr_t[:, :, :3]
                        rgb_t[~mask_t] = 0.0
                        norm_rgb_t = rgb_t / 127.5 - 1.0
                        norm_alpha_t = np.where(mask_t, 1.0, -1.0).astype(np.float32)[:, :, np.newaxis]
                        rgba_t = np.concatenate([norm_rgb_t, norm_alpha_t], axis=-1)
                        target_tensor = torch.from_numpy(rgba_t).permute(2, 0, 1).float()

                        dataset_samples.append({
                            "front_tensor": front_tensor,
                            "frame_idx": f_idx,
                            "target_tensor": target_tensor,
                            "char_id": char_var_id,
                            "png_name": png_filename
                        })
                        total_cut_frames += 1
            else:
                # 16x4: Mapear 48 frames de caminatas/giros a la grilla 8x12
                for r in range(16):
                    for c in range(4):
                        mapped_idx = get_8x12_index_from_16x4(r, c)
                        if mapped_idx < 0:
                            continue

                        x0, y0, x1, y1 = get_cell_coordinates(sw, sh, r, c, 16, 4)
                        cell = sheet_img.crop((x0, y0, x1, y1))
                        padded_tgt = pad_target_frame_canonical(cell, target_size=MODEL_RESOLUTION)

                        png_filename = f"{char_var_id}_frame_{mapped_idx:03d}.png"
                        txt_filename = f"{char_var_id}_frame_{mapped_idx:03d}.txt"
                        if not has_trainable_content(padded_tgt):
                            remove_stale_empty_sample(png_filename, txt_filename)
                            continue

                        padded_tgt.save(FRAMES_PNG_DIR / png_filename)

                        info = get_frame_semantic_info(mapped_idx, "8x12")
                        caption = (
                            f"pixel art of {char_name}, {var_desc}, facing {info['direction']}, "
                            f"{info['action_desc']}, {info['sub_phase']}, 16-bit retro game style, "
                            f"clean transparent background, masterwork pixel art, crisp outline"
                        )
                        with open(CAPTIONS_TXT_DIR / txt_filename, "w", encoding="utf-8") as f_txt:
                            f_txt.write(caption)

                        arr_t = np.array(padded_tgt).astype(np.float32)
                        mask_t = arr_t[:, :, 3] > 20
                        rgb_t = arr_t[:, :, :3]
                        rgb_t[~mask_t] = 0.0
                        norm_rgb_t = rgb_t / 127.5 - 1.0
                        norm_alpha_t = np.where(mask_t, 1.0, -1.0).astype(np.float32)[:, :, np.newaxis]
                        rgba_t = np.concatenate([norm_rgb_t, norm_alpha_t], axis=-1)
                        target_tensor = torch.from_numpy(rgba_t).permute(2, 0, 1).float()

                        dataset_samples.append({
                            "front_tensor": front_tensor,
                            "frame_idx": mapped_idx,
                            "target_tensor": target_tensor,
                            "char_id": char_var_id,
                            "png_name": png_filename
                        })
                        total_cut_frames += 1

    print("\n" + "=" * 70)
    print(f"  RESUMEN: {total_cut_frames} frames cortados y emparejados.")
    print("  Guardando cache optimizada para PyTorch...")
    cache_path = DATASET_OUT / "supervised_cache_8x12.pt"
    torch.save(dataset_samples, cache_path)
    print(f"  [OK] Cache guardada en: {cache_path} ({round(cache_path.stat().st_size / (1024*1024), 1)} MB)")
    mark_dataset_layout_updated(len(dataset_samples))
    print("  [OK] Estado marcado como DATASET_ACTUALIZADO: inicia una nueva era de entrenamiento.")
    print("=" * 70)
    return dataset_samples

if __name__ == "__main__":
    build_supervised_dataset()
