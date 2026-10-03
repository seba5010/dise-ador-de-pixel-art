"""
Sprite Studio - Unified Local Application Coordinator (sprite_studio.py)
Servidor local multi-hilo con API REST completa para el control de:
- Generacion de Spritesheets (Motor PyTorch UNet y Motor Forge SD1.5/ControlNet)
- Modo 4 Poses de Prueba y Modo Hoja Completa (96 frames 8x12 y 64 frames 16x4)
- Regeneracion quirurgica de frames individuales
- Reproductor de Animaciones en tiempo real
- Auditoria de Calidad (Quality Gate)
- Monitoreo en vivo de entrenamiento protegido (sin colision de GPU)
- Exportacion limpia para Unity
"""

import os
import sys
import time
import json
import socket
import urllib.parse
import urllib.request
import requests
import threading
import subprocess
import traceback
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image, ImageDraw
import numpy as np

# Configurar path local
SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.name.lower() == "stable-diffusion-webui-forge-main":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

# Importar modulos del motor
from pixel_ai_engine.config import (
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    MODEL_RESOLUTION,
    DEVICE,
    USE_AMP,
    get_phase_config
)
from pixel_ai_engine.dataset import (
    TemplateManager,
    adapt_front_to_chibi,
    pad_target_frame_canonical,
    place_in_cell,
    get_cell_coordinates
)
from pixel_ai_engine.models import PixelArtUNetGenerator
from pixel_ai_engine.enhancer import PixelArtEnhancer
from pixel_ai_engine.frame_map import (
    FORMAT_8X12,
    FORMAT_16X4,
    get_frame_semantic_info,
    get_all_frame_mappings
)

PORT = 8080
ACTIVE_JOB = {
    "job_id": None,
    "status": "idle",
    "progress": 0.0,
    "current_frame": 0,
    "total_frames": 0,
    "message": "",
    "logs": [],
    "run_dir": None,
    "error": None
}
JOB_LOCK = threading.Lock()


def get_free_port(start_port: int = 8080) -> int:
    """Busca un puerto libre a partir de start_port."""
    port = start_port
    while port < start_port + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start_port


def check_gpu_training_status() -> Tuple[bool, Dict[str, Any]]:
    """
    Verifica si hay un entrenamiento activo en la GPU para protegerlo
    y evitar colisiones de memoria VRAM (OOM).
    """
    proc_running = False
    proc_info = None
    try:
        import psutil
        for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            cmd = ' '.join(p.info.get('cmdline') or [])
            if 'train_forge_lora.py' in cmd or 'train_supervised.py' in cmd or 'train.py' in cmd:
                proc_running = True
                proc_info = p.info
                break
    except Exception:
        pass

    status_file = PROJECT_ROOT / "training_status.json"
    if status_file.exists():
        try:
            mtime = os.path.getmtime(status_file)
            age = time.time() - mtime
            with open(status_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            st = data.get("status", "")
            is_active = proc_running or ((age < 600) and (st in ["ENTRENANDO", "PAUSANDO", "DETENIENDO"]))
            if proc_running:
                data["status"] = "ENTRENANDO"
                # Si el proceso está corriendo, calcular tiempo real transcurrido
                if proc_info and "create_time" in proc_info:
                    live_elapsed = round(time.time() - proc_info["create_time"], 1)
                    if live_elapsed > data.get("elapsed_sec", 0):
                        data["elapsed_sec"] = live_elapsed
            return is_active, data
        except Exception:
            pass
    return proc_running, {"status": "ENTRENANDO" if proc_running else "IDLE"}


def check_forge_status(forge_url: str = "http://127.0.0.1:7860") -> Tuple[bool, str]:
    """Comprueba la disponibilidad del motor. Forge ha sido desactivado en favor de PyTorch UNet."""
    return False, "Desactivado (Motor PyTorch Canónico Activo)"


def scan_available_characters() -> List[Dict[str, Any]]:
    """Escanea todos los personajes y sus imagenes frontales disponibles."""
    chars = []
    seen_names = set()
    p_dirs = []
    for d in [PROJECT_ROOT / "personajes", PROJECT_ROOT.parent / "personajes"]:
        if d.exists() and d.is_dir():
            for c in d.iterdir():
                if c.is_dir() and c.name.lower() not in ["_drop", "_pruebas_y_obsoletos", ".git"]:
                    if c.name.lower() not in seen_names:
                        seen_names.add(c.name.lower())
                        p_dirs.append(c)

    for cdir in sorted(p_dirs, key=lambda x: x.name.lower()):
        files = list(cdir.glob("*.png"))
        fronts = []
        sheets = []
        for f in files:
            rel = os.path.relpath(f, PROJECT_ROOT).replace("\\", "/")
            if "movimiento" in f.name.lower():
                sheets.append({"filename": f.name, "path": rel})
            else:
                fronts.append({"filename": f.name, "path": rel})

        # Ordenar para que rnormal aparezca primero si existe
        fronts.sort(key=lambda x: (0 if "rnormal" in x["filename"].lower() else 1, x["filename"]))

        chars.append({
            "name": cdir.name,
            "folder": cdir.name,
            "fronts": fronts,
            "sheets": sheets,
            "default_front": fronts[0]["path"] if fronts else None,
            "has_ground_truth": len(sheets) > 0
        })
    return chars


def scan_checkpoints() -> Dict[str, List[Dict[str, Any]]]:
    """Escanea los checkpoints disponibles y los clasifica por formato en todos los directorios."""
    ckpts = {"8x12": [], "16x4": []}
    seen = set()
    dirs = [
        PROJECT_ROOT / "checkpoints",
        PROJECT_ROOT / "checkpoints" / "snapshots",
        PROJECT_ROOT / "checkpoints" / "supervised",
        SCRIPT_DIR / "checkpoints",
        PROJECT_ROOT.parent / "checkpoints"
    ]
    for d in dirs:
        if d.exists() and d.is_dir():
            for f in sorted(list(d.glob("*.pt")), key=lambda x: x.stat().st_mtime, reverse=True):
                if "cache" in f.name.lower():
                    continue
                if f.name in seen:
                    continue
                seen.add(f.name)
                sz_mb = round(f.stat().st_size / (1024 * 1024), 1)
                entry = {
                    "filename": f.name,
                    "path": str(f),
                    "size_mb": sz_mb,
                    "is_best": "best" in f.name.lower() or "latest" in f.name.lower()
                }
                if "16x4" in f.name.lower():
                    ckpts["16x4"].append(entry)
                else:
                    ckpts["8x12"].append(entry)
    return ckpts


# ============================================================================
# MOTOR DE INFERENCIA PYTORCH (UNet + Adaptador Chibi + Enhancer Quirurgico)
# ============================================================================

def run_pytorch_generation(
    character_name: str,
    front_image_path: Path,
    format_type: str = "8x12",
    mode: str = "full",
    single_frame_idx: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    alpha_threshold: int = 60
) -> Path:
    """
    Ejecuta la generacion de frames utilizando el generador PyTorch entrenado.
    Soporta:
      - Modo 'full': Todos los 96 frames (u 64 en 16x4)
      - Modo 'test_4poses': 4 poses de prueba canonicas (Frente, Perfil, Espalda, Accion)
      - Modo 'single_frame': Regenera unicamente el frame especificado
    """
    import torch
    from torch.cuda.amp import autocast

    phase_cfg = get_phase_config("2" if format_type == "8x12" else "1")
    cols = phase_cfg["grid_cols"]
    rows = phase_cfg["grid_rows"]
    total_frames = phase_cfg["total_frames"]
    canvas_w = phase_cfg["canvas_w"]
    canvas_h = phase_cfg["canvas_h"]
    cell_w = phase_cfg["cell_w"]
    cell_h = phase_cfg["cell_h"]

    # Determinar frames a generar
    if mode == "single_frame":
        target_frame_indices = [single_frame_idx]
    elif mode == "test_4poses":
        target_frame_indices = FORMAT_8X12["test_pose_indices"] if format_type == "8x12" else FORMAT_16X4["test_pose_indices"]
    else:
        target_frame_indices = list(range(total_frames))

    # Preparar directorio de ejecucion
    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUT_DIR / f"{character_name}_{format_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw_frames"
    enh_dir = run_dir / "enhanced_frames"
    raw_dir.mkdir(exist_ok=True)
    enh_dir.mkdir(exist_ok=True)
    with JOB_LOCK:
        ACTIVE_JOB["run_dir"] = str(run_dir.relative_to(PROJECT_ROOT)).replace(chr(92), "/")
        ACTIVE_JOB["character"] = character_name
        ACTIVE_JOB["format"] = format_type

    # 1. Cargar Checkpoint
    if checkpoint_path is None or not checkpoint_path.exists():
        # Auto-seleccion inteligente de checkpoint
        if format_type == "8x12":
            candidates = [
                CHECKPOINT_DIR / "best_generator.pt",
                CHECKPOINT_DIR / "latest_checkpoint.pt",
                CHECKPOINT_DIR / "base_generator_16x4.pt"
            ]
        else:
            candidates = [
                CHECKPOINT_DIR / "base_generator_16x4.pt",
                CHECKPOINT_DIR / "latest_checkpoint_16x4.pt"
            ]
        checkpoint_path = next((c for c in candidates if c.exists()), None)

    if checkpoint_path is None:
        raise FileNotFoundError(f"No se encontro ningun checkpoint compatible para formato {format_type}.")

    # 2. Cargar y acondicionar imagen frontal con Adaptador Chibi
    if front_image_path.is_dir() or not front_image_path.is_file():
        found = None
        if front_image_path.is_dir():
            for p in front_image_path.glob("*.png"):
                if "front" in p.name.lower() or "chibi" in p.name.lower():
                    found = p
                    break
            if not found:
                pngs = list(front_image_path.glob("*.png"))
                if pngs: found = pngs[0]
        if not found:
            for cand in [PROJECT_ROOT / "personajes" / "alex" / "alex_front.png", PROJECT_ROOT / "personajes" / "alex_front.png"]:
                if cand.exists(): found = cand; break
        if found:
            front_image_path = found
        else:
            raise FileNotFoundError(f"No se encontró archivo de imagen frontal válido: {front_image_path}")

    raw_front = Image.open(front_image_path)
    chibi_front = adapt_front_to_chibi(raw_front, target_size=MODEL_RESOLUTION)
    
    # Extraer paleta canonica para el Enhancer
    canonical_palette = PixelArtEnhancer.extract_palette(chibi_front, max_colors=40)

    # Normalizar tensor frontal [-1, 1] con cielo limpio
    arr_f = np.array(chibi_front).astype(np.float32)
    mask_f = arr_f[:, :, 3] > 20
    rgb_f = arr_f[:, :, :3]
    rgb_f[~mask_f] = 0.0
    front_tensor = torch.from_numpy(rgb_f / 127.5 - 1.0).permute(2, 0, 1).float().to(DEVICE)

    # 3. Cargar Plantilla de Poses
    tmpl_mgr = TemplateManager(
        template_path=phase_cfg["template_path"],
        target_size=MODEL_RESOLUTION,
        rows=rows,
        cols=cols,
        canvas_w=canvas_w,
        canvas_h=canvas_h
    )

    # 4. Cargar Generador
    generator = PixelArtUNetGenerator().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "generator" in ckpt:
        generator.load_state_dict(ckpt["generator"])
    else:
        generator.load_state_dict(ckpt)
    generator.eval()

    # 5. Inferencia frame por frame
    total_to_generate = len(target_frame_indices)
    with torch.no_grad():
        for i, f_idx in enumerate(target_frame_indices):
            with JOB_LOCK:
                ACTIVE_JOB["progress"] = round((i / max(1, total_to_generate)) * 100, 1)
                ACTIVE_JOB["current_frame"] = f_idx
                ACTIVE_JOB["total_frames"] = total_frames
                info = get_frame_semantic_info(f_idx, format_type)
                ACTIVE_JOB["message"] = f"Generando frame {f_idx + 1:02d}/{total_frames} ({info['action_desc']})..."
                ACTIVE_JOB["logs"].append(f"[PyTorch] Frame {f_idx:02d} ({info['action_desc']}) procesado.")

            pose_tensor = tmpl_mgr.get_frame_tensor(f_idx).to(DEVICE)
            cond = torch.cat([front_tensor, pose_tensor], dim=0).unsqueeze(0)  # (1, 6, H, W)

            with autocast(enabled=USE_AMP):
                output_tensor = generator(cond).squeeze(0)  # (4, H, W)

            arr = output_tensor.detach().cpu().permute(1, 2, 0).numpy()
            arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
            raw_frame_pil = Image.fromarray(arr, mode="RGBA")
            raw_frame_pil.save(raw_dir / f"frame_{f_idx:03d}.png")

            # Aplicar Enhancer Quirurgico
            enh_frame_pil = PixelArtEnhancer.enhance_frame(
                raw_frame_pil,
                palette=canonical_palette,
                snap_palette=True,
                remove_noise=True,
                binarize=True,
                sharpen_tattoos=True
            )
            enh_frame_pil.save(enh_dir / f"frame_{f_idx:03d}.png")

    # 6. Ensamblado del Spritesheet completo (si existen todos los frames o los que haya)
    reassemble_spritesheet(run_dir, format_type)

    # 7. Actualizar metadatos
    save_run_metadata(
        run_dir=run_dir,
        character=character_name,
        front_image=str(front_image_path),
        engine="pytorch",
        format_type=format_type,
        checkpoint=str(checkpoint_path),
        mode=mode
    )

    with JOB_LOCK:
        ACTIVE_JOB["progress"] = 100.0
        ACTIVE_JOB["status"] = "completed"
        ACTIVE_JOB["run_dir"] = f"output/{run_dir.name}"
        ACTIVE_JOB["message"] = f"¡Generacion finalizada exitosamente! Guardado en: {run_dir.name}"

    return run_dir



def run_forge_generation(
    character_name: str,
    front_image_path: Path,
    format_type: str = "8x12",
    mode: str = "full",
    single_frame_idx: int = 0,
    seed: Optional[int] = None,
    run_dir: Optional[Path] = None
) -> Path:
    """Ejecuta generacion delegada a Forge SD1.5 + LoRA entrenado via API local 127.0.0.1:7860."""
    import base64
    import io
    cfg = FORMAT_8X12 if format_type == "8x12" else FORMAT_16X4
    total_frames = cfg["total_frames"]

    if mode == "single_frame":
        target_indices = [single_frame_idx]
    elif mode == "test_4poses":
        target_indices = [0, 16, 32, 92] if format_type == "8x12" else [0, 8, 16, 24]
    else:
        target_indices = list(range(total_frames))

    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUT_DIR / f"{character_name}_{format_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    enh_dir = run_dir / "enhanced_frames"
    raw_dir = run_dir / "raw_frames"
    enh_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)

    with JOB_LOCK:
        ACTIVE_JOB["run_dir"] = str(run_dir.relative_to(PROJECT_ROOT)).replace(chr(92), "/")
        ACTIVE_JOB["character"] = character_name
        ACTIVE_JOB["format"] = format_type

    # Determinar descripcion de vestimenta segun el archivo frontal
    fname_lower = front_image_path.name.lower()
    if "rbchef" in fname_lower:
        outfit_desc = "uniforme de chef blanco con delantal"
    elif "rnchef" in fname_lower:
        outfit_desc = "uniforme de chef negro con delantal"
    elif "rnormal" in fname_lower:
        outfit_desc = "ropa normal casual comensal"
    else:
        outfit_desc = "uniforme de chef"

    # Refrescar LoRAs en Forge para asegurar que el ultimo checkpoint esta cargado
    try:
        requests.post("http://127.0.0.1:7860/sdapi/v1/refresh-loras", json={}, timeout=5)
    except Exception:
        pass

    for i, f_idx in enumerate(target_indices):
        with JOB_LOCK:
            ACTIVE_JOB["progress"] = round((i / max(1, len(target_indices))) * 100, 1)
            ACTIVE_JOB["current_frame"] = f_idx
            ACTIVE_JOB["total_frames"] = total_frames
            info = get_frame_semantic_info(f_idx, format_type)
            ACTIVE_JOB["message"] = f"[Forge SD1.5] Generando frame {f_idx + 1:02d}/{total_frames} ({info['action_desc']})..."

        prompt = (
            f"<lora:villa_del_chef_characters:0.8> pixel art of {character_name}, {outfit_desc}, "
            f"{info['action_desc']}, facing {info['direction']}, 16-bit retro game style, "
            f"clean transparent background, masterwork pixel art, crisp outline"
        )
        payload = {
            "prompt": prompt,
            "negative_prompt": "blurry, realistic, photographic, 3d render, vector, noisy, deformed, extra limbs, low quality",
            "steps": 20,
            "width": 512,
            "height": 512,
            "cfg_scale": 7.0,
            "sampler_name": "Euler a"
        }
        if seed is not None and str(seed).strip() != "":
            try:
                payload["seed"] = int(seed)
            except Exception:
                pass

        res = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload, timeout=60)
        if res.status_code == 200:
            img_data = res.json()["images"][0]
            gen_img_512 = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("RGBA")
            # Downsampling con remuestreo NEAREST a resolucion nativa 128x128
            gen_img_128 = gen_img_512.resize((128, 128), Image.Resampling.NEAREST)
            gen_img_128.save(raw_dir / f"frame_{f_idx:03d}.png")
            gen_img_128.save(enh_dir / f"frame_{f_idx:03d}.png")

    if mode != "single_frame":
        reassemble_spritesheet(run_dir, format_type)

    save_run_metadata(
        run_dir, character_name, str(front_image_path), "forge",
        format_type, "villa_del_chef_characters.safetensors", mode, seed
    )
    with JOB_LOCK:
        ACTIVE_JOB["status"] = "completed"
        ACTIVE_JOB["progress"] = 100.0
        ACTIVE_JOB["run_dir"] = f"output/{run_dir.name}"
        ACTIVE_JOB["message"] = f"Generacion Forge finalizada! Guardado en: {run_dir.name}"
    return run_dir

def reassemble_spritesheet(run_dir: Path, format_type: str = "8x12"):
    """
    Ensambla la hoja completa de spritesheet a partir de los frames en enhanced_frames/.
    Genera dos versiones:
      - spritesheet_clean.png: 100% transparente sin lineas de cuadricula.
      - spritesheet_grid.png: con lineas guia de cuadricula para revision de alineacion.
    """
    phase_cfg = get_phase_config("2" if format_type == "8x12" else "1")
    cols = phase_cfg["grid_cols"]
    rows = phase_cfg["grid_rows"]
    canvas_w = phase_cfg["canvas_w"]
    canvas_h = phase_cfg["canvas_h"]

    enh_dir = run_dir / "enhanced_frames"
    canvas_clean = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    canvas_grid = Image.new("RGBA", (canvas_w, canvas_h), (220, 224, 232, 255))
    draw_grid = ImageDraw.Draw(canvas_grid)

    # Dibujar cuadricula en canvas_grid
    for r in range(rows):
        for c in range(cols):
            x0, y0, x1, y1 = get_cell_coordinates(canvas_w, canvas_h, r, c, rows, cols)
            draw_grid.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(180, 185, 195, 255), width=1)

    for r in range(rows):
        for c in range(cols):
            f_idx = r * cols + c
            frame_path = enh_dir / f"frame_{f_idx:03d}.png"
            if frame_path.exists():
                frame_img = Image.open(frame_path).convert("RGBA")
                x0, y0, x1, y1 = get_cell_coordinates(canvas_w, canvas_h, r, c, rows, cols)
                cell_w = x1 - x0
                cell_h = y1 - y0
                cell_placed = place_in_cell(frame_img, cell_w=cell_w, cell_h=cell_h)
                canvas_clean.paste(cell_placed, (x0, y0), cell_placed)
                canvas_grid.paste(cell_placed, (x0, y0), cell_placed)

    canvas_clean.save(run_dir / "spritesheet_clean.png", format="PNG")
    canvas_grid.save(run_dir / "spritesheet_grid.png", format="PNG")

    # Generar metadata para Unity 2D Sprite Editor
    export_unity_metadata(run_dir, format_type)


def export_unity_metadata(run_dir: Path, format_type: str = "8x12"):
    """Exporta archivo JSON con la definicion de celdas y clips de animacion para Unity."""
    cfg = FORMAT_8X12 if format_type == "8x12" else FORMAT_16X4
    cols = cfg["cols"]
    rows = cfg["rows"]
    canvas_w = cfg["canvas_width"]
    canvas_h = cfg["canvas_height"]

    sprites = []
    for r in range(rows):
        for c in range(cols):
            f_idx = r * cols + c
            x0, y0, x1, y1 = get_cell_coordinates(canvas_w, canvas_h, r, c, rows, cols)
            info = get_frame_semantic_info(f_idx, format_type)
            sprites.append({
                "name": f"frame_{f_idx:03d}",
                "index": f_idx,
                "rect": {"x": x0, "y": canvas_h - y1, "width": x1 - x0, "height": y1 - y0},  # Unity Y bottom-left
                "pivot": {"x": 0.5, "y": 0.05},  # anclado a los pies
                "semantic": {
                    "direction": info["direction"],
                    "action": info["action"],
                    "sub_phase": info["sub_phase"]
                }
            })

    meta = {
        "format": format_type,
        "canvas_size": {"width": canvas_w, "height": canvas_h},
        "pixels_per_unit": 16,
        "filter_mode": "Point",
        "sprites": sprites,
        "animations": cfg["animation_clips"]
    }

    with open(run_dir / "unity_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def save_run_metadata(
    run_dir: Path,
    character: str,
    front_image: str,
    engine: str,
    format_type: str,
    checkpoint: str,
    mode: str,
    seed: Optional[int] = None
):
    """Guarda o actualiza el registro metadata.json de la ejecucion."""
    meta_path = run_dir / "metadata.json"
    enh_dir = run_dir / "enhanced_frames"
    total_expected = 96 if format_type == "8x12" else 64
    present_frames = [f.name for f in enh_dir.glob("frame_*.png")]
    is_complete = len(present_frames) == total_expected

    meta = {
        "character": character,
        "front_image": front_image,
        "engine": engine,
        "format": format_type,
        "checkpoint": checkpoint,
        "mode": mode,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_frames_expected": total_expected,
        "frames_generated_count": len(present_frames),
        "is_complete": is_complete,
        "is_draft": not is_complete
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def run_quality_audit(run_dir: Path, format_type: str = "8x12") -> Dict[str, Any]:
    """Auditoria quirurgica completa del spritesheet generado o en borrador con diagnostico en vivo."""
    run_dir = Path(run_dir)
    meta = {}
    meta_file = run_dir / "metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if "format" in meta:
                    format_type = meta["format"]
        except Exception:
            pass

    enh_dir = run_dir / "enhanced_frames"
    if not enh_dir.exists() and (run_dir / "raw_frames").exists():
        enh_dir = run_dir / "raw_frames"

    cfg = FORMAT_8X12 if format_type == "8x12" else FORMAT_16X4
    total = cfg["total_frames"]

    empty_cells = []
    border_touch_cells = []
    blurry_alpha_cells = []
    valid_frames = 0

    for f_idx in range(total):
        f_path = enh_dir / f"frame_{f_idx:03d}.png"
        info = get_frame_semantic_info(f_idx, format_type)
        desc = info.get("action_desc", f"Frame {f_idx}")
        action = info.get("action", "")
        direction = info.get("direction", "")

        if not f_path.exists():
            empty_cells.append({
                "frame": f_idx,
                "desc": desc,
                "action": action,
                "direction": direction
            })
            continue

        valid_frames += 1
        try:
            im = Image.open(f_path).convert("RGBA")
            arr = np.array(im)
            alpha = arr[:, :, 3]

            # Verificar celda vacia
            if (alpha > 20).sum() < 10:
                empty_cells.append({
                    "frame": f_idx,
                    "desc": desc,
                    "action": action,
                    "direction": direction
                })
                continue

            # Verificar toque de bordes (sangrado a celdas contiguas)
            top_touch = (alpha[0, :] > 20).any()
            bottom_touch = (alpha[-1, :] > 20).any()
            left_touch = (alpha[:, 0] > 20).any()
            right_touch = (alpha[:, -1] > 20).any()
            if top_touch or bottom_touch or left_touch or right_touch:
                border_touch_cells.append({
                    "frame": f_idx,
                    "desc": desc,
                    "action": action,
                    "direction": direction,
                    "top": bool(top_touch),
                    "bottom": bool(bottom_touch),
                    "left": bool(left_touch),
                    "right": bool(right_touch)
                })

            # Verificar binaridad del canal alfa
            # En pixel art retro certificado para Unity, el alfa debe ser estrictamente binario (0 o 255).
            # Cualquier valor intermedio (1 <= alpha <= 254) representa semitransparencia no permitida.
            semi = int(((alpha > 0) & (alpha < 255)).sum())
            solid = int((alpha == 255).sum())
            is_blurry = False
            ratio_val = 0.0

            if solid == 0 and semi > 0:
                # Caso critico: Personaje completamente fantasmal / semitransparente sin ningun pixel opaco
                is_blurry = True
                ratio_val = 1.0
            elif solid > 0 and semi > 0:
                # Mezcla pixeles solidos con semitransparencias (halos, bordes difusos o antialiasing)
                is_blurry = True
                ratio_val = round(float(semi / solid), 3)

            if is_blurry:
                blurry_alpha_cells.append({
                    "frame": f_idx,
                    "desc": desc,
                    "ratio": ratio_val,
                    "semi_pixels": semi,
                    "solid_pixels": solid
                })
        except Exception as e:
            empty_cells.append({
                "frame": f_idx,
                "desc": desc,
                "action": action,
                "direction": direction,
                "error": str(e)
            })

    completeness_score = round((valid_frames / total) * 100, 1)
    alpha_purity_score = round(max(0, 100 - len(blurry_alpha_cells) * 1.5), 1)
    border_safety_score = round(max(0, 100 - len(border_touch_cells) * 2.0), 1)

    is_ready = (
        completeness_score == 100.0 and
        len(border_touch_cells) == 0 and
        len(empty_cells) == 0 and
        len(blurry_alpha_cells) == 0 and
        alpha_purity_score >= 95.0
    )
    audit_logs = [
        f"[{time.strftime('%H:%M:%S')}] Iniciando auditoria quirurgica de {run_dir.name}...",
        f"[{time.strftime('%H:%M:%S')}] Formato de spritesheet: {format_type} ({total} frames esperados).",
        f"[{time.strftime('%H:%M:%S')}] Escaneando celdas en {enh_dir.name}...",
        f"[{time.strftime('%H:%M:%S')}] Frames generados validos: {valid_frames}/{total} ({completeness_score}% completitud).",
        f"[{time.strftime('%H:%M:%S')}] Pureza de canal alfa: {alpha_purity_score}% ({len(blurry_alpha_cells)} celdas con semitransparencia borrosa).",
        f"[{time.strftime('%H:%M:%S')}] Seguridad de margenes (anti-sangrado): {border_safety_score}% ({len(border_touch_cells)} celdas tocando bordes)."
    ]
    if len(empty_cells) > 0:
        audit_logs.append(f"[{time.strftime('%H:%M:%S')}] AVISO: {len(empty_cells)} celdas vacias pendientes de generacion.")
    if len(border_touch_cells) > 0:
        audit_logs.append(f"[{time.strftime('%H:%M:%S')}] AVISO: {len(border_touch_cells)} celdas con sangrado en bordes.")
    if len(blurry_alpha_cells) > 0:
        audit_logs.append(f"[{time.strftime('%H:%M:%S')}] AVISO: {len(blurry_alpha_cells)} celdas con canal alfa semitransparente/borroso (no apto para Unity).")
    if is_ready:
        audit_logs.append(f"[{time.strftime('%H:%M:%S')}] VEREDICTO: SPRITESHEET CERTIFICADO 100% PARA UNITY (ALFA PURO Y SIN SANGRADO).")
    else:
        audit_logs.append(f"[{time.strftime('%H:%M:%S')}] VEREDICTO: BORRADOR / REVISION REQUERIDA ({completeness_score}% completado, {len(blurry_alpha_cells)} celdas con alfa defectuoso).")

    return {
        "run_id": run_dir.name,
        "format": format_type,
        "character": meta.get("character", run_dir.name.split("_")[0]),
        "total_frames_expected": total,
        "valid_frames_count": valid_frames,
        "completeness_score": completeness_score,
        "empty_cells": empty_cells,
        "border_touch_cells": border_touch_cells,
        "border_touching_cells": border_touch_cells,
        "blurry_alpha_cells": blurry_alpha_cells,
        "alpha_purity_score": alpha_purity_score,
        "border_safety_score": border_safety_score,
        "is_ready_for_game": is_ready,
        "certified": is_ready,
        "metadata": meta,
        "logs": audit_logs
    }


# ============================================================================
# SERVIDOR HTTP REST API (SpriteStudioHandler)
# ============================================================================

class SpriteStudioHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def log_message(self, format, *args):
        pass  # Silenciar logs ruidosos

    def send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/status":
            training_active, train_data = check_gpu_training_status()
            forge_active, forge_model = check_forge_status()
            
            gpu_name = "CPU"
            vram_gb = 0.0
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)

            self.send_json({
                "device": str(DEVICE),
                "gpu_name": gpu_name,
                "vram_gb": vram_gb,
                "training_active": training_active,
                "training_info": train_data,
                "forge_connected": forge_active,
                "forge_model": forge_model,
                "checkpoints": scan_checkpoints(),
                "active_job": ACTIVE_JOB
            })
            return

        elif path == "/api/characters":
            self.send_json(scan_available_characters())
            return

        elif path == "/api/pose_map":
            fmt = query.get("format", ["8x12"])[0]
            self.send_json({
                "format": fmt,
                "config": FORMAT_8X12 if fmt == "8x12" else FORMAT_16X4,
                "frames": get_all_frame_mappings(fmt)
            })
            return

        elif path == "/api/train/snapshots":
            snaps_dir = PROJECT_ROOT / "checkpoints" / "snapshots"
            snaps = []
            if snaps_dir.exists():
                for p in sorted(snaps_dir.glob("checkpoint_epoch_*.pt")):
                    try:
                        ep = int(p.stem.replace("checkpoint_epoch_", ""))
                        snaps.append({
                            "epoch": ep,
                            "file": p.name,
                            "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                            "preview_url": f"/training_samples/audit_history/preview_epoch_{ep:03d}.png"
                        })
                    except Exception:
                        pass
            self.send_json(snaps)
            return

        elif path == "/api/runs":
            runs = []
            if OUTPUT_DIR.exists():
                for d in sorted(OUTPUT_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                    if d.is_dir() and ((d / "enhanced_frames").exists() or (d / "raw_frames").exists() or (d / "spritesheet_clean.png").exists()):
                        meta = {}
                        meta_file = d / "metadata.json"
                        if meta_file.exists():
                            try:
                                with open(meta_file, "r", encoding="utf-8") as f:
                                    meta = json.load(f)
                            except Exception:
                                pass
                        runs.append({
                            "run_id": d.name,
                            "path": str(d.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                            "clean_sheet": f"output/{d.name}/spritesheet_clean.png" if (d / "spritesheet_clean.png").exists() else None,
                            "grid_sheet": f"output/{d.name}/spritesheet_grid.png" if (d / "spritesheet_grid.png").exists() else None,
                            "metadata": meta
                        })
            self.send_json(runs)
            return

        elif path == "/api/quality_audit":
            r_dir_str = query.get("run_dir", [""])[0]
            fmt = query.get("format", ["8x12"])[0]
            
            # Resolucion automatica de la ejecucion mas reciente o activa
            if not r_dir_str or r_dir_str.lower() in ["latest", "auto", "null", "undefined", ""]:
                with JOB_LOCK:
                    if ACTIVE_JOB.get("run_dir"):
                        r_dir_str = ACTIVE_JOB["run_dir"]
                if not r_dir_str or r_dir_str.lower() in ["latest", "auto", "null", "undefined", ""]:
                    if OUTPUT_DIR.exists():
                        for d in sorted(OUTPUT_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                            if d.is_dir():
                                enh_c = len(list((d / "enhanced_frames").glob("*.png"))) if (d / "enhanced_frames").exists() else 0
                                raw_c = len(list((d / "raw_frames").glob("*.png"))) if (d / "raw_frames").exists() else 0
                                if enh_c > 0 or raw_c > 0 or (d / "spritesheet_clean.png").exists():
                                    r_dir_str = str(d.relative_to(PROJECT_ROOT)).replace(chr(92), "/")
                                    break

            if not r_dir_str:
                self.send_json({"error": "No se encontraron ejecuciones para auditar"}, status=404)
                return

            # Normalizacion robusta de ruta (evita problemas con barras iniciales)
            r_clean = r_dir_str.lstrip("/\\").replace(chr(92), "/")
            if (PROJECT_ROOT / r_clean).exists():
                r_path = PROJECT_ROOT / r_clean
            elif Path(r_dir_str).exists():
                r_path = Path(r_dir_str)
            elif (OUTPUT_DIR / r_clean.replace("output/", "")).exists():
                r_path = OUTPUT_DIR / r_clean.replace("output/", "")
            else:
                r_path = PROJECT_ROOT / r_clean

            if r_path.exists() and r_path.is_dir():
                # Auto-ensamblar spritesheet si aun no existe para que el visor siempre tenga imagen
                clean_path = r_path / "spritesheet_clean.png"
                grid_path = r_path / "spritesheet_grid.png"
                if not clean_path.exists() or not grid_path.exists():
                    try:
                        reassemble_spritesheet(r_path, fmt)
                    except Exception as e:
                        print(f"[QC] Auto-ensamblado al auditar fallo: {e}")

                audit = run_quality_audit(r_path, fmt)
                norm_run_dir = str(r_path.relative_to(PROJECT_ROOT)).replace(chr(92), "/")
                audit["run_dir"] = norm_run_dir
                audit["clean_sheet"] = f"{norm_run_dir}/spritesheet_clean.png" if clean_path.exists() else None
                audit["grid_sheet"] = f"{norm_run_dir}/spritesheet_grid.png" if grid_path.exists() else None
                with JOB_LOCK:
                    is_active = (ACTIVE_JOB.get("run_dir") == norm_run_dir and ACTIVE_JOB.get("status") == "running")
                    audit["is_active_job"] = is_active
                    audit["job_status"] = ACTIVE_JOB.get("status", "idle")
                    audit["job_progress"] = ACTIVE_JOB.get("progress", 100.0) if is_active else 100.0
                    audit["job_message"] = ACTIVE_JOB.get("message", "") if is_active else ""
                self.send_json(audit)
            else:
                self.send_json({"error": f"Directorio no encontrado: {r_dir_str}"}, status=404)
            return

        elif path in ["/", "/index.html"]:
            self.path = "/sprite_studio.html"
            try:
                return super().do_GET()
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass

        # Normalizar rutas estaticas que vengan con prefijo Windows o subdirectorios
        unquoted = urllib.parse.unquote(path).replace("\\", "/")
        for prefix in ["output/", "personajes/", "dataset_moldes/", "training_samples/", "dataset_supervisado/"]:
            if prefix in unquoted:
                idx = unquoted.find(prefix)
                rel_file = unquoted[idx:]
                local_path = PROJECT_ROOT / rel_file
                if local_path.exists() and local_path.is_file():
                    self.path = "/" + rel_file
                    try:
                        return super().do_GET()
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                        return

        try:
            return super().do_GET()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len).decode("utf-8", errors="replace")
        try:
            body = json.loads(post_data) if post_data else {}
        except Exception:
            body = {}

        if parsed.path == "/api/generate":
            with JOB_LOCK:
                if ACTIVE_JOB["status"] == "running":
                    self.send_json({"error": "Ya hay un trabajo de generacion en curso."}, status=409)
                    return
                # Reiniciar estado para nueva generacion
                ACTIVE_JOB["error"] = None

                training_active, _ = check_gpu_training_status()
                engine = body.get("engine", "pytorch")
                
                # Advertencia de seguridad de GPU
                if training_active and engine == "pytorch" and str(DEVICE) == "cuda":
                    # Si el usuario no especifico forzar, avisar
                    if not body.get("force_during_training", False):
                        self.send_json({
                            "status": "gpu_busy",
                            "message": "Entrenamiento activo en la GPU. La generacion queda pausada para proteger el modelo.",
                        })
                        return

                ACTIVE_JOB["status"] = "running"
                ACTIVE_JOB["progress"] = 0.0
                ACTIVE_JOB["message"] = "Iniciando motor de generacion..."
                ACTIVE_JOB["logs"] = ["[Studio] Trabajo de generacion iniciado."]
                ACTIVE_JOB["error"] = None

            char_name = body.get("character", "character")
            front_path = PROJECT_ROOT / body.get("front_image", "")
            format_type = body.get("format", "8x12")
            mode = body.get("mode", "full")
            single_f_idx = body.get("single_frame_idx", None)
            ckpt_str = body.get("checkpoint", None)
            ckpt_path = Path(ckpt_str) if ckpt_str else None
            run_dir_str = body.get("run_dir", None)
            target_run_dir = (PROJECT_ROOT / run_dir_str) if run_dir_str else None

            def worker():
                try:
                    # Siempre ejecutar con motor canonico PyTorch UNet Supervisado
                    run_pytorch_generation(
                        character_name=char_name,
                        front_image_path=front_path,
                        format_type=format_type,
                        mode=mode,
                        single_frame_idx=single_f_idx,
                        checkpoint_path=ckpt_path,
                        run_dir=target_run_dir
                    )
                except Exception as e:
                    traceback.print_exc()
                    with JOB_LOCK:
                        ACTIVE_JOB["status"] = "failed"
                        ACTIVE_JOB["error"] = str(e)
                        ACTIVE_JOB["message"] = f"Error en generacion: {e}"
                        ACTIVE_JOB["logs"].append(f"[Error] {e}")

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            self.send_json({"status": "started", "message": "Generacion iniciada en segundo plano."})
            return

        elif parsed.path == "/api/train/start":
            training_active, _ = check_gpu_training_status()
            if training_active:
                self.send_json({"error": "Ya hay un entrenamiento activo en la GPU."}, status=409)
                return
            
            # Limpiar banderas anteriores
            stop_flag = PROJECT_ROOT / "stop_training.flag"
            pause_flag = PROJECT_ROOT / "pause_training.flag"
            if stop_flag.exists(): 
                try: stop_flag.unlink()
                except Exception: pass
            if pause_flag.exists(): 
                try: pause_flag.unlink()
                except Exception: pass

            epochs = int(body.get("epochs", 50))
            target_engine = "pytorch"  # Siempre PyTorch Supervisado
            batch_size = int(body.get("batch_size", 4))
            
            python_exe = sys.executable
            embedded_python = PROJECT_ROOT / "webui forger" / "system" / "python" / "python.exe"
            if not embedded_python.exists():
                embedded_python = PROJECT_ROOT.parent / "webui forger" / "system" / "python" / "python.exe"
            if embedded_python.exists():
                python_exe = str(embedded_python)

            if target_engine == "forge_lora":
                train_script = PROJECT_ROOT / "pixel_ai_engine" / "train_forge_lora.py"
                phase_title = "LoRA Forge SD1.5 (Alex, Amaro, Conny, Dana, Belial)"
                phase_num = 4
                msg_title = "LoRA para Forge SD1.5"
            else:
                train_script = PROJECT_ROOT / "pixel_ai_engine" / "train_supervised.py"
                phase_title = "Supervisado (Alex, Amaro, Conny, Dana, Belial)"
                phase_num = 3
                msg_title = "Supervisado PyTorch UNet"
            
            past_eras = []
            status_path = PROJECT_ROOT / "training_status.json"
            if status_path.exists():
                try:
                    with open(status_path, "r", encoding="utf-8") as f:
                        prev_data = json.load(f)
                    past_eras = prev_data.get("past_eras", [])
                    old_hist = prev_data.get("history", [])
                    if old_hist and len(old_hist) > 0:
                        era_idx = len(past_eras) + 1
                        past_eras.append({
                            "era": era_idx,
                            "name": f"Era {era_idx}",
                            "timestamp": prev_data.get("timestamp", time.strftime("%H:%M:%S")),
                            "epochs": prev_data.get("epoch", len(old_hist)),
                            "initial_loss": prev_data.get("initial_loss"),
                            "best_loss": prev_data.get("best_loss"),
                            "history": old_hist
                        })
                except Exception:
                    pass

            init_status = {
                "epoch": 1,
                "total_epochs": epochs,
                "phase": phase_title,
                "phase_id": phase_num,
                "status": "ENTRENANDO",
                "g_loss": "--",
                "d_loss": "--",
                "l1_loss": "--",
                "edge_loss": "--",
                "gpu_temp": 55,
                "elapsed_sec": 0,
                "timestamp": time.strftime("%H:%M:%S"),
                "total_frames": 1008,
                "past_eras": past_eras
            }
            try:
                with open(status_path, "w", encoding="utf-8") as f:
                    json.dump(init_status, f, indent=2)
            except Exception:
                pass

            subprocess.Popen([python_exe, str(train_script), "--epochs", str(epochs), "--batch_size", str(batch_size), "--mode", "start"], cwd=str(PROJECT_ROOT))
            self.send_json({"status": "started", "message": f"Entrenamiento de {msg_title} iniciado ({epochs} épocas)."})
            return

        elif parsed.path == "/api/train/pause":
            pause_flag = PROJECT_ROOT / "pause_training.flag"
            pause_flag.touch(exist_ok=True)
            status_file = PROJECT_ROOT / "training_status.json"
            if status_file.exists():
                try:
                    with open(status_file, "r", encoding="utf-8") as f:
                        cur = json.load(f)
                    cur["status"] = "PAUSANDO..."
                    with open(status_file, "w", encoding="utf-8") as f:
                        json.dump(cur, f, indent=2)
                except Exception:
                    pass
            self.send_json({"status": "pausing", "message": "Señal de pausa enviada. Guardando checkpoint y liberando GPU..."})
            return

        elif parsed.path == "/api/train/resume":
            training_active, _ = check_gpu_training_status()
            if training_active:
                self.send_json({"error": "El entrenamiento ya se encuentra en ejecución activa."}, status=409)
                return

            stop_flag = PROJECT_ROOT / "stop_training.flag"
            pause_flag = PROJECT_ROOT / "pause_training.flag"
            if stop_flag.exists(): 
                try: stop_flag.unlink()
                except Exception: pass
            if pause_flag.exists(): 
                try: pause_flag.unlink()
                except Exception: pass

            epochs = int(body.get("epochs", 50))
            target_engine = "pytorch"  # Siempre PyTorch Supervisado
            batch_size = int(body.get("batch_size", 4))
            
            python_exe = sys.executable
            embedded_python = PROJECT_ROOT / "webui forger" / "system" / "python" / "python.exe"
            if not embedded_python.exists():
                embedded_python = PROJECT_ROOT.parent / "webui forger" / "system" / "python" / "python.exe"
            if embedded_python.exists():
                python_exe = str(embedded_python)

            if target_engine == "forge_lora":
                train_script = PROJECT_ROOT / "pixel_ai_engine" / "train_forge_lora.py"
                msg_title = "LoRA Forge SD1.5"
            else:
                train_script = PROJECT_ROOT / "pixel_ai_engine" / "train_supervised.py"
                msg_title = "Supervisado PyTorch UNet"
            
            status_file = PROJECT_ROOT / "training_status.json"
            if status_file.exists():
                try:
                    with open(status_file, "r", encoding="utf-8") as f:
                        cur = json.load(f)
                    cur["status"] = "ENTRENANDO"
                    cur["timestamp"] = time.strftime("%H:%M:%S")
                    with open(status_file, "w", encoding="utf-8") as f:
                        json.dump(cur, f, indent=2)
                except Exception:
                    pass

            subprocess.Popen([python_exe, str(train_script), "--epochs", str(epochs), "--batch_size", str(batch_size), "--mode", "resume"], cwd=str(PROJECT_ROOT))
            self.send_json({"status": "resumed", "message": f"Entrenamiento de {msg_title} reanudado desde el último checkpoint."})
            return

        elif parsed.path == "/api/train/stop":
            stop_flag = PROJECT_ROOT / "stop_training.flag"
            stop_flag.touch(exist_ok=True)
            status_file = PROJECT_ROOT / "training_status.json"
            if status_file.exists():
                try:
                    with open(status_file, "r", encoding="utf-8") as f:
                        cur = json.load(f)
                    cur["status"] = "DETENIENDO..."
                    with open(status_file, "w", encoding="utf-8") as f:
                        json.dump(cur, f, indent=2)
                except Exception:
                    pass
            self.send_json({"status": "stopping", "message": "Señal de detención enviada al entrenamiento."})
            return

        elif parsed.path == "/api/train/respawn":
            training_active, _ = check_gpu_training_status()
            if training_active:
                self.send_json({"error": "Por favor pausa o detén el entrenamiento actual antes de hacer Respawn."}, status=409)
                return

            stop_flag = PROJECT_ROOT / "stop_training.flag"
            pause_flag = PROJECT_ROOT / "pause_training.flag"
            if stop_flag.exists(): 
                try: stop_flag.unlink()
                except Exception: pass
            if pause_flag.exists(): 
                try: pause_flag.unlink()
                except Exception: pass

            target_epoch = int(body.get("epoch", 10))
            epochs = int(body.get("epochs", 50))
            batch_size = int(body.get("batch_size", 4))
            
            snap_file = PROJECT_ROOT / "checkpoints" / "snapshots" / f"checkpoint_epoch_{target_epoch:03d}.pt"
            if not snap_file.exists():
                snap_file = PROJECT_ROOT / "checkpoints" / "snapshots" / f"checkpoint_epoch_{target_epoch}.pt"
            if not snap_file.exists():
                self.send_json({"error": f"No se encontró el snapshot para la época {target_epoch}."}, status=404)
                return

            python_exe = sys.executable
            embedded_python = PROJECT_ROOT / "webui forger" / "system" / "python" / "python.exe"
            if not embedded_python.exists():
                embedded_python = PROJECT_ROOT.parent / "webui forger" / "system" / "python" / "python.exe"
            if embedded_python.exists():
                python_exe = str(embedded_python)

            train_script = PROJECT_ROOT / "pixel_ai_engine" / "train_supervised.py"
            
            status_file = PROJECT_ROOT / "training_status.json"
            if status_file.exists():
                try:
                    with open(status_file, "r", encoding="utf-8") as f:
                        cur = json.load(f)
                    cur["status"] = f"RESPAWN EPOCA {target_epoch}"
                    cur["epoch"] = target_epoch
                    cur["timestamp"] = time.strftime("%H:%M:%S")
                    with open(status_file, "w", encoding="utf-8") as f:
                        json.dump(cur, f, indent=2)
                except Exception:
                    pass

            subprocess.Popen([
                python_exe, str(train_script),
                "--epochs", str(epochs),
                "--batch_size", str(batch_size),
                "--mode", "resume",
                "--respawn_epoch", str(target_epoch)
            ], cwd=str(PROJECT_ROOT))
            self.send_json({
                "status": "respawned",
                "message": f"¡Respawn exitoso! Reanudando entrenamiento desde la Época {target_epoch}."
            })
            return

        elif parsed.path == "/api/regenerate_frame":
            # Regenerar un solo frame
            frame_idx = body.get("frame_idx")
            run_dir_str = body.get("run_dir")
            char_name = body.get("character", "character")
            front_path = PROJECT_ROOT / body.get("front_image", "")
            format_type = body.get("format", "8x12")

            if frame_idx is None or not run_dir_str:
                self.send_json({"error": "frame_idx y run_dir son obligatorios"}, status=400)
                return

            target_run_dir = PROJECT_ROOT / run_dir_str

            def single_worker():
                try:
                    run_pytorch_generation(
                        character_name=char_name,
                        front_image_path=front_path,
                        format_type=format_type,
                        mode="single_frame",
                        single_frame_idx=int(frame_idx),
                        run_dir=target_run_dir
                    )
                except Exception as e:
                    traceback.print_exc()

            t = threading.Thread(target=single_worker, daemon=True)
            t.start()
            self.send_json({"status": "started", "message": f"Regenerando frame {frame_idx}..."})
            return

        self.send_json({"error": "Ruta no encontrada"}, status=404)


def open_browser(port: int):
    import webbrowser
    time.sleep(1.0)
    url = f"http://localhost:{port}/sprite_studio.html"
    print(f"[Sprite Studio] Abriendo navegador en: {url}")
    webbrowser.open(url)


def main():
    global PORT
    PORT = get_free_port(8080)
    os.chdir(PROJECT_ROOT)

    print("=" * 70)
    print("  SPRITE STUDIO - VILLA DEL CHEF (ESTUDIO LOCAL DE PIXEL ART)")
    print(f"  Directorio: {PROJECT_ROOT}")
    print(f"  Servidor activo en: http://localhost:{PORT}")
    print("=" * 70)

    # Abrir navegador automaticamente
    t = threading.Thread(target=open_browser, args=(PORT,), daemon=True)
    t.start()

    server = HTTPServer(("", PORT), SpriteStudioHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Sprite Studio] Servidor detenido por el usuario.")


if __name__ == "__main__":
    main()
