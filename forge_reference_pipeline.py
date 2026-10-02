"""
Pipeline de Generación de Sprites Pixel Art mediante Stable Diffusion 1.5 + Forge + ControlNet (MOTOR B)

Este script implementa el SEGUNDO PIPELINE DE GENERACIÓN del proyecto, permitiendo
generar hojas de spritesheets completas (16x4 o 8x12) frame por frame mediante
la API HTTP local de Stable Diffusion WebUI Forge con ControlNet (Lineart o Canny).

NO reemplaza ni altera el Motor A (U-Net PyTorch personalizada existente).
"""

import os
import sys
import re
import time
import json
import base64
import random
import argparse
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image, ImageDraw
import cv2
import requests

# Añadir directorio raíz al sys.path para importar componentes del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importar módulos existentes del proyecto
from pixel_ai_engine.config import (
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    GRID_ROWS,
    GRID_COLS,
    TOTAL_FRAMES
)
from pixel_ai_engine.enhancer import PixelArtEnhancer
from pixel_ai_engine.generate_character_sheet import resolve_character_input
from pixel_ai_engine.dataset import get_cell_coordinates, place_in_cell, foreground_mask


# ==============================================================================
# 1. CONFIGURACIÓN DE GRILLAS, MOLDES Y TEST
# ==============================================================================

POSSIBLE_FORGE_DIRS = [
    PROJECT_ROOT / "webui forger" / "webui",
    PROJECT_ROOT / "stable-diffusion-webui-forge-main",
]

MOLD_DIRS = {
    "16x4": PROJECT_ROOT / "dataset_frames_individuales" / "00_MOLDES_POSES" / "molde_16x4_64_poses",
    "8x12": PROJECT_ROOT / "dataset_frames_individuales" / "00_MOLDES_POSES" / "molde_8x12_96_poses",
}

# Configuración de poses de prueba y sus etiquetas anatómicas según LEEME_ACCIONES.md
TEST_CONFIG = {
    "16x4": {
        "poses": [1, 17, 33, 49],
        "labels": ["FRENTE", "ESPALDA", "DERECHA", "IZQUIERDA"],
        "descriptions": [
            "Fila 01 - Frente / Sur",
            "Fila 05 - Espalda / Norte",
            "Fila 09 - Derecha / Este",
            "Fila 13 - Izquierda / Oeste"
        ]
    },
    "8x12": {
        "poses": [1, 33, 17, 65],
        "labels": ["FRENTE", "ESPALDA", "LATERAL", "COCINA"],
        "descriptions": [
            "Fila 01 - Frente / Sur",
            "Fila 05 - Espalda / Norte",
            "Fila 03 - Derecha / Este",
            "Fila 09 - Cocinar / Batir"
        ]
    }
}

GRID_CONFIGS = {
    "16x4": {
        "rows": 16,
        "cols": 4,
        "total_frames": 64,
        "canvas_w": 724,
        "canvas_h": 2172,
        "mold_dir": MOLD_DIRS["16x4"],
    },
    "8x12": {
        "rows": 12,
        "cols": 8,
        "total_frames": 96,
        "canvas_w": 1024,
        "canvas_h": 1536,
        "mold_dir": MOLD_DIRS["8x12"],
    },
}


# ==============================================================================
# 2. VALIDACIÓN NUMÉRICA Y CARGA DE MOLDES
# ==============================================================================

def load_and_validate_molds(mold_dir: Path, expected_count: int) -> Dict[int, Path]:
    """
    Carga y valida rigurosamente los moldes según su número entero (1..N).
    Evita depender del orden lexicográfico y asegura que no falte ninguna pose.
    """
    if not mold_dir.exists():
        raise FileNotFoundError(f"[ERROR] Directorio de moldes no encontrado: {mold_dir}")

    molds: Dict[int, Path] = {}
    pattern = re.compile(r"pose_(\d+)")
    for f in mold_dir.glob("pose_*.png"):
        m = pattern.search(f.name)
        if m:
            num = int(m.group(1))
            molds[num] = f

    missing = []
    for i in range(1, expected_count + 1):
        if i not in molds:
            missing.append(f"pose_{i:03d}")

    if missing:
        err_msg = (
            f"[ERROR] Moldes incompletos en '{mold_dir.name}'.\n"
            f"Se esperaban {expected_count} poses pero faltan {len(missing)}: "
            f"{missing[:8]}" + ("..." if len(missing) > 8 else "")
        )
        raise FileNotFoundError(err_msg)

    return molds


# ==============================================================================
# 3. DETECCIÓN AUTOMÁTICA DE MODELOS Y FORGE
# ==============================================================================

def detect_active_forge_installation() -> Optional[Path]:
    """Detecta la carpeta de Forge activa examinando ejecutables y modelos."""
    for fdir in POSSIBLE_FORGE_DIRS:
        if (fdir / "launch.py").exists() and (fdir / "models").exists():
            return fdir
    return None


def detect_required_models(forge_dir: Optional[Path] = None) -> Dict[str, Optional[Path]]:
    """
    Verifica la presencia en disco de:
    1. SD 1.5 base: v1-5-pruned-emaonly.safetensors
    2. ControlNet Lineart: control_v11p_sd15_lineart.pth
    3. ControlNet Canny: control_v11p_sd15_canny.pth
    """
    search_dirs = []
    if forge_dir:
        search_dirs.append(forge_dir)
    search_dirs.extend(POSSIBLE_FORGE_DIRS)
    search_dirs.append(PROJECT_ROOT / "reparacion")
    search_dirs.append(PROJECT_ROOT)

    models_info = {
        "sd15": {
            "name": "SD 1.5 Base",
            "file": "v1-5-pruned-emaonly.safetensors",
            "subfolder": "Stable-diffusion",
            "path": None
        },
        "lineart": {
            "name": "ControlNet Lineart",
            "file": "control_v11p_sd15_lineart.pth",
            "subfolder": "ControlNet",
            "path": None
        },
        "canny": {
            "name": "ControlNet Canny",
            "file": "control_v11p_sd15_canny.pth",
            "subfolder": "ControlNet",
            "path": None
        }
    }

    for key, item in models_info.items():
        fname = item["file"]
        sub = item["subfolder"]
        for base in search_dirs:
            candidates = [
                base / "models" / sub / fname,
                base / fname,
                PROJECT_ROOT / "reparacion" / fname,
            ]
            for c in candidates:
                if c.exists() and c.is_file():
                    item["path"] = c.resolve()
                    break
            if item["path"]:
                break

    return {k: v["path"] for k, v in models_info.items()}


def print_model_status(models: Dict[str, Optional[Path]]):
    """Muestra reporte claro y formateado de los modelos encontrados."""
    print("=" * 70)
    print("  VERIFICACIÓN DE MODELOS EN DISCO (STABLE DIFFUSION 1.5 + CONTROLNET)")
    print("=" * 70)
    
    names = {
        "sd15": ("Stable Diffusion 1.5", "v1-5-pruned-emaonly.safetensors"),
        "lineart": ("ControlNet Lineart", "control_v11p_sd15_lineart.pth"),
        "canny": ("ControlNet Canny", "control_v11p_sd15_canny.pth"),
    }
    
    all_ok = True
    for key, (label, fname) in names.items():
        p = models.get(key)
        if p and p.exists():
            rel = p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  [OK]    {label:<22} : {rel} ({size_mb:,.1f} MB)")
        else:
            print(f"  [FALTA] {label:<22} : Archivo '{fname}' no encontrado")
            all_ok = False
            
    print("=" * 70)
    return all_ok


# ==============================================================================
# 4. COMUNICACIÓN Y VALIDACIÓN CON LA API DE FORGE
# ==============================================================================

def check_forge_api_connection(forge_url: str = "http://127.0.0.1:7860") -> Tuple[bool, str]:
    """
    Comprueba si el servidor web de Forge está iniciado y accesible.
    Retorna (True, mensaje_éxito) o (False, mensaje_de_ayuda).
    """
    try:
        r = requests.get(f"{forge_url}/sdapi/v1/sd-models", timeout=3)
        if r.status_code == 200:
            models_data = r.json()
            return True, f"Conectado a Forge ({len(models_data)} modelos disponibles)"
    except Exception:
        pass

    help_msg = (
        f"\n{'!' * 70}\n"
        f"  [ERROR] Forge no está iniciado o no responde en {forge_url}\n"
        f"{'-' * 70}\n"
        f"  Para iniciar el servidor de Forge con la API activa:\n"
        f"    1. Abre una ventana de terminal o haz doble clic en:\n"
        f"       webui forger\\run.bat\n"
        f"    2. Espera que cargue el servidor (verás 'Running on local URL: http://127.0.0.1:7860')\n"
        f"    3. Vuelve a ejecutar este comando.\n"
        f"{'!' * 70}\n"
    )
    return False, help_msg


def get_forge_options(forge_url: str) -> Optional[Dict[str, Any]]:
    """Consulta /sdapi/v1/options de Forge."""
    try:
        r = requests.get(f"{forge_url}/sdapi/v1/options", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_forge_sd_models(forge_url: str) -> List[Dict[str, Any]]:
    """Consulta /sdapi/v1/sd-models de Forge."""
    try:
        r = requests.get(f"{forge_url}/sdapi/v1/sd-models", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def get_forge_controlnet_models(forge_url: str) -> List[str]:
    """Consulta /controlnet/model_list de Forge."""
    try:
        r = requests.get(f"{forge_url}/controlnet/model_list", timeout=5)
        if r.status_code == 200:
            return r.json().get("model_list", [])
    except Exception:
        pass
    return []


def verify_forge_endpoints(forge_url: str) -> Tuple[bool, str]:
    """Comprueba que todos los endpoints indispensables de Forge respondan con código 200."""
    endpoints = ["/sdapi/v1/options", "/sdapi/v1/sd-models", "/controlnet/model_list"]
    for ep in endpoints:
        try:
            r = requests.get(f"{forge_url}{ep}", timeout=5)
            if r.status_code != 200:
                return False, f"Endpoint indispensable '{ep}' respondió con código HTTP {r.status_code}."
        except Exception as e:
            return False, f"No se pudo conectar al endpoint indispensable '{ep}': {e}"
    return True, "Endpoints OK"


def ensure_sd15_checkpoint_loaded(forge_url: str, target_name: str = "v1-5-pruned-emaonly") -> Tuple[bool, str]:
    """
    Verifica cuál checkpoint está cargado en Forge y fuerza explícitamente SD 1.5.
    Si tiene cargado otro modelo (ej: SDXL o checkpoint personalizado), lo cambia
    mediante /sdapi/v1/options y espera a que Forge termine de cargarlo.
    """
    options = get_forge_options(forge_url)
    if not options:
        return False, "No se pudo consultar /sdapi/v1/options en Forge."

    current_ckpt = options.get("sd_model_checkpoint", "")
    print(f"[Checkpoint] Solicitado : {target_name}.safetensors")
    print(f"[Checkpoint] Actual    : {current_ckpt or '(ninguno)'}")

    # Comprobar si ya corresponde a SD 1.5
    if target_name.lower() in current_ckpt.lower():
        print(f"[Checkpoint] Estado    : [OK] SD 1.5 cargado")
        return True, current_ckpt

    # Si es diferente, buscar el título completo registrado en Forge
    print(f"[Checkpoint] Modelo diferente detectado.")
    print(f"[Checkpoint] Cambiando a Stable Diffusion 1.5...")
    
    sd_models = get_forge_sd_models(forge_url)
    target_title = None
    for m in sd_models:
        title = m.get("title", "")
        model_name = m.get("model_name", "")
        if target_name.lower() in title.lower() or target_name.lower() in model_name.lower():
            target_title = title
            break

    if not target_title:
        # Fallback a nombre de archivo
        target_title = f"{target_name}.safetensors"

    try:
        post_r = requests.post(f"{forge_url}/sdapi/v1/options", json={"sd_model_checkpoint": target_title}, timeout=30)
        if post_r.status_code != 200:
            return False, f"Error HTTP {post_r.status_code} al solicitar cambio de checkpoint."
    except Exception as e:
        return False, f"Excepción al solicitar cambio de checkpoint: {e}"

    # Esperar hasta que Forge termine de cargarlo (hasta 45 segundos)
    print("  [Cargando] Esperando a que Forge complete la carga del modelo...", end="", flush=True)
    start_time = time.time()
    while time.time() - start_time < 45:
        time.sleep(2)
        print(".", end="", flush=True)
        opt = get_forge_options(forge_url)
        if opt:
            new_ckpt = opt.get("sd_model_checkpoint", "")
            if target_name.lower() in new_ckpt.lower():
                print(" [OK]")
                print(f"[Checkpoint] [OK] Cambio completado exitosamente a: {new_ckpt}")
                return True, new_ckpt

    print(" [TIEMPO AGOTADO]")
    return False, f"Forge no pudo cargar {target_name}.safetensors a tiempo."


def verify_controlnet_in_api(forge_url: str, control_type: str) -> Tuple[bool, str]:
    """
    Verifica que el modelo ControlNet solicitado aparezca registrado en Forge.
    No continúa silenciosamente con nombres inventados.
    """
    m_list = get_forge_controlnet_models(forge_url)
    pattern = f"control_v11p_sd15_{control_type.lower()}"
    matched = None
    for m in m_list:
        if pattern in m.lower():
            matched = m
            break

    if not matched:
        return False, (
            f"[ERROR] El archivo ControlNet '{control_type}' existe en disco, pero Forge no lo tiene "
            f"cargado/registrado en su API (/controlnet/model_list).\n"
            f"Modelos reportados por Forge: {m_list}\n"
            f"Reinicia Forge mediante 'webui forger\\run.bat' para que indexe la carpeta models/ControlNet/."
        )
    return True, matched


# ==============================================================================
# 5. GESTIÓN DE SEMILLAS E IDENTIDAD DETERMINISTA
# ==============================================================================

def get_or_create_character_seed(char_name: str, char_forge_dir: Path, requested_seed: Optional[int] = None) -> int:
    """
    Obtiene o crea una semilla consistente para el personaje en <char_name>_seed.json.
    La semilla fija ayuda a reducir variación aleatoria, pero no garantiza identidad idéntica entre poses.
    """
    seed_file = char_forge_dir / f"{char_name}_seed.json"
    
    if requested_seed is not None and requested_seed > 0:
        seed_data = {"character": char_name, "seed": requested_seed}
        char_forge_dir.mkdir(parents=True, exist_ok=True)
        with open(seed_file, "w", encoding="utf-8") as f:
            json.dump(seed_data, f, indent=2)
        return requested_seed

    if seed_file.exists():
        try:
            with open(seed_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return int(data.get("seed", 12345678))
        except Exception:
            pass

    # Generar semilla determinista basada en el nombre o aleatoria reproducible
    new_seed = random.randint(100000000, 999999999)
    char_forge_dir.mkdir(parents=True, exist_ok=True)
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump({"character": char_name, "seed": new_seed}, f, indent=2)
    return new_seed


# ==============================================================================
# 6. CONVERSIÓN Y PROCESAMIENTO DE IMÁGENES
# ==============================================================================

def pil_to_base64(img: Image.Image) -> str:
    """Convierte una imagen PIL a string Base64 PNG."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_pil(b64_str: str) -> Image.Image:
    """Decodifica un string Base64 a imagen PIL."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_data = base64.b64decode(b64_str)
    return Image.open(BytesIO(img_data)).convert("RGBA")


def prepare_reference_input(ref_path: Path, target_size: int = 512) -> Tuple[Image.Image, np.ndarray]:
    """
    Prepara la imagen de referencia frontal:
    - Extrae la paleta canónica del personaje.
    - Centra el personaje sobre un fondo neutro para img2img.
    - Redimensiona con Nearest Neighbor al tamaño de generación (ej. 512x512).
    """
    raw_img = Image.open(ref_path).convert("RGBA")
    
    # Extraer paleta canónica del personaje
    palette = PixelArtEnhancer.extract_palette(raw_img, max_colors=32)
    
    # Aislar y centrar el personaje
    arr = np.array(raw_img)
    alpha = arr[:, :, 3]
    mask = alpha > 25
    if not np.any(mask):
        mask = np.ones((arr.shape[0], arr.shape[1]), dtype=bool)

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cropped = raw_img.crop((x0, y0, x1 + 1, y1 + 1))
    
    # Escalar proporcionalmente al área segura central
    cw, ch = cropped.size
    safe_size = int(target_size * 0.85)
    scale = min(safe_size / max(1, cw), safe_size / max(1, ch))
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    
    scaled = cropped.resize((nw, nh), Image.Resampling.NEAREST)
    
    # Colocar en lienzo centrado sobre fondo neutro (#7F7F7F) para que SD1.5 difunda bien
    canvas = Image.new("RGBA", (target_size, target_size), (128, 128, 128, 255))
    ox = (target_size - nw) // 2
    oy = max(8, target_size - nh - 16)  # Anclado hacia la base
    canvas.paste(scaled, (ox, oy), scaled)
    
    return canvas, palette


def prepare_controlnet_pose(pose_path: Path, control_type: str = "lineart", target_size: int = 512) -> Tuple[Image.Image, str]:
    """
    Procesa el frame de molde/pose individual para alimentar ControlNet:
    - Escala el molde con Nearest Neighbor.
    - Genera la guía estructural óptima (líneas blancas sobre fondo negro para module='none').
    Retorna (imagen_preparada_pil, nombre_modulo).
    """
    pose_img = Image.open(pose_path).convert("RGBA")
    pose_scaled = pose_img.resize((target_size, target_size), Image.Resampling.NEAREST)
    arr = np.array(pose_scaled)
    alpha = arr[:, :, 3]
    
    # Crear máscara binaria del cuerpo
    mask = (alpha > 35).astype(np.uint8) * 255
    
    if control_type.lower() == "canny":
        # Extraer bordes duros con Canny para guía estricta
        edges = cv2.Canny(mask, 100, 200)
        # Dilatar levemente para dar espesor legible al modelo
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        out_pil = Image.fromarray(dilated, mode="L").convert("RGB")
        return out_pil, "none"
    else:
        # Modo Lineart: Extraer contornos anatómicos nítidos
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        out_pil = Image.fromarray(gradient, mode="L").convert("RGB")
        return out_pil, "none"


# ==============================================================================
# 7. CONTROL DE CALIDAD Y EXTRACCIÓN DE TRANSPARENCIA
# ==============================================================================

def remove_background_and_recover_alpha(img_generated: Image.Image, bg_tolerance: int = 25) -> Image.Image:
    """
    Convierte la imagen generada por SD a PNG RGBA transparente:
    - Muestrea las 4 esquinas exteriores para identificar el color de fondo.
    - Realiza flood-fill o detección de conectividad desde los bordes para NO
      borrar píxeles idénticos dentro de la ropa o cuerpo del personaje.
    """
    arr = np.array(img_generated.convert("RGBA"))
    h, w, _ = arr.shape
    rgb = arr[:, :, :3].astype(np.float32)
    
    # Muestrear fondo en las esquinas
    sample_size = max(3, min(h, w) // 16)
    corners = np.concatenate([
        rgb[:sample_size, :sample_size].reshape(-1, 3),
        rgb[:sample_size, w - sample_size:].reshape(-1, 3),
        rgb[h - sample_size:, :sample_size].reshape(-1, 3),
        rgb[h - sample_size:, w - sample_size:].reshape(-1, 3),
    ], axis=0)
    bg_color = np.median(corners, axis=0)
    
    # Distancia euclídea al color de fondo
    dists = np.sqrt(np.sum((rgb - bg_color) ** 2, axis=-1))
    is_bg_candidate = (dists <= bg_tolerance).astype(np.uint8)
    
    # Flood fill desde los 4 bordes externos para no dañar píxeles interiores
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(is_bg_candidate, flood_mask, (0, 0), 2)
    cv2.floodFill(is_bg_candidate, flood_mask, (w - 1, 0), 2)
    cv2.floodFill(is_bg_candidate, flood_mask, (0, h - 1), 2)
    cv2.floodFill(is_bg_candidate, flood_mask, (w - 1, h - 1), 2)
    
    # Los píxeles marcados con 2 son fondo exterior conectado
    exterior_bg = (is_bg_candidate == 2)
    
    out_arr = arr.copy()
    out_arr[exterior_bg, 3] = 0
    return Image.fromarray(out_arr, mode="RGBA")


def validate_frame(frame_img: Image.Image, frame_idx: int) -> Tuple[bool, str, str]:
    """
    Verifica los requisitos mínimos de calidad de cada frame generado:
    Retorna (is_ok, status_level, message) donde status_level in ['PASS', 'WARNING', 'FAIL'].
    """
    arr = np.array(frame_img.convert("RGBA"))
    alpha = arr[:, :, 3]
    fg_count = int(np.count_nonzero(alpha > 30))
    total_pixels = arr.shape[0] * arr.shape[1]
    
    # Fallos críticos (FAIL)
    if fg_count < 40:
        return False, "FAIL", f"Frame {frame_idx:03d}: Alpha casi vacío ({fg_count} px visibles)."
        
    if fg_count > int(total_pixels * 0.95):
        return False, "FAIL", f"Frame {frame_idx:03d}: Alpha desbordado ({fg_count}/{total_pixels} px)."

    fg_rgb = arr[alpha > 30][:, :3]
    std_rgb = float(np.std(fg_rgb))
    if std_rgb < 4.0:
        return False, "FAIL", f"Frame {frame_idx:03d}: Sprite monocromático o corrupto (std={std_rgb:.1f})."

    coords = np.argwhere(alpha > 30)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    fh = y1 - y0 + 1
    fw = x1 - x0 + 1
    
    if fh < 16 or fw < 8:
        return False, "FAIL", f"Frame {frame_idx:03d}: Bounding box diminuto ({fw}x{fh} px)."

    # Avisos menores (WARNING)
    if fg_count < 100:
        return True, "WARNING", f"Frame {frame_idx:03d}: Cobertura de píxeles reducida ({fg_count} px)."

    return True, "PASS", "OK"


# ==============================================================================
# 8. LLAMADA A LA API DE FORGE (IMG2IMG + CONTROLNET)
# ==============================================================================

def generate_single_frame_forge(
    forge_url: str,
    init_b64: str,
    control_b64: str,
    control_model_name: str,
    control_module: str,
    control_weight: float,
    control_start: float,
    control_end: float,
    seed: int,
    steps: int,
    cfg_scale: float,
    denoise: float,
    width: int,
    height: int,
    sampler_name: str,
    prompt: str,
    negative_prompt: str,
) -> Image.Image:
    """Envía la petición a /sdapi/v1/img2img de Forge con la configuración ControlNet."""
    
    payload = {
        "init_images": [f"data:image/png;base64,{init_b64}"],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "denoising_strength": denoise,
        "width": width,
        "height": height,
        "sampler_name": sampler_name,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "enabled": True,
                        "module": control_module,
                        "model": control_model_name,
                        "weight": control_weight,
                        "image": f"data:image/png;base64,{control_b64}",
                        "guidance_start": control_start,
                        "guidance_end": control_end,
                        "control_mode": "Balanced",
                        "pixel_perfect": False,
                    }
                ]
            }
        },
    }

    url = f"{forge_url}/sdapi/v1/img2img"
    response = requests.post(url, json=payload, timeout=120)
    
    if response.status_code != 200:
        raise RuntimeError(f"Error de Forge HTTP {response.status_code}: {response.text[:300]}")

    data = response.json()
    images = data.get("images", [])
    if not images:
        raise RuntimeError("Forge no devolvió ninguna imagen en la respuesta.")

    return base64_to_pil(images[0])


# ==============================================================================
# 9. PIPELINE PRINCIPAL DE GENERACIÓN
# ==============================================================================

def run_forge_reference_pipeline(
    reference_input: str,
    format_type: str = "16x4",
    control_type: str = "lineart",
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    seed_strategy: str = "fixed",
    steps: int = 25,
    cfg: float = 7.0,
    denoise: float = 0.50,
    control_weight: float = 0.90,
    control_start: float = 0.0,
    control_end: float = 1.0,
    sampler: str = "Euler a",
    resolution: int = 512,
    forge_url: str = "http://127.0.0.1:7860",
    resume: bool = False,
    only_frame: Optional[int] = None,
    test_mode: bool = False,
    no_enhance: bool = False,
    check_only: bool = False,
) -> bool:
    """Función controladora del pipeline Forge."""
    
    # 1. Resolver ruta del personaje
    ref_path = resolve_character_input(reference_input)
    char_name = ref_path.stem.replace("_rnormal", "").replace("rnormal", "personaje")
    print(f"\n[Pipeline] Personaje detectado: '{char_name}'")
    print(f"[Pipeline] Imagen de referencia: {ref_path}")

    # 2. Configuración de grilla
    if format_type not in GRID_CONFIGS:
        raise ValueError(f"Formato desconocido '{format_type}'. Debe ser '16x4' o '8x12'.")
    cfg_grid = GRID_CONFIGS[format_type]
    rows = cfg_grid["rows"]
    cols = cfg_grid["cols"]
    total_frames = cfg_grid["total_frames"]
    canvas_w = cfg_grid["canvas_w"]
    canvas_h = cfg_grid["canvas_h"]
    mold_dir = cfg_grid["mold_dir"]

    # 3. Detectar modelos locales en disco
    forge_dir = detect_active_forge_installation()
    models = detect_required_models(forge_dir)
    print_model_status(models)

    ctrl_key = "lineart" if control_type.lower() == "lineart" else "canny"
    other_ctrl_key = "canny" if ctrl_key == "lineart" else "lineart"

    has_sd15 = bool(models.get("sd15"))
    has_req_ctrl = bool(models.get(ctrl_key))
    has_other_ctrl = bool(models.get(other_ctrl_key))

    # Resumen de requisitos del pipeline
    print("\n" + "=" * 70)
    print("  REQUISITOS DEL PIPELINE")
    print("=" * 70)
    print(f"  Stable Diffusion 1.5   : {'[OK]' if has_sd15 else '[FALTA]'}")
    print(f"  ControlNet solicitado  : {control_type.upper()}")
    print(f"  ControlNet {control_type.capitalize():<12}: {'[OK]' if has_req_ctrl else '[FALTA]'}")
    print(f"  ControlNet {other_ctrl_key.capitalize():<12}: {'[DISPONIBLE]' if has_other_ctrl else '[NO REQUERIDO]'}")
    print("-" * 70)

    # 4. Estado de la API de Forge
    is_alive, msg = check_forge_api_connection(forge_url)
    if is_alive:
        print(f"  Forge API              : ONLINE ({forge_url})")
        options = get_forge_options(forge_url)
        current_ckpt = options.get("sd_model_checkpoint", "") if options else "Desconocido"
        ckpt_status = "OK" if "v1-5-pruned-emaonly" in current_ckpt.lower() else "DIFERENTE (se cambiará al generar)"
        print(f"  Checkpoint activo      : {current_ckpt} [{ckpt_status}]")
        
        cn_ok, cn_info = verify_controlnet_in_api(forge_url, control_type)
        if cn_ok:
            print(f"  ControlNet API ({control_type}) : DETECTADO ({cn_info})")
        else:
            print(f"  ControlNet API ({control_type}) : NO REGISTRADO EN FORGE")
    else:
        print(f"  Forge API              : OFFLINE ({forge_url})")
        print(f"  Modelos locales        : {'OK' if (has_sd15 and has_req_ctrl) else 'INCOMPLETOS'}")
        print(f"  Generación             : Requiere iniciar Forge ('webui forger\\run.bat')")

    print("=" * 70)

    # Si se solicitó --check-only, validar requisitos obligatorios y salir con código claro
    if check_only:
        if not has_sd15:
            print("\n[ERROR] Falta el modelo base Stable Diffusion 1.5 (v1-5-pruned-emaonly.safetensors).")
            print("RESULTADO: ERROR - FALTAN MODELOS OBLIGATORIOS\n")
            return False
        if not has_req_ctrl:
            print(f"\n[ERROR] Falta el modelo ControlNet solicitado '{control_type}'.")
            print("RESULTADO: ERROR - FALTAN MODELOS OBLIGATORIOS\n")
            return False

        print("  RESULTADO: PIPELINE LISTO (Archivos y dependencias locales validados)\n")
        return True

    # Para generación real, verificar que Forge está activo
    if not is_alive:
        print(msg)
        return False

    # Verificar endpoints indispensables de Forge
    ep_ok, ep_msg = verify_forge_endpoints(forge_url)
    if not ep_ok:
        print(f"\n[ERROR] {ep_msg}")
        return False

    # Forzar explícitamente SD 1.5 en Forge
    sd_loaded, sd_msg = ensure_sd15_checkpoint_loaded(forge_url, target_name="v1-5-pruned-emaonly")
    if not sd_loaded:
        print(f"\n[ERROR] Forge no pudo cargar v1-5-pruned-emaonly.safetensors: {sd_msg}")
        return False

    # Verificar que el modelo ControlNet seleccionado aparece en la API
    cn_ok, cn_model_name = verify_controlnet_in_api(forge_url, control_type)
    if not cn_ok:
        print(f"\n{cn_model_name}")
        return False
    print(f"[ControlNet] Modelo asignado en Forge: '{cn_model_name}' (Modo: {control_type.upper()})")

    # 5. Cargar y validar numéricamente los moldes
    molds_map = load_and_validate_molds(mold_dir, total_frames)
    print(f"[Moldes] {len(molds_map)}/{total_frames} moldes numéricos validados correctamente en {mold_dir.name}.")

    # 6. Carpetas de trabajo organizadas
    base_out_dir = PROJECT_ROOT / "output" / char_name / "forge"
    raw_dir = base_out_dir / "raw"
    norm_dir = base_out_dir / "normalized"
    failed_dir = base_out_dir / "failed"
    previews_dir = base_out_dir / "previews"
    
    for d in [raw_dir, norm_dir, failed_dir, previews_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 7. Semilla determinista
    base_seed = get_or_create_character_seed(char_name, base_out_dir, seed)
    print(f"[Semilla] Base para '{char_name}': {base_seed} (Estrategia: {seed_strategy})")

    # 8. Preprocesar referencia frontal y paleta
    ref_canvas, char_palette = prepare_reference_input(ref_path, target_size=resolution)
    ref_b64 = pil_to_base64(ref_canvas)

    # Prompts base conservadores
    prompt = (
        "same character as reference image, exact same character design, exact same clothing, "
        "exact same hair, exact same skin tone, same accessories, same proportions, 2D pixel art "
        "videogame sprite, full body, clean hard pixel edges, consistent character design, 16-bit retro style"
    )
    negative_prompt = (
        "different character, redesigned outfit, different hair, different skin tone, extra arms, "
        "extra legs, duplicate limbs, malformed hands, missing face, cropped feet, cropped head, "
        "photorealistic, 3d render, smooth painting, blurry, anti-aliasing, text, watermark, "
        "background objects, complex background, blurry edges"
    )

    # 9. Determinar frames a procesar
    if test_mode:
        t_cfg = TEST_CONFIG[format_type]
        frame_indices = t_cfg["poses"]
        test_labels = t_cfg["labels"]
        test_descs = t_cfg["descriptions"]
        print(f"\n{'*' * 70}")
        print(f"  MODO TEST ACTIVO ({format_type}): Se generarán únicamente {len(frame_indices)} poses clave")
        for p_num, p_lbl, p_dsc in zip(frame_indices, test_labels, test_descs):
            print(f"    - Pose {p_num:02d} ({p_lbl}): {p_dsc}")
        print(f"{'*' * 70}\n")
    elif only_frame is not None:
        if only_frame < 1 or only_frame > total_frames:
            raise ValueError(f"--only-frame debe estar entre 1 y {total_frames}")
        frame_indices = [only_frame]
        print(f"\n[Regeneración Individual] Procesando exclusivamente el frame {only_frame}...")
    else:
        frame_indices = list(range(1, total_frames + 1))
        print(f"\n[Generación Completa] Total de frames a procesar: {total_frames} ({format_type})")

    generated_frames_map: Dict[int, Image.Image] = {}
    valid_frames: List[int] = []
    failed_frames: List[int] = []

    # 10. Bucle de generación frame por frame
    for idx in frame_indices:
        f_num = idx
        norm_file = norm_dir / f"frame_{f_num:03d}.png"
        raw_file = raw_dir / f"frame_{f_num:03d}.png"
        
        # Modo reanudación
        if resume and norm_file.exists() and only_frame is None:
            try:
                existing_img = Image.open(norm_file).convert("RGBA")
                ok, level, _ = validate_frame(existing_img, f_num)
                if ok and level in ["PASS", "WARNING"]:
                    print(f"  [REANUDAR] Frame {f_num:03d}/{total_frames}: Ya existe y es válido. Omitiendo.")
                    generated_frames_map[f_num] = existing_img
                    valid_frames.append(f_num)
                    continue
            except Exception:
                pass

        # Determinar semilla del frame
        if seed_strategy == "offset":
            current_seed = base_seed + f_num
        else:
            current_seed = base_seed

        # Cargar molde numérico correspondiente
        pose_file = molds_map[f_num]
        ctrl_img, ctrl_module = prepare_controlnet_pose(pose_file, control_type=control_type, target_size=resolution)
        ctrl_b64 = pil_to_base64(ctrl_img)

        print(f"  [Generando] Frame {f_num:03d}/{total_frames} (Molde: {pose_file.name}, Seed: {current_seed})...", end="", flush=True)
        t0 = time.time()

        try:
            # Petición a Forge
            raw_result = generate_single_frame_forge(
                forge_url=forge_url,
                init_b64=ref_b64,
                control_b64=ctrl_b64,
                control_model_name=cn_model_name,
                control_module=ctrl_module,
                control_weight=control_weight,
                control_start=control_start,
                control_end=control_end,
                seed=current_seed,
                steps=steps,
                cfg_scale=cfg,
                denoise=denoise,
                width=resolution,
                height=resolution,
                sampler_name=sampler,
                prompt=prompt,
                negative_prompt=negative_prompt,
            )
            raw_result.save(raw_file)

            # Extracción de fondo y transparencia RGBA
            transparent_img = remove_background_and_recover_alpha(raw_result, bg_tolerance=28)

            # Post-procesado Pixel Art
            if not no_enhance:
                # 1. Binarización alfa y despeckling
                enhanced = PixelArtEnhancer.binarize_alpha(transparent_img, threshold=60)
                enhanced = PixelArtEnhancer.remove_orphan_pixels(enhanced, min_connected_size=3)
                # 2. Encaje a paleta canónica para mantener identidad exacta
                if len(char_palette) > 0:
                    enhanced = PixelArtEnhancer.snap_to_palette(enhanced, char_palette, tolerance=32.0)
            else:
                enhanced = transparent_img

            # Redimensionar al tamaño canónico de celda con Nearest Neighbor
            cell_w = canvas_w // cols
            cell_h = canvas_h // rows
            normalized_frame = place_in_cell(enhanced, cell_w=cell_w, cell_h=cell_h)
            normalized_frame.save(norm_file)

            # Quality Gate con niveles PASS, WARNING, FAIL
            valid_ok, valid_level, valid_msg = validate_frame(normalized_frame, f_num)
            dt = time.time() - t0

            if valid_level == "PASS":
                print(f" OK ({dt:.1f}s)")
                generated_frames_map[f_num] = normalized_frame
                valid_frames.append(f_num)
            elif valid_level == "WARNING":
                print(f" [AVISO] {valid_msg} ({dt:.1f}s)")
                generated_frames_map[f_num] = normalized_frame
                valid_frames.append(f_num)
            else:  # FAIL
                print(f" [FALLO] {valid_msg} ({dt:.1f}s)")
                normalized_frame.save(failed_dir / f"frame_{f_num:03d}.png")
                if norm_file.exists():
                    try:
                        norm_file.unlink()
                    except Exception:
                        pass
                failed_frames.append(f_num)

        except Exception as e:
            print(f" ERROR: {e}")
            failed_frames.append(f_num)
            return False

    # 11. Modo TEST: Generar panel comparativo de validación de identidad
    if test_mode:
        print("\n[Modo Test] Creando panel comparativo de validación de identidad...")
        test_panel_path = previews_dir / f"test_comparison_4poses_{format_type}_{control_type}.png"
        
        preview_cell_size = 256
        panel_w = preview_cell_size * (len(frame_indices) + 1)
        panel_h = preview_cell_size + 40
        panel = Image.new("RGBA", (panel_w, panel_h), (240, 243, 248, 255))
        draw = ImageDraw.Draw(panel)

        # 1. Pegar Referencia
        ref_thumb = ref_canvas.resize((preview_cell_size - 16, preview_cell_size - 16), Image.Resampling.NEAREST)
        panel.paste(ref_thumb, (8, 30), ref_thumb)
        draw.text((10, 8), "1. REFERENCIA ORIGINAL", fill=(20, 20, 20))

        # 2. Pegar las 4 poses generadas con sus etiquetas semánticas
        t_labels = TEST_CONFIG[format_type]["labels"]
        for p_idx, f_num in enumerate(frame_indices):
            f_img = generated_frames_map.get(f_num)
            if f_img:
                fx = (p_idx + 1) * preview_cell_size
                thumb = f_img.resize((preview_cell_size - 16, preview_cell_size - 16), Image.Resampling.NEAREST)
                panel.paste(thumb, (fx + 8, 30), thumb)
                draw.text((fx + 10, 8), f"{p_idx + 2}. {t_labels[p_idx]} (F{f_num:02d})", fill=(20, 20, 20))

        panel.save(test_panel_path)
        print("=" * 70)
        print(f"  PANEL DE PRUEBA GUARDADO: {test_panel_path}")
        print("  Revisa esta imagen para evaluar la consistencia de identidad y pose.")
        print("  Si la prueba es satisfactoria, puedes generar la hoja completa sin el flag --test.")
        print("=" * 70)
        return True

    # 12. Resumen de Calidad y Ensamblado de Spritesheet Final
    if not test_mode and only_frame is None:
        print("\n" + "=" * 70)
        print("  RESULTADO DE GENERACIÓN")
        print("=" * 70)
        print(f"  Frames esperados : {total_frames}")
        print(f"  Frames válidos   : {len(valid_frames)}")
        print(f"  Frames fallidos  : {len(failed_frames)}")
        if failed_frames:
            print(f"  Fallidos         : {', '.join(f'frame_{f:03d}' for f in failed_frames)}")
            print("  Estado           : INCOMPLETO — ejecutar --resume o --only-frame <N>")
        else:
            print("  Estado           : COMPLETO")
        print("=" * 70)

        # Ensamblar lienzo
        print(f"\n[Ensamblado] Construyendo spritesheet ({cols} columnas x {rows} filas = {total_frames} frames)...")
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        for f_num in range(1, total_frames + 1):
            f_file = norm_dir / f"frame_{f_num:03d}.png"
            if f_num in failed_frames:
                f_img = Image.new("RGBA", (canvas_w // cols, canvas_h // rows), (0, 0, 0, 0))
            elif f_num in generated_frames_map:
                f_img = generated_frames_map[f_num]
            elif f_file.exists():
                f_img = Image.open(f_file).convert("RGBA")
            else:
                f_img = Image.new("RGBA", (canvas_w // cols, canvas_h // rows), (0, 0, 0, 0))

            frame_idx = f_num - 1
            row = frame_idx // cols
            col = frame_idx % cols
            x0, y0, x1, y1 = get_cell_coordinates(canvas_w, canvas_h, row, col, rows, cols)
            target_w = x1 - x0
            target_h = y1 - y0

            cell_sprite = place_in_cell(f_img, cell_w=target_w, cell_h=target_h)
            canvas.paste(cell_sprite, (x0, y0), cell_sprite)

        # Destino de salida
        if output_path is None:
            final_output = base_out_dir / f"{char_name}_spritesheet_{cols}x{rows}_{control_type}.png"
        else:
            final_output = Path(output_path)
            
        final_output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(final_output, format="PNG")

        # Vista previa clara para Windows Photos
        preview_output = final_output.with_name(f"{final_output.stem}_vista_previa.png")
        bg_preview = Image.new("RGBA", canvas.size, (230, 233, 240, 255))
        bg_preview.paste(canvas, (0, 0), canvas)
        bg_preview.convert("RGB").save(preview_output, format="PNG")

        if failed_frames:
            print("=" * 70)
            print(f"  [AVISO] SPRITESHEET GENERADO CON {len(failed_frames)} CELDAS INCOMPLETAS")
            print(f"  Archivo guardado: {final_output}")
            print(f"  Ejecuta con '--resume' para regenerar los frames fallidos sin reiniciar.")
            print("=" * 70)
        else:
            print("=" * 70)
            print(f"  SPRITESHEET ENSAMBLADO CON ÉXITO: {final_output}")
            print(f"  Dimensiones: {canvas_w} x {canvas_h} px | {cols}x{rows} ({total_frames} frames)")
            print(f"  Transparencia: RGBA nativo compatible con Unity / Godot")
            print(f"  Vista Previa Clara: {preview_output.name}")
            print("=" * 70)

    return len(failed_frames) == 0


# ==============================================================================
# 10. PARSER DE ARGUMENTOS CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Referencia SD 1.5 + Forge + ControlNet (MOTOR B)"
    )
    parser.add_argument("--reference", "-r", type=str, required=True,
                        help="Ruta o nombre de la imagen de referencia frontal (ej: tori, alex, personajes/tori/tori_rnormal.png)")
    parser.add_argument("--format", "-f", type=str, default="16x4", choices=["16x4", "8x12"],
                        help="Formato del spritesheet: 16x4 (64 frames) u 8x12 (96 frames)")
    parser.add_argument("--control", "-c", type=str, default="lineart", choices=["lineart", "canny"],
                        help="Modelo de ControlNet a utilizar: 'lineart' (recomendado) o 'canny'")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Ruta personalizada para el archivo PNG final")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Semilla base determinista (si no se define, se usa o crea en <char>_seed.json)")
    parser.add_argument("--seed-strategy", type=str, default="fixed", choices=["fixed", "offset"],
                        help="Estrategia de semilla: 'fixed' (igual en todos los frames) u 'offset' (seed + frame_idx)")
    parser.add_argument("--steps", type=int, default=25,
                        help="Pasos de difusión en Forge (defecto: 25)")
    parser.add_argument("--cfg", type=float, default=7.0,
                        help="Escala CFG (defecto: 7.0)")
    parser.add_argument("--denoise", type=float, default=0.50,
                        help="Fuerza de denoising en img2img (0.40 - 0.60 recomendado para preservar identidad)")
    parser.add_argument("--control-weight", type=float, default=0.90,
                        help="Peso de ControlNet (0.80 - 1.00)")
    parser.add_argument("--control-start", type=float, default=0.0,
                        help="Inicio de guía ControlNet (0.0)")
    parser.add_argument("--control-end", type=float, default=1.0,
                        help="Fin de guía ControlNet (1.0)")
    parser.add_argument("--sampler", type=str, default="Euler a",
                        help="Nombre del sampler en Forge (defecto: 'Euler a')")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Resolución de trabajo en Forge (512 recomendado para SD1.5)")
    parser.add_argument("--forge-url", type=str, default="http://127.0.0.1:7860",
                        help="URL base de la API HTTP de Forge (defecto: http://127.0.0.1:7860)")
    parser.add_argument("--resume", action="store_true",
                        help="Modo reanudación: omite frames que ya existan y pasen el control de calidad")
    parser.add_argument("--only-frame", type=int, default=None,
                        help="Generar o regenerar exclusivamente un frame específico (ej: 37)")
    parser.add_argument("--test", action="store_true",
                        help="Modo test: genera únicamente 4 poses clave para validar identidad")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Desactiva el post-procesado de pixel art (paleta, despeckling, binarizado alfa)")
    parser.add_argument("--check-only", action="store_true",
                        help="Comprueba la existencia de modelos, conectividad de Forge y sale con código de error si faltan requisitos")

    args = parser.parse_args()

    success = run_forge_reference_pipeline(
        reference_input=args.reference,
        format_type=args.format,
        control_type=args.control,
        output_path=args.output,
        seed=args.seed,
        seed_strategy=args.seed_strategy,
        steps=args.steps,
        cfg=args.cfg,
        denoise=args.denoise,
        control_weight=args.control_weight,
        control_start=args.control_start,
        control_end=args.control_end,
        sampler=args.sampler,
        resolution=args.resolution,
        forge_url=args.forge_url,
        resume=args.resume,
        only_frame=args.only_frame,
        test_mode=args.test,
        no_enhance=args.no_enhance,
        check_only=args.check_only,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
