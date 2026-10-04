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
import urllib.parse
import urllib.request
import requests
import threading
import subprocess
import traceback
from pathlib import Path
from http.server import HTTPServer, ThreadingHTTPServer, SimpleHTTPRequestHandler
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
from pixel_ai_engine.frame_quality_review import FrameQualityReviewManager, VALID_REVIEW_STATUSES

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
FRAME_REVIEW_MANAGER = FrameQualityReviewManager(base_dir=PROJECT_ROOT)
JOB_LOCK = threading.Lock()
GLOBAL_TRAINING_PROC = None
LAST_START_TIME = 0.0
SERVER_HOST = os.environ.get("SPRITE_STUDIO_HOST", "192.168.1.83")
TRAINING_LOG_FILE = PROJECT_ROOT / "training_logs" / "training.log"
ACCESS_LOG_FILE = PROJECT_ROOT / "training_logs" / "access.log"
SERVER_LOCK_FILE = PROJECT_ROOT / ".sprite_studio.lock"
ACCESS_LOG_LOCK = threading.Lock()
KNOWN_ACCESS_CLIENTS = set()


def write_access_event(client_ip: str, user_agent: str, event: str, target: str, details: str = "") -> None:
    safe_ip = str(client_ip).replace("\r", " ").replace("\n", " ")[:64]
    safe_agent = str(user_agent).replace("\r", " ").replace("\n", " ")[:180]
    safe_event = str(event).replace("\r", " ").replace("\n", " ")[:40]
    safe_target = str(target).replace("\r", " ").replace("\n", " ")[:160]
    safe_details = str(details).replace("\r", " ").replace("\n", " ")[:240]
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    client_key = (safe_ip, safe_agent)
    lines = []

    with ACCESS_LOG_LOCK:
        if client_key not in KNOWN_ACCESS_CLIENTS:
            KNOWN_ACCESS_CLIENTS.add(client_key)
            lines.append(f"[{timestamp}] [CONEXION] IP={safe_ip} | Navegador={safe_agent or 'desconocido'}")
        line = f"[{timestamp}] [{safe_event}] IP={safe_ip} | {safe_target}"
        if safe_details:
            line += f" | {safe_details}"
        lines.append(line)
        ACCESS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ACCESS_LOG_FILE, "a", encoding="utf-8") as access_log:
            for access_line in lines:
                print(access_line, flush=True)
                access_log.write(access_line + "\n")


def _is_live_sprite_studio_process(process_id: int) -> bool:
    if process_id <= 0 or process_id == os.getpid():
        return False
    try:
        import psutil
        process = psutil.Process(process_id)
        command = " ".join(process.cmdline()).lower()
        return process.is_running() and "sprite_studio.py" in command
    except Exception:
        return False


def acquire_server_lock() -> bool:
    for _ in range(2):
        try:
            descriptor = os.open(SERVER_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
                lock_file.write(str(os.getpid()))
            return True
        except FileExistsError:
            try:
                existing_pid = int(SERVER_LOCK_FILE.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                existing_pid = -1
            if _is_live_sprite_studio_process(existing_pid):
                return False
            try:
                SERVER_LOCK_FILE.unlink()
            except OSError:
                return False
    return False


def release_server_lock() -> None:
    try:
        owner_pid = int(SERVER_LOCK_FILE.read_text(encoding="utf-8").strip())
        if owner_pid == os.getpid():
            SERVER_LOCK_FILE.unlink()
    except (OSError, ValueError):
        pass

def _forward_training_output(process: subprocess.Popen) -> None:
    TRAINING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(TRAINING_LOG_FILE, "a", encoding="utf-8", buffering=1) as log_file:
        header = f"\n{'=' * 70}\n[Sprite Studio] Entrenamiento iniciado: {started_at}\n{'=' * 70}\n"
        sys.stdout.write(header)
        sys.stdout.flush()
        log_file.write(header)
        if process.stdout is not None:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
        return_code = process.wait()
        footer = f"\n[Sprite Studio] Entrenamiento finalizado con codigo {return_code}.\n"
        sys.stdout.write(footer)
        sys.stdout.flush()
        log_file.write(footer)


def launch_training_process(command: List[str], env: Dict[str, str]) -> subprocess.Popen:
    process_env = dict(env)
    process_env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=process_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    threading.Thread(target=_forward_training_output, args=(process,), daemon=True).start()
    return process


CACHED_HW_TELEMETRY = {
    "gpu_temp": 55,
    "cpu_temp": None,
    "vram_gb": 0.88,
    "vram_total_gb": 4.0,
    "vram_pct": 22.0,
    "gpu_util": 0,
    "gpu_name": "NVIDIA RTX 3050 Ti",
    "last_updated": 0.0,
    "last_attempt": 0.0,
    "source": "fallback",
    "stale": True,
    "last_cpu_check": 0.0
}


def update_hardware_telemetry() -> Dict[str, Any]:
    """
    Consulta telemetría viva de hardware de forma no bloqueante con caché interna:
    - VRAM real usada y total de la GPU (GB y %)
    - Temperatura de GPU (°C)
    - Temperatura de CPU (°C) vía WMI / ACPI nativo de Windows
    """
    global CACHED_HW_TELEMETRY
    now = time.time()
    if now - CACHED_HW_TELEMETRY["last_attempt"] < 2.5:
        return CACHED_HW_TELEMETRY
    CACHED_HW_TELEMETRY["last_attempt"] = now

    # 1. GPU via nvidia-smi
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu,name", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1.5
        )
        first_gpu_line = res.stdout.strip().splitlines()[0] if res.stdout.strip() else ""
        parts = [x.strip() for x in first_gpu_line.split(",")]
        if res.returncode == 0 and len(parts) >= 3:
            gpu_temp = int(parts[0])
            used_mb = int(parts[1])
            tot_mb = int(parts[2])
            CACHED_HW_TELEMETRY["gpu_temp"] = gpu_temp
            CACHED_HW_TELEMETRY["vram_gb"] = round(used_mb / 1024.0, 2)
            CACHED_HW_TELEMETRY["vram_total_gb"] = round(tot_mb / 1024.0, 1)
            CACHED_HW_TELEMETRY["vram_pct"] = round((used_mb / max(1, tot_mb)) * 100, 1)
            if len(parts) > 3 and parts[3].isdigit():
                CACHED_HW_TELEMETRY["gpu_util"] = int(parts[3])
            if len(parts) > 4 and parts[4]:
                CACHED_HW_TELEMETRY["gpu_name"] = parts[4]
            CACHED_HW_TELEMETRY["last_updated"] = now
            CACHED_HW_TELEMETRY["source"] = "nvidia-smi"
    except Exception:
        pass

    # 2. CPU via WMI ACPI ThermalZone (Windows nativo)
    if now - CACHED_HW_TELEMETRY.get("last_cpu_check", 0) > 6.0:
        CACHED_HW_TELEMETRY["last_cpu_check"] = now
        try:
            cmd = ['powershell', '-NoProfile', '-Command', '(Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction SilentlyContinue).CurrentTemperature']
            out = subprocess.check_output(cmd, text=True, timeout=2.0).strip()
            digits = [int(x.strip()) for x in out.split() if x.strip().isdigit()]
            if digits:
                c_temp = round(digits[0] / 10.0 - 273.15)
                if 20 <= c_temp <= 115:
                    CACHED_HW_TELEMETRY["cpu_temp"] = c_temp
        except Exception:
            pass

    last_updated = CACHED_HW_TELEMETRY.get("last_updated", 0.0)
    CACHED_HW_TELEMETRY["stale"] = last_updated <= 0.0 or now - last_updated > 6.0
    return CACHED_HW_TELEMETRY


def check_gpu_training_status() -> Tuple[bool, Dict[str, Any]]:
    """
    Verifica con precisión si hay un entrenamiento activo en la GPU.
    Evita bloqueos falsos (zombie/phantom locks) limpiando banderas y
    sincronizando estados si el proceso ha finalizado o fue detenido.
    """
    global GLOBAL_TRAINING_PROC, LAST_START_TIME
    proc_running = False
    proc_info = None

    if GLOBAL_TRAINING_PROC is not None:
        if GLOBAL_TRAINING_PROC.poll() is None:
            proc_running = True
        else:
            GLOBAL_TRAINING_PROC = None

    if not proc_running:
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

    startup_grace = (time.time() - LAST_START_TIME) < 8.0 if LAST_START_TIME > 0 else False
    is_active = proc_running or startup_grace

    status_file = PROJECT_ROOT / "training_status.json"
    pause_flag = PROJECT_ROOT / "pause_training.flag"
    stop_flag = PROJECT_ROOT / "stop_training.flag"

    data: Dict[str, Any] = {}
    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

    st = data.get("status", "")

    if proc_running:
        keep_status = (
            pause_flag.exists()
            or stop_flag.exists()
            or "PAUSANDO" in st
            or "DETENIENDO" in st
            or st in ["RECUPERANDO", "REANUDANDO"]
        )
        data["status"] = st if keep_status else "ENTRENANDO"
        if proc_info and "create_time" in proc_info:
            live_elapsed = round(time.time() - proc_info["create_time"], 1)
            if live_elapsed > data.get("elapsed_sec", 0):
                data["elapsed_sec"] = live_elapsed
    else:
        if not startup_grace:
            # Si el proceso ya no corre, limpiar cualquier bandera residual
            if pause_flag.exists():
                try: pause_flag.unlink()
                except Exception: pass
            if stop_flag.exists():
                try: stop_flag.unlink()
                except Exception: pass

            # Corregir estados transitorios que hayan quedado congelados
            state_changed = False
            if st in ["PAUSANDO", "PAUSANDO...", "ENTRENANDO"]:
                data["status"] = "PAUSADO"
                state_changed = True
            elif st in ["RECUPERANDO", "REANUDANDO"]:
                recovery = data.get("recovery", {})
                data["status"] = "RECUPERACION_LISTA" if recovery.get("active") else "PAUSADO"
                state_changed = True
            elif st in ["DETENIENDO", "DETENIENDO..."]:
                data["status"] = "DETENIDO"
                state_changed = True

            if state_changed and status_file.exists():
                try:
                    with open(status_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                except Exception:
                    pass

    return is_active, data


def get_active_recovery_checkpoint() -> Optional[Path]:
    status_file = PROJECT_ROOT / "training_status.json"
    if not status_file.exists():
        return None
    try:
        with open(status_file, "r", encoding="utf-8") as status_handle:
            status_data = json.load(status_handle)
        recovery = status_data.get("recovery", {})
        if not isinstance(recovery, dict) or not recovery.get("active"):
            return None
        candidate = Path(recovery.get("checkpoint", ""))
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        resolved = candidate.resolve()
        resolved.relative_to(CHECKPOINT_DIR.resolve())
        return resolved if resolved.is_file() else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def get_valid_snapshot_max_epoch() -> Optional[int]:
    status_file = PROJECT_ROOT / "training_status.json"
    if not status_file.exists():
        return None
    try:
        with open(status_file, "r", encoding="utf-8") as status_handle:
            status_data = json.load(status_handle)
        if status_data.get("recovery_events"):
            return int(status_data.get("epoch", 0))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return None


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
                get_active_recovery_checkpoint(),
                CHECKPOINT_DIR / "best_generator.pt",
                CHECKPOINT_DIR / "latest_checkpoint.pt",
                CHECKPOINT_DIR / "base_generator_16x4.pt"
            ]
        else:
            candidates = [
                CHECKPOINT_DIR / "base_generator_16x4.pt",
                CHECKPOINT_DIR / "latest_checkpoint_16x4.pt"
            ]
        checkpoint_path = next((c for c in candidates if c is not None and c.exists()), None)

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
                canvas_clean.alpha_composite(cell_placed, (x0, y0))
                canvas_grid.alpha_composite(cell_placed, (x0, y0))

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

    def end_headers(self):
        request_path = urllib.parse.urlparse(self.path).path.lower()
        if not request_path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def log_message(self, format, *args):
        pass  # Silenciar logs ruidosos

    def audit_access(self, event: str, target: str, details: str = "") -> None:
        client_ip = self.client_address[0] if self.client_address else "desconocida"
        user_agent = self.headers.get("User-Agent", "desconocido")
        write_access_event(client_ip, user_agent, event, target, details)

    def send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in ["/", "/index.html", "/sprite_studio.html"] or path.lower().endswith(".html"):
            self.audit_access("NAVEGACION", path or "/")
        elif path.startswith("/api/") and path not in ["/api/status", "/api/train/snapshots"]:
            self.audit_access("CONSULTA", path)

        if path == "/api/status":
            training_active, train_data = check_gpu_training_status()
            forge_active, forge_model = check_forge_status()
            hw = update_hardware_telemetry()
            
            # Sincronizar métricas de hardware vivas en training_info
            train_data["vram_gb"] = hw["vram_gb"]
            train_data["vram_total_gb"] = hw["vram_total_gb"]
            train_data["vram_pct"] = hw["vram_pct"]
            train_data["gpu_temp"] = hw["gpu_temp"]
            if hw["cpu_temp"] is not None:
                train_data["cpu_temp"] = hw["cpu_temp"]
            train_data["gpu_util"] = hw["gpu_util"]
            train_data["telemetry_stale"] = hw["stale"]
            train_data["telemetry_updated_at"] = hw["last_updated"]

            gpu_name = hw.get("gpu_name") or "CPU"
            if gpu_name == "CPU":
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)

            self.send_json({
                "device": str(DEVICE),
                "gpu_name": gpu_name,
                "vram_gb": hw["vram_total_gb"],
                "vram_used_gb": hw["vram_gb"],
                "vram_pct": hw["vram_pct"],
                "training_active": training_active,
                "training_info": train_data,
                "hardware_telemetry": hw,
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
            max_valid_epoch = get_valid_snapshot_max_epoch()
            if snaps_dir.exists():
                for p in sorted(snaps_dir.glob("checkpoint_epoch_*.pt")):
                    try:
                        ep = int(p.stem.replace("checkpoint_epoch_", ""))
                        if max_valid_epoch is not None and ep > max_valid_epoch:
                            continue
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

        elif path == "/api/frame_review/queue":
            status = query.get("status", [None])[0]
            rows = FRAME_REVIEW_MANAGER.get_queue(status=status or None)
            self.send_json({
                "queue": rows,
                "count": len(rows),
                "status": status,
                "valid_statuses": list(VALID_REVIEW_STATUSES),
            })
            return

        elif path == "/api/frame_review/record":
            review_id = query.get("review_id", [None])[0]
            if not review_id:
                self.send_json({"error": "review_id es obligatorio"}, status=400)
                return
            record = FRAME_REVIEW_MANAGER.get_record(review_id)
            if record is None:
                self.send_json({"error": "Review no encontrado"}, status=404)
                return
            self.send_json(record)
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
        global GLOBAL_TRAINING_PROC, LAST_START_TIME
        parsed = urllib.parse.urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len).decode("utf-8", errors="replace")
        try:
            body = json.loads(post_data) if post_data else {}
        except Exception:
            body = {}

        if parsed.path == "/api/audit/navigation":
            section_names = {
                "studio": "Generador y Editor",
                "animator": "Reproductor de Animaciones",
                "monitor": "Monitor de Entrenamiento",
                "quality": "Control de Calidad",
            }
            section_id = str(body.get("section", "desconocida"))[:32]
            section_name = section_names.get(section_id, "Seccion desconocida")
            self.audit_access("NAVEGACION", section_name, f"seccion={section_id}")
            self.send_json({"status": "recorded"})
            return

        action_names = {
            "/api/generate": "GENERAR SPRITES",
            "/api/train/start": "INICIAR ENTRENAMIENTO",
            "/api/train/pause": "PAUSAR ENTRENAMIENTO",
            "/api/train/resume": "REANUDAR ENTRENAMIENTO",
            "/api/train/stop": "DETENER ENTRENAMIENTO",
            "/api/train/respawn": "RESPAWN ENTRENAMIENTO",
            "/api/regenerate_frame": "REGENERAR FRAME",
            "/api/frame_review/register": "REGISTRAR REVIEW DE FRAME",
            "/api/frame_review/reevaluate": "REEVALUAR REVIEW DE FRAME",
            "/api/frame_review/approve": "APROBAR REVIEW DE FRAME",
            "/api/frame_review/reject": "RECHAZAR REVIEW DE FRAME",
        }
        action_name = action_names.get(parsed.path, "SOLICITUD")
        details = ""
        if parsed.path in ["/api/train/start", "/api/train/resume"]:
            details = f"epocas={body.get('epochs', 50)} | batch={body.get('batch_size', 4)}"
        elif parsed.path == "/api/train/respawn":
            details = f"epoca={body.get('epoch', '?')}"
        elif parsed.path == "/api/generate":
            details = f"personaje={body.get('character', 'desconocido')} | modo={body.get('mode', 'full')}"
        elif parsed.path == "/api/regenerate_frame":
            details = f"frame={body.get('frame_idx', '?')}"
        self.audit_access("ACCION", action_name, details)

        if parsed.path == "/api/frame_review/register":
            character_id = str(body.get("character_id") or body.get("character") or "unknown")
            variant = str(body.get("variant") or "default")
            frame_idx = body.get("frame_idx")
            if frame_idx is None:
                self.send_json({"error": "frame_idx es obligatorio"}, status=400)
                return
            generated_path = body.get("generated_frame_path")
            if not generated_path:
                self.send_json({"error": "generated_frame_path es obligatorio"}, status=400)
                return
            generated_abs = Path(generated_path)
            if not generated_abs.is_absolute():
                generated_abs = PROJECT_ROOT / generated_abs
            target_raw = body.get("target_frame_path")
            target_abs = None
            if target_raw:
                target_abs = Path(target_raw)
                if not target_abs.is_absolute():
                    target_abs = PROJECT_ROOT / target_abs
            try:
                review = FRAME_REVIEW_MANAGER.register_frame(
                    character_id=character_id,
                    variant=variant,
                    frame_idx=int(frame_idx),
                    generated_frame_path=str(generated_abs),
                    target_frame_path=str(target_abs) if target_abs is not None else None,
                    metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else {},
                )
            except ValueError as exc:
                self.send_json({"error": str(exc)}, status=400)
                return
            self.send_json({"status": "registered", "review": review})
            return

        elif parsed.path == "/api/frame_review/reevaluate":
            review_id = body.get("review_id")
            if not review_id:
                self.send_json({"error": "review_id es obligatorio"}, status=400)
                return
            try:
                review = FRAME_REVIEW_MANAGER.reevaluate_frame(
                    str(review_id),
                    quality=body.get("quality") if isinstance(body.get("quality"), dict) else None,
                    issues=list(body.get("issues", [])) if isinstance(body.get("issues"), list) else None,
                    metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
                )
            except KeyError as exc:
                self.send_json({"error": str(exc)}, status=404)
                return
            self.send_json({"status": "reevaluated", "review": review})
            return

        elif parsed.path == "/api/frame_review/approve":
            review_id = body.get("review_id")
            if not review_id:
                self.send_json({"error": "review_id es obligatorio"}, status=400)
                return
            try:
                review = FRAME_REVIEW_MANAGER.approve_frame(
                    str(review_id),
                    user=str(body.get("user") or "studio_user"),
                    comment=body.get("comment"),
                )
            except KeyError as exc:
                self.send_json({"error": str(exc)}, status=404)
                return
            self.send_json({"status": "approved", "review": review})
            return

        elif parsed.path == "/api/frame_review/reject":
            review_id = body.get("review_id")
            if not review_id:
                self.send_json({"error": "review_id es obligatorio"}, status=400)
                return
            try:
                review = FRAME_REVIEW_MANAGER.reject_frame(
                    str(review_id),
                    user=str(body.get("user") or "studio_user"),
                    reason=body.get("reason"),
                )
            except KeyError as exc:
                self.send_json({"error": str(exc)}, status=404)
                return
            self.send_json({"status": "rejected", "review": review})
            return

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
            prev_data = {}
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

            dataset_layout = prev_data.get("dataset_layout", {})
            if isinstance(dataset_layout, dict):
                dataset_layout = dict(dataset_layout)
                dataset_layout["requires_new_era"] = False
                dataset_layout["training_era_started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                dataset_layout = {}

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
                "total_frames": dataset_layout.get("sample_count", 1008),
                "past_eras": past_eras,
                "recovery_events": prev_data.get("recovery_events", []),
                "dataset_layout": dataset_layout,
            }
            try:
                with open(status_path, "w", encoding="utf-8") as f:
                    json.dump(init_status, f, indent=2)
            except Exception:
                pass

            LAST_START_TIME = time.time()
            train_env = os.environ.copy()
            train_env["PYTHONUNBUFFERED"] = "1"
            GLOBAL_TRAINING_PROC = launch_training_process(
                [python_exe, str(train_script), "--epochs", str(epochs), "--batch_size", str(batch_size), "--mode", "start"],
                train_env,
            )
            self.send_json({"status": "started", "message": f"Entrenamiento de {msg_title} iniciado ({epochs} épocas)."})
            return

        elif parsed.path == "/api/train/pause":
            proc_active, _ = check_gpu_training_status()
            pause_flag = PROJECT_ROOT / "pause_training.flag"
            status_file = PROJECT_ROOT / "training_status.json"
            if not proc_active:
                if pause_flag.exists():
                    try: pause_flag.unlink()
                    except Exception: pass
                if status_file.exists():
                    try:
                        with open(status_file, "r", encoding="utf-8") as f:
                            cur = json.load(f)
                        cur["status"] = "PAUSADO"
                        with open(status_file, "w", encoding="utf-8") as f:
                            json.dump(cur, f, indent=2)
                    except Exception:
                        pass
                self.send_json({"status": "paused", "message": "El entrenamiento ya se encuentra pausado y seguro."})
                return

            pause_flag.touch(exist_ok=True)
            if status_file.exists():
                try:
                    with open(status_file, "r", encoding="utf-8") as f:
                        cur = json.load(f)
                    cur["status"] = "PAUSANDO..."
                    with open(status_file, "w", encoding="utf-8") as f:
                        json.dump(cur, f, indent=2)
                except Exception:
                    pass
            self.send_json({"status": "pausing", "message": "Señal de pausa enviada. Completando época en curso para pausar en frontera limpia..."})
            return

        elif parsed.path == "/api/train/resume":
            training_active, training_status = check_gpu_training_status()
            if training_active:
                self.send_json({"error": "El entrenamiento ya se encuentra en ejecución activa."}, status=409)
                return
            if str(training_status.get("status", "")).upper() == "DATASET_ACTUALIZADO":
                self.send_json({
                    "error": (
                        "Los sprites y la caché cambiaron de escala. Pulsa Iniciar para crear una era nueva "
                        "compatible; Reanudar mezclaría el optimizador y las métricas anteriores."
                    )
                }, status=409)
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
                    recovery = cur.get("recovery", {})
                    cur["status"] = "RECUPERANDO" if isinstance(recovery, dict) and recovery.get("active") else "REANUDANDO"
                    cur["timestamp"] = time.strftime("%H:%M:%S")
                    with open(status_file, "w", encoding="utf-8") as f:
                        json.dump(cur, f, indent=2)
                except Exception:
                    pass

            LAST_START_TIME = time.time()
            train_env = os.environ.copy()
            train_env["PYTHONUNBUFFERED"] = "1"
            GLOBAL_TRAINING_PROC = launch_training_process(
                [python_exe, str(train_script), "--epochs", str(epochs), "--batch_size", str(batch_size), "--mode", "resume"],
                train_env,
            )
            self.send_json({"status": "resumed", "message": f"Entrenamiento de {msg_title} reanudado desde el checkpoint seguro disponible."})
            return

        elif parsed.path == "/api/train/stop":
            proc_active, _ = check_gpu_training_status()
            stop_flag = PROJECT_ROOT / "stop_training.flag"
            status_file = PROJECT_ROOT / "training_status.json"
            if not proc_active:
                if stop_flag.exists():
                    try: stop_flag.unlink()
                    except Exception: pass
                if status_file.exists():
                    try:
                        with open(status_file, "r", encoding="utf-8") as f:
                            cur = json.load(f)
                        cur["status"] = "DETENIDO"
                        with open(status_file, "w", encoding="utf-8") as f:
                            json.dump(cur, f, indent=2)
                    except Exception:
                        pass
                self.send_json({"status": "stopped", "message": "El entrenamiento ya se encuentra detenido."})
                return

            stop_flag.touch(exist_ok=True)
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
            max_valid_epoch = get_valid_snapshot_max_epoch()
            if max_valid_epoch is not None and target_epoch > max_valid_epoch:
                self.send_json({
                    "error": f"El snapshot de la época {target_epoch} pertenece a la trayectoria descartada. Máximo válido actual: {max_valid_epoch}."
                }, status=409)
                return
            
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

            LAST_START_TIME = time.time()
            train_env = os.environ.copy()
            train_env["PYTHONUNBUFFERED"] = "1"
            GLOBAL_TRAINING_PROC = launch_training_process([
                python_exe, str(train_script),
                "--epochs", str(epochs),
                "--batch_size", str(batch_size),
                "--mode", "resume",
                "--respawn_epoch", str(target_epoch)
            ], train_env)
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


def open_browser(host: str, port: int):
    import webbrowser
    time.sleep(1.0)
    url = f"http://{host}:{port}/sprite_studio.html"
    print(f"[Sprite Studio] Abriendo navegador en: {url}")
    webbrowser.open(url)


def main():
    global PORT
    PORT = 8080
    os.chdir(PROJECT_ROOT)

    if not acquire_server_lock():
        print("=" * 70)
        print(f"  [ERROR] Ya existe otro Sprite Studio activo en {SERVER_HOST}:{PORT}.")
        print("  Cierra la instancia anterior antes de iniciar otra.")
        print("  Esto evita que el entrenamiento envie sus logs a una terminal oculta.")
        print("=" * 70)
        raise SystemExit(1)

    try:
        server = ThreadingHTTPServer((SERVER_HOST, PORT), SpriteStudioHandler)
    except OSError as error:
        release_server_lock()
        print("=" * 70)
        print(f"  [ERROR] Ya existe otro Sprite Studio usando {SERVER_HOST}:{PORT}.")
        print("  Cierra la instancia anterior antes de iniciar otra.")
        print("  Esto evita que el entrenamiento envie sus logs a una terminal oculta.")
        print(f"  Detalle: {error}")
        print("=" * 70)
        raise SystemExit(1)

    print("=" * 70)
    print("  SPRITE STUDIO - VILLA DEL CHEF (ESTUDIO LOCAL DE PIXEL ART)")
    print(f"  Directorio: {PROJECT_ROOT}")
    print(f"  Servidor activo en: http://{SERVER_HOST}:{PORT}")
    print("=" * 70)

    # Abrir navegador automáticamente salvo en reinicios técnicos en segundo plano.
    if os.environ.get("SPRITE_STUDIO_NO_BROWSER") != "1":
        t = threading.Thread(target=open_browser, args=(SERVER_HOST, PORT), daemon=True)
        t.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Sprite Studio] Servidor detenido por el usuario.")
    finally:
        server.server_close()
        release_server_lock()


if __name__ == "__main__":
    main()
