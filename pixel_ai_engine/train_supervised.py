import os
import sys
import time
import json
import math
import shutil
from typing import Any, Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from PIL import Image
import numpy as np
import argparse

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pixel_ai_engine.config import (
    MODEL_RESOLUTION,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    DEVICE,
    USE_AMP
)
from pixel_ai_engine.models import PixelArtUNetGenerator, PixelArtPatchDiscriminator
from pixel_ai_engine.palette_remap import extract_character_palette, remap_image_to_palette, clean_orphan_pixels

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR = CHECKPOINT_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SAMPLES_DIR = PROJECT_ROOT / "training_samples"
TRAIN_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR = TRAIN_SAMPLES_DIR / "audit_history"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

STATUS_FILE = PROJECT_ROOT / "training_status.json"
STOP_FLAG_FILE = PROJECT_ROOT / "stop_training.flag"
PAUSE_FLAG_FILE = PROJECT_ROOT / "pause_training.flag"

CACHE_PATH = PROJECT_ROOT / "dataset_supervisado" / "supervised_cache_8x12.pt"

# -------------------------------------------------------------
# 1. PÉRDIDA DE BORDES ACELERADA EN GPU CON KORNIA (DE FORGE)
# -------------------------------------------------------------
try:
    import kornia
    class KorniaSobelLoss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, pred, target):
            p_rgb = pred[:, :3]
            t_rgb = target[:, :3]
            p_grad = kornia.filters.sobel(p_rgb)
            t_grad = kornia.filters.sobel(t_rgb)
            return nn.functional.smooth_l1_loss(p_grad, t_grad)
    HAS_KORNIA = True
except Exception:
    HAS_KORNIA = False

class FallbackSobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, pred, target):
        loss = 0.0
        for c in range(min(3, pred.shape[1])):
            p_c = pred[:, c:c+1]
            t_c = target[:, c:c+1]
            p_gx = nn.functional.conv2d(p_c, self.kx, padding=1)
            p_gy = nn.functional.conv2d(p_c, self.ky, padding=1)
            t_gx = nn.functional.conv2d(t_c, self.kx, padding=1)
            t_gy = nn.functional.conv2d(t_c, self.ky, padding=1)
            p_edge = torch.sqrt(torch.clamp(p_gx**2 + p_gy**2, min=1e-5))
            t_edge = torch.sqrt(torch.clamp(t_gx**2 + t_gy**2, min=1e-5))
            loss += nn.functional.smooth_l1_loss(p_edge, t_edge)
        return loss / 3.0

# -------------------------------------------------------------
# 2. DATASET CON AUMENTO INTELIGENTE DE DATOS (ALBUMENTATIONS)
# -------------------------------------------------------------
try:
    import albumentations as A
    try:
        # Desactivar explícitamente tono y saturación para conservar la identidad exacta del personaje
        AUG_PIPELINE = A.Compose([
            A.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.0, hue=0.0, p=0.4),
        ])
    except Exception:
        AUG_PIPELINE = A.Compose([
            A.ColorJitter(brightness=(0.97, 1.03), contrast=(0.97, 1.03), saturation=(1.0, 1.0), hue=(0.0, 0.0), p=0.4),
        ])
    HAS_ALBUMENTATIONS = True
except Exception:
    HAS_ALBUMENTATIONS = False

class SupervisedTensorDataset(Dataset):
    def __init__(self, cache_path, augment: bool = True):
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache no encontrada: {cache_path}. Ejecuta prepare_supervised_dataset.py primero.")
        self.samples = torch.load(cache_path)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        front = s["front_tensor"] # (3, H, W)
        f_idx = s["frame_idx"]
        target = s["target_tensor"] # (4, H, W)

        # Si el aumento esta activo, aplicar sutil variacion al frontal para robustez
        if self.augment and HAS_ALBUMENTATIONS and torch.rand(1).item() < 0.35:
            f_np = ((front.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            aug_np = AUG_PIPELINE(image=f_np)["image"]
            aug_t = torch.from_numpy(aug_np.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
            front = aug_t

        return front, f_idx, target, s["char_id"]

def get_gpu_temperature():
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, text=True, timeout=2
        )
        return int(res.stdout.strip().split("\n")[0])
    except Exception:
        return 55

def get_available_snapshots():
    snaps = []
    if SNAPSHOTS_DIR.exists():
        for p in sorted(SNAPSHOTS_DIR.glob("checkpoint_epoch_*.pt")):
            name = p.stem
            try:
                ep = int(name.replace("checkpoint_epoch_", ""))
                img_url = f"/training_samples/audit_history/preview_epoch_{ep:03d}.png"
                snaps.append({"epoch": ep, "file": p.name, "preview_url": img_url})
            except Exception:
                pass
    return snaps

def update_status(epoch, total_epochs, status_str, g_loss, d_loss, l1_val, edge_val, start_time, lr_val=1.5e-4):
    elapsed = round(time.time() - start_time, 1)
    history = []
    best_loss = None
    initial_loss = None

    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                prev = json.load(f)
                history = prev.get("history", [])
                best_loss = prev.get("best_loss")
                initial_loss = prev.get("initial_loss")
        except Exception:
            history = []

    def safe_num(v, default=0.0):
        try:
            f = float(v)
            return f if math.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    # Comprobar finitud de métricas para evitar NaN en JSON y estado
    is_finite_g = isinstance(g_loss, (int, float)) and math.isfinite(float(g_loss))
    if not is_finite_g and status_str == "ENTRENANDO":
        status_str = "ERROR_NAN"
        print(f"[!] ADVERTENCIA CRÍTICA: Gradiente o pérdida no finita detectada: g_loss={g_loss}")

    f_loss = round(safe_num(g_loss, 0.0), 4)
    f_d = round(safe_num(d_loss, 0.0), 4)
    f_l1 = round(safe_num(l1_val, 0.0), 4)
    f_edge = round(safe_num(edge_val, 0.0), 4)
    f_lr = safe_num(lr_val, 1.5e-4)

    if f_loss > 0 and math.isfinite(f_loss):
        if initial_loss is None or not math.isfinite(initial_loss):
            initial_loss = f_loss
        if best_loss is None or not math.isfinite(best_loss) or f_loss < best_loss:
            best_loss = f_loss

        history = [h for h in history if h.get("epoch") != int(epoch)]
        history.append({
            "epoch": int(epoch),
            "loss": f_loss,
            "g_loss": f_loss,
            "d_loss": f_d,
            "l1_loss": f_l1,
            "edge_loss": f_edge,
            "lr": f_lr,
            "gpu_temp": get_gpu_temperature(),
            "elapsed_sec": elapsed
        })
        history.sort(key=lambda x: x["epoch"])

    loss_reduction_pct = 0.0
    if initial_loss and f_loss and initial_loss > 0 and math.isfinite(initial_loss):
        loss_reduction_pct = round(((initial_loss - f_loss) / initial_loss) * 100.0, 1)

    sec_per_epoch = (elapsed / max(1, epoch)) if epoch > 0 else 0
    rem_epochs = max(0, total_epochs - epoch)
    eta_sec = round(sec_per_epoch * rem_epochs)
    fps = round(1008.0 / max(1.0, sec_per_epoch), 2) if sec_per_epoch > 0 else 0.0

    status_data = {
        "epoch": int(epoch),
        "total_epochs": int(total_epochs),
        "phase": "Supervisado Pix2Pix + Kornia GPU + 8bit AdamW",
        "phase_id": 3,
        "status": status_str,
        "g_loss": f_loss,
        "d_loss": f_d,
        "l1_loss": f_l1,
        "edge_loss": f_edge,
        "best_loss": round(best_loss, 4) if (best_loss is not None and math.isfinite(best_loss)) else None,
        "initial_loss": round(initial_loss, 4) if (initial_loss is not None and math.isfinite(initial_loss)) else None,
        "loss_reduction_pct": loss_reduction_pct,
        "lr": f_lr,
        "gpu_temp": get_gpu_temperature(),
        "elapsed_sec": elapsed,
        "eta_sec": eta_sec,
        "fps": fps,
        "timestamp": time.strftime("%H:%M:%S"),
        "total_frames": 1008,
        "snapshots": get_available_snapshots(),
        "history": history
    }
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)

def save_checkpoint(file_path: Path, epoch: int, loss: float, best_loss: float,
                    generator: nn.Module, discriminator: nn.Module,
                    opt_g: Any = None, opt_d: Any = None, scaler_g: Any = None, scaler_d: Any = None):
    """Guarda un checkpoint completo con formato unificado y preservación estricta de métricas y estados."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    f_loss = float(loss) if (loss is not None and math.isfinite(float(loss))) else 0.0
    f_best = float(best_loss) if (best_loss is not None and math.isfinite(float(best_loss))) else 999.0
    state = {
        "epoch": int(epoch),
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "loss": f_loss,
        "best_loss": f_best,
        "opt_g": opt_g.state_dict() if opt_g is not None else None,
        "opt_d": opt_d.state_dict() if opt_d is not None else None,
        "scaler_g": scaler_g.state_dict() if (scaler_g is not None and hasattr(scaler_g, "state_dict")) else None,
        "scaler_d": scaler_d.state_dict() if (scaler_d is not None and hasattr(scaler_d, "state_dict")) else None,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, file_path)

def generate_preview(generator, dataset, epoch_label=None):
    generator.eval()
    with torch.no_grad():
        sample_indices = [0, 96, 288, 384] if len(dataset) > 400 else list(range(min(4, len(dataset))))
        from pixel_ai_engine.dataset import TemplateManager
        tmpl_path = PROJECT_ROOT / "dataset_moldes" / "plantilla de los spritesheets.png"
        if not tmpl_path.exists():
            tmpl_path = PROJECT_ROOT / "plantilla de los spritesheets.png"
        tm = TemplateManager(tmpl_path, MODEL_RESOLUTION, 12, 8, 1024, 1536)

        cards = []
        for s_idx in sample_indices:
            # Evaluar siempre sobre la muestra fija sin aumentos de datos aleatorios
            s = dataset.samples[s_idx]
            front = s["front_tensor"]
            f_idx = s["frame_idx"]
            target = s["target_tensor"]
            char_id = s["char_id"]
            pose = tm.get_frame_tensor(f_idx)

            cond = torch.cat([front, pose], dim=0).unsqueeze(0).to(DEVICE)
            with autocast(enabled=USE_AMP):
                out = generator(cond).squeeze(0)

            # 1. Frontal Chibi (Referencia real original sin aumentos)
            f_np = ((front.permute(1,2,0).cpu().numpy() + 1.0) * 127.5).clip(0,255).astype(np.uint8)
            f_pil = Image.fromarray(f_np, "RGB").resize((128, 128), Image.Resampling.NEAREST)

            # 2. Pose (Molde geométrico)
            p_np = ((pose.permute(1,2,0).cpu().numpy() + 1.0) * 127.5).clip(0,255).astype(np.uint8)
            p_pil = Image.fromarray(p_np, "RGB").resize((128, 128), Image.Resampling.NEAREST)

            # 3. Prediccion IA Cruda (Sin retoques - para auditoría transparente de fallos)
            pred_np = ((out[:3].permute(1,2,0).cpu().numpy() + 1.0) * 127.5).clip(0,255).astype(np.uint8)
            alpha_np = ((out[3].cpu().numpy() + 1.0) * 127.5).clip(0,255).astype(np.uint8)
            pred_rgba = np.dstack([pred_np, alpha_np])
            raw_pred_pil = Image.fromarray(pred_rgba, "RGBA").resize((128, 128), Image.Resampling.NEAREST)
            
            # 4. Predicción IA con Remapeo de Paleta + Filtro Morfológico Anti-Hollín + Alfa Puro
            char_palette = extract_character_palette(f_pil, include_props=True)
            snapped_pil = remap_image_to_palette(raw_pred_pil, char_palette, tolerance=35.0, binarize_alpha=True)
            pred_pil = clean_orphan_pixels(snapped_pil, min_connected_size=3, binarize=True)

            # 5. Ground Truth Real
            tgt_np = ((target[:3].permute(1,2,0).cpu().numpy() + 1.0) * 127.5).clip(0,255).astype(np.uint8)
            t_alpha_np = ((target[3].cpu().numpy() + 1.0) * 127.5).clip(0,255).astype(np.uint8)
            tgt_rgba = np.dstack([tgt_np, t_alpha_np])
            tgt_pil = Image.fromarray(tgt_rgba, "RGBA").resize((128, 128), Image.Resampling.NEAREST)

            # Tira comparativa con 5 paneles: Frontal | Pose | IA Cruda | IA Remapeada | Ground Truth
            strip = Image.new("RGBA", (128 * 5 + 35, 128 + 20), (16, 18, 26, 255))
            strip.paste(f_pil, (5, 10))
            strip.paste(p_pil, (128 + 10, 10))
            strip.paste(raw_pred_pil, (256 + 15, 10))
            strip.paste(pred_pil, (384 + 20, 10))
            strip.paste(tgt_pil, (512 + 25, 10))
            cards.append(strip)

        total_w = cards[0].width
        total_h = sum(c.height for c in cards) + 30
        comp_img = Image.new("RGBA", (total_w, total_h), (11, 13, 19, 255))
        y = 10
        for c in cards:
            comp_img.paste(c, (0, y))
            y += c.height + 5

        comp_img.save(TRAIN_SAMPLES_DIR / "latest_detail_comparison.png")
        comp_img.save(TRAIN_SAMPLES_DIR / "latest_preview.png")
        if epoch_label is not None:
            comp_img.save(AUDIT_DIR / f"preview_epoch_{epoch_label:03d}.png")

def train_supervised_model(epochs: int = 150, batch_size: int = 4, lr: float = 1.5e-4, mode: str = "resume", respawn_epoch: int = None):
    # Limpiar banderas anteriores
    if STOP_FLAG_FILE.exists():
        try: STOP_FLAG_FILE.unlink()
        except Exception: pass
    if PAUSE_FLAG_FILE.exists():
        try: PAUSE_FLAG_FILE.unlink()
        except Exception: pass

    dataset = SupervisedTensorDataset(CACHE_PATH, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    from pixel_ai_engine.dataset import TemplateManager
    tmpl_path = PROJECT_ROOT / "dataset_moldes" / "plantilla de los spritesheets.png"
    if not tmpl_path.exists():
        tmpl_path = PROJECT_ROOT / "plantilla de los spritesheets.png"
    tmpl_mgr = TemplateManager(tmpl_path, MODEL_RESOLUTION, 12, 8, 1024, 1536)

    generator = PixelArtUNetGenerator().to(DEVICE)
    discriminator = PixelArtPatchDiscriminator().to(DEVICE)

    start_epoch = 1
    best_loss = None
    loaded_ckpt = None

    # Logica de Respawn o Reanudacion
    if respawn_epoch is not None and respawn_epoch > 0:
        respawn_candidate = SNAPSHOTS_DIR / f"checkpoint_epoch_{respawn_epoch:03d}.pt"
        if not respawn_candidate.exists():
            respawn_candidate = SNAPSHOTS_DIR / f"checkpoint_epoch_{respawn_epoch}.pt"
        if respawn_candidate.exists():
            loaded_ckpt = torch.load(respawn_candidate, map_location=DEVICE)
            generator.load_state_dict(loaded_ckpt["generator"])
            if "discriminator" in loaded_ckpt:
                discriminator.load_state_dict(loaded_ckpt["discriminator"])
            start_epoch = respawn_epoch + 1
            best_loss = loaded_ckpt.get("best_loss", loaded_ckpt.get("loss", None))
            print(f"\n[RESPAWN] == RESPAWN EXITOSO: Regresando al punto de guardado EPOCA {respawn_epoch} ==\n")
        else:
            print(f"[!] Aviso: No se encontro snapshot para epoca {respawn_epoch}. Iniciando estandar.")
    elif mode == "resume":
        ckpt_candidate = CHECKPOINT_DIR / "latest_checkpoint.pt"
        if not ckpt_candidate.exists():
            ckpt_candidate = CHECKPOINT_DIR / "best_generator.pt"
        if ckpt_candidate.exists():
            try:
                loaded_ckpt = torch.load(ckpt_candidate, map_location=DEVICE)
                if isinstance(loaded_ckpt, dict) and "generator" in loaded_ckpt:
                    generator.load_state_dict(loaded_ckpt["generator"])
                    if "discriminator" in loaded_ckpt:
                        discriminator.load_state_dict(loaded_ckpt["discriminator"])
                    start_epoch = loaded_ckpt.get("epoch", 0) + 1
                    best_loss = loaded_ckpt.get("best_loss", None)
                    print(f"[OK] Reanudando entrenamiento desde epoca {start_epoch}")
            except Exception as e:
                print(f"[!] Error al reanudar checkpoint: {e}")
    elif mode == "start":
        # Warm start: Si existe best_generator o base_generator, iniciar desde pesos entrenados
        warm_ckpt = CHECKPOINT_DIR / "best_generator.pt"
        if not warm_ckpt.exists():
            warm_ckpt = CHECKPOINT_DIR / "base_generator_16x4.pt"
        if warm_ckpt.exists():
            try:
                ckpt = torch.load(warm_ckpt, map_location=DEVICE)
                gen_state = ckpt.get("generator", ckpt) if isinstance(ckpt, dict) else ckpt
                generator.load_state_dict(gen_state, strict=False)
                if isinstance(ckpt, dict):
                    best_loss = ckpt.get("best_loss", ckpt.get("loss", None))
                print(f"[OK] Warm Start: Inicializando generador desde pesos existentes: {warm_ckpt.name}")
            except Exception as e:
                print(f"[!] Aviso: No se pudo cargar warm start: {e}. Iniciando desde inicializacion normal.")

    # Recuperar best_loss de best_generator.pt para archivos antiguos compatibles
    if best_loss is None or (isinstance(best_loss, (int, float)) and best_loss >= 990.0):
        best_ckpt_file = CHECKPOINT_DIR / "best_generator.pt"
        if best_ckpt_file.exists():
            try:
                b_data = torch.load(best_ckpt_file, map_location="cpu")
                if isinstance(b_data, dict):
                    cand_best = b_data.get("best_loss", b_data.get("loss", 999.0))
                    if cand_best is not None and math.isfinite(cand_best):
                        best_loss = float(cand_best)
                        print(f"[OK] best_loss historico recuperado de best_generator.pt: {best_loss:.4f}")
            except Exception:
                pass
    if best_loss is None or not math.isfinite(best_loss):
        best_loss = 999.0

    total_target_epochs = (start_epoch - 1) + epochs if mode == "resume" and respawn_epoch is None else (start_epoch - 1 + epochs if respawn_epoch else epochs)

    print("=" * 70)
    print("  ENTRENAMIENTO SUPERVISADO CON KORNIA GPU + 8-BIT ADAMW + RESPAWN")
    print(f"  Modo: {mode.upper()} | Epocas: {start_epoch} a {total_target_epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  Muestras: {len(dataset)} pares Ground-Truth | Dispositivo: {DEVICE} | Mejor Loss Inicial: {best_loss:.4f}")
    print("=" * 70)

    # Optimizadores 8-bit AdamW de Forge
    try:
        import bitsandbytes as bnb
        opt_g = bnb.optim.AdamW8bit(generator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
        opt_d = bnb.optim.AdamW8bit(discriminator.parameters(), lr=lr * 0.5, betas=(0.5, 0.999), weight_decay=1e-4)
        print("[OK] Optimizador BitsAndBytes 8-bit AdamW activo (VRAM: ~1.5 GB)")
    except Exception as e:
        opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))
        print(f"[!] Optimizador estandar PyTorch Adam activo: {e}")

    scaler_g = GradScaler(enabled=USE_AMP, init_scale=2048.0)
    scaler_d = GradScaler(enabled=USE_AMP, init_scale=2048.0)

    # Restaurar estados de optimizadores, escaladores y RNG si estan disponibles en el checkpoint
    if loaded_ckpt is not None:
        if "opt_g" in loaded_ckpt and loaded_ckpt["opt_g"] is not None:
            try:
                opt_g.load_state_dict(loaded_ckpt["opt_g"])
                print("[OK] Estado de optimizador opt_g restaurado con exito.")
            except Exception as e:
                print(f"[!] Aviso al restaurar estado de opt_g: {e}")
        if "opt_d" in loaded_ckpt and loaded_ckpt["opt_d"] is not None:
            try:
                opt_d.load_state_dict(loaded_ckpt["opt_d"])
                print("[OK] Estado de optimizador opt_d restaurado con exito.")
            except Exception as e:
                print(f"[!] Aviso al restaurar estado de opt_d: {e}")
        if "scaler_g" in loaded_ckpt and loaded_ckpt["scaler_g"] is not None and scaler_g is not None:
            try: scaler_g.load_state_dict(loaded_ckpt["scaler_g"])
            except Exception: pass
        if "scaler_d" in loaded_ckpt and loaded_ckpt["scaler_d"] is not None and scaler_d is not None:
            try: scaler_d.load_state_dict(loaded_ckpt["scaler_d"])
            except Exception: pass
        if "rng_state" in loaded_ckpt and loaded_ckpt["rng_state"] is not None:
            try: torch.set_rng_state(loaded_ckpt["rng_state"])
            except Exception: pass
        if "cuda_rng_state" in loaded_ckpt and loaded_ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
            try: torch.cuda.set_rng_state_all(loaded_ckpt["cuda_rng_state"])
            except Exception: pass

    criterion_l1 = nn.SmoothL1Loss()
    criterion_edge = KorniaSobelLoss().to(DEVICE) if HAS_KORNIA else FallbackSobelLoss().to(DEVICE)
    if HAS_KORNIA:
        print("[OK] Perdida de bordes Kornia Sobel acelerada en GPU activa")
    criterion_bce = nn.BCEWithLogitsLoss()

    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=max(1, total_target_epochs - start_epoch + 1), eta_min=1e-6)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=max(1, total_target_epochs - start_epoch + 1), eta_min=1e-6)

    start_time = time.time()
    avg_g = 0.0
    avg_d = 0.0
    avg_l1 = 0.0
    avg_edge = 0.0

    for epoch in range(start_epoch, total_target_epochs + 1):
        if PAUSE_FLAG_FILE.exists():
            print("\n[PAUSA] Senal de pausa recibida. Guardando checkpoint...")
            try: PAUSE_FLAG_FILE.unlink()
            except Exception: pass
            save_checkpoint(CHECKPOINT_DIR / 'latest_checkpoint.pt', epoch - 1, avg_g, best_loss, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d)
            update_status(epoch - 1, total_target_epochs, "PAUSADO", avg_g, avg_d, avg_l1, avg_edge, start_time, lr)
            return

        if STOP_FLAG_FILE.exists():
            print("\n[STOP] Senal de detencion recibida. Guardando...")
            try: STOP_FLAG_FILE.unlink()
            except Exception: pass
            save_checkpoint(CHECKPOINT_DIR / 'latest_checkpoint.pt', epoch - 1, avg_g, best_loss, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d)
            update_status(epoch - 1, total_target_epochs, "DETENIDO", avg_g, avg_d, avg_l1, avg_edge, start_time, lr)
            return

        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_l1 = 0.0
        epoch_edge = 0.0

        for batch_i, (fronts, f_indices, targets, _) in enumerate(dataloader):
            if batch_i % 8 == 0:
                curr_batches = max(1, batch_i)
                curr_g = epoch_g_loss / curr_batches
                curr_d = epoch_d_loss / curr_batches
                curr_l1 = epoch_l1 / curr_batches
                curr_edge = epoch_edge / curr_batches
                if PAUSE_FLAG_FILE.exists():
                    print('\n[PAUSA] Senal de pausa en lote. Guardando checkpoint...')
                    try: PAUSE_FLAG_FILE.unlink()
                    except Exception: pass
                    save_checkpoint(CHECKPOINT_DIR / 'latest_checkpoint.pt', epoch, curr_g, best_loss, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d)
                    update_status(epoch, total_target_epochs, 'PAUSADO', curr_g, curr_d, curr_l1, curr_edge, start_time, lr)
                    return
                if STOP_FLAG_FILE.exists():
                    print('\n[STOP] Senal de detencion en lote. Guardando checkpoint...')
                    try: STOP_FLAG_FILE.unlink()
                    except Exception: pass
                    save_checkpoint(CHECKPOINT_DIR / 'latest_checkpoint.pt', epoch, curr_g, best_loss, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d)
                    update_status(epoch, total_target_epochs, 'DETENIDO', curr_g, curr_d, curr_l1, curr_edge, start_time, lr)
                    return

            fronts = fronts.to(DEVICE)
            targets = targets.to(DEVICE)
            poses = torch.stack([tmpl_mgr.get_frame_tensor(idx) for idx in f_indices]).to(DEVICE)
            cond = torch.cat([fronts, poses], dim=1)

            # Generador
            opt_g.zero_grad()
            with autocast(enabled=USE_AMP):
                preds = generator(cond)
                l1_color = criterion_l1(preds[:, :3], targets[:, :3]) * 5.0
                l1_alpha = criterion_l1(preds[:, 3:], targets[:, 3:]) * 2.5
                edge_loss = criterion_edge(preds[:, :3], targets[:, :3]) * 1.5

                # Discriminador: orden canonico (condition, target)
                d_fake = discriminator(cond, preds)
                adv_loss = criterion_bce(d_fake.float().clamp(-30.0, 30.0), torch.ones_like(d_fake).float()) * 0.05
                total_g = l1_color + l1_alpha + edge_loss + adv_loss

            scaler_g.scale(total_g).backward()
            scaler_g.unscale_(opt_g)
            g_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            if torch.isfinite(g_norm):
                scaler_g.step(opt_g)
            scaler_g.update()

            # Discriminador (Aprende mas lento para no aplastar al generador)
            opt_d.zero_grad()
            with autocast(enabled=USE_AMP):
                # Discriminador: orden canonico (condition, target)
                d_real = discriminator(cond, targets)
                d_fake_det = discriminator(cond, preds.detach())
                loss_d_real = criterion_bce(d_real.float().clamp(-30.0, 30.0), torch.ones_like(d_real).float())
                loss_d_fake = criterion_bce(d_fake_det.float().clamp(-30.0, 30.0), torch.zeros_like(d_fake_det).float())
                total_d = (loss_d_real + loss_d_fake) * 0.5

            scaler_d.scale(total_d).backward()
            scaler_d.unscale_(opt_d)
            d_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
            if torch.isfinite(d_norm):
                scaler_d.step(opt_d)
            scaler_d.update()

            epoch_g_loss += total_g.item()
            epoch_d_loss += total_d.item()
            epoch_l1 += l1_color.item()
            epoch_edge += edge_loss.item()

        n_batches = max(1, len(dataloader))
        avg_g = epoch_g_loss / n_batches
        avg_d = epoch_d_loss / n_batches
        avg_l1 = epoch_l1 / n_batches
        avg_edge = epoch_edge / n_batches

        print(f"Epoca [{epoch:03d}/{total_target_epochs:03d}] - G_Loss: {avg_g:.4f} | Color_L1: {avg_l1:.4f} | Borde: {avg_edge:.4f} | D_Loss: {avg_d:.4f}")

        # Guardar preview y actualizar status cada epoca
        update_status(epoch, total_target_epochs, "ENTRENANDO", avg_g, avg_d, avg_l1, avg_edge, start_time, lr)
        
        # Muestra visual
        save_audit = (epoch % 10 == 0)
        generate_preview(generator, dataset, epoch_label=epoch if save_audit else None)

        # -------------------------------------------------------------
        # SISTEMA DE RESPAWN: Guardado cada 10 epocas con estado completo
        # -------------------------------------------------------------
        if epoch % 10 == 0:
            snapshot_file = SNAPSHOTS_DIR / f"checkpoint_epoch_{epoch:03d}.pt"
            save_checkpoint(snapshot_file, epoch, avg_g, best_loss, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d)
            gen_only_file = SNAPSHOTS_DIR / f"generator_epoch_{epoch:03d}.pt"
            torch.save(generator.state_dict(), gen_only_file)
            print(f"[RESPAWN SNAPSHOT] Punto de restauracion guardado: Epoca {epoch:03d} (Loss: {avg_g:.4f}, Best: {best_loss:.4f})")

        scheduler_g.step()
        scheduler_d.step()

        # Checkpoints de mejor rendimiento y ultimo (preservando siempre best_loss)
        if avg_g < best_loss and math.isfinite(avg_g):
            best_loss = avg_g
            save_checkpoint(CHECKPOINT_DIR / "best_generator.pt", epoch, avg_g, best_loss, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d)
            print(f"[*] ¡Nuevo mejor modelo registrado! G_Loss: {best_loss:.4f}")

        save_checkpoint(CHECKPOINT_DIR / "latest_checkpoint.pt", epoch, avg_g, best_loss, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d)

    update_status(total_target_epochs, total_target_epochs, "COMPLETADO", avg_g, avg_d, avg_l1, avg_edge, start_time, lr)
    print("\n[OK] Ciclo de entrenamiento supervisado finalizado exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--mode", type=str, default="resume", choices=["start", "resume"])
    parser.add_argument("--respawn_epoch", type=int, default=None, help="Epoca exacta a la cual rebobinar (Respawn)")
    args = parser.parse_args()
    train_supervised_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mode=args.mode,
        respawn_epoch=args.respawn_epoch
    )
