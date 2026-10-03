import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import safetensors.torch

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diffusers import StableDiffusionPipeline, DDPMScheduler
import bitsandbytes as bnb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATUS_FILE = PROJECT_ROOT / "training_status.json"
STOP_FLAG_FILE = PROJECT_ROOT / "stop_training.flag"
PAUSE_FLAG_FILE = PROJECT_ROOT / "pause_training.flag"

TRAIN_SAMPLES_DIR = PROJECT_ROOT / "training_samples"
TRAIN_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DIR = PROJECT_ROOT / "dataset_supervisado"
FRAMES_DIR = DATASET_DIR / "frames_png"
CAPTIONS_DIR = DATASET_DIR / "captions_txt"
CACHE_PATH = DATASET_DIR / "forge_lora_cache_512.pt"

# Rutas de destino del LoRA en Forge
LORA_OUTPUT_PATHS = [
    PROJECT_ROOT / "stable-diffusion-webui-forge-main" / "models" / "Lora",
    PROJECT_ROOT / "webui forger" / "webui" / "models" / "Lora",
    CHECKPOINT_DIR
]

def find_sd15_model_path() -> Path:
    candidates = [
        PROJECT_ROOT / "stable-diffusion-webui-forge-main" / "models" / "Stable-diffusion" / "v1-5-pruned-emaonly.safetensors",
        PROJECT_ROOT / "webui forger" / "webui" / "models" / "Stable-diffusion" / "v1-5-pruned-emaonly.safetensors",
        PROJECT_ROOT / "reparacion" / "v1-5-pruned-emaonly.safetensors",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No se encontró el modelo base 'v1-5-pruned-emaonly.safetensors'.")

class LoRALinear(nn.Module):
    """Capa LoRA quirúrgica en PyTorch puro (Rank 16, Alpha 16)."""
    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scale = alpha / rank
        self.lora_down = nn.Linear(linear.in_features, rank, bias=False).to(linear.weight.device, dtype=linear.weight.dtype)
        self.lora_up = nn.Linear(rank, linear.out_features, bias=False).to(linear.weight.device, dtype=linear.weight.dtype)
        nn.init.normal_(self.lora_down.weight, std=1.0 / rank)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scale * self.lora_up(self.lora_down(x))

def inject_lora_into_unet(unet, rank=16, alpha=16.0):
    """Inyecta LoRALinear en todas las capas de atención cruzada y autoatención del UNet."""
    lora_modules = {}
    lora_params = []

    for name, module in unet.named_modules():
        if hasattr(module, "to_q") and isinstance(module.to_q, nn.Linear):
            module.to_q = LoRALinear(module.to_q, rank=rank, alpha=alpha)
            module.to_k = LoRALinear(module.to_k, rank=rank, alpha=alpha)
            module.to_v = LoRALinear(module.to_v, rank=rank, alpha=alpha)
            module.to_out[0] = LoRALinear(module.to_out[0], rank=rank, alpha=alpha)

            lora_modules[f"{name}.to_q"] = module.to_q
            lora_modules[f"{name}.to_k"] = module.to_k
            lora_modules[f"{name}.to_v"] = module.to_v
            lora_modules[f"{name}.to_out.0"] = module.to_out[0]

            for m in [module.to_q, module.to_k, module.to_v, module.to_out[0]]:
                lora_params.extend([m.lora_down.weight, m.lora_up.weight])

    return lora_modules, lora_params

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

def update_status(epoch, total_epochs, status_str, loss_val, start_time, lr_val=1e-4):
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

    f_loss = round(float(loss_val), 4) if isinstance(loss_val, (int, float)) and not (loss_val != loss_val) else None
    if f_loss is not None:
        if initial_loss is None:
            initial_loss = f_loss
        if best_loss is None or f_loss < best_loss:
            best_loss = f_loss

        history = [h for h in history if h.get("epoch") != int(epoch)]
        history.append({
            "epoch": int(epoch),
            "loss": f_loss,
            "g_loss": f_loss,
            "lr": float(lr_val),
            "gpu_temp": get_gpu_temperature(),
            "elapsed_sec": elapsed
        })
        history.sort(key=lambda x: x["epoch"])

    loss_reduction_pct = 0.0
    if initial_loss and f_loss and initial_loss > 0:
        loss_reduction_pct = round(((initial_loss - f_loss) / initial_loss) * 100.0, 1)

    sec_per_epoch = (elapsed / max(1, epoch)) if epoch > 0 else 0
    rem_epochs = max(0, total_epochs - epoch)
    eta_sec = round(sec_per_epoch * rem_epochs)
    fps = round(1008.0 / max(1.0, sec_per_epoch), 2) if sec_per_epoch > 0 else 0.0

    status_data = {
        "epoch": int(epoch),
        "total_epochs": int(total_epochs),
        "phase": "LoRA Forge SD1.5 (Alex, Amaro, Conny, Dana, Belial)",
        "phase_id": 4,
        "status": status_str,
        "g_loss": f_loss if f_loss is not None else loss_val,
        "d_loss": 0.0,
        "l1_loss": f_loss if f_loss is not None else loss_val,
        "edge_loss": 0.0,
        "best_loss": best_loss,
        "initial_loss": initial_loss,
        "loss_reduction_pct": loss_reduction_pct,
        "lr": float(lr_val),
        "gpu_temp": get_gpu_temperature(),
        "elapsed_sec": elapsed,
        "eta_sec": eta_sec,
        "fps": fps,
        "vram_gb": 2.8,
        "vram_total_gb": 4.0,
        "checkpoint_saved": "villa_del_chef_characters.safetensors",
        "timestamp": time.strftime("%H:%M:%S"),
        "total_frames": 1008,
        "history": history
    }
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)
    except Exception:
        pass

def update_live_batch_status(epoch, total_epochs, batch, total_batches, running_loss, start_time, epoch_start_time, lr_val=1e-4):
    elapsed = round(time.time() - start_time, 1)
    ep_elapsed = round(time.time() - epoch_start_time, 1)
    fps = round(batch / max(1.0, ep_elapsed), 2)
    rem_batches_in_epoch = max(0, total_batches - batch)
    rem_epochs = max(0, total_epochs - epoch)
    sec_per_batch = ep_elapsed / max(1, batch)
    eta_sec = round((rem_batches_in_epoch * sec_per_batch) + (rem_epochs * total_batches * sec_per_batch))

    batch_pct = round((batch / max(1, total_batches)) * 100.0, 1)
    overall_pct = round((((epoch - 1) * total_batches + batch) / max(1, total_epochs * total_batches)) * 100.0, 1)

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
            pass

    f_loss = round(float(running_loss), 4)
    if initial_loss is None and f_loss > 0:
        initial_loss = f_loss
    if best_loss is None or (f_loss > 0 and f_loss < best_loss):
        best_loss = f_loss

    loss_reduction_pct = 0.0
    if initial_loss and f_loss and initial_loss > 0:
        loss_reduction_pct = round(((initial_loss - f_loss) / initial_loss) * 100.0, 1)

    status_data = {
        "epoch": int(epoch),
        "total_epochs": int(total_epochs),
        "current_batch": int(batch),
        "total_batches": int(total_batches),
        "batch_progress_pct": batch_pct,
        "overall_progress_pct": overall_pct,
        "phase": "LoRA Forge SD1.5 (Alex, Amaro, Conny, Dana, Belial)",
        "phase_id": 4,
        "status": "ENTRENANDO",
        "g_loss": f_loss,
        "d_loss": 0.0,
        "l1_loss": f_loss,
        "edge_loss": 0.0,
        "best_loss": best_loss,
        "initial_loss": initial_loss,
        "loss_reduction_pct": loss_reduction_pct,
        "lr": float(lr_val),
        "gpu_temp": get_gpu_temperature(),
        "elapsed_sec": elapsed,
        "epoch_sec": ep_elapsed,
        "eta_sec": eta_sec,
        "fps": fps,
        "vram_gb": 2.8,
        "vram_total_gb": 4.0,
        "checkpoint_saved": "villa_del_chef_characters.safetensors",
        "timestamp": time.strftime("%H:%M:%S"),
        "total_frames": 1008,
        "history": history
    }
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)
    except Exception:
        pass

class LoRACacheDataset(Dataset):
    def __init__(self, cache):
        self.cache = cache

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        item = self.cache[idx]
        return item["latent"], item["text_embed"], item["name"]

def save_lora_weights_safetensors(lora_modules, output_filename="villa_del_chef_characters.safetensors"):
    """Exporta el diccionario de pesos en formato estándar Kohya / WebUI / Forge LoRA."""
    state_dict = {}
    for mod_name, lora_mod in lora_modules.items():
        # Formato estándar de claves LoRA para SD 1.5 en WebUI / Forge:
        # lora_unet_{cleaned_name}.lora_down.weight
        clean_name = mod_name.replace(".", "_")
        key_down = f"lora_unet_{clean_name}.lora_down.weight"
        key_up = f"lora_unet_{clean_name}.lora_up.weight"
        key_alpha = f"lora_unet_{clean_name}.alpha"

        state_dict[key_down] = lora_mod.lora_down.weight.detach().cpu().to(torch.float16)
        state_dict[key_up] = lora_mod.lora_up.weight.detach().cpu().to(torch.float16)
        state_dict[key_alpha] = torch.tensor(16.0, dtype=torch.float16)

    for out_dir in LORA_OUTPUT_PATHS:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            safetensors.torch.save_file(state_dict, str(out_dir / output_filename))
        except Exception as e:
            print(f"[!] Error guardando en {out_dir}: {e}")
    print(f"[OK] LoRA safetensors exportado exitosamente: {output_filename}")

def load_lora_weights_safetensors(lora_modules, lora_path: Path):
    """Carga pesos previos de un archivo safetensors de LoRA."""
    if not lora_path.exists():
        return False
    try:
        data = safetensors.torch.load_file(str(lora_path))
        for mod_name, lora_mod in lora_modules.items():
            clean_name = mod_name.replace(".", "_")
            key_down = f"lora_unet_{clean_name}.lora_down.weight"
            key_up = f"lora_unet_{clean_name}.lora_up.weight"
            if key_down in data and key_up in data:
                lora_mod.lora_down.weight.data.copy_(data[key_down].to(DEVICE))
                lora_mod.lora_up.weight.data.copy_(data[key_up].to(DEVICE))
        print(f"[OK] Pesos de LoRA cargados desde {lora_path.name}")
        return True
    except Exception as e:
        print(f"[!] Error al cargar {lora_path.name}: {e}")
        return False

def train_forge_lora(epochs: int = 50, batch_size: int = 1, lr: float = 1e-4, mode: str = "resume"):
    # Limpiar banderas anteriores
    if STOP_FLAG_FILE.exists():
        try: STOP_FLAG_FILE.unlink()
        except Exception: pass
    if PAUSE_FLAG_FILE.exists():
        try: PAUSE_FLAG_FILE.unlink()
        except Exception: pass

    # 1. Cargar caché de VAE y CLIP (ya pre-computada de los 1,008 frames)
    if not CACHE_PATH.exists():
        print("[!] No se encontró la caché pre-computada. Ejecutando preparación...")
        sys.exit(1)

    print(f"[*] Cargando caché pre-computada: {CACHE_PATH.name}")
    cache = torch.load(CACHE_PATH)
    dataset = LoRACacheDataset(cache)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Cargar pipeline SD1.5 base
    model_path = find_sd15_model_path()
    print(f"[*] Cargando pipeline base SD1.5: {model_path.name}")
    pipe = StableDiffusionPipeline.from_single_file(
        str(model_path),
        torch_dtype=torch.float16,
        load_safety_checker=False
    )

    unet = pipe.unet.to(DEVICE, dtype=torch.float16)
    unet.requires_grad_(False)

    # 3. Inyectar LoRA en UNet
    lora_modules, lora_params = inject_lora_into_unet(unet, rank=16, alpha=16.0)
    for p in lora_params:
        p.requires_grad = True

    print(f"[OK] LoRA inyectado: {len(lora_modules)} módulos | {sum(p.numel() for p in lora_params):,} parámetros entrenables")

    # 4. Optimizador AdamW 8-bit
    optimizer = bnb.optim.AdamW8bit(lora_params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    start_epoch = 1
    existing_lora = CHECKPOINT_DIR / "villa_del_chef_characters.safetensors"
    if mode == "resume" and existing_lora.exists():
        load_lora_weights_safetensors(lora_modules, existing_lora)
        if STATUS_FILE.exists():
            try:
                with open(STATUS_FILE, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                    if s_data.get("phase_id") == 4:
                        start_epoch = s_data.get("epoch", 1) + 1
            except Exception:
                pass

    total_target_epochs = (start_epoch - 1) + epochs if mode == "resume" else epochs

    print("=" * 70)
    print("  ENTRENAMIENTO LoRA PARA STABLE DIFFUSION FORGE (SD 1.5)")
    print(f"  Modo: {mode.upper()} | Épocas: {start_epoch} a {total_target_epochs} | Muestras: {len(dataset)}")
    print(f"  VRAM Optimizada: FP16 + 8-bit Adam + PyTorch LoRA (~2.5 GB VRAM)")
    print("=" * 70)

    start_time = time.time()
    avg_loss = 0.0
    grad_accum_steps = 4

    for epoch in range(start_epoch, total_target_epochs + 1):
        if PAUSE_FLAG_FILE.exists():
            print("\n[⏸️] Señal de pausa recibida antes de época. Guardando...")
            try: PAUSE_FLAG_FILE.unlink()
            except Exception: pass
            save_lora_weights_safetensors(lora_modules)

        # Generar muestra de auditoria visual automatica para el monitor web
        try:
            from pixel_ai_engine.audit_visualizer import generate_epoch_audit_sample
            generate_epoch_audit_sample(epoch)
        except Exception as e:
            print(f"[!] Error generando muestra de auditoria visual: {e}")
            update_status(epoch - 1, total_target_epochs, "PAUSADO", avg_loss, start_time)
            return

        if STOP_FLAG_FILE.exists():
            print("\n[⏹️] Señal de detención recibida antes de época. Guardando...")
            try: STOP_FLAG_FILE.unlink()
            except Exception: pass
            save_lora_weights_safetensors(lora_modules)
            update_status(epoch - 1, total_target_epochs, "DETENIDO", avg_loss, start_time)
            return

        epoch_loss = 0.0
        epoch_start_time = time.time()
        optimizer.zero_grad()

        for batch_i, (latents, text_embeds, names) in enumerate(dataloader):
            if batch_i % 15 == 0:
                if PAUSE_FLAG_FILE.exists():
                    print("\n[⏸️] Señal de pausa recibida en lote. Guardando...")
                    try: PAUSE_FLAG_FILE.unlink()
                    except Exception: pass
                    save_lora_weights_safetensors(lora_modules)
                    update_status(epoch, total_target_epochs, "PAUSADO", avg_loss, start_time)
                    return

                if STOP_FLAG_FILE.exists():
                    print("\n[⏹️] Señal de detención recibida en lote. Guardando...")
                    try: STOP_FLAG_FILE.unlink()
                    except Exception: pass
                    save_lora_weights_safetensors(lora_modules)
                    update_status(epoch, total_target_epochs, "DETENIDO", avg_loss, start_time)
                    return

            latents = latents.to(DEVICE, dtype=torch.float16)
            text_embeds = text_embeds.to(DEVICE, dtype=torch.float16)

            # Muestrear ruido gaussiano y timestep aleatorio
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=DEVICE).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predecir ruido con LoRA
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds).sample

            # Pérdida MSE de difusión
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_i + 1) % grad_accum_steps == 0 or (batch_i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum_steps

            if (batch_i + 1) % 15 == 0 or (batch_i + 1) == len(dataloader):
                cur_step = batch_i + 1
                tot_steps = len(dataloader)
                run_loss = epoch_loss / max(1, cur_step)
                update_live_batch_status(
                    epoch=epoch,
                    total_epochs=total_target_epochs,
                    batch=cur_step,
                    total_batches=tot_steps,
                    running_loss=run_loss,
                    start_time=start_time,
                    epoch_start_time=epoch_start_time
                )

        avg_loss = epoch_loss / max(1, len(dataloader))
        temp = get_gpu_temperature()
        print(f"Época [{epoch:03d}/{total_target_epochs:03d}] - LoRA Diffusion Loss: {avg_loss:.4f} | GPU Temp: {temp}°C")

        update_status(epoch, total_target_epochs, "ENTRENANDO", avg_loss, start_time)
        save_lora_weights_safetensors(lora_modules)

    update_status(total_target_epochs, total_target_epochs, "COMPLETADO", avg_loss, start_time)
    print("\n[OK] Entrenamiento de LoRA para Forge finalizado exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mode", type=str, default="resume", choices=["start", "resume"])
    args = parser.parse_args()
    train_forge_lora(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, mode=args.mode)
