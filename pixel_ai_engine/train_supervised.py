import os
import sys
import time
import json
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

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_SAMPLES_DIR = PROJECT_ROOT / "training_samples"
TRAIN_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
STATUS_FILE = PROJECT_ROOT / "training_status.json"
STOP_FLAG_FILE = PROJECT_ROOT / "stop_training.flag"
PAUSE_FLAG_FILE = PROJECT_ROOT / "pause_training.flag"

CACHE_PATH = PROJECT_ROOT / "dataset_supervisado" / "supervised_cache_8x12.pt"

class SupervisedTensorDataset(Dataset):
    def __init__(self, cache_path):
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache no encontrada: {cache_path}. Ejecuta prepare_supervised_dataset.py primero.")
        self.samples = torch.load(cache_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        front = s["front_tensor"] # (3, H, W)
        f_idx = s["frame_idx"]
        target = s["target_tensor"] # (4, H, W)
        return front, f_idx, target, s["char_id"]

class SobelLoss(nn.Module):
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
            p_edge = torch.sqrt(p_gx**2 + p_gy**2 + 1e-6)
            t_edge = torch.sqrt(t_gx**2 + t_gy**2 + 1e-6)
            loss += nn.functional.smooth_l1_loss(p_edge, t_edge)
        return loss / 3.0

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

def update_status(epoch, total_epochs, status_str, g_loss, d_loss, l1_val, edge_val, start_time):
    elapsed = round(time.time() - start_time, 1)
    status_data = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "phase": "Supervisado (Alex, Amaro, Conny, Dana, Belial)",
        "phase_id": 3,
        "status": status_str,
        "g_loss": round(float(g_loss), 4),
        "d_loss": round(float(d_loss), 4),
        "l1_loss": round(float(l1_val), 4),
        "edge_loss": round(float(edge_val), 4),
        "gpu_temp": get_gpu_temperature(),
        "elapsed_sec": elapsed,
        "timestamp": time.strftime("%H:%M:%S"),
        "total_frames": 1008
    }
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)

def generate_preview(generator, dataset):
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
            front, f_idx, target, char_id = dataset[s_idx]
            pose = tm.get_frame_tensor(f_idx)

            cond = torch.cat([front, pose], dim=0).unsqueeze(0).to(DEVICE)
            with autocast(enabled=USE_AMP):
                out = generator(cond).squeeze(0)

            # 1. Frontal Chibi
            f_np = ((front.permute(1,2,0).cpu().numpy() + 1.0) * 127.5).clip(0,255).astype(np.uint8)
            f_pil = Image.fromarray(f_np, "RGB").resize((128, 128), Image.Resampling.NEAREST)

            # 2. Pose
            p_np = ((pose.permute(1,2,0).cpu().numpy() + 1.0) * 127.5).clip(0,255).astype(np.uint8)
            p_pil = Image.fromarray(p_np, "RGB").resize((128, 128), Image.Resampling.NEAREST)

            # 3. Prediccion IA
            pred_np = ((out[:3].permute(1,2,0).cpu().numpy() + 1.0) * 127.5).clip(0,255).astype(np.uint8)
            alpha_np = ((out[3].cpu().numpy() + 1.0) * 127.5).clip(0,255).astype(np.uint8)
            pred_rgba = np.dstack([pred_np, alpha_np])
            pred_pil = Image.fromarray(pred_rgba, "RGBA").resize((128, 128), Image.Resampling.NEAREST)

            # 4. Ground Truth Real
            tgt_np = ((target[:3].permute(1,2,0).cpu().numpy() + 1.0) * 127.5).clip(0,255).astype(np.uint8)
            t_alpha_np = ((target[3].cpu().numpy() + 1.0) * 127.5).clip(0,255).astype(np.uint8)
            tgt_rgba = np.dstack([tgt_np, t_alpha_np])
            tgt_pil = Image.fromarray(tgt_rgba, "RGBA").resize((128, 128), Image.Resampling.NEAREST)

            strip = Image.new("RGBA", (128 * 4 + 30, 128 + 20), (16, 18, 26, 255))
            strip.paste(f_pil, (5, 10))
            strip.paste(p_pil, (128 + 10, 10))
            strip.paste(pred_pil, (256 + 15, 10))
            strip.paste(tgt_pil, (384 + 20, 10))
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

def train_supervised_model(epochs: int = 150, batch_size: int = 4, lr: float = 1.5e-4, mode: str = "resume"):
    # Limpiar banderas anteriores
    if STOP_FLAG_FILE.exists(): STOP_FLAG_FILE.unlink()
    if PAUSE_FLAG_FILE.exists(): PAUSE_FLAG_FILE.unlink()

    dataset = SupervisedTensorDataset(CACHE_PATH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    from pixel_ai_engine.dataset import TemplateManager
    tmpl_path = PROJECT_ROOT / "dataset_moldes" / "plantilla de los spritesheets.png"
    if not tmpl_path.exists():
        tmpl_path = PROJECT_ROOT / "plantilla de los spritesheets.png"
    tmpl_mgr = TemplateManager(tmpl_path, MODEL_RESOLUTION, 12, 8, 1024, 1536)

    generator = PixelArtUNetGenerator().to(DEVICE)
    discriminator = PixelArtPatchDiscriminator().to(DEVICE)

    start_epoch = 1
    best_loss = 999.0
    ckpt_candidate = CHECKPOINT_DIR / "latest_checkpoint.pt"

    if mode == "resume" and ckpt_candidate.exists():
        try:
            ckpt = torch.load(ckpt_candidate, map_location=DEVICE)
            if isinstance(ckpt, dict) and "generator" in ckpt:
                generator.load_state_dict(ckpt["generator"])
                if "discriminator" in ckpt:
                    discriminator.load_state_dict(ckpt["discriminator"])
                start_epoch = ckpt.get("epoch", 0) + 1
                best_loss = ckpt.get("best_loss", 999.0)
                print(f"[OK] Reanudando entrenamiento desde época {start_epoch}")
        except Exception as e:
            print(f"[!] Error al reanudar checkpoint: {e}")

    total_target_epochs = (start_epoch - 1) + epochs if mode == "resume" else epochs

    print("=" * 70)
    print("  ENTRENAMIENTO SUPERVISADO DE CORRECCIÓN QUIRÚRGICA")
    print(f"  Modo: {mode.upper()} | Épocas: {start_epoch} a {total_target_epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  Muestras: {len(dataset)} pares Ground-Truth | Dispositivo: {DEVICE}")
    print("=" * 70)

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))
    scaler_g = GradScaler(enabled=USE_AMP)
    scaler_d = GradScaler(enabled=USE_AMP)

    criterion_l1 = nn.SmoothL1Loss()
    criterion_edge = SobelLoss().to(DEVICE)
    criterion_bce = nn.BCEWithLogitsLoss()

    start_time = time.time()
    avg_g = 0.0
    avg_d = 0.0
    avg_l1 = 0.0
    avg_edge = 0.0

    for epoch in range(start_epoch, total_target_epochs + 1):
        if PAUSE_FLAG_FILE.exists():
            print("\n[⏸️] Señal de pausa recibida. Guardando checkpoint y pausando...")
            PAUSE_FLAG_FILE.unlink()
            update_status(epoch - 1, total_target_epochs, "PAUSADO", avg_g, avg_d, avg_l1, avg_edge, start_time)
            return

        if STOP_FLAG_FILE.exists():
            print("\n[⏹️] Señal de detención recibida. Guardando y deteniendo...")
            STOP_FLAG_FILE.unlink()
            update_status(epoch - 1, total_target_epochs, "DETENIDO", avg_g, avg_d, avg_l1, avg_edge, start_time)
            return

        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_l1 = 0.0
        epoch_edge = 0.0

        for batch_i, (fronts, f_indices, targets, _) in enumerate(dataloader):
            if batch_i % 8 == 0:
                if PAUSE_FLAG_FILE.exists():
                    print('\n[⏸️] Señal de pausa recibida en lote. Guardando checkpoint...')
                    try: PAUSE_FLAG_FILE.unlink()
                    except Exception: pass
                    torch.save({'epoch': epoch, 'generator': generator.state_dict(), 'discriminator': discriminator.state_dict(), 'loss': avg_g, 'best_loss': best_loss}, CHECKPOINT_DIR / 'latest_checkpoint.pt')
                    update_status(epoch, total_target_epochs, 'PAUSADO', avg_g, avg_d, avg_l1, avg_edge, start_time)
                    return
                if STOP_FLAG_FILE.exists():
                    print('\n[⏹️] Señal de detención recibida en lote. Guardando checkpoint...')
                    try: STOP_FLAG_FILE.unlink()
                    except Exception: pass
                    torch.save({'epoch': epoch, 'generator': generator.state_dict(), 'discriminator': discriminator.state_dict(), 'loss': avg_g, 'best_loss': best_loss}, CHECKPOINT_DIR / 'latest_checkpoint.pt')
                    update_status(epoch, total_target_epochs, 'DETENIDO', avg_g, avg_d, avg_l1, avg_edge, start_time)
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

                d_fake = discriminator(preds, cond)
                adv_loss = criterion_bce(d_fake.float().clamp(-30.0, 30.0), torch.ones_like(d_fake).float()) * 0.05
                total_g = l1_color + l1_alpha + edge_loss + adv_loss

            scaler_g.scale(total_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            # Discriminador
            opt_d.zero_grad()
            with autocast(enabled=USE_AMP):
                d_real = discriminator(targets, cond)
                d_fake_det = discriminator(preds.detach(), cond)
                loss_d_real = criterion_bce(d_real.float().clamp(-30.0, 30.0), torch.ones_like(d_real).float())
                loss_d_fake = criterion_bce(d_fake_det.float().clamp(-30.0, 30.0), torch.zeros_like(d_fake_det).float())
                total_d = (loss_d_real + loss_d_fake) * 0.5

            scaler_d.scale(total_d).backward()
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

        print(f"Época [{epoch:03d}/{total_target_epochs:03d}] - G_Loss: {avg_g:.4f} | Color_L1: {avg_l1:.4f} | Borde: {avg_edge:.4f} | D_Loss: {avg_d:.4f}")

        # Guardar preview y actualizar status cada época o cada 2
        update_status(epoch, total_target_epochs, "ENTRENANDO", avg_g, avg_d, avg_l1, avg_edge, start_time)
        if epoch % 2 == 0 or epoch == start_epoch or epoch == total_target_epochs:
            generate_preview(generator, dataset)

        # Checkpoints
        if avg_g < best_loss:
            best_loss = avg_g
            torch.save({
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "best_loss": best_loss
            }, CHECKPOINT_DIR / "best_generator.pt")

        torch.save({
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "loss": avg_g
        }, CHECKPOINT_DIR / "latest_checkpoint.pt")

    update_status(total_target_epochs, total_target_epochs, "COMPLETADO", avg_g, avg_d, avg_l1, avg_edge, start_time)
    print("\n[OK] Ciclo de entrenamiento supervisado finalizado exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--mode", type=str, default="resume", choices=["start", "resume"])
    args = parser.parse_args()
    train_supervised_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, mode=args.mode)
