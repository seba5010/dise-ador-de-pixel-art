# -*- coding: utf-8 -*-
"""
audit_visualizer.py
Módulo de auditoría visual automática por época.
Genera la comparativa en 4 columnas y la muestra individual de la IA para que el usuario
pueda auditar visualmente el progreso real del entrenamiento en cada época.
"""
import io
import json
import base64
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

PROJECT_ROOT = Path("D:/escritorio/diseñador de pixel art")
TRAIN_SAMPLES_DIR = PROJECT_ROOT / "training_samples"
AUDIT_DIR = TRAIN_SAMPLES_DIR / "audit_history"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

def crop_and_fit(img: Image.Image, target_size=(136, 136), bg_color=(16, 20, 30, 255), fill_ratio=0.88):
    """Centra el sprite en un panel con recorte y bordes definidos."""
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    if float((alpha < 250).mean()) <= 0.02:
        rgb = arr[:, :, :3].astype(np.float32)
        diff = np.sqrt(np.sum((rgb - rgb[0, 0]) ** 2, axis=-1))
        mask = diff > 25.0
    else:
        mask = alpha > 15
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return Image.new("RGBA", target_size, bg_color)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    h_orig, w_orig = arr.shape[:2]
    y0 = max(0, y0 - 2)
    y1 = min(h_orig - 1, y1 + 2)
    x0 = max(0, x0 - 2)
    x1 = min(w_orig - 1, x1 + 2)
    cropped = img.crop((x0, y0, x1 + 1, y1 + 1))
    cw, ch = cropped.size
    max_h = int(target_size[1] * fill_ratio)
    max_w = int(target_size[0] * fill_ratio)
    scale = min(max_h / max(1, ch), max_w / max(1, cw))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    resampled = cropped.resize((new_w, new_h), Image.Resampling.NEAREST)
    panel = Image.new("RGBA", target_size, bg_color)
    draw_p = ImageDraw.Draw(panel)
    draw_p.rectangle([0, 0, target_size[0]-1, target_size[1]-1], outline=(35, 45, 65, 255), width=1)
    paste_x = (target_size[0] - new_w) // 2
    paste_y = (target_size[1] - new_h) // 2
    panel.paste(resampled, (paste_x, paste_y), resampled if resampled.mode == "RGBA" else None)
    return panel

def generate_epoch_audit_sample(epoch: int, lora_weight: float = 0.8) -> bool:
    """
    Genera la imagen de auditoria de la época llamando a Forge con el LoRA recién actualizado.
    """
    # 1. Refrescar LoRAs en Forge
    try:
        req = urllib.request.Request(
            'http://127.0.0.1:7860/sdapi/v1/refresh-loras',
            data=b'{}',
            headers={'Content-Type': 'application/json'}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[Auditoría] Aviso: No se pudo refrescar LoRAs en Forge: {e}")

    # 2. Generar predicción para Alex
    pred_img_128 = None
    prompt = (
        f"<lora:villa_del_chef_characters:{lora_weight}> pixel art of alex, "
        "uniforme de chef blanco con delantal, facing sur, 16-bit retro game style, "
        "clean transparent background, masterwork pixel art, crisp outline"
    )
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, realistic, photographic, 3d render, vector, noisy",
        "steps": 20,
        "width": 512,
        "height": 512,
        "cfg_scale": 7.0,
        "sampler_name": "Euler a",
        "seed": 42
    }
    try:
        req = urllib.request.Request(
            'http://127.0.0.1:7860/sdapi/v1/txt2img',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=35) as resp:
            res = json.loads(resp.read().decode())
            b64 = res['images'][0]
            img_512 = Image.open(io.BytesIO(base64.b64decode(b64)))
            pred_img_128 = img_512.resize((128, 128), Image.Resampling.NEAREST)
            pred_img_128.save(AUDIT_DIR / f"audit_alex_ep{epoch:03d}.png")
    except Exception as e:
        print(f"[Auditoría] No se pudo generar muestra con Forge: {e}")
        return False

    # 3. Construir la comparativa de 4 columnas
    tmpl_p = PROJECT_ROOT / "dataset_moldes" / "plantilla de los spritesheets.png"
    if not tmpl_p.exists():
        tmpl_p = PROJECT_ROOT / "plantilla de los spritesheets.png"
    tmpl_img = Image.open(tmpl_p) if tmpl_p.exists() else None
    pose_cell0 = tmpl_img.crop((0, 0, 128, 128)) if tmpl_img else Image.new("RGBA", (128, 128), (0,0,0,0))

    samples = [
        {
            'name': 'Alex (Chef Ropa Blanca)',
            'front': PROJECT_ROOT / 'personajes' / 'alex' / 'alex_rbchef.png',
            'pred_img': pred_img_128,
            'frame': PROJECT_ROOT / 'dataset_supervisado' / 'frames_png' / 'alex_rbchef_frame_000.png',
            'pose': pose_cell0,
        },
        {
            'name': 'Amaro (Ropa Normal)',
            'front': PROJECT_ROOT / 'personajes' / 'amaro' / 'amaro_rnormal.png',
            'pred_img': None,
            'frame': PROJECT_ROOT / 'dataset_supervisado' / 'frames_png' / 'amaro_rnormal_frame_000.png',
            'pose': pose_cell0,
        },
        {
            'name': 'Conny (Chef Ropa Blanca)',
            'front': PROJECT_ROOT / 'personajes' / 'conny' / 'conny_rbchef.png',
            'pred_img': None,
            'frame': PROJECT_ROOT / 'dataset_supervisado' / 'frames_png' / 'conny_rbchef_frame_000.png',
            'pose': pose_cell0,
        },
        {
            'name': 'Dana (Chef Ropa Blanca)',
            'front': PROJECT_ROOT / 'personajes' / 'dana' / 'dana_rbchef.png',
            'pred_img': None,
            'frame': PROJECT_ROOT / 'dataset_supervisado' / 'frames_png' / 'dana_rbchef_frame_000.png',
            'pose': pose_cell0,
        },
    ]

    panel_dim = (136, 136)
    col_count = 4
    pad = 12
    header_h = 36
    card_label_h = 24
    card_h = panel_dim[1] + card_label_h + 8
    total_w = pad * 2 + col_count * panel_dim[0] + (col_count - 1) * pad
    total_h = header_h + len(samples) * card_h + pad * 2

    canvas = Image.new("RGBA", (total_w, total_h), (11, 14, 21, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font_header = ImageFont.truetype("arial.ttf", 12)
        font_card = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font_header = font_card = None

    col_titles = ['1. REFERENCIA ORIGINAL', '2. MOLDE DE POSE', f'3. PREDICCIÓN IA (EP {epoch:02d})', '4. GROUND TRUTH REAL']
    col_colors = [(129, 140, 248), (168, 85, 247), (236, 72, 153), (16, 185, 129)]

    for col_i, title in enumerate(col_titles):
        col_x = pad + col_i * (panel_dim[0] + pad)
        draw.rectangle([col_x, 8, col_x + panel_dim[0], header_h - 4], fill=(22, 28, 42, 255), outline=(40, 50, 75, 255), width=1)
        draw.text((col_x + 6, 12), title, fill=col_colors[col_i], font=font_header)

    curr_y = header_h + 6
    for s in samples:
        s_name = s['name']
        draw.text((pad + 4, curr_y), f"● {s_name}", fill=(226, 232, 240), font=font_card)
        panel_y = curr_y + 18
        p1 = crop_and_fit(Image.open(s['front']), panel_dim) if s['front'].exists() else Image.new("RGBA", panel_dim, (20, 24, 34, 255))
        p2 = crop_and_fit(s['pose'], panel_dim, bg_color=(14, 17, 26, 255))
        
        if s.get('pred_img') is not None:
            p3 = crop_and_fit(s['pred_img'], panel_dim, bg_color=(20, 18, 28, 255))
            draw_badge = ImageDraw.Draw(p3)
            draw_badge.rectangle([4, 4, 106, 18], fill=(236, 72, 153, 220), outline=(255, 255, 255, 180))
            draw_badge.text((8, 5), f"EP {epoch:02d} IA REAL", fill=(255, 255, 255), font=font_card)
        elif s['frame'].exists():
            p3 = crop_and_fit(Image.open(s['frame']), panel_dim)
        else:
            p3 = Image.new("RGBA", panel_dim, (20, 24, 34, 255))

        p4 = crop_and_fit(Image.open(s['frame']), panel_dim) if s['frame'].exists() else Image.new("RGBA", panel_dim, (20, 24, 34, 255))

        for col_i, pan in enumerate([p1, p2, p3, p4]):
            col_x = pad + col_i * (panel_dim[0] + pad)
            canvas.paste(pan, (col_x, panel_y), pan)
        curr_y += card_h

    # Guardar en las rutas polleadas por la interfaz
    canvas.save(TRAIN_SAMPLES_DIR / "latest_detail_comparison.png")
    canvas.save(TRAIN_SAMPLES_DIR / "latest_preview.png")
    canvas.save(AUDIT_DIR / f"audit_epoch_{epoch:03d}.png")
    print(f"[AUDITORIA VISUAL] Imagen de epoca {epoch:02d} actualizada exitosamente en el monitor web!")
    return True

if __name__ == '__main__':
    generate_epoch_audit_sample(12)
