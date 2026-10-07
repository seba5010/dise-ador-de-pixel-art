"""
Pixel AI Engine - Dataset Loader & Preprocessing (Chibi Anatomical Adaptation)
Gestiona la lectura de personajes de muestra, corte de sub-sprites,
adaptacion anatomica de ilustraciones puras a proporciones Chibi (2 cabezas),
acondicionamiento con la Plantilla Universal 8x12 (96 frames) y normalizacion estricta.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from .config import (
    PROJECT_ROOT,
    TEMPLATE_PATH,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    GRID_ROWS,
    GRID_COLS,
    TOTAL_FRAMES,
    CELL_WIDTH,
    CELL_HEIGHT,
    MODEL_RESOLUTION,
    BATCH_SIZE,
    NUM_WORKERS,
    ROW_ANIMATION_NAMES,
    SHEET_CELL_TOLERANCE,
    get_phase_config
)


CANONICAL_SPRITE_HEIGHT = 112
CANONICAL_SPRITE_MAX_WIDTH = 116
CELL_SPRITE_HEIGHT_RATIO = CANONICAL_SPRITE_HEIGHT / 128.0
CANONICAL_FRONT_REFERENCE_HEIGHT = 208
CANONICAL_FRONT_REFERENCE_GROUND_Y = 232
TRAINING_FRAME_HEIGHT = 208
TRAINING_FRAME_MAX_WIDTH = 232


def get_cell_coordinates(canvas_w: int, canvas_h: int, row: int, col: int, total_rows: int = GRID_ROWS, total_cols: int = GRID_COLS):
    """Calcula las coordenadas exactas de corte con redondeo estricto."""
    x0 = int(round(col * canvas_w / total_cols))
    x1 = int(round((col + 1) * canvas_w / total_cols))
    y0 = int(round(row * canvas_h / total_rows))
    y1 = int(round((row + 1) * canvas_h / total_rows))
    return x0, y0, x1, y1


def foreground_mask(img: Image.Image, alpha_threshold: int = 15, diff_threshold: float = 28.0) -> np.ndarray:
    """Genera una mascara booleana del contenido solido de la figura."""
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]

    if float((alpha < 250).mean()) > 0.02:
        return alpha > alpha_threshold

    rgb = arr[:, :, :3].astype(np.float32)
    h, w = alpha.shape
    sample = max(2, min(h, w) // 12)
    corners = np.concatenate([
        rgb[:sample, :sample].reshape(-1, 3),
        rgb[:sample, w - sample:].reshape(-1, 3),
        rgb[h - sample:, :sample].reshape(-1, 3),
        rgb[h - sample:, w - sample:].reshape(-1, 3),
    ], axis=0)
    bg = np.median(corners, axis=0)
    diff = np.sqrt(np.sum((rgb - bg) ** 2, axis=-1))
    return diff > diff_threshold


def center_and_pad(img: Image.Image, target_size: int = MODEL_RESOLUTION) -> Tuple[Image.Image, Tuple[int, int, float, int, int]]:
    """Alinea y centra el contenido no vacio horizontalmente anclado abajo."""
    arr = np.array(img.convert("RGBA"))
    mask = foreground_mask(img)

    if float((arr[:, :, 3] < 250).mean()) <= 0.02:
        arr[~mask, 3] = 0
        img = Image.fromarray(arr, mode="RGBA")

    coords = np.argwhere(mask)
    if len(coords) == 0:
        return img.resize((target_size, target_size), Image.Resampling.NEAREST), (img.size[0], img.size[1], 1.0, 0, 0)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    content = img.crop((x0, y0, x1 + 1, y1 + 1))
    cw, ch = content.size

    safe_dim = int(target_size * 0.90)
    scale = min(safe_dim / max(1, cw), safe_dim / max(1, ch), 1.0)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))

    if scale < 0.999:
        resample = Image.Resampling.LANCZOS if (cw > 400 or ch > 400) else Image.Resampling.NEAREST
        content_scaled = content.resize((new_w, new_h), resample)
    else:
        content_scaled = content

    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    ox = (target_size - new_w) // 2
    oy = max(4, target_size - new_h - 12)
    content_scaled = content_scaled.convert("RGBA")
    canvas.paste(content_scaled, (ox, oy), content_scaled)

    return canvas, (cw, ch, scale, ox, oy)


def pad_to_square(img: Image.Image, target_size: int = MODEL_RESOLUTION) -> Tuple[Image.Image, Tuple]:
    """Alias compatible con codigo existente."""
    return center_and_pad(img, target_size)


def adapt_front_to_chibi(
    img: Image.Image,
    target_size: int = MODEL_RESOLUTION,
    target_h: int = CANONICAL_FRONT_REFERENCE_HEIGHT,
    ground_y: int = CANONICAL_FRONT_REFERENCE_GROUND_Y,
) -> Image.Image:
    """
    ADAPTADOR ANATOMICO CHIBI:
    Toma una ilustración pura realista (8 cabezas, 1024x1536) y la acondiciona
    en proporciones Chibi canónicas y la centra como referencia frontal de alta definición.
    Utiliza remapeo continuo y suave de coordenadas anatómicas sin cortes de guillotina,
    evitando extremidades seccionadas, hombros flotantes o islas de píxeles desconectadas.
    Garantiza que el cielo superior (y < 160) sea 100% transparente para erradicar
    de raíz cualquier posible nube de hollín o cabello fantasma.
    """
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    if float((alpha < 250).mean()) <= 0.02:
        rgb = arr[:, :, :3].astype(np.float32)
        corners = np.concatenate([rgb[:5, :5], rgb[:5, -5:], rgb[-5:, :5], rgb[-5:, -5:]], axis=0).reshape(-1, 3)
        bg = np.median(corners, axis=0)
        diff = np.sqrt(np.sum((rgb - bg) ** 2, axis=-1))
        mask = diff > 25.0
        arr[~mask, 3] = 0
        img = Image.fromarray(arr, mode="RGBA")
    else:
        mask = alpha > 15

    coords = np.argwhere(mask)
    if len(coords) == 0:
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cropped = img.crop((x0, y0, x1 + 1, y1 + 1))
    cw, ch = cropped.size
    crop_arr = np.array(cropped)

    # Punto de referencia anatómico de transición cabeza-cuerpo (cuello/clavícula ~21% de la altura realista)
    neck_orig_y = float(ch) * 0.21
    target_head_h = float(target_h) * 0.30
    target_body_h = float(target_h) - target_head_h

    aspect = cw / max(1.0, float(ch))
    min_target_w = max(30, int(round(target_h * 0.40)))
    max_target_w = max(min_target_w, int(round(target_h * 0.62)))
    target_w = max(min_target_w, min(max_target_w, int(round(target_h * aspect * 1.12))))

    map_x = np.zeros((target_h, target_w), dtype=np.float32)
    map_y = np.zeros((target_h, target_w), dtype=np.float32)

    cx_out = (target_w - 1) / 2.0
    cx_in = (cw - 1) / 2.0

    for y_out in range(target_h):
        if y_out <= target_head_h:
            t = y_out / max(1.0, target_head_h)
            y_orig = t * neck_orig_y
        else:
            t = (y_out - target_head_h) / max(1.0, target_body_h)
            y_orig = neck_orig_y + t * (ch - 1 - neck_orig_y)

        # Transición sigmoidal suave en cuello para dar amplitud chibi a mejillas/cabeza sin quebrar el cuerpo
        t_neck = (y_out - target_head_h) / 3.0
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(t_neck, -5.0, 5.0)))
        scale_x = (cw / float(target_w)) * (0.88 * (1.0 - sigmoid) + 1.0 * sigmoid)

        for x_out in range(target_w):
            dx = x_out - cx_out
            map_x[y_out, x_out] = cx_in + dx * scale_x
            map_y[y_out, x_out] = y_orig

    warped = cv2.remap(crop_arr, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    warped_pil = Image.fromarray(warped)

    # Recortar márgenes alfa transparentes exteriores residuales
    w_arr = np.array(warped_pil)
    w_mask = w_arr[:, :, 3] > 15
    w_coords = np.argwhere(w_mask)
    if len(w_coords) > 0:
        wy0, wx0 = w_coords.min(axis=0)
        wy1, wx1 = w_coords.max(axis=0)
        trimmed = warped_pil.crop((wx0, wy0, wx1 + 1, wy1 + 1))
    else:
        trimmed = warped_pil

    tw, th = trimmed.size
    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    ox = (target_size - tw) // 2
    oy = ground_y - th
    canvas.alpha_composite(trimmed, (ox, oy))

    return canvas


def pad_target_frame_canonical(
    frame_img: Image.Image,
    target_size: int = MODEL_RESOLUTION,
    target_h: int = TRAINING_FRAME_HEIGHT,
    target_max_w: int = TRAINING_FRAME_MAX_WIDTH,
    ground_y: Optional[int] = None,
) -> Image.Image:
    """
    Amplía y centra un frame objetivo dentro del lienzo de entrenamiento 256x256.
    La escala de exportación 128x128 se aplica después mediante place_in_cell.
    """
    rgba_frame = frame_img.convert("RGBA")
    arr = np.array(rgba_frame)
    mask = foreground_mask(rgba_frame, alpha_threshold=20)
    coords = np.argwhere(mask)
    if len(coords) < 40:
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    if (x1 - x0 + 1) < 4 or (y1 - y0 + 1) < 8:
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    clean_arr = arr.copy()
    clean_arr[~mask, 3] = 0
    clean_frame = Image.fromarray(clean_arr, mode="RGBA")
    fg = clean_frame.crop((x0, y0, x1 + 1, y1 + 1))
    fw, fh = fg.size

    safe_target_h = min(target_h, target_size - 14)
    safe_target_w = min(target_max_w, target_size - 16)
    scale = min(safe_target_h / max(1, fh), safe_target_w / max(1, fw))
    new_w = max(1, int(round(fw * scale)))
    new_h = max(1, int(round(fh * scale)))
    fg_scaled = fg.resize((new_w, new_h), Image.Resampling.NEAREST)

    scaled_alpha = np.array(fg_scaled)[:, :, 3]
    visible_coordinates = np.argwhere(scaled_alpha > 20)
    if len(visible_coordinates) > 0:
        visible_y0, visible_x0 = visible_coordinates.min(axis=0)
        visible_y1, visible_x1 = visible_coordinates.max(axis=0)
        fg_scaled = fg_scaled.crop((visible_x0, visible_y0, visible_x1 + 1, visible_y1 + 1))
        new_w, new_h = fg_scaled.size

    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    ox = (target_size - new_w) // 2
    if ground_y is None:
        oy = (target_size - new_h) // 2
    else:
        oy = min(target_size - new_h, max(0, ground_y - new_h))
    canvas.alpha_composite(fg_scaled, (ox, oy))
    return canvas


def get_8x12_index_from_16x4(r16: int, c16: int) -> int:
    """Mapea las filas de caminata/giro de hojas 16x4 al índice canónico 8x12 (0..95)."""
    mapping = {
        0: (0, 0),   # Frente Idle -> 8x12 Fila 0 (cols 0..3)
        1: (0, 4),   # Frente Pasos -> 8x12 Fila 0 (cols 4..7)
        2: (4, 0),   # Espalda Idle -> 8x12 Fila 4 (cols 0..3)
        3: (4, 4),   # Espalda Pasos -> 8x12 Fila 4 (cols 4..7)
        4: (2, 0),   # Lateral Der Idle -> 8x12 Fila 2 (cols 0..3)
        5: (2, 4),   # Lateral Der Pasos -> 8x12 Fila 2 (cols 4..7)
        6: (6, 0),   # Lateral Izq Idle -> 8x12 Fila 6 (cols 0..3)
        7: (6, 4),   # Lateral Izq Pasos -> 8x12 Fila 6 (cols 4..7)
        8: (1, 0),   # Diag Frontal Der -> 8x12 Fila 1 (cols 0..3)
        9: (7, 0),   # Diag Frontal Izq -> 8x12 Fila 7 (cols 0..3)
        10: (3, 0),  # Diag Trasera Der -> 8x12 Fila 3 (cols 0..3)
        11: (5, 0),  # Diag Trasera Izq -> 8x12 Fila 5 (cols 0..3)
    }
    if r16 in mapping:
        r8, c_offset = mapping[r16]
        return r8 * 8 + (c_offset + c16)
    return -1


def place_in_cell(padded_img: Image.Image, cell_w: int = CELL_WIDTH, cell_h: int = CELL_HEIGHT, pad_meta: Optional[Tuple] = None) -> Image.Image:
    """
    Toma el frame generado de 256x256, extrae la figura y la ubica perfectamente centrada
    con los pies apoyados en el suelo dentro de la celda canónica (128x128).
    """
    rgba_image = padded_img.convert("RGBA")
    arr = np.array(rgba_image)
    mask = foreground_mask(rgba_image, alpha_threshold=20, diff_threshold=20.0)

    try:
        from scipy.ndimage import label
        structure = np.ones((3, 3), dtype=int)
        labeled, num_features = label(mask, structure=structure)
        if num_features > 1:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            max_size = sizes.max()
            if max_size >= 40:
                valid_components = np.where(sizes >= max(20, int(max_size * 0.05)))[0]
                mask = np.isin(labeled, valid_components)
    except Exception:
        pass

    coords = np.argwhere(mask)
    if len(coords) < 40:
        return Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    if (x1 - x0 + 1) < 4 or (y1 - y0 + 1) < 8:
        return Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))

    clean_arr = arr.copy()
    clean_arr[~mask, 3] = 0
    clean_image = Image.fromarray(clean_arr, mode="RGBA")
    fg = clean_image.crop((x0, y0, x1 + 1, y1 + 1))
    fw, fh = fg.size

    max_fw = cell_w - 12
    max_fh = cell_h - 14
    target_fh = min(max_fh, max(1, int(round(cell_h * CELL_SPRITE_HEIGHT_RATIO))))
    scale = min(target_fh / max(1, fh), max_fw / max(1, fw))
    new_fw = max(1, int(round(fw * scale)))
    new_fh = max(1, int(round(fh * scale)))
    if (new_fw, new_fh) != (fw, fh):
        fg = fg.resize((new_fw, new_fh), Image.Resampling.NEAREST)
        fw, fh = fg.size

    cell = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
    cx = max(0, (cell_w - fw) // 2)
    cy = max(6, cell_h - fh - 7)
    cell.alpha_composite(fg, (cx, cy))
    return cell


def unpad_from_square(square_img: Image.Image, pad_info: Tuple) -> Image.Image:
    """Restaura a celda original."""
    return place_in_cell(square_img, CELL_WIDTH, CELL_HEIGHT, pad_info)


def isolate_character(front_img: Image.Image) -> Image.Image:
    """Detecta automaticamente el color de fondo y recorta la silueta."""
    img_rgb = front_img.convert("RGB")
    arr = np.array(img_rgb)
    h, w, _ = arr.shape
    sample_points = [arr[5, 5], arr[5, -5], arr[-5, 5], arr[-5, -5]]
    bg_color = np.median(sample_points, axis=0)
    diff = np.sqrt(np.sum((arr.astype(float) - bg_color) ** 2, axis=-1))
    mask = diff > 25.0
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return front_img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    margin_y = int((y1 - y0) * 0.02)
    margin_x = int((x1 - x0) * 0.02)
    return front_img.crop((max(0, x0 - margin_x), max(0, y0 - margin_y), min(w, x1 + margin_x), min(h, y1 + margin_y)))


def slice_spritesheet(sheet_img: Image.Image, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> List[Image.Image]:
    """Segmenta una hoja de spritesheet en celdas individuales."""
    w, h = sheet_img.size
    frames = []
    for r in range(rows):
        for c in range(cols):
            x0, y0, x1, y1 = get_cell_coordinates(w, h, r, c, rows, cols)
            cell = sheet_img.crop((x0, y0, x1, y1))
            frames.append(cell)
    return frames


class TemplateManager:
    """Carga y gestiona la Plantilla de Poses (8x12 - 96 frames con fondo estrictamente transparente)."""
    def __init__(self, template_path: Optional[Path] = None, target_size: int = MODEL_RESOLUTION,
                 rows: int = GRID_ROWS, cols: int = GRID_COLS,
                 canvas_w: int = CANVAS_WIDTH, canvas_h: int = CANVAS_HEIGHT):
        self.path = template_path or TEMPLATE_PATH
        if not self.path.exists():
            raise FileNotFoundError(f"No se encontró la plantilla en: {self.path}")
            
        self.template_img = Image.open(self.path).convert("RGBA")
        self.rows = rows
        self.cols = cols
        if self.template_img.size != (canvas_w, canvas_h):
            print(f"[TemplateManager] Redimensionando plantilla a {canvas_w}x{canvas_h}...")
            self.template_img = self.template_img.resize((canvas_w, canvas_h), Image.Resampling.NEAREST)
        self.target_size = target_size
        self.frames = slice_spritesheet(self.template_img, self.rows, self.cols)
        
        self.frame_tensors = []
        self.frame_masks = []
        self.padded_frames = []
        for frame in self.frames:
            padded_frame = pad_target_frame_canonical(frame, self.target_size)
            self.padded_frames.append(padded_frame)
            arr = np.array(padded_frame).astype(np.float32)
            alpha_mask = (arr[:, :, 3] > 20)
            rgb = arr[:, :, :3]
            rgb[~alpha_mask] = 0.0
            norm_rgb = rgb / 127.5 - 1.0
            tensor = torch.from_numpy(norm_rgb).permute(2, 0, 1).float()
            assert tensor.shape[0] == 3, f"Tensor de pose debe tener 3 canales."
            self.frame_tensors.append(tensor)
            mask_tensor = torch.from_numpy(alpha_mask.astype(np.float32)).unsqueeze(0)
            self.frame_masks.append(mask_tensor)

    def get_frame_tensor(self, frame_idx: int) -> torch.Tensor:
        return self.frame_tensors[frame_idx]

    def get_frame_mask(self, frame_idx: int) -> torch.Tensor:
        return self.frame_masks[frame_idx]

    def get_frame_pil(self, frame_idx: int) -> Image.Image:
        return self.frames[frame_idx]

    def get_frame_padded_pil(self, frame_idx: int) -> Image.Image:
        return self.padded_frames[frame_idx]



def scan_character_datasets(root_dir: Path) -> List[Dict[str, Path]]:
    """
    Escaneo inteligente de carpetas de personajes para entrenamiento supervised.
    Excluye estrictamente Tori para usarlo exclusivamente como evaluacion no vista.
    """
    pairs = []
    root = Path(root_dir)
    char_dirs = []
    
    for pdir in [root / "personajes", root.parent / "personajes"]:
        if pdir.exists() and pdir.is_dir():
            for d in pdir.iterdir():
                if d.is_dir() and d not in char_dirs:
                    char_dirs.append(d)
                    
    variants = ["rnormal", "rbchef", "rnchef"]
    
    seen_pairs = set()
    for cdir in char_dirs:
        # Tori es estrictamente de evaluacion
        cname = cdir.name.lower().replace(" ", "_")
        if cname in ["tori", "test", "_pruebas_y_obsoletos", "_drop"]:
            continue
        files = list(cdir.glob("*.png"))
        sheets = [f for f in files if "movimiento" in f.name.lower()]
        fronts = [f for f in files if "movimiento" not in f.name.lower()]
        
        for var in variants:
            key = (cname, var)
            if key in seen_pairs:
                continue
                
            s_cand = [f for f in sheets if var in f.name.lower()]
            if not s_cand and var == "rnormal":
                s_cand = [f for f in sheets if f.name.lower() in ["movimientos.png", "movimiento.png", "movimiento_rnormal.png"]]
                
            f_cand = [f for f in fronts if var in f.name.lower()]
            if not f_cand and var == "rnormal":
                f_cand = [f for f in fronts if f.name.lower() in ["front.png", "rnormal.png", f"{cdir.name.lower()}.png"]]
                
            if s_cand and f_cand:
                seen_pairs.add(key)
                pairs.append({
                    "character": cdir.name,
                    "variant": var,
                    "front_path": f_cand[0],
                    "sheet_path": s_cand[0]
                })
                
    return pairs


class PixelArtDataset(Dataset):
    """
    Dataset PyTorch Frame-Level Pose-Conditioned con Adaptador Anatomico Chibi.
    Input: [Frontal Chibi (3ch) + Pose Maniqui (3ch)] -> 6 canales
    Target: Sub-sprite pixel art canónico (RGBA) -> 4 canales
    """
    def __init__(self, root_dir: Optional[Path] = None, template_manager: Optional[TemplateManager] = None,
                 target_size: int = MODEL_RESOLUTION, augment: bool = True, phase_cfg: Optional[Dict] = None):
        self.root_dir = root_dir or PROJECT_ROOT
        self.target_size = target_size
        self.augment = augment
        self.phase_cfg = phase_cfg or get_phase_config("2")
        
        grid_rows = self.phase_cfg["grid_rows"]
        grid_cols = self.phase_cfg["grid_cols"]
        canvas_w = self.phase_cfg["canvas_w"]
        canvas_h = self.phase_cfg["canvas_h"]
        template_p = self.phase_cfg["template_path"]
        
        self.template_mgr = template_manager or TemplateManager(
            template_path=template_p,
            target_size=target_size,
            rows=grid_rows,
            cols=grid_cols,
            canvas_w=canvas_w,
            canvas_h=canvas_h
        )
        
        self.pairs = scan_character_datasets(self.root_dir)
        if not self.pairs:
            raise RuntimeError(f"No se encontraron pares validos en {self.root_dir}")
            
        print(f"[Dataset] Modo activo: {self.phase_cfg['phase_name']}")
        print(f"[Dataset] Pares encontrados en disco: {len(self.pairs)}")
        for p in self.pairs:
            print(f"  - {p['character']} ({p['variant']}): front='{p['front_path'].name}', sheet='{p['sheet_path'].name}'")
            
        cache_path = self.phase_cfg["cache_file"]
        if cache_path.exists():
            print(f"[Dataset] Cargando {cache_path.name} desde cache en disco...", flush=True)
            self.samples = torch.load(cache_path)
            print(f"[Dataset] Total de muestras cargadas: {len(self.samples)} sub-sprites.", flush=True)
            return

        self.samples = []
        print(f"[Dataset] Construyendo dataset canonico con Adaptador Anatomico Chibi...", flush=True)
        
        for pair in self.pairs:
            front_img = Image.open(pair["front_path"])
            chibi_front = adapt_front_to_chibi(front_img, target_size=self.target_size)
            arr_f = np.array(chibi_front).astype(np.float32)
            mask_f = arr_f[:, :, 3] > 20
            rgb_f = arr_f[:, :, :3]
            rgb_f[~mask_f] = 0.0
            front_tensor = torch.from_numpy(rgb_f / 127.5 - 1.0).permute(2, 0, 1).float()
            
            sheet_img = Image.open(pair["sheet_path"]).convert("RGBA")
            sw, sh = sheet_img.size
            ratio = sw / max(1, sh)
            is_8x12 = ratio >= 0.45
            
            if is_8x12:
                # Hoja nativa 8x12 (Alex, Amaro) -> 96 frames
                for r in range(12):
                    for c in range(8):
                        f_idx = r * 8 + c
                        x0, y0, x1, y1 = get_cell_coordinates(sw, sh, r, c, 12, 8)
                        cell = sheet_img.crop((x0, y0, x1, y1))
                        padded_tgt = pad_target_frame_canonical(cell, target_size=self.target_size)
                        arr_t = np.array(padded_tgt).astype(np.float32)
                        mask_t = arr_t[:, :, 3] > 20
                        rgb_t = arr_t[:, :, :3]
                        rgb_t[~mask_t] = 0.0
                        norm_rgb_t = rgb_t / 127.5 - 1.0
                        norm_alpha_t = np.where(mask_t, 1.0, -1.0).astype(np.float32)[:, :, np.newaxis]
                        rgba_t = np.concatenate([norm_rgb_t, norm_alpha_t], axis=-1)
                        target_tensor = torch.from_numpy(rgba_t).permute(2, 0, 1).float()
                        
                        self.samples.append({
                            "front_tensor": front_tensor,
                            "frame_idx": f_idx,
                            "target_tensor": target_tensor,
                            "char_id": f"{pair['character']}_{pair['variant']}"
                        })
            else:
                # Hoja 16x4 (Conny, Dana) -> Mapear 48 frames de caminatas/giros a 8x12
                for r in range(16):
                    for c in range(4):
                        mapped_idx = get_8x12_index_from_16x4(r, c)
                        if mapped_idx < 0:
                            continue
                        x0, y0, x1, y1 = get_cell_coordinates(sw, sh, r, c, 16, 4)
                        cell = sheet_img.crop((x0, y0, x1, y1))
                        padded_tgt = pad_target_frame_canonical(cell, target_size=self.target_size)
                        arr_t = np.array(padded_tgt).astype(np.float32)
                        mask_t = arr_t[:, :, 3] > 20
                        rgb_t = arr_t[:, :, :3]
                        rgb_t[~mask_t] = 0.0
                        norm_rgb_t = rgb_t / 127.5 - 1.0
                        norm_alpha_t = np.where(mask_t, 1.0, -1.0).astype(np.float32)[:, :, np.newaxis]
                        rgba_t = np.concatenate([norm_rgb_t, norm_alpha_t], axis=-1)
                        target_tensor = torch.from_numpy(rgba_t).permute(2, 0, 1).float()
                        
                        self.samples.append({
                            "front_tensor": front_tensor,
                            "frame_idx": mapped_idx,
                            "target_tensor": target_tensor,
                            "char_id": f"{pair['character']}_{pair['variant']}"
                        })
                        
        if self.samples:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.samples, cache_path)
            print(f"[Dataset] Cache guardado exitosamente en: {cache_path.name}")
            print(f"[Dataset] Total de muestras preparadas: {len(self.samples)} sub-sprites.", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        front = sample["front_tensor"].clone()
        frame_idx = sample["frame_idx"]
        target = sample["target_tensor"].clone()
        pose = self.template_mgr.get_frame_tensor(frame_idx).clone()
        condition = torch.cat([front, pose], dim=0)
        return condition, target, frame_idx


def get_dataloader(root_dir: Optional[Path] = None, batch_size: int = BATCH_SIZE, phase_cfg: Optional[Dict] = None) -> DataLoader:
    """Retorna DataLoader optimizado."""
    dataset = PixelArtDataset(root_dir=root_dir, phase_cfg=phase_cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
