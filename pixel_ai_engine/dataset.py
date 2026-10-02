"""
Pixel AI Engine - Dataset Loader & Preprocessing
Gestiona la lectura flexible de carpetas de personajes, corte de los 64 frames (16x4),
aislamiento de fondo, normalización matemática y acondicionamiento con la Plantilla Universal.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
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


def get_cell_coordinates(canvas_w: int, canvas_h: int, row: int, col: int, total_rows: int = GRID_ROWS, total_cols: int = GRID_COLS):
    """
    Calcula las coordenadas exactas de corte con redondeo estricto para no perder píxeles.
    """
    x0 = int(round(col * canvas_w / total_cols))
    x1 = int(round((col + 1) * canvas_w / total_cols))
    y0 = int(round(row * canvas_h / total_rows))
    y1 = int(round((row + 1) * canvas_h / total_rows))
    return x0, y0, x1, y1


def foreground_mask(img: Image.Image, alpha_threshold: int = 10, diff_threshold: float = 28.0) -> np.ndarray:
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
    """
    Alinea y centra el contenido no vacío del sprite horizontalmente con los pies apoyados
    en la base inferior. En 256x256 preserva el 100% de píxeles nativos sin reducción.
    Elimina fondos sólidos automáticamente haciendo la máscara 100% transparente.
    """
    arr = np.array(img.convert("RGBA"))
    mask = foreground_mask(img)

    # Si la imagen venía con fondo sólido (ej: fotos de estudio con fondo gris), hacer el fondo transparente
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

    # En 256x256, el área segura es de 230x230 píxeles. Si el sprite mide <= 230px, scale=1.0 (nativo 1:1)
    safe_dim = int(target_size * 0.90)
    scale = min(safe_dim / max(1, cw), safe_dim / max(1, ch), 1.0)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))

    # Si es un retrato grande (>400px), usar Lanczos para preservar detalles finos; si es pixel art, Nearest
    if scale < 0.999:
        resample = Image.Resampling.LANCZOS if (cw > 400 or ch > 400) else Image.Resampling.NEAREST
        content_scaled = content.resize((new_w, new_h), resample)
    else:
        content_scaled = content

    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    # Centrado horizontal exacto y anclaje inferior de pies consistente
    ox = (target_size - new_w) // 2
    oy = max(4, target_size - new_h - 12)
    content_scaled = content_scaled.convert("RGBA")
    canvas.paste(content_scaled, (ox, oy), content_scaled)

    return canvas, (cw, ch, scale, ox, oy)


def pad_to_square(img: Image.Image, target_size: int = MODEL_RESOLUTION) -> Tuple[Image.Image, Tuple]:
    """Alias compatible con código existente usando centrado y anclaje anatómico."""
    return center_and_pad(img, target_size)


def place_in_cell(padded_img: Image.Image, cell_w: int = CELL_WIDTH, cell_h: int = CELL_HEIGHT, pad_meta: Optional[Tuple] = None) -> Image.Image:
    """
    Toma el frame generado de 128x128, extrae la figura y la ubica perfectamente centrada
    con los pies apoyados en el suelo dentro de la celda canónica (181 x 136) para Unity.
    Funciona tanto con imágenes RGBA (salida del generador) como RGB (datos de entrenamiento).
    Si la figura es más grande que la celda, la escala proporcionalmente para que quepa.
    """
    arr = np.array(padded_img.convert("RGBA"))

    # Intentar detección por canal alfa primero
    alpha = arr[:, :, 3]
    mask = alpha > 20
    if mask.sum() < 4:
        # Fallback: detección por contraste RGB (fondo claro vs figura oscura o viceversa)
        rgb = arr[:, :, :3].astype(np.float32)
        # Muestrear esquinas para estimar el color de fondo
        bg = np.median([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], axis=0)
        diff = np.sqrt(np.sum((rgb - bg) ** 2, axis=-1))
        mask = diff > 20.0

    # ── FILTRADO DE PUNTOS DE RUIDO AISLADOS PARA EVITAR ACHICAMIENTO ARTIFICIAL ──
    # Si hay puntos de ruido sueltos en el fondo, los excluimos para que no expandan falsamente la caja del sprite
    try:
        from scipy.ndimage import label
        structure = np.ones((3, 3), dtype=int)
        labeled, num_features = label(mask, structure=structure)
        if num_features > 1:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0  # Ignorar fondo
            max_size = sizes.max()
            if max_size >= 40:
                # Solo considerar componentes significativos (al menos 5% del cuerpo o > 25 px)
                valid_components = np.where(sizes >= max(20, int(max_size * 0.05)))[0]
                mask = np.isin(labeled, valid_components)
    except Exception:
        pass

    coords = np.argwhere(mask)
    if len(coords) == 0:
        return Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    fg = padded_img.convert("RGBA").crop((x0, y0, x1 + 1, y1 + 1))

    # Si se conoce la escala original (datos de entrenamiento), restaurar al tamaño real
    if pad_meta and len(pad_meta) >= 3:
        cw, ch, scale = pad_meta[0], pad_meta[1], pad_meta[2]
        if scale < 0.99 and cw > 0 and ch > 0:
            fg = fg.resize((cw, ch), Image.Resampling.NEAREST)

    fw, fh = fg.size

    # Si la figura excede la celda, escalarla proporcionalmente para que quepa con margen seguro
    max_fw = cell_w - 8
    max_fh = cell_h - 14
    if fw > max_fw or fh > max_fh:
        scale_down = min(max_fw / max(1, fw), max_fh / max(1, fh))
        new_fw = max(1, int(round(fw * scale_down)))
        new_fh = max(1, int(round(fh * scale_down)))
        fg = fg.resize((new_fw, new_fh), Image.Resampling.NEAREST)
        fw, fh = fg.size

    cell = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
    cx = max(0, (cell_w - fw) // 2)
    # Margen seguro de 7px inferior y mínimo 6px superior: nunca sangra a filas adyacentes
    cy = max(6, cell_h - fh - 7)
    cell.paste(fg, (cx, cy), fg)
    return cell


def unpad_from_square(square_img: Image.Image, pad_info: Tuple) -> Image.Image:
    """Restaura a celda original (181 x 136)."""
    return place_in_cell(square_img, CELL_WIDTH, CELL_HEIGHT, pad_info)


def isolate_character(front_img: Image.Image) -> Image.Image:
    """
    Detecta automáticamente el color de fondo (incluso si no es blanco puro, ej. gris de Belial)
    y recorta la caja delimitadora (bounding box) centrada del personaje.
    """
    img_rgb = front_img.convert("RGB")
    arr = np.array(img_rgb)
    h, w, _ = arr.shape
    
    # Muestrear esquinas para determinar el color de fondo dominante
    sample_points = [arr[5, 5], arr[5, -5], arr[-5, 5], arr[-5, -5]]
    bg_color = np.median(sample_points, axis=0)
    
    # Calcular distancia euclídea al fondo
    diff = np.sqrt(np.sum((arr.astype(float) - bg_color) ** 2, axis=-1))
    mask = diff > 25.0
    coords = np.argwhere(mask)
    
    if len(coords) == 0:
        return front_img
        
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    
    # Margen de seguridad del 2%
    margin_y = int((y1 - y0) * 0.02)
    margin_x = int((x1 - x0) * 0.02)
    y0 = max(0, y0 - margin_y)
    x0 = max(0, x0 - margin_x)
    y1 = min(h, y1 + margin_y)
    x1 = min(w, x1 + margin_x)
    
    return front_img.crop((x0, y0, x1, y1))


def slice_spritesheet(sheet_img: Image.Image, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> List[Image.Image]:
    """
    Segmenta una hoja de spritesheet en 64 sub-sprites individuales (fila por fila, col 0 a 3).
    """
    w, h = sheet_img.size
    frames = []
    for r in range(rows):
        for c in range(cols):
            x0, y0, x1, y1 = get_cell_coordinates(w, h, r, c, rows, cols)
            cell = sheet_img.crop((x0, y0, x1, y1))
            frames.append(cell)
    return frames


class TemplateManager:
    """
    Carga y gestiona la Plantilla de Movimientos (adaptable a 16x4 o 8x12).
    """
    def __init__(self, template_path: Optional[Path] = None, target_size: int = MODEL_RESOLUTION,
                 rows: int = GRID_ROWS, cols: int = GRID_COLS,
                 canvas_w: int = CANVAS_WIDTH, canvas_h: int = CANVAS_HEIGHT):
        self.path = template_path or TEMPLATE_PATH
        if not self.path.exists():
            raise FileNotFoundError(f"No se encontró la plantilla en: {self.path}")
            
        self.template_img = Image.open(self.path).convert("RGB")
        self.rows = rows
        self.cols = cols
        # Auto-redimensionar plantilla si no coincide con el canvas esperado
        if self.template_img.size != (canvas_w, canvas_h):
            print(f"[TemplateManager] Plantilla en {self.template_img.size[0]}x{self.template_img.size[1]}, "
                  f"redimensionando a {canvas_w}x{canvas_h} para grilla {cols}x{rows}...")
            self.template_img = self.template_img.resize((canvas_w, canvas_h), Image.Resampling.NEAREST)
        self.target_size = target_size
        self.frames = slice_spritesheet(self.template_img, self.rows, self.cols)
        
        # Pre-procesar frames a tensores normalizados [-1, 1]
        self.frame_tensors = []
        for frame in self.frames:
            padded_frame, _ = pad_to_square(frame, self.target_size)
            padded_rgb = padded_frame.convert("RGB")
            arr = np.array(padded_rgb).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float()  # (3, H, W)
            assert tensor.shape[0] == 3, f"[TemplateManager] Error: tensor de pose tiene {tensor.shape[0]} canales, se esperaban 3."
            self.frame_tensors.append(tensor)

    def get_frame_tensor(self, frame_idx: int) -> torch.Tensor:
        return self.frame_tensors[frame_idx]

    def get_frame_pil(self, frame_idx: int) -> Image.Image:
        return self.frames[frame_idx]


def scan_character_datasets(root_dir: Path) -> List[Dict[str, Path]]:
    """
    Escaneo inteligente de carpetas de personajes en el workspace y en la subcarpeta 'personajes'.
    Maneja todas las convenciones reales del proyecto:
      - personajes/<nombre>/<variante>.png <-> personajes/<nombre>/movimientos_<variante>.png
      - belial/belial_rnormal.png <-> belial/movimientos_rnormal.png
      - Nombres singulares: movimiento_rnormal.png
      - Sin sufijo: movimientos.png
      - Variantes con puntos o prefijos: diego.vallenar_*, diegos_*, mati_*
    """
    pairs = []
    root = Path(root_dir)
    
    # Recolectar carpetas de personajes
    char_dirs = []
    personajes_dir = root / "personajes"
    if personajes_dir.exists() and personajes_dir.is_dir():
        char_dirs.extend([d for d in personajes_dir.iterdir() if d.is_dir()])
        
    for d in root.iterdir():
        if d.is_dir() and d.name not in ["personajes", "webui forger", "system", ".git", "checkpoints", "output", "training_samples", "PLANTILLAS_CONTROLNET", "pixel_ai_engine"]:
            if d not in char_dirs:
                char_dirs.append(d)
                
    variants = ["rnormal", "rbchef", "rnchef", "rbnormal"]
    
    for cdir in char_dirs:
        if cdir.name.lower() in ["tori", "test", "_pruebas_y_obsoletos"]:
            continue
        files = list(cdir.glob("*.png"))
        sheets = [f for f in files if "movimiento" in f.name.lower()]
        fronts = [f for f in files if "movimiento" not in f.name.lower()]
        
        for var in variants:
            # Buscar hoja de movimientos para esta variante
            s_cand = [f for f in sheets if var in f.name.lower()]
            if not s_cand and var == "rnormal":
                # Soporte para "movimientos.png" o "movimiento.png"
                s_cand = [f for f in sheets if f.name.lower() in ["movimientos.png", "movimiento.png", "movimiento_rnormal.png"]]
                
            # Buscar personaje frontal para esta variante
            f_cand = [f for f in fronts if var in f.name.lower()]
            if not f_cand and var == "rnormal":
                f_cand = [f for f in fronts if f.name.lower() in ["front.png", "rnormal.png", f"{cdir.name.lower()}.png"]]
                
            if s_cand and f_cand:
                pairs.append({
                    "character": cdir.name,
                    "variant": var,
                    "front_path": f_cand[0],
                    "sheet_path": s_cand[0]
                })
                
    return pairs


class PixelArtDataset(Dataset):
    """
    Dataset PyTorch para entrenamiento Frame-Level Pose-Conditioned Pix2Pix / U-Net.
    Cada muestra consiste en:
      - Input X (6 canales): [Personaje Frontal (3ch) + Pose de Plantilla Maniquí (3ch)]
      - Target Y (4 canales): Sub-sprite objetivo en formato RGBA [-1, 1]
    """
    def __init__(self, root_dir: Optional[Path] = None, template_manager: Optional[TemplateManager] = None,
                 target_size: int = MODEL_RESOLUTION, augment: bool = True, phase_cfg: Optional[Dict] = None):
        self.root_dir = root_dir or PROJECT_ROOT
        self.target_size = target_size
        self.augment = augment
        self.phase_cfg = phase_cfg or get_phase_config("1")
        
        grid_rows = self.phase_cfg["grid_rows"]
        grid_cols = self.phase_cfg["grid_cols"]
        total_frames = self.phase_cfg["total_frames"]
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
            raise RuntimeError(f"No se encontraron pares (front, sheet) válidos en {self.root_dir}")
            
        print(f"[Dataset] Modo activo: {self.phase_cfg['phase_name']}")
        print(f"[Dataset] Encontrados {len(self.pairs)} pares de personajes/variantes en disco:")
        for p in self.pairs:
            print(f"  - {p['character']} ({p['variant']}): front='{p['front_path'].name}', sheet='{p['sheet_path'].name}'")
            
        # Verificar si existe caché en disco para carga instantánea
        cache_path = self.phase_cfg["cache_file"]
        if cache_path.exists():
            print(f"[Dataset] Cargando {cache_path.name} desde caché en disco (inicio ultrarrápido)...", flush=True)
            self.samples = torch.load(cache_path)
            print(f"[Dataset] Total de muestras cargadas desde caché: {len(self.samples)} sub-sprites.", flush=True)
            return

        # Precargar y cortar hojas o cargar frames individuales pre-extraídos
        self.samples = []
        frames_dir = self.root_dir / "dataset_frames_individuales"

        if frames_dir.exists() and frames_dir.is_dir():
            print(f"[Dataset] Modo activo: {self.phase_cfg['phase_name']}")
            print(f"[Dataset] Cargando frames individuales desde {frames_dir.name}...")
            
            for char_dir in sorted(frames_dir.iterdir()):
                if not char_dir.is_dir() or char_dir.name.startswith("00_"):
                    continue
                for var_dir in sorted(char_dir.iterdir()):
                    if not var_dir.is_dir():
                        continue
                    
                    frame_files = sorted(list(var_dir.glob("frame_*.png")))
                    front_files = list(var_dir.glob("00_front*.png"))
                    
                    if not frame_files or not front_files:
                        continue
                        
                    # Filtrar por compatibilidad de fase (64 frames para Fase 1, 96 frames para Fase 2)
                    if len(frame_files) != total_frames:
                        continue
                        
                    # Cargar frontal (3 canales RGB [-1, 1])
                    front_img = Image.open(front_files[0]).convert("RGBA")
                    # Fondo transparente a fondo neutro/RGB
                    front_rgb = Image.new("RGB", front_img.size, (0, 0, 0))
                    front_rgb.paste(front_img, mask=front_img.split()[3])
                    front_arr = np.array(front_rgb).astype(np.float32) / 127.5 - 1.0
                    front_tensor = torch.from_numpy(front_arr).permute(2, 0, 1).float()
                    
                    for f_idx, f_file in enumerate(frame_files):
                        target_img = Image.open(f_file).convert("RGBA")
                        target_arr = np.array(target_img).astype(np.float32) / 127.5 - 1.0
                        target_tensor = torch.from_numpy(target_arr).permute(2, 0, 1).float()  # (4, H, W)
                        
                        self.samples.append({
                            "front_tensor": front_tensor,
                            "frame_idx": f_idx,
                            "target_tensor": target_tensor,
                            "char_id": char_dir.name
                        })
                    print(f"  [CARGADO] {char_dir.name}/{var_dir.name}: {len(frame_files)} frames individuales.")

        if not self.samples:
            # Fallback a segmentar hojas de personajes
            print(f"[Dataset] Segmentando hojas desde 'personajes' para {self.phase_cfg['phase_name']}...", flush=True)
            for pair in self.pairs:
                sheet_img = Image.open(pair["sheet_path"]).convert("RGBA")
                sw, sh = sheet_img.size
                
                ratio = sw / max(1, sh)
                is_16x4 = (ratio < 0.45)
                is_8x12 = (ratio >= 0.45)
                
                is_compatible = (grid_rows == 16 and is_16x4) or (grid_rows == 12 and is_8x12)
                
                if not is_compatible:
                    target_fmt = "16x4 (64 frames)" if grid_rows == 16 else "8x12 (96 frames)"
                    actual_fmt = "16x4" if is_16x4 else "8x12"
                    print(f"  [OMITIDO EN ESTA FASE] {pair['character']} ({pair['variant']}): sheet {sw}x{sh} es formato {actual_fmt} (se requiere {target_fmt}).")
                    continue
                
                # Procesar Frontal
                front_img = Image.open(pair["front_path"])
                clean_front = isolate_character(front_img).convert("RGB")
                padded_front, _ = pad_to_square(clean_front, self.target_size)
                padded_front_rgb = padded_front.convert("RGB")
                front_arr = np.array(padded_front_rgb).astype(np.float32) / 127.5 - 1.0
                front_tensor = torch.from_numpy(front_arr).permute(2, 0, 1).float()  # (3, H, W)
                assert front_tensor.shape[0] == 3, f"[Dataset] Error: front_tensor tiene {front_tensor.shape[0]} canales, se esperaban 3."
                
                # Cortar frames
                target_frames = slice_spritesheet(sheet_img, grid_rows, grid_cols)
                
                for frame_idx, frame_img in enumerate(target_frames):
                    padded_target, _ = pad_to_square(frame_img, self.target_size)
                    target_arr = np.array(padded_target).astype(np.float32) / 127.5 - 1.0
                    target_tensor = torch.from_numpy(target_arr).permute(2, 0, 1).float()  # (4, H, W)
                    
                    self.samples.append({
                        "front_tensor": front_tensor,
                        "frame_idx": frame_idx,
                        "target_tensor": target_tensor,
                        "char_id": pair["character"]
                    })
                    
        # Guardar en caché para que las futuras ejecuciones carguen en 1 segundo
        if self.samples:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.samples, cache_path)
            print(f"[Dataset] Caché guardado en: {cache_path.name}")
            print(f"[Dataset] Total de muestras preparadas para {self.phase_cfg['phase_name']}: {len(self.samples)} sub-sprites.", flush=True)
        else:
            print(f"[Aviso] No se encontraron muestras compatibles para {self.phase_cfg['phase_name']}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        front = sample["front_tensor"].clone()
        frame_idx = sample["frame_idx"]
        target = sample["target_tensor"].clone()
        pose = self.template_mgr.get_frame_tensor(frame_idx).clone()
        
        # Concatenar Front (3 canales) y Pose (3 canales) -> (6, H, W)
        condition = torch.cat([front, pose], dim=0)
        
        return condition, target, frame_idx


def get_dataloader(root_dir: Optional[Path] = None, batch_size: int = BATCH_SIZE, phase_cfg: Optional[Dict] = None) -> DataLoader:
    """
    Retorna el DataLoader optimizado para GPU con pin_memory.
    """
    dataset = PixelArtDataset(root_dir=root_dir, phase_cfg=phase_cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
