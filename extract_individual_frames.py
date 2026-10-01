"""
Extractor y Organizador de Frames Individuales de Pixel Art.
Corta cada frame de cada personaje y de cada molde, los limpia, los centra
en celdas estandarizadas de 128x128 con fondo transparente y los categoriza en carpetas.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Configuración de rutas
PROJECT_ROOT = Path(__file__).resolve().parent
PERSONAJES_DIR = PROJECT_ROOT / "personajes"
OUTPUT_BASE = PROJECT_ROOT / "dataset_frames_individuales"
TARGET_SIZE = 256

def foreground_mask(img: Image.Image, alpha_threshold: int = 15) -> np.ndarray:
    """Extrae la máscara del personaje eliminando fondos transparentes o fondos grises de estudio."""
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    if float((alpha < 250).mean()) > 0.02:
        return alpha > alpha_threshold
        
    # Detección de fondo por color (ej: fondo gris de Alex y Amaro)
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
    return diff > 24.0

def center_and_pad_frame(img: Image.Image, target_size: int = TARGET_SIZE, is_portrait: bool = False) -> Image.Image:
    """
    Alinea y centra la figura en un lienzo de 256x256.
    Para sprites de movimiento (pixel art 1:1): NUNCA los achica, preserva el 100% de píxeles exactos.
    Para retratos gigantes de 1536px: usa Lanczos para preservar toda la tinta y detalles de tatuajes.
    """
    arr = np.array(img.convert("RGBA"))
    mask = foreground_mask(img)
    
    # Si la imagen venía con fondo sólido, hacer el fondo 100% transparente
    if float((arr[:, :, 3] < 250).mean()) <= 0.02:
        arr[~mask, 3] = 0
        img = Image.fromarray(arr, mode="RGBA")

    coords = np.argwhere(mask)
    if len(coords) == 0:
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    fg = img.crop((x0, y0, x1, y1))
    fw, fh = fg.size

    # En 256x256, el área segura es de 230x230 píxeles
    max_h = int(target_size * 0.90)
    max_w = int(target_size * 0.90)

    if is_portrait or fw > max_w or fh > max_h:
        scale = min(max_w / max(1, fw), max_h / max(1, fh))
        new_w = max(1, int(round(fw * scale)))
        new_h = max(1, int(round(fh * scale)))
        # Para retratos grandes usamos Lanczos para no perder ni un trazo de tatuaje
        resample = Image.Resampling.LANCZOS if is_portrait else Image.Resampling.NEAREST
        fg = fg.resize((new_w, new_h), resample)
        fw, fh = fg.size

    out = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    pos_x = (target_size - fw) // 2
    # Apoyo de los pies a 12 píxeles del borde inferior
    pos_y = max(4, target_size - fh - 12)
    out.paste(fg, (pos_x, pos_y), fg)
    return out

def slice_and_save_sheet(sheet_path: Path, rows: int, cols: int, out_dir: Path, prefix: str = "frame"):
    """Corta una hoja en fila x columna y guarda cada frame individual limpio a 256x256 nativo."""
    sheet = Image.open(sheet_path).convert("RGBA")
    sw, sh = sheet.size
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for r in range(rows):
        for c in range(cols):
            x0 = int(round(c * sw / cols))
            x1 = int(round((c + 1) * sw / cols))
            y0 = int(round(r * sh / rows))
            y1 = int(round((r + 1) * sh / rows))

            cell = sheet.crop((x0, y0, x1, y1))
            clean_cell = center_and_pad_frame(cell, TARGET_SIZE, is_portrait=False)
            
            frame_idx = r * cols + c + 1
            filename = f"{prefix}_{frame_idx:03d}_r{r+1:02d}_c{c+1:02d}.png"
            clean_cell.save(out_dir / filename)
            count += 1

    return count

VAR_NAMES = {
    "rnormal": "ropa_normal",
    "rnchef": "ropa_negra_chef",
    "rbchef": "ropa_blanca_chef"
}

ACTION_GUIDE_16x4 = """# Guía de Acciones (Plantilla 16x4 - 64 Frames)
- Filas 01 a 04 (Frames 01 a 16): Caminata Frontal / Vista Sur (Idle y pasos)
- Filas 05 a 08 (Frames 17 a 32): Caminata Espalda / Vista Norte (Idle y pasos)
- Filas 09 a 12 (Frames 33 a 48): Caminata Lateral Derecha / Vista Este
- Filas 13 a 16 (Frames 49 a 64): Caminata Lateral Izquierda / Vista Oeste
"""

ACTION_GUIDE_8x12 = """# Guía de Acciones (Plantilla 8x12 - 96 Frames)
- Filas 01 a 08 (Frames 01 a 64): Rotación 360° y pasos en 8 direcciones
  * Fila 01: Sur (Frente)
  * Fila 02: Sureste
  * Fila 03: Este (Derecha)
  * Fila 04: Noreste
  * Fila 05: Norte (Espalda)
  * Fila 06: Noroeste
  * Fila 07: Oeste (Izquierda)
  * Fila 08: Suroeste
- Filas 09 a 10 (Frames 65 a 80): Cocina (Batir en bowl, revolver olla, picar en tabla)
- Fila 11 (Frames 81 a 88): Pensar (mano en barbilla) y Manipular caja
- Fila 12 (Frames 89 a 96): Servir plato y Celebración (manos arriba)
"""

def extract_all():
    print("=" * 70)
    print("  EXTRACTOR Y CATEGORIZADOR DE FRAMES INDIVIDUALES DE PIXEL ART")
    print(f"  Directorio destino: {OUTPUT_BASE}")
    print("=" * 70)

    total_extracted = 0

    # 1. EXTRAER MOLDES (MANIQUÍES DE REFERENCIA)
    print("\n[1/2] Extrayendo Moldes de Pose (Maniquíes)...")
    moldes_dir = OUTPUT_BASE / "00_MOLDES_POSES"
    
    # Molde 16x4 (64 frames)
    m16_path = PROJECT_ROOT / "plantilla_16x4.png"
    if m16_path.exists():
        dir_m16 = moldes_dir / "molde_16x4_64_poses"
        n = slice_and_save_sheet(m16_path, 16, 4, dir_m16, prefix="pose")
        (dir_m16 / "LEEME_ACCIONES.md").write_text(ACTION_GUIDE_16x4, encoding="utf-8")
        print(f"  -> Molde 16x4: {n} poses guardadas en 00_MOLDES_POSES/molde_16x4_64_poses")
        total_extracted += n

    # Molde 8x12 (96 frames)
    m8_path = PROJECT_ROOT / "plantilla de los spritesheets.png"
    if m8_path.exists():
        dir_m8 = moldes_dir / "molde_8x12_96_poses"
        n = slice_and_save_sheet(m8_path, 12, 8, dir_m8, prefix="pose")
        (dir_m8 / "LEEME_ACCIONES.md").write_text(ACTION_GUIDE_8x12, encoding="utf-8")
        print(f"  -> Molde 8x12: {n} poses guardadas en 00_MOLDES_POSES/molde_8x12_96_poses")
        total_extracted += n

    # 2. EXTRAER CADA PERSONAJE Y SUS VARIANTES
    print("\n[2/2] Extrayendo y categorizando personajes por carpetas...")
    chars = sorted([d for d in PERSONAJES_DIR.iterdir() if d.is_dir()])
    
    for cdir in chars:
        char_name = cdir.name
        print(f"\n[PERSONAJE] Procesando: {char_name.upper()}...")
        files = list(cdir.glob("*.png"))
        
        variants = ["rnormal", "rnchef", "rbchef"]
        for var in variants:
            sheet_cand = [f for f in files if "movimiento" in f.name.lower() and var in f.name.lower()]
            front_cand = [f for f in files if "movimiento" not in f.name.lower() and var in f.name.lower()]
            
            if not sheet_cand:
                continue

            sheet_file = sheet_cand[0]
            sheet_im = Image.open(sheet_file)
            sw, sh = sheet_im.size
            ratio = sw / max(1, sh)

            # Auto-detectar si es 16x4 o 8x12
            if ratio < 0.45:
                rows, cols = 16, 4
                guide_text = ACTION_GUIDE_16x4
            else:
                rows, cols = 12, 8
                guide_text = ACTION_GUIDE_8x12

            var_readable = VAR_NAMES.get(var, var)
            dest_folder = OUTPUT_BASE / char_name.upper() / f"{char_name}_{var_readable}"
            dest_folder.mkdir(parents=True, exist_ok=True)

            # Guardar Frontal limpio
            if front_cand:
                front_im = Image.open(front_cand[0]).convert("RGBA")
                clean_front = center_and_pad_frame(front_im, TARGET_SIZE, is_portrait=True)
                clean_front.save(dest_folder / "00_frontal_identidad.png")

            # Guardar guía explicativa de acciones
            (dest_folder / "LEEME_ACCIONES.md").write_text(guide_text, encoding="utf-8")

            # Cortar frames
            n_frames = slice_and_save_sheet(sheet_file, rows, cols, dest_folder, prefix="frame")
            print(f"  [OK] {char_name.upper()} -> {var_readable} ({rows}x{cols} = {n_frames} frames): {dest_folder.relative_to(PROJECT_ROOT)}")
            total_extracted += n_frames

    print("\n" + "=" * 70)
    print(f"  ¡ÉXITO TOTAL! {total_extracted} FRAMES INDIVIDUALES GENERADOS")
    print(f"  Ubicación: {OUTPUT_BASE.resolve()}")
    print("=" * 70)

if __name__ == "__main__":
    extract_all()

