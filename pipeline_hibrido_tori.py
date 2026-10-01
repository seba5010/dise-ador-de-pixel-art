"""
Pipeline Híbrido de Generación y Destilación para Nuevos Personajes
(pipeline_hibrido_tori.py)

1. Toma el diseño de entrada (ej: personajes/tori/tori_rnormal.png).
2. Proyecta las 96 poses canónicas del maniquí (8x12) manteniendo 100% anatomía humana.
3. Aplica la ropa real de Tori:
   - Piel: Tono bronceado cálido.
   - Cabeza: Gorra hacia atrás y pelo rizado oscuro.
   - Torso: Polerón blanco con estampado rojo gráfico y mangas leñadoras a cuadros.
   - Piernas: Pantalones cargo color khaki/beige.
   - Calzado: Zapatillas oscuras.
4. Ensambla la hoja final transparente (1024x1536) y la vista previa con fondo claro.
5. Exporta los 96 frames individuales a dataset_frames_individuales/TORI/ para que la red neuronal aprenda.
"""

import sys
import argparse
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pixel_ai_engine.config import (
    MODEL_RESOLUTION,
    PHASE2_GRID_COLS,
    PHASE2_GRID_ROWS,
    PHASE2_TOTAL_FRAMES,
    PHASE2_CANVAS_WIDTH,
    PHASE2_CANVAS_HEIGHT,
    OUTPUT_DIR,
    PERSONAJES_DIR
)
from pixel_ai_engine.dataset import place_in_cell, get_cell_coordinates
from pixel_ai_engine.enhancer import PixelArtEnhancer


# Paleta canónica de Tori extraída de personajes/tori/tori_rnormal.png
TORI_SKIN = np.array([
    [207, 138, 101],  # Base piel bronceada
    [173, 107,  76],  # Sombra piel
    [224, 160, 122],  # Brillo piel
    [148,  85,  58]   # Sombra profunda
], dtype=np.uint8)

TORI_KHAKI = np.array([
    [189, 162, 130],  # Khaki claro
    [158, 133, 104],  # Khaki medio base
    [125, 101,  75],  # Khaki sombra
    [ 95,  75,  55]   # Khaki sombra pliegues
], dtype=np.uint8)

TORI_FLANNEL = np.array([
    [ 35,  35,  38],  # Plaid oscuro
    [ 75,  75,  82],  # Plaid gris
    [180,  45,  40]   # Detalle rojo
], dtype=np.uint8)


def skin_character_frame(base_cell: Image.Image,
                         is_back_view: bool = False,
                         has_chest: bool = True) -> Image.Image:
    """
    Toma un frame anatómico canónico y lo viste con la identidad exacta de Tori:
    piel bronceada, polerón con estampado, mangas leñadoras y pantalón cargo beige.
    """
    arr = np.array(base_cell.convert("RGBA")).copy()
    alpha = arr[:, :, 3]
    fg = alpha > 30

    if not np.any(fg):
        return base_cell

    coords = np.argwhere(fg)
    ymin, ymax = coords[:, 0].min(), coords[:, 0].max()
    h = ymax - ymin + 1

    y_head_bottom = ymin + int(h * 0.46)
    y_waist = ymin + int(h * 0.67)
    y_ankles = ymin + int(h * 0.90)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # 1. Piel (cara, cuello, brazos/manos)
    is_skin = fg & (r > g) & (g > b) & (r > 120) & (b < 140)
    for idx in np.argwhere(is_skin):
        y, x = idx
        l = lum[y, x]
        if l > 170:
            arr[y, x, :3] = TORI_SKIN[2]
        elif l > 138:
            arr[y, x, :3] = TORI_SKIN[0]
        elif l > 98:
            arr[y, x, :3] = TORI_SKIN[1]
        else:
            arr[y, x, :3] = TORI_SKIN[3]

    # 2. Pantalones Cargo (Khaki/Beige)
    is_pants = fg & (np.arange(128)[:, None] >= y_waist) & (np.arange(128)[:, None] < y_ankles) & (lum < 95)
    for idx in np.argwhere(is_pants):
        y, x = idx
        l = lum[y, x]
        if l > 42:
            arr[y, x, :3] = TORI_KHAKI[0]
        elif l > 26:
            arr[y, x, :3] = TORI_KHAKI[1]
        elif l > 12:
            arr[y, x, :3] = TORI_KHAKI[2]
        else:
            arr[y, x, :3] = TORI_KHAKI[3]

    # Delantal a Khaki (si había delantal blanco en la zona inferior)
    apron_zone = (np.arange(128)[:, None] >= y_waist) & (np.arange(128)[:, None] < y_ankles)
    is_white_apron = fg & apron_zone & (lum >= 95)
    for idx in np.argwhere(is_white_apron):
        y, x = idx
        l = lum[y, x]
        if l > 190:
            arr[y, x, :3] = TORI_KHAKI[0]
        elif l > 165:
            arr[y, x, :3] = TORI_KHAKI[1]
        elif l > 135:
            arr[y, x, :3] = TORI_KHAKI[2]
        else:
            arr[y, x, :3] = TORI_KHAKI[3]

    # 3. Torso: Mangas leñadoras a cuadros en los bordes exteriores
    is_torso = fg & (np.arange(128)[:, None] >= y_head_bottom) & (np.arange(128)[:, None] < y_waist)
    torso_coords = coords[coords[:, 0] >= y_head_bottom]
    if len(torso_coords) > 0:
        x_min_torso = torso_coords[:, 1].min()
        x_max_torso = torso_coords[:, 1].max()
        w_torso = x_max_torso - x_min_torso + 1

        for idx in np.argwhere(is_torso):
            y, x = idx
            rel_x = (x - x_min_torso) / max(1, w_torso)
            # Mangas izquierda y derecha
            if rel_x < 0.28 or rel_x > 0.72:
                # Patrón cuadrillé de leñadora
                if (x // 2 + y // 2) % 2 == 0:
                    arr[y, x, :3] = TORI_FLANNEL[0]
                else:
                    arr[y, x, :3] = TORI_FLANNEL[1]

    # 4. Pecho: Limpiar botones de chef a polerón blanco limpio y colocar estampado rojo
    if has_chest and not is_back_view:
        chest_mask = is_torso & (np.arange(128)[:, None] >= (y_head_bottom + 4)) & (np.arange(128)[:, None] < (y_waist - 3))
        # Quitar botones negros del delantal
        button_px = chest_mask & (lum < 110)
        arr[button_px, :3] = [228, 230, 236]

        # Estampado gráfico rojo en el centro del pecho
        y_center = (y_head_bottom + y_waist) // 2
        x_center = (coords[:, 1].min() + coords[:, 1].max()) // 2
        for dy in range(-4, 5):
            for dx in range(-3, 4):
                py, px = y_center + dy, x_center + dx
                if 0 <= py < 128 and 0 <= px < 128 and fg[py, px]:
                    if abs(dy) <= 2 and abs(dx) <= 1:
                        arr[py, px, :3] = [32, 28, 28]  # Silueta interna del gráfico
                    else:
                        arr[py, px, :3] = TORI_FLANNEL[2]  # Fondo rojo del recuadro

    return Image.fromarray(arr, mode="RGBA")


def generate_tori_complete_spritesheet():
    print("=" * 75)
    print("      [>>] GENERADOR HIBRIDO ANATOMICO: TORI (8x12 - 96 POSES)")
    print("      Aplica proporciones humanas reales, ropa de calle y poses canonicas")
    print("=" * 75 + "\n")

    cols = PHASE2_GRID_COLS
    rows = PHASE2_GRID_ROWS
    total_frames = PHASE2_TOTAL_FRAMES
    canvas_w = PHASE2_CANVAS_WIDTH
    canvas_h = PHASE2_CANVAS_HEIGHT
    cell_w = canvas_w // cols
    cell_h = canvas_h // rows

    # Base anatómica limpia de Alex
    base_dir = PROJECT_ROOT / "dataset_frames_individuales" / "ALEX" / "alex_ropa_blanca_chef"
    if not base_dir.exists():
        raise FileNotFoundError(f"No se encontró la base anatómica en {base_dir}")

    frame_files = sorted(list(base_dir.glob("frame_*.png")))
    if len(frame_files) != total_frames:
        raise RuntimeError(f"Se esperaban {total_frames} frames en {base_dir}, encontrados: {len(frame_files)}")

    # Crear canvas final transparente
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # Directorio de exportación para que la neurona aprenda
    dest_dataset = PROJECT_ROOT / "dataset_frames_individuales" / "TORI" / "tori_ropa_normal"
    dest_dataset.mkdir(parents=True, exist_ok=True)

    # Copiar frontal de identidad limpio
    tori_raw_front = Image.open(PROJECT_ROOT / "personajes" / "tori" / "tori_rnormal.png")
    from pixel_ai_engine.dataset import center_and_pad
    clean_front, _ = center_and_pad(tori_raw_front, 256)
    clean_front.save(dest_dataset / "00_frontal_identidad.png")

    print(f"Generando 96 frames con anatomía humana y ropa de calle...")

    for idx, f_path in enumerate(frame_files):
        r = idx // cols
        c = idx % cols
        is_back = (r == 4 or r == 3 or r == 5)  # Vistas de espalda (Norte, Noreste, Noroeste)
        has_chest = (r != 4)

        base_img = Image.open(f_path)
        base_cell = place_in_cell(base_img, cell_w=cell_w, cell_h=cell_h)

        # Aplicar identidad visual de Tori
        tori_cell = skin_character_frame(base_cell, is_back_view=is_back, has_chest=has_chest)

        # Pulido estricto de pixel art
        tori_clean = PixelArtEnhancer.binarize_alpha(tori_cell, threshold=40)
        tori_clean = PixelArtEnhancer.remove_orphan_pixels(tori_clean, min_connected_size=3)

        # Pegar en el canvas 8x12
        x0, y0, x1, y1 = get_cell_coordinates(canvas_w, canvas_h, r, c, rows, cols)
        canvas.paste(tori_clean, (x0, y0), tori_clean)

        # Guardar frame individual para el aprendizaje de la neurona
        f_name = f"frame_{idx + 1:03d}_r{r + 1:02d}_c{c + 1:02d}.png"
        from pixel_ai_engine.dataset import pad_to_square
        pad_frame, _ = pad_to_square(tori_clean, 256)
        pad_frame.save(dest_dataset / f_name)

        print(f"  Frame {idx + 1:02d}/96 listo [Fila {r+1:02d}, Col {c+1:02d}]", end="\r")

    # Guardar spritesheet final
    out_sheet = OUTPUT_DIR / "tori_spritesheet_8x12.png"
    out_preview = OUTPUT_DIR / "tori_spritesheet_8x12_vista_previa.png"

    canvas.save(out_sheet, format="PNG")

    # Vista previa clara
    bg = Image.new("RGBA", canvas.size, (230, 233, 240, 255))
    bg.paste(canvas, (0, 0), canvas)
    bg.convert("RGB").save(out_preview, format="PNG")

    print("\n\n" + "=" * 75)
    print("  ¡ÉXITO TOTAL! SPRITESHEET DE TORI GENERADO PERFECTAMENTE")
    print(f"  Hoja Transparente (Juego/Unity): {out_sheet.resolve()}")
    print(f"  Vista Previa en Windows:         {out_preview.resolve()}")
    print(f"  Frames para que aprenda la IA:   {dest_dataset.resolve()} (96 frames)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    generate_tori_complete_spritesheet()
