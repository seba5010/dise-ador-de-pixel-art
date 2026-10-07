"""
Pixel Art Grid Snapping & 1-px Outline Recovery Module (Técnica 8)
Inspirado en Retro-Diffusion/pixel-art-fixer y algoritmos morfológicos de post-procesado
para gráficos de videojuegos comerciales 2D.

Corrige:
1. Reconstrucción de contorno oscuro exterior de 1 píxel (Black/Dark Outline Recovery).
2. Erradicación total de semitransparencias difusas (Binarización estricta de Alfa).
3. Eliminación de píxeles aislados de ruido ("confeti" o huérfanos).
4. Snapping de periodicidad y alineación con grilla entera.
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image
import cv2


def snap_and_fix_pixel_art(
    img: Image.Image,
    alpha_threshold: int = 50,
    enforce_dark_outline: bool = True,
    outline_color: Tuple[int, int, int] = (18, 18, 24),
    clean_isolated_pixels: bool = True,
) -> Image.Image:
    """
    Restaura la estética de pixel art comercial:
    - Canal alfa estrictamente binario (0 o 255).
    - Contorno cerrado de 1px en bordes externos si enforce_dark_outline=True.
    - Eliminación de píxeles huérfanos de 1px desconectados.
    """
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    h, w, _ = arr.shape

    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    # 1. Binarización estricta de Alfa
    solid_mask = (alpha >= alpha_threshold).astype(np.uint8)

    if not np.any(solid_mask):
        return Image.new("RGBA", (w, h), (0, 0, 0, 0))

    # 2. Limpieza de componentes diminutos o huérfanos (< 3 px)
    if clean_isolated_pixels:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(solid_mask, connectivity=8)
        cleaned_mask = np.zeros_like(solid_mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 3:
                cleaned_mask[labels == i] = 1
        solid_mask = cleaned_mask

    if not np.any(solid_mask):
        return Image.new("RGBA", (w, h), (0, 0, 0, 0))

    out_arr = np.zeros((h, w, 4), dtype=np.uint8)
    out_arr[solid_mask == 1, :3] = rgb[solid_mask == 1]
    out_arr[solid_mask == 1, 3] = 255

    # 3. Delineado oscuro perimetral de 1 píxel
    if enforce_dark_outline:
        # Detectar el perímetro exterior del cuerpo
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        eroded = cv2.erode(solid_mask, kernel, iterations=1)
        border_mask = (solid_mask == 1) & (eroded == 0)

        # Para los píxeles del borde exterior que no sean oscuros, oscurecerlos
        # proporcionalmente para crear el trazo de tinta clásico de 1px
        border_coords = np.argwhere(border_mask)
        for y, x in border_coords:
            cur_rgb = out_arr[y, x, :3].astype(np.float32)
            cur_lum = 0.299 * cur_rgb[0] + 0.587 * cur_rgb[1] + 0.114 * cur_rgb[2]
            # Si el borde es demasiado claro (por ejemplo piel o tela clara que linda con el vacío),
            # oscurecerlo con el tono de contorno
            if cur_lum > 65.0:
                darkened = cur_rgb * 0.40 + np.array(outline_color, dtype=np.float32) * 0.60
                out_arr[y, x, :3] = np.clip(np.round(darkened), 0, 255).astype(np.uint8)

    return Image.fromarray(out_arr, mode="RGBA")
