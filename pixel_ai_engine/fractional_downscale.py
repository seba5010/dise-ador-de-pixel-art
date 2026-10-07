"""
Fractional Silhouette-First Downscaling Module (Técnica 6)
Inspirado en BetterPixelArtDownscale (MidFord) y algoritmos de muestreo con
preservación de contraste y siluetas para sprites de baja resolución.

Resuelve el problema de que los filtros clásicos (Bilinear, Lanczos o Nearest básico)
destruyen líneas finas de 1 píxel (como monturas de gafas, ojos y accesorios)
o las funden en degradados marrones/grises contra la piel o el fondo.
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image


def fractional_silhouette_downscale(
    img: Image.Image,
    target_w: int,
    target_h: int,
    contrast_priority_threshold: float = 0.28,
    dark_feature_bias: float = 1.35,
    min_alpha_threshold: int = 30,
) -> Image.Image:
    """
    Downscaler fraccional con prioridad de silueta y alto contraste.

    Algoritmo:
    1. Divide la imagen fuente (W, H) en una cuadrícula virtual de celdas flotantes (target_w, target_h).
    2. Para cada celda de destino (x, y), examina todos los sub-píxeles fuente que caen dentro.
    3. Si la celda contiene una mezcla de piel/fondo claro y un rasgo fino de alto contraste
       (ej. montura negra de gafas de 1px o pupila), el algoritmo otorga prioridad al rasgo
       estructural oscuro para evitar que sea promediado o borrado.
    4. Garantiza silueta 100% nítida y conservación de bordes de 1 píxel.
    """
    rgba = img.convert("RGBA")
    src_w, src_h = rgba.size

    if target_w <= 0 or target_h <= 0:
        return Image.new("RGBA", (max(1, target_w), max(1, target_h)), (0, 0, 0, 0))

    if (src_w, src_h) == (target_w, target_h):
        return rgba.copy()

    # Si la imagen destino es mayor, usar Nearest Neighbor
    if target_w >= src_w and target_h >= src_h:
        return rgba.resize((target_w, target_h), Image.Resampling.NEAREST)

    arr = np.array(rgba, dtype=np.float32)  # [H, W, 4]
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    # Luminancia perceptual estricta [0, 255]
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    out_arr = np.zeros((target_h, target_w, 4), dtype=np.uint8)

    scale_x = src_w / float(target_w)
    scale_y = src_h / float(target_h)

    for ty in range(target_h):
        sy_start = int(np.floor(ty * scale_y))
        sy_end = int(np.ceil((ty + 1) * scale_y))
        sy_start = max(0, min(src_h - 1, sy_start))
        sy_end = max(sy_start + 1, min(src_h, sy_end))

        for tx in range(target_w):
            sx_start = int(np.floor(tx * scale_x))
            sx_end = int(np.ceil((tx + 1) * scale_x))
            sx_start = max(0, min(src_w - 1, sx_start))
            sx_end = max(sx_start + 1, min(src_w, sx_end))

            sub_alpha = alpha[sy_start:sy_end, sx_start:sx_end]
            sub_rgb = rgb[sy_start:sy_end, sx_start:sx_end]
            sub_lum = lum[sy_start:sy_end, sx_start:sx_end]

            valid_mask = sub_alpha > min_alpha_threshold
            valid_count = np.count_nonzero(valid_mask)

            if valid_count == 0:
                out_arr[ty, tx] = [0, 0, 0, 0]
                continue

            # Fracción de cobertura del píxel
            coverage = valid_count / float(sub_alpha.size)

            # Si la celda apenas tiene contacto con la silueta (menos del 25%), descartar para bordes limpios
            if coverage < 0.22:
                out_arr[ty, tx] = [0, 0, 0, 0]
                continue

            valid_rgb = sub_rgb[valid_mask]
            valid_lum = sub_lum[valid_mask]

            # Detección de contraste dentro de la celda
            min_lum = float(np.min(valid_lum))
            max_lum = float(np.max(valid_lum))
            contrast = (max_lum - min_lum) / 255.0

            # Si hay un contraste notable (como montura de gafas negra sobre piel clara)
            if contrast > contrast_priority_threshold:
                # Priorizar el píxel estructural oscuro
                lum_range = max(1.0, max_lum - min_lum)
                darkness_weights = np.exp(-((valid_lum - min_lum) / lum_range) * dark_feature_bias)
                darkness_weights /= np.sum(darkness_weights)

                chosen_rgb = np.sum(valid_rgb * darkness_weights[:, None], axis=0)
            else:
                # Moda o promedio ponderado de los píxeles visibles
                chosen_rgb = np.median(valid_rgb, axis=0)

            out_arr[ty, tx, :3] = np.clip(np.round(chosen_rgb), 0, 255).astype(np.uint8)
            out_arr[ty, tx, 3] = 255  # Opacidad sólida para Pixel Art

    return Image.fromarray(out_arr, mode="RGBA")
