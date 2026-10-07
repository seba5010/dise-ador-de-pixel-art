"""
Salience-Preserving Pixelization Module (Técnica 12)
Inspirado en Deep Unsupervised Pixelization (Wong et al., SIGGRAPH Asia).

Optimiza la preparación de ilustraciones frontales de alta resolución (como la de Mauricio de ~800px)
hacia la escala canónica del modelo (256x256 / 40px), asegurando que los detalles más distintivos
y prominentes (montura de las gafas, brillo de pupilas, degradado del cabello, suelas)
no se pierdan durante la reducción de escala.
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image
import cv2

from .fractional_downscale import fractional_silhouette_downscale


def prepare_salience_preserved_front(
    raw_front_img: Image.Image,
    target_size: int = 256,
    edge_boost: float = 1.45,
) -> Image.Image:
    """
    Acondiciona la imagen frontal con refuerzo de características salientes.
    1. Aísla el sujeto y extrae mapas de prominencia estructural (filtros Laplaciano y Sobel).
    2. Refuerza el contraste local en líneas finas (gafas, ojos, bordes de ropa).
    3. Aplica downscaling fraccional preservando la silueta y los bordes duros de 1px.
    """
    rgba = raw_front_img.convert("RGBA")
    arr = np.array(rgba)
    h, w, _ = arr.shape

    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    mask = alpha > 25
    if not np.any(mask):
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped_rgb = rgb[y0:y1+1, x0:x1+1]
    cropped_alpha = alpha[y0:y1+1, x0:x1+1]
    cw, ch = cropped_rgb.shape[1], cropped_rgb.shape[0]

    # Calcular mapa de saliencia mediante gradientes espaciales
    gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    grad_norm = np.clip(grad_mag / (grad_mag.max() + 1e-5), 0.0, 1.0)

    # Reforzar líneas oscuras salientes (gafas, ojos, contornos)
    boosted_rgb = cropped_rgb.astype(np.float32)
    is_dark_salient = (grad_norm > 0.35) & (gray < 110)
    boosted_rgb[is_dark_salient] = boosted_rgb[is_dark_salient] * (1.0 / edge_boost)
    boosted_rgb = np.clip(boosted_rgb, 0, 255).astype(np.uint8)

    boosted_rgba = np.dstack([boosted_rgb, cropped_alpha])
    boosted_pil = Image.fromarray(boosted_rgba, mode="RGBA")

    # Escalar proporcionalmente al tamaño seguro de target_size (~85%)
    safe_size = int(round(target_size * 0.85))
    scale = min(safe_size / max(1, cw), safe_size / max(1, ch))
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))

    scaled_pil = fractional_silhouette_downscale(boosted_pil, nw, nh)

    # Centrar en lienzo final
    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    ox = (target_size - nw) // 2
    oy = max(8, target_size - nh - 16)
    canvas.alpha_composite(scaled_pil, (ox, oy))

    return canvas
