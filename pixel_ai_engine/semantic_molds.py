"""
Semantic Part Maps / DensePose 2D Segmentation Module (Técnica 10)
Inspirado en PISE (Person Image Synthesis and Editing) y DensePose 2D.

Descompone el molde anatómico en zonas corporales semánticas diferenciadas
(Cabeza, Torso, Brazos/Manos, Piernas/Pies, Utensilios/Props), permitiendo a la red
aprender límites anatómicos estrictos y eliminando el desbordamiento de colores entre ropa y piel.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image
import cv2


# Paleta semántica canónica (RGB)
PART_COLORS = {
    "background": (0, 0, 0),
    "head": (255, 60, 60),        # Rojo: Cabeza y rostro
    "torso": (60, 255, 60),       # Verde: Torso y ropa superior
    "arms": (60, 100, 255),       # Azul: Brazos y manos
    "legs": (255, 230, 60),       # Amarillo: Piernas y pantalones
    "feet": (200, 100, 50),       # Naranja: Zapatos y pies
    "props": (255, 60, 255),      # Magenta: Utensilios (bowl, caja, plato)
}


def create_semantic_part_map(
    pose_img: Image.Image,
    row_idx: int = 0,
) -> Image.Image:
    """
    Genera un mapa de partes anatómicas semánticas a partir del molde de pose.
    Utiliza segmentación geométrica vertical y componentes conectados para delimitar
    cabeza, torso, extremidades y accesorios de cocina.
    """
    arr = np.array(pose_img.convert("RGBA"))
    alpha = arr[:, :, 3]
    h, w = alpha.shape

    mask = alpha > 30
    if not np.any(mask):
        return Image.new("RGB", (w, h), (0, 0, 0))

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    total_h = y1 - y0 + 1

    # Segmentación en proporciones canónicas chibi (32% cabeza, 35% torso, 33% piernas/pies)
    head_cutoff = y0 + int(round(total_h * 0.32))
    torso_cutoff = y0 + int(round(total_h * 0.67))
    legs_cutoff = y0 + int(round(total_h * 0.90))

    sem_arr = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Cabeza
    head_mask = mask & (np.arange(h)[:, None] < head_cutoff)
    sem_arr[head_mask] = PART_COLORS["head"]

    # 2. Torso y Brazos
    mid_mask = mask & (np.arange(h)[:, None] >= head_cutoff) & (np.arange(h)[:, None] < torso_cutoff)
    cx = (x0 + x1) // 2
    torso_width_half = max(3, int(round((x1 - x0) * 0.28)))
    is_central = np.abs(np.arange(w)[None, :] - cx) <= torso_width_half

    torso_mask = mid_mask & is_central
    arms_mask = mid_mask & (~is_central)
    sem_arr[torso_mask] = PART_COLORS["torso"]
    sem_arr[arms_mask] = PART_COLORS["arms"]

    # 3. Piernas y Pies
    lower_mask = mask & (np.arange(h)[:, None] >= torso_cutoff)
    legs_mask = lower_mask & (np.arange(h)[:, None] < legs_cutoff)
    feet_mask = lower_mask & (np.arange(h)[:, None] >= legs_cutoff)
    sem_arr[legs_mask] = PART_COLORS["legs"]
    sem_arr[feet_mask] = PART_COLORS["feet"]

    # 4. Detección de Utensilios/Props en filas de acción (filas 8-11 en 8x12)
    if row_idx >= 8:
        # En filas de acción, los píxeles a la altura del pecho que se extienden
        # horizontalmente suelen ser el bowl, caja o plato
        prop_area = mid_mask & (np.arange(h)[:, None] >= head_cutoff + 5) & (np.arange(w)[None, :] > cx + 4)
        if np.any(prop_area):
            sem_arr[prop_area] = PART_COLORS["props"]

    return Image.fromarray(sem_arr, mode="RGB")
