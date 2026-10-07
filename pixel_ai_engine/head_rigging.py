"""
Modular Paper Doll Head-Anchor Composition Module (Técnica 9)
Inspirado en Universal LPC Sprite Sheet Generator y Top Down Sprite Maker (TDSM).

Resuelve el problema de que una red convolucional promedia y difumina los rasgos
faciales en 96 poses diferentes. Al separar el cuerpo animado de la cabeza canónica
direccional, garantiza que los lentes, ojos, peinados y expresiones del personaje
se mantengan 100% nítidos en todas las animaciones.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import cv2

from .palette_remap import extract_character_palette, remap_image_to_palette


class HeadRiggingManager:
    """
    Gestiona la extracción, adaptación direccional y anclaje de cabezas de alta fidelidad.
    """

    def __init__(self):
        pass

    @staticmethod
    def extract_canonical_head(
        front_img: Image.Image,
        head_height_ratio: float = 0.32,
    ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        Extrae la cabeza del personaje frontal con canal alfa recortado.
        Retorna (head_img_pil, (x0, y0, x1, y1)).
        """
        arr = np.array(front_img.convert("RGBA"))
        alpha = arr[:, :, 3]
        mask = alpha > 25

        if not np.any(mask):
            return Image.new("RGBA", front_img.size, (0, 0, 0, 0)), (0, 0, 0, 0)

        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)

        char_h = y1 - y0 + 1
        head_h = max(8, int(round(char_h * head_height_ratio)))

        # Sub-máscara precisa de la cabeza para no incluir hombros de la ilustración
        head_mask = mask.copy()
        head_mask[y0 + head_h:, :] = False
        head_coords = np.argwhere(head_mask)
        if len(head_coords) > 0:
            hy0, hx0 = head_coords.min(axis=0)
            hy1, hx1 = head_coords.max(axis=0)
            head_crop = front_img.crop((hx0, hy0, hx1 + 1, hy1 + 1))
            bounds = (hx0, hy0, hx1, hy1)
        else:
            head_crop = front_img.crop((x0, y0, x1 + 1, y0 + head_h))
            bounds = (x0, y0, x1, y0 + head_h)

        return head_crop, bounds

    @staticmethod
    def get_template_head_anchor(
        pose_template: Image.Image,
    ) -> Optional[Tuple[int, int, int, int, int, int]]:
        """
        Localiza la región de la cabeza en el molde anatómico:
        Retorna (head_x, head_y, head_w, head_h, neck_x, neck_y) o None.
        """
        arr = np.array(pose_template.convert("RGBA"))
        alpha = arr[:, :, 3]
        mask = alpha > 30

        if not np.any(mask):
            return None

        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)

        char_h = y1 - y0 + 1
        head_h = max(6, int(round(char_h * 0.32)))

        # Sub-máscara de la cabeza
        head_mask = mask.copy()
        head_mask[y0 + head_h:, :] = False

        head_coords = np.argwhere(head_mask)
        if len(head_coords) == 0:
            return None

        hy0, hx0 = head_coords.min(axis=0)
        hy1, hx1 = head_coords.max(axis=0)
        hw = hx1 - hx0 + 1
        hh = hy1 - hy0 + 1

        neck_x = (hx0 + hx1) // 2
        neck_y = hy1

        return (hx0, hy0, hw, hh, neck_x, neck_y)

    @classmethod
    def composite_head_onto_frame(
        cls,
        body_frame: Image.Image,
        canonical_head: Image.Image,
        pose_template: Image.Image,
        direction_row: int,
        blend_strength: float = 0.85,
    ) -> Image.Image:
        """
        Monta la cabeza de alta fidelidad adaptada sobre el cuerpo animado.
        """
        bw, bh = body_frame.size
        pw, ph = pose_template.size
        if (pw, ph) != (bw, bh):
            pose_template = pose_template.resize((bw, bh), Image.Resampling.NEAREST)

        anchor = cls.get_template_head_anchor(pose_template)
        if anchor is None:
            return body_frame

        hx0, hy0, hw, hh, neck_x, neck_y = anchor

        # Adaptar cabeza según la dirección
        head_to_paste = canonical_head.copy()

        # Filas 6 y 7 miran a la izquierda -> voltear horizontalmente
        if direction_row in [6, 7]:
            head_to_paste = head_to_paste.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Filas 3, 4, 5 son de espalda -> atenuar rasgos faciales y resaltar cabello
        is_back_view = direction_row in [3, 4, 5]
        if is_back_view:
            arr_h = np.array(head_to_paste.convert("RGBA"))
            hair_sample = arr_h[:max(2, arr_h.shape[0] // 3), :, :3]
            hair_mask = arr_h[:max(2, arr_h.shape[0] // 3), :, 3] > 30
            if np.any(hair_mask):
                hair_color = np.median(hair_sample[hair_mask], axis=0).astype(np.uint8)
                arr_h[arr_h[:, :, 3] > 30, :3] = hair_color
                head_to_paste = Image.fromarray(arr_h, mode="RGBA")

        # Escalar la cabeza para coincidir con las dimensiones anatómicas
        target_hw = max(4, int(round(hw * 1.04)))
        target_hh = max(4, int(round(hh * 1.02)))
        head_scaled = head_to_paste.resize((target_hw, target_hh), Image.Resampling.NEAREST)

        # Posicionar sobre el cuello
        paste_x = neck_x - target_hw // 2
        paste_y = max(0, neck_y - target_hh + 2)

        composite = body_frame.convert("RGBA").copy()
        composite.alpha_composite(head_scaled, (paste_x, paste_y))

        return composite
