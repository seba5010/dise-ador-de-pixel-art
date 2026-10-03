# -*- coding: utf-8 -*-
"""
palette_remap.py
Módulo de Extracción y Remapeo Cuántico de Paleta de Color para Pixel Art.
Rescatado y adaptado de las técnicas de indexación de color de WebUI/Forge.
Garantiza que la IA nunca invente colores ajenos a la identidad frontal del personaje.
"""
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

def extract_character_palette(
    img: Image.Image,
    max_colors: int = 32,
    min_alpha: int = 30
) -> np.ndarray:
    """
    Extrae la paleta exacta de colores dominantes de la imagen frontal del personaje.
    Ignora píxeles transparentes o semitransparentes del fondo.
    Retorna un array (N, 3) con los valores RGB de la paleta.
    """
    rgba = np.array(img.convert("RGBA"))
    mask = rgba[:, :, 3] >= min_alpha
    if not np.any(mask):
        return np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float32)

    pixels = rgba[mask][:, :3]  # (K, 3)

    # Si hay pocos colores únicos (pixel art limpio), tomamos directamente los únicos
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    if len(unique_colors) <= max_colors:
        return unique_colors.astype(np.float32)

    # Si hay ligeras variaciones, usamos K-Means simple para obtener los centroides más representativos
    # Ordenar por frecuencia y tomar los más frecuentes
    sorted_indices = np.argsort(-counts)
    top_colors = unique_colors[sorted_indices[:max_colors]]
    return top_colors.astype(np.float32)

def remap_image_to_palette(
    img: Image.Image,
    palette: np.ndarray,
    min_alpha: int = 25
) -> Image.Image:
    """
    Remapea cada píxel del sprite generado al color más cercano en la paleta del personaje.
    Preserva el canal alfa estricto.
    """
    rgba = np.array(img.convert("RGBA"))
    h, w, _ = rgba.shape
    alpha = rgba[:, :, 3]
    visible_mask = alpha >= min_alpha

    if not np.any(visible_mask) or len(palette) == 0:
        return img

    rgb = rgba[:, :, :3].astype(np.float32)
    flat_rgb = rgb[visible_mask]  # (M, 3)

    # Distancia euclidiana vectorial: (M, 1, 3) - (1, P, 3)
    diff = flat_rgb[:, np.newaxis, :] - palette[np.newaxis, :, :]  # (M, P, 3)
    dist_sq = np.sum(diff ** 2, axis=2)  # (M, P)
    nearest_idx = np.argmin(dist_sq, axis=1)  # (M,)

    snapped_rgb = palette[nearest_idx].astype(np.uint8)

    # Reensamblar imagen
    out_rgba = np.zeros_like(rgba)
    out_rgba[:, :, 3] = np.where(visible_mask, alpha, 0)
    out_rgba[visible_mask, :3] = snapped_rgb

    return Image.fromarray(out_rgba, mode="RGBA")

class PaletteRemapper:
    """Clase utilitaria para almacenar la paleta de un personaje y remapear frames."""
    def __init__(self, front_img: Image.Image, max_colors: int = 32):
        self.palette = extract_character_palette(front_img, max_colors=max_colors)

    def process(self, frame_img: Image.Image) -> Image.Image:
        return remap_image_to_palette(frame_img, self.palette)
