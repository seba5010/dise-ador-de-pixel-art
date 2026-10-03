# -*- coding: utf-8 -*-
"""
palette_remap.py
Módulo de Extracción, Remapeo Cuántico y Limpieza Morfológica para Pixel Art.
Rescatado y adaptado de las técnicas de indexación de color y OpenCV de WebUI/Forge.
Garantiza que la IA nunca invente colores ajenos ni deje píxeles sueltos (hollín).
"""
import numpy as np
from PIL import Image
import cv2
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

    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    if len(unique_colors) <= max_colors:
        return unique_colors.astype(np.float32)

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

    diff = flat_rgb[:, np.newaxis, :] - palette[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    nearest_idx = np.argmin(dist_sq, axis=1)

    snapped_rgb = palette[nearest_idx].astype(np.uint8)

    out_rgba = np.zeros_like(rgba)
    out_rgba[:, :, 3] = np.where(visible_mask, alpha, 0)
    out_rgba[visible_mask, :3] = snapped_rgb

    return Image.fromarray(out_rgba, mode="RGBA")

def clean_orphan_pixels(
    img: Image.Image,
    min_connected_size: int = 4,
    alpha_thresh: int = 25
) -> Image.Image:
    """
    Filtro Morfologico Anti-Hollin:
    Utiliza componentes conectados de OpenCV para eliminar pequeños grupos de píxeles
    huérfanos (< min_connected_size) que queden flotando en el fondo transparente.
    """
    rgba = np.array(img.convert("RGBA"))
    alpha = rgba[:, :, 3]
    binary = (alpha >= alpha_thresh).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    mask_clean = np.zeros_like(binary, dtype=bool)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_connected_size:
            mask_clean |= (labels == i)

    rgba[~mask_clean, 3] = 0
    return Image.fromarray(rgba, mode="RGBA")

class PaletteRemapper:
    """Clase utilitaria para almacenar la paleta de un personaje, remapear y limpiar frames."""
    def __init__(self, front_img: Image.Image, max_colors: int = 32):
        self.palette = extract_character_palette(front_img, max_colors=max_colors)

    def process(self, frame_img: Image.Image, clean_soot: bool = True) -> Image.Image:
        snapped = remap_image_to_palette(frame_img, self.palette)
        if clean_soot:
            snapped = clean_orphan_pixels(snapped)
        return snapped
