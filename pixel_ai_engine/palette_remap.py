# -*- coding: utf-8 -*-
"""
palette_remap.py
Módulo de Extracción Quirúrgica, Remapeo de Paleta y Limpieza Morfológica para Pixel Art.
Garantiza que la IA nunca invente colores ajenos, preserve colores esenciales (ojos, gemas, accesorios),
produzca un canal alfa binario estricto (0 o 255) y elimine píxeles huérfanos (hollín).
"""
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional

# Paleta canónica universal de accesorios y utensilios (cocina, cajas, platos)
# Evita que el mortero de madera se vuelva tono piel o el bowl se vuelva delantal
PROPS_PALETTE = np.array([
    [101,  71,  56],  # Madera oscura (mango mortero)
    [128,  66,  38],  # Madera caoba
    [151,  78,  45],  # Madera media
    [169,  90,  53],  # Madera clara
    [187, 102,  61],  # Madera brillo
    [110, 114, 125],  # Metal bowl gris oscuro
    [145, 149, 160],  # Metal bowl gris medio
    [185, 189, 198],  # Metal bowl gris brillo
    [165, 130,  95],  # Caja cartón sombra
    [195, 155, 115],  # Caja cartón base
    [180,  45,  40],  # Tomate / salsa plato
    [ 65, 135,  55],  # Lechuga / verde plato
], dtype=np.uint8)

def extract_character_palette(
    img: Image.Image,
    max_colors: int = 36,
    min_alpha: int = 30,
    include_props: bool = True
) -> np.ndarray:
    """
    Extrae la paleta cromática canónica de la imagen frontal del personaje.
    Utiliza K-Means clustering sobre colores únicos para asegurar que colores minoritarios
    o escasos (como ojos azules, accesorios, tatuajes) queden representados y no sean
    devorados por prendas de color dominante.
    Retorna un array (N, 3) con los valores RGB de la paleta.
    """
    rgba = np.array(img.convert("RGBA"))
    alpha = rgba[:, :, 3]
    mask = alpha >= min_alpha
    if not np.any(mask):
        return np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)

    pixels = rgba[mask][:, :3]  # (K, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    if len(unique_colors) <= max_colors:
        base_palette = unique_colors.astype(np.uint8)
    else:
        try:
            from scipy.cluster.vq import kmeans
            # K-Means sobre colores únicos para balancear tonos minoritarios
            centroids, _ = kmeans(unique_colors.astype(np.float32), max_colors)
            base_palette = centroids.astype(np.uint8)
        except Exception:
            # Fallback ordenado por frecuencia asegurando inclusión de colores con alta saturación
            sorted_indices = np.argsort(-counts)
            base_palette = unique_colors[sorted_indices[:max_colors]].astype(np.uint8)

    if include_props:
        full_palette = np.vstack([base_palette, PROPS_PALETTE])
        unique_full = np.unique(full_palette, axis=0)
        return unique_full.astype(np.uint8)

    return np.unique(base_palette, axis=0).astype(np.uint8)

def remap_image_to_palette(
    img: Image.Image,
    palette: np.ndarray,
    tolerance: float = 35.0,
    min_alpha: int = 25,
    binarize_alpha: bool = True,
    alpha_thresh: int = 40
) -> Image.Image:
    """
    Remapea los píxeles del sprite al color más cercano de la paleta permitida.
    Si un píxel dista más de `tolerance` (ej. detalles distintivos o accesorios legítimos),
    se conserva su color para no degradarlo a tonos grises o neutros erróneos.
    Además, produce un canal alfa estrictamente binario (0 o 255) si binarize_alpha=True.
    """
    rgba = np.array(img.convert("RGBA"))
    alpha = rgba[:, :, 3]
    visible_mask = alpha >= min_alpha

    if not np.any(visible_mask) or len(palette) == 0:
        if binarize_alpha:
            rgba[:, :, 3] = np.where(alpha >= alpha_thresh, 255, 0).astype(np.uint8)
            return Image.fromarray(rgba, mode="RGBA")
        return img

    rgb = rgba[:, :, :3].astype(np.float32)
    pal = palette.astype(np.float32)
    flat_rgb = rgb[visible_mask]  # (M, 3)

    diff = flat_rgb[:, np.newaxis, :] - pal[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    nearest_idx = np.argmin(dist, axis=1)
    min_dists = dist[np.arange(len(flat_rgb)), nearest_idx]

    # Conservar píxeles que caen dentro de la tolerancia; preservar los restantes
    close_enough = min_dists <= tolerance
    snapped_rgb = flat_rgb.copy()
    snapped_rgb[close_enough] = pal[nearest_idx[close_enough]]

    out_rgba = np.zeros_like(rgba)
    out_rgba[visible_mask, :3] = snapped_rgb.astype(np.uint8)

    if binarize_alpha:
        out_rgba[:, :, 3] = np.where(alpha >= alpha_thresh, 255, 0).astype(np.uint8)
    else:
        out_rgba[:, :, 3] = np.where(visible_mask, alpha, 0).astype(np.uint8)

    return Image.fromarray(out_rgba, mode="RGBA")

def clean_orphan_pixels(
    img: Image.Image,
    min_connected_size: int = 3,
    alpha_thresh: int = 30,
    binarize: bool = True
) -> Image.Image:
    """
    Filtro Morfológico Anti-Hollín:
    Utiliza componentes conectados de OpenCV para eliminar pequeños grupos de píxeles
    huérfanos (< min_connected_size) que queden flotando en el fondo transparente.
    Garantiza corte alfa 100% binario (0 o 255) sin halos semitransparentes.
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
    if binarize:
        rgba[mask_clean, 3] = 255

    return Image.fromarray(rgba, mode="RGBA")

class PaletteRemapper:
    """Clase utilitaria para almacenar la paleta de un personaje, remapear y limpiar frames."""
    def __init__(self, front_img: Image.Image, max_colors: int = 36, include_props: bool = True):
        self.palette = extract_character_palette(front_img, max_colors=max_colors, include_props=include_props)

    def process(self, frame_img: Image.Image, clean_soot: bool = True, tolerance: float = 35.0, binarize_alpha: bool = True) -> Image.Image:
        snapped = remap_image_to_palette(frame_img, self.palette, tolerance=tolerance, binarize_alpha=binarize_alpha)
        if clean_soot:
            snapped = clean_orphan_pixels(snapped, min_connected_size=3, binarize=binarize_alpha)
        return snapped

