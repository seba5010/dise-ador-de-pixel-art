"""
Single-Cell Focused Pipeline Module (Técnica 5)
Inspirado en arquitecturas Pix2Pix clásicas celda por celda y LPC Sprite Generator.

En lugar de diluir la capacidad de la red (15M de parámetros) sobre un lienzo gigante
de 1024x1536 con 96 personajes diminutos de 40px, este módulo estructura el procesamiento
frame a frame con resolución focal máxima (256x256 centrado), concentrando toda la potencia
convolucional en los rasgos individuales de cada pose antes del ensamblado final.
"""

from typing import List, Tuple, Optional, Callable
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from .config import (
    MODEL_RESOLUTION,
    CELL_WIDTH,
    CELL_HEIGHT,
    GRID_ROWS,
    GRID_COLS,
    TOTAL_FRAMES,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    DEVICE,
    USE_AMP
)
from .dataset import place_in_cell, get_cell_coordinates


class SingleCellCoordinator:
    """
    Coordina la generación y ensamblado frame por frame con zoom focal.
    """

    def __init__(
        self,
        rows: int = GRID_ROWS,
        cols: int = GRID_COLS,
        cell_w: int = CELL_WIDTH,
        cell_h: int = CELL_HEIGHT,
        canvas_w: int = CANVAS_WIDTH,
        canvas_h: int = CANVAS_HEIGHT,
    ):
        self.rows = rows
        self.cols = cols
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.total_frames = rows * cols

    def assemble_cells_into_sheet(
        self,
        frame_images: List[Image.Image],
    ) -> Image.Image:
        """
        Ensambla una lista de frames individuales generados en la cuadrícula canónica del spritesheet.
        """
        canvas = Image.new("RGBA", (self.canvas_w, self.canvas_h), (0, 0, 0, 0))

        for idx, frame_img in enumerate(frame_images[:self.total_frames]):
            row = idx // self.cols
            col = idx % self.cols

            x0, y0, x1, y1 = get_cell_coordinates(
                self.canvas_w, self.canvas_h, row, col, self.rows, self.cols
            )
            target_w = x1 - x0
            target_h = y1 - y0

            # Centrar el sprite en la celda
            cell_sprite = place_in_cell(frame_img, cell_w=target_w, cell_h=target_h)
            canvas.alpha_composite(cell_sprite, (x0, y0))

        return canvas
