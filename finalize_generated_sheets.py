"""Convierte hojas generadas limpias a la geometría canónica sin costuras curvas."""

from pathlib import Path
import argparse

import numpy as np
from PIL import Image

from normalize_spritesheets import (
    CANVAS_HEIGHT, CANVAS_WIDTH, COLS, ROWS, alpha_bbox, fit_sprite,
    rounded_bounds, sheet_column_bounds,
)


def finalize(source: Path) -> Image.Image:
    image = Image.open(source).convert("RGBA")
    alpha = np.asarray(image.getchannel("A"))
    x_bounds = sheet_column_bounds(alpha)
    y_bounds = rounded_bounds(image.height, ROWS)
    target_x = rounded_bounds(CANVAS_WIDTH, 5)
    target_y = rounded_bounds(CANVAS_HEIGHT, ROWS)
    output = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0, 0))

    for row in range(ROWS):
        for col in range(COLS):
            frame = image.crop((x_bounds[col], y_bounds[row], x_bounds[col + 1], y_bounds[row + 1]))
            bbox = alpha_bbox(frame, 8)
            if bbox is None:
                raise RuntimeError(f"Celda vacía: fila {row + 1}, columna {col + 1}")
            sprite = frame.crop(bbox)
            tx0, tx1 = target_x[col + 1], target_x[col + 2]
            ty0, ty1 = target_y[row], target_y[row + 1]
            sprite = fit_sprite(sprite, tx1 - tx0 - 8, ty1 - ty0 - 8)
            px = tx0 + (tx1 - tx0 - sprite.width) // 2
            py = ty1 - sprite.height - 4
            output.alpha_composite(sprite, (px, py))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for source in sorted(args.root.glob("*movimiento*.png")):
        destination = args.out / source.name
        finalize(source).save(destination)
        print(destination)


if __name__ == "__main__":
    main()
