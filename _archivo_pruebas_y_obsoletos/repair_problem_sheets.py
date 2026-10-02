"""Repara las 36 hojas indicadas sin sobrescribir las fuentes maestras."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from normalize_spritesheets import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    COLS,
    ROWS,
    alpha_bbox,
    fit_sprite,
    rounded_bounds,
    sheet_column_bounds,
)


CHARACTERS = [
    "alex", "carlos", "conny", "diego serena", "diego_vallenar", "duvan",
    "erin", "jorge", "juan", "mario", "maty hermano", "millaray",
]
SOURCE_ROOT = Path("personajes")
OUTPUT_ROOT = Path("reorganizadas_revision")


def instance_markers(alpha: np.ndarray, x_bounds: list[int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Crea una semilla interior para cada uno de los 64 sprites."""
    height, width = alpha.shape
    solid = (alpha > 80).astype(np.uint8)
    distance = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    markers = np.zeros((height, width), dtype=np.int32)
    markers[alpha <= 6] = 1  # Fondo seguro y transparente.
    y_bounds = rounded_bounds(height, ROWS)
    seeds: list[tuple[int, int]] = []

    for row in range(ROWS):
        nominal_y0, nominal_y1 = y_bounds[row], y_bounds[row + 1]
        # Mantener la búsqueda dentro de su fila evita que dos posiciones usen
        # como semilla el mismo personaje cuando las siluetas se solapan.
        search_y0 = nominal_y0
        search_y1 = nominal_y1
        for col in range(COLS):
            x0, x1 = x_bounds[col], x_bounds[col + 1]
            inset = max(1, round((x1 - x0) * 0.05))
            region = distance[search_y0:search_y1, x0 + inset:x1 - inset]
            if not region.size or float(region.max()) <= 0:
                seeds.append((-1, -1))
                continue
            local_y, local_x = np.unravel_index(int(np.argmax(region)), region.shape)
            sy = search_y0 + int(local_y)
            sx = x0 + inset + int(local_x)
            label = 2 + row * COLS + col
            radius = max(2, min(5, int(distance[sy, sx] * 0.35)))
            cv2.circle(markers, (sx, sy), radius, int(label), -1)
            seeds.append((sx, sy))
    return markers, seeds


def split_instances(image: Image.Image) -> tuple[list[Image.Image | None], list[int]]:
    rgba = np.array(image.convert("RGBA"))
    alpha = rgba[:, :, 3]
    x_bounds = sheet_column_bounds(alpha)
    markers, seeds = instance_markers(alpha, x_bounds)

    # La transparencia crea los bordes principales; el color ayuda en contactos reales.
    composite = rgba[:, :, :3].copy()
    composite[alpha <= 6] = 0
    labels = cv2.watershed(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR), markers)

    frames: list[Image.Image | None] = []
    failed: list[int] = []
    for index in range(ROWS * COLS):
        label = index + 2
        mask = labels == label
        if mask.sum() < 40:
            frames.append(None)
            failed.append(index)
            continue
        isolated = rgba.copy()
        isolated[:, :, 3] = np.where(mask, alpha, 0)
        frame = Image.fromarray(isolated, "RGBA")
        bbox = alpha_bbox(frame, threshold=6)
        if bbox is None:
            frames.append(None)
            failed.append(index)
            continue
        frames.append(frame.crop(bbox))
    return frames, failed


def assemble(frames: list[Image.Image | None]) -> Image.Image:
    output = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0, 0))
    target_y = rounded_bounds(CANVAS_HEIGHT, ROWS)
    target_x = rounded_bounds(CANVAS_WIDTH, 5)
    for index, sprite in enumerate(frames):
        if sprite is None:
            continue
        row, col = divmod(index, COLS)
        y0, y1 = target_y[row], target_y[row + 1]
        x0, x1 = target_x[col + 1], target_x[col + 2]
        sprite = fit_sprite(sprite, x1 - x0 - 8, y1 - y0 - 8)
        px = x0 + (x1 - x0 - sprite.width) // 2
        py = y1 - sprite.height - 4
        output.alpha_composite(sprite, (px, py))
    return output


def preview(sheet: Image.Image) -> Image.Image:
    bg = Image.new("RGBA", sheet.size, (32, 35, 42, 255))
    bg.alpha_composite(sheet)
    draw = ImageDraw.Draw(bg)
    xs = rounded_bounds(CANVAS_WIDTH, 5)
    ys = rounded_bounds(CANVAS_HEIGHT, ROWS)
    for x in xs[1:5]:
        draw.line((x, 0, x, CANVAS_HEIGHT), fill=(0, 170, 255, 120), width=1)
    for y in ys:
        draw.line((xs[1], y, xs[5], y), fill=(0, 170, 255, 120), width=1)
    return bg.convert("RGB")


def main() -> int:
    reports = []
    for character in CHARACTERS:
        sources = sorted((SOURCE_ROOT / character).glob("*movimiento*.png"))
        for source in sources:
            frames, failed = split_instances(Image.open(source))
            sheet = assemble(frames)
            destination_dir = OUTPUT_ROOT / character
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / source.name
            sheet.save(destination)
            preview(sheet).save(destination.with_name(f"{destination.stem}_revision.jpg"), quality=92)
            status = "OK" if not failed else "REVISAR"
            print(f"[{status:7}] {source} -> {destination} | recuperados={64-len(failed)}/64")
            reports.append({
                "source": str(source),
                "output": str(destination),
                "recovered": 64 - len(failed),
                "failed_frames": failed,
                "status": status.lower(),
            })
    OUTPUT_ROOT.mkdir(exist_ok=True)
    (OUTPUT_ROOT / "repair_report.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0 if all(not item["failed_frames"] for item in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
