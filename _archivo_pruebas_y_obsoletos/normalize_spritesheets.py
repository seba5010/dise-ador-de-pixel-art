"""Normaliza las hojas de movimientos al lienzo canónico 724x2172.

Las hojas originales usan cuatro sprites distribuidos sobre cinco unidades
horizontales (centros aproximados en 20%, 40%, 60% y 80% del ancho), no cuatro
cuartos del lienzo. Este script encuentra los valles transparentes entre cada
figura, extrae los 64 sprites sin cortar y los coloca en la geometría de la
plantilla oficial.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CANVAS_WIDTH = 724
CANVAS_HEIGHT = 2172
ROWS = 16
COLS = 4
HORIZONTAL_UNITS = 5


@dataclass
class SheetReport:
    path: str
    original_size: tuple[int, int]
    output_size: tuple[int, int]
    frames: int
    empty_frames: list[int]
    source_border_contacts: list[int]
    output_border_contacts: list[int]
    status: str


def rounded_bounds(length: int, divisions: int) -> list[int]:
    return [round(i * length / divisions) for i in range(divisions + 1)]


def longest_low_run(values: np.ndarray, start: int, end: int) -> int:
    """Escoge el centro del mejor corredor transparente cerca de un separador."""
    start = max(1, start)
    end = min(len(values) - 1, end)
    segment = values[start:end]
    if segment.size == 0:
        return (start + end) // 2

    minimum = segment.min()
    # Admitir píxeles antialias muy débiles y buscar el corredor más ancho.
    threshold = max(minimum, np.percentile(segment, 12))
    low = segment <= threshold
    runs: list[tuple[int, int]] = []
    run_start = None
    for idx, active in enumerate(low):
        if active and run_start is None:
            run_start = idx
        if run_start is not None and (not active or idx == len(low) - 1):
            run_end = idx if active else idx - 1
            runs.append((run_start, run_end))
            run_start = None
    if not runs:
        return start + int(np.argmin(segment))
    a, b = max(runs, key=lambda run: (run[1] - run[0], -segment[run[0]:run[1] + 1].mean()))
    return start + (a + b) // 2


def foreground_runs(projection: np.ndarray, threshold: int, min_width: int) -> list[tuple[int, int]]:
    active = projection > threshold
    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(active) - 1):
            end = index if value else index - 1
            if end - start + 1 >= min_width:
                runs.append((start, end))
            start = None
    return runs


def sheet_column_bounds(alpha: np.ndarray) -> list[int]:
    """Detecta las cuatro bandas reales, aunque estén desplazadas en el lienzo."""
    height, width = alpha.shape
    projection = (alpha > 8).sum(axis=0)
    runs = foreground_runs(projection, max(2, round(height * 0.005)), max(4, round(width * 0.01)))
    if len(runs) != COLS:
        raise ValueError(f"Se esperaban 4 columnas de contenido y se detectaron {len(runs)}")

    centers = [(start + end) / 2 for start, end in runs]
    cuts = [max(0, round(centers[0] - (centers[1] - centers[0]) / 2))]
    for left, right in zip(runs, runs[1:]):
        cuts.append((left[1] + right[0]) // 2)
    cuts.append(min(width, round(centers[-1] + (centers[-1] - centers[-2]) / 2)))
    return cuts


def sheet_row_bounds(alpha: np.ndarray) -> list[int]:
    """Detecta el desplazamiento vertical común sin deformar el paso de 16 filas."""
    height, _ = alpha.shape
    projection = (alpha > 32).sum(axis=1)
    pitch = height / ROWS
    candidates: list[tuple[int, int]] = []
    for offset in range(-round(pitch * 0.48), round(pitch * 0.48) + 1):
        score = sum(int(projection[min(height - 1, max(0, round(row * pitch + offset)))]) for row in range(1, ROWS))
        candidates.append((score, offset))
    best_score = min(score for score, _ in candidates)
    # Si varios corredores son casi equivalentes, usar el de menor desplazamiento.
    near_best = [(abs(offset), offset) for score, offset in candidates if score <= best_score * 1.03 + 1]
    offset = min(near_best)[1]
    return [0] + [round(row * pitch + offset) for row in range(1, ROWS)] + [height]


def transparent_seam(alpha_band: np.ndarray, expected_y: float) -> np.ndarray:
    """Traza de izquierda a derecha una costura de alfa mínimo entre dos filas."""
    height, width = alpha_band.shape
    pitch = height / ROWS
    y0 = max(1, round(expected_y - pitch * 0.30))
    y1 = min(height - 1, round(expected_y + pitch * 0.30))
    values = alpha_band[y0:y1 + 1].astype(np.float32) / 255.0
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    distance = np.abs(ys - expected_y)[:, None] / pitch
    cost = values * 30.0 + distance * 0.08

    band_h = cost.shape[0]
    dp = np.full((band_h, width), np.inf, dtype=np.float32)
    parent = np.zeros((band_h, width), dtype=np.int16)
    dp[:, 0] = cost[:, 0]
    shifts = np.arange(-3, 4, dtype=np.int16)
    for x in range(1, width):
        previous = dp[:, x - 1]
        options = np.full((7, band_h), np.inf, dtype=np.float32)
        for index, shift in enumerate(shifts):
            if shift < 0:
                options[index, -shift:] = previous[:band_h + shift] + abs(int(shift)) * 0.12
            elif shift > 0:
                options[index, :band_h - shift] = previous[shift:] + int(shift) * 0.12
            else:
                options[index] = previous
        choices = np.argmin(options, axis=0)
        parent[:, x] = np.clip(np.arange(band_h) + shifts[choices], 0, band_h - 1)
        dp[:, x] = cost[:, x] + options[choices, np.arange(band_h)]

    path = np.empty(width, dtype=np.int32)
    path[-1] = int(np.argmin(dp[:, -1]))
    for x in range(width - 1, 0, -1):
        path[x - 1] = parent[path[x], x]
    return path + y0


def sheet_seams(alpha: np.ndarray, x_bounds: list[int]) -> list[list[np.ndarray]]:
    height, _ = alpha.shape
    base_bounds = sheet_row_bounds(alpha)
    result: list[list[np.ndarray]] = []
    for col in range(COLS):
        x0, x1 = x_bounds[col], x_bounds[col + 1]
        band = alpha[:, x0:x1]
        seams = [np.zeros(x1 - x0, dtype=np.int32)]
        seams.extend(transparent_seam(band, base_bounds[row]) for row in range(1, ROWS))
        seams.append(np.full(x1 - x0, height, dtype=np.int32))
        result.append(seams)
    return result


def alpha_bbox(frame: Image.Image, threshold: int = 8) -> tuple[int, int, int, int] | None:
    alpha = np.asarray(frame.getchannel("A"))
    points = np.argwhere(alpha > threshold)
    if not len(points):
        return None
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0)
    return int(x0), int(y0), int(x1 + 1), int(y1 + 1)


def fit_sprite(sprite: Image.Image, max_width: int, max_height: int) -> Image.Image:
    width, height = sprite.size
    scale = min(max_width / max(1, width), max_height / max(1, height), 1.0)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if new_size != sprite.size:
        sprite = sprite.resize(new_size, Image.Resampling.NEAREST)
    return sprite


def normalize_sheet(source: Path) -> tuple[Image.Image, SheetReport, Image.Image]:
    image = Image.open(source).convert("RGBA")
    width, height = image.size
    alpha = np.asarray(image.getchannel("A"))
    source_x = sheet_column_bounds(alpha)
    seams = sheet_seams(alpha, source_x)
    target_y = rounded_bounds(CANVAS_HEIGHT, ROWS)
    target_x = rounded_bounds(CANVAS_WIDTH, HORIZONTAL_UNITS)

    output = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0, 0))
    audit = Image.new("RGBA", output.size, (32, 35, 42, 255))
    empty: list[int] = []
    source_contacts: list[int] = []

    for row in range(ROWS):
        ty0, ty1 = target_y[row], target_y[row + 1]
        target_height = ty1 - ty0

        for col in range(COLS):
            frame_index = row * COLS + col
            sx0, sx1 = source_x[col], source_x[col + 1]
            upper = seams[col][row]
            lower = seams[col][row + 1]
            sy0, sy1 = int(upper.min()), int(lower.max())
            raw_array = np.array(image.crop((sx0, sy0, sx1, sy1)), copy=True)
            local_y = np.arange(sy0, sy1)[:, None]
            keep = (local_y >= upper[None, :]) & (local_y < lower[None, :])
            raw_array[:, :, 3] = np.where(keep, raw_array[:, :, 3], 0)
            raw = Image.fromarray(raw_array, mode="RGBA")
            bbox = alpha_bbox(raw)
            if bbox is None:
                empty.append(frame_index)
                continue

            bx0, by0, bx1, by1 = bbox
            if bx0 <= 1 or by0 <= 1 or bx1 >= raw.width - 1 or by1 >= raw.height - 1:
                source_contacts.append(frame_index)
            sprite = raw.crop(bbox)

            tx0, tx1 = target_x[col + 1], target_x[col + 2]
            cell_width = tx1 - tx0
            sprite = fit_sprite(sprite, cell_width - 6, target_height - 8)
            px = tx0 + (cell_width - sprite.width) // 2
            py = ty1 - sprite.height - 4
            output.alpha_composite(sprite, (px, py))
            audit.alpha_composite(sprite, (px, py))

    # Cuadrícula visible únicamente en la imagen de auditoría.
    draw = ImageDraw.Draw(audit)
    for x in target_x[1:5]:
        draw.line((x, 0, x, CANVAS_HEIGHT), fill=(70, 160, 255, 110), width=1)
    for y in target_y:
        draw.line((target_x[1], y, target_x[5], y), fill=(70, 160, 255, 110), width=1)

    output_contacts: list[int] = []
    output_alpha = np.asarray(output.getchannel("A"))
    for row in range(ROWS):
        y0, y1 = target_y[row], target_y[row + 1]
        for col in range(COLS):
            x0, x1 = target_x[col + 1], target_x[col + 2]
            cell = output_alpha[y0:y1, x0:x1]
            if cell.size and (cell[:, 0].max() > 8 or cell[:, -1].max() > 8 or cell[0].max() > 8):
                output_contacts.append(row * COLS + col)

    report = SheetReport(
        path=str(source),
        original_size=(width, height),
        output_size=(CANVAS_WIDTH, CANVAS_HEIGHT),
        frames=ROWS * COLS - len(empty),
        empty_frames=empty,
        source_border_contacts=sorted(set(source_contacts)),
        output_border_contacts=sorted(set(output_contacts)),
        status="ok" if not empty and not output_contacts else "review",
    )
    return output, report, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("personajes"))
    parser.add_argument("--backup", type=Path, default=Path("spritesheets_originales_backup"))
    parser.add_argument("--audit", type=Path, default=Path("normalization_audit"))
    parser.add_argument("--apply", action="store_true", help="Reemplaza las hojas después de respaldarlas")
    args = parser.parse_args()

    sheets = sorted(args.root.rglob("*movimiento*.png"))
    args.audit.mkdir(parents=True, exist_ok=True)
    reports: list[SheetReport] = []

    for source in sheets:
        normalized, report, audit = normalize_sheet(source)
        relative = source.relative_to(args.root)
        audit_path = args.audit / relative
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit.save(audit_path)

        if args.apply:
            backup_path = args.backup / relative
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                shutil.copy2(source, backup_path)
            normalized.save(source)
        reports.append(report)
        print(f"[{report.status.upper():6}] {relative} {report.original_size} -> {report.output_size}")

    report_path = args.audit / "report.json"
    report_path.write_text(json.dumps([asdict(item) for item in reports], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nHojas: {len(reports)} | OK: {sum(r.status == 'ok' for r in reports)} | Revisar: {sum(r.status != 'ok' for r in reports)}")
    print(f"Reporte: {report_path}")
    return 0 if all(r.status == "ok" for r in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
