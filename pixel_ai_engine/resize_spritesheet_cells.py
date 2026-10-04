import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARACTERS_DIR = PROJECT_ROOT / "personajes"
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "reparacion" / "backup_spritesheets_antes_ampliacion_20261003"
DEFAULT_PREVIEW_PATH = PROJECT_ROOT / "training_samples" / "sprite_scale_comparison.png"
TARGET_HEIGHT_RATIO = 0.875
MAX_WIDTH_RATIO = 0.90625


def get_cell_coordinates(
    canvas_w: int,
    canvas_h: int,
    row: int,
    col: int,
    total_rows: int,
    total_cols: int,
) -> Tuple[int, int, int, int]:
    x0 = int(round(col * canvas_w / total_cols))
    x1 = int(round((col + 1) * canvas_w / total_cols))
    y0 = int(round(row * canvas_h / total_rows))
    y1 = int(round((row + 1) * canvas_h / total_rows))
    return x0, y0, x1, y1


def foreground_mask(image: Image.Image, alpha_threshold: int = 15, diff_threshold: float = 28.0) -> np.ndarray:
    array = np.array(image.convert("RGBA"))
    alpha = array[:, :, 3]
    if float((alpha < 250).mean()) > 0.02:
        return alpha > alpha_threshold

    rgb = array[:, :, :3].astype(np.float32)
    height, width = alpha.shape
    sample = max(2, min(height, width) // 12)
    corners = np.concatenate(
        [
            rgb[:sample, :sample].reshape(-1, 3),
            rgb[:sample, width - sample :].reshape(-1, 3),
            rgb[height - sample :, :sample].reshape(-1, 3),
            rgb[height - sample :, width - sample :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(corners, axis=0)
    difference = np.sqrt(np.sum((rgb - background) ** 2, axis=-1))
    return difference > diff_threshold


def scale_cell_sprite(cell: Image.Image) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int]]:
    rgba_cell = cell.convert("RGBA")
    array = np.array(rgba_cell)
    mask = foreground_mask(rgba_cell)
    coordinates = np.argwhere(mask)
    if len(coordinates) < 40:
        empty = Image.new("RGBA", rgba_cell.size, (0, 0, 0, 0))
        return empty, (0, 0), (0, 0)

    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0)
    source_width = int(x1 - x0 + 1)
    source_height = int(y1 - y0 + 1)
    if source_width < 4 or source_height < 8:
        empty = Image.new("RGBA", rgba_cell.size, (0, 0, 0, 0))
        return empty, (source_width, source_height), (0, 0)

    clean_array = array.copy()
    clean_array[~mask, 3] = 0
    clean_image = Image.fromarray(clean_array, mode="RGBA")
    foreground = clean_image.crop((x0, y0, x1 + 1, y1 + 1))

    cell_width, cell_height = rgba_cell.size
    target_height = max(1, int(round(cell_height * TARGET_HEIGHT_RATIO)))
    max_width = max(1, int(round(cell_width * MAX_WIDTH_RATIO)))
    scale = min(target_height / source_height, max_width / source_width)
    target_width = max(1, int(round(source_width * scale)))
    target_height = max(1, int(round(source_height * scale)))
    foreground = foreground.resize((target_width, target_height), Image.Resampling.NEAREST)

    result = Image.new("RGBA", rgba_cell.size, (0, 0, 0, 0))
    offset_x = (cell_width - target_width) // 2
    offset_y = (cell_height - target_height) // 2
    result.alpha_composite(foreground, (offset_x, offset_y))
    return result, (source_width, source_height), (target_width, target_height)


def detect_grid(sheet: Image.Image) -> Tuple[int, int]:
    width, height = sheet.size
    return (12, 8) if width / max(1, height) >= 0.45 else (16, 4)


def process_sheet(sheet: Image.Image) -> Tuple[Image.Image, Dict[str, float], List[Tuple[Image.Image, Image.Image]]]:
    rows, cols = detect_grid(sheet)
    width, height = sheet.size
    output = Image.new("RGBA", sheet.size, (0, 0, 0, 0))
    source_heights: List[int] = []
    target_heights: List[int] = []
    examples: List[Tuple[Image.Image, Image.Image]] = []

    for row in range(rows):
        for col in range(cols):
            x0, y0, x1, y1 = get_cell_coordinates(width, height, row, col, rows, cols)
            source_cell = sheet.crop((x0, y0, x1, y1))
            scaled_cell, source_size, target_size = scale_cell_sprite(source_cell)
            output.alpha_composite(scaled_cell, (x0, y0))
            if source_size[1] > 0 and target_size[1] > 0:
                source_heights.append(source_size[1])
                target_heights.append(target_size[1])
                if len(examples) < 1:
                    examples.append((source_cell.copy(), scaled_cell.copy()))

    metrics = {
        "rows": rows,
        "cols": cols,
        "frames": len(source_heights),
        "source_height_median": float(np.median(source_heights)) if source_heights else 0.0,
        "target_height_median": float(np.median(target_heights)) if target_heights else 0.0,
    }
    return output, metrics, examples


def checkerboard(size: Tuple[int, int], square: int = 12) -> Image.Image:
    width, height = size
    image = Image.new("RGBA", size, (35, 39, 48, 255))
    draw = ImageDraw.Draw(image)
    for y in range(0, height, square):
        for x in range(0, width, square):
            if (x // square + y // square) % 2:
                draw.rectangle((x, y, x + square - 1, y + square - 1), fill=(53, 59, 72, 255))
    return image


def save_preview(examples: List[Tuple[str, Image.Image, Image.Image]], preview_path: Path) -> None:
    if not examples:
        return
    panel_size = 192
    header_height = 30
    preview = Image.new("RGBA", (panel_size * 2, (panel_size + header_height) * len(examples)), (20, 22, 28, 255))
    draw = ImageDraw.Draw(preview)
    for index, (label, before, after) in enumerate(examples):
        y = index * (panel_size + header_height)
        draw.text((8, y + 7), f"{label}: antes", fill=(235, 235, 240, 255))
        draw.text((panel_size + 8, y + 7), "después", fill=(196, 181, 253, 255))
        for column, cell in enumerate((before, after)):
            background = checkerboard((panel_size, panel_size))
            fitted = cell.resize((panel_size, panel_size), Image.Resampling.NEAREST)
            background.alpha_composite(fitted)
            preview.alpha_composite(background, (column * panel_size, y + header_height))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview.convert("RGB").save(preview_path, quality=95)


def find_sheets() -> List[Path]:
    return sorted(
        path
        for path in CHARACTERS_DIR.glob("*/*.png")
        if "movimiento" in path.name.lower()
    )


def resize_sheets(backup_dir: Path, apply_changes: bool, preview_path: Path) -> None:
    sheet_paths = find_sheets()
    if not sheet_paths:
        raise RuntimeError(f"No se encontraron hojas de movimiento en {CHARACTERS_DIR}")

    preview_examples: List[Tuple[str, Image.Image, Image.Image]] = []
    for sheet_path in sheet_paths:
        original = Image.open(sheet_path).convert("RGBA")
        resized, metrics, examples = process_sheet(original)
        relative_path = sheet_path.relative_to(PROJECT_ROOT)
        print(
            f"[{relative_path}] {int(metrics['frames'])} frames | "
            f"altura mediana {metrics['source_height_median']:.0f} -> {metrics['target_height_median']:.0f}px"
        )

        if examples and len(preview_examples) < 4:
            preview_examples.append((sheet_path.parent.name, examples[0][0], examples[0][1]))

        if not apply_changes:
            continue

        backup_path = backup_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if not backup_path.exists():
            shutil.copy2(sheet_path, backup_path)

        temp_path = sheet_path.with_suffix(".png.tmp")
        resized.save(temp_path, format="PNG", optimize=True)
        os.replace(temp_path, sheet_path)

    save_preview(preview_examples, preview_path)
    if apply_changes:
        print(f"[OK] Hojas ampliadas. Respaldo: {backup_dir}")
    else:
        print("[DRY RUN] No se modificaron hojas. Usa --apply para guardar los cambios.")
    print(f"[OK] Comparativa visual: {preview_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Amplía cada figura sin salir de su celda de spritesheet.")
    parser.add_argument("--apply", action="store_true", help="Guarda los cambios después de crear respaldos.")
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW_PATH)
    args = parser.parse_args()
    resize_sheets(args.backup_dir.resolve(), args.apply, args.preview.resolve())


if __name__ == "__main__":
    main()
