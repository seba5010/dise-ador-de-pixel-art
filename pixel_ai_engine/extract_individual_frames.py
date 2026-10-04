"""Extract and organize every character spritesheet into individual frames."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from importlib import import_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARACTERS_DIR = PROJECT_ROOT / "personajes"
OUTPUT_DIR = PROJECT_ROOT / "dataset_frames_individuales"
PREVIEW_DIR = PROJECT_ROOT / "training_samples" / "frame_extraction_previews"
MOLD_PREVIEW_DIR = PROJECT_ROOT / "training_samples" / "mold_extraction_previews"
MOLD_BACKUP_DIR = PROJECT_ROOT / "reparacion" / "backup_moldes_antes_ampliacion_20261003"
TARGET_SIZE = 256
SAFE_SIZE = 230
CHARACTER_SAFE_SIZE = 220
ALPHA_THRESHOLD = 15
ERIN_WHITE_INCOMPLETE_ROWS = (1, 3, 5, 7)
ERIN_WHITE_UPPER_HEIGHT_RATIO = 0.84
ERIN_DONOR_LOWER_START_RATIO = 0.72
ERIN_BACK_DONOR_BOTTOM_RATIO = 0.955
ERIN_NORMAL_BOTTOM_INTRUSION_SLOTS = tuple(range(5, 13))
NICO_DONOR_LOWER_START_RATIO = 0.80
NICO_BODY_REPAIR_SPECS: Dict[int, Tuple[int, float]] = {
    **{33 + column: (1 + column, 0.82) for column in range(4)},
    **{37 + column: (9 + column, 0.80) for column in range(4)},
    **{41 + column: (17 + column, 0.85) for column in range(4)},
    **{45 + column: (25 + column, 0.84) for column in range(4)},
    49: (1, 0.88),
    51: (3, 0.88),
    52: (4, 0.88),
}
SEBAS_DONOR_LOWER_START_RATIO = 0.90
SEBAS_UPPER_HEIGHT_RATIO = 0.92
SEBAS_FEET_REPAIR_SPECS: Dict[int, int] = {
    **{33 + column: 1 + column for column in range(4)},
    **{37 + column: 9 + column for column in range(4)},
    **{41 + column: 17 + column for column in range(4)},
}
MOLD_SAFE_SIZE = 220
MOLD_CELL_HEIGHT_RATIO = 0.86
MOLD_CELL_WIDTH_RATIO = 0.88

MOLD_SHEETS = (
    (PROJECT_ROOT / "plantilla_16x4.png", 16, 4, "molde_16x4_64_poses"),
    (PROJECT_ROOT / "plantilla de los spritesheets.png", 12, 8, "molde_8x12_96_poses"),
)

BASE_CHARACTERS = {"alex", "amaro", "belial", "conny", "dana"}

VARIANT_NAMES = {
    "rnormal": "ropa_normal",
    "rbchef": "ropa_blanca_chef",
    "rnchef": "ropa_negra_chef",
}

# Several uploaded sheets were intentionally trimmed and no longer use the
# original 8x12 or 4x16 aspect ratio. Their real grids must be explicit so a
# character is never split using the wrong row or column count.
GRID_OVERRIDES: Dict[str, Tuple[int, int]] = {
    "andrea/movimientos_rbchef.png": (12, 4),
    "andrea/movimientos_rnormal.png": (11, 8),
    "andres_arica/movimientos_rbchef.png": (11, 4),
    "andres_arica/movimientos_rnchef.png": (13, 4),
    "andres_arica/movimientos_rnormal.png": (10, 4),
    "bastian/movimientos_rbchef.png": (13, 4),
    "bastian/movimientos_rnchef.png": (13, 4),
    "diego serena/movimientos_rnormal.png": (9, 4),
    "diego_vallenar/movimientos_rnormal.png": (8, 4),
    "erin/movimientos_rbchef.png": (8, 4),
    "erin/movimientos_rnormal.png": (16, 4),
    "mario/movimientos_rnormal (2).png": (8, 4),
    "maty hermano/movimientos_rnormal.png": (16, 4),
    "nico/movimientos_rnormal.png": (15, 4),
    "tori/movimientos.png": (15, 4),
    "zack/movimientos_rbchef.png": (9, 8),
    "zack/movimientos_rnchef.png": (9, 8),
    "zack/movimientos_rnormal.png": (9, 8),
}

UNIFORM_GRID_SHEETS = {
    "erin/movimientos_rnormal.png",
}

EXCLUDED_FRAME_SLOTS: Dict[str, set[int]] = {
    "jorge/movimientos_rnormal.png": {64},
}

DUVAN_WHITE_SHEET = "duvan/movimientos_rbchef.png"
DUVAN_WHITE_COLUMN_EDGES = (0, 201, 363, 533, 724)
DUVAN_WHITE_ROW_EDGES = (
    0, 172, 319, 477, 626, 769, 918, 1073, 1226,
    1376, 1532, 1679, 1828, 1942, 2059, 2172, 2172,
)
DUVAN_WHITE_OVERLAP_START = 1820


@dataclass(frozen=True)
class SheetSpec:
    character: str
    variant: str
    sheet_path: Path
    front_path: Optional[Path]
    rows: int
    cols: int


def relative_sheet_key(path: Path) -> str:
    return path.relative_to(CHARACTERS_DIR).as_posix().lower()


def get_cell_coordinates(
    width: int,
    height: int,
    row: int,
    col: int,
    rows: int,
    cols: int,
) -> Tuple[int, int, int, int]:
    x0 = int(round(col * width / cols))
    x1 = int(round((col + 1) * width / cols))
    y0 = int(round(row * height / rows))
    y1 = int(round((row + 1) * height / rows))
    return x0, y0, x1, y1


def _contiguous_bands(values: np.ndarray) -> List[List[int]]:
    indices = np.flatnonzero(values)
    if len(indices) == 0:
        return []
    bands: List[List[int]] = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index - previous > 1:
            bands.append([start, previous])
            start = index
        previous = index
    bands.append([start, previous])
    return bands


def detect_column_edges(sheet: Image.Image, rows: int, cols: int) -> List[int]:
    """Find transparent valleys between columns, including shifted/cropped grids."""
    alpha = np.array(sheet.convert("RGBA"))[:, :, 3] > 128
    height, width = alpha.shape
    row_centers: List[List[float]] = []

    for row in range(rows):
        y0 = int(round(row * height / rows))
        y1 = int(round((row + 1) * height / rows))
        bands = _contiguous_bands(np.any(alpha[y0:y1], axis=0))
        while len(bands) > cols:
            gaps = [bands[index + 1][0] - bands[index][1] - 1 for index in range(len(bands) - 1)]
            merge_index = int(np.argmin(gaps))
            bands[merge_index : merge_index + 2] = [
                [bands[merge_index][0], bands[merge_index + 1][1]]
            ]
        if len(bands) == cols:
            row_centers.append([(start + end) / 2.0 for start, end in bands])

    if not row_centers:
        return [int(round(column * width / cols)) for column in range(cols + 1)]

    centers = np.median(np.asarray(row_centers), axis=0)
    projection = np.count_nonzero(alpha, axis=0)
    edges = [0]
    for column in range(cols - 1):
        midpoint = (centers[column] + centers[column + 1]) / 2.0
        spacing = max(4.0, centers[column + 1] - centers[column])
        radius = max(2, int(round(spacing * 0.12)))
        start = max(edges[-1] + 1, int(round(midpoint)) - radius)
        end = min(width - 1, int(round(midpoint)) + radius)
        candidates = np.arange(start, end + 1)
        costs = projection[candidates]
        minimum_cost = costs.min()
        best = candidates[costs == minimum_cost]
        boundary = int(best[np.argmin(np.abs(best - midpoint))])
        edges.append(boundary)
    edges.append(width)
    return edges


def detect_row_edges(
    sheet: Image.Image,
    rows: int,
    column_edges: List[int],
) -> List[int]:
    """Find horizontal valleys so tall poses never leak into adjacent frames."""
    alpha_channel = np.array(sheet.convert("RGBA"))[:, :, 3]
    height, _ = alpha_channel.shape
    row_bands: List[List[List[int]]] = []

    for threshold in (250, 245, 220, 180, 128):
        row_bands.clear()
        alpha = alpha_channel > threshold
        for column in range(len(column_edges) - 1):
            x0 = column_edges[column]
            x1 = column_edges[column + 1]
            bands = _contiguous_bands(np.any(alpha[:, x0:x1], axis=1))
            bands = [band for band in bands if band[1] - band[0] + 1 >= 8]
            if len(bands) == rows:
                row_bands.append(bands)
        if len(row_bands) >= max(1, (len(column_edges) - 1) // 2):
            break

    if not row_bands:
        return [int(round(row * height / rows)) for row in range(rows + 1)]

    band_array = np.asarray(row_bands)
    starts = np.median(band_array[:, :, 0], axis=0)
    ends = np.median(band_array[:, :, 1], axis=0)
    projection = np.count_nonzero(alpha_channel > 128, axis=1)
    edges = [0]
    for row in range(rows - 1):
        midpoint = (ends[row] + starts[row + 1]) / 2.0
        radius = 4
        start = max(edges[-1] + 1, int(round(midpoint)) - radius)
        end = min(height - 1, int(round(midpoint)) + radius)
        candidates = np.arange(start, end + 1)
        costs = projection[candidates]
        minimum_cost = costs.min()
        best = candidates[costs == minimum_cost]
        boundary = int(best[np.argmin(np.abs(best - midpoint))])
        edges.append(boundary)
    edges.append(height)
    return edges


def detect_grid(sheet_path: Path, image: Image.Image) -> Tuple[int, int]:
    override = GRID_OVERRIDES.get(relative_sheet_key(sheet_path))
    if override:
        return override
    width, height = image.size
    return (12, 8) if width / max(1, height) >= 0.45 else (16, 4)


def infer_variant(sheet_path: Path) -> str:
    name = sheet_path.name.lower()
    for variant in ("rbchef", "rnchef", "rnormal"):
        if variant in name:
            return variant
    return "rnormal"


def is_sheet(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".png" and any(
        token in name for token in ("movimiento", "spritesheet", "sprite_sheet")
    )


def find_front(character_dir: Path, variant: str) -> Optional[Path]:
    candidates = [path for path in character_dir.glob("*.png") if not is_sheet(path)]
    aliases = {
        "rnormal": ("rnormal", "frente"),
        "rbchef": ("rbchef", "rbnormal"),
        "rnchef": ("rnchef",),
    }[variant]
    for alias in aliases:
        match = next((path for path in candidates if alias in path.name.lower()), None)
        if match:
            return match
    if variant == "rnormal" and len(candidates) == 1:
        return candidates[0]
    return None


def discover_sheets(base_only: bool = False) -> List[SheetSpec]:
    specs: List[SheetSpec] = []
    for character_dir in sorted(
        (path for path in CHARACTERS_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    ):
        if base_only and character_dir.name.lower() not in BASE_CHARACTERS:
            continue
        for sheet_path in sorted(character_dir.glob("*.png"), key=lambda path: path.name.lower()):
            if not is_sheet(sheet_path):
                continue
            with Image.open(sheet_path) as source:
                rows, cols = detect_grid(sheet_path, source)
            variant = infer_variant(sheet_path)
            specs.append(
                SheetSpec(
                    character=character_dir.name,
                    variant=variant,
                    sheet_path=sheet_path,
                    front_path=find_front(character_dir, variant),
                    rows=rows,
                    cols=cols,
                )
            )
    return specs


def foreground_mask(image: Image.Image, alpha_threshold: int = ALPHA_THRESHOLD) -> np.ndarray:
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
    return difference > 26.0


def normalize_frame(
    image: Image.Image,
    target_size: int = TARGET_SIZE,
    portrait: bool = False,
    allow_upscale: bool = False,
    center_vertical: bool = False,
    safe_size: int = SAFE_SIZE,
) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]:
    rgba = image.convert("RGBA")
    array = np.array(rgba)
    mask = foreground_mask(rgba)
    coordinates = np.argwhere(mask)
    if len(coordinates) < 40:
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0)), None

    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0)
    clean_array = array.copy()
    clean_array[~mask, 3] = 0
    foreground = Image.fromarray(clean_array, mode="RGBA").crop((x0, y0, x1 + 1, y1 + 1))
    foreground_width, foreground_height = foreground.size

    if foreground_width > safe_size or foreground_height > safe_size or portrait or allow_upscale:
        maximum_scale = float("inf") if allow_upscale else 1.0
        scale = min(safe_size / foreground_width, safe_size / foreground_height, maximum_scale)
        target_width = max(1, int(round(foreground_width * scale)))
        target_height = max(1, int(round(foreground_height * scale)))
        if (target_width, target_height) != foreground.size:
            resampling = Image.Resampling.LANCZOS if portrait else Image.Resampling.NEAREST
            foreground = foreground.resize((target_width, target_height), resampling)
            foreground_width, foreground_height = foreground.size

    output = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    offset_x = (target_size - foreground_width) // 2
    if center_vertical:
        offset_y = (target_size - foreground_height) // 2
    else:
        offset_y = max(4, target_size - foreground_height - 12)
    output.alpha_composite(foreground, (offset_x, offset_y))
    return output, (int(x0), int(y0), int(x1), int(y1))


def clean_mold_cell(cell: Image.Image) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]:
    rgba = cell.convert("RGBA")
    array = np.array(rgba)
    mask = foreground_mask(rgba)
    border = max(2, min(cell.size) // 40)
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False
    coordinates = np.argwhere(mask)
    if len(coordinates) < 20:
        return Image.new("RGBA", cell.size, (0, 0, 0, 0)), None

    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0)
    clean_array = array.copy()
    clean_array[~mask, 3] = 0
    foreground = Image.fromarray(clean_array, mode="RGBA").crop((x0, y0, x1 + 1, y1 + 1))
    foreground_width, foreground_height = foreground.size
    cell_width, cell_height = cell.size
    target_height = max(1, int(round(cell_height * MOLD_CELL_HEIGHT_RATIO)))
    target_width_limit = max(1, int(round(cell_width * MOLD_CELL_WIDTH_RATIO)))
    scale = min(target_height / foreground_height, target_width_limit / foreground_width)
    target_width = max(1, int(round(foreground_width * scale)))
    target_height = max(1, int(round(foreground_height * scale)))
    foreground = foreground.resize((target_width, target_height), Image.Resampling.NEAREST)

    output = Image.new("RGBA", cell.size, (0, 0, 0, 0))
    offset_x = (cell_width - target_width) // 2
    offset_y = (cell_height - target_height) // 2
    output.alpha_composite(foreground, (offset_x, offset_y))
    return output, (int(x0), int(y0), int(x1), int(y1))


def backup_mold_sheet(sheet_path: Path) -> Path:
    MOLD_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = MOLD_BACKUP_DIR / sheet_path.name
    if not backup_path.exists():
        shutil.copy2(sheet_path, backup_path)
    return backup_path


def process_mold_sheet(sheet_path: Path, rows: int, cols: int) -> Image.Image:
    backup_mold_sheet(sheet_path)
    with Image.open(sheet_path) as source:
        sheet = source.convert("RGBA")
    width, height = sheet.size
    output = Image.new("RGBA", sheet.size, (0, 0, 0, 0))
    for row in range(rows):
        for col in range(cols):
            x0, y0, x1, y1 = get_cell_coordinates(width, height, row, col, rows, cols)
            scaled, _ = clean_mold_cell(sheet.crop((x0, y0, x1, y1)))
            output.alpha_composite(scaled, (x0, y0))
    output.save(sheet_path)
    return output


def extract_mold_frames(
    sheet: Image.Image,
    rows: int,
    cols: int,
    destination: Path,
) -> Dict[int, Image.Image]:
    destination.mkdir(parents=True, exist_ok=True)
    for old_frame in destination.glob("pose_*.png"):
        old_frame.unlink()

    width, height = sheet.size
    frames: Dict[int, Image.Image] = {}
    for row in range(rows):
        for col in range(cols):
            slot = row * cols + col
            x0, y0, x1, y1 = get_cell_coordinates(width, height, row, col, rows, cols)
            frame, bounds = normalize_frame(
                sheet.crop((x0, y0, x1, y1)),
                allow_upscale=True,
                center_vertical=True,
                safe_size=MOLD_SAFE_SIZE,
            )
            if bounds is None:
                continue
            filename = f"pose_{slot + 1:03d}_r{row + 1:02d}_c{col + 1:02d}.png"
            frame.save(destination / filename)
            frames[slot] = frame
    return frames


def save_mold_preview(
    label: str,
    rows: int,
    cols: int,
    frames: Dict[int, Image.Image],
) -> Path:
    tile_size = 96
    header_height = 34
    preview = Image.new(
        "RGBA",
        (cols * tile_size, header_height + rows * tile_size),
        (20, 22, 28, 255),
    )
    draw = ImageDraw.Draw(preview)
    draw.text((8, 10), f"{label} - ampliado y centrado", fill=(225, 230, 240, 255))
    for row in range(rows):
        for col in range(cols):
            slot = row * cols + col
            tile = checkerboard((tile_size, tile_size))
            frame = frames.get(slot)
            if frame:
                tile.alpha_composite(frame.resize((tile_size, tile_size), Image.Resampling.NEAREST))
            preview.alpha_composite(tile, (col * tile_size, header_height + row * tile_size))
    MOLD_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = MOLD_PREVIEW_DIR / f"{label}.png"
    preview.save(preview_path)
    return preview_path


def extract_molds() -> Dict[str, object]:
    results: List[Dict[str, object]] = []
    mold_root = OUTPUT_DIR / "00_MOLDES_POSES"
    for sheet_path, rows, cols, folder_name in MOLD_SHEETS:
        if not sheet_path.exists():
            continue
        sheet = process_mold_sheet(sheet_path, rows, cols)
        destination = mold_root / folder_name
        frames = extract_mold_frames(sheet, rows, cols, destination)
        preview_path = save_mold_preview(folder_name, rows, cols, frames)
        results.append(
            {
                "sheet": str(sheet_path.relative_to(PROJECT_ROOT)),
                "grid": [rows, cols],
                "frames": len(frames),
                "folder": str(destination.relative_to(PROJECT_ROOT)),
                "preview": str(preview_path.relative_to(PROJECT_ROOT)),
            }
        )
        print(f"[OK] {folder_name}: {len(frames)} moldes ampliados y centrados")

    summary = {
        "templates": len(results),
        "frames": sum(int(item["frames"]) for item in results),
        "target_size": [TARGET_SIZE, TARGET_SIZE],
        "target_visible_size": MOLD_SAFE_SIZE,
        "backup": str(MOLD_BACKUP_DIR.relative_to(PROJECT_ROOT)),
        "results": results,
    }
    summary_path = mold_root / "resumen_moldes.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Resumen de moldes: {summary_path}")
    return summary


def source_border_sides(
    bounds: Tuple[int, int, int, int],
    cell_width: int,
    cell_height: int,
) -> List[str]:
    x0, y0, x1, y1 = bounds
    sides: List[str] = []
    if x0 <= 1:
        sides.append("left")
    if y0 <= 1:
        sides.append("top")
    if x1 >= cell_width - 2:
        sides.append("right")
    if y1 >= cell_height - 2:
        sides.append("bottom")
    return sides


def extract_duvan_white_overlaps(
    sheet: Image.Image,
    column_edges: Sequence[int],
) -> Dict[Tuple[int, int], Image.Image]:
    """Separate Duvan's three overlapping final poses in each source column."""
    try:
        ndimage: Any = import_module("scipy.ndimage")
    except ImportError as error:
        raise RuntimeError(
            "La extraccion corregida de Duvan requiere scipy en este entorno."
        ) from error

    rgba = np.array(sheet.convert("RGBA"))
    alpha = rgba[:, :, 3]
    frames: Dict[Tuple[int, int], Image.Image] = {}
    structure = np.ones((3, 3), dtype=bool)

    for column in range(4):
        x0, x1 = column_edges[column : column + 2]
        alpha_region = alpha[DUVAN_WHITE_OVERLAP_START:, x0:x1]
        eroded: Any = ndimage.binary_erosion(
            alpha_region > 128,
            structure=np.ones((15, 15), dtype=bool),
        )
        label_result: Any = ndimage.label(eroded, structure=structure)
        seed_labels: Any = label_result[0]
        seeds: List[Tuple[int, int]] = []
        objects: Any = ndimage.find_objects(seed_labels)
        for seed_id, bounds in enumerate(objects, start=1):
            if bounds is None:
                continue
            area = int((seed_labels[bounds] == seed_id).sum())
            if area >= 500:
                seeds.append((seed_id, DUVAN_WHITE_OVERLAP_START + bounds[0].start))
        seeds.sort(key=lambda item: item[1])
        if len(seeds) != 3:
            raise RuntimeError(
                f"Se esperaban 3 poses solapadas en columna {column + 1}; "
                f"se detectaron {len(seeds)}. No se regeneraron los frames."
            )

        seed_ids = [seed_id for seed_id, _ in seeds]
        distance_result: Any = ndimage.distance_transform_edt(
            ~np.isin(seed_labels, seed_ids),
            return_indices=True,
        )
        nearest: Any = distance_result[1]
        owners: Any = seed_labels[tuple(nearest)]
        source_region = rgba[
            DUVAN_WHITE_OVERLAP_START:, x0:x1
        ].copy()
        for row_offset, (seed_id, _) in enumerate(seeds):
            mask = (alpha_region > ALPHA_THRESHOLD) & (owners == seed_id)
            components_result: Any = ndimage.label(mask, structure=structure)
            components: Any = components_result[0]
            component_count = int(components_result[1])
            component_sizes = np.bincount(components.ravel())
            largest_size = int(component_sizes[1:].max()) if component_count else 0
            keep_threshold = max(40, int(largest_size * 0.05))
            keep_components = np.flatnonzero(component_sizes >= keep_threshold)
            keep_components = keep_components[keep_components != 0]
            mask = np.isin(components, keep_components)
            ys, xs = np.where(mask)
            if len(ys) < 40:
                raise RuntimeError(
                    f"La pose {row_offset + 13}, columna {column + 1} quedo vacia."
                )

            frame_array = source_region.copy()
            frame_array[~mask] = 0
            frame = Image.fromarray(frame_array, mode="RGBA").crop(
                (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            )
            frames[(row_offset + 12, column)] = frame

    return frames


def safe_output_directory(spec: SheetSpec, output_base: Path) -> Path:
    variant_name = VARIANT_NAMES[spec.variant]
    character_name = re.sub(r"[^\w .-]+", "_", spec.character, flags=re.UNICODE).strip()
    return output_base / character_name.upper() / f"{character_name}_{variant_name}"


def clear_generated_variant(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if (
            path.name.startswith("frame_") and path.suffix.lower() == ".png"
        ) or path.name in {"00_frontal_identidad.png", "LEEME_ACCIONES.md", "manifest.json"}:
            path.unlink()


def action_guide(spec: SheetSpec) -> str:
    return (
        f"# Frames de {spec.character}\n\n"
        f"- Variante: {VARIANT_NAMES[spec.variant]}\n"
        f"- Hoja original: `{spec.sheet_path.relative_to(PROJECT_ROOT)}`\n"
        f"- Cuadrícula real: {spec.rows} filas × {spec.cols} columnas\n"
        "- Cada PNG conserva el cuerpo visible completo de su celda original.\n"
        "- Los frames vacíos de la hoja no se guardan.\n"
    )


def checkerboard(size: Tuple[int, int], square: int = 12) -> Image.Image:
    width, height = size
    image = Image.new("RGBA", size, (32, 35, 43, 255))
    draw = ImageDraw.Draw(image)
    for y in range(0, height, square):
        for x in range(0, width, square):
            if (x // square + y // square) % 2:
                draw.rectangle((x, y, x + square - 1, y + square - 1), fill=(49, 53, 64, 255))
    return image


def save_preview(spec: SheetSpec, frames: Dict[int, Image.Image], preview_dir: Path) -> Path:
    tile_size = 96
    header_height = 34
    preview = Image.new(
        "RGBA",
        (spec.cols * tile_size, header_height + spec.rows * tile_size),
        (20, 22, 28, 255),
    )
    draw = ImageDraw.Draw(preview)
    draw.text(
        (8, 10),
        f"{spec.character} - {VARIANT_NAMES[spec.variant]} - {spec.rows}x{spec.cols}",
        fill=(225, 230, 240, 255),
    )
    for row in range(spec.rows):
        for col in range(spec.cols):
            slot = row * spec.cols + col
            tile = checkerboard((tile_size, tile_size))
            frame = frames.get(slot)
            if frame is not None:
                tile.alpha_composite(frame.resize((tile_size, tile_size), Image.Resampling.NEAREST))
            preview.alpha_composite(tile, (col * tile_size, header_height + row * tile_size))

    preview_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{spec.character}__{spec.variant}.png".replace(" ", "_")
    path = preview_dir / filename
    preview.save(path)
    return path


def find_generated_frame(directory: Path, slot: int) -> Optional[Path]:
    return next(iter(sorted(directory.glob(f"frame_{slot:03d}_*.png"))), None)


def repair_erin_white_frame(target: Image.Image, donor: Image.Image, row: int) -> Image.Image:
    target = target.convert("RGBA")
    donor = donor.convert("RGBA")
    target_bounds = target.getchannel("A").getbbox()
    donor_bounds = donor.getchannel("A").getbbox()
    if target_bounds is None or donor_bounds is None:
        return target

    donor_height = donor_bounds[3] - donor_bounds[1]
    donor_lower = donor.copy()
    donor_alpha = donor_lower.getchannel("A")
    lower_start = donor_bounds[1] + round(donor_height * ERIN_DONOR_LOWER_START_RATIO)
    donor_alpha.paste(0, (0, 0, donor.width, lower_start))
    if row == 3:
        lower_end = donor_bounds[1] + round(donor_height * ERIN_BACK_DONOR_BOTTOM_RATIO)
        donor_alpha.paste(0, (0, lower_end, donor.width, donor.height))
    donor_lower.putalpha(donor_alpha)

    target_foreground = target.crop(target_bounds)
    repaired_upper_height = round(donor_height * ERIN_WHITE_UPPER_HEIGHT_RATIO)
    repaired_upper_width = round(
        target_foreground.width * repaired_upper_height / target_foreground.height
    )
    target_foreground = target_foreground.resize(
        (repaired_upper_width, repaired_upper_height),
        Image.Resampling.NEAREST,
    )

    repaired = Image.new("RGBA", target.size, (0, 0, 0, 0))
    repaired.alpha_composite(donor_lower)
    repaired.alpha_composite(
        target_foreground,
        ((target.width - repaired_upper_width) // 2, donor_bounds[1]),
    )
    normalized, _ = normalize_frame(
        repaired,
        allow_upscale=True,
        center_vertical=True,
        safe_size=CHARACTER_SAFE_SIZE,
    )
    return normalized


def repair_erin_white_frames(
    specs: Iterable[SheetSpec],
    manifests: Iterable[Dict[str, object]],
) -> List[int]:
    erin_specs = {
        spec.variant: spec
        for spec in specs
        if spec.character.lower() == "erin" and spec.variant in {"rbchef", "rnormal"}
    }
    if set(erin_specs) != {"rbchef", "rnormal"}:
        return []

    white_spec = erin_specs["rbchef"]
    normal_spec = erin_specs["rnormal"]
    white_directory = safe_output_directory(white_spec, OUTPUT_DIR)
    normal_directory = safe_output_directory(normal_spec, OUTPUT_DIR)
    repaired_slots: List[int] = []

    for row in ERIN_WHITE_INCOMPLETE_ROWS:
        for column in range(1, white_spec.cols + 1):
            slot = (row - 1) * white_spec.cols + column
            target_path = find_generated_frame(white_directory, slot)
            donor_path = find_generated_frame(normal_directory, slot)
            if target_path is None or donor_path is None:
                continue
            with Image.open(target_path) as target, Image.open(donor_path) as donor:
                repaired = repair_erin_white_frame(target, donor, row)
            repaired.save(target_path)
            repaired_slots.append(slot)

    white_manifest = next(
        (
            manifest
            for manifest in manifests
            if str(manifest["character"]).lower() == "erin" and manifest["variant"] == "rbchef"
        ),
        None,
    )
    if white_manifest is not None and repaired_slots:
        frames: Dict[int, Image.Image] = {}
        for record in white_manifest["frames"]:
            slot = int(record["slot"])
            frame_path = white_directory / str(record["file"])
            if frame_path.exists():
                with Image.open(frame_path) as frame:
                    frames[slot - 1] = frame.convert("RGBA")
        preview_path = save_preview(white_spec, frames, PREVIEW_DIR)
        white_manifest["preview"] = str(preview_path.relative_to(PROJECT_ROOT))
        white_manifest["repaired_incomplete_body_frames"] = repaired_slots
        white_manifest["repair_donor_variant"] = "rnormal"
        (white_directory / "manifest.json").write_text(
            json.dumps(white_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return repaired_slots


def remove_erin_bottom_intrusion(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    array = np.array(rgba)
    alpha = array[:, :, 3] > ALPHA_THRESHOLD
    coordinates = np.argwhere(alpha)
    if len(coordinates) < 40:
        return rgba

    y0 = int(coordinates[:, 0].min())
    y1 = int(coordinates[:, 0].max())
    rows = np.indices(alpha.shape)[0]
    rgb = array[:, :, :3]
    shoe_pixels = (
        alpha
        & (rows > y0 + (y1 - y0) * 0.55)
        & (rgb[:, :, 0] > 75)
        & (rgb[:, :, 1] > 55)
        & (rgb[:, :, 2] > 35)
        & (rgb.mean(axis=2) > 65)
    )
    shoe_coordinates = np.argwhere(shoe_pixels)
    if len(shoe_coordinates) == 0:
        return rgba

    shoe_bottom = int(shoe_coordinates[:, 0].max())
    array[shoe_bottom + 2 :, :, 3] = 0
    cleaned = Image.fromarray(array, mode="RGBA")
    normalized, _ = normalize_frame(
        cleaned,
        allow_upscale=True,
        center_vertical=True,
        safe_size=CHARACTER_SAFE_SIZE,
    )
    return normalized


def repair_erin_normal_frames(
    specs: Iterable[SheetSpec],
    manifests: Iterable[Dict[str, object]],
) -> List[int]:
    normal_spec = next(
        (
            spec
            for spec in specs
            if spec.character.lower() == "erin" and spec.variant == "rnormal"
        ),
        None,
    )
    if normal_spec is None:
        return []

    normal_directory = safe_output_directory(normal_spec, OUTPUT_DIR)
    repaired_slots: List[int] = []
    for slot in ERIN_NORMAL_BOTTOM_INTRUSION_SLOTS:
        frame_path = find_generated_frame(normal_directory, slot)
        if frame_path is None:
            continue
        with Image.open(frame_path) as frame:
            repaired = remove_erin_bottom_intrusion(frame)
        repaired.save(frame_path)
        repaired_slots.append(slot)

    normal_manifest = next(
        (
            manifest
            for manifest in manifests
            if str(manifest["character"]).lower() == "erin" and manifest["variant"] == "rnormal"
        ),
        None,
    )
    if normal_manifest is not None and repaired_slots:
        frames: Dict[int, Image.Image] = {}
        for record in normal_manifest["frames"]:
            slot = int(record["slot"])
            frame_path = normal_directory / str(record["file"])
            if frame_path.exists():
                with Image.open(frame_path) as frame:
                    frames[slot - 1] = frame.convert("RGBA")
        preview_path = save_preview(normal_spec, frames, PREVIEW_DIR)
        normal_manifest["preview"] = str(preview_path.relative_to(PROJECT_ROOT))
        normal_manifest["repaired_bottom_intrusion_frames"] = repaired_slots
        (normal_directory / "manifest.json").write_text(
            json.dumps(normal_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return repaired_slots


def repair_incomplete_body_frame(
    target: Image.Image,
    donor: Image.Image,
    upper_height_ratio: float,
    donor_lower_start_ratio: float,
) -> Image.Image:
    target = target.convert("RGBA")
    donor = donor.convert("RGBA")
    target_bounds = target.getchannel("A").getbbox()
    donor_bounds = donor.getchannel("A").getbbox()
    if target_bounds is None or donor_bounds is None:
        return target

    donor_height = donor_bounds[3] - donor_bounds[1]
    donor_lower = donor.copy()
    donor_alpha = donor_lower.getchannel("A")
    lower_start = donor_bounds[1] + round(donor_height * donor_lower_start_ratio)
    donor_alpha.paste(0, (0, 0, donor.width, lower_start))
    donor_lower.putalpha(donor_alpha)

    target_foreground = target.crop(target_bounds)
    repaired_upper_height = round(donor_height * upper_height_ratio)
    repaired_upper_width = round(
        target_foreground.width * repaired_upper_height / target_foreground.height
    )
    target_foreground = target_foreground.resize(
        (repaired_upper_width, repaired_upper_height),
        Image.Resampling.NEAREST,
    )

    repaired = Image.new("RGBA", target.size, (0, 0, 0, 0))
    repaired.alpha_composite(donor_lower)
    repaired.alpha_composite(
        target_foreground,
        ((target.width - repaired_upper_width) // 2, donor_bounds[1]),
    )
    normalized, _ = normalize_frame(
        repaired,
        allow_upscale=True,
        center_vertical=True,
        safe_size=CHARACTER_SAFE_SIZE,
    )
    return normalized


def repair_nico_frames(
    specs: Iterable[SheetSpec],
    manifests: Iterable[Dict[str, object]],
) -> List[int]:
    nico_spec = next(
        (
            spec
            for spec in specs
            if spec.character.lower() == "nico" and spec.variant == "rnormal"
        ),
        None,
    )
    if nico_spec is None:
        return []

    nico_directory = safe_output_directory(nico_spec, OUTPUT_DIR)
    repaired_slots: List[int] = []
    for slot, (donor_slot, upper_height_ratio) in NICO_BODY_REPAIR_SPECS.items():
        target_path = find_generated_frame(nico_directory, slot)
        donor_path = find_generated_frame(nico_directory, donor_slot)
        if target_path is None or donor_path is None:
            continue
        with Image.open(target_path) as target, Image.open(donor_path) as donor:
            repaired = repair_incomplete_body_frame(
                target,
                donor,
                upper_height_ratio,
                NICO_DONOR_LOWER_START_RATIO,
            )
        repaired.save(target_path)
        repaired_slots.append(slot)

    nico_manifest = next(
        (
            manifest
            for manifest in manifests
            if str(manifest["character"]).lower() == "nico" and manifest["variant"] == "rnormal"
        ),
        None,
    )
    if nico_manifest is not None and repaired_slots:
        frames: Dict[int, Image.Image] = {}
        for record in nico_manifest["frames"]:
            slot = int(record["slot"])
            frame_path = nico_directory / str(record["file"])
            if frame_path.exists():
                with Image.open(frame_path) as frame:
                    frames[slot - 1] = frame.convert("RGBA")
        preview_path = save_preview(nico_spec, frames, PREVIEW_DIR)
        nico_manifest["preview"] = str(preview_path.relative_to(PROJECT_ROOT))
        nico_manifest["repaired_incomplete_body_frames"] = repaired_slots
        nico_manifest["repair_donor_slots"] = {
            str(slot): donor_slot
            for slot, (donor_slot, _) in NICO_BODY_REPAIR_SPECS.items()
        }
        (nico_directory / "manifest.json").write_text(
            json.dumps(nico_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return repaired_slots


def repair_sebas_frames(
    specs: Iterable[SheetSpec],
    manifests: Iterable[Dict[str, object]],
) -> List[int]:
    sebas_spec = next(
        (
            spec
            for spec in specs
            if spec.character.lower() == "sebas" and spec.variant == "rnormal"
        ),
        None,
    )
    if sebas_spec is None:
        return []

    sebas_directory = safe_output_directory(sebas_spec, OUTPUT_DIR)
    repaired_slots: List[int] = []
    for slot, donor_slot in SEBAS_FEET_REPAIR_SPECS.items():
        target_path = find_generated_frame(sebas_directory, slot)
        donor_path = find_generated_frame(sebas_directory, donor_slot)
        if target_path is None or donor_path is None:
            continue
        with Image.open(target_path) as target, Image.open(donor_path) as donor:
            repaired = repair_incomplete_body_frame(
                target,
                donor,
                SEBAS_UPPER_HEIGHT_RATIO,
                SEBAS_DONOR_LOWER_START_RATIO,
            )
        repaired.save(target_path)
        repaired_slots.append(slot)

    sebas_manifest = next(
        (
            manifest
            for manifest in manifests
            if str(manifest["character"]).lower() == "sebas" and manifest["variant"] == "rnormal"
        ),
        None,
    )
    if sebas_manifest is not None and repaired_slots:
        frames: Dict[int, Image.Image] = {}
        for record in sebas_manifest["frames"]:
            slot = int(record["slot"])
            frame_path = sebas_directory / str(record["file"])
            if frame_path.exists():
                with Image.open(frame_path) as frame:
                    frames[slot - 1] = frame.convert("RGBA")
        preview_path = save_preview(sebas_spec, frames, PREVIEW_DIR)
        sebas_manifest["preview"] = str(preview_path.relative_to(PROJECT_ROOT))
        sebas_manifest["repaired_incomplete_feet_frames"] = repaired_slots
        sebas_manifest["repair_donor_slots"] = {
            str(slot): donor_slot
            for slot, donor_slot in SEBAS_FEET_REPAIR_SPECS.items()
        }
        (sebas_directory / "manifest.json").write_text(
            json.dumps(sebas_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return repaired_slots


def extract_sheet(spec: SheetSpec, output_base: Path, preview_dir: Path) -> Dict[str, object]:
    destination = safe_output_directory(spec, output_base)
    with Image.open(spec.sheet_path) as source:
        sheet = source.convert("RGBA")
    width, height = sheet.size
    sheet_key = relative_sheet_key(spec.sheet_path)
    is_duvan_white = sheet_key == DUVAN_WHITE_SHEET
    if is_duvan_white:
        column_edges = list(DUVAN_WHITE_COLUMN_EDGES)
        row_edges = list(DUVAN_WHITE_ROW_EDGES)
        overlap_frames = extract_duvan_white_overlaps(sheet, column_edges)
    elif sheet_key in UNIFORM_GRID_SHEETS:
        column_edges = [int(round(column * width / spec.cols)) for column in range(spec.cols + 1)]
        row_edges = [int(round(row * height / spec.rows)) for row in range(spec.rows + 1)]
        overlap_frames = {}
    else:
        column_edges = detect_column_edges(sheet, spec.rows, spec.cols)
        row_edges = detect_row_edges(sheet, spec.rows, column_edges)
        overlap_frames = {}

    clear_generated_variant(destination)
    frames: Dict[int, Image.Image] = {}
    frame_records: List[Dict[str, object]] = []
    empty_slots: List[int] = []
    excluded_slots = sorted(EXCLUDED_FRAME_SLOTS.get(relative_sheet_key(spec.sheet_path), set()))
    if is_duvan_white:
        excluded_slots = sorted(set(excluded_slots) | {61, 62, 63, 64})
    excluded_slot_set = set(excluded_slots)
    border_contacts = 0

    for row in range(spec.rows):
        for col in range(spec.cols):
            slot = row * spec.cols + col
            if slot + 1 in excluded_slot_set:
                continue
            if is_duvan_white and row >= 12:
                cell = overlap_frames.get((row, col))
                if cell is None:
                    raise RuntimeError(
                        f"No se pudo reconstruir el slot {slot + 1} de Duvan."
                    )
            else:
                x0 = column_edges[col]
                x1 = column_edges[col + 1]
                y0 = row_edges[row]
                y1 = row_edges[row + 1]
                cell = sheet.crop((x0, y0, x1, y1))
            normalized, bounds = normalize_frame(
                cell,
                allow_upscale=True,
                center_vertical=True,
                safe_size=CHARACTER_SAFE_SIZE,
            )
            if bounds is None:
                empty_slots.append(slot + 1)
                continue

            sides = (
                []
                if is_duvan_white and row >= 12
                else source_border_sides(bounds, cell.width, cell.height)
            )
            if sides:
                border_contacts += 1
            filename = f"frame_{slot + 1:03d}_r{row + 1:02d}_c{col + 1:02d}.png"
            normalized.save(destination / filename)
            frames[slot] = normalized
            frame_records.append(
                {
                    "slot": slot + 1,
                    "row": row + 1,
                    "column": col + 1,
                    "file": filename,
                    "source_border_contacts": sides,
                }
            )

    if spec.front_path:
        with Image.open(spec.front_path) as source_front:
            front, _ = normalize_frame(
                source_front.convert("RGBA"),
                portrait=True,
                allow_upscale=True,
                center_vertical=True,
                safe_size=CHARACTER_SAFE_SIZE,
            )
        front.save(destination / "00_frontal_identidad.png")

    (destination / "LEEME_ACCIONES.md").write_text(action_guide(spec), encoding="utf-8")
    preview_path = save_preview(spec, frames, preview_dir)
    manifest = {
        "character": spec.character,
        "variant": spec.variant,
        "variant_name": VARIANT_NAMES[spec.variant],
        "source_sheet": str(spec.sheet_path.relative_to(PROJECT_ROOT)),
        "source_front": str(spec.front_path.relative_to(PROJECT_ROOT)) if spec.front_path else None,
        "grid": {"rows": spec.rows, "columns": spec.cols},
        "column_edges": column_edges,
        "row_edges": row_edges,
        "slots": spec.rows * spec.cols,
        "frames_saved": len(frame_records),
        "empty_slots": empty_slots,
        "excluded_slots": excluded_slots,
        "source_border_contact_frames": border_contacts,
        "output_size": [TARGET_SIZE, TARGET_SIZE],
        "target_visible_size": CHARACTER_SAFE_SIZE,
        "preview": str(preview_path.relative_to(PROJECT_ROOT)),
        "frames": frame_records,
    }
    if is_duvan_white:
        manifest["source_reconstruction"] = {
            "method": "alpha_seed_watershed",
            "overlapping_rows_rebuilt": [13, 14, 15],
            "excluded_source_slots": [61, 62, 63, 64],
            "reason": "La hoja fuente termina antes de mostrar una fila 16 completa.",
        }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def summarize(manifests: Iterable[Dict[str, object]]) -> Dict[str, object]:
    manifest_list = list(manifests)
    characters = sorted({str(item["character"]) for item in manifest_list}, key=str.lower)
    return {
        "characters": len(characters),
        "character_names": characters,
        "variants": len(manifest_list),
        "frames_saved": sum(int(item["frames_saved"]) for item in manifest_list),
        "empty_slots": sum(len(item["empty_slots"]) for item in manifest_list),
        "output_size": [TARGET_SIZE, TARGET_SIZE],
        "target_visible_size": CHARACTER_SAFE_SIZE,
        "manifests": [
            {
                "character": item["character"],
                "variant": item["variant"],
                "frames_saved": item["frames_saved"],
                "slots": item["slots"],
            }
            for item in manifest_list
        ],
    }


def extract_all(base_only: bool = False) -> Dict[str, object]:
    specs = discover_sheets(base_only=base_only)
    manifests: List[Dict[str, object]] = []
    for spec in specs:
        manifest = extract_sheet(spec, OUTPUT_DIR, PREVIEW_DIR)
        manifests.append(manifest)
        print(
            f"[OK] {spec.character} / {VARIANT_NAMES[spec.variant]}: "
            f"{manifest['frames_saved']}/{manifest['slots']} frames ({spec.rows}x{spec.cols})"
        )

    repaired_erin_slots = repair_erin_white_frames(specs, manifests)
    if repaired_erin_slots:
        print(
            f"[OK] Erin / ropa_blanca_chef: {len(repaired_erin_slots)} "
            "frames con piernas y pies restaurados"
        )

    cleaned_erin_slots = repair_erin_normal_frames(specs, manifests)
    if cleaned_erin_slots:
        print(
            f"[OK] Erin / ropa_normal: {len(cleaned_erin_slots)} "
            "frames sin residuos bajo los pies"
        )

    repaired_nico_slots = repair_nico_frames(specs, manifests)
    if repaired_nico_slots:
        print(
            f"[OK] Nico / ropa_normal: {len(repaired_nico_slots)} "
            "frames con cuerpo inferior restaurado"
        )

    repaired_sebas_slots = repair_sebas_frames(specs, manifests)
    if repaired_sebas_slots:
        print(
            f"[OK] Sebas / ropa_normal: {len(repaired_sebas_slots)} "
            "frames con pies restaurados"
        )

    summary = summarize(manifests)
    summary_path = OUTPUT_DIR / "resumen_extraccion.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("-" * 72)
    print(
        f"[OK] {summary['characters']} personajes, {summary['variants']} variantes y "
        f"{summary['frames_saved']} frames guardados."
    )
    print(f"[OK] Resumen: {summary_path}")
    print(f"[OK] Vistas previas: {PREVIEW_DIR}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Extrae únicamente Alex, Amaro, Belial, Conny y Dana.",
    )
    parser.add_argument(
        "--molds-only",
        action="store_true",
        help="Amplía, centra y extrae únicamente los moldes transparentes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
    arguments = parse_args()
    if arguments.molds_only:
        extract_molds()
    else:
        extract_all(base_only=arguments.base_only)
