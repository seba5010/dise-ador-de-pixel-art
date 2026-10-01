from pathlib import Path
import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw


CANVAS_W = 1024
CANVAS_H = 1536
COLS = 8
ROWS = 12
CELL = 128


def visible_mask(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    if float((alpha < 250).mean()) > 0.02:
        return (alpha > 12).astype(np.uint8)

    rgb = arr[:, :, :3].astype(np.float32)
    h, w = alpha.shape
    sample = max(4, min(h, w) // 40)
    corners = np.concatenate(
        [
            rgb[:sample, :sample].reshape(-1, 3),
            rgb[:sample, w - sample :].reshape(-1, 3),
            rgb[h - sample :, :sample].reshape(-1, 3),
            rgb[h - sample :, w - sample :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corners, axis=0)
    diff = np.sqrt(np.sum((rgb - bg) ** 2, axis=-1))
    return (diff > 30).astype(np.uint8)


def assign_component_to_cell(x: int, y: int, w: int, h: int):
    # Use the lower body anchor for characters. This is more stable than center
    # when a sprite crosses a horizontal grid boundary.
    anchor_x = x + w / 2
    anchor_y = y + h * 0.72
    col = int(np.clip(anchor_x // CELL, 0, COLS - 1))
    row = int(np.clip(anchor_y // CELL, 0, ROWS - 1))
    return row, col


def fit_group(group_img: Image.Image, bottom_margin: int = 5, top_margin: int = 5, side_margin: int = 4):
    group_img = group_img.convert("RGBA")
    arr = np.array(group_img)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return Image.new("RGBA", (CELL, CELL), (0, 0, 0, 0))

    crop = group_img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    sw, sh = crop.size
    max_w = CELL - side_margin * 2
    max_h = CELL - top_margin - bottom_margin
    scale = min(max_w / max(1, sw), max_h / max(1, sh), 1.0)
    if scale < 0.999:
        crop = crop.resize((max(1, round(sw * scale)), max(1, round(sh * scale))), Image.Resampling.LANCZOS)
        sw, sh = crop.size

    out = Image.new("RGBA", (CELL, CELL), (0, 0, 0, 0))
    x = (CELL - sw) // 2
    y = max(top_margin, CELL - sh - bottom_margin)
    out.paste(crop, (x, y), crop)
    return out


def normalize_sheet(src: Path, dst: Path, preview: Path | None = None):
    original = Image.open(src).convert("RGBA")
    sheet = original.resize((CANVAS_W, CANVAS_H), Image.Resampling.LANCZOS)
    mask = visible_mask(sheet)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    groups: dict[tuple[int, int], Image.Image] = {
        (r, c): Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
        for r in range(ROWS)
        for c in range(COLS)
    }

    for label in range(1, num):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 24:
            continue

        row, col = assign_component_to_cell(x, y, w, h)
        comp_mask = labels[y : y + h, x : x + w] == label
        comp = sheet.crop((x, y, x + w, y + h))
        arr = np.array(comp)
        arr[:, :, 3] = np.minimum(arr[:, :, 3], (comp_mask * 255).astype(np.uint8))
        comp = Image.fromarray(arr, mode="RGBA")
        groups[(row, col)].paste(comp, (x, y), comp)

    out = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    for r in range(ROWS):
        for c in range(COLS):
            x0, y0 = c * CELL, r * CELL
            # Crop a generous local region around the assigned cell. Components
            # assigned to this cell may cross original borders.
            region = groups[(r, c)].crop(
                (
                    max(0, x0 - CELL),
                    max(0, y0 - CELL),
                    min(CANVAS_W, x0 + CELL * 2),
                    min(CANVAS_H, y0 + CELL * 2),
                )
            )
            fixed = fit_group(region)
            out.paste(fixed, (x0, y0), fixed)

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)

    if preview:
        bg = Image.new("RGBA", out.size, (215, 218, 226, 255))
        bg.paste(out, (0, 0), out)
        draw = ImageDraw.Draw(bg)
        for x in range(0, CANVAS_W + 1, CELL):
            draw.line((x, 0, x, CANVAS_H), fill=(0, 180, 255, 180), width=1)
        for y in range(0, CANVAS_H + 1, CELL):
            draw.line((0, y, CANVAS_W, y), fill=(0, 180, 255, 180), width=1)
        preview.parent.mkdir(parents=True, exist_ok=True)
        bg.convert("RGB").save(preview)


def main():
    parser = argparse.ArgumentParser(description="Normaliza hoja completa detectando componentes globales.")
    parser.add_argument("--input", type=Path, default=Path("personajes"))
    parser.add_argument("--output", type=Path, default=Path("normalizadas_8x12_global"))
    args = parser.parse_args()

    count = 0
    for src in sorted(args.input.glob("*/movimientos_*.png")):
        rel = src.relative_to(args.input)
        dst = args.output / rel
        preview = args.output / rel.parent / f"{src.stem}_preview.jpg"
        normalize_sheet(src, dst, preview)
        print(f"OK {src} -> {dst}")
        count += 1
    print(f"Total normalizadas: {count}")


if __name__ == "__main__":
    main()
