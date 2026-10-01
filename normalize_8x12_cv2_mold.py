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


def alpha_or_bg_mask(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    if float((alpha < 250).mean()) > 0.02:
        return (alpha > 12).astype(np.uint8)

    rgb = arr[:, :, :3].astype(np.float32)
    h, w = alpha.shape
    sample = max(2, min(h, w) // 10)
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


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def expand_bbox(bbox, pad_x: int = 28, pad_y: int = 32):
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(CELL - 1, x1 + pad_x),
        min(CELL - 1, y1 + pad_y),
    )


def component_filter_by_mold(cell: Image.Image, mold_cell: Image.Image) -> np.ndarray:
    mask = alpha_or_bg_mask(cell)
    mold_mask = alpha_or_bg_mask(mold_cell)
    mold_bbox = bbox_from_mask(mold_mask)
    if mold_bbox is None:
        return mask

    allowed_bbox = expand_bbox(mold_bbox)
    ax0, ay0, ax1, ay1 = allowed_bbox
    allowed = np.zeros((CELL, CELL), dtype=np.uint8)
    allowed[ay0 : ay1 + 1, ax0 : ax1 + 1] = 1

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.zeros_like(mask, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num > 1 else []
    max_area = int(max(areas)) if len(areas) else 0

    for label in range(1, num):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        comp = labels == label
        overlap = int((comp & (allowed > 0)).sum())
        overlap_ratio = overlap / max(1, area)

        cx = x + w / 2
        cy = y + h / 2
        center_allowed = ax0 <= cx <= ax1 and ay0 <= cy <= ay1
        large_main = area >= max(80, max_area * 0.18)

        # Keep components that belong to the mold's expected region. This preserves
        # separated hats/hands if they are in the pose area, while dropping fragments
        # from neighboring cells outside that area.
        if overlap_ratio >= 0.12 or (center_allowed and large_main):
            kept[comp] = 1

    if kept.sum() < 20:
        return mask
    return kept


def crop_with_mask(cell: Image.Image, mask: np.ndarray):
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    x0, y0, x1, y1 = bbox
    rgba = cell.convert("RGBA")
    arr = np.array(rgba.crop((x0, y0, x1 + 1, y1 + 1)))
    crop_mask = (mask[y0 : y1 + 1, x0 : x1 + 1] * 255).astype(np.uint8)
    arr[:, :, 3] = np.minimum(arr[:, :, 3], crop_mask)
    return Image.fromarray(arr, mode="RGBA")


def place(sprite: Image.Image, bottom_margin: int = 5, top_margin: int = 5, side_margin: int = 4):
    sprite = sprite.convert("RGBA")
    sw, sh = sprite.size
    max_w = CELL - side_margin * 2
    max_h = CELL - top_margin - bottom_margin
    scale = min(max_w / max(1, sw), max_h / max(1, sh), 1.0)
    if scale < 0.999:
        sprite = sprite.resize((max(1, round(sw * scale)), max(1, round(sh * scale))), Image.Resampling.LANCZOS)
        sw, sh = sprite.size
    out = Image.new("RGBA", (CELL, CELL), (0, 0, 0, 0))
    x = (CELL - sw) // 2
    y = max(top_margin, CELL - sh - bottom_margin)
    out.paste(sprite, (x, y), sprite)
    return out


def normalize_sheet(src: Path, mold: Image.Image, dst: Path, preview: Path | None = None):
    img = Image.open(src).convert("RGBA")
    w, h = img.size
    out = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))

    for r in range(ROWS):
        for c in range(COLS):
            x0 = round(c * w / COLS)
            x1 = round((c + 1) * w / COLS)
            y0 = round(r * h / ROWS)
            y1 = round((r + 1) * h / ROWS)
            cell = img.crop((x0, y0, x1, y1))
            mold_cell = mold.crop((c * CELL, r * CELL, (c + 1) * CELL, (r + 1) * CELL))
            mask = component_filter_by_mold(cell.resize((CELL, CELL), Image.Resampling.LANCZOS), mold_cell)
            sprite = crop_with_mask(cell.resize((CELL, CELL), Image.Resampling.LANCZOS), mask)
            fixed = place(sprite)
            out.paste(fixed, (c * CELL, r * CELL), fixed)

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
    parser = argparse.ArgumentParser(description="Normaliza 8x12 usando OpenCV y la mascara del molde por frame.")
    parser.add_argument("--input", type=Path, default=Path("personajes"))
    parser.add_argument("--output", type=Path, default=Path("normalizadas_8x12_cv2_mold"))
    parser.add_argument("--mold", type=Path, default=Path("plantilla de los spritesheets.png"))
    args = parser.parse_args()

    mold = Image.open(args.mold).convert("RGBA")
    if mold.size != (CANVAS_W, CANVAS_H):
        raise ValueError(f"Molde incorrecto: {mold.size}, se esperaba {(CANVAS_W, CANVAS_H)}")

    count = 0
    for src in sorted(args.input.glob("*/movimientos_*.png")):
        rel = src.relative_to(args.input)
        dst = args.output / rel
        preview = args.output / rel.parent / f"{src.stem}_preview.jpg"
        normalize_sheet(src, mold, dst, preview)
        print(f"OK {src} -> {dst}")
        count += 1
    print(f"Total normalizadas: {count}")


if __name__ == "__main__":
    main()
