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


def mask_from_rgba(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    if float((alpha < 250).mean()) > 0.02:
        return (alpha > 10).astype(np.uint8)

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


def fit_to_cell(sprite: Image.Image, top_margin: int = 4, bottom_margin: int = 5, side_margin: int = 4) -> Image.Image:
    sprite = sprite.convert("RGBA")
    arr = np.array(sprite)
    ys, xs = np.where(arr[:, :, 3] > 10)
    if len(xs) == 0:
        return Image.new("RGBA", (CELL, CELL), (0, 0, 0, 0))

    crop = sprite.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
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


def select_sprite_for_cell(sheet: Image.Image, row: int, col: int) -> Image.Image:
    x0 = col * CELL
    y0 = row * CELL

    # Generous area: enough to recover hats/feet crossing a guide line, but not
    # the whole column. This is the key difference from the failed global pass.
    pad_x = 24
    pad_top = 92
    pad_bottom = 92
    rx0 = max(0, x0 - pad_x)
    ry0 = max(0, y0 - pad_top)
    rx1 = min(CANVAS_W, x0 + CELL + pad_x)
    ry1 = min(CANVAS_H, y0 + CELL + pad_bottom)

    region = sheet.crop((rx0, ry0, rx1, ry1)).convert("RGBA")
    mask = mask_from_rgba(region)

    # Clean isolated specks but keep pixel-art edges.
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if int(clean.sum()) > 50:
        mask = clean

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return Image.new("RGBA", (CELL, CELL), (0, 0, 0, 0))

    # Target cell rectangle in region coordinates.
    tx0 = x0 - rx0
    ty0 = y0 - ry0
    tx1 = tx0 + CELL
    ty1 = ty0 + CELL
    target_cx = tx0 + CELL / 2
    target_foot_y = ty0 + CELL * 0.86

    best_label = None
    best_score = -1e18
    for label in range(1, num):
        lx, ly, lw, lh, area = [int(v) for v in stats[label]]
        if area < 28:
            continue
        ix0 = max(lx, tx0)
        iy0 = max(ly, ty0)
        ix1 = min(lx + lw, tx1)
        iy1 = min(ly + lh, ty1)
        overlap = 0 if ix1 <= ix0 or iy1 <= iy0 else int(((labels[iy0:iy1, ix0:ix1] == label).sum()))
        if overlap < 15:
            continue

        cx = lx + lw / 2
        foot_y = ly + lh
        distance_penalty = abs(cx - target_cx) * 3.0 + abs(foot_y - target_foot_y) * 1.3
        # Favor components that have real presence inside this cell, not just a
        # neighboring hat/foot fragment crossing the boundary.
        score = overlap * 5 + area * 0.25 - distance_penalty
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return Image.new("RGBA", (CELL, CELL), (0, 0, 0, 0))

    chosen = labels == best_label
    bx, by, bw, bh, _ = [int(v) for v in stats[best_label]]

    # Add tiny nearby companion pieces only if they are very close to the chosen
    # sprite (for example a detached shadow), not random bits from another frame.
    expanded = (max(0, bx - 8), max(0, by - 8), min(labels.shape[1], bx + bw + 8), min(labels.shape[0], by + bh + 8))
    for label in range(1, num):
        if label == best_label:
            continue
        lx, ly, lw, lh, area = [int(v) for v in stats[label]]
        if area > 350:
            continue
        if lx + lw < expanded[0] or lx > expanded[2] or ly + lh < expanded[1] or ly > expanded[3]:
            continue
        chosen |= labels == label

    arr = np.array(region)
    arr[:, :, 3] = np.minimum(arr[:, :, 3], (chosen * 255).astype(np.uint8))
    isolated = Image.fromarray(arr, mode="RGBA")
    return fit_to_cell(isolated)


def normalize_sheet(src: Path, dst: Path, preview: Path | None = None):
    original = Image.open(src).convert("RGBA")
    sheet = original.resize((CANVAS_W, CANVAS_H), Image.Resampling.LANCZOS)
    out = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))

    for row in range(ROWS):
        for col in range(COLS):
            fixed = select_sprite_for_cell(sheet, row, col)
            out.paste(fixed, (col * CELL, row * CELL), fixed)

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
        bg.convert("RGB").save(preview, quality=92)


def main():
    parser = argparse.ArgumentParser(description="Normaliza hojas 8x12 por detección local de silueta principal.")
    parser.add_argument("--input", type=Path, default=Path("personajes"))
    parser.add_argument("--output", type=Path, default=Path("normalizadas_8x12_local"))
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
