from pathlib import Path
import argparse

import numpy as np
from PIL import Image, ImageDraw


CANVAS_W = 1024
CANVAS_H = 1536
COLS = 8
ROWS = 12
CELL = 128


def foreground_mask(img: Image.Image, alpha_threshold: int = 12, diff_threshold: float = 30.0) -> np.ndarray:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]

    # If the image has real transparency, trust it.
    if float((alpha < 250).mean()) > 0.02:
        return alpha > alpha_threshold

    # Otherwise estimate the background from cell corners.
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
    return diff > diff_threshold


def keep_main_components(mask: np.ndarray, min_area: int = 24, keep_mode: str = "strict") -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[tuple[int, int, int, int, int, list[tuple[int, int]]]] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            pixels = []
            x0 = x1 = x
            y0 = y1 = y

            while stack:
                px, py = stack.pop()
                pixels.append((px, py))
                x0 = min(x0, px)
                x1 = max(x1, px)
                y0 = min(y0, py)
                y1 = max(y1, py)
                for nx in (px - 1, px, px + 1):
                    for ny in (py - 1, py, py + 1):
                        if nx == px and ny == py:
                            continue
                        if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((nx, ny))

            area = len(pixels)
            if area >= min_area:
                components.append((area, x0, y0, x1, y1, pixels))

    if not components:
        return mask

    # Prefer the upper/main body component. Neighbor-cell contamination usually appears
    # as a separated cap/head fragment near the bottom of the cell.
    components.sort(key=lambda item: (item[0], -item[2]), reverse=True)
    main = components[0]
    main_area, main_x0, main_y0, main_x1, main_y1, _ = main
    main_cx = (main_x0 + main_x1) / 2
    main_cy = (main_y0 + main_y1) / 2

    kept = np.zeros_like(mask, dtype=bool)
    for area, x0, y0, x1, y1, pixels in components:
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        near_main = abs(cx - main_cx) <= 42 and abs(cy - main_cy) <= 50
        significant = area >= max(120, main_area * 0.08)
        touches_main_box = not (x1 < main_x0 - 6 or x0 > main_x1 + 6 or y1 < main_y0 - 2 or y0 > main_y1 + 2)
        below_main_gap = y0 > main_y1 + 2
        above_main_gap = y1 < main_y0 - 2

        keep = area == main_area
        if not keep and keep_mode != "strict":
            keep = (near_main or touches_main_box) and significant
        elif not keep:
            # Strict mode keeps nearby/overlapping object pieces, but rejects clearly
            # separated fragments above or below the main body.
            keep = (near_main or touches_main_box) and significant and not below_main_gap and not above_main_gap

        # Keep body plus nearby objects/hands. Drop distant orphan fragments from neighboring cells.
        if keep:
            for px, py in pixels:
                kept[py, px] = True

    return kept


def extract_sprite(cell: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    rgba = cell.convert("RGBA")
    mask = keep_main_components(foreground_mask(rgba), keep_mode="strict")
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0)), None

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    sprite = rgba.crop((x0, y0, x1, y1))

    # Convert estimated mask into alpha so opaque backgrounds do not come along.
    mask_crop = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    sprite_arr = np.array(sprite)
    sprite_arr[:, :, 3] = np.minimum(sprite_arr[:, :, 3], mask_crop)
    return Image.fromarray(sprite_arr, mode="RGBA"), (x0, y0, x1, y1)


def place_sprite(sprite: Image.Image, top_margin: int = 6, bottom_margin: int = 5, side_margin: int = 4) -> Image.Image:
    sprite = sprite.convert("RGBA")
    sw, sh = sprite.size
    max_w = CELL - side_margin * 2
    max_h = CELL - top_margin - bottom_margin
    scale = min(max_w / max(1, sw), max_h / max(1, sh), 1.0)

    if scale < 0.999:
        sprite = sprite.resize(
            (max(1, round(sw * scale)), max(1, round(sh * scale))),
            Image.Resampling.LANCZOS,
        )
        sw, sh = sprite.size

    out = Image.new("RGBA", (CELL, CELL), (0, 0, 0, 0))
    x = (CELL - sw) // 2
    y = CELL - sh - bottom_margin
    if y < top_margin:
        y = top_margin
    out.paste(sprite, (x, y), sprite)
    return out


def normalize_sheet(src: Path, dst: Path, preview: Path | None = None) -> dict:
    img = Image.open(src).convert("RGBA")
    w, h = img.size
    out = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    report = {
        "source": str(src),
        "source_size": [w, h],
        "output": str(dst),
        "frames": [],
    }

    for r in range(ROWS):
        for c in range(COLS):
            x0 = round(c * w / COLS)
            x1 = round((c + 1) * w / COLS)
            y0 = round(r * h / ROWS)
            y1 = round((r + 1) * h / ROWS)
            cell = img.crop((x0, y0, x1, y1))
            sprite, bbox = extract_sprite(cell)
            fixed = place_sprite(sprite)
            out.paste(fixed, (c * CELL, r * CELL), fixed)
            report["frames"].append(
                {
                    "row": r + 1,
                    "col": c + 1,
                    "source_cell": [x0, y0, x1, y1],
                    "bbox": list(bbox) if bbox else None,
                    "sprite_size": list(sprite.size),
                }
            )

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

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Normaliza hojas 8x12 a celdas 128x128 sin sobrescribir originales.")
    parser.add_argument("--input", type=Path, default=Path("personajes"))
    parser.add_argument("--output", type=Path, default=Path("normalizadas_8x12_dynamic"))
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
