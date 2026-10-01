from pathlib import Path
import argparse
import csv
import json

import numpy as np
from PIL import Image, ImageDraw

from normalize_8x12_dynamic import foreground_mask, CELL


def components(mask: np.ndarray, min_area: int = 16):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            pts = []
            x0 = x1 = x
            y0 = y1 = y
            while stack:
                px, py = stack.pop()
                pts.append((px, py))
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
            area = len(pts)
            if area >= min_area:
                out.append({"area": area, "bbox": (x0, y0, x1, y1), "pixels": pts})
    out.sort(key=lambda c: c["area"], reverse=True)
    return out


def audit_cell(cell: Image.Image):
    mask = foreground_mask(cell)
    comps = components(mask)
    if not comps:
        return {
            "status": "empty",
            "component_count": 0,
            "main_area": 0,
            "orphan_count": 0,
            "bbox": None,
            "touches": "",
            "warnings": "empty",
        }

    main = comps[0]
    x0, y0, x1, y1 = main["bbox"]
    warnings = []
    touches = []
    if y0 <= 1:
        touches.append("top")
        warnings.append("touch_top")
    if y1 >= CELL - 2:
        touches.append("bottom")
        warnings.append("touch_bottom")
    if x0 <= 1:
        touches.append("left")
        warnings.append("touch_left")
    if x1 >= CELL - 2:
        touches.append("right")
        warnings.append("touch_right")

    main_area = main["area"]
    orphan_count = 0
    orphan_area = 0
    main_cx = (x0 + x1) / 2
    main_cy = (y0 + y1) / 2
    for comp in comps[1:]:
        area = comp["area"]
        bx0, by0, bx1, by1 = comp["bbox"]
        cx = (bx0 + bx1) / 2
        cy = (by0 + by1) / 2
        separated = abs(cx - main_cx) > 35 or abs(cy - main_cy) > 45
        sizeable = area >= max(80, main_area * 0.06)
        if separated and sizeable:
            orphan_count += 1
            orphan_area += area
    if orphan_count:
        warnings.append("orphan_fragment")

    h = y1 - y0 + 1
    w = x1 - x0 + 1
    if h > 122 or w > 122:
        warnings.append("almost_full_cell")
    if h < 32 or w < 20:
        warnings.append("very_small")

    return {
        "status": "ok" if not warnings else "check",
        "component_count": len(comps),
        "main_area": main_area,
        "orphan_count": orphan_count,
        "orphan_area": orphan_area,
        "bbox": (x0, y0, x1, y1),
        "sprite_w": w,
        "sprite_h": h,
        "touches": "|".join(touches),
        "warnings": "|".join(warnings),
    }


def draw_audit_preview(sheet: Image.Image, rows: list[dict], out_path: Path):
    preview = Image.new("RGBA", sheet.size, (215, 218, 226, 255))
    preview.paste(sheet, (0, 0), sheet)
    draw = ImageDraw.Draw(preview)
    for x in range(0, sheet.size[0] + 1, CELL):
        draw.line((x, 0, x, sheet.size[1]), fill=(0, 180, 255, 150), width=1)
    for y in range(0, sheet.size[1] + 1, CELL):
        draw.line((0, y, sheet.size[0], y), fill=(0, 180, 255, 150), width=1)

    for row in rows:
        if row["status"] == "ok":
            continue
        x0 = (int(row["col"]) - 1) * CELL
        y0 = (int(row["row"]) - 1) * CELL
        draw.rectangle((x0, y0, x0 + CELL - 1, y0 + CELL - 1), outline=(255, 40, 40, 255), width=3)
        draw.text((x0 + 3, y0 + 3), str(row["frame"]), fill=(255, 40, 40, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview.convert("RGB").save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Audita cada frame 8x12 y detecta cortes/fragmentos.")
    parser.add_argument("--input", type=Path, default=Path("normalizadas_8x12_dynamic_v3"))
    parser.add_argument("--map", type=Path, default=Path("frame_map_8x12.json"))
    parser.add_argument("--output", type=Path, default=Path("auditoria_frames_8x12"))
    args = parser.parse_args()

    frame_map = json.loads(args.map.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for sheet_path in sorted(args.input.glob("*/movimientos_*.png")):
        sheet = Image.open(sheet_path).convert("RGBA")
        rows = []
        for meta in frame_map:
            r = meta["row"] - 1
            c = meta["col"] - 1
            cell = sheet.crop((c * CELL, r * CELL, (c + 1) * CELL, (r + 1) * CELL))
            result = audit_cell(cell)
            row = {
                "sheet": str(sheet_path),
                "character": sheet_path.parent.name,
                "variant": sheet_path.stem.replace("movimientos_", ""),
                **meta,
                **result,
            }
            rows.append(row)
            all_rows.append(row)

        rel_dir = args.output / sheet_path.parent.name
        rel_dir.mkdir(parents=True, exist_ok=True)
        draw_audit_preview(sheet, rows, rel_dir / f"{sheet_path.stem}_audit.jpg")

    csv_path = args.output / "audit_frames.csv"
    fieldnames = [
        "sheet",
        "character",
        "variant",
        "index",
        "frame",
        "row",
        "col",
        "direction",
        "action",
        "phase",
        "status",
        "warnings",
        "component_count",
        "orphan_count",
        "orphan_area",
        "main_area",
        "bbox",
        "sprite_w",
        "sprite_h",
        "touches",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    check = [r for r in all_rows if r["status"] != "ok"]
    print(f"Frames auditados: {len(all_rows)}")
    print(f"Frames a revisar: {len(check)}")
    print(f"Reporte: {csv_path}")


if __name__ == "__main__":
    main()
