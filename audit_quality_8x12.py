from pathlib import Path
import argparse
import csv
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw


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


def components(mask: np.ndarray, min_area: int = 24):
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        comps.append(
            {
                "area": area,
                "bbox": (x, y, x + w - 1, y + h - 1),
                "center": (float(cents[label][0]), float(cents[label][1])),
                "label": label,
            }
        )
    comps.sort(key=lambda c: c["area"], reverse=True)
    return comps


def classify_cell(cell: Image.Image) -> dict:
    # Analyze at 128x128 so all sheets share the same thresholds.
    cell = cell.convert("RGBA").resize((CELL, CELL), Image.Resampling.LANCZOS)
    mask = visible_mask(cell)
    comps = components(mask)
    if not comps:
        return {"quality": "MALO", "reason": "empty", "components": 0, "bbox": "", "main_area": 0}

    main = comps[0]
    x0, y0, x1, y1 = main["bbox"]
    main_area = main["area"]
    reasons = []

    touches_top = y0 <= 1
    touches_bottom = y1 >= CELL - 2
    touches_left = x0 <= 1
    touches_right = x1 >= CELL - 2
    full_height = (y1 - y0 + 1) >= 124
    full_width = (x1 - x0 + 1) >= 124

    if touches_top:
        reasons.append("touch_top")
    if touches_bottom:
        reasons.append("touch_bottom")
    if touches_left:
        reasons.append("touch_left")
    if touches_right:
        reasons.append("touch_right")
    if full_height:
        reasons.append("full_height")
    if full_width:
        reasons.append("full_width")

    separated_big = 0
    below_big = 0
    above_big = 0
    mx0, my0, mx1, my1 = main["bbox"]
    mcx, mcy = main["center"]
    for comp in comps[1:]:
        area = comp["area"]
        if area < max(90, main_area * 0.05):
            continue
        cx, cy = comp["center"]
        bx0, by0, bx1, by1 = comp["bbox"]
        separated = abs(cx - mcx) > 38 or abs(cy - mcy) > 45
        if separated:
            separated_big += 1
        if by0 > my1 + 2:
            below_big += 1
        if by1 < my0 - 2:
            above_big += 1

    if separated_big:
        reasons.append(f"separated_big:{separated_big}")
    if below_big:
        reasons.append(f"below_big:{below_big}")
    if above_big:
        reasons.append(f"above_big:{above_big}")

    # If the cell spans top+bottom or has a large separated fragment below/above,
    # the frame is probably split between cells and cannot be fixed safely.
    if (touches_top and touches_bottom) or full_height or below_big or above_big:
        quality = "MALO"
    elif reasons:
        quality = "REPARABLE"
    else:
        quality = "OK"

    return {
        "quality": quality,
        "reason": "|".join(reasons),
        "components": len(comps),
        "bbox": main["bbox"],
        "main_area": main_area,
    }


def make_bad_contact(rows: list[dict], output: Path, limit: int = 80):
    bad = [r for r in rows if r["quality"] == "MALO"][:limit]
    if not bad:
        return
    cols = 8
    tile = 160
    title_h = 26
    rows_count = (len(bad) + cols - 1) // cols
    canvas = Image.new("RGBA", (cols * tile, rows_count * (tile + title_h)), (230, 233, 240, 255))
    draw = ImageDraw.Draw(canvas)
    for i, r in enumerate(bad):
        sheet = Image.open(r["sheet"]).convert("RGBA")
        sw, sh = sheet.size
        rr = int(r["row"]) - 1
        cc = int(r["col"]) - 1
        x0 = round(cc * sw / COLS)
        x1 = round((cc + 1) * sw / COLS)
        y0 = round(rr * sh / ROWS)
        y1 = round((rr + 1) * sh / ROWS)
        cell = sheet.crop((x0, y0, x1, y1)).resize((128, 128), Image.Resampling.LANCZOS)
        px = (i % cols) * tile
        py = (i // cols) * (tile + title_h)
        draw.text((px + 4, py + 4), f'{r["character"]} {r["variant"]} F{r["frame"]}', fill=(0, 0, 0, 255))
        canvas.paste(cell, (px + 16, py + title_h), cell)
        draw.rectangle((px + 16, py + title_h, px + 143, py + title_h + 127), outline=(255, 0, 0, 255), width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output)


def main():
    parser = argparse.ArgumentParser(description="Clasifica celdas originales 8x12 como OK/REPARABLE/MALO.")
    parser.add_argument("--input", type=Path, default=Path("personajes"))
    parser.add_argument("--map", type=Path, default=Path("frame_map_8x12.json"))
    parser.add_argument("--output", type=Path, default=Path("auditoria_calidad_8x12"))
    args = parser.parse_args()

    frame_map = json.loads(args.map.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for sheet_path in sorted(args.input.glob("*/movimientos_*.png")):
        sheet = Image.open(sheet_path).convert("RGBA")
        sw, sh = sheet.size
        for meta in frame_map:
            r = meta["row"] - 1
            c = meta["col"] - 1
            x0 = round(c * sw / COLS)
            x1 = round((c + 1) * sw / COLS)
            y0 = round(r * sh / ROWS)
            y1 = round((r + 1) * sh / ROWS)
            cell = sheet.crop((x0, y0, x1, y1))
            result = classify_cell(cell)
            all_rows.append(
                {
                    "sheet": str(sheet_path),
                    "character": sheet_path.parent.name,
                    "variant": sheet_path.stem.replace("movimientos_", ""),
                    **meta,
                    **result,
                }
            )

    csv_path = args.output / "quality_audit.csv"
    fieldnames = [
        "sheet",
        "character",
        "variant",
        "frame",
        "row",
        "col",
        "direction",
        "action",
        "phase",
        "quality",
        "reason",
        "components",
        "bbox",
        "main_area",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    make_bad_contact(all_rows, args.output / "bad_frames_contact.jpg")
    counts = {}
    for row in all_rows:
        counts[row["quality"]] = counts.get(row["quality"], 0) + 1
    print(f"Frames auditados: {len(all_rows)}")
    print(counts)
    print(f"Reporte: {csv_path}")


if __name__ == "__main__":
    main()
