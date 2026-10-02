from pathlib import Path
import argparse
import csv
import json

from PIL import Image

from normalize_8x12_dynamic import extract_sprite, place_sprite, COLS, ROWS, CELL


def safe_name(text: str) -> str:
    return text.replace(" ", "_").replace("/", "_").lower()


def sample_sheet(sheet_path: Path, frame_map: list[dict], output_root: Path) -> None:
    img = Image.open(sheet_path).convert("RGBA")
    w, h = img.size
    out_dir = output_root / sheet_path.parent.name / sheet_path.stem
    raw_dir = out_dir / "raw_cells"
    sprite_dir = out_dir / "detected_sprites"
    fixed_dir = out_dir / "fixed_128"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sprite_dir.mkdir(parents=True, exist_ok=True)
    fixed_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "row",
                "col",
                "direction",
                "action",
                "phase",
                "raw_cell",
                "detected_sprite",
                "fixed_128",
                "bbox",
                "sprite_w",
                "sprite_h",
            ],
        )
        writer.writeheader()

        for meta in frame_map:
            r = meta["row"] - 1
            c = meta["col"] - 1
            x0 = round(c * w / COLS)
            x1 = round((c + 1) * w / COLS)
            y0 = round(r * h / ROWS)
            y1 = round((r + 1) * h / ROWS)
            cell = img.crop((x0, y0, x1, y1))
            sprite, bbox = extract_sprite(cell)
            fixed = place_sprite(sprite)

            stem = f'{meta["frame"]:03d}_{safe_name(meta["direction"])}_{safe_name(meta["action"])}_{meta["phase"]:02d}'
            raw_path = raw_dir / f"{stem}.png"
            sprite_path = sprite_dir / f"{stem}.png"
            fixed_path = fixed_dir / f"{stem}.png"
            cell.save(raw_path)
            sprite.save(sprite_path)
            fixed.save(fixed_path)

            writer.writerow(
                {
                    "frame": meta["frame"],
                    "row": meta["row"],
                    "col": meta["col"],
                    "direction": meta["direction"],
                    "action": meta["action"],
                    "phase": meta["phase"],
                    "raw_cell": raw_path,
                    "detected_sprite": sprite_path,
                    "fixed_128": fixed_path,
                    "bbox": bbox,
                    "sprite_w": sprite.size[0],
                    "sprite_h": sprite.size[1],
                }
            )

    print(f"OK {sheet_path} -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta muestras frame por frame usando frame_map_8x12.json.")
    parser.add_argument("--input", type=Path, default=Path("personajes"))
    parser.add_argument("--output", type=Path, default=Path("muestras_frames_8x12"))
    parser.add_argument("--map", type=Path, default=Path("frame_map_8x12.json"))
    args = parser.parse_args()

    frame_map = json.loads(args.map.read_text(encoding="utf-8"))
    if len(frame_map) != 96:
        raise ValueError(f"El mapa debe tener 96 frames, tiene {len(frame_map)}")

    for sheet in sorted(args.input.glob("*/movimientos_*.png")):
        sample_sheet(sheet, frame_map, args.output)


if __name__ == "__main__":
    main()
