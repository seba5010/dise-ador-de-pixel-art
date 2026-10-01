"""Crea paneles de auditoría para las carpetas reportadas por el usuario."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


CHARACTERS = [
    "alex", "carlos", "conny", "diego serena", "diego_vallenar", "duvan",
    "erin", "jorge", "juan", "mario", "maty hermano", "millaray",
]

ROOT = Path("personajes")
OUT = Path("auditoria_hojas_problematicas")
THUMB_W = 300
THUMB_H = 900
LABEL_H = 44


def checkerboard(size: tuple[int, int], block: int = 12) -> Image.Image:
    image = Image.new("RGB", size, (44, 47, 54))
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], block):
        for x in range(0, size[0], block):
            if (x // block + y // block) % 2:
                draw.rectangle((x, y, x + block - 1, y + block - 1), fill=(54, 58, 66))
    return image


def make_panel(path: Path) -> Image.Image:
    source = Image.open(path).convert("RGBA")
    source.thumbnail((THUMB_W, THUMB_H), Image.Resampling.NEAREST)
    panel = checkerboard((THUMB_W, THUMB_H + LABEL_H))
    x = (THUMB_W - source.width) // 2
    panel.paste(source, (x, LABEL_H), source)
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, THUMB_W - 1, LABEL_H - 1), fill=(20, 22, 27))
    draw.text((8, 6), path.parent.name, fill="white")
    draw.text((8, 22), f"{path.name} | {Image.open(path).size}", fill=(180, 205, 255))
    # Guía esperada 16×4; no modifica la fuente.
    for col in range(5):
        gx = x + round(col * source.width / 4)
        draw.line((gx, LABEL_H, gx, LABEL_H + source.height), fill=(0, 190, 255), width=1)
    for row in range(17):
        gy = LABEL_H + round(row * source.height / 16)
        draw.line((x, gy, x + source.width, gy), fill=(0, 190, 255), width=1)
    return panel


def main() -> None:
    OUT.mkdir(exist_ok=True)
    paths = []
    for character in CHARACTERS:
        paths.extend(sorted((ROOT / character).glob("*movimiento*.png")))

    panels = [make_panel(path) for path in paths]
    columns = 4
    rows = (len(panels) + columns - 1) // columns
    overview = Image.new("RGB", (columns * THUMB_W, rows * (THUMB_H + LABEL_H)), (18, 20, 24))
    for index, panel in enumerate(panels):
        overview.paste(panel, ((index % columns) * THUMB_W, (index // columns) * (THUMB_H + LABEL_H)))
    overview.save(OUT / "resumen_12_carpetas.jpg", quality=92)
    for character in CHARACTERS:
        character_paths = sorted((ROOT / character).glob("*movimiento*.png"))
        character_panels = [make_panel(path) for path in character_paths]
        if not character_panels:
            continue
        character_overview = Image.new(
            "RGB",
            (len(character_panels) * THUMB_W, THUMB_H + LABEL_H),
            (18, 20, 24),
        )
        for index, panel in enumerate(character_panels):
            character_overview.paste(panel, (index * THUMB_W, 0))
        character_overview.save(OUT / f"{character.replace(' ', '_')}.png")
    print(f"Hojas auditadas: {len(paths)}")
    print(OUT / "resumen_12_carpetas.jpg")


if __name__ == "__main__":
    main()
