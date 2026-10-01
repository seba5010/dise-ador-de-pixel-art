"""Crea plantillas vacías 16×4 compatibles con el pipeline del proyecto."""

from pathlib import Path
from PIL import Image, ImageDraw


WIDTH = 724
HEIGHT = 2172
ROWS = 16
COLS = 4
OUT = Path("plantillas_edicion_manual")


def bounds(length: int, divisions: int) -> list[int]:
    return [round(i * length / divisions) for i in range(divisions + 1)]


def draw_grid(image: Image.Image, color: tuple[int, int, int, int], width: int = 2) -> None:
    draw = ImageDraw.Draw(image)
    xs = bounds(WIDTH, COLS)
    ys = bounds(HEIGHT, ROWS)
    for x in xs:
        draw.line((x, 0, x, HEIGHT - 1), fill=color, width=width)
    for y in ys:
        draw.line((0, y, WIDTH - 1, y), fill=color, width=width)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    dark = Image.new("RGBA", (WIDTH, HEIGHT), (31, 35, 43, 255))
    draw_grid(dark, (22, 166, 230, 255), 2)
    dark.save(OUT / "cuadricula_16x4_fondo_oscuro.png")

    transparent = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw_grid(transparent, (22, 166, 230, 210), 2)
    transparent.save(OUT / "cuadricula_16x4_transparente.png")

    blank = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    blank.save(OUT / "lienzo_16x4_transparente_sin_lineas.png")

    print(f"Dimensiones: {WIDTH}x{HEIGHT}")
    print(f"Columnas: {bounds(WIDTH, COLS)}")
    print(f"Filas: {bounds(HEIGHT, ROWS)}")
    for path in sorted(OUT.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
