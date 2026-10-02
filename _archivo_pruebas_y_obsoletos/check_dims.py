from PIL import Image
import os

# Verificar plantillas
p1 = "plantilla_16x4.png"
p2 = "plantilla de los spritesheets.png"
for p in [p1, p2]:
    if os.path.exists(p):
        img = Image.open(p)
        print(f"OK: {p} -> {img.size}")
    else:
        print(f"FALTA: {p}")

# Verificar Conny y Dana
for char in ["conny", "dana"]:
    d = f"personajes/{char}"
    if os.path.exists(d):
        files = os.listdir(d)
        print(f"\n{char}:")
        for f in files:
            img = Image.open(f"personajes/{char}/{f}")
            print(f"  {f} -> {img.size}")
    else:
        print(f"\nFALTA carpeta: {d}")
