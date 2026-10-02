"""
Script de verificación rápida del pipeline:
Ejecuta 1 época de entrenamiento para validar CUDA, mixed precision, VRAM y genera una hoja de prueba.
"""
import sys
from pathlib import Path

# Añadir directorio actual
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pixel_ai_engine.train import train
from pixel_ai_engine.generate_character_sheet import generate_spritesheet
from pixel_ai_engine.config import CHECKPOINT_DIR, OUTPUT_DIR

def run_smoke_test():
    print("Iniciando prueba rápida de verificación...")
    # Entrenar 1 época para verificar gradientes, loss y guardado
    train(epochs=1, batch_size=8)
    
    # Probar inferencia con belial_rnormal.png
    input_test = Path("belial/belial_rnormal.png")
    ckpt_test = CHECKPOINT_DIR / "latest_checkpoint.pt"
    out_test = OUTPUT_DIR / "test_smoke_belial_sheet.png"
    
    print("\nProbando generación con checkpoint de prueba...")
    generate_spritesheet(
        input_image_path=input_test,
        checkpoint_path=ckpt_test,
        output_image_path=out_test
    )
    print("\n¡Prueba de verificación superada exitosamente!")

if __name__ == "__main__":
    run_smoke_test()
