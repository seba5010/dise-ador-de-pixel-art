"""
Entrypoint principal para entrenar el modelo de Pixel Art
"""
import sys
from pathlib import Path

# Añadir directorio actual al path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pixel_ai_engine.train import train, EPOCHS, BATCH_SIZE
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar Generador de Pixel Art Spritesheets")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Número de épocas. Usa 0 para modo infinito.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Tamaño del batch")
    parser.add_argument("--resume", action="store_true", help="Reanudar desde último checkpoint")
    parser.add_argument("--infinite", action="store_true", help="Entrena sin límite de épocas hasta Ctrl+C")
    parser.add_argument("--phase", type=str, default="1", choices=["1", "2", "16x4", "8x12"], help="Fase: 1 (base 16x4) o 2 (transferencia 8x12)")
    parser.add_argument("--transfer", type=str, default="", help="Ruta al checkpoint de Fase 1 para transferir a Fase 2")
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=args.resume,
        infinite=args.infinite,
        phase=args.phase,
        transfer_from=args.transfer if args.transfer else None
    )
