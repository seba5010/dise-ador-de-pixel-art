"""
Entrypoint principal para entrenar el modelo de Pixel Art (Motor Canónico Supervisado PyTorch UNet).
"""
import sys
import argparse
from pathlib import Path

# Añadir directorio actual al path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pixel_ai_engine.train_supervised import repair_current_training_state, train_supervised_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento Supervisado de Pixel Art Spritesheets con 4 Candados")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas a entrenar")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=4, help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Tasa de aprendizaje (Learning Rate)")
    parser.add_argument("--mode", type=str, default="resume", choices=["start", "resume"], help="Modo de entrenamiento (start o resume)")
    parser.add_argument("--respawn_epoch", type=int, default=None, help="Época exacta a la cual rebobinar (Respawn)")
    parser.add_argument("--repair_state", action="store_true", help="Prepara una ruta segura de recuperacion sin entrenar")
    args = parser.parse_args()

    if args.repair_state:
        result = repair_current_training_state()
        if result is None:
            print("[OK] No se detecto una degradacion sostenida que requiera reparacion.")
        else:
            print(
                f"[OK] Recuperacion preparada: epoca {result['source_epoch']} | "
                f"checkpoint={result['checkpoint']} | LR={result['preferred_lr']}"
            )
        raise SystemExit(0)
    
    train_supervised_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mode=args.mode,
        respawn_epoch=args.respawn_epoch
    )
