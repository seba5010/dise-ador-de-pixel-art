"""
Script de Pulido de Alta Definición - Fase 1 (Base 16x4)
Inicia desde el mejor modelo de la época 500 (90.1% de calidad) y refina:
- Texturas de ropa y delantal
- Micro-detalles de 1px (tatuajes, pupilas, pliegues)
- Sombreado de píxeles nítidos (discretización sin difuminado suave)
- Peso adversarial controlado (LAMBDA_ADV = 0.15) para máxima estabilidad
"""

import sys
from pathlib import Path

# Asegurar codificación utf-8 en terminal de Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pixel_ai_engine.train import train

if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("      💎 PULIDO DE ALTA DEFINICIÓN - FASE 1 (BASE 16X4)")
    print("      Partiendo del punto óptimo de Época 500 (90.1% de calidad)")
    print("      Meta: 800 épocas (300 épocas de texturizado y micro-detalles)")
    print("      Learning Rate fino: 8e-5 | Peso Adversarial suavizado: 0.15")
    print("=" * 75 + "\n")

    train(
        epochs=800,
        batch_size=8,
        resume=True,
        lr_g=8e-5,
        lr_d=4e-5,
        phase="1"
    )
