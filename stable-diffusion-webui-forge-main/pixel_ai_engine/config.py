"""
Pixel AI Engine - Configuration Module
Optimizado para GPUs NVIDIA con 4 GB de VRAM (e.g. RTX 3050 Ti).
Formato de Spritesheet: 1365x2048 px (8 cols x 12 filas = 96 frames, celdas 170x170 px).
"""

import os
import sys
from pathlib import Path
import torch

# Rutas base
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT
PERSONAJES_DIR = PROJECT_ROOT / "personajes"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "output"
SAMPLES_DIR = PROJECT_ROOT / "training_samples"

# Crear directorios clave si no existen
for d in [CHECKPOINT_DIR, OUTPUT_DIR, SAMPLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FASE 1: Plantilla Chica (4 columnas x 16 filas = 64 frames)
# Diseños base: Conny, Dana (y caminatas limpias de otros personajes)
# ============================================================================
PHASE1_TEMPLATE_PATH = PROJECT_ROOT / "plantilla_16x4.png"
PHASE1_CANVAS_WIDTH  = 682
PHASE1_CANVAS_HEIGHT = 2048
PHASE1_GRID_ROWS     = 16
PHASE1_GRID_COLS     = 4
PHASE1_TOTAL_FRAMES  = 64
PHASE1_CELL_WIDTH    = PHASE1_CANVAS_WIDTH // PHASE1_GRID_COLS   # ~170 px
PHASE1_CELL_HEIGHT   = PHASE1_CANVAS_HEIGHT // PHASE1_GRID_ROWS  # 128 px
PHASE1_ROW_NAMES = [
    "01_caminata_frente_idle",
    "02_caminata_frente_pasos",
    "03_caminata_espalda_idle",
    "04_caminata_espalda_pasos",
    "05_caminata_lateral_derecho_idle",
    "06_caminata_lateral_derecho_pasos",
    "07_caminata_lateral_izquierdo_idle",
    "08_caminata_lateral_izquierdo_pasos",
    "09_diagonal_frontal_derecha",
    "10_diagonal_frontal_izquierda",
    "11_diagonal_trasera_derecha",
    "12_diagonal_trasera_izquierda",
    "13_reposo_respiracion",
    "14_acciones_estaticas",
    "15_variaciones_manos",
    "16_expresiones_cierre",
]

# ============================================================================
# FASE 2: Plantilla Grande Detallada (8 columnas x 12 filas = 96 frames)
# Transfer Learning: Alex, Amaro (cocina, pensar, cargar caja, servir, celebrar)
# ============================================================================
PHASE2_TEMPLATE_PATH = PROJECT_ROOT / "plantilla de los spritesheets.png"
PHASE2_CANVAS_WIDTH  = 1024
PHASE2_CANVAS_HEIGHT = 1536
PHASE2_GRID_ROWS     = 12
PHASE2_GRID_COLS     = 8
PHASE2_TOTAL_FRAMES  = 96
PHASE2_CELL_WIDTH    = PHASE2_CANVAS_WIDTH // PHASE2_GRID_COLS   # 128 px
PHASE2_CELL_HEIGHT   = PHASE2_CANVAS_HEIGHT // PHASE2_GRID_ROWS  # 128 px
PHASE2_ROW_NAMES = [
    "01_frente_sur",
    "02_diagonal_frontal_derecha_sureste",
    "03_lateral_derecho_este",
    "04_diagonal_trasera_derecha_noreste",
    "05_espalda_norte",
    "06_diagonal_trasera_izquierda_noroeste",
    "07_lateral_izquierdo_oeste",
    "08_diagonal_frontal_izquierda_suroeste",
    "09_cocina_bowl_frente_espalda",
    "10_cocina_bowl_laterales",
    "11_pensar_cargar_caja",
    "12_servir_plato_celebracion",
]

# Por defecto activo: FASE 2 CANONICA (8 cols x 12 filas = 96 frames, 1024x1536)
TEMPLATE_PATH = PHASE2_TEMPLATE_PATH
CANVAS_WIDTH  = PHASE2_CANVAS_WIDTH
CANVAS_HEIGHT = PHASE2_CANVAS_HEIGHT
GRID_ROWS     = PHASE2_GRID_ROWS
GRID_COLS     = PHASE2_GRID_COLS
TOTAL_FRAMES  = PHASE2_TOTAL_FRAMES
CELL_WIDTH    = PHASE2_CELL_WIDTH
CELL_HEIGHT   = PHASE2_CELL_HEIGHT
ROW_ANIMATION_NAMES = PHASE2_ROW_NAMES

# Tolerancia para auto-detección de formato de sheet (±35%)
SHEET_CELL_TOLERANCE = 0.35

def get_phase_config(phase: str):
    """Devuelve la configuración específica según la fase ('1'/'16x4' o '2'/'8x12')."""
    p = str(phase).strip().lower()
    if p in ["2", "8x12", "fase2"]:
        return {
            "phase": "2",
            "phase_name": "Fase 2 (Transferencia 8x12 - 96 Frames)",
            "template_path": PHASE2_TEMPLATE_PATH,
            "canvas_w": PHASE2_CANVAS_WIDTH,
            "canvas_h": PHASE2_CANVAS_HEIGHT,
            "grid_rows": PHASE2_GRID_ROWS,
            "grid_cols": PHASE2_GRID_COLS,
            "total_frames": PHASE2_TOTAL_FRAMES,
            "cell_w": PHASE2_CELL_WIDTH,
            "cell_h": PHASE2_CELL_HEIGHT,
            "row_names": PHASE2_ROW_NAMES,
            "ckpt_latest": CHECKPOINT_DIR / "latest_checkpoint.pt",
            "ckpt_best": CHECKPOINT_DIR / "best_generator.pt",
            "cache_file": CHECKPOINT_DIR / "dataset_cache_256_8x12.pt"
        }
    else:
        return {
            "phase": "1",
            "phase_name": "Fase 1 (Base 16x4 - 64 Frames)",
            "template_path": PHASE1_TEMPLATE_PATH,
            "canvas_w": PHASE1_CANVAS_WIDTH,
            "canvas_h": PHASE1_CANVAS_HEIGHT,
            "grid_rows": PHASE1_GRID_ROWS,
            "grid_cols": PHASE1_GRID_COLS,
            "total_frames": PHASE1_TOTAL_FRAMES,
            "cell_w": PHASE1_CELL_WIDTH,
            "cell_h": PHASE1_CELL_HEIGHT,
            "row_names": PHASE1_ROW_NAMES,
            "ckpt_latest": CHECKPOINT_DIR / "latest_checkpoint_16x4.pt",
            "ckpt_best": CHECKPOINT_DIR / "base_generator_16x4.pt",
            "cache_file": CHECKPOINT_DIR / "dataset_cache_256_16x4.pt"
        }

# Configuración de Deep Learning / Resolución
# 256x256 nativa: preserva 100% de los píxeles reales de tatuajes, ojos y sombreados
MODEL_RESOLUTION = 256
IN_CHANNELS = 6   # 3 canales Identidad (Personaje Frontal) + 3 canales Pose (Maniquí Plantilla)
OUT_CHANNELS = 4  # 4 canales (RGBA: R, G, B + Máscara Alfa estricta de recorte)

# Hardware & VRAM 4GB RTX 3050 Optimization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()  # Automatic Mixed Precision (float16)

# Hiperparámetros de Entrenamiento
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
NUM_WORKERS = 0  # 0 recomendado en Windows para evitar overhead de subprocesos
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 1e-4
BETA1 = 0.5
BETA2 = 0.999
WEIGHT_DECAY = 1e-5

# Pesos de la Función de Pérdida Personalizada
LAMBDA_L1 = 100.0     # Reconstrucción de color y silueta
LAMBDA_EDGE = 45.0    # Reforzado para líneas de 1px duras en tatuajes y contornos
LAMBDA_PERC = 2.0     # Pérdida perceptiva VGG-16 para coherencia semántica
LAMBDA_ADV = 0.15     # Pérdida adversarial suavizada para evitar colapso modal

EPOCHS = 200
SAVE_EVERY_EPOCHS = 10
SAMPLE_EVERY_EPOCHS = 5
