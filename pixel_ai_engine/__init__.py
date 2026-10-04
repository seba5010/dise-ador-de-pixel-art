"""
Pixel AI Engine - Paquete de Redes Neuronales, Control de Calidad y Generacion de Spritesheets
"""

from .config import (
    PROJECT_ROOT,
    PERSONAJES_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    SAMPLES_DIR,
    DEVICE,
    USE_AMP,
    get_phase_config
)
from .models import (
    PixelArtUNetGenerator,
    PixelArtPatchDiscriminator,
    PixelArtLoss,
    SilhouetteAlignmentLoss,
    MinibatchStdDev
)
from .dataset import (
    PixelArtDataset,
    TemplateManager,
    isolate_character,
    pad_to_square,
    unpad_from_square,
    place_in_cell,
    get_dataloader
)
from .enhancer import PixelArtEnhancer
from .phase3_critical_enhancer import Phase3CriticalReviewer
from .quality_gate import QualityGate
from .quality_guidance import (
    QualityGuidanceConfig,
    QualityGuidanceController,
    QualityTrendAnalyzer,
    QualityVector,
    build_quality_vector,
    diagnose_quality_bottleneck,
)
from .generate_character_sheet import generate_spritesheet
from .train import train

__all__ = [
    "PROJECT_ROOT",
    "PERSONAJES_DIR",
    "CHECKPOINT_DIR",
    "OUTPUT_DIR",
    "SAMPLES_DIR",
    "DEVICE",
    "USE_AMP",
    "get_phase_config",
    "PixelArtUNetGenerator",
    "PixelArtPatchDiscriminator",
    "PixelArtLoss",
    "SilhouetteAlignmentLoss",
    "MinibatchStdDev",
    "PixelArtDataset",
    "TemplateManager",
    "isolate_character",
    "pad_to_square",
    "unpad_from_square",
    "place_in_cell",
    "get_dataloader",
    "PixelArtEnhancer",
    "Phase3CriticalReviewer",
    "QualityGate",
    "QualityGuidanceController",
    "QualityGuidanceConfig",
    "QualityTrendAnalyzer",
    "QualityVector",
    "build_quality_vector",
    "diagnose_quality_bottleneck",
    "generate_spritesheet",
    "train",
]
