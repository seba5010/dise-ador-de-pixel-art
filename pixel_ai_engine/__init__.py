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
    MinibatchStdDev,
    LocalFaceDiscriminator,
    AppearanceFlowModule,
    PaletteHistogramLoss,
    DiscretePixelCodebook,
    MultiScaleDiscriminator,
    FeatureMatchingLoss,
    LaplacianPyramidLoss,
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
from .frame_quality_review import FrameQualityReviewManager, VALID_REVIEW_STATUSES
from .frame_regeneration import FrameRegenerationManager, rank_regeneration_candidates
from .hard_examples import HardExampleQueue, load_manual_sampling_weights
from .interactive_qc import InteractiveQualityControlService
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

# Módulos de la arquitectura avanzada de 12 técnicas
from .fractional_downscale import fractional_silhouette_downscale
from .pixel_art_fixer import snap_and_fix_pixel_art
from .head_rigging import HeadRiggingManager
from .semantic_molds import create_semantic_part_map
from .salience_pixelization import prepare_salience_preserved_front
from .cell_pipeline import SingleCellCoordinator

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
    "FrameQualityReviewManager",
    "FrameRegenerationManager",
    "HardExampleQueue",
    "InteractiveQualityControlService",
    "VALID_REVIEW_STATUSES",
    "load_manual_sampling_weights",
    "rank_regeneration_candidates",
    "QualityGuidanceController",
    "QualityGuidanceConfig",
    "QualityTrendAnalyzer",
    "QualityVector",
    "build_quality_vector",
    "diagnose_quality_bottleneck",
    "generate_spritesheet",
    "train",
    # 12 técnicas
    "fractional_silhouette_downscale",
    "snap_and_fix_pixel_art",
    "HeadRiggingManager",
    "create_semantic_part_map",
    "prepare_salience_preserved_front",
    "SingleCellCoordinator",
    "LocalFaceDiscriminator",
    "AppearanceFlowModule",
    "PaletteHistogramLoss",
    "DiscretePixelCodebook",
    "MultiScaleDiscriminator",
    "FeatureMatchingLoss",
    "LaplacianPyramidLoss",
]
