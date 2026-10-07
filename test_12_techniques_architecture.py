"""
Unit tests for the 12 Architecture Techniques in pixel_ai_engine.
"""

import pytest
import torch
import numpy as np
from PIL import Image

from pixel_ai_engine.fractional_downscale import fractional_silhouette_downscale
from pixel_ai_engine.pixel_art_fixer import snap_and_fix_pixel_art
from pixel_ai_engine.head_rigging import HeadRiggingManager
from pixel_ai_engine.semantic_molds import create_semantic_part_map, PART_COLORS
from pixel_ai_engine.salience_pixelization import prepare_salience_preserved_front
from pixel_ai_engine.cell_pipeline import SingleCellCoordinator
from pixel_ai_engine.models import (
    LocalFaceDiscriminator,
    AppearanceFlowModule,
    PaletteHistogramLoss,
    DiscretePixelCodebook,
    MultiScaleDiscriminator,
    FeatureMatchingLoss,
    LaplacianPyramidLoss,
)


def test_fractional_silhouette_downscale():
    # Crear imagen de prueba 64x64 con fondo transparente y rasgo fino oscuro (ej. gafas)
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    arr = np.array(img)
    # Cuerpo de piel clara
    arr[20:50, 20:44] = [230, 190, 150, 255]
    # Gafas negras de 1px
    arr[24:26, 22:42] = [10, 10, 15, 255]
    test_img = Image.fromarray(arr, mode="RGBA")

    # Reducir a 16x16
    downscaled = fractional_silhouette_downscale(test_img, 16, 16)
    assert downscaled.size == (16, 16)
    out_arr = np.array(downscaled)
    # Verificar que el rasgo oscuro sobrevive y no se borra
    dark_pixels = (out_arr[:, :, 3] > 0) & (out_arr[:, :, 0] < 80)
    assert np.any(dark_pixels)


def test_pixel_art_fixer():
    # Imagen con semitransparencias y píxel aislado
    img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    arr = np.array(img)
    arr[10:25, 10:25] = [200, 100, 50, 180]  # semitransparente
    arr[2, 2] = [255, 0, 0, 255]  # píxel huérfano
    test_img = Image.fromarray(arr, mode="RGBA")

    fixed = snap_and_fix_pixel_art(test_img, alpha_threshold=50, enforce_dark_outline=True)
    out_arr = np.array(fixed)

    # El alfa debe ser estrictamente 0 o 255
    alphas = np.unique(out_arr[:, :, 3])
    for a in alphas:
        assert a in (0, 255)

    # El píxel aislado en (2, 2) debe haber sido eliminado
    assert out_arr[2, 2, 3] == 0


def test_head_rigging():
    # Frontal sintético con cabeza definida
    front = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    arr_f = np.array(front)
    arr_f[10:20, 22:42] = [40, 25, 15, 255]   # pelo
    arr_f[20:30, 24:40] = [220, 180, 140, 255] # cara
    arr_f[30:55, 20:44] = [50, 80, 160, 255]  # ropa
    front_img = Image.fromarray(arr_f, mode="RGBA")

    head, bounds = HeadRiggingManager.extract_canonical_head(front_img)
    assert head.size[0] > 0 and head.size[1] > 0

    # Molde de pose sintético
    pose = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    arr_p = np.array(pose)
    arr_p[12:28, 24:40] = [120, 120, 120, 255] # cabeza pose
    arr_p[28:56, 20:44] = [160, 160, 160, 255] # cuerpo pose
    pose_img = Image.fromarray(arr_p, mode="RGBA")

    composite = HeadRiggingManager.composite_head_onto_frame(
        front_img, head, pose_img, direction_row=0
    )
    assert composite.size == (64, 64)


def test_semantic_molds():
    pose = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    arr_p = np.array(pose)
    arr_p[10:55, 20:44] = [150, 150, 150, 255]
    pose_img = Image.fromarray(arr_p, mode="RGBA")

    sem_map = create_semantic_part_map(pose_img, row_idx=0)
    assert sem_map.size == (64, 64)
    sem_arr = np.array(sem_map)

    # Debe contener rojo (cabeza) y verde (torso)
    assert np.any(np.all(sem_arr == PART_COLORS["head"], axis=-1))
    assert np.any(np.all(sem_arr == PART_COLORS["torso"], axis=-1))


def test_salience_pixelization():
    raw_front = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    arr = np.array(raw_front)
    arr[20:110, 35:95] = [200, 150, 100, 255]
    # Gafas con contraste fuerte
    arr[40:46, 45:85] = [10, 10, 10, 255]
    img = Image.fromarray(arr, mode="RGBA")

    salient_prep = prepare_salience_preserved_front(img, target_size=64)
    assert salient_prep.size == (64, 64)


def test_cell_pipeline():
    coord = SingleCellCoordinator(rows=2, cols=2, cell_w=32, cell_h=32, canvas_w=64, canvas_h=64)
    frames = [Image.new("RGBA", (32, 32), (100, 150, 200, 255)) for _ in range(4)]
    sheet = coord.assemble_cells_into_sheet(frames)
    assert sheet.size == (64, 64)


def test_neural_components():
    # 1. LocalFaceDiscriminator
    disc = LocalFaceDiscriminator(in_channels=4)
    dummy_rgba = torch.randn(2, 4, 64, 64)
    out_logits = disc(dummy_rgba)
    assert out_logits.ndim == 4
    # Test crop helper
    crop = LocalFaceDiscriminator.crop_head_patch(dummy_rgba, patch_size=32)
    assert crop.shape == (2, 4, 32, 32)

    # 2. AppearanceFlowModule
    flow_mod = AppearanceFlowModule(in_channels=6)
    source_rgb = torch.randn(2, 3, 32, 32)
    condition = torch.randn(2, 6, 32, 32)
    warped, flow = flow_mod(source_rgb, condition)
    assert warped.shape == (2, 3, 32, 32)
    assert flow.shape == (2, 2, 32, 32)

    # 3. PaletteHistogramLoss
    hist_loss_fn = PaletteHistogramLoss(num_subregions=4)
    pred_rgb = torch.randn(2, 3, 32, 32)
    target_rgb = torch.randn(2, 3, 32, 32)
    loss_val = hist_loss_fn(pred_rgb, target_rgb)
    assert loss_val.ndim == 0 and not torch.isnan(loss_val)

    # 4. DiscretePixelCodebook
    codebook = DiscretePixelCodebook(num_embeddings=32, embedding_dim=16)
    dummy_feat = torch.randn(2, 16, 8, 8)
    quantized, c_loss = codebook(dummy_feat)
    assert quantized.shape == (2, 16, 8, 8)
    assert c_loss.ndim == 0 and not torch.isnan(c_loss)

    # 5. MultiScaleDiscriminator & FeatureMatchingLoss
    multi_disc = MultiScaleDiscriminator(in_channels=7, num_scales=2)
    cond7 = torch.randn(2, 3, 64, 64)
    target4 = torch.randn(2, 4, 64, 64)
    multi_outs = multi_disc(cond7, target4)
    assert len(multi_outs) == 2

    fm_loss_fn = FeatureMatchingLoss()
    fm_loss = fm_loss_fn(multi_outs, multi_outs)
    assert fm_loss.item() == 0.0

    # 6. LaplacianPyramidLoss
    lap_pyr = LaplacianPyramidLoss(num_levels=3)
    lap_loss = lap_pyr(pred_rgb, target_rgb)
    assert lap_loss.ndim == 0 and not torch.isnan(lap_loss)
