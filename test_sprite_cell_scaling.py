import cv2
import numpy as np
from PIL import Image, ImageDraw

from pixel_ai_engine.dataset import (
    CANONICAL_FRONT_REFERENCE_HEIGHT,
    CANONICAL_SPRITE_HEIGHT,
    CANONICAL_SPRITE_MAX_WIDTH,
    TRAINING_FRAME_HEIGHT,
    TRAINING_FRAME_MAX_WIDTH,
    pad_target_frame_canonical,
    place_in_cell,
)
from pixel_ai_engine.resize_spritesheet_cells import CHARACTERS_DIR, find_sheets, scale_cell_sprite


def alpha_bbox(image: Image.Image):
    alpha = np.array(image.convert("RGBA"))[:, :, 3]
    coordinates = np.argwhere(alpha > 20)
    if len(coordinates) == 0:
        return None
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0)
    return int(x0), int(y0), int(x1), int(y1)


def make_sprite(canvas_size=(256, 256), sprite_size=(40, 70)) -> Image.Image:
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    width, height = sprite_size
    x0 = (canvas_size[0] - width) // 2
    y0 = (canvas_size[1] - height) // 2
    draw.rectangle((x0, y0, x0 + width - 1, y0 + height - 1), fill=(220, 80, 120, 255))
    return canvas


def test_place_in_cell_enlarges_and_preserves_margins():
    result = place_in_cell(make_sprite())
    bbox = alpha_bbox(result)
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert y1 - y0 + 1 == CANONICAL_SPRITE_HEIGHT
    assert x0 >= 6 and x1 <= 121
    assert y0 >= 6 and y1 <= 120


def test_preview_uses_character_bounds_instead_of_shrinking_canvas():
    source = make_sprite(canvas_size=(256, 256), sprite_size=(52, 104))
    direct_resize = source.resize((128, 128), Image.Resampling.NEAREST)
    fitted_preview = place_in_cell(source)

    direct_bbox = alpha_bbox(direct_resize)
    fitted_bbox = alpha_bbox(fitted_preview)
    assert direct_bbox is not None
    assert fitted_bbox is not None
    assert direct_bbox[3] - direct_bbox[1] + 1 == 52
    assert fitted_bbox[3] - fitted_bbox[1] + 1 == CANONICAL_SPRITE_HEIGHT


def test_front_reference_has_high_detail_scale_and_is_centered():
    assert CANONICAL_FRONT_REFERENCE_HEIGHT == 208
    top = 232 - CANONICAL_FRONT_REFERENCE_HEIGHT
    bottom_margin = 256 - 232
    assert top == bottom_margin


def test_place_in_cell_limits_wide_poses():
    result = place_in_cell(make_sprite(sprite_size=(140, 50)))
    bbox = alpha_bbox(result)
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert x1 - x0 + 1 == CANONICAL_SPRITE_MAX_WIDTH
    assert 0 <= x0 <= x1 < 128
    assert 0 <= y0 <= y1 < 128


def test_pad_target_frame_uses_canonical_scale():
    result = pad_target_frame_canonical(make_sprite())
    bbox = alpha_bbox(result)
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert y1 - y0 + 1 == TRAINING_FRAME_HEIGHT
    assert x1 - x0 + 1 <= TRAINING_FRAME_MAX_WIDTH
    assert abs((x0 + x1) - 255) <= 1
    assert abs((y0 + y1) - 255) <= 1
    assert 0 <= x0 <= x1 < 256
    assert 0 <= y0 <= y1 < 256


def test_tiny_artifacts_are_discarded():
    canvas = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    canvas.putpixel((128, 128), (255, 255, 255, 255))
    assert alpha_bbox(place_in_cell(canvas)) is None
    assert alpha_bbox(pad_target_frame_canonical(canvas)) is None


def test_source_cell_scaling_stays_centered_inside_cell():
    source = make_sprite(canvas_size=(128, 128), sprite_size=(30, 50))
    result, source_size, target_size = scale_cell_sprite(source)
    bbox = alpha_bbox(result)
    assert source_size == (30, 50)
    assert target_size[1] == 112
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert abs((x0 + x1) - 127) <= 1
    assert abs((y0 + y1) - 127) <= 1
    assert 0 <= x0 <= x1 < 128
    assert 0 <= y0 <= y1 < 128


def test_every_movement_sheet_is_included_in_scaling_pipeline():
    expected = {
        path.resolve()
        for path in CHARACTERS_DIR.glob("*/*.png")
        if "movimiento" in path.name.lower()
    }
    assert {path.resolve() for path in find_sheets()} == expected


def test_conny_frame_027_has_no_detached_floor_artifact():
    sheet = Image.open(CHARACTERS_DIR / "conny" / "movimientos_rnormal.png").convert("RGBA")
    width, height = sheet.size
    cell = sheet.crop(
        (
            round(3 * width / 4),
            round(10 * height / 16),
            width,
            round(11 * height / 16),
        )
    )
    mask = (np.array(cell)[:, :, 3] > 0).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    significant_components = [
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, component_count)
        if stats[label, cv2.CC_STAT_AREA] >= 20
    ]
    assert len(significant_components) == 1


def test_conny_frame_043_has_complete_rounded_head():
    sheet = Image.open(CHARACTERS_DIR / "conny" / "movimientos_rnormal.png").convert("RGBA")
    width, height = sheet.size
    cell = sheet.crop(
        (
            round(3 * width / 4),
            round(11 * height / 16),
            width,
            round(12 * height / 16),
        )
    )
    mask = np.array(cell)[:, :, 3] > 20
    coordinates = np.argwhere(mask)
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0)
    head_row_widths = [int(mask[row].sum()) for row in range(y0, min(y0 + 30, y1 + 1))]

    assert y1 - y0 + 1 == 112
    assert abs((x0 + x1) - (width / 4 - 1)) <= 1
    assert abs((y0 + y1) - (height / 16 - 1)) <= 1
    assert head_row_widths[0] <= max(head_row_widths) * 0.35


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
