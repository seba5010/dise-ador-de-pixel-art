import json

import torch
from PIL import Image

from pixel_ai_engine import train_supervised


class FakeGenerator:
    def __init__(self):
        self.batch_sizes = []

    def eval(self):
        return self

    def __call__(self, conditions):
        self.batch_sizes.append(conditions.shape[0])
        return torch.zeros((conditions.shape[0], 4, 2, 2))


class FakeTemplateManager:
    def get_frame_tensor(self, frame_idx):
        return torch.full((3, 2, 2), float(frame_idx))


def test_all_frame_comparison_covers_sorted_records_in_batches(tmp_path, monkeypatch):
    samples = [
        {"char_id": "zeta", "frame_idx": 1, "front_tensor": torch.zeros(3, 2, 2), "target_tensor": torch.zeros(4, 2, 2)},
        {"char_id": "alpha", "frame_idx": 10, "front_tensor": torch.zeros(3, 2, 2), "target_tensor": torch.zeros(4, 2, 2)},
        {"char_id": "alpha", "frame_idx": 2, "front_tensor": torch.zeros(3, 2, 2), "target_tensor": torch.zeros(4, 2, 2)},
        {"char_id": "zeta", "frame_idx": 4, "front_tensor": torch.zeros(3, 2, 2), "target_tensor": torch.zeros(4, 2, 2)},
        {"char_id": "middle", "frame_idx": 7, "front_tensor": torch.zeros(3, 2, 2), "target_tensor": torch.zeros(4, 2, 2)},
    ]
    monkeypatch.setattr(train_supervised, "_tensor_to_preview_image", lambda tensor, has_alpha=False: Image.new("RGBA" if has_alpha else "RGB", (2, 2)))
    monkeypatch.setattr(train_supervised, "_fit_preview_cell", lambda image: Image.new("RGBA", (128, 128)))
    monkeypatch.setattr(train_supervised, "extract_character_palette", lambda image, include_props=True: [])
    monkeypatch.setattr(train_supervised, "remap_image_to_palette", lambda image, palette, **kwargs: image)
    monkeypatch.setattr(train_supervised, "clean_orphan_pixels", lambda image, **kwargs: image)
    generator = FakeGenerator()

    manifest = train_supervised._generate_all_frame_comparison(
        generator,
        samples,
        FakeTemplateManager(),
        output_dir=tmp_path,
        epoch_label=12,
        batch_size=2,
        rows_per_chunk=2,
    )

    assert manifest["row_count"] == 3
    assert manifest["rows"] == [
        {"char_id": "alpha", "frame_idx": 2},
        {"char_id": "middle", "frame_idx": 7},
        {"char_id": "zeta", "frame_idx": 1},
    ]
    assert generator.batch_sizes == [2, 1]
    assert len(manifest["chunks"]) == 2
    assert all((tmp_path / filename).is_file() for filename in manifest["chunks"])
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest