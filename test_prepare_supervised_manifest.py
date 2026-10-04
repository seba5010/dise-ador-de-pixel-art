import json
from pathlib import Path

from pixel_ai_engine.prepare_supervised_dataset import (
    discover_variants,
    get_canonical_frame_index,
)


def test_maps_manifest_coordinates_to_canonical_indices():
    assert get_canonical_frame_index(
        {"rows": 12, "columns": 8}, {"row": 12, "column": 8, "slot": 96}
    ) == 95
    assert get_canonical_frame_index(
        {"rows": 16, "columns": 4}, {"row": 1, "column": 1, "slot": 1}
    ) == 0
    assert get_canonical_frame_index(
        {"rows": 16, "columns": 4}, {"row": 2, "column": 1, "slot": 5}
    ) == 4
    assert get_canonical_frame_index(
        {"rows": 16, "columns": 4}, {"row": 9, "column": 1, "slot": 33}
    ) == 8
    assert get_canonical_frame_index(
        {"rows": 16, "columns": 4}, {"row": 13, "column": 1, "slot": 49}
    ) == -1


def test_discovery_uses_manifest_and_omits_missing_files(tmp_path: Path, capsys):
    variant_dir = tmp_path / "ALEX" / "alex_ropa_normal"
    variant_dir.mkdir(parents=True)
    (variant_dir / "frame_001_r01_c01.png").touch()
    (variant_dir / "frame_002_r01_c02.png").touch()
    manifest = {
        "character": "alex",
        "variant": "rnormal",
        "variant_name": "ropa_normal",
        "grid": {"rows": 1, "columns": 8},
        "slots": 8,
        "frames_saved": 2,
        "frames": [
            {"slot": 1, "row": 1, "column": 1, "file": "frame_001_r01_c01.png"},
            {"slot": 2, "row": 1, "column": 2, "file": "frame_deleted_r01_c02.png"},
        ],
    }
    (variant_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    variants = discover_variants(tmp_path)

    assert len(variants) == 1
    assert [frame["path"].name for frame in variants[0]["frames"]] == [
        "frame_001_r01_c01.png"
    ]
    assert variants[0]["frames"][0]["frame_idx"] == 0
    output = capsys.readouterr().out
    assert "1 PNG declarados ya no existen" in output
    assert "1 PNG no declarados" in output