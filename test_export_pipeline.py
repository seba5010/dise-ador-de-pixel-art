import pytest
from pathlib import Path
from export_final_spritesheets import find_all_characters, get_best_available_checkpoint


def test_find_all_characters_locates_roster():
    chars = find_all_characters()
    assert len(chars) > 0
    # Mauricio, tori, etc should be present
    names = [c.parent.name for c in chars]
    assert "mauricio" in names or "belial" in names


def test_get_best_available_checkpoint_finds_valid_file():
    ckpt = get_best_available_checkpoint()
    assert ckpt.exists()
    assert ckpt.suffix == ".pt"
