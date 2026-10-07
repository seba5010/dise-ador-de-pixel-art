"""
Pruebas unitarias para el módulo de Auditoría Periódica y Auto-Inyección de Ejemplos Difíciles
"""

import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from pixel_ai_engine.hard_examples import HARD_EXAMPLES_FILE, load_manual_sampling_weights
from pixel_ai_engine.periodic_audit import inject_weak_frames_to_hard_examples


def test_inject_weak_frames_to_hard_examples(tmp_path, monkeypatch):
    test_jsonl = tmp_path / "hard_examples_test.jsonl"
    monkeypatch.setattr("pixel_ai_engine.periodic_audit.HARD_EXAMPLES_FILE", test_jsonl)

    dummy_chars_res = [
        {
            "character": "tori",
            "frames": [
                # Frame sano
                {"frame_idx": 10, "raw_anatomy": 90.0, "raw_outline": 95.0, "raw_stray_limbs": 0.0},
                # Frame débil por anatomía
                {"frame_idx": 64, "raw_anatomy": 60.0, "raw_outline": 90.0, "raw_stray_limbs": 0.01},
                # Frame débil por stray limbs
                {"frame_idx": 72, "raw_anatomy": 80.0, "raw_outline": 90.0, "raw_stray_limbs": 0.08},
            ]
        }
    ]

    res = inject_weak_frames_to_hard_examples(
        dummy_chars_res,
        threshold_anatomy=75.0,
        threshold_outline=75.0,
        max_stray_limbs=0.03,
        priority=2.0,
        dataset_character_ids=["tori_rnormal"]
    )

    assert res["total_injected"] == 2
    assert "tori" in res["characters_affected"]
    assert test_jsonl.exists()

    weights = load_manual_sampling_weights(test_jsonl)
    assert "tori_rnormal::frame_064" in weights
    assert weights["tori_rnormal::frame_064"] == 2.0
    assert "tori_rnormal::frame_072" in weights
    assert weights["tori_rnormal::frame_072"] == 2.0
    assert "tori_rnormal::frame_010" not in weights
