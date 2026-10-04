import subprocess
import time
from unittest.mock import patch

import sprite_studio


def reset_cache():
    sprite_studio.CACHED_HW_TELEMETRY.update(
        {
            "gpu_temp": 55,
            "cpu_temp": None,
            "vram_gb": 0.88,
            "vram_total_gb": 4.0,
            "vram_pct": 22.0,
            "gpu_util": 0,
            "gpu_name": "NVIDIA RTX 3050 Ti",
            "last_updated": 0.0,
            "last_attempt": 0.0,
            "source": "fallback",
            "stale": True,
            "last_cpu_check": time.time(),
        }
    )


def nvidia_result(used_mb: int):
    return subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout=f"51, {used_mb}, 4096, 37, NVIDIA GeForce RTX 3050 Ti Laptop GPU\n",
        stderr="",
    )


def test_vram_changes_after_cache_window():
    reset_cache()
    with patch("sprite_studio.subprocess.run", side_effect=[nvidia_result(256), nvidia_result(2048)]) as run_mock:
        first = dict(sprite_studio.update_hardware_telemetry())
        cached = dict(sprite_studio.update_hardware_telemetry())
        sprite_studio.CACHED_HW_TELEMETRY["last_attempt"] = 0.0
        second = dict(sprite_studio.update_hardware_telemetry())

    assert first["vram_gb"] == 0.25
    assert cached["vram_gb"] == 0.25
    assert second["vram_gb"] == 2.0
    assert second["vram_pct"] == 50.0
    assert second["source"] == "nvidia-smi"
    assert second["stale"] is False
    assert run_mock.call_count == 2


def test_failed_refresh_marks_old_sample_as_stale():
    reset_cache()
    sprite_studio.CACHED_HW_TELEMETRY["last_updated"] = time.time() - 10.0
    with patch("sprite_studio.subprocess.run", side_effect=OSError("nvidia-smi no disponible")):
        telemetry = sprite_studio.update_hardware_telemetry()

    assert telemetry["stale"] is True
    assert telemetry["source"] == "fallback"


if __name__ == "__main__":
    test_vram_changes_after_cache_window()
    print("[OK] test_vram_changes_after_cache_window")
    test_failed_refresh_marks_old_sample_as_stale()
    print("[OK] test_failed_refresh_marks_old_sample_as_stale")
