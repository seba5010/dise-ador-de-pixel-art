import os
import sys
import json
import math
import time
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import pixel_ai_engine.train_supervised as train_mod
from pixel_ai_engine.models import PixelArtUNetGenerator, PixelArtPatchDiscriminator
from pixel_ai_engine.train_supervised import (
    update_status,
    save_checkpoint
)
from sprite_studio import run_quality_audit

def test_nan_handling():
    print("\n--- TEST 1: NaN e Infinitos ---")
    tmp_dir = tempfile.mkdtemp()
    temp_status = Path(tmp_dir) / "training_status.json"
    orig_status_file = train_mod.STATUS_FILE
    train_mod.STATUS_FILE = temp_status

    try:
        t0 = time.time()
        # 1.1 Test update_status with NaN and ensure allow_nan=False works
        update_status(
            epoch=3,
            total_epochs=10,
            status_str="ERROR_NAN",
            g_loss=float("nan"),
            d_loss=float("inf"),
            l1_val=float("nan"),
            edge_val=0.01,
            start_time=t0,
            lr_val=1.5e-4,
            error_details={"epoch": 3, "batch": 12, "metric": "l1_color", "value": "nan"},
            skipped_amp=2
        )

        with open(temp_status, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["status"] == "ERROR_NAN", f"Status should be ERROR_NAN, got {data['status']}"
        assert data["g_loss"] is None, f"g_loss should be None, got {data['g_loss']}"
        assert data["d_loss"] is None, f"d_loss should be None, got {data['d_loss']}"
        assert data["l1_loss"] is None, f"l1_loss should be None, got {data['l1_loss']}"
        assert data["edge_loss"] == 0.01
        assert data["error_details"]["metric"] == "l1_color"
        assert data["skipped_amp_steps"] == 2
        print("[PASS] 1.1: update_status serializa JSON con allow_nan=False y convierte NaN/Inf a null.")

        # 1.2 Verify that raw allow_nan=False in json.dump would fail on actual NaN
        raw_dict = {"val": data["g_loss"]}
        json.dumps(raw_dict, allow_nan=False)
        print("[PASS] 1.2: allow_nan=False verificado con valores None/null.")

        # 1.3 Checkpoint protection on NaN
        ckpt_dir = Path(tmp_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = ckpt_dir / "best_generator.pt"
        
        dummy_gen = PixelArtUNetGenerator()
        torch.save({"generator": dummy_gen.state_dict(), "best_loss": 0.08}, best_ckpt)
        initial_mtime = os.path.getmtime(best_ckpt)

        # Calling update_status with status_str="COMPLETADO" while loss is NaN must force ERROR_NAN and not publish 0.0
        update_status(
            epoch=10,
            total_epochs=10,
            status_str="COMPLETADO",
            g_loss=float("nan"),
            d_loss=0.02,
            l1_val=None,
            edge_val=None,
            start_time=t0
        )
        with open(temp_status, "r", encoding="utf-8") as f:
            data2 = json.load(f)
        assert data2["g_loss"] is None
        assert data2["status"] == "ERROR_NAN", f"Invalid g_loss should force ERROR_NAN instead of COMPLETADO, got {data2['status']}"
        assert data2["g_loss"] != 0.0, "g_loss must never be published as 0.0 when invalid"
        assert os.path.getmtime(best_ckpt) == initial_mtime, "Checkpoint should not be modified on NaN"
        print("[PASS] 1.3: Checkpoint previo protegido y nunca se reporta 0.0 falso ni COMPLETADO ante NaN.")
    finally:
        train_mod.STATUS_FILE = orig_status_file


def _create_synthetic_run(base_dir: Path, run_name: str, frame_fn, total_frames=64, missing_frames=None):
    run_dir = base_dir / run_name
    enh_dir = run_dir / "enhanced_frames"
    enh_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"format": "16x4", "total_frames": total_frames}, f)

    missing = set(missing_frames or [])
    for idx in range(total_frames):
        if idx in missing:
            continue
        im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        frame_fn(im, idx)
        im.save(enh_dir / f"frame_{idx:03d}.png")
    return run_dir


def test_alpha_certification():
    print("\n--- TEST 2: Certificación del canal Alfa ---")
    tmp_dir = Path(tempfile.mkdtemp())
    
    # 2.1 Caso A: Contenido opaco (alfa 255), centrado y con márgenes -> APROBADO
    def draw_opaque(im, idx):
        # 32x32 square inside 64x64 (leaving 16px margins on all 4 sides)
        for y in range(16, 48):
            for x in range(16, 48):
                im.putpixel((x, y), (220, 50, 50, 255))
    run_a = _create_synthetic_run(tmp_dir, "run_opaque", draw_opaque)
    audit_a = run_quality_audit(run_a, format_type="16x4")
    assert audit_a["certified"] is True, f"Opaque with margins should be certified, got: {audit_a}"
    assert audit_a["alpha_purity_score"] == 100.0
    assert len(audit_a["blurry_alpha_cells"]) == 0
    print("[PASS] 2.1: Contenido opaco, centrado y con márgenes: APROBADO (certified=True, pureza=100%).")

    # 2.2 Caso B: Contenido completamente semitransparente con alfa 128 (ghost) -> RECHAZADO
    def draw_ghost(im, idx):
        # 32x32 square with alpha=128 (no single opaque pixel)
        for y in range(16, 48):
            for x in range(16, 48):
                im.putpixel((x, y), (200, 50, 50, 128))
    run_b = _create_synthetic_run(tmp_dir, "run_ghost", draw_ghost)
    audit_b = run_quality_audit(run_b, format_type="16x4")
    assert audit_b["certified"] is False, f"Ghost alpha 128 must be rejected, got: {audit_b}"
    assert len(audit_b["blurry_alpha_cells"]) == 64, f"All 64 cells should be in blurry_alpha_cells, got {len(audit_b['blurry_alpha_cells'])}"
    assert audit_b["alpha_purity_score"] == 4.0, f"Alpha purity expected 4.0 (100 - 64*1.5), got {audit_b['alpha_purity_score']}"
    print(f"[PASS] 2.2: Contenido 100% semitransparente (alfa 128): RECHAZADO (blurry_alpha_cells={len(audit_b['blurry_alpha_cells'])}, certified=False).")

    # 2.3 Caso C: Mezcla de alfa 128 y alfa 255 -> RECHAZADO
    def draw_mixed(im, idx):
        for y in range(16, 32):
            for x in range(16, 48):
                im.putpixel((x, y), (200, 50, 50, 255))
        for y in range(32, 48):
            for x in range(16, 48):
                im.putpixel((x, y), (200, 50, 50, 128))
    run_c = _create_synthetic_run(tmp_dir, "run_mixed", draw_mixed)
    audit_c = run_quality_audit(run_c, format_type="16x4")
    assert audit_c["certified"] is False, f"Mixed alpha must be rejected, got: {audit_c}"
    assert len(audit_c["blurry_alpha_cells"]) == 64
    print(f"[PASS] 2.3: Mezcla alfa 128 y 255: RECHAZADO (blurry_alpha_cells={len(audit_c['blurry_alpha_cells'])}, certified=False).")

    # 2.4 Caso D: Frames vacíos, faltantes o tocando bordes -> RECHAZADO
    def draw_touching_and_empty(im, idx):
        if idx % 2 == 0:
            # Touches left and top border (x=0, y=0)
            for y in range(0, 32):
                for x in range(0, 32):
                    im.putpixel((x, y), (100, 100, 255, 255))
        else:
            # Empty frame
            pass
    # Also simulate missing frames 60, 61, 62, 63
    run_d = _create_synthetic_run(tmp_dir, "run_defects", draw_touching_and_empty, missing_frames=[60, 61, 62, 63])
    audit_d = run_quality_audit(run_d, format_type="16x4")
    assert audit_d["certified"] is False
    assert len(audit_d["empty_cells"]) > 0, "Should detect empty/missing cells"
    assert len(audit_d["border_touching_cells"]) > 0, "Should detect border touching"
    print(f"[PASS] 2.4: Frames vacíos/faltantes ({len(audit_d['empty_cells'])}) y tocando bordes ({len(audit_d['border_touching_cells'])}): RECHAZADOS.")


def test_quality_guidance_summary():
    print("\n--- TEST 3: Guía anatómica de calidad ---")
    metrics = {
        "score_total": 91.2,
        "cuerpo_precision": 94.5,
        "fidelidad_paleta": 96.0,
        "gestos_ojos": 88.7,
        "ropa_delantal": 92.1,
        "objetos_utensilios": 79.4,
        "pelo_gorro": 93.0,
        "zapatos_pies": 90.8,
        "pureza_alfa": 99.2,
    }
    summary = train_mod.PixelArtEnhancer.build_quality_guide(metrics)
    assert summary["guide_score"] == 91.2
    assert summary["body_score"] == 94.5
    assert summary["face_score"] == 88.7
    assert summary["clothes_score"] == 92.1
    assert summary["accessory_score"] == 79.4
    assert "body" in summary["dominant_signal"]
    assert isinstance(summary["alerts"], list)
    print("[PASS] 3.1: Guía anatómica reintroducida con puntuación por cuerpo, rasgos, ropa y accesorios.")


def test_monitor_historical_metrics():
    print("\n--- TEST 4: Métricas Históricas del Monitor ---")
    tmp_dir = tempfile.mkdtemp()
    temp_status = Path(tmp_dir) / "training_status.json"
    orig_status_file = train_mod.STATUS_FILE
    train_mod.STATUS_FILE = temp_status

    try:
        t0 = time.time()
        # 3.1 Initial loss 0.0941 and subsequent 0.09 -> ~4.4% reduction
        update_status(
            epoch=1,
            total_epochs=10,
            status_str="ENTRENANDO",
            g_loss=0.0941,
            d_loss=0.02,
            l1_val=0.05,
            edge_val=0.01,
            start_time=t0
        )
        with open(temp_status, "r", encoding="utf-8") as f:
            d1 = json.load(f)
        assert math.isclose(d1["initial_loss"], 0.0941, rel_tol=1e-4)
        assert math.isclose(d1["best_loss"], 0.0941, rel_tol=1e-4)

        # Step 2: loss becomes 0.09
        update_status(
            epoch=2,
            total_epochs=10,
            status_str="ENTRENANDO",
            g_loss=0.09,
            d_loss=0.019,
            l1_val=0.048,
            edge_val=0.01,
            start_time=t0
        )
        with open(temp_status, "r", encoding="utf-8") as f:
            d2 = json.load(f)
        
        assert math.isclose(d2["initial_loss"], 0.0941, rel_tol=1e-4), f"initial_loss should be 0.0941, got {d2['initial_loss']}"
        assert math.isclose(d2["best_loss"], 0.09, rel_tol=1e-4), f"best_loss should be 0.09, got {d2['best_loss']}"
        assert math.isclose(d2["loss_reduction_pct"], 4.4, rel_tol=1e-2), f"reduction should be ~4.4%, got {d2['loss_reduction_pct']}%"
        print(f"[PASS] 3.1: Pérdida 0.0941 -> 0.09 produce initial_loss={d2['initial_loss']}, best_loss={d2['best_loss']}, reduccion={d2['loss_reduction_pct']}%.")

        # 3.2 Minimum preservation: Checkpoint with historical minimum 0.1 keeps 0.1 when next loss is 0.5
        temp_status_min = Path(tmp_dir) / "test_min.json"
        train_mod.STATUS_FILE = temp_status_min
        update_status(1, 10, "ENTRENANDO", 0.1, 0.05, 0.05, 0.01, t0)
        update_status(2, 10, "ENTRENANDO", 0.5, 0.06, 0.25, 0.05, t0)
        with open(temp_status_min, "r", encoding="utf-8") as f:
            d3 = json.load(f)
        assert math.isclose(d3["best_loss"], 0.1, rel_tol=1e-4), f"best_loss should stay 0.1, got {d3['best_loss']}"
        assert d3["g_loss"] == 0.5
        print(f"[PASS] 3.2: Mínimo histórico 0.1 preservado cuando la siguiente pérdida sube a 0.5 (best_loss={d3['best_loss']}).")

        # 3.3 Test legacy 0.0 migration when reading pre-existing history with 0.0s
        temp_status_legacy = Path(tmp_dir) / "test_legacy.json"
        train_mod.STATUS_FILE = temp_status_legacy
        legacy_data = {
            "status": "ENTRENANDO",
            "epoch": 2,
            "total_epochs": 10,
            "g_loss": 0.08,
            "d_loss": 0.02,
            "best_loss": 0.0,
            "initial_loss": 0.0,
            "history": [
                {"epoch": 1, "loss": 0.0941, "g_loss": 0.0941, "d_loss": 0.03},
                {"epoch": 2, "loss": 0.09, "g_loss": 0.09, "d_loss": 0.025}
            ]
        }
        with open(temp_status_legacy, "w", encoding="utf-8") as f:
            json.dump(legacy_data, f)

        # Call update_status for epoch 3 with g_loss=0.085
        update_status(3, 10, "ENTRENANDO", 0.085, 0.02, 0.04, 0.01, t0)
        with open(temp_status_legacy, "r", encoding="utf-8") as f:
            migrated = json.load(f)
        assert math.isclose(migrated["initial_loss"], 0.0941, rel_tol=1e-4), f"Migrated initial_loss should be 0.0941, got {migrated['initial_loss']}"
        assert math.isclose(migrated["best_loss"], 0.085, rel_tol=1e-4), f"Migrated best_loss should be 0.085, got {migrated['best_loss']}"
        print(f"[PASS] 3.3: Migración de ceros heredados recupera initial_loss=0.0941 y best_loss={migrated['best_loss']} desde historial previo.")
    finally:
        train_mod.STATUS_FILE = orig_status_file


def test_checkpoint_schedulers_and_rng():
    print("\n--- TEST 4: Schedulers y RNG en Checkpoints ---")
    tmp_dir = tempfile.mkdtemp()
    ckpt_path = Path(tmp_dir) / "test_ckpt.pt"

    gen = PixelArtUNetGenerator()
    disc = PixelArtPatchDiscriminator()
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-4)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=10)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=10)

    # Step schedulers once
    opt_g.step()
    sched_g.step()
    opt_d.step()
    sched_d.step()

    # Save checkpoint
    save_checkpoint(
        ckpt_path,
        epoch=3,
        loss=0.08,
        best_loss=0.075,
        generator=gen,
        discriminator=disc,
        opt_g=opt_g,
        opt_d=opt_d,
        scaler_g=None,
        scaler_d=None,
        sched_g=sched_g,
        sched_d=sched_d
    )

    # Load and verify contents
    loaded = torch.load(ckpt_path, map_location="cpu")
    assert "scheduler_g" in loaded, "scheduler_g missing in checkpoint"
    assert "scheduler_d" in loaded, "scheduler_d missing in checkpoint"
    assert "rng_state" in loaded, "rng_state missing in checkpoint"
    assert loaded["best_loss"] == 0.075
    
    # Verify RNG state is a CPU ByteTensor / uint8
    rng_s = loaded["rng_state"]
    assert isinstance(rng_s, torch.Tensor)
    assert rng_s.device.type == "cpu"
    assert rng_s.dtype == torch.uint8
    print("[PASS] 4.1: Schedulers guardados y RNG verificado en CPU (torch.uint8).")

    # Test backward compatibility with legacy checkpoint missing schedulers and rng
    legacy_ckpt_path = Path(tmp_dir) / "legacy_ckpt.pt"
    torch.save({
        "generator": gen.state_dict(),
        "discriminator": disc.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "epoch": 2,
        "best_loss": 0.08
    }, legacy_ckpt_path)
    
    loaded_legacy = torch.load(legacy_ckpt_path, map_location="cpu")
    assert loaded_legacy.get("scheduler_g") is None
    assert loaded_legacy.get("rng_state") is None
    print("[PASS] 4.2: Compatibilidad hacia atrás garantizada para checkpoints antiguos sin schedulers ni RNG.")


def test_gpu_specific_checks():
    print("\n--- TEST 5: Verificaciones Específicas de GPU (CUDA) ---")
    if not torch.cuda.is_available():
        print("[SKIP] CUDA no disponible en este entorno. Prueba GPU omitida.")
        return

    tmp_dir = tempfile.mkdtemp()
    ckpt_path = Path(tmp_dir) / "gpu_ckpt.pt"

    gen = PixelArtUNetGenerator().cuda()
    disc = PixelArtPatchDiscriminator().cuda()
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-4)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=10)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=10)
    scaler_g = torch.cuda.amp.GradScaler(enabled=True)
    scaler_d = torch.cuda.amp.GradScaler(enabled=True)

    # Save on GPU
    save_checkpoint(
        ckpt_path,
        epoch=1,
        loss=0.09,
        best_loss=0.09,
        generator=gen,
        discriminator=disc,
        opt_g=opt_g,
        opt_d=opt_d,
        scaler_g=scaler_g,
        scaler_d=scaler_d,
        sched_g=sched_g,
        sched_d=sched_d
    )

    loaded = torch.load(ckpt_path, map_location="cpu")
    assert "cuda_rng_state" in loaded, "cuda_rng_state debe estar presente al guardar en GPU"
    assert "scaler_g" in loaded and loaded["scaler_g"] is not None
    assert "scaler_d" in loaded and loaded["scaler_d"] is not None
    
    # Restore CUDA RNG
    torch.cuda.set_rng_state_all(loaded["cuda_rng_state"])
    print("[PASS] 5.1: Guardado y restauración de estado RNG CUDA y GradScalers verificado exitosamente.")


def test_past_eras_archiving():
    print("\n--- TEST 6: Archivo y Línea del Tiempo de Eras Anteriores ---")
    tmp_dir = tempfile.mkdtemp()
    temp_status = Path(tmp_dir) / "training_status.json"
    orig_status_file = train_mod.STATUS_FILE
    train_mod.STATUS_FILE = temp_status

    try:
        t0 = time.time()
        # Simular Era 1
        train_mod.update_status(1, 10, "ENTRENANDO", 0.12, 0.05, 0.05, 0.01, t0)
        train_mod.update_status(2, 10, "COMPLETADO", 0.095, 0.04, 0.04, 0.01, t0)

        with open(temp_status, "r", encoding="utf-8") as f:
            data_era1 = json.load(f)
        assert len(data_era1["history"]) == 2
        assert data_era1["best_loss"] == 0.095

        # Simular inicio de nuevo experimento (mode="start") archivando Era 1
        old_hist = data_era1.get("history", [])
        past_eras = data_era1.get("past_eras", [])
        era_idx = len(past_eras) + 1
        past_eras.append({
            "era": era_idx,
            "name": f"Era {era_idx}",
            "timestamp": "12:00:00",
            "epochs": 2,
            "initial_loss": data_era1.get("initial_loss"),
            "best_loss": data_era1.get("best_loss"),
            "history": old_hist
        })
        data_era1["past_eras"] = past_eras
        data_era1["history"] = []
        data_era1["initial_loss"] = None
        data_era1["best_loss"] = None
        data_era1["epoch"] = 0
        data_era1["status"] = "INICIANDO"
        with open(temp_status, "w", encoding="utf-8") as f:
            json.dump(data_era1, f, indent=2, allow_nan=False)

        # Ahora simular Era 2 con update_status
        train_mod.update_status(1, 10, "ENTRENANDO", 0.088, 0.03, 0.03, 0.01, t0)

        with open(temp_status, "r", encoding="utf-8") as f:
            data_era2 = json.load(f)

        assert "past_eras" in data_era2, "past_eras debe conservarse en training_status.json"
        assert len(data_era2["past_eras"]) == 1, f"Debe haber 1 era archivada, hay {len(data_era2['past_eras'])}"
        assert data_era2["past_eras"][0]["name"] == "Era 1"
        assert data_era2["past_eras"][0]["best_loss"] == 0.095
        assert len(data_era2["history"]) == 1, "La era actual debe tener su propio historial limpio (1 época)"
        assert data_era2["history"][0]["loss"] == 0.088
        print("[PASS] 6.1: Era anterior archivada con éxito y línea del tiempo comparativa verificada en JSON.")
    finally:
        train_mod.STATUS_FILE = orig_status_file


if __name__ == "__main__":
    print("==================================================")
    print("EJECUTANDO BATERIA DE PRUEBAS CPU Y GPU")
    print("==================================================")
    test_nan_handling()
    test_alpha_certification()
    test_monitor_historical_metrics()
    test_checkpoint_schedulers_and_rng()
    test_past_eras_archiving()
    test_gpu_specific_checks()
    print("\n==============================================")
    print("TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
    print("==============================================")

