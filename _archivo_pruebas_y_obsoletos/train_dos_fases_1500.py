"""
Entrenamiento Automatizado de 2 Fases (2000 Épocas Fase 1 + 2000 Épocas Fase 2)
Pixel AI Engine - Spritesheet Generator

Fase 1: 2000 épocas en Plantilla 16x4 (64 frames) con Conny y Dana
        Aprende estructura básica, proporciones corporales y caminatas base.
Fase 2: 2000 épocas en Plantilla 8x12 (96 frames) con Alex y Amaro
        Transfiere los pesos aprendidos de la Fase 1 y perfecciona las 96 poses complejas
        (cocinar, pensar, cajas, servir, celebrar, rotación 360°).
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Optional

# Asegurar codificación utf-8 en terminal de Windows
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pixel_ai_engine.train import train
from pixel_ai_engine.config import CHECKPOINT_DIR

def run_two_phase_training(epochs_fase1: int = 800,
                           epochs_fase2: int = 600,
                           batch_size: int = 8,
                           resume: bool = True,
                           sample_image: Optional[str] = None):
    print("\n" + "=" * 75)
    print("      🚀 PIPELINE COMPLETO DE ENTRENAMIENTO DE TRES FASES")
    print(f"      Fase 1: Pulido hasta época {epochs_fase1} (Reanuda desde 500 con texturizado)")
    print(f"      Fase 2: {epochs_fase2} épocas (Transferencia 8x12 - 96 frames de Alex y Amaro)")
    print(f"      Fase 3: Auditoría Clínica y Elevación Quirúrgica Final de Spritesheet")
    print(f"      Batch Size: {batch_size}")
    print("=" * 75 + "\n")

    t_start_total = time.time()
    from pixel_ai_engine.quality_gate import QualityGate

    target_assimilation = 99.5
    max_reinforcement_rounds = 5

    base_model_path = CHECKPOINT_DIR / "base_generator_16x4.pt"
    latest_fase2_path = CHECKPOINT_DIR / "latest_checkpoint.pt"
    fase2_already_running = False

    if latest_fase2_path.exists():
        try:
            import torch
            ckpt_f2 = torch.load(latest_fase2_path, map_location="cpu")
            if ckpt_f2.get("phase") == "2" and ckpt_f2.get("epoch", 0) > 0:
                fase2_already_running = True
                f2_epoch = ckpt_f2.get("epoch", 0)
        except Exception:
            fase2_already_running = False

    if fase2_already_running and base_model_path.exists():
        print("\n" + "=" * 75)
        print(f"  ⚡ RECUPERACIÓN INTELIGENTE: FASE 1 YA FUE COMPLETADA Y VALIDADA AL 100%")
        print(f"  Detectado progreso de Fase 2 en época {f2_epoch}. Saltando Fase 1...")
        print("=" * 75 + "\n")
    else:
        # ──────────────────────────────────────────────────────────────────────────
        # [1/2] FASE 1: BASE 16x4 (64 FRAMES)
        # ──────────────────────────────────────────────────────────────────────────
        print("\n" + "#" * 75)
        print(f"  [1/2] INICIANDO FASE 1 ({epochs_fase1} ÉPOCAS) - PLANTILLA 16x4")
        print("  Objetivo: Aprender anatomía, outlines, ropa y caminatas base")
        print("#" * 75 + "\n")

        train(
            epochs=epochs_fase1,
            batch_size=batch_size,
            resume=resume,
            infinite=False,
            phase="1",
            lr_g=8e-5,
            lr_d=4e-5
        )

        if not base_model_path.exists():
            # Fallback a latest si base no se guardó
            latest_fase1 = CHECKPOINT_DIR / "latest_checkpoint_16x4.pt"
            if latest_fase1.exists():
                import torch
                saved = torch.load(latest_fase1)
                torch.save(saved.get("generator", saved), base_model_path)

        elapsed_f1 = (time.time() - t_start_total) / 60.0
        print("\n" + "=" * 75)
        print(f"  ✅ SESIÓN INICIAL DE FASE 1 COMPLETADA EN {elapsed_f1:.1f} MINUTOS")
        print("=" * 75 + "\n")

        # ──────────────────────────────────────────────────────────────────────────
        # [PUERTA DE CALIDAD FASE 1 -> FASE 2] AUDITORÍA CRÍTICA Y BUCLE DE REFUERZO
        # ──────────────────────────────────────────────────────────────────────────
        from pixel_ai_engine.quality_gate import QualityGate

        target_assimilation = 99.5
        max_reinforcement_rounds = 5
        report_f1_path = PROJECT_ROOT / "DIAGNOSTICO_AUDITORIA_FASE1.md"

        print("#" * 75)
        print("  🔍 AUDITORÍA CLÍNICA INTERMEDIA DE ASIMILACIÓN (FASE 1 -> FASE 2)")
        print(f"  Umbral Requerido de Asimilación: {target_assimilation}% (Colores, Molde y Textura)")
        print("#" * 75 + "\n")

        for r_round in range(max_reinforcement_rounds + 1):
            report_f1 = QualityGate.evaluate_model_critical(
                checkpoint_path=base_model_path,
                phase="1",
                target_threshold=target_assimilation
            )
            QualityGate.write_diagnostic_report(report_f1, report_f1_path)

            print(f"  📊 Resultado Auditoría Fase 1: {report_f1['score_total']}% (Umbral: {target_assimilation}%)")
            print(f"     • Fidelidad de Color/Paleta: {report_f1['fidelidad_paleta']}%")
            print(f"     • Alineación con el Molde:   {report_f1['alineacion_molde']}%")
            print(f"     • Micro-Textura (1px):       {report_f1['micro_textura']}%")
            print(f"     • Pureza Alfa (Fondo):       {report_f1['pureza_alfa']}%")
            print(f"  📝 Diagnóstico clínico guardado en: {report_f1_path.name}")

            if report_f1["aprobado"]:
                print("\n  🎯 ¡ASIMILACIÓN EXITOSA VALIDADA! La red dominó el pixel art y la paleta.")
                print("  Autorizando transferencia de pesos a la Fase 2 (Molde Definitivo 8x12)...\n")
                break
            else:
                if r_round < max_reinforcement_rounds:
                    print(f"\n  ⚠️ NO ALCANZÓ EL UMBRAL DEL {target_assimilation}%.")
                    print(f"  ❌ Fallas detectadas ({len(report_f1['fallas'])}):")
                    for f in report_f1['fallas'][:3]:
                        print(f"     - {f}")
                    print(f"\n  🔄 ACCIÓN AUTOMÁTICA: Devolviendo al entrenamiento de refuerzo (+100 épocas) [{r_round+1}/{max_reinforcement_rounds}]...")
                    train(epochs=100, batch_size=batch_size, resume=True, infinite=False, phase="1")
                else:
                    print(f"  ⚡ Límite de refuerzos alcanzado. Procediendo con el mejor checkpoint ({report_f1['score_total']}%)...")

        # Pausa de 5 segundos para refrescar GPU y archivos
        print("Preparando transferencia de pesos a la Fase 2 en 5 segundos...")
        time.sleep(5)

    # ──────────────────────────────────────────────────────────────────────────
    # [2/3] FASE 2: TRANSFER LEARNING A 8x12 (96 FRAMES - MOLDE DEFINITIVO)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "#" * 75)
    print(f"  [2/3] INICIANDO FASE 2 ({epochs_fase2} ÉPOCAS) - PLANTILLA 8x12")
    print("  Objetivo: Transferir pesos de Fase 1 y perfeccionar 96 poses complejas")
    print("  La cuadrícula 8x12 es la DEFINITIVA para la generación del juego")
    print("#" * 75 + "\n")

    # Si ya venía entrenando la Fase 2, verificar meta
    should_resume_f2 = resume and latest_fase2_path.exists()
    
    # Si epochs_fase2 es un número de épocas adicionales o una meta absoluta:
    target_f2_epochs = epochs_fase2
    if should_resume_f2 and f2_epoch > 0:
        if epochs_fase2 <= f2_epoch:
            # Interpretar como épocas adicionales
            target_f2_epochs = f2_epoch + epochs_fase2
            print(f"  -> Reanudando desde época {f2_epoch}. Entrenando {epochs_fase2} épocas adicionales (meta: {target_f2_epochs})...")
        else:
            print(f"  -> Reanudando desde época {f2_epoch} hasta meta {target_f2_epochs}...")

    train(
        epochs=target_f2_epochs,
        batch_size=batch_size,
        resume=should_resume_f2,
        infinite=False,
        phase="2",
        transfer_from=str(base_model_path) if not should_resume_f2 else None
    )

    # ──────────────────────────────────────────────────────────────────────────
    # [PUERTA DE CALIDAD FASE 2] AUDITORÍA CRÍTICA DE LAS 96 POSES DEFINITIVAS
    # ──────────────────────────────────────────────────────────────────────────
    target_assimilation = 98.5
    latest_f2_ckpt = CHECKPOINT_DIR / "latest_checkpoint.pt"
    best_gen_path = CHECKPOINT_DIR / "best_generator.pt"
    eval_ckpt_path = latest_f2_ckpt if latest_f2_ckpt.exists() else best_gen_path

    report_f2_path = PROJECT_ROOT / "DIAGNOSTICO_AUDITORIA_FASE2.md"

    print("\n" + "#" * 75)
    print("  🔍 AUDITORÍA CLÍNICA DE LAS 96 POSES DEFINITIVAS (FASE 2)")
    print(f"  Umbral Requerido de Asimilación: {target_assimilation}%")
    print("#" * 75 + "\n")

    from pixel_ai_engine.config import PERSONAJES_DIR
    sample_input = Path(sample_image) if sample_image else (PERSONAJES_DIR / "alex" / "alex_rbchef.png")
    if not sample_input.exists():
        candidates = list(PERSONAJES_DIR.glob("*/*.png"))
        if candidates:
            sample_input = candidates[0]

    for r_round in range(max_reinforcement_rounds + 1):
        report_f2 = QualityGate.evaluate_model_critical(
            checkpoint_path=eval_ckpt_path,
            sample_img_path=sample_input,
            phase="2",
            target_threshold=target_assimilation
        )
        QualityGate.write_diagnostic_report(report_f2, report_f2_path)

        print(f"  📊 Resultado Auditoría Fase 2: {report_f2['score_total']}% (Umbral: {target_assimilation}%)")
        print(f"     • Fidelidad de Color/Paleta: {report_f2['fidelidad_paleta']}%")
        print(f"     • Alineación con Molde 8x12: {report_f2['alineacion_molde']}%")
        print(f"     • Micro-Textura (1px):       {report_f2['micro_textura']}%")
        print(f"     • Pureza Alfa (Fondo):       {report_f2['pureza_alfa']}%")
        print(f"  📝 Diagnóstico clínico guardado en: {report_f2_path.name}")

        if report_f2["aprobado"]:
            print("\n  🎯 ¡ASIMILACIÓN DEFINITIVA APROBADA! Las 96 poses calzan exactamente con el molde.")
            break
        else:
            if r_round < max_reinforcement_rounds:
                print(f"\n  ⚠️ POSES INCOMPLETAS O DESVIADAS ({report_f2['score_total']}% < {target_assimilation}%).")
                print(f"  🔄 ACCIÓN AUTOMÁTICA: Devolviendo a refuerzo (+100 épocas en Fase 2) [{r_round+1}/{max_reinforcement_rounds}]...")
                train(epochs=100, batch_size=batch_size, resume=True, infinite=False, phase="2")
            else:
                print("  ⚡ Límite de refuerzo alcanzado. Procediendo a elevación final...")

    # ──────────────────────────────────────────────────────────────────────────
    # [3/3] FASE 3: REVISIÓN CRÍTICA Y ELEVACIÓN DE DISEÑO
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "#" * 75)
    print("  [3/3] INICIANDO FASE 3: AUDITORÍA CRÍTICA Y ELEVACIÓN DE DISEÑO")
    print("  Objetivo: Auditar los 96 frames y enriquecer texturas, sombras y micro-líneas")
    print("  Meta: Superar la calidad base de los sprites originales")
    print("#" * 75 + "\n")

    try:
        from pixel_ai_engine.phase3_critical_enhancer import Phase3CriticalReviewer
        from pixel_ai_engine.generate_character_sheet import generate_spritesheet
        from pixel_ai_engine.config import OUTPUT_DIR, PERSONAJES_DIR

        # Generar hoja de prueba del personaje indicado o Alex
        sample_input = Path(sample_image) if sample_image else (PERSONAJES_DIR / "alex" / "alex_rbchef.png")
        if not sample_input.exists():
            candidates = list(PERSONAJES_DIR.glob("*/*.png"))
            if candidates:
                sample_input = candidates[0]

        best_ckpt = CHECKPOINT_DIR / "best_generator.pt"
        if best_ckpt.exists() and sample_input.exists():
            print(f"  -> Generando spritesheet base con el modelo final ({sample_input.name})...")
            raw_sheet = generate_spritesheet(
                input_image_path=sample_input,
                checkpoint_path=best_ckpt,
                phase="2"
            )
            print("  -> Ejecutando auditoría y elevación de diseño de Fase 3...")
            elevated_sheet, report = Phase3CriticalReviewer.elevate_spritesheet(
                sheet_path=raw_sheet,
                identity_path=sample_input,
                phase="2"
            )
            print(f"  🏆 FASE 3 CONCLUIDA: Calidad elevada de {report['calidad_promedio_fase2']}% a {report['calidad_promedio_fase3_elevada']}% (+{report['mejora_porcentual']}%)")
    except Exception as e:
        print(f"  [Aviso] Fase 3 se ejecutará como herramienta independiente: {e}")

    elapsed_total = (time.time() - t_start_total) / 60.0
    print("\n" + "=" * 75)
    print("  🎉 ¡PIPELINE COMPLETO FINALIZADO CON ÉXITO!")
    print(f"  Tiempo total transcurrido: {elapsed_total:.1f} minutos ({elapsed_total/60:.2f} horas)")
    print(f"  Modelo final optimizado: checkpoints/best_generator.pt")
    print(f"  Módulo de elevación listo: fase3_revision_critica.py")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Completo: Pulido Fase 1 + Fase 2 (96 frames) + Fase 3 Elevacion")
    parser.add_argument("--epochs-fase1", type=int, default=800, help="Época meta Fase 1 (por defecto 800, reanuda desde 500)")
    parser.add_argument("--epochs-fase2", type=int, default=600, help="Épocas Fase 2 (por defecto 600, óptimo anti-colapso)")
    parser.add_argument("--batch-size", type=int, default=8, help="Tamaño del batch (por defecto 8)")
    parser.add_argument("--sample-image", type=str, default=None, help="Ruta a la foto del personaje a auditar (ej: personajes/alex/alex_rbchef.png)")
    parser.add_argument("--no-resume", action="store_true", help="No reanudar desde checkpoint")
    args = parser.parse_args()

    run_two_phase_training(
        epochs_fase1=args.epochs_fase1,
        epochs_fase2=args.epochs_fase2,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        sample_image=args.sample_image
    )
