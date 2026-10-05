# Plan de Control de Calidad Interactivo por Frame

Fecha de inicio: 2026-10-04  
Metodología: desarrollo incremental con Scrum dentro de cada incremento  
Alcance autorizado actual: Incremento 1 solamente  
Progreso real del roadmap: 20% (Incremento 1 completado; 4 incrementos posteriores bloqueados)
Progreso real del Incremento 1: 100%

## Estado auditado

- [x] Se revisó el servidor y la UI actuales.
  - Evidencia: `sprite_studio.py` usa `ThreadingHTTPServer`/`SimpleHTTPRequestHandler`; `sprite_studio.html` ya posee un Quality Gate de hoja, pero no revisión persistente por frame.
- [x] Se revisó el stack de calidad y entrenamiento.
  - Evidencia: `quality_gate.py`, `quality_guidance.py`, `enhancer.py`, `anatomical_guidance.py`, `train_supervised.py` y `training_recovery.py`.
- [x] Se revisó la resolución de dataset y metadata.
  - Evidencia: manifests por personaje/variante en `dataset_frames_individuales`; frames normalizados y frontales en `dataset_supervisado`.
- [x] Se revisaron los scripts históricos solicitados.
  - Evidencia: se reutilizan conceptos de auditoría por celda; no se restaura el entrenador de dos fases.
- [x] Se identificó comportamiento inseguro heredado.
  - Evidencia: `QualityGate.evaluate_model_critical` aprueba con 99.5 cuando falta referencia y contiene selección histórica por nombres; debe corregirse sin reescribir el pipeline.

### Componentes reutilizables

- `PixelArtEnhancer.analyze_quality`: paleta, alfa, bordes, microdetalle y categorías visuales.
- `compute_anatomical_metrics`: geometría e IoU con target real.
- `build_quality_vector` y `diagnose_quality_bottleneck`: vector canónico y diagnóstico.
- metadata de cada ejecución, `frame_map` y manifests de dataset.
- servidor/API y pestaña de Control de Calidad existentes.

### Faltantes confirmados al inicio (resueltos en este incremento)

- registro estable por frame y vocabulario cerrado;
- persistencia JSONL atómica con historial;
- búsqueda segura de generado/target/referencia;
- evaluación QualityGate de un solo frame;
- consulta, reevaluación, aprobación y rechazo por API;
- controles y diagnóstico detallado en UI;
- resumen de review queue en monitor;
- pruebas de seguridad, persistencia, resume y endpoints.

## Roadmap

- [x] Incremento 1 — Review Queue
- [ ] Incremento 2 — Regeneración Individual (bloqueado hasta aprobación del Incremento 1)
- [ ] Incremento 3 — Manual Reinforcement (bloqueado)
- [ ] Incremento 4 — Batch QC (bloqueado)
- [ ] Incremento 5 — Learning Feedback Loop (bloqueado)

## Incremento 1 — Review Queue

Objetivo: revisar, reevaluar, aprobar y rechazar frames individuales, con persistencia y trazabilidad, sin regenerar ni entrenar.

### Sprint 1 — Dominio y persistencia

Sprint Goal: crear un registro seguro y reanudable por frame.

Backlog:

- [x] Definir `FrameReviewStatus` con vocabulario estable.
- [x] Crear `FrameQualityReviewManager` separado del servidor.
- [x] Implementar JSONL con escritura atómica y bloqueo de escritura multihilo.
- [x] Registrar identidad, variante, frame, pose, paths, scores, issues y timestamps.
- [x] Conservar historial append-only dentro de cada registro materializado.
- [x] Resolver rutas exclusivamente desde `run_id` y metadata confiable.
- [x] Rechazar traversal y rutas fuera de `OUTPUT_DIR`/dataset permitido.
- [x] Tests unitarios de estados, JSONL, persistencia, resume y safe paths.

Archivos previstos:

- `pixel_ai_engine/frame_quality_review.py`
- `test_frame_quality_review.py`

Funciones previstas:

- `evaluate_frame`
- `get_review`
- `list_reviews`
- `approve_frame`
- `reject_frame`
- `resolve_run_frame`
- `resolve_dataset_assets`

Riesgos:

- IDs ambiguos entre nombres con espacios y guiones bajos;
- JSONL interrumpido durante escritura;
- targets con numeración 0-based y archivos 1-based.

Criterios de aceptación / Definition of Done:

- [x] strings de estado arbitrarios son rechazados;
- [x] un reinicio reconstruye el mismo estado;
- [x] no se puede resolver una ruta fuera de raíces permitidas;
- [x] tests dirigidos y regresiones PASS;
- [x] evidencia y commit registrados (`f2c044e18`).

### Sprint 2 — QualityGate por frame

Sprint Goal: evaluar o reevaluar un frame sin recargar un checkpoint ni generar una imagen.

Backlog:

- [x] Implementar `QualityGate.evaluate_single_frame`.
- [x] Combinar Enhancer, anatomía, detalle crítico y QualityVector.
- [x] Reportar `audit_available=false`, score `null` y no aprobado sin target.
- [x] Incluir controles de alfa y bordes.
- [x] Eliminar aprobación por defecto cuando falta referencia.
- [x] Retirar selección por nombres hardcodeados del camino moderno.
- [x] Integrar evaluación con el manager e historial.
- [x] Tests de reevaluación, target lookup y target ausente.

Archivos previstos:

- `pixel_ai_engine/quality_gate.py`
- `pixel_ai_engine/frame_quality_review.py`
- `test_frame_quality_review.py`

Riesgos:

- métricas heurísticas no equivalen a identidad semántica;
- targets o generado con tamaños distintos;
- ausencia de referencia frontal.

Criterios de aceptación / Definition of Done:

- [x] reevaluar no modifica imagen, dataset ni pesos;
- [x] silhouette no se calcula si falta target y la auditoría queda no disponible;
- [x] `identity_confidence` es `null` sin métrica válida;
- [x] diagnóstico contiene vector e issues explicables;
- [x] tests y commit registrados (`f2c044e18`).

### Sprint 3 — API y UI

Sprint Goal: exponer la revisión por frame en Sprite Studio.

Backlog:

- [x] `GET /api/qc/review-queue`.
- [x] `GET /api/qc/frame/<review_id>`.
- [x] `POST /api/qc/frame/evaluate` para alta controlada.
- [x] `POST /api/qc/frame/<review_id>/reevaluate`.
- [x] `POST /api/qc/frame/<review_id>/approve`.
- [x] `POST /api/qc/frame/<review_id>/reject`.
- [x] Tarjetas por frame con estado, score, issues y acciones.
- [x] Diagnóstico detallado con target/generado.
- [x] Semáforo accesible por color y texto.
- [x] Panel de conteos en `monitor.html`.
- [x] Tests HTTP, validación de método y regresiones.

Archivos previstos:

- `sprite_studio.py`
- `sprite_studio.html`
- `monitor.html`
- `test_frame_quality_review.py`

Riesgos:

- servidor multihilo escribiendo simultáneamente;
- HTML grande y estado UI previo;
- acciones mutables expuestas accidentalmente por GET.

Criterios de aceptación / Definition of Done:

- [x] acciones mutables sólo aceptan POST;
- [x] UI actualiza sin recargar y muestra texto además de color;
- [x] aprobar/rechazar persiste usuario y fecha;
- [x] panel refleja pendientes/aprobados/rechazados;
- [x] suite completa PASS e Increment Review aprobado.

## Restricciones activas

- [x] Ningún frame generado se usará como ground truth.
- [x] El Incremento 1 no escribe `hard_examples.jsonl`.
- [x] El Incremento 1 no inicia entrenamiento.
- [x] El Incremento 1 no implementa regeneración ni acciones batch.
- [x] TrainingRecovery permanece sin reemplazo.
- [x] Los datasets son de solo lectura para QC.

## Feature flags

Estado objetivo del Incremento 1:

```python
ENABLE_FRAME_REVIEW = True
ENABLE_FRAME_REGENERATION = False
ENABLE_MANUAL_REINFORCEMENT = False
ENABLE_BATCH_QC_ACTIONS = False
```

## Matriz de trazabilidad

| ID | Requisito | Estado | Archivo | Test | Commit |
|---|---|---|---|---|---|
| FQC-001 | Review queue | COMPLETADO | `frame_quality_review.py` | `test_frame_quality_review.py` | `f2c044e18` |
| FQC-002 | Reevaluar | COMPLETADO | `frame_quality_review.py`, `quality_gate.py` | `test_reevaluate_updates_metrics_without_modifying_images` | `f2c044e18` |
| FQC-003 | Regenerar | BLOQUEADO (Inc. 2) | - | - | - |
| FQC-004 | Aprobar | COMPLETADO | `frame_quality_review.py`, `sprite_studio.py` | `test_approve_and_reject_are_persisted_with_actor` | `f2c044e18` |
| FQC-005 | Rechazar | COMPLETADO | `frame_quality_review.py`, `sprite_studio.py` | `test_approve_and_reject_are_persisted_with_actor` | `f2c044e18` |
| FQC-006 | Enviar a refuerzo | BLOQUEADO (Inc. 3) | - | - | - |
| FQC-007 | Hard examples | BLOQUEADO (Inc. 3) | - | - | - |
| FQC-008 | API segura | COMPLETADO | `sprite_studio.py` | `test_frame_review_rest_endpoints_validate_and_persist` | `f2c044e18` |
| FQC-009 | UI por frame | COMPLETADO | `sprite_studio.html`, `monitor.html` | QA Chrome sin errores | `36f36d6fc` |
| FQC-010 | Historial y resume | COMPLETADO | `frame_quality_review.py` | persistencia/resume | `f2c044e18` |

## Registro de decisiones

- ADR-FQC-001: JSONL antes que SQLite para trazabilidad simple; la actualización reescribe atómicamente el estado materializado.
- ADR-FQC-002: el cliente envía IDs (`run_id`, `review_id`, `frame_idx`), nunca rutas arbitrarias.
- ADR-FQC-003: un target faltante produce auditoría no disponible, nunca aprobación implícita.
- ADR-FQC-004: el review conserva paths del generado y target, pero sólo el target real podrá alimentar entrenamiento en incrementos futuros.
- ADR-FQC-005: las funciones históricas de regeneración existentes quedan fuera de esta integración hasta el Incremento 2.

## Historial de pruebas y commits

| Fecha | Sprint | Pruebas | Resultado | Commit |
|---|---|---|---|---|
| 2026-10-04 | Baseline previo | `python -m pytest -q` | 87 PASS, 5 warnings AMP | `2402d42cc` y anteriores |
| 2026-10-04 | Sprint 1-2 | `python -m pytest -q test_frame_quality_review.py` | 10 PASS | `f2c044e18` |
| 2026-10-04 | Sprint 3 / regresión | `python -m pytest -q` | 91 PASS, 5 warnings AMP heredados | `f2c044e18`, `36f36d6fc` |
| 2026-10-04 | QA visual | Chrome sobre Sprite Studio y Monitor | cola, target, métricas y resumen visibles; 0 errores de consola | `36f36d6fc` |
| 2026-10-04 | Arquitectura | revisión documental | flujo, API, seguridad y límites documentados | `ee8c95c8e` |

## Increment Review 1

Estado: COMPLETADO Y VERIFICADO.

Evidencia de cierre:

- 91 tests PASS; 0 FAIL; 5 warnings deprecados de AMP preexistentes.
- evaluación real del frame 0 de `alex_8x12_20261004_191852`: target resuelto desde manifest, score 87.8 y trazabilidad persistida;
- UI verificada en Chrome: tarjeta por frame, generado/target, métricas, diagnóstico y acciones;
- monitor verificado con 1 review pendiente y contadores de estados;
- regeneración, refuerzo, cambios de sampling, entrenamiento y batch permanecen desactivados.

El trabajo se detiene aquí según la regla del Incremento 1. El Incremento 2 sólo debe comenzar tras una nueva aprobación explícita del usuario.
