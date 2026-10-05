# Plan de Control de Calidad Interactivo por Frame

Fecha de inicio: 2026-10-04  
Metodología: desarrollo incremental con Scrum dentro de cada incremento  
Alcance autorizado actual: Incremento 1 solamente  
Progreso real: 8% (auditoría y plan maestro completados; implementación pendiente)

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

### Faltantes confirmados

- registro estable por frame y vocabulario cerrado;
- persistencia JSONL atómica con historial;
- búsqueda segura de generado/target/referencia;
- evaluación QualityGate de un solo frame;
- consulta, reevaluación, aprobación y rechazo por API;
- controles y diagnóstico detallado en UI;
- resumen de review queue en monitor;
- pruebas de seguridad, persistencia, resume y endpoints.

## Roadmap

- [ ] Incremento 1 — Review Queue
- [ ] Incremento 2 — Regeneración Individual (bloqueado hasta aprobación del Incremento 1)
- [ ] Incremento 3 — Manual Reinforcement (bloqueado)
- [ ] Incremento 4 — Batch QC (bloqueado)
- [ ] Incremento 5 — Learning Feedback Loop (bloqueado)

## Incremento 1 — Review Queue

Objetivo: revisar, reevaluar, aprobar y rechazar frames individuales, con persistencia y trazabilidad, sin regenerar ni entrenar.

### Sprint 1 — Dominio y persistencia

Sprint Goal: crear un registro seguro y reanudable por frame.

Backlog:

- [ ] Definir `FrameReviewStatus` con vocabulario estable.
- [ ] Crear `FrameQualityReviewManager` separado del servidor.
- [ ] Implementar JSONL con escritura atómica y bloqueo de proceso.
- [ ] Registrar identidad, variante, frame, pose, paths, scores, issues y timestamps.
- [ ] Conservar historial append-only dentro de cada registro materializado.
- [ ] Resolver rutas exclusivamente desde `run_id` y metadata confiable.
- [ ] Rechazar traversal y rutas fuera de `OUTPUT_DIR`/dataset permitido.
- [ ] Tests unitarios de estados, JSONL, persistencia, resume y safe paths.

Archivos previstos:

- `pixel_ai_engine/frame_quality_review.py`
- `test_frame_quality_review.py`

Funciones previstas:

- `create_or_update_review`
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

- [ ] strings de estado arbitrarios son rechazados;
- [ ] un reinicio reconstruye el mismo estado;
- [ ] no se puede resolver una ruta fuera de raíces permitidas;
- [ ] tests dirigidos y regresiones PASS;
- [ ] evidencia y commit registrados.

### Sprint 2 — QualityGate por frame

Sprint Goal: evaluar o reevaluar un frame sin recargar un checkpoint ni generar una imagen.

Backlog:

- [ ] Implementar `QualityGate.evaluate_single_frame`.
- [ ] Combinar Enhancer, anatomía, detalle crítico y QualityVector.
- [ ] Reportar `audit_available=false`, score `null` y no aprobado sin target.
- [ ] Incluir controles de alfa y bordes.
- [ ] Eliminar aprobación por defecto cuando falta referencia.
- [ ] Retirar selección por nombres hardcodeados del camino moderno.
- [ ] Integrar evaluación con el manager e historial.
- [ ] Tests de reevaluación, target lookup y target ausente.

Archivos previstos:

- `pixel_ai_engine/quality_gate.py`
- `pixel_ai_engine/frame_quality_review.py`
- `test_frame_quality_review.py`

Riesgos:

- métricas heurísticas no equivalen a identidad semántica;
- targets o generado con tamaños distintos;
- ausencia de referencia frontal.

Criterios de aceptación / Definition of Done:

- [ ] reevaluar no modifica imagen, dataset ni pesos;
- [ ] silhouette es `null` si falta target;
- [ ] `identity_confidence` es `null` sin métrica válida;
- [ ] diagnóstico contiene vector e issues explicables;
- [ ] tests y commit registrados.

### Sprint 3 — API y UI

Sprint Goal: exponer la revisión por frame en Sprite Studio.

Backlog:

- [ ] `GET /api/qc/review-queue`.
- [ ] `GET /api/qc/frame/<review_id>`.
- [ ] `POST /api/qc/frame/evaluate` para alta controlada.
- [ ] `POST /api/qc/frame/<review_id>/reevaluate`.
- [ ] `POST /api/qc/frame/<review_id>/approve`.
- [ ] `POST /api/qc/frame/<review_id>/reject`.
- [ ] Tarjetas por frame con estado, score, issues y acciones.
- [ ] Diagnóstico detallado con target/generado.
- [ ] Semáforo accesible por color y texto.
- [ ] Panel de conteos en `monitor.html`.
- [ ] Tests HTTP, validación de método y regresiones.

Archivos previstos:

- `sprite_studio.py`
- `sprite_studio.html`
- `monitor.html`
- `test_frame_quality_review_api.py`

Riesgos:

- servidor multihilo escribiendo simultáneamente;
- HTML grande y estado UI previo;
- acciones mutables expuestas accidentalmente por GET.

Criterios de aceptación / Definition of Done:

- [ ] acciones mutables sólo aceptan POST;
- [ ] UI actualiza sin recargar y muestra texto además de color;
- [ ] aprobar/rechazar persiste usuario y fecha;
- [ ] panel refleja pendientes/aprobados/rechazados;
- [ ] suite completa PASS e Increment Review aprobado.

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
| FQC-001 | Review queue | EN DESARROLLO | `frame_quality_review.py` | pendiente | pendiente |
| FQC-002 | Reevaluar | PENDIENTE | - | - | - |
| FQC-003 | Regenerar | BLOQUEADO (Inc. 2) | - | - | - |
| FQC-004 | Aprobar | PENDIENTE | - | - | - |
| FQC-005 | Rechazar | PENDIENTE | - | - | - |
| FQC-006 | Enviar a refuerzo | BLOQUEADO (Inc. 3) | - | - | - |
| FQC-007 | Hard examples | BLOQUEADO (Inc. 3) | - | - | - |
| FQC-008 | API segura | PENDIENTE | - | - | - |
| FQC-009 | UI por frame | PENDIENTE | - | - | - |
| FQC-010 | Historial y resume | PENDIENTE | - | - | - |

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

## Increment Review 1

Estado: PENDIENTE. No autoriza Incremento 2 hasta completar los tres Sprint Reviews, documentación, suite completa y verificación manual de UI/API.
