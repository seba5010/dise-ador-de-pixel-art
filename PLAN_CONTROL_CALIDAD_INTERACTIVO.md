# Plan de Control de Calidad Interactivo por Frame

Fecha de inicio y cierre: 2026-10-04

Alcance autorizado: Incrementos 1–5

Estado: COMPLETADO Y VERIFICADO

Progreso real: 100%

## Resultado

Sprite Studio dispone de un ciclo completo y trazable de control de calidad por frame:

1. evaluar y revisar contra el target real del dataset;
2. regenerar candidatos aislados y compararlos con el original;
3. aplicar únicamente un candidato seguro, con backup y actualización de la celda correspondiente;
4. enviar fallos validados a una cola de hard examples separada;
5. ejecutar operaciones batch con progreso, cancelación y recuperación tras reinicio;
6. ajustar el muestreo de entrenamientos futuros con prioridad acotada y decay.

El frame generado nunca se usa como ground truth. El botón de refuerzo no inicia entrenamiento.

## Roadmap cerrado

- [x] Incremento 1 — Review Queue
- [x] Incremento 2 — Regeneración Individual
- [x] Incremento 3 — Manual Reinforcement
- [x] Incremento 4 — Batch QC
- [x] Incremento 5 — Learning Feedback Loop

## Incremento 1 — Review Queue

- [x] Estados estables y registro persistente `frame_review_queue.jsonl`.
- [x] Escritura atómica, bloqueo multihilo, historial, actores y timestamps.
- [x] Resolución de generado, target y frontal desde IDs y metadata confiable.
- [x] Protección contra traversal y raíces no autorizadas.
- [x] `QualityGate.evaluate_single_frame` sin recargar modelos ni mutar datos.
- [x] Target ausente implica auditoría no disponible, nunca aprobación implícita.
- [x] API de consulta, evaluación, reevaluación, aprobación y rechazo.
- [x] UI por frame y resumen en monitor.

Evidencia histórica: commits `f2c044e18`, `36f36d6fc` y `ee8c95c8e`.

## Incremento 2 — Regeneración Individual

- [x] `FrameRegenerationManager` y generador PyTorch genérico, sin nombres de personajes hardcodeados.
- [x] Carga del checkpoint una vez por solicitud y generación de 1–8 alternativas.
- [x] Candidatos guardados fuera del dataset bajo `output/<run>/regeneration/`.
- [x] Auditoría individual y ranking por mejora dominante, score global, pisos críticos y tolerancia de regresión.
- [x] Diagnósticos sin métrica por frame, como `training_stability`, comparan el score global.
- [x] Candidatos sin auditoría, con alfa/bordes críticos o regresiones quedan bloqueados.
- [x] Aplicación explícita del mejor candidato, backup del original y parche sólo de la celda afectada.
- [x] Descartar conserva tanto el original como la evidencia de candidatos.
- [x] API y UI para regenerar, comparar, aplicar y descartar.

## Incremento 3 — Manual Reinforcement

- [x] Cola separada `hard_examples.jsonl`, materializada de forma atómica.
- [x] Validación fuerte de personaje, variante, frame y path contra el manifest real.
- [x] Deduplicación por personaje + variante + frame.
- [x] Prioridad acotada entre 1.0 y 2.0 según severidad y recurrencia.
- [x] El generado se registra sólo como evidencia; el target verificado es el único dato entrenable.
- [x] Marcar para refuerzo no arranca un proceso de entrenamiento.
- [x] API, UI, historial y métricas de casos enviados a refuerzo.

## Incremento 4 — Batch QC

- [x] Reevaluar todos los reviews elegibles.
- [x] Regenerar únicamente frames defectuosos con target válido.
- [x] Enviar una selección explícita a refuerzo con confirmación.
- [x] Trabajos persistentes con total, completados, fallos, progreso y errores.
- [x] Cancelación cooperativa y estado `INTERRUPTED` para trabajos abandonados tras reinicio.
- [x] Endpoints de inicio, consulta, listado y cancelación.
- [x] Seguimiento de progreso y cancelación desde la UI.

## Incremento 5 — Learning Feedback Loop

- [x] El entrenamiento carga sólo hard examples activos y verificados.
- [x] Los pesos manuales se combinan con hard-example mining automático usando el máximo, nunca una multiplicación descontrolada.
- [x] `WeightedRandomSampler` aplica prioridades máximas de 2.0 en entrenamientos futuros.
- [x] Reevaluaciones y candidatos aplicados reducen prioridad cuando mejoran y resuelven el caso al aprobar.
- [x] Métricas: rechazados, regenerados, porcentaje de mejora, enviados a refuerzo, recurrencia y tiempo medio hasta aprobación.
- [x] Monitor muestra hard examples activos y mejora de regeneración.

## Seguridad y operación

- [x] Datasets de sólo lectura para todos los flujos QC.
- [x] Acciones mutables disponibles sólo mediante `POST`.
- [x] Confirmación requerida para batch de regeneración o refuerzo.
- [x] Paths de candidatos revalidados al aplicar, incluso si la persistencia fuese manipulada.
- [x] Estado mutable QC excluido de Git.
- [x] Servidor ligado por defecto a `127.0.0.1`; acceso LAN requiere `SPRITE_STUDIO_HOST` explícito.
- [x] Requests malformados ya no fallan si `SimpleHTTPRequestHandler` aún no definió `self.path`.
- [x] TrainingRecovery permanece intacto.

## Feature flags

Activados por defecto y deshabilitables con variables de entorno:

```python
ENABLE_FRAME_REVIEW = True
ENABLE_FRAME_REGENERATION = True
ENABLE_MANUAL_REINFORCEMENT = True
ENABLE_BATCH_QC_ACTIONS = True
```

Variables: `PIXEL_AI_ENABLE_FRAME_REVIEW`, `PIXEL_AI_ENABLE_FRAME_REGENERATION`, `PIXEL_AI_ENABLE_MANUAL_REINFORCEMENT` y `PIXEL_AI_ENABLE_BATCH_QC_ACTIONS`.

## Matriz de trazabilidad

| ID | Requisito | Estado | Implementación | Prueba principal |
|---|---|---|---|---|
| FQC-001 | Review queue | COMPLETADO | `frame_quality_review.py` | `test_frame_quality_review.py` |
| FQC-002 | Reevaluar | COMPLETADO | `frame_quality_review.py` | reevaluación sin mutar imágenes |
| FQC-003 | Regenerar | COMPLETADO | `frame_regeneration.py` | aislamiento, ranking, backup y aplicación |
| FQC-004 | Aprobar/rechazar | COMPLETADO | manager + API | persistencia de actor y estado |
| FQC-005 | Refuerzo manual | COMPLETADO | `hard_examples.py` | target verificado y deduplicación |
| FQC-006 | Batch QC | COMPLETADO | `interactive_qc.py` | progreso, cancelación y recuperación |
| FQC-007 | Feedback de aprendizaje | COMPLETADO | `quality_guidance.py`, `train_supervised.py` | prioridad, decay y sampler |
| FQC-008 | API segura | COMPLETADO | `sprite_studio.py` | métodos, confirmación y request malformado |
| FQC-009 | UI y monitor | COMPLETADO | HTML de Studio y Monitor | QA visual sin errores de consola |
| FQC-010 | Historial y resume | COMPLETADO | tres persistencias atómicas | reconstrucción tras reinicio |

## Verificación final

- Suite completa: `105 passed`, `0 failed`, 5 warnings de deprecación AMP heredados.
- Pruebas dirigidas finales de persistencia, regeneración, hard examples, batch y API: `24 passed`.
- Compilación Python de servidor y módulos nuevos: PASS.
- QA visual en Chrome: controles, capacidades, copy de seguridad y métricas del monitor visibles; 0 errores o warnings de consola.
- `git diff --check`: debe permanecer limpio antes del commit final.

## Decisiones

- ADR-FQC-001: JSONL antes que SQLite para trazabilidad simple y recuperación local.
- ADR-FQC-002: el cliente transmite IDs, no rutas arbitrarias.
- ADR-FQC-003: falta de target significa auditoría no disponible.
- ADR-FQC-004: candidatos y generados son evidencia, nunca ground truth.
- ADR-FQC-005: aplicar es una acción explícita y recuperable mediante backup.
- ADR-FQC-006: refuerzo modifica el muestreo del próximo entrenamiento; no lanza entrenamiento.
- ADR-FQC-007: trabajos batch son persistentes, cancelables y reanudables como estado interrumpido.
