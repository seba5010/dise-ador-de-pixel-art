# PLAN CONTROL DE CALIDAD INTERACTIVO POR FRAME

## ROADMAP GENERAL

- [x] Incremento 1 — Review Queue + Reevaluación + Aprobación/Rechazo
- [ ] Incremento 2 — Regeneración individual por frame
- [ ] Incremento 3 — Aplicación de mejor candidato y descartar regeneración
- [ ] Incremento 4 — Envío a refuerzo y hard examples
- [ ] Incremento 5 — Integración con sampler y entrenamiento adaptativo
- [ ] Incremento 6 — Batch QC, monitor y trazabilidad completa

## ESTADO ACTUAL

- [x] Auditoría del repo y revisión de los módulos base.
- [x] Definida la estructura de datos de revisión por frame.
- [x] Implementado el manager y la serialización JSONL.
- [x] Validación con tests de frontera y regresión.

## EVIDENCIA

- Repositorio revisado: `pixel_ai_engine/quality_gate.py`, `pixel_ai_engine/quality_guidance.py`, `pixel_ai_engine/train_supervised.py`, `pixel_ai_engine/dataset.py`.
- Estado actual: queda implementado el Incremento 1 de review queue sin entrar en regeneración ni refuerzo, conforme a la especificación.
- Prueba directa: `& "d:/escritorio/diseñador de pixel art/.venv/Scripts/python.exe" -m pytest -q test_frame_quality_review.py` → `5 passed in 3.97s`.
- Validación de regresión: `& "d:/escritorio/diseñador de pixel art/.venv/Scripts/python.exe" -m pytest -q test_quality_guidance.py test_quality_guidance_integration.py` → `46 passed`.

## INCREMENTO 1 — REVIEW QUEUE + REEVALUACIÓN + APROBACIÓN/RECHAZO

### Sprint Goal

Crear la base de datos persistente, los estados estables, la reevaluación por frame y la decisión de aprobar o rechazar sin iniciar regeneración ni entrenamiento.

### Backlog

- [x] Definir estados autorizados y serialización JSONL.
- [x] Crear `FrameQualityReviewManager` con carga/guardado por fila.
- [x] Añadir reevaluación con `QualityGate.evaluate_single_frame`.
- [x] Añadir métodos `approve_frame` y `reject_frame`.
- [x] Añadir listado de cola y persistencia segura.
- [x] Ejecutar tests del Incremento 1 y documentar evidencia.

### Archivos afectados

- `pixel_ai_engine/frame_quality_review.py` (nuevo)
- `pixel_ai_engine/quality_gate.py` (modificar)
- `pixel_ai_engine/__init__.py` (modificar)
- `test_frame_quality_review.py` (nuevo)

### Funciones nuevas

- `FrameQualityReviewManager.register_frame(...)`
- `FrameQualityReviewManager.evaluate_frame(...)`
- `FrameQualityReviewManager.reevaluate_frame(...)`
- `FrameQualityReviewManager.approve_frame(...)`
- `FrameQualityReviewManager.reject_frame(...)`
- `FrameQualityReviewManager.get_queue(...)`
- `QualityGate.evaluate_single_frame(...)`

### Pruebas

- [x] Estado permitido y vocabulario estable.
- [x] Reevaluación por frame.
- [x] Aprobar y rechazar con persistencia.
- [x] Resolución segura de target y ausencia de referencia.
- [x] Validación del archivo JSONL y tratamiento de registros duplicados.

### Riesgos

- [x] Datos incompletos sin target.
- [x] Estados arbitrarios fuera del vocabulario.
- [x] Overwrite sobre rutas fuera del directorio de trabajo.
- [x] Persistencia no atómica ante fallos.

### Criterios de aceptación

- [x] Un frame puede registrarse con metadata mínima.
- [x] La reevaluación recarga métricas sin regenerar la imagen.
- [x] Aprobación y rechazo dejan un historial persistente.
- [x] Si falta target real, el sistema marca `audit_available` como falso y no inventa un score.
- [x] Las decisiones se guardan en JSONL con estados permitidos.

### Definition of Done

- [x] Código terminado.
- [x] Tests PASS.
- [x] Sin regresiones relevantes.
- [x] Documentado.
- [x] Evidencia disponible.

## SPRINT REVIEW

- [x] Sprint 1 aprobado y verificado.
- Evidencia: `test_frame_quality_review.py` → `5 passed in 3.97s`.
- Regresiones: ninguna relevante en la guía de calidad (`46 passed` en `test_quality_guidance.py` + `test_quality_guidance_integration.py`).

## DECISIONES TÉCNICAS

- Se usará JSONL como cola inicial por trazabilidad y simplicidad.
- Los estados quedarán restringidos a un vocabulario fijo.
- Las métricas ausentes se devolverán como `None`/falso, no como aprobación automática.
- La lógica interactiva se encapsula en un manager separado para no mezclar UI y lógica de negocio.
- Se detiene aquí, sin implementar regeneración, hard examples ni batch actions.

## PORCENTAJE REAL COMPLETADO

- 16.7% del roadmap total completado (1/6 incrementos validados).
- 100% del Incremento 1 validado y documentado.
- El resto del roadmap queda bloqueado a la siguiente aprobación del incremento actual, conforme a la restricción de la especificación.
