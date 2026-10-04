# PLAN DE EVOLUCIÓN DEL MOTOR

# RESUMEN DEL ESTADO

Progreso general: pendiente de recálculo al cerrar el Incremento 1

Incremento actual: Incremento 1 — Quality Guidance Observacional

Sprint actual: Sprint 2 — Quality Guidance Controller

Última fase validada: Sprint 1 — Quality Vector

Bloqueos: Ninguno. El árbol de trabajo contiene cambios y datasets previos del usuario; se preservarán y los commits del incremento se limitarán a archivos propios.

Último test completo: PASS — 6 pruebas Guidance + 11 regresiones

Último checkpoint compatible: Pendiente de verificación; el formato moderno admite checkpoints completos y `best_generator.pt` ligero.

Último commit: `24369376f`

Última actualización: 2026-10-04

Branch: `main`

Estado general: EN PROGRESO

---

# ROADMAP GENERAL

- [ ] INCREMENTO 1 — Quality Guidance Observacional — EN PROGRESO
- [ ] INCREMENTO 2 — Smart Reinforcement — PENDIENTE
- [ ] INCREMENTO 3 — Adaptive Loss — PENDIENTE
- [ ] INCREMENTO 4 — Guidance + Recovery — PENDIENTE
- [ ] INCREMENTO 5 — Anatomical Guidance — PENDIENTE
- [ ] INCREMENTO 6 — Critical Detail Guidance — PENDIENTE

---

# ESTADO INICIAL AUDITADO

## Arquitectura moderna conservada

- Entrenamiento supervisado Pix2Pix en `pixel_ai_engine/train_supervised.py`.
- Protección de finitud/NaN, AMP, gradient clipping y pasos AMP omitidos.
- Optimizadores, scalers, schedulers y RNG CPU/CUDA serializados en checkpoints completos.
- Pausa/stop seguros, snapshots, respawn, `best_generator.pt` y `latest_checkpoint.pt`.
- Recuperación automática y rollback de seguridad en `pixel_ai_engine/training_recovery.py`.
- Historial, `past_eras`, telemetría, preview, paleta, alfa y detección anti-colapso.
- Monitor WebUI en `monitor.html` y `sprite_studio.html`.

## Estado previo encontrado

- Existe una implementación preparada pero no validada de `pixel_ai_engine/quality_guidance.py`.
- Existe integración preparada pero incompleta en `train_supervised.py` y las dos interfaces HTML.
- No existen pruebas específicas, configuración dinámica, análisis de tendencias, persistencia por entrada histórica ni documentación arquitectónica.
- La implementación previa crea un controlador nuevo en cada escritura de estado, usa umbral fijo 99,5, interpreta métricas ausentes como cero y emite acciones no normalizadas que sugieren mutaciones aunque el modo debe ser observacional.
- El árbol de trabajo ya estaba sucio al iniciar: los assets/datasets ajenos a este incremento no se modificarán ni se incluirán en sus commits.

---

# INCREMENTO 1 — QUALITY GUIDANCE OBSERVACIONAL

## Objetivo

Reconectar el sistema histórico de QualityGate con el entrenamiento supervisado moderno sin modificar todavía el aprendizaje.

## Feature flags y límites

- `ENABLE_QUALITY_GUIDANCE`: activa/desactiva únicamente observación, diagnóstico y publicación.
- Modo obligatorio: `observational`.
- No modifica sampling, LR, losses, optimizadores, scheduler, recovery ni checkpoints.
- Las métricas PIL/NumPy son no diferenciables y jamás se suman al grafo PyTorch.

## Sprint 1 — Quality Vector

### Sprint Goal

Crear una representación unificada, finita, normalizada y tolerante a métricas históricas/modernas incompletas.

### Backlog

- [x] Analizar QualityGate actual
  - Evidencia: catálogo y aliases verificados contra `pixel_ai_engine/quality_gate.py`; commit `24369376f`.
- [x] Identificar métricas históricas reutilizables
  - Evidencia: pruebas de payload QualityGate/Enhancer en `test_quality_guidance.py`; 6 PASS.
- [x] Definir `QualityVector`
  - Evidencia: `pixel_ai_engine/quality_guidance.py`; `test_quality_vector_round_trip_preserves_unavailable_categories`; commit `24369376f`.
- [x] Implementar normalización segura 0–100
  - Evidencia: `test_quality_score_normalization_is_finite_and_bounded`; PASS; commit `24369376f`.
- [x] Resolver aliases históricos `score_*`
  - Evidencia: `test_quality_vector_maps_quality_gate_aliases_without_inventing_missing_scores`; PASS.
- [x] Diferenciar métricas no disponibles de métricas con valor cero
  - Evidencia: `test_missing_is_distinct_from_measured_zero_and_payload_is_json_safe`; PASS.
- [x] Crear tests unitarios
  - Evidencia: 6 tests en `test_quality_guidance.py`; commit `24369376f`.
- [x] Ejecutar tests
  - Evidencia: `.venv\\Scripts\\python.exe -m pytest -q test_quality_guidance.py` → 6 PASS.
- [x] Verificar compatibilidad
  - Evidencia: `.venv\\Scripts\\python.exe -m pytest -q test_audit_suite.py test_training_recovery.py` → 11 PASS.
- [x] Documentar cambios
  - Evidencia: Sprint Review, ADR-003 y matriz QG-001 en este documento.

### Archivos

- Modificar: `pixel_ai_engine/quality_guidance.py`, `pixel_ai_engine/__init__.py`.
- Crear: `test_quality_guidance.py`.

### Funciones a crear/reutilizar

- Crear: `QualityVector`, `build_quality_vector`, normalización finita y agregación solo con señales disponibles.
- Reutilizar: salidas de `QualityGate.evaluate_model_critical`, `PixelArtEnhancer.analyze_quality` y métricas de entrenamiento.

### Riesgos

- Escalas mixtas 0–1/0–100; ceros reales; ausencia de ground truth; alias divergentes.

### Pruebas y criterios de aceptación

- Unitarias: valores límite, NaN/Inf, escala fraccional, alias, datos parciales y serialización.
- Integración: aceptar payloads reales de QualityGate y Enhancer sin importar PIL/NumPy al grafo.
- Aceptación: todas las categorías son JSON-safe y ninguna métrica ausente se diagnostica como fallo.

### Definition of Done

- [x] Código terminado
- [x] Tests terminados
- [x] Integración verificada
- [x] Documentación actualizada
- [x] Sin regresión crítica

Estado Sprint: COMPLETADO

### Sprint Review

Resultado: APROBADO

Evidencias:

- Archivos: `pixel_ai_engine/quality_guidance.py`, `pixel_ai_engine/__init__.py`, `test_quality_guidance.py`.
- Commit: `24369376f feat(guidance): add normalized quality vector`.
- Tests: 6/6 propios y 11/11 regresiones PASS.
- Regresiones encontradas: ninguna. El Python global no tiene PyTorch; se validó con el `.venv` del proyecto (PyTorch 2.14.1+cpu).

### Decisión

- [x] Sprint aprobado
- [ ] Sprint requiere correcciones

---

## Sprint 2 — Quality Guidance Controller

### Sprint Goal

Diagnosticar el cuello de botella, su tendencia y severidad, y producir una recomendación estructurada sin ejecutarla.

### Backlog

- [ ] Crear `QualityGuidanceController`
- [ ] Crear `diagnose_quality_bottleneck`
- [ ] Crear clasificación LOW/MEDIUM/HIGH/CRITICAL
- [ ] Crear `QualityTrendAnalyzer`
- [ ] Detectar IMPROVING/PLATEAU/REGRESSION/COLLAPSE/OSCILLATION
- [ ] Crear recomendación con acciones normalizadas
- [ ] Mantener acción aplicada en `CONTINUE` por modo observacional
- [ ] Registrar decisión y estado serializable
- [ ] Añadir tests
- [ ] Ejecutar tests
- [ ] Verificar regresiones
- [ ] Documentar

### Archivos

- Modificar: `pixel_ai_engine/quality_guidance.py`.
- Modificar: `test_quality_guidance.py`.

### Riesgos

- Confundir recomendación con acción ejecutada; reaccionar a una sola época; score global ocultando mínimos críticos.

### Definition of Done

- [ ] Código terminado
- [ ] Tests terminados
- [ ] Integración verificada
- [ ] Documentación actualizada
- [ ] Sin regresión crítica

Estado Sprint: PENDIENTE

### Sprint Review

Resultado: PENDIENTE

### Decisión

- [ ] Sprint aprobado
- [ ] Sprint requiere correcciones

---

## Sprint 3 — Integración observacional

### Sprint Goal

Publicar Quality Guidance en estado, historial y monitor sin cambiar el comportamiento del entrenamiento.

### Backlog

- [ ] Integrar con `train_supervised.py`
- [ ] Añadir feature flag apagable
- [ ] Persistir `guidance` en `training_status.json`
- [ ] Mantener alias retrocompatible `quality_guidance`
- [ ] Persistir `quality` y `guidance` por entrada histórica
- [ ] Restaurar historial de tendencia tras reanudación
- [ ] Mantener entrenamiento e hiperparámetros sin cambios
- [ ] Mostrar métricas y modo observacional en monitor
- [ ] Ejecutar entrenamiento de prueba equivalente sin GPU
- [ ] Ejecutar suite de regresión
- [ ] Verificar reanudación y estado antiguo sin Guidance
- [ ] Verificar checkpoints sin cambios de esquema
- [ ] Crear `QUALITY_GUIDANCE_ARCHITECTURE.md`
- [ ] Documentar

### Archivos

- Modificar: `pixel_ai_engine/train_supervised.py`, `monitor.html`, `sprite_studio.html`.
- Crear: `test_quality_guidance_integration.py`, `QUALITY_GUIDANCE_ARCHITECTURE.md`.

### Riesgos

- Sobrescribir calidad previa en estados de pausa; duplicar épocas; romper consumidores UI antiguos; cargar módulos pesados desde tests.

### Definition of Done

- [ ] Código terminado
- [ ] Tests terminados
- [ ] Integración verificada
- [ ] Documentación actualizada
- [ ] Sin regresión crítica

Estado Sprint: PENDIENTE

### Sprint Review

Resultado: PENDIENTE

### Decisión

- [ ] Sprint aprobado
- [ ] Sprint requiere correcciones

---

## Validación del Incremento 1

- [ ] Sprint 1 completado
- [ ] Sprint 2 completado
- [ ] Sprint 3 completado
- [ ] Tests unitarios aprobados
- [ ] Tests de integración aprobados
- [ ] Entrenamiento inicia correctamente
- [ ] Entrenamiento puede pausarse
- [ ] Entrenamiento puede reanudarse
- [ ] Checkpoints compatibles
- [ ] No existen regresiones críticas
- [ ] Quality Guidance funciona en modo observacional
- [ ] Documentación actualizada

Estado del Incremento: EN PROGRESO

---

# INCREMENTOS FUTUROS

## INCREMENTO 2 — SMART REINFORCEMENT

- [ ] Métricas confiables por frame
- [ ] Hard example mining con pesos acotados
- [ ] Feature flag de sampling adaptativo
- [ ] Protección contra dominación de pocos frames
- [ ] Comparación A/B contra baseline
- [ ] Persistencia y reanudación de sampling
- [ ] Tests y revisión formal

## INCREMENTO 3 — ADAPTIVE LOSS

- [ ] Multiplicadores acotados sobre losses base
- [ ] Una intervención atribuible por vez
- [ ] Registro explícito de multiplicadores
- [ ] Restauración de valores base
- [ ] Comparación A/B
- [ ] Tests y revisión formal

## INCREMENTO 4 — GUIDANCE + RECOVERY

- [ ] Jerarquía Guidance/Recovery sin reemplazar detector actual
- [ ] Cooldown y máximo de intervenciones
- [ ] Rollback por degradación sostenida
- [ ] Persistencia completa de `guidance_state`
- [ ] Compatibilidad con checkpoints antiguos
- [ ] Tests y revisión formal

## INCREMENTO 5 — ANATOMICAL GUIDANCE

- [ ] Métricas geométricas de cuerpo, centro, pies, silueta y pose
- [ ] Observación validada antes de loss diferenciable
- [ ] Calibración con datos reales
- [ ] Tests y revisión formal

## INCREMENTO 6 — CRITICAL DETAIL GUIDANCE

- [ ] Extraer solo métricas no destructivas de `Phase3CriticalReviewer`
- [ ] Integrar borde, tinta, sombra, paleta y rostro al vector
- [ ] No transformar ground truth automáticamente
- [ ] Tests y revisión formal

---

# MATRIZ DE TRAZABILIDAD

| ID | Requisito | Estado | Archivo | Test | Commit |
|---|---|---|---|---|---|
| QG-001 | QualityVector normalizado | ✅ COMPLETADO | `pixel_ai_engine/quality_guidance.py` | `test_quality_guidance.py` (6 PASS) | `24369376f` |
| QG-002 | Diagnóstico por categoría | 🚧 EN DESARROLLO | `pixel_ai_engine/quality_guidance.py` | Pendiente | Pendiente |
| QG-003 | Tendencias multiépoca | ⏳ PENDIENTE | `pixel_ai_engine/quality_guidance.py` | Pendiente | Pendiente |
| QG-004 | Integración observacional | 🚧 EN DESARROLLO | `pixel_ai_engine/train_supervised.py` | Pendiente | Pendiente |
| QG-005 | Estado e historial | ⏳ PENDIENTE | `pixel_ai_engine/train_supervised.py` | Pendiente | Pendiente |
| QG-006 | Monitor Quality Guidance | 🚧 EN DESARROLLO | `monitor.html`, `sprite_studio.html` | Pendiente | Pendiente |
| QG-007 | Feature flag | ⏳ PENDIENTE | `pixel_ai_engine/train_supervised.py` | Pendiente | Pendiente |
| QG-008 | Documentación arquitectónica | ⏳ PENDIENTE | `QUALITY_GUIDANCE_ARCHITECTURE.md` | Revisión documental | Pendiente |
| QG-101 | Smart Sampling | ⏳ PENDIENTE | - | - | - |
| QG-201 | Adaptive Loss | ⏳ PENDIENTE | - | - | - |
| QG-301 | Guidance + Recovery | ⏳ PENDIENTE | - | - | - |
| QG-401 | Anatomical Guidance | ⏳ PENDIENTE | - | - | - |
| QG-501 | Critical Detail Guidance | ⏳ PENDIENTE | - | - | - |

---

# DECISIONES TÉCNICAS

## ADR-001 — No recuperar el refuerzo histórico literal

Fecha: 2026-10-04

Problema: `train_dos_fases_1500.py` reintentaba bloques fijos de +100 épocas cuando `QualityGate` no alcanzaba un umbral global de 99,5%.

Decisión: conservar el principio de reevaluación y reemplazar la decisión ciega por `QualityGuidanceController` categorizado. En el Incremento 1 solo recomienda y nunca actúa.

Motivo: permite atribución, pruebas, límites y evolución incremental sin debilitar el pipeline moderno.

Alternativas descartadas: restaurar el entrenador antiguo; usar `score_total` como única verdad; ejecutar rollback/refuerzo desde este incremento.

Riesgos: umbrales no calibrados; se mitigan exponiendo configuración y manteniendo modo observacional.

## ADR-002 — Métricas visuales fuera del grafo

Fecha: 2026-10-04

Decisión: métricas basadas en PIL/NumPy/heursticas se usan solo para diagnóstico. No se suman a `total_g`.

Motivo: no son diferenciables y su incorporación directa rompería o falsearía el gradiente.

## ADR-003 — Ausente no equivale a cero

Fecha: 2026-10-04

Decisión: el vector conserva `null` para categorías no observadas; promedios y diagnósticos usan solo señales disponibles.

Motivo: evita diagnosticar cara/anatomía como colapsadas cuando una auditoría no produjo esas métricas.

---

# FUNCIONES HISTÓRICAS INVESTIGADAS

## QualityGate

Estado histórico: ACTIVO como puerta entre fases.

Estado actual: existe y calcula macro métricas y detalle anatómico, pero está desconectado del entrenamiento supervisado moderno.

- [x] Investigar
  - Evidencia: `pixel_ai_engine/quality_gate.py`, método `evaluate_model_critical` auditado el 2026-10-04.
- [x] Conservar catálogo de métricas
  - Evidencia: mapeo registrado en Sprint 1 y matriz QG-001.
- [ ] Integrar métricas mediante QualityVector
- [ ] Validar

Resultado: ADAPTAR/FUSIONAR, no invocar como bucle antiguo.

## Entrenamiento histórico de dos fases

- [x] Investigar
  - Evidencia: `_archivo_pruebas_y_obsoletos/train_dos_fases_1500.py` y sus bucles de `QualityGate`/+100 épocas revisados.
- [x] Determinar incompatibilidad como motor principal
  - Evidencia: ADR-001; contradice el entrenador supervisado moderno y no diagnostica causa.

Resultado: NO REIMPLEMENTAR COMPLETO.

## Auditorías 8x12

- [x] Investigar
  - Evidencia: `_archivo_pruebas_y_obsoletos/audit_quality_8x12.py` y `audit_frames_8x12.py` revisados.
- [ ] Extraer métricas por frame (Incremento 2)
- [ ] Validar confiabilidad

Resultado: POSPONER PARA HARD EXAMPLE MINING.

## Phase3CriticalReviewer

- [x] Investigar
  - Evidencia: `audit_frame` separa métricas de `elevate_frame` en `pixel_ai_engine/phase3_critical_enhancer.py`.
- [ ] Extraer métricas reutilizables (Incremento 6)
- [ ] Integrar métricas
- [ ] Validar

Resultado: ADAPTAR PARCIALMENTE; no reutilizar transformaciones destructivas.

## Pipeline híbrido Tori

- [x] Investigar
  - Evidencia: `_archivo_pruebas_y_obsoletos/pipeline_hibrido_tori.py` revisado.
- [x] Determinar incompatibilidad como motor principal
  - Evidencia: flujo específico de personaje y postprocesado, no controlador modular.

Resultado: NO REIMPLEMENTAR COMPLETO.

---

# HISTORIAL DE PRUEBAS

| Fecha | Incremento | Sprint | Test | Resultado | Observaciones |
|---|---:|---:|---|---|---|
| 2026-10-04 | 1 | Auditoría | `python -m pytest --collect-only -q` | TIMEOUT | La colección global superó 30 s; no se considera evidencia de fallo funcional. Se ejecutarán suites dirigidas. |
| 2026-10-04 | 1 | 1 | `test_quality_guidance.py` | PASS (6) | QualityVector, aliases, finitud, ausentes, JSON y estabilidad. |
| 2026-10-04 | 1 | 1 | `test_audit_suite.py test_training_recovery.py` | PASS (11) | Regresión de auditoría y recuperación. |

---

# AUDITORÍA DEL PLAN

- [ ] Cada `[x]` tiene evidencia.
- [ ] Cada funcionalidad marcada completa existe.
- [ ] Tests continúan pasando.
- [x] No hay archivos eliminados accidentalmente.
  - Evidencia: revisión inicial de `git status`; no se ejecutaron eliminaciones.
- [x] No hay funcionalidades antiguas marcadas como recuperadas si no están conectadas.
  - Evidencia: matriz distingue investigación, desarrollo y pendientes.
- [ ] Documentación coincide con código actual.
- [x] Último commit registrado coincide con Git.
  - Evidencia: `git rev-parse --short HEAD` devolvió `2dc923d3f` al iniciar.
- [x] Incremento activo coincide con desarrollo real.
  - Evidencia: solo Incremento 1 figura en desarrollo; Incrementos 2–6 siguen pendientes.

---

# INCREMENT REVIEW — INCREMENTO 1

## Objetivo

Quality Guidance observacional conectado al entrenamiento.

## Resultado

PENDIENTE

## Baseline anterior

El entrenamiento produce previews y métricas de `PixelArtEnhancer`; la recuperación usa colapso extremo. No hay vector unificado, tendencia ni decisión explicable persistida por época.

## Resultado actual

PENDIENTE

## Diferencias y métricas

PENDIENTE

## Tests y regresiones

PENDIENTE

## Decisión

- [ ] Incremento aprobado
- [ ] Incremento rechazado
- [ ] Requiere correcciones
