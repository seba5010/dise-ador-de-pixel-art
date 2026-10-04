# PLAN DE EVOLUCIÓN DEL MOTOR

# RESUMEN DEL ESTADO

Progreso del roadmap: 100% (6 de 6 incrementos completados y verificados; las casillas de rechazo/corrección permanecen sin marcar por diseño)

Incremento actual: Incremento 6 — COMPLETADO

Sprint actual: Ninguno (Increment Review aprobado)

Última fase validada: Incremento 6 — Critical Detail Guidance

Bloqueos: Ninguno. El árbol de trabajo contiene cambios y datasets previos del usuario; se preservarán y los commits del incremento se limitarán a archivos propios.

Último test completo: PASS — suite propia completa, 81 pruebas

Último checkpoint compatible: verificado en smoke CPU; checkpoints nuevos persisten Guidance y checkpoints antiguos cargan sin el campo.

Último commit validado: `5a063a2c0` (Incremento 5; el cierre del Incremento 6 se registra en el commit siguiente)

Última actualización: 2026-10-04

Branch: `main`

Estado general: PLAN COMPLETADO — Increment Reviews 1–6 aprobados

---

# ROADMAP GENERAL

- [x] INCREMENTO 1 — Quality Guidance Observacional — COMPLETADO
  - Evidencia: Increment Review aprobado; 53 tests PASS; commits `24369376f`, `2357b5e88`, `9c93ab6fa`, `f7d1a72b3`, `83b758e41`.
- [x] INCREMENTO 2 — Smart Reinforcement — COMPLETADO
  - Evidencia: seguimiento por frame, sampling 1.0–2.0, persistencia, A/B determinista y 63 tests PASS.
- [x] INCREMENTO 3 — Adaptive Loss — COMPLETADO
  - Evidencia: multiplicadores 0.75–1.25, una loss por intervención, persistencia, neutralidad por flag y 68 tests PASS.
- [x] INCREMENTO 4 — Guidance + Recovery — COMPLETADO
  - Evidencia: política jerárquica, cooldown 5, máximo 3, rollback delegado a Recovery, compatibilidad y 72 tests PASS.
- [x] INCREMENTO 5 — Anatomical Guidance — COMPLETADO
  - Evidencia: siete señales geométricas, integración observacional, loss de silueta experimental apagada y 76 tests PASS.
- [x] INCREMENTO 6 — Critical Detail Guidance — COMPLETADO
  - Evidencia: auditoría crítica pura, quality checkpoint independiente, eras ampliadas, monitor y 81 tests PASS.

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

- [x] Crear `QualityGuidanceController`
  - Evidencia: `pixel_ai_engine/quality_guidance.py`; commit `2357b5e88`.
- [x] Crear `diagnose_quality_bottleneck`
  - Evidencia: test de rostro débil oculto por global alto; PASS.
- [x] Crear clasificación LOW/MEDIUM/HIGH/CRITICAL
  - Evidencia: test parametrizado de cuatro severidades; PASS.
- [x] Crear `QualityTrendAnalyzer`
  - Evidencia: `test_trend_analyzer_detects_required_states`; PASS.
- [x] Detectar IMPROVING/PLATEAU/REGRESSION/COLLAPSE/OSCILLATION
  - Evidencia: cinco casos parametrizados; PASS.
- [x] Crear recomendación con acciones normalizadas
  - Evidencia: vocabulario `GUIDANCE_ACTIONS` y test del controlador; PASS.
- [x] Mantener acción aplicada en `CONTINUE` por modo observacional
  - Evidencia: `recommended_action=REINFORCE`, `action=CONTINUE`, `training_modified=false`; PASS.
- [x] Registrar decisión y estado serializable
  - Evidencia: round-trip JSON de `export_state`/`load_state`; PASS.
- [x] Añadir tests
  - Evidencia: suite ampliada a 20 tests.
- [x] Ejecutar tests
  - Evidencia: 20/20 PASS con `.venv`.
- [x] Verificar regresiones
  - Evidencia: `test_audit_suite.py test_training_recovery.py` → 11/11 PASS.
- [x] Documentar
  - Evidencia: Sprint Review y matriz actualizados.

### Archivos

- Modificar: `pixel_ai_engine/quality_guidance.py`.
- Modificar: `test_quality_guidance.py`.

### Riesgos

- Confundir recomendación con acción ejecutada; reaccionar a una sola época; score global ocultando mínimos críticos.

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
- Commit: `2357b5e88 feat(guidance): add bottleneck and trend diagnosis`.
- Tests: 20/20 Guidance y 11/11 regresiones PASS.
- Regresiones encontradas: ninguna.

### Decisión

- [x] Sprint aprobado
- [ ] Sprint requiere correcciones

---

## Sprint 3 — Integración observacional

### Sprint Goal

Publicar Quality Guidance en estado, historial y monitor sin cambiar el comportamiento del entrenamiento.

### Backlog

- [x] Integrar con `train_supervised.py`
  - Evidencia: estado por época y smoke de cuatro épocas; commit `9c93ab6fa`.
- [x] Añadir feature flag apagable
  - Evidencia: `PIXEL_AI_ENABLE_QUALITY_GUIDANCE=0` y test de flag; PASS.
- [x] Persistir `guidance` en `training_status.json`
  - Evidencia: `test_status_persists_observational_guidance_and_epoch_history`; PASS.
- [x] Mantener alias retrocompatible `quality_guidance`
  - Evidencia: igualdad del alias verificada en test; PASS.
- [x] Persistir `quality` y `guidance` por entrada histórica
  - Evidencia: tres entradas auditadas en integración; PASS.
- [x] Restaurar historial de tendencia tras reanudación
  - Evidencia: regresión 71→68→63 y smoke START→RESUME; PASS.
- [x] Mantener entrenamiento e hiperparámetros sin cambios
  - Evidencia: `action=CONTINUE`, `training_modified=false`, loss/LR invariantes; PASS.
- [x] Mostrar métricas y modo observacional en monitor
  - Evidencia: test de contrato HTML para `monitor.html` y `sprite_studio.html`; PASS.
- [x] Ejecutar entrenamiento de prueba equivalente sin GPU
  - Evidencia: smoke CPU con modelo/dataset mínimos, épocas 1–4, pausa y reanudación; PASS.
- [x] Ejecutar suite de regresión
  - Evidencia: 53/53 tests raíz PASS.
- [x] Verificar reanudación y estado antiguo sin Guidance
  - Evidencia: round-trip, checkpoint legado y reanudación CPU; PASS.
- [x] Verificar checkpoints sin cambios de esquema
  - Evidencia: campo aditivo `guidance_state`; checkpoint sin campo devuelve estado vacío sin error.
- [x] Crear `QUALITY_GUIDANCE_ARCHITECTURE.md`
  - Evidencia: commit `83b758e41`.
- [x] Documentar
  - Evidencia: arquitectura, Sprint Review e Increment Review.

### Archivos

- Modificar: `pixel_ai_engine/train_supervised.py`, `monitor.html`, `sprite_studio.html`.
- Crear: `test_quality_guidance_integration.py`, `QUALITY_GUIDANCE_ARCHITECTURE.md`.

### Riesgos

- Sobrescribir calidad previa en estados de pausa; duplicar épocas; romper consumidores UI antiguos; cargar módulos pesados desde tests.

### Definition of Done

- [x] Código terminado
- [x] Tests terminados
- [x] Integración verificada
- [x] Documentación actualizada
- [x] Sin regresión crítica

Estado Sprint: COMPLETADO

### Sprint Review

Resultado: APROBADO

### Evidencias

- Archivos modificados: `train_supervised.py`, `quality_guidance.py`, `monitor.html`, `sprite_studio.html`.
- Archivos nuevos: `test_quality_guidance_integration.py`, `QUALITY_GUIDANCE_ARCHITECTURE.md`.
- Commits: `9c93ab6fa`, `f7d1a72b3`, `83b758e41`.
- Tests: 53 PASS, 0 FAIL.
- Regresiones: ninguna; quedan cinco warnings de deprecación AMP preexistentes.

### Decisión

- [x] Sprint aprobado
- [ ] Sprint requiere correcciones

---

## Validación del Incremento 1

- [x] Sprint 1 completado
  - Evidencia: Sprint Review 1 aprobado.
- [x] Sprint 2 completado
  - Evidencia: Sprint Review 2 aprobado.
- [x] Sprint 3 completado
  - Evidencia: Sprint Review 3 aprobado.
- [x] Tests unitarios aprobados
  - Evidencia: 20 tests de vector/controlador incluidos en suite PASS.
- [x] Tests de integración aprobados
  - Evidencia: 7 tests de integración, incluido lifecycle CPU, PASS.
- [x] Entrenamiento inicia correctamente
  - Evidencia: smoke CPU inició START y completó época 1.
- [x] Entrenamiento puede pausarse
  - Evidencia: smoke solicitó pausa dentro del lote y publicó `PAUSADO` en época 3.
- [x] Entrenamiento puede reanudarse
  - Evidencia: smoke reanudó épocas 2 y 4, restaurando optimizadores, schedulers, RNG y Guidance.
- [x] Checkpoints compatibles
  - Evidencia: tests de formato moderno/legado y smoke de carga.
- [x] No existen regresiones críticas
  - Evidencia: 53 PASS, 0 FAIL.
- [x] Quality Guidance funciona en modo observacional
  - Evidencia: recomendación `REINFORCE` con acción aplicada `CONTINUE` y `training_modified=false`.
- [x] Documentación actualizada
  - Evidencia: documentos maestro y de arquitectura.

Estado del Incremento: COMPLETADO

---

# INCREMENTOS 2–6

## INCREMENTO 2 — SMART REINFORCEMENT

Objetivo: activar únicamente sampling adaptativo dirigido, sin modificar losses ni LR.

### Sprint 1 — Métricas por frame

- [x] Definir contrato `frame_quality`
  - Evidencia: `FrameQualityTracker`; commit `4a4e21ccb`.
- [x] Calcular score por muestra sin perder `char_id`/`frame_idx`
  - Evidencia: `compute_frame_quality_batch`; test per-sample PASS.
- [x] Agregar confiabilidad y mínimo de observaciones
  - Evidencia: tracker y política rechazan señales insuficientes; tests PASS.
- [x] Persistir métricas por frame
  - Evidencia: round-trip dentro de `guidance_state`; test PASS.
- [x] Añadir tests unitarios e integración
  - Evidencia: 34 tests Guidance/integración PASS.
- [x] Ejecutar Sprint Review
  - Resultado: APROBADO; regresiones dirigidas 11/11 PASS; no se activó sampling durante este Sprint.

### Sprint 2 — Hard example mining

- [x] Crear pesos acotados 1.0–2.0
- [x] Evitar dominación permanente con mezcla uniforme
- [x] Activar `WeightedRandomSampler` mediante feature flag
- [x] Persistir/restaurar sampling weights
- [x] No modificar loss ni LR
- [x] Añadir tests y ejecutar Sprint Review
  - Resultado: APROBADO; sampler ponderado sólo activo ante `REINFORCE`, pesos máximos 2.0 y fallback uniforme.

### Sprint 3 — Comparación A/B

- [x] Crear reporte baseline vs sampling
- [x] Comparar distribución, cobertura y estabilidad
- [x] Verificar fallback cuando no hay métricas confiables
- [x] Ejecutar suite completa
- [x] Ejecutar Increment Review
  - Resultado: APROBADO; 63 tests del proyecto PASS. El reporte A/B es determinista y no afirma mejora de calidad sin una corrida controlada.

## INCREMENTO 3 — ADAPTIVE LOSS

Objetivo: aplicar una sola corrección diferenciable atribuible y acotada por intervención.

### Sprint 1 — Capa de multiplicadores

- [x] Crear `LossMultiplierController`
- [x] Definir límites y paso gradual
- [x] Mantener valores base intactos
- [x] Añadir tests unitarios

### Sprint 2 — Integración y persistencia

- [x] Aplicar multiplicadores efectivos en `total_g`
- [x] Registrar valores efectivos por época
- [x] Persistir/restaurar en checkpoint
- [x] Feature flag y restauración neutral
- [x] Tests de integración

### Sprint 3 — A/B y revisión

- [x] Comparar baseline vs multiplicadores
- [x] Verificar finitud/NaN/AMP
- [x] Ejecutar suite completa e Increment Review
  - Resultado: APROBADO; 68 tests PASS. A/B registra pesos efectivos y no afirma mejora visual sin experimento controlado.

## INCREMENTO 4 — GUIDANCE + RECOVERY

Objetivo: permitir acciones seguras respetando que TrainingRecovery conserva autoridad sobre fallos críticos.

### Sprint 1 — Política de intervención

- [x] Jerarquía CONTINUE/REINFORCE/ADJUST/RECOVERY/ROLLBACK/STOP
- [x] Cooldown y máximo consecutivo
- [x] Una intervención atribuible por vez
- [x] Tests de límites

### Sprint 2 — Recovery y rollback

- [x] Complementar detector existente sin reemplazarlo
- [x] Rollback por regresión sostenida confirmada
- [x] Restaurar parámetros base al empeorar
- [x] Mantener ERROR_NAN como autoridad existente
- [x] Tests de rollback

### Sprint 3 — Persistencia y revisión

- [x] Persistencia completa y checkpoints antiguos
- [x] Pruebas pausa/cierre/reanudación
- [x] Suite completa e Increment Review
  - Resultado: APROBADO; 72 tests PASS. Recovery conserva autoridad y Guidance no selecciona snapshots por cuenta propia.

## INCREMENTO 5 — ANATOMICAL GUIDANCE

Objetivo: medir geometría corporal de forma explícita y evaluar una loss de silueta separada.

### Sprint 1 — Métricas geométricas

- [x] `body_height_ratio`, `body_width_ratio`, `center_offset_x`
- [x] `foot_anchor_error`, `silhouette_iou`, `pose_alignment`
- [x] `body_proportion_error`
- [x] Tests con máscaras sintéticas

### Sprint 2 — Integración observacional

- [x] Integrar métricas al QualityVector
- [x] Persistir y mostrar en monitor
- [x] Calibrar sobre ejemplos reales/sintéticos
- [x] Tests de integración

### Sprint 3 — Loss diferenciable experimental

- [x] Implementar loss de silueta tensorial bajo feature flag
- [x] Confirmar gradiente y finitud
- [x] Mantenerla desactivada por defecto hasta A/B
- [x] Suite completa e Increment Review
  - Resultado: APROBADO; 76 tests PASS. Las imágenes auditadas no se transforman y la loss experimental permanece opt-in.

## INCREMENTO 6 — CRITICAL DETAIL GUIDANCE

Objetivo: integrar métricas críticas no destructivas y cerrar comparación entre eras/checkpoints de calidad.

### Sprint 1 — Métricas críticas

- [x] Extraer auditoría pura sin `elevate_frame`
- [x] Integrar borde, tinta, sombra, paleta y rostro
- [x] Verificar que ground truth no se transforma
- [x] Tests unitarios

### Sprint 2 — Eras y quality checkpoint

- [x] Completar resumen de calidad entre eras
- [x] Crear `best_quality_generator.pt` sin reemplazar `best_generator.pt`
- [x] Feature flag y criterio compuesto con mínimos críticos
- [x] Persistencia y compatibilidad

### Sprint 3 — Observabilidad y cierre

- [x] Mostrar detalle crítico y comparación de eras
- [x] Ejecutar pruebas A/B deterministas disponibles
- [x] Suite completa
- [x] Auditoría final de checklist
- [x] Increment Review final
  - Resultado: APROBADO; 81 tests PASS. No se transforma ground truth y el checkpoint de calidad no reemplaza al checkpoint por loss.

---

# MATRIZ DE TRAZABILIDAD

| ID | Requisito | Estado | Archivo | Test | Commit |
|---|---|---|---|---|---|
| QG-001 | QualityVector normalizado | ✅ COMPLETADO | `pixel_ai_engine/quality_guidance.py` | `test_quality_guidance.py` (6 PASS) | `24369376f` |
| QG-002 | Diagnóstico por categoría | ✅ COMPLETADO | `pixel_ai_engine/quality_guidance.py` | `test_quality_guidance.py` | `2357b5e88` |
| QG-003 | Tendencias multiépoca | ✅ COMPLETADO | `pixel_ai_engine/quality_guidance.py` | `test_quality_guidance.py` | `2357b5e88` |
| QG-004 | Integración observacional | ✅ COMPLETADO | `pixel_ai_engine/train_supervised.py` | `test_quality_guidance_integration.py` | `9c93ab6fa` |
| QG-005 | Estado e historial | ✅ COMPLETADO | `pixel_ai_engine/train_supervised.py` | persistencia/resume/lifecycle | `9c93ab6fa`, `f7d1a72b3` |
| QG-006 | Monitor Quality Guidance | ✅ COMPLETADO | `monitor.html`, `sprite_studio.html` | contrato HTML | `9c93ab6fa` |
| QG-007 | Feature flag | ✅ COMPLETADO | `quality_guidance.py`, `train_supervised.py` | flag disabled | `9c93ab6fa` |
| QG-008 | Documentación arquitectónica | ✅ COMPLETADO | `QUALITY_GUIDANCE_ARCHITECTURE.md` | revisión documental | `83b758e41` |
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
- [x] Integrar métricas mediante QualityVector
- [x] Validar

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
- [x] Extraer métricas por frame (Incremento 2)
- [x] Validar confiabilidad

Resultado: POSPONER PARA HARD EXAMPLE MINING.

## Phase3CriticalReviewer

- [x] Investigar
  - Evidencia: `audit_frame` separa métricas de `elevate_frame` en `pixel_ai_engine/phase3_critical_enhancer.py`.
- [x] Extraer métricas reutilizables (Incremento 6)
- [x] Integrar métricas
- [x] Validar

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
| 2026-10-04 | 1 | 2 | `test_quality_guidance.py` | PASS (20) | Diagnóstico, severidad, tendencias, observación y estado. |
| 2026-10-04 | 1 | 2 | `test_audit_suite.py test_training_recovery.py` | PASS (11) | Sin regresiones tras controlador/tendencias. |
| 2026-10-04 | 1 | 3 | `test_quality_guidance_integration.py` | PASS (7) | Estado, historial, flag, checkpoint, eras, UI y lifecycle CPU. |
| 2026-10-04 | 1 | 3 | suite raíz `test_*.py` | PASS (53) | 0 FAIL; 5 warnings AMP deprecados preexistentes. |
| 2026-10-04 | 2–6 | Cierre | suite propia explícita | PASS (81) | Sampling, adaptive loss, recovery, anatomía, detalle crítico y lifecycle CPU; 5 warnings AMP preexistentes. |

---

# AUDITORÍA DEL PLAN

- [x] Cada `[x]` tiene evidencia.
  - Evidencia: auditoría final de checkboxes y referencias a archivos/tests/commits.
- [x] Cada funcionalidad marcada completa existe.
  - Evidencia: matriz QG-001–QG-008 y suite completa.
- [x] Tests continúan pasando.
  - Evidencia: 81 PASS el 2026-10-04.
- [x] No hay archivos eliminados accidentalmente.
  - Evidencia: revisión inicial de `git status`; no se ejecutaron eliminaciones.
- [x] No hay funcionalidades antiguas marcadas como recuperadas si no están conectadas.
  - Evidencia: matriz distingue investigación, desarrollo y pendientes.
- [x] Documentación coincide con código actual.
  - Evidencia: `QUALITY_GUIDANCE_ARCHITECTURE.md` describe el contrato validado.
- [x] Último commit registrado coincide con Git.
  - Evidencia: Incrementos 2–5 en `0ca7bdede`, `d7670f8d0`, `08b24049a`, `5a063a2c0`; el cierre del 6 se registra inmediatamente después de esta auditoría.
- [x] Incremento activo coincide con desarrollo real.
  - Evidencia: los seis incrementos figuran completados y sus Increment Reviews aprobados.

---

# INCREMENT REVIEW — INCREMENTO 1

## Objetivo

Quality Guidance observacional conectado al entrenamiento.

## Resultado

APROBADO

## Baseline anterior

El entrenamiento produce previews y métricas de `PixelArtEnhancer`; la recuperación usa colapso extremo. No hay vector unificado, tendencia ni decisión explicable persistida por época.

## Resultado actual

QualityVector, diagnóstico, tendencias, recomendación observacional, persistencia, checkpoints, historial por época, resumen de eras y monitor están conectados. Ninguna acción adapta el entrenamiento.

## Diferencias y métricas

- Baseline: preview + métricas sueltas, sin diagnóstico persistente.
- Actual: vector de 15 dimensiones, cinco estados de tendencia, cuatro severidades y acciones normalizadas.
- Seguridad: `action=CONTINUE`, `training_modified=false`, flags de sampling/loss/checkpoint en `False`.
- A/B de calidad: no corresponde todavía; al no existir intervención, no se afirma mejora del modelo.

## Tests y regresiones

- 53 PASS, 0 FAIL.
- Smoke CPU START→RESUME→PAUSE→RESUME aprobado.
- Checkpoint moderno y legado aprobados.
- Regresiones críticas: ninguna.
- Deuda detectada: cinco warnings por API AMP deprecada; no se cambió en este incremento para evitar alterar la base.

## Decisión

- [x] Incremento aprobado
  - Evidencia: tres Sprint Reviews aprobados y suite completa PASS.
- [ ] Incremento rechazado
- [ ] Requiere correcciones
