# Quality Guidance Architecture

## Alcance actual

Este documento describe la arquitectura final tras los **Incrementos 1–6**. El sistema observa y diagnostica siempre; bajo flags puede ajustar sampling o una loss acotada, y puede solicitar Recovery ante colapso visual sostenido.

Las garantías centrales son:

```text
una sola intervención atribuible por vez
sampling entre 1.0 y 2.0
multiplicadores de loss entre 0.75 y 1.25
Recovery conserva autoridad sobre NaN, snapshots y rollback
ground truth nunca pasa por elevate_frame
```

## Responsabilidades

### QualityGate

Responde: **«¿qué tan bueno está el modelo?»**

`pixel_ai_engine/quality_gate.py` genera auditorías visuales con PIL/NumPy a partir de un checkpoint: score global, paleta, molde/silueta, microtextura, alfa y categorías anatómicas. Sus señales son no diferenciables.

### QualityGuidance

Responde: **«¿qué parece fallar y qué convendría hacer después?»**

`pixel_ai_engine/quality_guidance.py`:

1. normaliza señales históricas y modernas;
2. construye un `QualityVector` JSON-safe;
3. conserva `None` para métricas no observadas;
4. analiza varias auditorías mediante `QualityTrendAnalyzer`;
5. localiza el cuello de botella con `diagnose_quality_bottleneck`;
6. separa recomendación de acción aplicada;
7. exporta/restaura estado, métricas por frame y política de intervención.

### TrainingRecovery

Responde: **«¿qué hacer si el entrenamiento se vuelve inseguro?»**

`pixel_ai_engine/training_recovery.py` sigue siendo la autoridad para NaN, spikes, colapso extremo, snapshot sano y recuperación. Guidance sólo puede solicitar una ruta tras confirmar tendencia `COLLAPSE`; Recovery selecciona y materializa el snapshot.

## Flujo implementado

```text
train_supervised
  ├─ entrena una época con el comportamiento existente
  ├─ genera preview
  ├─ PixelArtEnhancer.analyze_quality
  ├─ QualityVector
  ├─ QualityTrendAnalyzer
  ├─ diagnose_quality_bottleneck
  ├─ QualityGuidanceController
  ├─ GuidanceInterventionPolicy (cooldown/máximo)
  ├─ sampling o adaptive loss (excluyentes)
  ├─ TrainingRecovery si corresponde
  ├─ best_quality_generator.pt si mejora el criterio compuesto
  └─ training_status.json
       ├─ quality (métricas originales)
       ├─ guidance (contrato actual)
       ├─ quality_guidance (alias retrocompatible)
       ├─ guidance_state
       └─ history[].quality / history[].guidance
```

El checkpoint completo y `best_generator.pt` reciben el campo aditivo `guidance_state`. La ausencia de ese campo en un checkpoint antiguo equivale a estado inicial vacío y no produce error.

## QualityVector

Categorías actuales:

- `global`
- `anatomy`
- `silhouette`
- `pose`
- `face`
- `hair`
- `clothing`
- `arms_hands`
- `feet`
- `props`
- `palette`
- `alpha`
- `micro_detail`
- `outline`
- `training_stability`

Las puntuaciones se limitan a 0–100. NaN, infinito, booleanos y valores no numéricos se consideran no disponibles. El valor `0.0` se conserva como medición real; nunca se confunde con ausencia.

El adaptador acepta, entre otros, los aliases `score_gestos_ojos`/`gestos_ojos`, `score_ropa_delantal`/`ropa_delantal`, `alineacion_molde`/`silueta_iou_real` y las métricas puras de `Phase3CriticalReviewer`, ya integradas al preview sin `elevate_frame`.

## Tendencias

`QualityTrendAnalyzer` necesita como mínimo tres observaciones y clasifica cada categoría como:

- `IMPROVING`
- `PLATEAU`
- `REGRESSION`
- `COLLAPSE`
- `OSCILLATION`
- `STABLE`
- `INSUFFICIENT_DATA`

Una regresión de rostro, por ejemplo 71→68→63, se reporta aunque el score global siga siendo aceptable. Una regresión escala la severidad de la recomendación, pero no activa una intervención.

## Diagnóstico y umbrales

El diagnóstico compara cada señal **disponible** contra targets configurables y revisa mínimos críticos independientes del promedio. Devuelve:

```json
{
  "primary_problem": "face",
  "secondary_problems": ["micro_detail"],
  "severity": "medium",
  "confidence": 0.82
}
```

Los valores por defecto son conservadores y configurables en `QualityGuidanceConfig`; no se consideran una calibración definitiva. Tras acumular el número configurado de observaciones, el controlador puede derivar targets relativos desde la línea base, limitados por los mínimos críticos. Una futura calibración deberá usar datos A/B reales.

## Acciones

El vocabulario estable admite:

- `CONTINUE`
- `REINFORCE`
- `ADJUST_WEIGHTS`
- `ADJUST_SAMPLING`
- `REDUCE_LR`
- `FREEZE`
- `ROLLBACK`
- `STOP`

`recommended_action` expresa el diagnóstico y `authorized_action` la decisión de la política. Cooldown, presupuesto y disponibilidad de snapshot pueden convertir una recomendación en `CONTINUE`. Sampling, loss y rollback nunca se aplican simultáneamente.

## Señales diferenciables y no diferenciables

Las métricas de QualityGate, Enhancer y Phase3 que dependen de PIL, NumPy, segmentación, conteos o reglas discretas son **no diferenciables**. Controlan exclusivamente observación y recomendación.

No se realiza ninguna operación equivalente a:

```python
total_g += quality_gate_score
```

Las métricas discretas no entran al grafo. Los multiplicadores adaptativos sólo escalan una loss existente y la loss de silueta tensorial permanece desactivada por defecto.

## Feature flags

En `pixel_ai_engine/quality_guidance.py`:

```python
ENABLE_QUALITY_GUIDANCE = True
ENABLE_SMART_SAMPLING = True
ENABLE_ADAPTIVE_LOSS = True
ENABLE_QUALITY_CHECKPOINT = True
ENABLE_SILHOUETTE_LOSS = False
```

Cada capacidad tiene variable `PIXEL_AI_ENABLE_*`. Al desactivar Guidance o una capacidad, sampling vuelve a uniforme y las losses a sus pesos base exactos.

## Recuperado del sistema histórico

- catálogo de métricas macro y anatómicas de QualityGate;
- idea de auditar después de entrenar y volver a evaluar;
- conceptos de target y rondas limitadas como metadatos compatibles;
- auditoría por frame como candidato futuro para hard example mining;
- separación entre métricas críticas y elevación visual de Phase3.

## Descartado o pospuesto

- bucles automáticos de +100 épocas;
- umbral universal de 99,5%;
- entrenador antiguo de dos fases como pipeline principal;
- transformaciones Phase3 sobre ground truth;
- cambios simultáneos de sampling y loss;
- cambios directos de LR desde Guidance (Recovery conserva esa responsabilidad);
- score global como única fuente de verdad.

## Persistencia y compatibilidad

`guidance_state` contiene versión, configuración, historiales, calidad por frame, sampling, multiplicadores y estado de cooldown/intervenciones. Checkpoints antiguos sin esos campos continúan iniciando con valores neutrales.

Al reanudar:

1. se prefiere el estado de `training_status.json`;
2. si falta, se usa el estado del checkpoint moderno;
3. si ambos faltan, se inicia vacío;
4. en respawn explícito se puede preferir el estado del snapshot seleccionado.

Al abrir una nueva era, el historial anterior se archiva con los mejores scores de todas las dimensiones disponibles. `best_quality_generator.pt` usa un compuesto ponderado con pisos críticos y nunca reemplaza `best_generator.pt`.

## Observabilidad

`monitor.html` y `sprite_studio.html` leen primero `guidance` y aceptan `quality_guidance` como alias. El monitor muestra anatomía, silueta, pose, rostro, borde, microdetalle, acción aplicada y mejor quality checkpoint.

## Estado de validación

Los seis incrementos están aprobados. La suite propia final contiene 81 pruebas, incluido START→RESUME→PAUSE→RESUME en CPU y compatibilidad con checkpoints antiguos. Persisten cinco warnings de deprecación AMP sin impacto funcional.
