# Quality Guidance Architecture

## Alcance actual

Este documento describe el **Incremento 1 — Quality Guidance Observacional**. El sistema observa, normaliza, diagnostica y recomienda. No cambia sampling, learning rate, losses, optimizadores, scheduler, recovery ni selección de checkpoints.

La garantía central del incremento es:

```text
recommended_action puede ser REINFORCE/ADJUST/ROLLBACK
action siempre es CONTINUE
training_modified siempre es false
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
7. exporta/restaura estado observacional.

### TrainingRecovery

Responde: **«¿qué hacer si el entrenamiento se vuelve inseguro?»**

`pixel_ai_engine/training_recovery.py` sigue siendo la autoridad para NaN, spikes, colapso extremo, snapshot sano y recuperación. Quality Guidance no reemplaza, llama ni modifica sus decisiones en este incremento.

## Flujo implementado

```text
train_supervised
  ├─ entrena una época con el comportamiento existente
  ├─ genera preview
  ├─ PixelArtEnhancer.analyze_quality
  ├─ QualityVector
  ├─ QualityTrendAnalyzer
  ├─ diagnose_quality_bottleneck
  ├─ QualityGuidanceController (observational)
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

El adaptador acepta, entre otros, los aliases `score_gestos_ojos`/`gestos_ojos`, `score_ropa_delantal`/`ropa_delantal`, `alineacion_molde`/`silueta_iou_real` y métricas de `Phase3CriticalReviewer` cuando estén presentes. Incorporar formalmente las métricas críticas al pipeline queda reservado al Incremento 6.

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

En este incremento solo se ejecuta `CONTINUE`. El campo `recommended_action` expresa qué evaluaría un incremento futuro. Esto permite validar el diagnóstico antes de otorgarle autoridad sobre el entrenamiento.

## Señales diferenciables y no diferenciables

Las métricas de QualityGate, Enhancer y Phase3 que dependen de PIL, NumPy, segmentación, conteos o reglas discretas son **no diferenciables**. Controlan exclusivamente observación y recomendación.

No se realiza ninguna operación equivalente a:

```python
total_g += quality_gate_score
```

Las losses L1, alpha, edge y adversarial existentes permanecen intactas. Los multiplicadores adaptativos pertenecen al Incremento 3 y requerirán límites, feature flag y pruebas A/B.

## Feature flags

En `pixel_ai_engine/quality_guidance.py`:

```python
ENABLE_QUALITY_GUIDANCE = True
ENABLE_SMART_SAMPLING = False
ENABLE_ADAPTIVE_LOSS = False
ENABLE_QUALITY_CHECKPOINT = False
```

`ENABLE_QUALITY_GUIDANCE` puede apagarse con `PIXEL_AI_ENABLE_QUALITY_GUIDANCE=0`. Al desactivarlo se sigue publicando un contrato mínimo con `enabled=false`, `action=CONTINUE` y `training_modified=false`.

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
- acciones reales de rollback, LR o refuerzo desde Quality Guidance;
- score global como única fuente de verdad.

## Persistencia y compatibilidad

`guidance_state` contiene versión, configuración, última decisión, historiales y placeholders neutrales para sampling/loss. Los placeholders permanecen en 1.0/vacíos y no afectan entrenamiento.

Al reanudar:

1. se prefiere el estado de `training_status.json`;
2. si falta, se usa el estado del checkpoint moderno;
3. si ambos faltan, se inicia vacío;
4. en respawn explícito se puede preferir el estado del snapshot seleccionado.

Al abrir una nueva era, el historial anterior se archiva con sus mejores scores observados (`global`, anatomía, rostro, paleta, silueta, microdetalle y alfa).

## Observabilidad

`monitor.html` y `sprite_studio.html` leen primero `guidance` y aceptan `quality_guidance` como alias. Muestran modo, problema, severidad y acción recomendada. La consola declara explícitamente `OBSERVATIONAL_ONLY` y que el entrenamiento no fue modificado.

## Próximo incremento permitido

Solo después de aprobar formalmente el Incremento 1 podrá comenzar el **Incremento 2 — Smart Reinforcement**, primero con métricas por frame y después sampling adaptativo acotado. Adaptive Loss seguirá desactivado durante ese incremento.
