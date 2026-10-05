# Arquitectura de Control de Calidad por Frame

Estado: Incrementos 1–5 implementados y validados el 2026-10-04.

## Invariantes

- El frame generado y sus candidatos son evidencia de calidad; nunca son ground truth.
- El único target entrenable es el archivo real resuelto desde el manifest para el mismo personaje, variante y frame.
- Reevaluar no modifica imágenes, datasets, checkpoints, sampling ni pesos.
- Regenerar sólo produce candidatos aislados. Aplicar es una acción explícita y recuperable.
- Enviar a refuerzo no inicia entrenamiento; prepara el muestreo de una ejecución futura.
- La UI transmite identificadores y acciones, no rutas arbitrarias.

## Flujo

```text
run_id + frame_idx
        |
        v
FrameQualityReviewManager
  resuelve generado + metadata + manifest + target + frontal
        |
        v
QualityGate.evaluate_single_frame
        |
        v
frame_review_queue.jsonl
   |          |                 |
   |          |                 +--> aprobar / rechazar / reevaluar
   |          |
   |          +--> HardExampleQueue --> hard_examples.jsonl
   |                   |
   |                   +--> pesos 1.0..2.0 del próximo entrenamiento
   |
   +--> FrameRegenerationManager
           genera y audita candidatos aislados
           rankea mejoras sin regresión crítica
           aplica explícitamente con backup

InteractiveQualityControlService coordina acciones individuales, métricas y
trabajos batch persistentes en qc_batch_jobs.json.
```

## Componentes

### `FrameQualityReviewManager`

Mantiene el registro materializado de cada review, valida estados, resuelve rutas bajo raíces permitidas, conserva historial y expone evaluación, reevaluación, aprobación, rechazo, regeneración y refuerzo. Las escrituras usan `RLock`, archivo temporal, `fsync` y `os.replace`.

### `QualityGate`

`evaluate_single_frame` combina análisis visual, métricas anatómicas, revisión crítica y el vector de calidad canónico. Si falta el target devuelve `audit_available=false`, score nulo y estado de error. Los fallos de alfa o contacto con bordes son explícitos y bloqueantes.

### `FrameRegenerationManager`

`TorchFrameCandidateGenerator` deriva frontal, formato, molde y checkpoint desde metadata, carga el modelo una vez y genera entre 1 y 8 alternativas. Cada candidato se guarda bajo:

```text
output/<run>/regeneration/frame_NNN/<generation_id>/candidate_NN.png
```

El ranking exige:

- auditoría disponible y score global válido;
- mejora positiva del problema dominante; diagnósticos no comparables usan `global`;
- pisos mínimos de anatomía, silueta, rostro, props, alfa y paleta;
- ninguna regresión mayor a 5 puntos;
- ningún fallo crítico de alfa o bordes.

Aplicar vuelve a validar que el candidato pertenezca al directorio aislado, copia el original como backup, reemplaza el frame de forma atómica y parchea únicamente su celda en las hojas existentes. Descartar cambia estado pero conserva evidencia.

### `HardExampleQueue`

Antes de encolar, vuelve a resolver el manifest y exige coincidencia exacta de target, personaje, variante y frame. Deduplica por esa identidad y asigna prioridad entre 1.0 y 2.0. El path generado queda como evidencia; `target_verified=true` identifica el target real.

`record_outcome` reduce la prioridad cuando hay mejora y resuelve el caso cuando supera el umbral o es aprobado.

### `InteractiveQualityControlService`

Coordina las acciones individuales y trabajos:

- `REEVALUATE_ALL`;
- `REGENERATE_DEFECTIVE`;
- `REGENERATE_SELECTED`, usado internamente por la acción individual asíncrona;
- `REINFORCE_SELECTED`.

Los jobs conservan progreso, completados, fallos y errores. La cancelación es cooperativa; jobs `RUNNING` o `CANCELLING` encontrados durante un reinicio pasan a `INTERRUPTED`.

### Integración con entrenamiento

`load_manual_sampling_weights` lee sólo registros activos con target verificado. `HardExampleMiningPolicy` combina el peso manual y el automático mediante el máximo. `WeightedRandomSampler` aplica el plan desde el siguiente entrenamiento, con peso máximo 2.0 y sin sustituir samples ni targets del dataset.

## Persistencia

| Archivo | Propósito | Mutado por |
|---|---|---|
| `frame_review_queue.jsonl` | estado e historial por frame | review manager |
| `hard_examples.jsonl` | prioridad manual validada | hard-example queue |
| `qc_batch_jobs.json` | progreso y recuperación batch | servicio QC |

Los tres son estado de ejecución local y están excluidos de Git.

## API

Consultas:

- `GET /api/qc/review-queue`
- `GET /api/qc/frame/<review_id>`
- `GET /api/qc/hard-examples`
- `GET /api/qc/batches`
- `GET /api/qc/batch/<job_id>`

Acciones `POST`:

- `/api/qc/frame/evaluate`
- `/api/qc/frame/<id>/reevaluate`
- `/api/qc/frame/<id>/approve`
- `/api/qc/frame/<id>/reject`
- `/api/qc/frame/<id>/regenerate`
- `/api/qc/frame/<id>/apply-best`
- `/api/qc/frame/<id>/discard-regeneration`
- `/api/qc/frame/<id>/reinforce`
- `/api/qc/batch/start`
- `/api/qc/batch/<job_id>/cancel`

Regeneración y refuerzo batch requieren confirmación explícita. Las capacidades activas se publican con la cola para que la UI deshabilite controles cuando un flag esté apagado.

## Métricas

El resumen expone frames rechazados, regenerados, regeneraciones medidas/mejoradas, porcentaje de mejora, enviados a refuerzo, fallos recurrentes y tiempo medio hasta aprobación. El monitor muestra hard examples activos y porcentaje de regeneraciones mejores.

## Seguridad del servidor

Sprite Studio escucha en `127.0.0.1` por defecto. Para acceso deliberado desde la red se debe definir `SPRITE_STUDIO_HOST`. `end_headers` usa un path vacío seguro porque una petición HTTP malformada puede provocar una respuesta de error antes de que `SimpleHTTPRequestHandler` asigne `self.path`.

## Feature flags

```python
ENABLE_FRAME_REVIEW = True
ENABLE_FRAME_REGENERATION = True
ENABLE_MANUAL_REINFORCEMENT = True
ENABLE_BATCH_QC_ACTIONS = True
```

Cada flag se puede desactivar con su variable `PIXEL_AI_ENABLE_*`. El backend aplica la restricción y la UI refleja las capacidades recibidas.

## Verificación

- 105 pruebas pasan en la suite completa.
- Las pruebas cubren safe paths, target ausente, persistencia, resume, ranking, regresión crítica, backups, descarte, deduplicación, decay, sampler, batch, cancelación, API y requests malformados.
- QA visual de Sprite Studio y Monitor completada sin errores ni warnings de consola.
