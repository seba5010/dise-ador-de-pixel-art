# Arquitectura de Control de Calidad por Frame

Estado: Incremento 1 implementado y validado el 2026-10-04. Regeneración, refuerzo manual y acciones batch permanecen desactivados.

## Propósito

El sistema separa tres objetos que nunca deben confundirse:

- **Generated frame**: salida del modelo. Es evidencia para revisión; nunca es ground truth.
- **Target frame**: sprite correcto resuelto desde el dataset para el mismo personaje, variante y `frame_idx`.
- **Reference front**: identidad frontal usada para medir paleta cuando existe. No sustituye al target de pose.

La cola de revisión registra la evaluación automática y la decisión humana sin escribir en los datasets ni modificar pesos del modelo.

## Flujo activo del Incremento 1

```text
run_id + frame_idx
        |
        v
resolver output/<run>/enhanced_frames/frame_NNN.png
        |
        v
leer metadata de la ejecución
        |
        v
resolver manifest de personaje + variante + slot (frame_idx + 1)
        |
        v
QualityGate.evaluate_single_frame(generado, target, frontal)
        |
        v
persistir/actualizar frame_review_queue.jsonl
        |
        +--> Reevaluar (mismo archivo, nuevas métricas)
        +--> Aprobar (requiere target y auditoría válida)
        +--> Rechazar (no inicia entrenamiento)
```

## Componentes

### `FrameQualityReviewManager`

Responsable de:

- validar el vocabulario de estados;
- resolver el frame generado únicamente desde `run_id` y `frame_idx`;
- derivar personaje y variante desde metadata/manifests, sin listas de nombres;
- localizar target y frontal de identidad;
- ejecutar la auditoría por frame;
- materializar la cola JSONL mediante reemplazo atómico;
- conservar historial, actores, timestamps y contadores;
- consultar registros y resumen para UI/monitor.

El bloqueo `RLock` serializa lecturas/modificaciones concurrentes dentro del proceso multihilo de Sprite Studio. Cada escritura usa un archivo temporal, `fsync` y `os.replace`, por lo que nunca queda un JSONL parcialmente reemplazado.

### `QualityGate`

`evaluate_single_frame` reutiliza:

- `PixelArtEnhancer.analyze_quality`;
- `compute_anatomical_metrics`;
- `Phase3CriticalReviewer.audit_frame`;
- `build_quality_vector`;
- `diagnose_quality_bottleneck`.

Devuelve un vector común con `global`, anatomía, silueta, pose, rostro, pelo, ropa, brazos/manos, pies, objetos, paleta, alfa, microdetalle y contorno. Además registra issues, severidad, diagnóstico, métricas crudas e `identity_confidence` sólo cuando existe una señal de identidad utilizable.

Los controles estructurales añaden issues explícitos:

- `ALPHA_FAIL` si existe alfa entre 1 y 254;
- `BORDER_TOUCH_TOP`;
- `BORDER_TOUCH_BOTTOM`;
- `BORDER_TOUCH_LEFT`;
- `BORDER_TOUCH_RIGHT`.

Si falta target, la auditoría devuelve `audit_available=false`, `score_total=null` y `aprobado=false`. No se calcula ni inventa precisión anatómica o de silueta.

`evaluate_generator_frame` acepta un generator ya cargado, `front_tensor`, `pose_tensor` y `target_tensor`. No reconstruye el modelo ni recarga checkpoint.

## Registro persistido

Cada línea de `frame_review_queue.jsonl` representa el estado materializado de un review. El historial completo permanece dentro del registro.

Campos centrales:

```json
{
  "review_id": "uuid",
  "run_id": "alex_8x12_20261004_191852",
  "character_id": "alex",
  "variant": "rnormal",
  "frame_idx": 0,
  "row": 0,
  "column": 0,
  "pose": {},
  "epoch": null,
  "status": "NEEDS_REVIEW",
  "generated_frame_path": "output/.../frame_000.png",
  "target_frame_path": "dataset_frames_individuales/.../frame_001_r01_c01.png",
  "reference_front_path": "dataset_frames_individuales/.../00_frontal_identidad.png",
  "quality": {},
  "score_total": 0,
  "issues": [],
  "severity": "high",
  "identity_confidence": null,
  "user_action": null,
  "times_failed": 0,
  "times_regenerated": 0,
  "times_reevaluated": 0,
  "sent_to_reinforcement": false,
  "created_at": "...",
  "updated_at": "...",
  "history": []
}
```

Estados permitidos: `OK`, `NEEDS_REVIEW`, `REEVALUATED`, `REGENERATED`, `SENT_TO_REINFORCEMENT`, `APPROVED`, `REJECTED`, `SUPERSEDED` y `ERROR`. Los estados futuros ya están reservados, pero sus acciones permanecen desactivadas.

## Resolución de identidad y target

1. Se lee `output/<run_id>/metadata.json`.
2. Personaje y variante salen de metadata; si falta la variante, se deriva del nombre del frontal mediante alias genéricos conocidos del esquema de dataset.
3. Se busca un `manifest.json` cuyo personaje y variante normalizados coincidan.
4. El target es la entrada cuyo `slot == frame_idx + 1`.
5. Como compatibilidad, se consulta `dataset_supervisado/frames_png` con la convención normalizada.
6. Si no existe un archivo real, el target queda ausente y se bloquea la aprobación.

## API

Consultas:

- `GET /api/qc/review-queue?status=<estado>&run_id=<run>`
- `GET /api/qc/frame/<review_id>`

Acciones mutables (sólo `POST`):

- `POST /api/qc/frame/evaluate` con `run_id` y `frame_idx`;
- `POST /api/qc/frame/<review_id>/reevaluate`;
- `POST /api/qc/frame/<review_id>/approve`;
- `POST /api/qc/frame/<review_id>/reject`.

Las rutas históricas `/api/frame_review/*` se mantienen temporalmente por compatibilidad, pero la UI nueva consume `/api/qc/*` y no confía en métricas enviadas por el cliente al reevaluar.

## UI y monitor

La pestaña Control de Calidad permite seleccionar una celda y crear/actualizar su review. Cada tarjeta muestra:

- estado con texto y color;
- generado y target;
- score y métricas por categoría;
- issues y severidad;
- acciones Reevaluar, Aprobar y Rechazar;
- diagnóstico e historial expandibles.

El monitor muestra pendientes, aprobados y rechazados. Regenerados y enviados a refuerzo aparecen explícitamente en cero y desactivados para no prometer funciones de incrementos posteriores.

## Seguridad e invariantes

- Los identificadores rechazan separadores, `..` y caracteres fuera del vocabulario permitido.
- El generado sólo se resuelve bajo `OUTPUT_DIR`.
- Targets y referencias sólo se resuelven bajo raíces de dataset configuradas.
- La UI nueva envía IDs, no paths.
- La cola puede escribirse; datasets y frames evaluados se abren en modo lectura.
- Aprobar requiere target existente y auditoría disponible.
- `ALPHA_FAIL` bloquea aprobación.
- Reevaluar no modifica imágenes, dataset, checkpoint, sampling ni pesos.
- Rechazar no encola entrenamiento.

## Feature flags

```python
ENABLE_FRAME_REVIEW = True
ENABLE_FRAME_REGENERATION = False
ENABLE_MANUAL_REINFORCEMENT = False
ENABLE_BATCH_QC_ACTIONS = False
```

## Evolución prevista (no implementada)

- Incremento 2: candidatos de regeneración aislados, ranking seguro, backup y aplicación explícita.
- Incremento 3: `hard_examples.jsonl`, validación fuerte del target y prioridad acotada/decay.
- Incremento 4: reevaluación y operaciones batch con progreso/cancelación.
- Incremento 5: integración controlada del feedback con sampling y ciclo de aprendizaje.

`hard_examples.jsonl` será una persistencia distinta. Incluso entonces, el generado defectuoso seguirá siendo evidencia; el entrenamiento utilizará exclusivamente el target correcto del dataset.

## Verificación del Incremento 1

- Suite automatizada: 91 pruebas PASS, 5 warnings deprecados de AMP ya existentes.
- Pruebas dirigidas: estados, persistencia, resume, target lookup, target ausente, safe paths, reevaluación sin mutar imágenes, aprobación/rechazo y API HTTP.
- Prueba real: el frame 0 de `alex_8x12_20261004_191852` resolvió `alex/rnormal` y su target de manifest.
- QA visual: cola y tarjeta renderizadas en Sprite Studio; resumen renderizado en `monitor.html`; consola del navegador sin errores.
