# INFORME TÉCNICO DE RESOLUCIÓN DE AUDITORÍA
**Proyecto**: Villa del Chef — Diseñador de Pixel Art / Sprite Studio  
**Fecha**: 3 de octubre de 2026  
**Estado**: 100% Reparado y Validado  
**Versión Base**: `c6afe2a` → `ce79d2c` → Versión Actual  

---

## 1. Introducción y Alcance

Este informe documenta exhaustivamente las causas raíz, soluciones técnicas y motivos de las 5 observaciones pendientes identificadas en la auditoría del repositorio de fecha 3 de octubre de 2026.

Las áreas intervenidas abarcan el ciclo de vida de los checkpoints de entrenamiento, la interfaz reactiva de Respawn, el motor del reproductor de animaciones, el sistema de auditoría quirúrgica de control de calidad (QC) y el pipeline de aumentación, remapeo de paleta y binarización alfa.

---

## 2. Detalle de Problemas, Causas Raíz y Soluciones

### Pendiente 1: Reanudación de Checkpoints y Preservación de `best_loss`

#### Problema Detectado
Al finalizar una época normal, `train_supervised.py` guardaba `latest_checkpoint.pt` omitiendo el campo `best_loss`. Al reanudar el entrenamiento posteriormente, el cargador asignaba por defecto `999.0`. Si el modelo había alcanzado previamente una pérdida óptima de `0.1` y la siguiente época de la sesión reanudada alcanzaba `0.5`, el sistema consideraba erróneamente `0.5 < 999.0` como una mejora histórica, sobrescribiendo `best_generator.pt` con unos pesos peores. Además, los snapshots omitían estados de los optimizadores (`opt_g`, `opt_d`), escaladores de gradiente AMP (`GradScaler`) y estados del generador de números aleatorios (RNG).

#### Causa Raíz
Discrepancia en la serialización entre `latest_checkpoint.pt` y `best_generator.pt`, sumada a la ausencia de lógica de recuperación retrocompatible de métricas históricas al cargar checkpoints antiguos.

#### Solución Implementada
1. **Función Unificada `save_checkpoint`**:
   Se centralizó el guardado en `train_supervised.py` para checkpoints periódicos, snapshots cada 10 épocas, banderas de pausa/detención y el mejor modelo histórico:
   ```python
   def save_checkpoint(file_path: Path, epoch: int, loss: float, best_loss: float,
                       generator: nn.Module, discriminator: nn.Module,
                       opt_g: Any = None, opt_d: Any = None, scaler_g: Any = None, scaler_d: Any = None):
       file_path.parent.mkdir(parents=True, exist_ok=True)
       f_loss = float(loss) if (loss is not None and math.isfinite(float(loss))) else 0.0
       f_best = float(best_loss) if (best_loss is not None and math.isfinite(float(best_loss))) else 999.0
       state = {
           "epoch": int(epoch),
           "generator": generator.state_dict(),
           "discriminator": discriminator.state_dict(),
           "loss": f_loss,
           "best_loss": f_best,
           "opt_g": opt_g.state_dict() if opt_g is not None else None,
           "opt_d": opt_d.state_dict() if opt_d is not None else None,
           "scaler_g": scaler_g.state_dict() if (scaler_g is not None and hasattr(scaler_g, "state_dict")) else None,
           "scaler_d": scaler_d.state_dict() if (scaler_d is not None and hasattr(scaler_d, "state_dict")) else None,
           "rng_state": torch.get_rng_state(),
           "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
       }
       torch.save(state, file_path)
   ```
2. **Recuperación Retrocompatible**:
   Si se reanuda desde un checkpoint sin `best_loss` o con valor `999.0`, el sistema consulta automáticamente `checkpoints/best_generator.pt` para restaurar la pérdida mínima histórica real.
3. **Restauración Completa de Estado**:
   Se recargan `opt_g`, `opt_d`, `scaler_g`, `scaler_d` y las semillas aleatorias si existen en el diccionario cargado.
4. **Protección Numérica contra NaN**:
   En `update_status`, se valida `math.isfinite` para todas las pérdidas y métricas, evitando escribir representaciones no estándar en `training_status.json`.

---

### Pendiente 2: Selector y Contador de Snapshots de Respawn

#### Problema Detectado
El backend contaba con el endpoint `/api/train/snapshots` y la telemetría incluía la propiedad `snapshots`. No obstante, el selector HTML `#respawnEpochSelect` se quedaba permanentemente con la opción estática *"Esperando Época 10..."* y el badge en *"0 Puntos Guardados"*, impidiendo seleccionar un snapshot para rebobinar.

#### Causa Raíz
El JavaScript cliente de `sprite_studio.html` no consumía la lista de snapshots en `updateMonitorTab` ni existía una función encargada de poblar dinámicamente el elemento `<select>`.

#### Solución Implementada
1. Se programó la función `updateRespawnSnapshots(snapshots)`:
   - Actualiza `#respawnCountBadge` con el formato `"N Punto(s) Guardado(s)"`.
   - Si no hay snapshots disponibles (`snapshots.length === 0`), muestra *"Esperando Época 10..."* y desactiva el botón `#btnRespawnTrigger`.
   - Si existen snapshots, genera un `<option>` por cada uno indicando número de época y nombre de archivo.
   - Retiene la selección activa si sigue siendo válida o preselecciona el snapshot más reciente.
   - Habilita `#btnRespawnTrigger`.
2. Se programó `fetchSnapshots()`, consumiendo `/api/train/snapshots` en la inicialización (`initApp`) y al cambiar a la pestaña `monitor`.
3. Se enlazó la actualización automática en `updateMonitorTab(info)` cuando llega la telemetría periódica.

---

### Pendiente 3: Reproductor — Formato Dinámico (64 vs 96 frames) y Redibujo en Pausa

#### Problema Detectado
1. Al seleccionar una hoja generada con formato `16x4` (64 frames), el reproductor conservaba la configuración de rejilla `8x12` (96 frames), provocando dimensiones erróneas de celda (85×170 px en lugar de 170×128 px).
2. El selector de animación `#animCycleSelect` y la función `drawAnimFrame` utilizaban un diccionario estático de 96 frames. Si el usuario seleccionaba "celebrar" (índice 92), en una hoja 16x4 el cálculo `y = 23 * 128 = 2944px` sobrepasaba la altura real de 2048px, generando un recorte inválido fuera de la imagen.
3. Al cambiar de hoja en el reproductor estando pausado, `drawAnimFrame` intentaba ejecutarse de forma sincrónica antes de que el navegador completara la carga asíncrona de `activeSheetImg.src`. Como la animación estaba pausada, el canvas permanecía en blanco o mostraba la imagen anterior sin redibujarse.

#### Causa Raíz
Falta de enlace entre los metadatos de la hoja (`metadata.format`) y el estado global `appState.format`, así como la ausencia de un manejador de evento `onload` en la imagen del spritesheet.

#### Solución Implementada
1. **Detección Dinámica de Formato**:
   En `loadRunSheet(runDir)`, se extrae `metadata.format` (por defecto `"8x12"`, o `"16x4"`) y se invoca `setAppFormat(detectedFmt)`.
2. **Diccionarios de Animación Especializados**:
   Se crearon `ANIMATION_CLIPS_8X12` (15 clips, frames 0..95) y `ANIMATION_CLIPS_16X4` (9 clips canónicos, frames 0..63). La función `updateAnimCyclesDropdown(format)` reconstruye las opciones del desplegable según el formato activo y reajusta `animCycle` para evitar índices incompatibles.
3. **Acotamiento Seguro en el Recorte**:
   En `drawAnimFrame(ctx)`, se aplica:
   ```javascript
   const maxFrames = cols * rows;
   const safeIdx = Math.max(0, Math.min(fIdx, maxFrames - 1));
   ```
   Garantizando que ningún fotograma sobrepase la altura ni anchura del lienzo.
4. **Redibujado Inmediato mediante `onload`**:
   `loadRunSheet` asigna callbacks `sheetImg.onload` y `sheetImg.onerror`. Cuando la imagen termina de cargarse en memoria, se ejecuta de inmediato `drawAnimFrame(ctx)` en el canvas y se actualiza la vista previa del inspector, reflejando el nuevo spritesheet al instante sin requerir pulsar reproducir.

---

### Pendiente 4: Control de Calidad — Rechazo Estricto de Alfa Defectuoso

#### Problema Detectado
En `sprite_studio.py`, la función `run_quality_audit` calculaba métricas de semitransparencias borrosas (`blurry_alpha_cells`), pero la condición `is_ready` omitía este parámetro:
```python
# Código anterior defectuoso
is_ready = (completeness_score == 100.0 and len(border_touch_cells) == 0 and len(empty_cells) == 0)
```
Como consecuencia, una hoja con 96 celdas con bordes borrosos y pureza alfa de 0% recibía el veredicto `"SPRITESHEET CERTIFICADO 100% PARA UNITY"`.

#### Causa Raíz
Omisión de la comprobación de `blurry_alpha_cells` en la expresión booleana final de certificación.

#### Solución Implementada
Se reformuló la condición en `run_quality_audit`:
```python
is_ready = (
    completeness_score == 100.0 and
    len(border_touch_cells) == 0 and
    len(empty_cells) == 0 and
    len(blurry_alpha_cells) == 0 and
    alpha_purity_score >= 95.0
)
```
Adicionalmente, se incorporó una alerta explícita en `audit_logs`:
```python
if len(blurry_alpha_cells) > 0:
    audit_logs.append(f"[{time.strftime('%H:%M:%S')}] AVISO: {len(blurry_alpha_cells)} celdas con canal alfa semitransparente/borroso (no apto para Unity).")
```
Y si la hoja incumple el estándar, el veredicto indica claramente `"VEREDICTO: BORRADOR / REVISION REQUERIDA (... celdas con alfa defectuoso)"`, impidiendo certificaciones erróneas.

---

### Pendiente 5: Colores y Acabado — Albumentations, Remapeo y Transparencia de Inferencia

#### Problema Detectado
1. En `train_supervised.py`, `A.ColorJitter` solo definía brillo y contraste. En versiones de Albumentations 2.x, el tono y la saturación aplican variaciones por defecto si no se anulan explícitamente, alterando la paleta del frontal de referencia frente al ground truth.
2. `extract_character_palette` en `palette_remap.py` seleccionaba únicamente los 32 colores más frecuentes mediante conteo simple. Esto eliminaba tonos minoritarios críticos (por ejemplo, 1 píxel de un ojo azul entre miles de píxeles grises). Tampoco incluía los colores estándar de accesorios (`PROPS_PALETTE`).
3. `remap_image_to_palette` forzaba todos los píxeles a la paleta sin umbral de tolerancia euclídea y no binarizaba el canal alfa (dejando valores semitransparentes como 128).
4. `generate_preview` usaba el dataset de entrenamiento con aumentaciones activas para extraer muestras y solo mostraba la imagen postprocesada, ocultando posibles fallos morfológicos de la red neuronal cruda.
5. El acabado morfológico de `palette_remap.py` solo se utilizaba en `generate_preview`, mientras que Sprite Studio usaba `PixelArtEnhancer`, generando discrepancias visuales.

#### Causa Raíz
Falta de parámetros de bloqueo de tono/saturación en `ColorJitter`, uso de ordenamiento por frecuencia simple en lugar de agrupamiento perceptual (K-Means), y falta de binarización alfa estricta compartida.

#### Solución Implementada
1. **Bloqueo Cromático en Albumentations**:
   En `train_supervised.py`:
   ```python
   AUG_PIPELINE = A.Compose([
       A.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.0, hue=0.0, p=0.4),
   ])
   ```
2. **Extracción Perceptual con K-Means y Paleta de Accesorios**:
   En `palette_remap.py`:
   - `extract_character_palette` utiliza K-Means clustering sobre colores únicos para asegurar que tonos minoritarios de alta saturación queden preservados en los centroides.
   - Se incorpora `PROPS_PALETTE` con tonos oficiales de maderas, bowls metálicos, cartones y verduras.
3. **Tolerancia Euclídea y Canal Alfa Binario**:
   En `remap_image_to_palette`, si un píxel dista más de `tolerance=35.0` (un color de accesorio o detalle único), se conserva su color intacto. Se aplica binarización estricta (`alpha >= 40 ? 255 : 0`).
4. **Limpieza Anti-Hollín y Binarización en Enhancer**:
   Tanto `clean_orphan_pixels` como `PixelArtEnhancer.enhance_frame` garantizan corte alfa binario de 0 o 255 mediante `binarize_alpha(threshold=40)`.
5. **Previsualización Transparente de 5 Paneles**:
   En `generate_preview`, se evalúan las muestras puras sin aumentaciones aleatorias (`dataset.samples[s_idx]`) y la tira comparativa muestra:
   - Panel 1: Frontal Chibi (Referencia Real)
   - Panel 2: Pose Molde (Geometría)
   - Panel 3: **Predicción IA Cruda** (Permite auditar artefactos o colapsos de la red neuronal sin posprocesamiento)
   - Panel 4: **IA Postprocesada** (Remapeada con alfa puro y anti-hollín)
   - Panel 5: Ground Truth Real (Objetivo)

---

## 3. Pruebas de Validación Ejecutadas

| Prueba | Comando / Método | Resultado |
|---|---|---|
| **Sintaxis Python** | `python -m py_compile sprite_studio.py pixel_ai_engine/train_supervised.py pixel_ai_engine/palette_remap.py pixel_ai_engine/enhancer.py` | Exitoso (código de salida `0`, 0 errores). |
| **Sintaxis JavaScript Inline** | `node --check` sobre el script de `sprite_studio.html` | Exitoso (código de salida `0`, 0 errores). |
| **Preservación de Paleta y Binarización** | Script Python con 32 grises dominantes y 1 píxel azul único con alfa 128 | Azul conservado (`RGB=[0, 100, 255]`) y alfa binarizado a `255`. |
| **Certificación QC de Alfa** | Simulación de auditoría con 96 celdas con semitransparencias borrosas | Hoja rechazada (`is_ready: False`, pureza alfa 0%). |
| **Adaptación de Formato en Reproductor** | Carga de hoja 16x4 en entorno DOM simulado | Formato actualizado a `16x4`, clips cambiados a `caminar_frente` (0..7), recorte seguro dentro de 2048px de alto. |
| **Selector de Respawn** | Simulación con 0 snapshots y con 2 snapshots | 0 snapshots: badge `"0 Puntos Guardados"`, botón desactivado. 2 snapshots: badge `"2 Puntos Guardados"`, botón habilitado y opciones cargadas. |

---

---

## 5. Resolución de Errores Reproducidos en la Segunda Auditoría

### 5.1. Pérdida NaN y Validación de Finitud al Completar
- **Causa Raíz:** En `update_status`, la transición a `ERROR_NAN` solo se evaluaba si `status_str == "ENTRENANDO"`. La llamada final del ciclo de entrenamiento solicitaba `"COMPLETADO"`, por lo que si una época previa producía `NaN`, este se enmascaraba como `0.0` y el estado final marcaba erróneamente un éxito perfecto.
- **Solución Implementada:**
  - `update_status` valida finitud de forma agnóstica al estado solicitado: si `g_loss` o `d_loss` son `NaN` o no finitos, fuerza incondicionalmente `status = "ERROR_NAN"`.
  - Las métricas no válidas se guardan como `None` (`null` en JSON) en lugar del valor ficticio `0.0`.
  - En `train_supervised_model`, se añade un guardián por época: si `avg_g` o `avg_d` no son finitos, se detiene el entrenamiento de inmediato con `ERROR_NAN`.
  - La llamada final solo emite `"COMPLETADO"` si `avg_g` y `avg_d` son estrictamente finitos y mayores a cero.

### 5.2. Control de Calidad (QC): Rechazo de Sprites Completamente Semitransparentes (Alfa 128)
- **Causa Raíz:** La condición `solid > 0 and (semi / solid) > 0.08` en `run_quality_audit` requería la existencia de al menos un píxel sólido (`solid > 0`). Si una celda completa contenía un sprite con alfa 128 (bloque fantasma sin ningún píxel sólido), `solid` era 0, evadiendo la detección y aprobando la hoja con pureza alfa del 100%.
- **Solución Implementada:**
  - En `sprite_studio.py`, se reformuló la verificación de binaridad: si `solid == 0 and semi >= 5`, se clasifica como celda defectuosa (`blurry_alpha_cells`) con ratio `1.0`.
  - Si existen píxeles sólidos pero la proporción no binaria supera el 2% (`semi / solid > 0.02`), o si la celda acumula más de 15 píxeles difusos, también se marca como defectuosa.
  - La prueba de validación con 96 celdas con bloques verdes de alfa 128 ahora resulta en: `is_ready_for_game: False`, 96 celdas marcadas y pureza alfa de `0.0%`.

### 5.3. Métricas Históricas del Monitor (`best_loss` e `initial_loss`)
- **Causa Raíz:** Si `training_status.json` tenía valores `0.0` heredados de ejecuciones interrumpidas, el cargador los aceptaba porque `0.0` es finito. Dado que ninguna pérdida real es menor que cero, `best_loss` e `initial_loss` quedaban congelados en `0.0` y el porcentaje de reducción en `0.0%`.
- **Solución Implementada:**
  - En `update_status`, se exige estrictamente `float(v) > 0.0` para aceptar métricas existentes.
  - Si `best_loss` o `initial_loss` no están inicializados o eran `0.0`, se recuperan de los valores válidos históricos presentes en `history`.
  - Se eliminó el uso de `0.0` como valor de reserva; si no hay mediciones válidas, se serializa como `null`.
  - Al iniciar un experimento con `mode == "start"`, se reinicia el historial y las métricas en `STATUS_FILE` para evitar contaminaciones de corridas previas.

### 5.4. Reanudación de Schedulers y Generador RNG en CPU
- **Causa Raíz:** Los optimizadores se guardaban, pero los `lr_scheduler` se reconstruían desde cero al reanudar. Además, el tensor de estado RNG de CPU podía presentar incompatibilidades si se cargaba directamente en GPU con `map_location=DEVICE`.
- **Solución Implementada:**
  - `save_checkpoint` ahora almacena `scheduler_g` y `scheduler_d`.
  - Al reanudar desde checkpoint, se restaura el estado de ambos schedulers con `load_state_dict`.
  - El estado del RNG de CPU se transfiere explícitamente a CPU (`torch.get_rng_state().cpu()`) y al restaurar se valida `rng.cpu()` y `torch.uint8` antes de invocar `torch.set_rng_state(rng)`.

---

## 6. Archivos Modificados en el Repositorio

- [`pixel_ai_engine/train_supervised.py`](file:///d:/escritorio/diseñador%20de%20pixel%20art/pixel_ai_engine/train_supervised.py)
- [`pixel_ai_engine/palette_remap.py`](file:///d:/escritorio/diseñador%20de%20pixel%20art/pixel_ai_engine/palette_remap.py)
- [`pixel_ai_engine/enhancer.py`](file:///d:/escritorio/diseñador%20de%20pixel%20art/pixel_ai_engine/enhancer.py)
- [`sprite_studio.py`](file:///d:/escritorio/diseñador%20de%20pixel%20art/sprite_studio.py)
- [`sprite_studio.html`](file:///d:/escritorio/diseñador%20de%20pixel%20art/sprite_studio.html)
- [`MANUAL_Y_DOCUMENTACION_SPRITE_STUDIO.md`](file:///d:/escritorio/diseñador%20de%20pixel%20art/MANUAL_Y_DOCUMENTACION_SPRITE_STUDIO.md)
- [`INFORME_RESOLUCION_AUDITORIA_OCTUBRE_2026.md`](file:///d:/escritorio/diseñador%20de%20pixel%20art/INFORME_RESOLUCION_AUDITORIA_OCTUBRE_2026.md)
