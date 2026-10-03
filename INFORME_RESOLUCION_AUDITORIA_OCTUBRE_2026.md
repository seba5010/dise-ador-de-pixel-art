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

## 5. Resolución Exhaustiva de Observaciones de Auditoría (Commit f3f8da8 → Actual)

### 5.1. Manejo Quirúrgico de Pérdidas NaN e Infinitas
- **Causa Raíz:** Durante el entrenamiento con precisión mixta (AMP), una pérdida o gradiente desbordado generaba `NaN` o `Inf`. Aunque se emitía temporalmente `ERROR_NAN`, la finalización del bucle llamaba a `update_status(..., "COMPLETADO", ...)`, sobrescribiendo el estado con un supuesto éxito y mostrando `g_loss: 0.0`. Además, `json.dump` sin `allow_nan=False` podía fallar o emitir `NaN` no estándar en JSON, y el checkpoint podía corromperse.
- **Solución Implementada:**
  1. **Comprobación de finitud de componentes:** En `train_supervised.py`, antes de acumular métricas o llamar a `backward()`, se valida cada componente de pérdida (`l1_color`, `l1_alpha`, `edge_loss`, `adv_loss`, `loss_d_real`, `loss_d_fake`). Si alguno no es finito, se captura el lote, época y métrica afectada (`error_details`).
  2. **Interrupción controlada:** Se detiene el bucle de inmediato emitiendo `status="ERROR_NAN"`, protegiendo el último checkpoint válido y preservando intacto `best_generator.pt`.
  3. **Serialización estándar:** En `update_status`, las métricas no finitas se convierten estrictamente a `None` (`null` en JSON). La llamada a `json.dump` usa obligatoriamente `allow_nan=False`.
  4. **Diferenciación de pasos omitidos por AMP:** Se registra y monitoriza `skipped_amp_steps` independientemente de los errores NaN, informando al usuario cuando el escalador de gradientes omite un paso de actualización sin catalogarlo como error crítico.
  5. **Monitor UI adaptado:** Tanto en `sprite_studio.html` como en `monitor.html`, los valores `null` se renderizan como `"Sin dato"` y se omiten de la gráfica de pérdidas sin romper el canvas ni las escalas.

### 5.2. Certificación Estricta del Canal Alfa en Control de Calidad (QC)
- **Causa Raíz:** `run_quality_audit` calculaba el ratio de píxeles borrosos condicionándolo a `solid > 0`. Una celda con un personaje completo con alfa 128 (bloque fantasma) tenía `solid == 0`, por lo que el ratio no se calculaba y la hoja recibía pureza alfa 100% y quedaba falsamente certificada.
- **Solución Implementada:**
  - Se estableció el estándar binario del proyecto: alfa exclusivamente `0` o `255`.
  - Detección de cualquier valor intermedio `1 <= alpha <= 254` (`semi = ((alpha > 0) & (alpha < 255)).sum()`).
  - Si `solid == 0 and semi > 0`, se clasifica como celda borrosa crítica (`blurry_alpha_cells`) con ratio `1.0`.
  - Si `solid > 0 and semi > 0`, se reporta el ratio `round(semi / solid, 3)`.
  - La certificación exige `len(blurry_alpha_cells) == 0`, `len(empty_cells) == 0`, `len(border_touch_cells) == 0` y `alpha_purity_score >= 95.0%`.

### 5.3. Reparación y Migración de Métricas Históricas del Monitor
- **Causa Raíz:** Ejecuciones interrumpidas previas habían dejado `best_loss = 0.0` e `initial_loss = 0.0` en `training_status.json`. Al reanudar, como `0.0` era un número finito, el monitor lo adoptaba, impidiendo que pérdidas válidas positivas (ej. 0.09) registraran una mejora, y congelando el porcentaje de reducción en `0.0%`.
- **Solución Implementada:**
  - `update_status` valida estrictamente `v > 0.0` y finitud para inicializar `initial_loss` y `best_loss`.
  - **Migración automática:** Si los valores heredados son `0.0` o `None`, se reconstruyen a partir de las mediciones válidas en `history` (`initial_loss = valid_losses[0]`, `best_loss = min(valid_losses)`).
  - Si la primera pérdida válida es `0.0941` y la siguiente `0.09`, el monitor calcula correctamente `initial_loss: 0.0941`, `best_loss: 0.09` y `loss_reduction_pct: 4.4%`.
  - Un checkpoint con mínimo histórico `0.1` mantiene intacto `0.1` si la siguiente pérdida sube a `0.5`.

### 5.4. Reanudación Robusta de Checkpoints, Schedulers y RNG
- **Causa Raíz:** Se omitían los `lr_scheduler` al guardar, por lo que el esquema de decaimiento se reiniciaba. El estado del generador aleatorio de CPU podía cargarse en el dispositivo incorrecto. Al pausar a mitad de época, se saltaban silenciosamente los lotes pendientes.
- **Solución Implementada:**
  - `save_checkpoint` serializa `scheduler_g` y `scheduler_d` junto a `opt_g`, `opt_d`, `scaler_g`, `scaler_d`.
  - El RNG de CPU se transfiere explícitamente a CPU (`torch.get_rng_state().cpu()`, `torch.uint8`). Al restaurar, se carga en CPU y los RNG de GPU se restauran mediante `torch.cuda.set_rng_state_all()`.
  - Se eliminaron las excepciones silenciadas (`except: pass`), reemplazándolas por avisos diagnósticos claros (`print("[OK] ...")` o `print("[!] Aviso: ...")`).
  - **Pausa en frontera de época:** Al solicitar la pausa (`pause_requested = True`), el sistema permite que el bucle complete los lotes de la época en curso, asegurando promedios precisos y guardando el checkpoint en una frontera limpia (`epoch`), de modo que al reanudar se empiece exactamente en la época `epoch + 1` sin perder lotes.

### 5.5. Límites de Reproducibilidad Documentados
- **Determinismo garantizado:** El estado de los tensores de pesos, optimizadores, schedulers, escaladores AMP y los generadores de números aleatorios de CPU y GPU se restauran con precisión de bit idéntica.
- **Límites conocidos:** 
  1. Las operaciones atómicas de reducción en GPU (acumulación float16 en núcleos Tensor Core) pueden tener variaciones menores de orden de suma (en el último bit de mantisa).
  2. Para determinismo 100% bit a bit en PyTorch se requeriría `torch.use_deterministic_algorithms(True)` a costa de una penalización severa de rendimiento (~30-40% más lento). Para el diseño de spritesheets de pixel art, la restauración de RNG y schedulers implementada es totalmente suficiente y mantiene el rendimiento óptimo de la RTX 3050 Ti.

---

## 6. Resultados de la Batería de Pruebas Automatizadas (`test_audit_suite.py`)

Se ejecutó la suite de pruebas completa distinguiendo casos de CPU y GPU:

| # | Módulo Evaluado | Prueba Específica | Tipo | Resultado |
|---|---|---|---|---|
| **1.1** | `update_status` | Inyección de NaN e Inf con `allow_nan=False` | CPU | **PASÓ** (Valores convertidos a `null`, no hubo crash). |
| **1.2** | Serialización JSON | Verificación estricta de `allow_nan=False` | CPU | **PASÓ** (Garantizada compatibilidad estándar). |
| **1.3** | Bucle / Checkpoints | NaN en pérdida no emite `COMPLETADO` ni sobrescribe modelo | CPU | **PASÓ** (Estado `ERROR_NAN`, checkpoint previo intacto). |
| **2.1** | Certificación Alfa | Contenido opaco (alfa 255), centrado y con márgenes | CPU | **PASÓ** (Aprobado: `certified=True`, pureza `100.0%`). |
| **2.2** | Certificación Alfa | Contenido 100% semitransparente con alfa 128 (ghost) | CPU | **PASÓ** (Rechazado: `certified=False`, 64 celdas borrosas). |
| **2.3** | Certificación Alfa | Contenido mixto (alfa 128 y alfa 255) | CPU | **PASÓ** (Rechazado: `certified=False`, celdas detectadas). |
| **2.4** | Certificación Alfa | Frames vacíos, faltantes o tocando bordes | CPU | **PASÓ** (Rechazado: 34 vacías, 30 tocando bordes). |
| **3.1** | Métricas del Monitor | Pérdida inicial 0.0941 y siguiente 0.09 | CPU | **PASÓ** (`initial_loss: 0.0941`, `best_loss: 0.09`, reducción `4.4%`). |
| **3.2** | Métricas del Monitor | Preservación de mínimo histórico 0.1 ante subida a 0.5 | CPU | **PASÓ** (`best_loss: 0.1` conservado). |
| **3.3** | Métricas del Monitor | Migración de ceros heredados desde historial | CPU | **PASÓ** (`0.0` corregido a `0.0941` e `0.085`). |
| **4.1** | Checkpoints | Guardado/carga de schedulers y RNG en CPU (torch.uint8) | CPU | **PASÓ** (Estados presentes y validados). |
| **4.2** | Checkpoints | Retrocompatibilidad con checkpoints heredados | CPU | **PASÓ** (Carga exitosa sin errores de clave faltante). |
| **5.1** | Checkpoints GPU | Guardado y restauración de RNG CUDA y `GradScaler` | GPU (CUDA) | **PASÓ** (`cuda_rng_state` restaurado en dispositivo 0). |

---

## 7. Instrucciones para Ejecución Local de 5–10 Épocas

Para iniciar una prueba de entrenamiento local de 5 a 10 épocas en PowerShell:

```powershell
# 1. Posicionarse en el directorio del proyecto
cd "d:\escritorio\diseñador de pixel art"

# 2. Entrenar 5 épocas iniciando desde el mejor checkpoint existente
& "webui forger\system\python\python.exe" pixel_ai_engine\train_supervised.py --epochs 5 --mode resume

# O para una prueba limpia de 10 épocas:
& "webui forger\system\python\python.exe" pixel_ai_engine\train_supervised.py --epochs 10 --mode start

# 3. Lanzar la interfaz de Sprite Studio y el monitor en segundo plano
Start-Process -FilePath "cmd.exe" -ArgumentList "/c iniciar_sprite_studio.bat"
```

---

## 8. Evaluación de Preparación y Lista de Validación Visual

### ¿Está el proyecto preparado para la prueba de 5–10 épocas?
**SÍ, AL 100%.** Todas las condiciones numéricas, arquitecturales y de persistencia están verificadas y validadas con pruebas unitarias y de integración que pasaron exitosamente.

### Aspectos que deben validarse visualmente en las salidas generadas:
1. **Rostro y Expresión:** Verificar nitidez de ojos, cejas y boca del personaje chibi sin borrones ni pérdida de píxeles clave.
2. **Fidelidad de Colores:** Confirmar que la paleta del uniforme (ej. chaqueta de chef blanca o negra, botones dorados) coincide exactamente con la ilustración frontal de referencia.
3. **Accesorios:** Corroborar presencia de sombreros, cinturones o herramientas sin artefactos difusos.
4. **Poses y Perspectiva:** Comprobar que las 4 direcciones cardinales (frente, espalda, izquierda, derecha) respetan la anatomía 16-bit.
5. **Proporción Constante:** La altura y volumen del personaje deben mantenerse estables a través de todos los fotogramas del ciclo.
6. **Alineación de Pies:** Los pies deben contactar de manera uniforme la línea base del molde geométrico para evitar la sensación de "flotación" al animarse.
7. **Ausencia de Recortes (Márgenes Seguros):** Ningún frame debe tocar los bordes exteriores de su celda de 64x64 píxeles (0 sangrado).

---

## 9. Archivos Modificados en el Repositorio

- [`pixel_ai_engine/train_supervised.py`](file:///d:/escritorio/diseñador%20de%20pixel%20art/pixel_ai_engine/train_supervised.py)
- [`sprite_studio.py`](file:///d:/escritorio/diseñador%20de%20pixel%20art/sprite_studio.py)
- [`sprite_studio.html`](file:///d:/escritorio/diseñador%20de%20pixel%20art/sprite_studio.html)
- [`monitor.html`](file:///d:/escritorio/diseñador%20de%20pixel%20art/monitor.html)
- [`test_audit_suite.py`](file:///d:/escritorio/diseñador%20de%20pixel%20art/test_audit_suite.py)
- [`MANUAL_Y_DOCUMENTACION_SPRITE_STUDIO.md`](file:///d:/escritorio/diseñador%20de%20pixel%20art/MANUAL_Y_DOCUMENTACION_SPRITE_STUDIO.md)
- [`INFORME_RESOLUCION_AUDITORIA_OCTUBRE_2026.md`](file:///d:/escritorio/diseñador%20de%20pixel%20art/INFORME_RESOLUCION_AUDITORIA_OCTUBRE_2026.md)

