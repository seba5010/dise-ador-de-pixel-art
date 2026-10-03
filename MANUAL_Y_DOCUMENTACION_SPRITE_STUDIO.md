# MANUAL OFICIAL Y ARQUITECTURA DE SPRITE STUDIO

> **Villa del Chef — Suite Integral de Generación, Animación y Calidad Pixel Art (16-bit Retro)**  
> **Versión**: 2.5 (Motor Canónico 100% PyTorch UNet — Supervisado con 4 Candados)

---

## 1. Resumen Ejecutivo y Decisión Arquitectural

**Sprite Studio** es la plataforma visual y operativa para la creación de spritesheets de personajes 2D en pixel art de alta precisión para el videojuego *Villa del Chef*.

### 1.1. Deprecación y Bloqueo Definitivo de Forge SD1.5 / LoRA
- **Causa**: Stable Diffusion Forge 1.5 opera mediante difusión probabilística latente a partir de texto. A pesar de múltiples calibraciones y entrenamientos de LoRA (hasta Época 13), el modelo presentaba:
  1. *Fantasías morfológicas*: deformación de manos, proporciones alteradas y fondos accidentales.
  2. *Latencia y dependencia externa*: requería un servidor secundario en el puerto 7860 y reintentos HTTP que retrasaban la interfaz.
  3. *Pesos corruptos / inestables*: checkpoints `.safetensors` no estructurados para rejillas fijas de 96 celdas.
- **Acción Realizada**: Se han **bloqueado y purgado** todas las referencias a Forge SD1.5 y a los checkpoints antiguos en la interfaz y en el backend.
- **Motor Canónico Activo**: **PyTorch UNet Supervisado (Pix2Pix)** ejecutado localmente sobre GPU CUDA, asegurando determinismo geométrico absoluto, canal alfa puro y cero sangrado entre celdas.

---

## 2. Los 4 Candados Canónicos de Estabilidad Geométrica

Para garantizar que el modelo nunca colapse en cuadros grises ni desborde las proporciones del juego:

1. **Candado 1: Entrada Dual Estricta (Identidad + Geometría)**:
   - La red no recibe texto subjetivo. Recibe dos tensores directos: la ilustración frontal RGB del personaje y el molde geométrico de la pose (40 píxeles anclado al suelo).
2. **Candado 2: Pérdida Quirúrgica de Bordes con Kornia Sobel**:
   - Evaluación tensorial directa en GPU (`kornia.filters.sobel`) que penaliza desvanecimientos del contorno pixel art sin recurrir a pasadas lentas por CPU.
3. **Candado 3: Aceleración de Memoria con AdamW de 8 Bits (`bitsandbytes`)**:
   - Optimización que reduce la huella de VRAM a solo ~1.5 GB, permitiendo entrenamiento e inferencia en GPUs de portátiles (RTX 3050 Ti) sin estrangulamiento térmico.
4. **Candado 4: Post-procesamiento y Remapeo de Paleta**:
   - Extracción de la paleta RGB exacta de la imagen frontal del personaje y remapeo de cada píxel generado al color más cercano (distancia Euclidiana en espacio RGB/CIELAB) eliminando cualquier color espurio o artefacto de compresión.

---

## 3. Desglose Detallado de los 4 Apartados (Pestañas)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 🎨 SPRITE STUDIO (Header)                              │
│  💻 GPU: RTX 3050 Ti (4.0 GB)  │  ⚪ GPU Libre  │  ⚡ Motor: PyTorch UNet (Activo)     │
├────────────────────┬──────────────────────────┬──────────────────────┬─────────────────┤
│ 🎮 Generador/Edit  │ 🎬 Reprod. Animaciones   │ 📈 Monitor Entrenam. │ 🔍 Control Cal. │
└────────────────────┴──────────────────────────┴──────────────────────┴─────────────────┘
```

---

### Pestaña 1: 🎮 Generador & Editor
La consola principal para la inferencia y producción de hojas de sprites completas o frames individuales.

- **Panel de Personaje y Referencia**:
  - `Personaje`: Desplegable con los personajes oficiales (`Alex`, `Amaro`, `Conny`, `Dana`, `Belial`, `Andrea`, etc.). La estrella (★) indica personajes con ground-truth disponible.
  - `Ilustración Frontal`: Selecciona el uniforme o atuendo (`rnormal`, `rbchef`, etc.) con vista previa miniatura y nombre de archivo.
- **Formato & Molde**:
  - `Formato`: **Canónico 8x12** (96 poses, 1024x1536 px) o **Legacy 16x4** (64 poses, 682x2048 px).
  - `Molde`: Selecciona la plantilla geométrica que ancla los pies y define la perspectiva de la cámara.
- **Motor & Checkpoints**:
  - Fijado de forma canónica en `⚡ PyTorch UNet (Motor Canónico Supervisado + 4 Candados)`.
  - `Pesos / Checkpoint`: Permite elegir el modelo exacto (`best_generator.pt`, `latest_checkpoint.pt` o snapshots guardados).
  - `Semilla (Seed)`: Control determinista opcional.
- **Acciones de Generación**:
  - `⚡ Probar 4 Poses`: Genera de inmediato las 4 orientaciones cardinales (Sur, Este, Norte, Oeste) para validación rápida.
  - `✨ Generar Hoja Completa (96 Frames)`: Produce la matriz completa de 96 frames con todas las direcciones y acciones.
  - `🔄 Reanudar`: Continúa una generación interrumpida desde el último frame guardado.
- **Visor Central del Spritesheet**:
  - Muestra la hoja en alta resolución con zoom dinámico (`0.4x` a `1.5x`).
  - Rejilla interactiva superpuesta (SVG): al hacer clic en cualquier celda, se selecciona para inspección o re-generación quirúrgica individual.
  - Barra de progreso en tiempo real con indicador porcentual y mensaje de estado.
- **Inspector Lateral de Celda**:
  - Desglosa la celda seleccionada: Número de frame, Fila, Columna, Acción (`caminar`, `cocinar`, `pensar`, etc.) y Dirección (`sur`, `norte`, etc.).
  - Botón `⚡ Re-generar este Frame` para correcciones puntuales sin repetir toda la hoja.

---

### Pestaña 2: 🎬 Reproductor de Animaciones
Visor dinámico para previsualizar los ciclos de animación del juego antes de exportar a Unity.

- **Selector de Hoja / Run (`animRunSelect`)**:
  - Lista todas las ejecuciones disponibles en `output/` con nombre del personaje, formato y número de frames generados.
  - Permite cambiar de personaje u hoja generada directamente desde el reproductor.
- **Ciclo de Animación (`animCycleSelect`)**:
  - Direcciones de Caminata: Sur, Sureste, Este, Noreste, Norte, Noroeste, Oeste, Suroeste (8 direcciones ortogonales e isométricas).
  - Acciones Especiales de Restaurante:
    - *Cocinar (Bowl)*
    - *Cocinar (Estación Lateral)*
    - *Pensar (Mano en barbilla)*
    - *Cargar Caja*
    - *Servir Plato*
    - *Celebrar (Manos arriba)*
    - *Reposo / Idle*
- **Control de Tiempo y Escala**:
  - Slider de FPS (1 a 24 fotogramas por segundo, predeterminado 8 FPS).
  - Slider de Zoom de visualización (1x a 8x con filtrado de píxel nearest-neighbor estricto).
- **Controles de Reproducción Cuadro a Cuadro**:
  - `⏮️`: Retrocede exactamente un frame en la secuencia (función `prevAnimFrame()`).
  - `⏸️ / ▶️`: Alterna entre reproducción continua y pausa (función `togglePlayPause()`).
  - `⏭️`: Avanza un frame en la secuencia (función `nextAnimFrame()`).
- **Navegación Cruzada**:
  - Botones directos para saltar a *Control de Calidad* o al *Generador & Editor*.

---

### Pestaña 3: 📈 Monitor de Entrenamiento
Panel de control y telemetría en tiempo real del modelo supervisado de Deep Learning.

- **Panel Superior de Control**:
  - `Motor`: Fijo en `⚡ PyTorch UNet (Supervisado Pix2Pix + Kornia GPU Edge)`.
  - `Épocas`: Selector de duración (25, 50, 100, 150, 250 épocas).
  - Botones de Ciclo de Vida:
    - `▶️ Iniciar`: Lanza `train_supervised.py` en segundo plano utilizando el intérprete de Python embebido.
    - `⏸️ Pausar`: Crea un checkpoint inmediato y pausa el optimizador.
    - `🔄 Reanudar`: Continúa el entrenamiento desde el último checkpoint sin reiniciar curvas.
    - `⏹️ Detener`: Finaliza el proceso y guarda el estado.
- **Métricas Clave (Tarjetas Numéricas)**:
  - *Época Actual y Progreso Porcentual*.
  - *Tiempo Transcurrido y Estimado (ETA)*.
  - *Pérdida del Generador (G-Loss)* y *Mínima Histórica (Best Loss)*.
  - *Pérdida L1 (Fidelidad de Píxel)* y *Pérdida de Bordes (Sobel Edge Loss)*.
  - *Velocidad de Entrenamiento (Frames / seg)* y *Consumo de VRAM*.
  - *Temperatura de GPU (°C)* con alerta visual.
- **Gráfico de Curvas de Pérdida en Tiempo Real**:
  - Canvas interactivo que dibuja la curva histórica de convergencia época por época con tooltip flotante.
- **Sistema de Respawn (Restauración Temporal cada 10 Épocas)**:
  - Desplegable de snapshots guardados (`checkpoint_epoch_10.pt`, `checkpoint_epoch_20.pt`, etc.).
  - Botón `⏪ Respawn`: revierte el modelo a un punto temporal anterior si se detecta sobreajuste en épocas avanzadas.
- **Muestras Visuales en Vivo**:
  - Muestra la predicción actual del generador (`latest_preview.png`) y la comparativa de detalle contra el ground-truth (`latest_detail_comparison.png`).

---

### Pestaña 4: 🔍 Control de Calidad Quirúrgico (Quality Gate)
Auditoría técnica automatizada que certifica si una hoja cumple con los estándares para su importación a Unity.

- **Selector de Hoja**:
  - Permite auditar la hoja activa o cualquier hoja generada previamente en `output/`.
  - Botón `⚡ Auditar Ahora` y casilla de `Auditoría en tiempo real`.
- **Veredicto General**:
  - Banner dinámico:
    - 🟢 `100% APTO PARA UNITY (CERTIFICADO)`: Cumple todos los requisitos.
    - 🟡 `REQUIERE REVISIÓN`: Señala celdas con incidencias.
- **Métricas de Aceptación**:
  - *Completitud*: Porcentaje de frames generados (ej. 96/96).
  - *Pureza Alfa*: 100% libre de halos semitransparentes o ruido de fondo.
  - *Seguridad de Márgenes (Anti-sangrado)*: Verifica que ningún píxel del personaje toque los bordes exteriores de la celda de 128x128 píxeles.
- **Lienzo SVG con Overlays Interactivos**:
  - Dibuja el spritesheet completo con capas vectoriales coloreadas:
    - 🔴 Rojo: Píxeles en contacto con el borde (con flechas indicando la dirección del desborde: superior, inferior, izquierda, derecha).
    - 🟡 Amarillo: Celdas vacías o no generadas.
    - 🟣 Morado: Canal alfa con semitransparencias difusas.
  - Al pulsar sobre cualquier celda problemática, se abre la opción de cargarla en el Editor o en el Reproductor para su ajuste.
- **Herramientas de Exportación para Unity**:
  - `📥 Spritesheet Limpio (PNG)`: Exporta la imagen transparente final sin rejilla.
  - `📥 Rejilla de Inspección (PNG)`: Exporta la imagen con líneas de corte para debug visual.
  - `📄 Unity Metadata (JSON)`: Exporta las coordenadas exactas de cada sprite (`x`, `y`, `width`, `height`, `pivot`) listas para importar mediante el script de Unity `ImportSpritesheet.cs`.

---

## 4. Referencia de la API Local (Puerto 8080)

| Endpoint | Método | Descripción |
| :--- | :--- | :--- |
| `/api/status` | `GET` | Devuelve el estado de la GPU, motor activo, progreso del trabajo y estado de entrenamiento. |
| `/api/characters` | `GET` | Lista todos los personajes disponibles en `personajes/` y sus ilustraciones frontales. |
| `/api/pose_map` | `GET` | Devuelve el mapeo de 96 poses canónicas (dirección, acción, fila, columna). |
| `/api/runs` | `GET` | Lista todas las ejecuciones y spritesheets generados en `output/`. |
| `/api/quality_audit` | `GET` | Ejecuta la auditoría matemática de completitud, pureza de alfa y bordes de una hoja. |
| `/api/generate` | `POST` | Inicia la generación con PyTorch UNet (parámetros: `character`, `front_image`, `mode`, `checkpoint`). |
| `/api/train/start` | `POST` | Inicia el entrenamiento supervisado en GPU con PyTorch UNet. |
| `/api/train/pause` | `POST` | Pausa el entrenamiento de forma segura guardando el checkpoint actual. |
| `/api/train/resume` | `POST` | Reanuda el entrenamiento supervisado. |
| `/api/train/stop` | `POST` | Detiene definitivamente el proceso de entrenamiento. |
| `/api/train/respawn` | `POST` | Restaura el modelo a un snapshot temporal anterior (cada 10 épocas). |

---

## 5. Verificación de Funcionamiento

- **Sintaxis JavaScript**: Validada con `node --check` (0 errores de análisis).
- **Prueba End-to-End con Navegador Headless (CDP)**:
  - Cambio fluido y sin excepciones entre las 4 pestañas.
  - Sincronización instantánea de las 25 ejecuciones en los desplegables.
  - Paso cuadro a cuadro verificado con `nextAnimFrame()` y `prevAnimFrame()`.
  - Latencia de respuesta de `/api/status`: inferior a 5 milisegundos.

---

## 6. Estabilidad de Entrenamiento y Corrección Numérica (Octubre 2026)

- **Corrección de Subnormales en MinibatchStdDev**: Reemplazado `sqrt(var + 1e-8)` por `sqrt(clamp(var, min=1e-4))` en `models.py` para prevenir colapso de derivadas infinitas (`NaN`) en float16 AMP.
- **Protección de Gradientes No Finitos**: Validación con `torch.isfinite(norm)` antes de aplicar pasos del optimizador con `GradScaler`.
- **Orden de Parámetros en Discriminador PatchGAN**: Restablecido orden canónico `(condition, target)` en `train_supervised.py`.
- **Verificación**: Entrenador operativo al 100% con métricas reales (`G_Loss: 0.0941`, `D_Loss: 0.6536`) y sin desbordes numéricos.

---

## 7. Resolución Quirúrgica de Auditoría Técnica (Octubre 2026)

A raíz de la auditoría exhaustiva del repositorio, se implementaron las siguientes 5 correcciones arquitecturales:

1. **Reanudación y Preservación de `best_loss` y Schedulers**:
   - `save_checkpoint` unifica el guardado de checkpoints normales, pausas, paradas y snapshots de 10 épocas.
   - Preserva `best_loss`, optimizadores `opt_g`/`opt_d` (8-bit AdamW / Adam), escaladores AMP `scaler_g`/`scaler_d`, schedulers CosineAnnealing (`scheduler_g`/`scheduler_d`) y estados de generadores de números aleatorios (RNG en CPU/CUDA con validación de tipo de tensor).
   - En caso de reanudar desde checkpoints antiguos que carezcan de `best_loss`, se recupera automáticamente desde `best_generator.pt`.
   - Protección integral contra `NaN`: `update_status` intercepta pérdidas no finitas independientemente del estado (`COMPLETADO` o `ENTRENANDO`), asignando `status: "ERROR_NAN"` y serializando métricas inválidas como `null` en lugar de `0.0`.
   - Recuperación de métricas históricas: descarta valores `0.0` heredados para `best_loss` e `initial_loss`, recuperándolos del historial válido.

2. **Sincronización Reactiva de Respawn**:
   - El selector `#respawnEpochSelect` se alimenta dinámicamente de `/api/train/snapshots` y de la telemetría en tiempo real (`info.snapshots`).
   - El badge `#respawnCountBadge` refleja el número exacto de puntos de restauración disponibles.
   - Si no existen snapshots, el desplegable muestra el placeholder desactivado; al generarse snapshots, se habilita inmediatamente y retiene la selección del usuario o preselecciona el snapshot más reciente.

3. **Reproductor Multiformato (8x12 y 16x4) y Redibujado en Pausa**:
   - `loadRunSheet` detecta el formato de la hoja (`8x12` de 96 frames o `16x4` de 64 frames) a partir de los metadatos de la ejecución y ajusta automáticamente la rejilla, la plantilla y el visor.
   - Se crearon diccionarios de clips independientes (`ANIMATION_CLIPS_8X12` y `ANIMATION_CLIPS_16X4`). Las hojas de 64 frames ya no solicitan índices fuera de rango (como celebrar en el frame 92).
   - `drawAnimFrame` implementa acotamiento estricto (`safeIdx`), garantizando que el recorte no sobrepase los límites de altura o anchura del spritesheet.
   - El manejador `sheetImg.onload` redibuja el fotograma en el lienzo inmediatamente tras la carga asíncrona de la imagen, garantizando actualización visual instantánea incluso si la animación está en pausa.

4. **Control de Calidad Quirúrgico de Canal Alfa**:
   - `run_quality_audit` incluye formalmente el análisis de `blurry_alpha_cells` en la condición de certificación `is_ready`.
   - Detecta y rechaza de forma inmediata celdas completamente semitransparentes (por ejemplo, bloques o sprites fantasma con alfa 128 donde `solid == 0 and semi >= 5`), marcándolas con ratio `1.0`.
   - Para celdas con píxeles opacos, rechaza cualquier sangrado con semitransparencias mayores al 2% (`semi / solid > 0.02`) o con más de 15 píxeles difusos.
   - Se exige que `len(blurry_alpha_cells) == 0` y `alpha_purity_score >= 95.0%`. Cualquier hoja con píxeles semitransparentes o bordes difuminados es calificada como `BORRADOR / REVISIÓN REQUERIDA`.

5. **Colores, Acabado Unificado y Transparencia de Inferencia**:
   - `Albumentations.ColorJitter` bloquea explícitamente `saturation=0.0` y `hue=0.0`, limitando las variaciones exclusivamente a cambios sutiles de brillo y contraste sin alterar los tonos de la ropa ni la piel del personaje.
   - `extract_character_palette` aplica K-Means clustering sobre los colores únicos del personaje, garantizando la preservación de tonos minoritarios (ojos azules, gemas, accesorios pequeños) junto con la paleta de props autorizada (`PROPS_PALETTE`).
   - `remap_image_to_palette` incorpora tolerancia euclídea (35.0) para no destruir colores de accesorios legítimos y binariza estrictamente el canal alfa a 0 o 255.
   - `PixelArtEnhancer.enhance_frame` sella la salida con `binarize_alpha(threshold=40)`, unificando el acabado entre la generación/exportación de Sprite Studio y el entrenamiento supervisado.
   - `generate_preview` evalúa muestras fijas sin aumentos aleatorios (`dataset.samples[s_idx]`) y renderiza una tira de 5 columnas: `[Frontal Chibi] | [Pose] | [Predicción IA Cruda] | [IA Remapeada con Alfa Puro] | [Ground Truth Real]`.

