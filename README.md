# Diseñador de Pixel Art — Sprite Studio

Motor local para generar, entrenar, revisar y corregir spritesheets de personajes 2D. El proyecto usa un generador supervisado píxel a píxel y una interfaz web de control de calidad orientada tanto a artistas como a desarrolladores.

**Estado de la documentación:** octubre de 2026
**Versión funcional documentada:** 3.0

## Objetivo

A partir de una ilustración frontal y un molde de pose, el sistema genera frames RGBA que deben conservar:

- identidad, rostro, cabello, ropa y accesorios;
- proporciones y postura del personaje;
- paleta de colores original;
- contornos nítidos propios del pixel art;
- fondo transparente sin halos;
- ubicación segura dentro de cada celda del spritesheet.

Los formatos principales son **8×12 (96 frames)** y **16×4 (64 frames)**.

## Arquitectura actual y Pipeline de Alta Fidelidad (Motor A)

El sistema utiliza **Motor A** (aprendizaje supervisado píxel a píxel en PyTorch) potenciado con una suite de **técnicas neuronales y matemáticas de alta fidelidad** diseñadas para superar el "techo del 30%" de calidad visual (borrosidad por regresión a la media en detalles de 1 píxel como ojos, pupilas, lentes y texturas).

> [!IMPORTANT]
> **Decisión de Diseño Crítica: Descarte de Collage / Recorte y Pegado.**
> Tras pruebas experimentales, el recorte y pegado manual de cabezas/rostros fue **descartado de forma definitiva**. Generaba un efecto de "muñeco de papel recortado / sticker flotante" visualmente inaceptable y sin integración anatómica. Toda la síntesis es **100% neuronal y matemática de extremo a extremo**.

```text
Frontal (RGB) + Pose (RGB) + Mapas Semánticos (5 canales)
                     |
                     v
       Preprocesamiento de Saliencia (Laplaciano)
                     |
                     v
   Generador U-Net con Flujo de Apariencia (Appearance Flow)
                     |
                     v
      Cuantización Vectorial (Discrete Pixel Codebook)
                     |
                     v
    Supervisión Multi-Escala + Laplaciana + Discriminador Local Facial
                     |
                     v
   Post-Proceso Morfológico: Snapping a Paleta + Pixel Art Fixer (1-px contour)
                     |
                     v
            Frame Final 128×128 (Nítido, Estricto en Grilla)
```

### Las Técnicas Neuronales y Matemáticas Integradas

1. **LocalFaceDiscriminator (`pixel_ai_engine/models.py`)**: Sub-discriminador PatchGAN quirúrgico enfocado en el bounding box de la cabeza (32×32 / 64×64). Fuerza al generador a plasmar lentes, cejas y pupilas nítidas sin difuminarlas.
2. **AppearanceFlowModule (`pixel_ai_engine/models.py`)**: Módulo de flujo de deformación densa 2D (`grid_sample`). Mapea y transfiere texturas directas de la ropa y rostro original en vez de recrear píxeles desde cero.
3. **PaletteHistogramLoss (`pixel_ai_engine/models.py`)**: Penalización diferenciable en el espacio de color. Obliga a que la distribución cromática del frame pertenezca exactamente a la paleta canónica del personaje, prohibiendo colores inventados o degradados fangosos.
4. **DiscretePixelCodebook (`pixel_ai_engine/models.py`)**: Cuantizador de vectores discretos estilo VQ-VAE con estimador straight-through (STE). Restringe las características latentes del generador a valores discretos propios del pixel art.
5. **SingleCellCoordinator (`pixel_ai_engine/cell_pipeline.py`)**: Pipeline desacoplado que aísla la generación celda por celda (128×128) con acondicionamiento de pose unitario, eliminando el colapso de generar 96 celdas simultáneamente en un lienzo gigante.
6. **Downscaling Fraccional Inteligente (`pixel_ai_engine/fractional_downscale.py`)**: Algoritmo de reducción sub-píxel guiado por silueta y contraste. Asegura que líneas de 1 píxel (como marcos de gafas) no se pierdan al reducir ilustraciones de alta resolución a sprite art.
7. **MultiScaleDiscriminator + FeatureMatchingLoss (`pixel_ai_engine/models.py`)**: Arquitectura Pix2PixHD con discriminadores a escala completa (1x) y media resolución (0.5x). La pérdida de feature matching estabiliza los detalles finos.
8. **PixelArtFixer (`pixel_ai_engine/pixel_art_fixer.py`)**: Módulo post-procesador morfológico. Ejecuta integer grid snapping, binarización estricta de canal alfa (sin halos semitransparentes), eliminación de píxeles huérfanos y reconstrucción de contorno exterior oscuro de 1 píxel.
9. *(Nota: La técnica de collage/recorte manual de cabeza fue descartada).*
10. **Semantic Molds / DensePose 2D (`pixel_ai_engine/semantic_molds.py`)**: Descomposición del molde de pose en 5 mapas de partes semánticas (Cabeza, Torso, Brazos, Piernas, Accesorios) para guiar la propagación del color y evitar que el color de la piel se filtre a la ropa.
11. **LaplacianPyramidLoss (`pixel_ai_engine/models.py`)**: Pérdida de pirámide Laplaciana multinivel que descompone el frame en bandas de frecuencia espacial, ponderando con 15× las microfrecuencias para eliminar el desenfoque inducido por L1 regular.
12. **Salience Pixelization Pre-processing (`pixel_ai_engine/salience_pixelization.py`)**: Algoritmo de realce de contraste de gradiente previo al generador para preservar micro-rasgos de alta saliencia.

Stable Diffusion/Forge y el LoRA experimental permanecen como herramientas auxiliares o históricas. No son el motor recomendado para producir spritesheets rígidos: tienden a cambiar la escala, inventar colores, contaminar el fondo y no respetar con precisión la cuadrícula.

## Dataset supervisado

El estado actual contiene **56 personajes** y **3124 pares de frames válidos** registrados por el tracker. Estas cantidades se leen dinámicamente; pueden crecer al incorporar nuevos personajes.

Cada muestra relaciona:

1. frontal de identidad;
2. pose o molde correspondiente;
3. target real del dataset;
4. identificador de personaje, variante y número de frame.

La comparación de calidad siempre debe usar el personaje y el frame exactos. No se considera suficiente comparar solamente contra el frontal ni utilizar siempre el frame 0.

## Entrenamiento y checkpoints

El entrenamiento se controla desde la pestaña **Monitor** de Sprite Studio.

Archivos principales:

- `checkpoints/latest_checkpoint.pt`: estado más reciente para reanudar;
- `checkpoints/best_generator.pt`: mejor punto aceptado por pérdida y reglas de seguridad;
- `checkpoints/best_quality_generator.pt`: mejor punto aceptado por calidad visual;
- `checkpoints/last_quality_healthy_checkpoint.pt`: último punto saludable conocido;
- `checkpoints/recovery_checkpoint.pt`: punto preparado para recuperación;
- `checkpoints/snapshots/checkpoint_epoch_XXX.pt`: snapshot completo cada 10 épocas;
- `checkpoints/snapshots/generator_epoch_XXX.pt`: pesos del generador en esa época.

El sistema puede:

- ajustar pesos de pérdida;
- aumentar el muestreo de ejemplos difíciles;
- reducir el learning rate;
- volver al último checkpoint saludable;
- pausar después de colapsos repetidos;
- reanudar con optimizador, schedulers y estados RNG restaurados.

Las escrituras de estado y checkpoints son atómicas. En Windows se utilizan temporales únicos y reintentos para tolerar bloqueos breves de antivirus o lectores del archivo, evitando el antiguo `PermissionError` al reemplazar `training_status.json`.

## Auditoría automática durante el entrenamiento

La auditoría ya no depende de cuatro previews fijos.

### Rotación normal

- Se evalúan **10 personajes por época**.
- Los personajes rotan; los 56 quedan cubiertos aproximadamente cada seis épocas.
- También rota el frame usado para cada personaje.
- La ventana acumulada abarca **14 épocas**.
- Para autorizar un rollback se exigen al menos **2 frames distintos por cada personaje**.
- El tracker debe alcanzar **100% de los frames registrados**.

### Certificación completa

Cada 10 épocas se revisan los 56 personajes y se genera una comparación completa. Una certificación aprobada significa que supera los pisos configurados en ese momento; no significa necesariamente que sea mejor que todos los checkpoints anteriores.

### Protección del rollback

Una recomendación `ROLLBACK` queda en espera hasta disponer de cobertura suficiente. Esto evita volver atrás por una conclusión obtenida de pocos personajes. Cuando la cobertura está completa, la política puede:

1. confirmar que el deterioro sea sostenido;
2. volver al checkpoint saludable exacto;
3. reducir la tasa de aprendizaje;
4. reanudar automáticamente las épocas restantes;
5. detener el ciclo si ya ocurrió un colapso anterior y la degradación se repite.

Cuando el estado pasa a `RECUPERACION_LISTA`, Sprite Studio valida el checkpoint, cambia a `RECUPERANDO` y lanza el entrenamiento en modo `resume` sin exigir que el usuario pulse el botón. Conserva el objetivo original: si el rollback vuelve a la época 20 de un ciclo de 500, programa las 480 épocas restantes.

La reanudación automática tiene un intento por cada ruta de recuperación. Nunca ignora una pausa o detención solicitada y no se activa para errores del proceso, checkpoints ausentes, `PAUSADO_QC` o ciclos ya terminados. Puede deshabilitarse iniciando el servidor con `SPRITE_STUDIO_AUTO_RESUME_QC=0`.

## Control de Calidad interactivo

La pestaña **Control de Calidad** tiene dos niveles diferentes.

### Auditoría de hoja

Comprueba la estructura completa del spritesheet:

- cantidad de frames;
- pureza del canal alfa;
- contacto con los bordes de las celdas;
- celdas vacías;
- comparación visual contra los targets.

Por eso una hoja puede tener 100% de completitud, alfa o márgenes y aun así fallar visualmente. Esos porcentajes técnicos no sustituyen anatomía, rostro, ropa o fidelidad de color.

### Revisión por frame

Cada tarjeta muestra:

- frame generado y target real;
- métricas principales traducidas al español;
- problemas detectados;
- acciones disponibles;
- diagnóstico e historial explicados en lenguaje natural.

El bloque **Ver explicación e historial** responde cuatro preguntas:

1. **¿Cuál es el resultado?** Por ejemplo: aprobada, rechazada, necesita correcciones o enviada a refuerzo.
2. **¿Qué está mal?** Explica si faltan rasgos del rostro, si los ojos no coinciden, si el cuerpo está deformado, si hay ruido, colores incorrectos, contornos cortados o problemas de transparencia.
3. **¿Qué conviene hacer?** Recomienda regenerar, reevaluar o enviar a refuerzo según el tipo de defecto.
4. **¿Qué ocurrió antes?** Traduce cada evaluación, rechazo, aprobación, regeneración y envío a refuerzo como una cronología legible.

Los datos JSON originales continúan disponibles en **Ver datos técnicos (para desarrolladores)**.

### Significado de las métricas

| Métrica | Explicación para el usuario |
|---|---|
| Calidad general | Resumen conservador de toda la comparación. |
| Cuerpo y anatomía | Forma, proporciones y presencia de las partes corporales. |
| Silueta | Coincidencia del contorno exterior con el target. |
| Rostro y ojos | Presencia, ubicación y fidelidad de los rasgos faciales. |
| Ropa | Conservación de prendas y detalles del vestuario. |
| Colores | Fidelidad respecto de la paleta original. |
| Transparencia | Ausencia de halos y píxeles semitransparentes no deseados. |
| Detalles pequeños | Ojos, adornos, pliegues y píxeles distintivos. |
| Contorno | Continuidad, grosor y limpieza de los bordes. |
| Objetos | Presencia y fidelidad de utensilios u otros props. |

La explicación evita afirmar que “falta la cara” cuando la métrica solo demuestra una diferencia moderada. Esa frase se reserva para puntuaciones faciales extremadamente bajas; en los demás casos se informa que los ojos, la boca o la forma no coinciden suficientemente.

### Estados y acciones de una muestra

- **Aprobar:** acepta una muestra que supera la calidad mínima y no posee fallos bloqueantes.
- **Rechazar:** la mueve fuera de la vista activa y la conserva en la pestaña **Rechazados** y en **Historial**. No se borra evidencia.
- **Enviar a refuerzo:** la registra como ejemplo difícil para el entrenamiento y la mueve a la vista **En refuerzo**.
- **Regenerar frame:** crea candidatos aislados; no reemplaza el frame actual hasta que se aplique explícitamente un candidato.
- **Reevaluar:** ejecuta nuevamente la comparación contra el target vigente.
- **Seleccionar todos:** selecciona o deselecciona todas las tarjetas visibles en el filtro actual.

Rechazar y enviar a refuerzo son acciones distintas. Una muestra rechazada no entra automáticamente al entrenamiento; debe enviarse a refuerzo cuando se quiera que el modelo aprenda de ella.

## Cómo interpretar QUALITY GUIDANCE

Ejemplo:

```text
Problema principal: palette
Severidad: CRITICAL
Acción recomendada: ROLLBACK
Acción autorizada: CONTINUE
rollback EN ESPERA
```

Significa que los colores son el mayor cuello de botella y el controlador recomienda volver a un punto anterior, pero todavía no reunió la cobertura requerida. `CONTINUE` no significa que la muestra sea buena; significa que la intervención está temporalmente bloqueada por confirmación, cobertura, cooldown o presupuesto de acciones.

## Sobreentrenamiento

El número solicitado de épocas es un máximo, no una meta obligatoria. Debe conservarse el checkpoint que mejora la calidad visual, aunque la ejecución tenga épocas restantes.

Señales de deterioro:

- `G_Loss` y `Color_L1` suben durante varias épocas;
- calidad general, anatomía o silueta caen de forma sostenida;
- una categoría mejora a costa de varias otras;
- aumenta la cantidad de pasos AMP omitidos;
- la calidad queda por debajo del mejor checkpoint durante cinco o más épocas.

Los snapshots, rollback y `best_quality_generator.pt` protegen el trabajo, pero la auditoría actual utiliza targets pertenecientes al dataset supervisado. Para medir sobreajuste clásico con total rigor todavía se recomienda incorporar un conjunto fijo de validación que nunca participe en el gradiente.

## Archivos principales

| Archivo | Responsabilidad |
|---|---|
| `sprite_studio.py` | Servidor web y API local. |
| `sprite_studio.html` | Interfaz de generación, entrenamiento y QC. |
| `pixel_ai_engine/train_supervised.py` | Ciclo de entrenamiento con LaplacianPyramidLoss y PaletteHistogramLoss. |
| `pixel_ai_engine/models.py` | Redes neuronales: U-Net, AppearanceFlow, Codebook, Face/MultiScale Discriminators. |
| `pixel_ai_engine/fractional_downscale.py` | Downscaling fraccional sub-píxel guiado por contraste y silueta. |
| `pixel_ai_engine/pixel_art_fixer.py` | Integer grid snapping, contorno exterior morfológico y limpieza binaria de alfa. |
| `pixel_ai_engine/semantic_molds.py` | Particionado semántico DensePose 2D (cabeza, torso, extremidades, accesorios). |
| `pixel_ai_engine/salience_pixelization.py` | Realce previo de gradiente espacial y saliencia visual. |
| `pixel_ai_engine/cell_pipeline.py` | Coordinador desacoplado celda por celda (SingleCellCoordinator). |
| `export_final_spritesheets.py` | Exportador final con remapeo cromático 100% estricto y PixelArtFixer. |
| `pixel_ai_engine/quality_guidance.py` | Diagnóstico, tendencias e intervenciones. |
| `pixel_ai_engine/quality_gate.py` | Evaluación visual por frame. |
| `pixel_ai_engine/frame_quality_review.py` | Persistencia y estados de revisión. |
| `pixel_ai_engine/interactive_qc.py` | Acciones individuales y por lote. |
| `pixel_ai_engine/frame_regeneration.py` | Candidatos de regeneración. |
| `pixel_ai_engine/hard_examples.py` | Cola de ejemplos enviados a refuerzo. |
| `pixel_ai_engine/training_recovery.py` | Escrituras atómicas y recuperación segura. |
| `training_status.json` | Estado, historial y QUALITY GUIDANCE actual. |
| `training_logs/training.log` | Registro legible de las ejecuciones. |
| `frame_review_queue.jsonl` | Historial persistente de revisiones por frame. |

## Puesta en marcha

Desde PowerShell, en la raíz del proyecto:

```powershell
& ".\webui forger\system\python\python.exe" ".\sprite_studio.py"
```

Abrir:

```text
http://localhost:8080/sprite_studio.html
```

## Pruebas

Ejecutar con el mismo Python utilizado por el proyecto:

```powershell
& ".\webui forger\system\python\python.exe" -m pytest -q
```

Las pruebas cubren calidad, recuperación, revisión por frame, regeneración, refuerzo, selección por lotes y endpoints del servidor.

## Regla operativa recomendada

No aprobar una hoja solo porque completitud, alfa y márgenes indiquen 100%. Primero revisar la comparación visual y las tarjetas por frame. Cuando una muestra falle:

1. leer la explicación natural;
2. rechazarla si no debe utilizarse;
3. regenerarla si el defecto es estructural;
4. enviarla a refuerzo si el fallo se repite;
5. reevaluar la nueva versión antes de aprobarla.
