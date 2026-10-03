# DOCUMENTACIÓN OFICIAL DEL PROYECTO: DISEÑADOR DE PIXEL ART
**Villa del Chef — Motor Neuronal de Spritesheets y Análisis Comparativo de Motores**  
*Fecha: Octubre 2026 | Versión: 2.5 (Arquitectura Blindada y Control de Respawn)*

---

## 1. Resumen Ejecutivo del Proyecto

El objetivo de este proyecto es desarrollar una herramienta de software e inteligencia artificial capaz de generar **hojas de sprites (spritesheets) completas de animación 2D en Pixel Art (formatos 16x4 de 64 frames y 8x12 de 96 frames)** a partir de **una única ilustración frontal de referencia** de un personaje (nuevo o existente), manteniendo:
1. **Identidad absoluta del personaje:** Tono de piel exacto, peinado, ropa, delantal y detalles anatómicos.
2. **Escala y anclaje canónico:** Personaje de ~32 a 48 píxeles de altura neta, anclado al piso en celdas de 128×128 (o 170×128 en 16x4).
3. **Fondo 100% transparente (Alpha limpio):** Sin halos, sin fondos difusos o artefactos de compresión.
4. **Rigidez de animación:** Cuadrícula matemática estricta con todas las direcciones (sur, este, oeste, norte, diagonales) y acciones complejas (caminar, cocinar con bowl, pensar, cargar caja, servir plato, celebrar).

---

## 2. Dictamen Técnico Crítico: Por qué el Motor de Difusión (Forge SD 1.5 + LoRA) NO Sirve para este Caso Específico

Durante la fase experimental se integró un entorno de **Stable Diffusion WebUI Forge (SD 1.5 con LoRA entrenado sobre los personajes del juego)**. Tras auditorías visuales y pruebas de generación directa, se concluye formalmente que **el motor de difusión generativa libre por texto es inviable y NO sirve para este caso de uso**.

A continuación se detallan las razones técnicas fundamentales de este fallo:

### 2.1. El Paradigma de "Alucinación Libre" vs la "Rigidez Matemática" del Pixel Art
* **Naturaleza de Stable Diffusion:** Los modelos de difusión (SD 1.5) son sistemas probabilísticos diseñados para "imaginar" e ilustrar arte conceptual libre en base a descripciones en lenguaje natural.
* **El Requisito del Videojuego 2D:** Un motor de spritesheet para videojuegos exige determinismo atómico:
  * El pie en el frame 2 **debe** subir 3 píxeles exactos, no 12 ni 0.
  * La paleta de color **debe** ser el valor hexadecimal original del personaje, no una reinterpretación estética.
* **El Resultado en Forge:** Al pedirle por texto generar un frame de caminata, SD 1.5 "alucina" poses arbitrarias, cambia las posiciones de las extremidades a su antojo y pierde la sincronía de la cuadrícula de animación.

### 2.2. Destrucción de la Escala y el Formato ("La Falla Dulce")
* **Pérdida Matemática Engañosa:** En el entrenamiento LoRA, la pérdida matemática reportaba una reducción del >90% (loss de 0.12 a 0.011), lo que en apariencia sugería un "punto dulce".
* **Realidad Visual:** Matemáticamente la red memorizó la relación texto-imagen, pero la representación generada fue un muñeco desproporcionado tipo "chibi gigante / entrenador Pokémon" que ocupaba el 85% de la celda de 128×128, en lugar del sprite estilizado y diminuto de ~40 px de Villa del Chef.
* **Contaminación de Fondos y Artefactos:** SD 1.5 trabaja en un espacio latente de 512×512 entrenado con imágenes RGB con fondos. Al forzar la generación de pixel art:
  * Genera fondos beige/grises con halos en los bordes.
  * Al hacer el reescalado a 128×128 con Nearest-Neighbor, los rostros colapsan en manchas negras (ojos que parecen gafas de sol rectangulares) y las manos pierden definición anatómica.

### 2.3. Ignorancia de la Imagen Frontal (Falta de Acondicionamiento Directo)
* El endpoint `txt2img` de Forge solo recibe cadenas de texto (`prompt`). Ignora por completo los píxeles reales de la imagen frontal seleccionada por el usuario.
* Si el usuario sube un personaje nuevo con delantal blanco o un peinado singular, Forge no puede transferir esos píxeles directamente; intenta inventar un personaje nuevo basado únicamente en palabras clave, cambiando colores arbitrariamente (por ejemplo, vistiendo al chef de rojo y negro).

> **Conclusión sobre Forge:** El motor de difusión SD 1.5 + LoRA queda documentado oficialmente como **OBSOLETO Y NO APTO** para la generación de spritesheets rígidos en este proyecto.

---

## 3. El Motor Válido: Generador Supervisado Directo (Pix2Pix / UNet 6 Canales)

La solución técnica que cumple al 100% con los requisitos del juego es el **Generador Supervisado Píxel a Píxel (`PixelArtUNetGenerator`)**.

### 3.1. Arquitectura de Entrada y Salida
A diferencia de los modelos de difusión, este modelo es un traductor directo de imagen a imagen:

```
[FRONTAL DEL PERSONAJE] (3 canales RGB: Identidad, Colores reales, Cabello, Ropa)
           +
[MOLDE CANÓNICO DE LA POSE] (3 canales RGB: Silueta exacta, Altura 40px, Suelo fijo)
           │
           ▼
     ┌───────────┐
     │  U-Net    │  (Entrada: 6 canales | Salida: 4 canales RGBA)
     └─────┬─────┘
           ▼
[SPRITE FINAL 128×128] (Píxeles fieles, fondo alfa transparente estricto)
```

### 3.2. Los 4 Candados de Seguridad Implementados
Para evitar los incidentes históricos de colapso modal (cuadros grises) o desbordes numéricos, el script `pixel_ai_engine/train_supervised.py` cuenta con 4 blindajes activos:

1. **🔒 Candado 1: Cero Texto, Cero Alucinación:**
   La red no procesa texto. Se alimenta de matrices de píxeles reales (Frontal + Molde). No tiene libertad física para cambiar el tamaño de la cabeza ni inventar poses extrañas.
2. **🔒 Candado 2: Supremacía del Generador sobre el Discriminador:**
   - Tasa de aprendizaje del discriminador a la mitad (`lr * 0.5`).
   - Peso adversarial reducido al 5% (`adv_loss * 0.05`). El 95% del entrenamiento lo dominan las pérdidas reconstructivas: **Smooth L1 Color (peso 5.0)**, **Smooth L1 Alpha (peso 2.5)** y **Sobel Edge Loss (peso 1.5)**.
   - Dataset supervisado de **1.008 frames** reales (Alex, Amaro, Belial, Conny, Dana) que impide que el discriminador memorice las muestras.
3. **🔒 Candado 3: Blindaje Numérico Anti-NaN:**
   Clamping estricto de logits a Float32 en rango `[-30.0, 30.0]` en todas las llamadas `BCEWithLogitsLoss`, eliminando desbordes numéricos en FP16 AMP.
4. **🔒 Candado 4: Auditoría Visual en Vivo (4 Columnas) y Récord Histórico:**
   Generación en cada época de la tira comparativa:
   `[Frontal] -> [Molde] -> [Predicción IA] -> [Ground Truth Real]`
   Guardado automático del mejor modelo histórico en `checkpoints/best_generator.pt`.

---

## 4. Nuevo Sistema: Control de Épocas y Puntos de Respawn (Snapshots cada 10 Épocas)

A propuesta del equipo de desarrollo, se implementó una red de seguridad contra sobreajuste o degradación tardía:

### 4.1. Guardado de Snapshots Periódicos
Cada 10 épocas completadas (10, 20, 30, 40, 50, ...), el sistema almacena en `checkpoints/snapshots/`:
- `checkpoint_epoch_XXX.pt`: Cerebro completo (generador, discriminador, optimizadores, pérdidas).
- `generator_epoch_XXX.pt`: Pesos limpios del generador listos para inferencia.
- `preview_epoch_XXX.png`: Captura visual en `training_samples/audit_history/` para inspeccionar la calidad en esa época.

### 4.2. Función de Respawn (Rollback / Vuelta en el Tiempo)
Si el usuario nota que en una época avanzada (ej. 45) el modelo empezó a perder nitidez en comparación con una época anterior (ej. 20 o 30):
- Abre el panel **⏪ Control de Respawn** en Sprite Studio.
- Selecciona el punto deseado en el desplegable.
- Pulsa **⏪ Ejecutar Respawn**: El sistema rebobina los pesos al snapshot seleccionado y reanuda el entrenamiento desde ese punto exacto, descartando las épocas deterioradas.

---

## 5. Manual Operativo de Sprite Studio (`sprite_studio.py`)

### 5.1. Puesta en Marcha
1. **Servidor Web:** Ejecutar `sprite_studio.py` con el entorno de Python de Forge:
   ```bash
   & "d:\escritorio\diseñador de pixel art\webui forger\system\python\python.exe" "d:\escritorio\diseñador de pixel art\sprite_studio.py"
   ```
2. **Acceso Web:** Abrir en el navegador:
   `http://localhost:8080`

### 5.2. Flujo de Trabajo Recomendado
1. **Entrenamiento:**
   - Ir a la pestaña **📈 Monitor de Entrenamiento**.
   - Seleccionar **Motor: PyTorch UNet (Generador Quirúrgico Local)**.
   - Pulsar **▶️ Iniciar**.
   - Auditar en vivo la columna 3 (*Predicción IA*) en la vista previa.
2. **Control de Calidad (QC):**
   - En la pestaña **Control de Calidad**, pulsar **Auditar Spritesheet**.
   - El sistema genera `spritesheet_clean.png` (transparente) y `spritesheet_grid.png` (con cuadrícula de alineación) y emite un informe métrico de alineación de cabeza, pies y saturación.

---

## 6. Políticas de Control de Versiones (Git y GitHub)

1. **Prohibición de Archivos Binarios > 50 MB:**
   - No comitear archivos `.pt` masivos (`supervised_cache_8x12.pt`, etc.).
   - Mantener `.gitignore` configurado para excluir caches binarias temporales.
2. **No usar Git LFS:** La cuota de GitHub LFS está al límite; los assets pixel art livianos se suben mediante Git estándar sin LFS.
