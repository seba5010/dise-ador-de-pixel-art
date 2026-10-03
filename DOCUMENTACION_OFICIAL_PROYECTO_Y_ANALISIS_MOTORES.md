# DOCUMENTACIÓN OFICIAL DEL PROYECTO: DISEÑADOR DE PIXEL ART
**Villa del Chef — Motor Neuronal de Spritesheets y Análisis Comparativo de Motores**  
*Fecha: Octubre 2026 | Versión: 2.6 (Arquitectura Híbrida Supervisada, Optimización 8-Bit y Remapeo de Paleta)*

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

> **Conclusión sobre Forge:** El motor de difusión libre SD 1.5 + LoRA queda documentado oficialmente como **OBSOLETO Y NO APTO** para la generación de spritesheets rígidos en este proyecto.

---

## 3. Rescate Tecnológico: Métodos, Clases y Diseños Salvados de Forge para el Motor Supervisado

Aunque la difusión libre no sea apta, el entorno de Forge y su pila de visión artificial contienen **métodos, clases y algoritmos de alto rendimiento** que han sido rescatados e integrados directamente en nuestro motor:

### 3.1. Optimizador de 8-Bits (`bitsandbytes.optim.AdamW8bit`)
* **Origen:** Forge utiliza `bitsandbytes` para entrenar modelos pesados en GPUs de gama media sin desbordar la VRAM.
* **Integración en `train_supervised.py`:** En lugar del optimizador Adam tradicional de PyTorch (que mantiene momentos en Float32 consumiendo casi 3 GB de VRAM), se activó `AdamW8bit`.
* **Beneficio:** Reduce el consumo de VRAM a **~1.5 GB**, dejando holgura total para la GPU de 4 GB (RTX 3050 Ti) y acelerando la tasa de entrenamiento por época.

### 3.2. Módulo de Cuantización y Remapeo de Paleta (`pixel_ai_engine/palette_remap.py`)
* **Origen:** Algoritmos de indexación de color y color snapping usados en pipelines profesionales de Pixel Art.
* **Integración:**
  1. `extract_character_palette(front_img)`: Extrae los centroides RGB exactos de la piel, ojos, cabello y prendas del frontal del personaje.
  2. `remap_image_to_palette(frame, palette)`: Proyecta cada píxel generado por la U-Net al color canónico más cercano mediante distancia euclidiana mínima en espacio tridimensional, preservando el canal alfa.
* **Beneficio:** Erradica al 100% cualquier color "inventado" o gradiente difuminado. El personaje mantiene una fidelidad cromática idéntica a su ilustración de entrada.

### 3.3. Acondicionamiento Semántico de Poses (`frame_map.py`)
* **Origen:** Mapeo estructurado de 96 frames con atributos ontológicos: dirección (sur, este, etc.), subfase (paso 1, 2, idle) y acción (batir, cargar, celebrar).
* **Beneficio:** Permite estructurar la correlación entre la pose geométrica del maniquí y la orientación anatómica que la red neuronal debe aprender.

### 3.4. Preprocesadores de Siluetas y Líneas de ControlNet
* **Origen:** Filtros Canny / Lineart de la suite ControlNet de Forge.
* **Beneficio:** Automatizan la extracción de contornos de nuevos personajes con ropas o accesorios especiales para incorporarlos a la plantilla de poses.

---

## 4. El Motor Válido: Generador Supervisado Directo (Pix2Pix / UNet 6 Canales)

La solución técnica definitiva es el **Generador Supervisado Píxel a Píxel (`PixelArtUNetGenerator`)**.

### 4.1. Arquitectura de Entrada y Salida
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
   [REMAPEO DE PALETA] (pixel_ai_engine/palette_remap.py)
           ▼
[SPRITE FINAL 128×128] (Píxeles fieles, fondo alfa transparente estricto)
```

### 4.2. Los 4 Candados de Seguridad Implementados
1. **🔒 Candado 1: Cero Texto, Cero Alucinación:** Entrada directa de 6 canales numéricos (Identidad + Pose).
2. **🔒 Candado 2: Supremacía del Generador sobre el Discriminador:** Tasa de aprendizaje reducida (`lr * 0.5`) y peso adversarial de solo 5% (`adv_loss * 0.05`), frente a un 95% dominado por **Smooth L1 Color**, **Smooth L1 Alpha** y **Sobel Edge Loss** sobre 1.008 frames.
3. **🔒 Candado 3: Blindaje Numérico Anti-NaN:** Clamping estricto de logits a Float32 en rango `[-30.0, 30.0]`.
4. **🔒 Candado 4: Auditoría Visual en Vivo (4 Columnas) y Récord Histórico:** Tira comparativa en cada época y guardado de `best_generator.pt`.

---

## 5. Control de Épocas y Sistema de Respawn (Snapshots cada 10 Épocas)

* **Snapshots cada 10 Épocas:** En `checkpoints/snapshots/` se guardan automáticamente `checkpoint_epoch_XXX.pt`, `generator_epoch_XXX.pt` y `preview_epoch_XXX.png`.
* **Botón y Función de Respawn:** Desde el monitor web de Sprite Studio (`http://localhost:8080`), el usuario puede rebobinar el modelo a cualquier decena anterior si en el futuro se detecta sobreajuste o pérdida de nitidez.

---

## 6. Manual Operativo de Sprite Studio (`sprite_studio.py`)

1. **Iniciar Servidor:**
   ```bash
   & "d:\escritorio\diseñador de pixel art\webui forger\system\python\python.exe" "d:\escritorio\diseñador de pixel art\sprite_studio.py"
   ```
2. **Acceso Web:** `http://localhost:8080`
3. **Selección de Motor:** En el Monitor de Entrenamiento, seleccionar **PyTorch UNet (Generador Quirúrgico Local)**.
4. **Control de Calidad (QC):** En la pestaña de QC, auditar la alineación métrica y la transparencia de las hojas ensambladas.
