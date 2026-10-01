# INFORME TÉCNICO Y GUÍA DE COMPONENTES DEL PROYECTO
**Diseñador de Pixel Art - Motor Neuronal y Pipeline de Spritesheets**  
*Fecha: Octubre 2026*

---

## 📋 Resumen Ejecutivo

Este proyecto combina dos arquitecturas tecnológicas para la creación y animación de spritesheets en Pixel Art:
1. **Un motor neuronal propio (`pixel_ai_engine`)**: Diseñado para predecir cuadrículas completas de animación (16x4 y 8x12) a partir de una única imagen frontal de referencia.
2. **Un entorno portátil de IA generativa (`webui forger`)**: Basado en Stable Diffusion Forge con soporte para ControlNet y LoRA de Pixel Art.

Debido a que el proyecto contiene tanto código fuente liviano como instaladores y pesos neuronales masivos (que superan los **14 GB**), este informe detalla **para qué sirve cada archivo, cuáles deben subirse a GitHub y cuáles deben excluirse**.

---

## 1. Archivos Gigantes e Instaladores (⚠️ NO subir a GitHub)

Estos archivos superan el límite de **100 MB de GitHub** o representan software externo precompilado. Si intentas subirlos, GitHub bloqueará la subida inmediatamente.

| Archivo / Carpeta | Tamaño | ¿Para qué sirve? | Recomendación GitHub |
| :--- | :--- | :--- | :--- |
| `webui_forge_cu121_torch231.7z` | **1.87 GB** | Archivo comprimido con el instalador base portátil de Stable Diffusion WebUI Forge (con CUDA 12.1 y PyTorch 2.3.1 preinstalados). | ❌ **Excluir** (es un instalador comprimido redundante). |
| `webui forger/` | **~5 GB** (60.000+ archivos) | Carpeta con la instalación descomprimida de Python 3.10, scripts ejecutables, bibliotecas de PyTorch, CUDA y el servidor web de Forge. Es el entorno de ejecución que alimenta los scripts. | ❌ **Excluir** (los entornos de software no se suben a Git; se documentan en un `requirements.txt`). |
| `control_v11p_sd15_canny.pth` | **1.44 GB** | Modelo neuronal de ControlNet (Canny Edge) para Stable Diffusion 1.5. Permite guiar la generación de imágenes siguiendo las líneas de contorno de las plantillas. | ❌ **Excluir** (se descarga externamente desde HuggingFace). |
| `pixel-art-xl-v1.1.safetensors` | **170 MB** | Modelo adaptador (LoRA) para Stable Diffusion XL especializado en estilizar imágenes con estética pixel art nítida. | ❌ **Excluir** (se descarga externamente desde Civitai o HuggingFace). |
| `stable-diffusion-webui-forge-main.zip` | **20 MB** | Código fuente comprimido de la interfaz web de Forge. | ❌ **Excluir** (es un archivo comprimido temporal). |

---

## 2. Checkpoints y Caché Neuronal (`checkpoints/`)

Archivos generados por el ciclo de entrenamiento (`train.py` y `train_dos_fases_1500.py`).

| Archivo | Tamaño | ¿Para qué sirve? | Recomendación GitHub |
| :--- | :--- | :--- | :--- |
| `dataset_cache_256_8x12.pt` | **710 MB** | Caché binaria pre-procesada de todos los tensores de imagen para la Fase 2 (8x12). Acelera el entrenamiento para no tener que leer del disco cada PNG. | ❌ **Excluir** (se regenera automáticamente si no existe). |
| `dataset_cache_256_16x4.pt` | **407 MB** | Caché binaria pre-procesada para la Fase 1 (16x4). | ❌ **Excluir** (se regenera automáticamente). |
| `latest_checkpoint.pt` | **222 MB** | Guarda el estado completo de entrenamiento: generador, discriminador, optimizadores y programadores de tasa de aprendizaje. Permite reanudar tras caídas o apagados. | ❌ **Excluir** (excede 100 MB). |
| `latest_checkpoint_16x4.pt` | **222 MB** | Último checkpoint guardado de la Fase 1. | ❌ **Excluir** (excede 100 MB). |
| `best_generator.pt` | **63 MB** | Contiene **únicamente los pesos del generador** que alcanzaron la menor pérdida histórica. Es el archivo que usa `generate_character_sheet.py` para la inferencia. | ⚠️ **Opcional** (pesa menos de 100 MB, pero se recomienda compartir modelos vía GitHub Releases o Google Drive). |
| `base_generator_16x4.pt` | **63 MB** | Pesos entrenados de la Fase 1, usados como punto de partida (Transfer Learning) hacia la Fase 2. | ⚠️ **Opcional** (pesa menos de 100 MB). |

---

## 3. El Paquete Central: `pixel_ai_engine/` (✅ SÍ subir a GitHub)

El núcleo del motor neuronal. Es 100% código Python limpio y liviano.

* **`config.py`**: Configuración central del sistema. Define rutas, dimensiones de lienzo (724x2172 en Fase 1, 1024x1536 en Fase 2), resolución interna de trabajo (256x256), hiperparámetros y dispositivos CUDA.
* **`models.py`**: Define la arquitectura de redes neuronales:
  * `PixelArtUNetGenerator`: Red tipo U-Net profunda con conexiones residuales (*skip-connections*), normalización de instancias y bloques convolucionales. Toma 6 canales (3 de identidad frontal + 3 de pose de plantilla) y predice 4 canales (RGBA).
  * `PatchGANDiscriminator`: Discriminador 70x70 que evalúa si los parches generados son pixel art auténtico o ruido.
* **`dataset.py`**: Gestión de datos y pre-procesamiento geométrico. Contiene funciones críticas como `isolate_character()`, `pad_to_square()`, `place_in_cell()` y la clase `TemplateManager` para recortar y mapear celdas de las plantillas.
* **`train.py`**: Motor de entrenamiento con control adaptativo de temperatura (monitoreo de GPU vía `nvidia-smi` y CPU vía WMI), micro-pausas anti-sobrecalentamiento, precisión mixta (AMP) y cálculo de pérdidas compuestas (L1 + Adversarial + Gradientes de borde).
* **`generate_character_sheet.py`**: Módulo de inferencia autónoma. Carga un PNG frontal de entrada, proyecta los 96 frames y ensambla el spritesheet final transparente.
* **`quality_gate.py`**: Auditor quirúrgico de calidad. Analiza fidelidad de paleta, alineación con el molde, nitidez de micro-textura y pureza del canal alfa, decidiendo si se aprueba o se envía a refuerzo.
* **`enhancer.py`**: Algoritmos de post-procesamiento clásico: binarización de alfa, eliminación de píxeles huérfanos y cuantización de color.
* **`phase3_critical_enhancer.py`**: Capa de elevación de diseño de Fase 3. Aplica sombreado dinámico, refuerzo de contornos de 1px y snapping cromático.

---

## 4. Scripts de Control y Ejecución en la Raíz (✅ SÍ subir a GitHub)

* **`train_dos_fases_1500.py`**: Orquestador del pipeline completo de entrenamiento. Ejecuta la Fase 1 (aprendizaje base), transfiere pesos a la Fase 2 (96 poses), audita con QualityGate y ejecuta la Fase 3.
* **`fase3_revision_critica.py`**: Herramienta CLI independiente para tomar cualquier spritesheet generado y aplicarle la elevación de contornos y micro-sombras.
* **`pipeline_hibrido_tori.py`**: Script de proyección geométrica y recorte de anatomía usado originalmente para vestir el maniquí con la ropa de Tori.
* **`audit_quality_8x12.py` / `audit_frames_8x12.py`**: Scripts para inspeccionar individualmente la resolución y consistencia de los 96 frames generados.
* **`normalize_spritesheets.py` / `normalize_8x12_cv2_mold.py`**: Scripts de normalización que centran y redimensionan spritesheets existentes para que calcen con la grilla de celdas estándar.
* **`extract_individual_frames.py`**: Descompone una hoja de spritesheet completa en carpetas con sus 96 frames sueltos individuales.
* **`check_dims.py` / `check_syntax.py`**: Utilidades rápidas de verificación de archivos y sintaxis.

---

## 5. Lanzadores por Lotes (`.bat`) (✅ SÍ subir a GitHub)

Scripts de Windows para ejecutar tareas con doble clic:
* **`run_generate.bat`**: Genera la hoja de sprites de un personaje con un solo comando.
* **`run_monitor.bat`**: Inicia el servidor de monitoreo web local.
* **`run_train_dos_fases_1500.bat`**: Lanza el pipeline completo de entrenamiento.
* **`run_train_fase1_16x4.bat` / `run_train_fase2_8x12.bat`**: Entrenamientos individuales por fase.
* **`run_train_infinite.bat`**: Modo de entrenamiento continuo sin límite de épocas.

---

## 6. Servidor y Monitor Web Visual (✅ SÍ subir a GitHub)

* **`monitor.html`**: Panel de control frontend moderno y responsivo. Muestra gráficas de pérdida, barras de calidad, lecturas de temperatura en vivo (GPU/CPU) y previsualización de la última época entrenada.
* **`monitor_server.py`**: Servidor HTTP en Python que lee `training_status.json` y sirve los datos al panel web en `http://localhost:8000`.
* **`training_status.json`**: Archivo de telemetría actualizado en tiempo real por `train.py`.

---

## 7. Plantillas, Diseños y Datasets (✅ SÍ subir a GitHub con prudencia)

* **`plantilla de los spritesheets.png`** (8x12, 1024x1536): El maniquí canónico de 96 poses con proporciones humanas exactas.
* **`plantilla_16x4.png`** (16x4, 724x2172): Plantilla de la Fase 1 con 64 poses.
* **`marco_8x12.png` / `marco_16x4.png`**: Guías visuales transparentes de las celdas para validar que ningún sprite se desborde.
* **`PLANTILLAS_CONTROLNET/`**: Versiones de las plantillas procesadas en alto contraste para usarse con modelos de ControlNet Canny.
* **`personajes/`**: Imágenes frontales originales de referencia (Tori, Alex, Amaro, Conny, Dana, Mauricio, etc.).
* **`dataset_frames_individuales/`**: Base de datos de entrenamiento organizada por personaje y tipo de ropa con sus frames desglosados.
* **`output/`**: Carpeta donde se depositan los spritesheets finales generados.

---

## 8. Informes y Bitácoras Técnicas (`.md`) (✅ SÍ subir a GitHub)

* **`BITACORA_TECNICA_APRENDIZAJE_Y_COLAPSO.md`**: Registro exhaustivo de los experimentos de entrenamiento, análisis de colapsos de gradiente y soluciones implementadas.
* **`DIAGNOSTICO_AUDITORIA_FASE1.md` / `DIAGNOSTICO_AUDITORIA_FASE2.md`**: Reportes clínicos generados automáticamente por el QualityGate evaluando fidelidad de color, alineación y pureza alfa.
* **`DATASET_REORGANIZATION_STATUS.md` / `SESSION_2026-09-26_8x12.md`**: Historial de versiones y evolución de la estructura de datos.

---

## 9. Archivo `.gitignore` Recomendado

Para que tu repositorio en GitHub quede impecable, profesional y suba en segundos sin errores de archivos gigantes, este es el contenido que debe tener tu archivo `.gitignore`:

```gitignore
# Archivos comprimidos y pesados (> 100MB)
*.7z
*.zip
*.rar
*.tar.gz

# Modelos pesados externos (SD, ControlNet, LoRA)
*.pth
*.safetensors
*.bin
*.onnx

# Entorno de ejecución portable
webui forger/
system/
webui_forge_cu121_torch231/

# Checkpoints y cachés de entrenamiento masivos
checkpoints/*.pt
!checkpoints/best_generator.pt
training_samples/preview_epoch_*.png
training_samples/latest_*.png

# Python y sistema
__pycache__/
*.pyc
*.pyo
.vscode/
.idea/
*.log
```

---
*Documento preparado por Antigravity IDE para el proyecto de diseño de spritesheets.*
