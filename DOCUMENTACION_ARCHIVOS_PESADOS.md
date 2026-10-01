# Documentación Técnica: Archivos Pesados (> 100 MB) y su Función en el Sistema

Este documento describe detalladamente los archivos que superan los **100 MB** en el proyecto **Diseñador de Pixel Art** (excluidos del repositorio de Git debido a las restricciones de tamaño de GitHub) y la función crítica que desempeña cada uno dentro de la arquitectura de entrenamiento y generación.

---

## 1. Resumen de Archivos Excluidos (> 100 MB)

| Archivo / Ruta | Tamaño aprox. | Categoría | Función Principal |
| :--- | :--- | :--- | :--- |
| `control_v11p_sd15_canny.pth` | **1.38 GB** | Modelo de Visión (ControlNet) | Guía espacial de bordes/líneas para Stable Diffusion v1.5 |
| `pixel-art-xl-v1.1.safetensors` | **162.6 MB** | Adaptador LoRA | Especialización estilística en Pixel Art para SDXL |
| `webui_forge_cu121_torch231.7z` | **1.74 GB** | Entorno Empaquetado | Instalador / backup portable con CUDA 12.1 y PyTorch 2.3.1 |
| `checkpoints/dataset_cache_256_16x4.pt` | **388.6 MB** | Caché de Tensores | Tensores preprocesados en memoria para resolución 256x256 (rejilla 16x4) |
| `checkpoints/dataset_cache_256_8x12.pt` | **677.5 MB** | Caché de Tensores | Tensores preprocesados completos para rejilla 8x12 (96 animaciones) |
| `checkpoints/latest_checkpoint.pt` | **212.2 MB** | Checkpoint de Entrenamiento | Estado completo de entrenamiento (Pesos + Optimizador Adam + Época actual) |
| `checkpoints/latest_checkpoint_16x4.pt` | **212.2 MB** | Checkpoint de Entrenamiento | Estado completo del optimizador y modelo en fase 16x4 |
| `webui forger/system/python/...` (`*.dll`) | **~2.8 GB** | Binarios de Aceleración CUDA | Librerías nativas de Nvidia (cuBLAS, cuDNN, cuFFT, Torch C++) |

---

## 2. Descripción Detallada por Componente

### A. Modelos de Difusión y Control Estructural

#### 1. `control_v11p_sd15_canny.pth` (1.38 GB)
* **¿Qué es?**: Modelo neuronal de la arquitectura **ControlNet v1.1** entrenado para detección de bordes (Canny Edge Detection).
* **Función en el programa**:
  * Funciona como el "esqueleto" que controla la generación de sprites.
  * Permite extraer los contornos de los sprites base (como Alex, Amaro, Connie) para asegurar que cualquier nueva animación o variación de ropa respete exactamente la silueta, postura y anatomía de 32x32 / 48x48 píxeles sin deformaciones.

#### 2. `pixel-art-xl-v1.1.safetensors` (162.6 MB)
* **¿Qué es?**: Un adaptador de bajo rango (**LoRA**) entrenado específicamente sobre imágenes pixel art nítidas.
* **Función en el programa**:
  * Aplica la identidad visual de pixel art retro sobre las generaciones.
  * Elimina el desenfoque suave típico de los modelos de difusión estándar, forzando paletas limitadas, clusters de píxeles sólidos y estética de videojuegos de 16-bit.

---

### B. Cachés de Entrenamiento de Alto Rendimiento

#### 3. `checkpoints/dataset_cache_256_16x4.pt` (388.6 MB) y `dataset_cache_256_8x12.pt` (677.5 MB)
* **¿Qué son?**: Archivos serializados en formato binario de PyTorch conteniendo miles de frames procesados.
* **Función en el programa**:
  * Durante el entrenamiento de la red generativa (`train.py` y `train_dos_fases_1500.py`), cargar imágenes sueltas del disco duro en cada época genera un cuello de botella severo de I/O (lectura de disco).
  * Estos cachés contienen todas las imágenes normalizadas, convertidas a tensores float normalizados `[-1, 1]`, con máscaras alfa procesadas y empaquetadas directamente para transferirse a la VRAM de la GPU a velocidad máxima.

---

### C. Estados de Optimización y Checkpoints

#### 4. `checkpoints/latest_checkpoint.pt` y `latest_checkpoint_16x4.pt` (212.2 MB c/u)
* **¿Qué son?**: Puntos de control completos del ciclo de entrenamiento de redes neuronales.
* **Diferencia con los modelos ligeros guardados en Git (`best_generator.pt` / `base_generator_16x4.pt` ~63 MB)**:
  * Los generadores en Git solo contienen los pesos de inferencia (`state_dict` del modelo G).
  * Los checkpoints pesados (`latest_checkpoint.pt`) contienen:
    1. Pesos del Generador.
    2. Pesos del Discriminador.
    3. Momentos y estado del optimizador **Adam** (matrices de velocidad de gradiente de primer y segundo orden).
    4. El número de época y los historiales de funciones de pérdida.
* **Función**: Permiten pausar y reanudar el entrenamiento en cualquier momento exacto sin perder el progreso del descenso de gradiente.

---

### D. Runtime y Librerías CUDA Portables

#### 5. `webui_forge_cu121_torch231.7z` y carpeta `webui forger/` (~3.5 GB)
* **¿Qué son?**: Distribución embebida de Python 3.10 con soporte acelerado para GPU Nvidia.
* **Componentes clave**:
  * `torch_cuda.dll` (835 MB): Núcleo de cálculo tensorial en GPU.
  * `cublasLt64_12.dll` (513 MB): Acelerador de multiplicación de matrices densas de Nvidia.
  * `cudnn_cnn_infer64_8.dll` (555 MB): Biblioteca de redes neuronales profundas para convoluciones optimizadas.
* **Función**: Proveer el entorno de ejecución autónomo (sin necesidad de configurar Python ni drivers globales en el sistema) para ejecutar Stable Diffusion Forge y la suite de generación en local.

---

## 3. Estado de Archivos Ligeros Subidos a GitHub

Todos los archivos de código fuente, herramientas de monitoreo y los modelos de inferencia menores a 100 MB se encuentran salvaguardados en el repositorio:

- Modelos de inferencia:
  - `checkpoints/best_generator.pt` (63 MB): El mejor generador guardado listo para producción.
  - `checkpoints/base_generator_16x4.pt` (63 MB): El generador base calibrado para 16x4.
- Scripts de pipeline, normalización, entrenamiento y utilidades (`*.py`, `*.bat`).
- Herramientas de visualización en tiempo real (`monitor.html`, `monitor_server.py`).
- Plantillas y mapas de rejilla canónicos (`frame_map_8x12.json`, `plantilla_16x4.png`, etc.).
