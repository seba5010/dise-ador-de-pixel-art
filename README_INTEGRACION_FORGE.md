# Guía de Integración y Uso: Stable Diffusion 1.5 + Forge + ControlNet (MOTOR B)

Este documento detalla la arquitectura, configuración y modo de uso del **Motor B**, un segundo pipeline de generación de spritesheets de pixel art que opera en paralelo al motor existente (**Motor A: U-Net PyTorch propia**).

---

## 1. Arquitectura del Sistema (Motor A vs Motor B)

El proyecto cuenta ahora con dos motores independientes para generación de personajes:

```
                                  [IMAGEN DE REFERENCIA FRONTAL]
                                 (ej. personajes/tori/tori_rnormal.png)
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     [MOTOR A: U-Net PyTorch Propia]                         [MOTOR B: SD 1.5 + Forge + ControlNet]
     - Script: generate_character_sheet.py                   - Script: forge_reference_pipeline.py
     - Pesos: checkpoints/best_generator.pt                  - Base: v1-5-pruned-emaonly.safetensors
     - Enfoque: Red neuronal ligera entrenada                - Guía: ControlNet (Lineart o Canny)
       directamente sobre el dataset del proyecto.           - Enfoque: Generación asistida frame a frame
     - Inferencia: Instantánea (2-5 segundos).                 vía API local HTTP de Forge.
```

> [!NOTE]
> **Integridad garantizada:** Ningún archivo del Motor A ha sido modificado, reentrenado ni borrado. Los checkpoints (`best_generator.pt`, `base_generator_16x4.pt`, `latest_checkpoint.pt`) y datasets originales permanecen intactos.

---

## 2. Modelos Integrados y Ubicaciones

La instalación activa y portable de Forge utilizada por el proyecto es:
`webui forger\webui\`

Los modelos se encuentran enlazados (utilizando enlaces duros NTFS para consumir **0 bytes adicionales** de disco) en:

| Componente | Archivo de Pesos | Ubicación en Forge | Estado |
| :--- | :--- | :--- | :--- |
| **SD 1.5 Base** | `v1-5-pruned-emaonly.safetensors` (~4.06 GB) | `webui forger\webui\models\Stable-diffusion\` | **[OK] Detectado** |
| **ControlNet Lineart** | `control_v11p_sd15_lineart.pth` (~1.38 GB) | `webui forger\webui\models\ControlNet\` | **[OK] Detectado** |
| **ControlNet Canny** | `control_v11p_sd15_canny.pth` (~1.38 GB) | `webui forger\webui\models\ControlNet\` | **[OK] Detectado** |

> [!WARNING]
> **Compatibilidad de modelos:** El archivo `pixel-art-xl-v1.1.safetensors` corresponde a la arquitectura SDXL y no debe mezclarse con este pipeline SD 1.5. Se mantiene intacto en la raíz del proyecto sin intervenir en este flujo.

---

## 3. Requisito Previo: Iniciar el Servidor de Forge

El nuevo pipeline interactúa con Forge a través de su **API HTTP local** (`http://127.0.0.1:7860`).

### Paso para iniciar Forge:
1. Haz doble clic en el lanzador:
   ```cmd
   webui forger\run.bat
   ```
2. Espera unos momentos hasta que en la ventana de consola aparezca:
   ```
   Running on local URL: http://127.0.0.1:7860
   ```
*(Nota: El archivo `webui forger\webui\webui-user.bat` ya fue configurado automáticamente con `--api` para habilitar todos los endpoints REST necesarios).*

---

## 4. Modos de Uso

### Opción 1: Lanzador Interactivo Windows (Recomendado)

Ejecuta con doble clic el nuevo script por lotes:
```cmd
run_generate_forge.bat
```
El asistente te solicitará de forma intuitiva:
1. **Nombre o ruta del personaje** (ejemplo: `tori`, `alex`, `conny`, `dana`, `amaro` o ruta directa a su PNG frontal).
2. **Formato:** `[1] 16x4` (64 frames) o `[2] 8x12` (96 frames canónicos).
3. **ControlNet:** `[1] Lineart` (recomendado) o `[2] Canny`.
4. **Modo:** `[1] TEST (4 poses rápidas)`, `[2] Completo` o `[3] Reanudar (--resume)`.

---

### Opción 2: Línea de Comandos (CLI Avanzado)

Puedes invocar `forge_reference_pipeline.py` directamente usando el intérprete Python de Forge:

```powershell
# 1. Comprobación rápida de modelos (sin generar)
& "webui forger\system\python\python.exe" forge_reference_pipeline.py --reference tori --check-only

# 2. Modo TEST (4 poses clave: frente, espalda, lateral y acción)
& "webui forger\system\python\python.exe" forge_reference_pipeline.py --reference tori --format 16x4 --control lineart --test

# 3. Generación completa de la hoja 16x4 (64 frames)
& "webui forger\system\python\python.exe" forge_reference_pipeline.py --reference tori --format 16x4 --control lineart

# 4. Generación completa de la plantilla canónica 8x12 (96 frames)
& "webui forger\system\python\python.exe" forge_reference_pipeline.py --reference tori --format 8x12 --control lineart

# 5. Reanudar una generación previa (omitiendo frames válidos ya creados)
& "webui forger\system\python\python.exe" forge_reference_pipeline.py --reference tori --format 16x4 --resume

# 6. Regenerar exclusivamente un único frame específico (ejemplo: frame 37)
& "webui forger\system\python\python.exe" forge_reference_pipeline.py --reference tori --format 16x4 --only-frame 37
```

---

## 5. Parámetros Disponibles

| Parámetro | Valor por Defecto | Descripción |
| :--- | :--- | :--- |
| `--reference`, `-r` | *(Requerido)* | Nombre o ruta a la imagen frontal de referencia del personaje. |
| `--format`, `-f` | `16x4` | Formato de salida: `16x4` (64 frames) u `8x12` (96 frames). |
| `--control`, `-c` | `lineart` | Tipo de ControlNet: `lineart` (mejor integración anatómica) o `canny` (bordes rígidos). |
| `--output`, `-o` | `None` (automático) | Ruta personalizada para el archivo PNG final. |
| `--test` | `False` | Genera únicamente 4 poses representativas y crea una imagen comparativa. |
| `--resume` | `False` | Salta los frames que ya existan y hayan pasado el control de calidad. |
| `--only-frame` | `None` | Regenera únicamente el frame con el índice indicado (ej: 37). |
| `--seed`, `-s` | `None` (fijo por json) | Semilla base. Si no se indica, se guarda/lee de `<personaje>_seed.json`. |
| `--seed-strategy` | `fixed` | `fixed` (misma semilla en todos los frames) u `offset` (`seed + frame_idx`). |
| `--denoise` | `0.50` | Fuerza de denoising en img2img (0.45 - 0.55 óptimo para preservar identidad). |
| `--cfg` | `7.0` | Escala de adherencia al prompt de pixel art. |
| `--steps` | `25` | Pasos de muestreo en Forge. |
| `--control-weight` | `0.90` | Influencia estructural del modelo ControlNet (0.80 - 1.00). |
| `--forge-url` | `http://127.0.0.1:7860` | Dirección del servidor local de Forge. |
| `--no-enhance` | `False` | Desactiva el post-procesado (paleta, despeckling, binarización alfa). |
| `--check-only` | `False` | Solo valida la presencia de modelos y sale. |

---

## 6. Organización de Archivos de Salida

Cada ejecución almacena los datos de forma determinista y estructurada en:

```
output/
  └── <nombre_personaje>/
      └── forge/
          ├── <nombre_personaje>_seed.json              # Semilla fija asignada al personaje
          ├── raw/                                      # Frames sin procesar devueltos por SD
          │   ├── frame_001.png
          │   └── ...
          ├── normalized/                               # Frames limpios, con transparencia y en su celda
          │   ├── frame_001.png
          │   └── ...
          ├── failed/                                   # Frames que no superaron el Quality Gate
          ├── previews/
          │   └── test_comparison_4poses_lineart.png    # Panel de validación de identidad (Modo TEST)
          ├── <nombre_personaje>_spritesheet_4x16_lineart.png
          └── <nombre_personaje>_spritesheet_4x16_lineart_vista_previa.png
```

---

## 7. Flujo Quirúrgico de Calidad y Pixel Art

1. **Generación Frame a Frame:** Python garantiza el orden, numeración matemática (64 o 96 frames) y posición canónica según los moldes de `dataset_frames_individuales/00_MOLDES_POSES/`.
2. **Extracción de Transparencia Reversible:** Se muestrean las esquinas exteriores y se aplica flood-fill desde los bordes para eliminar únicamente el fondo externo sin borrar píxeles idénticos dentro de la ropa o cuerpo del personaje.
3. **Postprocesado con `PixelArtEnhancer`:**
   - **Binarización alfa:** Elimina cualquier degradado suave o anti-aliasing residual.
   - **Despeckling:** Remueve píxeles flotantes huérfanos.
   - **Palette Snapping:** Ajusta los colores a la paleta canónica extraída de la foto frontal.
4. **Ensamblado Canónico:** Utiliza `place_in_cell` y `get_cell_coordinates` para anclar los pies al suelo de cada celda y construir el spritesheet transparente compatible directamente con Unity y Godot.

---

## 8. Calibración y Preservación de Identidad

- **ControlNet controla la POSE:** El molde de pose define exclusivamente la postura anatómica.
- **img2img controla la IDENTIDAD:**
  - Si la identidad se diluye (ropa o rostro cambian demasiado): **Bajar `--denoise` a `0.45` o `0.42`**.
  - Si el personaje no adopta bien la pose: **Subir `--control-weight` a `0.95` o subir `--denoise` a `0.55`**.
- **Consistencia temporal:** La semilla fija (`--seed-strategy fixed`) asegura que no existan mutaciones aleatorias de vestimenta entre frames contiguos.
