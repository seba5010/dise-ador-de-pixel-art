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

## 4. Flujo de Verificación Previa Recomendada

Para asegurar una ejecución limpia sin desperdiciar tiempo ni recursos en generaciones completas defectuosas, se recomienda seguir este flujo secuencial:

```
[1. Iniciar Forge] ──► [2. --check-only] ──► [3. Verificar SD 1.5] ──► [4. Ejecutar TEST] ──► [5. Revisar Comparativa] ──► [6. Generación Completa]
   webui forger\run.bat     Validar modelos         Asegurar checkpoint       4 poses clave             Validar identidad           64 o 96 frames
```

1. **Iniciar Forge:**
   Ejecutar `webui forger\run.bat` y esperar a que muestre `Running on local URL: http://127.0.0.1:7860`.
2. **Ejecutar `--check-only`:**
   Verificar que los archivos base (SD 1.5, ControlNet) estén en disco y que Forge responda en la API.
   ```powershell
   & "webui forger\system\python\python.exe" forge_reference_pipeline.py --reference tori --control lineart --check-only
   ```
3. **Verificar que SD 1.5 esté activo:**
   El pipeline comprobará automáticamente si `v1-5-pruned-emaonly` está cargado en memoria de GPU; de no ser así, solicitará el cambio por API antes de empezar.
4. **Ejecutar Modo TEST:**
   Generar rápidamente únicamente 4 poses clave para evaluar la adaptación visual:
   - Formato `16x4`: Frames 1 (Frente), 17 (Espalda), 33 (Derecha), 49 (Izquierda).
   - Formato `8x12`: Frames 1 (Frente), 33 (Espalda), 17 (Lateral), 65 (Cocina).
   ```powershell
   & "webui forger\system\python\python.exe" forge_reference_pipeline.py --reference tori --format 16x4 --control lineart --test
   ```
5. **Revisar Comparación:**
   Inspeccionar el panel generado en `output/<personaje>/forge/previews/test_comparison_4poses_<control>.png`.
6. **Ejecutar Generación Completa:**
   Una vez confirmada la coherencia de las 4 poses, proceder a la generación de todos los frames:
   ```powershell
   & "webui forger\system\python\python.exe" forge_reference_pipeline.py --reference tori --format 16x4 --control lineart
   ```

---

## 5. Modos de Uso

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

También soporta ejecución directa por argumentos con valores por defecto automáticos (`16x4` y `lineart`):
```cmd
run_generate_forge.bat tori
run_generate_forge.bat tori 8x12 canny
run_generate_forge.bat tori 16x4 lineart --test
```

---

### Opción 2: Línea de Comandos (CLI Avanzado)

Puedes invocar `forge_reference_pipeline.py` directamente usando el intérprete Python de Forge:

```powershell
# 1. Comprobación rápida de modelos (sin generar)
& "webui forger\system\python\python.exe" forge_reference_pipeline.py --reference tori --check-only

# 2. Modo TEST (4 poses clave: frente, espalda, lateral/derecha y acción/izquierda)
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

## 6. Parámetros Disponibles

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

## 7. Organización de Archivos de Salida

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
          ├── <nombre_personaje>_spritesheet_16x4_lineart.png
          └── <nombre_personaje>_spritesheet_16x4_lineart_vista_previa.png
```

---

## 8. Flujo Quirúrgico de Calidad y Clasificación (Quality Gate)

1. **Validación Numérica de Moldes:** Se extrae el índice entero de cada molde (`pose_001` -> 1) garantizando la presencia estricta de 1..64 (para `16x4`) o 1..96 (para `8x12`) sin saltos.
2. **Quality Gate Estricto:** Cada frame generado se evalúa con tres estados:
   - **`PASS`:** Sprite con canal alfa limpio, dimensiones coherentes y bounding box anatómico válido.
   - **`WARNING`:** Sprite utilizable pero con alertas leves (ej: contacto marginal con el borde).
   - **`FAIL`:** Frame con canal alfa vacío, imagen corrupta o dimensiones colapsadas. Se guarda en `failed/` y **no se incorpora a la hoja final**.
3. **Resumen de Integridad:** Si algún frame falla durante el proceso, el spritesheet se marca explícitamente como **`INCOMPLETO`** y se reportan los números exactos de frames a regenerar con `--resume` o `--only-frame <N>`.

---

## 9. Calibración y Preservación de Identidad

- **ControlNet controla principalmente estructura y pose:** El molde guía la silueta y líneas del cuerpo, pero no inyecta textura ni color del personaje.
- **img2img ayuda a conservar identidad:** Transfiere los rasgos, paleta y vestimenta desde la imagen de referencia.
  - Si la identidad se diluye (ropa o rostro cambian demasiado): **Bajar `--denoise` a `0.45` o `0.42`**.
  - Si el personaje no adopta bien la pose: **Subir `--control-weight` a `0.95` o subir `--denoise` a `0.55`**.
- **Consistencia de semilla:** La semilla fija (`--seed-strategy fixed`) **ayuda a reducir variación aleatoria, pero no garantiza identidad idéntica entre poses**. Ni ControlNet ni una semilla fija garantizan identidad perfecta; por ello, el control de calidad visual en el Modo TEST es mandatorio antes de ensamblar hojas completas.

---

## 10. Informe Técnico sobre Geometría 16x4 vs 8x12

### Análisis Matemático de Formatos 16x4 Históricos y Actual:

Ninguno de los formatos históricos o heredados de 16x4 posee celdas uniformes en ambas dimensiones; todos requieren `get_cell_coordinates()` o delimitación por proporciones:

1. **Formato Histórico 341 x 1024:**
   - Columnas: $341 / 4 = 85.25\text{ px}$ (fraccional).
   - Filas: $1024 / 16 = 64.0\text{ px}$ (entera).
   - *Resultado:* Columnas fraccionarias. Requiere interpolación proporcional.

2. **Formato Histórico 682 x 2048 (en plantillas manuales y movimientos originales):**
   - Columnas: $682 / 4 = 170.5\text{ px}$ (fraccional).
   - Filas: $2048 / 16 = 128.0\text{ px}$ (entera).
   - *Resultado:* Columnas fraccionarias. **NO es un formato de celdas uniformes**, ya que horizontalmente cada celda requiere $170.5\text{ px}$.

3. **Formato Actual 724 x 2172 (estándar heredado de `normalize_spritesheets.py`):**
   - Columnas: $724 / 4 = 181.0\text{ px}$ (entera).
   - Filas: $2172 / 16 = 135.75\text{ px}$ (fraccional).
   - *Resultado:* Filas fraccionarias.
   - *Estado en el Pipeline:* **Se mantiene estrictamente `724 x 2172` en `GRID_CONFIGS["16x4"]`** para conservar compatibilidad absoluta con todos los spritesheets existentes del proyecto, sin redimensionar assets, moldes ni scripts preexistentes.

### Ejemplos Matemáticamente Válidos para Celdas Enteras en 16x4 (Referencia Técnica):

Si en fases futuras del proyecto se decidiera migrar a un lienzo de 16x4 con celdas de tamaño entero estricto para Unity/Godot, las opciones matemáticamente válidas son:

- **Opción A (680 x 2048):**
  - Columnas: $680 / 4 = 170\text{ px}$
  - Filas: $2048 / 16 = 128\text{ px}$
  - Tamaño de Celda uniforme: $170 \times 128\text{ px}$.
- **Opción B (724 x 2176) — La más cercana al estándar actual:**
  - Columnas: $724 / 4 = 181\text{ px}$
  - Filas: $2176 / 16 = 136\text{ px}$
  - Tamaño de Celda uniforme: $181 \times 136\text{ px}$.
  - *(Solo requiere añadir 4 píxeles de altura respecto al estándar actual de 724x2172).*
- **Opción C (512 x 2048):**
  - Columnas: $512 / 4 = 128\text{ px}$
  - Filas: $2048 / 16 = 128\text{ px}$
  - Tamaño de Celda uniforme cuadrada: $128 \times 128\text{ px}$.

*(Nota: Estas opciones son solo informativas. No se migran assets ni se alteran las grillas actuales).*

---

### Formato Canónico 8x12 (`1024 x 1536`):

- **Cálculo de Celdas:**
  - Columnas: $1024 / 8 = 128\text{ px}$ exactos.
  - Filas: $1536 / 12 = 128\text{ px}$ exactos.
- **Ventaja de Integración:** Celdas perfectamente cuadradas y uniformes de **$128 \times 128\text{ px}$**. Es el formato estándar recomendado para corte automático inmediato en Unity Sprite Editor (*Grid by Cell Size: 128x128*) y Godot Engine sin desfases de píxel.
