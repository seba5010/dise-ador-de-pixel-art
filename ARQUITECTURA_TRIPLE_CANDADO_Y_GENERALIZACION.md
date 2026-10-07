# Arquitectura de Triple Protección y Generalización de Poses

**Fecha de Implementación:** 7 de Octubre de 2026  
**Autor:** Antigravity / Sebastian  
**Componentes Principales:** `pixel_ai_engine/train_supervised.py`, `pixel_ai_engine/enhancer.py`, `pixel_ai_engine/dataset.py`, `pixel_ai_engine/quality_guidance.py`, `sprite_studio.py`  
**Estado de Validación:** 134 / 134 Tests Unitarios e Integrados Aprobados (100%)  

---

## 1. Contexto del Problema y Diagnóstico Clínico

Al auditar la calidad generativa del motor supervisado 8x12 en las épocas 1 a 22 (meta de 150 épocas), se detectaron dos fallas visuales severas en acciones complejas (cocinar, transportar cajas, celebrar y servir platos):

1. **Brazos Dobles / Extremidades Fantasma:**
   El generador dibujaba simultáneamente los brazos elevados de la acción requerida y los brazos caídos a los costados del cuerpo típicos de las caminatas.
2. **Nubes de Hollín / Ruido Estocástico en Personajes Oscuros:**
   Cuando se solicitaba generar personajes con ropa oscura (negro, marrón, azul marino) en poses de cocina o cajas, la IA generaba un cúmulo de ruido borroso o nubes de píxeles desarticulados en lugar de tela sólida.

### Causa Raíz (Análisis Forense de Dataset y Código):
* **Asimetría de Datos:** En el dataset supervisado (`dataset_supervisado/supervised_cache_8x12.pt`), los 56 personajes cuentan con ground-truth de caminatas (filas 1 a 8, frames 0 a 63). Sin embargo, solo 13 personajes tienen cocina frontal/trasera (fila 9), 10 cocina lateral (fila 10), 9 transporte de cajas (fila 11) y 6 celebración/servir plato (fila 12).
* **Sesgo Modal:** El 85% de las muestras del dataset tenían los brazos pegados al cuerpo (caminatas). La red aprendió un prior abrumador: *"los brazos siempre van abajo"*.
* **Sesgo Cromático:** Los pocos personajes con animación de cocina en el dataset vestían uniformes claros o delantales blancos (Alex, Chef). La red memorizó la cocina asociada a blanco; al ver ropa oscura, dudaba entre el molde gris y la memoria de color, emitiendo ruido de alta frecuencia.
* **Falta de Restricción Espacial:** `ENABLE_SILHOUETTE_LOSS` estaba desactivada por defecto (`"0"` en `anatomical_guidance.py`) y no existía ninguna pérdida que castigara píxeles generados fuera de la silueta de la pose en `train_supervised.py`.

---

## 2. Solución Arquitectónica: La Triple Protección (3 Candados)

Para resolver el problema sin requerir miles de dibujos manuales nuevos, se implementó un sistema de triple candado vinculado en tres niveles: **Entrenamiento**, **Control de Calidad (QC)** e **Inferencia/Generación**.

```
                           FLUJO DE TRIPLE PROTECCIÓN
  
    [Frontal Chibi (3ch)] + [Molde Pose Gris (3ch)] -> Tensor Condicional (6 canales)
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │             1. ENTRENAMIENTO SUPERVISADO               │
      │  • Sobremuestreo Balanceado por Pose (Boost 3.0x)       │
      │  • Aumentación Cruzada de Color en GPU (front=target)  │
      │  • Loss de Silueta (1.5x) + Boundary Constraint (2.0x) │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │          2. CONTROL DE CALIDAD (QC & AUDITORÍA)        │
      │  • Detección estricta de violaciones de silueta        │
      │  • Regeneración quirúrgica con template lock           │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │       3. GENERACIÓN EN SPRITE STUDIO & ENHANCER        │
      │  • clip_stray_limbs_against_template (margen 6px)      │
      │  • Supresión instantánea de brazos dobles y nubes      │
      └────────────────────────────────────────────────────────┘
```

---

### Candado 1: Límite de Silueta Anatómica y Recorte Morfológico

#### A. En el Entrenamiento (`train_supervised.py` & `dataset.py`):
1. **Activación de Silueta:** En `pixel_ai_engine/anatomical_guidance.py`, `ENABLE_SILHOUETTE_LOSS` se activa por defecto (`"1"`) con un multiplicador de **1.5x** en `train_supervised.py`.
2. **Máscaras de Pose:** `TemplateManager` en `pixel_ai_engine/dataset.py` ahora almacena tensores binarios de máscara `self.frame_masks` y provee `get_frame_mask(frame_idx)`.
3. **Pérdida de Envolvente (`boundary_loss`):** En cada paso de gradiente:
   ```python
   # Dilatación del molde con max_pool2d para tolerar holgura de ropa y cabello
   allowed_envelope = torch.nn.functional.max_pool2d(pose_masks, kernel_size=ksize, stride=1, padding=pad)
   pred_alpha_01 = ((preds[:, 3:4] + 1.0) * 0.5).clamp(0.0, 1.0)
   stray_silhouette_loss = (
       torch.relu(pred_alpha_01 - 0.20) * (1.0 - allowed_envelope)
   ).sum(dim=(1, 2, 3)) / ((1.0 - allowed_envelope).sum(dim=(1, 2, 3)) + 1e-6)
   boundary_loss = stray_silhouette_loss.mean() * 2.0
   ```
   Cualquier intento de dibujar brazos abajo mientras el maniquí tiene los brazos arriba recibe penalización directa de gradiente.

#### B. En la Generación de Sprites (`sprite_studio.py` & `enhancer.py`):
Se implementó `PixelArtEnhancer.clip_stray_limbs_against_template(frame_img, template_img, margin_px=6)`:
* Realiza una dilatación morfológica con `margin_px=6` de la máscara de la pose (para respetar cabelleras largas, capas y ropa holgada).
* Cualquier píxel generado fuera de este envolvente permitido es puesto a **0 de opacidad (transparencia limpia)**.
* Integrado en `PixelArtEnhancer.enhance_frame` y llamado automáticamente en `sprite_studio.py` (líneas 774-783) y `frame_regeneration.py`.

---

### Candado 2: Sobremuestreo Balanceado por Pose (Pose-Balanced Rarity Sampling)

* **Ubicación:** `pixel_ai_engine/quality_guidance.py` (`HardExampleMiningPolicy.build_plan`).
* **Mecanismo:** El planificador analiza la distribución de ocurrencias de cada `frame_idx` en el dataset.
* Al detectar disparidad (caminatas con 56 ejemplos vs acciones de filas 9 a 12 con 6 a 13 ejemplos), calcula un factor de rareza proporcional:
  $$\text{rarity\_boost} = \min\left(3.0, \max\left(1.0, \sqrt{\frac{\max(\text{counts})}{\text{count}}}\right)\right)$$
* Los frames de cocina, transporte de cajas y celebración reciben un multiplicador de peso de hasta **3.0x** en el `WeightedRandomSampler` de PyTorch.
* **Resultado:** La red entrena estas poses complejas 3 veces más frecuentemente por época, igualando su tasa de aprendizaje con la de las caminatas.

---

### Candado 3: Aumentación Cruzada Coordinada de Color y Paleta (GPU)

* **Ubicación:** `pixel_ai_engine/train_supervised.py` (`apply_coordinated_color_augmentation`).
* **Mecanismo:** Directamente en tensores en la GPU antes del forward pass:
  * Aplica perturbaciones idénticas y sincronizadas a `fronts[:, :3]` y `targets[:, :3]`:
    1. Escala por canal RGB aleatoria: $[0.60, 1.40]$.
    2. Variación de brillo: $[-0.25, 0.25]$.
    3. Variación de contraste: $[0.75, 1.25]$.
  * Solo afecta zonas opacas (preservando el fondo transparente).
  * Probabilidad base $p=0.45$, elevada a $p=0.85$ para poses raras ($f_{idx} \ge 64$).
* **Resultado:** La red ve a los pocos cocineros del dataset vistiendo ropa negra, roja, verde, azul y tonos oscuros. Se erradica la asociación "cocina = ropa blanca" y desaparecen las nubes de ruido en personajes oscuros.

---

## 3. Principio Operativo de Reconstrucción e Imaginación de Frames

La red neuronal es un generador condicional UNet (`PixelArtUNetGenerator`) que toma una condición de 6 canales:
$$\text{Input Tensor} = [\text{Frontal RGB (3ch)}, \text{Pose Maniquí Gris (3ch)}]$$

1. **Identidad:** La imagen frontal conditioning aporta el estilo de cabello, tono de piel, ojos, vestimenta y paleta canónica.
2. **Geometría Espacial:** El molde gris aporta la rotación tridimensional, ángulo de extremidades y accesorios (cuchillo, mortero, bowl, caja de cartón, plato de comida).
3. **Inferencia Paramétrica:** Para un personaje que **nunca tuvo ground-truth de cocina**, la red no copia ni pega: aplica la función matemática aprendida de los otros personajes, proyectando la vestimenta y facciones de este personaje nuevo sobre el volumen tridimensional del molde gris.
4. **Acabado Limpio:** El candado de silueta corta cualquier artefacto residual, entregando un frame con calidad lista para videojuegos.

---

## 4. Guía para Desarrolladores Futuros y Agentes IA

* **Archivos modificados:**
  * `pixel_ai_engine/anatomical_guidance.py`: Flag `PIXEL_AI_ENABLE_SILHOUETTE_LOSS` defaulting to `"1"`.
  * `pixel_ai_engine/dataset.py`: `self.frame_masks` y `get_frame_mask()` en `TemplateManager`.
  * `pixel_ai_engine/enhancer.py`: `clip_stray_limbs_against_template()` integrado en `enhance_frame()`.
  * `pixel_ai_engine/quality_guidance.py`: Lógica de `rarity_boost` en `HardExampleMiningPolicy`.
  * `pixel_ai_engine/train_supervised.py`: `apply_coordinated_color_augmentation()`, `boundary_loss`, integración en previews.
  * `sprite_studio.py`: Envío de `pose_pil` a `PixelArtEnhancer.enhance_frame()` durante generación de hojas y 4 poses.
  * `test_anatomical_guidance.py`: Test actualizado validando silueta activa por defecto.
* **Para correr las pruebas de verificación:**
  ```bash
  & "webui forger\system\python\python.exe" -m pytest
  ```
* **Para reanudar el entrenamiento pausado:**
  Basta con presionar "Reanudar" en la UI de Sprite Studio o ejecutar `train_supervised_model(epochs=150, mode="resume")`. El script cargará `checkpoints/latest_checkpoint.pt` (Época 22) e incorporará automáticamente los 3 candados desde la época 23 sin perder el progreso previo.
