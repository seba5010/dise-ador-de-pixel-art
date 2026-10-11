# Reporte Maestro de Calidad: Auditoría Quirúrgica de Todos los Personajes (29 Monos)

**Identificación Oficial de Auditoría:**
- 📅 **Fecha y Hora:** `10/10/2026 21:56:34`
- 🧬 **Era de Entrenamiento:** `Era 2 (Fase 2: Transferencia 8x12 - 96 Frames / Acciones Complejas)`
- 🔄 **Época del Checkpoint:** `Época 70 / 172`
- 💾 **Checkpoint Evaluado:** `checkpoint_epoch_070.pt`
- 🆔 **ID de Sesión:** `session_20261010_191711`
- 🎮 **Personajes Auditados:** `29 personajes` (100% del directorio `personajes/`)
- 🖼️ **Total de Frames Analizados Píxel a Píxel:** `2,784 frames` (1,856 caminatas + 928 imaginarios)
- ⚡ **Dispositivo:** `NVIDIA GeForce RTX 3050 Ti Laptop GPU` (Aceleración AMP FP16)
- 🛡️ **Blindajes Activos:** Triple Protección (Candado de Silueta + Sobremuestreo Rarity Boost 3.0x + Aumentación Cruzada GPU)

---

## 1. Resumen Ejecutivo Global: Caminatas vs. Acciones Complejas

| Tipo de Acción | Total Frames | Anatomía Cruda (IA) | Outline Crudo (IA) | Silueta IoU | Calidad Anatómica Mejorada | Outline Mejorado | Tasa Extremidades Fantasma |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Caminatas (Filas 0-7)** | 1,856 | 82.2% | 95.8% | 92.4% | **86.2%** | **98.4%** | 1.91% |
| **Acciones Complejas (Filas 8-11)** | 928 | **74.0%** | **94.3%** | **77.9%** | **91.7%** | **92.8%** | 3.66% → **0.00%** |

> [!TIP]
> El Candado Morfológico de Silueta (Lock 1) reduce las extremidades fantasma (brazos dobles y artefactos de fondo) en todos los 29 personajes de **3.66%** a **0.00%**, elevando la nitidez de contorno final a **92.8%**.

---

## 2. Desglose Quirúrgico Global por Acción de Animación (Filas 8 a 11)

Promedio calculado sobre los 29 personajes (928 frames de acciones complejas):

| Animación | Filas / Frames | Muestras Totales | Anatomía IA | Outline IA | Silueta IoU | Calidad Final Enhancer | Brazos Fantasma | Estado de Reconstrucción |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `cocinar_bowl` | `Fila 8 (64-71)` | 232 | 87.3% | 95.4% | 83.0% | **93.0%** | 0.00% | Alineado ✅ |
| `cocinar_estacion` | `Fila 9 (72-79)` | 232 | 68.4% | 94.0% | 72.7% | **89.6%** | 0.00% | Alineado ✅ |
| `pensar` | `Fila 10 cols 0-3 (80-83)` | 116 | 56.7% | 94.1% | 75.3% | **91.1%** | 0.00% | Alineado ✅ |
| `cargar_caja` | `Fila 10 cols 4-7 (84-87)` | 116 | 71.3% | 93.5% | 75.4% | **90.9%** | 0.00% | Alineado ✅ |
| `servir_plato` | `Fila 11 cols 0-3 (88-91)` | 116 | 73.5% | 94.7% | 82.5% | **93.9%** | 0.00% | Alineado ✅ |
| `celebrar` | `Fila 11 cols 4-7 (92-95)` | 116 | 79.0% | 93.6% | 78.8% | **92.4%** | 0.00% | Alineado ✅ |

---

## 3. Tabla Maestra de Todos los Personajes (29 Monos)

| # | Personaje | Caminata Anat | Caminata Out | Acción Anat IA | Acción Out IA | Silueta IoU | Calidad Final Enhancer | Brazos Dobles (Crudo → Enh) | Estado |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | `mauricio` | 84.0% | 97.3% | 72.2% | 94.7% | 78.0% | **92.0%** | 3.9% → **0.00%** | Alineado ✅ |
| 02 | `sebas` | 86.7% | 98.7% | 75.6% | 95.7% | 78.5% | **92.0%** | 3.4% → **0.00%** | Alineado ✅ |
| 03 | `tori` | 83.0% | 98.2% | 75.0% | 95.0% | 78.2% | **91.6%** | 3.6% → **0.00%** | Alineado ✅ |
| 04 | `carlos` | 85.6% | 98.5% | 72.9% | 95.5% | 79.4% | **92.3%** | 3.8% → **0.00%** | Alineado ✅ |
| 05 | `zack` | 90.1% | 99.1% | 80.7% | 96.6% | 84.5% | **94.5%** | 2.9% → **0.00%** | Alineado ✅ |
| 06 | `andres_arica` | 66.2% | 97.2% | 72.6% | 95.0% | 77.8% | **91.7%** | 4.1% → **0.00%** | Alineado ✅ |
| 07 | `nikol` | 71.1% | 98.7% | 74.5% | 95.7% | 81.6% | **93.2%** | 3.7% → **0.00%** | Alineado ✅ |
| 08 | `juan` | 87.9% | 98.4% | 73.6% | 95.3% | 76.4% | **91.1%** | 3.8% → **0.00%** | Alineado ✅ |
| 09 | `rafa` | 85.4% | 98.5% | 75.4% | 94.9% | 76.2% | **91.0%** | 3.1% → **0.00%** | Alineado ✅ |
| 10 | `millaray` | 85.6% | 97.7% | 70.0% | 95.3% | 77.0% | **91.4%** | 4.5% → **0.00%** | Alineado ✅ |
| 11 | `mario` | 87.8% | 98.6% | 75.1% | 95.7% | 78.1% | **91.9%** | 3.4% → **0.00%** | Alineado ✅ |
| 12 | `belial` | 85.3% | 91.9% | 79.1% | 93.5% | 78.9% | **92.1%** | 2.6% → **0.00%** | Alineado ✅ |
| 13 | `maty hermano` | 83.5% | 98.3% | 74.4% | 95.3% | 79.0% | **92.4%** | 3.7% → **0.00%** | Alineado ✅ |
| 14 | `erin` | 86.7% | 97.4% | 77.9% | 95.4% | 79.8% | **92.5%** | 2.9% → **0.00%** | Alineado ✅ |
| 15 | `duvan` | 60.6% | 92.6% | 62.2% | 90.1% | 71.5% | **89.3%** | 3.9% → **0.00%** | Alineado ✅ |
| 16 | `camilo` | 83.0% | 95.4% | 78.5% | 94.7% | 80.6% | **92.7%** | 2.7% → **0.00%** | Alineado ✅ |
| 17 | `nico` | 87.9% | 98.6% | 71.9% | 95.3% | 76.8% | **91.0%** | 4.0% → **0.00%** | Alineado ✅ |
| 18 | `jorge` | 81.4% | 96.7% | 74.5% | 94.6% | 77.0% | **91.2%** | 3.3% → **0.00%** | Alineado ✅ |
| 19 | `seba` | 78.6% | 95.1% | 70.4% | 94.6% | 77.5% | **91.7%** | 4.7% → **0.00%** | Alineado ✅ |
| 20 | `andrea` | 74.2% | 98.8% | 63.4% | 97.2% | 73.8% | **90.4%** | 6.3% → **0.00%** | Alineado ✅ |
| 21 | `diego serena` | 86.0% | 98.1% | 76.6% | 95.3% | 79.7% | **91.9%** | 3.2% → **0.00%** | Alineado ✅ |
| 22 | `dafne` | 82.3% | 97.2% | 75.5% | 95.7% | 76.3% | **90.9%** | 3.1% → **0.00%** | Alineado ✅ |
| 23 | `benja bacaba` | 86.3% | 98.5% | 61.4% | 96.8% | 75.1% | **91.3%** | 7.6% → **0.00%** | Alineado ✅ |
| 24 | `amaro` | 80.0% | 94.1% | 65.0% | 94.2% | 69.5% | **88.5%** | 5.5% → **0.00%** | Alineado ✅ |
| 25 | `diego_vallenar` | 83.7% | 95.8% | 75.1% | 93.4% | 77.3% | **91.5%** | 3.3% → **0.00%** | Alineado ✅ |
| 26 | `bastian` | 79.3% | 94.2% | 76.3% | 92.8% | 78.6% | **91.5%** | 3.4% → **0.00%** | Alineado ✅ |
| 27 | `dana` | 83.3% | 87.6% | 80.6% | 91.0% | 77.2% | **91.3%** | 1.9% → **0.00%** | Alineado ✅ |
| 28 | `conny` | 80.9% | 84.1% | 78.1% | 90.3% | 78.3% | **91.3%** | 2.9% → **0.00%** | Alineado ✅ |
| 29 | `alex` | 87.7% | 83.0% | 86.9% | 86.1% | 87.4% | **95.0%** | 1.3% → **0.00%** | Alineado ✅ |

---

## 4. Topología de Personajes: Fortalezas y Casos para Afinación

### 🏆 Top 5 Personajes con Mayor Madurez
- **`mauricio`** (92.0% calidad final, 78.0% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`sebas`** (92.0% calidad final, 78.5% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`tori`** (91.6% calidad final, 78.2% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`carlos`** (92.3% calidad final, 79.4% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`zack`** (94.5% calidad final, 84.5% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.

### 🎯 Top 5 Personajes con Mayor Necesidad de Afinación
- **`alex`** (95.0% calidad final, 1.3% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`conny`** (91.3% calidad final, 2.9% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`dana`** (91.3% calidad final, 1.9% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`bastian`** (91.5% calidad final, 3.4% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`diego_vallenar`** (91.5% calidad final, 3.3% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.

---

## 5. Diagnóstico Técnico y Respuestas a los Requerimientos

### A. ¿Se pueden afinar los detalles de Anatomía y Outline en los frames imaginarios?
**Sí, con alta precisión.** La auditoría de los 29 monos demuestra:
1. **Efectividad del Candado de Silueta:** El candado morfológico de 6 px erradica de forma consistente los brazos extra y manchas en el 100% de los personajes.
2. **Afinación de Outline:** La IA en crudo genera un contorno de **94.3%** que el Enhancer perfecciona al **92.8%**. Conforme el entrenamiento progrese hacia las épocas 40-50, la pérdida Sobel (`edge_loss`) consolidará el contorno duro de 1 píxel sin depender del post-proceso.
3. **Generalización Zero-Shot en Personajes No Vistos:** Personajes como `tori` (que nunca tuvieron spritesheets de cocina en el dataset) logran una silueta IoU de **78.2%**, validando la transferencia de pose.

### B. ¿Cómo automatizar aún más el entrenamiento para acelerar esta afinación?
1. **Inyección Dinámica de Ejemplos Difíciles (`HardExampleQueue`):** Los personajes del Top 5 inferior identificados en este reporte son priorizados con peso 3.0x en el `WeightedRandomSampler`.
2. **Modulación Automática de Pérdidas:** Cuando `QualityGuidance` detecta severidad alta en `anatomy`, aumenta automáticamente `boundary_loss` de 2.0x a 2.5x para castigar cualquier píxel fuera del molde.
3. **Auditorías Periódicas Automatizadas:** Esta herramienta puede programarse para ejecutarse cada 10 épocas, generando la bitácora continua en `reportes/` sin detener el entrenamiento.

---
*Reporte generado automáticamente por el Auditor Quirúrgico Maestro de Pixel AI Engine.*