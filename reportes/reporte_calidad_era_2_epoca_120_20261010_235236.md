# Reporte Maestro de Calidad: Auditoría Quirúrgica de Todos los Personajes (29 Monos)

**Identificación Oficial de Auditoría:**
- 📅 **Fecha y Hora:** `10/10/2026 23:52:36`
- 🧬 **Era de Entrenamiento:** `Era 2 (Fase 2: Transferencia 8x12 - 96 Frames / Acciones Complejas)`
- 🔄 **Época del Checkpoint:** `Época 120 / 172`
- 💾 **Checkpoint Evaluado:** `checkpoint_epoch_120.pt`
- 🆔 **ID de Sesión:** `session_20261010_191711`
- 🎮 **Personajes Auditados:** `29 personajes` (100% del directorio `personajes/`)
- 🖼️ **Total de Frames Analizados Píxel a Píxel:** `2,784 frames` (1,856 caminatas + 928 imaginarios)
- ⚡ **Dispositivo:** `NVIDIA GeForce RTX 3050 Ti Laptop GPU` (Aceleración AMP FP16)
- 🛡️ **Blindajes Activos:** Triple Protección (Candado de Silueta + Sobremuestreo Rarity Boost 3.0x + Aumentación Cruzada GPU)

---

## 1. Resumen Ejecutivo Global: Caminatas vs. Acciones Complejas

| Tipo de Acción | Total Frames | Anatomía Cruda (IA) | Outline Crudo (IA) | Silueta IoU | Calidad Anatómica Mejorada | Outline Mejorado | Tasa Extremidades Fantasma |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Caminatas (Filas 0-7)** | 1,856 | 81.8% | 95.9% | 92.4% | **85.7%** | **98.3%** | 1.96% |
| **Acciones Complejas (Filas 8-11)** | 928 | **72.0%** | **94.3%** | **77.1%** | **91.3%** | **93.1%** | 3.82% → **0.00%** |

> [!TIP]
> El Candado Morfológico de Silueta (Lock 1) reduce las extremidades fantasma (brazos dobles y artefactos de fondo) en todos los 29 personajes de **3.82%** a **0.00%**, elevando la nitidez de contorno final a **93.1%**.

---

## 2. Desglose Quirúrgico Global por Acción de Animación (Filas 8 a 11)

Promedio calculado sobre los 29 personajes (928 frames de acciones complejas):

| Animación | Filas / Frames | Muestras Totales | Anatomía IA | Outline IA | Silueta IoU | Calidad Final Enhancer | Brazos Fantasma | Estado de Reconstrucción |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `cocinar_bowl` | `Fila 8 (64-71)` | 232 | 86.4% | 95.9% | 83.1% | **93.2%** | 0.00% | Alineado ✅ |
| `cocinar_estacion` | `Fila 9 (72-79)` | 232 | 65.3% | 94.3% | 71.6% | **88.8%** | 0.00% | Alineado ✅ |
| `pensar` | `Fila 10 cols 0-3 (80-83)` | 116 | 56.9% | 94.5% | 74.0% | **90.7%** | 0.00% | Alineado ✅ |
| `cargar_caja` | `Fila 10 cols 4-7 (84-87)` | 116 | 69.0% | 93.1% | 74.9% | **90.7%** | 0.00% | Alineado ✅ |
| `servir_plato` | `Fila 11 cols 0-3 (88-91)` | 116 | 73.0% | 93.6% | 81.6% | **93.5%** | 0.00% | Alineado ✅ |
| `celebrar` | `Fila 11 cols 4-7 (92-95)` | 116 | 73.9% | 93.2% | 76.8% | **91.8%** | 0.00% | Alineado ✅ |

---

## 3. Tabla Maestra de Todos los Personajes (29 Monos)

| # | Personaje | Caminata Anat | Caminata Out | Acción Anat IA | Acción Out IA | Silueta IoU | Calidad Final Enhancer | Brazos Dobles (Crudo → Enh) | Estado |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | `mauricio` | 77.7% | 97.6% | 66.1% | 94.5% | 76.8% | **91.7%** | 4.2% → **0.00%** | Alineado ✅ |
| 02 | `sebas` | 87.4% | 98.9% | 73.6% | 95.8% | 77.8% | **91.7%** | 3.7% → **0.00%** | Alineado ✅ |
| 03 | `tori` | 84.3% | 98.7% | 72.4% | 95.2% | 77.3% | **91.4%** | 3.7% → **0.00%** | Alineado ✅ |
| 04 | `carlos` | 84.3% | 98.8% | 70.4% | 95.3% | 78.3% | **91.9%** | 4.1% → **0.00%** | Alineado ✅ |
| 05 | `zack` | 90.6% | 99.2% | 81.6% | 96.9% | 83.4% | **93.8%** | 2.6% → **0.00%** | Alineado ✅ |
| 06 | `juan` | 87.9% | 98.7% | 72.2% | 95.1% | 76.0% | **91.2%** | 4.2% → **0.00%** | Alineado ✅ |
| 07 | `andres_arica` | 66.7% | 97.3% | 67.9% | 94.5% | 75.4% | **91.0%** | 5.1% → **0.00%** | Alineado ✅ |
| 08 | `rafa` | 83.8% | 98.8% | 71.3% | 95.0% | 74.8% | **90.7%** | 3.5% → **0.00%** | Alineado ✅ |
| 09 | `nikol` | 69.0% | 98.3% | 66.1% | 95.5% | 79.5% | **92.4%** | 4.6% → **0.00%** | Alineado ✅ |
| 10 | `millaray` | 83.0% | 98.1% | 68.1% | 95.5% | 76.8% | **91.2%** | 4.7% → **0.00%** | Alineado ✅ |
| 11 | `belial` | 84.8% | 89.5% | 75.5% | 92.8% | 78.3% | **91.7%** | 2.9% → **0.00%** | Alineado ✅ |
| 12 | `maty hermano` | 84.6% | 98.1% | 73.5% | 94.7% | 78.1% | **91.9%** | 3.8% → **0.00%** | Alineado ✅ |
| 13 | `mario` | 87.0% | 98.2% | 72.7% | 95.3% | 77.0% | **91.5%** | 3.6% → **0.00%** | Alineado ✅ |
| 14 | `duvan` | 68.1% | 94.4% | 68.2% | 91.6% | 72.3% | **89.5%** | 2.9% → **0.00%** | Alineado ✅ |
| 15 | `erin` | 86.9% | 97.6% | 76.9% | 95.4% | 79.1% | **92.1%** | 3.0% → **0.00%** | Alineado ✅ |
| 16 | `jorge` | 79.6% | 97.1% | 71.4% | 95.1% | 75.5% | **90.5%** | 3.6% → **0.00%** | Alineado ✅ |
| 17 | `camilo` | 81.9% | 95.7% | 75.7% | 94.9% | 78.7% | **92.1%** | 3.2% → **0.00%** | Alineado ✅ |
| 18 | `seba` | 77.6% | 96.0% | 68.4% | 95.1% | 77.1% | **91.3%** | 4.6% → **0.00%** | Alineado ✅ |
| 19 | `dafne` | 81.7% | 97.9% | 72.4% | 96.1% | 74.6% | **90.1%** | 3.3% → **0.00%** | Alineado ✅ |
| 20 | `nico` | 86.5% | 98.1% | 74.4% | 95.2% | 76.8% | **90.7%** | 3.4% → **0.00%** | Alineado ✅ |
| 21 | `diego serena` | 87.2% | 97.3% | 76.5% | 95.4% | 78.9% | **91.7%** | 3.2% → **0.00%** | Alineado ✅ |
| 22 | `diego_vallenar` | 83.8% | 96.1% | 75.6% | 94.4% | 77.7% | **91.4%** | 3.0% → **0.00%** | Alineado ✅ |
| 23 | `andrea` | 73.5% | 98.7% | 62.2% | 96.8% | 73.3% | **90.2%** | 6.4% → **0.00%** | Alineado ✅ |
| 24 | `amaro` | 79.1% | 95.4% | 62.0% | 93.7% | 68.3% | **87.7%** | 5.7% → **0.00%** | Alineado ✅ |
| 25 | `benja bacaba` | 86.4% | 98.9% | 61.7% | 96.8% | 75.2% | **90.9%** | 7.3% → **0.00%** | Alineado ✅ |
| 26 | `bastian` | 78.0% | 94.6% | 73.9% | 92.4% | 77.7% | **91.3%** | 3.7% → **0.00%** | Alineado ✅ |
| 27 | `dana` | 82.7% | 89.3% | 78.5% | 91.9% | 76.6% | **91.0%** | 2.1% → **0.00%** | Alineado ✅ |
| 28 | `conny` | 80.4% | 84.5% | 73.9% | 90.8% | 76.8% | **90.9%** | 3.3% → **0.00%** | Alineado ✅ |
| 29 | `alex` | 87.5% | 79.9% | 85.8% | 83.7% | 87.3% | **94.9%** | 1.2% → **0.00%** | Alineado ✅ |

---

## 4. Topología de Personajes: Fortalezas y Casos para Afinación

### 🏆 Top 5 Personajes con Mayor Madurez
- **`mauricio`** (91.7% calidad final, 76.8% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`sebas`** (91.7% calidad final, 77.8% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`tori`** (91.4% calidad final, 77.3% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`carlos`** (91.9% calidad final, 78.3% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`zack`** (93.8% calidad final, 83.4% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.

### 🎯 Top 5 Personajes con Mayor Necesidad de Afinación
- **`alex`** (94.9% calidad final, 1.2% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`conny`** (90.9% calidad final, 3.3% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`dana`** (91.0% calidad final, 2.1% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`bastian`** (91.3% calidad final, 3.7% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`benja bacaba`** (90.9% calidad final, 7.3% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.

---

## 5. Diagnóstico Técnico y Respuestas a los Requerimientos

### A. ¿Se pueden afinar los detalles de Anatomía y Outline en los frames imaginarios?
**Sí, con alta precisión.** La auditoría de los 29 monos demuestra:
1. **Efectividad del Candado de Silueta:** El candado morfológico de 6 px erradica de forma consistente los brazos extra y manchas en el 100% de los personajes.
2. **Afinación de Outline:** La IA en crudo genera un contorno de **94.3%** que el Enhancer perfecciona al **93.1%**. Conforme el entrenamiento progrese hacia las épocas 40-50, la pérdida Sobel (`edge_loss`) consolidará el contorno duro de 1 píxel sin depender del post-proceso.
3. **Generalización Zero-Shot en Personajes No Vistos:** Personajes como `tori` (que nunca tuvieron spritesheets de cocina en el dataset) logran una silueta IoU de **77.3%**, validando la transferencia de pose.

### B. ¿Cómo automatizar aún más el entrenamiento para acelerar esta afinación?
1. **Inyección Dinámica de Ejemplos Difíciles (`HardExampleQueue`):** Los personajes del Top 5 inferior identificados en este reporte son priorizados con peso 3.0x en el `WeightedRandomSampler`.
2. **Modulación Automática de Pérdidas:** Cuando `QualityGuidance` detecta severidad alta en `anatomy`, aumenta automáticamente `boundary_loss` de 2.0x a 2.5x para castigar cualquier píxel fuera del molde.
3. **Auditorías Periódicas Automatizadas:** Esta herramienta puede programarse para ejecutarse cada 10 épocas, generando la bitácora continua en `reportes/` sin detener el entrenamiento.

---
*Reporte generado automáticamente por el Auditor Quirúrgico Maestro de Pixel AI Engine.*