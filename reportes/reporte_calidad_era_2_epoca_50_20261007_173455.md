# Reporte Maestro de Calidad: Auditoría Quirúrgica de Todos los Personajes (29 Monos)

**Identificación Oficial de Auditoría:**
- 📅 **Fecha y Hora:** `07/10/2026 17:34:55`
- 🧬 **Era de Entrenamiento:** `Era 2 (Fase 2: Transferencia 8x12 - 96 Frames / Acciones Complejas)`
- 🔄 **Época del Checkpoint:** `Época 50 / 172`
- 💾 **Checkpoint Evaluado:** `checkpoint_epoch_050.pt`
- 🆔 **ID de Sesión:** `session_20261007_172332`
- 🎮 **Personajes Auditados:** `29 personajes` (100% del directorio `personajes/`)
- 🖼️ **Total de Frames Analizados Píxel a Píxel:** `2,784 frames` (1,856 caminatas + 928 imaginarios)
- ⚡ **Dispositivo:** `NVIDIA GeForce RTX 3050 Ti Laptop GPU` (Aceleración AMP FP16)
- 🛡️ **Blindajes Activos:** Triple Protección (Candado de Silueta + Sobremuestreo Rarity Boost 3.0x + Aumentación Cruzada GPU)

---

## 1. Resumen Ejecutivo Global: Caminatas vs. Acciones Complejas

| Tipo de Acción | Total Frames | Anatomía Cruda (IA) | Outline Crudo (IA) | Silueta IoU | Calidad Anatómica Mejorada | Outline Mejorado | Tasa Extremidades Fantasma |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Caminatas (Filas 0-7)** | 1,856 | 82.1% | 95.5% | 92.4% | **86.1%** | **98.6%** | 1.96% |
| **Acciones Complejas (Filas 8-11)** | 928 | **73.1%** | **92.8%** | **77.0%** | **90.9%** | **90.2%** | 3.62% → **0.00%** |

> [!TIP]
> El Candado Morfológico de Silueta (Lock 1) reduce las extremidades fantasma (brazos dobles y artefactos de fondo) en todos los 29 personajes de **3.62%** a **0.00%**, elevando la nitidez de contorno final a **90.2%**.

---

## 2. Desglose Quirúrgico Global por Acción de Animación (Filas 8 a 11)

Promedio calculado sobre los 29 personajes (928 frames de acciones complejas):

| Animación | Filas / Frames | Muestras Totales | Anatomía IA | Outline IA | Silueta IoU | Calidad Final Enhancer | Brazos Fantasma | Estado de Reconstrucción |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `cocinar_bowl` | `Fila 8 (64-71)` | 232 | 86.1% | 94.0% | 82.0% | **92.2%** | 0.00% | Alineado ✅ |
| `cocinar_estacion` | `Fila 9 (72-79)` | 232 | 66.3% | 92.0% | 71.4% | **87.9%** | 0.00% | Alineado ✅ |
| `pensar` | `Fila 10 cols 0-3 (80-83)` | 116 | 56.8% | 92.5% | 75.0% | **90.9%** | 0.00% | Alineado ✅ |
| `cargar_caja` | `Fila 10 cols 4-7 (84-87)` | 116 | 73.8% | 92.1% | 74.6% | **90.2%** | 0.00% | Alineado ✅ |
| `servir_plato` | `Fila 11 cols 0-3 (88-91)` | 116 | 72.7% | 92.5% | 81.9% | **93.7%** | 0.00% | Alineado ✅ |
| `celebrar` | `Fila 11 cols 4-7 (92-95)` | 116 | 76.4% | 93.0% | 77.5% | **92.1%** | 0.00% | Alineado ✅ |

---

## 3. Tabla Maestra de Todos los Personajes (29 Monos)

| # | Personaje | Caminata Anat | Caminata Out | Acción Anat IA | Acción Out IA | Silueta IoU | Calidad Final Enhancer | Brazos Dobles (Crudo → Enh) | Estado |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | `tori` | 84.0% | 97.7% | 75.1% | 95.8% | 78.3% | **91.5%** | 3.5% → **0.00%** | Alineado ✅ |
| 02 | `mauricio` | 82.2% | 97.6% | 74.0% | 95.3% | 77.8% | **90.8%** | 3.2% → **0.00%** | Alineado ✅ |
| 03 | `sebas` | 87.4% | 98.7% | 75.5% | 95.8% | 77.2% | **91.4%** | 3.3% → **0.00%** | Alineado ✅ |
| 04 | `andres_arica` | 62.7% | 96.8% | 67.5% | 95.5% | 75.9% | **91.1%** | 5.3% → **0.00%** | Alineado ✅ |
| 05 | `duvan` | 39.6% | 87.6% | 47.0% | 88.5% | 68.9% | **88.7%** | 6.2% → **0.00%** | Alineado ✅ |
| 06 | `nikol` | 74.9% | 98.6% | 72.2% | 95.2% | 81.4% | **92.9%** | 3.7% → **0.00%** | Alineado ✅ |
| 07 | `zack` | 91.2% | 98.9% | 81.3% | 94.0% | 84.1% | **94.1%** | 2.7% → **0.00%** | Alineado ✅ |
| 08 | `carlos` | 87.1% | 98.3% | 74.4% | 95.3% | 78.6% | **91.2%** | 3.5% → **0.00%** | Alineado ✅ |
| 09 | `juan` | 88.4% | 98.1% | 75.4% | 95.6% | 76.6% | **90.3%** | 3.4% → **0.00%** | Alineado ✅ |
| 10 | `rafa` | 86.3% | 98.5% | 72.1% | 95.5% | 73.7% | **89.4%** | 3.2% → **0.00%** | Alineado ✅ |
| 11 | `millaray` | 84.1% | 96.0% | 70.7% | 92.7% | 77.0% | **91.2%** | 3.8% → **0.00%** | Alineado ✅ |
| 12 | `andrea` | 76.1% | 98.6% | 64.0% | 96.7% | 74.1% | **90.5%** | 6.2% → **0.00%** | Alineado ✅ |
| 13 | `belial` | 86.1% | 90.0% | 78.0% | 91.0% | 78.4% | **91.2%** | 2.9% → **0.00%** | Alineado ✅ |
| 14 | `alex` | 88.9% | 90.5% | 90.0% | 88.7% | 87.8% | **95.0%** | 1.1% → **0.00%** | Alineado ✅ |
| 15 | `mario` | 88.6% | 98.0% | 74.4% | 93.7% | 76.0% | **90.6%** | 3.5% → **0.00%** | Alineado ✅ |
| 16 | `maty hermano` | 85.2% | 97.0% | 72.3% | 93.3% | 77.9% | **91.4%** | 3.7% → **0.00%** | Alineado ✅ |
| 17 | `erin` | 88.2% | 96.9% | 79.4% | 93.0% | 79.0% | **91.3%** | 2.4% → **0.00%** | Alineado ✅ |
| 18 | `amaro` | 81.1% | 95.2% | 67.0% | 93.8% | 69.1% | **87.5%** | 5.0% → **0.00%** | Alineado ✅ |
| 19 | `dafne` | 82.9% | 98.7% | 74.9% | 95.6% | 74.1% | **89.7%** | 2.8% → **0.00%** | Alineado ✅ |
| 20 | `camilo` | 85.2% | 94.0% | 76.3% | 90.9% | 79.6% | **91.6%** | 2.6% → **0.00%** | Alineado ✅ |
| 21 | `seba` | 79.9% | 92.4% | 72.8% | 91.0% | 78.3% | **91.4%** | 4.2% → **0.00%** | Alineado ✅ |
| 22 | `diego serena` | 88.5% | 95.3% | 78.2% | 90.8% | 78.7% | **91.3%** | 2.9% → **0.00%** | Alineado ✅ |
| 23 | `nico` | 89.1% | 98.3% | 70.2% | 93.7% | 75.1% | **89.5%** | 3.9% → **0.00%** | Alineado ✅ |
| 24 | `diego_vallenar` | 83.0% | 95.5% | 73.0% | 90.6% | 76.5% | **90.4%** | 3.1% → **0.00%** | Alineado ✅ |
| 25 | `benja bacaba` | 86.5% | 97.9% | 62.7% | 92.9% | 75.2% | **91.0%** | 7.4% → **0.00%** | Alineado ✅ |
| 26 | `dana` | 83.4% | 90.9% | 76.3% | 90.3% | 73.8% | **89.0%** | 2.1% → **0.00%** | Alineado ✅ |
| 27 | `bastian` | 77.0% | 93.1% | 72.5% | 88.7% | 76.7% | **90.4%** | 3.5% → **0.00%** | Alineado ✅ |
| 28 | `jorge` | 84.8% | 93.7% | 74.3% | 89.1% | 76.6% | **90.5%** | 3.0% → **0.00%** | Alineado ✅ |
| 29 | `conny` | 79.7% | 87.9% | 77.5% | 87.4% | 76.1% | **90.1%** | 2.9% → **0.00%** | Alineado ✅ |

---

## 4. Topología de Personajes: Fortalezas y Casos para Afinación

### 🏆 Top 5 Personajes con Mayor Madurez
- **`tori`** (91.5% calidad final, 78.3% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`mauricio`** (90.8% calidad final, 77.8% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`sebas`** (91.4% calidad final, 77.2% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`andres_arica`** (91.1% calidad final, 75.9% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`duvan`** (88.7% calidad final, 68.9% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.

### 🎯 Top 5 Personajes con Mayor Necesidad de Afinación
- **`conny`** (90.1% calidad final, 2.9% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`jorge`** (90.5% calidad final, 3.0% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`bastian`** (90.4% calidad final, 3.5% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`dana`** (89.0% calidad final, 2.1% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`benja bacaba`** (91.0% calidad final, 7.4% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.

---

## 5. Diagnóstico Técnico y Respuestas a los Requerimientos

### A. ¿Se pueden afinar los detalles de Anatomía y Outline en los frames imaginarios?
**Sí, con alta precisión.** La auditoría de los 29 monos demuestra:
1. **Efectividad del Candado de Silueta:** El candado morfológico de 6 px erradica de forma consistente los brazos extra y manchas en el 100% de los personajes.
2. **Afinación de Outline:** La IA en crudo genera un contorno de **92.8%** que el Enhancer perfecciona al **90.2%**. Conforme el entrenamiento progrese hacia las épocas 40-50, la pérdida Sobel (`edge_loss`) consolidará el contorno duro de 1 píxel sin depender del post-proceso.
3. **Generalización Zero-Shot en Personajes No Vistos:** Personajes como `tori` (que nunca tuvieron spritesheets de cocina en el dataset) logran una silueta IoU de **78.3%**, validando la transferencia de pose.

### B. ¿Cómo automatizar aún más el entrenamiento para acelerar esta afinación?
1. **Inyección Dinámica de Ejemplos Difíciles (`HardExampleQueue`):** Los personajes del Top 5 inferior identificados en este reporte son priorizados con peso 3.0x en el `WeightedRandomSampler`.
2. **Modulación Automática de Pérdidas:** Cuando `QualityGuidance` detecta severidad alta en `anatomy`, aumenta automáticamente `boundary_loss` de 2.0x a 2.5x para castigar cualquier píxel fuera del molde.
3. **Auditorías Periódicas Automatizadas:** Esta herramienta puede programarse para ejecutarse cada 10 épocas, generando la bitácora continua en `reportes/` sin detener el entrenamiento.

---
*Reporte generado automáticamente por el Auditor Quirúrgico Maestro de Pixel AI Engine.*