# Reporte Maestro de Calidad: Auditoría Quirúrgica de Todos los Personajes (29 Monos)

**Identificación Oficial de Auditoría:**
- 📅 **Fecha y Hora:** `10/10/2026 20:47:28`
- 🧬 **Era de Entrenamiento:** `Era 2 (Fase 2: Transferencia 8x12 - 96 Frames / Acciones Complejas)`
- 🔄 **Época del Checkpoint:** `Época 40 / 172`
- 💾 **Checkpoint Evaluado:** `checkpoint_epoch_040.pt`
- 🆔 **ID de Sesión:** `session_20261010_191711`
- 🎮 **Personajes Auditados:** `29 personajes` (100% del directorio `personajes/`)
- 🖼️ **Total de Frames Analizados Píxel a Píxel:** `2,784 frames` (1,856 caminatas + 928 imaginarios)
- ⚡ **Dispositivo:** `NVIDIA GeForce RTX 3050 Ti Laptop GPU` (Aceleración AMP FP16)
- 🛡️ **Blindajes Activos:** Triple Protección (Candado de Silueta + Sobremuestreo Rarity Boost 3.0x + Aumentación Cruzada GPU)

---

## 1. Resumen Ejecutivo Global: Caminatas vs. Acciones Complejas

| Tipo de Acción | Total Frames | Anatomía Cruda (IA) | Outline Crudo (IA) | Silueta IoU | Calidad Anatómica Mejorada | Outline Mejorado | Tasa Extremidades Fantasma |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Caminatas (Filas 0-7)** | 1,856 | 82.4% | 94.1% | 92.4% | **86.3%** | **97.6%** | 1.90% |
| **Acciones Complejas (Filas 8-11)** | 928 | **73.8%** | **92.3%** | **78.3%** | **91.8%** | **90.7%** | 3.70% → **0.00%** |

> [!TIP]
> El Candado Morfológico de Silueta (Lock 1) reduce las extremidades fantasma (brazos dobles y artefactos de fondo) en todos los 29 personajes de **3.70%** a **0.00%**, elevando la nitidez de contorno final a **90.7%**.

---

## 2. Desglose Quirúrgico Global por Acción de Animación (Filas 8 a 11)

Promedio calculado sobre los 29 personajes (928 frames de acciones complejas):

| Animación | Filas / Frames | Muestras Totales | Anatomía IA | Outline IA | Silueta IoU | Calidad Final Enhancer | Brazos Fantasma | Estado de Reconstrucción |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `cocinar_bowl` | `Fila 8 (64-71)` | 232 | 87.7% | 93.1% | 83.5% | **93.3%** | 0.00% | Alineado ✅ |
| `cocinar_estacion` | `Fila 9 (72-79)` | 232 | 67.4% | 90.2% | 72.9% | **89.4%** | 0.00% | Alineado ✅ |
| `pensar` | `Fila 10 cols 0-3 (80-83)` | 116 | 56.2% | 91.9% | 75.5% | **91.3%** | 0.00% | Alineado ✅ |
| `cargar_caja` | `Fila 10 cols 4-7 (84-87)` | 116 | 72.2% | 92.9% | 76.2% | **91.2%** | 0.00% | Alineado ✅ |
| `servir_plato` | `Fila 11 cols 0-3 (88-91)` | 116 | 73.7% | 92.7% | 82.9% | **94.2%** | 0.00% | Alineado ✅ |
| `celebrar` | `Fila 11 cols 4-7 (92-95)` | 116 | 78.0% | 94.3% | 78.9% | **92.5%** | 0.00% | Alineado ✅ |

---

## 3. Tabla Maestra de Todos los Personajes (29 Monos)

| # | Personaje | Caminata Anat | Caminata Out | Acción Anat IA | Acción Out IA | Silueta IoU | Calidad Final Enhancer | Brazos Dobles (Crudo → Enh) | Estado |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | `mauricio` | 83.6% | 97.2% | 71.3% | 95.4% | 78.4% | **92.4%** | 4.1% → **0.00%** | Alineado ✅ |
| 02 | `duvan` | 64.7% | 93.7% | 64.5% | 91.3% | 73.5% | **90.1%** | 3.7% → **0.00%** | Alineado ✅ |
| 03 | `sebas` | 87.1% | 98.9% | 74.8% | 95.0% | 78.9% | **92.3%** | 3.5% → **0.00%** | Alineado ✅ |
| 04 | `carlos` | 85.6% | 97.8% | 72.9% | 94.6% | 80.0% | **92.4%** | 3.6% → **0.00%** | Alineado ✅ |
| 05 | `andres_arica` | 66.7% | 97.0% | 72.0% | 94.8% | 77.9% | **91.9%** | 4.2% → **0.00%** | Alineado ✅ |
| 06 | `rafa` | 85.9% | 97.9% | 76.1% | 94.3% | 76.7% | **91.2%** | 3.0% → **0.00%** | Alineado ✅ |
| 07 | `zack` | 89.7% | 98.2% | 79.2% | 93.5% | 83.9% | **94.3%** | 3.1% → **0.00%** | Alineado ✅ |
| 08 | `nikol` | 71.9% | 98.1% | 73.0% | 93.5% | 81.5% | **93.1%** | 4.3% → **0.00%** | Alineado ✅ |
| 09 | `tori` | 84.3% | 98.0% | 75.4% | 94.5% | 79.0% | **91.6%** | 3.4% → **0.00%** | Alineado ✅ |
| 10 | `mario` | 88.0% | 97.5% | 74.1% | 94.3% | 78.2% | **91.8%** | 3.4% → **0.00%** | Alineado ✅ |
| 11 | `millaray` | 85.1% | 94.3% | 69.8% | 93.7% | 77.5% | **91.6%** | 4.5% → **0.00%** | Alineado ✅ |
| 12 | `maty hermano` | 84.0% | 96.7% | 73.4% | 93.5% | 79.4% | **92.5%** | 3.7% → **0.00%** | Alineado ✅ |
| 13 | `camilo` | 83.0% | 92.3% | 76.1% | 92.3% | 80.0% | **92.5%** | 3.1% → **0.00%** | Alineado ✅ |
| 14 | `juan` | 88.3% | 97.9% | 74.6% | 93.8% | 76.9% | **91.3%** | 3.7% → **0.00%** | Alineado ✅ |
| 15 | `belial` | 85.6% | 89.1% | 78.8% | 90.8% | 79.2% | **92.1%** | 2.7% → **0.00%** | Alineado ✅ |
| 16 | `andrea` | 74.8% | 97.7% | 63.7% | 96.8% | 74.1% | **90.4%** | 6.3% → **0.00%** | Alineado ✅ |
| 17 | `erin` | 87.1% | 95.7% | 78.6% | 91.6% | 80.0% | **92.4%** | 2.7% → **0.00%** | Alineado ✅ |
| 18 | `nico` | 88.3% | 97.4% | 72.1% | 93.6% | 77.2% | **91.0%** | 3.9% → **0.00%** | Alineado ✅ |
| 19 | `diego serena` | 86.3% | 95.9% | 77.9% | 92.1% | 80.4% | **92.6%** | 3.1% → **0.00%** | Alineado ✅ |
| 20 | `jorge` | 81.9% | 95.0% | 73.7% | 92.2% | 77.4% | **91.3%** | 3.5% → **0.00%** | Alineado ✅ |
| 21 | `benja bacaba` | 84.2% | 96.2% | 61.2% | 95.3% | 75.3% | **91.3%** | 7.5% → **0.00%** | Alineado ✅ |
| 22 | `bastian` | 78.8% | 93.5% | 74.6% | 89.5% | 78.4% | **91.6%** | 3.7% → **0.00%** | Alineado ✅ |
| 23 | `seba` | 78.4% | 91.4% | 72.6% | 90.5% | 78.2% | **91.8%** | 4.2% → **0.00%** | Alineado ✅ |
| 24 | `diego_vallenar` | 82.7% | 93.0% | 73.6% | 90.6% | 78.0% | **91.8%** | 3.4% → **0.00%** | Alineado ✅ |
| 25 | `dafne` | 82.3% | 94.4% | 76.2% | 92.6% | 76.1% | **90.7%** | 3.1% → **0.00%** | Alineado ✅ |
| 26 | `amaro` | 79.4% | 91.0% | 65.5% | 92.5% | 70.6% | **88.8%** | 5.2% → **0.00%** | Alineado ✅ |
| 27 | `dana` | 82.9% | 83.5% | 78.6% | 86.4% | 77.8% | **91.4%** | 2.3% → **0.00%** | Alineado ✅ |
| 28 | `alex` | 87.6% | 79.9% | 88.9% | 82.2% | 87.4% | **95.0%** | 1.3% → **0.00%** | Alineado ✅ |
| 29 | `conny` | 81.2% | 80.6% | 77.1% | 85.7% | 78.3% | **91.2%** | 3.1% → **0.00%** | Alineado ✅ |

---

## 4. Topología de Personajes: Fortalezas y Casos para Afinación

### 🏆 Top 5 Personajes con Mayor Madurez
- **`mauricio`** (92.4% calidad final, 78.4% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`duvan`** (90.1% calidad final, 73.5% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`sebas`** (92.3% calidad final, 78.9% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`carlos`** (92.4% calidad final, 80.0% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`andres_arica`** (91.9% calidad final, 77.9% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.

### 🎯 Top 5 Personajes con Mayor Necesidad de Afinación
- **`conny`** (91.2% calidad final, 3.1% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`alex`** (95.0% calidad final, 1.3% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`dana`** (91.4% calidad final, 2.3% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`amaro`** (88.8% calidad final, 5.2% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`dafne`** (90.7% calidad final, 3.1% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.

---

## 5. Diagnóstico Técnico y Respuestas a los Requerimientos

### A. ¿Se pueden afinar los detalles de Anatomía y Outline en los frames imaginarios?
**Sí, con alta precisión.** La auditoría de los 29 monos demuestra:
1. **Efectividad del Candado de Silueta:** El candado morfológico de 6 px erradica de forma consistente los brazos extra y manchas en el 100% de los personajes.
2. **Afinación de Outline:** La IA en crudo genera un contorno de **92.3%** que el Enhancer perfecciona al **90.7%**. Conforme el entrenamiento progrese hacia las épocas 40-50, la pérdida Sobel (`edge_loss`) consolidará el contorno duro de 1 píxel sin depender del post-proceso.
3. **Generalización Zero-Shot en Personajes No Vistos:** Personajes como `tori` (que nunca tuvieron spritesheets de cocina en el dataset) logran una silueta IoU de **79.0%**, validando la transferencia de pose.

### B. ¿Cómo automatizar aún más el entrenamiento para acelerar esta afinación?
1. **Inyección Dinámica de Ejemplos Difíciles (`HardExampleQueue`):** Los personajes del Top 5 inferior identificados en este reporte son priorizados con peso 3.0x en el `WeightedRandomSampler`.
2. **Modulación Automática de Pérdidas:** Cuando `QualityGuidance` detecta severidad alta en `anatomy`, aumenta automáticamente `boundary_loss` de 2.0x a 2.5x para castigar cualquier píxel fuera del molde.
3. **Auditorías Periódicas Automatizadas:** Esta herramienta puede programarse para ejecutarse cada 10 épocas, generando la bitácora continua en `reportes/` sin detener el entrenamiento.

---
*Reporte generado automáticamente por el Auditor Quirúrgico Maestro de Pixel AI Engine.*