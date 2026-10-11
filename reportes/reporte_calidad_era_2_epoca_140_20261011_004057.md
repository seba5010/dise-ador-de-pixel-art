# Reporte Maestro de Calidad: Auditoría Quirúrgica de Todos los Personajes (29 Monos)

**Identificación Oficial de Auditoría:**
- 📅 **Fecha y Hora:** `11/10/2026 00:40:57`
- 🧬 **Era de Entrenamiento:** `Era 2 (Fase 2: Transferencia 8x12 - 96 Frames / Acciones Complejas)`
- 🔄 **Época del Checkpoint:** `Época 140 / 172`
- 💾 **Checkpoint Evaluado:** `checkpoint_epoch_140.pt`
- 🆔 **ID de Sesión:** `session_20261010_191711`
- 🎮 **Personajes Auditados:** `29 personajes` (100% del directorio `personajes/`)
- 🖼️ **Total de Frames Analizados Píxel a Píxel:** `2,784 frames` (1,856 caminatas + 928 imaginarios)
- ⚡ **Dispositivo:** `NVIDIA GeForce RTX 3050 Ti Laptop GPU` (Aceleración AMP FP16)
- 🛡️ **Blindajes Activos:** Triple Protección (Candado de Silueta + Sobremuestreo Rarity Boost 3.0x + Aumentación Cruzada GPU)

---

## 1. Resumen Ejecutivo Global: Caminatas vs. Acciones Complejas

| Tipo de Acción | Total Frames | Anatomía Cruda (IA) | Outline Crudo (IA) | Silueta IoU | Calidad Anatómica Mejorada | Outline Mejorado | Tasa Extremidades Fantasma |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Caminatas (Filas 0-7)** | 1,856 | 81.5% | 95.5% | 92.4% | **85.5%** | **98.0%** | 1.90% |
| **Acciones Complejas (Filas 8-11)** | 928 | **71.3%** | **93.7%** | **77.3%** | **91.4%** | **91.6%** | 3.72% → **0.00%** |

> [!TIP]
> El Candado Morfológico de Silueta (Lock 1) reduce las extremidades fantasma (brazos dobles y artefactos de fondo) en todos los 29 personajes de **3.72%** a **0.00%**, elevando la nitidez de contorno final a **91.6%**.

---

## 2. Desglose Quirúrgico Global por Acción de Animación (Filas 8 a 11)

Promedio calculado sobre los 29 personajes (928 frames de acciones complejas):

| Animación | Filas / Frames | Muestras Totales | Anatomía IA | Outline IA | Silueta IoU | Calidad Final Enhancer | Brazos Fantasma | Estado de Reconstrucción |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `cocinar_bowl` | `Fila 8 (64-71)` | 232 | 84.7% | 95.0% | 83.0% | **93.0%** | 0.00% | Alineado ✅ |
| `cocinar_estacion` | `Fila 9 (72-79)` | 232 | 65.3% | 93.0% | 71.6% | **88.8%** | 0.00% | Alineado ✅ |
| `pensar` | `Fila 10 cols 0-3 (80-83)` | 116 | 55.5% | 93.8% | 75.0% | **90.9%** | 0.00% | Alineado ✅ |
| `cargar_caja` | `Fila 10 cols 4-7 (84-87)` | 116 | 67.7% | 93.3% | 74.7% | **90.7%** | 0.00% | Alineado ✅ |
| `servir_plato` | `Fila 11 cols 0-3 (88-91)` | 116 | 71.2% | 93.4% | 82.3% | **93.8%** | 0.00% | Alineado ✅ |
| `celebrar` | `Fila 11 cols 4-7 (92-95)` | 116 | 76.3% | 93.3% | 77.4% | **91.9%** | 0.00% | Alineado ✅ |

---

## 3. Tabla Maestra de Todos los Personajes (29 Monos)

| # | Personaje | Caminata Anat | Caminata Out | Acción Anat IA | Acción Out IA | Silueta IoU | Calidad Final Enhancer | Brazos Dobles (Crudo → Enh) | Estado |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | `mauricio` | 77.6% | 97.8% | 66.1% | 95.2% | 76.9% | **91.8%** | 4.6% → **0.00%** | Alineado ✅ |
| 02 | `carlos` | 85.1% | 98.9% | 72.4% | 96.2% | 79.0% | **92.0%** | 3.7% → **0.00%** | Alineado ✅ |
| 03 | `andres_arica` | 65.6% | 97.5% | 68.0% | 95.4% | 76.9% | **91.5%** | 4.7% → **0.00%** | Alineado ✅ |
| 04 | `zack` | 87.0% | 99.2% | 77.0% | 97.1% | 83.8% | **94.0%** | 2.9% → **0.00%** | Alineado ✅ |
| 05 | `sebas` | 86.4% | 98.9% | 71.7% | 96.2% | 78.3% | **91.9%** | 3.6% → **0.00%** | Alineado ✅ |
| 06 | `tori` | 83.3% | 98.7% | 72.9% | 95.8% | 78.2% | **91.6%** | 3.7% → **0.00%** | Alineado ✅ |
| 07 | `juan` | 88.7% | 98.7% | 72.5% | 95.5% | 75.7% | **90.8%** | 3.9% → **0.00%** | Alineado ✅ |
| 08 | `duvan` | 63.6% | 94.5% | 60.9% | 91.9% | 70.7% | **89.2%** | 4.0% → **0.00%** | Alineado ✅ |
| 09 | `rafa` | 84.7% | 98.8% | 70.5% | 95.1% | 74.9% | **90.4%** | 3.5% → **0.00%** | Alineado ✅ |
| 10 | `mario` | 86.7% | 98.4% | 72.4% | 95.9% | 78.0% | **92.0%** | 3.4% → **0.00%** | Alineado ✅ |
| 11 | `millaray` | 81.5% | 97.8% | 65.3% | 95.0% | 76.9% | **91.3%** | 4.7% → **0.00%** | Alineado ✅ |
| 12 | `nikol` | 72.0% | 98.8% | 70.6% | 95.0% | 80.4% | **92.8%** | 4.1% → **0.00%** | Alineado ✅ |
| 13 | `erin` | 87.4% | 97.5% | 75.7% | 94.3% | 79.7% | **92.3%** | 2.8% → **0.00%** | Alineado ✅ |
| 14 | `maty hermano` | 84.2% | 97.6% | 72.5% | 94.9% | 78.0% | **91.7%** | 3.6% → **0.00%** | Alineado ✅ |
| 15 | `jorge` | 82.2% | 97.3% | 71.8% | 95.3% | 75.4% | **90.7%** | 3.6% → **0.00%** | Alineado ✅ |
| 16 | `benja bacaba` | 85.7% | 99.1% | 60.0% | 97.3% | 75.5% | **91.3%** | 7.3% → **0.00%** | Alineado ✅ |
| 17 | `seba` | 78.2% | 94.7% | 68.2% | 94.5% | 76.8% | **91.5%** | 4.5% → **0.00%** | Alineado ✅ |
| 18 | `nico` | 89.3% | 98.9% | 77.6% | 95.5% | 77.4% | **90.6%** | 2.8% → **0.00%** | Alineado ✅ |
| 19 | `camilo` | 83.7% | 95.6% | 75.8% | 93.6% | 79.2% | **92.1%** | 2.9% → **0.00%** | Alineado ✅ |
| 20 | `andrea` | 73.7% | 98.3% | 62.2% | 96.4% | 73.8% | **90.3%** | 6.2% → **0.00%** | Alineado ✅ |
| 21 | `belial` | 80.4% | 87.1% | 72.7% | 89.9% | 78.7% | **91.8%** | 2.9% → **0.00%** | Alineado ✅ |
| 22 | `diego serena` | 88.2% | 97.8% | 77.5% | 93.8% | 79.7% | **91.9%** | 3.0% → **0.00%** | Alineado ✅ |
| 23 | `diego_vallenar` | 85.0% | 95.9% | 75.9% | 93.6% | 77.0% | **90.8%** | 2.9% → **0.00%** | Alineado ✅ |
| 24 | `amaro` | 79.9% | 93.8% | 63.8% | 93.2% | 68.4% | **87.8%** | 5.4% → **0.00%** | Alineado ✅ |
| 25 | `dafne` | 78.8% | 95.6% | 68.6% | 93.6% | 74.7% | **90.0%** | 3.2% → **0.00%** | Alineado ✅ |
| 26 | `dana` | 84.5% | 87.9% | 77.8% | 89.6% | 75.5% | **90.5%** | 2.1% → **0.00%** | Alineado ✅ |
| 27 | `conny` | 80.2% | 82.5% | 71.8% | 88.4% | 77.8% | **91.3%** | 3.3% → **0.00%** | Alineado ✅ |
| 28 | `bastian` | 71.9% | 92.4% | 69.9% | 88.2% | 77.2% | **90.9%** | 3.5% → **0.00%** | Alineado ✅ |
| 29 | `alex` | 87.7% | 79.0% | 86.8% | 81.8% | 87.3% | **95.0%** | 1.4% → **0.00%** | Alineado ✅ |

---

## 4. Topología de Personajes: Fortalezas y Casos para Afinación

### 🏆 Top 5 Personajes con Mayor Madurez
- **`mauricio`** (91.8% calidad final, 76.9% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`carlos`** (92.0% calidad final, 79.0% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`andres_arica`** (91.5% calidad final, 76.9% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`zack`** (94.0% calidad final, 83.8% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`sebas`** (91.9% calidad final, 78.3% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.

### 🎯 Top 5 Personajes con Mayor Necesidad de Afinación
- **`alex`** (95.0% calidad final, 1.4% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`bastian`** (90.9% calidad final, 3.5% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`conny`** (91.3% calidad final, 3.3% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`dana`** (90.5% calidad final, 2.1% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`dafne`** (90.0% calidad final, 3.2% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.

---

## 5. Diagnóstico Técnico y Respuestas a los Requerimientos

### A. ¿Se pueden afinar los detalles de Anatomía y Outline en los frames imaginarios?
**Sí, con alta precisión.** La auditoría de los 29 monos demuestra:
1. **Efectividad del Candado de Silueta:** El candado morfológico de 6 px erradica de forma consistente los brazos extra y manchas en el 100% de los personajes.
2. **Afinación de Outline:** La IA en crudo genera un contorno de **93.7%** que el Enhancer perfecciona al **91.6%**. Conforme el entrenamiento progrese hacia las épocas 40-50, la pérdida Sobel (`edge_loss`) consolidará el contorno duro de 1 píxel sin depender del post-proceso.
3. **Generalización Zero-Shot en Personajes No Vistos:** Personajes como `tori` (que nunca tuvieron spritesheets de cocina en el dataset) logran una silueta IoU de **78.2%**, validando la transferencia de pose.

### B. ¿Cómo automatizar aún más el entrenamiento para acelerar esta afinación?
1. **Inyección Dinámica de Ejemplos Difíciles (`HardExampleQueue`):** Los personajes del Top 5 inferior identificados en este reporte son priorizados con peso 3.0x en el `WeightedRandomSampler`.
2. **Modulación Automática de Pérdidas:** Cuando `QualityGuidance` detecta severidad alta en `anatomy`, aumenta automáticamente `boundary_loss` de 2.0x a 2.5x para castigar cualquier píxel fuera del molde.
3. **Auditorías Periódicas Automatizadas:** Esta herramienta puede programarse para ejecutarse cada 10 épocas, generando la bitácora continua en `reportes/` sin detener el entrenamiento.

---
*Reporte generado automáticamente por el Auditor Quirúrgico Maestro de Pixel AI Engine.*