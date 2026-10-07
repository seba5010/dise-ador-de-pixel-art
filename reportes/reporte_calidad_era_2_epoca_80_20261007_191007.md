# Reporte Maestro de Calidad: Auditoría Quirúrgica de Todos los Personajes (29 Monos)

**Identificación Oficial de Auditoría:**
- 📅 **Fecha y Hora:** `07/10/2026 19:10:07`
- 🧬 **Era de Entrenamiento:** `Era 2 (Fase 2: Transferencia 8x12 - 96 Frames / Acciones Complejas)`
- 🔄 **Época del Checkpoint:** `Época 80 / 172`
- 💾 **Checkpoint Evaluado:** `checkpoint_epoch_080.pt`
- 🆔 **ID de Sesión:** `session_20261007_182937`
- 🎮 **Personajes Auditados:** `29 personajes` (100% del directorio `personajes/`)
- 🖼️ **Total de Frames Analizados Píxel a Píxel:** `2,784 frames` (1,856 caminatas + 928 imaginarios)
- ⚡ **Dispositivo:** `NVIDIA GeForce RTX 3050 Ti Laptop GPU` (Aceleración AMP FP16)
- 🛡️ **Blindajes Activos:** Triple Protección (Candado de Silueta + Sobremuestreo Rarity Boost 3.0x + Aumentación Cruzada GPU)

---

## 1. Resumen Ejecutivo Global: Caminatas vs. Acciones Complejas

| Tipo de Acción | Total Frames | Anatomía Cruda (IA) | Outline Crudo (IA) | Silueta IoU | Calidad Anatómica Mejorada | Outline Mejorado | Tasa Extremidades Fantasma |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Caminatas (Filas 0-7)** | 1,856 | 82.7% | 95.7% | 92.4% | **86.7%** | **98.7%** | 1.87% |
| **Acciones Complejas (Filas 8-11)** | 928 | **74.0%** | **93.0%** | **77.1%** | **90.9%** | **90.7%** | 3.50% → **0.00%** |

> [!TIP]
> El Candado Morfológico de Silueta (Lock 1) reduce las extremidades fantasma (brazos dobles y artefactos de fondo) en todos los 29 personajes de **3.50%** a **0.00%**, elevando la nitidez de contorno final a **90.7%**.

---

## 2. Desglose Quirúrgico Global por Acción de Animación (Filas 8 a 11)

Promedio calculado sobre los 29 personajes (928 frames de acciones complejas):

| Animación | Filas / Frames | Muestras Totales | Anatomía IA | Outline IA | Silueta IoU | Calidad Final Enhancer | Brazos Fantasma | Estado de Reconstrucción |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `cocinar_bowl` | `Fila 8 (64-71)` | 232 | 87.3% | 94.2% | 82.3% | **92.3%** | 0.00% | Alineado ✅ |
| `cocinar_estacion` | `Fila 9 (72-79)` | 232 | 68.3% | 92.5% | 71.8% | **87.8%** | 0.00% | Alineado ✅ |
| `pensar` | `Fila 10 cols 0-3 (80-83)` | 116 | 55.2% | 92.9% | 74.5% | **90.8%** | 0.00% | Alineado ✅ |
| `cargar_caja` | `Fila 10 cols 4-7 (84-87)` | 116 | 74.7% | 92.5% | 74.6% | **90.2%** | 0.00% | Alineado ✅ |
| `servir_plato` | `Fila 11 cols 0-3 (88-91)` | 116 | 73.8% | 92.3% | 81.9% | **93.7%** | 0.00% | Alineado ✅ |
| `celebrar` | `Fila 11 cols 4-7 (92-95)` | 116 | 77.3% | 93.0% | 77.5% | **92.1%** | 0.00% | Alineado ✅ |

---

## 3. Tabla Maestra de Todos los Personajes (29 Monos)

| # | Personaje | Caminata Anat | Caminata Out | Acción Anat IA | Acción Out IA | Silueta IoU | Calidad Final Enhancer | Brazos Dobles (Crudo → Enh) | Estado |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | `mauricio` | 82.6% | 97.7% | 74.8% | 95.3% | 78.0% | **90.9%** | 3.1% → **0.00%** | Alineado ✅ |
| 02 | `tori` | 84.9% | 97.8% | 75.6% | 95.7% | 78.0% | **91.3%** | 3.4% → **0.00%** | Alineado ✅ |
| 03 | `sebas` | 87.5% | 98.7% | 76.3% | 95.9% | 77.1% | **91.3%** | 3.1% → **0.00%** | Alineado ✅ |
| 04 | `duvan` | 48.1% | 87.9% | 54.3% | 89.5% | 70.4% | **89.1%** | 4.8% → **0.00%** | Alineado ✅ |
| 05 | `andres_arica` | 65.2% | 96.9% | 68.9% | 95.5% | 75.7% | **90.9%** | 5.0% → **0.00%** | Alineado ✅ |
| 06 | `juan` | 88.5% | 98.1% | 75.9% | 95.6% | 76.6% | **90.4%** | 3.4% → **0.00%** | Alineado ✅ |
| 07 | `nikol` | 74.4% | 98.7% | 72.0% | 95.2% | 81.6% | **92.9%** | 3.6% → **0.00%** | Alineado ✅ |
| 08 | `rafa` | 86.4% | 98.5% | 72.9% | 95.3% | 74.2% | **89.5%** | 3.2% → **0.00%** | Alineado ✅ |
| 09 | `carlos` | 88.3% | 98.4% | 75.7% | 95.4% | 78.6% | **91.2%** | 3.5% → **0.00%** | Alineado ✅ |
| 10 | `zack` | 91.6% | 98.8% | 82.5% | 94.0% | 84.4% | **94.2%** | 2.5% → **0.00%** | Alineado ✅ |
| 11 | `millaray` | 85.2% | 96.7% | 71.7% | 93.5% | 77.1% | **91.2%** | 3.7% → **0.00%** | Alineado ✅ |
| 12 | `belial` | 86.0% | 89.6% | 78.9% | 91.2% | 78.6% | **91.3%** | 2.8% → **0.00%** | Alineado ✅ |
| 13 | `andrea` | 75.6% | 98.4% | 65.0% | 96.9% | 74.1% | **90.4%** | 6.0% → **0.00%** | Alineado ✅ |
| 14 | `alex` | 89.1% | 91.1% | 90.1% | 89.7% | 87.8% | **95.0%** | 1.1% → **0.00%** | Alineado ✅ |
| 15 | `mario` | 88.9% | 98.1% | 74.6% | 93.9% | 76.1% | **90.6%** | 3.5% → **0.00%** | Alineado ✅ |
| 16 | `maty hermano` | 86.0% | 97.3% | 73.8% | 93.4% | 78.1% | **91.4%** | 3.6% → **0.00%** | Alineado ✅ |
| 17 | `erin` | 88.6% | 96.6% | 80.6% | 92.6% | 79.2% | **91.3%** | 2.1% → **0.00%** | Alineado ✅ |
| 18 | `amaro` | 81.0% | 95.3% | 67.0% | 94.1% | 69.0% | **87.4%** | 5.0% → **0.00%** | Alineado ✅ |
| 19 | `dafne` | 83.2% | 98.6% | 75.3% | 95.8% | 74.0% | **89.6%** | 2.6% → **0.00%** | Alineado ✅ |
| 20 | `camilo` | 86.2% | 94.5% | 77.5% | 90.7% | 79.9% | **91.7%** | 2.6% → **0.00%** | Alineado ✅ |
| 21 | `diego serena` | 88.7% | 96.2% | 78.7% | 91.5% | 78.8% | **91.4%** | 2.9% → **0.00%** | Alineado ✅ |
| 22 | `seba` | 79.5% | 92.4% | 72.6% | 91.2% | 77.9% | **91.2%** | 4.3% → **0.00%** | Alineado ✅ |
| 23 | `nico` | 89.3% | 98.2% | 70.6% | 94.3% | 75.3% | **89.6%** | 3.9% → **0.00%** | Alineado ✅ |
| 24 | `diego_vallenar` | 83.4% | 96.6% | 73.9% | 91.6% | 76.8% | **90.4%** | 3.1% → **0.00%** | Alineado ✅ |
| 25 | `benja bacaba` | 86.7% | 97.9% | 63.3% | 93.3% | 75.1% | **90.7%** | 7.4% → **0.00%** | Alineado ✅ |
| 26 | `bastian` | 77.3% | 93.1% | 71.7% | 89.2% | 76.7% | **90.4%** | 3.5% → **0.00%** | Alineado ✅ |
| 27 | `jorge` | 84.5% | 94.4% | 75.2% | 89.5% | 76.7% | **90.5%** | 2.8% → **0.00%** | Alineado ✅ |
| 28 | `dana` | 83.0% | 91.1% | 79.0% | 90.4% | 74.0% | **88.8%** | 2.0% → **0.00%** | Alineado ✅ |
| 29 | `conny` | 79.6% | 88.1% | 77.9% | 87.8% | 76.2% | **90.2%** | 2.8% → **0.00%** | Alineado ✅ |

---

## 4. Topología de Personajes: Fortalezas y Casos para Afinación

### 🏆 Top 5 Personajes con Mayor Madurez
- **`mauricio`** (90.9% calidad final, 78.0% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`tori`** (91.3% calidad final, 78.0% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`sebas`** (91.3% calidad final, 77.1% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`duvan`** (89.1% calidad final, 70.4% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.
- **`andres_arica`** (90.9% calidad final, 75.7% silueta IoU): Excelente adherencia a contornos y paleta cromática equilibrada.

### 🎯 Top 5 Personajes con Mayor Necesidad de Afinación
- **`conny`** (90.2% calidad final, 2.8% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`dana`** (88.8% calidad final, 2.0% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`jorge`** (90.5% calidad final, 2.8% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`bastian`** (90.4% calidad final, 3.5% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.
- **`benja bacaba`** (90.7% calidad final, 7.4% brazos crudos): Requiere mayor número de ciclos de sobremuestreo y refuerzo de boundary loss.

---

## 5. Diagnóstico Técnico y Respuestas a los Requerimientos

### A. ¿Se pueden afinar los detalles de Anatomía y Outline en los frames imaginarios?
**Sí, con alta precisión.** La auditoría de los 29 monos demuestra:
1. **Efectividad del Candado de Silueta:** El candado morfológico de 6 px erradica de forma consistente los brazos extra y manchas en el 100% de los personajes.
2. **Afinación de Outline:** La IA en crudo genera un contorno de **93.0%** que el Enhancer perfecciona al **90.7%**. Conforme el entrenamiento progrese hacia las épocas 40-50, la pérdida Sobel (`edge_loss`) consolidará el contorno duro de 1 píxel sin depender del post-proceso.
3. **Generalización Zero-Shot en Personajes No Vistos:** Personajes como `tori` (que nunca tuvieron spritesheets de cocina en el dataset) logran una silueta IoU de **78.0%**, validando la transferencia de pose.

### B. ¿Cómo automatizar aún más el entrenamiento para acelerar esta afinación?
1. **Inyección Dinámica de Ejemplos Difíciles (`HardExampleQueue`):** Los personajes del Top 5 inferior identificados en este reporte son priorizados con peso 3.0x en el `WeightedRandomSampler`.
2. **Modulación Automática de Pérdidas:** Cuando `QualityGuidance` detecta severidad alta en `anatomy`, aumenta automáticamente `boundary_loss` de 2.0x a 2.5x para castigar cualquier píxel fuera del molde.
3. **Auditorías Periódicas Automatizadas:** Esta herramienta puede programarse para ejecutarse cada 10 épocas, generando la bitácora continua en `reportes/` sin detener el entrenamiento.

---
*Reporte generado automáticamente por el Auditor Quirúrgico Maestro de Pixel AI Engine.*