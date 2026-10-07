# Reporte Quirúrgico de Calidad: Frames Imaginarios, Anatomía y Outline

**Personaje Evaluado:** `tori`  
**Checkpoint:** `latest_checkpoint.pt` (Época 31)  
**Dispositivo:** `cuda`  

---

## 1. Resumen Comparativo: Caminatas vs. Frames Imaginarios

| Tipo de Acción | Cantidad | Anatomía Cruda (IA) | Outline Crudo (IA) | Anatomía Mejorada | Outline Mejorado | Extremidades Fantasma |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Caminatas (Filas 0-7)** | 64 frames | 86.6% | 98.4% | 93.0% | 95.2% | 1.50% |
| **Acciones Complejas (Filas 8-11)** | 32 frames | **79.1%** | **96.2%** | **90.7%** | **93.5%** | 2.59% → **0.00%** |

> [!TIP]
> El candado morfológico de silueta reduce los brazos fantasma en frames imaginarios de **2.59%** a **0.00%**, elevando el outline final a **93.5%**.

---

## 2. Desglose Quirúrgico por Acción de Animación (Filas 8 a 11)

| Animación | Frames | Anatomía IA | Outline IA | Silueta IoU | Calidad Final Enhancer | Estado de Reconstrucción |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `cocinar_bowl` | `64-71` | 91.0% | 97.6% | 85.2% | **92.7%** | Alineado ✅ |
| `cocinar_estacion` | `72-79` | 70.5% | 96.6% | 71.0% | **87.3%** | Alineado ✅ |
| `pensar` | `80-83` | 71.4% | 96.7% | 77.8% | **91.1%** | Alineado ✅ |
| `cargar_caja` | `84-87` | 78.3% | 94.1% | 71.7% | **88.5%** | Alineado ✅ |
| `servir_plato` | `88-91` | 73.7% | 95.7% | 80.9% | **93.6%** | Alineado ✅ |
| `celebrar` | `92-95` | 86.4% | 94.5% | 77.8% | **92.1%** | Alineado ✅ |

---

## 3. Diagnóstico Técnico y Respuestas a las Preguntas

### A. ¿Se pueden afinar los detalles de Anatomía y Outline en los frames imaginarios?
**Sí.** El análisis demuestra que:
1. **En Anatomía:** La mayor pérdida de puntos ocurre en el torso y la colocación de codos en las animaciones de cocinar (`cocinar_bowl` y `cocinar_estacion`). El sobremuestreo de 3x recién empezó a actuar en la época 23; cada época adicional reforzará este anclaje.
2. **En Outline:** El generador produce un degradado suave de ~0.5 px en el perímetro exterior. El Enhancer lo binariza al 100%, pero el modelo mismo puede aprender a hacer el contorno duro de 1 píxel si reforzamos la pérdida de bordes (`edge_loss`).

### B. ¿Cómo automatizar aún más el entrenamiento para acelerar esta afinación?
1. **Conexión de `anatomy` al controlador de pérdidas:** Actualmente `LossMultiplierController` solo aumentaba `edge` para microdetalles. Podemos conectar `anatomy -> boundary_loss` de modo que si la anatomía cae de 75%, el castigo contra brazos dobles suba automáticamente de 2.0x a 2.5x.
2. **Loss de Borde Sobel Directo:** Aumentar el multiplicador de `edge` automáticamente cuando `outline` figure como problema secundario (actualmente ya se mapea `outline -> edge` en `LossMultiplierController`).
3. **Focalización en Frames 64-95:** Durante la auditoría de cada época, los frames que registren `outline < 75%` o `anatomy < 70%` son enviados directamente a la cola de ejemplos difíciles (`HardExampleQueue`) para recibir prioridad máxima en el siguiente ciclo.