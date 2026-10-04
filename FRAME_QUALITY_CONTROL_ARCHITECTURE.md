# FRAME QUALITY CONTROL ARCHITECTURE

## Objetivo

Crear una capa interactiva de control de calidad por frame que permita valorar la salida generada antes de que se convierta en entrenamiento o en un artefacto aceptado.

## Arquitectura modular

- `pixel_ai_engine/quality_gate.py`: auditoría automática y score por frame.
- `pixel_ai_engine/frame_quality_review.py`: cola persistida, estados, historial, aprobaciones y rechazos.
- `frame_review_queue.jsonl`: cola de revisión humana.
- `output/`: artefactos generados y snapshots de reemplazos por frame.

## Flujo propuesto

1. Generar frame.
2. Evaluar automáticamente con `QualityGate.evaluate_single_frame`.
3. Registrar en `FrameQualityReviewManager`.
4. Usuario revisa y aprueba o rechaza.
5. El histórico queda persistido para trazabilidad.

## Diferencia entre generated, target y error

- `generated`: salida actual del modelo.
- `target`: referencia correcta del dataset o plantilla.
- `error`: evidencia del fallo visual sin convertirla en ground truth.

## Seguridad

- Sólo se aceptan rutas dentro del árbol del proyecto.
- No se altera el dataset ni la ground truth original.
- Las rutas externas se bloquean por diseño.

## Estado futuro

La siguiente capa (no activada en este incremento) añade:

- regeneración individual;
- ranking de candidatos;
- aplicación del mejor candidato;
- hard examples;
- integración con sampler y entrenamiento futuro.

Este incremento queda detenido justo después de QA y aprobación del panel de revisión por frame.
