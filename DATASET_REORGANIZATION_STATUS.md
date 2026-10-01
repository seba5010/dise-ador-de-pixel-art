# Estado de reorganización del dataset

Los archivos de `personajes/` son las fuentes maestras y no se sobrescriben durante la revisión.

## Reglas de integridad

- Conservar identidad, ropa, paleta, accesorios, postura y orden original.
- No redibujar una hoja que ya esté correctamente organizada.
- Trabajar primero en `reorganizadas_revision/`.
- Formato final: PNG RGBA, 724×2172, 16 filas × 4 columnas.
- Cada celda debe contener exactamente una figura completa sin invadir otra celda.
- Solo sustituir una fuente después de validación visual y respaldo.

## Clasificación

- `CONSERVAR`: ya cumple geometría y separación; se mantiene idéntica.
- `RECOLOCAR`: figuras completas, pero el lienzo o la posición no coincide con la plantilla.
- `RECONSTRUIR`: existen figuras fusionadas, cortadas o superpuestas y requieren revisión individual.

## Estado actual

- Las 48 fuentes maestras permanecen intactas.
- La salida de `normalization_audit/` es solamente diagnóstica y no pertenece al dataset.
- La primera candidata de Alex está en `reorganizadas_revision/alex/` y todavía no está aprobada.
- El entrenamiento permanece detenido hasta completar y validar la clasificación.
