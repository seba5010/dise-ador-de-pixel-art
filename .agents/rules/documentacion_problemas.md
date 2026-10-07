# Protocolo de Documentación Obligatoria de Incidentes y Mejoras Técnicas

Cada vez que ocurra un problema técnico en este proyecto (artefactos visuales, brazos dobles, nubes de ruido, sesgos en el dataset, anomalías en el entrenamiento, problemas de memoria VRAM, degradación de métricas de calidad o colapso modal):

1. **Investigación y Causa Raíz:**
   - Realizar un análisis forense del dataset, los tensores de PyTorch y el flujo de ejecución.
   - Identificar con precisión matemática por qué ocurrió la falla.

2. **Implementación Integral (Mínimo en 3 Niveles cuando aplique):**
   - **Entrenamiento:** Funciones de pérdida (loss), balanceo de muestreo o aumentaciones en GPU.
   - **Control de Calidad (QC):** Métricas estrictas, umbrales de rechazo y regeneración quirúrgica.
   - **Generación / Inferencia:** Post-procesado, filtros de silueta morfológicos y ensamble en Sprite Studio.

3. **Validación Automática:**
   - Ejecutar la suite de pruebas de PyTest para asegurar 0 regresiones (`& "webui forger\system\python\python.exe" -m pytest`).

4. **Documentación Obligatoria en el Repositorio:**
   - Registrar la entrada detallada en `BITACORA_TECNICA_APRENDIZAJE_Y_COLAPSO.md` con fecha, causa raíz y solución.
   - Para reformas estructurales, crear el documento arquitectónico correspondiente (`ARQUITECTURA_...md`).
   - Explicar qué pasó, cómo ocurrió, qué se implementó y cómo se solucionó para futuros desarrolladores o agentes de IA.

5. **Sincronización con Git:**
   - Hacer commit descriptivo y `git push origin main`.
