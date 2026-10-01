@echo off
chcp 65001 > nul
title Pipeline Completo: Pulido Fase 1 + Fase 2 (96 frames) + Elevacion Final

echo ======================================================================
echo   PIXEL ART AI ENGINE - PIPELINE COMPLETO AUTOMATIZADO (3 FASES)
echo   - 1. Pulido Fase 1 (Epocas 500 a 800): Texturas y micro-detalles 1px
echo   - 2. Puerta de Calidad y Transferencia a Fase 2 (Plantilla 8x12)
echo   - 3. Entrenamiento Fase 2: 96 poses con Alex y Amaro (600 epocas)
echo   - 4. Revision Critica y Elevacion Final Quirurgica del Spritesheet
echo ======================================================================
echo.
echo Monitor web en vivo disponible en: http://localhost:8787/monitor.html
echo.

set PYTHON_EXE="webui forger\system\python\python.exe"

if not exist %PYTHON_EXE% (
    echo [ERROR] No se encontro el interprete de Python en %PYTHON_EXE%
    pause
    exit /b 1
)

%PYTHON_EXE% train_dos_fases_1500.py --epochs-fase1 800 --epochs-fase2 1100 --batch-size 8

echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
