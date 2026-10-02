@echo off
setlocal
cd /d "%~dp0"
set PYTHON_EXE=webui forger\system\python\python.exe
if not exist "%PYTHON_EXE%" (
    echo Python de Forge no encontrado, utilizando python del sistema...
    set PYTHON_EXE=python
)

echo ======================================================================
echo   PIXEL ART AI - FASE 2: TRANSFER LEARNING A PLANTILLA 8x12 (96 FRAMES)
echo ======================================================================
echo Transfiriendo experiencia de Fase 1 hacia la plantilla completa...
echo Entrenando poses de cocina, pensar, caja, servir y celebrar...
echo Pesos se guardaran en: checkpoints\best_generator.pt
echo ======================================================================
"%PYTHON_EXE%" train.py --phase 2 --infinite %*
pause
