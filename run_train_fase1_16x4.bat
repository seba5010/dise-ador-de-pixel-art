@echo off
setlocal
cd /d "%~dp0"
set PYTHON_EXE=webui forger\system\python\python.exe
if not exist "%PYTHON_EXE%" (
    echo Python de Forge no encontrado, utilizando python del sistema...
    set PYTHON_EXE=python
)

echo ======================================================================
echo   PIXEL ART AI - FASE 1: BASE CON PLANTILLA CHICA (16x4 / 64 FRAMES)
echo ======================================================================
echo Entrenando base solida con Conny y Dana...
echo Pesos se guardaran en: checkpoints\base_generator_16x4.pt
echo ======================================================================
"%PYTHON_EXE%" train.py --phase 1 --infinite %*
pause
