@echo off
setlocal
cd /d "%~dp0"
set PYTHON_EXE=webui forger\system\python\python.exe
if not exist "%PYTHON_EXE%" (
    echo Python de Forge no encontrado, utilizando python del sistema...
    set PYTHON_EXE=python
)

cls
echo ======================================================================
echo   PREPARANDO NUEVO DATASET - TODOS LOS PERSONAJES
echo   Fuente: dataset_frames_individuales/ (50 variantes, 24 personajes)
echo   Destino: dataset_supervisado/
echo ======================================================================
echo.

"%PYTHON_EXE%" -m pixel_ai_engine.prepare_supervised_dataset

echo.
echo ======================================================================
echo   DATASET LISTO - Ahora ejecuta run_train.bat para entrenar
echo ======================================================================
pause