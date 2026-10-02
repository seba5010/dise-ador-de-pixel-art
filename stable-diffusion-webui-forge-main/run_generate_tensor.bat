@echo off
setlocal
cd /d "%~dp0"

set PYTHON_EXE=..\webui forger\system\python\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=webui forger\system\python\python.exe
)
if not exist "%PYTHON_EXE%" (
    echo Python embebido no encontrado, usando python del sistema...
    set PYTHON_EXE=python
)

echo ======================================================================
echo   PIXEL ART AI - GENERADOR RAPIDO POR RED TENSOR (U-NET PYTORCH)
echo ======================================================================
echo.

if not "%~1"=="" goto :direct_run

set /p CHAR="Nombre o ruta del personaje (ej: tori, alex, conny, amaro): "
if "%CHAR%"=="" (
    echo [ERROR] Debes especificar un personaje.
    pause
    exit /b 1
)

echo.
echo Formatos:
echo  [1] 8x12 (96 frames - Canonica Completa 1024x1536) [RECOMENDADO]
echo  [2] 16x4 (64 frames - Base chica 682x2048)
set /p FMT="Selecciona (1 o 2, defecto 1): "
set PHASE=2
if "%FMT%"=="2" set PHASE=1

"%PYTHON_EXE%" generate_character_sheet.py --input "%CHAR%" --phase "%PHASE%"
pause
exit /b 0

:direct_run
"%PYTHON_EXE%" generate_character_sheet.py %*
pause
exit /b 0
