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
echo   PIXEL ART AI - GENERADOR POR DIFUSION (SD 1.5 + CONTROLNET + LORA)
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
set FORMAT=8x12
if "%FMT%"=="2" set FORMAT=16x4

echo.
echo ControlNet:
echo  [1] lineart (Recomendado: mejor integracion anatomica)
echo  [2] canny
set /p CTRL="Selecciona (1 o 2, defecto 1): "
set CONTROL=lineart
if "%CTRL%"=="2" set CONTROL=canny

echo.
echo Modo:
echo  [1] TEST (4 poses clave en tu formato)
echo  [2] Completo (todos los frames y ensamblar hoja final)
echo  [3] Reanudar (--resume)
set /p MOD="Selecciona (1, 2 o 3, defecto 1): "
set EXTRA=--test
if "%MOD%"=="2" set EXTRA=
if "%MOD%"=="3" set EXTRA=--resume

"%PYTHON_EXE%" forge_reference_pipeline.py --reference "%CHAR%" --format "%FORMAT%" --control "%CONTROL%" %EXTRA%
pause
exit /b 0

:direct_run
"%PYTHON_EXE%" forge_reference_pipeline.py %*
pause
exit /b 0
