@echo off
setlocal
cd /d "%~dp0"

set PYTHON_EXE=webui forger\system\python\python.exe
if not exist "%PYTHON_EXE%" (
    echo Python de Forge no encontrado, utilizando python del sistema...
    set PYTHON_EXE=python
)

echo ======================================================================
echo   PIXEL ART AI - GENERADOR CON STABLE DIFFUSION + CONTROLNET (MOTOR B)
echo ======================================================================
echo   Pipeline de referencia visual img2img asistido por ControlNet
echo   No altera ni sobrescribe el motor neuronal PyTorch existente (Motor A)
echo ======================================================================
echo.

if not "%~1"=="" (
    set CHAR_ARG=%~1
    set FMT_ARG=16x4
    set CTRL_ARG=lineart
    if not "%~2"=="" set FMT_ARG=%~2
    if not "%~3"=="" set CTRL_ARG=%~3
    
    echo Ejecutando directamente:
    "%PYTHON_EXE%" forge_reference_pipeline.py --reference "%~1" --format "%~2" --control "%~3" %4 %5 %6 %7 %8 %9
    pause
    exit /b 0
)

echo Paso 1: Identidad del Personaje
echo ----------------------------------------------------------------------
echo Ingresa el nombre del personaje o la ruta a su imagen frontal:
echo (Ejemplos: tori, alex, conny, amaro, dana o personajes\tori\tori_rnormal.png)
set /p CHAR_INPUT="Nombre o ruta del personaje: "

if "%CHAR_INPUT%"=="" (
    echo [ERROR] Debes especificar un personaje. Operacion cancelada.
    pause
    exit /b 1
)

echo.
echo Paso 2: Formato de Spritesheet
echo ----------------------------------------------------------------------
echo  [1] 16x4 (64 frames - 4 columnas x 16 filas)
echo  [2] 8x12 (96 frames - 8 columnas x 12 filas - Canonica completa)
set /p FMT_CHOICE="Selecciona formato (1 o 2, defecto 1): "

set FORMAT=16x4
if "%FMT_CHOICE%"=="2" set FORMAT=8x12

echo.
echo Paso 3: Guia de Pose (ControlNet)
echo ----------------------------------------------------------------------
echo  [1] Lineart (Recomendado: mejor integracion anatomica y ropa)
echo  [2] Canny   (Bordes duros y silueta estricta)
set /p CTRL_CHOICE="Selecciona ControlNet (1 o 2, defecto 1): "

set CONTROL=lineart
if "%CTRL_CHOICE%"=="2" set CONTROL=canny

echo.
echo Paso 4: Modo de Ejecucion
echo ----------------------------------------------------------------------
echo  [1] Modo TEST (Genera solo 4 poses clave: frente, espalda, lateral y accion)
echo      * Altamente recomendado para verificar identidad antes de la hoja completa.
echo  [2] Generacion Completa (Procesar todos los frames y ensamblar spritesheet)
echo  [3] Reanudar Generacion (--resume: omite los frames ya generados validos)
set /p MODE_CHOICE="Selecciona modo (1, 2 o 3, defecto 1): "

set EXTRA_FLAGS=--test
if "%MODE_CHOICE%"=="2" set EXTRA_FLAGS=
if "%MODE_CHOICE%"=="3" set EXTRA_FLAGS=--resume

echo.
echo ======================================================================
echo   RESUMEN DE GENERACION:
echo     - Personaje : %CHAR_INPUT%
echo     - Formato   : %FORMAT%
echo     - ControlNet: %CONTROL%
echo     - Flags     : %EXTRA_FLAGS%
echo ======================================================================
echo.

"%PYTHON_EXE%" forge_reference_pipeline.py --reference "%CHAR_INPUT%" --format "%FORMAT%" --control "%CONTROL%" %EXTRA_FLAGS%

echo.
echo Presiona cualquier tecla para finalizar...
pause > nul
