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
echo   PIXEL ART AI - ENTRENAMIENTO DE FASE 2 (ESTRUCTURA 8X12 - 96 POSES)
echo ======================================================================
echo   - Plantilla Canonica: 1024x1536 px (8 columnas x 12 filas)
echo   - Base de Transferencia: base_generator_16x4.pt (90.1%% calidad)
echo   - Proteccion termica adaptativa dual (GPU / CPU)
echo   - Perdida de silueta estricta (IoU + L1) anti-deformacion
echo   - Minibatch StdDev anti-colapso
echo ======================================================================
echo.

if not "%~1"=="" goto :direct_run

echo Selecciona opcion:
echo  [1] Entrenar Fase 2 (250 epocas - Punto Optimo de Calidad) [RECOMENDADO]
echo  [2] Reanudar Fase 2 desde ultimo checkpoint (--resume)
echo  [3] Entrenar con cantidad personalizada de epocas
echo  [4] Entrenar modo continuo/infinito (hasta presionar Ctrl+C)
set /p OPT="Selecciona (1, 2, 3 o 4, defecto 1): "

if "%OPT%"=="2" (
    echo Reanudando desde ultimo checkpoint de Fase 2...
    "%PYTHON_EXE%" train.py --phase 2 --resume --epochs 250 --batch-size 8
) else if "%OPT%"=="3" (
    set /p EP="Cantidad de epocas deseadas: "
    "%PYTHON_EXE%" train.py --phase 2 --epochs %EP% --batch-size 8
) else if "%OPT%"=="4" (
    echo Modo infinito: presiona Ctrl+C para detener y guardar cuando desees.
    "%PYTHON_EXE%" train.py --phase 2 --infinite --batch-size 8
) else (
    echo Iniciando entrenamiento optimo de Fase 2 (250 epocas)...
    "%PYTHON_EXE%" train.py --phase 2 --epochs 250 --batch-size 8
)

echo.
echo Presiona cualquier tecla para salir...
pause > nul
exit /b 0

:direct_run
"%PYTHON_EXE%" train.py %*
pause
exit /b 0
