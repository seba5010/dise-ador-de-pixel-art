@echo off
setlocal
cd /d "%~dp0"
set PYTHON_EXE=webui forger\system\python\python.exe
if not exist "%PYTHON_EXE%" (
    echo Python de Forge no encontrado, utilizando python del sistema...
    set PYTHON_EXE=python
)

if not "%1"=="" (
    "%PYTHON_EXE%" train.py %*
    goto end
)

:menu
cls
echo ======================================================================
echo          PIXEL ART AI ENGINE - PANEL DE ENTRENAMIENTO
echo          Dataset: TODOS LOS PERSONAJES (dataset_frames_individuales)
echo ======================================================================
echo  [1] PREPARAR DATASET - Generar cache desde dataset_frames_individuales
echo  [2] ENTRENAR desde cero (--mode start) - Todos los personajes
echo  [3] REANUDAR entrenamiento existente (--mode resume)
echo  [4] Abrir Monitor Web en Vivo (http://localhost:8787/monitor.html)
echo  [5] Salir
echo ======================================================================
set /p opt="Elige una opcion (1-5): "

if "%opt%"=="1" goto preparar
if "%opt%"=="2" goto entrenar_nuevo
if "%opt%"=="3" goto reanudar
if "%opt%"=="4" goto monitor
if "%opt%"=="5" goto end
goto menu

:preparar
cls
echo ======================================================================
echo   PREPARANDO DATASET COMPLETO
echo   Fuente: dataset_frames_individuales/ (todos los personajes)
echo   Destino: dataset_supervisado/
echo ======================================================================
"%PYTHON_EXE%" -m pixel_ai_engine.prepare_supervised_dataset
echo.
echo [OK] Dataset listo. Ya puedes iniciar el entrenamiento.
pause
goto menu

:entrenar_nuevo
cls
echo ======================================================================
echo   INICIANDO ENTRENAMIENTO NUEVO DESDE CERO
echo   Dataset: TODOS los personajes de dataset_frames_individuales
echo   Checkpoints: checkpoints\latest_checkpoint.pt
echo ======================================================================
"%PYTHON_EXE%" train.py --mode start --epochs 200
pause
goto menu

:reanudar
cls
echo ======================================================================
echo   REANUDANDO ENTRENAMIENTO EXISTENTE
echo   Continuando desde ultimo checkpoint guardado
echo ======================================================================
"%PYTHON_EXE%" train.py --mode resume --epochs 200
pause
goto menu

:monitor
start "" "%PYTHON_EXE%" monitor_server.py
goto menu

:end