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
echo ======================================================================
echo  [1] FASE 1: Entrenar Base Chica (16x4 / 64 frames - Conny y Dana)
echo  [2] FASE 2: Transfer Learning a Plantilla 8x12 (96 frames - Alex y Amaro)
echo  [3] Abrir Monitor Web en Vivo (http://localhost:8787/monitor.html)
echo  [4] Salir
echo ======================================================================
set /p opt="Elige una opcion (1-4): "

if "%opt%"=="1" goto fase1
if "%opt%"=="2" goto fase2
if "%opt%"=="3" goto monitor
if "%opt%"=="4" goto end
goto menu

:fase1
cls
echo ======================================================================
echo   INICIANDO FASE 1: BASE CON PLANTILLA CHICA (16x4 / 64 FRAMES)
echo   Entrenando con Conny y Dana (Modo Infinito - Ctrl+C para detener)
echo   Pesos se guardaran en: checkpoints\base_generator_16x4.pt
echo ======================================================================
"%PYTHON_EXE%" train.py --phase 1 --infinite
pause
goto menu

:fase2
cls
echo ======================================================================
echo   INICIANDO FASE 2: TRANSFER LEARNING A PLANTILLA 8x12 (96 FRAMES)
echo   Cargando pesos de Fase 1 hacia Alex y Amaro
echo   Pesos se guardaran en: checkpoints\best_generator.pt
echo ======================================================================
"%PYTHON_EXE%" train.py --phase 2 --infinite
pause
goto menu

:monitor
start "" "%PYTHON_EXE%" monitor_server.py
goto menu

:end
