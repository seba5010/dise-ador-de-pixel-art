@echo off
setlocal
cd /d "%~dp0"
set PYTHON_EXE=webui forger\system\python\python.exe
if not exist "%PYTHON_EXE%" (
    echo Python de Forge no encontrado, utilizando python del sistema...
    set PYTHON_EXE=python
)

echo.
echo ================================================================
echo   PIXEL ART AI -- ENTRENAMIENTO INFINITO + MONITOR EN VIVO
echo ================================================================
echo.
echo   - El modelo entrena hasta que cierres esta ventana o Ctrl+C
echo   - Los checkpoints se guardan cada 10 epocas automaticamente
echo   - El mejor modelo se guarda en: checkpoints\best_generator.pt
echo   - Monitor: http://localhost:8787/monitor.html
echo.
echo ================================================================
echo.

:: Iniciar el servidor del monitor en ventana separada
start "Monitor Pixel AI" "%PYTHON_EXE%" monitor_server.py
timeout /t 2 /nobreak >nul

echo Iniciando entrenamiento infinito desde cero...
echo Presiona Ctrl+C en cualquier momento para detener (guardara el estado).
echo.

"%PYTHON_EXE%" train.py --infinite --batch-size 16

echo.
echo ================================================================
echo   Entrenamiento detenido. Checkpoint guardado.
echo   Para reanudar, vuelve a ejecutar este bat.
echo ================================================================
echo.
pause
