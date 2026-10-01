@echo off
chcp 65001 > nul
title Pulido de Alta Definicion - Fase 1 (Epocas 500 a 800)

echo ======================================================================
echo   PIXEL ART AI ENGINE - PULIDO DE ALTA DEFINICIÓN FASE 1
echo   - Reanudando desde epoca 500 (Modelo perfecto al 90.1%%)
echo   - Enfoque: Texturizado de 1px, bordes duros y micro-detalles
echo   - Peso Adversarial protegido (0.15) anti-colapso
echo ======================================================================
echo.
echo Monitor web en vivo disponible en: http://localhost:8787/monitor.html
echo.

set PYTHON_EXE="webui forger\system\python\python.exe"

if not exist %PYTHON_EXE% (
    echo [ERROR] No se encontro el interprete de Python en %PYTHON_EXE%
    pause
    exit /b 1
)

%PYTHON_EXE% pulir_fase1.py

echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
