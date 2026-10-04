@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title Sprite Studio - Villa del Chef

:: Posicionarse siempre en el directorio del propio script .bat
cd /d "%~dp0"

echo ======================================================================
echo           SPRITE STUDIO - ESTUDIO LOCAL DE PIXEL ART
echo ======================================================================
echo.

:: 1. Buscar el interprete Python en rutas relativas al script
set "PYTHON_EXE="

if exist "%~dp0webui forger\system\python\python.exe" (
    set "PYTHON_EXE=%~dp0webui forger\system\python\python.exe"
)
if not defined PYTHON_EXE if exist "%~dp0..\webui forger\system\python\python.exe" (
    set "PYTHON_EXE=%~dp0..\webui forger\system\python\python.exe"
)
if not defined PYTHON_EXE if exist "%~dp0webui_forge_cu121_torch231\system\python\python.exe" (
    set "PYTHON_EXE=%~dp0webui_forge_cu121_torch231\system\python\python.exe"
)

:: Si no esta en carpetas relativas, buscar en PATH del sistema
if not defined PYTHON_EXE (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        if not defined PYTHON_EXE set "PYTHON_EXE=%%i"
    )
)

if not defined PYTHON_EXE (
    echo [ERROR CRITICO] No se encontro el interprete de Python.
    echo Asegurate de que la carpeta 'webui forger\system\python' exista
    echo o que Python este instalado y agregado al PATH de Windows.
    echo.
    echo Presiona cualquier tecla para salir...
    pause > nul
    exit /b 1
)

echo [OK] Interprete Python detectado: "!PYTHON_EXE!"

:: 2. Localizar el coordinador sprite_studio.py
set "STUDIO_SCRIPT="
set "STUDIO_DIR="

if exist "%~dp0sprite_studio.py" (
    set "STUDIO_DIR=%~dp0"
    set "STUDIO_SCRIPT=%~dp0sprite_studio.py"
) else if exist "%~dp0stable-diffusion-webui-forge-main\sprite_studio.py" (
    set "STUDIO_DIR=%~dp0stable-diffusion-webui-forge-main"
    set "STUDIO_SCRIPT=%~dp0stable-diffusion-webui-forge-main\sprite_studio.py"
)

if not defined STUDIO_SCRIPT (
    echo [ERROR CRITICO] No se encontro el archivo 'sprite_studio.py'.
    echo Verifique que el archivo este en la carpeta del proyecto.
    echo.
    pause
    exit /b 1
)

:: 3. Validar dependencias esenciales sin reinstalar nada
echo [*] Verificando entorno y modulos requeridos...
"!PYTHON_EXE!" -c "import torch, PIL, numpy" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [AVISO DE DEPENDENCIAS] Faltan librerias esenciales en este entorno Python.
    echo Se requiere: torch, Pillow, numpy.
    echo.
    echo Si deseas instalarlas manualmente, ejecuta:
    echo   "!PYTHON_EXE!" -m pip install Pillow numpy requests
    echo.
    echo Presiona cualquier tecla para salir...
    pause > nul
    exit /b 1
)

echo [OK] Dependencias verificadas con exito.
echo.
echo ======================================================================
echo   Iniciando Sprite Studio...
echo   URL de red: http://192.168.1.83:8080/sprite_studio.html
echo   La aplicacion abrira automaticamente la interfaz en tu navegador.
echo   Para detenerla, cierra esta ventana o presiona Ctrl + C.
echo ======================================================================
echo.

cd /d "!STUDIO_DIR!"
set "SPRITE_STUDIO_HOST=192.168.1.83"
"!PYTHON_EXE!" "!STUDIO_SCRIPT!"

if errorlevel 1 (
    echo.
    echo [AVISO] Sprite Studio finalizo con codigo de error.
    pause
)

exit /b 0
