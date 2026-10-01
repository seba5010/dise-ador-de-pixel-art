@echo off
setlocal
cd /d "%~dp0"
set PYTHON_EXE=webui forger\system\python\python.exe
if not exist "%PYTHON_EXE%" (
    echo Python de Forge no encontrado, utilizando python del sistema...
    set PYTHON_EXE=python
)

echo ========================================================
echo   PIXEL ART AI - GENERADOR DE HOJA DE SPRITES (16x4)
echo ========================================================
if "%~1"=="" (
    echo Uso: run_generate.bat nombre_o_ruta_personaje [ruta_salida.png]
    echo.
    echo Ejemplos:
    echo   run_generate.bat mauricio
    echo   run_generate.bat andres_arica
    echo   run_generate.bat camilo
    echo   run_generate.bat personajes\mauricio\mauricio_rnormal.png
    pause
    exit /b 1
)

"%PYTHON_EXE%" generate_character_sheet.py --input "%~1" --output "%~2"
pause
