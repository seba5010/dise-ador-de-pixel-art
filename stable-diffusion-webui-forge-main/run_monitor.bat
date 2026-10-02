@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Servidor de Monitoreo Web - Pixel Art AI
echo Iniciando servidor de monitoreo en http://localhost:8787/monitor.html ...
start http://localhost:8787/monitor.html
"..\webui forger\system\python\python.exe" monitor_server.py
pause
