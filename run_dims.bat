@echo off
cd /d "%~dp0"
set PYTHON="webui forger\system\python\python.exe"
%PYTHON% check_dims.py
