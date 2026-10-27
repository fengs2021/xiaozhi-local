@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"
"%PYTHON%" run.py >> "xz-out.log" 2>> "xz-err.log"
