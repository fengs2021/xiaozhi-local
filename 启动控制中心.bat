@echo off
chcp 65001 >nul
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"

echo ================================================
echo   XiaoZhi local control center - silent launcher
echo ================================================

rem --- stop existing instance on port 8003 ---
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8003" ^| findstr "LISTENING"') do (
  echo stopping old pid %%a
  taskkill /f /pid %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

rem --- start service hidden ---
start "" /b "%PYTHON%" run.py
echo service starting...
timeout /t 12 /nobreak >nul

rem --- start cloudflare tunnel hidden if not running ---
tasklist /fi "imagename eq cloudflared.exe" | findstr /i cloudflared >nul
if errorlevel 1 (
  echo starting cloudflare tunnel...
  start "" /b "%CLOUDFLARED%" tunnel --no-autoupdate --protocol http2 run --token YOUR_TUNNEL_TOKEN
) else (
  echo cloudflare tunnel already running
)

echo.
echo done. open http://127.0.0.1:8003/
timeout /t 3 /nobreak >nul
