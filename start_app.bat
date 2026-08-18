@echo off
:: ============================================================
::  BertPCa STKLM0 — Comparison App + ngrok
:: ============================================================

set PYTHON=C:\Users\farinati.davide\AppData\Local\anaconda3\envs\hsr-gpu\python.exe
set PORT=8501
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

:: Kill anything already on the port
echo Killing any process on port %PORT%...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%PORT% "') do (
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo Starting Streamlit comparison app (hsr-gpu env)...
start "BertPCa Compare" cmd /k ""%PYTHON%" -m streamlit run "C:\Users\farinati.davide\OneDrive - NOVAIMS\Desktop\BertPCa-Extension\stklm0\compare_app.py" --server.port %PORT% --server.headless true"

echo Waiting for Streamlit to start...
timeout /t 8 /nobreak >nul

echo Starting ngrok...
start "ngrok tunnel" cmd /k "ngrok http 127.0.0.1:%PORT%"

echo.
echo Both windows are open.
echo Copy the https://...ngrok-free.app URL from the ngrok window and share it.
pause
