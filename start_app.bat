@echo off
:: ============================================================
::  BertPCa STKLM0 — start Streamlit + ngrok
::  Edit PROJECT_DIR below to match the path on this machine.
:: ============================================================

set PROJECT_DIR=C:\path\to\BertPCa Extension
set PORT=8501

echo Starting Streamlit...
start "BertPCa Streamlit" cmd /k "cd /d "%PROJECT_DIR%\stklm0" && streamlit run app.py --server.port %PORT% --server.headless true"

echo Waiting for Streamlit to start...
timeout /t 5 /nobreak > nul

echo Starting ngrok...
start "ngrok tunnel" cmd /k "ngrok http 127.0.0.1:%PORT%"

echo.
echo Both windows are open.
echo Copy the https://...ngrok-free.app URL from the ngrok window and share it.
pause
