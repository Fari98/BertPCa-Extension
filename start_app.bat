@echo off
:: ============================================================
::  BertPCa STKLM0 — start Streamlit + ngrok
:: ============================================================

set PYTHON=C:\Users\farinati.davide\AppData\Local\anaconda3\envs\hsr-gpu\python.exe
set APP=C:\Users\farinati.davide\OneDrive - NOVAIMS\Desktop\BertPCa-Extension\stklm0\app.py
set PORT=8501
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo Starting Streamlit (hsr-gpu env)...
start "BertPCa Streamlit" cmd /k ""%PYTHON%" -m streamlit run "%APP%" --server.port %PORT% --server.headless true"

echo Waiting for Streamlit to start...
timeout /t 8 /nobreak > nul

echo Starting ngrok...
start "ngrok tunnel" cmd /k "ngrok http 127.0.0.1:%PORT%"

echo.
echo Both windows are open.
echo Copy the https://...ngrok-free.app URL from the ngrok window and share it.
pause
