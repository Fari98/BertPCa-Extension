@echo off
:: ============================================================
::  BertPCa — Evaluate STKLM0 model on Milan (OSR) data
:: ============================================================
set PYTHON=C:\Users\farinati.davide\AppData\Local\anaconda3\envs\hsr-gpu\python.exe
set OUTFILE=%~dp0stklm0\scripts\eval_milan_output.txt
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo Running STKLM0 -> Milan (OSR) evaluation...
echo Output: %OUTFILE%

"%PYTHON%" "%~dp0stklm0\scripts\eval_milan.py" --outcome csm > "%OUTFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%OUTFILE%"

echo Done. Check %OUTFILE%
pause
