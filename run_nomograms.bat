@echo off
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
set OUTFILE=%~dp0stklm0\scripts\run_nomograms_output.txt

:: Try hsr-gpu env first, fall back to system python
set PYTHON=C:\Users\dfarinati\AppData\Local\anaconda3\envs\hsr-gpu\python.exe
if not exist "%PYTHON%" set PYTHON=C:\Users\farinati.davide\AppData\Local\anaconda3\envs\hsr-gpu\python.exe
if not exist "%PYTHON%" (
    for /f "tokens=*" %%i in ('where python 2^>nul') do set PYTHON=%%i
)

echo Python: %PYTHON% > "%OUTFILE%"
echo. >> "%OUTFILE%"
"%PYTHON%" "%~dp0stklm0\scripts\run_nomograms.py" >> "%OUTFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%OUTFILE%"
