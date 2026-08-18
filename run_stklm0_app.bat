@echo off
cd /d "%~dp0"
python -m streamlit run stklm0/compare_app.py %*
