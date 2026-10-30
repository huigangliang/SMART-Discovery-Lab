@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 goto :setup_error
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :setup_error
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
pause
exit /b
:setup_error
echo Setup did not complete. Please keep this window open to read the error.
echo Python 3.12 and an internet connection for package installation are required.
pause
exit /b 1
