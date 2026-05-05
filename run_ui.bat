@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON_CMD="
where python >nul 2>nul && set "PYTHON_CMD=python"
if not defined PYTHON_CMD where python3 >nul 2>nul && set "PYTHON_CMD=python3"
if not defined PYTHON_CMD where py >nul 2>nul && set "PYTHON_CMD=py -3"

if not defined PYTHON_CMD (
  echo [ERROR] Python 3.10+ not found. Install Python and enable "Add python.exe to PATH".
  pause
  exit /b 1
)

if not exist .venv\Scripts\python.exe (
  echo [SETUP] Creating Python virtual environment...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 goto fail
)

echo [SETUP] Installing/updating dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo [START] Opening local UI at http://127.0.0.1:8000
set "PYTHONPATH=%CD%\src"
start "" http://127.0.0.1:8000
.venv\Scripts\python.exe -m uvicorn iot_question_agent.ui_app:app --host 127.0.0.1 --port 8000
if errorlevel 1 goto fail
exit /b 0

:fail
echo [ERROR] Failed. See messages above.
pause
exit /b 1
