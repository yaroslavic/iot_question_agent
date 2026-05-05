@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="

where python >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=python"

if not defined PYTHON_CMD (
    where python3 >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python3"
)

if not defined PYTHON_CMD (
    where py >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD (
    echo [ERROR] Python was not found.
    echo Install Python 3.10+ from https://www.python.org/downloads/windows/
    echo During installation enable: Add python.exe to PATH
    goto error
)

echo [SETUP] Using Python command: %PYTHON_CMD%
%PYTHON_CMD% --version
if errorlevel 1 goto error

if not exist .venv (
    echo [SETUP] Creating Python virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto error
)

if not exist .venv\Scripts\python.exe (
    echo [ERROR] Virtual environment was not created correctly.
    echo Delete the .venv folder and run run.bat again.
    goto error
)

echo [SETUP] Installing/updating dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto error
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto error

echo [RUN] Starting IOT question agent...
.venv\Scripts\python.exe -m src.iot_question_agent.main --config config.yaml
if errorlevel 1 goto error

echo [OK] Finished. Check output folder.
pause
exit /b 0

:error
echo [ERROR] Failed. See messages above.
pause
exit /b 1
