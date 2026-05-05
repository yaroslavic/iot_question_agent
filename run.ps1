Set-Location -Path $PSScriptRoot

$PythonCmd = $null
$PythonArgs = @()

if (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PythonCmd = "python3"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCmd = "py"
    $PythonArgs = @("-3")
}

if (-not $PythonCmd) {
    Write-Host "[ERROR] Python was not found."
    Write-Host "Install Python 3.10+ from https://www.python.org/downloads/windows/"
    Write-Host "During installation enable: Add python.exe to PATH"
    exit 1
}

Write-Host "[SETUP] Using Python command: $PythonCmd $($PythonArgs -join ' ')"
& $PythonCmd @PythonArgs --version
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Test-Path ".venv")) {
    Write-Host "[SETUP] Creating Python virtual environment..."
    & $PythonCmd @PythonArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "[ERROR] Virtual environment was not created correctly."
    Write-Host "Delete the .venv folder and run run.ps1 again."
    exit 1
}

Write-Host "[SETUP] Installing/updating dependencies..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[RUN] Starting IOT question agent..."
.\.venv\Scripts\python.exe -m src.iot_question_agent.main --config config.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[OK] Finished. Check output folder."
