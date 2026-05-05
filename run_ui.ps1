$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Python {
    foreach ($cmd in @("python", "python3")) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found) { return $cmd }
    }
    $py = Get-Command "py" -ErrorAction SilentlyContinue
    if ($py) { return "py -3" }
    throw "Python 3.10+ not found. Install Python and enable Add python.exe to PATH."
}

$pythonCmd = Find-Python
if (!(Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[SETUP] Creating Python virtual environment..."
    Invoke-Expression "$pythonCmd -m venv .venv"
}

Write-Host "[SETUP] Installing/updating dependencies..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "[START] Opening local UI at http://127.0.0.1:8000"
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
Start-Process "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m uvicorn iot_question_agent.ui_app:app --host 127.0.0.1 --port 8000
