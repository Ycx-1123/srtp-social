param(
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot
$env:SOCI_MODE = "edge"
$env:SOCI_ALLOW_NETWORK = "0"
$env:SOCI_HOST = "127.0.0.1"
$env:SOCI_PORT = "8001"
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot "data\ultralytics"
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCommand) {
    Write-Host "[SOCI-AI] Python was not found. Install Python 3.11 or newer." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

& $PythonCommand.Source -c "import fastapi, pydantic, uvicorn, websockets, cv2, numpy, ultralytics, importlib.util; assert importlib.util.find_spec('mediapipe')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[SOCI-AI] Runtime dependencies are missing. Run:" -ForegroundColor Yellow
    Write-Host "python -m pip install -r requirements-live.txt" -ForegroundColor Cyan
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[SOCI-AI] Local realtime capability check:" -ForegroundColor Cyan
& $PythonCommand.Source -c "from pathlib import Path; from soci_ai.config import Settings; from soci_ai.live.capabilities import print_capabilities; print_capabilities(Settings.from_env(Path.cwd()))"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[SOCI-AI] Capability check failed." -ForegroundColor Red
    exit 1
}
if ($CheckOnly) {
    Write-Host "[SOCI-AI] Capability check complete." -ForegroundColor Green
    exit 0
}

Write-Host "[SOCI-AI] Starting local realtime self-check..." -ForegroundColor Cyan
$Server = Start-Process -FilePath $PythonCommand.Source -ArgumentList "-m", "soci_ai" -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru

$Ready = $false
for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
    try {
        $Health = Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/health" -TimeoutSec 1
        if ($Health.status -eq "ok") { $Ready = $true; break }
    } catch {
        Start-Sleep -Milliseconds 300
    }
}

if (-not $Ready) {
    if (-not $Server.HasExited) { Stop-Process -Id $Server.Id }
    Write-Host "[SOCI-AI] Service did not become ready. Run python -m soci_ai to inspect logs." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$DataDirectory = Join-Path $ProjectRoot "data"
New-Item -ItemType Directory -Force -Path $DataDirectory | Out-Null
$Server.Id | Set-Content -LiteralPath (Join-Path $DataDirectory "soci_ai.pid") -Encoding ascii
Write-Host "[SOCI-AI] Ready: http://127.0.0.1:8001" -ForegroundColor Green
try {
    Start-Process "http://127.0.0.1:8001" -ErrorAction Stop
} catch {
    Write-Host "Open this address in your browser: http://127.0.0.1:8001" -ForegroundColor Yellow
}
Write-Host "Closing the browser does not stop the service. Stop its Python process or restart the computer when finished."
