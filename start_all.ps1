$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

Write-Host "Starting API..."
Start-Process -NoNewWindow -FilePath "uvicorn" -ArgumentList "--factory", "ridm_ultra.api:create_app", "--port", "8000"

Start-Sleep -Seconds 3

Write-Host "Starting UI..."
Start-Process -NoNewWindow -FilePath "streamlit" -ArgumentList "run", "ridm_ultra/ui/app.py", "--server.port", "8501"

Write-Host "All services started."
while ($true) {
    Start-Sleep -Seconds 60
}
