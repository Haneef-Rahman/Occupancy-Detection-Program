# Find out whether your Lepton gives OpenCV real temperatures.
#
#   .\probe.ps1
#
# Run with the camera plugged in, before recording anything. Uses the venv
# Python, so it does not matter which Python versions are on your PATH.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$vpy = Join-Path (Split-Path -Parent $here) ".venv-win\Scripts\python.exe"

if (-not (Test-Path $vpy)) {
    Write-Host "No environment yet. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}
& $vpy (Join-Path $here "backend_probe.py")
