# Check every capture before sending it to Haneef.
#
#   .\check.ps1              # check everything in ..\logs\
#   .\check.ps1 -Dir ..\logs\capture_20260911_143000

param([string]$Dir = "")

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$thermal = Split-Path -Parent $here
$vpy = Join-Path $thermal ".venv-win\Scripts\python.exe"

if (-not (Test-Path $vpy)) {
    Write-Host "No environment yet. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

if ($Dir) { & $vpy (Join-Path $here "verify_capture.py") $Dir }
else       { & $vpy (Join-Path $here "verify_capture.py") --all }
