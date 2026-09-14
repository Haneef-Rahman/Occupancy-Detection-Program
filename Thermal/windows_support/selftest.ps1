# Prove the Windows support works. No camera needed.
#
#   .\selftest.ps1
#
# Run this FIRST, before plugging anything in. It exercises everything except
# the USB camera: imports, paths, the monkeypatches, writing a capture with a
# fake camera, and reading it back. ~20 seconds.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$vpy = Join-Path (Split-Path -Parent $here) ".venv-win\Scripts\python.exe"

if (-not (Test-Path $vpy)) {
    Write-Host "No environment yet. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}
& $vpy (Join-Path $here "selftest.py")
