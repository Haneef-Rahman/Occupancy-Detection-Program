# One-time setup for FLUXNET thermal capture on Windows.
#
#   .\setup.ps1
#
# Creates a virtual environment, installs what is needed, and checks the
# camera. Run it once. If anything goes wrong it says what to do rather than
# just failing.
#
# If PowerShell refuses to run this ("running scripts is disabled"), do:
#     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# and run it again. That lasts for this window only and changes nothing
# permanently.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$thermal = Split-Path -Parent $here

Write-Host ""
Write-Host "FLUXNET thermal capture - Windows setup" -ForegroundColor Cyan
Write-Host ("=" * 50)

# --- 1. find a real Python -------------------------------------------------
# The Microsoft Store ships a fake python.exe that sits FIRST on PATH and has
# no packages. It is the single most common reason "pip installed it but
# Python can't find it". The py launcher ignores it entirely, so we use that.
Write-Host "`n[1/4] Looking for Python..."
$py = $null
foreach ($v in @("-3.12", "-3.11", "-3.10", "-3")) {
    try {
        $ver = & py $v -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver) { $py = $v; break }
    } catch { }
}
if (-not $py) {
    Write-Host "  No Python found via the 'py' launcher." -ForegroundColor Red
    Write-Host "  Install Python 3.12 from python.org (NOT the Microsoft Store"
    Write-Host "  version), tick 'Add python.exe to PATH', then re-run this."
    exit 1
}
Write-Host "  found Python $ver (py $py)" -ForegroundColor Green

# --- 2. virtual environment ------------------------------------------------
$venv = Join-Path $thermal ".venv-win"
Write-Host "`n[2/4] Virtual environment..."
if (Test-Path (Join-Path $venv "Scripts\python.exe")) {
    Write-Host "  already exists at $venv" -ForegroundColor Green
} else {
    & py $py -m venv $venv
    Write-Host "  created $venv" -ForegroundColor Green
}
$vpy = Join-Path $venv "Scripts\python.exe"

# --- 3. packages -----------------------------------------------------------
# Deliberately NOT installing torch/ultralytics. Recording does not need them:
# record_win.py writes YOLO labels only if a model is present, and Adrian is
# collecting data, not training. Keeping the install small also keeps it fast
# and avoids the CPU-vs-CUDA torch confusion entirely.
Write-Host "`n[3/4] Packages (numpy, opencv)..."
& $vpy -m pip install --quiet --upgrade pip
& $vpy -m pip install --quiet -r (Join-Path $here "requirements-win.txt")
$check = & $vpy -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
Write-Host "  opencv $($check.Split()[0]), numpy $($check.Split()[1])" -ForegroundColor Green

# --- 4. camera -------------------------------------------------------------
Write-Host "`n[4/4] Camera check..."
Write-Host "  Plug in the PureThermal now, and close anything else that might"
Write-Host "  be using it (Teams, OBS, browser tabs, the FLIR app)."
Read-Host "  Press Enter when ready"
& $vpy (Join-Path $here "backend_probe.py")

Write-Host "`n$("=" * 50)"
Write-Host "Setup done." -ForegroundColor Cyan
Write-Host ""
Write-Host "To record:"
Write-Host "  .\record.ps1 -Operator adrian -Note `"cafeteria, 1430, 26C`""
Write-Host ""
Write-Host "After each session:"
Write-Host "  .\check.ps1"
Write-Host ""
