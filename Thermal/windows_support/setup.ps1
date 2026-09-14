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

param(
    # Throw the existing environment away and build it again. Use this if
    # setup reports the venv is on the wrong Python version.
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$thermal = Split-Path -Parent $here

# WHY 3.12 AND NOT NEWER. Recording needs only numpy and opencv and would run
# on anything. But preview.ps1 and the two YOLO tools need ultralytics, which
# drags in torch, and torch wheels lag new Python releases by months. 3.12 is
# the newest version the whole stack is known good on. Newer is not better
# here; it is the difference between preview.ps1 working and not.
$PREFERRED = "3.12"

Write-Host ""
Write-Host "FLUXNET thermal capture - Windows setup" -ForegroundColor Cyan
Write-Host ("=" * 50)

# --- 1. find a real Python -------------------------------------------------
# The Microsoft Store ships a fake python.exe that sits FIRST on PATH and has
# no packages. It is the single most common reason "pip installed it but
# Python can't find it". The py launcher ignores it entirely, so we use that.
Write-Host "`n[1/4] Looking for Python $PREFERRED..."
$py = $null
$ver = $null
foreach ($v in @("-3.12", "-3.11", "-3.10", "-3")) {
    try {
        $got = & py $v -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>$null
        if ($LASTEXITCODE -eq 0 -and $got) { $py = $v; $ver = $got.Trim(); break }
    } catch { }
}
if (-not $py) {
    Write-Host "  No Python found via the 'py' launcher." -ForegroundColor Red
    Write-Host "  Install Python $PREFERRED from python.org (NOT the Microsoft"
    Write-Host "  Store version), tick 'Add python.exe to PATH', then re-run."
    exit 1
}
if ($ver -eq $PREFERRED) {
    Write-Host "  found Python $ver (py $py)" -ForegroundColor Green
} else {
    Write-Host "  found Python $ver - wanted $PREFERRED" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  RECORDING WILL WORK. numpy and opencv are all it needs." -ForegroundColor Yellow
    Write-Host "  The LIVE PREVIEW WILL NOT. preview.ps1, w_live_yolo.py and"
    Write-Host "  w_integrated_launcher.py need ultralytics, which needs torch,"
    Write-Host "  and torch has no wheels for Python $ver yet."
    Write-Host ""
    Write-Host "  If you want the preview: install Python $PREFERRED from"
    Write-Host "  python.org, then run  .\setup.ps1 -Rebuild"
    Write-Host ""
}

# --- 2. virtual environment ------------------------------------------------
$venv = Join-Path $thermal ".venv-win"
$vpy = Join-Path $venv "Scripts\python.exe"
Write-Host "`n[2/4] Virtual environment..."

if ($Rebuild -and (Test-Path $venv)) {
    Write-Host "  removing the old environment (-Rebuild)..."
    Remove-Item -Recurse -Force $venv
}

if (Test-Path $vpy) {
    # An environment built earlier may be on a different Python than the one
    # just found - that is exactly how you end up unable to install
    # ultralytics into a venv that setup keeps reporting as fine.
    $vver = (& $vpy -c "import sys; print('.'.join(map(str, sys.version_info[:2])))").Trim()
    if ($vver -eq $ver) {
        Write-Host "  already exists at $venv (Python $vver)" -ForegroundColor Green
    } else {
        Write-Host "  exists, but it is on Python $vver while setup just found $ver." -ForegroundColor Yellow
        Write-Host "  Rebuild it with:  .\setup.ps1 -Rebuild"
    }
} else {
    & py $py -m venv $venv
    Write-Host "  created $venv (Python $ver)" -ForegroundColor Green
}

# --- 3. packages -----------------------------------------------------------
# Deliberately NOT installing torch/ultralytics. Recording does not need them:
# record_win.py writes YOLO labels only if a model is present, and Adrian is
# collecting data, not training. Keeping the install small also keeps it fast
# and avoids the CPU-vs-CUDA torch confusion entirely.
#
# If you DO want the live preview later, on Python 3.12:
#     ..\.venv-win\Scripts\python.exe -m pip install ultralytics
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
if ($ver -ne $PREFERRED) {
    Write-Host "Reminder: this environment is Python $ver, so preview.ps1 and" -ForegroundColor Yellow
    Write-Host "the YOLO tools will not install. Recording is unaffected." -ForegroundColor Yellow
    Write-Host ""
}
