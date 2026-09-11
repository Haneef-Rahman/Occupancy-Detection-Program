# Live preview with detections. Records nothing — use it to aim the camera.
#
#   .\preview.ps1
#   .\preview.ps1 -Conf 0.45
#
# Needs a model in Thermal\models\vN\best.pt and ultralytics installed.
# Neither is needed just to collect data.

param([double]$Conf = 0, [int]$Device = 0, [switch]$Tracker)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$vpy = Join-Path (Split-Path -Parent $here) ".venv-win\Scripts\python.exe"

if (-not (Test-Path $vpy)) {
    Write-Host "No environment yet. Run .\setup.ps1 first." -ForegroundColor Red; exit 1
}

$script = if ($Tracker) { "w_integrated_launcher.py" } else { "w_live_yolo.py" }
$a = @((Join-Path $here $script), "--device", $Device)
if ($Conf -gt 0) { $a += @("--conf", $Conf) }
if ($Tracker)    { $a += @("--mode", "yolo") }

& $vpy @a
