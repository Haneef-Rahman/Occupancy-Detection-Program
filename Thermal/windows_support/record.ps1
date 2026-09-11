# Record a capture session.
#
#   .\record.ps1 -Operator adrian -Note "cafeteria, 1430, 26C"
#   .\record.ps1 -Operator adrian -Note "cafeteria EMPTY, 1445, 26C"
#
# Keys in the window:  r = start/stop recording,  q = quit
# Captures land in Thermal\logs\ — the same place the Mac writes them.
#
# No administrator needed. The Mac needs sudo because macOS blocks camera
# access for libusb; Windows does not. If something asks for admin, it isn't this.

param(
    [string]$Operator = "",
    [string]$Note = "",
    [int]$Device = 0,
    [switch]$AllowAgc
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$vpy = Join-Path (Split-Path -Parent $here) ".venv-win\Scripts\python.exe"

if (-not (Test-Path $vpy)) {
    Write-Host "No environment yet. Run .\setup.ps1 first." -ForegroundColor Red; exit 1
}
if (-not $Operator) {
    Write-Host "Pass -Operator so captures can be traced to you:" -ForegroundColor Yellow
    Write-Host '  .\record.ps1 -Operator adrian -Note "where, when, ambient"'; exit 1
}
if (-not $Note) {
    Write-Host "Pass -Note describing the scene. Include the ambient temperature —" -ForegroundColor Yellow
    Write-Host "it is the one thing that cannot be recovered afterwards."
    Write-Host '  .\record.ps1 -Operator adrian -Note "cafeteria, 1430, 26C"'; exit 1
}

$a = @((Join-Path $here "w_dataset_recording.py"),
       "--operator", $Operator, "--note", $Note, "--device", $Device)
if ($AllowAgc) { $a += "--allow-agc" }
& $vpy @a
