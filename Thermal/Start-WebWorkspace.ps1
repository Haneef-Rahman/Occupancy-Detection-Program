# Run from any directory; the full Thermal folder and local weights are required.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv-web\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
if (Test-Path -LiteralPath $python) {
    & $python (Join-Path $PSScriptRoot 'web_workspace.py') @args
} else {
    & py -3.12 (Join-Path $PSScriptRoot 'web_workspace.py') @args
}
exit $LASTEXITCODE
