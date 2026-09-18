# Turn capture logs into an annotated dataset.
#
#   .\pipeline.ps1                      # all seven stages, from the start
#   .\pipeline.ps1 -From 4              # resume at annotation
#   .\pipeline.ps1 -From 4 -Live        # model-in-the-loop annotator
#
# This is the one tool here that is not about recording. Everything else in
# this folder gets frames onto disk; this turns them into labels.
#
# YOU DO NOT NEED A MODEL OR A GPU. Stages 1-4 and 6-7 never touch the network:
# you draw the boxes. Stage 5 is a cross-check against YOLO and says SKIPPED if
# there is no model, which costs you a second opinion and nothing else. Only
# -Live genuinely requires one, because it proposes the boxes for you.
#
# ANNOTATION KEYS (stage 4). Two left-clicks are OPPOSITE CORNERS, not a drag —
# on a 160x120 sensor a drag has to be held steady across ~90 screen pixels for
# a 15 px box, and any tremor moves the corner.
#
#   click, click   the two opposite corners of a head-and-shoulders box
#   hover + DEL    remove the box whose corner you are on (or press e)
#   u              undo            c   clear the frame
#   x              commit EMPTY - nobody in this cluster
#   g              commit HARD NEGATIVE - nobody here, AND the frame holds
#                  something that looks like it should fire: a hot printer,
#                  sunlit pavement, a radiator, a laptop
#   ENTER          commit and move on      b  back      d  drop the cluster
#   q              stop (your work so far is already written)
#
# `g` is the one worth caring about. The training set currently contains ZERO
# frames without a person, so the network has never once been shown a scene and
# told there is nobody in it — which is exactly why it invented people in a 3D
# printing lab. An empty corridor teaches it almost nothing. A hot printer
# certified as "no person" teaches it the thing it got wrong.

param(
    # Stage to start at, 1-7. 4 is annotation.
    [ValidateRange(1, 7)][int]$From = 1,

    # Merge directory to resume against. Default: the last one the tool made.
    [string]$Root = "",

    # Use annotate_live.py: hover a person and the model proposes the box,
    # click to accept, SPACE twice to draw one it missed. NEEDS ultralytics
    # and a model — see the check below.
    [switch]$Live,

    # Explicit weights. Default: the highest-numbered models\vN\best.pt.
    [string]$Weights = "",

    # Dataset output directory.
    [string]$Out = "",

    # Zoom factor for the annotation window. 6 suits a 160x120 sensor on a
    # laptop screen; go to 8 if the boxes feel fiddly.
    [int]$Scale = 6,

    # EVERYTHING ELSE, PASSED STRAIGHT THROUGH.
    #
    # dataset_pipeline.py takes thirteen flags and naming five of them here
    # would quietly make this a different tool from the Mac one — you would
    # reach for --sim or --max-cluster, find no -Sim, and conclude the Windows
    # side does not have it. It does. The named parameters above are only the
    # common ones; anything the parser accepts still works:
    #
    #   .\pipeline.ps1 -From 3 --sim 0.05 --max-cluster 8
    #   .\pipeline.ps1 -From 7 --val-fraction 0.2 --seed 1
    #   .\pipeline.ps1 -From 5 --conf 0.29
    #   .\pipeline.ps1 -From 4 -Live --live-conf 0.25 --dedup-iou 0.0
    #
    # Use the python-style double dash for these, not a PowerShell dash.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$thermal = Split-Path -Parent $here
$vpy = Join-Path $thermal ".venv-win\Scripts\python.exe"

if (-not (Test-Path $vpy)) {
    Write-Host "No environment yet. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

# --- captures to work on ---------------------------------------------------
$logs = Join-Path $thermal "logs"
if (-not (Test-Path $logs)) {
    Write-Host "No $logs yet - record something first with .\record.ps1" -ForegroundColor Yellow
    exit 1
}
$caps = @(Get-ChildItem -Path $logs -Directory -Filter "capture_*" -ErrorAction SilentlyContinue)
if ($caps.Count -eq 0 -and $From -le 2) {
    Write-Host "No capture_* folders in $logs - nothing to build from." -ForegroundColor Yellow
    exit 1
}
Write-Host "$($caps.Count) capture(s) in Thermal\logs" -ForegroundColor Green

# --- -Live needs things setup.ps1 deliberately does not install ------------
if ($Live) {
    $hasUl = & $vpy -c "import importlib.util,sys; sys.stdout.write('1' if importlib.util.find_spec('ultralytics') else '0')"
    if ($hasUl -ne "1") {
        Write-Host ""
        Write-Host "-Live needs ultralytics, which setup.ps1 skips on purpose." -ForegroundColor Yellow
        Write-Host "  It pulls in torch: about 2 GB, and recording never needs it."
        Write-Host ""
        Write-Host "  Install it:"
        Write-Host "    $vpy -m pip install ultralytics" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  Or drop -Live and draw the corners yourself. That needs"
        Write-Host "  nothing extra, and for frames the model gets wrong it is"
        Write-Host "  what you would end up doing anyway."
        exit 1
    }
    $models = Join-Path $thermal "models"
    $found = @(Get-ChildItem -Path $models -Directory -Filter "v*" -ErrorAction SilentlyContinue |
               Where-Object { Test-Path (Join-Path $_.FullName "best.pt") })
    if ($found.Count -eq 0 -and -not $Weights) {
        Write-Host ""
        Write-Host "-Live needs a model: it proposes the boxes with it." -ForegroundColor Yellow
        Write-Host "  Nothing at $models\vN\best.pt, and no -Weights given."
        Write-Host "  Ask Haneef for a best.pt, or drop -Live."
        exit 1
    }
}

# --- go --------------------------------------------------------------------
$a = @((Join-Path $here "w_dataset_pipeline.py"), "--from", $From,
       "--scale", $Scale)
if ($Live)    { $a += "--live" }
if ($Root)    { $a += @("--root", $Root) }
if ($Weights) { $a += @("--weights", $Weights) }
if ($Out)     { $a += @("--out", $Out) }
if ($Rest.Count) {
    $a += $Rest
    Write-Host "passing through: $($Rest -join ' ')" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "stage 4 keys:  click click = corners   DEL = delete   x = empty" -ForegroundColor Cyan
Write-Host "               g = HARD NEGATIVE   ENTER = next   b = back   q = stop" -ForegroundColor Cyan
if ($Live) {
    Write-Host "       -Live:  click = accept a proposal   SPACE SPACE = draw one" -ForegroundColor Cyan
    Write-Host "               [ ] = move the confidence gate   r = re-scan here" -ForegroundColor Cyan
}
Write-Host ""

& $vpy @a
