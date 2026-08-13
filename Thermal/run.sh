#!/usr/bin/env bash
# Launch the thermal tools with the venv's Python under sudo.
#
# Why sudo: on macOS the kernel UVC driver claims the camera, so libusb
# cannot claim the interface without elevated privileges. On Linux/Raspberry
# Pi this is not needed (a udev rule grants access instead).
#
# Usage:
#   ./run.sh                    -> thermal_detect.py
#   ./run.sh lepton_libuvc.py   -> any script in this folder
#   ./run.sh thermal_detect.py --note "2 abreast"

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find a usable interpreter: an active venv, then ./.venv, then ../.venv,
# then whatever python3 is on PATH.
VENV_PY=""
CANDIDATES=()
[ -n "${VIRTUAL_ENV:-}" ] && CANDIDATES+=("$VIRTUAL_ENV/bin/python")
CANDIDATES+=("$HERE/.venv/bin/python" "$(cd "$HERE/.." && pwd)/.venv/bin/python")
for cand in "${CANDIDATES[@]}"; do
    if [ -x "$cand" ]; then VENV_PY="$cand"; break; fi
done
if [ -z "$VENV_PY" ]; then
    VENV_PY="$(command -v python3 || true)"
    if [ -z "$VENV_PY" ]; then
        echo "No Python found. Create a venv:"
        echo "  python3 -m venv .venv && source .venv/bin/activate"
        echo "  pip install numpy opencv-python"
        exit 1
    fi
    echo "warning: no venv found, using system python ($VENV_PY)"
fi

# Fail early with a clear message if the interpreter lacks numpy.
if ! "$VENV_PY" -c "import numpy" 2>/dev/null; then
    echo "error: $VENV_PY has no numpy."
    echo "activate your venv first, or: $VENV_PY -m pip install numpy opencv-python"
    exit 1
fi

SCRIPT="${1:-thermal_detect.py}"
if [ $# -gt 0 ]; then shift; fi

echo "running: $SCRIPT $*  (as root, via $VENV_PY)"
sudo "$VENV_PY" "$HERE/$SCRIPT" "$@"

# Captured logs are written as root; hand them back to the user.
if [ -d "$HERE/logs" ]; then
    sudo chown -R "$(id -u):$(id -g)" "$HERE/logs" 2>/dev/null || true
fi
