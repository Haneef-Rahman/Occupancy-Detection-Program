#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "$HERE/.venv/bin/python" "$HERE/tracking_simulator.py" "$@"
