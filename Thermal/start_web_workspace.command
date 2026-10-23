#!/bin/zsh
set -e
cd -- "${0:A:h}"
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python web_workspace.py "$@"
fi
exec python3 web_workspace.py "$@"
