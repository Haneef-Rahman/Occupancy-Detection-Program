#!/usr/bin/env python3
"""
One answer to "which model?", for every tool in the project.

    from model_registry import resolve_weights
    weights = resolve_weights(args.weights)      # None -> newest models/vN

    python3 model_registry.py                    # show what is installed

WHY THIS IS A MODULE AND NOT A LINE OF GLOB IN EACH SCRIPT. It already was a
line of glob in each script, twice, and the two copies had already drifted:
dataset_recording.py resolved models/ against the CURRENT WORKING DIRECTORY,
so it silently found nothing when run from anywhere but Thermal/, while
dataset_pipeline.py resolved it against the script location and worked. Both
looked correct in isolation.

Worse, every other tool took --weights as REQUIRED and the docs said
`--weights models/v2/best.pt`. Train a v3 and those commands keep running
happily against v2 — no error, no warning, just the old model. That is the
failure this file exists to remove.

SORTING IS NUMERIC, NOT LEXICAL. "v10" sorts before "v9" as a string. There are
three versions today, which is exactly when that bug gets written and not
noticed for six months.
"""

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")


def installed():
    """[(version:int, path:str)] for every models/vN/best.pt, newest last."""
    out = []
    for p in glob.glob(os.path.join(MODELS_DIR, "v*", "best.pt")):
        m = re.fullmatch(r"v(\d+)", os.path.basename(os.path.dirname(p)))
        if m:
            out.append((int(m.group(1)), p))
    out.sort()
    return out


def latest_weights():
    """Path to the highest-numbered model, or None if none are installed."""
    got = installed()
    return got[-1][1] if got else None


def resolve_weights(explicit=None, quiet=False):
    """
    The weights a tool should use: the explicit one, else the newest.

    Exits with an actionable message rather than returning None, because every
    caller would otherwise write the same three lines and one of them would get
    it wrong. Prints what it picked — a tool that silently chooses a model is
    how you spend an afternoon evaluating last month's weights.
    """
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"weights not found: {explicit}")
        if not quiet:
            print(f"model: {explicit}  (explicit)")
        return explicit

    w = latest_weights()
    if not w:
        sys.exit(f"no model found in {MODELS_DIR}/vN/best.pt — train one, or "
                 f"pass --weights")
    if not quiet:
        n = len(installed())
        print(f"model: {os.path.relpath(w, HERE)}  "
              f"(newest of {n} installed)")
    return w


def main():
    got = installed()
    if not got:
        print(f"no models in {MODELS_DIR}")
        return
    print(f"{MODELS_DIR}\n")
    for v, p in got:
        mb = os.path.getsize(p) / 1e6
        mark = "  <- default" if p == got[-1][1] else ""
        print(f"  v{v:<3} {mb:5.1f} MB   {p}{mark}")


if __name__ == "__main__":
    main()
