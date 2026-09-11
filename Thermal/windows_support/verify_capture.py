#!/usr/bin/env python3
"""
Check a capture is sound BEFORE you send it. Run it after every session.

    py -3.12 verify_capture.py ..\\logs\\capture_20260911_143000
    py -3.12 verify_capture.py --all

WHAT IT IS LOOKING FOR. One thing above all: are the numbers in these files
degrees Celsius, or are they 0-255 grey levels? Everything downstream — the
15-45 C render, the person-plausibility checks, the merge with the Mac data —
assumes Celsius. A capture of grey levels looks completely normal until it is
merged, at which point it reads as a room at 14 degrees and quietly poisons the
dataset.

It also checks the things that are boring until they bite: matching file counts
across npy/png/labels, a readable manifest, and whether the scene had anything
warm in it at all.

There is no pass mark for "good data". A capture of an empty corridor is
legitimate and will look sparse. The check is for BROKEN, not for boring.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(os.path.dirname(HERE), "logs")


def verify(root):
    name = os.path.basename(root.rstrip("\\/"))
    print(f"\n{'=' * 66}\n{name}\n{'=' * 66}")
    problems, warnings = [], []

    if not os.path.isdir(root):
        print(f"  no such directory: {root}")
        return False

    counts = {}
    for sub, ext in (("npy", ".npy"), ("png", ".png"),
                     ("labels", ".txt"), ("review", ".png")):
        counts[sub] = len(glob.glob(os.path.join(root, sub, "*" + ext)))
    print(f"  frames: npy {counts['npy']}   png {counts['png']}   "
          f"labels {counts['labels']}   review {counts['review']}")

    if counts["npy"] == 0:
        problems.append("no npy/ frames at all — nothing was recorded")
        _report(problems, warnings)
        return False
    if counts["npy"] != counts["png"] or counts["npy"] != counts["labels"]:
        problems.append("npy / png / labels counts disagree — a write failed "
                        "or the recorder was killed mid-frame")
    if counts["npy"] < 100:
        warnings.append(f"only {counts['npy']} frames — short session")

    # ---- THE IMPORTANT CHECK -------------------------------------------
    files = sorted(glob.glob(os.path.join(root, "npy", "*.npy")))
    sample = files[:: max(1, len(files) // 60)][:60]
    arrs = [np.load(f) for f in sample]
    a = arrs[len(arrs) // 2]

    print(f"\n  dtype {a.dtype}   shape {a.shape}")
    if a.shape[:2] != (120, 160):
        problems.append(f"frame is {a.shape[:2]}, expected (120, 160)")

    med = float(np.median([np.median(x) for x in arrs]))
    mx = float(max(np.max(x) for x in arrs))
    mn = float(min(np.min(x) for x in arrs))
    print(f"  values: min {mn:.1f}   median {med:.1f}   max {mx:.1f}")

    if a.dtype != np.float32:
        problems.append(f"dtype is {a.dtype}, expected float32 degrees C")

    # HOW TO TELL GREY LEVELS FROM DEGREES. Not by the median — AGC puts that
    # wherever the scene's histogram happens to sit, and a fabricated 8-bit
    # capture with median 36.1 sailed past an earlier version of this check.
    #
    # The signature of auto-contrast is the STRETCH: it maps the coldest pixel
    # to exactly 0 and the hottest to exactly 255, every frame, and the result
    # is whole numbers. Real temperatures do neither. Haneef's hottest pixel
    # across every capture he owns is 38.8 C, and his values are fractional.
    stretched = abs(mx - 255.0) < 1.5 and abs(mn - 0.0) < 1.5
    integral = all(float(np.abs(x - np.round(x)).max()) < 1e-6 for x in arrs[:8])
    looks_grey = stretched or (integral and mx > 100.0)
    looks_celsius = (not looks_grey) and -20.0 < med < 60.0 and mx < 400.0

    if looks_grey:
        why = []
        if stretched:
            why.append(f"range is exactly {mn:.0f}-{mx:.0f}")
        if integral:
            why.append("all values are whole numbers")
        problems.append(
            "THESE ARE GREY LEVELS, NOT TEMPERATURES (" + ", ".join(why) + ").\n"
            "      The camera gave 8-bit AGC. Do NOT send this capture — it\n"
            "      cannot be merged with the Mac data, and it will not look\n"
            "      broken later, it will just look like a very cold room.\n"
            "      Run:  py -3.12 backend_probe.py")
    elif not looks_celsius:
        warnings.append(f"median {med:.1f}, max {mx:.1f} — unusual for degrees "
                        f"C. Worth a second look before sending.")
    else:
        print(f"  -> looks like CELSIUS. Ambient about {med:.1f} C.")

    # ---- was anything warm in frame? ------------------------------------
    if looks_celsius:
        excess = mx - med
        print(f"  warmest thing: {mx:.1f} C, i.e. +{excess:.1f} C over ambient")
        if excess < 3.0:
            warnings.append(f"nothing in this capture is more than "
                            f"+{excess:.1f} C above ambient. If people were "
                            f"present, the camera may have been pointed wrong "
                            f"or they were too far away. If it was meant to be "
                            f"an EMPTY capture, this is correct.")
        if mx > 45.0:
            print(f"  note: {mx:.1f} C exceeds the 15-45 C render span, so hot "
                  f"equipment\n        saturates to white in png/. That is "
                  f"expected and the npy/ keeps\n        the real value — "
                  f"these captures are especially useful.")

    # ---- labels ---------------------------------------------------------
    nbox = nempty = nother = 0
    other_eg = None
    for f in glob.glob(os.path.join(root, "labels", "*.txt")):
        lines = [l for l in open(f) if l.strip()]
        nbox += len(lines)
        if not lines:
            nempty += 1
        for l in lines:
            q = l.split()
            if q and q[0] != "1":
                nother += 1
                if other_eg is None:
                    other_eg = os.path.basename(f)
                break
    # ONE line, not one per file. An earlier version appended a problem per
    # offending file and produced 3402 identical lines on a classical capture.
    if nother:
        problems.append(
            f"{nother} frames contain a non-omega class (e.g. {other_eg}). "
            f"Captures from record_win.py should only ever contain class 1.")
    print(f"\n  labels: {nbox} omega boxes, {nempty} empty frames "
          f"({100.0 * nempty / max(1, counts['labels']):.0f}%)")
    if nempty == counts["labels"] and counts["labels"]:
        warnings.append("EVERY frame is empty — the model found nobody. Fine "
                        "for an empty-room capture, wrong otherwise.")

    # ---- provenance -----------------------------------------------------
    sp = os.path.join(root, "source.txt")
    if not os.path.exists(sp):
        problems.append("no source.txt — recorded by something other than "
                        "record_win.py / dataset_recording.py")
    else:
        src = open(sp).read()
        print("\n  provenance:")
        for line in src.strip().splitlines():
            print(f"    {line}")
        if "operator" not in src:
            warnings.append("source.txt has no operator — next time pass "
                            "--operator adrian")

    if not os.path.exists(os.path.join(root, "manifest.csv")):
        problems.append("no manifest.csv")

    return _report(problems, warnings)


def _report(problems, warnings):
    print()
    for w in warnings:
        print(f"  WARNING  {w}")
    for p in problems:
        print(f"  PROBLEM  {p}")
    if problems:
        print("\n  NOT OK — fix before sending.")
        return False
    if warnings:
        print("\n  OK, with notes above. Safe to send.")
    else:
        print("\n  OK. Safe to send.")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Sanity-check a capture before sending it.")
    ap.add_argument("capture_dir", nargs="?")
    ap.add_argument("--all", action="store_true",
                    help="check every capture in ../logs/")
    args = ap.parse_args()

    if args.all:
        dirs = sorted(d for d in glob.glob(os.path.join(LOGS, "capture_*"))
                      if os.path.isdir(d))
        if not dirs:
            sys.exit(f"no captures in {LOGS}")
        ok = all(verify(d) for d in dirs)
        print(f"\n{len(dirs)} captures checked.")
        sys.exit(0 if ok else 1)

    if not args.capture_dir:
        sys.exit("give a capture directory, or --all")
    sys.exit(0 if verify(args.capture_dir) else 1)


if __name__ == "__main__":
    main()
