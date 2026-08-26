#!/usr/bin/env python3
"""
Point fluxnet.yaml at wherever this folder now lives. Run once after unzipping.

    python setup_windows.py

The yaml carries an ABSOLUTE `path:`, written on the Mac that built the
dataset. Ultralytics resolves train/val against it, so on another machine it
points at a directory that does not exist and training dies with
"Dataset images not found". This rewrites it to this file's own directory.

Safe to run repeatedly, and on any OS despite the name.
"""
import os

here = os.path.dirname(os.path.abspath(__file__))
y = os.path.join(here, "fluxnet.yaml")
lines = open(y).read().split("\n")
out = [f"path: {here}" if l.startswith("path:") else l for l in lines]
open(y, "w").write("\n".join(out))

for split in ("train", "val"):
    d = os.path.join(here, split, "images")
    n = len(os.listdir(d)) if os.path.isdir(d) else 0
    print(f"  {split:<6} {n} images   {d}")
print(f"\nfluxnet.yaml path -> {here}")
print(f"\nyolo detect train model=yolo26n.pt data={y} imgsz=640 epochs=150 batch=32 device=0")
