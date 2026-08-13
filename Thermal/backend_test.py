#!/usr/bin/env python3
"""
Try to open the PureThermal with each OpenCV backend explicitly, and with
several format requests. On macOS the default auto-backend path throws inside
AVFoundation for this device; forcing the backend/format sometimes works.

Run:  python backend_test.py
"""
import os
# silence the Orbbec/OBSENSOR backend that spams and fails on macOS
os.environ["OPENCV_VIDEOIO_PRIORITY_OBSENSOR"] = "0"

import cv2
import numpy as np

BACKENDS = [
    ("CAP_AVFOUNDATION", cv2.CAP_AVFOUNDATION),
    ("CAP_ANY", cv2.CAP_ANY),
]
FORMATS = [
    ("default", None),
    ("Y16 ", "Y16 "),
    ("YUYV", "YUYV"),
    ("UYVY", "UYVY"),
    ("GREY", "GREY"),
]

print(f"opencv {cv2.__version__}\n")

for dev in (0, 1, 2):
    for bname, bid in BACKENDS:
        for fname, fourcc in FORMATS:
            try:
                cap = cv2.VideoCapture(dev, bid)
            except Exception as e:
                print(f"[{dev}] {bname:18s} {fname:8s} -> exception on open: {e}")
                continue

            if not cap.isOpened():
                cap.release()
                continue

            if fourcc:
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

            ok, f = cap.read()
            if ok and f is not None:
                h, w = f.shape[:2]
                lepton = "  <== LEPTON!" if (w, h) in ((160, 120), (160, 240)) else ""
                print(f"[{dev}] {bname:18s} {fname:8s} -> {w}x{h} {f.dtype} "
                      f"min={f.min()} max={f.max()}{lepton}")
            else:
                print(f"[{dev}] {bname:18s} {fname:8s} -> opened but NO FRAME")
            cap.release()
    print()

print("Looking for a 160x120 (or 160x240) result. If none appears, OpenCV cannot")
print("reach this camera on macOS and we use the libuvc path instead (see README).")
