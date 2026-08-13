#!/usr/bin/env python3
"""
Report exactly what each video device delivers, so we can see why detection
is misbehaving. Run:  python diagnose.py
"""
import cv2
import numpy as np
import platform

print(f"platform : {platform.system()} {platform.release()}")
print(f"opencv   : {cv2.__version__}\n")

for idx in range(6):
    for convert_rgb in (True, False):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            if convert_rgb:          # report once per index, then move on
                print(f"[{idx}] FAILED TO OPEN  (in use by another app, or backend error)")
            continue

        if not convert_rgb:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
            except Exception:
                pass

        ok, f = cap.read()
        tag = "CONVERT_RGB=1 (default)" if convert_rgb else "CONVERT_RGB=0 (raw/Y16)"

        if not ok or f is None:
            print(f"[{idx}] {tag:26s} -> NO FRAME")
        else:
            shape = f.shape
            dt = f.dtype
            lo, hi, med = float(f.min()), float(f.max()), float(np.median(f))
            print(f"[{idx}] {tag:26s} -> shape={shape} dtype={dt} "
                  f"min={lo:.0f} max={hi:.0f} median={med:.0f}")

            if dt == np.uint16:
                for scale, name in ((0.01, "centiK"), (0.1, "deciK")):
                    c_med = med * scale - 273.15
                    c_max = hi * scale - 273.15
                    flag = "  <-- plausible" if -40 < c_med < 80 else ""
                    print(f"      as {name}: median={c_med:7.1f}C  max={c_max:7.1f}C{flag}")
            elif dt == np.uint8:
                print("      8-bit: AGC image (relative brightness, NOT temperature)")

        cap.release()
    print()

print("What we want: a device with dtype=uint16 and a 'plausible' temperature line.")
print("If only uint8 appears, macOS is refusing raw Y16 — the program will use")
print("adaptive (percentile) thresholding instead of absolute temperatures.")
