#!/usr/bin/env python3
"""
Try capturing the PureThermal through PyAV (ffmpeg's AVFoundation input)
instead of OpenCV's. Photo Booth can see this camera, so AVFoundation itself
works — only OpenCV's wrapper fails. ffmpeg may well succeed.

    pip install av
    python pyav_test.py
"""
import sys

try:
    import av
except ImportError:
    print("PyAV not installed.  ->  pip install av")
    sys.exit(1)

import os

import numpy as np

print(f"PyAV {av.__version__}")
term = os.environ.get("TERM_PROGRAM", "(unknown)")
print(f"terminal: {term}"
      + ("   <-- Terminal.app has NO camera permission; use VS Code's terminal"
         if term == "Apple_Terminal" else ""))
print()

# List AVFoundation devices (ffmpeg reports them in the error text).
print("--- devices reported by ffmpeg ---")
try:
    av.open("", format="avfoundation", options={"list_devices": "true"})
except Exception as e:
    for line in str(e).splitlines():
        print("  " + line)
print()

# AVFoundation input expects "video:audio" — a bare "0" is ambiguous and
# generally returns EIO. "0:none" selects video device 0 with no audio.
candidates = ["0:none", "1:none", "2:none", "0", "default:none"]

for dev in candidates:
    for opts in ({}, {"pixel_format": "gray16le"}, {"pixel_format": "yuyv422"}):
        label = f"device '{dev}' opts={opts or 'default'}"
        try:
            container = av.open(dev, format="avfoundation", options=opts, timeout=3)
        except Exception as e:
            msg = str(e).splitlines()[0][:90]
            print(f"{label:52s} -> open failed: {msg}")
            continue

        try:
            got = False
            for frame in container.decode(video=0):
                arr = frame.to_ndarray()
                h, w = arr.shape[:2]
                lep = "   <== LEPTON!" if (w, h) in ((160, 120), (160, 240)) else ""
                print(f"{label:52s} -> {w}x{h} {arr.dtype} "
                      f"fmt={frame.format.name} min={arr.min()} max={arr.max()}{lep}")
                got = True
                break
            if not got:
                print(f"{label:52s} -> opened, no frames decoded")
        except Exception as e:
            print(f"{label:52s} -> decode failed: {str(e).splitlines()[0][:60]}")
        finally:
            container.close()

print("\nLooking for a 160x120 result. gray16le format = radiometric data.")
