#!/usr/bin/env python3
"""
Find which OpenCV backend reaches your Lepton, and whether it gives real degrees.

    .\probe.ps1

RUN THIS FIRST. Before recording anything, before installing anything else.
It answers the one question everything else depends on: does your Lepton hand
Windows RADIOMETRIC 16-bit data (real temperatures) or AGC 8-bit (a pretty
picture with the temperatures thrown away)?

WHY IT MATTERS MORE THAN IT SOUNDS. Every tool in this project assumes a frame
is degrees Celsius. The render span is 15-45 C. The person-plausibility checks
compare against ambient in C. If your camera gives 8-bit AGC instead, your
files will contain 0-255 grey levels, everything will still appear to work, and
the captures will be quietly unusable when merged with Haneef's.

That failure is genuinely hard to spot after the fact — a 0-255 file looks
exactly like a capture taken in a very cold room. Thirty seconds here saves an
afternoon of recording you would have to throw away.

This is the Windows counterpart of ../backend_test.py, which probes
AVFoundation for macOS. Same idea, different backends.
"""

import os

# The Orbbec backend spams and fails on some machines; switch it off first.
os.environ["OPENCV_VIDEOIO_PRIORITY_OBSENSOR"] = "0"

import sys                                                  # noqa: E402

import cv2                                                  # noqa: E402
import numpy as np                                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import thermal_detect as TD                                 # noqa: E402

BACKENDS = [
    ("CAP_DSHOW", cv2.CAP_DSHOW),      # DirectShow — usually best for Y16
    ("CAP_MSMF", cv2.CAP_MSMF),        # Media Foundation — Windows default
    ("CAP_ANY", cv2.CAP_ANY),          # let OpenCV choose
]
FORMATS = [
    ("Y16 ", "Y16 "),                  # the one we want: 16-bit radiometric
    ("default", None),
    ("GREY", "GREY"),
    ("YUYV", "YUYV"),
]

# 122/244 = telemetry enabled (2 extra rows per frame). Both are fine; the
# recorder strips them. What matters here is only that the stream is 16-bit.
LEPTON_SIZES = ((160, 120), (160, 122), (160, 240), (160, 244))


def describe(frame):
    """Say what this frame actually is, in plain terms."""
    h, w = frame.shape[:2]
    size_ok = (w, h) in LEPTON_SIZES
    tele = h in (122, 244)
    if frame.dtype == np.uint16:
        # Lepton TLinear: counts are kelvin*100 (or *10 in low-gain).
        # Median over the telemetry rows too would skew this, so drop them.
        core = frame[:120] if h == 122 else frame
        c = float(np.median(core)) * 0.01 - 273.15
        plausible = -20 < c < 60
        kind = "RADIOMETRIC 16-bit" + ("" if plausible else "  (but scale looks odd)")
        extra = f"median ~{c:.1f} C" if plausible else f"median raw {np.median(core):.0f}"
        if tele:
            extra += ", telemetry ON"
    else:
        kind = "AGC 8-bit — NO TEMPERATURES"
        extra = f"values {frame.min()}-{frame.max()}"
    return size_ok, kind, extra


def main():
    print(f"opencv {cv2.__version__}, python {sys.version.split()[0]}\n")
    winners = []

    for dev in range(4):
        for bname, bid in BACKENDS:
            for fname, fourcc in FORMATS:
                try:
                    cap = cv2.VideoCapture(dev, bid)
                except Exception as e:
                    print(f"[{dev}] {bname:11s} {fname:8s} -> open failed: {e}")
                    continue
                if not cap.isOpened():
                    cap.release()
                    continue

                if fourcc:
                    # CONVERT_RGB=0 is the critical one: it tells OpenCV not to
                    # helpfully convert the raw 16-bit data into a colour image.
                    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                    cap.set(cv2.CAP_PROP_FOURCC, TD.fourcc(fourcc))

                ok, f = cap.read()
                cap.release()
                if not ok or f is None:
                    print(f"[{dev}] {bname:11s} {fname:8s} -> opened, no frame")
                    continue

                size_ok, kind, extra = describe(f)
                h, w = f.shape[:2]
                tag = ""
                if size_ok and f.dtype == np.uint16:
                    tag = "   <== USE THIS"
                    winners.append((dev, bname, fname))
                elif size_ok:
                    tag = "   <== Lepton, but 8-bit"
                print(f"[{dev}] {bname:11s} {fname:8s} -> {w}x{h} {str(f.dtype):7s} "
                      f"{kind}, {extra}{tag}")
        print()

    print("=" * 68)
    if winners:
        dev, bname, fname = winners[0]
        print("GOOD. Radiometric data is reaching OpenCV.\n")
        print(f"  device {dev}, backend {bname}, format {fname}")
        print(f"\n  Record with:   .\\record.ps1 -Operator adrian "
              f"-Note \"where, time, ambient\"")
        print("  The recorder forces DSHOW then MSMF and asks the driver for")
        print("  its own frame size, so you should not need to pass anything")
        print(f"  else. If it picks the wrong camera: -Device {dev}")
    else:
        got_lepton = False
        print("NO RADIOMETRIC PATH FOUND.\n")
        print("  Do NOT start recording — captures would be 8-bit AGC and")
        print("  could not be merged with the Mac data.\n")
        print("  Things to try, in order:")
        print("   1. Close anything else using the camera (Teams, OBS, the")
        print("      FLIR app, a browser tab). Windows gives exclusive access.")
        print("   2. Unplug and replug the PureThermal, then re-run this.")
        print("   3. Try a different USB port — ideally USB 2.0, directly on")
        print("      the machine rather than through a hub.")
        print("   4. Check the PureThermal firmware is current (GroupGets).")
        print("\n  Then send this whole output to Haneef.")
    print("=" * 68)


if __name__ == "__main__":
    main()
