#!/usr/bin/env python3
"""
Record captures on Windows. Same output as the Mac, byte for byte compatible.

    py -3.12 record_win.py --operator adrian --note "cafeteria, 1430, 26C"

    r        start / stop recording
    q, ESC   quit

Everything else — flags, keys, on-screen display — is identical to
../dataset_recording.py, because this IS that program. It imports it and
replaces exactly two things:

  1. THE CAMERA OPENER. The Mac path tries libuvc first, which needs
     libuvc.dylib and does not exist on Windows, then falls back to
     cv2.VideoCapture(index) with no backend specified — leaving the choice to
     OpenCV's whim. On Windows that whim decides whether you get real
     temperatures or an 8-bit picture. So this forces DSHOW, then MSMF, and
     checks that what came back is actually 16-bit before returning it.

  2. PROVENANCE. With two people recording on two different Leptons, a capture
     you cannot attribute to a machine is a capture nobody can correct later.
     This writes hostname, platform, camera backend and operator into
     source.txt alongside what the Mac already records.

NOTHING IN THE PARENT DIRECTORY IS MODIFIED. That is deliberate: Haneef's
working setup stays exactly as it is, and this cannot break it. The pattern
(import the module, replace the function, call main) is the same one
recompensate.py already uses on annotate.py.

NO ADMINISTRATOR NEEDED. The Mac needs sudo because the macOS kernel claims the
camera and libusb has to fight it. Windows has no such problem — if something
asks you to run as admin, it is not this.
"""

import os
import platform
import socket
import sys
from datetime import datetime

os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")

import cv2                                                  # noqa: E402
import numpy as np                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

import thermal_detect as TD                                 # noqa: E402
import dataset_recording as DR                              # noqa: E402

BACKENDS = [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF)]
_chosen = {"backend": "unknown"}


class WindowsThermalCamera:
    """
    Lepton over UVC on Windows, with the backend named rather than guessed.

    Subclasses nothing and reimplements little: it opens the device the way
    Windows needs, then hands frames back in the SAME form TD.ThermalCamera
    does — read() returns (data, is_temp) with data in degrees Celsius when the
    stream is radiometric. Downstream code cannot tell the difference, which is
    the point.
    """

    def __init__(self, index=0, require_radiometric=True):
        self.cap = None
        self.scale = 0.01              # TLinear high gain: counts are K*100
        self.radiometric = False

        for name, bid in BACKENDS:
            cap = cv2.VideoCapture(index, bid)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            try:
                cap.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*"Y16 "))
            except Exception:
                pass
            ok, probe = cap.read()
            if ok and probe is not None and probe.dtype == np.uint16:
                med_c = float(np.median(probe)) * self.scale - 273.15
                if not (-40 < med_c < 80):
                    # low-gain mode reports K*10 instead
                    self.scale = 0.1
                self.cap = cap
                self.radiometric = True
                _chosen["backend"] = name
                print(f"  camera: {name}, RADIOMETRIC "
                      f"({self.scale} K/count, ambient "
                      f"~{float(np.median(probe)) * self.scale - 273.15:.1f} C)")
                return
            cap.release()

        if require_radiometric:
            sys.exit(
                "\nNo radiometric (16-bit) stream found on device "
                f"{index}.\n"
                "  Recording now would produce 8-bit AGC files with no\n"
                "  temperatures in them, which cannot be merged with the Mac\n"
                "  captures.\n\n"
                "  Run:  py -3.12 backend_probe.py\n"
                "  and send the output to Haneef.\n\n"
                "  If you understand the consequences and want to record\n"
                "  anyway, add --allow-agc.")

        # Explicitly permitted 8-bit fallback.
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            sys.exit(f"could not open camera {index} at all")
        self.cap = cap
        _chosen["backend"] = "AGC-8bit"
        print("  camera: AGC 8-bit — NO TEMPERATURES. Flagged in source.txt.")

    def read(self):
        """Returns (data, is_temp). data FIRST — same order as TD.ThermalCamera."""
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None, False
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.radiometric and frame.dtype == np.uint16:
            return frame.astype(np.float32) * self.scale - 273.15, True
        return frame.astype(np.float32), False

    def release(self):
        if self.cap:
            self.cap.release()


def open_camera_windows(args):
    return WindowsThermalCamera(
        args.device if args.device is not None else 0,
        require_radiometric=not getattr(args, "allow_agc", False))


_orig_open_capture = DR.open_capture


def open_capture_windows(note, conf, weights):
    """DR.open_capture, plus who and what recorded it."""
    d, fh, w = _orig_open_capture(note, conf, weights)
    with open(os.path.join(d, "source.txt"), "a") as f:
        f.write(f"operator   {OPERATOR}\n"
                f"host       {socket.gethostname()}\n"
                f"platform   {platform.platform()}\n"
                f"camera     OpenCV {_chosen['backend']} (Windows)\n"
                f"opencv     {cv2.__version__}\n")
    return d, fh, w


OPERATOR = "unknown"


def main():
    global OPERATOR

    # Pull our own flags out before dataset_recording's parser sees argv.
    argv = sys.argv[1:]
    if "--operator" in argv:
        i = argv.index("--operator")
        OPERATOR = argv[i + 1]
        del argv[i:i + 2]
    allow_agc = "--allow-agc" in argv
    if allow_agc:
        argv.remove("--allow-agc")
    sys.argv = [sys.argv[0]] + argv

    if OPERATOR == "unknown":
        print("note: no --operator given. Pass --operator adrian so captures\n"
              "      can be traced to a person and a sensor later.\n")

    # Work from the Thermal/ directory so logs/ and models/ resolve the same
    # way they do on the Mac.
    os.chdir(PARENT)

    def _open(args):
        args.allow_agc = allow_agc
        return open_camera_windows(args)

    DR.open_camera = _open
    DR.open_capture = open_capture_windows
    DR.main()


if __name__ == "__main__":
    main()
