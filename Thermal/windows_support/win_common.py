#!/usr/bin/env python3
"""
Shared Windows plumbing. Every w_*.py imports this; none of them duplicate it.

WHAT THE w_*.py WRAPPERS ARE. Each one imports the real tool from Thermal/,
replaces the two or three things that are Unix-specific, and calls its main().
None of them copy logic, and none of them modify anything in Thermal/. The Mac
setup is untouched and cannot be broken from here — which also means a fix
Haneef makes upstream arrives here for free.

THE THREE THINGS THAT ACTUALLY DIFFER ON WINDOWS:

  1. THE CAMERA. The Mac path tries libuvc first (needs libuvc.dylib, absent
     here), then falls back to cv2.VideoCapture(index) with NO backend
     specified. On Windows that choice decides whether you get real
     temperatures or an 8-bit picture, so we name the backend explicitly and
     verify the result is 16-bit before handing it back.

  2. os.geteuid DOES NOT EXIST ON WINDOWS. dataset_pipeline.hand_back() calls
     it unconditionally to chown files back after sudo. On Windows that is an
     AttributeError, and there is no sudo to undo anyway, so it becomes a
     no-op.

  3. SERIAL PORTS. The radar defaults are /dev/cu.usbserial-*. Windows uses
     COM ports. Only matters if the radar is attached.

WORKING DIRECTORY. dataset_recording.py defines LOG_DIR = "logs" relative to
the CURRENT DIRECTORY, so captures only land in Thermal/logs/ if we are
standing in Thermal/. enter_thermal() does that, and every wrapper calls it
first. Get this wrong and captures appear inside windows_support/ where nothing
looks for them.
"""

import os
import platform
import socket
import sys

os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")

import cv2                                                  # noqa: E402
import numpy as np                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
THERMAL = os.path.dirname(HERE)

# DirectShow first: it honours CAP_PROP_CONVERT_RGB=0 more reliably than Media
# Foundation, which is what lets the raw 16-bit frames through.
BACKENDS = [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF)]

STATE = {"backend": "unknown", "operator": "unknown", "radiometric": None}


def enter_thermal():
    """Put Thermal/ on sys.path AND make it the working directory."""
    if THERMAL not in sys.path:
        sys.path.insert(0, THERMAL)
    os.chdir(THERMAL)


# ---------------------------------------------------------------------------
class WindowsThermalCamera:
    """
    The Lepton over UVC, with the backend named rather than guessed.

    read() returns (data, is_temp) with data in degrees Celsius — the same
    contract as thermal_detect.ThermalCamera, so nothing downstream can tell
    the difference. That is the point: one camera class changes, zero
    algorithms do.
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
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
            except Exception:
                pass
            ok, probe = cap.read()
            if ok and probe is not None and probe.dtype == np.uint16:
                med = float(np.median(probe)) * self.scale - 273.15
                if not (-40 < med < 80):
                    self.scale = 0.1           # low-gain mode reports K*10
                    med = float(np.median(probe)) * self.scale - 273.15
                self.cap = cap
                self.radiometric = True
                STATE["backend"] = name
                STATE["radiometric"] = True
                print(f"  camera: {name}, RADIOMETRIC "
                      f"({self.scale} K/count, ambient ~{med:.1f} C)")
                return
            cap.release()

        if require_radiometric:
            sys.exit(
                f"\nNo radiometric (16-bit) stream on device {index}.\n"
                "  Recording now would produce 8-bit AGC files with no\n"
                "  temperatures in them. They would look fine and be unusable\n"
                "  when merged with the Mac captures.\n\n"
                "  Run:  py -3.12 backend_probe.py\n"
                "  and send the output to Haneef.\n\n"
                "  To record anyway (you almost certainly should not):\n"
                "  add --allow-agc\n")

        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            sys.exit(f"could not open camera {index} at all")
        self.cap = cap
        STATE["backend"] = "AGC-8bit"
        STATE["radiometric"] = False
        print("  camera: AGC 8-bit — NO TEMPERATURES. Flagged in source.txt.")

    def read(self):
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


def patch_camera(mod, allow_agc=False):
    """Replace a module's open_camera with the Windows one."""
    def _open(args):
        idx = getattr(args, "device", None)
        return WindowsThermalCamera(idx if idx is not None else 0,
                                    require_radiometric=not allow_agc)
    mod.open_camera = _open


def patch_provenance(mod):
    """
    Append who and what recorded a capture to source.txt.

    With two people on two different Leptons, a capture you cannot attribute to
    a machine is a capture nobody can correct later. If Adrian's sensor reads
    half a degree off Haneef's, this turns that from a mystery into a constant.
    """
    original = mod.open_capture

    def _open_capture(note, conf, weights):
        d, fh, w = original(note, conf, weights)
        with open(os.path.join(d, "source.txt"), "a") as f:
            f.write(f"operator   {STATE['operator']}\n"
                    f"host       {socket.gethostname()}\n"
                    f"platform   {platform.platform()}\n"
                    f"camera     OpenCV {STATE['backend']} (Windows)\n"
                    f"opencv     {cv2.__version__}\n"
                    f"radiometric {STATE['radiometric']}\n")
        return d, fh, w

    mod.open_capture = _open_capture


def patch_hand_back(mod):
    """
    Neutralise the sudo-ownership fixup.

    dataset_pipeline.hand_back() calls os.geteuid(), which does not exist on
    Windows — an AttributeError, not a graceful skip. There is also no sudo
    here to undo: the Mac needs it because macOS blocks camera access for
    libusb, and Windows does not.
    """
    mod.hand_back = lambda path: None


def take_flag(argv, name, has_value=True, default=None):
    """Pull one of OUR flags out of argv before the real parser sees it."""
    if name not in argv:
        return default, argv
    i = argv.index(name)
    if has_value:
        val = argv[i + 1] if i + 1 < len(argv) else default
        del argv[i:i + 2]
        return val, argv
    del argv[i]
    return True, argv


def standard_prelude(require_operator=False):
    """
    Shared argv handling for the w_*.py wrappers.

    Returns (allow_agc, argv-with-our-flags-removed).
    """
    argv = sys.argv[1:]
    op, argv = take_flag(argv, "--operator", True, None)
    allow_agc, argv = take_flag(argv, "--allow-agc", False, False)
    if op:
        STATE["operator"] = op
    elif require_operator:
        print("note: no --operator given. Pass --operator adrian so captures\n"
              "      can be traced to a person and a sensor later.\n")
    sys.argv = [sys.argv[0]] + argv
    return bool(allow_agc), argv
