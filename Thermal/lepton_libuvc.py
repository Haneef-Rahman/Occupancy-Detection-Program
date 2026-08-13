#!/usr/bin/env python3
"""
PureThermal / Lepton capture via libuvc — bypasses OpenCV and AVFoundation.

Why this exists: on macOS, OpenCV's AVFoundation backend cannot open the
PureThermal at all ("backend is generally available but can't be used to
capture by index"). libuvc talks to the board over raw USB instead, which
also unlocks true Y16 radiometric frames that AVFoundation would not give us.

Prerequisites (macOS):
    brew install cmake libusb
    git clone https://github.com/groupgets/libuvc.git
    cd libuvc && mkdir build && cd build && cmake .. && make && sudo make install

Standalone test:
    python lepton_libuvc.py
"""

import ctypes
import ctypes.util
import platform
import queue
import sys
from ctypes import (CDLL, POINTER, Structure, byref, c_int, c_uint8, c_uint16,
                    c_uint32, c_size_t, c_void_p, cast)

import numpy as np

PT_VENDOR_ID = 0x1E4E   # 7758 decimal — matches system_profiler output
PT_PRODUCT_ID = 0x0100  # 256 decimal

# libuvc frame-format enum values differ between the GroupGets fork and
# mainline libuvc; we try both.
FMT_Y16_CANDIDATES = [
    (13, "UVC_FRAME_FORMAT_Y16 (GroupGets fork)"),
    (10, "UVC_FRAME_FORMAT_GRAY16 (mainline libuvc)"),
]


# ---------------------------------------------------------------------------
# Minimal ctypes bindings
# ---------------------------------------------------------------------------
class uvc_context(Structure):
    _fields_ = [("usb_ctx", c_void_p), ("own_usb_ctx", c_uint8),
                ("open_devices", c_void_p), ("handler_thread", ctypes.c_ulong),
                ("kill_handler_thread", c_int)]


class uvc_device(Structure):
    _fields_ = [("ctx", POINTER(uvc_context)), ("ref", c_int), ("usb_dev", c_void_p)]


class uvc_stream_ctrl(Structure):
    _fields_ = [("bmHint", c_uint16), ("bFormatIndex", c_uint8),
                ("bFrameIndex", c_uint8), ("dwFrameInterval", c_uint32),
                ("wKeyFrameRate", c_uint16), ("wPFrameRate", c_uint16),
                ("wCompQuality", c_uint16), ("wCompWindowSize", c_uint16),
                ("wDelay", c_uint16), ("dwMaxVideoFrameSize", c_uint32),
                ("dwMaxPayloadTransferSize", c_uint32), ("dwClockFrequency", c_uint32),
                ("bmFramingInfo", c_uint8), ("bPreferredVersion", c_uint8),
                ("bMinVersion", c_uint8), ("bMaxVersion", c_uint8),
                ("bInterfaceNumber", c_uint8)]


class timeval(Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class uvc_frame(Structure):
    # Only the leading fields are read, so tail differences between libuvc
    # versions are harmless.
    _fields_ = [("data", POINTER(c_uint8)), ("data_bytes", c_size_t),
                ("width", c_uint32), ("height", c_uint32),
                ("frame_format", c_int), ("step", c_size_t),
                ("sequence", c_uint32), ("capture_time", timeval),
                ("source", POINTER(uvc_device)), ("library_owns_data", c_uint8)]


def _load_libuvc():
    names = ["libuvc.dylib", "libuvc.so", "libuvc.so.0", "libuvc.0.dylib"]
    prefixes = ["", "/usr/local/lib/", "/opt/homebrew/lib/", "/usr/lib/"]
    tried = []
    for p in prefixes:
        for n in names:
            path = p + n
            tried.append(path)
            try:
                return CDLL(path)
            except OSError:
                continue
    found = ctypes.util.find_library("uvc")
    if found:
        try:
            return CDLL(found)
        except OSError:
            pass
    raise OSError(
        "libuvc not found. Install it:\n"
        "  brew install cmake libusb\n"
        "  git clone https://github.com/groupgets/libuvc.git\n"
        "  cd libuvc && mkdir build && cd build && cmake .. && make && sudo make install\n"
        f"(searched: {', '.join(tried[:6])} ...)"
    )


PTR_FRAME_CB = ctypes.CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)


# ---------------------------------------------------------------------------
class LeptonUVC:
    """
    Streams Y16 radiometric frames from a PureThermal board.

    Usage:
        cam = LeptonUVC()
        temp_c, ok = cam.read()      # ok=True -> temp_c is float32 °C
        cam.release()
    """

    def __init__(self, width=160, height=120, fps=9, timeout_s=5.0):
        self.lib = _load_libuvc()
        self._configure_prototypes()

        self.ctx = c_void_p()
        self.dev = c_void_p()
        self.devh = c_void_p()
        self.ctrl = uvc_stream_ctrl()
        self.q = queue.Queue(maxsize=2)
        self.radiometric = True
        self.scale = 0.01           # centikelvin per count (TLinear default)
        self.timeout_s = timeout_s
        self._streaming = False

        if self.lib.uvc_init(byref(self.ctx), 0) < 0:
            raise RuntimeError("uvc_init failed")

        res = self.lib.uvc_find_device(self.ctx, byref(self.dev),
                                       PT_VENDOR_ID, PT_PRODUCT_ID, 0)
        if res < 0:
            self.lib.uvc_exit(self.ctx)
            raise RuntimeError(
                "PureThermal not found on USB. Check the cable, and close any "
                "other app using the camera (Photo Booth, GetThermal)."
            )

        if self.lib.uvc_open(self.dev, byref(self.devh)) < 0:
            self.lib.uvc_unref_device(self.dev)
            self.lib.uvc_exit(self.ctx)
            raise RuntimeError(
                "uvc_open failed — usually a permissions issue.\n"
                "On macOS try running once with sudo, or ensure no other app "
                "holds the camera."
            )

        # Negotiate Y16 at the requested size; try both enum conventions.
        chosen = None
        for fmt_val, fmt_name in FMT_Y16_CANDIDATES:
            r = self.lib.uvc_get_stream_ctrl_format_size(
                self.devh, byref(self.ctrl), fmt_val, width, height, fps)
            if r >= 0:
                chosen = (fmt_val, fmt_name)
                break
        if chosen is None:
            self._teardown()
            raise RuntimeError(
                f"Could not negotiate Y16 {width}x{height}@{fps}. "
                "The board may be in a non-radiometric mode."
            )
        self.format_used = chosen[1]

        self._cb = PTR_FRAME_CB(self._on_frame)
        if self.lib.uvc_start_streaming(self.devh, byref(self.ctrl),
                                        self._cb, None, 0) < 0:
            self._teardown()
            raise RuntimeError("uvc_start_streaming failed")
        self._streaming = True
        self.width, self.height = width, height

        # Sanity-check the temperature scale on the first frame.
        first, ok = self.read()
        if ok:
            med = float(np.median(first))
            if not (-40 < med < 80):
                self.scale = 0.1

    def _configure_prototypes(self):
        L = self.lib
        L.uvc_init.restype = c_int
        L.uvc_find_device.restype = c_int
        L.uvc_open.restype = c_int
        L.uvc_get_stream_ctrl_format_size.restype = c_int
        L.uvc_start_streaming.restype = c_int

    def _on_frame(self, frame_p, _user):
        f = frame_p.contents
        if not f.data or f.data_bytes == 0:
            return
        n = f.data_bytes // 2
        buf = cast(f.data, POINTER(c_uint16 * n)).contents
        arr = np.frombuffer(buf, dtype=np.uint16).copy()
        try:
            arr = arr.reshape(f.height, f.width)
        except ValueError:
            return
        if self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put(arr)

    def read(self):
        """Returns (temp_c float32 °C, ok)."""
        try:
            raw = self.q.get(timeout=self.timeout_s)
        except queue.Empty:
            return None, False
        return raw.astype(np.float32) * self.scale - 273.15, True

    def _teardown(self):
        try:
            if self._streaming:
                self.lib.uvc_stop_streaming(self.devh)
                self._streaming = False
            if self.devh:
                self.lib.uvc_close(self.devh)
            if self.dev:
                self.lib.uvc_unref_device(self.dev)
            if self.ctx:
                self.lib.uvc_exit(self.ctx)
        except Exception:
            pass

    def release(self):
        self._teardown()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"platform: {platform.system()}")
    try:
        cam = LeptonUVC()
    except Exception as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)

    print(f"connected. format: {cam.format_used}, scale {cam.scale} K/count")
    for i in range(5):
        t, ok = cam.read()
        if not ok:
            print(f"  frame {i}: TIMEOUT")
            continue
        print(f"  frame {i}: {t.shape} "
              f"min={t.min():6.1f}C  median={np.median(t):6.1f}C  max={t.max():6.1f}C")
    cam.release()
    print("\nIf the temperatures look sane (room ~20-28C, hand ~30-34C),")
    print("radiometric capture is working — run: python thermal_detect.py")
