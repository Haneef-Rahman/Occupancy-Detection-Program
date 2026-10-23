"""Explicit-scale radiometric adapters; never infer Celsius from 8-bit AGC."""
import platform
import queue
import threading
import time
import cv2
import numpy as np


def raw_to_celsius(raw, scale):
    if scale not in (.01, .1):
        raise ValueError('TLinear scale must be 0.01 or 0.1 kelvin per count.')
    if raw.dtype != np.uint16 or raw.ndim != 2 or raw.shape[1] != 160:
        raise ValueError('Lepton requires a native 16-bit Y16 frame, not 8-bit video.')
    if raw.shape[0] == 122:
        from thermal_detect import strip_telemetry
        raw, _ = strip_telemetry(raw)
    if raw.shape != (120, 160):
        raise ValueError(f'Unsupported Y16 frame shape {raw.shape}; expected 160×120 or 160×122.')
    return raw.astype(np.float32)*scale-273.15


class WindowsY16:
    """Windows UVC via DirectShow, then Media Foundation; fixed operator scale."""
    def __init__(self, index=0, scale=.01, factory=None):
        self.cap = None
        self.scale = scale
        factory = factory or cv2.VideoCapture
        failures = []
        for name, backend in [('DirectShow', cv2.CAP_DSHOW), ('Media Foundation', cv2.CAP_MSMF)]:
            for height in [122, 120]:
                cap = factory(index, backend)
                try:
                    if not cap.isOpened():
                        continue
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                    from thermal_detect import fourcc
                    cap.set(cv2.CAP_PROP_FOURCC, fourcc('Y16 '))
                    ok, raw = cap.read()
                    if not ok or raw is None:
                        raise ValueError('No frame')
                    data = raw_to_celsius(raw, scale)
                    self.cap = cap
                    self.first = data
                    self.backend = name
                    return
                except Exception as error:
                    failures.append(str(error))
                finally:
                    if self.cap is not cap:
                        cap.release()
        raise RuntimeError('No 16-bit Lepton stream on camera index '+str(index)+'. Close other camera apps, check the index and run windows_support/backend_probe.py. '+('; '.join(failures[-2:])))

    def read(self):
        if self.first is not None:
            data, self.first = self.first, None
            return data
        ok, raw = self.cap.read()
        return raw_to_celsius(raw, self.scale) if ok and raw is not None else None

    def close(self):
        if self.cap is not None:
            self.cap.release()


class LibUVCRaw:
    def __init__(self, scale=.01):
        from lepton_libuvc import LeptonUVC
        self.camera = LeptonUVC(timeout_s=.5)
        self.scale = scale
        self.backend = 'libuvc Y16'

    def read(self):
        # Read raw callback data so the adapter's explicit scale is authoritative.
        try:
            raw = self.camera.q.get(timeout=.25)
        except queue.Empty:
            return None
        while not self.camera.q.empty():
            raw = self.camera.q.get_nowait()
        return raw_to_celsius(raw, self.scale)

    def close(self):
        self.camera.release()


def open_adapter(index=0, scale=.01, system=None):
    if scale not in (.01, .1):
        raise ValueError('Invalid TLinear scale.')
    system = system or platform.system()
    return WindowsY16(index, scale) if system == 'Windows' else LibUVCRaw(scale)


class CameraStream:
    """Keep only the newest received frame; capture never blocks HTTP controls."""
    def __init__(self, index=0, scale=.01, opener=None):
        self.adapter = (opener or open_adapter)(index=index, scale=scale)
        self.backend = self.adapter.backend
        self.scale = scale
        self.frames = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._capture, daemon=True, name='web-lepton-capture')
        self.thread.start()

    def _capture(self):
        last = time.monotonic()
        try:
            while not self.stop.is_set():
                data = self.adapter.read()
                now = time.monotonic()
                if data is None:
                    if now-last > 15:
                        raise TimeoutError('No thermal frames for 15 seconds.')
                    self.stop.wait(.01)
                    continue
                if data.shape != (120, 160) or data.dtype.kind != 'f' or not np.isfinite(data).all():
                    raise ValueError('Invalid radiometric Celsius frame.')
                last = now
                if self.frames.full():
                    try:self.frames.get_nowait()
                    except queue.Empty:pass
                self.frames.put_nowait((data.copy(), now))
        except Exception as error:
            self.error = str(error)
        finally:
            self.adapter.close()

    def read(self):
        if self.error:
            raise RuntimeError(self.error)
        try:return self.frames.get(timeout=.1)
        except queue.Empty:return None

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        # Some UVC drivers unblock read() only when the handle is released.
        if self.thread.is_alive():
            self.adapter.close()
            self.thread.join(timeout=2)
