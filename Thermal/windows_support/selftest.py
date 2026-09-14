#!/usr/bin/env python3
"""
Prove the Windows support works, without needing the camera plugged in.

    py -3.12 selftest.py

WHY THIS EXISTS. None of this folder has ever run on a real Windows machine —
it was written on a Mac against the Windows docs. So the first time Adrian runs
it IS the test, and it is better to find out in 20 seconds at a desk than in
hour two of a capture session in a cafeteria.

Everything here runs against a FAKE camera that emits synthetic radiometric
frames — a warm blob on a cool background, in real degrees. That exercises the
entire path that matters: imports, working directory, the monkeypatches, frame
encoding, file writing, the manifest, and the verifier reading it all back.

WHAT IT CANNOT TEST. Whether your actual Lepton hands OpenCV 16-bit data. That
is the one genuinely unknown thing and no amount of cleverness here settles it.
Run backend_probe.py with the camera attached for that.

Writes to a temporary folder and cleans up. Nothing lands in Thermal/logs/.
"""

import os
import platform
import shutil
import sys
import tempfile
import traceback

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail or ""))
        print(f"  PASS  {name}" + (f"   {detail}" if detail else ""))
        return True
    except Exception as e:
        RESULTS.append((False, name, str(e)))
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-3:]:
            print(f"          {line}")
        return False


def main():
    print("\nFLUXNET Windows support — self test")
    print("=" * 66)
    print(f"  {platform.python_version()} on {platform.system()} "
          f"{platform.release()}")
    print(f"  geteuid present: {hasattr(os, 'geteuid')}  "
          f"(False is expected on Windows)")
    print()

    # ---- 1. environment -------------------------------------------------
    def _pkgs():
        import cv2
        import numpy
        return f"opencv {cv2.__version__}, numpy {numpy.__version__}"
    if not check("numpy and opencv import", _pkgs):
        print("\n  Stop here. Run .\\setup.ps1 first.")
        return 1

    # ---- 2. the wrappers can reach the real tools ------------------------
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import win_common as W

    THERMAL = W.THERMAL

    def _enter():
        W.enter_thermal()
        assert os.getcwd() == THERMAL, f"cwd is {os.getcwd()}, not {THERMAL}"
        return os.path.basename(THERMAL) + "/"
    check("enter_thermal() sets the working directory", _enter)

    def _imports():
        import dataset_recording, thermal_detect, model_registry   # noqa
        return "dataset_recording, thermal_detect, model_registry"
    if not check("parent modules import", _imports):
        print("\n  The w_*.py wrappers cannot see Thermal/. Is the folder intact?")
        return 1

    def _logdir():
        import dataset_recording as DR
        got = os.path.abspath(DR.LOG_DIR)
        want = os.path.join(THERMAL, "logs")
        assert got == want, f"LOG_DIR -> {got}, expected {want}"
        return "Thermal/logs"
    check("captures would land in Thermal/logs", _logdir)

    # ---- 3. the monkeypatches actually take -----------------------------
    def _patch_cam():
        import dataset_recording as DR
        before = DR.open_camera
        W.patch_camera(DR, allow_agc=False)
        assert DR.open_camera is not before
        return "open_camera replaced"
    check("camera patch applies", _patch_cam)

    def _patch_prov():
        import dataset_recording as DR
        before = DR.open_capture
        W.patch_provenance(DR)
        assert DR.open_capture is not before
        return "open_capture wrapped"
    check("provenance patch applies", _patch_prov)

    def _patch_hb():
        import dataset_pipeline as DP
        W.patch_hand_back(DP)
        DP.hand_back(tempfile.gettempdir())   # AttributeError if unpatched
        return "os.geteuid() no longer reached"
    check("dataset_pipeline geteuid fix", _patch_hb)

    # ---- 3b. telemetry ---------------------------------------------------
    def _telemetry():
        import numpy as np
        import thermal_detect as TD

        # A Lepton with telemetry enabled sends 122 rows, not 120. The two
        # extra rows are packed status data, not image — decoded as
        # temperatures they come out around -273 C and +344 C, which is what
        # Adrian saw on 2026-09-14. They can sit at either end depending on how
        # the module is configured, so both are tested.
        counts = np.full((120, 160), int((22.0 + 273.15) * 100), np.uint16)
        tel = np.array([[0, 65535] * 80, [65535, 0] * 80], np.uint16)

        cases = {
            "none":   (counts, 0),
            "footer": (np.vstack([counts, tel]), 2),
            "header": (np.vstack([tel, counts]), 2),
        }
        for label, (frame, want_n) in cases.items():
            out, n = TD.strip_telemetry(frame)
            assert out.shape == (120, 160), \
                f"{label}: got {out.shape}, expected (120, 160)"
            assert n == want_n, f"{label}: stripped {n}, expected {want_n}"
            c = float(np.median(out)) * 0.01 - 273.15
            assert 21 < c < 23, f"{label}: median {c:.1f} C, expected ~22"
        return "header, footer and none all reduce to 120 rows"
    check("telemetry rows are detected and stripped", _telemetry)

    def _native_size():
        # native_size() must not accept a driver's placeholder default.
        class FakeCap:
            def __init__(self, w, h):
                self.w, self.h = w, h
            def isOpened(self):
                return True
            def get(self, prop):
                import cv2
                return self.w if prop == cv2.CAP_PROP_FRAME_WIDTH else self.h
            def release(self):
                pass

        import cv2
        real = cv2.VideoCapture
        try:
            for (w, h), want in (((160, 122), (160, 122)),
                                 ((160, 120), (160, 120)),
                                 ((640, 480), None)):
                cv2.VideoCapture = lambda i, b, _w=w, _h=h: FakeCap(_w, _h)
                got = W.native_size(0, cv2.CAP_DSHOW)
                assert got == want, f"{w}x{h} -> {got}, expected {want}"
        finally:
            cv2.VideoCapture = real
        return "160x122 and 160x120 accepted, 640x480 rejected"
    check("native_size reads the camera's own setting", _native_size)

    # ---- 4. end to end, with a fake camera ------------------------------
    tmp = tempfile.mkdtemp(prefix="fluxnet_selftest_")

    def _roundtrip():
        import numpy as np
        import dataset_recording as DR

        # A synthetic scene in real degrees: 22 C room, 34 C blob.
        def frame(i):
            a = np.full((120, 160), 22.0, np.float32)
            a += np.random.normal(0, 0.05, a.shape).astype(np.float32)
            x = 60 + (i % 20)
            a[50:66, x:x + 14] = 34.0
            return a

        old_logdir = DR.LOG_DIR
        DR.LOG_DIR = tmp
        try:
            d, fh, w = DR.open_capture("selftest, synthetic", 0.37, "none")
            for i in range(12):
                DR.write_frame(d, w, i, frame(i),
                               [(60.0 + (i % 20), 50.0, 14.0, 16.0, 0.9)],
                               "selftest", False)
            fh.close()
        finally:
            DR.LOG_DIR = old_logdir

        n = len(os.listdir(os.path.join(d, "npy")))
        assert n == 12, f"wrote {n} npy files, expected 12"
        back = np.load(os.path.join(d, "npy", "cap_000000.npy"))
        assert back.dtype == np.float32, f"dtype {back.dtype}"
        assert 20 < float(np.median(back)) < 24, "temperatures did not survive"
        return f"12 frames written and read back, median "\
               f"{float(np.median(back)):.1f} C"
    ok_rt = check("record a capture end to end (fake camera)", _roundtrip)

    # ---- 5. the verifier agrees ------------------------------------------
    if ok_rt:
        cap = os.path.join(tmp, sorted(os.listdir(tmp))[0])

        def _verify_good():
            import importlib.util
            here = os.path.dirname(os.path.abspath(__file__))
            spec = importlib.util.spec_from_file_location(
                "vc", os.path.join(here, "verify_capture.py"))
            vc = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vc)
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                good = vc.verify(cap)
            assert good, "verifier rejected a valid synthetic capture:\n" + buf.getvalue()[-400:]
            return "accepted valid Celsius data"
        check("verify_capture accepts good data", _verify_good)

        def _verify_bad():
            import importlib.util
            import io
            import contextlib
            import numpy as np
            here = os.path.dirname(os.path.abspath(__file__))
            spec = importlib.util.spec_from_file_location(
                "vc2", os.path.join(here, "verify_capture.py"))
            vc = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vc)

            # Same capture, rescaled to 0-255 — exactly what AGC produces.
            bad = cap + "_AGC"
            shutil.copytree(cap, bad)
            for f in os.listdir(os.path.join(bad, "npy")):
                p = os.path.join(bad, "npy", f)
                a = np.load(p)
                lo, hi = a.min(), a.max()
                np.save(p, ((a - lo) / (hi - lo) * 255).astype(np.float32))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                good = vc.verify(bad)
            assert not good, ("verifier ACCEPTED 8-bit AGC data — this is the "
                              "failure the whole folder exists to prevent")
            return "rejected 8-bit AGC data"
        check("verify_capture rejects AGC data", _verify_bad)

    shutil.rmtree(tmp, ignore_errors=True)

    # ---- summary ---------------------------------------------------------
    failed = [r for r in RESULTS if not r[0]]
    print("\n" + "=" * 66)
    if failed:
        print(f"{len(failed)} of {len(RESULTS)} checks FAILED:\n")
        for _, name, detail in failed:
            print(f"  - {name}\n      {detail}")
        print("\nSend this whole output to Haneef before recording anything.")
        return 1

    print(f"All {len(RESULTS)} checks passed.\n")
    print("Everything works except the one thing this cannot test: whether")
    print("your Lepton gives OpenCV real temperatures. Plug it in and run:\n")
    print("    py -3.12 backend_probe.py\n")
    print("Then record ONE short capture and send it over before doing a")
    print("full session — see the README.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
