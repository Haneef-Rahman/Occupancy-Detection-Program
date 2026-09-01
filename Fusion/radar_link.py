#!/usr/bin/env python3
"""
The radar half of fusion, as one object the thermal launcher can own.

    link = RadarLink(cli=..., data=..., close_cfg=..., far_cfg=..., adaptive=True)
    link.start()
    ...
    boxes  = link.projected(cam, ext)     # radar tracks in Lepton pixels
    status = link.status()                # for the sidebar
    link.stop()

WHY A SEPARATE OBJECT. integrated_launcher.py already owns a camera, a CNN, a
tracker and a UI, and its main loop must not block. The radar runs at ~17 Hz
against the Lepton's 9, on a serial link that punishes anyone who stops reading
it -- at 921600 baud a stalled reader overflows the kernel buffer and the
parser then spends frames resynchronising on the magic word. So the radar owns
a thread, and the launcher only ever asks for the LATEST state. Nothing the
launcher does can corrupt the stream.

ADAPTIVE SWITCHING LIVES HERE TOO, for the same reason: reconfiguring takes
~1.8 s of blackout and must never happen on the thermal thread. When it does
happen every radar track id changes, so `generation` is bumped and the fusion
layer can invalidate its radar associations without mistaking new ids for new
people.
"""

import math
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "mmWave"),):
    if p not in sys.path:
        sys.path.insert(0, p)

import project as P                                    # noqa: E402
import adaptive as A                                   # noqa: E402


class RadarLink:
    def __init__(self, cli, data, close_cfg, far_cfg=None, adaptive=False,
                 baud=921600, cli_baud=115200, ti_common=None,
                 near_min=1.0, far_enter=6.0, far_exit=5.0, far_count=2,
                 hold=3.0, dwell=30.0, metric="ground", body_width=0.50,
                 on_event=None):
        self.cli, self.data = cli, data
        self.cfgs = {A.CLOSE: close_cfg, A.FAR: far_cfg or close_cfg}
        self.adaptive = adaptive and far_cfg is not None
        self.baud, self.cli_baud = baud, cli_baud
        self.ti_common = ti_common
        self.body_width = body_width
        self.on_event = on_event or (lambda s: None)
        self.rule = A.Rule(near_min, far_enter, far_exit, far_count, hold, dwell,
                           A.lateral if metric == "lateral" else A.ground_range)

        self._lock = threading.Lock()
        self._tracks = []
        self._stop = False
        self._thread = None
        self.mode = A.CLOSE
        self.generation = 0          # bumped on every reconfigure
        self.frames = 0
        self.err = None
        self.switching = False
        self.switches = 0
        self.last_blackout = 0.0
        self.t_last = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop = True

    @property
    def alive(self):
        return self._thread is not None and self._thread.is_alive()

    # -- what the launcher asks for ---------------------------------------

    def tracks(self):
        with self._lock:
            return list(self._tracks)

    def projected(self, cam, ext):
        """
        Radar tracks as 2D boxes in the camera's frame.

        Every track is returned, in view or not. `in_view` is what lets the
        fusion protocol tell "thermal looked and said no" from "thermal never
        looked", so filtering here would throw that distinction away.
        """
        out = []
        for t in self.tracks():
            b = P.project_track(t, cam, ext, self.body_width)
            if b:
                out.append(b)
        return out

    def status(self):
        return {"mode": self.mode, "frames": self.frames, "err": self.err,
                "switching": self.switching, "switches": self.switches,
                "generation": self.generation, "n": len(self.tracks()),
                "reason": self.rule.reason, "adaptive": self.adaptive,
                "blackout": self.last_blackout,
                "stale_s": (time.time() - self.t_last) if self.t_last else None}

    # -- the thread --------------------------------------------------------

    def _apply(self, mode):
        import stream as S
        self.switching = True
        t0 = time.time()
        self.on_event(f"radar -> {mode.upper()}")
        try:
            ok = S.send_config(self.cli, self.cfgs[mode], self.cli_baud,
                               verbose=False)
        except Exception as e:
            self.err = f"{type(e).__name__}: {e}"
            ok = False
        self.last_blackout = time.time() - t0
        self.switching = False
        if not ok:
            self.err = self.err or f"could not apply {os.path.basename(self.cfgs[mode])}"
            return False
        self.mode = mode
        self.generation += 1
        self.rule.committed(mode)
        with self._lock:
            self._tracks = []          # old ids are meaningless now
        self.on_event(f"radar {mode.upper()} ready ({self.last_blackout:.1f}s)")
        return True

    def _run(self):
        import serial
        import ti_track as T
        try:
            parseFrame = T.load_ti_parser(T.find_ti_common(self.ti_common), False)
        except SystemExit as e:
            self.err = str(e)
            return

        while not self._stop:
            if not self._apply(self.mode):
                return
            buf = b""
            want = None
            try:
                with serial.Serial(self.data, self.baud, timeout=0.4) as s:
                    while not self._stop and want is None:
                        chunk = s.read(4096)
                        if chunk:
                            buf += chunk
                            if len(buf) > (1 << 20):
                                buf = buf[-65536:]
                        got, upto = T.frames_from(buf, parseFrame.parseStandardFrame)
                        buf = buf[upto:]
                        for f in got:
                            rec = T.summarise(f)
                            with self._lock:
                                self._tracks = rec["tracks"]
                            self.frames += 1
                            self.t_last = time.time()
                            if self.adaptive:
                                d = self.rule.update(rec["tracks"])
                                if d:
                                    want = d
            except Exception as e:
                self.err = f"{type(e).__name__}: {e}"
                return
            if want is None:
                return
            self.switches += 1
            self.mode = want


def self_test():
    """Exercise everything that does not need a radio."""
    ok = True

    def check(label, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {label} {extra}")

    link = RadarLink("/dev/null", "/dev/null", "close.cfg", "far.cfg",
                     adaptive=True)
    check("starts in CLOSE", link.mode == A.CLOSE)
    check("generation starts at 0", link.generation == 0)
    st = link.status()
    check("status has what a sidebar needs",
          {"mode", "frames", "n", "switching", "generation"} <= set(st))

    cam = P.Camera()
    ext = P.Extrinsics(tz=-0.08)
    with link._lock:
        link._tracks = [
            {"id": 1, "x": 0.0, "y": 4.0, "z": 0.9, "z_min": 0.0, "z_max": 1.7,
             "vx": 0.0, "vy": -0.8, "vz": 0.0},
            {"id": 2, "x": 9.0, "y": 1.0, "z": 0.9, "z_min": 0.0, "z_max": 1.7,
             "vx": 0.0, "vy": 0.0, "vz": 0.0},          # ~84 deg, outside view
        ]
    b = link.projected(cam, ext)
    check("both tracks returned, not filtered", len(b) == 2, len(b))
    check("in_view flags differ",
          sorted(x["in_view"] for x in b) == [False, True],
          [x["in_view"] for x in b])
    check("3D velocity carried through", b[0]["vel"] == [0.0, -0.8, 0.0])
    check("range carried through", abs(b[0]["range_m"] - 4.0) < 0.01)

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(self_test())
