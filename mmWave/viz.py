#!/usr/bin/env python3
"""
Live 3D view of what the radar tracker sees.

    ./.venv/bin/python viz.py --cfg configs/IWR6843AOP_7m_staticRetention_lp.cfg
    ./.venv/bin/python viz.py --cfg configs/... --save logs/run1.jsonl
    ./.venv/bin/python viz.py --replay logs/run1.jsonl          # no hardware

    drag / arrows   orbit          T  top-down      1  point cloud on/off
    scroll / +-     zoom           F  front         2  trails on/off
    space           pause          I  isometric     3  room box on/off
    S               snapshot       R  reset view    Q / esc  quit

WHY A THREAD. The serial link runs at 921600 baud and the sensor does not wait
for anyone: if the draw loop blocks the reader for a few hundred milliseconds
the kernel buffer overflows, bytes are lost, and the parser spends the next
several frames resynchronising on the magic word. So the reader owns the port
in its own thread and only ever publishes the LATEST frame. Drawing slowly then
costs you dropped displays, which nobody notices, instead of a corrupted
stream, which looks like a hardware fault and wastes an afternoon.

WHAT YOU ARE LOOKING AT. Coloured dots are the point cloud, coloured by which
target the on-chip tracker assigned them to; grey dots are points it kept but
attributed to nobody. Each person is a labelled box drawn from their tracked
height. The wire box is the boundaryBox from your .cfg — the tracker discards
everything outside it before it ever allocates a target, so if a person is
standing outside that box they do not exist as far as the radar is concerned.
That single fact accounts for most "it sees nothing" sessions, which is why the
box is drawn rather than described.

ON THE VERTICAL AXIS. The demo is supposed to apply the sensorPosition tilt and
height on-chip, so z should arrive as metres above the FLOOR — matching the
0..3 z range in the .cfg's boundaryBox. If your points cluster around z=0 and
go negative, that did not happen and you want --tilt-fix, which applies the
rotation here instead. The HUD prints the observed z range so you can tell
which world you are in at a glance instead of guessing.
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv missing.  ./.venv/bin/pip install opencv-python")

import stream as S


# =============================================================================
# the .cfg tells us the room
# =============================================================================

def read_cfg_geometry(path):
    """
    Pull the scene geometry out of the chirp profile.

    Drawing the boundaryBox is not decoration. The tracker's first act on every
    frame is to throw away points outside it, so a box that does not contain
    the room is indistinguishable from a broken sensor.
    """
    g = {"boundary": None, "static": None, "sensor_z": 2.0,
         "az_tilt": 0.0, "el_tilt": 0.0, "frame_ms": None}
    if not path or not os.path.exists(path):
        return g
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == "boundaryBox" and len(p) >= 7:
            g["boundary"] = [float(x) for x in p[1:7]]
        elif p[0] == "staticBoundaryBox" and len(p) >= 7:
            g["static"] = [float(x) for x in p[1:7]]
        elif p[0] == "sensorPosition" and len(p) >= 4:
            g["sensor_z"], g["az_tilt"], g["el_tilt"] = (float(p[1]), float(p[2]),
                                                         float(p[3]))
        elif p[0] == "frameCfg" and len(p) >= 6:
            g["frame_ms"] = float(p[5])
    return g


def tilt_fix(pts, sensor_z, el_tilt_deg):
    """
    Rotate the sensor frame into the room frame, if the chip did not.

    A downward-tilted sensor reports a standing person as leaning away, because
    its own z axis is tilted with it. Rotating about x by the elevation tilt
    and then adding the mounting height puts the floor back at z=0.
    """
    if pts.size == 0:
        return pts
    t = math.radians(el_tilt_deg)
    c, s = math.cos(t), math.sin(t)
    y, z = pts[:, 1].copy(), pts[:, 2].copy()
    out = pts.copy()
    out[:, 1] = y * c + z * s
    out[:, 2] = -y * s + z * c + sensor_z
    return out


# =============================================================================
# camera: an orbit around the middle of the room
# =============================================================================

class Camera:
    def __init__(self, centre, dist=11.0):
        self.centre = np.array(centre, dtype=float)
        self.home = (float(dist), math.radians(-20), math.radians(24))
        self.reset()

    def reset(self):
        self.dist, self.yaw, self.pitch = self.home

    def eye(self):
        cp = math.cos(self.pitch)
        return self.centre + self.dist * np.array([
            cp * math.sin(self.yaw), -cp * math.cos(self.yaw),
            math.sin(self.pitch)])

    def basis(self):
        eye = self.eye()
        f = self.centre - eye
        f /= np.linalg.norm(f)
        up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up)
        n = np.linalg.norm(r)
        if n < 1e-6:                       # looking straight down: pick an axis
            r = np.array([1.0, 0.0, 0.0])
        else:
            r /= n
        u = np.cross(r, f)
        return eye, r, u, f

    def project(self, pts, w, h, fov=58.0):
        """
        World -> screen. Returns (xy, depth, visible).

        Points behind the eye must be dropped rather than projected: the
        perspective divide happily maps them to plausible-looking screen
        coordinates on the wrong side of the image.
        """
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        if pts.size == 0:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0, bool)
        eye, r, u, f = self.basis()
        d = pts - eye
        vx, vy, vz = d @ r, d @ u, d @ f
        vis = vz > 0.05
        fx = (w * 0.5) / math.tan(math.radians(fov) * 0.5)
        safe = np.where(vis, vz, 1.0)
        sx = w * 0.5 + fx * vx / safe
        sy = h * 0.5 - fx * vy / safe
        return np.stack([sx, sy], 1), vz, vis


def seg(cam, a, b, w, h):
    """Project a line segment, clipping it against the near plane."""
    eye, r, u, f = cam.basis()
    a, b = np.asarray(a, float), np.asarray(b, float)
    za, zb = (a - eye) @ f, (b - eye) @ f
    near = 0.05
    if za < near and zb < near:
        return None
    if za < near:
        a = a + (b - a) * ((near - za) / (zb - za))
    elif zb < near:
        b = b + (a - b) * ((near - zb) / (za - zb))
    xy, _, _ = cam.project(np.array([a, b]), w, h)
    p, q = xy[0], xy[1]
    if not (np.isfinite(p).all() and np.isfinite(q).all()):
        return None
    lim = 20000
    if max(abs(p[0]), abs(p[1]), abs(q[0]), abs(q[1])) > lim:
        return None
    return (int(p[0]), int(p[1])), (int(q[0]), int(q[1]))


def line3(img, cam, a, b, colour, thick=1):
    s = seg(cam, a, b, img.shape[1], img.shape[0])
    if s:
        cv2.line(img, s[0], s[1], colour, thick, cv2.LINE_AA)


def box3(img, cam, x0, x1, y0, y1, z0, z1, colour, thick=1):
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]:
        line3(img, cam, c[i], c[j], colour, thick)


# =============================================================================
# appearance
# =============================================================================

BG = (18, 18, 20)
GRID = (46, 46, 52)
# Cool and desaturated on purpose: the room and the sensor are scenery, and
# must not compete with the target colours for attention. These are BGR.
ROOM = (170, 132, 86)
STATIC = (112, 88, 62)
DIM = (108, 108, 116)
TEXT = (226, 226, 232)
MUTED = (140, 140, 150)
ACCENT = (255, 200, 120)

# Distinct at a glance and distinct in greyscale, so a screen recording of this
# is still readable. BGR.
PALETTE = [(80, 200, 120), (90, 160, 255), (200, 140, 255), (80, 220, 240),
           (140, 120, 255), (120, 235, 180), (200, 200, 90), (255, 150, 120)]


def colour_for(tid):
    return PALETTE[int(tid) % len(PALETTE)]


def text(img, s, org, colour=TEXT, scale=0.44, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick,
                cv2.LINE_AA)


def text_w(s, scale=0.44, thick=1):
    return cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]


# =============================================================================
# the reader thread
# =============================================================================

class Source:
    """Publishes the latest frame. Never queues: stale frames are worthless."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.n = 0
        self.bad = 0
        self.stop = False
        self.err = None
        self.dead = False
        self.t0 = time.time()

    def publish(self, f):
        with self.lock:
            self.frame = f
            self.n += 1

    def latest(self):
        with self.lock:
            return self.frame, self.n, self.bad

    def fps(self):
        return self.n / max(1e-6, time.time() - self.t0)


def serial_reader(src, port, baud, save_path):
    """
    EVERYTHING here is inside the try, including opening the log file.

    It was not, and a missing logs/ directory killed this thread on its first
    line — before the serial port was ever opened. The window then drew an
    empty room forever with no error anywhere, which looks exactly like a
    sensor that sees nothing. A background thread that can die silently is
    worse than no thread at all.
    """
    fh = None
    buf = b""
    try:
        import serial
        if save_path:
            d = os.path.dirname(save_path)
            if d:
                os.makedirs(d, exist_ok=True)
            fh = open(save_path, "a")
        with serial.Serial(port, baud, timeout=0.4) as s:
            while not src.stop:
                chunk = s.read(4096)
                if chunk:
                    buf += chunk
                while True:
                    i = buf.find(S.MAGIC)
                    if i < 0:
                        if len(buf) > 65536:
                            buf = buf[-4096:]
                        break
                    body = buf[i + len(S.MAGIC):]
                    fr, consumed = S.parse_frame(body)
                    if fr is None:
                        buf = buf[i:]
                        break
                    buf = body[consumed:]
                    if fr is False:
                        src.bad += 1
                        continue
                    if fh:
                        fh.write(json.dumps(fr) + "\n")
                    src.publish(fr)
    except Exception as e:
        src.err = f"{type(e).__name__}: {e}"
    finally:
        if fh:
            fh.close()
        src.dead = True


def replay_reader(src, path, speed):
    """Play a saved .jsonl back at wall-clock rate so timing bugs still show."""
    try:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if not rows:
            src.err = f"{path} is empty"
            return
        dt = 0.05 / max(0.01, speed)
        while not src.stop:
            for r in rows:
                if src.stop:
                    return
                src.publish(r)
                time.sleep(dt)
    except Exception as e:
        src.err = f"{type(e).__name__}: {e}"
    finally:
        src.dead = True


# =============================================================================
# drawing
# =============================================================================

def draw_floor(img, cam, bx):
    x0, x1, y0, y1 = bx[0], bx[1], bx[2], bx[3]
    for x in np.arange(math.floor(x0), math.ceil(x1) + 0.01, 1.0):
        line3(img, cam, (x, y0, 0), (x, y1, 0), GRID)
    for y in np.arange(math.floor(y0), math.ceil(y1) + 0.01, 1.0):
        line3(img, cam, (x0, y, 0), (x1, y, 0), GRID)


def draw_sensor(img, cam, z):
    """A small marker at the sensor, plus its boresight, for orientation."""
    line3(img, cam, (-0.18, 0, z), (0.18, 0, z), ACCENT, 2)
    line3(img, cam, (0, 0, z - 0.18), (0, 0, z + 0.18), ACCENT, 2)
    line3(img, cam, (0, 0, z), (0, 0.9, z), ACCENT, 1)
    xy, _, vis = cam.project(np.array([[0, 0, z]]), img.shape[1], img.shape[0])
    if vis[0]:
        text(img, "sensor", (int(xy[0][0]) + 8, int(xy[0][1]) - 6), ACCENT, 0.4)


def draw_points(img, cam, pts, assign):
    if len(pts) == 0:
        return
    P = np.array([[p["x"], p["y"], p["z"]] for p in pts], dtype=float)
    xy, depth, vis = cam.project(P, img.shape[1], img.shape[0])
    h, w = img.shape[:2]
    for k in range(len(P)):
        if not vis[k]:
            continue
        x, y = int(xy[k][0]), int(xy[k][1])
        if not (0 <= x < w and 0 <= y < h):
            continue
        tid = assign[k] if k < len(assign) else 255
        # 253-255 are the tracker's codes for noise / too weak to associate.
        col = DIM if tid >= 253 else colour_for(tid)
        r = 3 if depth[k] < 5 else 2
        cv2.circle(img, (x, y), r, col, -1, cv2.LINE_AA)


def draw_target(img, cam, t, trail, show_trail):
    col = colour_for(t["id"])
    x, y = t["x"], t["y"]
    z0 = t.get("z_min", 0.0)
    z1 = t.get("z_max", 1.75)
    if z1 - z0 < 0.3:                       # tracker has not settled on a height
        z0, z1 = 0.0, 1.75
    hw = 0.28
    box3(img, cam, x - hw, x + hw, y - hw, y + hw, z0, z1, col, 2)
    # A stalk to the floor: without it a person floats and you cannot judge
    # range from a perspective view at all.
    line3(img, cam, (x, y, z0), (x, y, 0), col, 1)

    if show_trail and len(trail) > 1:
        pts = list(trail)
        for i in range(1, len(pts)):
            line3(img, cam, (pts[i - 1][0], pts[i - 1][1], 0),
                  (pts[i][0], pts[i][1], 0), col, 1)

    speed = math.hypot(t["vx"], t["vy"])
    xy, _, vis = cam.project(np.array([[x, y, z1 + 0.12]]),
                             img.shape[1], img.shape[0])
    if vis[0]:
        px, py = int(xy[0][0]), int(xy[0][1])
        label = f"ID {t['id']}"
        sub = f"{y:.1f} m  {speed:.1f} m/s  {z1 - z0:.2f} m"
        wlab = max(text_w(label, 0.5, 2), text_w(sub, 0.38))
        cv2.rectangle(img, (px - wlab // 2 - 6, py - 34),
                      (px + wlab // 2 + 6, py + 4), (28, 28, 32), -1)
        cv2.rectangle(img, (px - wlab // 2 - 6, py - 34),
                      (px + wlab // 2 + 6, py + 4), col, 1)
        text(img, label, (px - wlab // 2, py - 18), col, 0.5, 2)
        text(img, sub, (px - wlab // 2, py - 3), MUTED, 0.38)


def draw_hud(img, src, frame, geo, view, paused, zrange):
    h, w = img.shape[:2]
    pad = 14
    panel_w = 232
    cv2.rectangle(img, (0, 0), (panel_w, h), (24, 24, 28), -1)
    cv2.line(img, (panel_w, 0), (panel_w, h), (44, 44, 50), 1)

    y = 30
    text(img, "IWR6843AOPEVM", (pad, y), TEXT, 0.52, 2); y += 17
    text(img, "3D people tracking", (pad, y), MUTED, 0.38); y += 26

    targets = frame.get("targets", []) if frame else []
    count = len(targets)
    cv2.rectangle(img, (pad, y - 4), (panel_w - pad, y + 46), (32, 32, 38), -1)
    col = ACCENT if count else MUTED
    text(img, str(count), (pad + 10, y + 34), col, 1.3, 2)
    text(img, "IN ROOM", (pad + 62, y + 20), MUTED, 0.42)
    if frame and frame.get("presence") is not None:
        text(img, f"presence {frame['presence']}", (pad + 62, y + 36), MUTED, 0.36)
    y += 66

    rows = [
        ("frame", str(frame.get("frame", "-")) if frame else "-"),
        ("points", str(len(frame.get("points", []))) if frame else "-"),
        ("fps", f"{src.fps():.1f}"),
        ("dropped", str(src.bad)),
        ("z range", zrange),
        ("view", view),
    ]
    for k, v in rows:
        text(img, k, (pad, y), MUTED, 0.4)
        text(img, v, (panel_w - pad - text_w(v, 0.4), y), TEXT, 0.4)
        y += 18

    y += 10
    text(img, "TRACKS", (pad, y), MUTED, 0.4); y += 8
    cv2.line(img, (pad, y), (panel_w - pad, y), (44, 44, 50), 1); y += 18
    if not targets:
        text(img, "none", (pad, y), DIM, 0.4); y += 18
    for t in sorted(targets, key=lambda t: t["y"])[:8]:
        c = colour_for(t["id"])
        cv2.rectangle(img, (pad, y - 9), (pad + 9, y), c, -1)
        text(img, f"ID {t['id']}", (pad + 16, y), TEXT, 0.4)
        d = math.hypot(t["x"], t["y"])
        rv = f"{d:.1f} m"
        text(img, rv, (panel_w - pad - text_w(rv, 0.4), y), MUTED, 0.4)
        y += 18

    keys = ["drag/arrows orbit", "scroll zoom", "T top  F front  I iso",
            "1 points  2 trails  3 box", "space pause   S snap   Q quit"]
    y = h - 14 - 16 * len(keys)
    for k in keys:
        text(img, k, (pad, y), (92, 92, 100), 0.36)
        y += 16

    if paused:
        text(img, "PAUSED", (w - 90, 30), (90, 160, 255), 0.6, 2)

    # A banner, not a log line. The failure mode being guarded against is a
    # dead reader looking identical to an empty room, so it has to be
    # impossible to miss while looking at the 3D view.
    msg = None
    if src.err:
        msg = f"READER FAILED — {src.err}"
    elif getattr(src, "dead", False):
        msg = "READER STOPPED — no longer listening to the sensor"
    elif src.n == 0:
        waited = time.time() - src.t0
        if waited > 3.0:
            msg = (f"NO FRAMES after {waited:.0f}s — the sensor is not sending. "
                   f"Did the profile end in sensorStart?")
    if msg:
        cx = (panel_w + w) // 2
        tw = text_w(msg, 0.46, 1)
        cv2.rectangle(img, (cx - tw // 2 - 16, h - 56), (cx + tw // 2 + 16, h - 22),
                      (26, 22, 40), -1)
        cv2.rectangle(img, (cx - tw // 2 - 16, h - 56), (cx + tw // 2 + 16, h - 22),
                      (110, 130, 255), 1)
        text(img, msg, (cx - tw // 2, h - 35), (140, 160, 255), 0.46)


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default="/dev/cu.usbserial-010821020")
    ap.add_argument("--data", default="/dev/cu.usbserial-010821021")
    ap.add_argument("--cfg")
    ap.add_argument("--no-config", action="store_true")
    ap.add_argument("--drop-handshake", action="store_true",
                    help="deassert DTR/RTS on open; on this board that SILENCES "
                         "the demo, so only for hardware that needs it")
    ap.add_argument("--no-check", action="store_true",
                    help="send the profile without first probing for the prompt")
    ap.add_argument("--replay", help="play a saved .jsonl instead of the sensor")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed")
    ap.add_argument("--save", help="also append every frame to this .jsonl")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--cli-baud", type=int, default=115200)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=760)
    ap.add_argument("--trail", type=int, default=60, help="trail length, frames")
    ap.add_argument("--tilt-fix", action="store_true",
                    help="apply sensorPosition tilt/height here, if the chip "
                         "did not (see the note at the top of this file)")
    args = ap.parse_args()

    geo = read_cfg_geometry(args.cfg or args.replay and None)
    bx = geo["boundary"] or [-3, 3, 0, 8, 0, 3]

    if not args.replay and not args.no_config:
        if not args.cfg:
            sys.exit("need --cfg, or --no-config, or --replay <file.jsonl>")
        if not os.path.exists(args.cfg):
            sys.exit(f"no such file: {args.cfg}")
        print(f"configuring from {args.cfg} ...")
        if not S.send_config(args.cli, args.cfg, args.cli_baud, True,
                             check=not args.no_check,
                             drop_handshake=args.drop_handshake):
            sys.exit(1)
        print("configured; sensor started\n")

    src = Source()
    if args.replay:
        th = threading.Thread(target=replay_reader,
                              args=(src, args.replay, args.speed), daemon=True)
    else:
        th = threading.Thread(target=serial_reader,
                              args=(src, args.data, args.baud, args.save),
                              daemon=True)
    th.start()

    centre = [(bx[0] + bx[1]) / 2, (bx[2] + bx[3]) / 2, 0.9]
    span = max(bx[1] - bx[0], bx[3] - bx[2])
    cam = Camera(centre, dist=span * 1.5)

    win = "FLUXNET  mmWave"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    drag = {"on": False, "x": 0, "y": 0}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            drag.update(on=True, x=x, y=y)
        elif event == cv2.EVENT_LBUTTONUP:
            drag["on"] = False
        elif event == cv2.EVENT_MOUSEMOVE and drag["on"]:
            cam.yaw += (x - drag["x"]) * 0.008
            cam.pitch = max(-1.45, min(1.45, cam.pitch + (y - drag["y"]) * 0.008))
            drag.update(x=x, y=y, moved=True)
        elif event == cv2.EVENT_MOUSEWHEEL:
            cam.dist = max(1.5, min(60.0, cam.dist * (0.9 if flags > 0 else 1.1)))

    cv2.setMouseCallback(win, on_mouse)

    trails = {}
    show_points = show_trails = show_box = True
    paused = False
    view = "iso"
    last = None
    shots = 0

    while True:
        # The label must follow the camera, not the last key pressed.
        if drag.pop("moved", False):
            view = "free"
        if not th.is_alive() and not src.dead:
            src.dead = True                # died without reaching its finally

        frame, _, _ = src.latest()
        if paused and last is not None:
            frame = last
        else:
            last = frame

        img = np.full((args.height, args.width, 3), BG, np.uint8)

        pts = frame.get("points", []) if frame else []
        targets = list(frame.get("targets", [])) if frame else []
        assign = frame.get("assign", []) if frame else []

        if args.tilt_fix and (pts or targets):
            if pts:
                P = tilt_fix(np.array([[p["x"], p["y"], p["z"]] for p in pts]),
                             geo["sensor_z"], geo["el_tilt"])
                pts = [dict(p, x=float(P[i][0]), y=float(P[i][1]),
                            z=float(P[i][2])) for i, p in enumerate(pts)]
            if targets:
                T = tilt_fix(np.array([[t["x"], t["y"], t["z"]] for t in targets]),
                             geo["sensor_z"], geo["el_tilt"])
                for i, t in enumerate(targets):
                    t["x"], t["y"], t["z"] = float(T[i][0]), float(T[i][1]), float(T[i][2])

        zs = [p["z"] for p in pts]
        zrange = f"{min(zs):+.1f}..{max(zs):+.1f}" if zs else "-"

        draw_floor(img, cam, bx)
        if show_box:
            box3(img, cam, bx[0], bx[1], bx[2], bx[3], bx[4], bx[5], ROOM, 1)
            if geo["static"]:
                s = geo["static"]
                box3(img, cam, s[0], s[1], s[2], s[3], s[4], s[5], STATIC, 1)
        draw_sensor(img, cam, geo["sensor_z"] if args.tilt_fix else 0.0)

        if show_points and pts:
            draw_points(img, cam, pts, assign)

        alive = set()
        for t in targets:
            alive.add(t["id"])
            trails.setdefault(t["id"], deque(maxlen=args.trail)).append((t["x"], t["y"]))
        for tid in list(trails):
            if tid not in alive:
                trails[tid].clear()
                if not trails[tid]:
                    del trails[tid]

        for t in sorted(targets, key=lambda t: -t["y"]):     # far first
            draw_target(img, cam, t, trails.get(t["id"], ()), show_trails)

        draw_hud(img, src, frame, geo, view, paused, zrange)
        cv2.imshow(win, img)

        k = cv2.waitKey(16) & 0xFF
        if k in (27, ord('q')):
            break
        elif k == ord(' '):
            paused = not paused
        elif k in (ord('t'), ord('T')):
            cam.yaw, cam.pitch, view = 0.0, math.radians(88), "top"
        elif k in (ord('f'), ord('F')):
            cam.yaw, cam.pitch, view = 0.0, math.radians(6), "front"
        elif k in (ord('i'), ord('I')):
            cam.yaw, cam.pitch, view = math.radians(-20), math.radians(24), "iso"
        elif k in (ord('r'), ord('R')):
            cam.reset(); view = "iso"
        elif k == ord('1'):
            show_points = not show_points
        elif k == ord('2'):
            show_trails = not show_trails
        elif k == ord('3'):
            show_box = not show_box
        elif k in (ord('+'), ord('=')):
            cam.dist = max(1.5, cam.dist * 0.9)
        elif k in (ord('-'), ord('_')):
            cam.dist = min(60.0, cam.dist * 1.1)
        elif k == 0:
            pass
        elif k in (ord('s'), ord('S')):
            shots += 1
            p = f"mmwave_view_{int(time.time())}_{shots}.png"
            cv2.imwrite(p, img)
            print(f"saved {p}")
        elif k == 81:
            cam.yaw -= 0.08
        elif k == 83:
            cam.yaw += 0.08
        elif k == 82:
            cam.pitch = min(1.45, cam.pitch + 0.06)
        elif k == 84:
            cam.pitch = max(-1.45, cam.pitch - 0.06)

    src.stop = True
    cv2.destroyAllWindows()
    print(f"\nframes {src.n}   dropped {src.bad}   {src.fps():.1f} fps")
    if src.err:
        print(f"reader error: {src.err}")
    elif src.n == 0:
        print("No frames arrived. Run stream.py first — it prints why.")


if __name__ == "__main__":
    main()
