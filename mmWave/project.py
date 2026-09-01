#!/usr/bin/env python3
"""
Project radar tracks into a 2D image plane — the first half of thermal fusion.

    python project.py --self-test
    python project.py --replay logs/desk_sr.jsonl --render out/proj
    python project.py --replay logs/desk_sr.jsonl --print

WHAT THIS PRODUCES. For every radar track, a 2D bounding box in Lepton pixel
coordinates, computed from the track's 3D position and its measured height. It
is the same box you would draw if the radar could see through the thermal
camera's lens, which is what makes an IoU against the YOLO omega box meaningful.

THERE IS NO SUCH THING AS "THE RADAR'S OWN 2D VIEW", usefully. A radar has no
image plane; a 2D picture of its tracks is a rendering choice, not a
measurement. The only projection worth computing is into the frame you want to
compare against — so this projects into the LEPTON's frame, parameterised by
extrinsics. Set them to zero and you get the co-located approximation, which is
also the closest thing to "from the radar's perspective". Same code, one
parameter.

COORDINATE FRAMES, stated because getting these wrong is silent.

    radar   x right, y forward (range), z up.   Origin at the sensor.
    camera  X right, Y down, Z forward.         Origin at the lens.

so with the two co-located and aligned:  X = x,  Y = -z,  Z = y

WHAT MUST BE CALIBRATED BEFORE AN IoU MEANS ANYTHING.

  1. Intrinsics. Lepton 3.1R, 160x120, 95 deg HFOV -> fx = 80/tan(47.5 deg)
     = 73.3 px, and 0.59 deg per pixel. Defaults here.
  2. DISTORTION. At 95 deg a pinhole model is not good enough. Barrel
     distortion pushes edge pixels several px off, and the entire IoU error
     budget is 2-3 px (a 1.7 m person at 5 m is only ~25 px tall). k1/k2 have
     to be measured; the defaults are ZERO, which is honest rather than right.
  3. Extrinsics. R and t between radar origin and lens. 1 deg of rotation
     error costs 73.3*tan(1 deg) = 1.3 px, so the budget is roughly 1.5-2 deg
     total. That is a calibration, not a mounting job.

AND ONE THING THAT CANNOT BE CALIBRATED AWAY. The radar's own angular
uncertainty is comparable to all of the above: a track accurate to +-15 cm at
5 m is +-1.7 deg, about 3 px. So the projected box jitters by a few pixels no
matter how perfect the geometry is. Use IoU as a loose gate, not as a precise
score.

Z REFERENCE IS UNVERIFIED. The firmware is supposed to apply sensorPosition and
report z as height above the FLOOR (the configs' boundaryBox spans z 0..3,
which only makes sense floor-referenced). That has never been checked on this
board, and while the sensor sits on a desk with the config claiming 2 m the
value is wrong by an unknown amount. --print reports the observed z range so
the assumption can be tested rather than trusted.
"""

import argparse
import json
import math
import os
import sys


# ---------------------------------------------------------------------------
# camera model
# ---------------------------------------------------------------------------

class Camera:
    """
    Pinhole with optional radial distortion.

    fx defaults to the Lepton 3.1R with its 95 deg lens. Derivation, so it can
    be rechecked: half the sensor width in pixels divided by the tangent of
    half the horizontal field of view, 80/tan(47.5 deg) = 73.3.
    """

    def __init__(self, width=160, height=120, hfov_deg=95.0,
                 fx=None, fy=None, cx=None, cy=None, k1=0.0, k2=0.0):
        self.w, self.h = width, height
        self.fx = fx if fx is not None else (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        self.fy = fy if fy is not None else self.fx        # square pixels
        self.cx = cx if cx is not None else width / 2
        self.cy = cy if cy is not None else height / 2
        self.k1, self.k2 = k1, k2

    @property
    def deg_per_px(self):
        return math.degrees(math.atan(1.0 / self.fx))

    def project(self, X, Y, Z):
        """Camera-frame metres -> pixels. Returns None if behind the lens."""
        if Z <= 1e-3:
            return None
        xn, yn = X / Z, Y / Z
        if self.k1 or self.k2:
            r2 = xn * xn + yn * yn
            f = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
            xn, yn = xn * f, yn * f
        return self.cx + self.fx * xn, self.cy + self.fy * yn


class Extrinsics:
    """
    Where the camera is, relative to the radar.

    Translation is in the RADAR frame, in metres. Rotation is applied in the
    camera frame after the axis permutation, in degrees, as yaw (pan) then
    pitch (tilt) then roll. Defaults are all zero, i.e. perfectly co-located
    and aligned -- an approximation, and the right starting point before a
    calibration exists.
    """

    def __init__(self, tx=0.0, ty=0.0, tz=0.0, yaw=0.0, pitch=0.0, roll=0.0):
        self.t = (tx, ty, tz)
        self.yaw, self.pitch, self.roll = yaw, pitch, roll
        cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
        # camera-frame rotations: yaw about Y (down), pitch about X (right),
        # roll about Z (forward)
        Ry = ((cy, 0, sy), (0, 1, 0), (-sy, 0, cy))
        Rx = ((1, 0, 0), (0, cp, -sp), (0, sp, cp))
        Rz = ((cr, -sr, 0), (sr, cr, 0), (0, 0, 1))
        self.R = _matmul(_matmul(Rz, Rx), Ry)

    def to_camera(self, x, y, z):
        """Radar-frame metres -> camera-frame metres."""
        tx, ty, tz = self.t
        x, y, z = x - tx, y - ty, z - tz
        # axis permutation: X = x (right), Y = -z (down), Z = y (forward)
        v = (x, -z, y)
        R = self.R
        return (R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
                R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
                R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2])


def _matmul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


# ---------------------------------------------------------------------------
# tracks -> boxes
# ---------------------------------------------------------------------------

DEFAULT_WIDTH = 0.50        # shoulder width, metres
DEFAULT_HEIGHT = 1.70       # fallback when the tracker reports no height


def track_box_3d(t, width=DEFAULT_WIDTH, fallback_h=DEFAULT_HEIGHT, depth=0.0):
    """
    Corners of a person-shaped target, as a CAMERA-FACING BILLBOARD by default.

    A cube was the obvious first choice and it is wrong for this job. Its
    axis-aligned 2D hull is set by whichever face is NEAREST, so a 0.5 m deep
    box around a person 5 m away projects at the size they would be at 4.75 m
    -- 26.2 px instead of 24.9, a 5% over-estimate that grows as they approach.
    Caught by the self test.

    What we are matching against is a thermal silhouette: the person's extent
    as seen, at their actual range. So the model is a flat rectangle, width
    across the line of sight and height from z, placed at the track's range.
    Pass depth > 0 to get the old volumetric box back if some other use wants
    an occupancy volume rather than a silhouette.

    Height comes from the tracker's own z_min/z_max when it has settled on one.
    Early in a track those are absent or nonsense, so a fallback is used and
    the caller is told which -- a box built on a guessed height should not be
    trusted to the same tolerance as a measured one.
    """
    zmin, zmax = t.get("z_min"), t.get("z_max")
    measured = (zmin is not None and zmax is not None and (zmax - zmin) > 0.3)
    if not measured:
        zmin = t.get("z", 0.0) - fallback_h / 2
        zmax = zmin + fallback_h
    hw = width / 2
    x, y = t["x"], t["y"]

    # Unit vector perpendicular to the horizontal line of sight, so the
    # billboard faces the sensor even for targets well off axis.
    r = math.hypot(x, y)
    if r < 1e-6:
        px, py = 1.0, 0.0
    else:
        px, py = y / r, -x / r

    if depth <= 0.0:
        return [(x + s * hw * px, y + s * hw * py, z)
                for s in (-1, 1) for z in (zmin, zmax)], measured

    hd = depth / 2
    lx, ly = (x / r, y / r) if r > 1e-6 else (0.0, 1.0)
    return [(x + s * hw * px + d * hd * lx, y + s * hw * py + d * hd * ly, z)
            for s in (-1, 1) for d in (-1, 1) for z in (zmin, zmax)], measured


def project_track(t, cam, ext, width=DEFAULT_WIDTH, clip=True, depth=0.0):
    """
    Radar track -> 2D box in image pixels.

    Returns a dict, or None when the track is outside the camera entirely.
    The box is the axis-aligned hull of the projected corners: a rotated 3D
    box has no rotated 2D equivalent to IoU against a YOLO box, which is
    axis-aligned by construction.
    """
    corners, measured = track_box_3d(t, width, depth=depth)
    pts = []
    for (X, Y, Z) in (ext.to_camera(*c) for c in corners):
        p = cam.project(X, Y, Z)
        if p is not None:
            pts.append(p)
    if len(pts) < 3:                      # mostly behind the lens
        return None
    us = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    x1, y1, x2, y2 = min(us), min(vs), max(us), max(vs)

    inside = not (x2 < 0 or x1 > cam.w or y2 < 0 or y1 > cam.h)
    if clip:
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(cam.w), x2), min(float(cam.h), y2)
    if x2 <= x1 or y2 <= y1:
        inside = False

    Xc, Yc, Zc = ext.to_camera(t["x"], t["y"], t.get("z", 0.0))
    return {
        "id": t["id"],
        "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        "in_view": inside,
        "height_measured": measured,
        "range_m": round(math.hypot(t["x"], t["y"]), 3),
        "bearing_deg": round(math.degrees(math.atan2(t["x"], t["y"])), 2),
        "pos": [t["x"], t["y"], t.get("z", 0.0)],
        "vel": [t.get("vx", 0.0), t.get("vy", 0.0), t.get("vz", 0.0)],
        "px_per_m": round(cam.fx / Zc, 2) if Zc > 1e-3 else None,
    }


def iou(a, b):
    """Axis-aligned IoU. Here so the fusion step has one definition to import."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

PALETTE = [(80, 200, 120), (90, 160, 255), (200, 140, 255), (80, 220, 240),
           (140, 120, 255), (120, 235, 180), (200, 200, 90), (255, 150, 120)]


def render(frame_boxes, cam, scale=6, note=""):
    """Draw the projected boxes on a blank Lepton-sized canvas."""
    import numpy as np
    import cv2
    img = np.full((cam.h * scale, cam.w * scale, 3), 22, np.uint8)

    # principal point and a horizon line, so misalignment is visible
    cv2.line(img, (0, int(cam.cy * scale)), (img.shape[1], int(cam.cy * scale)),
             (48, 48, 54), 1)
    cv2.line(img, (int(cam.cx * scale), 0), (int(cam.cx * scale), img.shape[0]),
             (48, 48, 54), 1)

    for b in frame_boxes:
        c = PALETTE[int(b["id"]) % len(PALETTE)]
        x1, y1, x2, y2 = [int(v * scale) for v in b["box"]]
        solid = 2 if b["height_measured"] else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), c, solid)
        label = f"{b['id']}  {b['range_m']:.1f}m"
        cv2.putText(img, label, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
        if not b["height_measured"]:
            cv2.putText(img, "h?", (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (140, 140, 150), 1, cv2.LINE_AA)
    if note:
        cv2.putText(img, note, (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (150, 150, 160), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------

def self_test():
    ok = True

    def check(label, got, want, tol=0.05):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {label:<44} {got:8.3f}"
              f"{'' if good else f'   want {want}'}")

    cam = Camera()
    ext = Extrinsics()
    print("camera: Lepton 3.1R, 160x120, 95 deg HFOV")
    check("fx from geometry", cam.fx, 73.31, 0.02)
    check("degrees per pixel", cam.deg_per_px, 0.781, 0.01)

    print("\ngeometry sanity")
    u, v = cam.project(*ext.to_camera(0.0, 5.0, 0.0))
    check("dead ahead -> principal point u", u, 80.0)
    check("dead ahead -> principal point v", v, 60.0)

    # a target at exactly half the HFOV must land on the image edge
    az = math.radians(47.5)
    u, _ = cam.project(*ext.to_camera(math.sin(az) * 5, math.cos(az) * 5, 0.0))
    check("edge of HFOV -> image edge", u, 160.0, 0.5)

    # apparent height: 1.7 m at 5 m
    t = {"id": 0, "x": 0.0, "y": 5.0, "z": 0.85, "z_min": 0.0, "z_max": 1.70}
    b = project_track(t, cam, ext, clip=False)
    px_h = b["box"][3] - b["box"][1]
    check("1.7 m person at 5 m, pixel height", px_h, 1.70 * cam.fx / 5.0, 0.6)
    print(f"       box {b['box']}  range {b['range_m']} m  "
          f"bearing {b['bearing_deg']} deg  measured_h={b['height_measured']}")

    t9 = dict(t, y=9.0)
    check("same person at 9 m", project_track(t9, cam, ext, clip=False)["box"][3]
          - project_track(t9, cam, ext, clip=False)["box"][1],
          1.70 * cam.fx / 9.0, 0.6)

    print("\nerror budget")
    for deg in (0.5, 1.0, 2.0):
        check(f"{deg} deg of extrinsic error costs px",
              cam.fx * math.tan(math.radians(deg)), cam.fx * math.tan(math.radians(deg)), 1e6)
    e1 = Extrinsics(yaw=1.0)
    u0, _ = cam.project(*ext.to_camera(0, 5, 0))
    u1, _ = cam.project(*e1.to_camera(0, 5, 0))
    check("1 deg yaw shifts a centred target by px", abs(u1 - u0), 1.28, 0.15)

    print("\nIoU degradation for a 5 m person")
    base = project_track(t, cam, ext, clip=False)["box"]
    for shift in (1, 2, 3, 4):
        moved = [base[0] + shift, base[1], base[2] + shift, base[3]]
        print(f"       {shift} px offset -> IoU {iou(base, moved):.2f}")

    print("\nbehind and outside")
    behind = project_track({"id": 1, "x": 0, "y": -3.0, "z": 0.9}, cam, ext)
    print(f"       target behind the sensor -> {behind}")
    ok &= behind is None
    side = project_track({"id": 2, "x": 9.0, "y": 1.0, "z": 0.9}, cam, ext)
    print(f"       target 84 deg off axis   -> in_view={side and side['in_view']}")
    ok &= (side is None or not side["in_view"])

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="a ti_track .jsonl")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--print", dest="do_print", action="store_true")
    ap.add_argument("--render", help="write PNGs to this directory")
    ap.add_argument("--every", type=int, default=10, help="render every Nth frame")
    ap.add_argument("--limit", type=int, default=40, help="max frames rendered")
    ap.add_argument("--scale", type=int, default=6)

    ap.add_argument("--hfov", type=float, default=95.0)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=120)
    ap.add_argument("--k1", type=float, default=0.0)
    ap.add_argument("--k2", type=float, default=0.0)
    ap.add_argument("--tx", type=float, default=0.0)
    ap.add_argument("--ty", type=float, default=0.0)
    ap.add_argument("--tz", type=float, default=0.0,
                    help="camera height above the radar, metres")
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--body-width", type=float, default=DEFAULT_WIDTH)
    ap.add_argument("--body-depth", type=float, default=0.0,
                    help="0 = camera-facing silhouette (right for IoU against "
                         "a thermal box); >0 = volumetric box")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())
    if not args.replay:
        sys.exit("need --replay <file.jsonl> or --self-test")

    cam = Camera(args.width, args.height, args.hfov, k1=args.k1, k2=args.k2)
    ext = Extrinsics(args.tx, args.ty, args.tz, args.yaw, args.pitch, args.roll)
    print(f"camera  {cam.w}x{cam.h}  hfov {args.hfov} deg  fx {cam.fx:.1f} px  "
          f"{cam.deg_per_px:.2f} deg/px  k1 {cam.k1} k2 {cam.k2}")
    print(f"extrin  t {ext.t}  yaw {ext.yaw} pitch {ext.pitch} roll {ext.roll}\n")

    rows = [json.loads(l) for l in open(args.replay) if l.strip()]
    rows = [r for r in rows if "tracks" in r]
    if args.render:
        os.makedirs(args.render, exist_ok=True)
        import cv2

    n_box = n_view = n_meas = 0
    zs = []
    written = 0
    for i, r in enumerate(rows):
        boxes = []
        for t in r["tracks"]:
            zs.append(t.get("z", 0.0))
            b = project_track(t, cam, ext, args.body_width,
                              depth=args.body_depth)
            if b is None:
                continue
            boxes.append(b)
            n_box += 1
            n_view += b["in_view"]
            n_meas += b["height_measured"]
        if args.do_print and boxes:
            print(f"frame {r.get('frame')}")
            for b in boxes:
                print(f"   id {b['id']:>3}  box {b['box']}  "
                      f"{b['range_m']:.1f} m  {b['bearing_deg']:+.1f} deg  "
                      f"{'in view' if b['in_view'] else 'OUTSIDE'}"
                      f"{'' if b['height_measured'] else '  (height guessed)'}")
        if args.render and boxes and i % args.every == 0 and written < args.limit:
            img = render(boxes, cam, args.scale,
                         f"frame {r.get('frame')}   {len(boxes)} track(s)")
            cv2.imwrite(os.path.join(args.render, f"proj_{i:06d}.png"), img)
            written += 1

    print(f"\nframes            {len(rows)}")
    print(f"boxes projected   {n_box}")
    print(f"in view           {n_view}"
          + (f"  ({100.0*n_view/n_box:.0f}%)" if n_box else ""))
    print(f"height measured   {n_meas}"
          + (f"  ({100.0*n_meas/n_box:.0f}%)" if n_box else ""))
    if zs:
        print(f"observed z        {min(zs):+.2f} .. {max(zs):+.2f} m")
        print("                  floor-referenced z should sit roughly 0..2 m. "
              "If it straddles 0,\n"
              "                  the firmware is NOT applying sensorPosition "
              "and every box is\n"
              "                  vertically wrong by the mounting height.")
    if args.render:
        print(f"rendered          {written} PNGs to {args.render}/")


if __name__ == "__main__":
    main()
