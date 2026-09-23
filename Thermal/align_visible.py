#!/usr/bin/env python3
"""
Warp a Sony ZV-1 frame into the Lepton's geometry, so they overlay.

    ./run.sh align_visible.py logs/capture_X/npy/cap_000123.npy shot.jpg
    ./run.sh align_visible.py --npy frame.npy --vis shot.jpg --save fig.png

WHY A SCALE-AND-SHIFT NEVER WORKS, AND WHAT DOES.

The two cameras do not just have different fields of view — they use different
PROJECTION MODELS, and no affine transform or homography can convert between
them because the difference is nonlinear in radius.

    Lepton 3.1R   EQUIDISTANT (f-theta):   r = f * theta
    Sony ZV-1     RECTILINEAR:             r = f * tan(theta)

The Lepton's is confirmed by its own datasheet numbers: 200 px along the
diagonal x 0.59375 deg/px = 118.75 deg, matching FLIR's published 119 deg
diagonal. That linear px-to-angle relationship is what equidistant means; a
rectilinear lens cannot produce it.

Align the centres perfectly and you are still out by tan(theta)/theta:

    10 deg off-axis    1%
    20 deg             4%
    30 deg            10%
    40 deg            20%

So the centre looks right and the edges drift — which is exactly what a
stretch-to-fit overlay looks like.

HOW THIS FIXES IT. For every Lepton pixel we compute the real-world direction
it looks along, then ask where that direction lands in the Sony's rectilinear
image, and sample there:

    1. pixel radius  ->  angle          theta = r_px * DEG_PER_PX   (f-theta)
    2. angle         ->  unit vector    d = (sin th cos ph, sin th sin ph, cos th)
    3. optional rotation R              mount misalignment: yaw, pitch, roll
    4. vector        ->  Sony pixel     u = f_v * d.x/d.z + cx      (rectilinear)

That is an exact inversion of the projection difference, not a fit. What it
does NOT fix is parallax: the two lenses are not at the same point, so near
objects need a different shift from far ones. No 2D warp can fix that. Expect
the background to line up while foreground objects sit a few pixels off, and
tune for the depth you care about.

FIELD OF VIEW, AND HOW LITTLE OVERLAP THERE ACTUALLY IS. The Lepton sees 95 deg
horizontally. A ZV-1 at its widest is about 74 deg on 3:2, and setting 4:3
CROPS THE SENSOR WIDTH (13.2 -> 11.7 mm), taking it to roughly 67 deg.

Measured on a 4000x3000 frame at 67.4 deg: the photo spans the middle 71% of
the thermal WIDTH and 74% of its HEIGHT — but that is only about **50% of the
thermal frame by AREA**, because both axes are clipped at once. Half the
thermal image has no visible counterpart and comes out black. That is correct,
not a bug. The tool measures and prints the real number at startup rather than
estimating it from the FOV ratio, which overstates it.

TUNING. Run it with a window and use the trackbars. --hfov is the one that
matters most and the one you cannot read off the camera reliably, because the
ZV-1 reports 35mm-equivalent focal length for a 3:2 frame. Get the background
to line up, then stop; do not chase the foreground, that is parallax.
"""

import argparse
import os
import sys

import cv2
import numpy as np

import thermal_detect as TD

# Lepton 3.1R, 95 deg HFOV across 160 px. Same constant predict_gui.py uses.
LEPTON_W, LEPTON_H = 160, 120
DEG_PER_PX = 95.0 / 160.0          # 0.59375


def rotation(yaw_deg, pitch_deg, roll_deg):
    """Small mount-misalignment rotation, applied to the look direction."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0],
                   [0, 0, 1]])
    return Rz @ Rx @ Ry


def build_maps(vis_shape, out_w, out_h, hfov_vis, yaw=0.0, pitch=0.0,
               roll=0.0, cx_off=0.0, cy_off=0.0):
    """
    Remap tables taking the VISIBLE image into LEPTON pixel space.

    out_w/out_h can be a multiple of 160x120 to supersample for a figure; the
    angular scale is divided to match, so the field of view is unchanged.
    """
    vh, vw = vis_shape[:2]
    # Rectilinear focal length in pixels, from the horizontal field of view.
    f_v = (vw / 2.0) / np.tan(np.radians(hfov_vis) / 2.0)
    cx_v, cy_v = vw / 2.0 + cx_off, vh / 2.0 + cy_off

    sx, sy = out_w / LEPTON_W, out_h / LEPTON_H
    deg_px = DEG_PER_PX / sx                      # finer grid, same total FOV

    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    dx = xx - (out_w - 1) / 2.0
    dy = yy - (out_h - 1) / 2.0

    # 1. f-theta: pixel radius IS the angle, linearly.
    r = np.hypot(dx, dy)
    theta = np.radians(r * deg_px)
    phi = np.arctan2(dy, dx)

    # 2. unit direction in the thermal camera's frame
    st = np.sin(theta)
    d = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=-1)

    # 3. mount misalignment
    if yaw or pitch or roll:
        d = d @ rotation(yaw, pitch, roll).T

    # 4. rectilinear projection into the Sony frame
    z = d[..., 2]
    behind = z <= 1e-6                 # >90 deg off-axis: no rectilinear image
    z = np.where(behind, 1.0, z)
    mx = (f_v * d[..., 0] / z + cx_v).astype(np.float32)
    my = (f_v * d[..., 1] / z + cy_v).astype(np.float32)
    mx[behind] = -1
    my[behind] = -1
    return mx, my


def warp(vis, out_w, out_h, **kw):
    mx, my = build_maps(vis.shape, out_w, out_h, **kw)
    return cv2.remap(vis, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def thermal_rgb(path, out_w, out_h):
    """The thermal frame as a colour image at the output size."""
    arr = np.load(path) if path.endswith(".npy") else None
    if arr is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            sys.exit(f"could not read {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(TD.colorize(arr), (out_w, out_h),
                      interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser(
        description="Warp a rectilinear photo into the Lepton's f-theta frame.")
    ap.add_argument("npy", help=".npy or thermal .png")
    ap.add_argument("vis", help="the Sony frame")
    ap.add_argument("--hfov", type=float, default=67.4,
                    help="visible camera horizontal FOV in degrees. ZV-1 at "
                         "widest: ~73.7 on 3:2, ~67.4 on 4:3 (4:3 crops the "
                         "sensor width). Zoomed in, it is smaller. This is the "
                         "main thing to tune.")
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--scale", type=int, default=6,
                    help="output is 160*scale x 120*scale")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--save", default=None, help="write the blend and stop")
    ap.add_argument("--save-warp", default=None,
                    help="write the warped VISIBLE frame alone")
    args = ap.parse_args()

    vis = cv2.imread(args.vis, cv2.IMREAD_COLOR)
    if vis is None:
        sys.exit(f"could not read {args.vis}")
    out_w, out_h = LEPTON_W * args.scale, LEPTON_H * args.scale
    therm = thermal_rgb(args.npy, out_w, out_h)

    print(f"visible {vis.shape[1]}x{vis.shape[0]}  ->  "
          f"lepton grid {out_w}x{out_h}")
    print(f"lepton  95.0 deg HFOV (f-theta, {DEG_PER_PX:.5f} deg/px)")
    print(f"visible {args.hfov:.1f} deg HFOV (rectilinear)")
    # Measure the overlap rather than estimating it from the FOV ratio: both
    # axes clip at once, so the area covered is far less than hfov/95 suggests.
    _mx, _my = build_maps(vis.shape, out_w, out_h, hfov_vis=args.hfov,
                          yaw=args.yaw, pitch=args.pitch, roll=args.roll)
    _v = ((_mx >= 0) & (_mx < vis.shape[1]) &
          (_my >= 0) & (_my < vis.shape[0]))
    print(f"overlap: the photo covers {100.0 * _v.mean():.0f}% of the thermal "
          f"frame by area; the rest is black by construction")

    if args.save or args.save_warp:
        w = warp(vis, out_w, out_h, hfov_vis=args.hfov, yaw=args.yaw,
                 pitch=args.pitch, roll=args.roll)
        if args.save_warp:
            cv2.imwrite(args.save_warp, w)
            print(f"warped visible -> {args.save_warp}")
        if args.save:
            cv2.imwrite(args.save,
                        cv2.addWeighted(w, 1 - args.alpha, therm, args.alpha, 0))
            print(f"blend -> {args.save}")
        return

    win = "align  (hfov/yaw/pitch/roll)   s=save  q=quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    # Trackbars are integers, so angles are stored x10 and offsets are
    # centred at the midpoint of their range.
    cv2.createTrackbar("hfov x10", win, int(args.hfov * 10), 1200, lambda v: None)
    cv2.createTrackbar("yaw +50", win, int(args.yaw + 50), 100, lambda v: None)
    cv2.createTrackbar("pitch +50", win, int(args.pitch + 50), 100, lambda v: None)
    cv2.createTrackbar("roll +50", win, int(args.roll + 50), 100, lambda v: None)
    cv2.createTrackbar("alpha %", win, int(args.alpha * 100), 100, lambda v: None)

    while True:
        hf = max(5.0, cv2.getTrackbarPos("hfov x10", win) / 10.0)
        yw = cv2.getTrackbarPos("yaw +50", win) - 50
        pt = cv2.getTrackbarPos("pitch +50", win) - 50
        rl = cv2.getTrackbarPos("roll +50", win) - 50
        al = cv2.getTrackbarPos("alpha %", win) / 100.0

        w = warp(vis, out_w, out_h, hfov_vis=hf, yaw=yw, pitch=pt, roll=rl)
        blend = cv2.addWeighted(w, 1 - al, therm, al, 0)
        cv2.putText(blend, f"hfov {hf:.1f}  yaw {yw}  pitch {pt}  roll {rl}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1, cv2.LINE_AA)
        cv2.imshow(win, blend)

        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            base = os.path.splitext(os.path.basename(args.vis))[0]
            cv2.imwrite(f"{base}_aligned.png", blend)
            cv2.imwrite(f"{base}_warped.png", w)
            print(f"\nsaved {base}_aligned.png and {base}_warped.png")
            print(f"  --hfov {hf} --yaw {yw} --pitch {pt} --roll {rl}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
