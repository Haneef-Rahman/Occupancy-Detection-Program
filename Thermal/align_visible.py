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


# ---------------------------------------------------------------------------
# Point-pair fitting
# ---------------------------------------------------------------------------
PARAM_KEYS = ("hfov_vis", "yaw", "pitch", "roll")


def therm_px_to_dir(px, py, out_w, out_h):
    """A pixel in the thermal output grid -> the direction it looks along."""
    dx = px - (out_w - 1) / 2.0
    dy = py - (out_h - 1) / 2.0
    sx = out_w / LEPTON_W
    theta = np.radians(np.hypot(dx, dy) * DEG_PER_PX / sx)
    phi = np.arctan2(dy, dx)
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


def dir_to_vis_px(d, vis_shape, hfov_vis, yaw=0.0, pitch=0.0, roll=0.0):
    """A direction -> where it lands in the ORIGINAL Sony frame."""
    vh, vw = vis_shape[:2]
    f = (vw / 2.0) / np.tan(np.radians(hfov_vis) / 2.0)
    if yaw or pitch or roll:
        d = rotation(yaw, pitch, roll) @ d
    z = d[2]
    if z <= 1e-6:
        return np.nan, np.nan
    return f * d[0] / z + vw / 2.0, f * d[1] / z + vh / 2.0


def residuals(pairs, out_w, out_h, vis_shape, params):
    """Per-pair error in SONY pixels. pairs = [((tx,ty),(vx,vy)), ...]."""
    out = []
    for (tx, ty), (vx, vy) in pairs:
        d = therm_px_to_dir(tx, ty, out_w, out_h)
        u, v = dir_to_vis_px(d, vis_shape, **params)
        out.append(np.hypot(u - vx, v - vy) if np.isfinite(u) else np.inf)
    return np.array(out)


def fit_params(pairs, out_w, out_h, vis_shape, init=None, iters=400):
    """
    Least-squares fit of hfov/yaw/pitch/roll to hand-clicked correspondences.

    WHY FIT THE PHYSICAL PARAMETERS AND NOT A FREE WARP. A homography or a
    thin-plate spline through the same points would snap every head into place
    exactly — and be wrong everywhere else, because it has enough freedom to
    absorb parallax into geometry that does not exist. Four parameters cannot
    do that: they can only express what the cameras can physically differ by,
    so whatever is left over in the residuals IS the parallax, and you can read
    it rather than hiding it.

    Coordinate descent with a shrinking step, not scipy: four parameters and a
    handful of points do not need more, and it keeps this file importable on a
    machine with numpy and nothing else.

    Two pairs is enough to be determined; four or more spread across the frame
    is enough to be trustworthy. Put them at DIFFERENT DEPTHS only if you want
    the fit to split the difference — heads at one depth will fit that depth.
    """
    if len(pairs) < 2:
        raise ValueError("need at least 2 point pairs")

    p = dict(init or {"hfov_vis": 67.4, "yaw": 0.0, "pitch": 0.0, "roll": 0.0})
    p = {k: float(p.get(k, 0.0)) for k in PARAM_KEYS}

    def cost(q):
        r = residuals(pairs, out_w, out_h, vis_shape, q)
        return float(np.sum(r ** 2)) if np.all(np.isfinite(r)) else 1e18

    best = cost(p)
    step = {"hfov_vis": 4.0, "yaw": 2.0, "pitch": 2.0, "roll": 2.0}
    for _ in range(iters):
        moved = False
        for k in PARAM_KEYS:
            for sgn in (1.0, -1.0):
                q = dict(p)
                q[k] += sgn * step[k]
                if k == "hfov_vis" and not (5.0 < q[k] < 170.0):
                    continue
                c = cost(q)
                if c < best - 1e-9:
                    p, best, moved = q, c, True
        if not moved:
            for k in step:
                step[k] *= 0.5
            if max(step.values()) < 1e-5:
                break
    return p, residuals(pairs, out_w, out_h, vis_shape, p)


def match_points(therm_pts, vis_pts, out_w, out_h, vis_shape, params,
                 gate_px=None):
    """
    Pair detections across the two cameras, using the current model as a prior.

    Project each thermal point into the Sony frame with the parameters we have,
    then pair it with the nearest visible detection. MUTUAL nearest only: A's
    closest must be B and B's closest must be A. One-way nearest happily maps
    three thermal heads onto one visible person, and every one of those pairs
    is then a control point pulling the spline somewhere wrong.

    `gate_px` rejects pairs further apart than a threshold, in Sony pixels.
    Default is 12% of the image width — generous enough to survive a bad
    starting hfov, tight enough that two different people do not get paired.

    Returns (pairs, unmatched_therm, unmatched_vis).
    """
    if not therm_pts or not vis_pts:
        return [], list(therm_pts), list(vis_pts)
    if gate_px is None:
        gate_px = 0.12 * vis_shape[1]

    proj = []
    for (tx, ty) in therm_pts:
        d = therm_px_to_dir(tx, ty, out_w, out_h)
        u, v = dir_to_vis_px(d, vis_shape, **params)
        proj.append((u, v))

    P = np.array(proj, np.float64)                 # (n,2) predicted
    Q = np.array(vis_pts, np.float64)              # (m,2) detected
    ok = np.isfinite(P).all(1)
    D = np.full((len(P), len(Q)), np.inf)
    if ok.any():
        D[ok] = np.linalg.norm(P[ok, None, :] - Q[None, :, :], axis=-1)

    pairs, used_t, used_v = [], set(), set()
    a_best = D.argmin(1)
    b_best = D.argmin(0)
    for i in range(len(P)):
        j = a_best[i]
        if not np.isfinite(D[i, j]) or D[i, j] > gate_px:
            continue
        if b_best[j] != i:                          # not mutual
            continue
        pairs.append((tuple(therm_pts[i]), tuple(vis_pts[j])))
        used_t.add(i)
        used_v.add(j)
    return (pairs,
            [p for i, p in enumerate(therm_pts) if i not in used_t],
            [p for j, p in enumerate(vis_pts) if j not in used_v])


def auto_align(therm_pts, vis_pts, out_w, out_h, vis_shape, init=None,
               rounds=3, gate_px=None, sweep=True):
    """
    Match, fit, re-match. The fit improves the prior, which improves the match.

    One pass is not enough when the starting hfov is wrong: a bad prior throws
    the projection far enough that half the heads fall outside the gate or pair
    with a neighbour. Fitting on the few that DID match tightens everything,
    so the next pass recovers the rest. Three rounds is comfortably past the
    point where the pairing stops changing on real data.

    OUTLIERS ARE PRUNED BETWEEN ROUNDS, and that is not optional. A head the
    other camera never saw — occluded, out of frame, missed by the detector —
    still gets mutually paired with whatever nearest survivor is inside the
    gate. One such pair drags the fit, which moves the projection, which keeps
    the bad pair inside the gate on the next round: the iteration reinforces
    its own mistake instead of escaping it. Measured on a synthetic scene with
    3 unmatched heads, un-pruned auto_align sat at 133 px RMS across all three
    rounds and recovered hfov as 66.8 against a true 61.0.

    The prune is median-based rather than mean-based for the same reason: the
    outlier is exactly what would inflate a mean and hide itself.

    WHY IT SWEEPS hfov FIRST. Pruning fixes ONE bad pair among good ones. It
    cannot fix a globally SHIFTED assignment, where a wrong starting hfov
    displaces every projection by about the same amount and each thermal head
    mutually pairs with its neighbour instead of itself. Every pair is then
    wrong in the same way, the median residual is large, and median-based
    pruning sees no outlier to remove — measured: hfov recovered as 66.8
    against a true 61.0, RMS frozen at 133 px for all three rounds.

    hfov is also the ONLY parameter we are genuinely ignorant about: yaw, pitch
    and roll start near zero because the cameras are bolted side by side, but
    the ZV-1 reports 35mm-equivalent focal length for a 3:2 frame while you
    shoot 4:3. So we try a range of hfov, run the whole match-and-fit for each,
    and keep the seed with the lowest RMS. A correct assignment converges to
    near zero and a shifted one does not, which makes RMS a sharp discriminator
    rather than a vague preference.

    Returns (pairs, params, history) where history is the per-round
    (n_pairs, rms_sony_px) so you can see it converge — or not.
    """
    if sweep:
        best = None
        for hf in np.arange(35.0, 105.0, 5.0):
            seed = dict(init or {})
            seed.update({"hfov_vis": float(hf)})
            seed.setdefault("yaw", 0.0)
            seed.setdefault("pitch", 0.0)
            seed.setdefault("roll", 0.0)
            pr, pa, hi = auto_align(therm_pts, vis_pts, out_w, out_h,
                                    vis_shape, init=seed, rounds=rounds,
                                    gate_px=gate_px, sweep=False)
            if len(pr) < 3 or not hi or not np.isfinite(hi[-1][1]):
                continue
            # Lowest RMS wins, but only among seeds that matched a comparable
            # number of points: two pairs at zero error is not better than ten
            # pairs at one pixel.
            key = (hi[-1][1] / max(1, len(pr)) ** 0.5, -len(pr))
            if best is None or key < best[0]:
                best = (key, pr, pa, hi)
        if best is not None:
            return best[1], best[2], best[3]

    params = dict(init or {"hfov_vis": 67.4, "yaw": 0.0, "pitch": 0.0,
                           "roll": 0.0})
    pairs, hist = [], []
    for _ in range(max(1, rounds)):
        pairs, _, _ = match_points(therm_pts, vis_pts, out_w, out_h,
                                   vis_shape, params, gate_px)
        if len(pairs) < 2:
            hist.append((len(pairs), float("nan")))
            break
        params, res = fit_params(pairs, out_w, out_h, vis_shape, init=params)

        # Drop pairs far worse than typical, then refit on what is left.
        # Never below 3 pairs: under that the "outlier" may be the signal.
        if len(pairs) > 3:
            med = float(np.median(res))
            keep = res <= max(3.0 * med, 5.0)
            if 3 <= int(keep.sum()) < len(pairs):
                pairs = [p for p, k in zip(pairs, keep) if k]
                params, res = fit_params(pairs, out_w, out_h, vis_shape,
                                         init=params)
        hist.append((len(pairs), float(np.sqrt(np.mean(res ** 2)))))
    return pairs, params, hist


# ---------------------------------------------------------------------------
# Thin-plate spline: the part that actually changes SHAPE
# ---------------------------------------------------------------------------
#
# hfov/yaw/pitch/roll can only move, turn and scale the sampling. They cannot
# change its SHAPE, so they cannot absorb:
#
#   * parallax — the lenses are centimetres apart, so a head at 2 m and a wall
#     at 8 m need DIFFERENT shifts. That is a depth-dependent deformation.
#   * residual lens distortion — neither lens is a perfect f-theta or a perfect
#     rectilinear; both have their own barrel/pincushion on top.
#
# A thin-plate spline can. It is the 2-D surface that passes through every
# control point while minimising bending energy, so it deforms exactly as much
# as the points demand and stays smooth everywhere else.
#
# APPLIED TO THE RESIDUAL, NOT THE WHOLE MAPPING. The physical model still does
# the projection conversion, because it extrapolates correctly to the parts of
# the frame you never clicked. The spline only carries what is left over, which
# is small and smooth. Fitting a spline to the whole mapping instead would make
# the corners — where you have no points and the f-theta/rectilinear difference
# is 30% — pure guesswork.
#
# lam is the smoothing. lam=0 passes exactly through every point, including
# your misclicks. Raising it trades exactness for a gentler field.


def _tps_kernel(r2):
    """U(r) = r^2 log r^2, the 2-D biharmonic kernel. Zero at r=0."""
    return np.where(r2 > 1e-12, r2 * np.log(np.maximum(r2, 1e-12)), 0.0)


def tps_fit(P, V, lam=0.0):
    """
    Solve a thin-plate spline from control points P (n,2) to values V (n,k).

    Returns None when the system is degenerate — fewer than 3 points, or all of
    them collinear. The caller falls back to a plain translation, which is the
    honest answer when the data cannot support a deformation.
    """
    P = np.asarray(P, np.float64)
    V = np.asarray(V, np.float64)
    n = len(P)
    if n < 3:
        return None
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    K = _tps_kernel(d2) + lam * np.eye(n)
    Pm = np.hstack([np.ones((n, 1)), P])
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K
    L[:n, n:] = Pm
    L[n:, :n] = Pm.T
    Y = np.zeros((n + 3, V.shape[1]))
    Y[:n] = V
    try:
        sol = np.linalg.solve(L, Y)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(L, Y, rcond=None)
    if not np.all(np.isfinite(sol)):
        return None
    return sol, P


def tps_eval(fit, X, Y):
    """Evaluate a fitted spline over grids X, Y -> (..., k)."""
    sol, P = fit
    n = len(P)
    pts = np.stack([X.ravel(), Y.ravel()], -1).astype(np.float64)
    d2 = ((pts[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    out = _tps_kernel(d2) @ sol[:n] \
        + np.hstack([np.ones((len(pts), 1)), pts]) @ sol[n:]
    return out.reshape(X.shape + (sol.shape[1],))


def residual_field(pairs, out_w, out_h, vis_shape, params, lam=0.0,
                   coarse=4):
    """
    A per-pixel correction (dx, dy) in SONY pixels, from the point residuals.

    Computed on a grid `coarse` times smaller and resized up. The spline is
    smooth by construction, so evaluating it coarsely costs almost nothing in
    accuracy and turns an O(pixels x points) solve into something interactive.
    Measured at 960x720 with 8 control points, against the full-resolution
    field, in the unit that matters (1 thermal px = 0.59 deg, a head is 4-15):

        coarse    time     worst error
             2     43 ms     0.030 thermal px
             4     11 ms     0.075 thermal px   <- default
             8      4 ms     0.177 thermal px
            16      2 ms     0.371 thermal px
          full    167 ms     --

    4 keeps the error under a tenth of a thermal pixel while staying live
    under a slider.
    """
    if not pairs:
        return None
    P, V = [], []
    for (tx, ty), (vx, vy) in pairs:
        d = therm_px_to_dir(tx, ty, out_w, out_h)
        u, v = dir_to_vis_px(d, vis_shape, **params)
        if not np.isfinite(u):
            continue
        P.append((tx, ty))
        V.append((vx - u, vy - v))          # what the model got WRONG
    if not P:
        return None
    P, V = np.array(P), np.array(V)

    cw, ch = max(8, out_w // coarse), max(8, out_h // coarse)
    gx, gy = np.meshgrid(np.linspace(0, out_w - 1, cw),
                         np.linspace(0, out_h - 1, ch))

    fit = tps_fit(P, V, lam)
    if fit is None:
        # Under 3 points, or collinear: a spline is not determined. A uniform
        # shift by the mean residual is the most the data supports.
        mean = V.mean(0)
        dx = np.full((ch, cw), mean[0], np.float32)
        dy = np.full((ch, cw), mean[1], np.float32)
    else:
        f = tps_eval(fit, gx, gy)
        dx, dy = f[..., 0].astype(np.float32), f[..., 1].astype(np.float32)

    return (cv2.resize(dx, (out_w, out_h), interpolation=cv2.INTER_CUBIC),
            cv2.resize(dy, (out_w, out_h), interpolation=cv2.INTER_CUBIC))


def build_maps_morph(vis_shape, out_w, out_h, params, pairs=None, lam=0.0,
                     strength=1.0):
    """build_maps() plus the spline correction, blended by `strength`."""
    mx, my = build_maps(vis_shape, out_w, out_h, **params)
    if pairs and strength > 0.0:
        f = residual_field(pairs, out_w, out_h, vis_shape, params, lam)
        if f is not None:
            valid = mx >= 0
            mx = np.where(valid, mx + strength * f[0], mx).astype(np.float32)
            my = np.where(valid, my + strength * f[1], my).astype(np.float32)
    return mx, my


def rms_in_thermal_px(res_sony_px, vis_w, hfov_vis):
    """
    Convert a Sony-pixel residual into THERMAL pixels, which is the unit that
    means something here: 1 thermal px is 0.59 deg, and a head is 4-15 px.
    """
    deg_per_sony_px = hfov_vis / float(vis_w)
    return float(np.sqrt(np.mean(np.square(res_sony_px)))
                 * deg_per_sony_px / DEG_PER_PX)


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
