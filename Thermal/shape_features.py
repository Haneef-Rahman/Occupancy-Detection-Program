#!/usr/bin/env python3
"""
Measurable invariants of the "human family" of density distributions.

Theory (see SHAPE_THEORY.md): a thermal frame IS a density distribution
rho(x,y). There is no single "human rho" — every person, pose, distance and
viewing angle produces a different one. What we call human shape is a FAMILY
of such functions, and recognition means asking whether a given rho falls
inside it.

Since the family cannot be enumerated, it is characterised by INVARIANTS:
quantities that stay nearly constant across it and differ outside it. Three
sets of physical constraints make the family small enough to have invariants
at all, and this module is organised around them:

  STRUCTURAL   articulated rigid segments, bilateral symmetry, fixed
               anthropometric proportions, limbs projecting from a trunk
               -> solidity can never reach 1.0, defects are guaranteed

  THERMAL      heat rises from a near-uniform internal core and escapes
               through layers of differing insulation, diffusing on the way
               -> several smooth warm centres, not one sharp-edged source

  PROJECTIVE   the sensor sees a 2-D projection from a fixed geometry
               -> overhead and forward views are genuinely different
                  families, so ratios are tested per view

IMPORTANT: nothing here filters. It measures and returns numbers. Thresholds
come later, from the measured distributions — that is the difference between
engineering and tuning.
"""

import numpy as np
import cv2

SIGMA_SB = 5.670374419e-8
T_CORE_C = 37.0

# Anthropometric ratios (ANSUR II class figures, adult). Scale-invariant, so
# they hold regardless of range — which is why they are usable when the
# absolute pixel size is unknown.
ANTHRO = {
    "shoulder_over_stature": 0.23,   # biacromial breadth / height
    "head_over_shoulder": 0.38,      # head breadth / biacromial breadth
    "head_over_stature": 0.13,       # ~7.5 heads tall
}


# ---------------------------------------------------------------------------
# Skeleton (Zhang-Suen thinning, no external dependency)
# ---------------------------------------------------------------------------
def skeletonize(binary):
    """
    Zhang-Suen thinning. Returns a 1-px-wide skeleton.

    Viable only when limbs are wide enough to survive thinning: at a 2.4 m
    ceiling a limb is ~9 px and the skeleton is stable; above ~3.5 m limbs
    fall to 3-4 px and it degenerates into noise. Check the pixel budget for
    your mount height before trusting the topology numbers.
    """
    img = (binary > 0).astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2 = p[:-2, 1:-1]; P3 = p[:-2, 2:]; P4 = p[1:-1, 2:]
            P5 = p[2:, 2:];    P6 = p[2:, 1:-1]; P7 = p[2:, :-2]
            P8 = p[1:-1, :-2]; P9 = p[:-2, :-2]
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
                    for i in range(8))
            if step == 0:
                cond = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                cond = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            rm = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if rm.any():
                img[rm] = 0
                changed = True
    return img


def skeleton_topology(skel, prune_px=3):
    """
    Count structure in the skeleton: endpoints (limb tips), junctions
    (trunk branch points), and total length.

    A body branches — head, arms, legs radiate from a trunk. A manufactured
    object thins to a plain line or a T: few endpoints, no real junction.
    """
    s = (skel > 0).astype(np.uint8)
    if s.sum() == 0:
        return dict(sk_endpoints=0, sk_junctions=0, sk_length=0)

    # Count neighbours with a hollow kernel (centre 0) in float32 — a uint8
    # kernel silently loses the centre weight in cv2.filter2D.
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32)
    nb = cv2.filter2D(s.astype(np.float32), cv2.CV_32F, k,
                      borderType=cv2.BORDER_CONSTANT)
    on = s == 1
    endpoints = int((on & (nb == 1)).sum())      # limb tip
    junctions = int((on & (nb >= 3)).sum())      # trunk branch point

    # Prune: adjacent junction pixels are one junction, not several.
    if junctions:
        jm = (on & (nb >= 3)).astype(np.uint8)
        n_j, _ = cv2.connectedComponents(jm)
        junctions = max(0, n_j - 1)

    return dict(sk_endpoints=endpoints, sk_junctions=junctions,
                sk_length=int(s.sum()))


# ---------------------------------------------------------------------------
# Peura & Iivarinen descriptors, via Zhang & Lu (2004) §3.1.1
#
# The survey's own recommendation for these: "simple global descriptors ...
# are usually used as FILTERS TO ELIMINATE FALSE HITS or combined with other
# shape descriptors" — exactly the laptop problem. It also warns (Fig. 2)
# that no single one suffices: eccentricity fails on some shapes where
# circularity succeeds and vice versa, so they must be combined.
# ---------------------------------------------------------------------------
def contour_variances(contour):
    """
    circular_variance : spread of boundary radii about their mean.
        circle -> 0, ellipse -> small, rectangle -> larger.

    elliptic_variance : the same spread measured against the BEST-FIT
        ELLIPSE, using Mahalanobis distance under the boundary's own
        covariance. Every point of a perfect ellipse has identical
        Mahalanobis radius, so an ellipse scores 0 regardless of how
        elongated it is — while a rectangle's four corners sit far outside
        its inscribed ellipse and drive the value up.

        This is the discriminator for the laptop case: a body seen from
        above is approximately elliptical (head + shoulders); a laptop is
        not, no matter its aspect ratio or rotation.
    """
    out = {}
    p = contour.reshape(-1, 2).astype(np.float64)
    if p.shape[0] < 5:
        return out

    c = p.mean(axis=0)
    d = p - c

    # --- circular variance: deviation from a circle
    r = np.sqrt((d ** 2).sum(axis=1))
    mu_r = r.mean()
    if mu_r > 1e-9:
        out["circular_var"] = float(((r - mu_r) ** 2).mean() / (mu_r ** 2))

    # --- elliptic variance: deviation from the best-fit ellipse
    C = np.cov(d.T)
    if C.shape == (2, 2) and abs(np.linalg.det(C)) > 1e-12:
        Ci = np.linalg.inv(C)
        # Mahalanobis radius of every boundary point
        rE = np.sqrt(np.einsum("ij,jk,ik->i", d, Ci, d))
        mu_E = rE.mean()
        if mu_E > 1e-9:
            out["elliptic_var"] = float(((rE - mu_E) ** 2).mean() / (mu_E ** 2))
    return out


def bending_energy(contour, smooth=3):
    """
    Mean squared curvature along the boundary, plus the count of sharp
    curvature peaks.

    A rectangle concentrates ALL of its turning into four points: high peak
    curvature, four corners, low curvature everywhere else. A body turns
    gradually and continuously, so its curvature is spread out. `n_corners`
    is therefore close to 4 for any rectangle at any rotation, and rarely
    exactly 4 for a person.
    """
    out = {}
    p = contour.reshape(-1, 2).astype(np.float64)
    n = p.shape[0]
    if n < 12:
        return out

    # Smooth the boundary before differentiating — curvature on a raw
    # pixel-stepped contour is dominated by staircase noise.
    k = max(3, smooth | 1)
    pad = np.vstack([p[-k:], p, p[:k]])
    ker = np.ones(k) / k
    xs = np.convolve(pad[:, 0], ker, mode="same")[k:-k]
    ys = np.convolve(pad[:, 1], ker, mode="same")[k:-k]

    dx = np.gradient(xs); dy = np.gradient(ys)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5
    denom[denom < 1e-9] = 1e-9
    curv = np.abs(dx * ddy - dy * ddx) / denom

    perim = float(cv2.arcLength(contour, True))
    scale = perim / max(1, n)                      # normalise for size
    out["bending_energy"] = float((curv ** 2).mean() * scale)
    out["curv_max"] = float(curv.max() * scale)

    # Corner count: curvature peaks well above the boundary's own median.
    thr = max(curv.mean() + 2.0 * curv.std(), 0.15)
    peaks = curv > thr
    if peaks.any():
        lab = np.diff(np.concatenate(([0], peaks.view(np.int8), [0])))
        out["n_corners"] = int((lab == 1).sum())
    else:
        out["n_corners"] = 0
    return out


# ---------------------------------------------------------------------------
# STRUCTURAL — articulation, symmetry, non-convexity
# ---------------------------------------------------------------------------
def structural_features(mask):
    out = {}
    m8 = (mask > 0).astype(np.uint8)
    # CHAIN_APPROX_NONE, not SIMPLE: SIMPLE collapses straight runs to their
    # endpoints, so an axis-aligned rectangle arrives as FOUR points and every
    # contour-based descriptor silently returns nothing. The boundary-sampling
    # descriptors below need every pixel of the outline.
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return out
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    per = float(cv2.arcLength(c, True))
    if area <= 0 or per <= 0:
        return out

    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    x, y, w, h = cv2.boundingRect(c)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)

    out["area"] = area
    out["compactness"] = 4.0 * np.pi * area / (per * per)   # circle=1, limbs<<1
    out["solidity"] = area / hull_area if hull_area > 0 else 0.0
    out["extent"] = area / float(w * h)
    out["rect_fill"] = area / (rw * rh) if rw > 1 and rh > 1 else 1.0
    out["aspect"] = h / float(max(1, w))

    # Convexity defects = the gaps articulation guarantees: armpits, the space
    # between legs, the neck notch. A convex object has none.
    n_def, deepest, mean_def = 0, 0.0, 0.0
    if len(c) > 3:
        hi = cv2.convexHull(c, returnPoints=False)
        if hi is not None and len(hi) > 3:
            try:
                d = cv2.convexityDefects(c, np.sort(hi[:, 0])[::-1][:, None])
                if d is not None:
                    depths = d[:, 0, 3] / 256.0
                    sig = depths[depths > 1.0]
                    n_def = int(sig.size)
                    deepest = float(depths.max())
                    mean_def = float(sig.mean()) if sig.size else 0.0
            except cv2.error:
                pass
    out["n_defects"] = n_def
    out["deepest_defect"] = deepest
    out["mean_defect"] = mean_def

    # Bilateral symmetry about the principal axis — bodies are symmetric,
    # most clutter is not.
    M = cv2.moments(m8, binaryImage=True)
    if M["m00"] > 0:
        mu20, mu02, mu11 = M["mu20"], M["mu02"], M["mu11"]
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        H, W = m8.shape

        # Rotate the principal axis vertical AND translate the centroid to the
        # image centre in one transform. Without the translation, cv2.flip
        # mirrors about the IMAGE axis rather than the body's own axis, and
        # the overlap measures nothing.
        rot = cv2.getRotationMatrix2D((cx, cy), np.degrees(theta), 1.0)
        rot[0, 2] += W / 2.0 - cx
        rot[1, 2] += H / 2.0 - cy
        al = cv2.warpAffine(m8, rot, (W, H), flags=cv2.INTER_NEAREST)

        flip = cv2.flip(al, 1)                   # mirror about the body axis
        inter = float((al & flip).sum())
        union = float((al | flip).sum())
        out["symmetry"] = inter / union if union > 0 else 0.0

        tr = mu20 + mu02
        det = mu20 * mu02 - mu11 * mu11
        disc = max(0.0, tr * tr / 4.0 - det)
        l1, l2 = tr / 2 + np.sqrt(disc), tr / 2 - np.sqrt(disc)
        out["eccentricity"] = float(np.sqrt(1 - l2 / l1)) if l1 > 0 else 0.0

    # Peura & Iivarinen descriptors (Zhang & Lu 2004, §3.1.1)
    out.update(contour_variances(c))
    out.update(bending_energy(c))
    out["convexity"] = float(cv2.arcLength(hull, True) / per) if per > 0 else 1.0

    # Skeleton topology — how many limbs radiate from a trunk.
    out.update(skeleton_topology(skeletonize(m8)))
    out["sk_len_over_area"] = out["sk_length"] / area if area > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# THERMAL — internal source, insulation layers, diffusion
# ---------------------------------------------------------------------------
def thermal_features(data, mask, ambient_c, prominence=2.5, min_sep=2):
    out = {}
    ys, xs = np.where(mask > 0)
    if xs.size < 4:
        return out
    v = data[ys, xs]

    out["t_max"] = float(v.max())
    out["t_mean"] = float(v.mean())
    out["t_std"] = float(v.std())
    out["t_excess"] = float(v.mean() - ambient_c)

    # Warm centres. A body has several (face, neck, hands, clothing gaps);
    # a single manufactured source has one.
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = np.full((y1 - y0, x1 - x0), -999.0, np.float32)
    patch[ys - y0, xs - x0] = v
    k = 2 * min_sep + 1
    dil = cv2.dilate(patch, np.ones((k, k), np.uint8))
    pk = ((patch >= dil - 1e-6) & (patch >= v.max() - prominence)).astype(np.uint8)
    n, _ = cv2.connectedComponents(pk)
    out["peaks"] = max(1, n - 1)

    # Diffusion smoothness: heat conducted through tissue has soft gradients;
    # a manufactured source has sharp edges. Measured as mean gradient
    # magnitude inside the blob, normalised by its temperature range.
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    valid = patch > -900
    inner = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if inner.sum() > 4:
        g = np.sqrt(gx[inner] ** 2 + gy[inner] ** 2)
        rng = max(0.1, float(v.max() - v.min()))
        out["grad_mean"] = float(g.mean() / rng)
        out["grad_p90"] = float(np.percentile(g, 90) / rng)

    # Total excess radiated power (Stefan-Boltzmann) — a physical quantity,
    # not an arbitrary score.
    t_k, a_k = v + 273.15, ambient_c + 273.15
    out["radiant_excess"] = float(0.98 * SIGMA_SB * np.sum(t_k ** 4 - a_k ** 4))
    return out


# ---------------------------------------------------------------------------
# CONTEXT — the shape of the warm region SURROUNDING the hot blob
#
# A laptop does not present as a rectangle at the detection threshold: only
# its vent and CPU corner clear ambient+4, and that hot patch is an irregular
# blob which passes every shape test. The rectangle exists one temperature
# level down — the whole chassis sits a degree or two above ambient.
#
# So the object's true geometry is recovered by re-thresholding LOWER and
# measuring the region that contains the hot blob. This is a level-set /
# scale-space argument (Zhang & Lu §3.2.4) applied to temperature rather than
# to spatial scale: examine the shape across the parameter and ask whether it
# stays self-consistent.
#
#   person  : the warm footprint is a slightly larger body — still organic,
#             still non-convex. Shape character is PRESERVED across levels.
#   laptop  : a blobby hot patch sits inside a hard rectangle. Shape character
#             CHANGES completely between levels — that change is the signal.
# ---------------------------------------------------------------------------
def thermal_context(data, blob_mask, ambient_c, low_delta=1.5, tmax=None,
                    grow_px=6):
    """
    Re-threshold at ambient+low_delta, isolate the warm region containing this
    blob, and describe it. Returns ctx_* features plus the growth ratio.
    """
    out = {}
    if blob_mask.sum() == 0:
        return out

    warm = data > (ambient_c + low_delta)
    if tmax is not None:
        warm &= data <= (tmax + 6.0)     # allow hotter than the human band:
                                         # the point is to SEE the equipment
    warm = warm.astype(np.uint8)
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    n, lab = cv2.connectedComponents(warm)
    if n <= 1:
        return out

    # Which warm component contains the hot blob?
    b = blob_mask > 0
    ids, counts = np.unique(lab[b & (lab > 0)], return_counts=True)
    if ids.size == 0:
        return out
    host = int(ids[np.argmax(counts)])
    host_mask = (lab == host).astype(np.uint8)

    hot_area = float(b.sum())
    ctx_area = float(host_mask.sum())
    out["ctx_area"] = ctx_area
    out["ctx_growth"] = ctx_area / max(1.0, hot_area)

    cnts, _ = cv2.findContours(host_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return out
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    per = float(cv2.arcLength(c, True))
    if area <= 0 or per <= 0:
        return out

    hull = cv2.convexHull(c)
    ha = float(cv2.contourArea(hull))
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)

    out["ctx_solidity"] = area / ha if ha > 0 else 0.0
    out["ctx_rect_fill"] = area / (rw * rh) if rw > 1 and rh > 1 else 1.0
    out["ctx_compactness"] = 4.0 * np.pi * area / (per * per)
    out.update({f"ctx_{k}": v for k, v in contour_variances(c).items()})

    # Does the hot patch sit INSIDE a much larger rigid body? That pairing —
    # small irregular hot spot within a large high-rect-fill region — is the
    # signature of powered equipment, and does not occur for a person, whose
    # warm footprint is only modestly larger than their hot regions.
    out["is_equipment_like"] = float(
        out["ctx_growth"] > 2.5 and out["ctx_rect_fill"] > 0.80
    )
    return out


# ---------------------------------------------------------------------------
# PROJECTIVE — view-dependent anthropometric ratios
# ---------------------------------------------------------------------------
def projective_features(mask, view="vertical"):
    """
    vertical (overhead): head is a distance-transform peak nested inside the
        shoulder region — a concentric 'bullseye'. Test the ratio of the
        inner peak's extent to the outer extent against head/shoulder ~0.38.

    horizontal (forward): the row-projection profile carries the
        head -> shoulders -> torso -> legs signature. Test the width step.
    """
    out = {}
    m8 = (mask > 0).astype(np.uint8)
    if m8.sum() < 10:
        return out

    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    out["dt_max"] = float(dist.max())

    ys, xs = np.where(m8 > 0)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1

    if view == "vertical":
        # Concentric structure: how much of the blob lies within the core
        # (>60% of peak distance). A head-in-shoulders gives a small, compact
        # core; a flat slab gives a long ridge instead.
        core = dist > 0.6 * dist.max()
        out["core_frac"] = float(core.sum()) / float(m8.sum())
        if core.sum() > 0:
            cy, cx = np.where(core)
            core_w = cx.max() - cx.min() + 1
            core_h = cy.max() - cy.min() + 1
            out["core_over_span"] = float(max(core_w, core_h)) / float(max(w, h))
            out["core_round"] = float(min(core_w, core_h)) / float(max(1, max(core_w, core_h)))
    else:
        # Row widths from top to bottom = the vertical silhouette profile.
        prof = m8[ys.min():ys.max() + 1, :].sum(axis=1).astype(np.float32)
        if prof.max() > 0:
            p = prof / prof.max()
            top = p[:max(1, len(p) // 5)]          # head band, top 20%
            body = p[max(1, len(p) // 5):]
            out["head_band_w"] = float(top.mean())
            out["body_band_w"] = float(body.mean()) if body.size else 0.0
            out["head_over_body"] = (out["head_band_w"] / out["body_band_w"]
                                     if out["body_band_w"] > 0 else 0.0)
            # a step (narrow head over wide shoulders) is the human signature
            out["profile_step"] = float(p.max() - top.mean())
    out["span_ratio"] = float(min(w, h)) / float(max(w, h))
    return out


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# OMEGA (head-shoulder) detection
#
# After Li, Zhang, Huang & Tan, "Rapid and Robust Human Detection and Tracking
# Based on Omega-Shape Features", ICIP 2009. Their motivation is exactly ours:
# "full-body human detection often suffers from occlusions among individuals
# and scenes in which people are not necessarily standing", and their key
# observation is that "the head-shoulder part has a distinctive omega-like
# shape in almost all view angles".
#
# Their IMPLEMENTATION is a Viola-Jones cascade followed by a local-HOG
# AdaBoost classifier — both trained, which we cannot use without importing a
# training set and the per-site calibration burden that comes with it. The
# GEOMETRY, however, needs no training: an omega is a raised dome flanked by
# two outward-flaring slopes, and that is directly measurable on the upper
# contour of a thermal blob.
#
# Why it earns its place alongside body-extension:
#   * SITTING people have no legs to find, but still have head and shoulders.
#   * MERGED people produce one blob with TWO domes — so counting omegas
#     recovers the count that full-body segmentation loses.
#
# Note the omega is a forward/oblique-view feature. Directly overhead there is
# no dome, and the equivalent structure is the concentric head-inside-shoulders
# already measured by core_frac / core_over_span in projective_features().
# ---------------------------------------------------------------------------
def upper_profile(mask):
    """
    Height of the silhouette's top edge for each column: h(x).

    The omega lives on the upper contour. Taking the topmost filled pixel per
    column reduces the 2-D shape to a 1-D curve in which each head becomes a
    local maximum — which is what makes multi-person counting cheap.
    """
    m = (mask > 0)
    cols = np.where(m.any(axis=0))[0]
    if cols.size == 0:
        return None, None
    top = np.argmax(m[:, cols], axis=0).astype(np.float32)
    h = (top.max() - top)          # invert: heads become peaks
    return cols, h


def find_omegas(mask, min_prominence=0.22, min_sep_frac=0.18):
    """
    Locate head-shoulder omegas on the upper contour.

    Returns a list of dicts: {x, height, width, shoulder_w, ratio, score}.
    More than one means the blob contains more than one person — the merged
    case that full-body detection cannot resolve.
    """
    cols, h = upper_profile(mask)
    if cols is None or cols.size < 9:
        return []

    span = float(cols.max() - cols.min() + 1)
    hmax = float(h.max())
    if hmax < 3:
        return []

    # Smooth, then find domes as local maxima with real prominence, so
    # boundary noise does not register as a head.
    k = max(3, int(span * 0.08) | 1)
    hs = np.convolve(np.pad(h, k, mode="edge"), np.ones(k) / k, mode="same")[k:-k]

    peaks = []
    for i in range(1, len(hs) - 1):
        if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1]:
            # prominence: drop to the lower of the two flanking minima
            left = hs[:i].min() if i else hs[i]
            right = hs[i + 1:].min() if i + 1 < len(hs) else hs[i]
            prom = hs[i] - max(left, right)
            if prom >= min_prominence * hmax:
                peaks.append((int(cols[i]), float(hs[i]), float(prom)))

    # Suppress peaks closer than a shoulder-width apart: one head, not two.
    peaks.sort(key=lambda p: -p[2])
    kept = []
    for x, hh, pr in peaks:
        if all(abs(x - k2[0]) >= min_sep_frac * span for k2 in kept):
            kept.append((x, hh, pr))

    m = (mask > 0)
    rows = np.where(m.any(axis=1))[0]
    if rows.size == 0:
        return []
    y0, y1 = int(rows.min()), int(rows.max())
    height = max(1, y1 - y0 + 1)

    out = []
    for x, hh, pr in kept:
        # Head width: the run of columns near this dome that are within 25% of
        # its height. Shoulder width: the widest row in the band below it.
        near = np.abs(cols - x) < 0.5 * span
        dome = near & (hs > hh - 0.25 * hmax)
        head_w = float(dome.sum())

        band0 = y0 + int(0.18 * height)
        band1 = y0 + int(0.55 * height)
        sub = m[band0:max(band0 + 1, band1), :]
        shoulder_w = float(sub.sum(axis=1).max()) if sub.size else head_w

        ratio = head_w / max(1.0, shoulder_w)

        # Anthropometric gate: head breadth / biacromial breadth ~= 0.38.
        # Score peaks at that ratio and falls off either side.
        r_ok = np.exp(-((ratio - ANTHRO["head_over_shoulder"]) ** 2) / (2 * 0.16 ** 2))
        p_ok = min(1.0, pr / (0.35 * hmax))          # dome distinctness
        f_ok = 1.0 if shoulder_w > head_w else 0.4   # shoulders must flare out
        score = float(r_ok * p_ok * f_ok)

        out.append(dict(x=int(x), height=float(hh), prominence=float(pr),
                        head_w=head_w, shoulder_w=shoulder_w,
                        ratio=float(ratio), score=score))
    out.sort(key=lambda d: -d["score"])
    return out


def omega_score(mask):
    """Best single omega match for a blob, plus how many were found."""
    om = find_omegas(mask)
    if not om:
        return dict(omega_score=0.0, omega_count=0, omega_ratio=0.0)
    return dict(omega_score=om[0]["score"], omega_count=len(om),
                omega_ratio=om[0]["ratio"])


def suppress_small_omegas(dets, min_ratio=0.40, min_ref_px=6.0, min_ref_score=0.35):
    """
    Frame-level scale consistency for omegas.

    A head has a real physical size. Within one frame, every head sits at a
    comparable range, so head breadths cluster. A laptop's hot patch can trace
    a dome that passes the local anthropometric ratio test, but it does so at
    the wrong SCALE — the dome is a few pixels where a real head is ten or
    more. Local shape descriptors cannot see this, because scale invariance is
    exactly what they were designed to throw away. Comparing omegas against
    each other puts the scale information back.

    Rule: if the frame contains a confident, reasonably large omega, discard
    every omega narrower than `min_ratio` of it.

    The cost is explicit and worth stating: head breadth scales as 1/range, so
    `min_ratio` also sets how much range spread survives. At the 0.40 default a
    person 2.5x further away than the nearest one is still kept; beyond that
    they are discarded with the laptop. In a corridor that is generous; across
    a large hall it is not, and the ratio should be lowered.

    Requires a reference of at least `min_ref_px` and score `min_ref_score` —
    with no big confident omega in frame there is nothing to be small relative
    TO, and the filter correctly does nothing. This is what stops it wiping out
    a lone distant person in an otherwise empty frame.

    Mutates dets in place; returns (n_dropped, reference_width).
    """
    widths = [om["head_w"] for d in dets for om in d.get("omegas", [])
              if om.get("score", 0.0) >= min_ref_score]
    if not widths:
        return 0, 0.0

    ref = max(widths)
    if ref < min_ref_px:
        return 0, float(ref)

    cut = min_ratio * ref
    dropped = 0
    for d in dets:
        oms = d.get("omegas", [])
        if not oms:
            continue
        keep = [om for om in oms if om["head_w"] >= cut]
        dropped += len(oms) - len(keep)
        d["omegas"] = keep
        d["omega_count"] = len(keep)
        d["omega_score"] = float(keep[0]["score"]) if keep else 0.0
    return dropped, float(ref)


def body_extension(data, hot_mask, ambient_c, low_delta=2.0, tmax=None,
                   min_growth=1.6, max_growth=25.0):
    """
    Recover the cool, clothed parts of a body that the detection threshold
    discards, and validate that hot + cool together form a plausible person.

    The problem: a hoodie, jeans or shoes sit only 2-4 C above ambient, well
    below ambient+4, so only the head and hands survive thresholding. The
    detection is then a small blob with few warm centres — which the p-filter
    then removes, discarding a real person.

    The method is the same level-set trick used to identify equipment, run in
    reverse. Re-threshold at ambient+low_delta, take the warm region that
    contains the hot blob, and ask whether it is body-shaped:

        equipment : rectangular, convex, rigid, skeleton does not branch
        body      : elongated, concave, articulated, skeleton BRANCHES
                    and the hot part sits at one END of it (the head)

    If body-shaped, the expanded region is returned so that peaks, shape and
    position are recomputed over the WHOLE person. Peak count rises naturally
    because torso and limbs contribute their own warm centres — which is the
    point: it lets a real person clear the p-filter on merit rather than by
    lowering the filter.

    Returns (is_body, expanded_mask, info).
    """
    info = {}
    if hot_mask.sum() == 0:
        return False, hot_mask, info

    warm = data > (ambient_c + low_delta)
    if tmax is not None:
        warm &= data <= tmax
    warm = warm.astype(np.uint8)
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    n, lab = cv2.connectedComponents(warm)
    if n <= 1:
        return False, hot_mask, info

    b = hot_mask > 0
    ids, counts = np.unique(lab[b & (lab > 0)], return_counts=True)
    if ids.size == 0:
        return False, hot_mask, info
    host = (lab == int(ids[np.argmax(counts)])).astype(np.uint8)

    hot_a = float(b.sum())
    ext_a = float(host.sum())
    growth = ext_a / max(1.0, hot_a)
    info["ext_growth"] = growth
    info["ext_area"] = ext_a
    if not (min_growth <= growth <= max_growth):
        return False, hot_mask, info

    cnts, _ = cv2.findContours(host, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return False, hot_mask, info
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    per = float(cv2.arcLength(c, True))
    if area < 20 or per <= 0:
        return False, hot_mask, info

    hull = cv2.convexHull(c)
    ha = float(cv2.contourArea(hull))
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    rfill = area / (rw * rh) if rw > 1 and rh > 1 else 1.0
    sol = area / ha if ha > 0 else 1.0
    info["ext_rect_fill"] = rfill
    info["ext_solidity"] = sol

    # --- reject the equipment reading of the same region
    if rfill > 0.80 or sol > 0.94:
        return False, hot_mask, info

    # --- articulation: a clothed body still branches into limbs
    topo = skeleton_topology(skeletonize(host))
    info["ext_endpoints"] = topo["sk_endpoints"]
    info["ext_junctions"] = topo["sk_junctions"]
    branched = topo["sk_junctions"] >= 1 or topo["sk_endpoints"] >= 3

    # --- the hot part should sit at one END, as a head does on a body.
    # Measured as the offset between the hot centroid and the full-region
    # centroid, in units of the region's own radius — scale-free, so it holds
    # at any range.
    ys, xs = np.where(host > 0)
    hy, hx = np.where(b)
    d = ((hx.mean() - xs.mean()) ** 2 + (hy.mean() - ys.mean()) ** 2) ** 0.5
    radius = max(1.0, 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min()))
    info["ext_offset"] = float(d / radius)
    peripheral = info["ext_offset"] > 0.18

    is_body = branched and (peripheral or growth > 3.0)
    info["is_body_like"] = float(is_body)
    return bool(is_body), (host * 255).astype(np.uint8), info


def equipment_score(f):
    """
    Combined evidence that a blob is powered equipment rather than a body.

    Zhang & Lu (2004) §3.1.1 is explicit that simple descriptors "are not
    suitable to be standalone shape descriptors" and must be combined — and we
    reproduced that empirically: every single descriptor overlapped between
    people and laptops. So this accumulates weak evidence instead of applying
    one threshold, and each term states what it is testing.

    Returns (score, reasons). Score >= 3.0 is strong evidence.
    """
    why = []

    # ---- PRIMARY: context. Up to 3.5, and REQUIRED — see the gate below.
    #
    # This is the only measure that actually distinguishes the two cases. A
    # head seen from above is genuinely convex, smooth, unarticulated and
    # single-peaked: by every rigidity measure it resembles a machine, and
    # scoring those alone rejected real scalps. What a scalp does NOT have is
    # a large rigid body around it — its warm surround is itself (growth 1.0),
    # whereas a laptop's hot vent sits inside a chassis ten times its size.
    g = f.get("ctx_growth", 1.0)
    crf = f.get("ctx_rect_fill", 0.0)
    csol = f.get("ctx_solidity", 0.0)

    ctx = 0.0
    if g > 2.5 and crf > 0.80:
        ctx = 3.5
        why.append(f"hot spot inside rigid surround (x{g:.1f}, fill {crf:.2f})")
    elif g > 2.0 and crf > 0.88:
        ctx = 3.0
        why.append(f"hot spot inside rectangular surround (x{g:.1f})")
    elif g > 3.5 and csol > 0.95:
        ctx = 2.5
        why.append(f"hot spot inside large convex body (x{g:.1f})")

    if ctx == 0.0:
        # No enclosing rigid body -> not equipment, regardless of how smooth
        # or convex the blob itself happens to be.
        return 0.0, []

    # ---- SUPPORTING: corroboration only. Capped so it can never reject on
    # its own; a blob must first be shown to sit inside something rigid.
    sup = 0.0
    if f.get("rect_fill", 0) > 0.86:
        sup += 0.5; why.append("blob fills its rotated box")
    if f.get("solidity", 0) > 0.95:
        sup += 0.4; why.append("blob has no concavities")
    if f.get("peaks", 9) <= 1 and f.get("area", 0) > 60:
        sup += 0.4; why.append("single warm centre")
    if f.get("grad_p90", 0) > 2.5:
        sup += 0.4; why.append("sharp thermal edges")
    if f.get("n_defects", 9) == 0 and f.get("area", 0) > 60:
        sup += 0.3; why.append("no articulation gaps")
    if f.get("sk_junctions", 9) == 0 and f.get("area", 0) > 150:
        sup += 0.3; why.append("skeleton does not branch")
    sup = min(sup, 1.5)

    return ctx + sup, why


def extract(data, mask, ambient_c, view="vertical", context=True,
            low_delta=1.5, tmax=None):
    """Full invariant vector for one blob. Measures only — never filters."""
    f = {}
    f.update(structural_features(mask))
    f.update(thermal_features(data, mask, ambient_c))
    f.update(projective_features(mask, view))
    if context:
        f.update(thermal_context(data, mask, ambient_c, low_delta, tmax))
    # Omega is a forward/oblique-view feature; directly overhead the analogous
    # structure is the concentric core already measured in projective_features.
    if view in ("horizontal", "any"):
        f.update(omega_score(mask))
    return f


FEATURE_ORDER = [
    # structural
    "area", "compactness", "solidity", "extent", "rect_fill", "aspect",
    "n_defects", "deepest_defect", "mean_defect", "symmetry", "eccentricity",
    "circular_var", "elliptic_var", "convexity", "bending_energy", "curv_max",
    "n_corners",
    "sk_endpoints", "sk_junctions", "sk_length", "sk_len_over_area",
    # thermal
    "t_max", "t_mean", "t_std", "t_excess", "peaks", "grad_mean", "grad_p90",
    "radiant_excess",
    # projective
    "dt_max", "core_frac", "core_over_span", "core_round",
    "head_band_w", "body_band_w", "head_over_body", "profile_step",
    "span_ratio",
    # context (warm region surrounding the hot blob)
    "ctx_area", "ctx_growth", "ctx_solidity", "ctx_rect_fill",
    "ctx_compactness", "ctx_circular_var", "ctx_elliptic_var",
    "is_equipment_like",
    # omega (head-shoulder)
    "omega_score", "omega_count", "omega_ratio",
]
