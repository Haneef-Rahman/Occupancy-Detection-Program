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
# STRUCTURAL — articulation, symmetry, non-convexity
# ---------------------------------------------------------------------------
def structural_features(mask):
    out = {}
    m8 = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
def extract(data, mask, ambient_c, view="vertical"):
    """Full invariant vector for one blob. Measures only — never filters."""
    f = {}
    f.update(structural_features(mask))
    f.update(thermal_features(data, mask, ambient_c))
    f.update(projective_features(mask, view))
    return f


FEATURE_ORDER = [
    # structural
    "area", "compactness", "solidity", "extent", "rect_fill", "aspect",
    "n_defects", "deepest_defect", "mean_defect", "symmetry", "eccentricity",
    "sk_endpoints", "sk_junctions", "sk_length", "sk_len_over_area",
    # thermal
    "t_max", "t_mean", "t_std", "t_excess", "peaks", "grad_mean", "grad_p90",
    "radiant_excess",
    # projective
    "dt_max", "core_frac", "core_over_span", "core_round",
    "head_band_w", "body_band_w", "head_over_body", "profile_step",
    "span_ratio",
]
