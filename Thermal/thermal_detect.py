#!/usr/bin/env python3
"""
FLUXNET — thermal person detection baseline (classical CV, no training).

Works in two modes, chosen automatically:

  RADIOMETRIC (uint16 Y16/TLinear, typically on Linux/Raspberry Pi)
      Every pixel is an absolute temperature. Detection = pixels warmer than
      (ambient + delta). Stable, physically meaningful, the real target.

  AGC (uint8, typically what macOS UVC allows)
      The board hands over an auto-gain 8-bit image: brightness is RELATIVE and
      rescales as the scene changes, so absolute thresholds are meaningless.
      Detection instead uses an adaptive threshold (high percentile + spread),
      which tracks the rescaling. Good enough to develop against; not a
      substitute for radiometric data.

Controls (in the video window):
    q / ESC   quit
    s         save current frame (raw .npy + preview .png)
    l         toggle continuous logging
    [ / ]     make detection less / more sensitive
    a         re-estimate background now
    r         cycle view: color -> mask -> both
    h         print this help

Usage:
    python diagnose.py                       # first, see what the camera gives
    python thermal_detect.py                 # auto-find the camera
    python thermal_detect.py --device 1      # force a device index
    python thermal_detect.py --list          # list candidate devices
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np

try:
    import shape_features as SF
except ImportError:
    SF = None

try:
    from tracker import MultiTracker
except ImportError:
    MultiTracker = None

# ---------------------------------------------------------------------------
# Every processing stage is independently switchable, so each can be A/B'd
# against the same footage. Key -> (flag name, label, default).
# ---------------------------------------------------------------------------
TOGGLES = [
    ("t", "band",     "TempBand",  True),   # absolute human temperature band
    ("w", "split",    "Watershed", True),   # split touching bodies
    ("g", "merge",    "Merge",     True),   # rejoin body fragments
    ("u", "cluster",  "ClustSml",  True),   # cluster small patches
    ("y", "shape",    "ShapeGate", True),   # aspect / extent envelope
    ("k", "peaks",    "MinPeaks",  True),   # >=N warm centres
    ("p", "pfilter",  "P-Filter",  False),  # drop blobs with <= p_min peaks
    ("e", "equip",    "EquipRej",  True),   # reject powered equipment
    ("i", "skin",     "SkinPrio",  True),   # skin band grants immunity
    ("x", "extend",   "BodyExt",   False),  # grow onto clothed torso/legs
    ("z", "omega",    "Omega",     True),   # head-shoulder detection
    ("j", "OmScale",  "OmScale",   True),   # drop omegas too small for frame
    ("S", "osplit",   "OmSplit",   True),   # one box per omega inside a blob
    ("N", "orot",     "OmRot",     True),   # rotation-invariant omega search
    ("F", "farfield", "FarField",  True),   # distant small-blob pass (horiz)
    ("D", "gradfar",  "GradFarF",  True),   # graded range model (horizontal)
    ("f", "kalman",   "Kalman",    False),  # temporal tracking
    ("R", "rigid",    "NonRigid",  False),  # reject rigid peak constellations
    ("G", "ground",   "GndPlane",  False),  # ground-plane scale consistency
    ("A", "gait",     "Gait",      False),  # periodic at a walking cadence
    ("V", "vgrad",    "VGrad",     False),  # head-warm / legs-cool gradient
    ("B", "static",   "StaticSup", True),   # suppress unmoving objects
]
FLAG_KEYS = {k: name for k, name, _, _ in TOGGLES}
FLAG_LABEL = {name: lab for _, name, lab, _ in TOGGLES}
DEFAULT_FLAGS = {name: dflt for _, name, _, dflt in TOGGLES}

# ----------------------------------------------------------------------------
LEPTON_W, LEPTON_H = 160, 120
DEFAULT_DELTA_C = 4.0      # radiometric: °C above ambient
DEFAULT_PCTL = 96.0        # AGC: percentile of brightness treated as "hot"

# Human surface temperature band (absolute, °C).
#
#   Exposed skin (face, hands) ....... 30-35
#   Clothed torso / limbs ............ 27-32
#   Hair, thick clothing ............. 26-29
#   ---- band edges chosen just outside those ----
#   Equipment that must be excluded: laptop ~40, lamp ~50, radiator ~60+
#
# The band is applied IN ADDITION to the relative (ambient + delta) threshold,
# so a pixel must be both warmer than the room AND physiologically plausible.
# The relative test alone fails in cold rooms (clutter passes) and warm rooms
# (bodies barely exceed ambient); the absolute band fixes both ends.
DEFAULT_TMIN_C = 27.0
DEFAULT_TMAX_C = 36.0

# Exposed-skin priority band. 33-35 C is the temperature of a face, neck or
# hand — the most specific human signature in the whole scene. Almost nothing
# in a room sits there: equipment runs hotter, furnishings cooler. So a blob
# containing genuine skin pixels is treated as a person even when the generic
# filters (shape envelope, peak count) would have rejected it. This buys back
# the false negatives those filters cost — partly-occluded people, odd poses,
# distant targets — without loosening anything for non-skin blobs.
# The priority band is DERIVED FROM AMBIENT, not fixed.
#
# A body surface sits between core temperature and the room, at a position set
# by how well that surface is insulated:
#
#     T_surface = ambient + coupling * (T_CORE - ambient)
#
#   coupling ~0.70-0.90  exposed skin (face, hands) — well coupled to core
#   coupling ~0.45-0.62  thin clothing
#   coupling ~0.28-0.50  HAIR / scalp — an insulator, much nearer ambient
#
# This matters most for the overhead view, where the camera sees the TOP OF
# THE HEAD — hair, not skin. A fixed 33-35 C band implicitly assumes a face is
# visible and misses hair entirely, especially in a cool room.
#
# Sanity check at ambient 24 C: skin -> 33.1-35.7 C, which reproduces the
# 33-35 placeholder. The model agrees with the fixed band where the fixed band
# was valid, and adapts where it was not.
T_CORE_C = 37.0
COUPLING = {
    "skin":     (0.70, 0.90),
    "clothing": (0.45, 0.62),
    "hair":     (0.28, 0.50),
}
SKIN_EPS_C = 0.05
SKIN_MIN_FRAC = 0.04      # 4% of blob pixels in-band = a real patch
SKIN_MIN_PX = 3           # ...and at least this many, so noise cannot qualify

# Set per frame from the measured ambient; overridden by --skin-lo/--skin-hi.
SKIN_BAND_C = (33.0, 35.0)
SKIN_BAND_FIXED = False


def surface_band(ambient_c, kind="skin", core_c=T_CORE_C):
    """Temperature range a given body surface should occupy in this room."""
    lo_c, hi_c = COUPLING[kind]
    return (ambient_c + lo_c * (core_c - ambient_c),
            ambient_c + hi_c * (core_c - ambient_c))

MIN_BLOB_AREA = 6          # lowered: a person at range is only a few px
MAX_BLOB_AREA = 14000

# Bodies are never box-shaped. A filled rectangle has extent 1.0, a perfect
# ellipse 0.785; real people sit at 0.4-0.75. Anything above this is a
# manufactured object (laptop, monitor, panel heater) regardless of its
# temperature or peak count.
MAX_EXTENT = 0.82

# Same idea but rotation-invariant, measured against the minimum-area rect.
# This is the one that actually catches a laptop lying at an angle.
MAX_RECT_FILL = 0.86
DISPLAY_SCALE = 6      # overridden by --scale

# Fixed span for --capture-png-raw. A per-frame stretch would make the same
# scene encode differently from one frame to the next, which is exactly the
# nuisance variation a learned model must not be taught.
PNG_SPAN_C = (15.0, 45.0)
LOG_DIR = "logs"

# Two mounting geometries, switchable live with the 'v' key (and later
# selectable automatically from the IMU's gravity vector).
#
#   VERTICAL   camera axis points down (ceiling / overhead mount).
#              A person is head+shoulders from above: roughly round.
#              Two people may be separated along ANY image direction,
#              because they stand on a floor plane seen in plan view —
#              so a split blob is never re-merged.
#
#   HORIZONTAL camera axis points forward (wall / lintel mount).
#              A person is a standing body: tall ellipse.
#              Two people separate side-by-side only; a vertical split is
#              head/torso/legs of ONE body and must be merged back.
#
# 'any' is a permissive fallback used before the mount is known.
VIEW_MODES = ["vertical", "horizontal", "any"]

# min_sep_px: two people cannot stand closer than about shoulder width, so a
# split producing centres nearer than this is one body that fragmented, not
# two people. Geometry-dependent — at a 2.4 m ceiling this is roughly 20 px;
# tune with --min-sep once the real mount height is known.
SHAPE_RULES = {
    # aspect = height/width of the blob;  extent = blob area / bbox area
    # cluster_small / small_area: overhead, one body appears as several small
    # patches (head, shoulders, arms) sitting close together, while clutter sits
    # alone. Clustered small blobs become one person; isolated ones are dropped.
    # vertical: delta 4.0 and a mandatory 2-peak test at ALL blob sizes —
    # overhead, even a distant person shows head + shoulder warm centres, so
    # a single-peak blob is a hot object rather than a body.
    "vertical":   dict(aspect_min=0.45, aspect_max=2.2, extent_min=0.35,
                       merge_axis="any",       # any direction, but only if close
                       min_sep_px=26,
                       # reach far enough to gather torso/clothing patches onto
                       # the body they belong to before anything is discarded
                       cluster_small=True, small_area=55, cluster_radius=22,
                       # the 2-peak test applies only to blobs large enough to
                       # HAVE two centres; a cool clothing patch legitimately
                       # has one, and demanding two deleted real people
                       # an overhead head is a smooth dome — it cannot show as many
                       # warm centres as a whole body, so the bar is lower here.
                       # The laptop is rejected by MAX_EXTENT, not by peaks.
                       delta=4.0, min_peaks=2, peak_check_area=200,
                       surface="hair",       # overhead you see the scalp
                       reject_equipment=True,
                       label="VERT (down)"),
    # Retuned for RANGE. The old gate (aspect_min 1.1, extent_min 0.25, 3 peaks
    # above 150 px) assumed you always see a whole standing body: taller than
    # wide, multi-peaked. Past roughly 2.5 m you do not — the clothed torso
    # falls below threshold and what survives is a head plus hands, which is
    # roughly SQUARE. aspect_min 1.1 rejected that outright, so detection
    # stopped abruptly instead of degrading.
    #
    # Relaxing shape is safer here than it looks: distant blobs are small, and
    # small blobs are already held in by the absolute temperature band, which
    # is what rejects clutter. cluster_small is now on so a face and two hands
    # rejoin into one person rather than being three fragments that each fail.
    "horizontal": dict(aspect_min=0.65, aspect_max=6.0, extent_min=0.18,
                       merge_axis="vertical",  # stacked = one body
                       min_sep_px=0,
                       cluster_small=True, small_area=48, cluster_radius=18,
                       delta=3.0, min_peaks=2, peak_check_area=420,
                       surface="skin",       # forward you see faces/hands
                       reject_equipment=True,
                       label="HORIZ (fwd)"),
    "any":        dict(aspect_min=0.30, aspect_max=7.0, extent_min=0.20,
                       merge_axis="vertical",
                       min_sep_px=0,
                       cluster_small=False, small_area=0, cluster_radius=0,
                       delta=4.0, min_peaks=3, peak_check_area=150,
                       surface="skin", reject_equipment=True,
                       label="ANY"),
}


# ----------------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------------
class ThermalCamera:
    """PureThermal/Lepton UVC wrapper. Prefers radiometric, falls back to AGC."""

    def __init__(self, device_index):
        self.index = device_index
        self.radiometric = False
        self.scale = 0.01

        # Try raw 16-bit first.
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video device {device_index}")
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
        except Exception:
            pass

        ok, probe = cap.read()
        if ok and probe is not None and probe.dtype == np.uint16:
            self.cap = cap
            self.radiometric = True
            med = float(np.median(probe))
            if not (-40 < med * 0.01 - 273.15 < 80):
                self.scale = 0.1
            print(f"  mode: RADIOMETRIC (uint16, {self.scale} K/count)")
            return

        # Raw failed or gave 8-bit: reopen cleanly in normal (AGC) mode.
        cap.release()
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not reopen device {device_index}")
        ok, probe = cap.read()
        if not ok or probe is None:
            raise RuntimeError("Device opened but returned no frames")
        self.cap = cap
        print("  mode: AGC 8-bit (no absolute temperatures)")
        print("        macOS commonly blocks raw Y16; use a Linux host/Pi for")
        print("        radiometric work. Adaptive thresholding is used instead.")

    def read(self):
        """Returns (data, is_temp). data is float32 °C if radiometric, else 0-255."""
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None, False
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.radiometric and frame.dtype == np.uint16:
            return frame.astype(np.float32) * self.scale - 273.15, True
        return frame.astype(np.float32), False

    def release(self):
        self.cap.release()


def scan_devices(max_index=6):
    out = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        ok, f = cap.read()
        if ok and f is not None:
            h, w = f.shape[:2]
            looks_lepton = (w == LEPTON_W and h in (LEPTON_H, LEPTON_H * 2)) or f.dtype == np.uint16
            out.append((i, w, h, str(f.dtype), looks_lepton))
        cap.release()
    return out


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def compute_threshold(data, is_temp, sensitivity):
    """
    Returns (threshold_value, background_level).

    radiometric : threshold = ambient + sensitivity  (sensitivity in °C)
    AGC         : threshold = percentile(sensitivity) of the frame, floored at
                  median + 2*MAD so an empty room doesn't self-detect noise.
    """
    if is_temp:
        bg = float(np.median(data))
        return bg + sensitivity, bg

    bg = float(np.median(data))
    mad = float(np.median(np.abs(data - bg))) or 1.0
    pctl_thr = float(np.percentile(data, sensitivity))
    floor_thr = bg + 4.0 * mad
    return max(pctl_thr, floor_thr), bg


def merge_nearby(dets, gap_px, merge_axis="vertical", min_sep_px=0):
    """
    Fragments of one body (arm, shoulder, head separated by cooler clothing)
    arrive as separate components. Merge detections whose bounding boxes are
    within gap_px of each other — verified necessary on real frames, where a
    single rotating arm split into 5 boxes inside one 39x82 px region.
    """
    if gap_px <= 0 or len(dets) < 2:
        return dets

    def near(a, b):
        # Regions sharing a parent component were cut apart by watershed.
        # Whether to re-merge them is decided by the axis of separation:
        # two people stand SIDE BY SIDE, while the parts of a single body
        # (head / torso / legs) stack VERTICALLY. So a mainly-horizontal
        # split is two bodies (keep separate); a mainly-vertical one is
        # fragments of one body (merge back).
        pa, pb = a.get("parent_cc"), b.get("parent_cc")
        if pa is not None and pa == pb:
            dxc = abs(a["centroid"][0] - b["centroid"][0])
            dyc = abs(a["centroid"][1] - b["centroid"][1])
            if merge_axis == "any":
                # VERTICAL (overhead): people separate in any image direction,
                # so the axis carries no information. Use the physical limit
                # instead — two bodies cannot be closer than shoulder width.
                if (dxc ** 2 + dyc ** 2) ** 0.5 >= min_sep_px:
                    return False        # far enough apart: genuinely two people
            elif dxc > dyc:
                # HORIZONTAL: side-by-side means two people; stacked means
                # head/torso/legs of one, which should merge back.
                return False
        ax, ay, aw, ah = a["bbox"]
        bx, by, bw, bh = b["bbox"]
        dx = max(0, max(ax, bx) - min(ax + aw, bx + bw))
        dy = max(0, max(ay, by) - min(ay + ah, by + bh))
        return dx <= gap_px and dy <= gap_px

    groups = []
    for d in dets:
        placed = False
        for g in groups:
            if any(near(d, m) for m in g):
                g.append(d)
                placed = True
                break
        if not placed:
            groups.append([d])

    # one more pass: groups can become adjacent after absorbing members
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if any(near(a, b) for a in groups[i] for b in groups[j]):
                    groups[i] += groups[j]
                    del groups[j]
                    changed = True
                    break
            if changed:
                break

    merged = []
    for g in groups:
        if len(g) == 1:
            merged.append(g[0])
            continue
        x0 = min(d["bbox"][0] for d in g)
        y0 = min(d["bbox"][1] for d in g)
        x1 = max(d["bbox"][0] + d["bbox"][2] for d in g)
        y1 = max(d["bbox"][1] + d["bbox"][3] for d in g)
        area = sum(d["area_px"] for d in g)
        merged.append({
            "bbox": (x0, y0, x1 - x0, y1 - y0),
            "centroid": (sum(d["centroid"][0] * d["area_px"] for d in g) / area,
                         sum(d["centroid"][1] * d["area_px"] for d in g) / area),
            "area_px": area,
            "val_max": max(d["val_max"] for d in g),
            "val_mean": sum(d["val_mean"] * d["area_px"] for d in g) / area,
            "fragments": len(g),
            "priority": any(d.get("priority") for d in g),
            "skin_px": sum(d.get("skin_px", 0) for d in g),
            "skin_frac": sum(d.get("skin_px", 0) for d in g) / area,
        })
    merged.sort(key=lambda d: d["area_px"], reverse=True)
    return merged


def split_touching(mask, min_sep=3, neck_ratio=0.72):
    """
    Split blobs formed by two adjacent bodies.

    Two people standing close merge into one connected component, and no
    downstream filter can undo that — the blob must be cut. A distance
    transform gives each body a peak at its centre with a valley between;
    watershed cuts along the valley. This is the shape-domain version of the
    head-peaks / shoulder-valley separation the depth sensor performs.

    Returns a label image (0 = background, 1..n = separated regions).
    """
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros_like(binary, np.int32), 0

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Local maxima of the distance map = body centres. Comparing against a
    # dilation finds them without a global threshold, so blobs of different
    # sizes are handled correctly.
    k = 2 * min_sep + 1
    dil = cv2.dilate(dist, np.ones((k, k), np.uint8))
    peaks = ((dist >= dil - 1e-6) & (dist >= min_sep)).astype(np.uint8)

    n_seeds, seeds = cv2.connectedComponents(peaks)
    if n_seeds <= 2:                      # 0 or 1 real seed: nothing to split
        n, labels = cv2.connectedComponents(binary)
        return labels, n - 1

    # Saddle test — decide whether a pair of peaks is really two bodies.
    #
    # Two lobes of ONE body (torso + shoulder, head + neck) are joined by a
    # THICK neck: the distance transform stays high all along the path between
    # their centres. Two people merely touching are joined by a THIN neck.
    # So peaks whose connecting path never thins below `neck_ratio` of the
    # smaller peak are fused into a single seed, and no split occurs there.
    cents = []
    for lab in range(1, n_seeds):
        ys_, xs_ = np.where(seeds == lab)
        if xs_.size:
            cents.append((lab, float(xs_.mean()), float(ys_.mean()),
                          float(dist[ys_, xs_].max())))

    parent = {lab: lab for lab, _, _, _ in cents}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            la, xa, ya, pa = cents[i]
            lb, xb, yb, pb = cents[j]
            steps = max(2, int(((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5))
            saddle = min(
                float(dist[int(round(ya + (yb - ya) * t / steps)),
                           int(round(xa + (xb - xa) * t / steps))])
                for t in range(steps + 1)
            )
            if saddle >= neck_ratio * min(pa, pb):
                ra, rb = find(la), find(lb)
                if ra != rb:
                    parent[rb] = ra

    if len(set(find(l) for l, _, _, _ in cents)) <= 1:
        # every peak belongs to one body: do not split at all
        n, labels = cv2.connectedComponents(binary)
        return labels, n - 1

    remap = {}
    fused = np.zeros_like(seeds)
    for lab, _, _, _ in cents:
        root = find(lab)
        if root not in remap:
            remap[root] = len(remap) + 1
        fused[seeds == lab] = remap[root]
    seeds = fused
    peaks = (fused > 0).astype(np.uint8)

    # Marker convention for cv2.watershed:
    #   0        = unknown, to be filled
    #   1        = background
    #   2..n+1   = seeds
    # Leaving non-seed foreground at 0 is essential — that is the region
    # watershed grows the seeds into.
    markers = np.zeros(binary.shape, np.int32)
    markers[binary == 0] = 1
    markers[peaks > 0] = seeds[peaks > 0] + 1

    img3 = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img3, markers)

    out = np.zeros_like(markers)
    idx = 0
    for lab in np.unique(markers):
        if lab <= 1:                      # background (1) and ridges (-1)
            continue
        region = (markers == lab) & (binary > 0)
        if region.sum() == 0:
            continue
        idx += 1
        out[region] = idx
    return out, idx


def peak_points(data, ys, xs, prominence=2.5, min_sep=2):
    """
    Locations of the warm centres, not just how many.

    thermal_texture answers "how many peaks"; this answers "where". The count
    is what equipment can imitate — give a laptop more hot spots and it scores
    like a person. The GEOMETRY of the constellation, tracked over time, is
    what it cannot imitate, because a rigid object's hot spots cannot move
    relative to one another.
    """
    if xs.size < 4:
        return np.empty((0, 2), np.float64)
    patch = np.full(data.shape, -999.0, np.float32)
    patch[ys, xs] = data[ys, xs]
    k = 2 * min_sep + 1
    mx = cv2.dilate(patch, np.ones((k, k), np.uint8))
    reg = np.zeros(data.shape, bool)
    reg[ys, xs] = True
    pk = (patch >= mx - 1e-6) & (patch > data[ys, xs].max() - prominence) & reg
    n, lab = cv2.connectedComponents(pk.astype(np.uint8))
    pts = []
    for i in range(1, n):
        yy, xx = np.where(lab == i)
        pts.append((float(xx.mean()), float(yy.mean())))
    return np.asarray(pts, np.float64) if pts else np.empty((0, 2), np.float64)


def thermal_texture(data, ys, xs, prominence=2.5, min_sep=2):
    """
    Measure the internal thermal structure of a blob.

    Theory under test: a human is a CLUSTER of several warm regions — face,
    neck, hands, gaps in clothing — each at a slightly different temperature.
    A single manufactured hot object (lamp, laptop, radiator, reflection) is
    one homogeneous patch. So counting distinct local maxima inside a blob
    should separate bodies from single hot objects.

    Parameters were swept against real captured humans and synthetic objects:

        prominence  min_sep |  human peaks (min/mean) | object peaks (max)
        ------------------- | ---------------------- | ------------------
              0.6        2  |        2 / 7.2         |        1
              1.5        2  |        5 / 11.9        |        1
              2.5        2  |        5 / 12.5        |        1     <- chosen
              2.5        1  |        8 / 31.6        |        3

    min_sep=1 yields far more peaks but lets objects reach 3, eroding the
    margin; min_sep=2 pins objects at exactly 1, so the human/object ratio is
    the widest and the threshold is unambiguous.

    Returns (n_peaks, std_c, range_c).
    """
    if xs.size < 4:
        return 1, 0.0, 0.0

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    patch = np.full((y1 - y0, x1 - x0), -999.0, np.float32)
    patch[ys - y0, xs - x0] = data[ys, xs]

    vals = data[ys, xs]
    std_c = float(vals.std())
    range_c = float(vals.max() - vals.min())

    # Local maxima: a pixel equal to the max of its neighbourhood, and within
    # `prominence` of the blob's own peak (so we count real warm centres, not
    # every noise wobble).
    k = 2 * min_sep + 1
    dil = cv2.dilate(patch, np.ones((k, k), np.uint8))
    peak_mask = (patch >= dil - 1e-6) & (patch >= vals.max() - prominence)
    n_peaks, _ = cv2.connectedComponents(peak_mask.astype(np.uint8))
    return max(1, n_peaks - 1), std_c, range_c


def cluster_small(dets, small_area, radius, min_sep_px=0,
                  body_min_area=14, body_min_c=29.0):
    """
    Handle small blobs by company, not by size alone.

    Seen from above, one person breaks into several small warm patches — head,
    shoulders, arms — that sit close together. Genuine clutter (a reflection, a
    warm fitting, sensor speckle) sits alone. So:

      * small blobs near other blobs  -> CLUSTERED into a single detection
      * small blobs that stay alone   -> DROPPED as isolated heat sources

    Two people are protected by min_sep_px: clusters are never formed across a
    gap wider than shoulder width.
    """
    if not dets:
        return dets

    n = len(dets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    def sep(a, b):
        return ((a["centroid"][0] - b["centroid"][0]) ** 2
                + (a["centroid"][1] - b["centroid"][1]) ** 2) ** 0.5

    for i in range(n):
        for j in range(i + 1, n):
            a, b = dets[i], dets[j]
            # Only small blobs are clustered this way; two large bodies that
            # happen to be near each other stay distinct.
            if a["area_px"] >= small_area and b["area_px"] >= small_area:
                continue
            d = sep(a, b)
            if d > radius:
                continue
            if min_sep_px and d >= min_sep_px:
                continue          # far enough apart to be two people
            union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(dets[i])

    out = []
    for g in groups.values():
        if len(g) == 1:
            d = g[0]
            # Alone and small -> isolated heat source, not a person.
            #
            # Two exemptions, both added because this rule was deleting real
            # people whose torso/clothing reads cool and patchy:
            #   * skin      — a distant face is small, alone, and real
            #   * body-warm — a patch sitting solidly in the clothed-body range
            #                 with a credible size is a person part, not noise.
            #                 Speckle is tiny AND barely above the floor; this
            #                 is neither.
            if d["area_px"] < small_area and not d.get("priority"):
                body_like = (d["area_px"] >= body_min_area
                             and d.get("val_mean", 0) >= body_min_c)
                if not body_like:
                    continue
            out.append(d)
            continue

        x0 = min(d["bbox"][0] for d in g)
        y0 = min(d["bbox"][1] for d in g)
        x1 = max(d["bbox"][0] + d["bbox"][2] for d in g)
        y1 = max(d["bbox"][1] + d["bbox"][3] for d in g)
        area = sum(d["area_px"] for d in g)
        out.append({
            "bbox": (x0, y0, x1 - x0, y1 - y0),
            "centroid": (sum(d["centroid"][0] * d["area_px"] for d in g) / area,
                         sum(d["centroid"][1] * d["area_px"] for d in g) / area),
            "area_px": area,
            "val_max": max(d["val_max"] for d in g),
            "val_mean": sum(d["val_mean"] * d["area_px"] for d in g) / area,
            "aspect": (y1 - y0) / max(1, x1 - x0),
            "extent": area / max(1, (x1 - x0) * (y1 - y0)),
            "parent_cc": g[0].get("parent_cc"),
            "fragments": sum(d.get("fragments", 1) for d in g),
            "priority": any(d.get("priority") for d in g),
            "skin_px": sum(d.get("skin_px", 0) for d in g),
            "skin_frac": sum(d.get("skin_px", 0) for d in g) / area,
        })
    return out


SIGMA_SB = 5.670374419e-8      # Stefan-Boltzmann constant, W m^-2 K^-4


def density_map(data, region_mask, ambient_c, kind="binary", emissivity=0.98):
    """
    Build Hu's density distribution rho(x, y) for one blob.

    Hu (1962) defines the moments over a "density distribution function"
    rho(x,y) but never fixes what it is — assigning mass to pixels is a
    modelling choice. Three options, in increasing physical content:

      'binary'   rho = 1 inside the region, 0 outside.
                 Pure geometry; comparable with the shape-recognition
                 literature, discards all temperature information.

      'excess'   rho = max(0, T - T_ambient).
                 Moments now describe WHERE THE HEAT IS, not just where the
                 outline is: a head-heavy body differs from a vent-heavy
                 laptop even at similar outlines.

      'radiant'  rho = eps * sigma * (T^4 - T_ambient^4)   [W/m^2]
                 The physically exact version. Stefan-Boltzmann makes this
                 the excess radiated power per unit area, and skin emissivity
                 ~0.98 (Steketee 1973) makes the approximation tight. m00 is
                 then total excess radiated power and the centroid is the
                 true centre of radiated energy.

    Ambient MUST be subtracted. Raw temperature gives every background pixel
    a mass of ~24, so the moments would describe the image frame rather than
    the person — and Hu's uniqueness theorem requires rho to be non-zero only
    over a finite region, which only the ambient-referenced forms satisfy.
    """
    m = region_mask > 0
    rho = np.zeros(data.shape, np.float64)

    if kind == "binary":
        rho[m] = 1.0
    elif kind == "excess":
        rho[m] = np.maximum(0.0, data[m] - ambient_c)
    elif kind == "radiant":
        t_k = data[m] + 273.15
        a_k = ambient_c + 273.15
        rho[m] = np.maximum(0.0, emissivity * SIGMA_SB * (t_k ** 4 - a_k ** 4))
    else:
        raise ValueError(f"unknown density kind: {kind}")
    return rho


def shape_moments(data, region_mask, ambient_c, kind="binary"):
    """
    Hu's seven invariants plus the classical region descriptors, computed on
    the chosen density map.

    Returns a dict. Hu values are log-compressed (sign-preserving) because the
    raw invariants span many orders of magnitude and are unusable as features
    otherwise.
    """
    rho = density_map(data, region_mask, ambient_c, kind)
    M = cv2.moments(rho.astype(np.float32), binaryImage=False)
    if M["m00"] <= 0:
        return None

    hu = cv2.HuMoments(M).flatten()
    hu_log = [float(-np.sign(h) * np.log10(abs(h))) if h != 0 else 0.0 for h in hu]

    out = {f"hu{i+1}": v for i, v in enumerate(hu_log)}
    out["mass"] = float(M["m00"])          # area / total excess / total watts
    out["cx"] = float(M["m10"] / M["m00"])
    out["cy"] = float(M["m01"] / M["m00"])

    # Classical region descriptors, on the binary support of the same region.
    m8 = (region_mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        per = float(cv2.arcLength(c, True))
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        out["compactness"] = (4.0 * np.pi * area / (per * per)) if per > 0 else 0.0
        out["solidity"] = (area / hull_area) if hull_area > 0 else 0.0

        # Convexity defects = the gaps a body has and a box does not:
        # armpits, the space between legs, the neck notch.
        n_def, deepest = 0, 0.0
        if len(c) > 3:
            hull_i = cv2.convexHull(c, returnPoints=False)
            if hull_i is not None and len(hull_i) > 3:
                try:
                    d = cv2.convexityDefects(c, np.sort(hull_i[:, 0])[::-1][:, None])
                    if d is not None:
                        depths = d[:, 0, 3] / 256.0
                        n_def = int((depths > 1.0).sum())
                        deepest = float(depths.max())
                except cv2.error:
                    pass
        out["n_defects"] = n_def
        out["deepest_defect"] = deepest

        # Second-order shape: eccentricity from the covariance of the region.
        mu20, mu02, mu11 = M["mu20"], M["mu02"], M["mu11"]
        tr = mu20 + mu02
        det = mu20 * mu02 - mu11 * mu11
        disc = max(0.0, tr * tr / 4.0 - det)
        l1, l2 = tr / 2.0 + np.sqrt(disc), tr / 2.0 - np.sqrt(disc)
        out["eccentricity"] = float(np.sqrt(1 - l2 / l1)) if l1 > 0 else 0.0
    return out


def rect_fill(region_mask):
    """
    How completely a blob fills its MINIMUM-AREA (rotated) rectangle.

    The axis-aligned bounding box is useless for a laptop lying at an angle —
    a tilted rectangle only half-fills its upright box, so it slips through an
    extent test. minAreaRect rotates with the object:

        filled rectangle (any angle) -> ~1.00
        perfect ellipse              -> ~0.785
        person with limbs/gaps       -> 0.45-0.72

    Returns fill in 0..1 (1.0 if the contour is too small to measure).
    """
    m = (region_mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 1.0
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5 or cv2.contourArea(c) < 12:
        return 1.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    if rw < 1 or rh < 1:
        return 1.0
    return float(cv2.contourArea(c) / (rw * rh))


class StaticSuppressor:
    """
    Reject things that never move.

    A laptop, monitor or radiator holds the same temperature in the same place
    indefinitely; a person does not. A slowly-adapting per-pixel background is
    learned, and pixels matching it are treated as furniture.

    The time constant is deliberately long (~40 s at 9 fps) so that someone
    standing still for a few seconds is not absorbed. Anyone genuinely
    stationary for a minute WILL fade — that is the accepted limit of a
    thermal-only detector, and precisely the gap radar breathing-detection
    fills in the fused system.
    """

    def __init__(self, alpha=0.0028, tol_c=0.8):
        self.bg = None
        self.alpha = alpha
        self.tol_c = tol_c
        self.enabled = True
        self.frames = 0

    def update(self, data):
        if self.bg is None:
            self.bg = data.astype(np.float32).copy()
        else:
            self.bg += self.alpha * (data - self.bg)
        self.frames += 1

    def moving_mask(self, data):
        """True where the scene differs from its long-term self."""
        if self.bg is None or not self.enabled or self.frames < 30:
            return np.ones(data.shape, bool)      # warm-up: trust everything
        return np.abs(data - self.bg) > self.tol_c

    def reset(self):
        self.bg = None
        self.frames = 0


def skin_score(vals):
    """
    Fraction of a blob's pixels in the exposed-skin band, and whether that is
    enough to grant priority. Returns (fraction, n_pixels, is_priority).
    """
    if vals.size == 0:
        return 0.0, 0, False
    lo = SKIN_BAND_C[0] - SKIN_EPS_C      # inclusive of 33.0
    hi = SKIN_BAND_C[1] + SKIN_EPS_C      # inclusive of 35.0
    n = int(((vals >= lo) & (vals <= hi)).sum())
    frac = n / float(vals.size)
    return frac, n, (n >= SKIN_MIN_PX and frac >= SKIN_MIN_FRAC)


def shape_ok(w, h, area, rules, priority=False):
    """
    View-dependent shape gate. Returns (passed, aspect, extent).

    A blob carrying exposed-skin temperatures gets a widened envelope: a real
    person in an awkward pose or partly occluded still reads as skin, and
    rejecting them on proportions alone would be a false negative we can see
    is wrong.
    """
    aspect = h / max(1, w)                       # >1 = taller than wide
    extent = area / max(1, w * h)                # blob fill of its bbox
    amin, amax = rules["aspect_min"], rules["aspect_max"]
    emin = rules["extent_min"]
    if priority:
        amin, amax, emin = amin * 0.55, amax * 1.8, emin * 0.6

    # Rectangularity ceiling — applies even to priority blobs.
    #
    # Manufactured objects are rectangles and fill their bounding box almost
    # completely (extent -> 1.0). A body never does: a perfect ellipse is
    # 0.785, and real people with limbs, gaps and soft edges land well below
    # that. A laptop reading 29-31 C passed the temperature band, the shape
    # envelope and the peak test; extent is what actually separates it.
    if extent > MAX_EXTENT:
        return False, aspect, extent

    ok = (amin <= aspect <= amax and extent >= emin)
    return ok, aspect, extent


def detect_people(data, threshold, merge_gap=6, tmax=None, view="any",
                  min_area=MIN_BLOB_AREA, split=True, min_sep=None,
                  cohesion=1, tmin=None, min_peaks=None, peak_check_area=None,
                  body_min_area=14, body_min_c=29.0, moving=None,
                  equip_thresh=3.0, p_filter=False, p_min=5,
                  body_extend=False, body_delta=2.0, flags=None,
                  omega_scale_ratio=0.40, omega_delta=2.0,
                  gnd_horizon=0.0, gnd_gain=0.0, gnd_tol=0.5,
                  omega_max_tilt=40.0,
                  far_delta=2.0, far_min_area=5, far_max_area=90,
                  far_contrast=2.5, far_row_max=0.55,
                  gfar_ref_row=25.0, gfar_ref_h=7.0, gfar_ref_range=8.5,
                  gfar_horizon=0.0, gfar_lo=0.55, gfar_hi=1.9,
                  gfar_delta=2.5, gfar_max_range=12.0,
                  gfar_contrast_near=3.5, gfar_contrast_far=2.5,
                  gfar_structure_h=12.0,
                  vgrad_max=-0.15):
    """
    Band-threshold -> clean -> split touching bodies -> shape filter.

    tmax     : upper temperature bound (°C). Rejects equipment, which is what
               makes a low min_area safe. None disables (AGC mode).
    view     : 'vertical' | 'horizontal' | 'any' — shape + merge rules.
    cohesion : anti-fragmentation strength, 0..4. Raising it does three
               things at once, which are the three ways a body fragments:
                 1. larger CLOSE kernel  — bridges gaps before labelling
                 2. larger merge gap     — rejoins separated pieces
                 3. coarser watershed seeds — fewer spurious splits
               Adjust live with 'm' / 'n'.
    """
    cohesion = int(max(0, min(4, cohesion)))
    F = dict(DEFAULT_FLAGS)
    if flags:
        F.update(flags)
    if not F["band"]:
        tmin = tmax = None
    if not F["extend"]:
        body_extend = False
    if not F["pfilter"]:
        p_filter = False
    # Relative test (warmer than this room) AND absolute band (plausibly human).
    #
    # Guard: in a warm room, ambient+delta can rise above tmax and the band
    # collapses to nothing — zero detections forever, with no visible cause.
    # Cap the relative threshold so a usable window always remains. Contrast
    # genuinely degrades in warm rooms, but the failure should be graceful.
    if tmax is not None and threshold > tmax - 2.0:
        threshold = tmax - 2.0

    keep = data > threshold
    if tmin is not None:
        keep &= data >= tmin
    if tmax is not None:
        keep &= data <= tmax
    # Drop anything that has not changed in a long time: furniture and
    # electronics hold still, people do not.
    if moving is not None:
        keep &= moving
    mask = keep.astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)

    # Lever 1: the CLOSE kernel bridges cool gaps (clothing, hair) that break a
    # body into pieces. This runs BEFORE labelling, so it is the most effective
    # of the three — pieces never become separate components at all.
    ck = 5 + 2 * cohesion                      # 5, 7, 9, 11, 13
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)

    # Parent connected components, recorded before any splitting, so that
    # merge_nearby can tell "fragments of one body" from "two bodies we cut".
    _, cc_labels = cv2.connectedComponents((mask > 0).astype(np.uint8))

    if split and F["split"]:
        # Lever 3: coarser seeds mean only substantial cores start a region,
        # so thin fragments stop generating their own split.
        labels, n = split_touching(mask, min_sep=3 + cohesion)
    else:
        n_cc, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
        n = n_cc - 1

    rules = SHAPE_RULES.get(view, SHAPE_RULES["any"])

    # Priority band for THIS room, derived from the measured ambient.
    #
    # ALWAYS the skin coupling, never hair — even overhead. Hair sits only a
    # few degrees above ambient (27.6-30.5 C in a 24 C room), which is exactly
    # where warm electronics live: a MacBook at 29-31 C fell inside the hair
    # band and was granted priority, bypassing every other filter. Skin is far
    # above ambient and genuinely rare in a room, so it is the only surface
    # specific enough to justify overriding the filters. A scalp is still
    # DETECTED normally; it just does not get a free pass.
    global SKIN_BAND_C
    if not SKIN_BAND_FIXED and tmin is not None:
        ambient_now = float(np.median(data))
        SKIN_BAND_C = surface_band(ambient_now, "skin")
    dets = []
    for i in range(1, n + 1):
        ys, xs = np.where(labels == i)
        if xs.size == 0:
            continue
        area = int(xs.size)
        if area < min_area or area > MAX_BLOB_AREA:
            continue
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)

        blob = data[ys, xs]
        sfrac, spx, priority = skin_score(blob)
        if not F["skin"]:
            priority = False

        # Rotation-invariant boxiness: a laptop at ANY angle fills its
        # minimum-area rectangle. Applies to priority blobs too.
        sub = np.zeros((h, w), np.uint8)
        sub[ys - y, xs - x] = 255
        rfill = rect_fill(sub)
        if rfill > MAX_RECT_FILL:
            continue

        ok, aspect, extent = shape_ok(w, h, area, rules, priority)
        if not F["shape"]:
            ok = True
        if not ok:
            continue

        dets.append({
            "skin_frac": float(sfrac),
            "skin_px": int(spx),
            "priority": bool(priority),
            "rect_fill": float(rfill),
            "bbox": (x, y, w, h),
            "centroid": (float(xs.mean()), float(ys.mean())),
            "area_px": area,
            "val_max": float(blob.max()),
            "val_mean": float(blob.mean()),
            "aspect": float(aspect),
            "extent": float(extent),
            "parent_cc": int(cc_labels[ys[0], xs[0]]),
        })

    # Lever 2: a wider gap rejoins pieces that survived as separate components.
    eff_min_sep = min_sep if min_sep is not None else rules.get("min_sep_px", 0)
    if not F["merge"]:
        merge_gap = -1                      # gap < 0 disables merging
    dets = merge_nearby(dets, merge_gap + 3 * cohesion,
                        rules.get("merge_axis", "vertical"), eff_min_sep)

    # Cluster surviving small patches into bodies, and discard the ones that
    # remain alone. Radius grows with cohesion, like the other levers.
    if rules.get("cluster_small") and F["cluster"]:
        dets = cluster_small(dets,
                             small_area=rules.get("small_area", 40),
                             radius=rules.get("cluster_radius", 14) + 2 * cohesion,
                             min_sep_px=eff_min_sep,
                             body_min_area=body_min_area, body_min_c=body_min_c)

    # ---- equipment rejection (vertical mode) --------------------------------
    # A laptop is not rejected by the shape of its HOT PATCH: only the vent and
    # CPU corner clear the threshold, and that patch is irregular. The rectangle
    # lives one temperature level down. So each surviving blob is re-examined
    # against the warm region that contains it, combined with rigidity and
    # thermal-structure evidence.
    if SF is not None and rules.get("reject_equipment") and F["equip"] and tmin is not None:
        kept = []
        for d in dets:
            x, y, w, h = d["bbox"]
            sub = np.zeros(mask.shape, np.uint8)
            sub[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
            try:
                feats = SF.extract(data, sub, float(np.median(data)),
                                   view=view, context=True, tmax=tmax)
                score, why = SF.equipment_score(feats)
            except Exception:
                score, why, feats = 0.0, [], {}
            d["equip_score"] = float(score)
            d["ctx_growth"] = float(feats.get("ctx_growth", 0.0))
            d["ctx_rect_fill"] = float(feats.get("ctx_rect_fill", 0.0))
            # Skin priority still wins: a face is a person whatever the
            # surroundings look like.
            if score >= equip_thresh and not d.get("priority"):
                d["reject_reason"] = "; ".join(why[:3])
                continue
            kept.append(d)
        dets = kept

    amb_now = float(np.median(data))
    _warm_lbl = [None]        # warm-silhouette labelling, computed on demand

    def _warm_labels():
        if _warm_lbl[0] is None:
            wlo = amb_now + omega_delta
            wm = ((data >= wlo) & (data <= tmax)) if tmax is not None \
                else (data >= wlo)
            wm = cv2.morphologyEx(wm.astype(np.uint8), cv2.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8))
            _warm_lbl[0] = cv2.connectedComponents(wm)[1]
        return _warm_lbl[0]

    _owner = [None]

    def _owner_map(all_dets):
        """
        Assign every warm pixel to its NEAREST hot blob.

        Clipping each blob's warm region to a box around its own hot pixels
        fails in both directions: a coated person's hot bbox is just the face,
        so the swinging hands fall outside it and the articulation signal is
        lost; while a loose box lets a person and adjacent equipment share
        limbs. Nearest-seed assignment splits a merged warm silhouette along
        the natural boundary instead of an arbitrary rectangle.
        """
        if _owner[0] is None:
            dists = []
            for d in all_dets:
                x, y, w, h = d["bbox"]
                seed = np.ones(mask.shape, np.uint8)
                seed[y:y + h, x:x + w] = 1 - (binm[y:y + h, x:x + w] > 0)
                dists.append(cv2.distanceTransform(seed, cv2.DIST_L2, 3))
            _owner[0] = (np.argmin(np.stack(dists, 0), axis=0)
                         if dists else np.zeros(mask.shape, np.int32))
        return _owner[0]

    def _warm_of(hot_sub):
        """The connected warm body containing this hot blob."""
        lbl = _warm_labels()
        ids, cnt = np.unique(lbl[hot_sub > 0], return_counts=True)
        keep = [i for i in ids[np.argsort(-cnt)] if i != 0]
        return (lbl == keep[0]).astype(np.uint8) * 255 if keep else hot_sub

    # ---- BODY EXTENSION (recover clothed torso / legs / feet) ---------------
    # A hoodie or jeans sits only 2-4 C above ambient, below the detection
    # threshold, so only the head survives and the blob has too few warm
    # centres to clear the p-filter. Re-threshold lower, and if the warm region
    # containing this blob is body-shaped (branching, concave, head at one
    # end) rather than equipment-shaped (rectangular, rigid), adopt it. Peaks
    # are then recounted over the whole person and rise on merit.
    if SF is not None and body_extend and tmin is not None:
        grown = []
        for d in dets:
            x, y, w, h = d["bbox"]
            sub = np.zeros(mask.shape, np.uint8)
            sub[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
            try:
                ok, ext, info = SF.body_extension(data, sub, amb_now,
                                                  low_delta=body_delta, tmax=tmax)
            except Exception:
                ok, ext, info = False, sub, {}
            d["extended"] = bool(ok)
            d.update({k: float(v) for k, v in info.items()})
            if ok:
                ys2, xs2 = np.where(ext > 0)
                if xs2.size:
                    x0, y0 = int(xs2.min()), int(ys2.min())
                    d["bbox"] = (x0, y0, int(xs2.max() - x0 + 1),
                                 int(ys2.max() - y0 + 1))
                    d["area_px"] = int(xs2.size)
                    d["centroid"] = (float(xs2.mean()), float(ys2.mean()))
                    d["_ext_mask"] = ext          # peaks recounted on this
            grown.append(d)
        dets = grown

    # Thermal texture is measured on the FINAL body (after merging/clustering),
    # not on individual fragments, so the peak count reflects the whole person.
    binm = mask > 0
    for di, d in enumerate(dets):
        x, y, w, h = d["bbox"]
        if d.get("_ext_mask") is not None:
            # count warm centres over the WHOLE body, not just the hot head
            ys_, xs_ = np.where(d.pop("_ext_mask") > 0)
        else:
            sub = np.zeros_like(binm)
            sub[y:y + h, x:x + w] = binm[y:y + h, x:x + w]
            ys_, xs_ = np.where(sub)
        if xs_.size:
            npk, sd, rg = thermal_texture(data, ys_, xs_)
        else:
            npk, sd, rg = 1, 0.0, 0.0
        d["peaks"], d["std_c"], d["range_c"] = int(npk), float(sd), float(rg)

        # Vertical thermal gradient. A body cools monotonically downward: the
        # head is exposed, the torso insulated, the legs both clothed and
        # furthest from the core. That is the Fanger heat balance applied down
        # the body, not a heuristic. Equipment has an arbitrary profile set by
        # where its vents happen to be. Reported as the correlation between row
        # index and mean row temperature -- negative means cooling downward.
        if ys_.size > 12:
            rr = np.unique(ys_)
            if rr.size >= 4:
                means = np.array([data[ys_[ys_ == r], xs_[ys_ == r]].mean()
                                  for r in rr])
                if means.std() > 1e-6 and rr.std() > 1e-6:
                    rc = np.corrcoef(rr.astype(np.float64), means)[0, 1]
                    d["vgrad"] = float(rc) if np.isfinite(rc) else 0.0
                else:
                    d["vgrad"] = 0.0
            else:
                d["vgrad"] = 0.0
        else:
            d["vgrad"] = 0.0
        # Peak GEOMETRY for the temporal rigidity test — measured on the WARM
        # silhouette, not the hot mask. The hot mask is face and hands only;
        # its peaks sit inside the head and barely move relative to each other,
        # so a walking person scores nearly as rigid as a laptop. The limbs are
        # where articulation actually shows, and they only exist in the warm
        # mask.
        if tmin is not None and xs_.size:
            hs = np.zeros(mask.shape, np.uint8)
            hs[y:y + h, x:x + w] = binm[y:y + h, x:x + w] * 255
            wsub = _warm_of(hs)
            wsub = wsub * (_owner_map(dets) == di)
            wys, wxs = np.where(wsub > 0)
            d["peak_pts"] = (peak_points(data, wys, wxs) if wxs.size
                             else np.empty((0, 2), np.float64))
        else:
            d["peak_pts"] = (peak_points(data, ys_, xs_) if xs_.size
                             else np.empty((0, 2), np.float64))

        # Head-shoulder omegas.
        #
        # Fitted to the WARM silhouette (ambient + omega_delta), not the hot
        # detection mask. Hair and scalp are 3-6 C cooler than the face, and
        # clothed shoulders cooler still, so the head-and-shoulder OUTLINE only
        # exists in the warm mask. The hot face is then the anchor that must
        # sit inside the dome — which is what a laptop cannot reproduce.
        if SF is not None and F["omega"] and view in ("horizontal", "any"):
            x, y, w, h = d["bbox"]
            hot_sub = np.zeros(mask.shape, np.uint8)
            hot_sub[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
            # The warm silhouette must be the WHOLE connected warm body, not a
            # padded crop of the hot head — otherwise the shoulder measurement
            # is taken across the head itself and the ratio is meaningless.
            warm_sub = _warm_of(hot_sub)
            try:
                if F["orot"]:
                    oms = SF.find_omegas_rotated(warm_sub, hot_sub,
                                                 max_tilt=omega_max_tilt)
                else:
                    oms = SF.find_omegas_thermal(warm_sub, hot_sub)
            except Exception:
                oms = []
            d["omegas"] = oms
            d["omega_count"] = len(oms)
            d["omega_score"] = float(oms[0]["score"]) if oms else 0.0

    # Scale consistency across the frame. A laptop hotspot can trace a dome
    # that passes the local ratio test but at a fraction of head size; only a
    # comparison BETWEEN omegas exposes that, since each descriptor on its own
    # is deliberately scale-invariant.
    if SF is not None and F["omega"] and F["OmScale"]:
        try:
            SF.suppress_small_omegas(dets, min_ratio=omega_scale_ratio)
        except Exception:
            pass

    # ---- OMEGA SPLIT (toggle 'S') ------------------------------------------
    # Two people standing shoulder to shoulder merge into one connected blob,
    # and no per-blob descriptor can undo that — the evidence that there are
    # two of them is that there are two heads. Watershed already cuts on the
    # distance transform, which fails when the bodies overlap enough that no
    # valley forms between them; the omega detector works on the upper contour
    # instead and still sees two domes.
    #
    # So: if a blob contains more than one omega, cut it. Each pixel goes to
    # the nearest omega axis, which is the right rule for this geometry because
    # people standing together separate side by side. Every piece is then
    # re-measured on its own — area, peaks, temperature — so it has to earn its
    # place through the later filters rather than inheriting the parent's.
    if F["osplit"] and SF is not None and view in ("horizontal", "any"):
        pieces = []
        for d in dets:
            oms = d.get("omegas") or []
            if len(oms) < 2:
                pieces.append(d)
                continue
            x, y, w, h = d["bbox"]
            sub = np.zeros(mask.shape, bool)
            sub[y:y + h, x:x + w] = binm[y:y + h, x:x + w]
            ys_, xs_ = np.where(sub)
            if xs_.size == 0:
                pieces.append(d)
                continue

            axes = np.array(sorted({int(om["x"]) for om in oms}), np.float64)
            if axes.size < 2:
                pieces.append(d)
                continue
            owner = np.argmin(np.abs(xs_[:, None] - axes[None, :]), axis=1)

            made = []
            for k, ax in enumerate(axes):
                sel = owner == k
                if int(sel.sum()) < max(4, min_area // 2):
                    continue
                yy, xx = ys_[sel], xs_[sel]
                bx0, by0 = int(xx.min()), int(yy.min())
                bx1, by1 = int(xx.max()), int(yy.max())

                mine = [om for om in oms if int(om["x"]) == int(ax)]
                # the box must ENCLOSE its omega: the head box comes from the
                # warm silhouette and can reach above/beside the hot pixels
                for om in mine:
                    hb = om.get("head_box")
                    if hb:
                        hx, hy, hw_, hh_ = hb
                        bx0, by0 = min(bx0, hx), min(by0, hy)
                        bx1, by1 = max(bx1, hx + hw_ - 1), max(by1, hy + hh_ - 1)
                bx0, by0 = max(0, bx0), max(0, by0)
                bx1 = min(mask.shape[1] - 1, bx1)
                by1 = min(mask.shape[0] - 1, by1)

                nd = dict(d)
                nd["bbox"] = (bx0, by0, bx1 - bx0 + 1, by1 - by0 + 1)
                nd["area_px"] = int(sel.sum())
                # Re-measure the geometry. Copying the parent's aspect/extent/
                # rect_fill would describe the merged pair, not this person.
                pw, ph = bx1 - bx0 + 1, by1 - by0 + 1
                nd["aspect"] = float(ph) / max(1.0, float(pw))
                nd["extent"] = float(sel.sum()) / max(1.0, float(pw * ph))
                pts = np.stack([xx, yy], axis=1).astype(np.int32)
                try:
                    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
                    nd["rect_fill"] = float(sel.sum()) / max(1.0, rw * rh)
                except Exception:
                    nd["rect_fill"] = nd["extent"]
                nd["centroid"] = (float(xx.mean()), float(yy.mean()))
                nd["val_max"] = float(data[yy, xx].max())
                npk2, sd2, rg2 = thermal_texture(data, yy, xx)
                nd["peaks"] = int(npk2)
                nd["std_c"] = float(sd2)
                nd["range_c"] = float(rg2)
                nd["peak_pts"] = peak_points(data, yy, xx)
                nd["omegas"] = mine
                nd["omega_count"] = len(mine)
                nd["omega_score"] = float(mine[0]["score"]) if mine else 0.0
                nd["omega_split"] = True
                made.append(nd)

            # A failed split must leave the original detection intact, never
            # delete it. Below shoulder width the two domes merge, confidence
            # collapses, and the pieces would be thrown out by MinPeaks — which
            # would turn one correct detection into zero. Only accept a split
            # where every piece carries a confident omega.
            ok = (len(made) >= 2
                  and all(m.get("omega_score", 0.0) >= 0.5 for m in made))
            pieces.extend(made if ok else [d])
        dets = pieces

    # Single-hot-object rejection.
    #
    # Measured on real captures vs synthetic objects:
    #   humans          peaks 2-10   (face, neck, hands, gaps in clothing)
    #   uniform objects peaks 1      (lamp, laptop, reflection, hotspot)
    # Temperature variance does NOT separate them — a gaussian hotspot has
    # higher std than a real person — so the count of distinct warm centres is
    # the discriminator, not how varied the blob is.
    #
    # Only applied above peak_check_area: a distant person is a few pixels and
    # legitimately has one peak, so small blobs are exempt.
    mp = rules.get("min_peaks", min_peaks) if min_peaks is None else min_peaks
    pca = rules.get("peak_check_area", peak_check_area) if peak_check_area is None \
        else peak_check_area
    if mp and mp > 1:
        # Skin-priority blobs are exempt: a face at 33-35 C is a person even
        # if it presents as a single warm centre.
        # An omega-backed split piece is exempt. Splitting only happens because
        # two head-and-shoulder domes were found with a hot face inside each;
        # that is stronger evidence than the warm-centre count, and a person
        # cut out of a merged pair legitimately carries fewer centres than the
        # pair did.
        dets = [d for d in dets
                if d.get("priority") or d.get("far_field")
                or d["area_px"] < pca or d["peaks"] >= mp
                or (d.get("omega_split") and d.get("omega_score", 0.0) >= 0.5)]

    # ---- P-FILTER (all view modes, toggled with 'p') ------------------------
    # Reject anything with p_min or fewer warm centres. A body radiates from
    # several places at once — face, neck, hands, gaps in clothing — while a
    # manufactured heat source has one or two. Applied AFTER merging so the
    # count refers to the assembled body, not a fragment.
    #
    # Unconditional by design: it overrides skin priority too. That is the
    # trade — it is the strongest clutter rejector available, at the cost of
    # distant or heavily-clothed people who cannot show enough structure.
    if p_filter:
        if F.get("rigid"):
            # Bonus system active. Do NOT drop here — a rejected blob still
            # needs a track so it can accumulate the motion history that earns
            # it readmission. Tag it and let the main loop decide once the
            # tracker has evidence.
            for d in dets:
                d["p_reject"] = (d.get("peaks", 0) <= p_min
                                 and not d.get("far_field"))
        else:
            dets = [d for d in dets
                    if d.get("peaks", 0) > p_min or d.get("far_field")]

    # ---- GROUND-PLANE SCALE PRIOR (toggle 'G') -----------------------------
    # With a fixed camera, a person standing on the floor has a DETERMINED
    # apparent height for their image position: nearer feet sit lower in the
    # frame and subtend more pixels. Under the usual linear approximation,
    #     h_expected(y_feet) = gain * (y_feet - horizon)
    # This is the geometric prior that carried pre-deep-learning pedestrian
    # detection, and it rejects clutter for a reason no thermal or shape
    # descriptor can reach: a laptop sits on a DESK, not on the floor, so it is
    # the wrong size for where it appears. Same for wall heaters and monitors.
    #
    # Only meaningful in the forward-looking view; overhead has no ground plane
    # in this sense.
    if F["ground"] and view in ("horizontal", "any") and gnd_gain > 0:
        keep = []
        for d in dets:
            x, y, w, h = d["bbox"]
            y_feet = y + h
            h_exp = gnd_gain * max(1.0, y_feet - gnd_horizon)
            d["gnd_exp"] = float(h_exp)
            d["gnd_err"] = float(abs(h - h_exp) / max(1.0, h_exp))
            if d["gnd_err"] <= gnd_tol:
                keep.append(d)
        dets = keep

    if F["vgrad"] and view in ("horizontal", "any"):
        # Negative correlation = cooling downward = body-like.
        dets = [d for d in dets
                if d.get("area_px", 0) < 60 or d.get("vgrad", 0.0) <= vgrad_max]

    # ---- FAR FIELD (horizontal only, toggle 'F') ---------------------------
    # Runs LAST, on what the near-body gates discarded. Placing it earlier
    # meant a distant blob was still sitting in dets when the pass built its
    # exclusion mask, so the pass skipped the very blobs it exists to rescue,
    # and only got rid of them a few lines later.
    #
    # A person at range is not a small version of a person up close. The
    # clothed torso drops below threshold first, then the hands, and what
    # survives is a few pixels of face — mixed with cool background inside each
    # pixel, so even the peak temperature falls. By the time someone is across
    # a room they present as a compact warm island of 5-60 px with one warm
    # centre, which every gate tuned for a near body rejects: too small, too
    # few peaks, no meaningful aspect ratio.
    #
    # So this is a SEPARATE pass with its own evidence, not a relaxation of the
    # main one. The discriminator for a blob this small is LOCAL CONTRAST: a
    # distant body is a warm island standing clear of the surface behind it,
    # while sensor noise and gradual warm patches are not. Anything found here
    # is flagged far_field and is exempt from the near-body gates, because
    # those gates measure structure that genuinely is not resolvable.
    if F["farfield"] and view == "horizontal" and tmin is not None:
        # Exclude only what a SURVIVING detection already claims, not the whole
        # threshold mask. A distant person usually does clear the main
        # threshold — they are then thrown out by the near-body gates. Masking
        # off the entire mask meant those blobs were excluded from the pass
        # built to rescue them.
        # Exclude only what a SURVIVING detection already claims, not the whole
        # threshold mask — a distant person usually does clear the threshold and
        # is then thrown out by the near-body gates, so masking the whole mask
        # would skip the very blobs this pass exists to rescue.
        #
        # The margin is generous because the dominant false positive here is a
        # HAND or FOOT of an already-detected person poking outside their box.
        # That is not a second person, and at this blob size nothing about the
        # fragment itself says so — only its proximity does.
        taken = np.zeros(mask.shape, np.uint8)
        near_boxes = []
        for d in dets:
            bx_, by_, bw_, bh_ = d["bbox"]
            pad = max(6, int(0.35 * max(bw_, bh_)))
            taken[max(0, by_ - pad):by_ + bh_ + pad,
                  max(0, bx_ - pad):bx_ + bw_ + pad] = 1
            near_boxes.append((bx_, by_, bw_, bh_, pad))
        flo = amb_now + far_delta
        fhi = tmax if tmax is not None else 60.0
        fm = ((data >= flo) & (data <= fhi) & (taken == 0)).astype(np.uint8)
        if moving is not None:
            # A distant person crossing a room is moving; a warm patch on a
            # wall is not. At this blob size motion is the only other evidence
            # available, and without it the pass fires on room clutter.
            fm = fm * (moving > 0).astype(np.uint8)
        fm = cv2.morphologyEx(fm, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        nlab, lab, stats, cents = cv2.connectedComponentsWithStats(fm, 8)

        IMH = float(data.shape[0])
        for li in range(1, nlab):
            a = int(stats[li, cv2.CC_STAT_AREA])
            bx = int(stats[li, cv2.CC_STAT_LEFT]); by = int(stats[li, cv2.CC_STAT_TOP])
            bw = int(stats[li, cv2.CC_STAT_WIDTH]); bh = int(stats[li, cv2.CC_STAT_HEIGHT])

            # ---- PERSPECTIVE GATE ----------------------------------------
            # Looking forward, distance grows as you go UP the frame: the
            # bottom of the image is the floor at your feet, the top is the far
            # wall. So a blob's expected size is a function of its row, and
            # "small" only means "distant" in the upper part of the frame.
            #
            # A 20 px blob near the bottom is NOT a distant person — at that
            # range a person fills a third of the frame. It is a hand, a foot,
            # or clutter. Rejecting it here is what stops the far-field pass
            # from firing all over the near field.
            #
            # In the upper band the ceiling on area grows linearly downward,
            # because a person one row lower is one step closer and subtends
            # proportionally more pixels.
            fy = (by + bh) / max(1.0, IMH)          # feet row, normalised
            if fy > far_row_max:
                continue                             # near field: not our job
            span = max(1e-3, far_row_max)
            cap = far_max_area * (0.35 + 1.30 * (fy / span))
            if not (far_min_area <= a <= cap):
                continue

            # A distant body is blobby, not a thin streak along a warm edge.
            if a / max(1.0, float(bw * bh)) < 0.42:
                continue
            if max(bw, bh) / max(1.0, min(bw, bh)) > 3.2:
                continue

            m_i = lab == li
            ys_i, xs_i = np.where(m_i)
            vals = data[ys_i, xs_i]

            # LOCAL CONTRAST: blob against the ring immediately around it.
            # Referencing the ring rather than the frame ambient is what makes
            # this work on a warm wall or a sunlit floor.
            gx0, gy0 = max(0, bx - 4), max(0, by - 4)
            gx1, gy1 = min(data.shape[1], bx + bw + 4), min(data.shape[0], by + bh + 4)
            ring = np.ones((gy1 - gy0, gx1 - gx0), bool)
            ring[by - gy0:by - gy0 + bh, bx - gx0:bx - gx0 + bw] = False
            rv = data[gy0:gy1, gx0:gx1][ring]
            if rv.size < 8:
                continue
            contrast = float(vals.mean() - np.median(rv))
            if contrast < far_contrast:
                continue

            # second proximity guard, on the blob's own centroid
            fcx, fcy = float(xs_i.mean()), float(ys_i.mean())
            if any(bx_ - pad <= fcx <= bx_ + bw_ + pad and
                   by_ - pad <= fcy <= by_ + bh_ + pad
                   for bx_, by_, bw_, bh_, pad in near_boxes):
                continue

            npk3, sd3, rg3 = thermal_texture(data, ys_i, xs_i)
            dets.append(dict(
                bbox=(bx, by, bw, bh), area_px=a,
                centroid=(float(xs_i.mean()), float(ys_i.mean())),
                val_max=float(vals.max()), val_mean=float(vals.mean()),
                aspect=float(bh) / max(1.0, float(bw)),
                extent=a / max(1.0, float(bw * bh)),
                rect_fill=a / max(1.0, float(bw * bh)),
                peaks=int(npk3), std_c=float(sd3), range_c=float(rg3),
                peak_pts=peak_points(data, ys_i, xs_i),
                omegas=[], omega_count=0, omega_score=0.0,
                far_field=True, far_contrast=contrast,
                priority=False, extended=False))

    # ---- GRADED FAR FIELD (horizontal only, toggle 'D') --------------------
    # FarField uses a hard band: above a row, accept small blobs. That leaves
    # mid range badly served — too far to survive the near-body gates, too big
    # and too low in the frame for the far-field band. The gap is exactly where
    # detection was failing.
    #
    # This pass replaces the band with a CONTINUOUS model. Under linear
    # perspective a standing person's pixel height and the row of their feet
    # are both inversely proportional to range, so
    #
    #     (y_feet - horizon)  =  k * h        with  k = (y_ref - horizon) / h_ref
    #
    # One measured person calibrates it. From the 8.5 m reference (h_ref px
    # tall, feet at y_ref) every other range follows: a person at 4 m is
    # 8.5/4 times taller with their feet proportionally further down.
    #
    # A blob is accepted when its height MATCHES what its row predicts. That is
    # a much sharper test than "small and high up", because it rejects things
    # that are the wrong size for where they are in both directions — a big
    # blob high in the frame and a small blob low in it are equally impossible.
    #
    # The contrast requirement is graded too: apparent contrast falls with
    # range as background mixes into each pixel, so demanding a fixed margin
    # from a distant target would silently reimpose a range limit.
    if F["gradfar"] and view == "horizontal" and tmin is not None:
        IMH = float(data.shape[0])
        k_ref = max(1e-3, (gfar_ref_row - gfar_horizon) / max(1.0, gfar_ref_h))

        taken2 = np.zeros(mask.shape, np.uint8)
        claimed = []
        for d in dets:
            bx_, by_, bw_, bh_ = d["bbox"]
            pad = max(6, int(0.35 * max(bw_, bh_)))
            taken2[max(0, by_ - pad):by_ + bh_ + pad,
                   max(0, bx_ - pad):bx_ + bw_ + pad] = 1
            claimed.append((bx_, by_, bw_, bh_, pad))

        glo = amb_now + gfar_delta
        ghi = tmax if tmax is not None else 60.0
        gm = ((data >= glo) & (data <= ghi) & (taken2 == 0)).astype(np.uint8)
        gm = cv2.morphologyEx(gm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        nl2, lb2, st2, _ = cv2.connectedComponentsWithStats(gm, 8)

        for li in range(1, nl2):
            a = int(st2[li, cv2.CC_STAT_AREA])
            if a < 4:
                continue
            bx = int(st2[li, cv2.CC_STAT_LEFT]); by = int(st2[li, cv2.CC_STAT_TOP])
            bw = int(st2[li, cv2.CC_STAT_WIDTH]); bh = int(st2[li, cv2.CC_STAT_HEIGHT])

            y_feet = float(by + bh)
            if y_feet <= gfar_horizon + 1.0:
                continue                       # at or above the horizon
            h_exp = (y_feet - gfar_horizon) / k_ref
            if h_exp < 2.0:
                continue
            ratio = bh / h_exp
            if not (gfar_lo <= ratio <= gfar_hi):
                continue                       # wrong size for its row

            # implied range, for the caller and for grading the contrast test
            rng_m = gfar_ref_range * (gfar_ref_h / max(1.0, float(bh)))
            if rng_m > gfar_max_range:
                continue

            # width must be plausible for a body of that height
            if bw > bh * 1.6 or bw < bh * 0.20:
                continue

            m_i = lb2 == li
            ys_i, xs_i = np.where(m_i)
            vals = data[ys_i, xs_i]

            gx0, gy0 = max(0, bx - 4), max(0, by - 4)
            gx1, gy1 = min(data.shape[1], bx + bw + 4), min(data.shape[0], by + bh + 4)
            ring = np.ones((gy1 - gy0, gx1 - gx0), bool)
            ring[by - gy0:by - gy0 + bh, bx - gx0:bx - gx0 + bw] = False
            rv = data[gy0:gy1, gx0:gx1][ring]
            if rv.size < 8:
                continue
            contrast = float(vals.mean() - np.median(rv))

            # graded requirement: full margin near, relaxed at range
            frac = min(1.0, rng_m / max(1e-3, gfar_max_range))
            need = gfar_contrast_near + (gfar_contrast_far - gfar_contrast_near) * frac
            if contrast < need:
                continue

            fcx, fcy = float(xs_i.mean()), float(ys_i.mean())
            if any(bx_ - pad <= fcx <= bx_ + bw_ + pad and
                   by_ - pad <= fcy <= by_ + bh_ + pad
                   for bx_, by_, bw_, bh_, pad in claimed):
                continue

            # EQUIPMENT CONTEXT. Static suppression is off in this view by
            # design — a person at range barely moves in image space — but that
            # was also what kept powered equipment out. So the rigid-surround
            # test has to be applied here explicitly, or every warm appliance
            # at a plausible row becomes a person.
            if SF is not None and F["equip"] and bh >= 6:
                sub_i = np.zeros(mask.shape, np.uint8)
                sub_i[ys_i, xs_i] = 255
                try:
                    ctx = SF.thermal_context(data, sub_i, amb_now, tmax=tmax)
                    if SF.equipment_score(ctx) >= equip_thresh:
                        continue
                except Exception:
                    pass

            npk4, sd4, rg4 = thermal_texture(data, ys_i, xs_i)
            dets.append(dict(
                bbox=(bx, by, bw, bh), area_px=a,
                centroid=(fcx, fcy),
                val_max=float(vals.max()), val_mean=float(vals.mean()),
                aspect=float(bh) / max(1.0, float(bw)),
                extent=a / max(1.0, float(bw * bh)),
                rect_fill=a / max(1.0, float(bw * bh)),
                peaks=int(npk4), std_c=float(sd4), range_c=float(rg4),
                peak_pts=peak_points(data, ys_i, xs_i),
                omegas=[], omega_count=0, omega_score=0.0,
                # The p-filter exemption is EARNED BY SIZE, not by being a
                # far-field hit. A blob 20 px tall is a person at ~3 m and can
                # show warm centres, so it must; only once a body is small
                # enough that structure is unresolvable does the exemption
                # apply. Without this the pass hands every mid-range warm
                # object — a laptop at 3 m, say — a free pass around the one
                # filter that actually rejects equipment.
                far_field=bool(bh <= gfar_structure_h), grad_far=True,
                far_contrast=contrast,
                range_m=float(rng_m), h_expected=float(h_exp),
                priority=False, extended=False))

    dets.sort(key=lambda d: (d.get("priority", False), d["area_px"]), reverse=True)
    return dets, mask


# ----------------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------------
def colorize(data):
    lo = float(np.percentile(data, 1))
    hi = float(np.percentile(data, 99))
    if hi - lo < 1e-3:
        hi = lo + 1.0
    norm = np.clip((data - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def draw_omega(vis, dets, S=1):
    """
    Draw the omega on the anatomy it actually describes:

      * dome    -> the HEAD, traced on the warm silhouette (hair/scalp)
      * legs    -> the SHOULDERS, flaring out and down from the dome
      * marker  -> the HOT FACE, which must sit inside the dome

    Previously the dome was drawn around the face because it was fitted to the
    hot mask. The face is a patch within the head, not its outline.
    """
    for d in dets:
        for om in d.get("omegas", []):
            x = int(om["x"]) * S
            hw = max(2, int(om["head_w"] / 2)) * S
            sw = max(3, int(om["shoulder_w"] / 2)) * S

            hb = om.get("head_box")
            if hb:
                bx, by, bw, bh = [v * S for v in hb]
                cxh, cyh = bx + bw // 2, by + bh // 2
                ax, ay = max(3, bw // 2), max(3, bh // 2)
            else:
                cxh, cyh, ax, ay = x, (d["bbox"][1] * S) + hw, hw, hw

            # Draw at the angle the omega was actually found at, so a leaning
            # person gets a leaning omega rather than an upright one that does
            # not match the body.
            rot = float(om.get("angle", 0.0))
            th = np.radians(-rot)
            cs, sn = np.cos(th), np.sin(th)

            def rp(dx, dy):
                return (int(cxh + dx * cs - dy * sn), int(cyh + dx * sn + dy * cs))

            cv2.ellipse(vis, (cxh, cyh), (ax, ay), -rot, 180, 360,
                        (0, 220, 255), 2, cv2.LINE_AA)
            cv2.line(vis, rp(-ax, 0), rp(-sw, ay), (0, 220, 255), 2, cv2.LINE_AA)
            cv2.line(vis, rp(ax, 0), rp(sw, ay), (0, 220, 255), 2, cv2.LINE_AA)
            cv2.line(vis, rp(-sw, ay), rp(sw, ay), (0, 180, 210), 1, cv2.LINE_AA)

            # the hot face anchor, inside the dome
            if "core_x" in om:
                cv2.drawMarker(vis, (int(om["core_x"] * S), int(om["core_y"] * S)),
                               (60, 60, 255), cv2.MARKER_TILTED_CROSS, 12, 2)

            rr = om.get("angle", 0.0)
            lbl = f"{om['score']:.2f}" + (f" {rr:+.0f}deg" if abs(rr) >= 1 else "")
            cv2.putText(vis, lbl, (cxh - 16, max(14, cyh - ay - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, lbl, (cxh - 16, max(14, cyh - ay - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
    return vis


def draw_probe(vis, data, mx, my, S, is_temp):
    """
    Cursor temperature probe.

    Reads the raw sensor array, not the colourised image: the palette is
    re-normalised per frame by percentile, so identical colours mean different
    temperatures from one frame to the next. Only the underlying value is
    trustworthy, and being able to point at hair, face and coat and read the
    actual numbers is how the coupling constants get checked against reality
    rather than assumed.

    Reports the 3x3 mean alongside the pixel value — single Lepton pixels are
    noisy (~50 mK NETD) and the neighbourhood mean is the more honest figure.
    """
    if mx is None:
        return vis
    sx, sy = mx // S, my // S
    if not (0 <= sx < data.shape[1] and 0 <= sy < data.shape[0]):
        return vis
    v = float(data[sy, sx])
    y0, y1 = max(0, sy - 1), min(data.shape[0], sy + 2)
    x0, x1 = max(0, sx - 1), min(data.shape[1], sx + 2)
    v3 = float(data[y0:y1, x0:x1].mean())

    px, py = sx * S + S // 2, sy * S + S // 2
    cv2.line(vis, (px, 0), (px, vis.shape[0]), (200, 200, 200), 1)
    cv2.line(vis, (0, py), (vis.shape[1], py), (200, 200, 200), 1)
    cv2.rectangle(vis, (sx * S, sy * S), (sx * S + S, sy * S + S), (0, 255, 255), 2)

    unit = "C" if is_temp else ""
    txt = f"{v:.2f}{unit}  3x3 {v3:.2f}{unit}" if is_temp else f"{v:.0f}  3x3 {v3:.0f}"
    sub = f"px ({sx},{sy})"
    tx = min(px + 14, vis.shape[1] - 210)
    ty = max(28, py - 14)
    cv2.rectangle(vis, (tx - 6, ty - 20), (tx + 200, ty + 16), (16, 16, 20), -1)
    cv2.rectangle(vis, (tx - 6, ty - 20), (tx + 200, ty + 16), (0, 255, 255), 1)
    cv2.putText(vis, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, sub, (tx, ty + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (170, 170, 175), 1, cv2.LINE_AA)
    return vis


def draw_boxes(vis, dets, is_temp, S=1, show_boxes=True, show_ids=False):
    """Detection boxes on the upscaled canvas — full-resolution text."""
    unit = "C" if is_temp else ""
    for d in (dets if show_boxes else []):
        x, y, w, h = [v * S for v in d["bbox"]]
        pri = d.get("priority", False)
        col = (0, 255, 255) if pri else (0, 255, 0)      # yellow = skin priority
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
        cx, cy = int(d["centroid"][0] * S), int(d["centroid"][1] * S)
        cv2.drawMarker(vis, (cx, cy), col, cv2.MARKER_CROSS, 14, 2)
        bn = d.get("peaks_bonus", 0)
        tag = (f"{d['val_max']:.0f}{unit}/{d.get('peaks', 0)}p"
               + (f"+{bn}" if bn else "") + ("*" if pri else ""))
        cv2.putText(vis, tag, (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, tag, (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        if show_ids and "track_id" in d:
            idt = f"#{d['track_id']}"
            iy = y + h + 18 if y + h + 18 < vis.shape[0] - 4 else y + h - 6
            cv2.putText(vis, idt, (x, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, idt, (x, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 160, 60), 1, cv2.LINE_AA)
    return vis


# -- sidebar -----------------------------------------------------------------
SIDEBAR_W = 330
_C_DIM   = (120, 120, 130)
_C_TEXT  = (225, 225, 230)
_C_ON    = (110, 255, 140)
_C_OFF   = (95, 95, 110)
_C_WARN  = (70, 70, 255)
_C_HEAD  = (255, 190, 90)


def _line(img, y, text, colour=_C_TEXT, x=12, sc=0.44, th=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, colour, th,
                cv2.LINE_AA)
    return y


def _rule(img, y, label=None):
    cv2.line(img, (12, y), (SIDEBAR_W - 12, y), (55, 55, 65), 1)
    if label:
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.rectangle(img, (18, y - 7), (18 + tw + 8, y + 7), (24, 24, 30), -1)
        _line(img, y + 4, label, _C_HEAD, x=22, sc=0.36)
    return y


def draw_sidebar(h, dets, bg, thr, sens, is_temp, fps, logging_on, view,
                 cohesion, band_lo, band_hi, p_min, flags, show_boxes,
                 show_omega, capture_on, capture_n, tracking,
                 probe_on=False, probe_xy=None, data=None, probe_temp=True):
    """
    Status panel rendered at native display resolution.

    The stage list used to be 4-character stubs painted onto a 160x120 sensor
    frame and then magnified 5x with nearest-neighbour — unreadable by
    construction. Drawing it here instead means the text is never scaled.
    """
    # Draw onto a canvas taller than any possible layout, then crop to the
    # ink. Clipping was previously possible because the panel was forced to the
    # frame's height regardless of how much it had to say.
    SCRATCH = 2000
    sb = np.full((SCRATCH, SIDEBAR_W, 3), 18, np.uint8)
    y = 26
    mode = "RAD" if is_temp else "AGC"
    _line(sb, y, "FLUXNET", _C_HEAD, sc=0.58); _line(sb, y, mode, _C_DIM, x=120, sc=0.42)
    _line(sb, y, f"{fps:4.1f} fps", _C_DIM, x=200, sc=0.42)

    y += 40
    ncol = _C_ON if dets else _C_DIM
    _line(sb, y, f"n = {len(dets)}", ncol, sc=1.05, th=2)
    if tracking:
        _line(sb, y, "tracked", (255, 160, 60), x=140, sc=0.40)
    nom = sum(d.get("omega_count", 0) for d in dets)
    _line(sb, y + 20, f"omegas {nom}", _C_DIM, sc=0.40)

    y += 44
    _rule(sb, y, "SCENE")
    vlabel = SHAPE_RULES.get(view, SHAPE_RULES["any"])["label"]
    vcolor = {"vertical": (80, 220, 255),
              "horizontal": (255, 200, 80)}.get(view, (200, 200, 200))
    y += 24; _line(sb, y, f"[v] {vlabel}", vcolor, sc=0.46)
    y += 20
    # cohesion >= 3 merges two people standing close: flag it loudly
    ccol = _C_WARN if cohesion >= 3 else _C_TEXT
    _line(sb, y, f"cohesion {cohesion}" + ("  MERGES PAIRS" if cohesion >= 3 else ""),
          ccol, sc=0.42)
    y += 20; _line(sb, y, f"bg {bg:5.1f}   thr {thr:5.1f}", _C_TEXT, sc=0.42)
    y += 20
    sens_s = f"+{sens:.1f}C" if is_temp else f"p{sens:.0f}"
    band = f"band {band_lo:.0f}-{band_hi:.0f}" if (is_temp and band_lo is not None) else ""
    _line(sb, y, f"sens {sens_s}   {band}", _C_DIM, sc=0.40)

    y += 22
    _rule(sb, y, "STAGES")
    y += 6
    for kkey, name, lab, _d in TOGGLES:
        y += 19
        on = flags.get(name, False)
        c = _C_ON if on else _C_OFF
        cv2.rectangle(sb, (12, y - 9), (24, y + 2), c, -1 if on else 1)
        _line(sb, y, kkey, _C_DIM, x=32, sc=0.40)
        _line(sb, y, lab, c, x=52, sc=0.44)
        if name == "pfilter" and on:
            _line(sb, y, f"<={p_min}", _C_DIM, x=170, sc=0.38)
        _line(sb, y, "ON" if on else "off", c, x=240, sc=0.40)

    y += 22
    _rule(sb, y, "DISPLAY")
    y += 20
    for lab, on, kkey in (("boxes", show_boxes, "o"), ("omega", show_omega, "O"),
                          ("probe", probe_on, "T"), ("logging", logging_on, "l")):
        _line(sb, y, f"{kkey}  {lab}", _C_TEXT if on else _C_OFF, sc=0.40)
        _line(sb, y, "ON" if on else "off", _C_ON if on else _C_OFF, x=240, sc=0.40)
        y += 17

    if probe_on and probe_xy and data is not None and probe_xy.get("x") is not None:
        sx = probe_xy["x"] // DISPLAY_SCALE
        sy = probe_xy["y"] // DISPLAY_SCALE
        if 0 <= sx < data.shape[1] and 0 <= sy < data.shape[0]:
            y += 6
            v = float(data[sy, sx])
            _line(sb, y, f"probe  {v:.2f}" + ("C" if probe_temp else ""),
                  (0, 255, 255), sc=0.48)
            y += 16
            _line(sb, y, f"({sx},{sy})", _C_DIM, sc=0.38)
            y += 8

    if capture_on:
        y += 6
        cv2.circle(sb, (18, y - 4), 5, (0, 0, 255), -1)
        _line(sb, y, f"REC  {capture_n} frames", (80, 80, 255), x=32, sc=0.44)
        y += 16

    # Per-detection detail flows below DISPLAY and is clipped to whatever room
    # is left, so it can never collide with the block above it.
    if dets:
        y += 20
        _rule(sb, y, "DETECTIONS")
        y += 20
        room = 8
        for d in dets[:room]:
            pri = "*" if d.get("priority") else " "
            oc = d.get("omega_count", 0)
            tid = f"#{d['track_id']}" if "track_id" in d else "  "
            rg = d.get("rigidity")
            rgs = f" r{rg:.3f}" if rg is not None else ""
            bn = d.get("peaks_bonus", 0)
            _line(sb, y, f"{tid:>4}{pri} {d['val_max']:4.1f}C "
                         f"{d.get('peaks', 0):2d}p{('+%d' % bn) if bn else '  '} "
                         f"{d['area_px']:4d}px"
                         + (f" {oc}w" if oc else "") + rgs,
                  (0, 255, 255) if d.get("priority") else _C_TEXT, sc=0.38)
            y += 16
        if len(dets) > room:
            _line(sb, y, f"+{len(dets) - room} more", _C_DIM, sc=0.36)
    return sb[:max(h, y + 14)]


HELP = """
  q/ESC quit | s save frame | l toggle logging
  [ less sensitive   ] more sensitive
  v cycle MOUNTING MODE:  vertical (down) / horizontal (fwd) / any
  1 = vertical (overhead)   2 = horizontal (wall)   3 = any
  m MORE cohesion (fewer fragments)   n LESS cohesion (more separation)
  b reset static background   B toggle static suppression on/off
  p toggle P-FILTER (drop blobs with <= --p-min warm centres)
  x toggle BODY-EXTENSION (recover cool clothed torso/legs)
  c START/STOP continuous capture  |  o body boxes on/off  |  O omega on/off

  STAGE TOGGLES (each shown along the bottom of the view):
    t TempBand   w Watershed  g Merge      u ClustSml   y ShapeGate
    k MinPeaks   p P-Filter   e EquipRej   i SkinPrio   x BodyExt
    z Omega      j OmScale    S OmSplit    N OmRot      F FarField
    D GradFarF
    f Kalman
    R NonRigid
    G GndPlane   A Gait       V VGrad      B StaticSup
    G GndPlane   A Gait       V VGrad

  C sample ground-plane calibration point (walk to several distances)

  T cursor TEMPERATURE PROBE (hover to read the exact pixel value)
  a print background/threshold | r cycle display (color/mask/both)
"""


# ----------------------------------------------------------------------------
def main():
    global SKIN_BAND_C            # retunable from the command line
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--delta", type=float, default=None,
                    help="radiometric: °C above ambient (default 4.0)")
    ap.add_argument("--pctl", type=float, default=None,
                    help="AGC: brightness percentile treated as hot (default 96)")
    ap.add_argument("--note", type=str, default="")
    ap.add_argument("--opencv", action="store_true",
                    help="skip libuvc and force the OpenCV capture path")
    ap.add_argument("--view", choices=VIEW_MODES, default="any",
                    help="mounting geometry: vertical (ceiling/down), "
                         "horizontal (wall/forward), any. Switch live with 'v'.")
    ap.add_argument("--tmax", type=float, default=DEFAULT_TMAX_C,
                    help=f"upper bound of the human band °C (default "
                         f"{DEFAULT_TMAX_C}; rejects laptops/lamps/radiators)")
    ap.add_argument("--tmin", type=float, default=DEFAULT_TMIN_C,
                    help=f"lower bound of the human band °C (default "
                         f"{DEFAULT_TMIN_C}; rejects sun-warmed surfaces)")
    ap.add_argument("--min-area", type=int, default=MIN_BLOB_AREA,
                    help="minimum blob area in px (lower = longer range)")
    ap.add_argument("--no-split", action="store_true",
                    help="disable watershed splitting of touching bodies")
    ap.add_argument("--min-sep", type=int, default=None,
                    help="vertical mode: min px between two people's centres "
                         "(shoulder width at your mount height; default 26)")
    ap.add_argument("--skin-lo", type=float, default=None,
                    help=f"low edge of the exposed-skin priority band °C "
                         f"(default {SKIN_BAND_C[0]}). Blobs containing skin "
                         f"bypass the shape and peak filters.")
    ap.add_argument("--skin-hi", type=float, default=None,
                    help=f"high edge of the skin priority band °C "
                         f"(default {SKIN_BAND_C[1]})")
    ap.add_argument("--min-peaks", type=int, default=None,
                    help="reject blobs with fewer distinct warm centres than "
                         "this (default 2). A human has several; a lamp or "
                         "laptop has one. Set 1 to disable.")
    ap.add_argument("--peak-check-area", type=int, default=None,
                    help="only apply --min-peaks above this blob area (px); "
                         "distant people legitimately have one peak")
    ap.add_argument("--kalman", action="store_true",
                    help="start with Kalman tracking on (toggle with 'f')")
    ap.add_argument("--no-review", action="store_true",
                    help="skip the review/ folder (boxes drawn, for spotting "
                         "false positives by eye). Never feed review/ to a "
                         "trainer.")
    ap.add_argument("--capture-png-overlay", action="store_true",
                    help="save the annotated display instead of a clean frame "
                         "(demo footage, NOT training data)")
    ap.add_argument("--capture-png-raw", action="store_true",
                    help="save 16-bit linear PNGs over a fixed temperature span "
                         "instead of a colour map — preserves the radiometry, "
                         "recommended if the model should see temperature")
    ap.add_argument("--capture-every", type=int, default=1,
                    help="during capture, save every Nth frame (default 1)")
    ap.add_argument("--scale", type=int, default=DISPLAY_SCALE,
                    help="display magnification (default 6 -> 960x720 view)")
    ap.add_argument("--rigid-thresh", type=float, default=0.005,
                    help="peak-constellation deformation below this is treated "
                         "as a rigid object (toggle 'R'). Synthetic: walking "
                         "person 0.011, static laptop 0.000, STILL person "
                         "0.000. MEASURE both classes on your own footage "
                         "first — the sidebar shows the live value as 'r'. "
                         "Sensor noise puts a nonzero floor under equipment "
                         "that this synthetic cannot predict.")
    ap.add_argument("--gait-frac", type=float, default=0.35,
                    help="fraction of width-oscillation power that must fall in "
                         "the 0.6-3.0 Hz walking band (toggle 'A'). Tracks "
                         "without enough history are never judged.")
    ap.add_argument("--gnd-horizon", type=float, default=0.0,
                    help="image row of the horizon for the ground-plane prior "
                         "(toggle 'G')")
    ap.add_argument("--gnd-gain", type=float, default=0.0,
                    help="expected person pixel-height per row below the "
                         "horizon. 0 disables. Calibrate with 'C': walk the "
                         "space and the console prints (y_feet, height) pairs "
                         "plus the fitted gain.")
    ap.add_argument("--gnd-tol", type=float, default=0.5,
                    help="fractional tolerance on expected height (0.5 = accept "
                         "half to 1.5x, generous on purpose)")
    ap.add_argument("--vgrad-max", type=float, default=-0.15,
                    help="max row/temperature correlation to accept as a body "
                         "(toggle 'V'). Negative = cools downward.")
    ap.add_argument("--rigid-bonus", type=int, default=4,
                    help="peaks credited to a track that demonstrates "
                         "articulation. Lets a heavily clothed person clear "
                         "the p-filter on MOTION evidence when the coat has "
                         "hidden the warm centres. Requires 'R' + 'f'.")
    ap.add_argument("--rigid-frames", type=int, default=8,
                    help="frames of history before the rigidity test is "
                         "allowed to judge a track")
    ap.add_argument("--probe", action="store_true",
                    help="start with the cursor temperature probe on "
                         "(toggle with 'T')")
    ap.add_argument("--omega-delta", type=float, default=2.0,
                    help="degrees above ambient defining the WARM silhouette "
                         "the omega is fitted to (hair/scalp + clothed "
                         "shoulders). The hot face anchors it from inside.")
    ap.add_argument("--far-delta", type=float, default=2.0,
                    help="degrees above ambient for the far-field pass "
                         "(toggle 'F', horizontal only)")
    ap.add_argument("--far-min-area", type=int, default=5)
    ap.add_argument("--far-max-area", type=int, default=90,
                    help="area ceiling at the BOTTOM of the far band; the "
                         "actual ceiling scales with row, from 35%% of this at "
                         "the top of the frame to 165%% at the far-field line")
    ap.add_argument("--far-contrast", type=float, default=2.5,
                    help="minimum degrees the blob must stand above the ring "
                         "around it — the discriminator at this size. Measured "
                         "knee: on 332 real frames containing NO distant "
                         "people, false positives run 292 at 2.0, 7 at 2.5, "
                         "0 at 3.0. Room clutter tops out near 2 C of local "
                         "contrast; a body should clear it comfortably.")
    ap.add_argument("--far-row-max", type=float, default=0.55,
                    help="far-field only considers blobs whose FEET sit above "
                         "this fraction of frame height. Below it the scene is "
                         "near field, where a small blob cannot be a distant "
                         "person. Raise if the camera is pitched further down.")
    ap.add_argument("--gfar-ref-row", type=float, default=25.0,
                    help="CALIBRATION: image row of the reference person's FEET")
    ap.add_argument("--gfar-ref-h", type=float, default=7.0,
                    help="CALIBRATION: pixel height of the reference person")
    ap.add_argument("--gfar-ref-range", type=float, default=8.5,
                    help="CALIBRATION: true distance to the reference person (m)")
    ap.add_argument("--gfar-horizon", type=float, default=0.0,
                    help="image row of the horizon; 0 = top of frame")
    ap.add_argument("--gfar-lo", type=float, default=0.55,
                    help="accept blobs from this fraction of predicted height")
    ap.add_argument("--gfar-hi", type=float, default=1.9,
                    help="...up to this multiple. Wide because a seated or "
                         "partly-occluded person is shorter than predicted.")
    ap.add_argument("--gfar-delta", type=float, default=2.5,
                    help="degrees above ambient for the graded pass")
    ap.add_argument("--gfar-max-range", type=float, default=12.0)
    ap.add_argument("--gfar-structure-h", type=float, default=12.0,
                    help="blobs TALLER than this (px) must still pass the "
                         "warm-centre filter; only smaller ones are exempt, "
                         "because below this a body cannot resolve structure. "
                         "12 px is about 5 m on the default calibration.")
    ap.add_argument("--gfar-contrast-near", type=float, default=3.5,
                    help="local contrast required at close range")
    ap.add_argument("--gfar-contrast-far", type=float, default=2.5,
                    help="...relaxed to this at max range, because apparent "
                         "contrast falls as background mixes into each pixel")
    ap.add_argument("--h-static", action="store_true",
                    help="keep static suppression ON in horizontal view "
                         "(off by default there: a seated or standing person "
                         "at range barely moves in image space and gets "
                         "absorbed into the background)")
    ap.add_argument("--h-strict", action="store_true",
                    help="restore the old close-range horizontal gates "
                         "(aspect>=1.1, extent>=0.25, 3 peaks above 150 px)")
    ap.add_argument("--omega-max-tilt", type=float, default=40.0,
                    help="degrees of head-tilt the rotated omega search covers "
                         "(toggle 'N'). Wider costs proportionally more time.")
    ap.add_argument("--omega-scale-ratio", type=float, default=0.40,
                    help="omegas narrower than this fraction of the frame's "
                         "largest confident omega are discarded (toggle 'j'). "
                         "Also caps range spread: 0.40 keeps a person 2.5x "
                         "further than the nearest one.")
    ap.add_argument("--omega-only", action="store_true",
                    help="start with body boxes hidden and only the omega "
                         "overlay drawn — the capture view")
    ap.add_argument("--body-extend", action="store_true",
                    help="recover cool clothed torso/legs below the detection "
                         "threshold when they form a body-shaped region "
                         "(toggle live with 'x')")
    ap.add_argument("--body-delta", type=float, default=2.0,
                    help="degrees above ambient used to find clothed body "
                         "parts (default 2.0)")
    ap.add_argument("--p-filter", action="store_true",
                    help="start with the p-filter ON: drop blobs with <= "
                         "--p-min warm centres (toggle live with 'p')")
    ap.add_argument("--p-min", type=int, default=4,
                    help="p-filter cutoff: blobs with this many peaks or "
                         "fewer are removed (default 4)")
    ap.add_argument("--equip-thresh", type=float, default=3.0,
                    help="equipment-evidence score above which a blob is "
                         "rejected in vertical mode (default 3.0; raise to be "
                         "more permissive, 99 disables)")
    ap.add_argument("--no-static", action="store_true",
                    help="disable static-object suppression (keeps furniture "
                         "and electronics as detections)")
    ap.add_argument("--cohesion", type=int, default=1,
                    help="anti-fragmentation strength 0-4 (default 1). "
                         "Bench-measured: 4 for vertical single-occupant, "
                         "2 max when multiple people may stand close "
                         "(>=3 merges two people). Live: m / n")
    args = ap.parse_args()

    globals()['DISPLAY_SCALE'] = max(2, int(args.scale))
    if args.h_strict:
        SHAPE_RULES["horizontal"].update(
            aspect_min=1.1, extent_min=0.25, min_peaks=3,
            peak_check_area=150, cluster_small=False, delta=4.0)
    global SKIN_BAND_FIXED
    if args.skin_lo is not None or args.skin_hi is not None:
        lo = args.skin_lo if args.skin_lo is not None else SKIN_BAND_C[0]
        hi = args.skin_hi if args.skin_hi is not None else SKIN_BAND_C[1]
        SKIN_BAND_C = (lo, hi)
        SKIN_BAND_FIXED = True

    if args.list:
        for i, w, h, dt, lep in scan_devices():
            print(f"  [{i}] {w}x{h} {dt}" + ("   <-- looks like Lepton" if lep else ""))
        return

    # Preferred path: libuvc (works where OpenCV/AVFoundation cannot, and is
    # the only way to get radiometric Y16 on macOS).
    cam = None
    if not args.opencv:
        try:
            from lepton_libuvc import LeptonUVC
            cam = LeptonUVC()
            print(f"  mode: RADIOMETRIC via libuvc ({cam.format_used})")
        except Exception as e:
            print(f"  libuvc unavailable: {e}")
            print("  falling back to OpenCV capture ...")

    if cam is None:
        dev = args.device
        if dev is None:
            cands = scan_devices()
            lep = [c for c in cands if c[4]]
            if not lep:
                print("No Lepton-like device found. Seen:")
                for i, w, h, dt, _ in cands:
                    print(f"  [{i}] {w}x{h} {dt}")
                print("Run diagnose.py / backend_test.py, then pass --device N")
                return
            dev = lep[0][0]
        print(f"Opening device {dev} via OpenCV ...")
        cam = ThermalCamera(dev)

    sens = (args.delta if args.delta is not None else DEFAULT_DELTA_C) if cam.radiometric \
        else (args.pctl if args.pctl is not None else DEFAULT_PCTL)

    os.makedirs(LOG_DIR, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LOG_DIR, f"session_{session}.csv")
    fcsv = open(csv_path, "w", newline="")
    wr = csv.writer(fcsv)
    wr.writerow(["timestamp", "frame", "mode", "view", "n_det", "background",
                 "threshold", "cx", "cy", "area_px", "val_max", "aspect",
                 "extent", "peaks", "std_c", "skin_px", "priority", "note"])

    flags = dict(DEFAULT_FLAGS)
    flags["pfilter"] = args.p_filter
    flags["extend"] = args.body_extend
    flags["kalman"] = args.kalman
    flags["static"] = not args.no_static
    tracker = MultiTracker() if MultiTracker else None

    # cursor probe state, filled by the OpenCV mouse callback
    gnd_cal = []
    probe_on = args.probe
    probe_xy = {"x": None, "y": None}

    def _on_mouse(event, mxx, myy, fl, param):
        probe_xy["x"], probe_xy["y"] = mxx, myy

    WIN = "FLUXNET thermal baseline"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, _on_mouse)

    logging_on = False
    # Continuous capture: one keypress starts it, frames stream to disk with a
    # manifest. Building a few thousand training frames by pressing 's' is not
    # realistic; recording a walk-through and letting it run is.
    capture_on = False
    capture_n = 0
    capture_dir = None
    capture_manifest = None
    capture_writer = None
    show_boxes = not args.omega_only
    show_omega = True

    suppressor = StaticSuppressor()
    cohesion = args.cohesion       # anti-fragmentation strength 0..4
    view_mode = args.view          # 'vertical' | 'horizontal' | 'any'
    view = 0                       # display style: color / mask / blended
    frame_i = 0
    fps = 0.0
    t_last = time.time()
    print(HELP)

    while True:
        data, is_temp = cam.read()
        if data is None:
            print("Frame read failed; stopping.")
            break
        frame_i += 1

        eff_sens = sens
        if is_temp and args.delta is None:
            eff_sens = SHAPE_RULES.get(view_mode, {}).get("delta", DEFAULT_DELTA_C)
        thr, bg = compute_threshold(data, is_temp, eff_sens)
        suppressor.update(data)
        # Horizontal sees people at range, where a whole body spans a handful
        # of pixels and a seated or standing person moves almost nothing in
        # image space. Static suppression absorbs them into the background,
        # which is precisely the mid- and far-field case this view exists for.
        suppressor.enabled = flags["static"] and (view_mode != "horizontal"
                                                  or args.h_static)
        moving = suppressor.moving_mask(data) if is_temp else None
        dets, mask = detect_people(
            data, thr,
            # the absolute band only means anything on real temperatures
            tmax=(args.tmax if is_temp else None),
            tmin=(args.tmin if is_temp else None),
            view=view_mode,
            min_area=args.min_area,
            split=not args.no_split,
            min_sep=args.min_sep,
            cohesion=cohesion,
            min_peaks=args.min_peaks,
            peak_check_area=args.peak_check_area,
            moving=moving,
            equip_thresh=args.equip_thresh,
            p_filter=flags['pfilter'],
            p_min=args.p_min,
            body_extend=flags['extend'],
            body_delta=args.body_delta,
            flags=flags,
            omega_scale_ratio=args.omega_scale_ratio,
            omega_delta=args.omega_delta,
            gnd_horizon=args.gnd_horizon, gnd_gain=args.gnd_gain,
            gnd_tol=args.gnd_tol, vgrad_max=args.vgrad_max,
            omega_max_tilt=args.omega_max_tilt,
            far_delta=args.far_delta, far_min_area=args.far_min_area,
            far_max_area=args.far_max_area, far_contrast=args.far_contrast,
            far_row_max=args.far_row_max,
            gfar_ref_row=args.gfar_ref_row, gfar_ref_h=args.gfar_ref_h,
            gfar_ref_range=args.gfar_ref_range, gfar_horizon=args.gfar_horizon,
            gfar_lo=args.gfar_lo, gfar_hi=args.gfar_hi,
            gfar_delta=args.gfar_delta, gfar_max_range=args.gfar_max_range,
            gfar_contrast_near=args.gfar_contrast_near,
            gfar_contrast_far=args.gfar_contrast_far,
            gfar_structure_h=args.gfar_structure_h,
        )

        if not (flags["kalman"] and tracker is not None):
            dets = [d for d in dets if not d.get("p_reject")]

        # ---- Kalman tracking -------------------------------------------
        # A person persists; clutter flickers. Requiring a track to survive
        # several frames rejects transients no single frame can catch, and
        # coasting through the Lepton's FFC pause stops one person being
        # re-acquired as a new one.
        tracks = []
        if flags["kalman"] and tracker is not None:
            tracker.update(dets)
            tracks = tracker.confirmed()
            dets = [dict(t.det, bbox=t.bbox, centroid=t.centroid,
                         track_id=t.id, track_hits=t.hits, speed=t.speed,
                         rigidity=t.rigidity(min_frames=args.rigid_frames))
                    for t in tracks]
            # NON-RIGID GATE, in two directions.
            #
            # Demonstrated articulation is positive evidence of a body, so it
            # pays a BONUS onto the peak count. This is what rescues a heavily
            # clothed person: the coat hides the warm centres the p-filter
            # wants, but it cannot hide the fact that the limbs swing. Motion
            # substitutes for the thermal structure the clothing removed.
            #
            # Measured rigidity is negative evidence and still rejects.
            # rigidity() returns None until there is enough history — treated
            # as UNKNOWN, never as rigid, so nothing is dropped or credited on
            # insufficient evidence.
            if flags["gait"]:
                for t_ in tracks:
                    g = t_.gait(fps=max(1.0, fps))
                    for d in dets:
                        if d.get("track_id") == t_.id:
                            d["gait_hz"] = None if g is None else g[0]
                            d["gait_frac"] = None if g is None else g[1]
                dets = [d for d in dets
                        if d.get("gait_frac") is None
                        or d["gait_frac"] >= args.gait_frac]

            if flags["rigid"]:
                for d in dets:
                    r = d.get("rigidity")
                    bonus = args.rigid_bonus if (r is not None and
                                                 r >= args.rigid_thresh) else 0
                    d["peaks_bonus"] = bonus
                    d["peaks_eff"] = d.get("peaks", 0) + bonus
                dets = [d for d in dets
                        if (d.get("rigidity") is None
                            or d["rigidity"] >= args.rigid_thresh)
                        and (not d.get("p_reject")
                             or d.get("peaks_eff", 0) > args.p_min)]
            else:
                dets = [d for d in dets if not d.get("p_reject")]

        now = time.time()
        dt = now - t_last
        t_last = now
        if dt > 0:
            fps = (0.9 * fps + 0.1 / dt) if fps else 1.0 / dt

        if logging_on:
            ts = datetime.now().isoformat(timespec="milliseconds")
            mode = "RAD" if is_temp else "AGC"
            if dets:
                for d in dets:
                    wr.writerow([ts, frame_i, mode, view_mode, len(dets),
                                 f"{bg:.2f}", f"{thr:.2f}",
                                 f"{d['centroid'][0]:.1f}", f"{d['centroid'][1]:.1f}",
                                 d["area_px"], f"{d['val_max']:.2f}",
                                 f"{d.get('aspect', 0):.2f}", f"{d.get('extent', 0):.2f}",
                                 d.get('peaks', ''), f"{d.get('std_c', 0):.2f}",
                                 d.get('skin_px', 0), int(d.get('priority', False)),
                                 args.note])
            else:
                wr.writerow([ts, frame_i, mode, view_mode, 0, f"{bg:.2f}",
                             f"{thr:.2f}", "", "", "", "", "", "", "", "", "", "", args.note])

        color = colorize(data)
        maskc = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # ---- continuous capture ------------------------------------------
        if capture_on and frame_i % max(1, args.capture_every) == 0:
            fn = f"cap_{capture_n:06d}"
            np.save(os.path.join(capture_dir, "npy", fn + ".npy"), data)

            # PNG for the learned branch — CLEAN. No boxes, no text, no
            # sidebar, no omega arc, no upscaling. Detection training takes a
            # clean image plus a separate label file; a box drawn into the
            # pixels teaches the network to find rectangles and cannot be read
            # by any standard pipeline.
            if args.capture_png_overlay:
                png = vis
            elif args.capture_png_raw:
                # 16-bit linear over a FIXED span, so a pixel value means the
                # same temperature across the whole dataset. The live palette is
                # percentile-stretched per frame, which is exactly the nuisance
                # variation a model must not be taught.
                lo, hi = PNG_SPAN_C
                png = (np.clip((data - lo) / (hi - lo), 0, 1) * 65535).astype(np.uint16)
            else:
                png = colorize(data)
            cv2.imwrite(os.path.join(capture_dir, "png", fn + ".png"), png)

            # YOLO labels: "<class> <cx> <cy> <w> <h>", all normalised 0-1.
            #   class 0 = person        (full body blob)
            #   class 1 = head_shoulder (the omega region, if one was found)
            # Two classes rather than one because the omega box is a genuinely
            # different target from the body box, and which one a model should
            # learn is an open question worth keeping both options for.
            IH, IW = data.shape[:2]
            lines = []
            for de in dets:
                bx, by, bw, bh = de["bbox"]
                lines.append(f"0 {(bx + bw / 2) / IW:.6f} {(by + bh / 2) / IH:.6f} "
                             f"{bw / IW:.6f} {bh / IH:.6f}")
                for om in de.get("omegas", []):
                    hb = om.get("head_box")
                    if not hb:
                        continue
                    hx, hy, hw_, hh_ = hb
                    lines.append(f"1 {(hx + hw_ / 2) / IW:.6f} {(hy + hh_ / 2) / IH:.6f} "
                                 f"{hw_ / IW:.6f} {hh_ / IH:.6f}")
            with open(os.path.join(capture_dir, "labels", fn + ".txt"), "w") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))

            # QA copy with the boxes drawn, kept OUT of png/ so it can never be
            # fed to a trainer by accident. Flip through this folder, delete the
            # frames that are wrong, then run prune.py to drop the matching
            # npy/png/labels files.
            if not args.no_review:
                R = 4
                rv = cv2.resize(colorize(data), (IW * R, IH * R),
                                interpolation=cv2.INTER_NEAREST)
                for de in dets:
                    bx, by, bw, bh = [q * R for q in de["bbox"]]
                    cv2.rectangle(rv, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.putText(rv, f"{de.get('peaks', 0)}p", (bx, max(12, by - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(rv, f"{de.get('peaks', 0)}p", (bx, max(12, by - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                    for om in de.get("omegas", []):
                        hb = om.get("head_box")
                        if hb:
                            hx, hy, hw_, hh_ = [q * R for q in hb]
                            cv2.rectangle(rv, (hx, hy), (hx + hw_, hy + hh_),
                                          (0, 220, 255), 1)
                # stamp the frame id so a deleted file is always traceable back
                cv2.rectangle(rv, (0, 0), (rv.shape[1], 20), (14, 14, 17), -1)
                cv2.putText(rv, f"{fn}    n={len(dets)}", (6, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 220), 1,
                            cv2.LINE_AA)
                cv2.imwrite(os.path.join(capture_dir, "review", fn + ".png"), rv)
            if capture_writer:
                capture_writer.writerow([
                    fn, datetime.now().isoformat(timespec="milliseconds"),
                    view_mode, f"{bg:.2f}", f"{thr:.2f}", len(dets),
                    sum(d.get("omega_count", 0) for d in dets),
                    max([d.get("omega_score", 0.0) for d in dets], default=0.0),
                    "+".join(FLAG_LABEL[n] for n in flags if flags[n]),
                    args.note,
                ])
            capture_n += 1

        base = color if view == 0 else (maskc if view == 1 else
                                        cv2.addWeighted(color, 0.7, maskc, 0.3, 0))
        # Upscale FIRST, then annotate: every glyph is drawn at display
        # resolution instead of being magnified 5x with nearest-neighbour.
        S = DISPLAY_SCALE
        frame = cv2.resize(base, (base.shape[1] * S, base.shape[0] * S),
                           interpolation=cv2.INTER_NEAREST)
        frame = draw_boxes(frame, dets, is_temp, S, show_boxes,
                           show_ids=flags["kalman"])
        if show_omega:
            frame = draw_omega(frame, dets, S)
        if probe_on:
            frame = draw_probe(frame, data, probe_xy["x"], probe_xy["y"], S, is_temp)

        sb = draw_sidebar(frame.shape[0], dets, bg, thr, eff_sens, is_temp, fps,
                          logging_on, view_mode, cohesion,
                          args.tmin if is_temp else None,
                          args.tmax if is_temp else None,
                          args.p_min, flags, show_boxes, show_omega,
                          capture_on, capture_n, flags["kalman"],
                          probe_on, probe_xy, data, is_temp)
        # The panel may be taller than the frame; pad the frame rather than
        # truncate the panel.
        H = max(frame.shape[0], sb.shape[0])
        if frame.shape[0] < H:
            frame = np.vstack([frame, np.full((H - frame.shape[0],
                                               frame.shape[1], 3), 12, np.uint8)])
        if sb.shape[0] < H:
            sb = np.vstack([sb, np.full((H - sb.shape[0], SIDEBAR_W, 3),
                                        18, np.uint8)])
        vis = np.hstack([frame, sb])
        if capture_on:
            cv2.circle(vis, (frame.shape[1] - 18, 18), 7, (0, 0, 255), -1)
        cv2.imshow(WIN, vis)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("s"):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            np.save(os.path.join(LOG_DIR, f"frame_{stamp}.npy"), data)
            cv2.imwrite(os.path.join(LOG_DIR, f"frame_{stamp}.png"), vis)
            print(f"  saved frame_{stamp}  (n={len(dets)}, thr={thr:.1f})")
        elif k == ord("l"):
            logging_on = not logging_on
            print(f"  logging {'ON' if logging_on else 'OFF'} -> {csv_path}")
        elif k == ord("["):
            sens = max(0.5, sens - 0.5) if is_temp else min(99.5, sens + 0.5)
            print(f"  sensitivity: {sens}")
        elif k == ord("]"):
            sens = sens + 0.5 if is_temp else max(50.0, sens - 0.5)
            print(f"  sensitivity: {sens}")
        elif k == ord("a"):
            print(f"  background = {bg:.2f}, threshold = {thr:.2f}")
        elif k == ord("r"):
            view = (view + 1) % 3
        elif k == ord("v"):
            view_mode = VIEW_MODES[(VIEW_MODES.index(view_mode) + 1) % len(VIEW_MODES)]
            print(f"  mounting mode -> {SHAPE_RULES[view_mode]['label']}")
        elif k == ord("m"):
            cohesion = min(4, cohesion + 1)
            print(f"  cohesion -> {cohesion} (less fragmentation: "
                  f"close={5+2*cohesion}px, gap={6+3*cohesion}px, seed={3+cohesion}px)")
            if cohesion >= 3:
                print("  ** note: cohesion >=3 merges two people standing close "
                      "into one detection.")
                print("     Correct for SINGLE-occupant vertical mode (4 is the "
                      "bench-measured best); avoid if two people may be present.")
        elif k == ord("n"):
            cohesion = max(0, cohesion - 1)
            print(f"  cohesion -> {cohesion} (more separation: "
                  f"close={5+2*cohesion}px, gap={6+3*cohesion}px, seed={3+cohesion}px)")
        elif k in (ord("1"), ord("2"), ord("3")):
            view_mode = VIEW_MODES[k - ord("1")]
            print(f"  mounting mode -> {SHAPE_RULES[view_mode]['label']}")
        elif k == ord("c"):
            capture_on = not capture_on
            if capture_on:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                capture_dir = os.path.join(LOG_DIR, f"capture_{stamp}")
                for sub in ("npy", "png", "labels", "review"):
                    os.makedirs(os.path.join(capture_dir, sub), exist_ok=True)
                os.makedirs(os.path.join(capture_dir, "review"), exist_ok=True)
                with open(os.path.join(capture_dir, "classes.txt"), "w") as fh:
                    fh.write("person\nhead_shoulder\n")
                capture_manifest = open(os.path.join(capture_dir, "manifest.csv"),
                                        "w", newline="")
                capture_writer = csv.writer(capture_manifest)
                capture_writer.writerow(["file", "timestamp", "view", "background",
                                         "threshold", "n_det", "omega_count",
                                         "omega_score", "flags", "note"])
                capture_n = 0
                print(f"  RECORDING -> {capture_dir}  (press c again to stop)")
            else:
                if capture_manifest:
                    capture_manifest.close()
                print(f"  recording stopped: {capture_n} frames in {capture_dir}")
        elif k == ord("o"):
            show_boxes = not show_boxes
            print(f"  body boxes {'ON' if show_boxes else 'OFF'}")
        elif k == ord("C"):
            # Ground-plane calibration: collect (y_feet, height) pairs and fit
            # the gain, so the prior is measured on YOUR mount rather than
            # guessed from trigonometry.
            for d in dets:
                x, y, w, h = d["bbox"]
                gnd_cal.append((y + h, h))
            if len(gnd_cal) >= 4:
                a = np.array(gnd_cal, float)
                gain = float(np.sum(a[:, 0] * a[:, 1]) / max(1e-6, np.sum(a[:, 0] ** 2)))
                print(f"  ground-plane: {len(gnd_cal)} samples -> "
                      f"--gnd-gain {gain:.4f}   (horizon assumed row 0)")
            else:
                print(f"  ground-plane: {len(gnd_cal)} samples "
                      f"(need 4+, walk to several distances then press C)")
        elif k == ord("T"):
            probe_on = not probe_on
            print(f"  cursor probe {'ON' if probe_on else 'OFF'}")
        elif k == ord("O"):
            show_omega = not show_omega
            print(f"  omega overlay {'ON' if show_omega else 'OFF'}")
        elif 0 < k < 256 and chr(k) in FLAG_KEYS:
            name = FLAG_KEYS[chr(k)]
            flags[name] = not flags[name]
            if name == "kalman" and tracker is not None:
                tracker.reset()
            print(f"  {FLAG_LABEL[name]:<10} {'ON ' if flags[name] else 'OFF'}"
                  f"   [{' '.join(FLAG_LABEL[n] for n in flags if flags[n])}]")
        elif k == ord("b"):
            suppressor.reset()
            print("  static background reset — relearning")
        elif k == ord("h"):
            print(HELP)

    if capture_manifest:
        capture_manifest.close()
        print(f"capture: {capture_n} frames in {capture_dir}")
    fcsv.close()
    cam.release()
    cv2.destroyAllWindows()
    print(f"\nSession log: {csv_path}")


if __name__ == "__main__":
    main()
