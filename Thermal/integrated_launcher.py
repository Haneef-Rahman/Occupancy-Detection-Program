#!/usr/bin/env python3
"""
YOLO to find people, Kalman + classical blobs to follow them, YOLO again on doubt.

The CNN is the accurate detector and the expensive one: ~50 ms per frame on this
Mac, against ~3 ms for a classical blob scan. Running it on every frame spends
almost all the budget re-answering a question that has not changed — the person
in the left of frame is still the same person one frame later. So it runs when
identity is genuinely in question, and cheap machinery covers the rest.

THREE STATES PER FRAME:

    1. classical blob scan          always, ~3 ms
    2. Kalman predict + associate   always, microseconds
    3. YOLO inference               only on a TRIGGER

TRIGGERS, and why each one exists:

    unmatched blob   a warm blob that no track claims. Somebody walked in, or a
                     track drifted off its subject. Either way identity is
                     unknown and only the CNN can settle it. This is what stops
                     a re-detect-on-loss-only design from being blind to new
                     arrivals: an entering person shows up as an unclaimed blob
                     long before any existing track dies.
    track lost       a confirmed track exceeded its miss budget. The person did
                     not necessarily leave — they may have merged with someone,
                     or gone briefly cold to the threshold.
    stale            a track has coasted `--max-stale` frames without a fresh
                     CNN confirmation. Bounds how long a wrong identity can
                     survive on Kalman momentum alone.

WHY THE CLASSICAL PATH IS THE CHEAP ONE HERE. It is not a second opinion — its
job is to answer "did anything move that nobody is tracking?", which is a much
easier question than "is that a person?". So it runs with the horizontal-tuned
rules, which are the most sensitive to small distant blobs, and its false
positives are harmless: a spurious blob just triggers a CNN call that then
declines to confirm it.

TWO MODES:

    --mode hybrid   (default)  classical blobs carry the cheap frames, the CNN
                    fires on the triggers above. ~25% CNN duty measured on real
                    captures. Kalman both smooths AND substitutes for detection.

    --mode yolo     CNN omega on EVERY frame; the classical path is not run at
                    all. Kalman does identity and smoothing only, never
                    detection. This is the accuracy ceiling — every box comes
                    from a fresh inference at conf 0.374 — and the honest
                    baseline to judge hybrid against.

ONLY class 1 (omega) ever becomes a track, in both modes. Class 0 (person) is
read for its height, which feeds the range model, and is not drawn unless you
ask with --show-person. The box drawn for a track defaults to whatever that
mode actually measured: omega in yolo mode, full body in hybrid.

    ./run.sh integrated_launcher.py --weights models/v2/best.pt
    ./run.sh integrated_launcher.py --weights models/v2/best.pt --mode yolo

Keys
    q / ESC   quit          space  pause        h  hide sidebar
    s         save frame    l      log to CSV
    y         force a YOLO frame now
    b         classical blob overlay (what triggers a re-detect)
    t         track trails
    p         cycle box: body / omega / both
    o         show/hide the CNN person boxes
    d         dedup on/off
    r         reset all track identities (do this after changing a parameter)

    TAB       select a parameter
    [ ]       decrease / increase it

Live parameters: conf, gate px, dup confirm, reid px, coast trigger, merge m,
dedup iou. Tuning while watching beats guessing from a table — and `r` clears
the ids so the count you judge is not inherited from the previous setting.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

import thermal_detect as TD
from tracker import MultiTracker, KalmanTrack


SPAN_C = (15.0, 45.0)      # MUST match make_dataset.py — the model's encoding
SIDEBAR_W = 340

# One palette, named, so a colour means the same thing everywhere. BGR.
C_BG      = (18, 17, 16)
C_PANEL   = (30, 28, 26)
C_RULE    = (58, 54, 50)
C_TEXT    = (232, 230, 226)
C_DIM     = (140, 136, 130)
C_FAINT   = (96, 93, 89)
C_OK      = (120, 220, 130)
C_WARN    = (70, 190, 255)
C_BAD     = (90, 95, 250)
C_ACCENT  = (255, 214, 130)
F = cv2.FONT_HERSHEY_SIMPLEX

TRACK_COLS = [(120, 240, 150), (90, 220, 255), (255, 180, 110), (215, 150, 255),
              (150, 210, 255), (170, 255, 215), (255, 225, 150)]


# ---------------------------------------------------------------------------
# drawing helpers
#
# HERSHEY fonts are PROPORTIONAL, so the obvious f"{name:<12}{value:>7}" does
# not align anything — the columns wander by several pixels per row and the
# panel reads as sloppy. Every value here is positioned by measuring its own
# width and subtracting from the right edge, which is the only way to get a
# straight column out of a proportional font.
# ---------------------------------------------------------------------------

def _tw(txt, sc, th=1):
    return cv2.getTextSize(txt, F, sc, th)[0][0]


def row(img, y, label, value=None, col=None, sc=0.44, vcol=None, pad=14):
    cv2.putText(img, label, (pad, y), F, sc, col or C_DIM, 1, cv2.LINE_AA)
    if value is not None:
        v = str(value)
        cv2.putText(img, v, (img.shape[1] - pad - _tw(v, sc), y), F, sc,
                    vcol or C_TEXT, 1, cv2.LINE_AA)
    return y


def head(img, y, txt, pad=14):
    """Small caps section label with a hairline under it."""
    cv2.putText(img, txt, (pad, y), F, 0.38, C_FAINT, 1, cv2.LINE_AA)
    cv2.line(img, (pad, y + 6), (img.shape[1] - pad, y + 6), C_RULE, 1)
    return y


def chip(img, x, y, txt, col, sc=0.4):
    """Filled pill — reads as a state, not as another line of text."""
    w = _tw(txt, sc) + 16
    cv2.rectangle(img, (x, y - 13), (x + w, y + 6), col, -1)
    cv2.putText(img, txt, (x + 8, y + 1), F, sc, C_BG, 1, cv2.LINE_AA)
    return x + w + 7


def meter(img, x, y, w, frac, col, h=5):
    """Thin progress bar. A number tells you the value; a bar tells you where
    it sits in its range, which is what you actually want at a glance."""
    cv2.rectangle(img, (x, y), (x + w, y + h), C_PANEL, -1)
    f = int(w * max(0.0, min(1.0, frac)))
    if f > 0:
        cv2.rectangle(img, (x, y), (x + f, y + h), col, -1)


def spark(img, x, y, w, h, series, col, vmax=None):
    """Occupancy over the last N frames — shows flicker that a single number
    hides. A count alternating 1,2,1,2 and a steady 2 look identical live."""
    cv2.rectangle(img, (x, y), (x + w, y + h), C_PANEL, -1)
    if len(series) < 2:
        return
    m = max(1, vmax or max(series))
    n = len(series)
    pts = [(x + int(w * i / (n - 1)), y + h - int((h - 2) * v / m))
           for i, v in enumerate(series)]
    cv2.polylines(img, [np.array(pts, np.int32)], False, col, 1, cv2.LINE_AA)


def corners(img, p0, p1, col, th=2, frac=0.28):
    """
    Corner ticks instead of a closed rectangle.

    The thermal image is 160x120 upscaled; a person can be 15 px tall, so a
    full rectangle drawn around them covers much of the evidence you are trying
    to look at. Ticks mark the same extent while leaving the subject visible.
    """
    x0, y0 = p0
    x1, y1 = p1
    lx = max(3, int((x1 - x0) * frac))
    ly = max(3, int((y1 - y0) * frac))
    for (ax, ay, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                             (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (ax, ay), (ax + dx * lx, ay), col, th, cv2.LINE_AA)
        cv2.line(img, (ax, ay), (ax, ay + dy * ly), col, th, cv2.LINE_AA)


# Parameters adjustable while running. live_yolo.py had [ and ] for confidence
# alone; here there are six things worth turning, so [ and ] act on whichever
# one TAB has selected rather than each getting its own key.
#
#   name        attribute on args        step   min    max    fmt
TUNABLES = [
    ("conf",          "conf",           0.02,  0.05,  0.95,  "{:.2f}"),
    ("gate px",       "gate",           2.0,   6.0,   80.0,  "{:.0f}"),
    ("dup confirm",   "dup_confirm",    1,     1,     10,    "{:d}"),
    ("reid px",       "reid_px",        5.0,   0.0,   80.0,  "{:.0f}"),
    ("coast trig",    "coast_trigger",  1,     1,     8,     "{:d}"),
    ("merge m",       "merge_dist",     0.25,  0.0,   5.0,   "{:.2f}"),
    ("dedup iou",     "dedup_iou",      0.05,  0.0,   0.9,   "{:.2f}"),
]


def bump(args, idx, direction):
    """Nudge the selected tunable and return a message for the console."""
    name, attr, step, lo, hi, fmt = TUNABLES[idx]
    cur = getattr(args, attr)
    new = cur + direction * step
    new = max(lo, min(hi, type(step)(new) if isinstance(step, int) else new))
    if isinstance(step, int):
        new = int(round(new))
    setattr(args, attr, new)
    return f"{name} = {fmt.format(new)}"


def render_for_cnn(data):
    lo, hi = SPAN_C
    v = np.clip((data.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    g = (v * 255.0).astype(np.uint8)
    return cv2.merge([g, g, g])


def open_camera(args):
    if not args.opencv:
        try:
            from lepton_libuvc import LeptonUVC
            print("  libuvc: radiometric Y16")
            return LeptonUVC()
        except Exception as e:
            print(f"  libuvc unavailable: {e}\n  falling back to OpenCV ...")
    return TD.ThermalCamera(args.device if args.device is not None else 0)


# Fraction of a standing body's height occupied by head-and-shoulders, and the
# height of that region's centre above the body top. Measured against the v2
# labels, where every person carries both boxes.
OMEGA_H_FRAC = 0.30
OMEGA_C_FRAC = 0.16


def yolo_omegas(model, data, conf, imgsz):
    """
    OMEGA is the tracked primitive; person boxes are auxiliary.

    Omega localises slightly worse in isolation (mAP50-95 0.706 vs 0.831 for
    person), so this is not the accuracy-optimal choice frame by frame. It is
    the right choice for TRACKING, because of what happens when people get
    close: two bodies at 0.5 m share one warm silhouette and the classical
    scan sees a single merged blob, but their heads remain distinct. A
    primitive that survives crowding beats one that localises better in the
    easy case, since the easy case is handled by Kalman anyway.
    """
    res = model.predict(render_for_cnn(data), verbose=False,
                        conf=conf, imgsz=imgsz)[0]
    dets, bodies = [], []
    if len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        for (x0, y0, x1, y1), c, v in zip(xyxy, cls, cf):
            w, h = float(x1 - x0), float(y1 - y0)
            box = (float(x0), float(y0), w, h)
            if c == 1:                      # omega -> a track
                dets.append({"bbox": box,
                             "centroid": (float(x0) + w / 2, float(y0) + h / 2),
                             "area_px": w * h, "conf": float(v), "src": "cnn",
                             "body_h": h / OMEGA_H_FRAC})
            else:                           # person -> display + range only
                bodies.append((box, float(v)))
    return dets, bodies


def dedup(dets, iou_thresh=0.15, centre_frac=1.4):
    """
    Collapse multiple omega boxes that describe the same head.

    Ultralytics runs NMS at IoU 0.7 by default, which is deliberately permissive
    so that genuinely adjacent objects both survive. On a 160x120 thermal frame
    an omega is 4-15 px across, and two boxes on the same head routinely overlap
    at 0.4-0.6 — under the NMS threshold, so both come through. The val_batch
    previews show this directly: labels reading "omeomega0.8" are two boxes
    stacked on one person.

    Downstream that is not a cosmetic problem. One box matches the existing
    track, the other is unmatched, and an unmatched detection creates a NEW
    track with a NEW id and a new colour. Next frame the duplicate lands
    elsewhere and the ids swap. That is the colour churn.

    Two boxes are merged if they overlap past `iou_thresh` OR if one's centre
    sits within `centre_frac` of the other's half-size — the second test catches
    a small box nested inside a larger one, where IoU stays low but they are
    plainly the same head.

    Thresholds swept on a simulated walk with 30% duplicate boxes. 0.35/0.6
    (the obvious first guess) left 8 distinct ids over 300 frames; 0.15/1.4
    leaves 4. Going further to 0.10 gives 3, but starts merging detections only
    8 px apart — 0.49 m at 2 m range, which is two people standing shoulder to
    shoulder, so that is where the tuning stops. At 0.15/1.4 the closest pair
    that survives as two is 6 px, or 0.37 m: tighter than people stand.
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i].get("conf", 0.0))
    keep = []
    for i in order:
        a = dets[i]
        ax, ay, aw, ah = a["bbox"]
        dup = False
        for j in keep:
            b = dets[j]
            bx, by, bw, bh = b["bbox"]
            ix0, iy0 = max(ax, bx), max(ay, by)
            ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            union = aw * ah + bw * bh - inter
            if union > 0 and inter / union >= iou_thresh:
                dup = True
                break
            acx, acy = a["centroid"]
            bcx, bcy = b["centroid"]
            if (abs(acx - bcx) <= centre_frac * max(aw, bw) / 2 and
                    abs(acy - bcy) <= centre_frac * max(ah, bh) / 2):
                dup = True
                break
        if not dup:
            keep.append(i)
    return [dets[i] for i in sorted(keep)]


class StickyTracker(MultiTracker):
    """
    MultiTracker plus a second association pass for coasting tracks.

    The base class matches every detection against every track with one gate,
    then spawns a new track for whatever is left. That is fine while detections
    are continuous, and wrong the moment one drops: the track starts coasting
    and drifts under constant velocity, so when the detection returns it can sit
    outside the gate. A new track is created, the person changes id and colour,
    and the old track lingers for max_misses frames as a ghost.

    So: match live tracks first with the normal gate, then give still-unmatched
    detections a second chance against COASTING tracks with a gate widened in
    proportion to how long they have coasted — a track that has predicted
    forward for 5 frames has 5 frames of accumulated uncertainty and deserves a
    correspondingly wider window. Only what survives both passes is new.
    """

    def __init__(self, *a, revive_scale=1.8, ghost_frames=40,
                 dup_confirm=3, reid_px=0.0, **kw):
        super().__init__(*a, **kw)
        self.revive_scale = revive_scale
        self.n_new = 0
        self.n_revived = 0
        self.n_suppressed = 0
        self.n_reid = 0
        # DIAGNOSTIC ONLY. Dead tracks are remembered but never re-attached —
        # the point is to find out how often a "new" person is really a
        # returning one, before deciding whether re-identification is worth
        # building. Turning this into a fix would be a different change.
        self.ghost_frames = ghost_frames
        self.dup_confirm = dup_confirm
        self.reid_px = reid_px
        self.ghosts = []          # (id, x, y, w, h, frame_died, hits)
        self.pending = []         # provisional births: [x, y, count, frame]
        self.events = []          # rows for track_events.csv
        self.frame = 0

    def _classify(self, d, live_before):
        """
        Why did this detection not belong to an existing track?

        Distinguishes the three failure modes that need different fixes:
          gate_miss   a live track was near but outside the gate -> gate/cost
          contested   two or more tracks were within reach -> greedy stole it
          returning   a track died near here recently -> needs re-identification
          genuine     nothing nearby, alive or dead -> a real new person
        """
        dx, dy = d["centroid"]
        near = sorted(((((t.centroid[0] - dx) ** 2 +
                         (t.centroid[1] - dy) ** 2) ** 0.5), t)
                      for t in live_before)
        g_near = sorted((((gx - dx) ** 2 + (gy - dy) ** 2) ** 0.5,
                         gid, self.frame - fd, gh)
                        for gid, gx, gy, gw, gh_, fd, gh in self.ghosts)
        nd, nt = (near[0][0], near[0][1]) if near else (None, None)
        gd, gid, gage, ghits = (g_near[0] if g_near else (None, None, None, None))
        contested = sum(1 for dist, _ in near if dist <= self.gate_px * 2)

        # ORDER MATTERS. A track sitting 7 px away with misses==0 was matched
        # THIS frame — it was never out of reach, it was already taken, and the
        # extra detection is a duplicate the deduper missed. Calling that
        # "gate_miss" sent me looking at the gate, which was not the problem.
        if nt is not None and nd <= self.gate_px and nt.misses == 0:
            cause = "duplicate"
        elif gd is not None and gd <= self.gate_px * 2:
            cause = "returning"
        elif nd is not None and nd <= self.gate_px * 3:
            cause = "contested" if contested >= 2 else "gate_miss"
        else:
            cause = "genuine"
        return {"cause": cause,
                "near_track": nt.id if nt else "",
                "near_dist": f"{nd:.1f}" if nd is not None else "",
                "near_misses": nt.misses if nt else "",
                "ghost_id": gid if gid is not None else "",
                "ghost_dist": f"{gd:.1f}" if gd is not None else "",
                "ghost_age": gage if gage is not None else "",
                "n_within_2gate": contested}

    def update(self, dets):
        self.frame += 1
        live_before = [t for t in self.tracks if t.hits >= self.min_hits]
        snap = {t.id: (t.centroid, t.misses) for t in self.tracks}
        for t in self.tracks:
            t.predict()

        def associate(tracks, det_idx, gate_for):
            pairs = []
            for t in tracks:
                tx, ty = t.centroid
                for di in det_idx:
                    dx, dy = dets[di]["centroid"]
                    d = ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5
                    if d <= gate_for(t):
                        pairs.append((d, id(t), di, t))
            pairs.sort(key=lambda p: p[0])
            used_t, used_d = set(), set()
            for d, tid, di, t in pairs:
                if tid in used_t or di in used_d:
                    continue
                t.update(dets[di])
                used_t.add(tid)
                used_d.add(di)
            return used_d

        left = set(range(len(dets)))
        live = [t for t in self.tracks if t.misses <= 1]
        left -= associate(live, list(left), lambda t: self.gate_px)

        coasting = [t for t in self.tracks if t.misses > 1]
        if left and coasting:
            before = len(left)
            left -= associate(
                coasting, list(left),
                lambda t: self.gate_px * min(self.revive_scale, 1.0 + 0.18 * t.misses))
            self.n_revived += before - len(left)

        for i in sorted(left):
            info = self._classify(dets[i], live_before)

            # FIX A — provisional birth.
            # A detection landing inside the gate of a track that was already
            # matched this frame is, on the evidence, a duplicate box on one
            # head: measured at 7.2 px from a matched track, alive 10 frames,
            # 2 hits. But it could also be a second person walking up to the
            # first, and those look identical for one frame. So neither trust
            # nor discard it — require it to persist. A duplicate flickers and
            # never reaches the count; a real person standing there does.
            if info["cause"] == "duplicate" and self.dup_confirm > 1:
                x, y = dets[i]["centroid"]
                hit = None
                for pnd in self.pending:
                    if abs(pnd[0] - x) <= self.gate_px * 0.5 and \
                       abs(pnd[1] - y) <= self.gate_px * 0.5 and \
                       self.frame - pnd[3] <= 2:
                        hit = pnd
                        break
                if hit is None:
                    self.pending.append([x, y, 1, self.frame])
                    self.n_suppressed += 1
                    continue
                hit[0], hit[1], hit[3] = x, y, self.frame
                hit[2] += 1
                if hit[2] < self.dup_confirm:
                    self.n_suppressed += 1
                    continue
                self.pending.remove(hit)
                info["cause"] = "duplicate_promoted"

            t = KalmanTrack(dets[i])

            # FIX B — re-identification.
            # Measured: a track died, the person was gone 33 frames, came back
            # 3.8 px from where it died, and was issued a new id. For occupancy
            # counting that is not cosmetic — it reads as a departure plus an
            # arrival when nobody entered or left.
            if self.reid_px > 0:
                x, y = dets[i]["centroid"]
                cand = [(((gx - x) ** 2 + (gy - y) ** 2) ** 0.5, k, g)
                        for k, g in enumerate(self.ghosts)
                        for gid, gx, gy, gw, gh_, fd, gh in [g]]
                cand = [c for c in cand if c[0] <= self.reid_px]
                if cand:
                    cand.sort()
                    _, k, g = cand[0]
                    t.id = g[0]
                    KalmanTrack._next_id -= 1
                    self.ghosts.pop(k)
                    self.n_reid += 1
                    info["cause"] = info["cause"] + "_reid"

            self.tracks.append(t)
            self.n_new += 1
            x, y = dets[i]["centroid"]
            self.events.append(dict(frame=self.frame, event="birth", id=t.id,
                                    x=f"{x:.1f}", y=f"{y:.1f}",
                                    n_dets=len(dets), n_tracks=len(self.tracks),
                                    **info))

        dead = [t for t in self.tracks if t.misses > self.max_misses]
        for t in dead:
            x, y, w, h = t.bbox
            self.ghosts.append((t.id, t.centroid[0], t.centroid[1], w, h,
                                self.frame, t.hits))
            self.events.append(dict(frame=self.frame, event="death", id=t.id,
                                    x=f"{t.centroid[0]:.1f}",
                                    y=f"{t.centroid[1]:.1f}",
                                    n_dets=len(dets), n_tracks=len(self.tracks),
                                    cause=f"age{t.age}_hits{t.hits}",
                                    near_track="", near_dist="", near_misses="",
                                    ghost_id="", ghost_dist="", ghost_age="",
                                    n_within_2gate=""))
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        self.ghosts = [g for g in self.ghosts
                       if self.frame - g[5] <= self.ghost_frames]
        self.pending = [q for q in self.pending if self.frame - q[3] <= 2]
        return self.tracks


def omega_to_body(x, y, w, h, det=None):
    """
    The full-body box to DRAW for a track whose state is an omega.

    Height inverts cleanly (omega is a fixed fraction of body height), but
    width does not: body_to_omega clamps omega width to at most 1.6x its
    height, so an 11 px-wide body and a 6 px-wide one can produce the same
    omega. Reconstructing by formula gave 11 px -> 5.3 px, a body drawn half
    its real width.

    So the last MEASURED body box wins when there is one — classical dets carry
    body_bbox, CNN dets carry body_h — and the formula is only the fallback for
    a track that has never seen either. The box is re-centred on the track so it
    follows the Kalman estimate rather than the stale detection position.

    Display only; the tracker never sees this.
    """
    bh = h / OMEGA_H_FRAC
    bw = max(w, bh * 0.34)
    if det is not None:
        bb = det.get("body_bbox")
        if bb is not None:
            bw, bh = float(bb[2]), float(bb[3])
        elif det.get("body_h"):
            bh = float(det["body_h"])
            bw = max(w, bh * 0.34)
    return x + w / 2.0 - bw / 2.0, y, bw, bh


def body_to_omega(d):
    """
    Put a full-body classical blob into OMEGA coordinates.

    Without this the two measurement sources disagree about where the object
    IS: a body centroid sits at torso height, an omega centroid near the head,
    tens of pixels apart. Feeding both to one Kalman filter makes every track
    jump vertically whenever the source switches, which reads as motion the
    person never made and corrupts the velocity estimate.

    So the classical blob is reduced to its head-and-shoulder region before it
    ever reaches the tracker. Both sources then measure the same point.
    """
    x, y, w, h = d["bbox"]
    oh = h * OMEGA_H_FRAC
    ow = min(w, oh * 1.6)
    ox = x + (w - ow) / 2.0
    out = dict(d)
    out["bbox"] = (ox, y, ow, oh)
    out["centroid"] = (x + w / 2.0, y + h * OMEGA_C_FRAC)
    out["body_h"] = float(h)
    out["body_bbox"] = (x, y, w, h)
    return out


def classical_blobs(data, is_temp, args):
    """
    Cheap 'did anything move that nobody is tracking?' scan.

    Deliberately the horizontal ruleset: it is the most sensitive to small
    distant blobs, and over-triggering costs one CNN call, while under-
    triggering costs a missed person.
    """
    thr, _ = TD.compute_threshold(data, is_temp, args.delta)
    dets, mask = TD.detect_people(
        data, thr,
        tmin=args.tmin if is_temp else None,
        tmax=args.tmax if is_temp else None,
        view="horizontal", min_area=args.min_area,
        cohesion=args.cohesion, flags=dict(TD.DEFAULT_FLAGS),
        p_filter=False, p_min=0)
    for d in dets:
        d["src"] = "blob"
    return dets, mask


def range_m(body_h_px, ref_h=7.0, ref_range=8.5):
    """
    Range in metres from a body's pixel height, using the GradFarF calibration.

        rng = ref_range * ref_h / h_px

    Same relation detect_people uses internally, anchored on the person you
    measured at 8.5 m who stood 7 px tall. Under linear perspective pixel
    height is inversely proportional to distance, so one measured subject fixes
    the whole curve.
    """
    return ref_range * ref_h / max(1.0, float(body_h_px))


def m_per_px(rng, ref_h=7.0, ref_range=8.5, body_m=1.70):
    """
    Ground-plane metres per pixel at a given range.

    A body of `body_m` spans ref_h*ref_range/rng pixels there, so one pixel is
    body_m divided by that. At the 8.5 m reference this gives 0.24 m/px, which
    is the check to make if these numbers ever look wrong.
    """
    h_px = ref_h * ref_range / max(1e-3, rng)
    return body_m / max(1e-3, h_px)


def close_pairs(dets, max_m, ref_h, ref_range):
    """
    Pairs of detections standing within `max_m` of each other.

    This is the crowding trigger. Two people converging is the one situation
    the cheap path cannot survive: their warm silhouettes merge into a single
    blob, the tracker sees one object where there are two, and no amount of
    Kalman smoothing recovers the lost identity. Catching it BEFORE the merge
    is what makes the fallback useful — once merged, the CNN has less to work
    with too.

    Range comes from each blob's own pixel height, so the metre threshold holds
    across the room rather than being a pixel distance that means 0.5 m at the
    back wall and 3 m at the front.
    """
    out = []
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            a, b = dets[i], dets[j]
            ha = a.get("body_h") or a["bbox"][3]
            hb = b.get("body_h") or b["bbox"][3]
            ra, rb = range_m(ha, ref_h, ref_range), range_m(hb, ref_h, ref_range)
            # a pair far apart in depth is not "close" however near in the image
            if abs(ra - rb) > max_m:
                continue
            scale = m_per_px((ra + rb) / 2.0, ref_h, ref_range)
            ax, ay = a["centroid"]; bx, by = b["centroid"]
            d_px = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            d_m = d_px * scale
            if d_m <= max_m:
                out.append((i, j, d_m))
    return out


def unmatched(blobs, tracks, gate):
    """Blobs no confirmed track sits near — the entry signal."""
    out = []
    for b in blobs:
        bx, by = b["centroid"]
        if not any(((bx - t.centroid[0]) ** 2 + (by - t.centroid[1]) ** 2) ** 0.5
                   <= gate for t in tracks):
            out.append(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.374,
                    help="peak-F1 threshold measured for the v2 model")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--mode", choices=("hybrid", "yolo"), default="hybrid",
                    help="hybrid: classical blobs carry the cheap frames and "
                         "the CNN fires on triggers (~25%% duty). "
                         "yolo: CNN omega on EVERY frame, Kalman used purely "
                         "for identity and smoothing, classical path disabled "
                         "entirely. yolo is the accuracy ceiling and the "
                         "baseline hybrid should be judged against.")
    ap.add_argument("--always-yolo", action="store_true",
                    help="alias for --mode yolo")
    ap.add_argument("--max-stale", type=int, default=20,
                    help="force a CNN re-confirmation if a track has coasted "
                         "this many frames. Bounds how long a wrong identity "
                         "can live on Kalman momentum. ~2.3 s at 8.7 fps.")
    ap.add_argument("--dup-confirm", type=int, default=3,
                    help="a detection appearing inside the gate of an ALREADY "
                         "MATCHED track must persist this many frames before it "
                         "gets its own id. Duplicate boxes flicker and never "
                         "qualify; a second person standing there does. 1 "
                         "disables.")
    ap.add_argument("--reid-px", type=float, default=25.0,
                    help="a new detection within this many px of a recently "
                         "dead track reclaims its id instead of getting a new "
                         "one. 0 disables. Measured need: 3.8 px after a "
                         "33-frame absence.")
    ap.add_argument("--ghost-frames", type=int, default=45,
                    help="how long a dead track stays re-identifiable")
    ap.add_argument("--revive", type=float, default=1.8,
                    help="how far the association gate may widen for a coasting "
                         "track, as a multiple of --gate. This is what stops a "
                         "one-frame detection dropout from becoming a new id "
                         "and a new colour.")
    ap.add_argument("--no-dedup", action="store_true",
                    help="disable omega de-duplication (for diagnosing)")
    ap.add_argument("--dedup-iou", type=float, default=0.15)
    ap.add_argument("--dedup-centre", type=float, default=1.4)
    ap.add_argument("--gate", type=float, default=30.0,
                    help="px radius for blob-to-track association")
    ap.add_argument("--max-misses", type=int, default=8)
    ap.add_argument("--coast-trigger", type=int, default=3,
                    help="frames a CONFIRMED track may coast before the CNN is "
                         "called. 1 is far too eager: classical blobs flicker, "
                         "so a track drops a frame constantly and duty cycle "
                         "measured 62%% on real data. 3 costs ~0.3 s of "
                         "staleness and settles it.")
    ap.add_argument("--merge-dist", type=float, default=1.0,
                    help="metres. Two blobs closer than this trigger a CNN "
                         "omega pass, because their silhouettes are about to "
                         "merge into one and the classical scan will under-"
                         "count. Range comes from GradFarF, so this is a real "
                         "distance, not a pixel gap.")
    ap.add_argument("--gfar-ref-h", type=float, default=7.0,
                    help="GradFarF calibration: pixel height of the reference "
                         "person")
    ap.add_argument("--gfar-ref-range", type=float, default=8.5,
                    help="GradFarF calibration: their range in metres")
    ap.add_argument("--min-hits", type=int, default=2,
                    help="frames before a track counts. Low because the CNN "
                         "already vetted it — this is not a raw blob.")
    ap.add_argument("--delta", type=float, default=2.5)
    ap.add_argument("--tmin", type=float, default=TD.DEFAULT_TMIN_C)
    ap.add_argument("--tmax", type=float, default=TD.DEFAULT_TMAX_C)
    ap.add_argument("--min-area", type=int, default=TD.MIN_BLOB_AREA)
    ap.add_argument("--cohesion", type=int, default=2)
    ap.add_argument("--show-person", action="store_true",
                    help="also outline the CNN's class-0 person boxes. OFF by "
                         "default: omega is the tracked primitive, and drawing "
                         "person boxes anyway put a full-body rectangle on "
                         "screen that no part of the pipeline was using.")
    ap.add_argument("--box", choices=("body", "omega", "both"), default=None,
                    help="which rectangle to draw for a track. 'body' shows the "
                         "full-body box implied by the tracked omega, which is "
                         "what live_yolo.py drew; 'omega' shows the head-and-"
                         "shoulders box actually being tracked. Display only — "
                         "the tracker is unaffected either way.")
    ap.add_argument("--scale", type=int, default=5)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--opencv", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    if args.always_yolo:
        args.mode = "yolo"
    pure = args.mode == "yolo"
    if args.box is None:
        # Show what the mode actually measures. In yolo mode every detection is
        # an omega, so drawing an inferred full body would be showing something
        # the detector never produced. In hybrid the classical scan does measure
        # a real full-body blob, so a body box is honest there.
        args.box = "omega" if pure else "body"

    from ultralytics import YOLO
    if not os.path.exists(args.weights):
        sys.exit(f"no such file: {args.weights}")
    print(f"loading {args.weights} ...")
    model = YOLO(args.weights)

    cam = open_camera(args)
    data, is_temp = cam.read()
    if data is None:
        sys.exit("no frame from the camera")
    if not is_temp:
        print("\n  WARNING: not radiometric. The CNN expects a FIXED 15-45 C\n"
              "  span; an AGC image is a different encoding.\n")

    os.makedirs(TD.LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(TD.LOG_DIR, f"integrated_{stamp}")
    csv_path = os.path.join(TD.LOG_DIR, f"integrated_{stamp}.csv")
    fh = open(csv_path, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["frame", "timestamp", "n_tracks", "n_confirmed", "n_blobs",
                 "yolo_ran", "trigger", "infer_ms", "loop_ms", "ambient_c",
                 "note"])

    mt = StickyTracker(max_misses=args.max_misses, min_hits=args.min_hits,
                       gate_px=args.gate, revive_scale=args.revive,
                       ghost_frames=args.ghost_frames,
                       dup_confirm=args.dup_confirm, reid_px=args.reid_px)
    S = max(2, args.scale)
    WIN = "FLUXNET  integrated"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    paused = logging_on = False
    show_bar, show_blobs, show_trails = True, False, True
    frame_i = saved = n_yolo = 0
    sel = 0                          # which TUNABLE [ and ] act on
    occ_hist = []                    # confirmed count history, for the sparkline
    force_yolo = True                 # first frame always seeds from the CNN
    last_cnn_frame = -999
    infer_ms = blob_ms = loop_ms = 0.0
    trigger = "seed"
    prev_conf_ids = set()
    bodies = []
    blobs = []
    trail = {}

    print(f"\nlogging to {csv_path}")
    print("q quit  space pause  y force YOLO  b blobs  t trails  l log\n")

    while True:
        if not paused:
            data, is_temp = cam.read()
            if data is None:
                continue
            frame_i += 1
            t_loop = time.time()

            # ---- 1. cheap classical scan ----------------------------------
            # Skipped entirely in yolo mode: nothing consumes it there, and
            # running it anyway would charge ~3 ms/frame for a result that is
            # thrown away.
            if pure:
                blobs = []
            else:
                t0 = time.time()
                blobs, _ = classical_blobs(data, is_temp, args)
                blob_ms = 0.9 * blob_ms + 0.1 * (1000 * (time.time() - t0))

            # ---- 2. decide whether the CNN is needed ----------------------
            live = [t for t in mt.tracks if t.misses == 0]
            newcomers = [] if pure else unmatched(blobs, live, args.gate)
            stale = (frame_i - last_cnn_frame) >= args.max_stale and mt.tracks
            near = [] if pure else close_pairs(
                blobs, args.merge_dist, args.gfar_ref_h, args.gfar_ref_range)
            # LOST means gone, not "dropped one frame". A confirmed track that
            # vanished between frames, or one coasting past --coast-trigger.
            # Firing on misses>0 measured 62% duty on real capture data: the
            # classical blob scan is jittery enough that some track is almost
            # always mid-coast.
            now_ids = {t.id for t in mt.tracks if t.hits >= args.min_hits}
            died = prev_conf_ids - now_ids
            coasting = [t for t in mt.tracks
                        if t.hits >= args.min_hits
                        and t.misses >= args.coast_trigger]
            lost = bool(died) or bool(coasting)

            run_yolo = pure or force_yolo
            if run_yolo:
                trigger = "forced" if force_yolo and not pure else "every frame"
            elif near:
                # crowding outranks the others: it is the only trigger that
                # fires BEFORE the failure rather than after it
                run_yolo = True
                trigger = f"close {min(d for _, _, d in near):.1f}m"
            elif newcomers:
                run_yolo, trigger = True, f"unmatched x{len(newcomers)}"
            elif lost:
                run_yolo, trigger = (True, f"lost #{sorted(died)[0]}" if died
                                     else f"coasting x{len(coasting)}")
            elif stale:
                run_yolo, trigger = True, "stale"
            else:
                trigger = "-"
            force_yolo = False

            # ---- 3. measure, then track -----------------------------------
            if run_yolo:
                t0 = time.time()
                dets, bodies = yolo_omegas(model, data, args.conf, args.imgsz)
                if not args.no_dedup:
                    dets = dedup(dets, args.dedup_iou, args.dedup_centre)
                infer_ms = 0.9 * infer_ms + 0.1 * (1000 * (time.time() - t0))
                last_cnn_frame = frame_i
                n_yolo += 1
            else:
                # CNN-free frame: classical FULL-BODY blobs, reduced to omega
                # coordinates so they measure the same point the CNN does.
                # They keep confirmed tracks alive and positioned; they cannot
                # create a NEW confirmed track, because an unclaimed blob would
                # have triggered the CNN above.
                dets = [body_to_omega(b) for b in blobs]

            mt.update(dets)
            prev_conf_ids = {t.id for t in mt.tracks if t.hits >= args.min_hits}
            for t in mt.tracks:
                trail.setdefault(t.id, []).append(t.centroid)
                if len(trail[t.id]) > 40:
                    trail[t.id].pop(0)
            live_ids = {t.id for t in mt.tracks}
            for k in list(trail):
                if k not in live_ids:
                    del trail[k]

            loop_ms = 1000 * (time.time() - t_loop)
            if logging_on:
                wr.writerow([frame_i,
                             datetime.now().isoformat(timespec="milliseconds"),
                             len(mt.tracks), len(mt.confirmed()), len(blobs),
                             int(run_yolo), trigger, f"{infer_ms:.1f}",
                             f"{loop_ms:.1f}",
                             f"{float(np.median(data)):.2f}", args.note])
                fh.flush()

        # ---------------- draw ----------------
        vis = cv2.resize(TD.colorize(data),
                         (data.shape[1] * S, data.shape[0] * S),
                         interpolation=cv2.INTER_NEAREST)

        if show_blobs:
            for b in blobs:
                x, y, w, h = b["bbox"]
                cv2.rectangle(vis, (int(x * S), int(y * S)),
                              (int((x + w) * S), int((y + h) * S)),
                              C_FAINT, 1)

        if args.show_person:
            for (x, y, w, h), v in bodies:
                cv2.rectangle(vis, (int(x * S), int(y * S)),
                              (int((x + w) * S), int((y + h) * S)),
                              (72, 68, 64), 1)

        for t in mt.tracks:
            # Unconfirmed tracks are not drawn: a one-frame track would flash a
            # new colour that reads as an id change though nothing was counted.
            if t.hits < args.min_hits:
                continue
            col = TRACK_COLS[t.id % len(TRACK_COLS)]
            ox, oy, ow, oh = t.bbox
            solid = t.misses == 0

            # trail first, so boxes and labels sit on top of it
            tr = trail.get(t.id, [])
            if show_trails and len(tr) > 1:
                # fade with age: a uniform line implies the whole path is
                # equally current, when only the head of it is
                for i in range(1, len(tr)):
                    f = i / len(tr)
                    c = tuple(int(C_BG[k] + (col[k] - C_BG[k]) * (0.15 + 0.85 * f))
                              for k in range(3))
                    cv2.line(vis,
                             (int(tr[i - 1][0] * S), int(tr[i - 1][1] * S)),
                             (int(tr[i][0] * S), int(tr[i][1] * S)),
                             c, 1, cv2.LINE_AA)

            def draw_box(bx, by, bw_, bh_, thick):
                corners(vis, (int(bx * S), int(by * S)),
                        (int((bx + bw_) * S), int((by + bh_) * S)), col, thick)

            if args.box in ("body", "both"):
                draw_box(*omega_to_body(ox, oy, ow, oh, t.det), 2 if solid else 1)
            if args.box in ("omega", "both"):
                draw_box(ox, oy, ow, oh, 2 if solid else 1)

            x, y, w, h = (omega_to_body(ox, oy, ow, oh, t.det)
                          if args.box == "body" else (ox, oy, ow, oh))
            p0 = (int(x * S), int(y * S))
            p1 = (int((x + w) * S), int((y + h) * S))

            cf = t.det.get("conf")
            tag = f"{t.id}"
            sub = (f"{cf:.2f}" if cf is not None and solid else
                   f"coast {t.misses}" if not solid else "")
            # id badge: a filled square reads at a glance and cannot be lost
            # against a bright thermal background the way thin text can
            bs = 17
            bx0, by0 = p0[0], max(0, p0[1] - bs - 3)
            cv2.rectangle(vis, (bx0, by0), (bx0 + bs, by0 + bs), col, -1)
            cv2.putText(vis, tag, (bx0 + 6 - (2 if len(tag) > 1 else 0),
                                   by0 + bs - 4), F, 0.42, C_BG, 1, cv2.LINE_AA)
            if sub:
                sw = _tw(sub, 0.38) + 8
                cv2.rectangle(vis, (bx0 + bs + 3, by0),
                              (bx0 + bs + 3 + sw, by0 + bs), (26, 24, 22), -1)
                cv2.putText(vis, sub, (bx0 + bs + 7, by0 + bs - 4), F, 0.38,
                            col if solid else C_DIM, 1, cv2.LINE_AA)

        occ_hist.append(len(mt.confirmed()))
        if len(occ_hist) > 120:
            occ_hist.pop(0)

        if show_bar:
            H = vis.shape[0]
            bar = np.full((H, SIDEBAR_W, 3), C_BG, np.uint8)
            conf_n = len(mt.confirmed())
            duty = 100.0 * n_yolo / max(1, frame_i)
            W = SIDEBAR_W

            # ---- title -------------------------------------------------
            cv2.rectangle(bar, (0, 0), (W, 34), C_PANEL, -1)
            cv2.putText(bar, "FLUXNET", (14, 22), F, 0.5, C_TEXT, 1, cv2.LINE_AA)
            mtxt = "YOLO" if pure else "HYBRID"
            cv2.putText(bar, mtxt, (W - 14 - _tw(mtxt, 0.42), 22), F, 0.42,
                        C_WARN if pure else C_OK, 1, cv2.LINE_AA)

            # ---- occupancy ---------------------------------------------
            cv2.rectangle(bar, (14, 48), (W - 14, 132), C_PANEL, -1)
            cv2.rectangle(bar, (14, 48), (17, 132), C_OK, -1)
            cv2.putText(bar, "OCCUPANCY", (28, 68), F, 0.38, C_FAINT, 1, cv2.LINE_AA)
            big = str(conf_n)
            cv2.putText(bar, big, (28, 118), F, 1.9, C_OK, 3, cv2.LINE_AA)
            cv2.putText(bar, f"tracks {len(mt.tracks)}", (28 + _tw(big, 1.9) + 16, 100),
                        F, 0.4, C_DIM, 1, cv2.LINE_AA)
            cv2.putText(bar, f"blobs {'-' if pure else len(blobs)}",
                        (28 + _tw(big, 1.9) + 16, 118), F, 0.4, C_DIM, 1, cv2.LINE_AA)
            spark(bar, W - 26 - 84, 58, 84, 24, occ_hist, C_OK,
                  vmax=max(2, max(occ_hist) if occ_hist else 2))

            # The bottom block is fixed height; everything above it flows.
            # Without a floor the flowing rows overwrite the tunables at small
            # --scale, which is exactly what happened at scale 5 / 600 px.
            floor = H - 40 - 19 * len(TUNABLES) - 26

            def fits(y, need=20):
                return y + need <= floor

            y = 158
            head(bar, y, "DETECTOR"); y += 24
            dcol = C_OK if duty < 25 else C_WARN if duty < 60 else C_BAD
            row(bar, y, "cnn duty", f"{duty:.0f}%", vcol=dcol); y += 8
            meter(bar, 14, y, W - 28, duty / 100.0, dcol); y += 20
            if fits(y): row(bar, y, "ran", f"{n_yolo}/{frame_i}"); y += 20
            if fits(y): row(bar, y, "trigger", trigger[:16], vcol=C_ACCENT); y += 20
            if fits(y): row(bar, y, "since", f"{frame_i - last_cnn_frame} f"); y += 28

            if fits(y, 44):
                head(bar, y, "TIMING"); y += 24
                row(bar, y, "inference", f"{infer_ms:.0f} ms"); y += 20
                if fits(y): row(bar, y, "classical",
                                "off" if pure else f"{blob_ms:.0f} ms"); y += 20
                if fits(y): row(bar, y, "loop", f"{loop_ms:.0f} ms",
                                vcol=C_OK if loop_ms < 115 else C_WARN); y += 20
                saved_ms = (1 - n_yolo / max(1, frame_i)) * infer_ms
                if fits(y) and not pure:
                    row(bar, y, "saved/frame", f"{saved_ms:.0f} ms", vcol=C_OK)
                    y += 20
                y += 8

            if fits(y, 44):
                head(bar, y, "IDENTITY"); y += 24
                row(bar, y, "births", mt.n_new); y += 20
                if fits(y): row(bar, y, "dups held", mt.n_suppressed); y += 20
                if fits(y): row(bar, y, "ids reclaimed", mt.n_reid,
                                vcol=C_OK if mt.n_reid else C_TEXT); y += 20
                y += 8
            if fits(y): row(bar, y, "ambient",
                            f"{float(np.median(data)):.1f} C"); y += 20

            # ---- state chips, always shown; they are state, not detail ---
            cy = min(y + 12, floor + 16)
            cx = 14
            if paused:
                cx = chip(bar, cx, cy, "PAUSED", C_WARN)
            if logging_on:
                cx = chip(bar, cx, cy, "REC", C_BAD)
            if args.no_dedup:
                cx = chip(bar, cx, cy, "NO DEDUP", C_DIM)

            # ---- tunables, pinned to the bottom -------------------------
            ty = floor + 26
            head(bar, ty, "TUNE      TAB select    [ ]  adjust"); ty += 22
            for i, (nm, attr, _st, _lo, _hi, fmt) in enumerate(TUNABLES):
                on = i == sel
                val = fmt.format(getattr(args, attr))
                if on:
                    cv2.rectangle(bar, (8, ty - 12), (W - 8, ty + 5), C_PANEL, -1)
                    cv2.rectangle(bar, (8, ty - 12), (10, ty + 5), C_ACCENT, -1)
                cv2.putText(bar, nm, (16, ty), F, 0.42,
                            C_ACCENT if on else C_DIM, 1, cv2.LINE_AA)
                cv2.putText(bar, val, (W - 16 - _tw(val, 0.42), ty), F, 0.42,
                            C_ACCENT if on else C_TEXT, 1, cv2.LINE_AA)
                ty += 19

            cv2.line(bar, (14, H - 32), (W - 14, H - 32), C_RULE, 1)
            cv2.putText(bar, "p box   o person   d dedup   r reset   s save",
                        (14, H - 19), F, 0.34, C_FAINT, 1, cv2.LINE_AA)
            cv2.putText(bar, "q quit   space pause   y yolo   b blobs   t trails",
                        (14, H - 6), F, 0.34, C_FAINT, 1, cv2.LINE_AA)

            sep = np.full((H, 1, 3), C_RULE, np.uint8)
            vis = np.hstack([vis, sep, bar])

        cv2.imshow(WIN, vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            paused = not paused
        elif k == ord("h"):
            show_bar = not show_bar
        elif k == ord("b"):
            show_blobs = not show_blobs
        elif k == ord("t"):
            show_trails = not show_trails
        elif k == ord("y"):
            force_yolo = True
        elif k == 9:                                    # TAB
            sel = (sel + 1) % len(TUNABLES)
            print(f"selected: {TUNABLES[sel][0]}")
        elif k in (ord("["), ord("]")):
            msg = bump(args, sel, -1 if k == ord("[") else 1)
            # the tracker holds its own copies, so push the live values through
            mt.gate_px = args.gate
            mt.dup_confirm = args.dup_confirm
            mt.reid_px = args.reid_px
            print(msg)
        elif k == ord("p"):
            order = ("body", "omega", "both")
            args.box = order[(order.index(args.box) + 1) % 3]
            print(f"box = {args.box}")
        elif k == ord("o"):
            args.show_person = not args.show_person
        elif k == ord("d"):
            args.no_dedup = not args.no_dedup
            print("dedup OFF" if args.no_dedup else "dedup ON")
        elif k == ord("r"):
            # Reset identities without restarting. Essential when tuning: after
            # changing the gate you want a clean count, not ids inherited from
            # the old settings.
            mt.tracks.clear()
            mt.ghosts.clear()
            mt.pending.clear()
            trail.clear()
            KalmanTrack._next_id = 1
            print("tracker reset")
        elif k == ord("l"):
            logging_on = not logging_on
            print("logging ON" if logging_on else "logging paused")
        elif k == ord("s"):
            os.makedirs(out_dir, exist_ok=True)
            stem = f"int_{saved:06d}"
            np.save(os.path.join(out_dir, stem + ".npy"), data)
            cv2.imwrite(os.path.join(out_dir, stem + ".png"), vis)
            saved += 1
            print(f"saved {stem}")

    cv2.destroyAllWindows()
    fh.close()

    ev_path = os.path.join(TD.LOG_DIR, f"integrated_{stamp}_events.csv")
    cols = ["frame", "event", "id", "x", "y", "cause", "near_track",
            "near_dist", "near_misses", "ghost_id", "ghost_dist", "ghost_age",
            "n_within_2gate", "n_dets", "n_tracks"]
    with open(ev_path, "w", newline="") as efh:
        w = csv.DictWriter(efh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in mt.events:
            w.writerow(e)
    duty = 100.0 * n_yolo / max(1, frame_i)
    print(f"\nframes           {frame_i}")
    print(f"CNN ran          {n_yolo}  ({duty:.1f}% duty)")
    print(f"inference        {infer_ms:.0f} ms   classical {blob_ms:.1f} ms")
    print(f"mean loop        {loop_ms:.0f} ms")
    print(f"log              {csv_path}")
    print(f"track events     {ev_path}")
    births = [e for e in mt.events if e["event"] == "birth"]
    if births:
        import collections as _c
        tally = _c.Counter(e["cause"] for e in births)
        print(f"\ntrack births     {len(births)}")
        for k, v in tally.most_common():
            print(f"  {k:<12} {v:4d}  ({100.0 * v / len(births):.0f}%)")
        print(f"  suppressed dups  {mt.n_suppressed}")
        print(f"  ids reclaimed    {mt.n_reid}")
        print(f"\n  python3 track_report.py {ev_path}")
    if saved:
        print(f"saved frames     {out_dir}")
    print(f"\nCompare against --always-yolo on the same scene: if the counts\n"
          f"match, the {100 - duty:.0f}% of frames the CNN skipped cost nothing.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
