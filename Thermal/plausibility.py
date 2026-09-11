#!/usr/bin/env python3
"""
The classical detector, demoted from perception to judgement — where it is good.

    python3 plausibility.py logs/merged_20260910_021320 --evaluate
    python3 plausibility.py logs/merged_20260910_021320 --evaluate --prior fixture

THE POINT. The classical path lost to YOLO at deciding "is this shape a
person". It never lost at physics. Those are different jobs, and the physics
was thrown out with the pattern matching. This puts it back, around the CNN
instead of instead of it.

A CNN sees one 160x120 frame with no memory, no scene and no geometry. It
cannot know that the warm rectangle at (20,88) has been there since Tuesday,
that a 30 px omega high in the frame would imply a two-metre-wide head, or that
a 41 C blob is a laptop. You do not need it to learn any of that. You can
state it.

EVERY PRIOR IS SOFT. Each returns a multiplier on confidence, never a veto.
Hard rules fail exactly where you need them: someone lying down, a child, a
person half through a doorway. A soft layer degrades; a rule layer deletes the
interesting case and tells you nothing. If a prior is so certain it wants to
veto, it is not a prior, it is a sensor fault.

EVERY PRIOR EXPLAINS ITSELF. Each returns a reason string. When a detection is
suppressed you read WHY:

    0.71 -> 0.24   scale: h=28px at row 41 implies 1.6m range,
                          geometry says 6.4m (expected 5-8px)

That is what makes it debuggable, and defensible to somebody who asks.

WHAT THIS IS NOT. It is not common sense. It is a list of things you thought
of. It will not handle the situation you did not anticipate, and no amount of
adding priors changes that — it only moves the boundary. Keep the multipliers
gentle so the unanticipated case survives to be noticed.
"""

import argparse
import collections
import csv
import glob
import math
import os
import sys

import cv2
import numpy as np

import thermal_detect as TD

# Geometry. Matches mmWave/project.py — Lepton 3.1R, 95 deg HFOV.
IMG_W, IMG_H = 160, 120
FX = (IMG_W / 2) / math.tan(math.radians(95.0) / 2)      # 73.3 px
CY = IMG_H / 2

# Mount. These are ASSUMPTIONS until the tape-measure capture is done, and the
# scale prior is only as good as they are. They are stated here, once, rather
# than buried in a formula.
MOUNT_H_M = 2.00            # camera height above floor
MOUNT_TILT_DEG = 15.0       # downward tilt of the optical axis

# Head height above floor. A RANGE, not a value: 1.15 m is someone seated at a
# desk, 1.90 m is a tall person standing. Using a single number here would
# reject every seated person in the dataset, which is most of them.
HEAD_H_MIN_M, HEAD_H_MAX_M = 1.10, 1.95

# Omega physical size: head plus shoulders. OMEGA_H_FRAC = 0.30 of body height
# in integrated_launcher.py; at 1.7 m that is ~0.51 m. Range again, because
# people differ and the box is drawn by hand.
OMEGA_H_MIN_M, OMEGA_H_MAX_M = 0.32, 0.62

OMEGA_CLASS = 1


# ---------------------------------------------------------------------------
# Scene memory
# ---------------------------------------------------------------------------
class SceneMemory:
    """
    What this room looks like with nobody in it, maintained online.

    review.py builds this offline from 150 frames spread across a whole
    capture, which is right for review and useless at inference — you do not
    have the future. This keeps a running per-pixel median instead, so it works
    on a live stream and converges within a minute or two of running.

    WHY MEDIAN AND NOT MEAN. A person crossing a pixel raises its mean forever
    in proportion to how long they stood there. The median ignores them as long
    as they occupy the pixel less than half the time. review.py's docstring
    already measured what happens without this: 2.6% of every frame is
    permanently above ambient, and a naive check rediscovered those fixtures in
    all 3726 frames — a 100% false-positive rate.
    """

    def __init__(self, keep=64, stride=3):
        self.keep = keep          # frames retained for the median
        self.stride = stride      # sample every Nth frame; fixtures are slow
        self.buf = []
        self.n = 0
        self._bg = None

    def update(self, arr):
        self.n += 1
        if self.n % self.stride:
            return
        self.buf.append(arr.astype(np.float32))
        if len(self.buf) > self.keep:
            self.buf.pop(0)
        self._bg = None

    @property
    def background(self):
        if self._bg is None and len(self.buf) >= 8:
            self._bg = np.median(np.stack(self.buf), axis=0)
        return self._bg

    def ready(self):
        return self.background is not None


# ---------------------------------------------------------------------------
# Priors. Each: (multiplier, reason) — multiplier in roughly [0.1, 1.2].
# ---------------------------------------------------------------------------
def prior_fixture(box, arr, scene, **kw):
    """
    Was this warm before anybody arrived? Then it is furniture.

    The strongest prior available, and the one a human applies without noticing.
    A radiator is indistinguishable from a torso on a single frame and trivially
    distinguishable across a hundred.
    """
    if not scene or not scene.ready():
        return 1.0, ""
    x0, y0, x1, y1 = _clip(box)
    bg = scene.background[y0:y1, x0:x1]
    cur = arr[y0:y1, x0:x1]
    if bg.size == 0:
        return 1.0, ""
    amb = float(np.median(scene.background))
    bg_warm = float(np.median(bg)) - amb
    excess = float(np.median(cur)) - float(np.median(bg))

    # Warm in the background AND barely warmer now == it never left.
    if bg_warm > TD.DEFAULT_DELTA_C and excess < 0.8:
        return 0.25, (f"fixture: warm in background (+{bg_warm:.1f}C) and only "
                      f"+{excess:.1f}C now")
    if bg_warm > TD.DEFAULT_DELTA_C:
        return 0.7, f"fixture: sits on a permanently warm region (+{bg_warm:.1f}C)"
    return 1.0, ""


def prior_temperature(box, arr, **kw):
    """
    Is it warm enough, RELATIVE TO THIS ROOM, to be a person?

    THE MISTAKE THIS REPLACES, BECAUSE IT IS EASY TO MAKE TWICE. The first
    version tested the absolute band alone: 27-36 C from thermal_detect.py. On
    capture_20260909_133031 the frame median is 14.2 C and the hottest pixel
    anywhere is 25.3 C, so NOTHING in the capture reaches 27 C and the prior
    rejected every box in the file — people included. The same capture has
    boxes reading 19.0 C against a 14.2 C ambient: +4.8 C, which is exactly
    what a person looks like.

    The Lepton's absolute calibration drifts between sessions and the room
    temperature is not a constant. thermal_detect.py always knew this — it
    tests "warmer than THIS room AND physiologically plausible", and it is the
    first half that does the work. Keeping only the absolute half was the bug.

    So: excess above ambient is the test. The absolute band survives only as a
    weak upper check for genuinely hot objects, and it ABSTAINS entirely when
    the whole frame sits below the band, because that means the sensor is
    offset, not that the room is empty.
    """
    x0, y0, x1, y1 = _clip(box)
    patch = arr[y0:y1, x0:x1]
    if patch.size == 0:
        return 1.0, ""

    amb = float(np.median(arr))
    t = float(np.percentile(patch, 75))
    excess = t - amb

    # Primary test: relative. A person is a few degrees above the room.
    if excess < 0.5:
        return 0.35, f"temp: only +{excess:.1f}C above ambient ({amb:.1f}C)"
    if excess < TD.DEFAULT_DELTA_C * 0.5:
        return 0.75, f"temp: weak, +{excess:.1f}C above ambient"

    # Secondary: absolute, and only in the direction that survives a
    # calibration offset. Too hot is still too hot; "too cold" is unusable
    # when the whole frame reads cold.
    frame_max = float(np.percentile(arr, 99.9))
    if frame_max >= TD.DEFAULT_TMIN_C and t > TD.DEFAULT_TMAX_C + 4.0:
        return 0.4, (f"temp: {t:.1f}C, too hot for skin "
                     f"(>{TD.DEFAULT_TMAX_C}C) — likely equipment")
    return 1.0, ""


def range_from_row(row, head_h):
    """
    How far away a head at `head_h` metres would be to appear at image `row`.

    The camera looks down at MOUNT_TILT_DEG. A head below camera height sits at
    angle atan(dh / R) below horizontal, i.e. (tilt - that) ABOVE the optical
    axis. Rearranged for R. Returns None when the geometry has no solution —
    a head at that row would have to be above the camera, or infinitely far.
    """
    dh = MOUNT_H_M - head_h
    ang_above_axis = math.atan((CY - row) / FX)              # +ve = above axis
    ang_below_horizon = math.radians(MOUNT_TILT_DEG) - ang_above_axis
    if ang_below_horizon <= 1e-3:
        return None
    if dh <= 0:                                              # head above camera
        return None
    return dh / math.tan(ang_below_horizon)


def row_from_range(R, head_h):
    """Image row a head at `head_h` metres and range `R` would appear at."""
    dh = MOUNT_H_M - head_h
    if R <= 0.1:
        return None
    below = math.atan(dh / R) if dh > 0 else -math.atan(-dh / R)
    return CY - FX * math.tan(math.radians(MOUNT_TILT_DEG) - below)


def prior_scale(box, **kw):
    """
    Is this box's SIZE consistent with where it sits in the frame?

    DIRECTION MATTERS, AND THE OBVIOUS ONE IS WRONG. The first version of this
    went row -> range -> expected size. That inversion is ill-conditioned in
    this geometry: the camera is at 2.0 m and heads at ~1.7 m, so dh = 0.3 m,
    the depression angle is tiny, and the whole 4-9 m range compresses into
    THREE image rows (46.1 -> 43.0). One pixel of box-centre error moved the
    implied range by metres, which made the prior noise.

    So it runs the other way. Size -> range is well conditioned: R = fx*h/h_px,
    where a 1 px error on a 10 px box is a 10% range error, not 300%. From that
    range it predicts the row band and compares.

    WHAT THIS CAN AND CANNOT DO. Row saturates near the horizon (~43 at any
    range beyond 6 m), so this cannot tell 6 m from 9 m and does not try. What
    it catches is the box whose size says "1.5 m away" sitting at a row that
    says "far" — the confident absurdity, which is exactly the failure the CNN
    produces and cannot detect itself.
    """
    x0, y0, x1, y1 = box[:4]
    w_px = x1 - x0
    h_px = y1 - y0
    row = (y0 + y1) / 2.0
    if h_px < 2:
        return 1.0, ""

    # FORESHORTENING. Directly beneath the camera you see the top of a head and
    # shoulders, not their profile: the box goes wide and flat, and its HEIGHT
    # stops encoding range — which is the inversion this whole prior rests on.
    #
    # Measured on 1546 human labels: boxes this prior rejected had median
    # aspect 1.75 at row 111, against 1.09 at row 60 for the ones it kept, and
    # the median aspect across all labels jumps from ~1.0 in rows 60-104 to
    # 1.73 in rows 105-119. Every rejection was a real person walking under the
    # sensor. Abstain rather than be confidently wrong about them.
    if w_px / max(1.0, h_px) > 1.45:
        return 1.0, ""

    # Range implied by apparent size, across plausible omega sizes.
    R_lo = FX * OMEGA_H_MIN_M / h_px
    R_hi = FX * OMEGA_H_MAX_M / h_px

    rows = []
    for R in (R_lo, R_hi):
        for head_h in (HEAD_H_MIN_M, HEAD_H_MAX_M):
            r = row_from_range(R, head_h)
            if r is not None:
                rows.append(r)
    if not rows:
        return 1.0, ""
    lo_row, hi_row = min(rows) - 6.0, max(rows) + 6.0   # generous tolerance

    if lo_row <= row <= hi_row:
        return 1.0, ""
    off = (lo_row - row) if row < lo_row else (row - hi_row)
    mult = max(0.25, 1.0 / (1.0 + 0.12 * off))
    return mult, (f"scale: h={h_px:.0f}px implies {R_lo:.1f}-{R_hi:.1f}m, "
                  f"which should sit at row {lo_row:.0f}-{hi_row:.0f}, "
                  f"but it is at {row:.0f}")


def prior_anatomy(box, arr, **kw):
    """
    Does this sit on top of a warm mass, the way a head sits on a body?

    An omega is the TOP of something. A box floating in the middle of a torso,
    or below the warm region's centre, is anatomically backwards even when the
    texture looks right. Cheap, and independent of everything else here.

    Abstains when the blob is small — at 9 m a whole person is barely larger
    than the omega, and there is no "below" to measure.
    """
    x0, y0, x1, y1 = _clip(box)
    bh = y1 - y0
    if bh < 2:
        return 1.0, ""
    # a strip of the same width, one box-height below
    sy0, sy1 = min(IMG_H, y1), min(IMG_H, y1 + int(bh * 1.5))
    if sy1 - sy0 < 2:
        return 1.0, ""                        # at the frame edge; abstain
    amb = float(np.median(arr))
    below = arr[sy0:sy1, x0:x1]
    inside = arr[y0:y1, x0:x1]
    if below.size == 0 or inside.size == 0:
        return 1.0, ""
    warm_below = float((below > amb + TD.DEFAULT_DELTA_C).mean())
    warm_in = float((inside > amb + TD.DEFAULT_DELTA_C).mean())
    if warm_in < 0.15:
        return 0.6, f"anatomy: only {warm_in:.0%} of the box is warm"
    if warm_below < 0.05 and warm_in > 0.4:
        return 0.75, "anatomy: warm box with nothing warm beneath it"
    return 1.0, ""


def prior_exclusion(box, all_boxes, **kw):
    """
    Heads do not stack. cap_000463 produced five omegas over two or three people.

    Suppresses the weaker of a heavily-overlapping pair. This duplicates what
    dedup() already does inside integrated_launcher, and is kept separate so
    the evaluation can measure how much of the benefit is really just dedup —
    otherwise every other prior gets credit for it.
    """
    x0, y0, x1, y1, conf = box[0], box[1], box[2], box[3], box[4]
    worst = 1.0
    reason = ""
    for other in all_boxes:
        if other is box:
            continue
        if _iou((x0, y0, x1, y1), other[:4]) < 0.45:
            continue
        if other[4] > conf:
            worst = 0.4
            reason = f"exclusion: overlaps a stronger omega ({other[4]:.2f})"
    return worst, reason


PRIORS = {
    "fixture": prior_fixture,
    "temperature": prior_temperature,
    "scale": prior_scale,
    "anatomy": prior_anatomy,
    "exclusion": prior_exclusion,
}


# ---------------------------------------------------------------------------
class Judge:
    """
    Applies the priors. Advisory by default: it reports, it does not change.

    `apply=False` exists so you can watch what it WOULD have done for a session
    before letting it touch anything. Every automated filter feels obviously
    right until you see the list of things it removed.
    """

    def __init__(self, enabled=None, apply=True, floor=0.05):
        self.enabled = list(enabled or PRIORS)
        self.apply = apply
        self.floor = floor
        self.scene = SceneMemory()

    def observe(self, arr):
        self.scene.update(arr)

    def judge(self, boxes, arr):
        """
        boxes: [(x0, y0, x1, y1, conf), ...] in sensor px.
        Returns [(box, adjusted_conf, [reasons])] in the same order.
        """
        out = []
        for b in boxes:
            mult, reasons = 1.0, []
            for name in self.enabled:
                m, why = PRIORS[name](box=b, arr=arr, scene=self.scene,
                                      all_boxes=boxes)
                mult *= m
                if why:
                    reasons.append(why)
            adj = max(self.floor, b[4] * mult) if self.apply else b[4]
            out.append((b, adj, reasons))
        return out


# ---------------------------------------------------------------------------
def _clip(box):
    x0 = max(0, int(box[0])); y0 = max(0, int(box[1]))
    x1 = min(IMG_W, int(math.ceil(box[2]))); y1 = min(IMG_H, int(math.ceil(box[3])))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _load_boxes(path):
    out = []
    if not os.path.exists(path):
        return out
    for ln in open(path):
        q = ln.split()
        if len(q) == 5 and int(q[0]) == OMEGA_CLASS:
            cx, cy, w, h = (float(v) for v in q[1:])
            out.append(((cx - w/2) * IMG_W, (cy - h/2) * IMG_H,
                        (cx + w/2) * IMG_W, (cy + h/2) * IMG_H))
    return out


# ---------------------------------------------------------------------------

def _frames(root, limit=None):
    """Frames in a capture, with an error message that names the actual problem."""
    if not os.path.isdir(root):
        near = sorted(glob.glob(os.path.join("logs", os.path.basename(root) + "*")))
        hint = f"\n  did you mean: {near[0]}" if near else \
               "\n  (paths are relative to Thermal/ — capture dirs live under logs/)"
        sys.exit(f"no such directory: {root}{hint}")
    npys = sorted(glob.glob(os.path.join(root, "npy", "*.npy")))
    if not npys:
        sys.exit(f"{root} exists but has no npy/ — not a capture log")
    return npys[:limit] if limit else npys

def evaluate(root, weights, conf, priors, limit=None):
    """
    Score each prior against YOUR labels, one at a time and all together.

    The whole point of building this as separate functions. A prior that sounds
    obviously right can still cost recall, and the only way to know is to run
    it against frames a human has already judged. labels_human/ is that ground
    truth.
    """
    from ultralytics import YOLO
    from model_registry import resolve_weights
    model = YOLO(resolve_weights(weights))

    npys = _frames(root, limit)
    hd = os.path.join(root, "labels_human")
    if not os.path.isdir(hd) or not os.listdir(hd):
        sys.exit(f"{root} has no labels_human/ — --evaluate needs ground "
                 f"truth to score against.\n"
                 f"  For an unannotated capture use --inspect instead: it "
                 f"applies the priors to\n"
                 f"  the boxes already in labels/ and explains each verdict, "
                 f"no labels required.")

    # One warm-up pass so the scene memory is populated before anything is
    # judged. At inference this happens naturally over the first minute.
    scene = SceneMemory()
    for f in npys[::max(1, len(npys)//120)][:120]:
        scene.update(np.load(f))

    sets = [("baseline (no priors)", [])] + \
           [(p, [p]) for p in priors] + \
           [("ALL", list(priors))]

    print(f"{len(npys)} frames   model conf {conf}\n")
    print(f"  {'configuration':<26} {'TP':>5} {'FP':>5} {'FN':>5} "
          f"{'prec':>6} {'rec':>6} {'F1':>6}")
    print("  " + "-" * 66)

    cache = {}
    for label, names in sets:
        j = Judge(enabled=names, apply=bool(names))
        j.scene = scene
        tp = fp = fn = 0
        for f in npys:
            stem = os.path.splitext(os.path.basename(f))[0]
            arr = np.load(f)
            if stem not in cache:
                lo, hi = TD.PNG_SPAN_C
                v = np.clip((arr.astype(np.float32)-lo)/(hi-lo), 0, 1)
                img = cv2.merge([(v*255).astype(np.uint8)]*3)
                r = model.predict(img, verbose=False, conf=0.01, imgsz=640)[0]
                dets = []
                if len(r.boxes):
                    for (a, b_, c_, d), cl, s in zip(
                            r.boxes.xyxy.cpu().numpy(),
                            r.boxes.cls.cpu().numpy().astype(int),
                            r.boxes.conf.cpu().numpy()):
                        if cl == OMEGA_CLASS:
                            dets.append((float(a), float(b_), float(c_),
                                         float(d), float(s)))
                cache[stem] = dets
            judged = j.judge(cache[stem], arr)
            kept = [bx for bx, adj, _ in judged if adj >= conf]
            truth = _load_boxes(os.path.join(hd, stem + ".txt"))
            used = set()
            for k in kept:
                hit = None
                for i, t in enumerate(truth):
                    if i in used:
                        continue
                    if _iou(k[:4], t) >= 0.3:
                        hit = i
                        break
                if hit is None:
                    fp += 1
                else:
                    used.add(hit)
                    tp += 1
            fn += len(truth) - len(used)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        print(f"  {label:<26} {tp:5d} {fp:5d} {fn:5d} "
              f"{prec:6.3f} {rec:6.3f} {f1:6.3f}")
    print("\n  A prior earns its place by raising F1. One that lifts precision "
          "\n  while costing more recall is making the detector worse and "
          "looking\n  better while doing it.")


def evaluate_empty(root, weights, conf, priors, limit=None):
    """
    Score the priors on a capture with NOBODY IN IT. Every detection is wrong.

    THE CLEANEST MEASUREMENT AVAILABLE, and the only one free of both biases
    that afflict the labelled captures:

      * no feedback loop. labels_human/ in every merged_* log was seeded from
        the model's own boxes, so a false positive you accepted is now "truth"
        and a prior that kills it is scored as LOSING recall — penalising
        exactly the behaviour you want.
      * no training contamination. Every labelled capture is inside v3a.

    Here there is nothing to bias: the room is empty, so the correct output is
    zero boxes, and any prior that removes one is unambiguously right. Recall
    is not measurable and does not need to be — you measure that separately on
    a capture with people in it.
    """
    from ultralytics import YOLO
    from model_registry import resolve_weights
    model = YOLO(resolve_weights(weights))

    npys = _frames(root, limit)

    scene = SceneMemory()
    for f in npys[::max(1, len(npys)//120)][:120]:
        scene.update(np.load(f))

    dets_by_frame = {}
    for f in npys:
        arr = np.load(f)
        lo, hi = TD.PNG_SPAN_C
        v = np.clip((arr.astype(np.float32)-lo)/(hi-lo), 0, 1)
        img = cv2.merge([(v*255).astype(np.uint8)]*3)
        r = model.predict(img, verbose=False, conf=0.01, imgsz=640)[0]
        d = []
        if len(r.boxes):
            for (a, b_, c_, dd), cl, s in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.cls.cpu().numpy().astype(int),
                    r.boxes.conf.cpu().numpy()):
                if cl == OMEGA_CLASS:
                    d.append((float(a), float(b_), float(c_), float(dd), float(s)))
        dets_by_frame[f] = d

    sets = [("baseline (no priors)", [])] + \
           [(p, [p]) for p in priors] + [("ALL", list(priors))]

    print(f"EMPTY ROOM — {len(npys)} frames, model conf {conf}")
    print("every detection is a false positive by construction\n")
    print(f"  {'configuration':<26} {'FPs':>7} {'per frame':>11} {'removed':>9}")
    print("  " + "-" * 56)
    base = None
    for label, names in sets:
        j = Judge(enabled=names, apply=bool(names))
        j.scene = scene
        fp = 0
        for f in npys:
            arr = np.load(f)
            fp += sum(1 for _, adj, _ in j.judge(dets_by_frame[f], arr)
                      if adj >= conf)
        if base is None:
            base = fp
        cut = "" if base == 0 else f"{100.0*(base-fp)/base:8.1f}%"
        print(f"  {label:<26} {fp:7d} {fp/len(npys):11.3f} {cut:>9}")
    print("\n  Now measure the same priors on a capture WITH people, to see "
          "what\n  each one costs in recall. A prior is only worth wiring in "
          "if it\n  removes more false positives than true ones.")


def inspect(root, priors, limit=None, show=25):
    """
    Apply the priors to the boxes ALREADY in labels/, and explain each verdict.

    No ground truth needed and no inference run: labels/ is what the model
    already emitted, so the priors can be scored against it directly. This is
    the mode for a capture you have not annotated — you read the reasons and
    judge for yourself whether the suppressed boxes were false positives.

    It cannot compute precision or recall, because nothing here knows the
    truth. It answers a narrower and still useful question: what would this
    layer have removed, and on what grounds.
    """
    npys = _frames(root, limit)

    scene = SceneMemory()
    for f in npys[::max(1, len(npys) // 120)][:120]:
        scene.update(np.load(f))
    if not scene.ready():
        print("  scene memory did not converge — fixture prior will abstain")

    j = Judge(enabled=priors, apply=True)
    j.scene = scene

    fired = collections.Counter()
    mults = []
    flagged = []
    n_box = 0
    for f in npys:
        stem = os.path.splitext(os.path.basename(f))[0]
        arr = np.load(f)
        boxes = _load_boxes(os.path.join(root, "labels", stem + ".txt"))
        if not boxes:
            continue
        # labels/ carries no confidence, so assume 1.0 and read the MULTIPLIER.
        bs = [(b[0], b[1], b[2], b[3], 1.0) for b in boxes]
        for (b, adj, reasons) in j.judge(bs, arr):
            n_box += 1
            mults.append(adj)
            for r in reasons:
                # reason prefixes are not always the prior name ("temp:" vs
                # "temperature"), so map explicitly rather than by string.
                key = r.split(":")[0]
                fired[{"temp": "temperature"}.get(key, key)] += 1
            if adj < 0.9:
                flagged.append((adj, stem, b, reasons))

    print(f"{len(npys)} frames, {n_box} boxes from labels/\n")
    if not n_box:
        return

    mults = np.array(mults)
    print("  multiplier distribution")
    for lo, hi, tag in ((0.0, 0.3, "crushed  <0.3"),
                        (0.3, 0.6, "heavy  0.3-0.6"),
                        (0.6, 0.9, "mild   0.6-0.9"),
                        (0.9, 1.01, "untouched >=0.9")):
        n = int(((mults >= lo) & (mults < hi)).sum())
        bar = "#" * int(50.0 * n / n_box)
        print(f"    {tag:<16} {n:5d} ({100.0*n/n_box:5.1f}%)  {bar}")

    print(f"\n  which prior fired, and how often")
    for name in priors:
        c = fired.get(name, 0)
        print(f"    {name:<14} {c:5d}  ({100.0*c/n_box:5.1f}% of boxes)")

    flagged.sort(key=lambda t: t[0])
    print(f"\n  {len(flagged)} boxes suppressed. Worst {min(show, len(flagged))}:")
    for adj, stem, b, reasons in flagged[:show]:
        print(f"    x{adj:.2f}  {stem}  "
              f"[{b[0]:.0f},{b[1]:.0f} {b[2]-b[0]:.0f}x{b[3]-b[1]:.0f}px]")
        for r in reasons:
            print(f"           {r}")
    print(f"\n  Open review/{flagged[0][1]}.png if you want to see the worst "
          f"one.\n  These are v2's own detections — anything crushed here is "
          f"a box that\n  became a LABEL if it survived annotation.")


def main():
    ap = argparse.ArgumentParser(
        description="Physical priors over CNN detections. Judgement, not perception.")
    ap.add_argument("capture_dir", nargs="?")
    ap.add_argument("--evaluate", action="store_true",
                    help="score priors against labels_human/")
    ap.add_argument("--inspect", action="store_true",
                    help="apply priors to the boxes already in labels/ and "
                         "explain each verdict. No ground truth needed.")
    ap.add_argument("--empty", action="store_true",
                    help="capture contains NOBODY. Every detection is a false "
                         "positive; no labels needed. The unbiased test.")
    ap.add_argument("--prior", action="append", default=None,
                    choices=list(PRIORS),
                    help="evaluate only these, repeatable")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=0.374)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.inspect:
        if not args.capture_dir:
            sys.exit("--inspect needs a capture directory")
        inspect(args.capture_dir.rstrip("/"), args.prior or list(PRIORS),
                args.limit)
        return

    if args.empty:
        if not args.capture_dir:
            sys.exit("--empty needs a capture directory")
        evaluate_empty(args.capture_dir.rstrip("/"), args.weights, args.conf,
                       args.prior or list(PRIORS), args.limit)
        return

    if args.evaluate:
        if not args.capture_dir:
            sys.exit("--evaluate needs a capture directory")
        evaluate(args.capture_dir.rstrip("/"), args.weights, args.conf,
                 args.prior or list(PRIORS), args.limit)
        return

    # No capture given: show the geometry the scale prior is built on, since
    # it is the prior most dependent on assumptions.
    print(f"mount {MOUNT_H_M} m, tilt {MOUNT_TILT_DEG} deg, fx {FX:.1f} px")
    print(f"head height {HEAD_H_MIN_M}-{HEAD_H_MAX_M} m  ->  dh as little as "
          f"{MOUNT_H_M - HEAD_H_MAX_M:+.2f} m\n")
    print(f"  {'range':>7} {'omega px':>12} {'row (standing)':>16}")
    for R in (1.5, 2, 3, 4, 5, 6, 7, 8, 9):
        lo = FX * OMEGA_H_MIN_M / R
        hi = FX * OMEGA_H_MAX_M / R
        r = row_from_range(R, 1.70)
        print(f"  {R:6.1f}m {lo:7.0f}-{hi:<4.0f} {r:16.1f}")
    print("\n  Note how little the row moves beyond 5 m — the camera is nearly")
    print("  level with people's heads, so row carries almost no range")
    print("  information out there. SIZE does. That is why the scale prior")
    print("  works size -> range and not the other way round.")
    print("\nrun with a capture dir and --evaluate to score the priors "
          "against labels_human/")


if __name__ == "__main__":
    main()
