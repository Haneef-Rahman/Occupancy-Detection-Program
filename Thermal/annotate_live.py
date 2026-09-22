#!/usr/bin/env python3
"""
EXPERIMENTAL. Annotate by pointing at people, not by drawing rectangles.

    ./run.sh annotate_live.py logs/merged_YYYYMMDD_HHMMSS

    hover a hot object   the model proposes an omega box under your cursor
    LEFT CLICK           accept it
    SPACE, SPACE         draw a box yourself: two opposite corners at the
                         crosshair. Works at any time — the proposals stay
                         live, so you can click the two heads the model found
                         and draw the third it missed without changing mode.
    DRAG a box           move one you already have. Only starts when there is
                         no new proposal under the cursor, so accepting always
                         wins over grabbing.
    ctrl-C / ctrl-V      copy this frame's boxes, paste them onto another.
                         For a cluster where everyone barely moved: paste,
                         then drag the two or three that shifted. Paste APPENDS
                         and skips exact duplicates.
    ENTER (picked none)  accept everything currently previewed, and move on
    ENTER (picked some)  commit exactly what you picked
    x                    commit EMPTY — nobody in this cluster
    g                    commit as a HARD NEGATIVE — nobody here, and the
                         frame contains something that looks like it should
                         fire (hot printer, sunlit pavement, radiator)
    r                    RE-SCAN just the object you are hovering
    hover + e / DEL      delete a box you accepted
    [ ]                  lower / raise the confidence gate, live
    u undo (cancels a half-placed corner first)   c clear
    x empty   g negative   ENTER next   b back   d drop   q quit

WHAT IS DIFFERENT. annotate.py and dataset_pipeline.py use the model PASSIVELY:
you draw, and later something checks your work at 0.29. Here the model is in the
loop while you annotate. You never place a corner — you point at a person and
take the box the model already has.

WHY THAT IS BETTER, WHEN IT IS. Two corner clicks on a 15 px omega is four
opportunities to be a pixel off, and that error goes straight into the labels as
geometric noise. The model's box is derived from the image every time and is
consistent frame to frame. When it is right, accepting it is strictly better
than redrawing it — faster AND more precise.

WHY SPACE EXISTS. Everything above is true only while the model is right. It was
once the case here that you could ONLY accept what the model proposed, so a head
it could not find at any confidence could not be labelled at all — and a MISSING
box is the worst label error there is: it teaches the network that a person is
background. review.py sorts misses ahead of every bad-IoU box for that reason.
The old escape hatch was to mark the cluster `d` and redo it in
dataset_pipeline.py, which meant leaving the tool to fix the tool's one gap.

Space closes it, IN ADDITION to the proposals rather than instead of them. The
first version of this put corner-drawing on a mode key that took over the mouse
button, which defeats the point: the frame where you need to draw is usually a
frame where the model also got two people right, and you want both without
switching. Corners are on the space bar so nothing has to be given up.

Two opposite corners, not click-and-drag, for the reasons in Pad's docstring.
Boxes you draw join the same list as proposals you accept and commit by the same
path; nothing downstream can tell which came from where, and nothing should.

WHAT `r` IS FOR. Prefer it to `m`. A box the model derives from the image is
consistent frame to frame; a box you draw carries your hand's error into the
labels as geometric noise.

`r` is the answer to that. It re-runs inference on a window around your cursor
alone, so the object gets a second look on its own terms: NMS is not competing
across the whole frame, the head fills far more of the input, and the
surrounding warm clutter is absent. If something is there, this is what finds
it. Point at the person, press r, then press [ down to see what came back.

If a person survives that and still has no box, draw it with SPACE. Do not
leave the person unboxed.

THE SCAN WINDOW SIZES ITSELF. It is not a fixed square: it is the bounding box
of the warm blob under your cursor, squared up and padded. A fixed window is
wrong at both ends of the room — at 2 m a body overflows it and the crop cuts
the head off its shoulders; at 9 m the same window is mostly cold wall, which
is the clutter the crop existed to remove. The rectangle on screen turns green
when it has locked onto a blob and stays grey when it has fallen back to the
fixed --crop, so you always know which you are getting.

Zoom follows from that: the window is rendered to a constant --target size, so
a person at 2 m and a person at 9 m both reach the model at the same apparent
scale. A fixed zoom factor would have undone the point of the auto-window.

HOW INFERENCE IS SPENT. One full-frame pass per cluster at --floor (default
0.01), cached. `[` and `]` re-filter that cache rather than re-running, so the
gate responds instantly. `r` is the only thing that spends another forward
pass, on demand, on one object.

CONTRACT. Identical to dataset_pipeline.py stage 4: reads triage.csv, writes
labels_human/, propagates to cluster members with motion compensation, snaps
propagated boxes to YOLO on their own frame. The representative is written
verbatim. Nothing here edits a frame you drew on. `g` additionally appends to
negatives.csv, by the same helper stage 4 uses.
"""

import argparse
import collections
import csv
import os
import sys

import cv2
import numpy as np

import thermal_detect as TD
import dataset_pipeline as DP
from model_registry import resolve_weights

OMEGA_CLASS = DP.OMEGA_CLASS
OMEGA_COL = DP.OMEGA_COL          # accepted boxes
CAND_COL = (255, 255, 255)        # the proposal under your cursor
DOOMED_COL = (70, 70, 255)        # what e/DEL would remove
DIM_COL = (95, 95, 105)           # other proposals above the gate
ARM_COL = (120, 220, 140)         # proposals ENTER is about to accept wholesale
MANUAL_COL = (255, 120, 255)      # the box you are drawing by hand
NEG_COL = (150, 150, 255)         # hard-negative banner

# ctrl-C and ctrl-V arrive from HighGUI as the raw control codes 3 and 22.
# Shift-C / shift-V are accepted as well: on the macOS Cocoa backend control
# characters are not always delivered, and a copy key that silently does
# nothing is worse than a second way to press it. Lower-case c and v are NOT
# here — they are clear and contrast, and must keep working.
COPY_KEYS = (3, ord("C"))
PASTE_KEYS = (22, ord("V"))

CONF_STEP = 0.01
CONF_MIN, CONF_MAX = 0.01, 0.95


# ---------------------------------------------------------------------------
def infer(model, arr, floor, imgsz):
    """
    Every omega the model can see, down to `floor`, sorted best first.

    Run once per frame and cached. The confidence gate is applied later, at
    display time, so moving the gate costs nothing. Running at 0.01 and
    filtering up is the same set as running at the gate directly — YOLO's conf
    argument is a threshold on the same scores, not a different computation.
    """
    lo, hi = DP.SPAN_C
    v = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    img = cv2.merge([(v * 255).astype(np.uint8)] * 3)
    res = model.predict(img, verbose=False, conf=floor, imgsz=imgsz)[0]
    out = []
    if len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        for (x0, y0, x1, y1), c, s in zip(xyxy, cls, cf):
            if c == OMEGA_CLASS:
                out.append((float(x0), float(y0), float(x1), float(y1),
                            float(s)))
    out.sort(key=lambda d: -d[4])
    return out


def blob_window(an, arr, cursor, pad, min_half, max_half, search=6):
    """
    Bound the warm object under the cursor, so the scan window fits the person.

    A fixed 48x48 is wrong at both ends of the room. At 2 m a body overflows it
    and the crop cuts the head off its shoulders; at 9 m the same window is
    mostly cold wall, which is the clutter the crop was supposed to remove. The
    object's own extent is the only sensible window, and the frame already
    tells us it.

    Thresholding is annotate.warm_mask — imported, not reimplemented, so this
    cannot drift from what review.py scores against. It is an ABSOLUTE ambient
    threshold rather than background subtraction, deliberately: a person seated
    still IS their own background, and subtracting it deleted 1842 valid labels
    when that was measured.

    Returns (x0, y0, x1, y1) in sensor pixels, or None when the cursor is not
    on anything warm — the caller then falls back to the fixed window.
    """
    H, W = arr.shape[:2]
    cx, cy = int(round(cursor[0])), int(round(cursor[1]))
    if not (0 <= cx < W and 0 <= cy < H):
        return None

    mask = an.warm_mask(arr)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None

    lid = int(lab[cy, cx])
    if lid == 0:
        # Not exactly on the blob. Take the nearest labelled pixel within a
        # few px rather than giving up — you point at a person, not at a
        # specific pixel of one, and the omega is often a cool patch of hair
        # sitting inside the warm silhouette.
        best_d, best_l = None, 0
        y0, y1 = max(0, cy - search), min(H, cy + search + 1)
        x0, x1 = max(0, cx - search), min(W, cx + search + 1)
        sub = lab[y0:y1, x0:x1]
        ys, xs = np.nonzero(sub)
        for yy, xx in zip(ys, xs):
            d = (yy + y0 - cy) ** 2 + (xx + x0 - cx) ** 2
            if best_d is None or d < best_d:
                best_d, best_l = d, int(sub[yy, xx])
        lid = best_l
        if lid == 0:
            return None

    bx = st[lid, cv2.CC_STAT_LEFT]
    by = st[lid, cv2.CC_STAT_TOP]
    bw = st[lid, cv2.CC_STAT_WIDTH]
    bh = st[lid, cv2.CC_STAT_HEIGHT]

    # Square it about the blob centre. A tall thin body would otherwise give a
    # tall thin crop, and the model has never seen an omega in one of those.
    ccx, ccy = bx + bw / 2.0, by + bh / 2.0
    half = max(bw, bh) / 2.0 + pad
    half = float(min(max(half, min_half), max_half))
    return (max(0.0, ccx - half), max(0.0, ccy - half),
            min(float(W), ccx + half), min(float(H), ccy + half))


def scan_window(an, arr, cursor, args):
    """The blob's own extent, or a fixed square when the cursor is on nothing."""
    w = blob_window(an, arr, cursor, args.pad, args.min_half, args.max_half)
    if w is not None:
        return w
    H, W = arr.shape[:2]
    cx, cy = cursor
    return (max(0.0, cx - args.crop), max(0.0, cy - args.crop),
            min(float(W), cx + args.crop), min(float(H), cy + args.crop))


def refresh_at(model, arr, window, target, max_zoom, floor, imgsz):
    """
    Re-run the model on a window around the cursor. One object at a time.

    WHY THIS EXISTS. The full-frame pass is a single look at the whole 160x120
    array. A head it does not find there cannot be annotated at all, and a
    missing box is the worst label error available — it teaches the network
    that a person is background. This is a second opinion, aimed where you are
    pointing, and it is the answer to that gap.

    WHAT ACTUALLY CHANGES, AND WHAT DOES NOT. Cropping does not lower the
    threshold: the floor is the same. Three things do differ, and any of them
    can surface a box the full frame missed:

      * NMS no longer competes across the whole frame. Two heads 5 px apart
        can suppress each other in one pass; alone in a crop, neither does.
      * the object occupies far more of the input, so small-object recall
        improves in the way it usually does for detectors.
      * surrounding warm clutter is simply absent from the tensor.

    IT CAN ALSO FIND NOTHING, OR FIND RUBBISH. The model was trained on whole
    frames, so a zoomed crop is off its training distribution and the scores
    coming back are not calibrated against the ones from the full pass. Treat a
    refreshed proposal as a suggestion to look harder, not as a better number.
    Both are shown; you decide.

    Returns detections in FRAME coordinates.
    """
    H, W = arr.shape[:2]
    wx0, wy0, wx1, wy1 = window
    x0, y0 = int(max(0, wx0)), int(max(0, wy0))
    x1, y1 = int(min(W, wx1)), int(min(H, wy1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []

    crop = arr[y0:y1, x0:x1]

    # Zoom is DERIVED, not fixed. The window now varies with the person's size,
    # so a constant factor would present a near subject huge and a far one
    # small — reintroducing the scale variation the auto-window removes. Scale
    # to a constant target instead, and the model sees an omega at the same
    # apparent size whether the person is at 2 m or 9 m.
    zoom = target / float(max(x1 - x0, y1 - y0))
    zoom = float(min(max(zoom, 1.0), max_zoom))

    lo, hi = DP.SPAN_C
    v = np.clip((crop.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    img = cv2.merge([(v * 255).astype(np.uint8)] * 3)
    img = cv2.resize(img, None, fx=zoom, fy=zoom,
                     interpolation=cv2.INTER_CUBIC)

    res = model.predict(img, verbose=False, conf=floor, imgsz=imgsz)[0]
    out = []
    if len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        for (bx0, by0, bx1, by1), c, s in zip(xyxy, cls, cf):
            if c != OMEGA_CLASS:
                continue
            # crop pixels -> frame pixels
            out.append((x0 + bx0 / zoom, y0 + by0 / zoom,
                        x0 + bx1 / zoom, y0 + by1 / zoom, float(s)))
    return out


def merge_dets(dets, new, thresh=0.55):
    """
    Add refreshed detections, keeping the better score where they agree.

    A refresh usually re-finds what the full pass already had. Appending blindly
    would stack near-duplicates and make the proposal count meaningless, so an
    overlapping pair collapses to whichever scored higher. Returns
    (merged, n_new) where n_new counts only genuinely new objects — that number
    is the whole point of pressing r.
    """
    out = list(dets)
    n_new = 0
    for nd in new:
        hit = None
        for i, od in enumerate(out):
            if DP.iou(nd[:4], od[:4]) >= thresh:
                hit = i
                break
        if hit is None:
            out.append(nd)
            n_new += 1
        elif nd[4] > out[hit][4]:
            out[hit] = nd
    out.sort(key=lambda d: -d[4])
    return out, n_new


def candidate(dets, conf, cursor):
    """
    Which proposal the cursor is on, or None.

    Containment first, highest confidence wins. When boxes nest — and they do;
    cap_000463 has five omegas over two or three people — the confident one is
    the one you meant. Falling back to nearest-centre would let a distant box
    win just because nothing contains the cursor, so it does not.
    """
    cx, cy = cursor
    best = None
    for (x0, y0, x1, y1, s) in dets:
        if s < conf:
            continue
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            if best is None or s > best[4]:
                best = (x0, y0, x1, y1, s)
    return best


def already_have(boxes, cand, thresh=0.60):
    """
    True if this proposal is one you have already accepted.

    Without it, clicking a box twice silently stores two near-identical
    rectangles for the same head, and duplicates in the labels are worse than
    they look — they double that head's weight in the loss.
    """
    for b in boxes:
        if DP.iou((b[1], b[2], b[3], b[4]),
                  (cand[0], cand[1], cand[2], cand[3])) >= thresh:
            return True
    return False


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Model-in-the-loop omega annotation. Experimental.")
    ap.add_argument("capture_dir")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=0.29,
                    help="starting confidence gate; [ and ] move it live")
    ap.add_argument("--floor", type=float, default=0.01,
                    help="inference floor. Everything above this is cached, so "
                         "[ can reach it without re-running the model.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--pad", type=float, default=4.0,
                    help="sensor px of margin added around the warm blob r "
                         "scans. Some context helps; a tight crop on a head "
                         "loses the shoulders that make it an omega.")
    ap.add_argument("--min-half", type=float, default=10.0,
                    help="floor on the auto window half-width, so a small warm "
                         "fragment does not produce a crop with no context")
    ap.add_argument("--max-half", type=float, default=40.0,
                    help="cap on it, so two merged bodies do not give a window "
                         "so large the zoom stops helping")
    ap.add_argument("--crop", type=int, default=24,
                    help="FALLBACK half-width used when the cursor is not on "
                         "anything warm")
    ap.add_argument("--target", type=float, default=192.0,
                    help="render the scan window at about this many px before "
                         "inference, so apparent object size stays constant")
    ap.add_argument("--max-zoom", type=float, default=8.0)
    ap.add_argument("--max-shift", type=float, default=16.0)
    ap.add_argument("--min-corr", type=float, default=0.55)
    ap.add_argument("--dedup-iou", type=float, default=DP.DEDUP_IOU)
    args = ap.parse_args()

    root = args.capture_dir.rstrip("/")
    weights = resolve_weights(args.weights)

    tri = os.path.join(root, "triage.csv")
    if not os.path.exists(tri):
        sys.exit(f"no triage.csv in {root} — run triage.py first")
    rows = list(csv.DictReader(open(tri)))
    byc = collections.OrderedDict()
    for r in rows:
        byc.setdefault(int(r["cluster"]), []).append(r)
    clusters = list(byc)

    an = DP.load_module("annotate.py")
    human_dir = os.path.join(root, "labels_human")
    os.makedirs(human_dir, exist_ok=True)

    print(f"floor {args.floor}   starting gate {args.conf}")
    from ultralytics import YOLO
    model = YOLO(weights)

    done = sum(1 for c in clusters
               if os.path.exists(os.path.join(
                   human_dir, byc[c][0]["file"] + ".txt")))
    print(f"{len(clusters)} clusters, {done} already annotated\n")

    win = "annotate_live (experimental)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    # CORNERS ARE ON THE SPACE BAR, NOT THE MOUSE BUTTON.
    #
    # The obvious design is a mode key that swaps the left button between
    # "accept the proposal" and "place a corner". It was written that way first
    # and it was wrong: the two are meant to be available AT THE SAME TIME.
    # Half the value of drawing by hand is drawing the head the model missed
    # while still clicking the three it found, and a mode makes you leave one
    # to reach the other.
    #
    # Space resolves it with no mode at all. The cursor is already tracked for
    # the hover proposal and the crosshair already shows which sensor pixel you
    # are on, so a keypress has somewhere unambiguous to land. It is also
    # steadier than clicking: on a trackpad, pressing the button moves the
    # pointer, and at 6x zoom that is most of a sensor pixel.
    boxes = []            # the live list, mutated in place — see below
    clipboard = []        # ctrl-C / ctrl-V between frames

    state = {"cursor": (0, 0), "click": False,
             "first": None, "pending": None,
             "drag": None,      # (index, off_x, off_y, w, h) while dragging
             "cand_ok": False}  # is there a NEW proposal under the cursor?

    def box_at(p):
        """Index of the topmost accepted box containing p, or None."""
        x, y = p
        for i_ in range(len(boxes) - 1, -1, -1):   # newest first
            _, x0, y0, x1, y1 = boxes[i_]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i_
        return None

    def on_mouse(ev, mx, my, flags, _):
        p = (mx / args.scale, my / args.scale)
        state["cursor"] = p

        if ev == cv2.EVENT_LBUTTONDOWN:
            # ACCEPTING A PROPOSAL WINS OVER STARTING A DRAG.
            #
            # In a crowd a proposal frequently overlaps a box you already
            # accepted, and if drag took priority there you could never accept
            # the second of two overlapping people — the click would grab the
            # first box instead. So a click only becomes a drag when there is
            # nothing new under the cursor to accept. cand_ok is computed once
            # per redraw in the main loop and parked here.
            if state["cand_ok"]:
                state["click"] = True
                return
            hit = box_at(p)
            if hit is None:
                state["click"] = True
                return
            _, x0, y0, x1, y1 = boxes[hit]
            state["drag"] = (hit, p[0] - x0, p[1] - y0, x1 - x0, y1 - y0)

        elif ev == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
            j, ox, oy, w, h = state["drag"]
            if j < len(boxes):
                nx, ny = p[0] - ox, p[1] - oy
                boxes[j] = (OMEGA_CLASS, nx, ny, nx + w, ny + h)
            else:
                state["drag"] = None      # list shrank under us

        elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            state["drag"] = None

    cv2.setMouseCallback(win, on_mouse)


    def drop_corner():
        """Space: first press arms a corner, second completes the box."""
        p = state["cursor"]
        if state["first"] is None:
            state["first"] = p
        else:
            state["pending"] = (state["first"], p)
            state["first"] = None

    conf = args.conf
    view = [0]
    i = 0
    deleted = set()
    snapped_total = 0
    neg_total = 0

    while 0 <= i < len(clusters):
        cid = clusters[i]
        members = byc[cid]
        rep = members[0]
        arr = np.load(os.path.join(root, "npy", rep["file"] + ".npy"))
        H, W = arr.shape[:2]

        dets = infer(model, arr, args.floor, args.imgsz)

        hp = os.path.join(human_dir, rep["file"] + ".txt")
        # MUTATED IN PLACE, NEVER REBOUND. The mouse callback holds a
        # reference to this exact list so it can move a box under the cursor;
        # rebinding it per cluster would leave the callback writing into the
        # frame you just left. Every site below uses [:] / .clear() / .pop()
        # for the same reason.
        boxes[:] = ([b for b in an.load_labels(hp, W, H)
                     if b[0] == OMEGA_CLASS] if os.path.exists(hp) else [])

        S = args.scale
        advance = None
        msg = ""
        state["first"] = None      # a half-drawn box never crosses a cluster
        state["pending"] = None
        state["drag"] = None       # nor does a drag
        refreshed = set()      # coarse cells already given a second look
        while advance is None:
            cand = candidate(dets, conf, state["cursor"])
            dupe = cand is not None and already_have(boxes, cand)
            # The callback fires between redraws and cannot recompute this.
            state["cand_ok"] = bool(cand) and not dupe

            # what e/DEL would remove
            pad = DP.Pad()
            pad.boxes, pad.cursor = boxes, state["cursor"]
            doomed, how = pad.target()

            vis = cv2.resize(DP.colorize_view(arr, view[0]), None, fx=S, fy=S,
                             interpolation=cv2.INTER_NEAREST)

            above = [d for d in dets if d[4] >= conf]

            # ARMED: nothing accepted yet, but the gate is showing proposals,
            # so ENTER will take all of them. Draw them as what they are about
            # to become, not as background detail — pressing ENTER for "next"
            # and silently writing five boxes you never looked at is exactly
            # the failure this colour exists to prevent.
            # A half-placed corner means you are mid-box, so ENTER is "finish
            # what I am doing", never "accept all five proposals and move on".
            arming = ((not boxes) and bool(above)
                      and state["first"] is None)

            for (x0, y0, x1, y1, s) in above:
                if cand and (x0, y0, x1, y1) == cand[:4]:
                    continue
                col = ARM_COL if arming else DIM_COL
                cv2.rectangle(vis, (int(x0 * S), int(y0 * S)),
                              (int(x1 * S), int(y1 * S)), col,
                              2 if arming else 1)
                if arming:
                    cv2.putText(vis, f"{s:.2f}",
                                (int(x0 * S), max(12, int(y0 * S) - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, ARM_COL, 1,
                                cv2.LINE_AA)

            dragging = (state["drag"][0] if state["drag"] is not None
                        else None)
            for bi, (_, x0, y0, x1, y1) in enumerate(boxes):
                c = (DOOMED_COL if bi == doomed
                     else MANUAL_COL if bi == dragging else OMEGA_COL)
                cv2.rectangle(vis, (int(x0 * S), int(y0 * S)),
                              (int(x1 * S), int(y1 * S)), c, 2)
                if bi == doomed:
                    for px, py in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
                        cv2.drawMarker(vis, (int(px * S), int(py * S)),
                                       DOOMED_COL, cv2.MARKER_TILTED_CROSS,
                                       10, 2)
                    cv2.putText(vis, f"DEL removes ({how})",
                                (int(x0 * S), max(12, int(y0 * S) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, DOOMED_COL,
                                1, cv2.LINE_AA)

            if cand:
                x0, y0, x1, y1, s = cand
                col = (120, 200, 120) if dupe else CAND_COL
                cv2.rectangle(vis, (int(x0 * S), int(y0 * S)),
                              (int(x1 * S), int(y1 * S)), col, 2)
                tag = f"{s:.2f}" + ("  already have" if dupe else "  CLICK")
                cv2.putText(vis, tag,
                            (int(x0 * S), max(12, int(y0 * S) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1,
                            cv2.LINE_AA)

            # the window r would re-run on — sized to the blob you are on, so
            # you can see it snap to the person before you press anything
            cxp, cyp = state["cursor"]
            sw = scan_window(an, arr, state["cursor"], args)
            auto = blob_window(an, arr, state["cursor"], args.pad,
                               args.min_half, args.max_half) is not None
            cv2.rectangle(vis,
                          (int(sw[0] * S), int(sw[1] * S)),
                          (int(sw[2] * S), int(sw[3] * S)),
                          (90, 130, 90) if auto else (60, 60, 68), 1)

            # First corner placed, second not yet: mark it and rubber-band to
            # the cursor, so the box you are about to make is visible before
            # you commit to it. Same feedback as dataset_pipeline.py.
            if state["first"]:
                fx, fy = state["first"]
                cv2.drawMarker(vis, (int(fx * S), int(fy * S)), MANUAL_COL,
                               cv2.MARKER_CROSS, 16, 2)
                ux, uy = state["cursor"]
                cv2.rectangle(vis, (int(fx * S), int(fy * S)),
                              (int(ux * S), int(uy * S)), MANUAL_COL, 1)

            mx, my = int(cxp * S), int(cyp * S)
            cv2.line(vis, (mx, 0), (mx, vis.shape[0]), (90, 90, 100), 1)
            cv2.line(vis, (0, my), (vis.shape[1], my), (90, 90, 100), 1)

            n_above = len(above)
            bar = np.full((92, vis.shape[1], 3), 16, np.uint8)
            cv2.putText(bar, f"cluster {i + 1}/{len(clusters)}   "
                             f"{rep['file']}   {len(members)} frames",
                        (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (220, 220, 230), 1, cv2.LINE_AA)
            mode_txt = ("SPACE sets the opposite corner   [first corner set]"
                        if state["first"] else "")
            clip_txt = f"   clip {len(clipboard)}" if clipboard else ""
            cv2.putText(bar, f"accepted {len(boxes)}    gate {conf:.2f}  "
                             f"[ ]    proposals {n_above}/{len(dets)} "
                             f"(floor {args.floor:.2f}){clip_txt}",
                        (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        OMEGA_COL, 1, cv2.LINE_AA)


            # Say out loud what ENTER does right now. It has three outcomes and
            # they are not interchangeable; the one that writes boxes you never
            # inspected must never be the silent one.
            if arming:
                enter_txt = f"ENTER accepts all {n_above} previewed"
                enter_col = ARM_COL
            elif boxes:
                enter_txt = f"ENTER commits your {len(boxes)}"
                enter_col = OMEGA_COL
            else:
                enter_txt = "ENTER commits EMPTY (nobody here)"
                enter_col = (150, 150, 160)
            (tw, _), _ = cv2.getTextSize(enter_txt, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.46, 1)
            cv2.putText(bar, enter_txt, (bar.shape[1] - tw - 12, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, enter_col, 1,
                        cv2.LINE_AA)
            if mode_txt:
                cv2.putText(bar, mode_txt, (12, 66),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, MANUAL_COL, 1,
                            cv2.LINE_AA)

            hint = ("CLICK accept / DRAG a box   SPACE x2 corners   "
                    "^C copy  ^V paste   e/DEL remove   r re-scan   [ ] gate  "
                    "v contrast   u undo  c clear  x empty  g NEGATIVE   "
                    "ENTER next  b back  d drop  q quit")
            cv2.putText(bar, hint, (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        MANUAL_COL if state["first"] else (140, 140, 152), 1,
                        cv2.LINE_AA)
            if msg:
                (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX,
                                             0.44, 1)
                cv2.putText(bar, msg, (bar.shape[1] - tw - 12, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 220, 140),
                            1, cv2.LINE_AA)
            cv2.imshow(win, np.vstack([vis, bar]))

            if state["click"]:
                state["click"] = False
                if cand and not dupe:
                    boxes.append((OMEGA_CLASS, cand[0], cand[1],
                                  cand[2], cand[3]))

            if state["pending"]:
                (ax, ay), (bx, by) = state["pending"]
                state["pending"] = None
                # A double-click, or two clicks on the same sensor pixel, would
                # otherwise produce a zero-area box: invisible on screen, a
                # NaN-width entry in the label file, and a training sample that
                # means nothing. One sensor pixel minimum in both axes.
                if abs(bx - ax) >= 1.0 and abs(by - ay) >= 1.0:
                    boxes.append((OMEGA_CLASS, min(ax, bx), min(ay, by),
                                  max(ax, bx), max(ay, by)))
                else:
                    msg = "box too small - two OPPOSITE corners"

            k = cv2.waitKey(20) & 0xFF
            if k == 255:
                continue

            if k == ord("r"):
                win_ = scan_window(an, arr, state["cursor"], args)
                new = refresh_at(model, arr, win_, args.target,
                                 args.max_zoom, args.floor, args.imgsz)
                dets, n_new = merge_dets(dets, new)
                refreshed.add((int(state["cursor"][0] // args.crop),
                               int(state["cursor"][1] // args.crop)))
                best_new = max((d[4] for d in new), default=0.0)
                msg = (f"r: +{n_new} new, {len(new)} found here, "
                       f"best {best_new:.2f}"
                       + ("   <- press [ to see it" if n_new and
                          best_new < conf else ""))
            elif k == ord("["):
                conf = max(CONF_MIN, round(conf - CONF_STEP, 3))
            elif k == ord("]"):
                conf = min(CONF_MAX, round(conf + CONF_STEP, 3))
            elif k in DP.DELETE_KEYS and doomed is not None:
                boxes.pop(doomed)
            elif k == ord(" "):
                drop_corner()
            elif k in COPY_KEYS:
                clipboard[:] = [tuple(b) for b in boxes]
                msg = (f"copied {len(clipboard)} boxes" if clipboard
                       else "nothing to copy")
            elif k in PASTE_KEYS:
                if not clipboard:
                    msg = "clipboard empty - ctrl-C on a frame first"
                else:
                    # APPEND, not replace. Pasting onto a frame you have
                    # already started must not silently discard that work, and
                    # "paste then delete the two that do not fit" is the whole
                    # workflow this exists for. Exact duplicates are skipped so
                    # a double press cannot stack boxes invisibly on top of
                    # each other.
                    have = {tuple(b) for b in boxes}
                    added = [b for b in clipboard if tuple(b) not in have]
                    boxes.extend(added)
                    msg = (f"pasted {len(added)} of {len(clipboard)}"
                           + ("  (drag to nudge)" if added
                              else "  - already here"))
            elif k == ord("v"):
                # Display only — the model still sees render_for_cnn at SPAN_C,
                # so changing this cannot change what it proposes.
                view[0] = (view[0] + 1) % len(DP.VIEWS)
                msg = f"view: {DP.VIEWS[view[0]][0]}"
            elif k == ord("m"):
                # There WAS an m here for one revision, as a mode toggle. It
                # was the wrong design — it took the mouse button away from the
                # proposals — so corners moved to SPACE and both are live at
                # once. Anyone who learned `m` in that window gets told, rather
                # than pressing a dead key.
                msg = "corners are on SPACE now - proposals stay clickable"
            elif k == ord("u"):
                # Cancel a half-placed corner before undoing a finished box.
                # Otherwise `u` after one stray click silently removes the last
                # GOOD box and leaves the stray corner armed.
                if state["first"]:
                    state["first"] = None
                elif boxes:
                    boxes.pop()
            elif k == ord("c"):
                boxes.clear()
                state["first"] = None
            elif k in (13, 10, ord("n")):
                if arming:
                    # Nothing hand-picked, but the gate is showing proposals:
                    # take the preview as-is. This is the fast path for a frame
                    # the model already got right, which is most of them.
                    # `x` remains the way to say "genuinely nobody here" —
                    # without this branch ENTER-on-empty just duplicated it.
                    advance = ("commit", [(OMEGA_CLASS, d[0], d[1], d[2], d[3])
                                          for d in above])
                else:
                    advance = ("commit", list(boxes))
            elif k == ord("x"):
                advance = ("commit", [])
            elif k == ord("g"):
                advance = ("negative", [])
            elif k == ord("b"):
                advance = ("back", None)
            elif k == ord("d"):
                advance = ("drop", None)
            elif k in (ord("q"), 27):
                advance = ("quit", None)

        what, payload = advance
        if what in ("commit", "negative"):
            _, n = DP.commit(an, root, human_dir, members, payload, arr, H, W,
                             args.max_shift, args.min_corr, args.dedup_iou)
            snapped_total += n
            if what == "negative":
                neg_total += DP.mark_negative(root, members, cid, "hard",
                                              "annotate_live")
            i += 1
        elif what == "back":
            i = max(0, i - 1)
        elif what == "drop":
            deleted.update(m["file"] for m in members)
            i += 1
        else:
            break

    cv2.destroyAllWindows()
    print(f"\nhuman labels -> {human_dir}")
    if snapped_total:
        print(f"snapped to YOLO geometry: {snapped_total} propagated boxes. "
              f"Representatives untouched.")
    if neg_total:
        print(f"hard negatives: {neg_total} frames certified -> "
              f"{os.path.join(root, DP.NEGATIVES)}")
    if deleted:
        print(f"marked for deletion: {len(deleted)} frames "
              f"(run prune.py to remove)")
    DP.hand_back(root)


if __name__ == "__main__":
    main()
