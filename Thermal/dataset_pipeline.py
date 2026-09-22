#!/usr/bin/env python3
"""
Capture logs in, trained-ready dataset out. One command, seven stages.

    ./run.sh dataset_pipeline.py

Launch through run.sh, not bare python3. Stage 5 needs ultralytics, which lives
in Thermal/.venv — and run.sh resolves that to an absolute path rather than
trusting whichever venv happens to be active. (mmWave/.venv has numpy and cv2
but no ultralytics, so a stale activation gets you all the way to stage 5
before failing.) Deactivate any active venv first, or run.sh will prefer it
over the local one. Anything written under sudo is chowned back to you.

    1  pick capture logs          you choose, by index
    2  merge                      one working directory, provenance kept
    3  triage                     sim 0.03, max-cluster 5
    4  annotate                   omega only, every cluster
                                  --live hands this to annotate_live.py
    5  verify                     YOLO @ 0.29 audits PROPAGATED boxes only
    6  review                     worst-IoU first
    7  prune + build              cluster-level random split

Each stage can be re-entered: --from 5 resumes at verification against the
merge directory recorded in .pipeline_state.json.

WHAT THIS DOES NOT TOUCH. annotate.py and recompensate.py are imported for
their compensate()/load_labels()/save_labels() and never modified — the same
idiom recompensate.py already uses on annotate.py, so the propagation maths
cannot drift between tools. triage.py, review.py, prune.py and make_dataset.py
are invoked as subprocesses, unmodified.

THE ONE RULE THIS FILE ENFORCES. Stage 5 deletes label boxes. It only ever
deletes boxes on frames that were PROPAGATED — never on a frame a human drew.
A representative frame is human ground truth and is not auditable by a model.
That check is in `verify_propagated()` and is not optional; if the set of
hand-drawn frames cannot be determined, the stage refuses to run rather than
guess.

WHY CLUSTER-LEVEL SPLITTING. triage groups near-identical frames. A random
FRAME split therefore puts near-copies of the same moment in both train and
val, and the val score rises for a reason that has nothing to do with
generalisation. Splitting whole clusters keeps every copy of a moment on one
side of the line. This costs you a flattering number and buys you a true one.
"""

import argparse
import collections
import csv
import glob
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime

import cv2
import numpy as np

import thermal_detect as TD
from model_registry import latest_weights, resolve_weights  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
STATE = os.path.join(HERE, ".pipeline_state.json")

SPAN_C = (15.0, 45.0)
OMEGA_CLASS = 1
OMEGA_COL = (60, 220, 255)
# A YOLO box you have not touched yet. Drawn thinner and cooler than your own
# work, because on load EVERY box is one of these and they are not evidence of
# anything — they are the machine's opening bid. Pressing ENTER promotes them
# to gold labels verbatim, so "which of these have I actually looked at" is a
# question the screen has to answer.
SEED_COL = (150, 170, 120)

TRIAGE_SIM = 0.03          # default 0.10 — tighter, so propagation drifts less
TRIAGE_MAX_CLUSTER = 5     # default 30

# Minimum IoU for a PROPAGATED box to snap to a YOLO box on the same frame.
# 0.0 means ANY contact is enough; a strictly zero overlap never matches.
#
# WHY CONTACT AND NOT A REAL THRESHOLD. Inside a tight cluster the frames are
# near-identical, so a propagated box lands close to the right place but not
# exactly on it. Against a small omega — 15 px — a few pixels of drift drops
# IoU below 0.5 while the two boxes are plainly the same head. Requiring 0.5
# meant the drifted copy was kept precisely in the cases the snap existed for.
#
# THE RISK THIS ACCEPTS. At zero threshold a propagated box that merely grazes
# a spurious detection can adopt it. cap_000463 has five omega boxes for two
# or three people, one of them on wall clutter. The mitigation is that the
# match is the HIGHEST-IoU intersecting box, not the first: a graze loses to a
# real overlap. Raise --dedup-iou if over-detection starts winning anyway.
#
# WHICH FRAMES THIS TOUCHES, AND WHICH IT NEVER DOES. Cluster members only.
# The representative is what Haneef drew and is written verbatim — an earlier
# version applied this at the representative and destroyed correct boxes,
# because "compare the human box to the model" and "overwrite a hand-annotated
# frame" are the same operation there. A member's box is not his work: it is a
# copy of his box shifted by a correlation estimate that nobody has looked at.
# Asking the model about THAT is the same thing stage 5 does.
DEDUP_IOU = 0.0
VERIFY_CONF = 0.29
VERIFY_IOU = 0.30          # a YOLO detection must overlap this much to vouch

# Cursor-to-corner distance, in SENSOR pixels, that counts as hovering a corner.
# An omega box is ~15 px across, so its corners are ~15 px apart and 3 px can
# never be ambiguous between two corners of the same box. Defined in sensor
# units, not screen units, so it does not change meaning when you alter --scale.
CORNER_R = 3.0

# Keys that delete the hovered box. Several, because HighGUI does not agree
# with itself across platforms about what Backspace and Delete send — on macOS
# you may get 8 or 127 depending on the build. `e` for erase is the fallback
# that always works. Not `d`: that drops the whole cluster, and the two must
# never be one fumbled keypress apart.
DELETE_KEYS = (8, 127, ord("e"))

# Deliberate negatives: frames a human looked at and certified contain NO
# person, recorded here as a manifest rather than inferred from an empty label
# file.
#
# WHY A MANIFEST AND NOT JUST AN EMPTY .txt. YOLO's convention is that an empty
# label file means "background", which is already what `x` writes. But an empty
# file is also what you get from a frame nobody has annotated yet, from a
# cluster that was skipped, and from a bug. Three very different things with
# identical bytes on disk. If negatives are going to be used as evidence
# against false positives, they have to be distinguishable from an absence.
#
# The distinction between the two kinds matters more than it looks. v3a's
# training set contains ZERO frames without a person, so the network has never
# been shown a room and told "nothing here" — but an empty corridor teaches
# almost nothing, because nothing in it was ever going to fire. A hot 3D
# printer is the one that pays: it is the exact object the model invented a
# person from. Counting them separately is what lets you tell whether you have
# collected enough of the kind that matters.
# The frames a human actually drew on, recorded at commit time rather than
# re-derived later.
#
# review.py's --skip-drawn used to infer the representative with annotate.py's
# rule (fewest person boxes, first wins a tie) while BOTH annotators here take
# members[0] unconditionally. They agree on every capture this project records,
# because dataset_recording.py writes no person boxes so n_person is all-zero
# and min() returns the first element. They stop agreeing the moment a merge
# includes an older capture that does have person boxes — and the failure is
# silent and exactly backwards: review would hide a propagated frame and show
# you the one you drew.
REPRESENTATIVES = "representatives.txt"

NEGATIVES = "negatives.csv"
NEG_COLS = ("file", "cluster", "kind", "note", "marked")
NEG_KINDS = ("hard", "plain")


# ---------------------------------------------------------------------------
# Reuse, never fork
# ---------------------------------------------------------------------------
def load_module(name):
    """Import a sibling script as a module without modifying it on disk."""
    path = os.path.join(HERE, name)
    spec = importlib.util.spec_from_file_location("_" + name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(cmd, cwd=HERE):
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    return subprocess.call([str(c) for c in cmd], cwd=cwd)


def hand_back(path):
    """
    Return anything written under sudo to the invoking user.

    run.sh launches under sudo (the macOS kernel UVC driver claims the camera,
    so libusb needs privileges), and it chowns logs/ afterwards — but only
    logs/. Datasets land in datasets/, which it does not touch, so without this
    a dataset built through run.sh comes out root-owned and the next
    non-sudo run cannot write to it.

    Doing it here rather than editing run.sh keeps the fix with the tool that
    creates the problem, and means it still works if you launch some other way.
    """
    if os.geteuid() != 0:
        return
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if not uid:
        return
    uid, gid = int(uid), int(gid or uid)
    for root_, dirs, files in os.walk(path):
        for p in [root_] + [os.path.join(root_, f) for f in dirs + files]:
            try:
                os.chown(p, uid, gid)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 1. Choose capture logs
# ---------------------------------------------------------------------------
def capture_time(d):
    """
    Prefer the manifest's first timestamp; fall back to the directory name.

    The directory name is when recording STARTED, which is what you remember.
    But a merged or hand-renamed directory may not parse, and mtime is a lie
    after any file is touched — so the manifest wins when it exists.
    """
    man = os.path.join(d, "manifest.csv")
    if os.path.exists(man):
        try:
            with open(man) as fh:
                r = csv.DictReader(fh)
                row = next(r, None)
                if row and row.get("timestamp"):
                    return datetime.fromisoformat(row["timestamp"])
        except Exception:
            pass
    m = re.search(r"(\d{8})_(\d{6})", os.path.basename(d))
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None


def frame_paths(d):
    """
    Frames in a capture, whichever layout it uses.

    Captures recorded before the npy/png/labels split keep their .npy files
    flat in the capture directory. Matching only npy/*.npy silently omitted
    five sessions and 2847 frames, which is exactly the kind of quiet
    disappearance that gets discovered three weeks later.
    """
    sub = sorted(glob.glob(os.path.join(d, "npy", "*.npy")))
    if sub:
        return sub, "nested"
    flat = sorted(glob.glob(os.path.join(d, "*.npy")))
    return flat, ("flat" if flat else "none")


def list_captures():
    """
    Every capture directory, usable or not, with the reason when not.

    Unusable ones are LISTED rather than filtered out. A capture that silently
    fails to appear looks identical to a capture that was never recorded.
    """
    rows = []
    for d in sorted(glob.glob(os.path.join(LOG_DIR, "*"))):
        if not os.path.isdir(d):
            continue
        frames, layout = frame_paths(d)
        if not frames:
            continue
        has_labels = os.path.isdir(os.path.join(d, "labels"))
        why = ""
        if layout == "flat":
            why = "old flat layout"
        if not has_labels:
            why = (why + "; " if why else "") + "no labels/"
        rows.append({
            "dir": d,
            "name": os.path.basename(d),
            "time": capture_time(d),
            "n": len(frames),
            "layout": layout,
            "omega_only": os.path.exists(os.path.join(d, "source.txt")),
            "usable": has_labels,
            "why": why,
        })
    return rows


def show_captures(rows):
    print(f"\n{'#':>4}  {'capture log':<34} {'time':<9} {'day':<10} "
          f"{'date':<12} {'frames':>7}  src")
    print("  " + "-" * 88)
    # Only usable captures get an index, and the index is its position among
    # usable ones — so what you type is never off-by-one against what you read.
    k = 0
    for r in rows:
        t = r["time"]
        hhmm = t.strftime("%I:%M%p") if t else "--:--"
        day = t.strftime("%A") if t else "-"
        date = t.strftime("%d-%m-%Y") if t else "-"
        src = "yolo" if r["omega_only"] else "classical"
        if r["usable"]:
            idx = f"{k:>4}"
            k += 1
        else:
            idx = "   -"
        line = (f"{idx}  {r['name']:<34} {hhmm:<9} {day:<10} {date:<12} "
                f"{r['n']:>7}  {src}")
        print(line if r["usable"] else line + f"   UNUSABLE: {r['why']}")
    bad = [r for r in rows if not r["usable"]]
    if bad:
        print(f"\n  {len(bad)} capture(s) cannot be used "
              f"({sum(r['n'] for r in bad)} frames). They have no labels/ to "
              f"seed annotation from and triage has no boxes to count.")


def parse_selection(raw, n):
    """Accept '0 2 5', '0,2,5', '0-4', 'all', and any mixture."""
    if raw.strip().lower() in ("all", "*"):
        return list(range(n))
    out = []
    for tok in re.split(r"[,\s]+", raw.strip()):
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    seen, uniq = set(), []
    for i in out:
        if i in seen or not (0 <= i < n):
            continue
        seen.add(i)
        uniq.append(i)
    return uniq


def choose(rows):
    show_captures(rows)
    usable = [r for r in rows if r["usable"]]
    if not usable:
        sys.exit("no usable capture logs")
    while True:
        try:
            raw = input("\nindices to merge (e.g. 0 2 5, or 3-7, or all): ")
        except EOFError:
            sys.exit("\nno selection")
        pick = parse_selection(raw, len(usable))
        if not pick:
            print("  nothing matched — try again")
            continue
        total = sum(usable[i]["n"] for i in pick)
        print(f"\n  {len(pick)} captures, {total} frames")
        for i in pick:
            print(f"    {i:>3}  {usable[i]['name']}  ({usable[i]['n']})")
        if input("  ok? [Y/n]: ").strip().lower() in ("", "y", "yes"):
            return [usable[i] for i in pick]


# ---------------------------------------------------------------------------
# 2. Merge
# ---------------------------------------------------------------------------
def merge(sel):
    """
    Concatenate into one working directory with sequential stems.

    Stems are renumbered because cap_000000 exists in every capture; copying
    without renumbering silently overwrites. sources.csv records the mapping so
    any frame in the merged set can be traced to the session and stem it came
    from — without it, a bad frame found in review is untraceable.

    Ordering is by capture time, then original stem. triage clusters
    SEQUENTIALLY, so the order here becomes the cluster structure: frames from
    different sessions must not interleave, or triage will group across a scene
    change.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(LOG_DIR, f"merged_{stamp}")
    for sub in ("npy", "png", "labels", "review"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    sel = sorted(sel, key=lambda r: (r["time"] or datetime.min, r["name"]))

    src_rows = []
    man_rows = []
    n = 0
    for r in sel:
        d = r["dir"]
        man = {}
        mp = os.path.join(d, "manifest.csv")
        if os.path.exists(mp):
            with open(mp) as fh:
                for row in csv.DictReader(fh):
                    man[row["file"]] = row
        frames, _ = frame_paths(d)
        for npy in frames:
            old = os.path.splitext(os.path.basename(npy))[0]
            new = f"cap_{n:06d}"
            shutil.copy2(npy, os.path.join(out, "npy", new + ".npy"))
            for sub, ext in (("png", ".png"), ("labels", ".txt"),
                             ("review", ".png")):
                s = os.path.join(d, sub, old + ext)
                if os.path.exists(s):
                    shutil.copy2(s, os.path.join(out, sub, new + ext))
            src_rows.append([new, r["name"], old])
            if old in man:
                row = dict(man[old])
                row["file"] = new
                man_rows.append(row)
            n += 1
        print(f"  merged {r['name']}  ->  {n} frames total")

    with open(os.path.join(out, "classes.txt"), "w") as fh:
        fh.write("person\nhead_shoulder\n")
    with open(os.path.join(out, "sources.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "capture", "orig_file"])
        w.writerows(src_rows)
    if man_rows:
        cols = list(man_rows[0].keys())
        with open(os.path.join(out, "manifest.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for row in man_rows:
                w.writerow(row)

    print(f"\nmerged -> {out}  ({n} frames)")
    return out


# ---------------------------------------------------------------------------
# 4. Omega-only annotation
# ---------------------------------------------------------------------------
class Pad:
    """
    Same interaction model as annotate.py: TWO CLICKS, OPPOSITE CORNERS.

    Not click-and-drag. On a 160x120 sensor scaled up 6x, a drag has to be held
    accurately across ~90 screen pixels for a 15 px box, and any tremor moves
    the corner. Two independent clicks let you place each corner deliberately,
    and the crosshair shows exactly which sensor pixel you are on.

    The one difference from annotate.py: there is no class step. It has to wait
    for 1 or 2 after the second click because a box could be either class. Here
    everything is omega, so the second click commits.
    """

    def __init__(self):
        self.boxes = []      # (cls, x0, y0, x1, y1) in SENSOR pixels
        self.first = None    # first corner click
        self.cursor = (0, 0)

    def corner_hit(self, r=CORNER_R):
        """
        Index of the box with a corner under the cursor, or None.

        WHY CORNERS AND NOT JUST CONTAINMENT. Right-clicking inside a box
        deletes the topmost one containing that point, which is fine until two
        omegas overlap — then "topmost" is an ordering detail you cannot see,
        and you delete the wrong one with no indication it happened. Corners
        are unambiguous: every box has four, they are visible, and at 3 px
        they cannot be confused with a neighbour's.

        Searched newest-first so the tie-break matches the containment path.
        """
        cx, cy = self.cursor
        best, best_d = None, r
        for i in range(len(self.boxes) - 1, -1, -1):
            _, x0, y0, x1, y1 = self.boxes[i]
            for px, py in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if d <= best_d:
                    best, best_d = i, d
        return best

    def target(self):
        """
        The box DEL would remove, or None. One function, two callers.

        The renderer highlights whatever this returns and the delete key acts
        on whatever this returns, so the preview cannot promise one box and
        remove another. An earlier version highlighted corner hits only while
        the delete also fell back to containment — so a click could silently
        remove a box that was never marked.

        Corner first because it is the specific gesture; containment second so
        a box you are simply pointing at is still reachable without hunting
        for a corner.
        """
        hit = self.corner_hit()
        if hit is not None:
            return hit, "corner"
        cx, cy = self.cursor
        for i in range(len(self.boxes) - 1, -1, -1):
            _, x0, y0, x1, y1 = self.boxes[i]
            if min(x0, x1) <= cx <= max(x0, x1) and \
               min(y0, y1) <= cy <= max(y0, y1):
                return i, "inside"
        return None, None


def annotate_omega(root, scale=6, max_shift=16.0, min_corr=0.55,
                   dedup_iou=DEDUP_IOU):
    """
    One representative frame per cluster, omega class only.

    This is annotate.py's model of work — draw once, propagate to the cluster —
    with the person class removed rather than hidden. There is no key that
    produces a class-0 box, so a full-body label cannot enter this dataset by a
    slip of the finger.

    Propagation reuses annotate.compensate() by import, so the box-shifting
    maths here is the same code annotate.py runs, not a copy of it.
    """
    an = load_module("annotate.py")

    tri = os.path.join(root, "triage.csv")
    if not os.path.exists(tri):
        sys.exit(f"no triage.csv in {root} — run stage 3 first")
    rows = list(csv.DictReader(open(tri)))
    by_cluster = collections.OrderedDict()
    for r in rows:
        by_cluster.setdefault(int(r["cluster"]), []).append(r)

    human_dir = os.path.join(root, "labels_human")
    os.makedirs(human_dir, exist_ok=True)

    # Every cluster, in order. `--all` in annotate.py terms: you asked for the
    # highest n, and with omega-only captures there are no person boxes for a
    # cut-off to act on anyway.
    clusters = list(by_cluster.keys())
    done = {c for c in clusters
            if os.path.exists(os.path.join(
                human_dir, by_cluster[c][0]["file"] + ".txt"))}
    print(f"\n{len(clusters)} clusters, {len(done)} already annotated")

    pad = Pad()
    view = [0]          # list so the inner loop can rebind it
    win = "annotate (omega only)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(ev, mx, my, flags, _):
        S = scale
        pad.cursor = (mx / S, my / S)
        if ev == cv2.EVENT_LBUTTONDOWN:
            if pad.first is None:
                pad.first = (mx / S, my / S)
            else:
                x0, y0 = pad.first
                x1, y1 = mx / S, my / S
                pad.first = None
                pad.boxes.append((OMEGA_CLASS,
                                  min(x0, x1), min(y0, y1),
                                  max(x0, x1), max(y0, y1)))
        # Deliberately no right-button handler. Deletion is on the keyboard —
        # see DELETE_KEYS. On a trackpad a right-click is a two-finger tap or
        # ctrl-click, which moves the cursor at exactly the moment you have it
        # parked on a 3 px corner.

    cv2.setMouseCallback(win, on_mouse)

    i = 0
    deleted = set()
    snapped_total = 0
    neg_total = 0
    while 0 <= i < len(clusters):
        cid = clusters[i]
        members = by_cluster[cid]
        rep = members[0]
        arr = np.load(os.path.join(root, "npy", rep["file"] + ".npy"))
        H, W = arr.shape[:2]

        # The model's boxes for this frame, kept separately from the pad so a
        # box you edit can still be compared against what the model said.
        mp = os.path.join(root, "labels", rep["file"] + ".txt")
        seed_boxes = ([b for b in an.load_labels(mp, W, H)
                       if b[0] == OMEGA_CLASS] if os.path.exists(mp) else [])

        hp = os.path.join(human_dir, rep["file"] + ".txt")
        if os.path.exists(hp):
            pad.boxes = [b for b in an.load_labels(hp, W, H)
                         if b[0] == OMEGA_CLASS]
            # Already committed once, so every box here is yours — including
            # any YOLO box you accepted by pressing ENTER over it.
            untouched = set()
        else:
            # Seed from the YOLO silver label so you are correcting, not
            # starting from nothing. Class 0 is dropped on the way in.
            pad.boxes = list(seed_boxes)
            untouched = {_key(b) for b in seed_boxes}
        pad.first = None

        S = scale
        hint_msg = ""
        while True:
            vis = cv2.resize(colorize_view(arr, view[0]), None, fx=S, fy=S,
                             interpolation=cv2.INTER_NEAREST)

            # Computed ONCE per frame and used for both the highlight and the
            # delete key, so what you see and what happens cannot disagree.
            hover, how = pad.target()

            for bi, b in enumerate(pad.boxes):
                _, x0, y0, x1, y1 = b
                doomed = (bi == hover)
                seed = _key(b) in untouched
                col = ((70, 70, 255) if doomed
                       else SEED_COL if seed else OMEGA_COL)
                cv2.rectangle(vis, (int(x0 * S), int(y0 * S)),
                              (int(x1 * S), int(y1 * S)), col,
                              1 if seed and not doomed else 2)
                if doomed:
                    # mark all four corners so it is obvious WHICH box is armed,
                    # not merely that something is
                    for px, py in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
                        cv2.drawMarker(vis, (int(px * S), int(py * S)),
                                       (70, 70, 255), cv2.MARKER_TILTED_CROSS,
                                       10, 2)
                    cv2.putText(vis, f"DEL removes ({how})",
                                (int(min(x0, x1) * S),
                                 max(12, int(min(y0, y1) * S) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 255),
                                1, cv2.LINE_AA)

            # first corner placed, second not yet: mark it, and rubber-band to
            # the cursor so the box you are about to make is visible before you
            # commit to it
            if pad.first:
                cx, cy = pad.first
                cv2.drawMarker(vis, (int(cx * S), int(cy * S)), (255, 255, 255),
                               cv2.MARKER_CROSS, 16, 2)
                ux, uy = pad.cursor
                cv2.rectangle(vis, (int(cx * S), int(cy * S)),
                              (int(ux * S), int(uy * S)), (255, 255, 255), 1)

            # crosshair, because a 160x120 sensor pixel is easy to miss by one
            mx, my = int(pad.cursor[0] * S), int(pad.cursor[1] * S)
            cv2.line(vis, (mx, 0), (mx, vis.shape[0]), (90, 90, 100), 1)
            cv2.line(vis, (0, my), (vis.shape[1], my), (90, 90, 100), 1)

            bar = np.full((72, vis.shape[1], 3), 16, np.uint8)
            cv2.putText(bar, f"cluster {i + 1}/{len(clusters)}   "
                             f"{rep['file']}   {len(members)} frames", (12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 230), 1,
                        cv2.LINE_AA)
            n_seed = sum(1 for b in pad.boxes if _key(b) in untouched)
            n_mine = len(pad.boxes) - n_seed
            cv2.putText(bar, f"omega {len(pad.boxes)}   "
                             f"{n_seed} from YOLO (thin), {n_mine} yours"
                             + ("   [first corner set]" if pad.first else ""),
                        (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        OMEGA_COL, 1, cv2.LINE_AA)
            cv2.putText(bar, "L-click x2 = corners   hover + DEL/e = delete   "
                             "u undo  c clear  x empty  g NEGATIVE  "
                             "v contrast  ENTER next  b back  d drop  q quit",
                        (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (140, 140, 152), 1, cv2.LINE_AA)
            if hint_msg:
                (tw, _), _ = cv2.getTextSize(hint_msg,
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.putText(bar, hint_msg, (bar.shape[1] - tw - 12, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 200, 255), 1,
                            cv2.LINE_AA)

            cv2.imshow(win, np.vstack([vis, bar]))
            k = cv2.waitKey(20) & 0xFF

            # Keys that belong to the OTHER stage-4 tool. Pressing one here
            # used to do nothing at all, which reads as a broken feature rather
            # than as the wrong window — there is no live model in this tool to
            # have a confidence gate over, and never was.
            if k in (ord("["), ord("]"), ord("m")):
                hint_msg = "that key is --live only (annotate_live.py)"
            if k == ord("v"):
                view[0] = (view[0] + 1) % len(VIEWS)
                hint_msg = f"view: {VIEWS[view[0]][0]}"

            if k in (13, 10, ord("n")):
                # pad.boxes verbatim to the representative; members get
                # propagated and then snapped to YOLO.
                _, n = commit(an, root, human_dir, members, pad.boxes, arr,
                              H, W, max_shift, min_corr, dedup_iou)
                snapped_total += n
                i += 1
                break
            if k == ord("b"):
                i = max(0, i - 1)
                break
            if k in DELETE_KEYS and hover is not None:
                pad.boxes.pop(hover)
            if k == ord("u"):
                if pad.first:
                    pad.first = None      # cancel a half-placed box first
                elif pad.boxes:
                    pad.boxes.pop()
            if k == ord("c"):
                pad.boxes = []
                pad.first = None
            if k == ord("x"):
                commit(an, root, human_dir, members, [], arr, H, W,
                       max_shift, min_corr, dedup_iou)
                i += 1
                break
            if k == ord("g"):
                # HARD NEGATIVE. Same labels on disk as `x` — empty, which is
                # what YOLO reads as background — but additionally certified in
                # negatives.csv so it can be counted and audited. Use it on the
                # frames that look like they should fire and must not: hot
                # printers, sunlit pavement, radiators, laptops.
                commit(an, root, human_dir, members, [], arr, H, W,
                       max_shift, min_corr, dedup_iou)
                neg_total += mark_negative(root, members, cid, "hard",
                                           "annotate_omega")
                i += 1
                break
            # No 'k' (keep) key, deliberately. annotate.py needs one because it
            # opens a frame with no boxes loaded, so "accept the machine's work"
            # is a distinct action. Here the YOLO boxes are already seeded into
            # the pad on load, so pressing ENTER without editing IS accepting
            # them — and it promotes them into labels_human/ exactly the same
            # way. A second key doing the same thing would only make you wonder
            # which one you were supposed to press.
            if k == ord("d"):
                deleted.update(m["file"] for m in members)
                i += 1
                break
            if k in (ord("q"), 27):
                cv2.destroyAllWindows()
                print(f"\nstopped at cluster {i + 1}/{len(clusters)}")
                if neg_total:
                    print(f"hard negatives so far: {neg_total} frames")
                return deleted

    cv2.destroyAllWindows()
    print(f"\nhuman labels -> {human_dir}")
    if snapped_total:
        print(f"snapped to YOLO geometry: {snapped_total} propagated boxes "
              f"(IoU >= {dedup_iou}). Representatives untouched.")
    if neg_total:
        print(f"hard negatives: {neg_total} frames certified -> "
              f"{os.path.join(root, NEGATIVES)}")
    return deleted


# Display stretches for the ANNOTATION WINDOW ONLY.
#
# thermal_detect.colorize() stretches percentile 1-99, which is right for a
# room with two people in it: most of the palette goes to separating a warm
# body from a cool wall. In a lecture hall it is exactly wrong. p1 is cold
# ceiling and p99 is somebody's face, so the whole colour map is spent on a
# distinction you can already make by eye, and the one you actually need —
# this person's shoulder against the next person's shoulder — is compressed
# into a few grey levels. Measured on capture_20260921_100706: body-vs-room is
# 7.7 C out of a 18.6 C scene span.
#
# NOTHING HERE TOUCHES THE TRAINING DATA. The network is fed by
# render_for_cnn() at the fixed SPAN_C; this only changes what your eyes get.
# So you can switch it per frame with no consequence for the dataset at all.
VIEWS = (
    ("auto   p1-p99", 1.0, 99.0),       # the old behaviour, unchanged
    ("crowd  p40-p99.5", 40.0, 99.5),   # palette on the warm half only
    ("dense  p70-p100", 70.0, 100.0),   # bodies only; room clips to black
)


def colorize_view(arr, mode=0):
    """colorize(), but with the percentile window chosen by the operator."""
    _, plo, phi = VIEWS[mode % len(VIEWS)]
    lo = float(np.percentile(arr, plo))
    hi = float(np.percentile(arr, phi))
    if hi - lo < 1e-3:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)


def _key(b):
    """Geometry key for 'is this still exactly the box YOLO gave me'."""
    return tuple(round(float(v), 4) for v in b[1:5])


def note_representative(root, stem):
    """Record that a human drew this frame. Idempotent."""
    path = os.path.join(root, REPRESENTATIVES)
    have = set()
    if os.path.exists(path):
        have = {ln.strip() for ln in open(path) if ln.strip()}
    if stem in have:
        return
    with open(path, "a") as fh:
        fh.write(stem + "\n")


def read_representatives(root):
    path = os.path.join(root, REPRESENTATIVES)
    if not os.path.exists(path):
        return set()
    return {ln.strip() for ln in open(path) if ln.strip()}


def _read_negatives(root):
    path = os.path.join(root, NEGATIVES)
    if not os.path.exists(path):
        return {}
    return {r["file"]: r for r in csv.DictReader(open(path))}


def _write_negatives(root, rows):
    with open(os.path.join(root, NEGATIVES), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(NEG_COLS)
        for f in sorted(rows):
            w.writerow([rows[f].get(c, "") for c in NEG_COLS])


def mark_negative(root, members, cid, kind="hard", note=""):
    """
    Certify every frame in a cluster as containing no person.

    Keyed by file, so marking the same cluster twice updates rather than
    duplicating — you can go back with `b` and change your mind.
    """
    rows = _read_negatives(root)
    stamp = datetime.now().isoformat(timespec="seconds")
    for m in members:
        rows[m["file"]] = {"file": m["file"], "cluster": str(cid),
                           "kind": kind, "note": note, "marked": stamp}
    _write_negatives(root, rows)
    return len(members)


def unmark_negative(root, members):
    """
    Drop a cluster from the negative manifest.

    Called whenever a commit writes at least one box. Without this, marking a
    cluster negative and then going back and annotating it leaves a manifest
    that says "certified no person" about frames that now carry a person box —
    and the manifest is the thing downstream tools are meant to trust.
    """
    rows = _read_negatives(root)
    gone = [m["file"] for m in members if m["file"] in rows]
    if not gone:
        return 0
    for f in gone:
        rows.pop(f)
    _write_negatives(root, rows)
    return len(gone)


def snap_to_yolo(propagated, yolo, thresh=DEDUP_IOU):
    """
    Replace a propagated box with the YOLO box describing the same object.

    A propagated box is your box from the representative, translated by a
    normalised-correlation estimate. On the member frame the model has its own,
    independently derived detection of that head. Where the two agree on WHICH
    object they mean, the model's box is the better estimate of WHERE it is on
    this particular frame — it looked at this frame; the propagated copy only
    inferred it.

    THE TRAP. Building the result as "all YOLO boxes, plus propagated boxes
    that matched nothing" RESURRECTS every false positive you deleted at the
    representative. You delete a radiator once; YOLO still finds it on all four
    members, and it has no propagated counterpart — indistinguishable from a
    box you never touched. Worse than at the representative, because one
    deletion silently comes back N times.

    So the result is built from the PROPAGATED list, never YOLO's:

        matched   -> emit YOLO's box   (better geometry, same object)
        unmatched -> emit the propagated box

    A YOLO box matching nothing propagated is something you removed. It is
    never emitted.

    Returns (boxes, n_snapped).
    """
    out, snapped, used = [], 0, set()
    for pb in propagated:
        best, best_i = 0.0, None
        for i, yb in enumerate(yolo):
            if i in used:
                continue
            v = iou((pb[1], pb[2], pb[3], pb[4]), (yb[1], yb[2], yb[3], yb[4]))
            if v > best:
                best, best_i = v, i
        # `best > 0` is separate from the threshold on purpose: at thresh 0.0
        # the rule is CONTACT, and two boxes that do not touch have IoU exactly
        # 0.0, which would otherwise satisfy `>= 0.0` and match everything.
        if best_i is not None and best > 0.0 and best >= thresh:
            out.append(yolo[best_i])
            used.add(best_i)
            snapped += 1
        else:
            out.append(pb)
    return out, snapped


def commit(an, root, human_dir, members, boxes, ref_arr, H, W,
           max_shift, min_corr, dedup_iou=DEDUP_IOU):
    """
    Write the representative, then propagate to the rest of the cluster.

    The representative is written verbatim — it is what you drew. Every other
    member gets each box shifted by how far ITS OWN subject moved, per box, not
    per frame: two people in one cluster move independently.

    compensate() returns (new_box, distance), or None. On None the box is kept
    verbatim — which is what annotate.py does, and what you want.

    DO NOT READ `unshifted` AS A FAILURE RATE. None has six causes and only
    three are failures (patch under 3x3, weak correlation, peak on the search
    edge). The other three are successes: the shift exceeded max_shift, the box
    would leave the frame, or — overwhelmingly the common one — `d < 0.5`,
    meaning the best match sat where the box already was.

    Measured over 32 propagated omega boxes at sim 0.03: 84.4% `d < 0.5`,
    15.6% genuinely shifted, and ZERO failures of any kind. That is the tight
    clustering working. An earlier version of this docstring called all of them
    refusals and made a 66% success rate look like a 66% failure rate.
    """
    # THE REPRESENTATIVE, VERBATIM. Written before anything else and never
    # passed through snap_to_yolo. What you drew is what is stored.
    rep = members[0]
    an.save_labels(os.path.join(human_dir, rep["file"] + ".txt"), boxes, W, H)
    note_representative(root, rep["file"])

    # Any box at all means this cluster is not a negative any more.
    if boxes:
        unmark_negative(root, members)

    unshifted = snapped = 0
    for m in members[1:]:
        tgt = np.load(os.path.join(root, "npy", m["file"] + ".npy"))
        moved = []
        for b in boxes:
            try:
                r = an.compensate(b, ref_arr, tgt, H, W, max_shift, min_corr)
            except Exception:
                r = None
            if r is None:
                moved.append(b)
                unshifted += 1
            else:
                moved.append(r[0])

        # Members only: swap each propagated copy for the model's own detection
        # of the same object ON THIS FRAME, where they agree it is the same
        # object. Deletions are preserved — see snap_to_yolo.
        mp = os.path.join(root, "labels", m["file"] + ".txt")
        ylabels = ([b for b in an.load_labels(mp, W, H) if b[0] == OMEGA_CLASS]
                   if os.path.exists(mp) else [])
        moved, n = snap_to_yolo(moved, ylabels, dedup_iou)
        snapped += n

        an.save_labels(os.path.join(human_dir, m["file"] + ".txt"),
                       moved, W, H)
    return unshifted, snapped


# ---------------------------------------------------------------------------
# 5. YOLO verification of PROPAGATED boxes only
# ---------------------------------------------------------------------------
def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def representatives(root):
    """
    The frames a human actually drew on: first member of each cluster.

    Derived from triage.csv, the same way annotate.py picks them. If triage.csv
    is missing this returns None and verification refuses to run — deleting
    boxes without knowing which frames are hand-drawn is how you destroy a
    week of annotation.
    """
    tri = os.path.join(root, "triage.csv")
    if not os.path.exists(tri):
        return None
    first = {}
    for r in csv.DictReader(open(tri)):
        c = int(r["cluster"])
        if c not in first:
            first[c] = r["file"]
    return set(first.values())


def verify_propagated(root, weights, conf=VERIFY_CONF, min_iou=VERIFY_IOU,
                      imgsz=640, apply=False):
    """
    Audit propagated boxes against the model; drop the ones it will not vouch for.

    A propagated box is a machine product: annotate.py copied it from a
    neighbouring frame and shifted it by a correlation estimate. Nobody looked
    at it. So it is fair to ask a second opinion.

    A box on a REPRESENTATIVE frame is not a machine product. It is what Haneef
    drew, and it is ground truth by definition — no model gets a vote on it.
    Those frames are skipped, always.

    The model runs at a floor of 0.01 so every candidate has a score, and a
    propagated box is kept if some detection overlapping it by >= min_iou
    scores >= conf. No overlap at all counts as unsupported.
    """
    reps = representatives(root)
    if reps is None:
        sys.exit("no triage.csv — refusing to verify without knowing which "
                 "frames are hand-drawn")

    an = load_module("annotate.py")
    from ultralytics import YOLO
    model = YOLO(weights)

    human_dir = os.path.join(root, "labels_human")
    files = sorted(glob.glob(os.path.join(human_dir, "*.txt")))
    print(f"\nverifying {len(files)} label files "
          f"({len(reps)} hand-drawn frames will be skipped)")
    print(f"model {weights}   conf {conf}   min IoU {min_iou}")

    kept = dropped = skipped = 0
    per_frame = []
    for i, f in enumerate(files):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem in reps:
            skipped += 1
            continue
        arr = np.load(os.path.join(root, "npy", stem + ".npy"))
        H, W = arr.shape[:2]
        boxes = an.load_labels(f, W, H)
        if not boxes:
            continue

        lo, hi = SPAN_C
        v = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        img = cv2.merge([(v * 255).astype(np.uint8)] * 3)
        res = model.predict(img, verbose=False, conf=0.01, imgsz=imgsz)[0]

        dets = []
        if len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cf = res.boxes.conf.cpu().numpy()
            for (x0, y0, x1, y1), c, s in zip(xyxy, cls, cf):
                if c == OMEGA_CLASS:
                    dets.append((float(x0), float(y0), float(x1), float(y1),
                                 float(s)))

        survivors = []
        for b in boxes:
            best = 0.0
            for (dx0, dy0, dx1, dy1, s) in dets:
                if iou((b[1], b[2], b[3], b[4]), (dx0, dy0, dx1, dy1)) >= min_iou:
                    best = max(best, s)
            if best >= conf:
                survivors.append(b)
                kept += 1
            else:
                dropped += 1
        if len(survivors) != len(boxes):
            per_frame.append((stem, len(boxes), len(survivors)))
        if apply:
            an.save_labels(f, survivors, W, H)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)}")

    print(f"\n{'APPLIED' if apply else 'DRY RUN'}")
    print(f"  hand-drawn frames skipped   {skipped}")
    print(f"  propagated boxes kept       {kept}")
    print(f"  propagated boxes dropped    {dropped}")
    if kept + dropped:
        print(f"  drop rate                   "
              f"{100.0 * dropped / (kept + dropped):.1f}%")
    if per_frame[:10]:
        print("\n  first frames changed:")
        for stem, a, b in per_frame[:10]:
            print(f"    {stem}  {a} -> {b}")
    if not apply:
        print("\n  re-run with --apply to write these deletions")
    return dropped


# ---------------------------------------------------------------------------
# 7. Split + build
# ---------------------------------------------------------------------------
def apply_to_delete(root):
    """
    Bridge review.py's verdict to prune.py's mechanism. They do not share one.

    THE BUG THIS FIXES. review.py records rejections as stems in to_delete.txt
    and explicitly does not remove anything. prune.py takes review/ as the KEEP
    list and drops `npy - review`, i.e. it expects you to have deleted the QA
    PNGs by hand. Run back to back, review marks 679 frames and prune reports
    "to drop: 0 frames (0.0%)" — no error, no warning, and a dataset built from
    every frame you just rejected.

    That is what happened to merged_20260910_021320: 679 deletions marked,
    1854 frames in the dataset.

    Deleting the review/ PNG for each rejected stem is the missing step. It
    uses prune.py's own contract rather than working around it, so prune stays
    the one thing that removes npy/png/labels.

    Returns the number of frames now queued for pruning.
    """
    td = os.path.join(root, "to_delete.txt")
    rev = os.path.join(root, "review")
    if not os.path.exists(td):
        return 0
    if not os.path.isdir(rev):
        print(f"  to_delete.txt has entries but there is no review/ — "
              f"prune.py cannot act. Skipping.")
        return 0

    stems = [ln.strip() for ln in open(td) if ln.strip()]
    gone = 0
    for s in stems:
        p = os.path.join(rev, s + ".png")
        if os.path.exists(p):
            try:
                os.remove(p)
                gone += 1
            except OSError as e:
                print(f"  could not remove {p}: {e}")
    print(f"  to_delete.txt: {len(stems)} rejected, {gone} review/ entries "
          f"removed -> prune.py will now see them")
    return gone


def build(root, out, val_fraction, seed):
    """
    Cluster-level random split.

    --split-by cluster is the whole point: see the module docstring. --seed is
    recorded so the split is reproducible; a dataset you cannot rebuild is a
    dataset you cannot debug.

    --train-source both: gold (human) where it exists, silver (YOLO) elsewhere.
    You asked for these not to be isolated, on the grounds that YOLO has earned
    it against the classical detector. Note this applies to TRAIN only —
    make_dataset.py builds validation from labels_human/ regardless, which is
    what keeps the val number honest.
    """
    return run([sys.executable, "make_dataset.py", root,
                "--out", out,
                "--val-fraction", val_fraction,
                "--split-by", "cluster",
                "--seed", seed,
                "--train-source", "both",
                "--lo", SPAN_C[0], "--hi", SPAN_C[1]])


# ---------------------------------------------------------------------------
def save_state(root, live=None):
    """
    Remember the merge directory AND which annotator was chosen.

    WHY --live IS PERSISTED. It used not to be, and that was a trap: you run
    stage 4 with --live, learn its keys, stop for the night, resume tomorrow
    with `--from 4`, and land silently in the corner annotator instead. Same
    window title prefix, same frame, but SPACE does nothing, there is no
    confidence gate, and clicking draws a corner rather than accepting a
    proposal. It reads exactly like the tool reverted. Reported 2026-09-21.
    """
    prev = _load_state_raw()
    d = {"root": root, "live": prev.get("live") if live is None else bool(live)}
    json.dump(d, open(STATE, "w"))


def _load_state_raw():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE)) or {}
        except Exception:
            return {}
    return {}


def load_state():
    return _load_state_raw().get("root")


def load_state_live():
    return bool(_load_state_raw().get("live"))


def main():
    ap = argparse.ArgumentParser(
        description="Capture logs -> reviewed, split dataset.")
    ap.add_argument("--from", dest="start", type=int, default=1,
                    choices=range(1, 8), metavar="1-7",
                    help="resume at this stage against the last merge dir")
    ap.add_argument("--root", default=None,
                    help="merge directory to resume against "
                         "(default: last one this tool made)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=VERIFY_CONF)
    ap.add_argument("--sim", type=float, default=TRIAGE_SIM)
    ap.add_argument("--max-cluster", type=int, default=TRIAGE_MAX_CLUSTER)
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--val-fraction", type=float, default=None,
                    help="held-out fraction. Omitted: you are asked.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="dataset output directory")
    ap.add_argument("--no-live", dest="no_live", action="store_true",
                    help="force the corner annotator even if the last run "
                         "used --live. Without this, --live is remembered "
                         "across resumes.")
    ap.add_argument("--live", action="store_true",
                    help="stage 4: use annotate_live.py — hover a person and "
                         "the model proposes the box, click to accept. Press "
                         "SPACE twice draws a box by hand when the model "
                         "cannot see a person at any confidence; the "
                         "proposals stay live while you do.")
    ap.add_argument("--live-conf", type=float, default=0.29,
                    help="with --live, the starting confidence gate. [ and ] "
                         "move it during annotation.")
    ap.add_argument("--dedup-iou", type=float, default=DEDUP_IOU,
                    help="IoU at which a PROPAGATED box is replaced by the "
                         "YOLO box on that frame. 0 disables snapping. "
                         "Never applies to representatives.")
    ap.add_argument("--review-drawn", action="store_true",
                    help="also review the frames you hand-annotated. Off by "
                         "default: they are ground truth, not candidates.")
    ap.add_argument("--review-clusters", action="store_true",
                    help="review one frame per cluster instead of every frame. "
                         "Faster, but a propagated box is then never looked at "
                         "on the frame it actually landed on.")
    args = ap.parse_args()

    # NOT required=True. Only stage 5 (verify) and --live need a model at all;
    # stages 1-4 and 6-7 never touch one. Resolving hard here killed the whole
    # run before stage 1 on any machine without models/vN/best.pt — which is
    # every collaborator's machine, since the weights are not in the repo. The
    # stages that genuinely need it say so when they are reached.
    weights = resolve_weights(args.weights, required=False)

    # --live sticks across resumes unless you say otherwise, so `--from 4`
    # tomorrow gives you the same annotator you used today.
    if args.no_live:
        args.live = False
    elif not args.live and load_state_live():
        args.live = True
        print("note: using --live (remembered from the last run). "
              "Pass --no-live for the corner annotator.")

    root = args.root or load_state()
    if args.start > 2 and not root:
        sys.exit("nothing to resume — run from stage 1")

    # Record the choice now, not only at stage 2. Resuming with
    # `--from 4 --live` has to stick too, or the next resume forgets again.
    if root:
        save_state(root, live=args.live)

    # ---- 1 + 2 -----------------------------------------------------------
    if args.start <= 2:
        rows = list_captures()
        if not rows:
            sys.exit(f"no capture logs in {LOG_DIR}")
        sel = choose(rows)
        print("\n" + "=" * 60 + "\nSTAGE 2  merge\n" + "=" * 60)
        root = merge(sel)
        hand_back(root)
        save_state(root, live=args.live)

    print(f"\nworking directory: {root}")

    # ---- 3 triage --------------------------------------------------------
    if args.start <= 3:
        print("\n" + "=" * 60 + "\nSTAGE 3  triage\n" + "=" * 60)
        print(f"sim {args.sim} / max-cluster {args.max_cluster} — tighter than "
              f"the 0.10/30 default, so frames inside a cluster are "
              f"near-identical and propagation barely has to move a box.")
        print("Captures from dataset_recording.py carry no person boxes, so "
              "triage's box histogram will read all-zero and every cluster "
              "gets flagged. That is intended: you are annotating all of it.")
        run([sys.executable, "triage.py", root,
             "--sim", args.sim, "--max-cluster", args.max_cluster,
             "--max-boxes", 0])

    # ---- 4 annotate ------------------------------------------------------
    if args.start <= 4:
        mode = ("live, model-in-the-loop (+ SPACE for manual corners)"
                if args.live else "corner drawing")
        print("\n" + "=" * 60 +
              f"\nSTAGE 4  annotate (omega only — {mode})\n" + "=" * 60)
        if args.live and not weights:
            sys.exit("--live needs a model: it proposes boxes with it. "
                     "Put one at models/vN/best.pt, pass --weights, or drop "
                     "--live and draw the corners by hand.")
        if args.live:
            # Subprocess, not an import: annotate_live.py imports THIS module
            # for commit()/snap_to_yolo()/Pad, so importing it back would be a
            # cycle. A subprocess also means a crash in the experimental tool
            # cannot take the pipeline down mid-run.
            cmd = [sys.executable, "annotate_live.py", root,
                   "--scale", args.scale,
                   "--conf", args.live_conf,
                   "--dedup-iou", args.dedup_iou,
                   "--max-shift", 16.0, "--min-corr", 0.55]
            if args.weights:
                cmd += ["--weights", args.weights]
            if run(cmd) != 0:
                print("\nannotate_live.py exited non-zero. Stopping here so "
                      "you can inspect labels_human/ rather than pressing on "
                      "into verification with a half-annotated set.")
                return
        else:
            annotate_omega(root, scale=args.scale, dedup_iou=args.dedup_iou)
        hand_back(root)      # annotation is long; don't leave it root-owned

    # ---- 5 verify --------------------------------------------------------
    if args.start <= 5:
        print("\n" + "=" * 60 + "\nSTAGE 5  verify propagated boxes\n" + "=" * 60)
        if not weights:
            # A cross-check, not a producer of labels. Skipping it costs you
            # the second opinion on propagated boxes; it does not invalidate
            # anything already in labels_human/. Better than refusing to build
            # a dataset at all on a machine with no model.
            print("SKIPPED - no model. This stage only re-checks propagated\n"
                  "  boxes against YOLO; your human labels are unaffected.\n"
                  "  Put one at models/vN/best.pt and re-run with --from 5\n"
                  "  to get the cross-check.")
        else:
            n = verify_propagated(root, weights, conf=args.conf, apply=False)
            if n:
                ans = input(f"\napply {n} deletions? [y/N]: ").strip().lower()
                if ans in ("y", "yes"):
                    verify_propagated(root, weights, conf=args.conf,
                                      apply=True)
                    hand_back(root)
                else:
                    print("  left unchanged")

    # ---- 6 review --------------------------------------------------------
    if args.start <= 6:
        print("\n" + "=" * 60 + "\nSTAGE 6  review\n" + "=" * 60)
        print("Ordered worst-first: frames with a warm blob and no box sort "
              "ahead of everything, then ascending box-vs-blob IoU.")
        cmd = [sys.executable, "review.py", root, "--scale", args.scale]
        if not args.review_drawn and not args.review_clusters:
            # The frames you drew on are not on trial. Reviewing them is asking
            # a tool that scores box-vs-blob overlap to second-guess a human
            # who looked at the frame — and its MISS check fires on any warm
            # surface, so it will disagree with you about frames that are fine.
            cmd.append("--skip-drawn")
        if not args.review_clusters:
            # EVERY frame, not one per cluster. review.py defaults to showing a
            # cluster's worst frame as its proxy, which is the right economy
            # when a cluster is a verbatim copy. It is the wrong economy here:
            # compensation SHIFTS each propagated box independently, so the
            # frames differ from one another and a box can be wrong on frame 11
            # while the cluster's proxy frame looks fine.
            cmd.append("--frames")
        run(cmd)

    # ---- 7 prune + build -------------------------------------------------
    if args.start <= 7:
        print("\n" + "=" * 60 + "\nSTAGE 7  prune + build\n" + "=" * 60)
        apply_to_delete(root)
        run([sys.executable, "prune.py", root])
        if input("apply prune? [y/N]: ").strip().lower() in ("y", "yes"):
            run([sys.executable, "prune.py", root, "--apply"])
        else:
            # Saying no here means the rejected frames stay in the dataset.
            # That is a legitimate choice, but not a quiet one.
            n = sum(1 for ln in open(os.path.join(root, "to_delete.txt"))
                    if ln.strip()) if os.path.exists(
                        os.path.join(root, "to_delete.txt")) else 0
            if n:
                print(f"\n  NOT pruned. {n} frames you rejected in review will "
                      f"be included in the dataset.")

        vf = args.val_fraction
        if vf is None:
            raw = input("\ntrain:val ratio — val fraction [0.20]: ").strip()
            vf = float(raw) if raw else 0.20
        out = args.out or os.path.join(
            HERE, "datasets", f"ds_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"\nsplit  train {100 * (1 - vf):.0f} : val {100 * vf:.0f}   "
              f"by CLUSTER, seed {args.seed}")
        build(root, out, vf, args.seed)
        hand_back(out)
        hand_back(root)
        print(f"\ndataset -> {out}")


if __name__ == "__main__":
    main()
