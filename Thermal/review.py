#!/usr/bin/env python3
"""
Judge the human labels worst-first, and stop when they stop being wrong.

Flipping through 4000 frames in filename order spends your attention uniformly
on a problem that is not uniform. Measured on capture_20260824_152959, of 431
annotated clusters:

     87  a person-sized warm blob with no box on it   (MISS)
      7  a box touching no warm pixels at all         (IoU 0.00)
     27  a box scoring below 0.20
    ---
    121  worth opening (28%)   |   310 the score can find no fault with

Ordering by badness turns a five-hour census into a forty-minute pass. It does
not turn it into no work: 121 items is 121 judgements, and the ranking only
guarantees you meet them worst-first, not that you can stop early.

    python3 review.py logs/capture_X
    python3 review.py logs/capture_X --limit 50        # worst 50 only
    python3 review.py logs/capture_X --min-score 0.20  # only the 121 flagged
    python3 review.py logs/capture_X --frames          # every frame, not clusters
    python3 review.py logs/capture_X --frames --skip-drawn   # only the copies

CLUSTERS, NOT FRAMES, BY DEFAULT. You only ever DREW 431 times; the other 3295
labels are byte-identical copies. Reviewing them one by one re-judges the same
431 decisions twelve times over. Cluster mode shows the WORST-scoring frame in
each cluster — if the box survives its hardest frame it survives the rest — and
your verdict applies to the whole cluster. Use --frames when you have reason to
believe a cluster is not uniform.

WHAT THE SCORE MEANS. Two different errors, one ordering:

  * a box on the wrong thing -> IoU of the box against the best-overlapping
    warm blob. 0.00 means the box touches no warm pixels at all.
  * a person with no box     -> a person-sized warm blob covered by no human
    box. IoU cannot see this, because a missing box has no score to be bad;
    it is the same survivorship bias that made the machine labels untrustworthy.

Frames with a MISS sort first, ahead of even the zero-IoU boxes, because a
missing person poisons a detector worse than a sloppy rectangle does.

IoU IS NOT A GRADE. Your box encloses the visible person; a warm blob is often
a fragment of them, or two people merged. Median IoU runs ~0.37 on labels that
are perfectly good. Read the ORDER, not the number — the ranking is meaningful
even though the absolute value is not.

Keys
    ENTER / n     approve, next
    x             mark for deletion
    b             back one, and un-decide it
    u             undo the last decision, stay here
    g             toggle warm-blob outlines (see WHY the score is low)
    q             save and quit

Nothing is deleted. Decisions go to review.csv and the deletions to
to_delete.txt, both rewritten after every keypress so a crash costs nothing.
"""

import argparse
import collections
import csv
import glob
import os
import sys

import cv2
import numpy as np


CLASS_NAMES = ["person", "omega"]
CLASS_COLOR = [(90, 255, 120), (60, 220, 255)]

DELTA, TMAX = 2.0, 36.0
BLOB_MIN = 25          # px, below this a warm component is noise
MISS_MIN = 120         # px, a warm blob this big with no box is a probable miss
MISS_COVER = 0.25      # a blob is "boxed" if this much of it falls in some box
MISS_ASPECT = 0.28     # h/w below this is a ceiling band or a pipe, not a person


def build_background(npys, sample=150):
    """
    Per-pixel temporal median: what this room looks like with nobody in it.

    A scene-median threshold cannot tell a person from a radiator, because both
    are simply "warmer than average". Measured on capture_20260824_152959, 2.6%
    of every frame is permanently above ambient — a 134px strip down the left
    edge, a 109px band across the top — and a naive MISS check rediscovered
    those fixtures in all 3726 frames, a 100% false-positive rate.

    A person moves; a pipe does not. Taking the median over frames spread
    across the whole capture leaves the fixtures in and the people out, so
    subtracting it leaves only what moved.
    """
    if not npys:
        return None
    step = max(1, len(npys) // sample)
    stack = []
    for f in npys[::step][:sample]:
        try:
            stack.append(np.load(f).astype(np.float32))
        except Exception:
            continue
    if len(stack) < 5:
        return None
    return np.median(np.stack(stack, 0), 0)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def warm_mask(arr):
    """
    Absolute threshold against scene ambient — deliberately NOT background
    subtracted.

    Subtracting the temporal background looks like the obvious upgrade and is a
    trap here: a person seated at a desk for most of the capture IS their own
    background, so subtraction deletes them. Measured, it turned 1842 perfectly
    good labels into "box on empty space". The background is still useful, but
    as a VETO on false alarms (see fixture_mask), never as the threshold.
    """
    amb = float(np.median(arr))
    m = ((arr >= amb + DELTA) & (arr <= TMAX)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def fixture_mask(bg):
    """Pixels that are warm in EVERY frame: radiators, vents, light fittings."""
    if bg is None:
        return None
    return (bg >= float(np.median(bg)) + DELTA)


def warm_blobs(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < BLOB_MIN:
            continue
        x, y, w, h = (st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP],
                      st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
        out.append({"box": [x, y, x + w, y + h],
                    "area": int(st[i, cv2.CC_STAT_AREA]), "id": i})
    return out, lab


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) +
                   (b[2] - b[0]) * (b[3] - b[1]) - i)


def load_labels(path, W, H):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        p = line.split()
        if len(p) != 5:
            continue
        try:
            c = int(p[0])
            cx, cy, bw, bh = (float(v) for v in p[1:])
        except ValueError:
            continue
        out.append((c, (cx - bw / 2) * W, (cy - bh / 2) * H,
                    (cx + bw / 2) * W, (cy + bh / 2) * H))
    return out


def score_frame(arr, boxes, fix=None, miss_min=MISS_MIN, do_miss=True):
    """
    Return (score, reason, blobs). Lower score = look at this sooner.

    Person boxes only. Omega boxes describe head-and-shoulders, a deliberate
    sub-region of the body, so they score low against a full-body blob by
    construction — including them would bury the real errors under 60% noise.
    (Measured: scoring all classes flagged 62% of clusters; scoring person
    boxes only flagged 9%, and the 9% were real.)
    """
    H, W = arr.shape[:2]
    mask = warm_mask(arr)
    blobs, lab = warm_blobs(mask)
    people = [b[1:] for b in boxes if b[0] == 0]

    # --- error 1: a person-sized warm blob that no box covers ---------------
    for bl in (blobs if do_miss else []):
        if bl["area"] < miss_min:
            continue
        bx0, by0, bx1, by1 = bl["box"]
        bw, bh = max(1, bx1 - bx0), max(1, by1 - by0)
        # a wide flat smear is a vent or a light fitting; people are not 4:1
        if bh / bw < MISS_ASPECT:
            continue
        px = (lab == bl["id"])
        tot = int(px.sum())
        if tot == 0:
            continue
        # VETO: mostly sitting on something that is warm in every frame, so it
        # is furniture, not an unlabelled person. Without this the check found
        # the same radiator in all 3726 frames — a 100% false-positive rate.
        if fix is not None and int((px & fix).sum()) / tot > 0.50:
            continue
        covered = 0
        for (x0, y0, x1, y1) in people:
            ax0, ay0 = max(0, int(min(x0, x1))), max(0, int(min(y0, y1)))
            ax1, ay1 = min(W, int(max(x0, x1)) + 1), min(H, int(max(y0, y1)) + 1)
            if ax1 <= ax0 or ay1 <= ay0:
                continue
            covered = max(covered, int(px[ay0:ay1, ax0:ax1].sum()))
        if covered / tot < MISS_COVER:
            return -1.0, f"MISS: {bl['area']}px blob unboxed", blobs

    # --- error 2: a box that does not sit on a warm body --------------------
    if not people:
        return 2.0, "empty label, nothing warm unboxed", blobs
    worst, wb = 1.0, None
    for p in people:
        best = max((iou(p, bl["box"]) for bl in blobs), default=0.0)
        if best < worst:
            worst, wb = best, p
    if worst == 0.0:
        return 0.0, "box on empty space", blobs
    return worst, f"worst box IoU {worst:.2f}", blobs


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def colorize(a):
    lo, hi = float(np.percentile(a, 1)), float(np.percentile(a, 99))
    if hi - lo < 1e-3:
        hi = lo + 1.0
    n = np.clip((a - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def render(arr, boxes, blobs, S, show_blobs, header, sub, colour):
    H, W = arr.shape[:2]
    vis = cv2.resize(colorize(arr), (W * S, H * S),
                     interpolation=cv2.INTER_NEAREST)
    if show_blobs:
        for bl in blobs:
            x0, y0, x1, y1 = bl["box"]
            cv2.rectangle(vis, (int(x0 * S), int(y0 * S)),
                          (int(x1 * S), int(y1 * S)), (70, 70, 90), 1)
    for c, x0, y0, x1, y1 in boxes:
        col = CLASS_COLOR[c] if c < len(CLASS_COLOR) else (200, 200, 200)
        p0 = (int(min(x0, x1) * S), int(min(y0, y1) * S))
        p1 = (int(max(x0, x1) * S), int(max(y0, y1) * S))
        cv2.rectangle(vis, p0, p1, col, 2)
        nm = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
        for th, cc in ((3, (0, 0, 0)), (1, col)):
            cv2.putText(vis, nm, (p0[0], max(12, p0[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, cc, th, cv2.LINE_AA)
    bar = np.full((86, vis.shape[1], 3), 16, np.uint8)
    cv2.putText(bar, header, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (225, 225, 235), 1, cv2.LINE_AA)
    cv2.putText(bar, sub, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                colour, 1, cv2.LINE_AA)
    cv2.putText(bar, "ENTER approve   x delete   b back   u undo   "
                     "g blobs   q quit", (12, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 152), 1, cv2.LINE_AA)
    return np.vstack([vis, bar])


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--scale", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None,
                    help="review only the worst N items and stop")
    ap.add_argument("--frames", action="store_true",
                    help="one item per FRAME instead of per cluster")
    ap.add_argument("--min-score", type=float, default=None,
                    help="skip anything scoring at or above this, e.g. 0.20")
    ap.add_argument("--skip-drawn", action="store_true",
                    help="with --frames, omit the one frame per cluster you "
                         "actually drew on, leaving only the propagated "
                         "copies. 3726 -> 3295 on capture_20260824_152959.")
    ap.add_argument("--no-miss", action="store_true",
                    help="disable MISS detection and rank by IoU alone. The "
                         "MISS check is the noisy half of this tool: it fires "
                         "on any large warm surface the fixture veto did not "
                         "catch — sunlit walls, equipment that warms up "
                         "mid-capture. Use this if it is wasting your time.")
    ap.add_argument("--miss-min", type=int, default=MISS_MIN,
                    help=f"px area a warm blob must reach to count as a "
                         f"possible missed person (default {MISS_MIN}). Raise "
                         f"it to cut false alarms, lower it to catch distant "
                         f"people.")
    ap.add_argument("--restart", action="store_true",
                    help="discard previous decisions and start over")
    args = ap.parse_args()

    root = args.capture_dir.rstrip("/")
    hd = os.path.join(root, "labels_human")
    if not os.path.isdir(hd):
        sys.exit(f"no labels_human/ in {root} — nothing to review")

    # HUMAN LABELS ONLY. labels/ is the machine's and is not on trial here.
    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(hd, "*.txt")))
    if not stems:
        sys.exit(f"labels_human/ is empty in {root}")

    cl_of = {}
    tri = os.path.join(root, "triage.csv")
    if os.path.exists(tri):
        for r in csv.DictReader(open(tri)):
            cl_of[r["file"]] = int(r["cluster"])

    allnpy = sorted(glob.glob(os.path.join(root, "npy", "*.npy")))
    print(f"building background from {min(150, len(allnpy))} frames ...")
    fix = fixture_mask(build_background(allnpy))
    if fix is None:
        print("  could not build one — MISS detection will report fixtures")
    else:
        print(f"  {int(fix.sum())} permanently-warm pixels vetoed "
              f"({100.0 * fix.mean():.1f}% of frame)")

    print(f"{len(stems)} human-labelled frames — scoring ...")
    scored = []
    for i, st in enumerate(stems):
        f = os.path.join(root, "npy", st + ".npy")
        if not os.path.exists(f):
            continue
        arr = np.load(f).astype(np.float32)
        H, W = arr.shape[:2]
        boxes = load_labels(os.path.join(hd, st + ".txt"), W, H)
        s, why, _ = score_frame(arr, boxes, fix, args.miss_min, not args.no_miss)
        scored.append({"file": st, "cluster": cl_of.get(st, -1),
                       "score": s, "why": why})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(stems)}")

    if args.frames:
        items = sorted(scored, key=lambda d: d["score"])
        if args.skip_drawn:
            # The representative is the frame annotate.py opened and you drew
            # on, picked the same way it picks: fewest person boxes, first one
            # wins a tie, in triage.csv row order. Re-deriving it here rather
            # than storing it keeps the two files from disagreeing.
            drawn = set()
            if os.path.exists(tri):
                per = collections.OrderedDict()
                for r in csv.DictReader(open(tri)):
                    per.setdefault(int(r["cluster"]), []).append(r)
                for c, mem in per.items():
                    drawn.add(min(mem, key=lambda r: int(r["n_person"]))["file"])
            if not drawn:
                print("  --skip-drawn needs triage.csv; nothing skipped")
            before = len(items)
            items = [d for d in items if d["file"] not in drawn]
            print(f"  --skip-drawn: {before} -> {len(items)} "
                  f"({before - len(items)} frames you drew, omitted)")
    elif args.skip_drawn:
        print("  --skip-drawn only applies with --frames; ignored")
    else:
        # one entry per cluster, represented by its WORST frame: a box that
        # survives its hardest frame survives the easy ones too
        best = {}
        for d in scored:
            k = d["cluster"]
            if k not in best or d["score"] < best[k]["score"]:
                best[k] = d
        items = sorted(best.values(), key=lambda d: d["score"])

    if args.min_score is not None:
        items = [d for d in items if d["score"] < args.min_score]
    if args.limit:
        items = items[:args.limit]
    if not items:
        sys.exit("nothing to review at that threshold")

    members = collections.defaultdict(list)
    for d in scored:
        members[d["cluster"]].append(d["file"])

    out_csv = os.path.join(root, "review.csv")
    done = {}
    if os.path.exists(out_csv) and not args.restart:
        for r in csv.DictReader(open(out_csv)):
            done[r["file"]] = r["decision"]
        if done:
            print(f"resuming: {len(done)} already decided")

    def flush():
        tmp = out_csv + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "cluster", "score", "why", "decision"])
            for d in items:
                if d["file"] in done:
                    w.writerow([d["file"], d["cluster"], f"{d['score']:.4f}",
                                d["why"], done[d["file"]]])
        os.replace(tmp, out_csv)
        dl = []
        for d in items:
            if done.get(d["file"]) == "delete":
                dl += (members[d["cluster"]] if not args.frames else [d["file"]])
        tmp2 = os.path.join(root, "to_delete.txt") + ".tmp"
        with open(tmp2, "w") as fh:
            fh.write("\n".join(sorted(set(dl))) + ("\n" if dl else ""))
        os.replace(tmp2, os.path.join(root, "to_delete.txt"))

    todo = [d for d in items if d["file"] not in done]
    if not todo:
        print("every item already decided — use --restart to go again")
        return

    WIN = "FLUXNET review"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    S = max(2, args.scale)
    show_blobs = True
    idx = 0
    history = []
    n_ok = sum(1 for v in done.values() if v == "approve")
    n_del = sum(1 for v in done.values() if v == "delete")

    print(f"\n{len(todo)} to review, worst first")
    print("ENTER approve   x delete   b back   u undo   g blobs   q quit\n")

    while 0 <= idx < len(todo):
        d = todo[idx]
        arr = np.load(os.path.join(root, "npy", d["file"] + ".npy")).astype(np.float32)
        H, W = arr.shape[:2]
        boxes = load_labels(os.path.join(hd, d["file"] + ".txt"), W, H)
        s, why, blobs = score_frame(arr, boxes, fix, args.miss_min, not args.no_miss)
        colour = ((80, 90, 255) if s < 0.10 else
                  (70, 190, 255) if s < 0.25 else (120, 220, 130))
        n = len(members[d["cluster"]]) if not args.frames else 1
        head = (f"{idx + 1}/{len(todo)}   {d['file']}   "
                f"cluster {d['cluster']} ({n} frames)")
        sub = f"{why}    approved {n_ok}  deleted {n_del}"
        cv2.imshow(WIN, render(arr, boxes, blobs, S, show_blobs,
                               head, sub, colour))
        k = cv2.waitKey(20) & 0xFF

        if k in (13, 10, ord("n")):
            done[d["file"]] = "approve"
            history.append(d["file"])
            n_ok += 1
            flush()
            idx += 1
        elif k == ord("x"):
            done[d["file"]] = "delete"
            history.append(d["file"])
            n_del += 1
            flush()
            idx += 1
        elif k == ord("b"):
            if idx > 0:
                idx -= 1
                prev = todo[idx]["file"]
                if prev in done:
                    n_ok -= done[prev] == "approve"
                    n_del -= done[prev] == "delete"
                    del done[prev]
                    if prev in history:
                        history.remove(prev)
                    flush()
        elif k == ord("u"):
            if history:
                last = history.pop()
                n_ok -= done.get(last) == "approve"
                n_del -= done.get(last) == "delete"
                done.pop(last, None)
                flush()
                idx = max(0, idx - 1)
        elif k == ord("g"):
            show_blobs = not show_blobs
        elif k in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    flush()

    dl = sum(1 for line in open(os.path.join(root, "to_delete.txt"))
             if line.strip())
    print(f"\napproved            {n_ok}")
    print(f"marked for deletion {n_del}"
          + (f"  ({dl} frames once clusters expand)" if not args.frames else ""))
    print(f"undecided           {len(todo) - n_ok - n_del}")
    print(f"\ndecisions  {out_csv}")
    print(f"deletions  {os.path.join(root, 'to_delete.txt')}  "
          f"— nothing removed yet")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted — decisions saved after the last keypress")
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        try:
            d = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "."
            with open(os.path.join(d, "review_crash.log"), "a") as fh:
                fh.write(f"\n===== {__import__('datetime').datetime.now()} =====\n")
                fh.write(tb)
            print(f"traceback appended to {d}/review_crash.log")
        except Exception:
            pass
        raise
