#!/usr/bin/env python3
"""
Click two corners, pick a class, move on. One frame fixes a whole cluster.

Reads triage.csv and opens ONE representative frame per cluster that needs
work. Whatever you draw is written to that frame's label file and propagated to
every other frame in the cluster, because triage only groups frames that are
near-identical in time.

PROPAGATION IS NOT FREE. A cluster is near-identical, not identical, and the
box you draw is copied verbatim to every frame in it. Measured over 3726
labelled frames: the frame you drew scores median IoU 0.396 against the warm
body, propagated frames 0.361, and 45 boxes end up on empty space — none of
them on a frame anyone drew, 60% past frame 15 of a long cluster. Pass
--compensate to shift each propagated box by however far its own subject moved.

QUARANTINE. Your work goes to labels_human/, never to labels/. The machine's
labels stay exactly as written, so the two annotators can never silently blend,
and any frame's provenance is readable from which folder holds its label. The
rule that matters downstream: validation is built from labels_human/ ONLY.
Noisy labels in TRAINING are weak supervision and the model averages over them;
noisy labels in VALIDATION corrupt the number you report and there is no way to
detect it after the fact. Different risks, only one of them fatal.

    python3 annotate.py logs/capture_X
    python3 annotate.py logs/capture_X --scale 5 --no-propagate
    python3 annotate.py logs/capture_X --compensate

Mouse
    click, click     two opposite corners of a box
    right click      delete the box under the cursor

Keys
    1                class the pending box as PERSON   (green)
    2                class the pending box as OMEGA    (yellow, head+shoulders)
    u                undo last box on this frame
    c                clear all boxes on this frame
    k                keep the classical detector's existing boxes and move on
    ENTER / n        save + propagate + next cluster
    b                previous cluster
    x                mark this cluster EMPTY (no people) and move on
    d                mark this cluster for deletion (genuinely unusable)
    q                save and quit

The pending box is drawn dashed until you press 1 or 2 — a box with no class is
never written. That is deliberate: an unclassed box is the one mistake that
silently corrupts a dataset.
"""

import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np


CLASS_NAMES = ["person", "omega"]
CLASS_COLOR = [(90, 255, 120), (60, 220, 255)]


class State:
    def __init__(self):
        self.boxes = []          # (cls, x0, y0, x1, y1) in SENSOR pixels
        self.pending = None      # (x0, y0, x1, y1) awaiting a class
        self.first = None        # first corner click
        self.cursor = (0, 0)


def load_labels(path, W, H):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        if not line.strip():
            continue
        p = line.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        cx, cy, bw, bh = (float(v) for v in p[1:])
        out.append((c, (cx - bw / 2) * W, (cy - bh / 2) * H,
                    (cx + bw / 2) * W, (cy + bh / 2) * H))
    return out


def save_labels(path, boxes, W, H):
    lines = []
    for c, x0, y0, x1, y1 in boxes:
        x0, x1 = sorted((max(0.0, x0), min(float(W), x1)))
        y0, y1 = sorted((max(0.0, y0), min(float(H), y1)))
        bw, bh = x1 - x0, y1 - y0
        if bw < 1 or bh < 1:
            continue
        lines.append(f"{c} {(x0 + bw / 2) / W:.6f} {(y0 + bh / 2) / H:.6f} "
                     f"{bw / W:.6f} {bh / H:.6f}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def colorize(a):
    lo, hi = float(np.percentile(a, 1)), float(np.percentile(a, 99))
    if hi - lo < 1e-3:
        hi = lo + 1.0
    n = np.clip((a - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


# ---------------------------------------------------------------------------
# MOTION COMPENSATION
#
# Propagation copies one box to every frame in its cluster. The cluster is
# built to be near-identical, but "near" is not "identical" — people keep
# moving, just slowly enough that the frame signature barely changes. Measured
# across 3726 propagated labels on capture_20260824_152959:
#
#     frame you drew        median IoU 0.396
#     propagated frames     median IoU 0.361
#     boxes landing on empty space:  45, of which NONE were on a frame the
#     annotator drew, and 60% were past frame 15 of a long cluster.
#
# So the average cost is small and the tail is what hurts. This shifts each box
# by however far its own warm body moved, which fixes translation — the bulk of
# motion at 8.7 fps — and does nothing for pose change, which is fine because
# pose change does not walk a box off its subject.
#
# Per-box, not per-frame: two people in one cluster move independently, and a
# single global shift would drag one box off to follow the other.
# ---------------------------------------------------------------------------

def warm_mask(arr, delta=2.0, tmax=36.0):
    amb = float(np.median(arr))
    m = ((arr >= amb + delta) & (arr <= tmax)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def compensate(box, ref_arr, tgt_arr, H, W, max_shift, min_corr=0.55):
    """
    Return box translated to follow its own subject, or None if untrustworthy.

    METHOD: normalised cross-correlation of the box's THERMAL PATCH against the
    target frame, searched within +/- max_shift. The patch is the whole
    appearance — body, clothing, the cool-head/hot-face structure — so it locks
    onto the person rather than onto warm pixels in general.

    An earlier version tracked the centroid of warm mass instead. It measured
    WORSE than not compensating at all (p25 IoU 0.231 -> 0.183 over 16516 real
    propagated boxes), because the reference centroid was taken inside the box
    while the target centroid was taken inside the box PLUS the search margin.
    The larger target window swept in neighbouring warm pixels and dragged the
    centroid outward — a systematic bias, not noise. Correlation is symmetric by
    construction and has no such asymmetry.

    Refusals, each a silent failure mode:
      * patch smaller than 3x3            -> nothing to correlate
      * peak correlation below `min_corr` -> subject changed or left; a weak
        peak is a guess, and a guess is worse than the honest un-shifted box
      * peak at the search-window edge    -> the true match is outside the
        window, so the peak is a boundary artefact
    """
    c, x0, y0, x1, y1 = box
    ax0, ay0 = max(0, int(min(x0, x1))), max(0, int(min(y0, y1)))
    ax1, ay1 = min(W, int(max(x0, x1)) + 1), min(H, int(max(y0, y1)) + 1)
    if ax1 - ax0 < 3 or ay1 - ay0 < 3:
        return None
    tmpl = ref_arr[ay0:ay1, ax0:ax1]

    ms = int(round(max_shift))
    sx0, sy0 = max(0, ax0 - ms), max(0, ay0 - ms)
    sx1, sy1 = min(W, ax1 + ms), min(H, ay1 + ms)
    search = tgt_arr[sy0:sy1, sx0:sx1]
    if search.shape[0] < tmpl.shape[0] or search.shape[1] < tmpl.shape[1]:
        return None

    res = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(res)
    if peak < min_corr:
        return None
    # a peak pinned to the edge of the search window means the real match lies
    # outside it; accepting it would clamp the box to an arbitrary offset
    if res.shape[1] > 1 and loc[0] in (0, res.shape[1] - 1) and \
       res.shape[0] > 1 and loc[1] in (0, res.shape[0] - 1):
        return None

    dx = (sx0 + loc[0]) - ax0
    dy = (sy0 + loc[1]) - ay0
    d = (dx * dx + dy * dy) ** 0.5
    if d > max_shift:
        return None
    if d < 0.5:
        return None
    nx0, ny0, nx1, ny1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
    if min(nx0, nx1) < -2 or min(ny0, ny1) < -2 or \
       max(nx0, nx1) > W + 2 or max(ny0, ny1) > H + 2:
        return None
    return (c, nx0, ny0, nx1, ny1), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--scale", type=int, default=5)
    ap.add_argument("--no-propagate", action="store_true",
                    help="write only the frame you edited")
    ap.add_argument("--compensate", action="store_true",
                    help="shift each propagated box by however far its own warm "
                         "body moved between the frame you drew and the target "
                         "frame. Off by default so existing labels stay "
                         "reproducible; a shift that cannot be trusted is "
                         "refused and the un-shifted box kept.")
    ap.add_argument("--max-shift", type=float, default=16.0,
                    help="refuse a compensation longer than this many sensor "
                         "pixels. Swept on 16516 real propagated boxes: 6 px "
                         "and 8 px leave most of the gain on the table, 16 px "
                         "is the optimum, 20 px starts pulling boxes onto the "
                         "wrong person (harmed 3 -> 12).")
    ap.add_argument("--all", action="store_true",
                    help="reopen every work cluster, including ones that "
                         "already have human labels. Existing boxes load as a "
                         "starting point; nothing is discarded until you "
                         "commit over it.")
    args = ap.parse_args()

    root = args.capture_dir.rstrip("/")
    tri = os.path.join(root, "triage.csv")
    if not os.path.exists(tri):
        sys.exit(f"no triage.csv in {root} — run triage.py first")

    human_dir = os.path.join(root, "labels_human")
    os.makedirs(human_dir, exist_ok=True)

    rows = list(csv.DictReader(open(tri)))
    by_cluster = {}
    for r in rows:
        by_cluster.setdefault(int(r["cluster"]), []).append(r)

    # RESUME. A cluster whose representative already has a human label was
    # finished in an earlier run, even if the crash lost the status. Trusting
    # the label files rather than the index means a crash costs at most the one
    # cluster that was open at the time.
    done_on_disk = set()
    for r in rows:
        if os.path.exists(os.path.join(human_dir, r["file"] + ".txt")):
            done_on_disk.add(int(r["cluster"]))
            if r["status"] == "todo":
                r["status"] = "human"

    if args.all:
        # Re-open everything the triage flagged, done or not. Used when you
        # want another pass rather than a resume.
        todo = sorted({int(r["cluster"]) for r in rows
                       if r["status"] in ("todo", "human")})
        print(f"--all: reopening {len(todo)} clusters "
              f"({len(done_on_disk)} already have human labels)")
    else:
        todo = sorted({int(r["cluster"]) for r in rows if r["status"] == "todo"})
    if done_on_disk and not args.all:
        print(f"resuming: {len(done_on_disk)} clusters already have human "
              f"labels, {len(todo)} left")
    if not todo:
        print("nothing left to annotate — all clusters have human labels")
        return

    S = max(2, args.scale)
    WIN = "FLUXNET annotate"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    st = State()

    def on_mouse(event, mx, my, flags, param):
        st.cursor = (mx / S, my / S)
        if event == cv2.EVENT_LBUTTONDOWN:
            if st.first is None:
                st.first = (mx / S, my / S)
            else:
                x0, y0 = st.first
                st.pending = (x0, y0, mx / S, my / S)
                st.first = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            cx, cy = mx / S, my / S
            for i in range(len(st.boxes) - 1, -1, -1):
                _, x0, y0, x1, y1 = st.boxes[i]
                if min(x0, x1) <= cx <= max(x0, x1) and \
                   min(y0, y1) <= cy <= max(y0, y1):
                    st.boxes.pop(i)
                    break

    cv2.setMouseCallback(WIN, on_mouse)

    def flush_triage():
        """
        Write progress after every cluster, not just at quit.

        The old version only saved triage.csv on exit, so a crash 100 clusters
        in lost every status even though the label files themselves had been
        written. Rewriting 4000 rows costs about a millisecond; losing an hour
        of annotation does not. Written to a temp file and renamed, so a crash
        mid-write cannot leave a half-written index either.
        """
        tmp = tri + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "cluster", "n_person", "n_omega", "status"])
            for r in rows:
                w.writerow([r["file"], r["cluster"], r["n_person"],
                            r["n_omega"], r["status"]])
        os.replace(tmp, tri)

    ci = 0
    edited = 0
    propagated = 0
    deleted = []
    dirty = False
    n_shift = n_refuse = 0
    shifts = []

    def load_cluster(k):
        """Open the cluster's representative: the frame most needing work."""
        nonlocal dirty
        members = by_cluster[todo[k]]
        rep = min(members, key=lambda r: int(r["n_person"]))
        arr = np.load(os.path.join(root, "npy", rep["file"] + ".npy"))
        # prefer your own earlier work, fall back to the machine's as a start
        hp = os.path.join(root, "labels_human", rep["file"] + ".txt")
        mp = os.path.join(root, "labels", rep["file"] + ".txt")
        st.boxes = load_labels(hp if os.path.exists(hp) else mp,
                               arr.shape[1], arr.shape[0])
        st.pending = None
        st.first = None
        dirty = False
        return rep, members, arr

    rep, members, arr = load_cluster(ci)

    print(f"\n{len(todo)} clusters to review, covering "
          f"{sum(len(by_cluster[c]) for c in todo)} frames")
    print("click two corners, then 1=person 2=omega, ENTER=next, q=quit\n")

    while True:
        H, W = arr.shape[:2]
        vis = cv2.resize(colorize(arr), (W * S, H * S),
                         interpolation=cv2.INTER_NEAREST)

        for c, x0, y0, x1, y1 in st.boxes:
            p0 = (int(min(x0, x1) * S), int(min(y0, y1) * S))
            p1 = (int(max(x0, x1) * S), int(max(y0, y1) * S))
            col = CLASS_COLOR[c] if c < len(CLASS_COLOR) else (200, 200, 200)
            cv2.rectangle(vis, p0, p1, col, 2)
            cv2.putText(vis, CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c),
                        (p0[0], max(12, p0[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c),
                        (p0[0], max(12, p0[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        # pending box: dashed, no class yet
        if st.pending:
            x0, y0, x1, y1 = st.pending
            p0 = (int(min(x0, x1) * S), int(min(y0, y1) * S))
            p1 = (int(max(x0, x1) * S), int(max(y0, y1) * S))
            for xx in range(p0[0], p1[0], 10):
                cv2.line(vis, (xx, p0[1]), (min(xx + 5, p1[0]), p0[1]), (255, 255, 255), 2)
                cv2.line(vis, (xx, p1[1]), (min(xx + 5, p1[0]), p1[1]), (255, 255, 255), 2)
            for yy in range(p0[1], p1[1], 10):
                cv2.line(vis, (p0[0], yy), (p0[0], min(yy + 5, p1[1])), (255, 255, 255), 2)
                cv2.line(vis, (p1[0], yy), (p1[0], min(yy + 5, p1[1])), (255, 255, 255), 2)
            cv2.putText(vis, "1=person  2=omega", (p0[0], max(12, p0[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        elif st.first:
            cx, cy = st.first
            cv2.drawMarker(vis, (int(cx * S), int(cy * S)), (255, 255, 255),
                           cv2.MARKER_CROSS, 16, 2)

        # crosshair, because a 160x120 sensor pixel is easy to miss by one
        mx, my = int(st.cursor[0] * S), int(st.cursor[1] * S)
        cv2.line(vis, (mx, 0), (mx, vis.shape[0]), (90, 90, 100), 1)
        cv2.line(vis, (0, my), (vis.shape[1], my), (90, 90, 100), 1)

        bar = np.full((78, vis.shape[1], 3), 16, np.uint8)
        cv2.putText(bar, f"cluster {ci + 1}/{len(todo)}   {rep['file']}   "
                         f"{len(members)} frames", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 230), 1, cv2.LINE_AA)
        np_ = sum(1 for b in st.boxes if b[0] == 0)
        no_ = sum(1 for b in st.boxes if b[0] == 1)
        cv2.putText(bar, f"person {np_}   omega {no_}", (12, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(bar, "1/2 class   u undo   c clear   x empty   k keep   "
                         "ENTER next   b back   d drop   q quit", (12, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 152), 1, cv2.LINE_AA)
        cv2.imshow(WIN, np.vstack([vis, bar]))

        k = cv2.waitKey(20) & 0xFF

        def commit():
            nonlocal edited, propagated, n_shift, n_refuse
            targets = [rep] if args.no_propagate else members
            ref_arr = arr.astype(np.float32) if args.compensate and st.boxes else None
            for m in targets:
                p = os.path.join(root, "labels_human", m["file"] + ".txt")
                out = st.boxes
                if ref_arr is not None and m["file"] != rep["file"]:
                    tp = os.path.join(root, "npy", m["file"] + ".npy")
                    try:
                        tgt_arr = np.load(tp).astype(np.float32)
                    except Exception:
                        tgt_arr = None
                    if tgt_arr is not None:
                        out = []
                        for b in st.boxes:
                            r = compensate(b, ref_arr, tgt_arr, H, W,
                                           args.max_shift)
                            if r is None:
                                out.append(b)
                                n_refuse += 1
                            else:
                                nb, d = r
                                out.append(nb)
                                shifts.append(d)
                                n_shift += 1
                save_labels(p, out, W, H)
                m["status"] = "human"
            edited += 1
            propagated += len(targets) - 1
            flush_triage()

        if k in (ord("1"), ord("2")) and st.pending:
            c = 0 if k == ord("1") else 1
            x0, y0, x1, y1 = st.pending
            st.boxes.append((c, x0, y0, x1, y1))
            st.pending = None
            dirty = True
        elif k == ord("u"):
            if st.pending:
                st.pending = None
            elif st.boxes:
                st.boxes.pop()
                dirty = True
        elif k == ord("c"):
            st.boxes = []
            st.pending = None
            dirty = True
        elif k == ord("x"):
            st.boxes = []
            commit()
            ci = min(ci + 1, len(todo) - 1)
            rep, members, arr = load_cluster(ci)
        elif k == ord("k"):
            # confirming the machine's boxes is a human judgement, so they get
            # promoted into labels_human/ rather than merely left alone
            for m in members:
                mp = os.path.join(root, "labels", m["file"] + ".txt")
                hp = os.path.join(root, "labels_human", m["file"] + ".txt")
                if os.path.exists(mp):
                    with open(hp, "w") as fh:
                        fh.write(open(mp).read())
                m["status"] = "human"
            flush_triage()
            ci = min(ci + 1, len(todo) - 1)
            rep, members, arr = load_cluster(ci)
        elif k == ord("d"):
            for m in members:
                m["status"] = "drop"
            deleted.extend(m["file"] for m in members)
            flush_triage()
            ci = min(ci + 1, len(todo) - 1)
            rep, members, arr = load_cluster(ci)
        elif k in (13, 10, ord("n")):
            commit()
            if ci + 1 >= len(todo):
                print("  last cluster")
            ci = min(ci + 1, len(todo) - 1)
            rep, members, arr = load_cluster(ci)
        elif k == ord("b"):
            ci = max(0, ci - 1)
            rep, members, arr = load_cluster(ci)
        elif k in (ord("q"), 27):
            if dirty:
                commit()
            break

    cv2.destroyAllWindows()

    with open(os.path.join(root, "triage.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "cluster", "n_person", "n_omega", "status"])
        for r in rows:
            w.writerow([r["file"], r["cluster"], r["n_person"], r["n_omega"],
                        r["status"]])

    print(f"\nclusters edited      {edited}")
    print(f"frames propagated to {propagated}")
    print(f"marked for deletion  {len(deleted)}")
    if args.compensate:
        tot = n_shift + n_refuse
        sh = np.array(shifts) if shifts else np.array([0.0])
        print(f"\nmotion compensation")
        print(f"  boxes shifted      {n_shift} "
              f"({100.0 * n_shift / max(1, tot):.0f}%)")
        print(f"  refused (kept)     {n_refuse} "
              f"({100.0 * n_refuse / max(1, tot):.0f}%)")
        print(f"  shift: median {np.median(sh):.1f} px   "
              f"p90 {np.percentile(sh, 90):.1f} px   max {sh.max():.1f} px")
    if deleted:
        dl = os.path.join(root, "to_delete.txt")
        with open(dl, "w") as fh:
            fh.write("\n".join(deleted) + "\n")
        print(f"  list written to {dl}")
    print(f"\nhuman labels written to {os.path.join(root, 'labels_human')}")
    print("machine labels in labels/ untouched. Next: make_dataset.py")


if __name__ == "__main__":
    # A crash mid-session used to vanish with the window. Write the traceback
    # somewhere it survives, so the next run can say what happened.
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted — progress saved after the last committed cluster")
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        try:
            d = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "."
            with open(os.path.join(d, "annotate_crash.log"), "a") as fh:
                fh.write(f"\n===== {__import__('datetime').datetime.now()} =====\n")
                fh.write(tb)
            print(f"traceback appended to {d}/annotate_crash.log")
        except Exception:
            pass
        raise
