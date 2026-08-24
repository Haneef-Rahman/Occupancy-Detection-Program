#!/usr/bin/env python3
"""
Click two corners, pick a class, move on. One frame fixes a whole cluster.

Reads triage.csv and opens ONE representative frame per cluster that needs
work. Whatever you draw is written to that frame's label file and propagated to
every other frame in the cluster, because triage only groups frames that are
near-identical in time.

    python3 annotate.py logs/capture_X
    python3 annotate.py logs/capture_X --scale 5 --no-propagate

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--scale", type=int, default=5)
    ap.add_argument("--no-propagate", action="store_true",
                    help="write only the frame you edited")
    args = ap.parse_args()

    root = args.capture_dir.rstrip("/")
    tri = os.path.join(root, "triage.csv")
    if not os.path.exists(tri):
        sys.exit(f"no triage.csv in {root} — run triage.py first")

    rows = list(csv.DictReader(open(tri)))
    by_cluster = {}
    for r in rows:
        by_cluster.setdefault(int(r["cluster"]), []).append(r)

    todo = sorted({int(r["cluster"]) for r in rows if r["status"] == "todo"})
    if not todo:
        print("nothing marked todo — already done?")
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

    ci = 0
    edited = 0
    propagated = 0
    deleted = []
    dirty = False

    def load_cluster(k):
        """Open the cluster's representative: the frame most needing work."""
        nonlocal dirty
        members = by_cluster[todo[k]]
        rep = min(members, key=lambda r: int(r["n_person"]))
        arr = np.load(os.path.join(root, "npy", rep["file"] + ".npy"))
        st.boxes = load_labels(os.path.join(root, "labels", rep["file"] + ".txt"),
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
            nonlocal edited, propagated
            targets = [rep] if args.no_propagate else members
            for m in targets:
                p = os.path.join(root, "labels", m["file"] + ".txt")
                save_labels(p, st.boxes, W, H)
                m["status"] = "done"
            edited += 1
            propagated += len(targets) - 1

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
            for m in members:
                m["status"] = "kept"
            ci = min(ci + 1, len(todo) - 1)
            rep, members, arr = load_cluster(ci)
        elif k == ord("d"):
            for m in members:
                m["status"] = "drop"
            deleted.extend(m["file"] for m in members)
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
    if deleted:
        dl = os.path.join(root, "to_delete.txt")
        with open(dl, "w") as fh:
            fh.write("\n".join(deleted) + "\n")
        print(f"  list written to {dl}")
    print("\nlabels updated in place. Next: make_dataset.py")


if __name__ == "__main__":
    main()
