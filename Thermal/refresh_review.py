#!/usr/bin/env python3
"""
Re-render review/ from the labels as they stand NOW.

review/ is written once, at capture time, with the classical detector's boxes
burned into the PNG. It does not change when you annotate — so after a session
with annotate.py, flipping through review/ shows you the boxes you replaced, not
the ones you drew. Anyone doing a final visual QA pass would be checking stale
pictures and would not know it.

This rebuilds review/ from npy/ plus whichever label is authoritative for each
frame, and marks the provenance on the image so a glance tells you who drew it:

    GOLD   solid box, "H" tag    from labels_human/ — you drew or confirmed it
    silver dashed box, "m" tag   from labels/       — still the machine's

    python3 refresh_review.py logs/capture_X
    python3 refresh_review.py logs/capture_X --only human

Safe to run repeatedly. prune.py still works against the refreshed folder, so
the delete-what-looks-wrong workflow is unchanged — except now you are deleting
based on what the dataset really holds.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np


CLASS_NAMES = ["person", "omega"]
CLASS_COLOR = [(90, 255, 120), (60, 220, 255)]


def load(path, W, H):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        p = line.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        cx, cy, bw, bh = (float(v) for v in p[1:])
        out.append((c, (cx - bw / 2) * W, (cy - bh / 2) * H,
                    (cx + bw / 2) * W, (cy + bh / 2) * H))
    return out


def colorize(a):
    lo, hi = float(np.percentile(a, 1)), float(np.percentile(a, 99))
    if hi - lo < 1e-3:
        hi = lo + 1.0
    n = np.clip((a - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def dashed(img, p0, p1, col, step=9, th=2):
    for x in range(p0[0], p1[0], step):
        cv2.line(img, (x, p0[1]), (min(x + step // 2, p1[0]), p0[1]), col, th)
        cv2.line(img, (x, p1[1]), (min(x + step // 2, p1[0]), p1[1]), col, th)
    for y in range(p0[1], p1[1], step):
        cv2.line(img, (p0[0], y), (p0[0], min(y + step // 2, p1[1])), col, th)
        cv2.line(img, (p1[0], y), (p1[0], min(y + step // 2, p1[1])), col, th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--only", choices=("all", "human", "machine"), default="all",
                    help="render only frames of one provenance")
    args = ap.parse_args()

    root = args.capture_dir.rstrip("/")
    npys = sorted(glob.glob(os.path.join(root, "npy", "*.npy")))
    if not npys:
        sys.exit(f"no npy/ frames in {root}")
    rev = os.path.join(root, "review")
    os.makedirs(rev, exist_ok=True)

    S = max(2, args.scale)
    n_gold = n_silver = n_none = 0

    for f in npys:
        stem = os.path.splitext(os.path.basename(f))[0]
        hp = os.path.join(root, "labels_human", stem + ".txt")
        mp = os.path.join(root, "labels", stem + ".txt")
        if os.path.exists(hp):
            lab, src = hp, "gold"
        elif os.path.exists(mp):
            lab, src = mp, "silver"
        else:
            lab, src = None, "none"

        if args.only == "human" and src != "gold":
            continue
        if args.only == "machine" and src != "silver":
            continue

        arr = np.load(f).astype(np.float32)
        H, W = arr.shape[:2]
        vis = cv2.resize(colorize(arr), (W * S, H * S),
                         interpolation=cv2.INTER_NEAREST)
        boxes = load(lab, W, H) if lab else []

        for c, x0, y0, x1, y1 in boxes:
            p0 = (int(x0 * S), int(y0 * S))
            p1 = (int(x1 * S), int(y1 * S))
            col = CLASS_COLOR[c] if c < len(CLASS_COLOR) else (200, 200, 200)
            if src == "gold":
                cv2.rectangle(vis, p0, p1, col, 2)
            else:
                dashed(vis, p0, p1, col)
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
            cv2.putText(vis, name, (p0[0], max(12, p0[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, name, (p0[0], max(12, p0[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

        tag = {"gold": ("H", (110, 255, 140)),
               "silver": ("m", (150, 150, 160)),
               "none": ("?", (80, 80, 255))}[src]
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 20), (14, 14, 17), -1)
        cv2.putText(vis, f"{stem}   {len(boxes)} boxes", (6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 220), 1, cv2.LINE_AA)
        cv2.putText(vis, tag[0], (vis.shape[1] - 18, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, tag[1], 2, cv2.LINE_AA)

        cv2.imwrite(os.path.join(rev, stem + ".png"), vis)
        if src == "gold":
            n_gold += 1
        elif src == "silver":
            n_silver += 1
        else:
            n_none += 1

    total = n_gold + n_silver + n_none
    print(f"re-rendered {total} frames into {rev}")
    print(f"  H  human  (solid)   {n_gold}")
    print(f"  m  machine (dashed) {n_silver}")
    if n_none:
        print(f"  ?  no label at all  {n_none}")
    print("\nflip through it; delete what is wrong; then prune.py")


if __name__ == "__main__":
    main()
