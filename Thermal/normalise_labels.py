#!/usr/bin/env python3
"""
Make the classical detector's boxes mean the same thing yours do.

The problem this solves. The detector boxes the THRESHOLDED HOT BLOB; a human
boxes the PERSON THEY CAN SEE, including the cooler clothing the threshold
missed. Two annotators, two conventions, same object. Measured on a real
capture: the classical box is 0.90x the width and 0.91x the height of the warm
body, covering 81% of its area — and in 15% of cases less than half of it,
which is flatly contradictory supervision rather than mere noise.

Re-annotating thousands of frames by hand is not the answer. Converting the
machine to the human convention once, automatically, is.

CONVENTION, stated so both sides can follow it:

    A box encloses the WARM SILHOUETTE of one person — every pixel at least
    `--delta` above ambient that belongs to that person — not just the pixels
    hot enough to have triggered detection.

Two guards, because naive expansion is worse than no expansion:

  * ownership. Adjacent people share one warm region. Every warm pixel is
    assigned to its NEAREST hot blob, so a box grows onto its own clothing and
    stops at its neighbour, instead of swallowing them.
  * a cap. Growth beyond `--max-growth` in area is refused and the original
    box kept. A runaway expansion across a warm floor is silent and poisonous;
    refusing is recoverable.

    python3 normalise_labels.py logs/capture_X            # dry run, reports
    python3 normalise_labels.py logs/capture_X --apply

Omega (class 1) boxes are left alone — they describe the head-and-shoulder
region, which is a different thing from the body and already human-shaped.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np


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
        out.append([c, (cx - bw / 2) * W, (cy - bh / 2) * H,
                    (cx + bw / 2) * W, (cy + bh / 2) * H])
    return out


def save(path, boxes, W, H):
    lines = []
    for c, x0, y0, x1, y1 in boxes:
        bw, bh = x1 - x0, y1 - y0
        if bw < 1 or bh < 1:
            continue
        lines.append(f"{c} {(x0 + bw / 2) / W:.6f} {(y0 + bh / 2) / H:.6f} "
                     f"{bw / W:.6f} {bh / H:.6f}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--delta", type=float, default=2.0,
                    help="degrees above ambient defining the visible warm body")
    ap.add_argument("--tmax", type=float, default=36.0)
    ap.add_argument("--max-growth", type=float, default=3.0,
                    help="refuse an expansion that multiplies box area by more "
                         "than this; the original box is kept instead")
    args = ap.parse_args()

    root = args.capture_dir.rstrip("/")
    npys = sorted(glob.glob(os.path.join(root, "npy", "*.npy")))
    if not npys:
        sys.exit(f"no npy/ frames in {root}")

    grown, refused, kept, nbox = 0, 0, 0, 0
    ratios = []

    for f in npys:
        stem = os.path.splitext(os.path.basename(f))[0]
        lp = os.path.join(root, "labels", stem + ".txt")
        arr = np.load(f).astype(np.float32)
        H, W = arr.shape[:2]
        boxes = load(lp, W, H)
        people = [b for b in boxes if b[0] == 0]
        if not people:
            continue

        amb = float(np.median(arr))
        warm = ((arr >= amb + args.delta) & (arr <= args.tmax)).astype(np.uint8)
        warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # CONNECTIVITY FIRST. Only the warm blob a person is actually standing
        # in counts. Without this, ownership hands every warm pixel in the room
        # — floor, walls, radiators — to the nearest person, and the box grows
        # to fill the frame. Measured: 83% of expansions were refused for
        # runaway growth before this guard existed.
        ncc, cc, ccst, _ = cv2.connectedComponentsWithStats(warm, 8)

        # OWNERSHIP SECOND: adjacent people share one warm component, so the
        # nearest hot box wins each pixel — a body grows onto its own clothing
        # and stops at its neighbour rather than swallowing them.
        dists = []
        for _, x0, y0, x1, y1 in people:
            seed = np.ones((H, W), np.uint8)
            seed[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1] = 0
            dists.append(cv2.distanceTransform(seed, cv2.DIST_L2, 3))
        owner = np.argmin(np.stack(dists, 0), axis=0)

        out = []
        for bi, b in enumerate(boxes):
            if b[0] != 0:
                out.append(b)
                continue
            pi = sum(1 for q in boxes[:bi] if q[0] == 0)
            c, x0, y0, x1, y1 = b
            nbox += 1
            # which warm components does this person's HOT box actually touch?
            hx0, hy0 = max(0, int(x0)), max(0, int(y0))
            hx1, hy1 = min(W, int(x1) + 1), min(H, int(y1) + 1)
            ids = np.unique(cc[hy0:hy1, hx0:hx1])
            ids = ids[ids != 0]
            if ids.size == 0:
                out.append(b)
                kept += 1
                continue
            m = np.isin(cc, ids) & (owner == pi)
            ys, xs = np.where(m)
            if xs.size == 0:
                out.append(b)
                kept += 1
                continue
            nx0, ny0 = float(min(x0, xs.min())), float(min(y0, ys.min()))
            nx1, ny1 = float(max(x1, xs.max())), float(max(y1, ys.max()))
            a_old = max(1.0, (x1 - x0) * (y1 - y0))
            a_new = (nx1 - nx0) * (ny1 - ny0)
            if a_new / a_old > args.max_growth:
                out.append(b)
                refused += 1
                continue
            ratios.append(a_new / a_old)
            if a_new > a_old * 1.02:
                grown += 1
            else:
                kept += 1
            out.append([c, nx0, ny0, nx1, ny1])

        if args.apply:
            save(lp, out, W, H)

    r = np.array(ratios) if ratios else np.array([1.0])
    print(f"person boxes            {nbox}")
    print(f"  expanded              {grown} ({100.0 * grown / max(1, nbox):.0f}%)")
    print(f"  already correct       {kept}")
    print(f"  refused (>{args.max_growth}x growth)  {refused}")
    print(f"\narea change: median {np.median(r):.2f}x   "
          f"p90 {np.percentile(r, 90):.2f}x   max {r.max():.2f}x")
    print("\n" + ("labels REWRITTEN in place" if args.apply
                  else "DRY RUN — re-run with --apply"))


if __name__ == "__main__":
    main()
