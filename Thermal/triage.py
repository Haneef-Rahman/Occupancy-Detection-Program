#!/usr/bin/env python3
"""
Sort a capture into "trust it" and "needs a human", and group the work.

The old workflow deleted every frame the classical detector got wrong. That
throws away exactly the examples a learned model most needs — and worse, it
biases what survives: false positives are visible in review and get deleted,
misses look identical to an empty frame and quietly stay. The surviving labels
therefore under-count, which punishes any model that detects more than its
teacher.

So: don't delete, correct. This step decides what to correct and, crucially,
groups the work so one correction covers many frames.

CLUSTERING. At ~8.7 fps consecutive frames are nearly identical, so a box drawn
on one is valid for its neighbours. Clusters are built SEQUENTIALLY in time: a
frame joins the running cluster while it stays within `--sim` of that cluster's
first frame, otherwise a new cluster starts. Two consequences worth knowing:

  * clusters are contiguous in time, so propagation never jumps across a cut
  * a person who moves breaks the cluster, which is what we want — propagation
    is only safe while the scene is genuinely static

Similarity is mean absolute temperature difference on a 20x15 downsample, in
degrees C. Downsampling is what makes 4000 frames tractable and also what stops
sensor noise from splitting clusters.

CHOOSING --sim. Propagation is only valid while people stay put, so the right
threshold is whatever keeps within-cluster movement below a body width. Measured
on a real 4190-frame capture (a person is 15-35 px wide there):

    --sim   clusters   you annotate   leverage   drift p90   clusters >20px
     0.30      145           84         18.1x      27.9 px        13%
     0.20      231          108         14.1x      14.4 px        10%
     0.15      344          146         10.4x       8.9 px         5%
     0.10      555          221          6.9x       7.2 px         1%   <- default
     0.06      951          346          4.4x       5.8 px         0%

0.30 looks tempting at 18x, but 13% of its clusters move a person clean out of
the propagated box — silently, since nobody re-checks a propagated frame. 0.10
keeps that at 1% for a third of the leverage, which is the right trade when the
failure is invisible. This script measures the drift for YOUR capture and prints
it, so the choice is never blind.

    python3 triage.py logs/capture_X                  # asks for the cut-off
    python3 triage.py logs/capture_X --max-boxes 1    # unattended

Writes triage.csv in the capture folder. Moves and deletes nothing.
"""

import argparse
import csv
import glob
import os
import sys

import collections

import cv2
import numpy as np


def count_boxes(label_path):
    n = {0: 0, 1: 0}
    if not os.path.exists(label_path):
        return n
    for line in open(label_path):
        if not line.strip():
            continue
        try:
            c = int(line.split()[0])
        except (ValueError, IndexError):
            continue
        n[c] = n.get(c, 0) + 1
    return n


def signature(arr, w=20, h=15):
    """Small enough to compare 4000 frames, big enough to see a person move."""
    return cv2.resize(arr.astype(np.float32), (w, h),
                      interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--max-boxes", type=int, default=None,
                    help="frames with this many PERSON boxes or fewer go for "
                         "manual annotation. Omit and you are asked "
                         "interactively, after seeing what each choice costs.")
    ap.add_argument("--sim", type=float, default=0.10,
                    help="mean abs temperature difference (C) below which two "
                         "frames count as the same scene. Lower = more, "
                         "tighter clusters = safer propagation, more work.")
    ap.add_argument("--max-cluster", type=int, default=30,
                    help="cap on cluster size, so one bad propagation cannot "
                         "poison hundreds of frames")
    args = ap.parse_args()

    root = args.capture_dir.rstrip("/")
    npys = sorted(glob.glob(os.path.join(root, "npy", "*.npy")))
    if not npys:
        sys.exit(f"no npy/ frames in {root}")

    print(f"{len(npys)} frames — reading signatures ...")
    stems, sigs, counts = [], [], []
    for i, f in enumerate(npys):
        stem = os.path.splitext(os.path.basename(f))[0]
        stems.append(stem)
        sigs.append(signature(np.load(f)))
        counts.append(count_boxes(os.path.join(root, "labels", stem + ".txt")))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(npys)}")

    # ---- sequential clustering ------------------------------------------
    cluster = np.zeros(len(stems), np.int32)
    cid = 0
    anchor = sigs[0]
    size = 0
    for i in range(len(stems)):
        d = float(np.abs(sigs[i] - anchor).mean())
        if i > 0 and (d > args.sim or size >= args.max_cluster):
            cid += 1
            anchor = sigs[i]
            size = 0
        cluster[i] = cid
        size += 1
    n_clusters = cid + 1

    # ---- choose the cut-off ----------------------------------------------
    # "Poorly annotated" has no principled definition yet, so rather than bake
    # in a guess, show what each threshold actually costs and let the operator
    # decide with the numbers in front of them.
    hist = collections.Counter(counts[i][0] for i in range(len(stems)))
    work_for = {}
    for cut in range(0, max(hist) + 1):
        nd = sum(v for k, v in hist.items() if k <= cut)
        wc = len({int(cluster[i]) for i in range(len(stems))
                  if counts[i][0] <= cut})
        work_for[cut] = (nd, wc)

    print(f"\nPERSON BOXES PER FRAME (as the classical detector left them)")
    for k in sorted(hist):
        bar = "#" * int(60.0 * hist[k] / max(hist.values()))
        print(f"  {k:2d} boxes  {hist[k]:5d}  {bar}")

    print(f"\nWHAT EACH CUT-OFF COSTS")
    print(f"  {'cut':>4} {'frames flagged':>15} {'clusters to open':>18} "
          f"{'leverage':>10}")
    for cut in sorted(work_for):
        nd, wc = work_for[cut]
        if nd == 0:
            continue
        print(f"  <={cut:<2} {nd:15d} {wc:18d} "
              f"{nd / max(1, wc):9.1f}x")

    cut = args.max_boxes
    if cut is None:
        while True:
            try:
                raw = input("\ncut-off — frames with N or fewer person boxes "
                            "go for annotation [1]: ").strip()
            except EOFError:
                raw = ""
            if raw == "":
                cut = 1
                break
            try:
                cut = int(raw)
            except ValueError:
                print("  enter a whole number")
                continue
            if cut < 0:
                print("  must be 0 or more")
                continue
            nd, wc = work_for.get(cut, (0, 0))
            ans = input(f"  that flags {nd} frames across {wc} clusters "
                        f"({nd / max(1, wc):.1f}x). ok? [Y/n] ").strip().lower()
            if ans in ("", "y", "yes"):
                break
    print(f"\nusing cut-off <= {cut}")

    need = [counts[i][0] <= cut for i in range(len(stems))]

    # a cluster is worth opening if ANY frame in it needs work; the rest of the
    # cluster then comes along for free via propagation
    work_clusters = sorted({int(cluster[i]) for i in range(len(stems)) if need[i]})

    out = os.path.join(root, "triage.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "cluster", "n_person", "n_omega", "status"])
        for i, stem in enumerate(stems):
            w.writerow([stem, int(cluster[i]), counts[i][0], counts[i].get(1, 0),
                        "todo" if need[i] else "ok"])

    n_need = sum(need)
    sizes = np.bincount(cluster)
    print(f"\n{'='*60}")
    print(f"frames                {len(stems)}")
    print(f"  already annotated   {len(stems) - n_need}  "
          f"({100.0 * (len(stems) - n_need) / len(stems):.0f}%)  -> normal review")
    print(f"  need correction     {n_need}  "
          f"({100.0 * n_need / len(stems):.0f}%)  <= {cut} person box")
    print(f"\nclusters              {n_clusters}   "
          f"(median {int(np.median(sizes))} frames, largest {int(sizes.max())})")
    print(f"  containing work     {len(work_clusters)}")
    if work_clusters:
        covered = int(sum(sizes[c] for c in work_clusters))
        print(f"  frames they cover   {covered}")
        print(f"\nYOU ANNOTATE {len(work_clusters)} FRAMES, NOT {n_need}.")
        print(f"  leverage: {n_need / max(1, len(work_clusters)):.1f}x")
    # ---- self-check: is propagation actually safe here? ------------------
    # A box drawn on one frame is copied to the whole cluster, so the number
    # that matters is how far a person moves WITHIN a cluster. Nobody re-checks
    # a propagated frame, so this failure would be silent.
    byc = collections.defaultdict(list)
    for i in range(len(stems)):
        byc[int(cluster[i])].append(i)

    def person_centres(i):
        p = os.path.join(root, "labels", stems[i] + ".txt")
        out = []
        if not os.path.exists(p):
            return out
        for line in open(p):
            q = line.split()
            if len(q) == 5 and q[0] == "0":
                out.append((float(q[1]) * 160.0, float(q[2]) * 120.0))
        return out

    drift = []
    for c, idx in byc.items():
        pts = [person_centres(i)[0] for i in idx if len(person_centres(i)) == 1]
        if len(pts) < 3:
            continue
        a = np.array(pts)
        drift.append(float(np.hypot(np.ptp(a[:, 0]), np.ptp(a[:, 1]))))
    if drift:
        d = np.array(drift)
        risky = int((d > 20).sum())
        print(f"\npropagation check   {len(d)} clusters measurable")
        print(f"  person moves       median {np.median(d):.1f} px, "
              f"p90 {np.percentile(d, 90):.1f} px")
        print(f"  clusters >20 px    {risky} ({100.0 * risky / len(d):.0f}%)"
              + ("   <-- lower --sim" if risky > 0.05 * len(d) else "   OK"))
    print(f"{'='*60}")
    print(f"\nwrote {out}")
    print(f"next:  python3 annotate.py {root}")


if __name__ == "__main__":
    main()
