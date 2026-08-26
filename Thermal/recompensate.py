#!/usr/bin/env python3
"""
Apply motion compensation to labels that were already propagated without it.

`annotate.py --compensate` only acts at commit time, so a capture annotated
before the flag existed keeps its verbatim copies. This re-derives the
propagation from the same representative frames, shifting each box by however
far its own subject moved. It re-runs the copy; it does not re-run YOU.

WHY THIS IS SAFE HERE. annotate.py has no way to edit a non-representative
frame: every member of a cluster receives the identical byte string. So a
cluster whose members are all identical carries no human work beyond the
representative, and regenerating it loses nothing. A cluster whose members
DIFFER contains something this script cannot account for, so it is skipped and
reported rather than overwritten. That check runs on every cluster, every time,
and is not optional — it is the only thing standing between this script and
silently destroying hand corrections.

Measured on 16516 propagated boxes from capture_20260824_152959:

    boxes on empty space (IoU 0)   159 -> 102
    boxes IoU < 0.10               848 -> 606
    p25 IoU                      0.231 -> 0.245
    rescued 63, harmed 3

    python3 recompensate.py logs/capture_X              # dry run
    python3 recompensate.py logs/capture_X --apply

Frames the annotator actually drew on are never touched.
"""

import argparse
import collections
import csv
import importlib.util
import os
import sys

import numpy as np


def _load_annotate():
    """Reuse annotate.py's compensate() so the two can never drift apart."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "annotate.py")
    spec = importlib.util.spec_from_file_location("_annotate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-shift", type=float, default=16.0)
    ap.add_argument("--min-corr", type=float, default=0.55)
    args = ap.parse_args()

    an = _load_annotate()
    root = args.capture_dir.rstrip("/")
    tri = os.path.join(root, "triage.csv")
    if not os.path.exists(tri):
        sys.exit(f"no triage.csv in {root}")
    hd = os.path.join(root, "labels_human")
    if not os.path.isdir(hd):
        sys.exit(f"no labels_human/ in {root} — nothing to recompensate")

    rows = list(csv.DictReader(open(tri)))
    byc = collections.OrderedDict()
    for r in rows:
        byc.setdefault(int(r["cluster"]), []).append(r)

    def read(stem):
        p = os.path.join(hd, stem + ".txt")
        return open(p).read() if os.path.exists(p) else None

    n_clu = n_skip_nolabel = n_skip_mixed = n_done = 0
    n_shift = n_refuse = 0
    shifts = []
    mixed = []
    pending = []          # (path, boxes, W, H) — nothing written until the end

    for c, mem in byc.items():
        n_clu += 1
        texts = [(m["file"], read(m["file"])) for m in mem]
        have = [t for t in texts if t[1] is not None]
        if not have:
            n_skip_nolabel += 1
            continue
        if len(have) != len(texts) or len({t[1] for t in have}) != 1:
            # Either partially labelled or not uniform: something happened to
            # these frames that this script cannot reconstruct. Leave them.
            n_skip_mixed += 1
            mixed.append(c)
            continue

        # the representative, chosen exactly as annotate.py chooses it
        rep = min(mem, key=lambda r: int(r["n_person"]))
        rf = os.path.join(root, "npy", rep["file"] + ".npy")
        if not os.path.exists(rf):
            n_skip_nolabel += 1
            continue
        ref = np.load(rf).astype(np.float32)
        H, W = ref.shape[:2]
        boxes = an.load_labels(os.path.join(hd, rep["file"] + ".txt"), W, H)
        if not boxes:
            n_done += 1          # an empty cluster is already correct
            continue

        for m in mem:
            if m["file"] == rep["file"]:
                continue         # never touch the frame that was drawn on
            tf = os.path.join(root, "npy", m["file"] + ".npy")
            if not os.path.exists(tf):
                continue
            tgt = np.load(tf).astype(np.float32)
            out = []
            for b in boxes:
                r = an.compensate(b, ref, tgt, H, W, args.max_shift,
                                  min_corr=args.min_corr)
                if r is None:
                    out.append(b)
                    n_refuse += 1
                else:
                    nb, d = r
                    out.append(nb)
                    shifts.append(d)
                    n_shift += 1
            pending.append((os.path.join(hd, m["file"] + ".txt"), out, W, H))
        n_done += 1

    sh = np.array(shifts) if shifts else np.array([0.0])
    tot = n_shift + n_refuse
    print(f"{root}")
    print(f"  clusters                  {n_clu}")
    print(f"  recompensated             {n_done}")
    print(f"  skipped, no human label   {n_skip_nolabel}")
    print(f"  skipped, members DIFFER   {n_skip_mixed}"
          + ("   <-- hand-edited, left alone" if n_skip_mixed else ""))
    if mixed:
        print(f"    clusters: {mixed[:20]}{' ...' if len(mixed) > 20 else ''}")
    print(f"\n  frames to rewrite         {len(pending)}")
    print(f"  boxes shifted             {n_shift} "
          f"({100.0 * n_shift / max(1, tot):.0f}%)")
    print(f"  boxes refused (unchanged) {n_refuse} "
          f"({100.0 * n_refuse / max(1, tot):.0f}%)")
    print(f"  shift: median {np.median(sh):.1f} px   "
          f"p90 {np.percentile(sh, 90):.1f} px   max {sh.max():.1f} px")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply")
        return

    for path, boxes, W, H in pending:
        an.save_labels(path, boxes, W, H)
    print(f"\n{len(pending)} label files rewritten in {hd}")
    print(f"representative frames untouched")
    print(f"\nnext:  python3 refresh_review.py {root}")


if __name__ == "__main__":
    main()
