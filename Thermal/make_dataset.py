#!/usr/bin/env python3
"""
Build a YOLO-ready dataset from captured .npy frames.

Why this exists rather than using png/ directly: the live palette is
percentile-stretched per frame, so the same scene encodes to different pixel
values depending on what else was in view. A network trained on that learns
that a person's brightness is arbitrary. Here every frame is mapped through the
SAME fixed temperature span, so a pixel value means one temperature across the
entire dataset — which is the whole advantage of a radiometric sensor over a
webcam, and it is thrown away by a per-frame stretch.

Output is 8-bit 3-channel PNG (the single thermal channel replicated), which is
what Ultralytics expects and what lets COCO-pretrained weights transfer.

    python3 make_dataset.py logs/capture_A logs/capture_B --out datasets/v1
    python3 make_dataset.py logs/capture_* --out datasets/v1 --val-session 2

Sessions are kept whole: a validation split taken at random from shuffled
frames is meaningless here, because consecutive frames are near-duplicates and
the model would be validated on frames it effectively trained on. --val-session
holds out an entire capture instead.
"""

import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np


SPAN_C = (15.0, 45.0)     # must match PNG_SPAN_C in thermal_detect.py


def render(arr, lo, hi):
    """Fixed-span 8-bit render. No per-frame normalisation, deliberately."""
    v = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    g = (v * 255.0).astype(np.uint8)
    return cv2.merge([g, g, g])


def collect(capture_dir):
    npys = sorted(glob.glob(os.path.join(capture_dir, "npy", "*.npy")))
    out = []
    for n in npys:
        stem = os.path.splitext(os.path.basename(n))[0]
        lab = os.path.join(capture_dir, "labels", stem + ".txt")
        if os.path.exists(lab):
            out.append((n, lab, stem))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-session", type=int, default=None,
                    help="index (0-based) of the capture to hold out for "
                         "validation. Omit and the LAST capture is used.")
    ap.add_argument("--lo", type=float, default=SPAN_C[0])
    ap.add_argument("--hi", type=float, default=SPAN_C[1])
    ap.add_argument("--every", type=int, default=1,
                    help="keep every Nth frame. Consecutive frames are near "
                         "duplicates; 2 or 3 costs little and halves the disk.")
    args = ap.parse_args()

    caps = [c.rstrip("/") for c in args.captures]
    if len(caps) < 2:
        print("WARNING: one capture only. Train and val will come from the same\n"
              "         session, so the val score will be optimistic and will\n"
              "         not tell you whether the model generalises.\n")
    vi = args.val_session if args.val_session is not None else len(caps) - 1

    # MUST be "images", not "png": Ultralytics finds a label file by string-
    # replacing os.sep + "images" + os.sep with os.sep + "labels" + os.sep in
    # the image path. Any other folder name and the swap silently does nothing,
    # every image is counted as a background, and training dies with
    # "No labels found".
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(args.out, split, sub), exist_ok=True)

    counts = {"train": 0, "val": 0}
    boxes = {"train": 0, "val": 0}
    per_class = {0: 0, 1: 0}
    empty = 0

    for ci, cap in enumerate(caps):
        split = "val" if ci == vi else "train"
        items = collect(cap)
        if not items:
            print(f"  {os.path.basename(cap)}: no npy/label pairs, skipped")
            continue
        tag = os.path.basename(cap)
        kept = 0
        for i, (npy, lab, stem) in enumerate(items):
            if i % max(1, args.every):
                continue
            arr = np.load(npy)
            img = render(arr, args.lo, args.hi)
            name = f"{tag}_{stem}"
            cv2.imwrite(os.path.join(args.out, split, "images", name + ".png"), img)
            shutil.copy(lab, os.path.join(args.out, split, "labels", name + ".txt"))
            with open(lab) as fh:
                lines = [l for l in fh.read().split("\n") if l.strip()]
            if not lines:
                empty += 1
            boxes[split] += len(lines)
            for l in lines:
                try:
                    per_class[int(l.split()[0])] += 1
                except (ValueError, IndexError, KeyError):
                    pass
            counts[split] += 1
            kept += 1
        print(f"  {tag:<34} -> {split:<5} {kept} frames")

    yaml = os.path.join(args.out, "fluxnet.yaml")
    with open(yaml, "w") as fh:
        fh.write(f"path: {os.path.abspath(args.out)}\n"
                 f"train: train/images\nval: val/images\n\n"
                 f"names:\n  0: person\n  1: head_shoulder\n")

    print(f"\ntrain {counts['train']} frames / {boxes['train']} boxes")
    print(f"val   {counts['val']} frames / {boxes['val']} boxes")
    print(f"classes: person {per_class[0]}, head_shoulder {per_class[1]}")
    print(f"empty frames (no object): {empty} "
          f"({100.0 * empty / max(1, sum(counts.values())):.0f}%)")
    print(f"span: {args.lo}-{args.hi} C, fixed across the whole dataset")
    print(f"\nwrote {yaml}")
    if counts["val"] == 0:
        print("\nNO VALIDATION FRAMES — check --val-session")
        sys.exit(1)
    print("\nyolo detect train model=yolo26n.pt "
          f"data={yaml} imgsz=640 epochs=150 batch=32 device=0")


if __name__ == "__main__":
    main()
