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
    python3 make_dataset.py logs/capture_* --out datasets/v2 --val-sessions 1,2,3

Sessions are kept whole: a validation split taken at random from shuffled
frames is meaningless here, because consecutive frames are near-duplicates and
the model would be validated on frames it effectively trained on. --val-session
holds out an entire capture instead.

HOW MANY SESSIONS TO HOLD OUT. One is rarely enough. A 203-frame capture
sounds respectable until you cluster it: 12 clusters over 23 seconds, so the
effective sample is ~12 independent moments and one bad cluster swings the
score by 8%. Worse, a single short recording covers one occupancy range and one
ambient temperature — capture_20260821_155600 never contains an empty room, so
it cannot measure the false-positive rate at all, which for an occupancy sensor
is the error that matters. --val-sessions pools several recordings so the held-
out set spans more scenes than the one you happened to record last.

TWO SEPARATE RULES, often confused:

  * VALIDATION IS HUMAN-ONLY, ALWAYS. Held-out captures draw from
    labels_human/ and nothing else, whatever --train-source says. Noisy labels
    in training are weak supervision the model averages over; noisy labels in
    validation corrupt the number you report, undetectably.
  * --train-source chooses the TRAINING side only. `human` means both sides are
    hand-labelled; `both` adds machine labels to training and leaves val clean.
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


def collect(capture_dir, source="both"):
    """
    Resolve each frame's label, and say where it came from.

    QUARANTINE. labels_human/ is yours, labels/ is the machine's; they are never
    merged in a single file. A frame with a human label uses it and is tagged
    gold, otherwise the machine label is used and tagged silver.

    source="human"  only gold frames — a small, clean set
    source="both"   gold where it exists, silver elsewhere
    """
    npys = sorted(glob.glob(os.path.join(capture_dir, "npy", "*.npy")))
    out = []
    for n in npys:
        stem = os.path.splitext(os.path.basename(n))[0]
        hum = os.path.join(capture_dir, "labels_human", stem + ".txt")
        mac = os.path.join(capture_dir, "labels", stem + ".txt")
        if os.path.exists(hum):
            out.append((n, hum, stem, "gold"))
        elif source == "both" and os.path.exists(mac):
            out.append((n, mac, stem, "silver"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-session", type=int, default=None,
                    help="index (0-based) of the capture to hold out for "
                         "validation. Omit and the LAST capture is used.")
    ap.add_argument("--val-sessions", type=str, default=None,
                    help="comma-separated indices to hold out, e.g. '1,2,3'. "
                         "Several short recordings make a far better test set "
                         "than one: a 200-frame capture is only ~12 clusters, "
                         "so a single bad cluster swings the score by 8%%.")
    ap.add_argument("--lo", type=float, default=SPAN_C[0])
    ap.add_argument("--hi", type=float, default=SPAN_C[1])
    ap.add_argument("--train-source", choices=("human", "both"), default="both",
                    help="'both' trains on machine labels too (weak supervision "
                         "— noisy, but the model averages over it). 'human' "
                         "trains only on verified frames. VALIDATION IS ALWAYS "
                         "HUMAN-ONLY regardless: this flag governs TRAIN only, "
                         "and cannot loosen the val rule.")
    ap.add_argument("--every", type=int, default=1,
                    help="keep every Nth frame. Consecutive frames are near "
                         "duplicates; 2 or 3 costs little and halves the disk.")
    args = ap.parse_args()

    caps = [c.rstrip("/") for c in args.captures]
    if len(caps) < 2:
        print("WARNING: one capture only. Train and val will come from the same\n"
              "         session, so the val score will be optimistic and will\n"
              "         not tell you whether the model generalises.\n")
    if args.val_sessions:
        try:
            val_idx = {int(x) for x in args.val_sessions.split(",") if x.strip()}
        except ValueError:
            sys.exit(f"--val-sessions must be comma-separated integers, "
                     f"got {args.val_sessions!r}")
        bad = [i for i in val_idx if not 0 <= i < len(caps)]
        if bad:
            sys.exit(f"--val-sessions index out of range: {bad} "
                     f"(there are {len(caps)} captures, 0..{len(caps)-1})")
        if len(val_idx) >= len(caps):
            sys.exit("--val-sessions holds out every capture; nothing left to "
                     "train on")
    elif args.val_session is not None:
        val_idx = {args.val_session}
    else:
        val_idx = {len(caps) - 1}

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

    prov = {"train": {"gold": 0, "silver": 0}, "val": {"gold": 0, "silver": 0}}
    # A held-out capture that silently contributes nothing is the dangerous
    # case: the run still succeeds, and the val score you quote came from
    # fewer sessions than you think. Collected here and shouted about at the end.
    missing = []

    for ci, cap in enumerate(caps):
        split = "val" if ci in val_idx else "train"
        # Validation is human-only ALWAYS. Noisy labels in TRAINING are weak
        # supervision the model averages over; noisy labels in VALIDATION
        # corrupt the reported number with no way to detect it afterwards.
        # --train-source governs TRAIN only: `human` gives a fully hand-
        # labelled dataset on both sides, `both` still leaves val clean.
        items = collect(cap, "human" if split == "val" else args.train_source)
        if not items:
            why = ("no labels_human/ — annotate it or drop it from "
                   "--val-sessions" if split == "val"
                   else "no labels_human/ — use --train-source both to fall "
                        "back on machine labels" if args.train_source == "human"
                   else "no npy/label pairs")
            print(f"  {os.path.basename(cap):<34} -> {split:<5} SKIPPED ({why})")
            missing.append((cap, split, why))
            continue
        tag = os.path.basename(cap)
        kept = 0
        for i, (npy, lab, stem, src) in enumerate(items):
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
            prov[split][src] += 1
            kept += 1
        print(f"  {tag:<34} -> {split:<5} {kept} frames "
              f"({'held out' if split == 'val' else 'training'})")

    yaml = os.path.join(args.out, "fluxnet.yaml")
    with open(yaml, "w") as fh:
        fh.write(f"path: {os.path.abspath(args.out)}\n"
                 f"train: train/images\nval: val/images\n\n"
                 f"names:\n  0: person\n  1: omega\n")

    print(f"\ntrain {counts['train']} frames / {boxes['train']} boxes")
    print(f"val   {counts['val']} frames / {boxes['val']} boxes")
    print(f"classes: person {per_class[0]}, omega {per_class[1]}")
    print(f"empty frames (no object): {empty} "
          f"({100.0 * empty / max(1, sum(counts.values())):.0f}%)")
    print(f"provenance: train gold {prov['train']['gold']} / "
          f"silver {prov['train']['silver']}   |   "
          f"val gold {prov['val']['gold']} (human-only by construction)")
    print(f"span: {args.lo}-{args.hi} C, fixed across the whole dataset")
    print(f"\nwrote {yaml}")
    if missing:
        print(f"\n{'!' * 60}")
        print(f"{len(missing)} capture(s) contributed NOTHING:")
        for cap, split, why in missing:
            print(f"  {os.path.basename(cap)}  ({split})  {why}")
        print("The dataset above was built WITHOUT them.")
        print("!" * 60)
    if counts["val"] == 0:
        print("\nNO VALIDATION FRAMES.\n"
              "  Validation is built from labels_human/ only, so every held-out\n"
              "  session needs annotating first:")
        for i in sorted(val_idx):
            print(f"    python3 triage.py {caps[i]}")
            print(f"    python3 annotate.py {caps[i]}")
        sys.exit(1)
    if counts["train"] == 0:
        print("\nNO TRAINING FRAMES.\n"
              "  With --train-source human, every training capture needs\n"
              "  labels_human/ entries. Either annotate them or use\n"
              "  --train-source both.")
        sys.exit(1)
    print("\nyolo detect train model=yolo26n.pt "
          f"data={yaml} imgsz=640 epochs=150 batch=32 device=0")


if __name__ == "__main__":
    main()
