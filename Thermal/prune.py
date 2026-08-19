#!/usr/bin/env python3
"""
Drop the frames you rejected during review.

Workflow this supports:

    1. Record with `c`. Every frame is written four times:
         npy/     the measurement          (archive, never regenerable)
         png/     clean image              (model input)
         labels/  YOLO boxes               (auto-generated, may be wrong)
         review/  boxes drawn on the image (for your eyes only)

    2. Flip through review/ in Finder at large icon size. Delete any frame
       where the detector got it wrong — a false positive on warm clutter, a
       box on half a person, a miss.

    3. Run this. Whatever you deleted from review/ is deleted from npy/, png/
       and labels/ too, so the dataset matches what you approved.

The point of the indirection: deleting from review/ is safe, because review/
is disposable. Deleting directly from npy/ would destroy measurements you
cannot recapture. This makes the destructive step explicit and dry-run by
default.

    python3 prune.py logs/capture_20260819_204512            # report only
    python3 prune.py logs/capture_20260819_204512 --apply     # actually delete
"""

import argparse
import csv
import os
import sys


SUBS = ("npy", "png", "labels")
EXT = {"npy": ".npy", "png": ".png", "labels": ".txt"}


def stems(d):
    """Frame ids present in a subfolder, e.g. {'cap_000000', ...}."""
    if not os.path.isdir(d):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(d)
            if not f.startswith(".")}


def main():
    ap = argparse.ArgumentParser(
        description="Delete frames rejected during review from a capture folder")
    ap.add_argument("capture_dir")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default is a dry run)")
    ap.add_argument("--keep-labels", action="store_true",
                    help="prune npy/ and png/ but leave labels/ alone")
    args = ap.parse_args()

    root = args.capture_dir.rstrip("/")
    review = os.path.join(root, "review")
    if not os.path.isdir(review):
        sys.exit(f"no review/ folder in {root} — nothing to prune against")

    keep = stems(review)
    have = stems(os.path.join(root, "npy"))
    if not have:
        sys.exit(f"no npy/ folder in {root}")

    drop = sorted(have - keep)
    extra = sorted(keep - have)

    print(f"capture : {root}")
    print(f"recorded: {len(have)} frames")
    print(f"approved: {len(keep)} frames in review/")
    print(f"to drop : {len(drop)} frames "
          f"({100.0 * len(drop) / max(1, len(have)):.1f}%)")
    if extra:
        print(f"warning : {len(extra)} frames in review/ have no npy — "
              f"already pruned? {extra[:3]}")
    if not drop:
        print("\nnothing to do.")
        return

    print("\nfirst few:", ", ".join(drop[:8]) + (" ..." if len(drop) > 8 else ""))

    subs = [s for s in SUBS if not (args.keep_labels and s == "labels")]
    n = 0
    for stem in drop:
        for sub in subs:
            f = os.path.join(root, sub, stem + EXT[sub])
            if os.path.exists(f):
                if args.apply:
                    os.remove(f)
                n += 1

    # keep the manifest consistent with what survived
    man = os.path.join(root, "manifest.csv")
    if args.apply and os.path.exists(man):
        with open(man, newline="") as fh:
            rows = list(csv.reader(fh))
        if rows:
            head, body = rows[0], rows[1:]
            body = [r for r in body if r and r[0] not in set(drop)]
            with open(man, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(head)
                w.writerows(body)
            print(f"manifest: {len(body)} rows kept")

    if args.apply:
        print(f"\ndeleted {n} files across {', '.join(subs)}/")
        print("review/ left intact — it is now the record of what you approved.")
    else:
        print(f"\nDRY RUN — would delete {n} files. Re-run with --apply.")


if __name__ == "__main__":
    main()
