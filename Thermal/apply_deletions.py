#!/usr/bin/env python3
"""
Carry out the deletions decided in review.py — reversibly.

review.py records verdicts and touches nothing. This is the step that acts on
them, by MOVING the rejected human labels to labels_human_rejected/ rather than
removing them. Two reasons the move beats a delete:

  * make_dataset.py resolves a frame's label by looking for it in
    labels_human/. A label that is not there is simply not in the dataset, so
    moving is functionally identical to deleting for every downstream tool.
  * a wrong verdict stays recoverable. `--undo` puts them all back.

The .npy frames are never touched. They are the irreplaceable measurement; a
label is an opinion about one and can be redrawn, so only opinions get moved.
Machine labels in labels/ are also left alone — they are a separate quarantine
and are not what was under review.

    python3 apply_deletions.py logs/capture_X            # dry run
    python3 apply_deletions.py logs/capture_X --apply
    python3 apply_deletions.py logs/capture_X --undo
"""

import argparse
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--undo", action="store_true",
                    help="move everything in labels_human_rejected/ back")
    args = ap.parse_args()

    root = args.capture_dir.rstrip("/")
    hd = os.path.join(root, "labels_human")
    rd = os.path.join(root, "labels_human_rejected")

    if args.undo:
        if not os.path.isdir(rd):
            sys.exit(f"nothing to undo — no {rd}")
        names = sorted(n for n in os.listdir(rd) if n.endswith(".txt"))
        print(f"restoring {len(names)} labels to labels_human/")
        if not args.apply:
            print("DRY RUN — add --apply")
            return
        os.makedirs(hd, exist_ok=True)
        for n in names:
            shutil.move(os.path.join(rd, n), os.path.join(hd, n))
        print(f"restored {len(names)}")
        return

    dl = os.path.join(root, "to_delete.txt")
    if not os.path.exists(dl):
        sys.exit(f"no to_delete.txt in {root} — run review.py first")
    stems = [l.strip() for l in open(dl) if l.strip()]
    if not stems:
        sys.exit("to_delete.txt is empty — nothing rejected")

    before = len([n for n in os.listdir(hd) if n.endswith(".txt")]) \
        if os.path.isdir(hd) else 0
    present = [s for s in stems if os.path.exists(os.path.join(hd, s + ".txt"))]
    missing = len(stems) - len(present)

    print(f"{root}")
    print(f"  rejected in review.csv    {len(stems)}")
    print(f"  present in labels_human/  {len(present)}")
    if missing:
        print(f"  already gone              {missing}")
    print(f"\n  labels_human/  {before} -> {before - len(present)}")
    print(f"  moved to       {rd}")
    print(f"  .npy frames    untouched")

    if not args.apply:
        print("\nDRY RUN — re-run with --apply")
        return

    os.makedirs(rd, exist_ok=True)
    n = 0
    for s in present:
        shutil.move(os.path.join(hd, s + ".txt"), os.path.join(rd, s + ".txt"))
        n += 1
    after = len([x for x in os.listdir(hd) if x.endswith(".txt")])
    print(f"\nmoved {n} labels")
    print(f"labels_human/ now holds {after}")
    print(f"\nreversible:  python3 apply_deletions.py {root} --undo --apply")


if __name__ == "__main__":
    main()
