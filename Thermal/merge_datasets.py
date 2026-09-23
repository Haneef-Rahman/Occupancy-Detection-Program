#!/usr/bin/env python3
"""
Combine finished YOLO datasets into one, without silently corrupting either.

    python3 merge_datasets.py datasets/v2 datasets/v3 --out datasets/v4
    python3 merge_datasets.py datasets/v2 datasets/v3 --out datasets/v4 --keep-all

WHAT IT CHECKS BEFORE IT COPIES ANYTHING. Merging image folders is trivial; the
reasons not to are not, and none of them announce themselves at train time:

  image geometry     two sources at different resolutions train fine and
                     degrade quietly
  class names        same index meaning different things in each source is
                     unrecoverable once merged
  CLASS COVERAGE     the one that actually bites — see below
  filename clashes   cap_000000.png exists in every capture ever recorded

CLASS COVERAGE, AND WHY IT MATTERS MORE THAN IT LOOKS. A detector learns from
absence. Every frame with no box for class C is evidence that class C does not
appear there. So if v2 labels `person` and v3 does not, merging them tells the
model that the 1175 people in v3 are NOT people — 1175 unlabelled positives
aimed straight at the class v2 was trying to teach.

This is not a warning you can reasonably ignore, so the default is to DROP any
class that is not labelled consistently across every source. --keep-all
overrides it, and prints what you are agreeing to.

WHY CLASS INDICES ARE NOT RENUMBERED. Dropping `person` leaves index 0 unused
and omega at 1, which looks untidy and is deliberate. dataset_recording.py,
annotate_live.py and dataset_pipeline.py all filter on `cls == 1` for omega.
Compacting to `0: omega` would train a model whose omega predictions every one
of those tools would discard as person boxes, and nothing would raise an error
— you would just get zero detections and go looking in the wrong place.

WHY THERE IS NO --resplit. Each source was split by CLUSTER, using triage.csv
to keep near-identical frames on one side. That information does not exist in a
built dataset — by here a frame is a PNG with no memory of which moment it came
from. Re-splitting could only shuffle at frame level, which puts near-copies in
both halves and inflates val. So train stays train and val stays val. To change
the ratio, rebuild from the capture logs.
"""

import argparse
import collections
import glob
import os
import shutil
import stat
import sys

SPLITS = ("train", "val")


def read_yaml_names(path):
    """Minimal `names:` reader. Avoids a PyYAML dependency for four lines."""
    names, in_names = {}, False
    if not os.path.exists(path):
        return names
    for ln in open(path):
        s = ln.rstrip("\n")
        if s.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not s.startswith(" ") and s.strip():
                break
            if ":" in s:
                k, v = s.split(":", 1)
                try:
                    names[int(k.strip())] = v.strip()
                except ValueError:
                    pass
    return names


def survey(root):
    """Everything about a dataset that merging could break."""
    from PIL import Image
    info = {"root": root, "name": os.path.basename(root.rstrip("/")),
            "splits": {}, "classes": collections.Counter(),
            "sizes": collections.Counter(), "modes": collections.Counter(),
            "frames_with_any_box": 0, "frames": 0}
    for sp in SPLITS:
        imgs = sorted(glob.glob(os.path.join(root, sp, "images", "*")))
        info["splits"][sp] = imgs
        info["frames"] += len(imgs)
        for p in imgs[:40]:            # sampling: geometry is uniform or broken
            with Image.open(p) as im:
                info["sizes"][im.size] += 1
                info["modes"][im.mode] += 1
        for p in imgs:
            lp = os.path.join(root, sp, "labels",
                              os.path.splitext(os.path.basename(p))[0] + ".txt")
            any_box = False
            if os.path.exists(lp):
                for ln in open(lp):
                    q = ln.split()
                    if len(q) == 5:
                        info["classes"][int(q[0])] += 1
                        any_box = True
            if any_box:
                info["frames_with_any_box"] += 1
    yml = glob.glob(os.path.join(root, "*.yaml"))
    info["names"] = read_yaml_names(yml[0]) if yml else {}
    return info


def hand_back(path):
    """
    Return anything written under sudo to the invoking user.

    run.sh launches EVERY tool under sudo — the macOS kernel UVC driver claims
    the camera, so libusb needs privileges — and afterwards it chowns logs/.
    Only logs/. A dataset built here lands in datasets/, which it never
    touched, so the files came out root-owned at mode 0600 and the next
    non-sudo thing to look at them failed:

        error: open("Thermal/datasets/v5/train/images/...png"): Permission denied
        fatal: Unable to process path ...

    That was git, on 2026-09-23, refusing to add 4600 files it could not read.
    dataset_pipeline.py already does this; this tool did not, which is why
    stage 7 output was fine and merged output was not.

    Also forces u+rw, because shutil.copy2 preserves the SOURCE mode — so a
    0600 file copied by root stays 0600 even after the chown.
    """
    if os.geteuid() != 0:
        return
    uid = os.environ.get("SUDO_UID")
    if not uid:
        return
    uid, gid = int(uid), int(os.environ.get("SUDO_GID") or uid)
    n = 0
    for root_, dirs, files in os.walk(path):
        for p_ in [root_] + [os.path.join(root_, f) for f in dirs + files]:
            try:
                os.chown(p_, uid, gid)
                mode = os.stat(p_).st_mode
                want = stat.S_IMODE(mode) | stat.S_IRUSR | stat.S_IWUSR
                if os.path.isdir(p_):
                    want |= stat.S_IXUSR
                os.chmod(p_, want)
                n += 1
            except OSError:
                pass
    if n:
        print(f"handed {n} paths back to uid {uid}")


def main():
    ap = argparse.ArgumentParser(
        description="Merge built YOLO datasets, checking they are compatible.")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-all", action="store_true",
                    help="keep classes that are not labelled in every source. "
                         "Read what it prints before using this.")
    ap.add_argument("--drop-class", type=int, action="append", default=[],
                    help="drop a class explicitly, repeatable")
    ap.add_argument("--link", action="store_true",
                    help="hardlink images instead of copying. Saves disk; "
                         "breaks if a source is later deleted.")
    ap.add_argument("--path", default=None,
                    help="absolute root written into fluxnet.yaml. Defaults to "
                         "the output dir as this process sees it — override "
                         "when the machine that TRAINS mounts it elsewhere, or "
                         "ultralytics will look for a path that does not exist "
                         "and report only 'no images found'.")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation")
    args = ap.parse_args()

    srcs = [survey(s.rstrip("/")) for s in args.sources]
    for s in srcs:
        if s["frames"] == 0:
            sys.exit(f"{s['root']} has no images — is it a built dataset?")

    print("SOURCES")
    for s in srcs:
        sz = ", ".join(f"{w}x{h}" for (w, h) in s["sizes"])
        print(f"  {s['name']:<28} {s['frames']:>5} frames "
              f"(train {len(s['splits']['train'])}, val {len(s['splits']['val'])})"
              f"   {sz}  {'/'.join(s['modes'])}")
        print(f"  {'':28} boxes: " +
              "  ".join(f"{s['names'].get(c, c)}={n}"
                        for c, n in sorted(s["classes"].items())) or "none")

    # ---- geometry -------------------------------------------------------
    sizes = set().union(*[set(s["sizes"]) for s in srcs])
    modes = set().union(*[set(s["modes"]) for s in srcs])
    if len(sizes) > 1:
        sys.exit(f"\nimage sizes differ across sources: {sizes}. Refusing — a "
                 f"model trained on mixed resolutions degrades without saying "
                 f"so. Rebuild the odd one out.")
    if len(modes) > 1:
        sys.exit(f"\nimage modes differ: {modes}. Refusing for the same reason.")

    # ---- class names ----------------------------------------------------
    merged_names = {}
    for s in srcs:
        for k, v in s["names"].items():
            if k in merged_names and merged_names[k] != v:
                sys.exit(f"\nclass {k} means '{merged_names[k]}' in one source "
                         f"and '{v}' in {s['name']}. Refusing — this is not "
                         f"recoverable after merging.")
            merged_names[k] = v

    # ---- class coverage: the one that matters ---------------------------
    all_classes = sorted(set().union(*[set(s["classes"]) for s in srcs]))
    inconsistent = []
    for c in all_classes:
        missing = [s for s in srcs if s["classes"].get(c, 0) == 0
                   and s["frames_with_any_box"] > 0]
        if missing:
            inconsistent.append((c, missing))

    drop = set(args.drop_class)
    if inconsistent:
        print("\nCLASS COVERAGE PROBLEM")
        for c, missing in inconsistent:
            nm = merged_names.get(c, c)
            who = ", ".join(f"{m['name']} ({m['frames_with_any_box']} labelled "
                            f"frames, 0 '{nm}' boxes)" for m in missing)
            print(f"  '{nm}' (class {c}) is labelled in some sources but not "
                  f"in: {who}")
            print(f"      Those frames would train the model that the objects "
                  f"in them are NOT '{nm}'.")
            if not args.keep_all:
                drop.add(c)
        if args.keep_all:
            print("\n  --keep-all given: merging anyway. You are asserting the "
                  "unlabelled frames genuinely contain none of that class.")
        else:
            print(f"\n  Dropping {sorted(drop)} so every remaining class is "
                  f"labelled consistently. --keep-all overrides.")

    keep = [c for c in all_classes if c not in drop]
    if not keep:
        sys.exit("\nnothing left after dropping — check --drop-class")

    # ---- plan -----------------------------------------------------------
    total = {sp: sum(len(s["splits"][sp]) for s in srcs) for sp in SPLITS}
    kept_boxes = sum(sum(s["classes"].get(c, 0) for c in keep) for s in srcs)
    dropped_boxes = sum(sum(s["classes"].get(c, 0) for c in drop) for s in srcs)
    print(f"\nPLAN -> {args.out}")
    print(f"  train {total['train']}   val {total['val']}   "
          f"(splits preserved from each source, never reshuffled)")
    print(f"  keeping classes {[merged_names.get(c, c) for c in keep]}"
          f"   {kept_boxes} boxes")
    if dropped_boxes:
        print(f"  dropping        {[merged_names.get(c, c) for c in sorted(drop)]}"
              f"   {dropped_boxes} boxes discarded")
    print(f"  class indices UNCHANGED — omega stays {1 if 1 in keep else '?'}, "
          f"because every tool in Thermal/ filters on cls == 1")

    if not args.yes:
        if input("\nproceed? [y/N]: ").strip().lower() not in ("y", "yes"):
            sys.exit("aborted")

    # ---- copy -----------------------------------------------------------
    for sp in SPLITS:
        os.makedirs(os.path.join(args.out, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, sp, "labels"), exist_ok=True)

    seen = set()
    n_img = n_box = 0
    for s in srcs:
        for sp in SPLITS:
            for p in s["splits"][sp]:
                stem = os.path.splitext(os.path.basename(p))[0]
                # Prefix unconditionally. There are no collisions between v2
                # and v3 today because make_dataset.py already prefixes by
                # capture — but a future source built differently would
                # overwrite silently, and one lost frame is invisible.
                new = f"{s['name']}__{stem}"
                if new in seen:
                    sys.exit(f"duplicate after prefixing: {new}")
                seen.add(new)

                dst = os.path.join(args.out, sp, "images",
                                   new + os.path.splitext(p)[1])
                if args.link:
                    try:
                        os.link(p, dst)
                    except OSError:
                        shutil.copy2(p, dst)
                else:
                    shutil.copy2(p, dst)
                n_img += 1

                lp = os.path.join(s["root"], sp, "labels", stem + ".txt")
                lines = []
                if os.path.exists(lp):
                    for ln in open(lp):
                        q = ln.split()
                        if len(q) == 5 and int(q[0]) in keep:
                            lines.append(ln.rstrip("\n"))
                            n_box += 1
                with open(os.path.join(args.out, sp, "labels",
                                       new + ".txt"), "w") as fh:
                    fh.write("\n".join(lines) + ("\n" if lines else ""))

    out_abs = args.path or os.path.abspath(args.out)
    with open(os.path.join(args.out, "fluxnet.yaml"), "w") as fh:
        fh.write(f"path: {out_abs}\ntrain: train/images\nval: val/images\n\n"
                 f"names:\n")
        for c in sorted(merged_names):
            fh.write(f"  {c}: {merged_names[c]}"
                     + ("" if c in keep else "   # dropped: not labelled in "
                                             "every source") + "\n")

    with open(os.path.join(args.out, "SOURCES.txt"), "w") as fh:
        fh.write("merged by merge_datasets.py\n\n")
        for s in srcs:
            fh.write(f"{s['name']}  train {len(s['splits']['train'])}  "
                     f"val {len(s['splits']['val'])}  from {s['root']}\n")
        fh.write(f"\nclasses kept    {sorted(keep)}\n")
        fh.write(f"classes dropped {sorted(drop)}\n")
        fh.write("splits preserved from sources; not reshuffled\n")

    print(f"\nwrote {n_img} images, {n_box} boxes -> {args.out}")
    print(f"      {args.out}/fluxnet.yaml")
    print(f"      {args.out}/SOURCES.txt   (provenance)")
    hand_back(args.out)


if __name__ == "__main__":
    main()
