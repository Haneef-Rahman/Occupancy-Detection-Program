#!/usr/bin/env python3
"""
Train the omega detector on v5. Windows / CUDA.

    python train_v5.py

You do NOT have to edit the yaml this time. preflight() rewrites `path:` to
wherever you actually unzipped v5, because getting that wrong is the single
most common way to lose an evening: ultralytics reports a bad root as "no
images found" and you go looking in the dataloader.

================================================================================
WHAT WENT WRONG LAST TIME, AND WHAT IS DIFFERENT HERE
================================================================================

The v4 run took TEN HOURS and was killed. The epochs were not slow — they were
27 seconds each, 18 minutes of compute for the whole run. The hours disappeared
BETWEEN epochs, and there were four contributing causes. All four are fixed
below. Read this section before changing any of them back.

1. cache="ram" WAS THE KILLER.  Ultralytics does not cache at the source size.
   BaseDataset.load_image resizes the long side to imgsz BEFORE caching, so a
   160x120 PNG is held as 640x480x3:

       on disk, PNG              26 KB
       decoded at source     56.25 KB
       CACHED AT imgsz          900 KB     <- 16x what you would estimate

       v5 train  4647 imgs      3.99 GB
       v5 val    1162 imgs      1.00 GB
       total                    4.99 GB

   The machine had ~1.0 GB free of 31.2 GB. So a 5 GB cache went to the page
   file on a SATA SSD and every epoch boundary re-read it. That is the ten
   hours. (The v4 script said "~266 MB" in its docstring. That number was wrong
   — it was computed at 160x120. Do not trust it if you see it again.)

   NOW cache=False, and here is why that costs nothing:

       cv2.imread of one of these PNGs      1.96 ms   (measured, 400 files)
       one epoch, 4647 images, 1 worker     9.1 s
       across 6 workers, overlapped         ~1.5 s
       GPU time for that same epoch         27 s

   The decode hides completely behind the forward pass. cache="disk" is the
   fallback if you ever move to larger source frames — it writes .npy at SOURCE
   size, so 310 MB for all of v5, not 5 GB — but it litters the dataset tree
   with 5809 .npy files that you then have to remember to delete before any
   file count or merge is meaningful. At 160x120 it buys ~1.5 s an epoch. Not
   worth the cleanup.

2. workers=8 ON A BOX THAT DID NOT HAVE 8 FREE.  Stray python.exe processes
   from earlier annotation sessions were still holding cores. Dropped to 6, and
   preflight() now counts python processes and tells you before you commit to
   150 epochs.

3. RAZER CORTEX WAS RUNNING, "for training performance boost". It is a game
   booster: it reprioritises and de-prioritises processes on its own schedule
   and has no idea what a dataloader is. It did not cause the ten hours — the
   RAM cache did — but it is uncontrolled interference in a run you want to
   compare against v4. preflight() looks for it. Close it.

4. THE GPU WAS NOT THE PROBLEM. Measured at 87% utilisation. I had argued from
   FLOPs-versus-peak that it must be idle; that reasoning is invalid for a
   memory-bound model this small, and the measurement beat it. So batch stays
   at 32 and there is no case here for a CPU upgrade. If you want this faster,
   the lever is batch size, not the host.

================================================================================
WHAT v5 IS
================================================================================

v4 (= v3a + the 2026-09-21 lecture hall) plus 962 frames from the 2026-09-22
and 09-23 sessions. Measured on the actual label files, not estimated:

    train  4647 frames  19914 boxes   mean 4.29/frame  max 32  empty 332
    val    1162 frames   4864 boxes   mean 4.19/frame  max 32  empty  81

THE HEADLINE CHANGE IS THE 413 EMPTY FRAMES.  v4 had ONE, across 4847 images.
The network had therefore never been shown a scene and told there is nobody in
it, which is precisely why it invented people in a 3D printing lab. These are
not blank corridors: they are HARD negatives, certified by hand as containing
no person while containing something that should plausibly fire — a hot
printer, a radiator, sunlit floor, a laptop. That is the thing v4 got wrong,
and it is now 7.1% of train and 7.0% of val.

7% background is inside the range ultralytics expects (it documents roughly
0-10%), so no special handling is needed. If you ever push it past ~15%,
recall will start to sag and you will want to subsample the negatives rather
than wonder why.

SECOND CHANGE: the crowd share went up again.

                    v4                    v5
    sparse (<=5)    3661 f  10478 b       4212 f  10825 b
    crowd  (> 5)     215 f   5991 b        435 f   9089 b
    crowd box share  36.4%                 45.6%

Nearly half the box regression and classification loss now comes from frames
with more than five people, off 9.4% of the images. That is deliberate — a
model that has never seen more than five people cannot represent a lecture
hall, and its objectness prior actively suppresses detections once a frame gets
busy — but it is now the dominant term and it comes from ESSENTIALLY ONE ROOM.
Overfitting risk is real: the network can satisfy half its loss by memorising
that hall. Watch sparse recall, not aggregate mAP.

================================================================================
WHY YOU CANNOT COMPARE THIS mAP TO v4's
================================================================================

Two independent reasons, and they push in opposite directions:

  * The val split is 44% crowd boxes rather than 36%. Harder.
  * The val split now contains 81 images with NO objects. An empty image can
    only ever contribute false positives, so it can lower precision and can
    never raise recall. Different denominator.

So a v5-vs-v4 mAP delta means nothing. The comparison that means something is
at the bottom of this file: run both weights over the SAME held-out capture.

CLASS INDICES. names = {0: person, 1: omega}, and all 24778 labels are class 1
— verified on v5's label files, both splits. Index 0 is deliberately unused. Do
NOT set single_cls=True and do NOT renumber: dataset_recording.py,
annotate_live.py, dataset_pipeline.py, predict_gui.py and annotate.py all
filter detections on `cls == 1`. A model emitting omega as class 0 would have
every prediction silently dropped by every tool in the repo, with no error.

imgsz=640 must match inference. dataset_recording.py and annotate_live.py both
predict at 640; training at anything else changes apparent object scale between
train and deploy.

AUGMENTATION. Unchanged from v4 and the reasoning is unchanged, but the one
that matters is worth repeating because an ultralytics upgrade can silently
restore the default: hsv_v=0.0. Ultralytics defaults it to 0.4, random
brightness of +/-40%. On a photograph that is free robustness. Here a pixel
value IS a temperature — make_dataset.py renders a FIXED 15-45 C span, so grey
level maps linearly to degrees. hsv_v=0.4 presents a 30 C person as anywhere
from ~18 C to ~42 C and teaches the model to ignore absolute temperature: the
most discriminative feature the sensor has, and the whole basis of the
classical detector. Every augmentation below is pinned explicitly for the same
reason — so a version bump cannot change training silently.
"""

import os
import subprocess
import sys

# Point this at wherever you unzipped v5.zip. The yaml INSIDE it will be
# rewritten to match — you do not need to edit it by hand.
DATA = "D:/fluxnet/v5/fluxnet.yaml"

EXPECT_ULTRALYTICS = "8.4.125"   # what v3/v4/v3a's best.pt were all built with


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def patch_yaml_path(yaml_path):
    """
    Force `path:` to the directory the yaml actually lives in.

    v5/fluxnet.yaml ships with the Mac path baked in. A relative path is NOT a
    fix: ultralytics resolves a relative `path:` against its own settings
    DATASETS_DIR, not against the yaml's parent, so it would break somewhere
    else instead. Absolute and self-correcting is the only honest option.
    """
    root = os.path.dirname(os.path.abspath(yaml_path)).replace("\\", "/")
    with open(yaml_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    out, seen, old = [], False, None
    for ln in lines:
        if not seen and ln.lstrip().startswith("path:"):
            old = ln.split(":", 1)[1].strip()
            out.append(f"path: {root}\n")
            seen = True
        else:
            out.append(ln)
    if not seen:
        out.insert(0, f"path: {root}\n")

    new = "".join(out)
    if new != "".join(lines):
        with open(yaml_path, "w", encoding="utf-8") as fh:
            fh.write(new)
        print(f"  yaml path   rewritten: {old}  ->  {root}")
    else:
        print(f"  yaml path   {root}  (already correct)")


def free_ram_gb():
    """Physical RAM free, without pulling in psutil."""
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 1024 ** 3, m.ullTotalPhys / 1024 ** 3
    except Exception:
        return None, None


def running_processes():
    try:
        raw = subprocess.run(["tasklist", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    return [ln.split('","')[0].lstrip('"').lower()
            for ln in raw.splitlines() if ln.strip()]


def preflight():
    print("=" * 66)
    print("preflight")
    print("=" * 66)
    ok = True

    # --- the dataset -------------------------------------------------------
    if not os.path.exists(DATA):
        print(f"  DATA        MISSING: {DATA}")
        print("              edit DATA at the top of this file.")
        return False
    print(f"  DATA        {DATA}")
    patch_yaml_path(DATA)

    root = os.path.dirname(os.path.abspath(DATA))
    for split, want in (("train", 4647), ("val", 1162)):
        d = os.path.join(root, split, "images")
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        flag = "" if n == want else f"   <- expected {want}"
        print(f"  {split:11s} {n} images{flag}")
        if n == 0:
            ok = False

    stray = [f for f in os.listdir(os.path.join(root, "train", "images"))
             if f.endswith(".npy")] if os.path.isdir(
                 os.path.join(root, "train", "images")) else []
    if stray:
        print(f"  .npy litter {len(stray)} files from an old cache='disk' run.")
        print("              they are stale if you re-ran make_dataset. delete:")
        print(f'                Get-ChildItem "{root}" -Recurse -Filter *.npy '
              '| Remove-Item')

    # --- ultralytics -------------------------------------------------------
    try:
        import ultralytics
        v = ultralytics.__version__
        note = "" if v == EXPECT_ULTRALYTICS else (
            f"   <- v3/v3a/v4 were built on {EXPECT_ULTRALYTICS}; "
            "check the augmentation defaults below still hold")
        print(f"  ultralytics {v}{note}")
    except Exception as e:
        print(f"  ultralytics NOT IMPORTABLE: {e}")
        return False

    # --- the GPU -----------------------------------------------------------
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  cuda        {torch.cuda.get_device_name(0)}  "
                  f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
        else:
            print("  cuda        NOT AVAILABLE — this will train on CPU and "
                  "take days.")
            ok = False
    except Exception as e:
        print(f"  torch       {e}")
        ok = False

    # --- the four things that ate ten hours --------------------------------
    avail, total = free_ram_gb()
    if avail is not None:
        print(f"  RAM         {avail:.1f} GB free of {total:.1f} GB")
        if avail < 4.0:
            print("              tight, but cache=False needs almost none. "
                  "This is exactly the condition that made cache='ram' page "
                  "to disk for ten hours — leave the cache off.")

    procs = running_processes()
    if procs:
        py = sum(1 for p in procs if p.startswith("python"))
        if py > 1:
            print(f"  python.exe  {py} running (one of them is this script).")
            print("              close leftover annotation windows — they hold "
                  "the cores the 6 dataloader workers want.")
        cortex = [p for p in procs if "razer" in p or "cortex" in p]
        if cortex:
            print(f"  Razer       {', '.join(sorted(set(cortex)))}")
            print("              CLOSE IT. A game booster reprioritises "
                  "processes on its own schedule; it is uncontrolled "
                  "interference in a run you want to compare against v4.")

    print("=" * 66)
    return ok


# ---------------------------------------------------------------------------
def main():
    from ultralytics import YOLO

    if not preflight():
        print("\npreflight failed — fix the above before starting 150 epochs.")
        sys.exit(1)

    model = YOLO("yolo26n.pt")      # COCO-pretrained base, NOT v4/best.pt

    model.train(
        data=DATA,
        imgsz=640,           # must match inference
        epochs=150,
        patience=30,         # stop when val plateaus
        batch=32,            # GPU measured at 87% util here; it is fed
        device=0,
        workers=6,           # was 8; leave headroom for stray processes
        seed=0,              # reproducible split shuffling and init
        deterministic=False, # ~15% faster; seed still fixes the split
        cache=False,         # READ SECTION 1 OF THE DOCSTRING BEFORE CHANGING
        project="fluxnet",
        name="v5",

        # --- augmentation, tuned for radiometric thermal -------------------
        hsv_h=0.0,           # no-op on greyscale, pinned anyway
        hsv_s=0.0,           # ditto
        hsv_v=0.0,           # CRITICAL: brightness == temperature here
        degrees=0.0,         # fixed mount, no roll
        translate=0.1,       # harmless, helps edge cases
        scale=0.2,           # was 0.5; apparent size encodes range
        shear=0.0,
        perspective=0.0,
        flipud=0.0,          # nobody is upside down
        fliplr=0.5,          # bodies are roughly symmetric — keep
        mosaic=1.0,          # small-object recall; an omega is 4-15 px
        close_mosaic=25,     # four crowd tiles make ~128-box frames
        mixup=0.0,           # blends two scenes; meaningless temperatures
        copy_paste=0.0,      # needs segmentation masks
    )

    m = model.val(data=DATA, imgsz=640, device=0, max_det=300)
    print("\nVALIDATION  — NOT comparable to v4's numbers. The val split is")
    print("              44% crowd boxes (v4: 36%) AND now holds 81 empty")
    print("              frames, which can only ever cost precision.")
    print(f"  mAP50      {m.box.map50:.4f}")
    print(f"  mAP50-95   {m.box.map:.4f}")
    print(f"  precision  {m.box.mp:.4f}")
    print(f"  recall     {m.box.mr:.4f}")
    print("\nbest weights: fluxnet/v5/weights/best.pt")
    print("copy to Thermal/models/v5/best.pt on the Mac —")
    print("model_registry.py picks the highest-numbered models/vN automatically.")

    # THE TWO NUMBERS THAT ACTUALLY ANSWER 'DID v5 HELP'. Aggregate mAP hides
    # both: crowd frames carry 46% of the boxes, so a gain there masks a loss
    # on the sparse scenes that are 91% of the images and all of the
    # deployment target.
    print("""
NEXT, in this order:

1. FALSE POSITIVES ON HARD NEGATIVES. This is what v5 was built to fix and the
   val mAP will not tell you. Run v5 and v4 over a capture with no people in it
   but plenty of warm clutter — the 3D printing lab. Count detections. v4
   invented people there; if v5 still does, the 413 negatives were not enough
   and the answer is more of them, not more epochs.

2. SPARSE RECALL. Run v5/best.pt AND models/v4/best.pt over the SAME held-out
   sparse capture (e.g. logs/capture_20260824_152959) and compare recall. If it
   fell, the crowd frames are overfitting one lecture hall and want subsampling
   with dataset_pipeline.py's merge ratio, not a different learning rate.
""")


if __name__ == "__main__":
    # Required on Windows: dataloader workers spawn a new interpreter, which
    # re-imports this file. Without the guard each worker starts its own
    # training run — which is also, on its own, a plausible way to turn 18
    # minutes of compute into an overnight job.
    main()
