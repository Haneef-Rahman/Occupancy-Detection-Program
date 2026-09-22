#!/usr/bin/env python3
"""
Train the omega detector on v4. Windows / CUDA.

    python train_v4.py

BEFORE YOU RUN IT, fix the path inside the yaml. v4/fluxnet.yaml was written
on the Mac and still says:

    path: /Users/haneefrahman2650/Developer/.../datasets/v4

Change it to wherever you unzipped it, e.g. `path: D:/fluxnet/v4`. Ultralytics
does not report a missing root usefully — you get "no images found" and go
looking in the wrong place.

WHAT v4 IS. v3a (v2 + v3, makerspaces and hangar) plus 270 frames from a
lecture hall on 2026-09-21. Measured:

    train  3876 frames  16469 boxes   mean 4.25/frame  max 32  empty 1
    val     971 frames   4097 boxes   mean 4.22/frame  max 32  empty 0

THE ONE THING THAT IS DIFFERENT FROM v3a, AND IT IS NOT THE FRAME COUNT.
v3a never contained a frame with more than FIVE people. v4 goes to 32. Those
270 crowd frames are 6% of the images and 36% OF ALL BOXES:

    sparse (<=5 people)   3661 train frames   10478 boxes
    crowd  (> 5 people)    215 train frames    5991 boxes

So the crowd scenes carry roughly a third of the box regression and
classification loss while being a twentieth of the images. That is deliberate —
a model that has never seen more than five people cannot represent a lecture
hall, and its learned objectness prior actively suppresses detections once a
frame gets busy. But it means two things worth watching in the logs:

  * If sparse-scene recall DROPS relative to v3a's numbers, the crowd frames
    are dominating. They come from ONE room across two or three genuinely
    distinct scenes, so there is real overfitting risk: the network can satisfy
    that third of the loss by memorising this hall rather than learning
    density.
  * The val split carries the same 36% crowd share, so a v4-vs-v3a mAP
    comparison is NOT like-for-like. Different val sets, different difficulty.
    Compare per-class-free mAP only as a sanity check; for "did it get better",
    run both models over the same held-out capture.

WHAT CARRIES OVER FROM THE v3a SCRIPT, AND WHY.

1. AUGMENTATION THAT DESTROYS TEMPERATURE IS OFF. Still the important one.
   Ultralytics defaults to hsv_v=0.4 — random brightness of plus/minus 40%.
   On a photograph that is free robustness. On this dataset a pixel value IS a
   temperature: make_dataset renders a FIXED 15-45 C span, so grey level maps
   linearly to degrees. hsv_v=0.4 takes a 30 C person and presents them as
   anywhere from roughly 18 C to 42 C, and the model learns to ignore absolute
   temperature — the single most discriminative feature the sensor has, and the
   entire basis of the classical detector. Set to 0.

   hsv_h and hsv_s are no-ops on greyscale-rendered-to-RGB (saturation is
   already 0), but they are zeroed too so nobody has to re-derive that.

2. LESS SCALE JITTER. Default scale=0.5 is plus/minus 50%. The sensor is fixed
   at 2 m and 15 degrees down, so apparent omega size encodes RANGE —
   predict_gui.py estimates distance from box width. Heavy scale augmentation
   teaches the model to discard that. Reduced, not removed.

3. NO VERTICAL FLIP, NO ROTATION. People are gravity-aligned and the mount does
   not roll. Both defaults are already 0; stated explicitly so an upgrade of
   ultralytics cannot quietly turn them on.

4. MOSAIC ON, CLOSED EARLY — AND IT MATTERS MORE NOW. Mosaic composes four
   scenes into one frame. With v3a that capped out around 20 boxes per mosaic;
   with v4 four crowd tiles can reach ~128, which is well past anything the
   sensor can produce and past what the assigner sees at inference. It still
   earns its place for small-object recall (an omega is 4-15 px), so it stays
   on, but close_mosaic is raised 15 -> 25 so the model spends longer finishing
   on real frames. If crowd recall looks erratic, this is the first knob.

5. EPOCHS 150 WITH EARLY STOPPING, patience=30. Unchanged. 150 is a ceiling.

6. cache="ram". 4847 images at 160x120x3 uint8 is ~266 MB decoded. Holding it
   in RAM removes disk I/O from the loop entirely.

7. max_det AT VALIDATION. Ultralytics defaults to 300, which is fine for 32
   people — but it is now within two orders of magnitude of the real count
   rather than sixty times it, so it is pinned explicitly. If you ever record a
   genuinely packed hall, raise it before concluding recall collapsed.

CLASS INDICES. v4 has names {0: person, 1: omega} and ZERO person boxes —
verified: all 20566 labels are class 1. Index 0 is deliberately unused. Do NOT
set single_cls=True and do not renumber: dataset_recording.py, annotate_live.py
and dataset_pipeline.py all filter detections on `cls == 1`. A model that emits
omega as class 0 would have every prediction silently discarded by those tools.

imgsz=640 must match inference. dataset_recording.py and annotate_live.py both
predict at 640; training at anything else changes the apparent object scale
between train and deploy.
"""

from ultralytics import YOLO

DATA = "D:/fluxnet/v4/fluxnet.yaml"      # <- and fix `path:` inside it too


def main():
    model = YOLO("yolo26n.pt")           # COCO-pretrained base, NOT v3/best.pt

    model.train(
        data=DATA,
        imgsz=640,          # must match inference
        epochs=150,
        patience=30,        # stop when val plateaus
        batch=16,
        device=0,
        workers=8,
        seed=0,             # reproducible split shuffling and init
        cache="ram",        # ~266 MB decoded; kills dataloader I/O
        project="fluxnet",
        name="v4",

        # --- augmentation, tuned for radiometric thermal ------------------
        hsv_h=0.0,          # no-op on greyscale, pinned anyway
        hsv_s=0.0,          # ditto
        hsv_v=0.0,          # CRITICAL: brightness == temperature here
        degrees=0.0,        # fixed mount, no roll
        translate=0.1,      # harmless, helps edge cases
        scale=0.2,          # was 0.5; apparent size encodes range
        shear=0.0,
        perspective=0.0,
        flipud=0.0,         # nobody is upside down
        fliplr=0.5,         # bodies are roughly symmetric — keep
        mosaic=1.0,
        close_mosaic=25,    # was 15; four crowd tiles make ~128-box frames
        mixup=0.0,          # blends two scenes; meaningless temperatures
        copy_paste=0.0,     # needs segmentation masks
    )

    # Validate explicitly so the numbers land in the log next to the run,
    # rather than only in whatever the last training epoch happened to print.
    m = model.val(data=DATA, imgsz=640, device=0, max_det=300)
    print("\nVALIDATION  (val is 36% crowd boxes — not comparable to v3a's)")
    print(f"  mAP50      {m.box.map50:.4f}")
    print(f"  mAP50-95   {m.box.map:.4f}")
    print(f"  precision  {m.box.mp:.4f}")
    print(f"  recall     {m.box.mr:.4f}")
    print("\nbest weights: fluxnet/v4/weights/best.pt")
    print("copy to Thermal/models/v4/best.pt on the Mac to use it —")
    print("model_registry.py picks the highest-numbered models/vN automatically.")

    # THE NUMBER THAT ACTUALLY ANSWERS 'DID THE CROWD DATA HELP OR HURT'.
    # Aggregate mAP hides it: crowd frames carry a third of the boxes, so a
    # gain there can mask a loss on the sparse scenes that are 94% of the
    # images and all of the deployment target.
    print("\nNEXT: run this model AND models/v3/best.pt over the same sparse\n"
          "capture (e.g. logs/capture_20260824_152959) and compare recall.\n"
          "If sparse recall fell, the 270 crowd frames are overfitting one\n"
          "lecture hall and want subsampling with make_dataset.py --every.")


if __name__ == "__main__":
    # Required on Windows: dataloader workers spawn a new interpreter, which
    # re-imports this file. Without the guard each worker starts its own
    # training run.
    main()
