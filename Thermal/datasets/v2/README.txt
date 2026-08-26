FLUXNET dataset v2
==================
Source     : capture_20260824_152959 (FLIR Lepton 3.1R, 160x120 radiometric)
Labels     : HUMAN ONLY. 324 frames rejected in review were removed; no
             machine-generated labels are present on either side.
Split      : 80/20 random by FRAME, seed 0, taken within the one capture.
Classes    : 0 person, 1 omega (head+shoulders)
Encoding   : 8-bit PNG, single thermal channel replicated to 3.
             Fixed span 15.0-45.0 C across every frame — a pixel value means
             the same temperature everywhere in the dataset. Do NOT re-normalise.

train 2722 frames / 13885 boxes
val    680 frames /  3460 boxes
person 8869 boxes, omega 8476 boxes
occupancy per frame: 2-5 people, no empty-room frames

READ THIS BEFORE QUOTING A NUMBER
Train and val come from the SAME 8-minute recording and the split is random by
frame, so near-duplicate frames sit on both sides. The val score will be
optimistic and does not measure generalisation to a new room, a new day, or a
new thermal background. v2 exists to prove the annotation pipeline produces
something trainable, nothing more.

FIRST STEP ON WINDOWS
    python setup_windows.py
