#!/usr/bin/env python3
"""
Live detection preview on Windows. Records nothing.

    py -3.12 w_live_yolo.py
    py -3.12 w_live_yolo.py --conf 0.45 --scale 6

Use this to aim the camera and sanity-check the view before recording. Needs a
model in ../models/vN/best.pt and ultralytics installed — neither is required
for recording, so if you only came to collect data you can ignore this file.
"""

import win_common as W

W.enter_thermal()

import live_yolo as LY                                      # noqa: E402


def main():
    allow_agc, _ = W.standard_prelude()
    W.patch_camera(LY, allow_agc)
    LY.main()


if __name__ == "__main__":
    main()
