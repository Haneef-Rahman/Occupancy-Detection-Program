#!/usr/bin/env python3
"""
The full tracker on Windows: detection, Kalman tracking, the tuning sidebar.

    py -3.12 w_integrated_launcher.py --mode yolo
    py -3.12 w_integrated_launcher.py                 # hybrid

Needs a model and ultralytics. Not required for data collection.

RADAR PORTS. The upstream defaults are macOS device paths
(/dev/cu.usbserial-*). If an IWR6843 is attached here, pass Windows COM ports:

    --radar-cli COM3 --radar-data COM4

Device Manager lists them under Ports (COM & LPT); the XDS110 shows two, and
the LOWER-numbered one is normally the CLI. Without --radar* the radar path
never runs, so the mismatch is harmless when you are thermal-only.
"""

import win_common as W

W.enter_thermal()

import integrated_launcher as IL                            # noqa: E402


def main():
    allow_agc, argv = W.standard_prelude()
    W.patch_camera(IL, allow_agc)
    IL.main(argv)          # this one takes argv explicitly


if __name__ == "__main__":
    main()
