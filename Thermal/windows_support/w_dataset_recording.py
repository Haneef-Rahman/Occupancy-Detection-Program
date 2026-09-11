#!/usr/bin/env python3
"""
Record capture logs on Windows. Output is byte-compatible with the Mac.

    py -3.12 w_dataset_recording.py --operator adrian --note "cafeteria, 1430, 26C"

    r        start / stop recording
    q, ESC   quit

All flags from ../dataset_recording.py work unchanged (--conf, --device,
--capture-every, --no-review, --weights). Two extra:

    --operator NAME   who is recording. Goes into source.txt.
    --allow-agc       record even without real temperatures. Don't.

Captures land in Thermal/logs/capture_YYYYMMDD_HHMMSS, exactly where the Mac
puts them and exactly where dataset_pipeline.py looks for them.
"""

import win_common as W

W.enter_thermal()

import dataset_recording as DR                              # noqa: E402


def main():
    allow_agc, _ = W.standard_prelude(require_operator=True)
    W.patch_camera(DR, allow_agc)
    W.patch_provenance(DR)
    DR.main()


if __name__ == "__main__":
    main()
