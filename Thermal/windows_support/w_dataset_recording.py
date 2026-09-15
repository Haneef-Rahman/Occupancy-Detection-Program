#!/usr/bin/env python3
"""
Record capture logs on Windows. Output is byte-compatible with the Mac.

    .\record.ps1 -Operator adrian -Note "cafeteria, 1430, 26C" -NoModel

    r        start / stop recording
    q, ESC   quit

All flags from ../dataset_recording.py work unchanged (--conf, --device,
--capture-every, --no-review, --weights). Two extra:

    --operator NAME   who is recording. Goes into source.txt.
    --allow-agc       record even without real temperatures. Don't.
    --no-model        record without YOLO. Added automatically below if
                      ultralytics is not installed, which it is not by
                      default — see setup.ps1.

Captures land in Thermal/logs/capture_YYYYMMDD_HHMMSS, exactly where the Mac
puts them and exactly where dataset_pipeline.py looks for them.
"""

import sys

import win_common as W

W.enter_thermal()

import dataset_recording as DR                              # noqa: E402


def main():
    allow_agc, _ = W.standard_prelude(require_operator=True)

    # setup.ps1 deliberately does not install ultralytics — it needs torch,
    # which is a 2 GB download that recording has no use for. So unless the
    # caller asked for a model explicitly, say so up front rather than letting
    # dataset_recording.py fail on the import.
    import importlib.util
    has_ul = importlib.util.find_spec("ultralytics") is not None
    if not has_ul and "--no-model" not in sys.argv:
        print("ultralytics is not installed, so recording without YOLO.\n"
              "  The temperature data is exactly the same; frames just\n"
              "  arrive unlabelled for Haneef to annotate. This is normal.\n")
        sys.argv.append("--no-model")

    W.patch_camera(DR, allow_agc)
    W.patch_provenance(DR)
    DR.main()


if __name__ == "__main__":
    main()
