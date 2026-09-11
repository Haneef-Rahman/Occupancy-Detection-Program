#!/usr/bin/env python3
"""
RENAMED. This is now w_dataset_recording.py, to match w_live_yolo.py,
w_integrated_launcher.py and w_dataset_pipeline.py.

Kept as a shim so an old command still works rather than failing obscurely.
"""

import sys

print("record_win.py is now w_dataset_recording.py — forwarding.\n",
      file=sys.stderr)

from w_dataset_recording import main                        # noqa: E402

if __name__ == "__main__":
    main()
