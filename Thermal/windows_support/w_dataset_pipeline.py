#!/usr/bin/env python3
"""
Turn capture logs into a dataset, on Windows. No camera involved.

    py -3.12 w_dataset_pipeline.py
    py -3.12 w_dataset_pipeline.py --from 4 --live

Identical to ../dataset_pipeline.py — same seven stages, same prompts, same
output. Two Windows differences, both handled here:

  1. hand_back() calls os.geteuid(), which DOES NOT EXIST on Windows. Not a
     graceful skip — an AttributeError that would kill the run at the end of
     stage 2, after the merge but before anything was saved. It exists to undo
     sudo file ownership, and there is no sudo here, so it becomes a no-op.

  2. Working directory, so logs/ and models/ resolve the same as on the Mac.

You probably do not need this. Adrian collects, Haneef builds. But if you want
to annotate your own captures, it works.
"""

import win_common as W

W.enter_thermal()

import dataset_pipeline as DP                               # noqa: E402


def main():
    W.standard_prelude()
    W.patch_hand_back(DP)
    DP.main()


if __name__ == "__main__":
    main()
