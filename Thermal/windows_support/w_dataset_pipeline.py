#!/usr/bin/env python3
"""
Turn capture logs into a dataset, on Windows. No camera involved.

    ..\.venv-win\Scripts\python.exe w_dataset_pipeline.py
    ..\.venv-win\Scripts\python.exe w_dataset_pipeline.py --from 4 --live

Identical to ../dataset_pipeline.py — same seven stages, same prompts, same
output. Two Windows differences, both handled here:

  1. hand_back() calls os.geteuid(), which DOES NOT EXIST on Windows. Not a
     graceful skip — an AttributeError that would kill the run at the end of
     stage 2, after the merge but before anything was saved. It exists to undo
     sudo file ownership, and there is no sudo here, so it becomes a no-op.

  2. Working directory, so logs/ and models/ resolve the same as on the Mac.

--live NEEDS ULTRALYTICS AND A MODEL, neither of which setup.ps1 installs — it
proposes boxes with the network, so there is nothing to propose without one.
Plain (no --live) annotation needs neither: you draw the corners. Stage 5
(verify) also wants a model and now says SKIPPED rather than refusing to run,
so a dataset can still be built end to end on a machine with no weights.

In the stage-4 annotator: two left-clicks are opposite corners, hover a corner
and press DEL/e to remove a box, x commits empty, and g commits a HARD NEGATIVE
— a frame certified to contain no person despite holding something that looks
like it should fire. Hot printers, sunlit pavement, radiators. Those are the
captures the model most needs and the training set has none of.

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
