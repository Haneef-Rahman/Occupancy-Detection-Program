# Native tracking simulator

Python/Tk desktop application for testing saved thermal captures through `integrated_launcher_v2.py`. No browser, web server, camera, desktop capture or screen recording is used. The simulator imports `TrackingSession`, `CrowdDetector`, `FastConfig` and the renderer from the launcher. The launcher also uses that same session class, so changes to its detector/tracker path apply to both.

## Open it

Double-click `Thermal/run_tracking_simulator.command`, or from `Thermal` run:

```bash
.venv/bin/python tracking_simulator.py
```

No sudo is needed. Requires the existing numpy, OpenCV, Pillow and Ultralytics environment, plus Python's Tk support. Tk 8.6 was verified on this Mac. Python GUI programs need a normal desktop session; a restricted/headless process cannot validate a native window.

1. Select a discovered `logs/capture_...` folder, use **Browse**, or paste its path.
2. Choose **Automatic**, **Precise K** or **Faster K**. Forced modes are for comparisons, not automatic deployment routing.
3. Set confidence/recovery parameters if desired. **Load / new test** applies them and starts paused.
4. **Play**, **Step**, **Back**, **Restart**, and the timeline control playback. Space toggles playback; right arrow steps when an input field is not focused.
5. **Save summary** updates the current report; **Open results folder** shows its files. Reports are also saved automatically.

Changing a setting does not silently change an existing run: reload it as a new test. Backward seeking restarts processing from frame zero to rebuild temporal state, creating another results folder. Forward seeking processes every intervening frame. Pause is applied after any in-flight inference finishes.

The viewer uses 0.25×–16× playback speed. The tracker always receives the recorded time interval. No frames are skipped to catch up, so inference can make playback slower than requested. This avoids making a slow replay look like slow human motion or losing important intervening detections.

## Capture formats and integrity

Pass either the capture root or its `npy` directory. Frames must be finite 120×160 floating-point Celsius arrays. Raw sensor counts, colour PNG previews and resized arrays are rejected rather than silently reinterpreted.

Timestamp priority:

1. `timestamps.json` from the new launcher.
2. Thermal host-receipt timestamps in Faster capture `pairs.json`.
3. ISO timestamps keyed by `file` in older `manifest.csv` captures.
4. A user-entered **FPS override**, explicitly labelled as an assumed rate, if intended.

File modification times are never used to invent a frame rate. Missing, duplicated or non-increasing timestamps stop the load with an explanation; use an explicit FPS override only if that assumption is justified. Manifest save times and host-receipt times are not exact exposure timestamps. Recording gaps remain gaps in the replay.

All source arrays, annotations and metadata are read-only. Results go to new folders under `Thermal/logs/tracking_simulations/`; a results directory inside the imported capture is rejected. Existing annotation files are not fed to the detector or treated as identity ground truth.

## Reports

Each run creates:

- `frames.jsonl`: original frame time/path, hash of the loaded array, detector boxes, observed identities, unresolved counts, local recovery regions, mode switches, and measured processing times.
- `summary.json`: source metadata/hash, weights hash, engine source hashes, parameters, processed-frame count, mode usage, observed count range, number of distinct observed IDs, and median/p95 timings.

Distinct observed IDs are **not an ID-switch accuracy metric**. Accuracy needs independently reviewed people/identity annotations. Timing is measured on the current computer, includes cold first-frame inference, and excludes playback delays, rendering and log writes.

The native window displays the current thermal frame with the launcher's boxes and search regions. The plot is rendered from stored sensor data, not captured from the user's screen.

## Headless test

```bash
.venv/bin/python tracking_simulator.py --headless \
  --capture logs/capture_20260921_100706 --tracker auto
```

Use `--tracker precise` or `--tracker faster` on the same recording for separate comparisons. This runs to completion with recorded timing and writes a fresh report.

## Verification — 23 October 2026

- Installed launcher matched its pre-edit backup before the controlled shared-session refactor. The legacy `integrated_launcher.py` remains unchanged.
- 41 automated tests passed after the native UI conversion: the prior 32 tracking tests plus manifest and timestamp import, malformed input handling, shared engine execution, play/step/seek/restart, unchanged time intervals at 4× viewing speed, and source file preservation.
- The full `capture_20260921_100706` sequence (247 frames) completed with real v5 inference through the simulator engine before the UI conversion. It is a functional replay check, not ground-truth accuracy validation.
- Native Tk window checks also passed after installation: real v5 rendering, folder loading, frame stepping, 4× playback, pause, backward seeking with history reconstruction, visible controls, and no Tk callback errors. Evidence: `results/native-ui-check.json`. The initial GUI test harness used a nested `update()` loop and stalled; it was replaced with scheduled callbacks through Tk's normal `mainloop()`, which is also what the application uses.
- The abandoned browser interface is archived only in the project experiment folder and is not installed with the native simulator.

Evidence and test outputs: project workspace `experiments/tracking_simulator/results/`.

### Timeline update — 23 October 2026

Click or drag to select a frame; the handle stays at the requested frame while temporal history is rebuilt. The status above the image shows reconstruction progress. With the timeline focused, Left/Right select adjacent frames and Home/End select the endpoints. The speed selector also includes 16×; every frame still runs through inference with its original timestamp.

Verification for this update: all 41 automated tests passed, including frame preservation and original timestamps at 16×. Native event tests covered clicking, dragging, keyboard navigation and a stationary handle during pending seeks. A real-v5 native replay passed load, step, 16× selection/playback, pause and backward history reconstruction. This verifies functionality, not achievement of 16× inference throughput. Evidence: workspace `experiments/tracking_simulator/slider_fix/timeline-check.json` and `experiments/tracking_simulator/results/native-ui-slider-check.json`.
