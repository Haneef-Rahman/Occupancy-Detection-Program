# Experimental tracking launcher — 23 October 2026

Prototype for field testing, not a validated replacement for the existing launcher.
`integrated_launcher.py` is unchanged. Launch the new `integrated_launcher_v2.py` explicitly.

## What changed

The old `StickyTracker` is extracted into `precise_k.py` as `PreciseK`. Its sparse-scene association, occlusion/crouch heuristics and ledger are retained. One deterministic tie-break fixes a crash when two tracks are exactly equally distant. A regression trace verifies parity with the original sparse tracker away from that crash. Historical claims in the extracted comments are inherited, not revalidated here.

`faster_k.py` is a separate experimental tracker. Its four-state position/velocity Kalman filter uses elapsed seconds, rather than assuming one constant interval per frame. It associates detections only when they satisfy image-space displacement, covariance distance, search-radius and size-change gates. Association is mutual nearest with a uniqueness margin; it refuses ambiguous identities instead of forcing matches. This is intentionally conservative and can undercount observed tracks. It is not Hungarian assignment or a full ByteTrack implementation.

Default automatic routing uses **normal-confidence omega detections**, after only near-identical duplicate suppression:

- At most four separated detections: precise K.
- More than four, or any touching/overlapping omega boxes: faster K.
- Return to precise K only after a one-second stable sparse period, with no unresolved tracks and at most four retained crowd tracks. A temporary drop in YOLO count must not immediately change tracker.
- Currently observed IDs survive handover. Pixel/frame and pixel/second velocity units are explicitly converted. Old hidden/coasting identities and the old ledger are not imported into faster K; this is recorded in switch events. A return to precise K starts its hidden memory afresh. This is continuity of observed IDs, not guaranteed identity preservation through occlusion.

The new launcher runs a normal-confidence omega inference on every processed thermal frame. This deliberately gives the router a fresh count; it does not reproduce the old hybrid blob-trigger schedule. It is currently **thermal-only**: radar fusion, window correction and the old hybrid detector are not ported. Native confidence/tracker controls and legacy-format thermal dataset capture are available as described below. Use the legacy launcher for those comparisons. RGB is not used for deployed inference.

## Missing detection and local recovery

1. Match normal-confidence detections first.
2. A confirmed unmatched track last seen away from the frame boundary becomes unresolved. Predict an internal circular search region, capped at 14 native thermal pixels.
3. Run a lower-confidence YOLO pass on a crop around that region. Crop padding supplies context, but only detection centres **inside the actual circle** qualify. The full-frame pass stays at its normal confidence.
4. Apply the same motion/size/ambiguity checks. Recovery cannot steal an already matched normal detection or create a new identity. Two consecutive recovery measurements are required before displaying a recovered box.
5. The recovery opportunity expires two seconds after the last **normal full-frame confirmation**. Crop results cannot keep renewing that deadline. At most six crops are processed per frame, with rotating priority if more people are unresolved.

Faster K draws only current detector rectangles. It never draws a predicted or hidden-person box. Optional search circles are labelled as searches, not detections. The display separates observed people and unresolved tracks; the latter are not silently added to the crowd count. Precise K's legacy occupancy estimate remains labelled separately.

Lowering the threshold does not recover a fully invisible person. It can only reveal a weak detector response. Crop resizing also changes detector context and apparent scale; this must be tested on real thermal footage.

## Initial settings — all need field calibration

| Setting | Starting value | Meaning |
| --- | --- | --- |
| Normal confidence | 0.374 | Inherited launcher setting; not established optimal for v5 |
| Recovery confidence | 0.18 | Only crop recovery, never global lowering |
| Recovery deadline | 2 seconds | Engineering starting point, not research-established optimum |
| Maximum search radius | 14 px | Native 160×120 thermal coordinates |
| Maximum displacement rate | 30 px/s + 2 px jitter | Image-space association gate, **not metres/s** |
| Acceleration uncertainty | 100 px/s² | Kalman process-noise parameter, not a physical hard limit |
| Maximum width/height ratio | 1.8 | Reject large apparent size changes |
| Confirmation | 2 normal frames / 2 consecutive recovery frames | Weak evidence never creates a new track |

Without calibrated depth and camera geometry, these pixel limits cannot establish real-world human speed. The proposed 2–3 second horizon is a maximum retention/recovery period, not permission for a circle to grow across the whole room.

## Run tomorrow

From the repository's `Thermal` directory, with the Lepton connected and its fixed TLinear scale verified as 0.01 K/count:

```bash
./run.sh integrated_launcher_v2.py --tracker auto --kelvin-per-count 0.01 --show-search --save-thermal
```

`--save-thermal` saves native Celsius arrays for replay; it is not screen recording. Omit it to retain only numerical JSON logs. No video or desktop recording is started. `q` quits; space pauses; `x` resets tracking; `c` toggles search circles. `r` starts/stops dataset recording; `s` saves one frame. Live resume resets stale tracking state because movement during the pause was unobserved.

The scale argument declares a hardware setting already checked by the operator; this launcher does not re-query the TLinear controls. Do not use AGC frames or unsigned raw-count arrays as Celsius input.

Replay saved Faster captures:

```bash
.venv/bin/python integrated_launcher_v2.py --replay paired_recordings/capture_20261021_164034_583167/npy --headless --tracker auto
```

Replay uses the capture's recorded thermal host-receipt timestamps from `pairs.json`, `timestamps.json` written by this launcher, or the original recorder's `manifest.csv`. Pass the capture root or its `npy` directory. For recordings without usable timing, `--replay-fps` is an explicit override and is labelled as such in the log. Host receipt is not exact exposure time. Approximately 1.75-second-spaced paired stills are insufficient to evaluate full-rate occlusion recovery.

Each run creates a new `logs/crowd_test_...` directory with model hash, settings, mode changes, detections, search regions, observed IDs, unresolved count and measured detector/recovery/tracker times. No model training or weights changes occur.

## Validation and remaining work

- 32 automated tests passed: disappearance without fabricated boxes, bounded/expiring recovery, confidence and circle filtering, no weak births, no detection stealing, ambiguity, speed/size gates, mode hysteresis, identity handover, sparse legacy parity and timestamp replay.
- Real v5 CPU inference completed on 12 stored thermal frames using recorded timestamps, including ROI inference. This validates software plumbing only. There is no manual identity ground truth for these samples, so no tracking accuracy or occupancy accuracy is claimed.
- A separate synthetic tracker-only timing run on this arm64 Mac used 20 warm-up and 200 measured updates per size: median 0.356 / 0.832 / 2.163 / 6.418 ms at 4 / 8 / 16 / 32 stationary detections. Excludes YOLO, recovery crops, camera and GUI. These are local measurements, not Raspberry Pi estimates or proof that faster K is faster than precise K.
- Evidence in the project workspace: `experiments/crowd_tracking/results/`. The earlier six-frame smoke used an explicit average replay rate; the later 12-frame run uses recorded timestamps.

Field test on 24 October: record sparse separated people, five-plus people, touching heads, crossing paths, a short partial occlusion, full occlusion/crouching, entry/exit, fast motion, and empty-scene clutter. Use the same recorded frames for mode comparisons. Manually mark true counts/identities and occlusion intervals, then measure ID switches, false births, count error, misses, recovery delay and inference cost. Tune on one sequence and evaluate on a separate sequence. C2 is not complete until those checks are done.

## Evidence informing the design

- Zhang et al., **ByteTrack**, peer-reviewed ECCV 2022: high-confidence association followed by association of lower-confidence detections to existing tracklets provides an academic basis for considering weak detections during recovery. Its published benchmark numbers do **not** validate this custom thermal tracker or its settings. [Published paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf).
- **Ultralytics official prediction documentation**: `conf`, `classes`, batched image input and detection output coordinates. The spatial threshold restriction is implemented in our crop/filter layer, not a built-in per-pixel YOLO confidence setting. [Official documentation](https://docs.ultralytics.com/modes/predict/).


## Native controls and dataset recording — 23 October 2026

The launcher now has a native OpenCV control panel, implemented in Python. It is not a browser UI. All buttons have keyboard equivalents:

| Key | Action |
| --- | --- |
| `[` / `]` | Decrease / increase normal detection confidence by 0.01 |
| `;` / `'` | Decrease / increase local search confidence by 0.01 |
| `1` / `2` / `3` | Automatic / Precise K / Faster K |
| `T` | Cycle tracker policy |
| `R` | Start / stop continuous thermal dataset recording |
| `S` | Save one current frame; avoid duplicating it if already saved in this capture |
| `Space` / `N` | Pause or resume / advance one frame |
| `X` | Reset tracker state (R now means record) |
| `C` | Show / hide search circles; recovery continues either way |
| `B` / `I` / `G` | Tracked boxes / IDs and scores / raw full-frame detector boxes |
| `P` / `H` | Cycle Inferno, Viridis, grayscale / show help |
| `Q` or Escape | Finish, close captures and save logs |

Use lowercase letter keys, without Shift. Confidence values are constrained to `0 < search < normal < 1`. Changes apply to the next processed frame; a paused frame's recorded confidence and labels stay those actually used to infer it. Changing search confidence updates both ROI inference and Faster K's recovery filter without discarding tracks. Switching tracker policy uses the existing observed-ID handover; it cannot guarantee identity correctness through occlusion. Resuming live capture after a pause resets stale tracking history. Replay preserves its recorded frame times.

Recording creates unique `Thermal/logs/capture_.../` directories, separately from diagnostic `crowd_test_...` logs. These contain `npy/`, fixed-span 16-bit `png/`, omega-only class-1 `labels/`, `review/`, `manifest.csv`, `classes.txt` and `source.txt`. The original `dataset_recording.write_frame` encoder writes these files. Only full-frame YOLO detections become proposed labels; recovery boxes, predicted boxes and tracker IDs do not. Empty-detection frames are retained for review. Manual shots have a separate capture when continuous recording is off. Repeated S on the same paused frame does not duplicate it. Each R start creates a new folder, without overwriting another capture started in the same second.

`timestamps.json` preserves original sequence timing for simulation. `capture_frames.jsonl` records exact per-frame confidence, tracker policy and source frame. The legacy manifest confidence column retains its original two-decimal format. Diagnostic `frames.jsonl` includes both active thresholds and policy; `controls.jsonl` records control actions. No screen or RGB recording is involved.

Optional startup flags: `--record` starts dataset recording immediately; `--capture-every N` saves one in N **processed** frames; `--note "..."` adds a capture note. YOLO throughput limits how many live camera frames are processed. Pause stops automatic saving; N processes one frame, and S can save the displayed frame once. `--save-thermal` is a separate all-processed-frame diagnostic dump, not the dataset-recording toggle.

Review the resulting capture with the existing `dataset_pipeline.py` (including its usual `--live` workflow). No Faster pipeline or RGB camera is required.

Verification: 46 automated tests passed (32 tracking, 9 simulator, 5 controls/recording). Tests cover live threshold propagation, ID handover, unchanged raw arrays, fixed PNG encoding, class-1 labels without tracker/recovery leakage, duplicate prevention, unique capture folders, timestamp import and actual `dataset_pipeline.list_captures()` compatibility. Real v5 CPU replay processed eight frames and recorded four with `--capture-every 2`. A separate native-window replay exercised keyboard settings during pause, R/S recording, palettes, overlays and clean shutdown. This is software validation on stored footage; physical-camera recording and field tracking accuracy remain to be checked. Evidence: project workspace `experiments/launcher_controls/`.


## Earlier bitmap theme — 23 October 2026 (superseded below)

The launcher uses the supplied dashboard and slider snippets as visual references: rounded cards, segmented tracker selection, keycaps, hover states and value bubbles. The interface is black, white and gray. Thermal palettes and detection overlays retain their semantic colors. This is an original Python/Pillow/OpenCV implementation in `launcher_theme.py`; no React/TypeScript, npm dependencies, web server, stock photos or CodeImage checkout are used. Pillow is already present in the project environment. Existing launch commands and keyboard shortcuts still apply.

Both confidence rails support click/drag and show the proposed value while held. Releasing commits it for the next processed frame. Search confidence remains below detection confidence. Keys `[ ]` and `; '` still adjust the same controls and briefly display their values. UI updates are serviced between inference calls, so active inference can limit drag feedback speed; paused interaction refreshes more frequently. H opens the shortcut panel. Help is closed initially to keep the thermal view unobstructed.

Metric cards show actual observed tracks, unresolved tracks and processing time from the current frame; no demo metrics are used. Processing time excludes UI rendering and saving. The new theme changes neither recorded data nor tracking algorithms.

Verification: 48 automated tests passed, including slider preview-versus-commit, clamping, detector/recovery-filter propagation, bounds at supported scales and the existing tracking/recording/simulator regressions. A native real-v5 replay checked both slider mouse interactions, keyboard controls, recording and shutdown. Preview images were rendered directly from saved thermal data, not the desktop. Evidence: workspace `experiments/launcher_theme/`.


## Inferno annotation contrast — 23 October 2026

Click **Contrast** or press **V** to cycle Auto (p1–p99), Crowd (p40–p99.5) and Dense (p70–p100). Inferno starts in Auto. This calls `dataset_pipeline.colorize_view()` directly, including its flat-frame fallback. The active palette and percentile window appear above the thermal image. V also returns from another palette to Inferno; P still cycles palettes. Viridis/grayscale retain their fixed 15–45°C display span.

Contrast is display-only. Model input, confidences, tracking, raw Celsius arrays, fixed-span dataset PNGs and proposed labels are unaffected. Control events record the selected contrast and palette. Pixel-parity checks cover all annotation presets on varying, constant and outlier-heavy frames, plus the button/key cycle and unchanged inference settings.


## Monochrome occupancy area chart — 23 October 2026

The **History** button / **O** toggles a chart above the thermal image. It adapts the supplied screenshot design with a black card, dotted grid, white gradient area, grayscale lines, hover markers and a tooltip. Hiding it restores the larger thermal image and metric cards. It is displayed initially.

Like the original launcher's 120-sample sparkline, the chart plots the occupancy estimate. It also includes a dashed observed-track trace. The estimate is `row['estimate']`: Precise K's legacy presence estimate, or Faster K's currently observed count. Unresolved Faster K tracks are never added to occupancy. The caption and tooltip identify this distinction; it is an algorithm estimate, not a validated ground-truth count. Traces may coincide in Faster K.

The buffer retains only the last 120 processed frames. Samples are added in the processing loop, never on display refresh. The horizontal axis uses actual elapsed timestamps, so pauses and irregular recording intervals are not treated as a constant frame rate. Hover selects the nearest measured frame and displays elapsed time, frame number, estimate, observed count and active tracker. Step-shaped lines preserve integer counts without smooth-curve overshoot. X clears history with the tracker reset; a new or reversed replay clock starts a new history. Hiding the graph does not stop collection.

Checks: three history tests passed (capacity/timestamps, duplicate/reset handling, hover values and monochrome rendering), along with all nine launcher control tests. A native real-v5 replay verified processing-loop updates, paused redraw behavior, hover, hide/show and clean shutdown. The visual preview uses the prior 247-frame real replay, displaying its last 120 samples. Evidence: workspace `experiments/launcher_graph/`. No screen recording was used.


## Sharp native dashboard and independent inference — 23 October 2026

The default interactive launcher now uses `launcher_desktop.py` and `launcher_canvas.py`: native Tk text, vector controls and a separate inference worker. The 1240×840 design follows the user's supplied FLUXNET screenshot: matching card positions, rounded corners, grayscale controls, keycaps, confidence rails, hover tooltip and gradient occupancy area. Only the thermal sensor image is a bitmap; labels, boxes, search circles, graph and interface text are native drawing objects. Sensor pixels are enlarged with nearest-neighbour interpolation and retain the source's 160×120 resolution. The image cache is refreshed only for a new frame, palette/contrast change or resize. Chrome stays monochrome; thermal overlays keep the reference's semantic colors.

The old implementation composed the whole dashboard into one Pillow image and displayed it through OpenCV. That made interface sharpness depend on bitmap/window scaling and made input handling wait for inference. The new main-thread event loop processes display controls independently. One worker owns the camera, YOLO, tracker and file writes; ordered queued commands apply between frames. An in-flight frame finishes before a pause or threshold change takes effect. Acknowledgements keep an older worker snapshot from overwriting a just-edited slider. Published frame arrays are copied and read-only. No web server, React deployment or screen recording is used.

Existing command-line flags and shortcuts remain. The window can grow from a minimum 1120×840 layout; `--scale` selects its initial width. Tab cycles buttons/sliders, Return activates a focused button, and arrow keys adjust a focused slider. A finite replay remains open for display inspection after processing finishes; inference controls cease applying at that point. Quit waits for the current frame and output files to finish. `--headless` runs the same capture/inference code without Tk. The older bitmap renderer remains for its existing regression checks, not as the default live UI.

Validation: all 53 existing tracking, simulator, recording, contrast and chart tests passed; three new thread-boundary tests passed (snapshot ownership, ordered commands during a busy worker, and surfaced errors with cleanup). A real v5 CPU replay tested the new 1240×840 canvas, both confidence changes, Faster K selection, pause/step, R/S recording, contrast, search circles, history hover/toggle and clean shutdown: eight frames processed, two legacy-format dataset frames saved and successfully re-imported. With an artificial 800 ms delay added to every inference, 335 scheduled 20 ms UI heartbeat intervals during inference had median 20.69 ms and maximum 67.00 ms. These are local event-loop measurements, not inference latency or a hardware-independent guarantee. A headless three-frame run with `--capture-every 2` also completed. A separate canvas export used the existing 247-frame replay and its last 120 actual history samples for visual comparison; it was not a desktop capture.

Evidence in the project workspace: `experiments/launcher_native/`, specifically `check-20261023-072055/report.json`, `headless-check/session.json`, `native_check.py`, and the vector-canvas reference export. Physical-camera operation in the new worker and field tracking accuracy still require a live check. A web UI could also provide resolution-independent text, but would still need inference kept off its interaction path; switching to a browser alone would not accelerate YOLO.

## Shared Launcher / Simulator tabs — 23 October 2026

The native header now contains Launcher and Simulator tabs, with the absolute Thermal directory beneath them. Branding is removed. Left-hand analytics show current detector, recovery and tracking durations, mean observed detection confidence and currently recovered observations. These are diagnostics, not accuracy scores.

Simulator reuses the existing simulator worker with original timestamps, sequential inference, backward-seek history rebuilding, result summaries and protected input captures. Controls include Open capture, Play/Pause, Step, Back, Reset, timeline seek, Export and speeds 0.25× through 16×. Options retains recovery window, radius, speed gate and explicit FPS override; applying options restarts replay. Confidence, tracker mode, palettes, contrast and thermal recording use the shared dashboard. Replay dataset output is separate from input captures. Control changes are included in result metadata.

Switching to Simulator queues a launcher pause; returning leaves it paused until resumed. Leaving Simulator pauses replay. Embedded simulator resets preserve the shared global ID allocator. The standalone tracking_simulator.py command also opens this dashboard directly on Simulator, without a camera; headless replay remains supported. Physical-camera operation needs a live check.

Validation: 56 existing tests passed; the nine simulator regressions passed again after final adjustments. Workspace experiments/tabbed_launcher/check.py passed real v5 CPU replay with confidence and tracker changes, recording, backward seek, 16× speed, export and tab switching. Its canvas export was used for layout inspection, without screen recording.

## Simulator raw heat diagnostic — 23 October 2026

Heat settings configures a simulator-only graph diagnostic. Initial, unvalidated test parameters are 25–38°C inclusive, bottom weight 1, top weight 3, vertical exponent 1, and 400 weighted pixels per occupancy unit. This is not an established human-temperature band or a calibrated person-area conversion. No research benchmark is claimed. Use native Celsius arrays, never the contrast-colored image. Pixels outside the band and nonfinite pixels contribute zero.

For normalized vertical coordinate y=0 at the top and y=1 at the bottom, weight(y)=near+(far-near)*(1-y)^power. rFh=sum(mask*weight)/area. Detected=len(full-frame base detections)+max(observed,estimate), exactly the requested overlapping diagnostic sum; it is not a deduplicated count. Fh=a*rFh+b*Detected, initially a=.25, b=.75. Linked sliders enforce 0<=a<.5<b<=1 and a+b=1. This blend is not guaranteed to reduce noise or improve occupancy accuracy.

Default eligibility requires five continuous recorded seconds in Faster K and at most one entry into Precise K within the previous 30 seconds. Gating thresholds are editable. Any Precise K frame immediately returns null heat values and hides all three extra graph traces/metrics. Backward seeks and resets rebuild gating history. Settings changes restart the gate and apply to subsequent frames; earlier samples retain their original values. Simulator reports include raw heat, blended heat, Detected, band pixel count, active state and configuration provenance. Detector, tracker, occupancy outputs and training labels are unchanged by the diagnostic.

Compressed time mode anchors the x-axis at zero and maintains an at-most-240-point decimated overview. Moving mode retains the last 120 processed samples. The header toggles these modes; neither changes frame timestamps. EXPERIMENTAL and LIVE/REPLAY badges have been removed.

Validation: 56 existing tests plus six heat tests passed (band inclusivity, finite pixels, direction weighting, exact blend, gating, reversal and invalid configuration). Native real-v5 replay verified settings opening, axis toggling, eligible heat metrics and immediate hiding after a switch to Precise K. Eligibility warmup was set to zero only for that short UI check; production defaults remain five seconds. Workspace evidence: experiments/heat_diagnostic/validation/report.json. Field calibration and any improvement in stability or accuracy remain unvalidated.


## Heat diagnostic enabled in Launcher — 23 October 2026

The approved heat diagnostic now runs in integrated_launcher_v2.py as well as Simulator. Both tabs expose Heat settings and use the same formula, provisional defaults, continuous Faster K eligibility and immediate Precise K exclusion. Launcher runs the calculation on native Celsius frames before recording history; detector/tracker outputs and dataset labels are unaffected. Settings changes are queued between frames, logged in controls.jsonl and reflected in session.json. Tracker resets, including live resume resets, restart heat eligibility. Settings dialogs retain their originating tab as the command destination.

Validation: 62 automated tests passed. Native Launcher replay with real v5 CPU inference confirmed eligible heat metrics, the settings window, Precise K suppression and continuous recording (three processed frames, three dataset frames). Warmup was zero only in this short check; default remains five seconds. Evidence: workspace experiments/launcher_heat/validation-104241/report.json. Physical live-camera operation was not exercised in this check.


## Integer heat and revised Detected — 23 October 2026

Both tabs now calculate baseline=raw full-frame detections+max(observed,estimate), Fh=floor(a*rFh+b*baseline), and displayed Detected=Fh+raw full-frame detections. The original blend input is retained as blend_detected in frame logs to avoid circular feedback through the revised Detected output. rFh remains continuous; Fh and Detected are integers in new logs, graph metrics and tooltips. Gating and detector/tracker behavior are unchanged. This supersedes earlier Detected/Fh formulas above.
