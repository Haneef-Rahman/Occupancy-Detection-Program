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
