# Thermal web workspace

A separate local browser application for Lepton capture and capture replay. Entry point: `web_workspace.py`. It does not launch or import `integrated_launcher_v2.py`, and does not replace the native workspace. Shared detection, tracking, heat metrics and recording modules remain the source of the underlying algorithms.

The Python backend runs **on the computer connected to Lepton**; open `http://127.0.0.1:8780/` on that computer. Copy the entire Thermal folder, including a thermal omega model under `models/vN/best.pt`, to use it on another computer. A webpage by itself cannot provide this application's raw radiometric USB stream. No hosted service, Node build, cloud upload or screen recording is involved. Internet is unnecessary once dependencies and weights are installed.

## Mac

Use the existing Thermal environment:

```sh
cd /Users/haneefrahman2650/Developer/Occupancy-Detection-Program/Thermal
.venv/bin/python web_workspace.py
```

Or double-click `start_web_workspace.command`. The existing libuvc installation is reused. If Connect Lepton reports a USB permission error, stop this backend with Ctrl+C and run:

```sh
sudo .venv/bin/python web_workspace.py --no-browser
```

Then open the URL manually. Only one program can own Lepton at a time: disconnect/close the native launcher, GetThermal and other camera applications first. Simulation does not require sudo or a connected camera.

## Windows

In PowerShell, from the copied Thermal folder, either use an existing compatible Python environment or create an isolated one (Python 3.12 recommended):

```powershell
py -3.12 -m venv .venv-web
.\.venv-web\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv-web\Scripts\python.exe web_workspace.py
```

`Start-WebWorkspace.ps1` chooses `.venv-web`, then `.venv`, then `py -3.12`. It forwards command-line options. No machine-wide execution-policy changes are required; invoking Python directly works if script execution is restricted. Ultralytics supplies Torch; CUDA is optional. This implementation was tested with the existing Mac dependencies, not a fresh Windows dependency installation.

Click **Connect Lepton**. Windows tries DirectShow and then Media Foundation at 160×122 / 160×120. Select the camera index and **the scale actually configured on the camera**: 0.01 K/count is the UI default, 0.1 is also supported. This selector does not configure or verify TLinear in the hardware. A 16-bit frame alone does not prove that the board is outputting TLinear: ensure radiometric TLinear is enabled on the board before using absolute-temperature heat metrics. The adapter rejects 8-bit / colour video rather than treating it as temperature. Telemetry rows are removed before conversion.

If no raw stream opens, run the repository's `windows_support/backend_probe.py` using that environment and inspect its output. Camera privacy settings must permit desktop camera applications. Do not replace a working UVC driver with a different USB driver just for this app.

**Validation boundary:** recorded capture replay and actual thermal YOLO inference were exercised on this Mac. Windows capture negotiation, frame conversion, telemetry, resource release and platform dispatch are covered by simulated driver tests; physical Windows Lepton capture remains to be verified with the camera attached. No claim of hardware validation on both operating systems is made.

## Controls

- **Launcher:** Connect Lepton, pause / step, Auto / Precise / Faster tracking, detection and local search confidence, search circles, boxes, IDs/scores and raw detections.
- **Simulator:** Open a capture by the discovered list or local folder path; optional FPS override when timestamps are unavailable. Play at ¼×, ½×, 1×, 2×, 4× or 16×; seek / back / reset, export frame metrics, adjust tracker options. Replay processes intermediate frames when seeking so tracking state has a valid history. Requested speed is limited by inference; no frames are skipped to manufacture 16× throughput.
- **History:** Estimate / Observed plus Detected, rFₕ, Fₕ and pD while the shared Faster K heat gate is active. Moving 120-frame history or compressed overview; hover to inspect values. Heat metrics remain diagnostic estimates, not ground truth.
- **Heat settings:** default 28–37°C, a=0.3, b=0.7. The temperature-band button shows only in-range pixels without changing inference or the recorded temperatures. Contrast cycles the annotation-stage Inferno presets; palette cycles Inferno / Viridis / grayscale.
- **Recording:** R toggles continuous capture; S saves a single frame. Uses the original recorder's `logs/capture_...` schema, float Celsius NPY, fixed-span 16-bit PNG, timestamps and class-1 omega labels from full-frame YOLO only. Recovery / predicted tracking boxes are not written as training labels. Compatible with the existing dataset pipeline; this is not the paired ESP-CAM Faster recorder.
- **Reports:** per-session configuration, model SHA256 and frame metrics go to `logs/web_workspace/session_...`. Export downloads session JSON. Old capture folders are read-only inputs. Changing tracker options restarts the processed history; switching tabs pauses the previous workspace.

Other keys: Space play/pause, N step, X reset, 1/2/3 tracking policy, [/] detection confidence, ;/' search confidence, C circles, B boxes, I IDs, G raw detections, O history, P palette, V contrast, H help, Q disconnect source. Q releases the source; Ctrl+C in the backend terminal stops the server.

## Rendering and runtime

Labels, controls, charts and overlays are browser text / SVG, avoiding enlargement of rasterized UI text. The thermal raster itself remains the sensor's native 160×120 resolution. Capture runs separately from inference and HTTP, keeping only the newest live frame; the two workspace workers have independent tracking and heat state. Camera timestamps represent host receipt, not measured exposure time.

The server binds only to loopback, checks local Host / Origin, requires a per-run command token and serves bundled static files only. Leave it local; it is not a remote-access or multi-user server. Use one operator/browser session. No analytics or external fonts/assets are loaded.

Optional CLI: `--port 8780`, `--weights <local-best.pt>`, `--thermal-root <Thermal-folder>`, `--no-browser`. Models load lazily on first connection or replay, so first-frame initialization may take substantially longer than subsequent frames. No fixed latency guarantee is asserted.

## Tests

From Thermal:

```sh
python -m unittest discover -s tests -p 'test_web_workspace.py' -v
```

The suite exercises raw-scale validation, 8-bit rejection, telemetry, Windows backend fallback, capture cleanup, native tracker parity, replay/seek, legacy recording schema, heat/pD, settings, snapshot isolation and HTTP control protection. Camera tests use simulated frames, not live hardware.

Primary implementation references: [OpenCV video I/O flags](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html), [GroupGets PureThermal UVC capture examples](https://github.com/groupgets/purethermal1-uvc-capture), and the project's existing Mac / Windows adapters. The references describe interfaces, not measured performance or this build's hardware validation.
