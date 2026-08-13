# FLUXNET — thermal detection baseline

Classical (untrained) person detection from a FLIR Lepton 3.1R on a PureThermal 3.
This is the **baseline** any future ML model must beat — and the source of the
validation data you'd need to prove it did.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "opencv-python==4.10.0.84" numpy
```

### macOS: the capture problem, and the recipe that works

**Nothing in macOS's camera stack can reach this board.** Tested and ruled out:

| Path | Result |
|---|---|
| OpenCV (2 backends × 5 formats, `backend_test.py`) | `backend is generally available but can't be used to capture by index` |
| PyAV / ffmpeg avfoundation (`pyav_test.py`) | `[Errno 5] Input/output error` on every device |
| Terminal.app | never appears in Privacy → Camera, so cannot be granted access |

The MacBook and iPhone cameras open fine in all of the above, so it is specific
to the PureThermal. **libuvc over raw USB is the working path** — and it also
delivers true radiometric Y16, which AVFoundation would never have given us.

Working recipe (no Homebrew — `ghcr.io` is throttled from some regions, which
breaks the Homebrew installer):

```bash
pip install cmake

cd ~
curl -LO https://github.com/libusb/libusb/releases/download/v1.0.27/libusb-1.0.27.tar.bz2
tar xf libusb-1.0.27.tar.bz2
cd libusb-1.0.27
./configure --prefix=/usr/local && make && sudo make install

git clone https://github.com/groupgets/libuvc.git
cd libuvc && mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/usr/local -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make && sudo make install
```

`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is required because cmake 4.x dropped
support for libuvc's old `cmake_minimum_required`.

**Run with sudo** — the macOS kernel UVC driver claims the camera, so libusb
cannot claim the interface as a normal user:

```bash
./run.sh                          # thermal_detect.py
./run.sh lepton_libuvc.py         # capture self-test
./run.sh thermal_detect.py --note "2 abreast"
```

Expect output like `median=26.0C max=34.1C` — room temperature and skin.

On **Linux / Raspberry Pi** none of this applies: V4L2 exposes the board
natively including Y16, and a udev rule replaces the need for sudo.

### Diagnostics

| Script | Purpose |
|---|---|
| `diagnose.py` | what each video device delivers, in both capture modes |
| `backend_test.py` | every backend × format combination, to prove what works |
| `lepton_libuvc.py` | standalone libuvc capture test (prints frame temperatures) |

## Run

```bash
python thermal_detect.py --list          # see which device index is the Lepton
python thermal_detect.py                 # auto-detect and go
python thermal_detect.py --device 1 --delta 4.0 --note "2-person walkthrough"
```

### Keys

| Key | Action |
|---|---|
| `q` / ESC | quit |
| `s` | save frame — raw `.npy` (temperatures) + `.png` preview |
| `l` | toggle logging to CSV |
| `[` `]` | lower / raise threshold (°C above ambient) |
| `a` | re-estimate ambient now |
| `r` | toggle color view ↔ binary mask (useful for tuning) |

## How the detection works

1. **Radiometric capture** — asks the board for 16-bit Y16/TLinear so each pixel
   is an absolute temperature, not an auto-gain brightness. (AGC drifts as the
   scene changes, which makes any fixed threshold meaningless.)
2. **Ambient estimate** — median pixel of the frame; people are a minority of
   pixels, so the median tracks room temperature.
3. **Threshold** at `ambient + delta` (default +4 °C).
4. **Morphology** — open then close: removes speckle, fills pinholes.
5. **Connected components** → blobs, filtered by area and aspect ratio
   (rejects hot pipes, light strips, tiny reflections).

Every detection is explainable: a temperature and a shape. No weights, no
training, no per-site calibration.

## Known limits (these are the fusion arguments, not bugs)

- **Warm clutter** — lamps, laptops, radiators can exceed the threshold. Radar
  (motion + breathing micro-motion) is what disambiguates, not a bigger model.
- **Touching people** — two adjacent bodies merge into one blob. The SPAD's
  depth map (head peaks, shoulder valleys) is what separates them.
- **Glass** — LWIR does not pass through glass or eyewear. A glazed door is a
  wall to this sensor.
- **FFC** — the shutter recalibrates periodically; the image freezes ~1 s and
  clicks. Expected behaviour, not a fault.
- **macOS** — the UVC layer often refuses Y16; the program warns and falls back
  to 8-bit AGC. Use a Linux host / Raspberry Pi for real radiometric work.

## Collecting the validation set

Run with `-l` logging on and a descriptive `--note`, then walk scripted
scenarios: empty room · one person crossing · two abreast · one still/seated ·
warm clutter only. Save frames with `s` at interesting moments. That gives you
ground truth to quote accuracy against in the report — and the dataset you'd
need later if a model ever becomes justified.
