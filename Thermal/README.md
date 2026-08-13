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
| **`v`** | **cycle mounting mode** (vertical → horizontal → any) |
| **`1` `2` `3`** | **jump to** vertical / horizontal / any |
| `a` | print current background and threshold |
| `r` | cycle display: colour → mask → blended (useful for tuning) |

## How the detection works

1. **Radiometric capture** — 16-bit Y16/TLinear, so each pixel is an absolute
   temperature rather than an auto-gain brightness. (AGC drifts as the scene
   changes, which makes any fixed threshold meaningless.)
2. **Ambient estimate** — median pixel; people are a minority of the frame, so
   the median tracks room temperature.
3. **Temperature band** — a pixel must be both warmer than the room *and*
   physiologically plausible: `T > ambient + delta` **and** `27 ≤ T ≤ 36 °C`.

   | Source | Typical °C | In band? |
   |---|---|---|
   | Exposed skin (face, hands) | 30 – 35 | ✅ |
   | Clothed torso / limbs | 27 – 32 | ✅ |
   | Sun-warmed surface | ~26 | ❌ below |
   | Laptop | ~41 | ❌ above |
   | Lamp | ~52 | ❌ above |
   | Radiator | ~60 | ❌ above |

   Because clutter is rejected by *temperature*, the minimum blob area can be
   small (6 px) — that is what gives range on distant targets. Tune with
   `--tmin` / `--tmax`.

   **Warm rooms:** if `ambient + delta` rises above `tmax`, the band would
   collapse and produce zero detections silently. The relative threshold is
   therefore capped at `tmax − 2`. Contrast genuinely degrades in a warm room
   (a clothed body may sit only 2 °C above ambient), so also lower `--delta`
   there — the band, not the delta, is what keeps clutter out.
4. **Morphology** — open (despeckle) then close (fill gaps).
5. **Watershed split** — two people standing close merge into one connected
   component, and no filter can undo that; the blob must be cut. A distance
   transform gives each body a peak with a valley between, and watershed cuts
   along the valley. This is the shape-domain analogue of the head-peak /
   shoulder-valley separation the depth sensor performs.
6. **Mounting mode** — the two geometries need different rules, and they are
   switchable live (later: selected automatically from the IMU).

   | mode | key | aspect (h/w) | extent | person looks like |
   |---|---|---|---|---|
   | `vertical` (ceiling, looking down) | `1` | 0.45 – 2.2 | ≥ 0.35 | head + shoulders: roughly round |
   | `horizontal` (wall, looking forward) | `2` | 1.1 – 6.0 | ≥ 0.25 | standing body: tall ellipse |
   | `any` (mount unknown) | `3` | 0.30 – 7.0 | ≥ 0.20 | permissive fallback |

   Press `v` to cycle, or `1`/`2`/`3` to jump. The active mode is shown in the
   overlay and recorded in the CSV, so logged runs stay interpretable.

7. **Fragment merge** — a body broken up by cool clothing must rejoin, but two
   adjacent people must not. The rule differs by mode, and this is the crux:

   - **horizontal**: two people are separated **side by side**; head/torso/legs
     stack **vertically**. So a horizontal split stays split, a vertical one
     merges back.
   - **vertical**: seen in plan view, two people can be separated along *any*
     image direction, so the axis carries no information. The physical limit is
     used instead — two bodies cannot be closer than shoulder width, so splits
     whose centres are nearer than `--min-sep` px (default 20) merge back.
     Tune this to your mount height: it is the pixel distance corresponding to
     roughly 40 cm on the floor.

Every detection is explainable: a temperature, a shape, and a geometry rule.
No weights, no training, no per-site calibration.

### Choosing settings

```bash
./run.sh thermal_detect.py --view vertical
./run.sh thermal_detect.py --view horizontal --min-area 4
./run.sh thermal_detect.py --view vertical --min-sep 28
./run.sh thermal_detect.py --tmax 45
./run.sh thermal_detect.py --no-split
```

Modes can also be switched live with `v` / `1` / `2` / `3`, so a single run can
cover both mounting tests.

Longer range comes from lowering `--min-area`; the `--tmax` band is what keeps
that safe. If distant people are still missed, lower the detection threshold
too (`[` / `]` live, or `--delta 2.5`) — at range, atmospheric attenuation and
pixel mixing reduce apparent contrast.

### Measured settings (bench-tested)

| Scenario | Settings | Why |
|---|---|---|
| **Vertical, single occupant** | `--view vertical --cohesion 4` | Max fragmentation resistance. The usual penalty of high cohesion — merging two adjacent people — cannot occur with one occupant, so it is free here. **Measured best on the bench.** |
| Vertical, multiple occupants | `--view vertical --cohesion 2` | Hard limit: at cohesion ≥3 two people standing close merge into one detection. |
| Horizontal, any occupancy | `--view horizontal --cohesion 1` | Vertical-stack merging already handles head/torso/legs; extra cohesion is not needed. |
| Long range | `--min-area 4 --delta 2.5` | Distant targets are few pixels and lower contrast. |

**Cohesion is not monotonic.** A larger CLOSE kernel changes blob topology,
which changes where watershed cuts, so raising cohesion can occasionally *add*
a detection. Tune by watching the mask view (`r`), not by assuming higher is
always calmer.

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
