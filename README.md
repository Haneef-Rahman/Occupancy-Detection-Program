# Occupancy-Detection-Program

**FLUXNET** — privacy-first occupancy detection. A multi-sensor node that counts
people without cameras, microphones, or per-site training.

Sensing is done with a **radiometric thermal camera** (body heat), **mmWave
radar** (motion, micro-motion, absolute range and radial velocity), and a
planned **dToF SPAD depth sensor** (head/shoulder valley separation). No
modality can capture an identifiable image of anyone — privacy is a property of
the physics, not of a policy.

![FLUXNET thermal person detection](docs/hero_thermal_detection.png)

*Live thermal output. The warm square in frame is rejected on every frame
because a body radiates from several places at once and a heat source radiates
from one.*

---

## Status

| Sensor | State | Headline |
|---|---|---|
| **Thermal** | working, trained model deployed | classical **94.0 % recall / 96.9 % precision**; YOLO26n **mAP@50 0.991** |
| **mmWave** | working, headless pipeline | 12 m tracking at 16.7 fps, on-chip tracker, no host DSP; vital signs demonstrated (**10 breaths/min, 61.9 bpm**) |
| **Fusion** | **working** | radar projected into the thermal frame, IoU association, one persistent identity per person with measured 3D position and velocity |
| **Depth** | not started | ST VL53L8CX / VL53L9CX |

### Thermal

Two detectors. A **classical, untrained** one — threshold above ambient →
morphology → connected components → geometry, where every detection traces to a
temperature and a shape — and a **YOLO26n** trained on labels the classical
detector proposed and a human corrected.

The learned model went from mAP@50 **0.491 → 0.991** between v1 and v2. Not from
a bigger network or more epochs: v1's validation box loss *rose* while training
loss fell, which is a model fitting label noise. The fix was better labels, and
most of [`Thermal/README.md`](Thermal/README.md) is the annotation pipeline that
produced them.

### mmWave

TI IWR6843AOPEVM running TI's *3D People Tracking* demo. The group tracker runs
**on the radar**, so the host decodes a few hundred bytes at ~17 Hz and does no
signal processing — which is why this deploys to a Raspberry Pi unchanged.
Decoding uses TI's own parser, imported from the Radar Toolbox and verified to
have no Qt dependency.

See [`mmWave/README.md`](mmWave/README.md), including a documented firmware
constraint TI does not state anywhere: `fineMotionCfg` requires 48 chirp loops,
not 96.

### Fusion

`Fusion/fuse.py` is the entry point for the whole system. It assembles the two
sensors rather than reimplementing either: the thermal launcher supplies the
camera, CNN, Kalman tracking and UI; `radar_link` supplies the radar thread and
adaptive config switching; the fusion layer supplies association and the
adjudication protocol.

**Thermal is the authority on what is a person. Radar owns motion; thermal owns
continuity.** Doppler velocity is blended into the thermal Kalman filter, which
keeps owning identity — because both sensors' own IDs are short-lived handles
that get reused, reset on reconfigure, and churn under occlusion.

```bash
cd Fusion && sudo ../Thermal/.venv/bin/python fuse.py --live --adaptive
```

See [`Fusion/README.md`](Fusion/README.md).

---

## Repository layout

| Folder | Contents |
|---|---|
| `Thermal/` | FLIR Lepton 3.1R + PureThermal 3 — capture, annotation pipeline, classical + learned detection, tracking |
| `mmWave/` | TI IWR6843AOPEVM — flashing, configuration, TLV decoding, logging |
| `Fusion/` | cross-sensor association, the adjudication protocol, the fused entry point |
| `docs/` | project-level figures |
| _(planned)_ `Depth/` | ST VL53L8CX / VL53L9CX multizone depth |

## Hardware

| Role | Part |
|---|---|
| Thermal | FLIR Lepton 3.1R (160×120 radiometric, 95° HFOV) + PureThermal 3 |
| Radar | TI IWR6843AOPEVM (60 GHz FMCW, antenna-on-package, ±70° FoV) |
| Depth | ST VL53L8CX (8×8) / VL53L9CX (2.3k zones) |
| Host | Raspberry Pi 5 (target) · macOS (development) |

---

## Design principles

**No per-site training.** Detection must generalise across rooms without
recalibration — the burden that has sunk comparable thermal and CSI systems.
The classical detector needs none by construction. The learned model is trained
once on diverse data and deployed frozen, never fitted per installation.

**Measure, don't assume.** Every processing stage in the thermal detector is
individually toggleable so its contribution can be measured, and the active
configuration is written into every capture manifest so results stay
attributable. Several ideas that sounded good were measured and discarded:
warm-centroid propagation compensation (worse than nothing — p25 IoU 0.231 →
0.183), background subtraction as a detection threshold (deleted 1842 valid
labels, because a person seated at a desk *is* their own background), and a 9 m
radar config with static retention (measurably worse than TI's stock).

**Sensors that fail differently.** All measured on this hardware: a radiator is
a permanent thermal false positive and invisible to radar; an oscillating fan is
a dense, high-SNR, unambiguously moving radar target with no thermal signature;
and with static retention enabled the radar held four motionless "targets" —
furniture — for 8-30 s each in a real workspace. The failure modes do not
overlap, so requiring spatial agreement kills both classes with one test. It
also lets each sensor run *more* sensitively than it safely could alone.

---

## Quick start

```bash
# thermal — live classical detection
cd Thermal && ./run.sh thermal_detect.py --view any --cohesion 2 --delta 2.5

# thermal — trained model
cd Thermal && ./run.sh live_yolo.py --weights models/v2/best.pt --conf 0.374

# radar — headless tracking
cd mmWave && ./.venv/bin/python ti_track.py --cfg configs/AOP_9m_default.cfg

# both — fused, one identity per person, 3D position and velocity
cd Fusion && sudo ../Thermal/.venv/bin/python fuse.py --live --adaptive
```

Full setup, including the macOS libuvc recipe the thermal camera requires, is in
each sensor's README.
