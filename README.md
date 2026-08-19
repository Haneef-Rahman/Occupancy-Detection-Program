# Occupancy-Detection-Program

Privacy-first occupancy detection: a multi-sensor node that counts people
without cameras, microphones, or per-site training.

Sensing is done with **mmWave radar** (motion, breathing micro-motion),
a **radiometric thermal camera** (body heat), and a **dToF SPAD depth sensor**
(head/shoulder separation). No modality can capture an identifiable image —
privacy is a property of the physics, not of a policy.

![FLUXNET thermal person detection](docs/hero_thermal_detection.png)

*Live output from the classical thermal detector. Every detection traces to a
temperature and a shape — no weights, no training set, no per-site calibration.
The warm square in frame is rejected on every frame because a body radiates from
several places at once and a heat source radiates from one. Measured over 332
frames: **94.0 % recall, 96.9 % precision**, and clutter false positives reduced
from 209 to 0 against the same run without the warm-centre filter. Reproduce it
from [`Thermal/datasets/`](Thermal/datasets/).*

## Repository layout

| Folder | Contents |
|---|---|
| `Thermal/` | FLIR Lepton 3.1R + PureThermal 3 — radiometric capture and person detection |
| _(planned)_ `Radar/` | TI IWR6843AOP — point cloud capture, config, TLV parsing |
| _(planned)_ `Depth/` | ST VL53L8CX / VL53L9CX — multizone depth |
| _(planned)_ `Fusion/` | multi-sensor association, tracking, crossing counts |

## Status

**Thermal — working.** Radiometric Y16 capture via libuvc, with a classical
(untrained) detector: threshold above ambient → morphology → connected
components → geometry filters → fragment merging. Measured on real frames:
ambient 23.7–25.8 °C, skin peaks 31.4–33.4 °C, **7.3 °C average contrast**.

See [`Thermal/README.md`](Thermal/README.md) for setup, the macOS capture
problem and its solution, and the validation procedure.

## Design principle: no per-site training

Detection is deliberately classical rather than learned. Every detection
traces to a temperature and a shape, so it is explainable and it generalises
across rooms without retraining — the calibration burden that has sunk
comparable thermal/CSI systems. Machine learning stays available for
sub-problems the baseline provably cannot solve, trained once on diverse data
and deployed frozen, never fitted per installation.

## Hardware

| Role | Part |
|---|---|
| Thermal | FLIR Lepton 3.1R (160×120 radiometric, 95° HFOV) + PureThermal 3 |
| Radar | TI IWR6843AOPEVM (60 GHz FMCW, antenna-on-package) |
| Depth | ST VL53L8CX (8×8) / VL53L9CX (2.3k zones) |
| Host | Raspberry Pi 5 (target) · macOS (development) |
