# FLUXNET — thermal ↔ mmWave fusion

One identity per person, with measured 3D position and velocity.

`fuse.py` is the entry point for the whole system. It doesn't reimplement
either sensor — it assembles them:

| module | role |
|---|---|
| `../Thermal/integrated_launcher.py` | camera, omega CNN, Kalman tracking, occlusion memory, re-identification, live tuning, the UI |
| `radar_link.py` | radar thread, adaptive config switching, projection |
| `fuse.py` | association and the adjudication protocol |
| `../mmWave/project.py` | radar → camera-plane geometry |
| `../mmWave/adaptive.py` | the config-switching rule |

```bash
cd Fusion
../Thermal/.venv/bin/python fuse.py --self-test        # no hardware
sudo ../Thermal/.venv/bin/python fuse.py --live --adaptive
```

`sudo` because libuvc must claim the Lepton from the macOS kernel driver. The
Thermal venv is used because it already has torch, ultralytics and OpenCV; it
needs `pip install pyserial` once. Any flag `fuse.py` doesn't recognise is
forwarded verbatim to `integrated_launcher.py`.

---

## Why fuse these two

The sensors' failure modes **do not overlap**, and all four cases below were
measured on this hardware rather than argued from first principles:

| | thermal | radar |
|---|---|---|
| person sitting perfectly still | trivial — body heat doesn't depend on motion | **loses them**; static retention is a heuristic patch |
| radiator, laptop, lamp | **false positive**, permanently | invisible — doesn't move |
| oscillating fan | invisible — room temperature | **false positive**; dense, high-SNR, unambiguously moving |
| chair, monitor, desk | invisible | **false positive** with `fineMotionCfg` — four held 8–30 s each in a real workspace |
| absolute range, radial velocity | estimated from pixel height | **measured** |
| darkness, glare, backlight | fine | fine |

Requiring agreement kills two whole classes of false positive with one test.
Equally, it lets both sensors run *more* sensitively than either could alone —
`AOP_9m_sensitive` produces more clutter, and the clutter has no thermal
signature.

---

## The adjudication protocol

**Thermal is the authority on what is a person.** Radar answers *is something
there, where, and how fast* superbly, and *is it a person* not at all.

| situation | verdict |
|---|---|
| thermal confirms | **person** |
| thermal sees them, radar lost them (stationary) | **person** |
| thermal could see, never confirmed, moving | `UNCONFIRMED` — not a person |
| thermal could see, never confirmed, motionless | `CLUTTER` — furniture |
| radar-only, **outside** the camera's 95° | `UNSEEN` — not counted, not rejected |
| previously confirmed, thermal blinks < `--grace` frames | still a person |

Two qualifiers that the bare rule doesn't state, and both matter:

**Denial only counts where thermal could see.** The radar's field of view is
±70°; the Lepton's is 95° total. A radar track at 84° off-axis has not been
rejected — it has not been *examined*. Calling that "not a person" would put
words in thermal's mouth, so those are held as `UNSEEN`, excluded from the
count, and reported separately rather than silently deleted.

**A thermal blink is not a verdict.** The FFC shutter freezes the image about
a second, and brief occlusions happen. A track thermal has already vouched for
keeps the benefit of the doubt for `--grace 9` frames before the veto bites.

`--no-thermal-veto` restores the pre-protocol behaviour for an A/B.

---

## Division of labour

**Radar owns motion. Thermal owns continuity.**

The thermal Kalman filter infers velocity by differencing noisy pixel
positions — which is what produced ID churn in the first place. The radar
*measures* radial velocity from Doppler. So Doppler is blended into the
filter's velocity state at `--assist-alpha 0.6`, while the filter keeps owning
identity: births, re-identification, occlusion memory.

**The assist is lateral only, deliberately.** Converting m/s to px/frame needs
a range, and horizontal range is trustworthy while the vertical is not — see
the open z-reference question below. Assisting an axis we cannot yet interpret
would inject a bias dressed up as a measurement.

### Identity lives here, not in either sensor

Both sensors' IDs are short-lived handles:

- TI's group tracker **reuses** IDs after freeing them, so someone who drops
  out and returns comes back as a different ID — and their old one may later
  belong to somebody else.
- Reconfiguring the radar restarts the tracker and **resets every ID**.
  `radar_link` bumps a `generation` counter so this is never mistaken for
  people arriving.
- The thermal tracker churns IDs of its own under occlusion.

So the fused track owns a persistent ID and holds `(thermal_id, radar_id)` as
associations that go stale independently.

### Coasting a stationary person is nearly exact

When the radar drops someone **because they stopped moving**, their range has
not changed — that is the same fact that caused the dropout. So the fused track
keeps the last measured range and takes fresh bearing from the thermal box.
Range from the sensor that measures range; bearing from the sensor that
measures bearing.

---

## Association, and its error budget

Radar tracks are projected into the Lepton's image plane as **camera-facing
billboards** — width across the line of sight, height from the tracker's
`z_min`/`z_max` — and matched to thermal boxes by IoU, highest first, with a
centre-distance second pass.

**The second pass is not optional.** At 0.78°/px a 1.7 m person is ~25 px tall
at 5 m, the projected box jitters a few px from the radar's own angular noise,
and IoU collapses fast:

| offset | IoU |
|---|---|
| 1 px | 0.76 |
| 2 px | 0.57 |
| 3 px | 0.42 |
| 4 px | 0.29 |

A pure IoU gate would reject true pairs at exactly the range this system is
for. **IoU decides which match is best; distance decides whether a weak one is
real.**

What that budget costs:

- **Extrinsics.** 1° of rotation error = `73.3·tan(1°)` ≈ **1.3 px**, so the
  budget is roughly 1.5–2° total. A calibration, not a mounting job.
- **Distortion.** At 95° a pinhole model is not good enough; barrel distortion
  spends the entire budget before any other error. `--k1`/`--k2` default to
  **zero**, which is honest rather than right.
- **Irreducible.** A radar track accurate to ±15 cm at 5 m is ±1.7° ≈ 3 px. The
  box jitters no matter how perfect the geometry.

### Calibrating by eye

Stand at 3–4 m and nudge `--pitch` a degree at a time until the projected boxes
sit on the thermal ones. That single number absorbs both the mount tilt and the
z-reference offset, and is far quicker than resolving the firmware's
convention. Horizontal should already line up — **bearing does not depend on z**.

---

## Open question: the z reference

Measured on a real capture, the track centroid sits **below** the height TLV's
`z_min`:

```
z centroid   median  -0.14 m
z_min        median  +0.88 m
z_max        median  +1.16 m
```

The centre of a person cannot be beneath the bottom of them, so `z` and the
height TLV are **not in the same reference frame**. `z_min`/`z_max` look
floor-referenced and torso-shaped; the centroid looks sensor-referenced. If so,
the offset between them is the mounting height — about **1.3 m** by that data.

Until it's settled with a controlled capture (known mount height and tilt,
tape-measured distance, standing still), everything vertical carries an unknown
offset. Everything horizontal is unaffected, which is why the velocity assist
and the association both lean on bearing.

---

## Display

Each track carries the thermal confidence in its badge — that number is what
the veto protocol rests on and it stays visible whatever the radar says. Radar
output goes on its own lines beneath:

```
  p +1.2 +4.0 +0.9 m
  v -0.3 +0.7 +0.0 m/s
```

The motion arrow is drawn **in 3D**: the track's position and the point it
reaches in `--arrow-secs` are both projected, then joined. Someone walking
straight at the sensor therefore gets a short arrow and a growing box, which is
what is physically happening — a 2D arrow from `vx, vy` would be the same
length regardless of depth. Below `--arrow-min 0.15` m/s nothing is drawn, so
standing people don't twitch.

The sidebar gains a **RADAR** panel: mode, live target count, `matched/total`,
`rejected` (thermal looked and said no), `unseen` (outside the camera), the
velocity-assist alpha, and switch count with a `reconfig...` flag during the
~1.8 s blackout.

---

## Testing

Everything except the camera and the radio runs headless:

```bash
python fuse.py --self-test          # association, identity, the protocol
python radar_link.py                # projection, FoV flags, 3D passthrough
../mmWave/project.py --self-test    # geometry, error budget, IoU falloff
../mmWave/adaptive.py --replay logs/session.jsonl
```

Three real bugs were caught this way rather than in the field: a volumetric box
whose 2D hull was set by its nearest face (5 % oversize at 5 m), a `Rule` that
seeded its clocks from `time.monotonic()` and so could never switch under an
injected clock, and a fusion path feeding raw temperatures to a model trained
on a fixed 15–45 °C span.
