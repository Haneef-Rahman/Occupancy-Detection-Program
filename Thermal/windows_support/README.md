# FLUXNET thermal capture — Windows

Hi Adrian. This folder is everything you need to record captures on your
machine that drop straight into the project alongside the Mac ones. You should
never need to touch anything outside this folder.

---

## The short version

```powershell
cd Thermal\windows_support
.\setup.ps1                                              # once, ~2 min
.\record.ps1 -Operator adrian -Note "cafeteria, 1430, 26C"
.\check.ps1                                              # before sending
```

Then zip `Thermal\logs\capture_*` and send them over.

In the recording window: **`r`** starts and stops recording, **`q`** quits.

---

## First time: test it in five steps

None of this has ever run on a real Windows machine — it was written on a Mac
against the docs. So please walk these in order rather than going straight out
to collect. Total time about ten minutes, and it fails cheaply at a desk
instead of expensively in hour two of a session.

**1. Self test — no camera needed.**
```powershell
.\selftest.ps1
```
Ten checks: imports, paths, the patches, and writing then re-reading a capture
using a fake camera that emits synthetic temperatures. If anything fails, stop
and send Haneef the output. Nothing here needs the Lepton.

**2. Camera probe — Lepton plugged in.**
```powershell
.\probe.ps1
```
This answers the one question the self test cannot: does your Lepton hand
OpenCV real temperatures? You want a line ending `<== USE THIS`. If nothing
does, **stop** and send the output — do not record.

**3. Look at the image.**
```powershell
.\preview.ps1
```
Skip this if you haven't got a model — it needs ultralytics, so it also needs
the environment to be on Python 3.12. Otherwise: point
it at yourself, check you're a bright blob, check the aim.

**4. A thirty-second capture, then verify it.**
```powershell
.\record.ps1 -Operator adrian -Note "desk test, 1500, ambient 24C"
.\check.ps1
```
Press `r`, wave at it for thirty seconds, press `r` again, then `q`. The check
should say `OK`. Look at the ambient it reports — if it says 24 °C and your
room is 24 °C, the whole chain is working.

**5. Send Haneef that one capture before doing anything longer.**

This is the step worth not skipping. He'll merge it with the Mac data and
confirm it lines up. If your Lepton reads half a degree off his, or the render
is subtly different, it shows up in one small file rather than after you've
collected four hours across six buildings.

---

## What you're actually collecting

A Lepton 3.1R gives a 160×120 grid of **temperatures** (160×122 if telemetry
is enabled — the two extra rows are stripped on read), not a picture. We record
the raw temperature array per frame, and the project uses them to detect the
head-and-shoulders outline of a person — never a face, never anything
identifiable. That's the whole point of FLUXNET: privacy comes from the physics,
not from a policy.

Your job is variety. The model was trained almost entirely inside HKU
makerspaces at 22–24 °C, and it fell over the first time it met a room full of
running 3D printers — it had literally never seen anything hotter than a human
face. **Different rooms, different temperatures, different clutter is worth far
more than more frames of the same corridor.**

---

## The one thing that can go wrong

Everything in this project assumes the numbers in a frame are **degrees
Celsius**.

Windows can hand OpenCV the Lepton stream in two forms: real 16-bit
radiometric data, or 8-bit "AGC" — a nice-looking auto-contrast picture with
the temperatures thrown away. If you record AGC, your files hold 0–255 grey
levels instead of degrees. Nothing will look broken. The recorder will run, the
window will show people, the files will be the right size.

It only shows up weeks later when the data is merged and your captures read as
a room at 14 °C.

So there are two guards, and please don't skip them:

- **`backend_probe.py`** (run by `setup.ps1`) tells you before you start.
- **the recorder refuses to start** if it can't get a radiometric stream.

If it refuses, don't force it — send Haneef the probe output.

---

## Files here

| file | what it does |
|---|---|
| `setup.ps1` | one-time: creates the environment, installs numpy + opencv, checks the camera |
| `selftest.ps1` | proves everything works without a camera — run this first |
| `probe.ps1` | checks the Lepton gives real temperatures — run this second |
| `record.ps1` | records a session |
| `check.ps1` | verifies captures before you send them |
| `pipeline.ps1` | turn your captures into annotated labels — no model needed |
| `preview.ps1` | live preview / tracker |
| `selftest.py` | the ten checks |
| `backend_probe.py` | finds which camera backend gives real temperatures |
| `verify_capture.py` | checks a capture is sound before you send it |
| `win_common.py` | shared Windows plumbing — camera, provenance, the geteuid fix |
| `w_dataset_recording.py` | the recorder |
| `w_live_yolo.py` | live preview with detections (needs a model) |
| `w_integrated_launcher.py` | full tracker (needs a model) |
| `w_dataset_pipeline.py` | build a dataset from captures (driven by `pipeline.ps1`) |
| `requirements-win.txt` | numpy and opencv, nothing else |

No PyTorch, no ultralytics, no CUDA. Recording doesn't need them — it saves
the raw temperature frames and leaves the labelling to Haneef — and leaving
them out turns a 2 GB install into a 60-second one. `record.ps1` notices they
are absent and records unlabelled automatically; you don't have to pass
anything. To be explicit about it, add `-NoModel`.

**Nothing in the parent folder is modified.** Every `w_*.py` imports the real
tool from `Thermal\` and swaps out only what's Unix-specific — the camera
opener, and `os.geteuid` which doesn't exist on Windows at all. No logic is
copied. Haneef's setup can't be broken from here, and any fix he makes upstream
arrives for free.

**Captures still land in `Thermal\logs\`**, exactly where the Mac writes them
and where the dataset pipeline looks. Nothing here writes to its own folder.

---

## Recording well

**Mount it around 2 m, tilted about 15° down.** That's the geometry the whole
project assumes, and honestly nobody has verified it properly yet — so if you
can measure yours and put it in the note, that's genuinely useful.

**Always put the ambient temperature in the note.** A thermometer, the weather
app, whatever. It's the one piece of context that can't be recovered later, and
it's the difference between a capture we can interpret and one we can't.

```powershell
.\record.ps1 -Operator adrian -Note "library 3F, 1430, ambient 24C, mounted 2.1m"
```

**Record an EMPTY version of every scene.** Same spot, same mount, nobody in
frame — warm chairs, laptops, radiators, sunlit floor. Thirty seconds is
plenty.

These are worth more than the ones with people in them. The training set
currently contains **zero** frames without a person, which means the model has
never once been shown a room and told "nothing here" — and that's exactly why
it invents people in a 3D printing lab.

```powershell
.\record.ps1 -Operator adrian -Note "library 3F EMPTY, 1445, ambient 24C"
```

**Anywhere with hot equipment is gold.** Print labs, server rooms, kitchens,
machine shops. Things above 45 °C saturate in the preview and look like a white
blob — that's expected, and the raw file keeps the true temperature. Those are
the captures that fix the actual known failure.

**Sessions of 5–15 minutes.** At 9 fps that's 2,700–8,000 frames, which is a
sensible chunk to review. Several short sessions in different places beat one
long one.

---

## If something breaks

**"running scripts is disabled on this system"**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
That window only. Changes nothing permanently.

**Python version — 3.12, please**
Recording needs only numpy and opencv and runs on anything. `preview.ps1`,
`w_live_yolo.py` and `w_integrated_launcher.py` need ultralytics, which needs
torch, and torch wheels lag new Python releases by months. `setup.ps1` prefers
3.12 and warns if it had to settle for something else. If it did and you want
the preview: install 3.12 from python.org, then

```powershell
.\setup.ps1 -Rebuild
```

**`ModuleNotFoundError` even though pip said it installed**
The Microsoft Store ships a fake `python.exe` that sits first on PATH with no
packages. `setup.ps1` avoids it by using the `py` launcher. To fix it properly:
Settings → Apps → Advanced app settings → App execution aliases → turn **off**
`python.exe` and `python3.exe`.

**`cd /d` doesn't work** — that's Command Prompt syntax. PowerShell just wants
`cd D:\path`.

**Camera opens but no frames / probe finds nothing**
Something else has it. Windows gives exclusive access — close Teams, OBS, the
FLIR app, any browser tab that ever asked for a camera. Then unplug and replug.

**Frames are 160x122, or temperatures like -273 C / +344 C appear**
That is telemetry: your Lepton has it enabled, so it sends two extra rows of
status data per frame. Nothing to change — the camera is asked what size it is
already set to and those rows are stripped before anything interprets the
frame. `source.txt` records that it happened. Found by you, 2026-09-14.

**"No radiometric stream found"**
Don't use `-AllowAgc` to get past it unless Haneef says so. That flag exists for
one rare case and it produces data we can't merge.

---

## Annotating (optional, and very welcome)

Recording is the job. But labelling is the bottleneck — one person drawing
boxes on thousands of clusters — so if you have time, this is where it goes
furthest.

```powershell
.\pipeline.ps1 -From 4
```

**You do not need a model, a GPU, or anything setup.ps1 skipped.** You draw the
boxes yourself; stages that use the network say SKIPPED and carry on.

Two left-clicks are OPPOSITE CORNERS of the head-and-shoulders box, not a drag.
Hover a corner and press DEL to remove a box, `u` undoes, `ENTER` commits and
moves on, `q` stops (everything already committed is saved).

The key worth knowing is **`g`**:

> `g` commits the frame as a HARD NEGATIVE — nobody in it, **and** it contains
> something that looks like it should have fired. A running 3D printer, sunlit
> pavement, a radiator, a laptop vent.

The training set contains **zero** frames without a person. The network has
never once been shown a scene and told there is nobody in it, which is exactly
why it invented people in a room full of hot printers. An empty corridor
teaches it almost nothing — nothing in it was ever going to fire. A hot printer
certified "no person" teaches it the thing it actually got wrong.

So the captures you were asked for in the section above are the same ones worth
annotating with `g`. Record the hot room, then mark it.

---

## Sending captures

```powershell
.\check.ps1
```

Fix anything marked `PROBLEM`; `WARNING` is usually fine (an empty-room capture
correctly warns that nothing is warm). Then zip `Thermal\logs\capture_*` and
send.

Each capture carries a `source.txt` recording your name, your machine, the
camera backend and your note — so if your Lepton turns out to read half a degree
different from Haneef's, it's a correction we can apply rather than a mystery.

Speaking of which: **worth doing once, early.** Put both Leptons side by side
looking at the same scene, record thirty seconds on each simultaneously, and
note it. Two sensors are two instruments, and knowing the offset between them
beforehand is much easier than inferring it afterwards.

Thanks for doing this — the variety is the whole value.
