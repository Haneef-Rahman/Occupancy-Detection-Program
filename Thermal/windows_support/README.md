# FLUXNET thermal capture — Windows

Hi Adrian. This folder is everything you need to record captures on your
machine that drop straight into the project alongside the Mac ones. You should
never need to touch anything outside this folder.

---

## The short version

```powershell
cd Thermal\windows_support
.\setup.ps1                                              # once
.\record.ps1 -Operator adrian -Note "cafeteria, 1430, 26C"
.\check.ps1                                              # before sending
```

Then zip `Thermal\logs\capture_*` and send them over.

In the recording window: **`r`** starts and stops recording, **`q`** quits.

---

## What you're actually collecting

A Lepton 3.1R gives a 160×120 grid of **temperatures**, not a picture. We record
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
- **`record_win.py` refuses to record** if it can't get a radiometric stream.

If it refuses, don't force it — send Haneef the probe output.

---

## Files here

| file | what it does |
|---|---|
| `setup.ps1` | one-time: creates the environment, installs numpy + opencv, checks the camera |
| `record.ps1` | records a session |
| `check.ps1` | verifies captures before you send them |
| `backend_probe.py` | finds which camera backend gives real temperatures |
| `record_win.py` | the recorder (Windows camera handling + provenance) |
| `verify_capture.py` | the checker |
| `requirements-win.txt` | numpy and opencv, nothing else |

No PyTorch, no ultralytics, no CUDA. Recording doesn't need them, and leaving
them out turns a 2 GB install into a 60-second one.

**Nothing in the parent folder is modified.** `record_win.py` imports the Mac
recorder and swaps out two functions at runtime, so Haneef's setup can't be
broken by anything here.

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

**"No radiometric stream found"**
Don't use `-AllowAgc` to get past it unless Haneef says so. That flag exists for
one rare case and it produces data we can't merge.

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
