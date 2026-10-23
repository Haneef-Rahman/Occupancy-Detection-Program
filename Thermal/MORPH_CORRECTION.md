# RGB → thermal annotation correction

Updated 23 October 2026. Applies to development dataset annotation only.

`windows_pipeline.py annotate` now uploads the selected displacement map to the Windows job. `rgb_annotation_worker.py` applies it after the existing RGB→IR alignment and before final clipping/label export. The default is `calibration/morph-20261023.json`, exported from the accepted cumulative 69-landmark calibration with all older fit inputs retained.

For a provisional IR box with centre **c**, evaluate **d(c)** once and add the same displacement to every corner. Box size is preserved during correction; only sensor-boundary clipping can reduce it. No RGB/thermal pixels, temperatures or original recordings are warped. Coordinates and displacements use native thermal pixels, not enlarged preview pixels.

The saved NumPy thin-plate RBF model includes centres, fitted weights, affine coefficients, normalization, support hull and source hashes. It reproduces the original fitted field without refitting or requiring SciPy at runtime. Current strength is **1.0**. Full-frame extrapolation is enabled; supported landmark-hull coverage is **67.48%**, not full-frame measured accuracy. Extrapolated proposals are flagged for review. A fixed 2D field remains an approximation to depth-dependent parallax.

The field's alignment hash must match the recording snapshot exactly. A mismatch fails before job upload on Mac and before model loading on Windows. Recalibrate/export a matching field after changing mount geometry or alignment; do not bypass the check by editing hashes.

Returned capture folders retain the usual `npy/`, `png/`, `labels/`, `manifest.csv` and class-1 head/shoulder labels. They also contain:

- `morph.json`: exact map used for this job.
- `rgb_annotation.json`: map hash, strength, support, per-box provisional/translated/final bounds and displacement, extrapolation/clipping/rejection details, transfer-stage durations and model timing.
- `review/`: thermal proposal previews.
- `READY_FOR_REVIEW`: completion marker, not human approval.

The source capture remains unchanged. No `labels_human/` are synthesized. Use `dataset_pipeline.py --live --capture-seeds` to review transferred proposals individually, remove thermally unrecognizable boxes and correct mistakes. Dataset/training requirements remain unchanged.

For the current experimental YOLOE candidate, the existing trial invocation gains morph automatically:

```sh
/Users/haneefrahman2650/Developer/Occupancy-Detection-Program/Thermal/.venv/bin/python \
  /Users/haneefrahman2650/Developer/Occupancy-Detection-Program/Thermal/windows_pipeline.py \
  annotate /absolute/path/to/capture_FOLDER --experimental --experimental-annotator
```

Run transfers as the normal Mac user, without sudo. Optional `--morph /absolute/path/to/map.json` selects a matching versioned map. The RGB annotator itself is still unvalidated; earlier omega prompt trials produced no boxes. Successful geometry tests do not establish detector competence, and empty proposals must not be treated as verified negative examples.
