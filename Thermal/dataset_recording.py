#!/usr/bin/env python3
"""
Record a capture log with YOLO omega labels and nothing else.

    python3 dataset_recording.py                 # latest model, conf 0.37
    python3 dataset_recording.py --conf 0.40
    python3 dataset_recording.py --weights models/v2/best.pt

    r        start / stop recording
    q, ESC   quit

WHY OMEGA ONLY. The person box and the omega box are different targets, and
thermal_detect.py wrote both because which one a model should learn was an open
question. It is no longer open: omega is what the tracker consumes, because two
people at 0.5 m share one warm silhouette but keep distinct heads. Writing a
person box here would put a class into the dataset that nothing downstream
reads, and every one of them would have to be reviewed by a human who then
discards it. So class 0 is never emitted.

The label file still uses CLASS INDEX 1 for omega, and classes.txt still lists
both names. That is deliberate. triage.py, annotate.py, review.py and
make_dataset.py all assume omega == 1; renumbering it to 0 here would silently
turn every omega in this capture into a person everywhere downstream.

ONE CONSEQUENCE, STATED UP FRONT. triage.py chooses its annotation cut-off from
the number of PERSON boxes per frame. A capture recorded here has none, so every
frame reads as "0 person boxes" and every cluster gets flagged for annotation.
That is the intended outcome — you are annotating all of it — but it means
triage's cut-off histogram and its propagation-drift self-check are both
uninformative on these captures. dataset_pipeline.py accounts for this.

WHAT IS NOT RECORDED. No classical detection runs. No background model, no
threshold, no watershed, no Kalman. The manifest keeps those columns so the file
stays readable by the same tools, and writes the values it can honestly fill.
"""

import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime

import cv2
import numpy as np

import thermal_detect as TD

SPAN_C = (15.0, 45.0)     # MUST match make_dataset.py and thermal_detect.PNG_SPAN_C
OMEGA_CLASS = 1           # index in classes.txt — do not change
LOG_DIR = "logs"

BOX_COL = (60, 220, 255)  # same yellow annotate.py and review.py use for omega


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------
def latest_weights(models_dir="models"):
    """
    Highest-numbered models/vN/best.pt.

    Sorted numerically, not lexically: v10 must beat v9, and a string sort puts
    "v10" before "v9". There is only v1 and v2 today, which is exactly when this
    kind of bug gets written and not noticed.
    """
    cands = []
    for p in glob.glob(os.path.join(models_dir, "v*", "best.pt")):
        m = re.search(r"v(\d+)", os.path.basename(os.path.dirname(p)))
        if m:
            cands.append((int(m.group(1)), p))
    if not cands:
        return None
    return max(cands)[1]


# ---------------------------------------------------------------------------
# Camera + encoding
# ---------------------------------------------------------------------------
def render_for_cnn(data):
    """The encoding the model was trained on. Fixed span, 3-channel, no AGC."""
    lo, hi = SPAN_C
    v = np.clip((data.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    g = (v * 255.0).astype(np.uint8)
    return cv2.merge([g, g, g])


def open_camera(args):
    """read() returns (data, flag) — data FIRST. `data is None` is the failure."""
    if not args.opencv:
        try:
            from lepton_libuvc import LeptonUVC
            print("  libuvc: radiometric Y16")
            return LeptonUVC()
        except Exception as e:
            print(f"  libuvc unavailable: {e}\n  falling back to OpenCV ...")
    return TD.ThermalCamera(args.device if args.device is not None else 0)


def yolo_omegas(model, data, conf, imgsz):
    """
    Returns [(x0, y0, w, h, confidence), ...] for class 1 only.

    Class 0 detections are dropped here rather than filtered later, so there is
    no path by which a person box reaches the label file.
    """
    res = model.predict(render_for_cnn(data), verbose=False,
                        conf=conf, imgsz=imgsz)[0]
    out = []
    if len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        for (x0, y0, x1, y1), c, v in zip(xyxy, cls, cf):
            if c != OMEGA_CLASS:
                continue
            out.append((float(x0), float(y0),
                        float(x1 - x0), float(y1 - y0), float(v)))
    return out


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def label_at(img, text, x, y, col):
    """Draw a label clear of the box edge, clamped to stay on screen."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    ly = max(th + 3, y - 4)
    lx = min(max(0, x), img.shape[1] - tw - 2)
    cv2.putText(img, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                col, 1, cv2.LINE_AA)


def draw(data, omegas, S, recording, n_frames, conf, capture_name):
    vis = cv2.resize(TD.colorize(data), None, fx=S, fy=S,
                     interpolation=cv2.INTER_NEAREST)
    for (x0, y0, w, h, v) in omegas:
        p0 = (int(round(x0 * S)), int(round(y0 * S)))
        p1 = (int(round((x0 + w) * S)), int(round((y0 + h) * S)))
        cv2.rectangle(vis, p0, p1, BOX_COL, 2)
        label_at(vis, f"{v:.2f}", p0[0], p0[1], BOX_COL)

    bar_h = 26
    cv2.rectangle(vis, (0, 0), (vis.shape[1], bar_h), (14, 14, 17), -1)
    left = f"omega x{len(omegas)}   conf>={conf:.2f}"
    cv2.putText(vis, left, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (210, 210, 220), 1, cv2.LINE_AA)

    if recording:
        txt = f"REC  {n_frames} frames   {capture_name}"
        cv2.circle(vis, (vis.shape[1] - 14, 13), 6, (80, 80, 255), -1)
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.putText(vis, txt, (vis.shape[1] - 28 - tw, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 255), 1, cv2.LINE_AA)
    else:
        hint = "r = record"
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.putText(vis, hint, (vis.shape[1] - 10 - tw, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 140, 150), 1,
                    cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# Capture log
# ---------------------------------------------------------------------------
MANIFEST_COLS = ["file", "timestamp", "view", "background", "threshold",
                 "n_det", "omega_count", "omega_score", "flags", "note"]


def open_capture(note, conf, weights):
    """Create logs/capture_YYYYMMDD_HHMMSS in the layout every other tool expects."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(LOG_DIR, f"capture_{stamp}")
    for sub in ("npy", "png", "labels", "review"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    with open(os.path.join(d, "classes.txt"), "w") as fh:
        fh.write("person\nhead_shoulder\n")
    # Provenance, so a capture can always be traced to the model that labelled
    # it. Without this a dataset built from several sessions has no record of
    # which weights produced which silver labels.
    with open(os.path.join(d, "source.txt"), "w") as fh:
        fh.write(f"recorder   dataset_recording.py\n"
                 f"weights    {weights}\n"
                 f"conf       {conf}\n"
                 f"classes    omega only (class {OMEGA_CLASS}); no person boxes\n"
                 f"started    {datetime.now().isoformat(timespec='seconds')}\n"
                 f"note       {note}\n")
    fh = open(os.path.join(d, "manifest.csv"), "w", newline="")
    w = csv.writer(fh)
    w.writerow(MANIFEST_COLS)
    return d, fh, w


def write_frame(d, w, n, data, omegas, note, no_review):
    fn = f"cap_{n:06d}"
    np.save(os.path.join(d, "npy", fn + ".npy"), data)

    # 16-bit linear over a FIXED span, so a pixel value means the same
    # temperature across the whole dataset. Never the live palette — that is
    # percentile-stretched per frame, which is exactly the nuisance variation a
    # model must not be taught.
    lo, hi = SPAN_C
    png = (np.clip((data - lo) / (hi - lo), 0, 1) * 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(d, "png", fn + ".png"), png)

    IH, IW = data.shape[:2]
    lines = [f"{OMEGA_CLASS} {(x + w_ / 2) / IW:.6f} {(y + h_ / 2) / IH:.6f} "
             f"{w_ / IW:.6f} {h_ / IH:.6f}"
             for (x, y, w_, h_, _) in omegas]
    with open(os.path.join(d, "labels", fn + ".txt"), "w") as lf:
        lf.write("\n".join(lines) + ("\n" if lines else ""))

    # QA copy with boxes drawn, kept OUT of png/ so it can never be fed to a
    # trainer by accident.
    if not no_review:
        R = 4
        rv = cv2.resize(TD.colorize(data), (IW * R, IH * R),
                        interpolation=cv2.INTER_NEAREST)
        for (x, y, w_, h_, v) in omegas:
            cv2.rectangle(rv, (int(x * R), int(y * R)),
                          (int((x + w_) * R), int((y + h_) * R)), BOX_COL, 1)
            label_at(rv, f"{v:.2f}", int(x * R), int(y * R), BOX_COL)
        cv2.rectangle(rv, (0, 0), (rv.shape[1], 20), (14, 14, 17), -1)
        cv2.putText(rv, f"{fn}    omega={len(omegas)}", (6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 220), 1,
                    cv2.LINE_AA)
        cv2.imwrite(os.path.join(d, "review", fn + ".png"), rv)

    scores = [v for (_, _, _, _, v) in omegas]
    w.writerow([fn, datetime.now().isoformat(timespec="milliseconds"),
                "horizontal", "", "", 0, len(omegas),
                f"{max(scores, default=0.0):.4f}", "YOLO-omega", note])


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Record a capture log labelled by YOLO, omega class only.")
    ap.add_argument("--weights", default=None,
                    help="path to best.pt. Default: highest models/vN/best.pt")
    ap.add_argument("--conf", type=float, default=0.37,
                    help="detection threshold for the RECORDED labels. These "
                         "become silver labels, so this is a precision "
                         "decision: too low and you hand the annotator "
                         "rubbish to delete.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--scale", type=int, default=5)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--opencv", action="store_true")
    ap.add_argument("--capture-every", type=int, default=1,
                    help="record 1 frame in N. The Lepton runs at 9 fps and "
                         "consecutive frames are near-identical, so >1 costs "
                         "little diversity and a lot of disk.")
    ap.add_argument("--no-review", action="store_true",
                    help="skip the review/ QA renders")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    weights = args.weights or latest_weights()
    if not weights or not os.path.exists(weights):
        sys.exit("no weights found — pass --weights, or put a model at "
                 "models/vN/best.pt")
    print(f"model: {weights}")

    from ultralytics import YOLO
    model = YOLO(weights)

    cam = open_camera(args)
    print(f"conf:  {args.conf}   omega only, class {OMEGA_CLASS}")
    print("r = record, q = quit")

    win = "dataset recording"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    recording = False
    n = 0
    frame_i = 0
    cap_dir = cap_fh = cap_w = None
    finished = []

    try:
        while True:
            data, _ = cam.read()
            if data is None:
                print("camera returned no frame — stopping")
                break

            omegas = yolo_omegas(model, data, args.conf, args.imgsz)

            if recording and frame_i % max(1, args.capture_every) == 0:
                write_frame(cap_dir, cap_w, n, data, omegas, args.note,
                            args.no_review)
                n += 1
            frame_i += 1

            name = os.path.basename(cap_dir) if cap_dir else ""
            cv2.imshow(win, draw(data, omegas, args.scale, recording, n,
                                 args.conf, name))

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("r"):
                if not recording:
                    cap_dir, cap_fh, cap_w = open_capture(args.note, args.conf,
                                                          weights)
                    n = 0
                    recording = True
                    print(f"recording -> {cap_dir}")
                else:
                    recording = False
                    cap_fh.close()
                    finished.append((cap_dir, n))
                    print(f"stopped: {n} frames in {cap_dir}")
                    cap_dir = cap_fh = cap_w = None
    finally:
        if recording and cap_fh:
            cap_fh.close()
            finished.append((cap_dir, n))
        cam.release()
        cv2.destroyAllWindows()

    if not finished:
        print("\nno frames recorded")
        return

    print("\n" + "=" * 60)
    print("CAPTURE LOG" + ("S" if len(finished) > 1 else ""))
    for d, cnt in finished:
        print(f"  {os.path.basename(d)}    {cnt} frames")
    print("=" * 60)
    print("\nrecord more, then:  python3 dataset_pipeline.py")


if __name__ == "__main__":
    main()
