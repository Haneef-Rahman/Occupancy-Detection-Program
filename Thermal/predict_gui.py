#!/usr/bin/env python3
"""
Drop a frame on the window, see what the model does with it.

    ./run.sh predict_gui.py                    # newest models/vN
    ./run.sh predict_gui.py --weights models/v2/best.pt

WHY THIS EXISTS. live_yolo.py runs the model on the camera, and that is the
right tool when the camera is plugged in. It is the wrong tool for the far more
common question: "what does the model think of THIS frame?" — a .npy you saved
six weeks ago, a .png out of a dataset split, a frame a collaborator emailed.
Until now that meant writing four lines of throwaway script every time, and the
throwaway script always used the wrong encoding.

ENCODING IS THE WHOLE POINT. The model was trained on a FIXED 15-45 C span
(make_dataset.py, thermal_detect.PNG_SPAN_C, live_yolo.SPAN_C). A per-frame
percentile stretch looks better on screen and silently destroys accuracy,
because the same person then encodes to different pixel values depending on
what else is in frame. This file keeps the two renders strictly separate:

    render_for_cnn()   fixed span, 3-channel, no AGC   -> the model
    render_for_eyes()  per-frame stretch + colormap    -> you

If you only remember one thing from this file, remember that they are different
functions and only one of them is allowed near the model.

RADIOMETRIC vs ALREADY-RENDERED. A .npy off this project is degrees Celsius and
must go through the fixed span. A .png has ALREADY been through it and must
not go through it twice -- doing so would clip almost everything to black. The
loader decides which it is holding and says so in the status bar, because
silently guessing is how you spend an afternoon debugging a model that is fine.
"""

import argparse
import os
import sys
import threading
import traceback

import cv2
import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

# Drag-and-drop is optional. The Browse button always works; without tkinterdnd2
# you simply lose the drop target, and the status bar says how to get it back.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAVE_DND = True
except Exception:
    HAVE_DND = False


# ---------------------------------------------------------------- constants

SPAN_C = (15.0, 45.0)          # MUST match make_dataset.py / live_yolo.py
IFOV = np.radians(95.0 / 160)  # 10.363 mrad/px. Lepton 3.1R (p/n 500-0758-03),
                               # 95 deg HFOV, f-theta: FLIR's published 119 deg
                               # diagonal matches 200 px x 0.59375 deg = 118.75.
                               # NOT the 3.5 -- that lens is 57 deg and every
                               # range here would be 1.67x too large.
W_NOM = 0.489                  # mean apparent omega width, m (5-point calibration)

# Measured box-width correction, from the 2026-09-20 five-point range study.
# raw box width (px) -> px to ADD to recover the true omega width. The model
# shrinks large boxes toward its training distribution and the optics inflate
# small ones, so the correction changes sign. np.interp clamps outside the
# anchors, which is the saturating behaviour we measured, not an accident.
CORR_X = np.array([8.47, 10.43, 13.90, 25.85])
CORR_Y = np.array([-2.0, -2.0, 0.0, +4.0])

# Display stretches for the EYE render only. render_for_eyes() used a fixed
# p1-p99.5 window, which is right for a room with two people in it: most of the
# palette separates a warm body from a cool wall. In a crowded room it is
# exactly wrong — p1 is cold ceiling, p99.5 is somebody's face, and the
# distinction you actually need (this shoulder against the next) is squeezed
# into a few grey levels. Measured on capture_20260921_100706: bodies span
# 7.7 C inside an 18.6 C scene.
#
# NONE OF THIS REACHES THE MODEL. render_for_cnn() is untouched and still uses
# the fixed SPAN_C, so cycling this cannot change a single detection.
VIEWS = (
    ("auto   p1-p99.5", 1.0, 99.5),
    ("crowd  p40-p99.5", 40.0, 99.5),
    ("dense  p70-p100", 70.0, 100.0),
)

# Face-temperature overlay.
#
# The hottest compact region of a person is the face: uncovered skin, no
# clothing between it and the sensor. Everything else about a human in thermal
# is attenuated by what they are wearing, which is why absolute body
# temperature is useless here and the face is not.
#
# There is no universal number for it. The published ~34 C for exposed skin is
# an indoor-at-rest figure and it moves with ambient, airflow, exertion,
# vasoconstriction, how long someone has been sitting, and how far they are
# from the sensor (a distant face fills less than a pixel and averages with the
# wall behind it). Hence a slider rather than a constant — you set it per scene
# by watching the count.
FACE_MIN_PX = 2                 # smaller than this is sensor noise, not a face
FACE_BAND_C = 2.5               # width of the band above the threshold

NAMES = ["person", "omega"]
COLOR = {"person": (120, 255, 90), "omega": (255, 220, 60)}   # RGB
SUPPORTED = (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def width_correction(raw_px):
    """px to add to a raw box width to recover the true omega width."""
    return float(np.interp(raw_px, CORR_X, CORR_Y))


def implied_range(raw_px):
    """Slant range in metres from a raw box width, or None if unusable."""
    w = raw_px + width_correction(raw_px)
    if w <= 0.5:
        return None
    return W_NOM / (w * IFOV)


# ---------------------------------------------------------------- rendering

def render_for_cnn(celsius):
    """The encoding the model was trained on. Fixed span, 3-channel, no AGC."""
    lo, hi = SPAN_C
    v = np.clip((celsius.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    g = (v * 255.0).astype(np.uint8)
    return cv2.merge([g, g, g])


def render_for_eyes(celsius, view=0):
    """Per-frame stretch + colormap. For your eyes only -- never for the model."""
    a = celsius.astype(np.float32)
    _, plo, phi = VIEWS[view % len(VIEWS)]
    lo, hi = np.percentile(a, plo), np.percentile(a, phi)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    v = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    g = (v * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def face_blobs(celsius, t_face, band=FACE_BAND_C, min_px=FACE_MIN_PX):
    """
    Compact regions in [t_face, t_face + band] — candidate faces.

    A BAND, not a floor. With a floor, raising the slider past the hottest
    pixel makes every blob vanish at once and you learn nothing; with a band
    you sweep a window through the temperature distribution and watch which
    structures survive. It also rejects genuinely hot clutter above the band —
    a laptop vent at 45 C is not a face and should not become one just because
    you set the threshold low.

    Returns [(cx, cy, radius_px, peak_C), ...].
    """
    a = celsius.astype(np.float32)
    mask = ((a >= t_face) & (a <= t_face + band)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):                       # 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_px:
            continue
        cx, cy = cent[i]
        peak = float(a[lab == i].max())
        out.append((float(cx), float(cy), max(1.5, (area / np.pi) ** 0.5), peak))
    return out


# ---------------------------------------------------------------- loading

class Frame:
    """One loadable thing: the array the model sees and the array you see."""

    def __init__(self, cnn_rgb, eye_rgb, kind, note, stack=None, index=0,
                 celsius=None):
        self.cnn_rgb = cnn_rgb      # uint8 HxWx3, fixed-span encoding
        self.eye_rgb = eye_rgb      # uint8 HxWx3, pretty
        self.kind = kind            # "radiometric" | "pre-rendered"
        self.note = note
        self.stack = stack          # (N,H,W) celsius, or None
        self.index = index
        self.celsius = celsius      # HxW float32 C, or None if pre-rendered


def _celsius_from(arr):
    """
    Interpret a raw array as degrees Celsius.

    A Lepton speaks Y16: centi-kelvin as uint16, so ~29600 counts is room
    temperature. Anything already in a plausible Celsius range is left alone.
    Getting this backwards produces an all-black frame and a model that finds
    nothing, which looks like a model bug and is very much not one.
    """
    a = arr.astype(np.float32)
    med = float(np.median(a))
    if med > 10000:
        return a / 100.0 - 273.15, "centi-kelvin -> C"
    if 150 < med < 400:
        return a - 273.15, "kelvin -> C"
    return a, "already Celsius"


def _looks_like_raw_y16(img):
    """
    Distinguish a RAW Y16 frame from a 16-bit frame that has already been
    span-encoded to 0..65535.

    This project's dataset_recording.py writes the SECOND kind:

        png = (clip((data - 15) / 30, 0, 1) * 65535).astype(uint16)

    Assuming such a file is raw Y16 and subtracting 273.15 K sends a median of
    ~17500 to -97 C, the span clips the whole frame to black, and the model
    reports one spurious box. That was a real bug here, so the test is on the
    RELATIVE spread rather than dtype: true centi-kelvin sits in a narrow band
    around room temperature (a few percent of its median), whereas a
    span-encoded frame is stretched across most of the 16-bit range.
    """
    a = img.astype(np.float32)
    med = float(np.median(a))
    if not (20000 < med < 40000):          # outside 200-400 K: not centi-kelvin
        return False
    spread = float(a.max() - a.min()) / max(med, 1.0)
    return spread < 0.25


def _from_prerendered(g8, w, h, note):
    cnn = cv2.merge([g8, g8, g8])
    eye = cv2.cvtColor(cv2.applyColorMap(g8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    return Frame(cnn, eye, "pre-rendered", f"{w}x{h}  {note}")


def load_path(path, index=0, interpret="auto"):
    """
    interpret: "auto" | "radiometric" | "pre-rendered"

    The override exists because a silent wrong guess here is indistinguishable
    from a broken model, and costs an afternoon. Whatever is chosen is reported
    back in Frame.note so it is never invisible.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        stack = None
        if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
            stack = arr
            arr = arr[min(index, len(arr) - 1)]
        elif arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"expected a 2-D frame or a stack, got shape {arr.shape}")

        if interpret == "pre-rendered":
            g = _to_u8_full_range(arr)
            f = _from_prerendered(g, arr.shape[1], arr.shape[0],
                                  f"{arr.dtype}  forced pre-rendered")
            f.stack, f.index = stack, index
            return f

        cel, how = _celsius_from(arr)
        note = f"{arr.shape[1]}x{arr.shape[0]}  {arr.dtype}  {how}"
        if stack is not None:
            note += f"  |  frame {index + 1} of {len(stack)}"
            cel_stack = np.stack([_celsius_from(f)[0] for f in stack])
        else:
            cel_stack = None
        return Frame(render_for_cnn(cel), render_for_eyes(cel),
                     "radiometric", note, cel_stack, index, celsius=cel)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("could not read that file as an image")
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    if img.dtype == np.uint16:
        raw = _looks_like_raw_y16(img) if interpret == "auto" \
            else (interpret == "radiometric")
        if raw:
            cel, how = _celsius_from(img)
            return Frame(render_for_cnn(cel), render_for_eyes(cel),
                         "radiometric", f"{w}x{h}  16-bit  raw, {how}", celsius=cel)
        # 16-bit already span-encoded to 0..65535 (dataset_recording.py writes
        # these). Rescale to 8 bits; do NOT put it through the span again.
        #
        # Integer-divide by 257, do not scale-and-round. 65535 = 255 * 257, and
        # both encoders TRUNCATE:
        #     npy path   u8  = floor(255 * v)
        #     png path   u16 = floor(255 * v * 257)
        # so u16 // 257 == floor(255 * v) exactly, for every v. Rounding instead
        # leaves a half-count offset that flips borderline boxes at the
        # confidence threshold and makes the same frame score differently
        # depending on which file you happened to open.
        g = (img // 257).astype(np.uint8)
        return _from_prerendered(g, w, h, "16-bit  span already applied (//257)")

    if interpret == "radiometric":
        cel, how = _celsius_from(img)
        return Frame(render_for_cnn(cel), render_for_eyes(cel),
                     "radiometric", f"{w}x{h}  8-bit  forced raw, {how}", celsius=cel)
    # 8-bit: already through the fixed span. Re-applying it would clip to black.
    return _from_prerendered(img.astype(np.uint8), w, h,
                             "8-bit  span already applied")


def _to_u8_full_range(arr):
    """Rescale an arbitrary array to 0..255 assuming it is already an encoding."""
    a = arr.astype(np.float32)
    if arr.dtype == np.uint16:
        return (a / 65535.0 * 255.0).round().astype(np.uint8)
    if arr.dtype == np.uint8:
        return arr
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return ((a - lo) / (hi - lo) * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------- inference

class Predictor:
    """Loads the model once, lazily, so the window appears immediately."""

    def __init__(self, weights):
        self.weights = weights
        self.model = None
        self.names = {}

    def ensure(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.weights)
            self.names = getattr(self.model, "names", None) or \
                {i: n for i, n in enumerate(NAMES)}
        return self.model

    def run(self, cnn_rgb, conf, imgsz):
        m = self.ensure()
        r = m.predict(cnn_rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append({
                "cls": self.names.get(int(b.cls), str(int(b.cls))),
                "conf": float(b.conf),
                "xyxy": (x1, y1, x2, y2),
                "w": x2 - x1,
                "h": y2 - y1,
            })
        out.sort(key=lambda d: -d["conf"])
        return out


def draw(rgb, dets, show_classes, scale):
    """
    Overlay boxes. Omega labels go above their box and person labels below,
    because an omega sits at the top of its person and the two labels would
    otherwise land on the same pixels -- the thing that makes Ultralytics' own
    val_batch previews unreadable.
    """
    big = cv2.resize(rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale),
                     interpolation=cv2.INTER_NEAREST)
    for d in dets:
        if show_classes != "both" and d["cls"] != show_classes:
            continue
        x1, y1, x2, y2 = (v * scale for v in d["xyxy"])
        col = COLOR.get(d["cls"], (255, 255, 255))
        cv2.rectangle(big, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        txt = f"{d['cls']} {d['conf']:.2f}  {d['w']:.1f}px"
        if d["cls"] == "omega":
            rng = implied_range(d["w"])
            if rng:
                txt += f"  ~{rng:.1f}m"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        above = d["cls"] == "omega"
        ly = max(th + 4, int(y1) - 5) if above else min(big.shape[0] - 4, int(y2) + th + 6)
        lx = min(max(0, int(x1)), big.shape[1] - tw - 4)
        cv2.rectangle(big, (lx, ly - th - 4), (lx + tw + 6, ly + 3), (18, 18, 20), -1)
        cv2.putText(big, txt, (lx + 3, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    col, 1, cv2.LINE_AA)
    return big


# ---------------------------------------------------------------- the window

BG = "#16161a"
FG = "#e8e8ec"
MUTED = "#8b8b96"


class App:
    def __init__(self, root, predictor, args):
        self.root = root
        self.pred = predictor
        self.args = args
        self.frame = None
        self.path = None
        self.dets = []
        self.show = "both"
        self.view = 0            # eye-render stretch, VIEWS index
        self.faces_on = False    # face-temperature overlay
        self.face_blobs = []
        self._face_temp_set = False

        root.title("Omega predictor")
        root.configure(bg=BG)
        root.geometry("1120x760")

        top = tk.Frame(root, bg=BG)
        top.pack(fill="x", padx=12, pady=(12, 6))

        tk.Button(top, text="Browse…", command=self.browse,
                  bg="#2a2a33", fg=FG, relief="flat", padx=14, pady=6,
                  activebackground="#3a3a46", activeforeground=FG,
                  highlightthickness=0, bd=0).pack(side="left")

        self.save_btn = tk.Button(top, text="Export ▾", command=self.export_menu,
                                  bg="#2a2a33", fg=FG, relief="flat", padx=14, pady=6,
                                  activebackground="#3a3a46", activeforeground=FG,
                                  highlightthickness=0, bd=0, state="disabled")
        self.save_btn.pack(side="left", padx=(8, 0))

        tk.Label(top, text="confidence", bg=BG, fg=MUTED).pack(side="left", padx=(20, 6))
        self.conf = tk.DoubleVar(value=args.conf)
        s = ttk.Scale(top, from_=0.01, to=0.95, variable=self.conf,
                      command=lambda _=None: self.on_conf(), length=180)
        s.pack(side="left")
        self.conf_lbl = tk.Label(top, text=f"{args.conf:.2f}", bg=BG, fg=FG, width=5)
        self.conf_lbl.pack(side="left")

        self.cls_btn = tk.Button(top, text="classes: both", command=self.cycle_classes,
                                 bg="#2a2a33", fg=FG, relief="flat", padx=12, pady=6,
                                 activebackground="#3a3a46", activeforeground=FG,
                                 highlightthickness=0, bd=0)
        self.cls_btn.pack(side="left", padx=(16, 0))

        # An override, because a silent wrong guess about the encoding looks
        # exactly like a broken model. Auto is right for everything this
        # project writes; the other two are for frames from elsewhere.
        self.interpret = "auto"
        self.int_btn = tk.Button(top, text="input: auto", command=self.cycle_interpret,
                                 bg="#2a2a33", fg=FG, relief="flat", padx=12, pady=6,
                                 activebackground="#3a3a46", activeforeground=FG,
                                 highlightthickness=0, bd=0)
        self.int_btn.pack(side="left", padx=(8, 0))

        # Contrast. Display only — says so on the tin, because anything in this
        # window that looked like it might touch the model would be a menace.
        self.view_btn = tk.Button(top, text="view: auto", command=self.cycle_view,
                                  bg="#2a2a33", fg=FG, relief="flat", padx=12,
                                  pady=6, activebackground="#3a3a46",
                                  activeforeground=FG, highlightthickness=0, bd=0)
        self.view_btn.pack(side="left", padx=(8, 0))

        self.face_btn = tk.Button(top, text="faces: off", command=self.toggle_faces,
                                  bg="#2a2a33", fg=FG, relief="flat", padx=12,
                                  pady=6, activebackground="#3a3a46",
                                  activeforeground=FG, highlightthickness=0, bd=0)
        self.face_btn.pack(side="left", padx=(8, 0))

        # Face temperature. Second row, because it is only meaningful with the
        # overlay on and it needs the width.
        row2 = tk.Frame(root, bg=BG)
        row2.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(row2, text="face temp", bg=BG, fg=MUTED).pack(side="left",
                                                               padx=(0, 6))
        self.face_t = tk.DoubleVar(value=args.face_temp)
        ttk.Scale(row2, from_=15.0, to=40.0, variable=self.face_t,
                  command=lambda _=None: self.on_face_temp(),
                  length=260).pack(side="left")
        self.face_lbl = tk.Label(row2, text="", bg=BG, fg=FG, width=34,
                                 anchor="w", font=("Menlo", 10))
        self.face_lbl.pack(side="left", padx=(8, 0))
        self.face_auto_btn = tk.Button(row2, text="auto", command=self.auto_face_temp,
                                       bg="#2a2a33", fg=FG, relief="flat",
                                       padx=10, pady=3, activebackground="#3a3a46",
                                       activeforeground=FG, highlightthickness=0,
                                       bd=0)
        self.face_auto_btn.pack(side="left", padx=(8, 0))

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=6)

        self.canvas = tk.Label(body, bg="#0e0e12", fg=MUTED,
                               text=("drop a .npy or .png here" if HAVE_DND
                                     else "click Browse to open a .npy or .png"),
                               font=("Helvetica", 15))
        self.canvas.pack(side="left", fill="both", expand=True)

        side = tk.Frame(body, bg=BG, width=300)
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)
        self.info = tk.Text(side, bg="#0e0e12", fg=FG, relief="flat", width=36,
                            font=("Menlo", 11), wrap="none", padx=10, pady=10,
                            insertbackground=FG)
        self.info.pack(fill="both", expand=True)

        self.frame_bar = tk.Frame(root, bg=BG)
        self.frame_var = tk.IntVar(value=0)
        self.frame_scale = ttk.Scale(self.frame_bar, from_=0, to=1,
                                     variable=self.frame_var,
                                     command=lambda _=None: self.on_frame())
        self.frame_scale.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.frame_lbl = tk.Label(self.frame_bar, text="", bg=BG, fg=MUTED, width=18)
        self.frame_lbl.pack(side="left")

        self.status = tk.Label(root, text=self._ready_text(), bg=BG, fg=MUTED,
                               anchor="w", font=("Menlo", 10))
        self.status.pack(fill="x", padx=12, pady=(4, 10))

        if HAVE_DND:
            for w in (root, self.canvas):
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<Drop>>", self.on_drop)

        root.bind("<Command-o>", lambda e: self.browse())
        root.bind("v", lambda e: self.cycle_view())
        root.bind("f", lambda e: self.toggle_faces())
        # Cmd-S keeps its old meaning (write the overlay); Cmd-E opens the
        # full export menu. Both no-op harmlessly before a frame is loaded.
        root.bind("<Command-s>", lambda e: self.frame and self.export("overlay"))
        root.bind("<Command-e>", lambda e: self.export_menu())

    # -------------------------------------------------------------- helpers

    def _ready_text(self):
        bits = [f"model {os.path.relpath(self.args.weights)}",
                f"imgsz {self.args.imgsz}",
                f"span {SPAN_C[0]:.0f}-{SPAN_C[1]:.0f} C"]
        if not HAVE_DND:
            bits.append("drag-and-drop off (pip install tkinterdnd2)")
        return "   |   ".join(bits)

    def set_status(self, msg):
        self.status.config(text=msg)
        self.root.update_idletasks()

    def on_conf(self):
        self.conf_lbl.config(text=f"{self.conf.get():.2f}")
        if self.frame is not None:
            self.infer()

    def cycle_classes(self):
        self.show = {"both": "person", "person": "omega", "omega": "both"}[self.show]
        self.cls_btn.config(text=f"classes: {self.show}")
        self.redraw()

    def cycle_view(self):
        self.view = (self.view + 1) % len(VIEWS)
        self.view_btn.config(text=f"view: {VIEWS[self.view][0].split()[0]}")
        self._reeye()
        self.redraw()

    def _reeye(self):
        """Re-render the EYE image at the current stretch. Model untouched."""
        f = self.frame
        if f is None:
            return
        if f.celsius is not None:
            f.eye_rgb = render_for_eyes(f.celsius, self.view)
        else:
            # Pre-rendered input: no degrees to work with, so stretch the grey
            # levels instead. Still eyes-only, still never near the model.
            g = cv2.cvtColor(f.cnn_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            f.eye_rgb = render_for_eyes(g, self.view)

    def toggle_faces(self):
        self.faces_on = not self.faces_on
        self.face_btn.config(text=f"faces: {'on' if self.faces_on else 'off'}")
        if self.faces_on and self.frame is not None \
                and self.frame.celsius is not None \
                and not self._face_temp_set:
            self.auto_face_temp()
            return
        self.recount_faces()

    def on_face_temp(self):
        self._face_temp_set = True
        self.recount_faces()

    def auto_face_temp(self):
        """
        Start the slider where the faces probably are: the 99th percentile.

        Not a physiological constant — a property of THIS frame. In a room
        where the hottest thing is a face, p99 lands on faces; where it is a
        laptop, it does not, and you will see that immediately in the count.
        """
        f = self.frame
        if f is None or f.celsius is None:
            return
        self.face_t.set(round(float(np.percentile(f.celsius, 99.0)) - 0.5, 1))
        self._face_temp_set = True
        self.recount_faces()

    def recount_faces(self):
        f = self.frame
        if f is None or not self.faces_on:
            self.face_blobs = []
            self.face_lbl.config(text="")
            self.redraw()
            return
        if f.celsius is None:
            self.face_blobs = []
            self.face_lbl.config(text="needs radiometric (.npy)")
            self.redraw()
            return
        t = float(self.face_t.get())
        self.face_blobs = face_blobs(f.celsius, t)
        self.face_lbl.config(
            text=f"{t:.1f}-{t + FACE_BAND_C:.1f} C   {len(self.face_blobs)} blobs")
        self.redraw()

    def cycle_interpret(self):
        self.interpret = {"auto": "pre-rendered",
                          "pre-rendered": "radiometric",
                          "radiometric": "auto"}[self.interpret]
        self.int_btn.config(text=f"input: {self.interpret}")
        if self.path:
            self.open(self.path, index=int(self.frame_var.get())
                      if self.frame is not None and self.frame.stack is not None else 0)

    def on_drop(self, event):
        raw = event.data.strip()
        # Tk quotes paths containing spaces, and hands several as one string.
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        path = raw.split("} {")[0].strip("{}").strip()
        self.open(path)

    def browse(self):
        p = filedialog.askopenfilename(
            title="Open a frame",
            filetypes=[("Frames", "*.npy *.png *.jpg *.jpeg *.tif *.tiff"),
                       ("NumPy", "*.npy"), ("Images", "*.png *.jpg *.jpeg *.tif *.tiff"),
                       ("All files", "*.*")])
        if p:
            self.open(p)

    def on_frame(self):
        if self.frame is None or self.frame.stack is None:
            return
        i = int(self.frame_var.get())
        if i == self.frame.index:
            return
        self.open(self.path, index=i)

    # -------------------------------------------------------------- pipeline

    def open(self, path, index=0):
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED:
            messagebox.showerror("Unsupported", f"{ext or 'that file'} is not one of "
                                 + " ".join(SUPPORTED))
            return
        try:
            self.set_status(f"loading {os.path.basename(path)} …")
            self.frame = load_path(path, index, self.interpret)
            self.path = path
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Could not load", str(e))
            self.set_status(self._ready_text())
            return

        if self.frame.stack is not None:
            n = len(self.frame.stack)
            self.frame_scale.config(to=n - 1)
            self.frame_var.set(self.frame.index)
            self.frame_lbl.config(text=f"frame {self.frame.index + 1} / {n}")
            self.frame_bar.pack(fill="x", padx=12, pady=(0, 4), before=self.status)
        else:
            self.frame_bar.pack_forget()

        # A new file arrives rendered at view 0; carry the operator's choice
        # over rather than silently resetting it.
        if self.view:
            self._reeye()
        # New scene, new temperatures. Re-derive unless the slider was moved
        # by hand, in which case it is a deliberate value and stays put.
        if self.faces_on and not self._face_temp_set:
            self.auto_face_temp()
        else:
            self.recount_faces()

        self.save_btn.config(state="normal")
        self.infer()

    def infer(self):
        self.set_status("running the model …")
        try:
            self.dets = self.pred.run(self.frame.cnn_rgb, self.conf.get(), self.args.imgsz)
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Inference failed", str(e))
            self.set_status(self._ready_text())
            return
        self.redraw()
        self.write_info()
        self.set_status(f"{os.path.basename(self.path)}   |   {self.frame.note}"
                        f"   |   {len(self.dets)} detections   |   {self._ready_text()}")

    def redraw(self):
        if self.frame is None:
            return
        h, w = self.frame.eye_rgb.shape[:2]
        avail_w = max(320, self.canvas.winfo_width() or 760)
        avail_h = max(240, self.canvas.winfo_height() or 560)
        scale = max(1, int(min(avail_w / w, avail_h / h)))
        big = draw(self.frame.eye_rgb, self.dets, self.show, scale)
        if self.faces_on and self.face_blobs:
            big = big.copy()
            for (cx, cy, r, peak) in self.face_blobs:
                # Deliberately a CIRCLE, not a box. A box here would read as a
                # detection, and these are not detections — they are pixels in
                # a temperature band, with no model involved and no claim that
                # they are people.
                cv2.circle(big, (int(cx * scale), int(cy * scale)),
                           max(4, int(r * scale * 1.6)), (90, 200, 255), 1,
                           cv2.LINE_AA)
                cv2.circle(big, (int(cx * scale), int(cy * scale)), 1,
                           (90, 200, 255), -1)
        self.photo = ImageTk.PhotoImage(Image.fromarray(big))
        self.canvas.config(image=self.photo, text="")

    def write_info(self):
        t = self.info
        t.config(state="normal")
        t.delete("1.0", "end")
        f = self.frame
        t.insert("end", f"{os.path.basename(self.path)}\n")
        t.insert("end", f"{f.note}\n")
        t.insert("end", f"source: {f.kind}\n")
        if f.kind == "pre-rendered":
            t.insert("end", "  (8-bit: fixed span already\n   applied, not re-applied)\n")
        if f.celsius is not None:
            a = f.celsius
            t.insert("end", f"scene  {a.min():.1f} - {a.max():.1f} C\n")
            t.insert("end", f"       median {np.median(a):.1f} C, "
                            f"p99 {np.percentile(a, 99):.1f} C\n")
        if self.faces_on:
            t.insert("end", "\n" + "-" * 34 + "\n")
            if f.celsius is None:
                t.insert("end", "\nface overlay needs a .npy\n"
                                "(a .png has no degrees in it)\n")
            else:
                tt = float(self.face_t.get())
                t.insert("end", f"\nface band {tt:.1f}-{tt + FACE_BAND_C:.1f} C\n")
                t.insert("end", f"blobs >= {FACE_MIN_PX}px: "
                                f"{len(self.face_blobs)}\n")
                if self.face_blobs:
                    pk = [b[3] for b in self.face_blobs]
                    t.insert("end", f"peak {min(pk):.1f}-{max(pk):.1f} C\n")
                t.insert("end", "\nnot detections - just pixels\nin a band. "
                                "sweep the slider\nand watch what survives.\n")

        t.insert("end", "\n" + "-" * 34 + "\n")
        if not self.dets:
            t.insert("end", f"\nno detections at conf >= {self.conf.get():.2f}\n")
        for i, d in enumerate(self.dets, 1):
            t.insert("end", f"\n{i}. {d['cls']}   conf {d['conf']:.3f}\n")
            t.insert("end", f"   box   {d['w']:6.2f} x {d['h']:6.2f} px\n")
            if d["cls"] == "omega":
                c = width_correction(d["w"])
                rng = implied_range(d["w"])
                t.insert("end", f"   corr  {c:+.2f} px -> {d['w'] + c:6.2f} px\n")
                if rng:
                    t.insert("end", f"   range ~{rng:5.2f} m  (slant)\n")
        if any(d["cls"] == "omega" for d in self.dets):
            t.insert("end", "\n" + "-" * 34 + "\n")
            t.insert("end", "\nrange is provisional: from a\n5-point calibration, W=0.489 m,\n"
                            "+/-23%. Facing direction is\n72% of that variance.\n")
        t.config(state="disabled")

    # -------------------------------------------------------------- export
    #
    # Six things worth getting out of here, and they are genuinely different
    # files. "Raw" is ambiguous -- the pretty picture, the bytes the CNN saw,
    # and the temperatures are three separate artefacts and a figure caption
    # that mixes them up is a figure caption that is wrong. Each option below
    # says exactly which one it writes.

    EXPORT_SCALE = 6

    def export_menu(self):
        if self.frame is None:
            return
        m = tk.Menu(self.root, tearoff=0, bg="#22222a", fg=FG,
                    activebackground="#3a3a46", activeforeground=FG, bd=0)
        m.add_command(label="Overlay  — boxes on the colour render",
                      command=lambda: self.export("overlay"))
        m.add_separator()
        m.add_command(label="Clean render  — no boxes, upscaled",
                      command=lambda: self.export("clean_big"))
        m.add_command(label="Clean render  — no boxes, native size",
                      command=lambda: self.export("clean_native"))
        m.add_separator()
        m.add_command(label="Model input  — exact bytes the CNN saw",
                      command=lambda: self.export("cnn"))
        m.add_command(label="Radiometric .npy  — degrees Celsius",
                      command=lambda: self.export("npy"),
                      state="normal" if self.frame.celsius is not None else "disabled")
        m.add_separator()
        m.add_command(label="Detections .txt  — YOLO label format",
                      command=lambda: self.export("labels"),
                      state="normal" if self.dets else "disabled")
        try:
            m.tk_popup(self.save_btn.winfo_rootx(),
                       self.save_btn.winfo_rooty() + self.save_btn.winfo_height())
        finally:
            m.grab_release()

    def export(self, kind):
        f = self.frame
        base = os.path.splitext(os.path.basename(self.path))[0]

        if kind == "overlay":
            arr = draw(f.eye_rgb, self.dets, self.show, self.EXPORT_SCALE)
            suffix, ext, what = "_overlay", ".png", "overlay"
        elif kind == "clean_big":
            arr = cv2.resize(f.eye_rgb,
                             (f.eye_rgb.shape[1] * self.EXPORT_SCALE,
                              f.eye_rgb.shape[0] * self.EXPORT_SCALE),
                             interpolation=cv2.INTER_NEAREST)
            suffix, ext, what = "_render", ".png", "clean colour render, upscaled"
        elif kind == "clean_native":
            arr = f.eye_rgb
            suffix, ext, what = "_render_native", ".png", "clean colour render, native"
        elif kind == "cnn":
            # Greyscale, native size, fixed span. Byte-for-byte what predict()
            # was handed -- so a reviewer can rerun the detection from it.
            arr = f.cnn_rgb[..., 0]
            suffix, ext, what = "_cnn_input", ".png", "model input (grey, fixed span)"
        elif kind == "npy":
            p = filedialog.asksaveasfilename(defaultextension=".npy",
                                             initialfile=f"{base}_celsius.npy",
                                             filetypes=[("NumPy", "*.npy")])
            if not p:
                return
            np.save(p, f.celsius.astype(np.float32))
            self.set_status(f"saved radiometric Celsius → {p}")
            return
        elif kind == "labels":
            p = filedialog.asksaveasfilename(defaultextension=".txt",
                                             initialfile=f"{base}_pred.txt",
                                             filetypes=[("YOLO labels", "*.txt")])
            if not p:
                return
            h, w = f.cnn_rgb.shape[:2]
            lines = []
            for d in self.dets:
                ci = NAMES.index(d["cls"]) if d["cls"] in NAMES else 0
                x1, y1, x2, y2 = d["xyxy"]
                lines.append(f"{ci} {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            with open(p, "w") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            self.set_status(f"saved {len(lines)} detections → {p}")
            return
        else:
            return

        p = filedialog.asksaveasfilename(defaultextension=ext,
                                         initialfile=f"{base}{suffix}{ext}",
                                         filetypes=[("PNG", "*.png")])
        if not p:
            return
        out = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(p, out)
        self.set_status(f"saved {what} → {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None,
                    help="path to best.pt. Default: newest models/vN")
    ap.add_argument("--conf", type=float, default=0.374,
                    help="starting confidence threshold (the slider moves it)")
    ap.add_argument("--face-temp", type=float, default=30.0,
                    help="starting face-temperature threshold, C. There is no "
                         "right value: it moves with ambient, clothing, "
                         "airflow and range. The 'auto' button sets it from "
                         "the frame's own p99.")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="inference size. Training ran at 640 on 160x120 frames, "
                         "so leave this alone unless you retrained.")
    ap.add_argument("open", nargs="?", help="a file to open on launch")
    args = ap.parse_args()

    try:
        from model_registry import resolve_weights
        args.weights = resolve_weights(args.weights)
    except Exception:
        if not args.weights:
            print("no model_registry and no --weights: cannot pick a model", file=sys.stderr)
            return 2

    root = TkinterDnD.Tk() if HAVE_DND else tk.Tk()
    app = App(root, Predictor(args.weights), args)
    if args.open:
        root.after(120, lambda: app.open(os.path.abspath(args.open)))
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
