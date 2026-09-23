#!/usr/bin/env python3
"""
Drop a Sony frame and a Lepton .npy on the window, see them overlaid.

    ./run.sh align_gui.py
    ./run.sh align_gui.py shot.jpg cap_000123.npy

The geometry lives in align_visible.py and is imported, not copied — this file
is only the window. See that module's docstring for why a scale-and-shift can
never align these two cameras: the Lepton is EQUIDISTANT (r = f*theta) and the
ZV-1 is RECTILINEAR (r = f*tan theta), so the mismatch is nonlinear in radius
and reaches 32% at the thermal frame's edge.

WHAT IT DOES. Every Lepton pixel is turned into a look direction, that
direction is projected into the Sony's rectilinear image, and the photo is
sampled there. The result is in LEPTON pixel space, so the thermal frame is
never resampled — it is the reference, and the photograph is the thing that
gets bent. That is the right way round: the thermal data is what you are
studying, and interpolating 160x120 to match a 4000 px photo would invent
detail the sensor never had.

DROP EITHER FILE IN EITHER ORDER. .npy is unambiguously thermal; an image file
is treated as the visible frame unless nothing thermal is loaded yet and it is
small enough to be a thermal render.

WHAT WILL STILL BE WRONG, AND IS NOT A BUG.

  * Only about HALF the thermal frame has any photo in it. The Lepton sees
    95 deg horizontally; a ZV-1 at 4:3 sees roughly 67. The rest is black.
  * Parallax. The two lenses are not in the same place, so near objects need a
    different shift from far ones. Line the BACKGROUND up and stop — chasing a
    foreground subject just breaks the background again. The status bar keeps
    reminding you.
"""

import argparse
import json
import os
import sys
import traceback

import cv2
import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import align_visible as AV
import thermal_detect as TD

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAVE_DND = True
except Exception:
    HAVE_DND = False

BG = "#16161a"
FG = "#e8e8ec"
MUTED = "#8b8b96"

# ---------------------------------------------------------------------------
# The calibrated rig, 2026-09-23
# ---------------------------------------------------------------------------
# Solved by Auto from 7 thermal + 15 visible detections -> 5 pairs, then
# hand-checked. 1.12 thermal px RMS on a 160x120 sensor where a head is 4-15 px
# across, with the Sony at 4864x3648.
#
# THESE FOUR ARE PROPERTIES OF THE RIG, not of the scene, so they are reusable
# on every future pair of frames — until something physically moves. Re-run
# Auto and re-pin if you remount either camera, change the ZV-1's zoom, or
# swap the lens.
#
# hfov landing at 69.4 rather than the 67.4 predicted for a 4:3 crop is worth
# noting: 2 degrees wider than the sensor-geometry estimate. Either the zoom
# was not quite at the wide stop, or Sony's stated 35mm-equivalent is
# approximate. The measured number wins.
#
# WHAT IS NOT REUSABLE: the point pairs, and therefore the deformation itself.
# Parallax depends on where people are standing, so the spline has to be
# re-derived per scene — press Auto. morph strength and smoothing ARE saved,
# because they say how much to trust that spline, which is a preference.
DEFAULTS = {
    "hfov": 69.4,
    "yaw": 0.37,
    "pitch": 2.59,
    "roll": 1.69,
    "morph": 0.56,
    "smooth": 5.4,          # lam = 10^5.4, heavy: a gentle nudge, not a snap
    "alpha": 0.50,
}
DEFAULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "align_defaults.json")


def load_defaults():
    """Built-ins, overridden by align_defaults.json if you have re-pinned."""
    d = dict(DEFAULTS)
    try:
        if os.path.exists(DEFAULTS_PATH):
            d.update({k: float(v) for k, v in
                      json.load(open(DEFAULTS_PATH)).items() if k in DEFAULTS})
    except Exception:
        pass
    return d


THERMAL_EXT = (".npy",)
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


class App:
    SCALE = 6                      # output is 160*6 x 120*6

    def __init__(self, root, args):
        self.root = root
        self.vis = None            # the Sony frame, BGR
        self.vis_path = None
        self.arr = None            # the Lepton frame, degrees C
        self.arr_path = None
        self.mode = "blend"

        # Hand-clicked correspondences. Stored as
        #   ((tx, ty) in THERMAL OUTPUT-GRID px, (vx, vy) in ORIGINAL Sony px)
        # so they stay valid when SCALE changes and when the sliders move.
        self.pairs = []
        self.pending_t = None
        self.pending_v = None
        self.picking = False
        self.coco_weights = args.coco

        root.title("Thermal / visible alignment")
        root.configure(bg=BG)
        root.geometry("1180x820")

        top = tk.Frame(root, bg=BG)
        top.pack(fill="x", padx=12, pady=(12, 6))

        def btn(parent, text, cmd, **kw):
            return tk.Button(parent, text=text, command=cmd, bg="#2a2a33",
                             fg=FG, relief="flat", padx=12, pady=6,
                             activebackground="#3a3a46", activeforeground=FG,
                             highlightthickness=0, bd=0, **kw)

        btn(top, "Visible…", lambda: self.browse("vis")).pack(side="left")
        btn(top, "Thermal…", lambda: self.browse("arr")).pack(side="left",
                                                              padx=(8, 0))
        self.mode_btn = btn(top, "view: blend", self.cycle_mode)
        self.mode_btn.pack(side="left", padx=(16, 0))
        self.pick_btn = btn(top, "points: off", self.toggle_pick)
        self.pick_btn.pack(side="left", padx=(16, 0))
        self.auto_btn = btn(top, "Auto", self.do_auto, state="disabled")
        self.auto_btn.pack(side="left", padx=(8, 0))
        self.fit_btn = btn(top, "Fit", self.do_fit, state="disabled")
        self.fit_btn.pack(side="left", padx=(8, 0))

        self.save_btn = btn(top, "Export ▾", self.export_menu, state="disabled")
        self.save_btn.pack(side="left", padx=(8, 0))
        btn(top, "Reset", self.reset_params).pack(side="left", padx=(8, 0))

        # --- sliders ------------------------------------------------------
        grid = tk.Frame(root, bg=BG)
        grid.pack(fill="x", padx=12, pady=(0, 4))

        self.var = {}
        # hfov FIRST and widest: it is the one you cannot read off the camera,
        # because the ZV-1 reports 35mm-equivalent focal length for a 3:2 frame
        # while you are shooting 4:3, which crops the sensor width.
        for i, (key, label, lo, hi, init, fmt) in enumerate((
                ("hfov", "visible HFOV", 20.0, 120.0, args.hfov, "{:.1f}°"),
                ("alpha", "thermal opacity", 0.0, 1.0, args.alpha, "{:.2f}"),
                ("yaw", "yaw", -20.0, 20.0, args.yaw, "{:+.2f}°"),
                ("pitch", "pitch", -20.0, 20.0, args.pitch, "{:+.2f}°"),
                ("roll", "roll", -20.0, 20.0, args.roll, "{:+.2f}°"),
                ("morph", "morph strength", 0.0, 1.0,
                 load_defaults()["morph"], "{:.2f}"),
                ("smooth", "morph smoothing", 0.0, 6.0,
                 load_defaults()["smooth"], "1e{:.1f}"))):
            r, c = divmod(i, 2)
            cell = tk.Frame(grid, bg=BG)
            cell.grid(row=r, column=c, sticky="we", padx=(0, 18), pady=2)
            grid.columnconfigure(c, weight=1)
            tk.Label(cell, text=label, bg=BG, fg=MUTED, width=14,
                     anchor="w").pack(side="left")
            v = tk.DoubleVar(value=init)
            self.var[key] = v
            ttk.Scale(cell, from_=lo, to=hi, variable=v, length=240,
                      command=lambda _=None: self.redraw()).pack(side="left")
            lbl = tk.Label(cell, text=fmt.format(init), bg=BG, fg=FG, width=8,
                           anchor="w", font=("Menlo", 10))
            lbl.pack(side="left", padx=(6, 0))
            v.trace_add("write",
                        lambda *_, k=key, l=lbl, f=fmt:
                        l.config(text=f.format(self.var[k].get())))

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=6)

        self.canvas = tk.Label(
            body, bg="#0e0e12", fg=MUTED,
            text=("drop a Sony frame and a Lepton .npy here"
                  if HAVE_DND else "use the two buttons above"),
            font=("Helvetica", 15))
        self.canvas.pack(side="left", fill="both", expand=True)

        # Picking needs both images in their OWN geometry, side by side: the
        # thermal in its output grid, and the Sony UNWARPED. A correspondence
        # clicked on the warped photo would be a correspondence to whatever the
        # current (wrong) parameters happen to say, which is circular.
        self.pick_frame = tk.Frame(body, bg=BG)
        self.t_pane = tk.Label(self.pick_frame, bg="#0e0e12", fg=MUTED,
                               text="thermal")
        self.t_pane.pack(side="left", fill="both", expand=True)
        self.v_pane = tk.Label(self.pick_frame, bg="#0e0e12", fg=MUTED,
                               text="visible (unwarped)")
        self.v_pane.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.t_pane.bind("<Button-1>", self.click_thermal)
        self.v_pane.bind("<Button-1>", self.click_visible)

        side = tk.Frame(body, bg=BG, width=300)
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)
        self.info = tk.Text(side, bg="#0e0e12", fg=FG, relief="flat", width=36,
                            font=("Menlo", 11), wrap="none", padx=10, pady=10)
        self.info.pack(fill="both", expand=True)

        self.status = tk.Label(root, text=self._ready(), bg=BG, fg=MUTED,
                               anchor="w", font=("Menlo", 10))
        self.status.pack(fill="x", padx=12, pady=(4, 10))

        if HAVE_DND:
            for w in (root, self.canvas):
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<Drop>>", self.on_drop)

        root.bind("<Command-o>", lambda e: self.browse("vis"))
        root.bind("m", lambda e: self.cycle_mode())
        root.bind("p", lambda e: self.toggle_pick())
        root.bind("f", lambda e: self.do_fit())
        root.bind("a", lambda e: self.do_auto())
        root.bind("<BackSpace>", self.undo_point)
        root.bind("<Escape>", self.clear_points)

    # ---------------------------------------------------------------- state
    def _ready(self):
        bits = [f"lepton {AV.LEPTON_W}x{AV.LEPTON_H} @ 95° f-theta",
                f"out {AV.LEPTON_W * self.SCALE}x{AV.LEPTON_H * self.SCALE}"]
        if not HAVE_DND:
            bits.append("drag-and-drop off (pip install tkinterdnd2)")
        return "   |   ".join(bits)

    def set_status(self, msg):
        self.status.config(text=msg)
        self.root.update_idletasks()

    def reset_params(self):
        # Back to the PINNED rig, not to zeros. Zeros are not a neutral
        # starting point here, they are a wrong one.
        for k, v in load_defaults().items():
            if k in self.var:
                self.var[k].set(v)
        self.redraw()

    def cycle_mode(self):
        self.mode = {"blend": "thermal", "thermal": "visible",
                     "visible": "blend"}[self.mode]
        self.mode_btn.config(text=f"view: {self.mode}")
        self.redraw()

    # -------------------------------------------------------------- points
    PANE_W = 520

    def toggle_pick(self):
        self.picking = not self.picking
        self.pick_btn.config(text=f"points: {'on' if self.picking else 'off'}")
        if self.picking:
            self.canvas.pack_forget()
            self.pick_frame.pack(side="left", fill="both", expand=True)
        else:
            self.pick_frame.pack_forget()
            self.canvas.pack(side="left", fill="both", expand=True)
        self.redraw()

    def click_thermal(self, ev):
        if self.arr is None:
            return
        self.pending_t = (ev.x / self._t_disp, ev.y / self._t_disp)
        self._maybe_pair()

    def click_visible(self, ev):
        if self.vis is None:
            return
        self.pending_v = (ev.x / self._v_disp, ev.y / self._v_disp)
        self._maybe_pair()

    def _maybe_pair(self):
        """A pair completes when BOTH sides have a pending click."""
        if self.pending_t is not None and self.pending_v is not None:
            self.pairs.append((self.pending_t, self.pending_v))
            self.pending_t = self.pending_v = None
            self.set_status(f"{len(self.pairs)} pairs — click the same head "
                            f"on both sides")
        self.fit_btn.config(state="normal" if len(self.pairs) >= 2
                            else "disabled")
        self.redraw()

    def undo_point(self, _=None):
        if self.pending_t or self.pending_v:
            self.pending_t = self.pending_v = None
        elif self.pairs:
            self.pairs.pop()
        self._maybe_pair()

    def clear_points(self, _=None):
        self.pairs.clear()
        self.pending_t = self.pending_v = None
        self._maybe_pair()

    def live_residuals(self, ow, oh):
        """
        Error at each control point AS CURRENTLY DISPLAYED — model plus morph.

        Reporting the model-only residual while the screen shows a morphed
        image would be reporting a different picture from the one you are
        looking at.
        """
        res = AV.residuals(self.pairs, ow, oh, self.vis.shape, self.params())
        st = self.var["morph"].get()
        if st <= 0:
            return res
        lam = 0.0 if self.var["smooth"].get() <= 0 \
            else 10.0 ** self.var["smooth"].get()
        f = AV.residual_field(self.pairs, ow, oh, self.vis.shape,
                              self.params(), lam=lam)
        if f is None:
            return res
        out = []
        for i, ((tx, ty), (vx, vy)) in enumerate(self.pairs):
            d = AV.therm_px_to_dir(tx, ty, ow, oh)
            u, v = AV.dir_to_vis_px(d, self.vis.shape, **self.params())
            xi, yi = int(round(tx)), int(round(ty))
            if 0 <= yi < oh and 0 <= xi < ow:
                u += st * f[0][yi, xi]
                v += st * f[1][yi, xi]
            out.append(np.hypot(u - vx, v - vy))
        return np.array(out)

    # Where the HEAD sits inside each detector's box, as a fraction of box
    # height from the top. The two detectors do not frame the same thing:
    #
    #   omega  = head AND shoulders, so the head centre is about a third down
    #   person = the whole body, so the head is in the top tenth
    #
    # Getting these wrong does not break the fit — a constant offset is
    # absorbed as pitch — but it does put the control points somewhere other
    # than the heads, and the spline then deforms around the wrong anchors.
    HEAD_FRAC_OMEGA = 0.35
    HEAD_FRAC_PERSON = 0.10

    def detect_thermal(self):
        """Head points in THERMAL OUTPUT-GRID coordinates, via your own model."""
        from ultralytics import YOLO
        from model_registry import resolve_weights
        import dataset_pipeline as DP
        w = resolve_weights(None, quiet=True)
        img = AV.TD.render_for_cnn(self.arr) if hasattr(AV.TD, "render_for_cnn") \
            else cv2.merge([np.clip(
                (self.arr - DP.SPAN_C[0]) /
                (DP.SPAN_C[1] - DP.SPAN_C[0]) * 255, 0, 255
            ).astype(np.uint8)] * 3)
        res = YOLO(w).predict(img, verbose=False, conf=0.25, imgsz=640)[0]
        pts = []
        for (x0, y0, x1, y1), c in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int)):
            if c != DP.OMEGA_CLASS:
                continue
            # boxes come back in the SOURCE image's pixels (160x120), so scale
            # to the output grid the pairs are stored in
            cx = (x0 + x1) / 2.0 * self.SCALE
            cy = (y0 + (y1 - y0) * self.HEAD_FRAC_OMEGA) * self.SCALE
            pts.append((cx, cy))
        return pts

    def detect_visible(self, weights="yolo11n.pt"):
        """Head points in ORIGINAL Sony pixels, via a COCO person detector."""
        from ultralytics import YOLO
        res = YOLO(weights).predict(self.vis, verbose=False, conf=0.35,
                                    imgsz=1280)[0]
        names = res.names
        pts = []
        for (x0, y0, x1, y1), c in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int)):
            if names.get(c, "") != "person":
                continue
            cx = (x0 + x1) / 2.0
            cy = y0 + (y1 - y0) * self.HEAD_FRAC_PERSON
            pts.append((float(cx), float(cy)))
        return pts

    def do_auto(self):
        """Detect in both, match, fit, and leave the pairs for you to edit."""
        if self.vis is None or self.arr is None:
            return
        self.set_status("detecting …")
        try:
            tp = self.detect_thermal()
            vp = self.detect_visible(self.coco_weights)
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror(
                "Auto failed",
                f"{e}\n\nNeeds ultralytics, a model in models/vN/best.pt for "
                f"the thermal side, and a COCO model for the visible side "
                f"(downloads on first use).")
            self.set_status(self._ready())
            return

        if len(tp) < 2 or len(vp) < 2:
            self.set_status(f"only {len(tp)} thermal / {len(vp)} visible "
                            f"detections — need 2 of each. Click them by hand.")
            return

        ow, oh = AV.LEPTON_W * self.SCALE, AV.LEPTON_H * self.SCALE
        pairs, got, hist = AV.auto_align(tp, vp, ow, oh, self.vis.shape)
        if len(pairs) < 2:
            self.set_status(f"{len(tp)} thermal, {len(vp)} visible, but "
                            f"nothing matched — check they are the same scene")
            return

        # REPLACE rather than extend: mixing an automatic pass into points you
        # placed by hand leaves you unable to tell which is which when one is
        # wrong. Undo restores nothing here, so the old set is worth keeping.
        self.pairs = list(pairs)
        for k, v in (("hfov", got["hfov_vis"]), ("yaw", got["yaw"]),
                     ("pitch", got["pitch"]), ("roll", got["roll"])):
            self.var[k].set(v)
        self._maybe_pair()

        after = self.live_residuals(ow, oh)
        self.set_status(
            f"auto: {len(tp)} thermal + {len(vp)} visible -> "
            f"{len(pairs)} pairs, hfov {got['hfov_vis']:.1f}°, "
            f"{AV.rms_in_thermal_px(after, self.vis.shape[1], got['hfov_vis']):.2f}"
            f" thermal px. Turn points on to check and fix them.")

    def do_fit(self):
        if len(self.pairs) < 2:
            return
        ow, oh = AV.LEPTON_W * self.SCALE, AV.LEPTON_H * self.SCALE
        try:
            got, res = AV.fit_params(
                self.pairs, ow, oh, self.vis.shape,
                init={"hfov_vis": self.var["hfov"].get(),
                      "yaw": self.var["yaw"].get(),
                      "pitch": self.var["pitch"].get(),
                      "roll": self.var["roll"].get()})
        except Exception as e:
            messagebox.showerror("Fit failed", str(e))
            return
        self.var["hfov"].set(got["hfov_vis"])
        self.var["yaw"].set(got["yaw"])
        self.var["pitch"].set(got["pitch"])
        self.var["roll"].set(got["roll"])
        ow, oh = AV.LEPTON_W * self.SCALE, AV.LEPTON_H * self.SCALE
        after = self.live_residuals(ow, oh)
        t_before = AV.rms_in_thermal_px(res, self.vis.shape[1],
                                        got["hfov_vis"])
        t_after = AV.rms_in_thermal_px(after, self.vis.shape[1],
                                       got["hfov_vis"])
        self.set_status(
            f"{len(self.pairs)} pairs — 4-param fit {t_before:.2f} "
            f"thermal px, + morph {t_after:.2f}  "
            f"(a head is 4-15 thermal px wide)")
        self.redraw()

    def save_points(self):
        p = filedialog.asksaveasfilename(defaultextension=".json",
                                         initialfile="align_points.json")
        if p:
            json.dump({"scale": self.SCALE, "pairs": self.pairs},
                      open(p, "w"), indent=1)
            self.set_status(f"wrote {os.path.basename(p)}")

    def load_points(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not p:
            return
        d = json.load(open(p))
        k = self.SCALE / float(d.get("scale", self.SCALE))
        self.pairs = [((t[0] * k, t[1] * k), (v[0], v[1]))
                      for t, v in d["pairs"]]
        self._maybe_pair()

    # ---------------------------------------------------------------- load
    def on_drop(self, event):
        raw = event.data.strip()
        # Tk hands several paths as one brace-quoted string.
        parts = [p.strip("{}").strip() for p in raw.split("} {")] \
            if "} {" in raw else [raw.strip("{}").strip()]
        for p in parts:
            self.open(p, redraw=False)
        self.redraw()

    def browse(self, which):
        if which == "arr":
            p = filedialog.askopenfilename(
                title="Lepton frame",
                filetypes=[("Lepton", "*.npy"), ("All", "*.*")])
        else:
            p = filedialog.askopenfilename(
                title="Sony frame",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff"),
                           ("All", "*.*")])
        if p:
            self.open(p)

    def open(self, path, redraw=True):
        """
        Work out which camera a dropped file came from and load it.

        .npy is unambiguous. An ordinary image is the VISIBLE frame — unless
        nothing thermal is loaded yet and the image is small enough to be a
        thermal render, in which case treating it as thermal is the only
        interpretation that produces anything.
        """
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in THERMAL_EXT:
                self.arr = np.load(path, allow_pickle=False)
                if self.arr.ndim == 3:
                    self.arr = self.arr[..., 0] if self.arr.shape[-1] <= 4 \
                        else self.arr[0]
                self.arr_path = path
                self.set_status(f"thermal {os.path.basename(path)} "
                                f"{self.arr.shape[1]}x{self.arr.shape[0]}")
            elif ext in IMAGE_EXT:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("could not read that image")
                small = max(img.shape[:2]) <= 512
                if self.arr is None and small:
                    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    self.arr = g.astype(np.float32)
                    self.arr_path = path
                    self.set_status(f"thermal (8-bit render) "
                                    f"{os.path.basename(path)}")
                else:
                    self.vis, self.vis_path = img, path
                    self.set_status(f"visible {os.path.basename(path)} "
                                    f"{img.shape[1]}x{img.shape[0]}")
            else:
                messagebox.showerror("Unsupported", f"{ext or 'that file'}")
                return
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Could not load", str(e))
            return

        if self.vis is not None and self.arr is not None:
            self.save_btn.config(state="normal")
            self.auto_btn.config(state="normal")
        if redraw:
            self.redraw()

    # ---------------------------------------------------------------- draw
    def params(self):
        return dict(hfov_vis=self.var["hfov"].get(),
                    yaw=self.var["yaw"].get(),
                    pitch=self.var["pitch"].get(),
                    roll=self.var["roll"].get())

    def compose(self, scale=None):
        """(blend, warped_visible, thermal_rgb, coverage) at the given scale."""
        s = scale or self.SCALE
        ow, oh = AV.LEPTON_W * s, AV.LEPTON_H * s
        therm = cv2.resize(TD.colorize(self.arr), (ow, oh),
                           interpolation=cv2.INTER_NEAREST)
        # lam is on a LOG slider: the useful range spans six orders of
        # magnitude, and a linear slider would be unusable at the low end
        # where the interesting behaviour is.
        lam = 0.0 if self.var["smooth"].get() <= 0 \
            else 10.0 ** self.var["smooth"].get()
        mx, my = AV.build_maps_morph(self.vis.shape, ow, oh, self.params(),
                                     pairs=self.pairs, lam=lam,
                                     strength=self.var["morph"].get())
        w = cv2.remap(self.vis, mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        cover = float(((mx >= 0) & (mx < self.vis.shape[1]) &
                       (my >= 0) & (my < self.vis.shape[0])).mean())
        a = self.var["alpha"].get()
        blend = cv2.addWeighted(w, 1.0 - a, therm, a, 0.0)
        return blend, w, therm, cover

    def _mark(self, img, x, y, n, col, pending=False):
        """A numbered crosshair. Hollow, so it never hides the thing you aimed at."""
        x, y = int(round(x)), int(round(y))
        cv2.circle(img, (x, y), 9, col, 1, cv2.LINE_AA)
        cv2.line(img, (x - 14, y), (x - 4, y), col, 1, cv2.LINE_AA)
        cv2.line(img, (x + 4, y), (x + 14, y), col, 1, cv2.LINE_AA)
        cv2.line(img, (x, y - 14), (x, y - 4), col, 1, cv2.LINE_AA)
        cv2.line(img, (x, y + 4), (x, y + 14), col, 1, cv2.LINE_AA)
        cv2.putText(img, "?" if pending else str(n), (x + 11, y - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    def _show(self, widget, img, attr):
        """Fit an image into a pane, remembering the scale for click mapping."""
        k = self.PANE_W / img.shape[1]
        disp = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)),
                          interpolation=cv2.INTER_AREA)
        setattr(self, attr, k)
        photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        widget.config(image=photo, text="")
        widget.image = photo            # keep a reference or Tk frees it

    def redraw(self):
        if self.vis is None or self.arr is None:
            return
        blend, w, therm, cover = self.compose()

        if self.picking:
            t = therm.copy()
            for i, ((tx, ty), _) in enumerate(self.pairs, 1):
                self._mark(t, tx, ty, i, (90, 230, 255))
            if self.pending_t:
                self._mark(t, *self.pending_t, 0, (255, 255, 255), True)
            self._show(self.t_pane, t, "_t_disp")

            v = self.vis.copy()
            for i, (_, (vx, vy)) in enumerate(self.pairs, 1):
                self._mark(v, vx, vy, i, (90, 230, 255))
            if self.pending_v:
                self._mark(v, *self.pending_v, 0, (255, 255, 255), True)
            self._show(self.v_pane, v, "_v_disp")
        else:
            img = {"blend": blend, "thermal": therm, "visible": w}[self.mode]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.config(image=self.photo, text="")
        self.write_info(cover)

    def write_info(self, cover):
        t = self.info
        t.config(state="normal")
        t.delete("1.0", "end")
        t.insert("end", f"{os.path.basename(self.vis_path or '')}\n")
        t.insert("end", f"  {self.vis.shape[1]}x{self.vis.shape[0]} rectilinear\n")
        t.insert("end", f"{os.path.basename(self.arr_path or '')}\n")
        t.insert("end", f"  {self.arr.shape[1]}x{self.arr.shape[0]} f-theta\n")
        t.insert("end", "\n" + "-" * 34 + "\n\n")

        hf = self.var["hfov"].get()
        f_v = (self.vis.shape[1] / 2.0) / np.tan(np.radians(hf) / 2.0)
        t.insert("end", f"visible HFOV  {hf:6.1f}°\n")
        t.insert("end", f"  f          {f_v:8.1f} px\n")
        t.insert("end", f"lepton HFOV    95.0°  ({AV.DEG_PER_PX:.5f}°/px)\n")
        t.insert("end", f"\noverlap    {100 * cover:5.1f}% of the\n"
                        f"           thermal frame by area\n")

        if self.arr.dtype.kind == "f" and self.arr.max() < 200:
            t.insert("end", f"\nscene {self.arr.min():.1f} – "
                            f"{self.arr.max():.1f} °C\n")

        if self.pairs:
            ow, oh = AV.LEPTON_W * self.SCALE, AV.LEPTON_H * self.SCALE
            res = self.live_residuals(ow, oh)
            tpx = AV.rms_in_thermal_px(res, self.vis.shape[1],
                                       self.var["hfov"].get())
            t.insert("end", "\n" + "-" * 34 + "\n")
            t.insert("end", f"\n{len(self.pairs)} point pairs\n")
            t.insert("end", f"RMS {np.sqrt(np.mean(res**2)):7.1f} sony px\n")
            t.insert("end", f"    {tpx:7.2f} thermal px\n\n")
            # Per-point, so an outlier is visible. A pair that is far worse
            # than the rest is either a misclick or a head at a very different
            # depth from the others — both worth knowing before you trust the
            # fit.
            for i, r in enumerate(res, 1):
                bar = "#" * min(20, int(r / 10))
                t.insert("end", f"  {i:2d} {r:7.1f} px {bar}\n")

        t.insert("end", "\n" + "-" * 34 + "\n")
        t.insert("end", "\nthe residual you CANNOT\nremove here is parallax:\n"
                        "the lenses are not in the\nsame place, so near and\n"
                        "far need different shifts.\n\n"
                        "align the BACKGROUND and\nstop there.\n")
        t.config(state="disabled")

    # -------------------------------------------------------------- export
    EXPORT_SCALE = 10

    def export_menu(self):
        if self.vis is None or self.arr is None:
            return
        m = tk.Menu(self.root, tearoff=0, bg="#22222a", fg=FG,
                    activebackground="#3a3a46", activeforeground=FG, bd=0)
        m.add_command(label="Blend  — the overlay as shown",
                      command=lambda: self.export("blend"))
        m.add_command(label="Warped visible  — photo alone, thermal geometry",
                      command=lambda: self.export("warped"))
        m.add_command(label="Thermal  — colour render alone",
                      command=lambda: self.export("thermal"))
        m.add_separator()
        m.add_command(label="Copy the flags for align_visible.py",
                      command=self.copy_flags)
        m.add_separator()
        m.add_command(label="Save point pairs…", command=self.save_points,
                      state="normal" if self.pairs else "disabled")
        m.add_command(label="Load point pairs…", command=self.load_points)
        m.add_separator()
        m.add_command(label="Pin these sliders as the default",
                      command=self.pin_defaults)
        try:
            m.tk_popup(self.save_btn.winfo_rootx(),
                       self.save_btn.winfo_rooty()
                       + self.save_btn.winfo_height())
        finally:
            m.grab_release()

    def pin_defaults(self):
        """
        Write the current sliders to align_defaults.json.

        Only the rig parameters and the morph preferences — never the point
        pairs. Pairs are scene-specific; saving them here would silently apply
        one room's parallax to every other room.
        """
        d = {k: round(float(self.var[k].get()), 3)
             for k in ("hfov", "yaw", "pitch", "roll", "morph", "smooth",
                       "alpha") if k in self.var}
        try:
            json.dump(d, open(DEFAULTS_PATH, "w"), indent=1)
        except OSError as e:
            messagebox.showerror("Could not write defaults", str(e))
            return
        self.set_status(f"pinned to {os.path.basename(DEFAULTS_PATH)}: "
                        + "  ".join(f"{k} {v}" for k, v in d.items()))

    def copy_flags(self):
        p = self.params()
        s = (f"--hfov {p['hfov_vis']:.2f} --yaw {p['yaw']:.2f} "
             f"--pitch {p['pitch']:.2f} --roll {p['roll']:.2f}")
        self.root.clipboard_clear()
        self.root.clipboard_append(s)
        self.set_status(f"copied:  {s}")

    def export(self, kind):
        # Export at a HIGHER scale than the screen: the maps are analytic, so
        # a finer grid is genuinely finer, not an upscale of what you saw.
        blend, w, therm, _ = self.compose(self.EXPORT_SCALE)
        arr = {"blend": blend, "warped": w, "thermal": therm}[kind]
        base = os.path.splitext(os.path.basename(self.vis_path))[0]
        p = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=f"{base}_{kind}.png",
            filetypes=[("PNG", "*.png")])
        if not p:
            return
        cv2.imwrite(p, arr)
        self.set_status(f"wrote {os.path.basename(p)}  "
                        f"({arr.shape[1]}x{arr.shape[0]})")


def main():
    ap = argparse.ArgumentParser(
        description="Overlay a Sony frame on a Lepton frame, correctly.")
    d = load_defaults()
    ap.add_argument("files", nargs="*", help="a .jpg and a .npy, either order")
    ap.add_argument("--hfov", type=float, default=d["hfov"])
    ap.add_argument("--alpha", type=float, default=d["alpha"])
    ap.add_argument("--yaw", type=float, default=d["yaw"])
    ap.add_argument("--pitch", type=float, default=d["pitch"])
    ap.add_argument("--roll", type=float, default=d["roll"])
    ap.add_argument("--coco", default="yolo11n.pt",
                    help="COCO detector for the VISIBLE frame's people. "
                         "Downloads on first use. The thermal side uses your "
                         "own omega model from models/vN.")
    args = ap.parse_args()

    root = TkinterDnD.Tk() if HAVE_DND else tk.Tk()
    app = App(root, args)
    for f in args.files:
        root.after(120, lambda p=os.path.abspath(f): app.open(p))
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
