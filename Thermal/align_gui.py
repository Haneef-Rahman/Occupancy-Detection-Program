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
                ("roll", "roll", -20.0, 20.0, args.roll, "{:+.2f}°"))):
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
        for k, v in (("hfov", 67.4), ("alpha", 0.5), ("yaw", 0.0),
                     ("pitch", 0.0), ("roll", 0.0)):
            self.var[k].set(v)
        self.redraw()

    def cycle_mode(self):
        self.mode = {"blend": "thermal", "thermal": "visible",
                     "visible": "blend"}[self.mode]
        self.mode_btn.config(text=f"view: {self.mode}")
        self.redraw()

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
        mx, my = AV.build_maps(self.vis.shape, ow, oh, **self.params())
        w = cv2.remap(self.vis, mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        cover = float(((mx >= 0) & (mx < self.vis.shape[1]) &
                       (my >= 0) & (my < self.vis.shape[0])).mean())
        a = self.var["alpha"].get()
        blend = cv2.addWeighted(w, 1.0 - a, therm, a, 0.0)
        return blend, w, therm, cover

    def redraw(self):
        if self.vis is None or self.arr is None:
            return
        blend, w, therm, cover = self.compose()
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
        try:
            m.tk_popup(self.save_btn.winfo_rootx(),
                       self.save_btn.winfo_rooty()
                       + self.save_btn.winfo_height())
        finally:
            m.grab_release()

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
    ap.add_argument("files", nargs="*", help="a .jpg and a .npy, either order")
    ap.add_argument("--hfov", type=float, default=67.4)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--roll", type=float, default=0.0)
    args = ap.parse_args()

    root = TkinterDnD.Tk() if HAVE_DND else tk.Tk()
    app = App(root, args)
    for f in args.files:
        root.after(120, lambda p=os.path.abspath(f): app.open(p))
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
