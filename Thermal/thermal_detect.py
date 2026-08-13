#!/usr/bin/env python3
"""
FLUXNET — thermal person detection baseline (classical CV, no training).

Works in two modes, chosen automatically:

  RADIOMETRIC (uint16 Y16/TLinear, typically on Linux/Raspberry Pi)
      Every pixel is an absolute temperature. Detection = pixels warmer than
      (ambient + delta). Stable, physically meaningful, the real target.

  AGC (uint8, typically what macOS UVC allows)
      The board hands over an auto-gain 8-bit image: brightness is RELATIVE and
      rescales as the scene changes, so absolute thresholds are meaningless.
      Detection instead uses an adaptive threshold (high percentile + spread),
      which tracks the rescaling. Good enough to develop against; not a
      substitute for radiometric data.

Controls (in the video window):
    q / ESC   quit
    s         save current frame (raw .npy + preview .png)
    l         toggle continuous logging
    [ / ]     make detection less / more sensitive
    a         re-estimate background now
    r         cycle view: color -> mask -> both
    h         print this help

Usage:
    python diagnose.py                       # first, see what the camera gives
    python thermal_detect.py                 # auto-find the camera
    python thermal_detect.py --device 1      # force a device index
    python thermal_detect.py --list          # list candidate devices
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np

# ----------------------------------------------------------------------------
LEPTON_W, LEPTON_H = 160, 120
DEFAULT_DELTA_C = 4.0      # radiometric: °C above ambient
DEFAULT_PCTL = 96.0        # AGC: percentile of brightness treated as "hot"
MIN_BLOB_AREA = 20
MAX_BLOB_AREA = 14000
MAX_ASPECT = 6.0
DISPLAY_SCALE = 5
LOG_DIR = "logs"


# ----------------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------------
class ThermalCamera:
    """PureThermal/Lepton UVC wrapper. Prefers radiometric, falls back to AGC."""

    def __init__(self, device_index):
        self.index = device_index
        self.radiometric = False
        self.scale = 0.01

        # Try raw 16-bit first.
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video device {device_index}")
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
        except Exception:
            pass

        ok, probe = cap.read()
        if ok and probe is not None and probe.dtype == np.uint16:
            self.cap = cap
            self.radiometric = True
            med = float(np.median(probe))
            if not (-40 < med * 0.01 - 273.15 < 80):
                self.scale = 0.1
            print(f"  mode: RADIOMETRIC (uint16, {self.scale} K/count)")
            return

        # Raw failed or gave 8-bit: reopen cleanly in normal (AGC) mode.
        cap.release()
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not reopen device {device_index}")
        ok, probe = cap.read()
        if not ok or probe is None:
            raise RuntimeError("Device opened but returned no frames")
        self.cap = cap
        print("  mode: AGC 8-bit (no absolute temperatures)")
        print("        macOS commonly blocks raw Y16; use a Linux host/Pi for")
        print("        radiometric work. Adaptive thresholding is used instead.")

    def read(self):
        """Returns (data, is_temp). data is float32 °C if radiometric, else 0-255."""
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None, False
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.radiometric and frame.dtype == np.uint16:
            return frame.astype(np.float32) * self.scale - 273.15, True
        return frame.astype(np.float32), False

    def release(self):
        self.cap.release()


def scan_devices(max_index=6):
    out = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        ok, f = cap.read()
        if ok and f is not None:
            h, w = f.shape[:2]
            looks_lepton = (w == LEPTON_W and h in (LEPTON_H, LEPTON_H * 2)) or f.dtype == np.uint16
            out.append((i, w, h, str(f.dtype), looks_lepton))
        cap.release()
    return out


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def compute_threshold(data, is_temp, sensitivity):
    """
    Returns (threshold_value, background_level).

    radiometric : threshold = ambient + sensitivity  (sensitivity in °C)
    AGC         : threshold = percentile(sensitivity) of the frame, floored at
                  median + 2*MAD so an empty room doesn't self-detect noise.
    """
    if is_temp:
        bg = float(np.median(data))
        return bg + sensitivity, bg

    bg = float(np.median(data))
    mad = float(np.median(np.abs(data - bg))) or 1.0
    pctl_thr = float(np.percentile(data, sensitivity))
    floor_thr = bg + 4.0 * mad
    return max(pctl_thr, floor_thr), bg


def merge_nearby(dets, gap_px):
    """
    Fragments of one body (arm, shoulder, head separated by cooler clothing)
    arrive as separate components. Merge detections whose bounding boxes are
    within gap_px of each other — verified necessary on real frames, where a
    single rotating arm split into 5 boxes inside one 39x82 px region.
    """
    if gap_px <= 0 or len(dets) < 2:
        return dets

    def near(a, b):
        ax, ay, aw, ah = a["bbox"]
        bx, by, bw, bh = b["bbox"]
        dx = max(0, max(ax, bx) - min(ax + aw, bx + bw))
        dy = max(0, max(ay, by) - min(ay + ah, by + bh))
        return dx <= gap_px and dy <= gap_px

    groups = []
    for d in dets:
        placed = False
        for g in groups:
            if any(near(d, m) for m in g):
                g.append(d)
                placed = True
                break
        if not placed:
            groups.append([d])

    # one more pass: groups can become adjacent after absorbing members
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if any(near(a, b) for a in groups[i] for b in groups[j]):
                    groups[i] += groups[j]
                    del groups[j]
                    changed = True
                    break
            if changed:
                break

    merged = []
    for g in groups:
        if len(g) == 1:
            merged.append(g[0])
            continue
        x0 = min(d["bbox"][0] for d in g)
        y0 = min(d["bbox"][1] for d in g)
        x1 = max(d["bbox"][0] + d["bbox"][2] for d in g)
        y1 = max(d["bbox"][1] + d["bbox"][3] for d in g)
        area = sum(d["area_px"] for d in g)
        merged.append({
            "bbox": (x0, y0, x1 - x0, y1 - y0),
            "centroid": (sum(d["centroid"][0] * d["area_px"] for d in g) / area,
                         sum(d["centroid"][1] * d["area_px"] for d in g) / area),
            "area_px": area,
            "val_max": max(d["val_max"] for d in g),
            "val_mean": sum(d["val_mean"] * d["area_px"] for d in g) / area,
            "fragments": len(g),
        })
    merged.sort(key=lambda d: d["area_px"], reverse=True)
    return merged


def detect_people(data, threshold, merge_gap=6):
    mask = (data > threshold).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dets = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
            continue
        if max(w, h) / max(1, min(w, h)) > MAX_ASPECT:
            continue
        blob = data[labels == i]
        dets.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "centroid": (float(cents[i][0]), float(cents[i][1])),
            "area_px": int(area),
            "val_max": float(blob.max()),
            "val_mean": float(blob.mean()),
        })
    dets = merge_nearby(dets, merge_gap)
    dets.sort(key=lambda d: d["area_px"], reverse=True)
    return dets, mask


# ----------------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------------
def colorize(data):
    lo = float(np.percentile(data, 1))
    hi = float(np.percentile(data, 99))
    if hi - lo < 1e-3:
        hi = lo + 1.0
    norm = np.clip((data - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def draw_overlay(vis, dets, bg, thr, sens, is_temp, fps, logging_on):
    unit = "C" if is_temp else ""
    for d in dets:
        x, y, w, h = d["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cx, cy = int(d["centroid"][0]), int(d["centroid"][1])
        cv2.drawMarker(vis, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 6, 1)
        cv2.putText(vis, f"{d['val_max']:.0f}{unit}", (x, max(8, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

    mode = "RAD" if is_temp else "AGC"
    sens_s = f"+{sens:.1f}C" if is_temp else f"p{sens:.0f}"
    cv2.putText(vis, f"{mode} n={len(dets)} bg={bg:.1f} thr={thr:.1f} {sens_s} {fps:.0f}fps",
                (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
    if logging_on:
        cv2.circle(vis, (vis.shape[1] - 8, 8), 4, (0, 0, 255), -1)
    return vis


HELP = """
  q/ESC quit | s save frame | l toggle logging
  [ less sensitive   ] more sensitive
  a re-estimate background | r cycle view (color/mask/both)
"""


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--delta", type=float, default=None,
                    help="radiometric: °C above ambient (default 4.0)")
    ap.add_argument("--pctl", type=float, default=None,
                    help="AGC: brightness percentile treated as hot (default 96)")
    ap.add_argument("--note", type=str, default="")
    ap.add_argument("--opencv", action="store_true",
                    help="skip libuvc and force the OpenCV capture path")
    args = ap.parse_args()

    if args.list:
        for i, w, h, dt, lep in scan_devices():
            print(f"  [{i}] {w}x{h} {dt}" + ("   <-- looks like Lepton" if lep else ""))
        return

    # Preferred path: libuvc (works where OpenCV/AVFoundation cannot, and is
    # the only way to get radiometric Y16 on macOS).
    cam = None
    if not args.opencv:
        try:
            from lepton_libuvc import LeptonUVC
            cam = LeptonUVC()
            print(f"  mode: RADIOMETRIC via libuvc ({cam.format_used})")
        except Exception as e:
            print(f"  libuvc unavailable: {e}")
            print("  falling back to OpenCV capture ...")

    if cam is None:
        dev = args.device
        if dev is None:
            cands = scan_devices()
            lep = [c for c in cands if c[4]]
            if not lep:
                print("No Lepton-like device found. Seen:")
                for i, w, h, dt, _ in cands:
                    print(f"  [{i}] {w}x{h} {dt}")
                print("Run diagnose.py / backend_test.py, then pass --device N")
                return
            dev = lep[0][0]
        print(f"Opening device {dev} via OpenCV ...")
        cam = ThermalCamera(dev)

    sens = (args.delta if args.delta is not None else DEFAULT_DELTA_C) if cam.radiometric \
        else (args.pctl if args.pctl is not None else DEFAULT_PCTL)

    os.makedirs(LOG_DIR, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LOG_DIR, f"session_{session}.csv")
    fcsv = open(csv_path, "w", newline="")
    wr = csv.writer(fcsv)
    wr.writerow(["timestamp", "frame", "mode", "n_det", "background", "threshold",
                 "cx", "cy", "area_px", "val_max", "note"])

    logging_on = False
    view = 0
    frame_i = 0
    fps = 0.0
    t_last = time.time()
    print(HELP)

    while True:
        data, is_temp = cam.read()
        if data is None:
            print("Frame read failed; stopping.")
            break
        frame_i += 1

        thr, bg = compute_threshold(data, is_temp, sens)
        dets, mask = detect_people(data, thr)

        now = time.time()
        dt = now - t_last
        t_last = now
        if dt > 0:
            fps = (0.9 * fps + 0.1 / dt) if fps else 1.0 / dt

        if logging_on:
            ts = datetime.now().isoformat(timespec="milliseconds")
            mode = "RAD" if is_temp else "AGC"
            if dets:
                for d in dets:
                    wr.writerow([ts, frame_i, mode, len(dets), f"{bg:.2f}", f"{thr:.2f}",
                                 f"{d['centroid'][0]:.1f}", f"{d['centroid'][1]:.1f}",
                                 d["area_px"], f"{d['val_max']:.2f}", args.note])
            else:
                wr.writerow([ts, frame_i, mode, 0, f"{bg:.2f}", f"{thr:.2f}",
                             "", "", "", "", args.note])

        color = colorize(data)
        maskc = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        base = color if view == 0 else (maskc if view == 1 else
                                        cv2.addWeighted(color, 0.7, maskc, 0.3, 0))
        vis = draw_overlay(base.copy(), dets, bg, thr, sens, is_temp, fps, logging_on)
        cv2.imshow("FLUXNET thermal baseline",
                   cv2.resize(vis, (vis.shape[1] * DISPLAY_SCALE, vis.shape[0] * DISPLAY_SCALE),
                              interpolation=cv2.INTER_NEAREST))

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("s"):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            np.save(os.path.join(LOG_DIR, f"frame_{stamp}.npy"), data)
            cv2.imwrite(os.path.join(LOG_DIR, f"frame_{stamp}.png"), vis)
            print(f"  saved frame_{stamp}  (n={len(dets)}, thr={thr:.1f})")
        elif k == ord("l"):
            logging_on = not logging_on
            print(f"  logging {'ON' if logging_on else 'OFF'} -> {csv_path}")
        elif k == ord("["):
            sens = max(0.5, sens - 0.5) if is_temp else min(99.5, sens + 0.5)
            print(f"  sensitivity: {sens}")
        elif k == ord("]"):
            sens = sens + 0.5 if is_temp else max(50.0, sens - 0.5)
            print(f"  sensitivity: {sens}")
        elif k == ord("a"):
            print(f"  background = {bg:.2f}, threshold = {thr:.2f}")
        elif k == ord("r"):
            view = (view + 1) % 3
        elif k == ord("h"):
            print(HELP)

    fcsv.close()
    cam.release()
    cv2.destroyAllWindows()
    print(f"\nSession log: {csv_path}")


if __name__ == "__main__":
    main()
