#!/usr/bin/env python3
"""
Run the trained CNN on the live camera. Nothing else.

live_compare.py runs the classical detector and the CNN side by side and asks
you to type ground truth. That is the right tool for scoring, and the wrong one
for simply watching the model work: half the screen and most of the failure
modes belong to a detector you are not testing. This drops the classical path,
the truth-typing and the scoring, and shows one panel.

    ./run.sh live_yolo.py --weights models/v2/best.pt
    ./run.sh live_yolo.py --weights models/v2/best.pt --conf 0.374 --scale 6

ENCODING. The image handed to the model is built with the same fixed 15-45 C
span make_dataset.py used, NOT the per-frame percentile stretch that looks nicer
on screen. This matters more than it sounds: with a per-frame stretch the same
person encodes to different pixel values depending on what else is in view, so
the model sees an input distribution it never trained on and quietly degrades.
The colourful picture you see is a separate render for your eyes only.

LABEL PLACEMENT. An omega box sits at the top of its person box, so drawing
both labels at their own top-left corners puts them on the same pixels and one
overprints the other — which is exactly what makes Ultralytics' own val_batch
previews hard to read. Here person labels go below their box and omega labels
above, so both stay legible.

Keys
    q / ESC   quit
    space     pause
    s         save this frame (npy + annotated png)
    l         start/stop logging counts to CSV
    [ / ]     confidence threshold down / up by 0.05
    p         cycle which classes are drawn: both -> person -> omega
    h         hide/show the sidebar
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

import thermal_detect as TD


SPAN_C = (15.0, 45.0)     # MUST match make_dataset.py and thermal_detect.PNG_SPAN_C
SIDEBAR_W = 300
NAMES = ["person", "omega"]
COLOR = [(90, 255, 120), (60, 220, 255)]      # BGR: person green, omega amber


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


def label(img, text, x, y, col, above=True):
    """Draw a label clear of the box edge, clamped to stay on screen."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    ly = max(th + 3, y - 4) if above else min(img.shape[0] - 3, y + th + 5)
    lx = min(max(0, x), img.shape[1] - tw - 2)
    cv2.rectangle(img, (lx, ly - th - 3), (lx + tw + 4, ly + 2), (18, 18, 20), -1)
    cv2.putText(img, text, (lx + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                col, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--conf", type=float, default=0.374,
                    help="confidence threshold. 0.374 is where F1 peaked for "
                         "the v2 model; the curve is flat from 0.10 to 0.80 so "
                         "this is not a delicate choice.")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="inference size. Training ran at 640 on 160x120 "
                         "frames, so leave this alone unless you retrained.")
    ap.add_argument("--scale", type=int, default=5)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--opencv", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    from ultralytics import YOLO
    print(f"loading {args.weights} ...")
    if not os.path.exists(args.weights):
        sys.exit(f"no such file: {args.weights}")
    model = YOLO(args.weights)
    names = getattr(model, "names", None) or {}
    print(f"  classes: {names}")

    cam = open_camera(args)
    data, is_temp = cam.read()
    if data is None:
        sys.exit("no frame from the camera")
    if not is_temp:
        print("\n  WARNING: not radiometric. The model was trained on a FIXED\n"
              "  15-45 C span; an AGC image is a different encoding entirely\n"
              "  and detections will be unreliable. Use the libuvc path.\n")

    os.makedirs(TD.LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(TD.LOG_DIR, f"yolo_{stamp}")
    csv_path = os.path.join(TD.LOG_DIR, f"yolo_{stamp}.csv")
    fh = open(csv_path, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["frame", "timestamp", "n_person", "n_omega",
                 "conf_mean", "conf_min", "ambient_c", "thresh", "note"])

    S = max(2, args.scale)
    WIN = "FLUXNET  live YOLO"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    paused = False
    logging_on = False
    show_bar = True
    draw_mode = 0                 # 0 both, 1 person only, 2 omega only
    frame_i = saved = 0
    t0, fps, infer_ms = time.time(), 0.0, 0.0
    hist = []                     # recent person counts, for a stability read

    print(f"\nlogging to {csv_path}")
    print("q quit   space pause   s save   l log   [ ] conf   p classes\n")

    boxes = []
    while True:
        if not paused:
            data, is_temp = cam.read()
            if data is None:
                continue
            frame_i += 1
            dt = time.time() - t0
            t0 = time.time()
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            img = render_for_cnn(data)
            t1 = time.time()
            res = model.predict(img, verbose=False, conf=args.conf,
                                imgsz=args.imgsz)[0]
            infer_ms = 0.9 * infer_ms + 0.1 * (1000.0 * (time.time() - t1))

            boxes = []
            if len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                cf = res.boxes.conf.cpu().numpy()
                boxes = [(int(c), float(v), *map(float, b))
                         for b, c, v in zip(xyxy, cls, cf)]

            n_p = sum(1 for b in boxes if b[0] == 0)
            n_o = sum(1 for b in boxes if b[0] == 1)
            cf_all = [b[1] for b in boxes]
            hist.append(n_p)
            if len(hist) > 60:
                hist.pop(0)

            if logging_on:
                wr.writerow([frame_i,
                             datetime.now().isoformat(timespec="milliseconds"),
                             n_p, n_o,
                             f"{np.mean(cf_all):.3f}" if cf_all else "",
                             f"{min(cf_all):.3f}" if cf_all else "",
                             f"{float(np.median(data)):.2f}",
                             f"{args.conf:.3f}", args.note])
                fh.flush()

        # ---------------- draw ----------------
        vis = cv2.resize(TD.colorize(data),
                         (data.shape[1] * S, data.shape[0] * S),
                         interpolation=cv2.INTER_NEAREST)
        for c, cf, x0, y0, x1, y1 in boxes:
            if draw_mode == 1 and c != 0:
                continue
            if draw_mode == 2 and c != 1:
                continue
            col = COLOR[c] if c < len(COLOR) else (200, 200, 200)
            p0 = (int(x0 * S), int(y0 * S))
            p1 = (int(x1 * S), int(y1 * S))
            cv2.rectangle(vis, p0, p1, col, 2)
            nm = NAMES[c] if c < len(NAMES) else str(c)
            # person below its box, omega above: an omega sits at the top of a
            # person, so same-corner labels would land on the same pixels
            label(vis, f"{nm} {cf:.2f}", p0[0], p1[1] if c == 0 else p0[1],
                  col, above=(c != 0))

        if show_bar:
            bar = np.full((vis.shape[0], SIDEBAR_W, 3), 16, np.uint8)
            n_p = sum(1 for b in boxes if b[0] == 0)
            n_o = sum(1 for b in boxes if b[0] == 1)
            cf_all = [b[1] for b in boxes]

            def put(y, txt, col=(215, 215, 225), sc=0.46, th=1):
                cv2.putText(bar, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, sc,
                            col, th, cv2.LINE_AA)

            put(30, "PEOPLE", (150, 150, 165), 0.42)
            cv2.putText(bar, str(n_p), (12, 84), cv2.FONT_HERSHEY_SIMPLEX,
                        1.9, COLOR[0], 3, cv2.LINE_AA)
            put(116, f"omega    {n_o}", COLOR[1])
            put(140, f"boxes    {len(boxes)}")
            y = 176
            put(y, "CONFIDENCE", (150, 150, 165), 0.42); y += 24
            put(y, f"threshold  {args.conf:.3f}"); y += 22
            if cf_all:
                put(y, f"mean       {np.mean(cf_all):.2f}"); y += 22
                put(y, f"min        {min(cf_all):.2f}"); y += 22
            else:
                put(y, "no detections", (130, 130, 140)); y += 22
            y += 14
            put(y, "STABILITY", (150, 150, 165), 0.42); y += 24
            if len(hist) > 5:
                # a count that flickers frame to frame is the failure you can
                # see without ground truth; sd over the last ~60 frames shows it
                sd = float(np.std(hist))
                col = (120, 220, 130) if sd < 0.3 else \
                      (70, 190, 255) if sd < 0.8 else (80, 90, 255)
                put(y, f"sd(60f)    {sd:.2f}", col); y += 22
                put(y, f"mode       {int(np.bincount(hist).argmax())}"); y += 22
            y += 14
            put(y, "SPEED", (150, 150, 165), 0.42); y += 24
            put(y, f"infer      {infer_ms:.1f} ms"); y += 22
            put(y, f"loop       {fps:.1f} fps"); y += 22
            y += 14
            put(y, f"ambient    {float(np.median(data)):.1f} C"); y += 22
            put(y, f"saved      {saved}"); y += 22
            if logging_on:
                put(y, "LOGGING", (80, 90, 255), 0.5, 2); y += 22
            if paused:
                put(y, "PAUSED", (70, 190, 255), 0.5, 2)
            mode = ("both", "person only", "omega only")[draw_mode]
            cv2.putText(bar, f"showing: {mode}", (12, bar.shape[0] - 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 152), 1,
                        cv2.LINE_AA)
            cv2.putText(bar, "q quit  s save  l log  [ ] conf  p  h",
                        (12, bar.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (110, 110, 122), 1, cv2.LINE_AA)
            vis = np.hstack([vis, bar])

        cv2.imshow(WIN, vis)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            paused = not paused
        elif k == ord("h"):
            show_bar = not show_bar
        elif k == ord("p"):
            draw_mode = (draw_mode + 1) % 3
        elif k == ord("["):
            args.conf = max(0.01, args.conf - 0.05)
        elif k == ord("]"):
            args.conf = min(0.99, args.conf + 0.05)
        elif k == ord("l"):
            logging_on = not logging_on
            print("logging ON" if logging_on else "logging paused")
        elif k == ord("s"):
            os.makedirs(out_dir, exist_ok=True)
            stem = f"yolo_{saved:06d}"
            np.save(os.path.join(out_dir, stem + ".npy"), data)
            cv2.imwrite(os.path.join(out_dir, stem + ".png"), vis)
            saved += 1
            print(f"saved {stem}")

    cv2.destroyAllWindows()
    fh.close()
    print(f"\nframes           {frame_i}")
    print(f"inference        {infer_ms:.1f} ms  ({fps:.1f} fps loop)")
    print(f"log              {csv_path}")
    if saved:
        print(f"saved frames     {out_dir}")
    print("\nNo ground truth was recorded — this tool only shows what the model\n"
          "sees. For a scored comparison against a human, use live_compare.py.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
