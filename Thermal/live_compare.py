#!/usr/bin/env python3
"""
Live head-to-head: classical detector vs trained CNN, scored against a HUMAN.

Why this exists. The CNN was trained on labels the classical detector produced,
so any offline comparison between them is circular: disagreement is scored as a
CNN error even when the CNN is right and the teacher was wrong. Pruning does not
fix it, because pruning removes false positives you can SEE — a missed person
looks identical to an empty frame in review, so the surviving labels
systematically under-count, which punishes a model that detects more.

The only unbiased referee is a person who can see the room. So: both detectors
run on the same frame, and you type the true occupancy with the number keys.
Every frame is logged with three numbers — truth, classical, CNN — and the
summary at exit scores BOTH against truth, not against each other.

    ./run.sh live_compare.py --weights best.pt --view horizontal

Keys
    0-9   set the TRUE number of people in view (stays until you change it)
    space pause
    l     start/stop logging to CSV
    s     save the current frame (npy + annotated png)
    q     quit and print the scoreboard

Record honestly: set the count BEFORE someone enters or leaves, and pause while
you reposition. A wrong truth value is worse than no data.
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


PANEL_W = 300
SPAN_C = (15.0, 45.0)      # MUST match make_dataset.py, or the CNN sees a
                           # different image encoding than it was trained on


def render_for_cnn(data):
    """Exactly the encoding the model was trained on. Fixed span, 3-channel."""
    lo, hi = SPAN_C
    v = np.clip((data.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    g = (v * 255.0).astype(np.uint8)
    return cv2.merge([g, g, g])


def open_camera(args):
    if not args.opencv:
        try:
            from lepton_libuvc import LeptonUVC
            print("  libuvc: radiometric Y16")
            return LeptonUVC(), True
        except Exception as e:
            print(f"  libuvc unavailable: {e}\n  falling back to OpenCV ...")
    cam = TD.ThermalCamera(args.device if args.device is not None else 0)
    return cam, getattr(cam, "is_temp", False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--view", default="horizontal", choices=TD.VIEW_MODES)
    ap.add_argument("--cohesion", type=int, default=1)
    ap.add_argument("--delta", type=float, default=2.5)
    ap.add_argument("--tmin", type=float, default=TD.DEFAULT_TMIN_C)
    ap.add_argument("--tmax", type=float, default=TD.DEFAULT_TMAX_C)
    ap.add_argument("--min-area", type=int, default=TD.MIN_BLOB_AREA)
    ap.add_argument("--p-filter", action="store_true", default=True)
    ap.add_argument("--p-min", type=int, default=4)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--opencv", action="store_true")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    from ultralytics import YOLO
    print(f"loading {args.weights} ...")
    model = YOLO(args.weights)

    cam, is_temp = open_camera(args)
    if not is_temp:
        print("\n  WARNING: not radiometric. The CNN was trained on a FIXED\n"
              "  temperature span; an AGC image is a different encoding and the\n"
              "  comparison will be meaningless. Use the libuvc path.\n")

    os.makedirs(TD.LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(TD.LOG_DIR, f"compare_{stamp}.csv")
    fh = open(csv_path, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["frame", "timestamp", "truth", "classical", "cnn",
                 "cnn_conf_mean", "ambient_c", "note"])

    flags = dict(TD.DEFAULT_FLAGS)
    flags["pfilter"] = args.p_filter

    S = max(2, args.scale)
    WIN = "FLUXNET  classical  vs  CNN"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    truth = 0
    logging_on = False
    paused = False
    frame_i = 0
    # rows are (truth, classical, cnn), appended only while logging
    tally = []
    t0, fps = time.time(), 0.0

    print(f"\nlogging to {csv_path}")
    print("press 0-9 to set the TRUE count, l to start logging, q to finish\n")

    while True:
        if not paused:
            ok, data = cam.read()
            if not ok:
                continue
            frame_i += 1
            dt = time.time() - t0
            t0 = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

            # ---- classical ------------------------------------------------
            thr, bg = TD.compute_threshold(data, is_temp, args.delta)
            dets, mask = TD.detect_people(
                data, thr,
                tmin=args.tmin if is_temp else None,
                tmax=args.tmax if is_temp else None,
                view=args.view, min_area=args.min_area,
                cohesion=args.cohesion, flags=flags,
                p_filter=args.p_filter, p_min=args.p_min)
            n_cls = len(dets)

            # ---- CNN ------------------------------------------------------
            img = render_for_cnn(data)
            res = model.predict(img, verbose=False, conf=args.conf)[0]
            cls_ids = res.boxes.cls.cpu().numpy() if len(res.boxes) else np.array([])
            confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.array([])
            xyxy = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
            keep = cls_ids == 0                      # class 0 = person
            n_cnn = int(keep.sum())
            cmean = float(confs[keep].mean()) if n_cnn else 0.0

            if logging_on:
                wr.writerow([frame_i,
                             datetime.now().isoformat(timespec="milliseconds"),
                             truth, n_cls, n_cnn, f"{cmean:.3f}",
                             f"{float(np.median(data)):.2f}", args.note])
                tally.append((truth, n_cls, n_cnn))

        # ---- draw: same frame twice, classical left, CNN right ------------
        base = TD.colorize(data)
        left = cv2.resize(base, (base.shape[1] * S, base.shape[0] * S),
                          interpolation=cv2.INTER_NEAREST)
        left = TD.draw_boxes(left, dets, is_temp, S, True)
        cv2.rectangle(left, (0, 0), (left.shape[1], 22), (14, 14, 17), -1)
        cv2.putText(left, f"CLASSICAL   n = {n_cls}", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 150), 1, cv2.LINE_AA)

        right = cv2.resize(base, (base.shape[1] * S, base.shape[0] * S),
                           interpolation=cv2.INTER_NEAREST)
        for (x0, y0, x1, y1), c, k in zip(xyxy, confs, cls_ids):
            if k != 0:
                continue
            p0 = (int(x0 * S / 4), int(y0 * S / 4))
            p1 = (int(x1 * S / 4), int(y1 * S / 4))
            cv2.rectangle(right, p0, p1, (255, 170, 60), 2)
            t = f"{c:.2f}"
            cv2.putText(right, t, (p0[0], max(14, p0[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(right, t, (p0[0], max(14, p0[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 170, 60), 1, cv2.LINE_AA)
        cv2.rectangle(right, (0, 0), (right.shape[1], 22), (14, 14, 17), -1)
        cv2.putText(right, f"CNN   n = {n_cnn}", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 170, 60), 1, cv2.LINE_AA)

        H = left.shape[0]
        sb = np.full((H, PANEL_W, 3), 18, np.uint8)
        y = 34
        cv2.putText(sb, "TRUTH", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 210), 1, cv2.LINE_AA)
        cv2.putText(sb, str(truth), (150, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += 46
        for lab, val, col in (("classical", n_cls, (120, 255, 150)),
                              ("cnn", n_cnn, (255, 170, 60))):
            hit = val == truth
            cv2.putText(sb, lab, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        col, 1, cv2.LINE_AA)
            cv2.putText(sb, str(val), (150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        col if hit else (80, 80, 255), 1, cv2.LINE_AA)
            cv2.putText(sb, "OK" if hit else "X", (210, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (120, 255, 150) if hit else (80, 80, 255), 1, cv2.LINE_AA)
            y += 26

        if tally:
            a = np.array(tally)
            y += 16
            cv2.line(sb, (12, y - 10), (PANEL_W - 12, y - 10), (55, 55, 65), 1)
            cv2.putText(sb, f"SCORED  {len(a)} frames", (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 190, 90), 1, cv2.LINE_AA)
            y += 24
            for lab, col_i, col in (("classical", 1, (120, 255, 150)),
                                    ("cnn", 2, (255, 170, 60))):
                acc = 100.0 * float((a[:, col_i] == a[:, 0]).mean())
                bias = float((a[:, col_i] - a[:, 0]).mean())
                cv2.putText(sb, f"{lab:<10} {acc:5.1f}%  bias {bias:+.2f}",
                            (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1,
                            cv2.LINE_AA)
                y += 22

        y = H - 92
        cv2.line(sb, (12, y - 12), (PANEL_W - 12, y - 12), (55, 55, 65), 1)
        for txt, on in (("0-9  set truth", True),
                        (f"l    logging {'ON' if logging_on else 'off'}", logging_on),
                        (f"spc  {'PAUSED' if paused else 'running'}", paused),
                        ("q    quit + score", False)):
            cv2.putText(sb, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (120, 255, 150) if on else (140, 140, 152), 1, cv2.LINE_AA)
            y += 20

        cv2.putText(sb, f"{fps:4.1f} fps", (PANEL_W - 84, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 132), 1, cv2.LINE_AA)
        cv2.imshow(WIN, np.hstack([left, right, sb]))

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif ord("0") <= k <= ord("9"):
            truth = k - ord("0")
            print(f"  truth -> {truth}")
        elif k == ord(" "):
            paused = not paused
        elif k == ord("l"):
            logging_on = not logging_on
            print(f"  logging {'ON' if logging_on else 'OFF'}")
        elif k == ord("s"):
            st = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            np.save(os.path.join(TD.LOG_DIR, f"cmp_{st}.npy"), data)
            cv2.imwrite(os.path.join(TD.LOG_DIR, f"cmp_{st}.png"),
                        np.hstack([left, right, sb]))
            print(f"  saved cmp_{st}")

    fh.close()
    cam.release()
    cv2.destroyAllWindows()

    if not tally:
        print("\nno frames scored — press 'l' to log next time")
        return
    a = np.array(tally)
    n = len(a)
    print(f"\n{'='*58}\nSCORED AGAINST HUMAN TRUTH — {n} frames\n{'='*58}")
    print(f"{'':<12}{'exact':>8}{'off by 1':>10}{'off 2+':>9}{'bias':>9}{'MAE':>8}")
    for lab, col in (("classical", 1), ("cnn", 2)):
        d = a[:, col] - a[:, 0]
        print(f"{lab:<12}{100*float((d==0).mean()):7.1f}%"
              f"{100*float((np.abs(d)==1).mean()):9.1f}%"
              f"{100*float((np.abs(d)>1).mean()):8.1f}%"
              f"{float(d.mean()):+9.2f}{float(np.abs(d).mean()):8.2f}")
    agree = 100.0 * float((a[:, 1] == a[:, 2]).mean())
    both = 100.0 * float(((a[:, 1] == a[:, 0]) & (a[:, 2] == a[:, 0])).mean())
    cnn_only = 100.0 * float(((a[:, 2] == a[:, 0]) & (a[:, 1] != a[:, 0])).mean())
    cls_only = 100.0 * float(((a[:, 1] == a[:, 0]) & (a[:, 2] != a[:, 0])).mean())
    print(f"\nthe two agree with each other on {agree:.1f}% of frames")
    print(f"  both right      {both:5.1f}%")
    print(f"  only CNN right  {cnn_only:5.1f}%")
    print(f"  only classical  {cls_only:5.1f}%")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
