#!/usr/bin/env python3
"""
Send a .cfg to the sensor and read the point cloud back.

The demo firmware does nothing until it is configured. On boot it sits at its
prompt waiting; you push a chirp profile down the CLI port line by line, finish
with sensorStart, and it begins streaming binary frames out the DATA port.
Power-cycle the board and it forgets everything, so this runs every session.

    python3 stream.py --cfg profile.cfg
    python3 stream.py --cfg profile.cfg --save logs/run1.jsonl
    python3 stream.py --no-config          # already configured, just listen

WHY LINE BY LINE, WITH A WAIT. The CLI acknowledges each command with "Done"
and rejects the whole profile if it receives the next line before it has
finished the last. Blasting the file at it looks like it works — the port
accepts the bytes — and then the sensor never starts, with no error anywhere
obvious. So each line is sent, the reply is read, and anything that is not
"Done" stops the run and prints what the sensor actually said.

THE FRAME FORMAT. Every frame opens with an 8-byte magic word, then a 40-byte
header, then a series of type-length-value blocks. Resynchronising on the magic
word rather than trusting the stream to stay aligned matters: a dropped byte on
a 921600-baud link would otherwise desynchronise the parser permanently.
"""

import argparse
import json
import os
import struct
import sys
import time
from datetime import datetime

try:
    import serial
except ImportError:
    sys.exit("pyserial missing.  ./.venv/bin/pip install pyserial")


MAGIC = bytes([2, 1, 4, 3, 6, 5, 8, 7])
HDR = struct.Struct("<IIIIIII")          # after the magic word: 28 bytes
HDR_LEN = 28

TLV_DETECTED_POINTS = 1
TLV_RANGE_PROFILE = 2
TLV_NOISE_PROFILE = 3
TLV_AZIMUTH_HEATMAP = 4
TLV_RANGE_DOPPLER_HEATMAP = 5
TLV_STATS = 6
TLV_SIDE_INFO = 7

TLV_NAMES = {1: "points", 2: "range_profile", 3: "noise", 4: "azimuth_heatmap",
             5: "range_doppler", 6: "stats", 7: "side_info"}


def send_config(port, path, verbose=True):
    """Push the profile down the CLI, one line at a time, checking each reply."""
    with open(path) as fh:
        lines = [l.strip() for l in fh]
    lines = [l for l in lines if l and not l.startswith("%")]

    with serial.Serial(port, 115200, timeout=0.6) as s:
        s.reset_input_buffer()
        # A bare newline first: if the sensor is mid-run from a previous
        # session it must be stopped, or every subsequent command is refused.
        s.write(b"sensorStop\n")
        time.sleep(0.1)
        s.reset_input_buffer()

        for i, line in enumerate(lines, 1):
            s.write((line + "\n").encode())
            s.flush()
            time.sleep(0.05)
            reply = s.read(256).decode("ascii", "ignore")
            ok = "Done" in reply or "Ignored" in reply
            if verbose:
                mark = "  " if ok else "!!"
                print(f"{mark} {i:3d}/{len(lines)}  {line[:52]}")
            if not ok:
                print(f"\nSensor rejected line {i}:\n  {line}\n"
                      f"  reply: {reply.strip()!r}")
                print("\nCommon causes: the .cfg is for a different device "
                      "(ISK vs AOP), or for a different demo than the one "
                      "flashed. The profile must match the firmware.")
                return False
    return True


def parse_frame(buf):
    """buf starts immediately after the magic word. Returns (dict, consumed)."""
    if len(buf) < HDR_LEN:
        return None, 0
    (version, total_len, platform, frame_no,
     cpu_cycles, num_obj, num_tlv) = HDR.unpack_from(buf, 0)

    body_len = total_len - len(MAGIC) - HDR_LEN
    if body_len < 0 or len(buf) < HDR_LEN + body_len:
        return None, 0                       # wait for the rest

    out = {"frame": frame_no, "n_obj": num_obj, "n_tlv": num_tlv,
           "points": [], "tlvs": []}
    off = HDR_LEN
    end = HDR_LEN + body_len
    for _ in range(num_tlv):
        if off + 8 > end:
            break
        t_type, t_len = struct.unpack_from("<II", buf, off)
        off += 8
        payload_end = off + t_len - 8 if t_len >= 8 else off + t_len
        payload_end = min(payload_end, end)
        out["tlvs"].append(TLV_NAMES.get(t_type, str(t_type)))

        if t_type == TLV_DETECTED_POINTS:
            n = (payload_end - off) // 16
            for k in range(n):
                x, y, z, v = struct.unpack_from("<ffff", buf, off + k * 16)
                out["points"].append({"x": round(x, 3), "y": round(y, 3),
                                      "z": round(z, 3), "v": round(v, 3)})
        off = payload_end
    return out, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default="/dev/cu.usbserial-010821020",
                    help="CLI port (from find_ports.py)")
    ap.add_argument("--data", default="/dev/cu.usbserial-010821021",
                    help="DATA port (from find_ports.py)")
    ap.add_argument("--cfg", help="chirp profile to send")
    ap.add_argument("--no-config", action="store_true",
                    help="skip configuration; the sensor is already running")
    ap.add_argument("--save", help="append each frame to this .jsonl")
    ap.add_argument("--seconds", type=float, default=0,
                    help="stop after this long (0 = run until Ctrl-C)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.no_config:
        if not args.cfg:
            sys.exit("need --cfg <profile.cfg>, or --no-config if already running")
        if not os.path.exists(args.cfg):
            sys.exit(f"no such file: {args.cfg}")
        print(f"configuring from {args.cfg} ...")
        if not send_config(args.cli, args.cfg, verbose=not args.quiet):
            sys.exit(1)
        print("configured; sensor started\n")

    fh = open(args.save, "a") if args.save else None
    buf = b""
    n_frames = 0
    n_points = 0
    t0 = time.time()
    last_report = t0

    print(f"reading {args.data} at 921600 — Ctrl-C to stop\n")
    try:
        with serial.Serial(args.data, 921600, timeout=0.5) as s:
            while True:
                chunk = s.read(4096)
                if chunk:
                    buf += chunk

                # Resync on the magic word every time rather than assuming the
                # stream stays aligned — one dropped byte would otherwise
                # desynchronise us for good.
                while True:
                    i = buf.find(MAGIC)
                    if i < 0:
                        if len(buf) > 65536:
                            buf = buf[-4096:]      # nothing usable; don't grow
                        break
                    body = buf[i + len(MAGIC):]
                    frame, consumed = parse_frame(body)
                    if frame is None:
                        buf = buf[i:]              # incomplete; wait for more
                        break
                    buf = body[consumed:]
                    n_frames += 1
                    n_points += len(frame["points"])
                    if fh:
                        fh.write(json.dumps(frame) + "\n")
                        fh.flush()
                    if not args.quiet:
                        now = time.time()
                        if now - last_report >= 0.5:
                            fps = n_frames / max(1e-6, now - t0)
                            print(f"\rframe {frame['frame']:>7}   "
                                  f"points {len(frame['points']):>4}   "
                                  f"tlvs {','.join(frame['tlvs'])[:34]:<34} "
                                  f"{fps:5.1f} fps", end="", flush=True)
                            last_report = now

                if args.seconds and time.time() - t0 > args.seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if fh:
            fh.close()

    dt = time.time() - t0
    print(f"\n\nframes      {n_frames}   ({n_frames / max(1e-6, dt):.1f} fps)")
    print(f"points      {n_points}   ({n_points / max(1, n_frames):.1f} per frame)")
    if args.save:
        print(f"saved       {args.save}")
    if n_frames == 0:
        print("\nNo frames. Either the sensor was never started (send a .cfg "
              "ending in sensorStart), or the CLI and DATA ports are swapped.")


if __name__ == "__main__":
    main()
