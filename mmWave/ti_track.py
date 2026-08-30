#!/usr/bin/env python3
"""
Headless people tracking, using TI's own parser.

    ./.venv/bin/python ti_track.py --cfg configs/AOP_6m_staticRetention.cfg
    ./.venv/bin/python ti_track.py --cfg configs/... --save logs/run1.jsonl
    ./.venv/bin/python ti_track.py --no-config            # already running
    ./.venv/bin/python ti_track.py --self-test            # no hardware needed

WHY THIS REPLACES MY PARSER. stream.py decoded the frame format by walking
every plausible header length and TLV convention until one landed exactly on
the end of the packet. That worked — it independently derived TI's real layout
from live data — but it was reverse engineering, and reverse engineering has a
shelf life. TI ships the authoritative decoder in the Radar Toolbox, it is 1361
lines, it handles every TLV this firmware can emit, and it is Qt-free, so it
imports with nothing but numpy. Using it is strictly better than being clever.

WHAT IS STILL MINE. The serial handling. TI's visualiser sends a config with no
retry and no reply classification, which is why a UART framing glitch there
surfaces as "Parsing .cfg file failed. Did you select the right file?" when the
config was never opened. send_config() in stream.py distinguishes a corrupted
line from a rejected command and retries only the former. That part earned its
keep, so it stays.

WHY IT MATTERS FOR THE PI. Nothing here imports Qt, OpenGL, or a display. The
tracker runs on the radar itself, so this decodes a few hundred bytes at 20 Hz
and nothing more — it will idle on a Pi Zero. Point --ti-common at the toolbox
on whatever machine you deploy to and the same file runs.
"""

import argparse
import glob
import json
import logging
import math
import os
import struct
import sys
import time

# ---------------------------------------------------------------------------
# find TI's parser
# ---------------------------------------------------------------------------

CANDIDATES = [
    "~/Developer/radar_toolbox_*/tools/visualizers/Applications_Visualizer/common",
    "~/Downloads/radar_toolbox_*/tools/visualizers/Applications_Visualizer/common",
    "~/ti/radar_toolbox_*/tools/visualizers/Applications_Visualizer/common",
    "/opt/ti/radar_toolbox_*/tools/visualizers/Applications_Visualizer/common",
]


def find_ti_common(explicit=None):
    """
    Locate Applications_Visualizer/common, newest toolbox first.

    Searched rather than hard-coded because this has to run on the Mac and on
    the Pi, where the toolbox will not live in the same place.
    """
    if explicit:
        p = os.path.expanduser(explicit)
        if not os.path.isfile(os.path.join(p, "parseFrame.py")):
            sys.exit(f"no parseFrame.py in {p}")
        return p
    hits = []
    for pat in CANDIDATES:
        hits += glob.glob(os.path.expanduser(pat))
    hits = [h for h in hits if os.path.isfile(os.path.join(h, "parseFrame.py"))]
    if not hits:
        sys.exit(
            "Could not find TI's parser.\n"
            "  Expected Applications_Visualizer/common inside a radar_toolbox\n"
            "  install. Pass --ti-common <path> if it lives somewhere else."
        )
    return sorted(hits)[-1]


def load_ti_parser(common_dir, verbose=True):
    """
    Import TI's parser.

    Its modules import each other by bare name (`from parseTLVs import *`), so
    the directory has to be on sys.path — it cannot be imported as a package.
    Nothing here pulls in Qt; that has been verified by importing it in an
    environment with no PySide2 present at all.
    """
    if common_dir not in sys.path:
        sys.path.insert(0, common_dir)
    try:
        import parseFrame
    except ImportError as e:
        sys.exit(f"TI parser failed to import from {common_dir}\n  {e}\n"
                 f"  numpy is the only dependency it needs.")
    if verbose:
        print(f"TI parser: {common_dir}")
    return parseFrame


MAGIC = bytes([2, 1, 4, 3, 6, 5, 8, 7])
HDR_LEN = struct.calcsize("Q8I")          # 40, matching TI's headerStruct


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

def frames_from(buf, parse):
    """
    Pull whole frames out of a byte buffer and hand each to TI's parser.

    Yields (parsed, consumed_upto). TI's parseStandardFrame expects the frame
    INCLUDING its magic word, because its header struct starts with a Q that
    swallows the magic — so the slice starts at the magic, not after it.

    Resynchronising on the magic word every time is deliberate: at 921600 baud
    a single dropped byte would otherwise desynchronise the reader for good.
    """
    out = []
    i = 0
    while True:
        j = buf.find(MAGIC, i)
        if j < 0:
            break
        if len(buf) - j < HDR_LEN:
            break                                   # header not here yet
        total = struct.unpack_from("<I", buf, j + 12)[0]   # totalPacketLen
        if not HDR_LEN <= total <= 65536:
            i = j + 1                               # false magic; step past
            continue
        if len(buf) - j < total:
            break                                   # wait for the rest
        out.append(parse(buf[j:j + total]))
        i = j + total
    return out, i


def summarise(f):
    """Flatten TI's numpy output into something printable and JSON-safe."""
    tracks = []
    heights = {}
    if f.get("numDetectedHeights"):
        for h in f["heightData"]:
            heights[int(h[0])] = (round(float(h[2]), 3), round(float(h[1]), 3))
    for t in f.get("trackData", [])[:int(f.get("numDetectedTracks", 0))]:
        tid = int(t[0])
        zmin, zmax = heights.get(tid, (None, None))
        tracks.append({
            "id": tid,
            "x": round(float(t[1]), 3), "y": round(float(t[2]), 3),
            "z": round(float(t[3]), 3),
            "vx": round(float(t[4]), 3), "vy": round(float(t[5]), 3),
            "vz": round(float(t[6]), 3),
            "speed": round(math.hypot(float(t[4]), float(t[5])), 3),
            "range": round(math.hypot(float(t[1]), float(t[2])), 3),
            "conf": round(float(t[11]), 3),
            "z_min": zmin, "z_max": zmax,
        })
    return {
        "frame": int(f.get("frameNum", -1)),
        "error": int(f.get("error", 0)),
        "n_points": int(f.get("numDetectedPoints", 0)),
        "tracks": tracks,
        "assigned": int((assignment(f) < 253).sum()),
    }


def assignment(f):
    """
    Which track each point belongs to, as a length-numDetectedPoints array.

    TI does NOT fold this into the point cloud. The target-index TLV lands in
    its own 'trackIndexes' array while pointCloud column 6 stays at its 255
    initialiser, so reading column 6 quietly reports that nothing was ever
    assigned to anybody. Caught by the self test, which is the entire reason
    it exists. 253-255 are TI's codes for noise or too weak to associate.
    """
    import numpy as np
    n = int(f.get("numDetectedPoints", 0))
    idx = f.get("trackIndexes")
    if idx is not None and len(idx):
        out = np.full(n, 255.0)
        m = min(n, len(idx))
        out[:m] = idx[:m]
        return out
    pc = f.get("pointCloud")
    if pc is not None and n:
        return np.asarray(pc[:n, 6], dtype=float)
    return np.zeros(0)


def points_of(f):
    """Full point cloud as dicts, only when asked — it is the bulky part."""
    pc = f.get("pointCloud")
    n = int(f.get("numDetectedPoints", 0))
    if pc is None or not n:
        return []
    tid = assignment(f)
    return [{"x": round(float(p[0]), 3), "y": round(float(p[1]), 3),
             "z": round(float(p[2]), 3), "v": round(float(p[3]), 3),
             "snr": round(float(p[4]), 1), "tid": int(tid[k])}
            for k, p in enumerate(pc[:n])]


# ---------------------------------------------------------------------------
# self test — proves the wiring without hardware
# ---------------------------------------------------------------------------

def self_test(parseFrame):
    """
    Build a frame the way the firmware does and check TI's parser reads it.

    Worth having because every failure today looked like a sensor problem and
    was not one. If this passes, the parser, the framing and the summariser are
    all fine, and anything still broken is the serial link or the config.
    """
    import numpy as np

    def tlv(t, payload):
        return struct.pack("<II", t, len(payload)) + payload

    targets = b""
    for i, (x, y) in enumerate([(-1.2, 3.0), (1.4, 5.5)]):
        vals = [x, y, 0.9, 0.1, 0.4, 0.0] + [0.0] * 3 + [0.0] * 16 + [1.0, 0.87]
        targets += struct.pack("<I27f", 10 + i, *vals)
    heights = b"".join(struct.pack("<I2f", 10 + i, 1.78, 0.02) for i in range(2))

    units = struct.pack("<5f", 0.01, 0.01, 0.05, 0.02, 1.0)
    pts = units + b"".join(struct.pack("<bbhHH", 5, -10, 20, 200 + k, 30)
                           for k in range(6))
    idx = bytes([0, 0, 1, 1, 255, 255])

    body = (tlv(1020, pts) + tlv(1010, targets) +
            tlv(1011, idx) + tlv(1012, heights))
    n_tlv = 4
    raw = 8 + 32 + len(body)
    pad = (-raw) % 32
    total = raw + pad
    hdr = struct.pack("<8I", 3, total, 0x6843, 4242, 0, 6, n_tlv, 0)
    frame = MAGIC + hdr + body + b"\x00" * pad

    got = parseFrame.parseStandardFrame(frame)
    s = summarise(got)
    ok = True

    def check(label, actual, expect):
        nonlocal ok
        good = actual == expect
        ok &= good
        print(f"  {'ok ' if good else 'FAIL'} {label:<22} {actual!r}"
              f"{'' if good else f'  expected {expect!r}'}")

    print("self test — synthetic people-tracking frame through TI's parser")
    check("frame number", s["frame"], 4242)
    check("parser error code", s["error"], 0)
    check("tracks", len(s["tracks"]), 2)
    check("track ids", [t["id"] for t in s["tracks"]], [10, 11])
    check("points", s["n_points"], 6)
    check("points assigned", s["assigned"], 4)
    check("height of id 10", s["tracks"][0]["z_max"], 1.78)
    t0 = s["tracks"][0]
    check("position of id 10", (t0["x"], t0["y"]), (-1.2, 3.0))
    check("confidence", t0["conf"], 0.87)

    # and the same frame split across chunk boundaries, with junk in front
    stream = b"\x11\x22" + frame + frame[:24]
    buf = b""
    seen = 0
    for k in range(0, len(stream), 7):
        buf += stream[k:k + 7]
        fs, upto = frames_from(buf, parseFrame.parseStandardFrame)
        buf = buf[upto:]
        seen += len(fs)
    check("reassembled from 7-byte chunks", seen, 1)

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default="/dev/cu.usbserial-010821020")
    ap.add_argument("--data", default="/dev/cu.usbserial-010821021")
    ap.add_argument("--cfg")
    ap.add_argument("--no-config", action="store_true",
                    help="skip configuration; the sensor is already running")
    ap.add_argument("--ti-common", help="path to Applications_Visualizer/common")
    ap.add_argument("--save", help="append each frame to this .jsonl")
    ap.add_argument("--points", action="store_true",
                    help="also save the full point cloud, not just tracks")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--cli-baud", type=int, default=115200)
    ap.add_argument("--no-reset", action="store_true",
                    help="do not issue resetDevice before sending the profile")
    ap.add_argument("--trace", action="store_true",
                    help="dump every byte written to and read from the CLI")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # TI's parser is chatty at INFO about TLVs it chooses not to decode.
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(filename)s:%(lineno)d %(message)s")

    parseFrame = load_ti_parser(find_ti_common(args.ti_common),
                                verbose=not args.quiet)

    if args.self_test:
        sys.exit(self_test(parseFrame))

    try:
        import serial
    except ImportError:
        sys.exit("pyserial missing.  ./.venv/bin/pip install pyserial")
    import stream as S               # for the hardened config sender only
    S.TRACE = args.trace

    if not args.no_config:
        if not args.cfg:
            sys.exit("need --cfg <profile.cfg>, or --no-config")
        if not os.path.exists(args.cfg):
            sys.exit(f"no such file: {args.cfg}")
        print(f"\nconfiguring from {args.cfg} ...")
        if not S.send_config(args.cli, args.cfg, args.cli_baud, not args.quiet,
                             reset=not args.no_reset):
            sys.exit(1)
        print("configured; sensor started\n")

    fh = None
    if args.save:
        d = os.path.dirname(args.save)
        if d:
            os.makedirs(d, exist_ok=True)
        fh = open(args.save, "a")

    buf = b""
    n = n_err = 0
    peak = 0
    ids = set()
    occupied = 0
    t0 = time.time()
    last = t0

    print(f"reading {args.data} at {args.baud} — Ctrl-C to stop\n")
    try:
        with serial.Serial(args.data, args.baud, timeout=0.4) as s:
            while True:
                chunk = s.read(4096)
                if chunk:
                    buf += chunk
                    if len(buf) > 1 << 20:
                        buf = buf[-65536:]          # runaway guard
                got, upto = frames_from(buf, parseFrame.parseStandardFrame)
                buf = buf[upto:]

                for f in got:
                    rec = summarise(f)
                    if args.points:
                        rec["points"] = points_of(f)
                    n += 1
                    n_err += 1 if rec["error"] else 0
                    k = len(rec["tracks"])
                    peak = max(peak, k)
                    occupied += 1 if k else 0
                    ids.update(t["id"] for t in rec["tracks"])
                    if fh:
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()

                if got and not args.quiet:
                    now = time.time()
                    if now - last >= 0.5:
                        r = summarise(got[-1])
                        who = ",".join(str(t["id"]) for t in r["tracks"]) or "-"
                        near = min((t["range"] for t in r["tracks"]), default=0)
                        print(f"\rframe {r['frame']:>7}  people {len(r['tracks'])} "
                              f"[{who:<9}]  nearest {near:4.1f} m  "
                              f"points {r['n_points']:>3} ({r['assigned']} assigned)  "
                              f"{n / max(1e-6, now - t0):5.1f} fps",
                              end="", flush=True)
                        last = now

                if args.seconds and time.time() - t0 > args.seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if fh:
            fh.close()

    dt = max(1e-6, time.time() - t0)
    print(f"\n\nframes        {n}   ({n / dt:.1f} fps)")
    print(f"parse errors  {n_err}")
    print(f"peak people   {peak}   ({len(ids)} distinct ids: "
          f"{sorted(ids) if len(ids) < 12 else str(len(ids)) + ' ids'})")
    print(f"occupied      {100.0 * occupied / max(1, n):.0f}% of frames")
    if args.save:
        print(f"saved         {args.save}")

    if n == 0:
        print("\nNo frames. The sensor is not sending — either it was never "
              "started (the profile must end in sensorStart), or the CLI and "
              "DATA ports are swapped. find_ports.py settles it.")
    elif peak == 0:
        print("\nFrames arrived but the tracker allocated nobody. Check that "
              "boundaryBox in the .cfg contains where you actually stood, and "
              "that sensorPosition matches the real mounting height and tilt — "
              "points outside the box are discarded before allocation.")


if __name__ == "__main__":
    main()
