#!/usr/bin/env python3
"""
Send a .cfg to the sensor and read back what it produces.

The demo firmware does nothing until it is configured. On boot it sits at its
prompt waiting; you push a chirp profile down the CLI port line by line, finish
with sensorStart, and it begins streaming binary frames out the DATA port.
Power-cycle the board and it forgets everything, so this runs every session.

    python3 stream.py --cfg profile.cfg
    python3 stream.py --cfg profile.cfg --save logs/run1.jsonl
    python3 stream.py --cfg profile.cfg --verbose      # print every target
    python3 stream.py --no-config                      # already running, listen

TWO DEMOS, ONE PARSER. The out-of-box demo emits a raw detected-point list and
nothing else. The 3D People Tracking demo runs a group tracker on-chip and
emits a TARGET LIST instead — id, position, velocity, acceleration, and the
tracker's own covariance — plus a compressed point cloud in SPHERICAL
coordinates and a per-point target assignment. Occupancy comes from the target
list; the point cloud is only useful for looking at what the tracker was given.
Both are handled here, because which one you get depends on which .bin is
flashed and it is easy to lose an afternoon to that.

WHY LINE BY LINE, WITH A WAIT. The CLI acknowledges each command with "Done"
and rejects the whole profile if it receives the next line before it has
finished the last. Blasting the file at it looks like it works — the port
accepts the bytes — and then the sensor never starts, with no error anywhere
obvious. So each line is sent, the reply is read, and anything that is not
"Done" stops the run and prints what the sensor actually said.

WHY THE HEADER LENGTH IS MEASURED, NOT ASSUMED. TI's frame header is 36 bytes
in some demos and 40 in others, and the TLV length field counts the 8-byte TLV
header in some builds and not in others. Four combinations, all of which parse
*something* from any given frame, three of which yield silent nonsense. So the
first frame is walked under every combination and the one whose TLV chain lands
exactly on the end of the packet with every type recognised wins. That is
decided once and reported, then reused.
"""

import argparse
import collections
import json
import math
import os
import struct
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial missing.  ./.venv/bin/pip install pyserial")


MAGIC = bytes([2, 1, 4, 3, 6, 5, 8, 7])

# ---- TLV types --------------------------------------------------------------
# Low numbers are the out-of-box demo. The 10xx block is the group tracker,
# which is what the People Tracking .bin emits.
TLV_DETECTED_POINTS = 1
TLV_RANGE_PROFILE = 2
TLV_NOISE_PROFILE = 3
TLV_AZIMUTH_HEATMAP = 4
TLV_RANGE_DOPPLER_HEATMAP = 5
TLV_STATS = 6
TLV_SIDE_INFO = 7
TLV_AZIMUTH_ELEV_HEATMAP = 8
TLV_TEMPERATURE = 9

TLV_TARGET_LIST = 1010          # the tracker's output: one entry per person
TLV_TARGET_INDEX = 1011         # which target each point was assigned to
TLV_TARGET_HEIGHT = 1012        # per target: max and min z
TLV_POINT_CLOUD_SPHERE = 1020   # compressed, spherical, with a unit block
TLV_PRESENCE = 1021             # a single uint32: is anyone in the zone

TLV_NAMES = {
    1: "points", 2: "range_profile", 3: "noise", 4: "azimuth_heatmap",
    5: "range_doppler", 6: "stats", 7: "side_info", 8: "az_el_heatmap",
    9: "temperature",
    1010: "targets", 1011: "target_index", 1012: "target_height",
    1020: "point_cloud", 1021: "presence",
}

# 4 + 27 floats: tid, pos xyz, vel xyz, acc xyz, 4x4 error covariance, g,
# confidence. Some builds drop the confidence field, giving 108.
TARGET_112 = struct.Struct("<I27f")
TARGET_108 = struct.Struct("<I26f")

POINT_UNIT = struct.Struct("<5f")        # elev, azim, doppler, range, snr scale
POINT_C = struct.Struct("<bbhHH")        # 8 bytes per compressed point


# =============================================================================
# configuration
# =============================================================================

def _diagnose_silence(raw):
    """
    Explain a non-ASCII reply from what is supposed to be a command line.

    Worth doing properly, because the three causes look identical from the
    terminal and each wastes a different afternoon. A reply that is all NULs or
    all 0xFF is a line-level fault — wrong baud, or nothing driving the wire.
    A reply full of high bytes is the DATA port, which streams binary and never
    answers anything. An empty reply is a port that exists but has nobody
    behind it, which after a flash almost always means the board never left the
    bootloader.
    """
    if not raw:
        return ("nothing came back at all",
                "The port opened but the demo is not running. After changing "
                "the SOP jumpers back to functional mode the board must be "
                "POWER-CYCLED — SOP is only sampled at reset, so flipping it "
                "while powered leaves the ROM bootloader in charge and it "
                "answers nothing. Unplug the USB, replug, try again.")
    uniq = set(raw)
    if uniq <= {0x00} or uniq <= {0xFF}:
        return (f"{len(raw)} bytes, all 0x{raw[0]:02x}",
                "A line stuck high or low, not a talking device. Either the "
                "baud is wrong, or this is the wrong half of the CP2105. "
                "Run find_ports.py — it probes both and reports which one "
                "answers the prompt.")
    printable = sum(1 for b in raw if 32 <= b < 127 or b in (10, 13))
    if printable < len(raw) * 0.6:
        return (f"{len(raw)} bytes, mostly non-text",
                "Two things look like this and they need opposite fixes.\n"
                "  (a) This is the DATA port, which streams binary frames and "
                "never answers commands — run find_ports.py to check, since "
                "the two CP2105 halves are not interchangeable.\n"
                "  (b) This IS the CLI, but the sensor is running and a "
                "low-power build gates the UART clock between frames, "
                "corrupting whatever is in flight. Power-cycle the board so it "
                "comes up stopped, then configure it before anything starts "
                "it. Repeated 0xF0/0x00 bytes point at this one.")
    return (repr(raw.decode("ascii", "ignore").strip()[:70]),
            "The port is talking but did not acknowledge. If it printed a "
            "banner, the demo may still be booting — wait a second and retry.")


def open_cli(port, baud=115200, timeout=0.6, drop_handshake=False):
    """
    Open the command port, ONCE, leaving the handshake lines alone.

    Opening a CP2105 asserts DTR and RTS, and on this board that glitches a
    garbage character onto the sensor's receive line. It sits in the demo's
    buffer until the next newline and then gets glued to the front of a real
    command, which the demo rejects — so a command you sent perfectly comes
    back as '""udfeDataOutputMode'. Every open costs another byte, hence
    opening once here instead of probing on a second handle.

    DO NOT "fix" that by deasserting DTR/RTS before opening. Measured on this
    board: with the lines at their defaults the demo answers the prompt every
    time; held low it goes completely silent, because deasserted RTS tells a
    device using hardware flow control not to transmit. The glitch character
    is harmless once clear_line() has flushed it, and that is the whole fix.
    drop_handshake exists only for a board that actually needs it.
    """
    s = serial.Serial()
    s.port = port
    s.baudrate = baud
    s.timeout = timeout
    if drop_handshake:
        s.dtr = False
        s.rts = False
    s.open()
    time.sleep(0.25)
    s.reset_input_buffer()
    return s


def clear_line(s, settle=0.25):
    """
    Send a bare newline to terminate whatever partial line the sensor is
    holding, and swallow the complaint it makes about it.

    This is the antidote to the glitch byte. The demo only parses on newline,
    so a stray character sits in its buffer indefinitely; giving it a newline
    makes it reject that fragment on its own and return to a clean prompt.
    """
    s.reset_input_buffer()
    s.write(b"\n")
    s.flush()
    time.sleep(settle)
    return s.read(s.in_waiting or 256)


def probe_cli(port, baud=115200):
    """Standalone check that a port is the CLI. Opens its own handle."""
    with open_cli(port, baud) as s:
        clear_line(s)                      # first newline flushes the junk
        raw = clear_line(s)                # second gets a clean prompt
    txt = raw.decode("ascii", "ignore")
    return ("mmwDemo" in txt or ">" in txt), raw


def _read_reply(s, timeout=1.2):
    """
    Read until the demo reprints its prompt, or we run out of patience.

    A fixed sleep was wrong: this firmware idles the device between frames and
    some commands take far longer than others to acknowledge, so a short sleep
    truncates a good reply into an apparent failure while a long one makes
    36 lines crawl. Waiting for the prompt costs exactly as long as it needs.
    """
    end = time.monotonic() + timeout
    buf = b""
    while time.monotonic() < end:
        chunk = s.read(s.in_waiting or 1)
        if chunk:
            buf += chunk
        txt = buf.decode("ascii", "ignore")
        if "mmwDemo" in txt or txt.rstrip().endswith(">"):
            break
    return buf


def reply_kind(raw):
    """
    Classify a reply. The category that was missing is 'garbage'.

    Treating unreadable bytes as a rejection is what made this blame the .cfg
    for a UART framing error. A reply of 0xF0 / 0x00 patterns is a line fault —
    the demo never saw the command and never said anything about it — and the
    right response is to send it again, not to give up and accuse the profile.
    """
    if not raw:
        return "silent"
    printable = sum(1 for b in raw if 32 <= b < 127 or b in (9, 10, 13))
    if printable < len(raw) * 0.7:
        return "garbage"
    txt = raw.decode("ascii", "ignore")
    if "Done" in txt or "Ignored" in txt:
        return "ok"
    if "not recognized" in txt:
        return "unknown"
    if "already stopped" in txt or "not running" in txt:
        return "ok"
    return "error"


def _send_line(s, line, attempts=3):
    """
    Send one command, retrying only when the LINE failed, never when the
    command was understood and refused.

    Three attempts, not one: a device that idles its UART will corrupt the
    occasional byte no matter how carefully you drive it, and losing a
    36-line profile to one bad character is not acceptable. But a readable
    rejection is a real answer and gets no retries at all, so a genuine
    firmware mismatch still fails on the first try.
    """
    for k in range(attempts):
        s.write((line + "\n").encode())
        s.flush()
        raw = _read_reply(s)
        kind = reply_kind(raw)
        if kind in ("ok", "unknown", "error"):
            return kind, raw.decode("ascii", "ignore"), raw, k
        clear_line(s, 0.2)                 # garbage or silence: resynchronise
    return kind, raw.decode("ascii", "ignore"), raw, attempts - 1


def stop_sensor(s, tries=6, verbose=True):
    """
    Get the sensor genuinely stopped before reconfiguring it.

    This matters more than it looks. A running Low Power build idles the
    device between frames, which stops the UART clock and corrupts whatever is
    in flight — so every command sent to a running sensor is a coin flip. Once
    it is stopped the line is quiet and reliable. Previously this was assumed
    rather than verified, and the whole session was then fighting corruption
    that a working sensorStop would have removed.
    """
    for k in range(tries):
        clear_line(s, 0.15)
        kind, reply, raw, _ = _send_line(s, "sensorStop", attempts=1)
        if kind in ("ok", "error"):
            # "error" here is almost always "not running", which is what we want.
            return True, kind
        time.sleep(0.25)
    if verbose:
        print("  sensorStop never acknowledged — the line stayed corrupted.")
    return False, kind


def send_config(port, path, baud=115200, verbose=True, check=True,
                drop_handshake=False):
    """Push the profile down the CLI, one line at a time, checking each reply."""
    with open(path) as fh:
        lines = [l.strip() for l in fh]
    lines = [l for l in lines if l and not l.startswith("%")]

    with open_cli(port, baud, drop_handshake=drop_handshake) as s:
        clear_line(s)                      # flush the open-glitch

        # STOP BEFORE PROBING. A running low-power build garbles its own
        # prompt, so checking for the prompt first diagnoses a perfectly good
        # CLI as the wrong port. sensorStop is the one command worth pushing
        # through the noise, because succeeding at it makes the noise stop.
        stopped, _ = stop_sensor(s, verbose=verbose)
        if verbose and stopped:
            print("sensor stopped")

        raw = clear_line(s)                # now the prompt should be clean
        if check:
            txt = raw.decode("ascii", "ignore")
            if "mmwDemo" not in txt and ">" not in txt:
                what, why = _diagnose_silence(raw)
                print(f"\n{port} does not answer like the demo's command line.")
                print(f"  sent: a bare newline")
                print(f"  got:  {what}\n")
                print(f"  {why}")
                print("\n  (--no-check sends the profile anyway.)")
                return False
            if verbose:
                print(f"CLI responding: {txt.strip().splitlines()[-1][:40]!r}\n")

        for i, line in enumerate(lines, 1):
            if line.split()[0] == "sensorStop":
                # Already handled above, and the profiles all open with it.
                if verbose:
                    print(f"   {i:3d}/{len(lines)}  {line[:52]}  (done above)")
                continue

            kind, reply, raw, retries = _send_line(s, line)

            if verbose:
                mark = "  " if kind == "ok" else "!!"
                note = f"  (retry {retries})" if retries and kind == "ok" else ""
                print(f"{mark} {i:3d}/{len(lines)}  {line[:52]}{note}")

            if kind == "ok":
                continue

            print(f"\nStopped at line {i}:\n  {line}")
            print(f"  reply: {reply.strip()!r}")
            print(f"  raw:   {raw[:48].hex(' ')}")

            if kind in ("garbage", "silent"):
                print("\nThat is not a rejection — it is a UART framing fault. "
                      "The demo never received a readable command, so nothing "
                      "is wrong with your .cfg.")
                print("  This build idles the device between frames, which "
                      "stops the UART clock. Power-cycle the board so it comes "
                      "up stopped and idle, then run this again before "
                      "anything else has configured it.")
            elif kind == "unknown":
                print("\nThe demo does not have this command at all — this is "
                      "the wrong .bin for this .cfg. A People Tracking profile "
                      "sent to the out-of-box demo dies on the first tracking "
                      "command; an out-of-box profile sent to People Tracking "
                      "dies on dfeDataOutputMode.")
            else:
                print("\nThe command exists but the arguments were refused: "
                      "same lab, wrong device. An AOP config carries "
                      "antGeometry/antPhaseRot lines that an ISK build will "
                      "not accept, and vice versa.")
            return False
    return True


# =============================================================================
# frame parsing
# =============================================================================

# Decided once from the first good frame, then reused: (header_len,
# tlv_len_includes_its_own_header).
_LAYOUT = None
_LAYOUT_TRIED = []


def _walk(body, hdr_len, inc, num_tlv, end):
    """
    Walk the TLV chain under one candidate interpretation.

    Returns (blocks, leftover) or None. A candidate only counts if every TLV
    fits inside the packet and every type is one we know — a wrong header
    length lands mid-payload and produces types in the millions, which is what
    makes this discriminating rather than a coin flip.
    """
    off = hdr_len
    out = []
    for _ in range(num_tlv):
        if off + 8 > end:
            return None
        t_type, t_len = struct.unpack_from("<II", body, off)
        payload = t_len - 8 if inc else t_len
        if payload < 0 or off + 8 + payload > end:
            return None
        if t_type not in TLV_NAMES:
            return None
        out.append((t_type, off + 8, off + 8 + payload))
        off = off + 8 + payload
    leftover = end - off          # the demo pads the packet to 32 bytes
    if leftover < 0 or leftover > 31:
        return None
    return out, leftover


def _detect_layout(body, total_len, end):
    """Try every combination and keep the one that lands cleanest."""
    best = None
    for hdr_len in (32, 28, 36):
        if end < hdr_len:
            continue
        fields = struct.unpack_from("<%dI" % (hdr_len // 4), body, 0)
        if fields[1] != total_len:
            continue
        num_tlv = fields[6]
        if not 0 < num_tlv <= 32:
            continue
        for inc in (False, True):
            got = _walk(body, hdr_len, inc, num_tlv, end)
            if got is None:
                continue
            _, leftover = got
            if best is None or leftover < best[2]:
                best = (hdr_len, inc, leftover)
    return best


def _spherical(payload, body):
    """
    Compressed point cloud: a 20-byte unit block, then 8 bytes per point.

    Everything is sent as small integers scaled by those units, which is how a
    few hundred points fit in a frame at 30 Hz. Angles arrive in RADIANS and
    the elevation is measured from the horizon, not from vertical, so y is the
    forward axis and z is up — matching the boundaryBox in the .cfg.
    """
    p0, p1 = payload
    if p1 - p0 < POINT_UNIT.size:
        return []
    eu, au, du, ru, su = POINT_UNIT.unpack_from(body, p0)
    off = p0 + POINT_UNIT.size
    pts = []
    while off + POINT_C.size <= p1:
        el, az, dop, rng, snr = POINT_C.unpack_from(body, off)
        off += POINT_C.size
        elev, azim = el * eu, az * au
        r = rng * ru
        pts.append({
            "x": round(r * math.cos(elev) * math.sin(azim), 3),
            "y": round(r * math.cos(elev) * math.cos(azim), 3),
            "z": round(r * math.sin(elev), 3),
            "v": round(dop * du, 3),
            "snr": round(snr * su, 1),
        })
    return pts


def parse_frame(body):
    """
    body starts immediately after the magic word.

    Returns (dict, consumed). consumed is measured from the start of body, so
    the caller drops magic + consumed. Returns (None, 0) if the frame is not
    yet complete, and (False, n) if it is complete but unparseable — the caller
    must skip those or it will stall on one bad packet forever.
    """
    global _LAYOUT
    if len(body) < 12:
        return None, 0
    total_len = struct.unpack_from("<I", body, 4)[0]
    if not 40 <= total_len <= 65536:
        return False, 1                       # garbage magic; step past it
    end = total_len - len(MAGIC)
    if len(body) < end:
        return None, 0                        # wait for the rest

    if _LAYOUT is None:
        found = _detect_layout(body, total_len, end)
        if found is None:
            _LAYOUT_TRIED.append(total_len)
            return False, end                 # complete but unreadable; drop it
        _LAYOUT = (found[0], found[1])

    hdr_len, inc = _LAYOUT
    fields = struct.unpack_from("<%dI" % (hdr_len // 4), body, 0)
    frame_no, num_obj, num_tlv = fields[3], fields[5], fields[6]

    got = _walk(body, hdr_len, inc, num_tlv, end)
    if got is None:
        # A single odd frame should not invalidate a layout that has been
        # working, so this drops the frame rather than re-detecting.
        return False, end
    blocks, _ = got

    out = {"frame": frame_no, "n_obj": num_obj, "tlvs": [],
           "points": [], "targets": [], "heights": {}, "assign": [],
           "presence": None}

    for t_type, p0, p1 in blocks:
        out["tlvs"].append(TLV_NAMES.get(t_type, str(t_type)))
        n = p1 - p0

        if t_type == TLV_DETECTED_POINTS:
            for k in range(n // 16):
                x, y, z, v = struct.unpack_from("<ffff", body, p0 + k * 16)
                out["points"].append({"x": round(x, 3), "y": round(y, 3),
                                      "z": round(z, 3), "v": round(v, 3)})

        elif t_type == TLV_POINT_CLOUD_SPHERE:
            out["points"] = _spherical((p0, p1), body)

        elif t_type == TLV_TARGET_LIST:
            st = TARGET_112 if n % TARGET_112.size == 0 else TARGET_108
            for k in range(n // st.size):
                f = st.unpack_from(body, p0 + k * st.size)
                out["targets"].append({
                    "id": f[0],
                    "x": round(f[1], 3), "y": round(f[2], 3), "z": round(f[3], 3),
                    "vx": round(f[4], 3), "vy": round(f[5], 3), "vz": round(f[6], 3),
                    "ax": round(f[7], 3), "ay": round(f[8], 3), "az": round(f[9], 3),
                    "conf": round(f[27], 3) if len(f) > 27 else None,
                })

        elif t_type == TLV_TARGET_HEIGHT:
            for k in range(n // 12):
                tid, maxz, minz = struct.unpack_from("<Iff", body, p0 + k * 12)
                out["heights"][str(tid)] = [round(minz, 3), round(maxz, 3)]

        elif t_type == TLV_TARGET_INDEX:
            # One byte per point. 253-255 mean noise or too weak to assign.
            out["assign"] = list(body[p0:p1])

        elif t_type == TLV_PRESENCE and n >= 4:
            out["presence"] = struct.unpack_from("<I", body, p0)[0]

    # The tracker reports height separately from position; fold it in so a
    # saved frame is self-contained.
    for t in out["targets"]:
        h = out["heights"].get(str(t["id"]))
        if h:
            t["z_min"], t["z_max"] = h
    return out, end


# =============================================================================
# main
# =============================================================================

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
    ap.add_argument("--baud", type=int, default=921600, help="DATA port baud")
    ap.add_argument("--cli-baud", type=int, default=115200)
    ap.add_argument("--verbose", action="store_true",
                    help="print every target every frame instead of a summary")
    ap.add_argument("--drop-handshake", action="store_true",
                    help="deassert DTR/RTS on open; on this board that SILENCES "
                         "the demo, so only for hardware that needs it")
    ap.add_argument("--no-check", action="store_true",
                    help="send the profile without first probing for the prompt")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.no_config:
        if not args.cfg:
            sys.exit("need --cfg <profile.cfg>, or --no-config if already running")
        if not os.path.exists(args.cfg):
            sys.exit(f"no such file: {args.cfg}")
        print(f"configuring from {args.cfg} ...")
        if not send_config(args.cli, args.cfg, args.cli_baud, not args.quiet,
                           check=not args.no_check,
                           drop_handshake=args.drop_handshake):
            sys.exit(1)
        print("configured; sensor started\n")

    fh = None
    if args.save:
        d = os.path.dirname(args.save)
        if d:
            os.makedirs(d, exist_ok=True)
        fh = open(args.save, "a")
    buf = b""
    n_frames = n_bad = n_points = n_targets = 0
    peak = 0
    seen_ids = set()
    tlv_seen = collections.Counter()
    t0 = time.time()
    last_report = t0
    announced = False

    print(f"reading {args.data} at {args.baud} — Ctrl-C to stop\n")
    try:
        with serial.Serial(args.data, args.baud, timeout=0.5) as s:
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
                    if frame is False:
                        n_bad += 1
                        continue

                    n_frames += 1
                    n_points += len(frame["points"])
                    n_targets += len(frame["targets"])
                    peak = max(peak, len(frame["targets"]))
                    seen_ids.update(t["id"] for t in frame["targets"])
                    tlv_seen.update(frame["tlvs"])

                    if not announced and _LAYOUT:
                        hdr, inc = _LAYOUT
                        kind = ("people tracking"
                                if "targets" in frame["tlvs"] else
                                "out-of-box point cloud")
                        print(f"layout: {hdr + 8}-byte header, TLV length "
                              f"{'includes' if inc else 'excludes'} its header"
                              f"   demo: {kind}\n")
                        announced = True

                    if fh:
                        fh.write(json.dumps(frame) + "\n")
                        fh.flush()

                    if args.verbose and frame["targets"]:
                        print(f"frame {frame['frame']}")
                        for t in frame["targets"]:
                            print(f"   id {t['id']:>3}  "
                                  f"x{t['x']:+6.2f} y{t['y']:+6.2f} z{t['z']:+6.2f} m   "
                                  f"v{math.hypot(t['vx'], t['vy']):5.2f} m/s   "
                                  f"h {t.get('z_min', 0):.2f}-{t.get('z_max', 0):.2f}")
                    elif not args.quiet:
                        now = time.time()
                        if now - last_report >= 0.5:
                            fps = n_frames / max(1e-6, now - t0)
                            ids = ",".join(str(t["id"]) for t in frame["targets"])
                            print(f"\rframe {frame['frame']:>7}   "
                                  f"people {len(frame['targets']):>2} [{ids:<11}] "
                                  f"points {len(frame['points']):>4}   "
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
    print(f"targets     {n_targets}  ({n_targets / max(1, n_frames):.2f} per frame, "
          f"peak {peak}, {len(seen_ids)} distinct ids)")
    if n_bad:
        print(f"dropped     {n_bad} unparseable frames")
    if tlv_seen:
        print("tlvs        " + ", ".join(f"{k} x{v}" for k, v in tlv_seen.most_common()))
    if args.save:
        print(f"saved       {args.save}")

    if n_frames == 0:
        print("\nNo frames.")
        if _LAYOUT_TRIED:
            print(f"  {len(_LAYOUT_TRIED)} packets arrived but none could be "
                  f"parsed under any header layout. The link is alive, so this "
                  f"is a format mismatch, not a wiring problem.")
        else:
            print("  Either the sensor was never started (the .cfg must end in "
                  "sensorStart), or the CLI and DATA ports are swapped — run "
                  "find_ports.py.")
    elif n_targets == 0 and peak == 0:
        print("\nFrames arrived but the tracker reported nobody. If you were in "
              "the room: check the .cfg's boundaryBox and staticBoundaryBox "
              "actually contain where you stood, and that sensorPosition "
              "matches the real mounting height and tilt — the tracker discards "
              "everything outside the box before it ever allocates a target.")


if __name__ == "__main__":
    main()
