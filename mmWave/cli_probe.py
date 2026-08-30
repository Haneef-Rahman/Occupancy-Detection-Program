#!/usr/bin/env python3
"""
Minimal CLI probe. Opens the port EXACTLY the way TI's visualiser does.

    ./.venv/bin/python cli_probe.py                 # TI's exact open
    ./.venv/bin/python cli_probe.py --mine          # the way stream.py opens it

The point is to bisect. TI's gui_parser.connectComPorts succeeds on this board
while stream.py's send_config gets zero bytes, minutes apart, with the ports
free. Either the board is intermittent or my open differs from theirs in a way
that matters. This strips away every bit of my logic -- no clearing newlines,
no retries, no reply classification -- and leaves only the two open calls, so
whichever answer comes back is unambiguous.

TI, verbatim from gui_parser.py:
    serial.Serial(port, baud, parity=serial.PARITY_NONE,
                  stopbits=serial.STOPBITS_ONE, timeout=0.6)
then plain readline() per command, no retry, no prompt check.
"""

import argparse
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial missing.  ./.venv/bin/pip install pyserial")


def open_ti(port, baud):
    """Exactly TI's call. Constructor form, opens immediately."""
    return serial.Serial(port, baud, parity=serial.PARITY_NONE,
                         stopbits=serial.STOPBITS_ONE, timeout=0.6)


def open_mine(port, baud):
    """stream.py's form: build, configure, then open."""
    s = serial.Serial()
    s.port = port
    s.baudrate = baud
    s.timeout = 0.6
    s.open()
    time.sleep(0.25)
    s.reset_input_buffer()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbserial-010821020")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--mine", action="store_true",
                    help="use stream.py's open instead of TI's")
    ap.add_argument("--both-ports", action="store_true",
                    help="also open the DATA port first, as TI's GUI does -- "
                         "it opens both in one call and we never have")
    ap.add_argument("--data", default="/dev/cu.usbserial-010821021")
    ap.add_argument("-n", type=int, default=6, help="how many probes")
    args = ap.parse_args()

    held = None
    if args.both_ports:
        held = serial.Serial(args.data, 921600, parity=serial.PARITY_NONE,
                             stopbits=serial.STOPBITS_ONE, timeout=0.6)
        print(f"holding DATA open: {args.data}")

    opener = open_mine if args.mine else open_ti
    print(f"opening {args.port} @ {args.baud} using "
          f"{'stream.py' if args.mine else 'TI'} style")
    s = opener(args.port, args.baud)
    print(f"open ok. dtr={s.dtr} rts={s.rts} cts={s.cts} dsr={s.dsr}")

    got = 0
    for i in range(args.n):
        cmd = b"\n" if i % 2 == 0 else b"sensorStop\n"
        s.write(cmd)
        s.flush()
        line = s.readline()          # TI's exact read
        got += len(line)
        print(f"  {i}: sent {cmd!r:14} -> {len(line):3d} bytes  {line[:60]!r}")
        time.sleep(0.2)

    s.close()
    if held:
        held.close()

    print()
    if got:
        print("CLI IS ANSWERING. The board is fine; the fault is in "
              "stream.py's send path, not the hardware.")
    else:
        print("Zero bytes across every probe. Not a code-path difference -- "
              "the board is not talking to anyone right now. Power-cycle "
              "(unplug, five seconds, replug) and run this again FIRST, "
              "before any other program touches the port.")


if __name__ == "__main__":
    main()
