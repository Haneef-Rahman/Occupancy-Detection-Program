#!/usr/bin/env python3
"""
Reset the sensor without touching the USB cable.

    ./.venv/bin/python power_cycle.py              # try everything, verify
    ./.venv/bin/python power_cycle.py --identify   # which handshake line is NRST?
    ./.venv/bin/python power_cycle.py --method cli
    ./.venv/bin/python power_cycle.py --method lines

WHY THIS EXISTS. A failed sensorStart wedges this board: the CLI stops
answering and stays dead until USB power is removed. That makes every config
experiment cost a physical unplug, which is fine once and intolerable when you
are bisecting a parameter.

TWO CANDIDATE MECHANISMS, and the script does not assume either works.

  cli     The flashed demo exposes a 'resetDevice' command (confirmed present
          in pcount3D_cli.c, "No arguments"). Costs nothing to try, but it
          needs a CLI that still answers -- so it is useless in the exact case
          we care about, a wedged board. Try it first anyway; it is the clean
          path when the sensor is merely running rather than hung.

  lines   Many TI EVMs wire the USB-UART bridge's DTR or RTS to NRST so that
          flashing tools can reset the board.

          MEASURED ON THIS EVM: they do not. --identify pulsed DTR alone, RTS
          alone, and both together; none produced a boot banner and the CLI
          kept answering throughout. So on the IWR6843AOPEVM these lines are
          neither NRST nor flow control -- driving them does nothing at all.
          Kept only so the test can be repeated on other hardware.

HOW WE KNOW IT ACTUALLY RESET. Not by the CLI answering -- a CLI that was
never dead also answers. The demo prints a boot banner:

    ***********************************
    IWR68xx Indoor people counting demo

Seeing that banner is proof the processor restarted. Nothing else is. So every
method here is judged on whether the banner appears, and --identify pulses each
line separately to find out which one, if either, is wired to reset.
"""

import argparse
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial missing.  ./.venv/bin/pip install pyserial")


BANNER_HINTS = (b"Indoor people counting", b"*****", b"mmwDemo")
BOOT_MARKERS = (b"Indoor people counting", b"*****")


def drain(s, seconds, want=BOOT_MARKERS):
    """Read for a while, returning everything and whether a boot marker showed."""
    end = time.monotonic() + seconds
    buf = b""
    while time.monotonic() < end:
        chunk = s.read(s.in_waiting or 1)
        if chunk:
            buf += chunk
            if any(m in buf for m in want):
                # keep reading briefly so the whole banner is captured
                end = min(end, time.monotonic() + 0.4)
    return buf, any(m in buf for m in want)


def alive(s, timeout=1.5):
    """Does the CLI answer a bare newline with its prompt?"""
    s.reset_input_buffer()
    s.write(b"\n")
    s.flush()
    buf, _ = drain(s, timeout, want=(b"mmwDemo",))
    return b"mmwDemo" in buf, buf


def method_cli(s, verbose=True):
    """Ask the firmware to reset itself."""
    ok, _ = alive(s)
    if not ok:
        if verbose:
            print("  cli    : CLI is not answering, so resetDevice cannot be "
                  "delivered. Skipping (this is the wedged case).")
        return False
    s.reset_input_buffer()
    s.write(b"resetDevice\n")
    s.flush()
    buf, booted = drain(s, 6.0)
    if verbose:
        print(f"  cli    : sent resetDevice, "
              f"{'BANNER SEEN' if booted else 'no banner'} "
              f"({len(buf)} bytes: {buf[:60]!r})")
    return booted


def pulse(s, dtr=None, rts=None, hold=0.6, verbose=True, label=""):
    """
    Drive one or both handshake lines low, hold, release, and watch for a boot.

    Held low is the assertion direction for these signals as pyserial exposes
    them, so False is 'asserted' at the connector on a board that wires them to
    NRST. The hold has to outlast the device's reset filter; 600 ms is well
    past anything reasonable.
    """
    was = (s.dtr, s.rts)
    try:
        if dtr is not None:
            s.dtr = dtr
        if rts is not None:
            s.rts = rts
        time.sleep(hold)
    finally:
        s.dtr, s.rts = was
    s.reset_input_buffer()
    buf, booted = drain(s, 6.0)
    if verbose:
        print(f"  {label:<7}: {'BANNER SEEN' if booted else 'no banner'} "
              f"({len(buf)} bytes: {buf[:60]!r})")
    return booted


def identify(port, baud):
    """
    Find out which handshake line, if any, is wired to NRST on this board.

    Worth running once. The answer is not in TI's documentation for this EVM
    and guessing it wrong cost real time earlier: holding both lines low made
    the demo silent, which I read as flow control when it may have been reset.
    """
    print(f"identify: pulsing each line on {port}\n")
    results = {}
    for label, kw in (("DTR", dict(dtr=False)),
                      ("RTS", dict(rts=False)),
                      ("both", dict(dtr=False, rts=False))):
        with serial.Serial(port, baud, timeout=0.4) as s:
            time.sleep(0.4)
            s.reset_input_buffer()
            results[label] = pulse(s, label=label, **kw)
            ok, _ = alive(s)
            print(f"           CLI answering afterwards: {ok}")
        time.sleep(1.0)

    print()
    hits = [k for k, v in results.items() if v]
    if hits:
        print(f"RESET LINE FOUND: {', '.join(hits)} produced a boot banner.")
        print("  Use --method lines from now on; no unplugging needed.")
    else:
        print("No line produced a boot banner. On this board DTR and RTS are")
        print("  almost certainly UART handshake only, not wired to NRST, so a")
        print("  software reset over the cable is not possible. Options left:")
        print("    - press the NRST button on the EVM (a reset, not a power")
        print("      cycle, so it will not clear a brown-out state)")
        print("    - unplug the USB")
        print("    - a USB hub with per-port power switching (uhubctl), which")
        print("      is a genuine power cycle and scriptable")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default="/dev/cu.usbserial-010821020")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--method", choices=["auto", "cli", "lines"], default="auto")
    ap.add_argument("--identify", action="store_true",
                    help="test each handshake line separately and report which "
                         "one (if any) is wired to NRST")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.identify:
        identify(args.cli, args.baud)
        return

    print(f"resetting via {args.cli}")
    booted = False
    with serial.Serial(args.cli, args.baud, timeout=0.4) as s:
        time.sleep(0.4)
        s.reset_input_buffer()

        if args.method in ("auto", "cli"):
            booted = method_cli(s, not args.quiet)

        if not booted and args.method in ("auto", "lines"):
            booted = pulse(s, dtr=False, rts=False, label="lines",
                           verbose=not args.quiet)

        ok, _ = alive(s)

    print()
    if booted:
        print("RESET CONFIRMED — boot banner seen. The sensor is unconfigured "
              "and idle, which is the state a config send wants.")
    elif ok:
        print("No boot banner, but the CLI is answering. The board was "
              "probably not wedged in the first place; nothing was reset. "
              "Safe to send a config.")
    else:
        print("No banner and the CLI is silent. Nothing here reached the "
              "board — unplug the USB, wait five seconds, replug.")
        print("Run --identify once to learn whether a handshake line can do "
              "this at all on your EVM.")
    sys.exit(0 if (booted or ok) else 1)


if __name__ == "__main__":
    main()
