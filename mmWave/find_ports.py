#!/usr/bin/env python3
"""
Find the IWR6843AOPEVM's two serial ports and work out which is which.

The board presents a Silicon Labs CP2105 dual UART, so it appears as TWO
devices. They are not interchangeable and nothing in the name tells you which
is which:

    CLI  / Enhanced   115200 baud   you send the .cfg here
    DATA / Standard   921600 baud   the point cloud comes back here

On Windows they are labelled "Enhanced" and "Standard". On macOS you get two
/dev/tty.usbserial-* nodes distinguished only by a trailing digit, and which
one is the CLI is NOT guaranteed to be the lower number. So this probes them
instead of guessing: the CLI port answers a bare newline with the demo's
"mmwDemo:/>" prompt, and the data port does not.

    python3 find_ports.py
    python3 find_ports.py --all      # list every serial port, not just TI ones

Probing is safe. A newline on the CLI is a no-op that just re-prints the
prompt, and the data port is only read from.
"""

import argparse
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial missing.  pip install pyserial")


# CP2105 as fitted to the AOP EVM. Matching on VID/PID rather than on the
# device name keeps this working across macOS versions, which have changed
# the /dev naming more than once.
SILABS_VID = 0x10C4
CP2105_PID = 0xEA70


def candidates(show_all=False):
    out = []
    for p in list_ports.comports():
        ti = (p.vid == SILABS_VID and p.pid == CP2105_PID)
        if ti or show_all:
            out.append((p, ti))
    return out


def probe_cli(dev, timeout=1.2):
    """
    Does this port answer like the demo's command line?

    A bare newline makes the CLI reprint its prompt. Anything that replies
    with 'mmwDemo' is the config port. The data port stays silent or emits
    binary, so the test separates them without needing to send a real command.
    """
    try:
        with serial.Serial(dev, 115200, timeout=timeout) as s:
            s.reset_input_buffer()
            s.write(b"\n")
            s.flush()
            time.sleep(0.4)
            data = s.read(s.in_waiting or 64)
        txt = data.decode("ascii", "ignore")
        return ("mmwDemo" in txt or ">" in txt), txt.strip()[:60]
    except Exception as e:
        return False, f"error: {e}"


def looks_like_data(dev, timeout=1.5):
    """
    The demo streams frames continuously once configured, each beginning with
    a fixed 8-byte magic word. Seeing it proves both that this is the data
    port AND that the sensor is already running.
    """
    MAGIC = bytes([2, 1, 4, 3, 6, 5, 8, 7])
    try:
        with serial.Serial(dev, 921600, timeout=timeout) as s:
            time.sleep(0.6)
            buf = s.read(s.in_waiting or 4096)
        return (MAGIC in buf), len(buf)
    except Exception:
        return False, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="list every serial port, not just the CP2105 pair")
    ap.add_argument("--no-probe", action="store_true",
                    help="just list; do not open the ports")
    args = ap.parse_args()

    found = candidates(args.all)
    if not found:
        print("No CP2105 found.\n")
        print("  1. is the board plugged in and powered?")
        print("  2. macOS needs the Silicon Labs CP210x VCP driver on older")
        print("     releases; recent macOS includes it. If nothing appears at")
        print("     all, install it from silabs.com and reboot.")
        print("  3. run with --all to see every serial device present.")
        return

    print(f"{len(found)} port(s):\n")
    cli = data = None
    for p, is_ti in found:
        tag = "CP2105" if is_ti else "other"
        print(f"  {p.device}")
        print(f"      {tag}  vid:pid {p.vid:04x}:{p.pid:04x}" if p.vid else
              f"      {tag}")
        print(f"      {p.description}")
        if args.no_probe or not is_ti:
            print()
            continue

        ok, reply = probe_cli(p.device)
        if ok:
            cli = p.device
            print(f"      -> CLI PORT   (answered: {reply!r})")
        else:
            streaming, n = looks_like_data(p.device)
            if streaming:
                data = p.device
                print(f"      -> DATA PORT  (magic word seen, {n} bytes waiting)")
            else:
                data = data or p.device
                print(f"      -> probably DATA  (silent; sensor not started yet)")
        print()

    print("-" * 58)
    if cli:
        print(f"CLI  (send .cfg here, 115200):  {cli}")
    else:
        print("CLI  not identified — no port answered the prompt.")
        print("     If the board was just flashed, check the SOP switch is back")
        print("     in FUNCTIONAL mode and power-cycle. In flash mode the demo")
        print("     never runs, so nothing answers.")
    if data:
        print(f"DATA (read here, 921600):       {data}")
    print("-" * 58)


if __name__ == "__main__":
    main()
