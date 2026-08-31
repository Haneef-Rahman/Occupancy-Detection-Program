#!/usr/bin/env python3
"""
Switch radar configs at runtime based on where the targets actually are.

    ./.venv/bin/python adaptive.py --save logs/adaptive.jsonl
    ./.venv/bin/python adaptive.py --dry-run        # decide, log, never switch
    ./.venv/bin/python adaptive.py --seconds 600

THE IDEA. Close range and long range want different radar configs and no single
profile is good at both:

    CLOSE   AOP_6m_staticRetention   8 m box, fineMotionCfg on. Holds a person
                                     who sits still. Measured 99% occupied at
                                     desk distance.
    FAR     AOP_9m_sensitive         12 m box, lower CFAR thresholds, no static
                                     retention. Finds distant and weak targets
                                     that the 6 m profile never reports.

So run CLOSE by default and go FAR only when the scene actually needs it.

THE RULE, as specified:

    CLOSE -> FAR   every target is at least --near-min from the sensor
                   AND at least 2 targets are beyond --far-enter
    FAR -> CLOSE   no target beyond --far-exit

Both conditions are observable from within the config that has to evaluate
them, which is not automatic and is worth checking before inventing a rule:
the 6 m profile's boundaryBox reaches 8 m, so "someone is past 6 m" is visible
while CLOSE is running. A rule that could only be evaluated from the config you
are trying to switch INTO would be unimplementable.

WHY --near-min EXISTS. The 9 m profile starts its boundaryBox at 0.5 m and its
staticBoundaryBox at 2 m, so a person standing on top of the sensor is outside
the region it detects in. Measured at 0.65 m, the 9 m configs reported 0-15%
occupancy while the 6 m config reported 99%. Switching to FAR while anyone is
that close trades a tracked person for a distant one.

THREE THINGS THAT MAKE THIS HARDER THAN IT SOUNDS.

1. A SWITCH IS A BLACKOUT. Reconfiguring means sensorStop, flushCfg, ~30 config
   lines, then sensorStart -- which runs boot calibration. Measured, that is
   several seconds during which the radar reports nothing at all. The blackout
   is logged for every switch so it can be budgeted rather than guessed at.

2. EVERY RADAR ID RESETS. A new config restarts the group tracker, so track ids
   are meaningless across a switch. This is not a detail to paper over: it is
   why identity has to live in the FUSION layer, above the radar, carried by
   the thermal track through the blackout and re-associated by position
   afterwards. A "switch" record is written into the log at the moment it
   happens so nothing downstream mistakes a new id for a new person.

3. OSCILLATION IS THE FAILURE MODE. A person loitering at exactly the threshold
   would otherwise switch repeatedly, and each switch costs seconds of
   blindness plus a full identity reset -- far worse than simply running the
   wrong config. Three defences, all on by default:
     * separate enter/exit thresholds (--far-enter 6.0, --far-exit 5.0)
     * the condition must hold continuously for --hold seconds
     * a minimum --dwell in a config before any switch is considered
"""

import argparse
import collections
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ti_track as T


CLOSE = "close"
FAR = "far"

DEFAULT_CFG = {
    CLOSE: "configs/AOP_6m_staticRetention.cfg",
    FAR: "configs/AOP_9m_sensitive.cfg",
}


def ground_range(t):
    """Distance in the horizontal plane. 'How far away' in the everyday sense."""
    return math.hypot(t["x"], t["y"])


def lateral(t):
    """Sideways offset only. The other reading of 'horizontally'."""
    return abs(t["x"])


class Rule:
    """
    Decide which config the scene wants, with hysteresis and debouncing.

    Kept separate from the serial and parsing code so it can be exercised
    against a recorded log (--replay) without any hardware. A switching policy
    that has only ever been tested live is a policy nobody can tune.
    """

    def __init__(self, near_min=1.0, far_enter=6.0, far_exit=5.0,
                 far_count=2, hold=3.0, dwell=30.0, metric=ground_range):
        self.near_min = near_min
        self.far_enter = far_enter
        self.far_exit = far_exit
        self.far_count = far_count
        self.hold = hold
        self.dwell = dwell
        self.metric = metric
        self.mode = CLOSE
        self.wants = None
        self.reason = "startup"
        # Clocks start UNSET and are seeded by the first update(). Seeding them
        # from time.monotonic() here silently breaks every caller that injects
        # its own time -- the tests and --replay both count from 0, so a dwell
        # check against seconds-since-boot never passes and no switch is ever
        # possible. Found by the tests; the replay path was broken too.
        self.since_switch = None
        self.wants_since = None

    def want(self, tracks):
        """What the CURRENT frame argues for, ignoring timing."""
        if self.mode == CLOSE:
            if not tracks:
                return CLOSE, "no targets"
            rs = [self.metric(t) for t in tracks]
            nearest = min(rs)
            n_far = sum(1 for r in rs if r > self.far_enter)
            if nearest < self.near_min:
                return CLOSE, f"nearest {nearest:.1f} m inside {self.near_min:.1f} m"
            if n_far >= self.far_count:
                return FAR, f"{n_far} targets beyond {self.far_enter:.1f} m"
            return CLOSE, f"{n_far} beyond {self.far_enter:.1f} m, need {self.far_count}"
        else:
            if not tracks:
                return CLOSE, "no targets"
            rs = [self.metric(t) for t in tracks]
            if max(rs) <= self.far_exit:
                return CLOSE, f"all within {self.far_exit:.1f} m"
            if min(rs) < self.near_min:
                return CLOSE, f"nearest {min(rs):.1f} m inside {self.near_min:.1f} m"
            return FAR, f"farthest {max(rs):.1f} m"

    def update(self, tracks, now=None):
        """
        Returns the config to switch to, or None.

        The debounce is deliberately on the WANT, not on the switch: a
        condition that flickers never accumulates hold time, so a person
        walking back and forth across the threshold cannot drive a switch no
        matter how many times they cross it.
        """
        now = now if now is not None else time.monotonic()
        if self.since_switch is None:
            self.since_switch = now
        want, why = self.want(tracks)
        if want != self.wants or self.wants_since is None:
            self.wants, self.wants_since = want, now
        self.reason = why
        if want == self.mode:
            return None
        if now - self.wants_since < self.hold:
            return None
        if now - self.since_switch < self.dwell:
            return None
        return want

    def committed(self, mode, now=None):
        self.mode = mode
        self.since_switch = now if now is not None else time.monotonic()
        self.wants, self.wants_since = None, self.since_switch


def run(args):
    parseFrame = T.load_ti_parser(T.find_ti_common(args.ti_common),
                                  verbose=not args.quiet)
    import serial
    import stream as S

    metric = lateral if args.near_metric == "lateral" else ground_range
    rule = Rule(args.near_min, args.far_enter, args.far_exit, args.far_count,
                args.hold, args.dwell, metric)

    cfgs = {CLOSE: args.close_cfg, FAR: args.far_cfg}
    for m, p in cfgs.items():
        if not os.path.exists(p):
            sys.exit(f"missing {m} config: {p}")

    fh = None
    if args.save:
        d = os.path.dirname(args.save)
        if d:
            os.makedirs(d, exist_ok=True)
        fh = open(args.save, "a")

    def emit(rec):
        if fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    t0 = time.time()
    switches = 0
    blackouts = []
    per_mode = collections.Counter()
    mode = CLOSE

    try:
        while True:
            # ---- (re)configure ------------------------------------------
            t_cfg = time.time()
            if not args.quiet:
                print(f"\n=== {mode.upper()}  {os.path.basename(cfgs[mode])} ===")
            ok = S.send_config(args.cli, cfgs[mode], args.cli_baud,
                               verbose=args.verbose_cfg)
            if not ok:
                emit({"event": "config_failed", "mode": mode, "t": time.time() - t0})
                sys.exit(f"could not apply {cfgs[mode]}")
            blackout = time.time() - t_cfg
            blackouts.append(blackout)
            rule.committed(mode)
            emit({"event": "switch", "to": mode, "cfg": cfgs[mode],
                  "blackout_s": round(blackout, 2), "t": round(time.time() - t0, 2),
                  "note": "all radar track ids reset here"})
            if not args.quiet:
                print(f"    blackout {blackout:.1f}s — radar ids reset\n")

            # ---- read until the rule says otherwise ---------------------
            buf = b""
            want_switch = None
            with serial.Serial(args.data, args.baud, timeout=0.4) as s:
                last = 0.0
                while want_switch is None:
                    chunk = s.read(4096)
                    if chunk:
                        buf += chunk
                        if len(buf) > (1 << 20):
                            buf = buf[-65536:]
                    got, upto = T.frames_from(buf, parseFrame.parseStandardFrame)
                    buf = buf[upto:]

                    for f in got:
                        rec = T.summarise(f)
                        rec["mode"] = mode
                        per_mode[mode] += 1
                        emit(rec)
                        decision = rule.update(rec["tracks"])
                        if decision and not args.dry_run:
                            want_switch = decision
                        elif decision and args.dry_run:
                            emit({"event": "would_switch", "to": decision,
                                  "why": rule.reason,
                                  "t": round(time.time() - t0, 2)})
                            if not args.quiet:
                                print(f"\n  [dry-run] would switch to "
                                      f"{decision.upper()} — {rule.reason}")
                            rule.committed(mode)   # pretend, keep evaluating

                    if got and not args.quiet:
                        now = time.time()
                        if now - last >= 0.5:
                            r = T.summarise(got[-1])
                            rs = [metric(t) for t in r["tracks"]]
                            span = (f"{min(rs):.1f}-{max(rs):.1f} m" if rs else "-")
                            print(f"\r  {mode:<5} people {len(r['tracks'])} "
                                  f"range {span:<12} {rule.reason:<38}",
                                  end="", flush=True)
                            last = now

                    if args.seconds and time.time() - t0 > args.seconds:
                        want_switch = "__stop__"

            if want_switch == "__stop__":
                break
            switches += 1
            mode = want_switch
            if not args.quiet:
                print(f"\n  -> switching to {mode.upper()}: {rule.reason}")

    except KeyboardInterrupt:
        pass
    finally:
        if fh:
            fh.close()

    dt = max(1e-6, time.time() - t0)
    print(f"\n\nran           {dt:.0f}s")
    print(f"switches      {switches}"
          + (f"   (one every {dt/switches:.0f}s)" if switches else ""))
    if blackouts:
        print(f"blackout      {sum(blackouts):.1f}s total, "
              f"{sum(blackouts)/dt*100:.1f}% of the session, "
              f"mean {sum(blackouts)/len(blackouts):.1f}s per switch")
    for m, n in per_mode.items():
        print(f"frames {m:<6}{n}")
    if args.save:
        print(f"saved         {args.save}")


def replay(args):
    """
    Run the rule over a recorded log. No hardware, no switching.

    This is how thresholds get tuned: capture a real session with ti_track.py,
    then see how often THIS policy would have switched, before letting it cost
    real blackouts.
    """
    metric = lateral if args.near_metric == "lateral" else ground_range
    rule = Rule(args.near_min, args.far_enter, args.far_exit, args.far_count,
                args.hold, args.dwell, metric)
    rows = [json.loads(l) for l in open(args.replay) if l.strip()]
    rows = [r for r in rows if "tracks" in r]
    dt = 1.0 / max(1e-6, args.fps)
    t = 0.0
    events = []
    mode_frames = collections.Counter()
    for r in rows:
        mode_frames[rule.mode] += 1
        d = rule.update(r["tracks"], now=t)
        if d:
            events.append((t, rule.mode, d, rule.reason))
            rule.committed(d, now=t)
        t += dt

    print(f"{args.replay}: {len(rows)} frames, {t:.0f}s at {args.fps} fps")
    print(f"thresholds: near-min {args.near_min}  far-enter {args.far_enter}  "
          f"far-exit {args.far_exit}  count {args.far_count}  "
          f"hold {args.hold}s  dwell {args.dwell}s\n")
    if not events:
        print("no switches — the policy would have stayed in CLOSE throughout.")
    for tt, a, b, why in events:
        print(f"  {tt:7.1f}s  {a:>5} -> {b:<5}  {why}")
    print(f"\nswitches {len(events)}"
          + (f"  (one every {t/len(events):.0f}s)" if events else ""))
    for m, n in mode_frames.items():
        print(f"  {m:<6}{100.0*n/max(1,len(rows)):.0f}% of frames")
    if events:
        cost = len(events) * args.assume_blackout
        print(f"\nat {args.assume_blackout:.0f}s per switch that is {cost:.0f}s "
              f"({100.0*cost/max(1e-6,t):.0f}% of the session) blind.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default="/dev/cu.usbserial-010821020")
    ap.add_argument("--data", default="/dev/cu.usbserial-010821021")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--cli-baud", type=int, default=115200)
    ap.add_argument("--ti-common")
    ap.add_argument("--close-cfg", default=DEFAULT_CFG[CLOSE])
    ap.add_argument("--far-cfg", default=DEFAULT_CFG[FAR])

    ap.add_argument("--near-min", type=float, default=1.0,
                    help="never go FAR while a target is closer than this")
    ap.add_argument("--near-metric", choices=["ground", "lateral"],
                    default="ground",
                    help="'horizontally' as ground range (default) or as "
                         "sideways offset only")
    ap.add_argument("--far-enter", type=float, default=6.0)
    ap.add_argument("--far-exit", type=float, default=5.0,
                    help="lower than --far-enter on purpose: hysteresis")
    ap.add_argument("--far-count", type=int, default=2,
                    help="how many targets must be beyond --far-enter")
    ap.add_argument("--hold", type=float, default=3.0,
                    help="seconds the condition must hold before switching")
    ap.add_argument("--dwell", type=float, default=30.0,
                    help="minimum seconds in a config before switching again")

    ap.add_argument("--save")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="evaluate and log decisions, never actually switch")
    ap.add_argument("--replay", help="run the rule over a saved .jsonl instead")
    ap.add_argument("--fps", type=float, default=16.7, help="for --replay timing")
    ap.add_argument("--assume-blackout", type=float, default=2.0,
                    help="seconds per switch, for the --replay cost estimate")
    ap.add_argument("--verbose-cfg", action="store_true",
                    help="print every config line as it is sent")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.far_exit > args.far_enter:
        sys.exit("--far-exit must be <= --far-enter, or it will oscillate")

    if args.replay:
        replay(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
