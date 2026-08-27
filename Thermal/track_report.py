#!/usr/bin/env python3
"""
Read a track-events log and say WHY identities changed.

An id change is always an association failure — the detector produced a box and
the tracker declined to give it to an existing track. But there are four
different reasons for that, and they need four different fixes. Guessing which
one is happening wastes effort on the wrong one, so integrated_launcher.py
records the geometry at every track birth and this reads it back.

    genuine      nothing was near, alive or dead. A real new person.
                 -> not a bug. This is the tracker working.

    duplicate    a track sitting INSIDE the gate had already been matched this
                 frame — a second box on the same head, not a missed gate.
                 The first version of this report called these "gate_miss" and
                 recommended fixing the gate, which would have changed nothing.
                 -> FIX: provisional births (--dup-confirm), so a box must
                    persist before it earns an id.

    gate_miss    a live track was nearby but outside the association gate.
                 -> the gate is the problem: a fixed radius that ignores how
                    uncertain the filter actually is. FIX: Mahalanobis gate
                    from the Kalman covariance, so a confident track gets a
                    tight window and a coasting one gets a wide one.

    contested    two or more tracks were within reach of the same detection.
                 -> greedy association let the wrong one claim it first, and
                    the right one was left to spawn a new id. FIX: Hungarian
                    assignment, plus a size term in the cost so a near person
                    and a far one are not interchangeable.

    returning    a track died near here recently. The person left the frame,
                 was occluded, or the detector blinked, and came back a
                 stranger.
                 -> FIX: re-identification. Keep dead tracks in a buffer and
                    let a new detection reclaim its old id. This is the one
                    that matters for occupancy counting, because a returning
                    person must not count as an arrival.

    python3 track_report.py logs/integrated_20260827_143000_events.csv
"""

import collections
import csv
import sys


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().splitlines()[-1])
    rows = list(csv.DictReader(open(sys.argv[1])))
    births = [r for r in rows if r["event"] == "birth"]
    deaths = [r for r in rows if r["event"] == "death"]
    if not births:
        sys.exit("no track births in that log — nothing to diagnose")

    frames = max(int(r["frame"]) for r in rows)
    tally = collections.Counter(r["cause"] for r in births)

    print(f"{sys.argv[1]}")
    print(f"  frames            {frames}")
    print(f"  track births      {len(births)}")
    print(f"  track deaths      {len(deaths)}")
    print(f"  births per 100 f  {100.0 * len(births) / max(1, frames):.1f}")

    print(f"\nWHY IDS WERE CREATED")
    print(f"  {'cause':<12}{'n':>6}{'share':>8}   what it means")
    meaning = {
        "genuine":   "a real new person — not a bug",
        "duplicate": "2nd box on one head; track was already matched",
        "duplicate_promoted": "persisted -> probably a real 2nd person",
        "gate_miss": "live track nearby but outside the gate",
        "contested": "2+ tracks in reach; greedy picked wrong",
        "returning": "a track died here recently",
    }
    for k, v in tally.most_common():
        print(f"  {k:<12}{v:>6}{100.0 * v / len(births):>7.0f}%   {meaning.get(k,'')}")

    real = tally.get("genuine", 0) + tally.get("duplicate_promoted", 0)
    churn = len(births) - real
    print(f"\n  churn (all but 'genuine')  {churn}  "
          f"({100.0 * churn / len(births):.0f}% of births)")

    # ---- how far off were the misses? tells you how much gate to add -------
    gm = [float(r["near_dist"]) for r in births
          if r["cause"] == "gate_miss" and r["near_dist"]]
    if gm:
        gm.sort()
        print(f"\nGATE MISSES — distance to the track that should have claimed it")
        for p in (50, 75, 90, 100):
            i = min(len(gm) - 1, int(len(gm) * p / 100.0))
            print(f"  p{p:<4} {gm[i]:6.1f} px")
        print(f"  a gate this size would have caught them all: {gm[-1]:.0f} px")

    ms = [int(r["near_misses"]) for r in births
          if r["cause"] == "gate_miss" and r["near_misses"] not in ("", None)]
    if ms:
        c = collections.Counter(ms)
        print(f"  coast state of that track: " +
              ", ".join(f"{k} miss:{v}" for k, v in sorted(c.items())))

    # ---- returning: how long were they gone? sizes the re-id buffer --------
    rg = [(float(r["ghost_dist"]), int(r["ghost_age"])) for r in births
          if r["cause"] == "returning" and r["ghost_dist"] and r["ghost_age"]]
    if rg:
        d = sorted(x[0] for x in rg)
        a = sorted(x[1] for x in rg)
        print(f"\nRETURNING — a dead track was near")
        print(f"  distance   median {d[len(d)//2]:.1f} px   max {d[-1]:.1f} px")
        print(f"  gone for   median {a[len(a)//2]} frames   max {a[-1]} frames")
        print(f"  a re-id buffer of {a[-1]} frames would cover every case here")

    # ---- what died, and had it earned an identity? ------------------------
    if deaths:
        short = sum(1 for r in deaths if "hits1" in r["cause"] or "hits2" in r["cause"])
        print(f"\nDEATHS")
        print(f"  total          {len(deaths)}")
        print(f"  short-lived    {short}  (<=2 hits — never really established)")

    # ---- the verdict ------------------------------------------------------
    print(f"\n{'=' * 62}")
    if churn == 0:
        print("No churn. Every id was a genuine new person.")
    else:
        top = max((k for k in tally if k != "genuine"),
                  key=lambda k: tally[k], default=None)
        fix = {"duplicate": "lower --dedup-iou / raise --dup-confirm",
               "duplicate_promoted": "nothing — these look like real people",
               "gate_miss": "Mahalanobis gate from the Kalman covariance "
                            "(replaces the fixed --gate radius)",
               "contested": "Hungarian assignment + a size term in the cost",
               "returning": "re-identification buffer for dead tracks"}
        print(f"Dominant cause: {top}  ({tally[top]} of {len(births)} births)")
        print(f"Fix worth building first: {fix.get(top, '?')}")
    print("=" * 62)


if __name__ == "__main__":
    main()
