#!/usr/bin/env python3
"""
TI's Industrial Visualizer, with adaptive config switching bolted on.

    conda activate tiviz
    python gui_adaptive.py
    python gui_adaptive.py --dry-run          # decide and log, never switch

Same GUI, same plots, same TI code. The only addition is that the switching
rule from adaptive.py watches every parsed frame and reconfigures the sensor
when the scene calls for it.

WHY A LAUNCHER RATHER THAN A PATCH. TI's tree gets replaced wholesale when the
Radar Toolbox is reinstalled or updated, so anything written into it is lost
silently. This file replicates what gui_main.py does -- build the QApplication,
construct Window, show it -- and then attaches the rule from outside. Nothing
in radar_toolbox is modified.

HOW THE HOOK WORKS. Core.updateGraph(outputDict) is called once per parsed
frame, and gui_core connects it to the UART thread only when the user clicks
Connect. Replacing the attribute at startup therefore gets picked up by that
later connect, and the original is still called first so the GUI behaves
exactly as before.

HOW A SWITCH IS PERFORMED. Core.selectCfg() cannot be used -- it opens a
QFileDialog and would prompt the operator every time. Core.parseCfg(path) takes
a filename directly and does everything needed: reads the file, sets self.cfg
and parser.cfg, and updates the demo's boundary-box drawing so the 3D view
redraws the new room. Then sendCfg() transmits it and restarts the parse timer.

    parseTimer.stop()  ->  parseCfg(path)  ->  sendCfg()

WHAT TO WATCH FOR. Every switch resets the group tracker, so all track ids in
the GUI change at that moment. Measured blackout on this board is ~1.8 s, during
which the plot is empty. Both are expected; see adaptive.py for why the rule is
debounced so heavily.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import adaptive as A                      # noqa: E402  (our rule)


def find_visualizer():
    """Locate Applications_Visualizer, newest toolbox first."""
    import glob
    pats = [
        "~/Developer/radar_toolbox_*/tools/visualizers/Applications_Visualizer",
        "~/Downloads/radar_toolbox_*/tools/visualizers/Applications_Visualizer",
        "~/ti/radar_toolbox_*/tools/visualizers/Applications_Visualizer",
        "/opt/ti/radar_toolbox_*/tools/visualizers/Applications_Visualizer",
    ]
    hits = []
    for p in pats:
        hits += glob.glob(os.path.expanduser(p))
    hits = [h for h in hits if os.path.isfile(os.path.join(h, "common", "gui_core.py"))]
    if not hits:
        sys.exit("Could not find Applications_Visualizer. Pass --visualizer.")
    return sorted(hits)[-1]


def tracks_from(outputDict):
    """
    TI's trackData is an (N,16) array: [id, x, y, z, vx, vy, vz, ...].

    numDetectedTracks is authoritative for how many rows are valid -- the array
    itself is allocated larger, so slicing by len() would feed the rule
    uninitialised rows.
    """
    n = int(outputDict.get("numDetectedTracks", 0) or 0)
    td = outputDict.get("trackData")
    if td is None or n <= 0:
        return []
    return [{"x": float(r[1]), "y": float(r[2])} for r in td[:n]]


def attach(core, close_cfg, far_cfg, rule, dry_run=False, log=None):
    """Wrap Core.updateGraph so the rule sees every frame."""
    paths = {A.CLOSE: close_cfg, A.FAR: far_cfg}
    original = core.updateGraph
    state = {"busy": False, "switches": 0, "last_reason": ""}

    def say(msg):
        line = f"[adaptive] {msg}"
        print(line, flush=True)
        if log:
            log.write(line + "\n")
            log.flush()

    def do_switch(target):
        path = paths[target]
        say(f"switching to {target.upper()}  {os.path.basename(path)}"
            f"  ({rule.reason})")
        t0 = time.time()
        state["busy"] = True
        try:
            core.parseTimer.stop()
            core.parseCfg(path)          # NOT selectCfg -- that opens a dialog
            core.sendCfg()               # sends and restarts the timer
            rule.committed(target)
            state["switches"] += 1
            say(f"blackout {time.time()-t0:.1f}s — all track ids have reset")
        except Exception as e:
            say(f"switch FAILED: {type(e).__name__}: {e}")
            say("the sensor may now be unconfigured; press Start and Send "
                "Configuration to recover")
        finally:
            state["busy"] = False

    def hooked(outputDict):
        original(outputDict)                     # GUI behaves exactly as before
        if state["busy"]:
            return
        try:
            decision = rule.update(tracks_from(outputDict))
        except Exception as e:
            say(f"rule error, switching disabled for this frame: {e}")
            return
        if not decision:
            return
        if dry_run:
            say(f"[dry-run] would switch to {decision.upper()} ({rule.reason})")
            rule.committed(decision)             # pretend, keep evaluating
            return
        do_switch(decision)

    core.updateGraph = hooked
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visualizer", help="path to Applications_Visualizer")
    ap.add_argument("--close-cfg",
                    default=os.path.join(HERE, A.DEFAULT_CFG[A.CLOSE]))
    ap.add_argument("--far-cfg",
                    default=os.path.join(HERE, A.DEFAULT_CFG[A.FAR]))
    ap.add_argument("--near-min", type=float, default=1.0)
    ap.add_argument("--near-metric", choices=["ground", "lateral"],
                    default="ground")
    ap.add_argument("--far-enter", type=float, default=6.0)
    ap.add_argument("--far-exit", type=float, default=5.0)
    ap.add_argument("--far-count", type=int, default=2)
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--dwell", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", help="append switch events to this file")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()

    if args.far_exit > args.far_enter:
        sys.exit("--far-exit must be <= --far-enter, or it will oscillate")

    # Resolve every user-supplied path against the ORIGINAL working directory,
    # before the chdir into TI's tree below. Otherwise a relative --log or
    # --close-cfg silently retargets into radar_toolbox, where it does not
    # exist -- which is exactly how this failed the first time it was run.
    cwd = os.getcwd()
    args.close_cfg = os.path.abspath(os.path.join(cwd, args.close_cfg))
    args.far_cfg = os.path.abspath(os.path.join(cwd, args.far_cfg))
    if args.log:
        args.log = os.path.abspath(os.path.join(cwd, args.log))
        d = os.path.dirname(args.log)
        if d:
            os.makedirs(d, exist_ok=True)

    for p in (args.close_cfg, args.far_cfg):
        if not os.path.exists(p):
            sys.exit(f"missing config: {p}")

    av = os.path.expanduser(args.visualizer) if args.visualizer else find_visualizer()
    common = os.path.join(av, "common")
    iv = os.path.join(av, "Industrial_Visualizer")
    if not os.path.isfile(os.path.join(common, "gui_core.py")):
        sys.exit(f"not an Applications_Visualizer directory: {av}\n"
                 f"  expected {os.path.join(common, 'gui_core.py')}")
    sys.path.insert(1, common)
    # gui_core loads images by relative path and cached_data writes a 'cache'
    # file into the working directory, so run from where TI expects to be.
    os.chdir(iv if os.path.isdir(iv) else av)

    try:
        from PySide2.QtCore import Qt
        from PySide2.QtWidgets import QApplication
        from PySide2.QtGui import QPalette, QColor
    except ImportError:
        sys.exit("PySide2 missing. This launcher needs TI's GUI environment:\n"
                 "  conda activate tiviz")

    from gui_core import Window
    from demo_defines import DEVICE_DEMO_DICT, BUSINESS_DEMOS

    # Same demo filtering gui_main.py applies for the Industrial build.
    for key in DEVICE_DEMO_DICT.keys():
        DEVICE_DEMO_DICT[key]["demos"] = [
            x for x in DEVICE_DEMO_DICT[key]["demos"]
            if x in BUSINESS_DEMOS["Industrial"]]

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    if args.dark:
        app.setStyle("Fusion")
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(53, 53, 53))
        pal.setColor(QPalette.WindowText, Qt.white)
        pal.setColor(QPalette.Base, QColor(25, 25, 25))
        pal.setColor(QPalette.Text, Qt.white)
        pal.setColor(QPalette.Button, QColor(53, 53, 53))
        pal.setColor(QPalette.ButtonText, Qt.white)
        app.setPalette(pal)

    screen = app.primaryScreen()
    win = Window(size=screen.size(), title="Industrial Visualizer — adaptive")
    win.show()

    metric = A.lateral if args.near_metric == "lateral" else A.ground_range
    rule = A.Rule(args.near_min, args.far_enter, args.far_exit, args.far_count,
                  args.hold, args.dwell, metric)
    log = open(args.log, "a") if args.log else None

    attach(win.core, args.close_cfg, args.far_cfg, rule,
           dry_run=args.dry_run, log=log)

    print(f"[adaptive] CLOSE {os.path.basename(args.close_cfg)}")
    print(f"[adaptive] FAR   {os.path.basename(args.far_cfg)}")
    print(f"[adaptive] near-min {args.near_min} ({args.near_metric})  "
          f"enter {args.far_enter}  exit {args.far_exit}  "
          f"count {args.far_count}  hold {args.hold}s  dwell {args.dwell}s")
    if args.dry_run:
        print("[adaptive] DRY RUN — decisions logged, sensor never reconfigured")
    print("[adaptive] load the CLOSE config in the GUI and press Start and "
          "Send Configuration to begin.\n")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
