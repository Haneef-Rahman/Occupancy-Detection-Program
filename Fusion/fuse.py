#!/usr/bin/env python3
"""
Thermal + mmWave fusion. This is the entry point for the whole pipeline.

    python fuse.py --self-test
    python fuse.py --live --adaptive
    python fuse.py --live --pitch -2.0 --assist-alpha 0.4

ASSEMBLED FROM LIBRARIES, not reimplemented:

    integrated_launcher   camera, omega CNN, Kalman tracking, occlusion
                          memory, re-identification, live tuning, the UI
    Fusion.radar_link     radar thread, adaptive config switching
    Fusion.fuse (here)    association and the thermal-authority protocol
    mmWave.project        radar -> camera-plane geometry
    mmWave.adaptive       the switching rule

Anything this parser does not recognise is forwarded verbatim to
integrated_launcher, so every flag that file accepts stays reachable.

WHAT IT DOES. Radar tracks are projected into the Lepton's image plane, matched
to the thermal YOLO boxes by IoU, and the two are bound to a single FUSED id
that outlives both. From the radar side that id carries a 3D position and
velocity; from the thermal side it survives the radar losing a person who stops
moving.

WHY IDENTITY LIVES HERE AND NOT IN EITHER SENSOR. Both sensors' ids are
short-lived handles:

  * TI's group tracker REUSES ids after a track is freed, so a person who drops
    out and returns comes back as someone else, and their old id may later
    belong to a different person.
  * Reconfiguring the radar restarts the tracker and resets every id.
  * The thermal Kalman tracker churns ids of its own under occlusion.

So neither is an identity. The fused track owns one, and holds
(thermal_id, radar_id) as associations that may each go stale independently.

THE DIVISION OF LABOUR, which is the whole reason to fuse these two:

    stationary person   radar loses them, thermal sees them trivially
    warm clutter        thermal false-positives forever, radar sees nothing
    static clutter      radar retains furniture, thermal sees nothing warm
    darkness, glare     both fine

Measured on this hardware: a radiator is a permanent thermal false positive and
invisible to radar; an oscillating fan is a dense moving radar target with no
thermal signature; and with fineMotionCfg enabled the radar held four
motionless "targets" in a workspace for 30 s each. The failure modes do not
overlap, which is what makes requiring agreement worth more than either sensor
alone.

COASTING A THERMAL-ONLY TRACK IS NEARLY EXACT, for a pleasing reason. When the
radar drops someone BECAUSE they stopped moving, their range has not changed --
that is the same fact that caused the dropout. So the fused track keeps the
last measured range and takes fresh bearing from the thermal box, which the
camera measures precisely. Range from radar, bearing from camera, each from the
sensor that is good at it.
"""

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "mmWave"))
sys.path.insert(0, os.path.join(ROOT, "Thermal"))

import project as P                                     # noqa: E402


# ---------------------------------------------------------------------------
# association
# ---------------------------------------------------------------------------

def centre(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def associate(radar, thermal, iou_min=0.15, max_centre_px=25.0):
    """
    Match projected radar boxes to thermal boxes.

    Greedy on IoU, highest first, then a centre-distance pass for the pairs
    that overlap too little to score but are obviously the same person.

    THE SECOND PASS IS NOT OPTIONAL. At 0.78 deg per pixel a person at 5 m is
    about 25 px tall, the projected box jitters a few px from the radar's own
    angular noise alone, and IoU falls off fast: 2 px of offset takes a good
    match from 1.0 to 0.57, 4 px to 0.29. A pure IoU gate would drop true
    pairs at exactly the range this system is for. IoU decides WHICH match is
    best; distance decides whether a weak one is still real.

    Returns (matches, radar_unmatched, thermal_unmatched) where matches is a
    list of (radar_index, thermal_index, iou, centre_distance).
    """
    pairs = []
    for i, r in enumerate(radar):
        for j, t in enumerate(thermal):
            v = P.iou(r["box"], t["box"])
            rc, tc = centre(r["box"]), centre(t["box"])
            d = math.dist(rc, tc)
            if v >= iou_min or d <= max_centre_px:
                pairs.append((v, -d, i, j))
    pairs.sort(reverse=True)

    used_r, used_t, out = set(), set(), []
    for v, negd, i, j in pairs:
        if i in used_r or j in used_t:
            continue
        used_r.add(i)
        used_t.add(j)
        out.append((i, j, round(v, 3), round(-negd, 1)))
    return (out,
            [i for i in range(len(radar)) if i not in used_r],
            [j for j in range(len(thermal)) if j not in used_t])


def bearing_of(u, cam):
    """Image column -> azimuth in radians. The camera's precise measurement."""
    return math.atan2((u - cam.cx) / cam.fx, 1.0)


# ---------------------------------------------------------------------------
# fused tracks
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# THE ADJUDICATION PROTOCOL: thermal is the authority on what is a person.
#
# Radar answers "is something there, where, and how fast" superbly and "is it a
# person" not at all. It cannot distinguish a person from a chair, a fan or a
# multipath ghost -- measured on this hardware, with fineMotionCfg enabled it
# held four motionless "targets" in a workspace for up to 30 s each. Thermal
# answers "is this a warm human-shaped thing" and is the only sensor here with
# any claim to that judgement.
#
# So: thermal confirmation is REQUIRED before a track counts as a person, and
# thermal denial overrides the radar.
#
# WITH ONE QUALIFIER, which matters. Denial only counts where thermal could
# actually see. The radar's field of view is +-70 deg; the Lepton's is 95 deg
# total, and it cannot see through a desk. A radar track outside the camera's
# frustum has not been rejected by thermal -- it has not been examined. Those
# are held as UNSEEN rather than dismissed, so the protocol never silently
# deletes a person the camera was never pointed at.
# ---------------------------------------------------------------------------

CONFIRMED = "confirmed"     # thermal has vouched for this, now or recently
THERMAL_ONLY = "thermal"    # thermal sees them, radar lost them (stationary)
UNCONFIRMED = "unconfirmed"  # in thermal's view, thermal says no -> NOT a person
UNSEEN = "unseen"           # radar-only, outside thermal's view -> unadjudicated
CLUTTER = "clutter"         # unconfirmed AND motionless -> furniture
RADAR_ONLY = "radar"        # legacy state, used when --no-thermal-veto


class FusedTrack:
    _next = 0

    def __init__(self, box, radar=None, thermal_id=None):
        FusedTrack._next += 1
        self.id = FusedTrack._next
        self.box = list(box)
        self.radar_id = radar["id"] if radar else None
        self.thermal_id = thermal_id
        self.pos = list(radar["pos"]) if radar else None
        self.vel = list(radar["vel"]) if radar else None
        self.range_m = radar["range_m"] if radar else None
        self.state = CONFIRMED if (radar and thermal_id is not None) else (
            RADAR_ONLY if radar else THERMAL_ONLY)
        self.age = 1
        self.hits_both = 1 if self.state == CONFIRMED else 0
        self.misses = 0
        self.no_radar = 0
        self.no_thermal = 0
        self.origin = tuple(self.pos[:2]) if self.pos else None
        self.max_move = 0.0
        self.in_fov = bool(radar["in_view"]) if radar else True
        self.ever_confirmed = (self.state == CONFIRMED)

    def _moved(self):
        if self.pos and self.origin:
            self.max_move = max(self.max_move,
                                math.dist(self.origin, tuple(self.pos[:2])))

    def update(self, box=None, radar=None, thermal_id=None, cam=None):
        self.age += 1
        self.misses = 0
        if box is not None:
            self.box = list(box)
        if radar is not None:
            self.in_fov = bool(radar["in_view"])
            self.radar_id = radar["id"]
            self.pos = list(radar["pos"])
            self.vel = list(radar["vel"])
            self.range_m = radar["range_m"]
            self.no_radar = 0
            self._moved()
        else:
            self.no_radar += 1
        if thermal_id is not None:
            self.thermal_id = thermal_id
            self.no_thermal = 0
        else:
            self.no_thermal += 1

        if radar is not None and thermal_id is not None:
            self.state = CONFIRMED
            self.hits_both += 1
            self.ever_confirmed = True
        elif thermal_id is not None:
            self.state = THERMAL_ONLY
            # Coast in 3D: keep the last measured range (unchanged, because
            # standing still is why the radar lost them) and take fresh bearing
            # from the thermal box, which the camera measures well.
            if self.range_m is not None and cam is not None and box is not None:
                u = (box[0] + box[2]) / 2.0
                az = bearing_of(u, cam)
                self.pos = [self.range_m * math.sin(az),
                            self.range_m * math.cos(az),
                            self.pos[2] if self.pos else 0.0]
                self.vel = [0.0, 0.0, 0.0]
        else:
            self.state = RADAR_ONLY

    def adjudicate(self, ghost_frames, ghost_move, grace, veto=True):
        """
        Apply the protocol. Called once per frame, after update().

        Order matters: a track that thermal has vouched for keeps the benefit
        of the doubt for `grace` frames, because the thermal tracker blinking
        during an FFC shutter or a brief occlusion is not a verdict. Only after
        that does the veto bite.
        """
        if not veto:
            return self.mark_clutter(ghost_frames, ghost_move)

        if self.state in (CONFIRMED, THERMAL_ONLY):
            return False

        # radar-only from here on
        if not self.in_fov:
            self.state = UNSEEN          # never examined; not a denial
            return False
        if self.ever_confirmed and self.no_thermal <= grace:
            self.state = CONFIRMED       # blink, not a verdict
            return False

        # thermal could see it and did not call it a person
        if self.max_move < ghost_move and self.no_thermal >= ghost_frames:
            self.state = CLUTTER
        else:
            self.state = UNCONFIRMED
        return True

    def mark_clutter(self, ghost_frames, ghost_move):
        """
        A radar-only track that has never been confirmed and has not gone
        anywhere is furniture, not a person.

        This is the specific failure measured in a real workspace: with
        fineMotionCfg on, four separate 'targets' were held for 8-30 s each
        while moving less than 0.6 m. Static retention retains anything static,
        and only the thermal channel can tell a chair from a person sitting in
        one.
        """
        if (self.state == RADAR_ONLY and self.hits_both == 0
                and self.no_thermal >= ghost_frames
                and self.max_move < ghost_move):
            self.state = CLUTTER
        return self.state == CLUTTER

    def speed(self):
        return math.hypot(self.vel[0], self.vel[1]) if self.vel else 0.0


class Fusion:
    def __init__(self, cam, iou_min=0.15, max_centre_px=25.0,
                 max_misses=8, ghost_frames=60, ghost_move=0.6,
                 thermal_veto=True, grace=9):
        self.cam = cam
        self.iou_min = iou_min
        self.max_centre_px = max_centre_px
        self.max_misses = max_misses
        self.ghost_frames = ghost_frames
        self.ghost_move = ghost_move
        self.thermal_veto = thermal_veto
        self.grace = grace
        self.tracks = []
        self.last = {"matches": [], "radar_only": 0, "thermal_only": 0}

    def step(self, radar_boxes, thermal_boxes):
        """
        radar_boxes  : from project.project_track
        thermal_boxes: [{"box":[x1,y1,x2,y2], "id":<kalman id>}, ...]
        """
        matches, r_un, t_un = associate(radar_boxes, thermal_boxes,
                                        self.iou_min, self.max_centre_px)
        claimed = set()

        # existing tracks first, by whichever handle still points at them
        by_thermal = {t.thermal_id: t for t in self.tracks
                      if t.thermal_id is not None}
        by_radar = {t.radar_id: t for t in self.tracks if t.radar_id is not None}

        for (ri, tj, v, d) in matches:
            r, t = radar_boxes[ri], thermal_boxes[tj]
            trk = by_thermal.get(t["id"]) or by_radar.get(r["id"])
            if trk is None or trk in claimed:
                trk = FusedTrack(t["box"], r, t["id"])
                self.tracks.append(trk)
            else:
                trk.update(t["box"], r, t["id"], self.cam)
            claimed.add(trk)

        for tj in t_un:
            t = thermal_boxes[tj]
            trk = by_thermal.get(t["id"])
            if trk is None or trk in claimed:
                trk = FusedTrack(t["box"], None, t["id"])
                self.tracks.append(trk)
            else:
                trk.update(t["box"], None, t["id"], self.cam)
            claimed.add(trk)

        for ri in r_un:
            r = radar_boxes[ri]
            trk = by_radar.get(r["id"])
            if trk is None or trk in claimed:
                trk = FusedTrack(r["box"], r, None)
                self.tracks.append(trk)
            else:
                trk.update(r["box"], r, None, self.cam)
            claimed.add(trk)

        for trk in self.tracks:
            if trk not in claimed:
                trk.misses += 1
            trk.adjudicate(self.ghost_frames, self.ghost_move,
                           self.grace, self.thermal_veto)

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        self.last = {"matches": matches,
                     "radar_only": len(r_un), "thermal_only": len(t_un)}
        return self.tracks

    def people(self):
        """
        Who counts as a person.

        Under the protocol, thermal confirmation is required: a track thermal
        has never vouched for is not a person, however convincing the radar
        finds it. UNSEEN tracks -- radar-only and outside the camera's view --
        are excluded from the count too, because nothing has judged them; they
        are reported separately rather than silently promoted or discarded.
        """
        if not self.thermal_veto:
            return [t for t in self.tracks if t.state != CLUTTER]
        return [t for t in self.tracks
                if t.ever_confirmed and t.state in (CONFIRMED, THERMAL_ONLY)]

    def unseen(self):
        """Radar tracks outside thermal's view: unadjudicated, not rejected."""
        return [t for t in self.tracks if t.state == UNSEEN]

    def rejected(self):
        """Thermal looked and said no."""
        return [t for t in self.tracks if t.state in (UNCONFIRMED, CLUTTER)]


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------

def self_test():
    ok = True

    def check(label, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {label} {extra}")

    cam = P.Camera()
    ext = P.Extrinsics(tz=-0.08)          # camera 8 cm below the radar

    def rbox(tid, x, y, z=0.9):
        t = {"id": tid, "x": x, "y": y, "z": z, "z_min": 0.0, "z_max": 1.75,
             "vx": 0.0, "vy": 0.0, "vz": 0.0}
        return P.project_track(t, cam, ext, clip=False)

    print("association")
    r = [rbox(3, 0.0, 5.0)]
    t = [{"box": list(r[0]["box"]), "id": 11}]
    m, ru, tu = associate(r, t)
    check("perfect overlap matches", m and m[0][2] > 0.95, m)

    off = list(r[0]["box"])
    off[0] += 3; off[2] += 3
    m, ru, tu = associate(r, [{"box": off, "id": 11}])
    check("3 px offset still matches via distance", len(m) == 1,
          f"iou {m[0][2] if m else '-'}")

    far = [b + 60 for b in r[0]["box"]]
    m, ru, tu = associate(r, [{"box": far, "id": 11}])
    check("60 px away does NOT match", not m and ru == [0] and tu == [0])

    print("\ntwo people are not swapped")
    r2 = [rbox(1, -1.2, 5.0), rbox(2, 1.2, 5.0)]
    t2 = [{"box": r2[1]["box"], "id": 21}, {"box": r2[0]["box"], "id": 22}]
    m, _, _ = associate(r2, t2)
    pair = {(i, j) for i, j, _, _ in m}
    check("nearest-first pairing correct", pair == {(0, 1), (1, 0)}, pair)

    print("\nfused identity")
    F = Fusion(cam)
    F.step(r2, t2)
    ids = sorted((x.thermal_id, x.radar_id) for x in F.tracks)
    check("two fused tracks bound to both handles", len(F.tracks) == 2, ids)
    fid = {t.thermal_id: t.id for t in F.tracks}

    # radar drops the LEFT person (they stopped moving); thermal keeps them
    for _ in range(5):
        F.step([r2[1]], t2)
    left = [t for t in F.tracks if t.thermal_id == 22][0]
    check("fused id survives radar dropout", left.id == fid[22],
          f"id {left.id}")
    check("state becomes thermal-only", left.state == THERMAL_ONLY, left.state)
    check("range retained through the dropout",
          abs(left.range_m - 5.14) < 0.3, f"{left.range_m}")
    check("velocity zeroed while coasting", left.speed() == 0.0)

    print("\nprotocol: thermal is the authority")
    F2 = Fusion(cam, ghost_frames=10, ghost_move=0.6)
    chair = rbox(9, 0.2, 3.0)
    for _ in range(15):
        F2.step([chair], [])
    check("radar-only, in view, never confirmed -> not a person",
          len(F2.people()) == 0, F2.tracks[0].state)
    check("and it is reported as rejected, not lost",
          len(F2.rejected()) == 1)
    check("motionless one is called clutter",
          F2.tracks[0].state == CLUTTER, F2.tracks[0].state)

    F3 = Fusion(cam, ghost_frames=10, ghost_move=0.6)
    for k in range(15):
        F3.step([rbox(9, 0.2, 3.0 + 0.15 * k)], [])
    check("a WALKING radar-only target is still not a person",
          len(F3.people()) == 0, F3.tracks[0].state)
    check("but it is UNCONFIRMED, not clutter",
          F3.tracks[0].state == UNCONFIRMED, F3.tracks[0].state)

    F4 = Fusion(cam, ghost_frames=10, ghost_move=0.6)
    for _ in range(15):
        rb = rbox(9, 0.2, 3.0)
        F4.step([rb], [{"box": rb["box"], "id": 5}])
    check("motionless but thermally confirmed IS a person",
          len(F4.people()) == 1 and F4.tracks[0].state == CONFIRMED)

    print("\noutside the camera: not examined, not denied")
    F5 = Fusion(cam, ghost_frames=10, ghost_move=0.6)
    for _ in range(15):
        F5.step([rbox(9, 9.0, 1.0)], [])          # ~84 deg off axis
    check("radar track outside thermal FoV -> UNSEEN",
          F5.tracks[0].state == UNSEEN, F5.tracks[0].state)
    check("not counted as a person", len(F5.people()) == 0)
    check("and NOT counted as rejected either", len(F5.rejected()) == 0)
    check("reported separately", len(F5.unseen()) == 1)

    print("\na thermal blink is not a verdict")
    F6 = Fusion(cam, ghost_frames=10, ghost_move=0.6, grace=9)
    rb = rbox(9, 0.0, 4.0)
    for _ in range(10):
        F6.step([rb], [{"box": rb["box"], "id": 7}])
    for _ in range(5):
        F6.step([rb], [])                          # thermal drops out briefly
    check("still a person during grace", len(F6.people()) == 1,
          F6.tracks[0].state)
    for _ in range(12):
        F6.step([rb], [])                          # gone well past grace
    check("demoted once the blink outlasts grace", len(F6.people()) == 0,
          F6.tracks[0].state)

    print("\nveto can be switched off")
    F7 = Fusion(cam, ghost_frames=10, ghost_move=0.6, thermal_veto=False)
    for k in range(15):
        F7.step([rbox(9, 0.2, 3.0 + 0.15 * k)], [])
    check("--no-thermal-veto counts a moving radar-only track",
          len(F7.people()) == 1, F7.tracks[0].state)

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------

def live(args, extra):
    """
    Run the real pipeline by DELEGATING, not reimplementing.

    An earlier version of this function opened its own camera, ran its own
    YOLO call, built its own tracker and drew its own window. That duplicated
    integrated_launcher.py badly: no occlusion memory, no re-identification,
    no duplicate suppression, no live tuning, and a YOLO call that fed raw
    temperatures to a model trained on a fixed 15-45 C span.

    So the pipeline is assembled from libraries instead:

        integrated_launcher   camera, CNN, Kalman tracking, occlusion, the UI
        Fusion.radar_link     radar thread, adaptive config switching
        Fusion.fuse           association and the thermal-authority protocol
        mmWave.project        radar -> camera-plane geometry

    integrated_launcher.main() takes an argv, so this builds one and hands
    over. Every flag that file accepts stays reachable: anything this parser
    does not recognise is forwarded verbatim.
    """
    import integrated_launcher as IL

    argv = [
        "--weights", args.weights,
        "--mode", args.mode,
        "--conf", str(args.conf),
        "--radar",
        "--radar-cli", args.cli,
        "--radar-data", args.data,
        "--radar-close-cfg", args.close_cfg,
        "--radar-far-cfg", args.far_cfg,
        "--radar-hfov", str(args.hfov),
        "--radar-tz", str(args.tz),
        "--radar-pitch", str(args.pitch),
        "--radar-yaw", str(args.yaw),
        "--radar-iou", str(args.iou_min),
        "--radar-centre-px", str(args.max_centre_px),
        "--assist-alpha", str(args.assist_alpha),
    ]
    if args.box:
        argv += ["--box", args.box]
    if args.adaptive:
        argv += ["--radar-adaptive"]
    if args.no_thermal_veto:
        argv += ["--no-thermal-veto"]
    if not args.arrow:
        argv += ["--no-radar-arrow"]
    if not args.vectors:
        argv += ["--no-radar-vectors"]
    if not args.velocity_assist:
        argv += ["--no-velocity-assist"]
    argv += extra

    print("FLUXNET fusion")
    print(f"  thermal   {args.weights}  mode={args.mode}")
    print(f"  radar     {os.path.basename(args.close_cfg)}"
          + (f" <-> {os.path.basename(args.far_cfg)}" if args.adaptive else "")
          + f"   adaptive={args.adaptive}")
    print(f"  protocol  thermal is the authority on what is a person")
    print(f"  motion    Doppler drives velocity (alpha {args.assist_alpha}), "
          f"thermal keeps identity\n")
    IL.main(argv)


def main():
    ap = argparse.ArgumentParser(
        description="FLUXNET thermal + mmWave fusion. Unrecognised arguments "
                    "are forwarded to integrated_launcher.py.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--live", action="store_true")

    ap.add_argument("--weights", default="../Thermal/models/v2/best.pt")
    ap.add_argument("--mode", choices=("yolo", "hybrid"), default="yolo",
                    help="yolo: omega CNN every frame, Kalman for identity "
                         "only, no classical detection")
    ap.add_argument("--box", choices=("body", "omega", "both"), default="omega")
    ap.add_argument("--conf", type=float, default=0.374)

    ap.add_argument("--cli", default="/dev/cu.usbserial-010821020")
    ap.add_argument("--data", default="/dev/cu.usbserial-010821021")
    ap.add_argument("--close-cfg",
                    default="../mmWave/configs/AOP_6m_staticRetention.cfg")
    ap.add_argument("--far-cfg",
                    default="../mmWave/configs/AOP_9m_sensitive.cfg")
    ap.add_argument("--adaptive", action="store_true",
                    help="switch radar configs at runtime by target geometry")

    ap.add_argument("--hfov", type=float, default=95.0)
    ap.add_argument("--tz", type=float, default=-0.08,
                    help="camera offset relative to the radar, metres "
                         "(negative: the Lepton sits below the AOP)")
    ap.add_argument("--pitch", type=float, default=0.0,
                    help="nudge until projected boxes sit on the thermal ones")
    ap.add_argument("--yaw", type=float, default=0.0)

    ap.add_argument("--iou-min", type=float, default=0.15)
    ap.add_argument("--max-centre-px", type=float, default=25.0)
    ap.add_argument("--assist-alpha", type=float, default=0.6,
                    help="how much of the Kalman velocity comes from Doppler")
    ap.add_argument("--no-arrow", dest="arrow", action="store_false")
    ap.add_argument("--no-vectors", dest="vectors", action="store_false")
    ap.add_argument("--no-velocity-assist", dest="velocity_assist",
                    action="store_false")
    ap.add_argument("--no-thermal-veto", action="store_true",
                    help="count radar-only tracks as people (pre-protocol)")
    args, extra = ap.parse_known_args()

    if args.self_test:
        sys.exit(self_test())
    if args.live:
        for p in (args.weights, args.close_cfg):
            if not os.path.exists(p):
                sys.exit(f"missing: {p}")
        live(args, extra)
    else:
        ap.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
