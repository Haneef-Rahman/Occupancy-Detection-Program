"""Precise K: extracted legacy StickyTracker for sparse scenes.

Source: integrated_launcher.py, repository commit 5ac36979a.
Legacy heuristics and historical comments are retained, not newly validated.
Do not treat legacy timing/accuracy comments as prototype benchmark evidence.
"""
import numpy as np
from tracker import MultiTracker, KalmanTrack

def dedup(dets, iou_thresh=0.15, centre_frac=1.4):
    """
    Collapse multiple omega boxes that describe the same head.

    Ultralytics runs NMS at IoU 0.7 by default, which is deliberately permissive
    so that genuinely adjacent objects both survive. On a 160x120 thermal frame
    an omega is 4-15 px across, and two boxes on the same head routinely overlap
    at 0.4-0.6 — under the NMS threshold, so both come through. The val_batch
    previews show this directly: labels reading "omeomega0.8" are two boxes
    stacked on one person.

    Downstream that is not a cosmetic problem. One box matches the existing
    track, the other is unmatched, and an unmatched detection creates a NEW
    track with a NEW id and a new colour. Next frame the duplicate lands
    elsewhere and the ids swap. That is the colour churn.

    Two boxes are merged if they overlap past `iou_thresh` OR if one's centre
    sits within `centre_frac` of the other's half-size — the second test catches
    a small box nested inside a larger one, where IoU stays low but they are
    plainly the same head.

    Thresholds swept on a simulated walk with 30% duplicate boxes. 0.35/0.6
    (the obvious first guess) left 8 distinct ids over 300 frames; 0.15/1.4
    leaves 4. Going further to 0.10 gives 3, but starts merging detections only
    8 px apart — 0.49 m at 2 m range, which is two people standing shoulder to
    shoulder, so that is where the tuning stops. At 0.15/1.4 the closest pair
    that survives as two is 6 px, or 0.37 m: tighter than people stand.
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i].get("conf", 0.0))
    keep = []
    for i in order:
        a = dets[i]
        ax, ay, aw, ah = a["bbox"]
        dup = False
        for j in keep:
            b = dets[j]
            bx, by, bw, bh = b["bbox"]
            ix0, iy0 = max(ax, bx), max(ay, by)
            ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            union = aw * ah + bw * bh - inter
            if union > 0 and inter / union >= iou_thresh:
                dup = True
                break
            acx, acy = a["centroid"]
            bcx, bcy = b["centroid"]
            if (abs(acx - bcx) <= centre_frac * max(aw, bw) / 2 and
                    abs(acy - bcy) <= centre_frac * max(ah, bh) / 2):
                dup = True
                break
        if not dup:
            keep.append(i)
    return [dets[i] for i in sorted(keep)]


class PreciseK(MultiTracker):
    """
    MultiTracker plus a second association pass for coasting tracks.

    The base class matches every detection against every track with one gate,
    then spawns a new track for whatever is left. That is fine while detections
    are continuous, and wrong the moment one drops: the track starts coasting
    and drifts under constant velocity, so when the detection returns it can sit
    outside the gate. A new track is created, the person changes id and colour,
    and the old track lingers for max_misses frames as a ghost.

    So: match live tracks first with the normal gate, then give still-unmatched
    detections a second chance against COASTING tracks with a gate widened in
    proportion to how long they have coasted — a track that has predicted
    forward for 5 frames has 5 frames of accumulated uncertainty and deserves a
    correspondingly wider window. Only what survives both passes is new.
    """

    def __init__(self, *a, revive_scale=1.8, ghost_frames=40,
                 dup_confirm=3, reid_px=0.0, exit_margin=12.0,
                 occl_timeout=350, occl_clear=6, frame_wh=(160, 120),
                 occl_enabled=True, use_ledger=True, ledger_decay=600,
                 exit_edges=("left", "right", "top", "bottom"),
                 exit_zones=(), **kw):
        super().__init__(*a, **kw)
        self.revive_scale = revive_scale
        self.n_new = 0
        self.n_revived = 0
        self.n_suppressed = 0
        self.n_reid = 0
        # DIAGNOSTIC ONLY. Dead tracks are remembered but never re-attached —
        # the point is to find out how often a "new" person is really a
        # returning one, before deciding whether re-identification is worth
        # building. Turning this into a fix would be a different change.
        self.ghost_frames = ghost_frames
        self.dup_confirm = dup_confirm
        self.reid_px = reid_px
        # ---- occlusion memory -------------------------------------------
        # A person cannot evaporate from the middle of a room. They enter and
        # leave through the frame boundary, so WHERE a track dies is evidence
        # about WHY. Dying near an edge is consistent with walking out; dying
        # in the interior is not, and means the view was blocked.
        # A separate switch, not exit_margin=0. Setting the margin to zero
        # makes at_edge() almost never true, so EVERY death looks interior and
        # occlusion memory runs at full strength — the exact opposite of off.
        self.occl_enabled = occl_enabled
        self.exit_margin = exit_margin
        self.exit_edges = set(exit_edges)
        self.exit_zones = list(exit_zones)
        self.n_crouch = 0
        self.no_crouch_veto = False
        self.occl_timeout = occl_timeout
        self.occl_clear = occl_clear
        # Only applied where the scene has never hidden anyone. Much more
        # patient than the old value, because empty floor is weak evidence.
        self.occl_clear_hard = occl_clear * 5
        self.occl_map = {}
        self.occl_learn = 1
        self.use_ledger = use_ledger
        self.ledger_decay = ledger_decay
        self.ledger = 0
        self.n_entries = 0
        self.n_exits = 0
        self.n_decayed = 0
        self.surplus_since = 0
        self.frame_wh = frame_wh
        self.occluded = []        # people known present but not currently seen
        self.n_occluded = 0
        self.n_recovered = 0
        self.n_exited = 0
        self.n_gaveup = 0
        self.ghosts = []          # (id, x, y, w, h, frame_died, hits)
        self.pending = []         # provisional births: [x, y, count, frame]
        self.events = []          # rows for track_events.csv
        self.frame = 0

    def at_edge(self, x, y):
        """
        Is this position outside the room, as the sensor defines the room?

        The field of view IS the room here. Anything that crosses out of it has
        stopped being observable, and an occupancy count over an unobservable
        space is a guess. So a track ending at the boundary is an exit
        regardless of how it got there.

        --exit-edges exists for the unusual install where one border must not
        count — a view that overruns into an adjoining space, say — but the
        default is all four, because the ordinary case is that the frame is
        the whole of what this sensor is responsible for.
        """
        W, H = self.frame_wh
        m = self.exit_margin
        if "left" in self.exit_edges and x <= m:
            return True
        if "right" in self.exit_edges and x >= W - m:
            return True
        if "top" in self.exit_edges and y <= m:
            return True
        if "bottom" in self.exit_edges and y >= H - m:
            return True
        for zx0, zy0, zx1, zy1 in self.exit_zones:
            if zx0 - m <= x <= zx1 + m and zy0 - m <= y <= zy1 + m:
                return True
        return False

    @staticmethod
    def loss_kind(t, n=8, grow_max=1.15, min_move=1.2):
        """
        WHICH WAY was this person going when we lost them?

        Direction beats position. A rule based on where a track ends needs to
        know which borders are doors and which are walls, and no amount of
        thermal data reveals that. But how someone was MOVING when they
        vanished is measured, not configured, and it separates the two cases
        directly:

            sideways   they walked out of the room. Whether that happened at
                       the frame border or in the middle of it does not matter
                       — lateral travel ending in nothing means gone.
            downward   they crouched, sat, or dropped behind furniture. They
                       are still here.

        Downward is qualified by scale. Walking toward the camera also carries
        the box down the frame, but a nearer head subtends more pixels, so it
        GROWS. A head that sinks while holding its size ducked; one that sinks
        while growing walked into the lens and out of frame.

        Returns "exit", "hidden", or "unclear" — the last when the track simply
        stopped without going anywhere, which is a detector dropout and gets
        decided by position instead.
        """
        hist = t.history[-n:] if len(t.history) >= 3 else []
        hh = getattr(t, "_hh", [])[-n:]
        if len(hist) < 3:
            return "unclear"
        dx = hist[-1][0] - hist[0][0]
        dy = hist[-1][1] - hist[0][1]
        adx, ady = abs(dx), abs(dy)
        if max(adx, ady) < min_move:
            return "unclear"                 # stood still and blinked out
        grow = 1.0
        if len(hh) >= 3:
            h0 = max(1e-6, sum(hh[:2]) / 2.0)
            grow = (sum(hh[-2:]) / 2.0) / h0
        if ady > adx and dy > 0:
            # went down: hidden unless the growth says they came at the camera
            return "exit" if grow >= grow_max else "hidden"
        if adx >= ady:
            return "exit"                    # travelled sideways and stopped
        return "unclear"                     # went up — walked away into depth

    @staticmethod
    def crouch_signature(t, n=8, grow_max=1.15):
        """
        Did this track go DOWN, or did it walk TOWARDS the camera and out?

        Both end with the box drifting down the frame, so vertical motion alone
        cannot separate them — a first attempt used it and vetoed every genuine
        exit through the bottom edge. Scale is what distinguishes them:

            crouching     head sinks, apparent size barely changes
                          (you are not getting closer, only lower)
            approaching   head sinks AND grows, because range is shrinking
                          and a nearer head subtends more pixels

        So the test is downward motion WITHOUT growth. A head that drops while
        staying the same size ducked behind something; one that drops while
        growing walked toward the lens.

        Note this reads the tracked OMEGA. A crouching person's head does not
        get smaller — the body silhouette shortens, but the head is the same
        head — so a shrink test on the tracked box would find nothing. It is
        the absence of growth that carries the information.

        (t.w_hist holds WIDTH, not height — using it here was the other half of
        the original bug, so the height series is kept separately.)
        """
        hist = t.history[-n:] if len(t.history) >= 2 else []
        hh = getattr(t, "_hh", [])[-n:]
        if len(hist) < 3 or len(hh) < 3:
            return False
        dy = hist[-1][1] - hist[0][1]           # + is downward in image coords
        h0 = max(1e-6, sum(hh[:2]) / 2.0)
        h1 = sum(hh[-2:]) / 2.0
        grow = h1 / h0
        sank = dy > max(0.8, 0.10 * float(t.x[5]))
        return sank and grow < grow_max

    def resolve_occluded(self, dets, cnn_ran):
        """
        Decide whether an out-of-sight person is still out of sight.

        THE MISTAKE THIS REPLACES. The first version dropped a hidden person
        once the CNN looked at their last position and saw nothing, reasoning
        that a clear view of empty space proves absence. It does not. The thing
        hiding someone is usually FURNITURE, and furniture is at room
        temperature — invisible to a thermal detector and to any check that
        asks "is another track nearby?". Crouch behind a table in open floor
        and that test sees empty space and declares you gone, which is exactly
        backwards: a person who ducks behind a desk is the case the memory
        exists for.

        What actually bounds the room is the BOUNDARY. People enter and leave
        through the frame edge, so absence is proven by an observed exit, not
        by an empty patch of floor. A hidden person is therefore kept until
        one of three things happens: they reappear, an exit is observed that
        must have been them, or the timeout expires as a last resort.

        The clear-space signal is kept only as a WEAK vote, and only where the
        scene has never hidden anyone before. If a track has previously
        vanished and reappeared at this spot, the location is a known occluder
        and empty space there means nothing.
        """
        keep = []
        for o in self.occluded:
            age = self.frame - o["frame_lost"]
            if age > self.occl_timeout:
                self.n_gaveup += 1
                self.events.append(dict(
                    frame=self.frame, event="giveup", id=o["id"],
                    x=f"{o['x']:.1f}", y=f"{o['y']:.1f}",
                    cause=f"timeout{age}f", near_track="", near_dist="",
                    near_misses="", ghost_id="", ghost_dist="", ghost_age=age,
                    n_within_2gate="", n_dets=len(dets),
                    n_tracks=len(self.tracks)))
                continue
            if self.is_occluder_zone(o["x"], o["y"]):
                o["clear"] = 0            # known hiding place: prove nothing
            else:
                blocked = any(((d["centroid"][0] - o["x"]) ** 2 +
                               (d["centroid"][1] - o["y"]) ** 2) ** 0.5
                              <= self.gate_px * 1.5 for d in dets) or any(
                    ((t.centroid[0] - o["x"]) ** 2 +
                     (t.centroid[1] - o["y"]) ** 2) ** 0.5 <= self.gate_px * 1.5
                    for t in self.tracks)
                if blocked:
                    o["clear"] = 0
                elif cnn_ran:
                    o["clear"] += 1
                    if o["clear"] >= self.occl_clear_hard:
                        self.n_gaveup += 1
                        continue
            keep.append(o)
        self.occluded = keep

    # ---- learned occluder map --------------------------------------------
    # Where have people vanished and then come back? Those places hide people:
    # a desk, a partition, a pillar. The map is built from the tracker's own
    # mistakes, so it needs no configuration and adapts to whatever room the
    # sensor is pointed at. A 16x12 grid over a 160x120 sensor is 10 px per
    # cell — about a body width at mid range.
    def _cell(self, x, y):
        W, H = self.frame_wh
        return (min(15, max(0, int(x * 16 / max(1, W)))),
                min(11, max(0, int(y * 12 / max(1, H)))))

    def note_occluder(self, x, y):
        c = self._cell(x, y)
        self.occl_map[c] = self.occl_map.get(c, 0) + 1

    def is_occluder_zone(self, x, y):
        return self.occl_map.get(self._cell(x, y), 0) >= self.occl_learn
        self.ghosts = []          # (id, x, y, w, h, frame_died, hits)
        self.pending = []         # provisional births: [x, y, count, frame]
        self.events = []          # rows for track_events.csv
        self.frame = 0

    def _classify(self, d, live_before):
        """
        Why did this detection not belong to an existing track?

        Distinguishes the three failure modes that need different fixes:
          gate_miss   a live track was near but outside the gate -> gate/cost
          contested   two or more tracks were within reach -> greedy stole it
          returning   a track died near here recently -> needs re-identification
          genuine     nothing nearby, alive or dead -> a real new person
        """
        dx, dy = d["centroid"]
        near = sorted(
            [((((t.centroid[0] - dx) ** 2 + (t.centroid[1] - dy) ** 2) ** 0.5), t)
             for t in live_before], key=lambda pair: (pair[0], pair[1].id))
        g_near = sorted((((gx - dx) ** 2 + (gy - dy) ** 2) ** 0.5,
                         gid, self.frame - fd, gh)
                        for gid, gx, gy, gw, gh_, fd, gh in self.ghosts)
        nd, nt = (near[0][0], near[0][1]) if near else (None, None)
        gd, gid, gage, ghits = (g_near[0] if g_near else (None, None, None, None))
        contested = sum(1 for dist, _ in near if dist <= self.gate_px * 2)

        # ORDER MATTERS. A track sitting 7 px away with misses==0 was matched
        # THIS frame — it was never out of reach, it was already taken, and the
        # extra detection is a duplicate the deduper missed. Calling that
        # "gate_miss" sent me looking at the gate, which was not the problem.
        if nt is not None and nd <= self.gate_px and nt.misses == 0:
            cause = "duplicate"
        elif gd is not None and gd <= self.gate_px * 2:
            cause = "returning"
        elif nd is not None and nd <= self.gate_px * 3:
            cause = "contested" if contested >= 2 else "gate_miss"
        else:
            cause = "genuine"
        return {"cause": cause,
                "near_track": nt.id if nt else "",
                "near_dist": f"{nd:.1f}" if nd is not None else "",
                "near_misses": nt.misses if nt else "",
                "ghost_id": gid if gid is not None else "",
                "ghost_dist": f"{gd:.1f}" if gd is not None else "",
                "ghost_age": gage if gage is not None else "",
                "n_within_2gate": contested}

    def update(self, dets):
        self.frame += 1
        live_before = [t for t in self.tracks if t.hits >= self.min_hits]
        snap = {t.id: (t.centroid, t.misses) for t in self.tracks}
        for t in self.tracks:
            t.predict()

        def associate(tracks, det_idx, gate_for):
            pairs = []
            for t in tracks:
                tx, ty = t.centroid
                for di in det_idx:
                    dx, dy = dets[di]["centroid"]
                    d = ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5
                    if d <= gate_for(t):
                        pairs.append((d, id(t), di, t))
            pairs.sort(key=lambda p: p[0])
            used_t, used_d = set(), set()
            for d, tid, di, t in pairs:
                if tid in used_t or di in used_d:
                    continue
                t.update(dets[di])
                used_t.add(tid)
                used_d.add(di)
            return used_d

        born_edge = 0
        died_edge = 0
        left = set(range(len(dets)))
        live = [t for t in self.tracks if t.misses <= 1]
        left -= associate(live, list(left), lambda t: self.gate_px)

        coasting = [t for t in self.tracks if t.misses > 1]
        if left and coasting:
            before = len(left)
            left -= associate(
                coasting, list(left),
                lambda t: self.gate_px * min(self.revive_scale, 1.0 + 0.18 * t.misses))
            self.n_revived += before - len(left)

        for i in sorted(left):
            info = self._classify(dets[i], live_before)

            # FIX A — provisional birth.
            # A detection landing inside the gate of a track that was already
            # matched this frame is, on the evidence, a duplicate box on one
            # head: measured at 7.2 px from a matched track, alive 10 frames,
            # 2 hits. But it could also be a second person walking up to the
            # first, and those look identical for one frame. So neither trust
            # nor discard it — require it to persist. A duplicate flickers and
            # never reaches the count; a real person standing there does.
            if info["cause"] == "duplicate" and self.dup_confirm > 1:
                x, y = dets[i]["centroid"]
                hit = None
                for pnd in self.pending:
                    if abs(pnd[0] - x) <= self.gate_px * 0.5 and \
                       abs(pnd[1] - y) <= self.gate_px * 0.5 and \
                       self.frame - pnd[3] <= 2:
                        hit = pnd
                        break
                if hit is None:
                    self.pending.append([x, y, 1, self.frame])
                    self.n_suppressed += 1
                    continue
                hit[0], hit[1], hit[3] = x, y, self.frame
                hit[2] += 1
                if hit[2] < self.dup_confirm:
                    self.n_suppressed += 1
                    continue
                self.pending.remove(hit)
                info["cause"] = "duplicate_promoted"

            t = KalmanTrack(dets[i])

            # FIX C — recover an occluded person.
            # Checked BEFORE ordinary re-identification and with a wider
            # radius, because an occluded person is not merely someone we saw
            # recently: they are someone we have positive reason to believe
            # never left. They also drift while hidden — the sitter shifts,
            # the hidden person leans — so the window has to be generous.
            if self.occl_enabled and self.occluded:
                x, y = dets[i]["centroid"]
                cands = sorted(
                    ((((o["x"] - x) ** 2 + (o["y"] - y) ** 2) ** 0.5, k)
                     for k, o in enumerate(self.occluded)))
                if cands and cands[0][0] <= self.gate_px * 2.5:
                    o = self.occluded.pop(cands[0][1])
                    t.id = o["id"]
                    KalmanTrack._next_id -= 1
                    self.n_recovered += 1
                    # This spot demonstrably hides people. Remember it, so the
                    # next disappearance here is not second-guessed.
                    self.note_occluder(o["x"], o["y"])
                    info["cause"] = "occlusion_recovered"
                    self.events.append(dict(
                        frame=self.frame, event="recover", id=t.id,
                        x=f"{x:.1f}", y=f"{y:.1f}",
                        cause=f"hidden{self.frame - o['frame_lost']}f",
                        near_track=o["occluder"] or "", near_dist="",
                        near_misses="", ghost_id="", ghost_dist="",
                        ghost_age=self.frame - o["frame_lost"],
                        n_within_2gate="", n_dets=len(dets),
                        n_tracks=len(self.tracks)))
                    self.tracks.append(t)
                    self.n_new += 1
                    continue

            # FIX B — re-identification.
            # Measured: a track died, the person was gone 33 frames, came back
            # 3.8 px from where it died, and was issued a new id. For occupancy
            # counting that is not cosmetic — it reads as a departure plus an
            # arrival when nobody entered or left.
            if self.reid_px > 0:
                x, y = dets[i]["centroid"]
                cand = [(((gx - x) ** 2 + (gy - y) ** 2) ** 0.5, k, g)
                        for k, g in enumerate(self.ghosts)
                        for gid, gx, gy, gw, gh_, fd, gh in [g]]
                cand = [c for c in cand if c[0] <= self.reid_px]
                if cand:
                    cand.sort()
                    _, k, g = cand[0]
                    t.id = g[0]
                    KalmanTrack._next_id -= 1
                    self.ghosts.pop(k)
                    self.n_reid += 1
                    info["cause"] = info["cause"] + "_reid"

            self.tracks.append(t)
            self.n_new += 1
            x, y = dets[i]["centroid"]
            self.events.append(dict(frame=self.frame, event="birth", id=t.id,
                                    x=f"{x:.1f}", y=f"{y:.1f}",
                                    n_dets=len(dets), n_tracks=len(self.tracks),
                                    **info))

        dead = [t for t in self.tracks if t.misses > self.max_misses]
        for t in dead:
            x, y, w, h = t.bbox
            cx, cy = t.centroid
            self.ghosts.append((t.id, cx, cy, w, h, self.frame, t.hits))

            # BOUNDARY OUTRANKS DIRECTION.
            #
            # Leaving the frame is leaving the room, whatever the manner of
            # going — sideways, downward, crouching through the doorway, any
            # of it. The sensor's field of view IS the room as far as this
            # system is concerned, and something that crosses out of it is no
            # longer being counted. Reading direction there was over-thinking:
            # it let a person crouch on their way out of frame and stay on the
            # books forever.
            #
            # Direction is the right tool for the OTHER question — a track that
            # died in open view, with the boundary nowhere near. There, going
            # down means they ducked behind something and going sideways means
            # the detector lost someone who was walking out.
            if self.at_edge(cx, cy):
                kind = "exit"
            elif self.no_crouch_veto:
                kind = "hidden"
            else:
                kind = self.loss_kind(t)
                if kind == "unclear":
                    # No direction to read and not at the boundary: a person
                    # does not evaporate from open floor, so assume hidden.
                    kind = "hidden"
            if kind == "hidden" and t.hits >= self.min_hits:
                self.n_crouch += 1
            if (self.occl_enabled and t.hits >= self.min_hits
                    and kind == "hidden"):
                # Interior death. Who was next to them when they vanished? If
                # another track was within merging distance, that is positive
                # evidence of occlusion rather than an unexplained dropout.
                occluder = None
                best = 1e9
                for u in self.tracks:
                    if u is t or u.hits < self.min_hits:
                        continue
                    d = ((u.centroid[0] - cx) ** 2 + (u.centroid[1] - cy) ** 2) ** 0.5
                    if d < best:
                        best, occluder = d, u.id
                self.occluded.append(
                    {"id": t.id, "x": cx, "y": cy, "w": w, "h": h,
                     "frame_lost": self.frame, "clear": 0, "det": t.det,
                     "occluder": occluder if best <= self.gate_px * 2 else None})
                self.n_occluded += 1
            elif t.hits >= self.min_hits:
                self.n_exited += 1
                died_edge += 1
                self.events.append(dict(
                    frame=self.frame, event="exit", id=t.id,
                    x=f"{cx:.1f}", y=f"{cy:.1f}", cause=kind,
                    near_track="", near_dist="", near_misses="",
                    ghost_id="", ghost_dist="", ghost_age="",
                    n_within_2gate="", n_dets=len(dets),
                    n_tracks=len(self.tracks)))
            self.events.append(dict(frame=self.frame, event="death", id=t.id,
                                    x=f"{t.centroid[0]:.1f}",
                                    y=f"{t.centroid[1]:.1f}",
                                    n_dets=len(dets), n_tracks=len(self.tracks),
                                    cause=f"age{t.age}_hits{t.hits}",
                                    near_track="", near_dist="", near_misses="",
                                    ghost_id="", ghost_dist="", ghost_age="",
                                    n_within_2gate=""))
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        self.ghosts = [g for g in self.ghosts
                       if self.frame - g[5] <= self.ghost_frames]
        for t in self.tracks:
            if not hasattr(t, "_hh"):
                t._hh = []
            # ONLY on frames that were actually measured. A coasting track
            # predicts w and h forward unchanged, so recording every frame
            # fills the window with identical values and every departure then
            # looks like "sank without growing" — i.e. a crouch. The last 8
            # real observations are what carry the scale information.
            if t.misses == 0:
                t._hh.append(float(t.x[5]))
                if len(t._hh) > 24:
                    t._hh.pop(0)

        self.pending = [q for q in self.pending if self.frame - q[3] <= 2]
        if self.use_ledger:
            self.ledger_update(born_edge, died_edge, 0)
        return self.tracks

    def ledger_update(self, born_ids, died_edge, died_interior):
        """
        Count the room by what crossed its edge, not by what is visible.

        A detector reports what it can see. An occupancy sensor is asked
        something else: how many people are in the room. Those differ exactly
        when someone is hidden, which is the case that matters. The boundary is
        what makes the second question answerable — a person can only join the
        room by crossing into frame and can only leave by crossing out.

        So: births at the edge add, deaths at the edge subtract, and anything
        that happens in the interior changes nothing. A person who crouches
        behind a desk in the middle of the room never touches the ledger.

        The failure mode of a ledger is drift: miss one exit and the count is
        permanently high. Two guards. If more people are visible than the
        ledger believes, the ledger is raised at once — seeing is stronger than
        bookkeeping. And a surplus that is never corroborated decays after
        --ledger-decay frames, logged, so a stuck count self-heals instead of
        needing a restart.
        """
        self.n_entries += born_ids
        self.n_exits += died_edge
        self.ledger = max(0, self.n_entries - self.n_exits)

        vis = len(self.confirmed())
        if vis > self.ledger:
            self.ledger = vis
            self.n_entries = self.ledger + self.n_exits
            self.surplus_since = self.frame
        elif self.ledger > vis + len(self.occluded):
            if self.frame - self.surplus_since > self.ledger_decay:
                self.n_exits += 1
                self.n_decayed += 1
                self.surplus_since = self.frame
                self.events.append(dict(
                    frame=self.frame, event="decay", id="",
                    x="", y="", cause="unverified surplus",
                    near_track="", near_dist="", near_misses="",
                    ghost_id="", ghost_dist="", ghost_age="",
                    n_within_2gate="", n_dets="", n_tracks=len(self.tracks)))
        else:
            self.surplus_since = self.frame

    def present(self):
        """
        Everyone in the room: visible plus known-but-hidden.

        This is the number an occupancy sensor should report. Counting only
        what is visible means a person sitting behind another reads as having
        left, and reappears as an arrival — two errors in the log for an event
        that never happened.
        """
        if self.use_ledger:
            return max(self.ledger, len(self.confirmed()))
        return len(self.confirmed()) + len(self.occluded)

