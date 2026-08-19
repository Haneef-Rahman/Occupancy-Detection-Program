#!/usr/bin/env python3
"""
Constant-velocity Kalman tracking for thermal detections.

Why tracking belongs here rather than as a nicety:

  * A person is ONE object persisting through time; fragments and clutter
    flicker. Requiring a track to be confirmed over several frames rejects
    transient false positives that no single-frame descriptor can catch.
  * The Lepton pauses ~0.5-1 s for its FFC shutter, and blobs drop out during
    pose changes. A predicted track survives those gaps; a per-frame detector
    loses the person and re-acquires them as someone new.
  * Counting crossings needs identity, not just presence. A track has an ID
    and a trajectory, so "entered" and "left" become answerable.

State is [x, y, vx, vy, w, h] with position/size measured and velocity
inferred. No external dependency — the matrices are small enough to write out.
"""

import numpy as np


class KalmanTrack:
    """One tracked object under a constant-velocity motion model."""

    _next_id = 1

    def __init__(self, det, dt=1.0, q_pos=1.0, q_vel=2.0, r_meas=4.0):
        cx, cy = det["centroid"]
        x, y, w, h = det["bbox"]
        self.id = KalmanTrack._next_id
        KalmanTrack._next_id += 1

        # state: x, y, vx, vy, w, h
        self.x = np.array([cx, cy, 0.0, 0.0, float(w), float(h)], np.float64)

        self.F = np.eye(6)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        self.H = np.zeros((4, 6))
        self.H[0, 0] = self.H[1, 1] = 1.0     # measure x, y
        self.H[2, 4] = self.H[3, 5] = 1.0     # measure w, h

        self.P = np.diag([10.0, 10.0, 100.0, 100.0, 10.0, 10.0])
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_pos, q_pos])
        self.R = np.diag([r_meas, r_meas, r_meas * 2, r_meas * 2])

        self.age = 1            # frames since created
        self.hits = 1           # frames matched to a detection
        self.misses = 0         # consecutive frames unmatched
        self.det = det          # most recent detection payload
        self.history = [(cx, cy)]

        # Peak constellations, centred per frame so global translation is
        # removed and only INTERNAL deformation remains.
        self.pk_hist = []
        self._push_peaks(det)

    def _push_peaks(self, det):
        pts = det.get("peak_pts")
        if pts is None or len(pts) < 3:
            self.pk_hist.append(None)
        else:
            p = np.asarray(pts, np.float64)
            self.pk_hist.append(p - p.mean(axis=0))
        if len(self.pk_hist) > 40:
            self.pk_hist.pop(0)

    def rigidity(self, min_frames=8, min_pts=3):
        """
        How much the internal peak geometry deforms, as a coefficient of
        variation of pairwise peak distances.

        A rigid body cannot deform: its hot spots are fixed in the object, so
        every pairwise distance is constant and this returns ~0 regardless of
        how many peaks it has. An articulated body cannot help deforming --
        arms swing, the head turns, hands cross the torso -- so it returns a
        clearly positive value.

        This is the one measurement that does not care about peak COUNT, which
        is why it escapes the trap that raising the count also raises it for
        equipment.

        Returns None until there is enough history to be meaningful; the caller
        must treat None as "unknown", never as "rigid".
        """
        frames = [p for p in self.pk_hist if p is not None]
        if len(frames) < min_frames:
            return None
        m = min(len(p) for p in frames)
        if m < min_pts:
            return None

        rows = []
        for p in frames:
            # order top-to-bottom so the same physical peak lines up across
            # frames without needing full correspondence solving
            q = p[np.argsort(p[:, 1])][:m]
            d = np.sqrt(((q[:, None, :] - q[None, :, :]) ** 2).sum(-1))
            rows.append(d[np.triu_indices(m, 1)])
        D = np.asarray(rows)
        return float(np.mean(D.std(axis=0) / (D.mean(axis=0) + 1e-6)))

    # -- prediction / update -------------------------------------------------
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.misses += 1
        return self.x

    def update(self, det):
        cx, cy = det["centroid"]
        _, _, w, h = det["bbox"]
        z = np.array([cx, cy, float(w), float(h)], np.float64)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        self.hits += 1
        self.misses = 0
        self.det = det
        self._push_peaks(det)
        self.history.append((self.x[0], self.x[1]))
        if len(self.history) > 60:
            self.history.pop(0)

    # -- accessors -----------------------------------------------------------
    @property
    def centroid(self):
        return float(self.x[0]), float(self.x[1])

    @property
    def bbox(self):
        cx, cy, _, _, w, h = self.x
        return (int(cx - w / 2), int(cy - h / 2), int(max(1, w)), int(max(1, h)))

    @property
    def speed(self):
        return float((self.x[2] ** 2 + self.x[3] ** 2) ** 0.5)

    def confirmed(self, min_hits):
        return self.hits >= min_hits


class MultiTracker:
    """
    Greedy nearest-neighbour association with a gate, plus track lifecycle.

    Greedy rather than Hungarian on purpose: at most a handful of people are
    in frame, the cost matrix is tiny, and the gate does the real work of
    preventing bad matches. Hungarian would be optimal but indistinguishable
    here, and this stays readable.
    """

    def __init__(self, max_misses=6, min_hits=3, gate_px=28.0):
        self.tracks = []
        self.max_misses = max_misses      # ~0.7 s of coasting at 8.7 fps
        self.min_hits = min_hits          # frames before a track is trusted
        self.gate_px = gate_px

    def update(self, dets):
        for t in self.tracks:
            t.predict()

        unmatched = list(range(len(dets)))
        if self.tracks and dets:
            pairs = []
            for ti, t in enumerate(self.tracks):
                tx, ty = t.centroid
                for di, d in enumerate(dets):
                    dx, dy = d["centroid"]
                    dist = ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5
                    if dist <= self.gate_px:
                        pairs.append((dist, ti, di))
            pairs.sort()
            used_t, used_d = set(), set()
            for dist, ti, di in pairs:
                if ti in used_t or di in used_d:
                    continue
                self.tracks[ti].update(dets[di])
                used_t.add(ti)
                used_d.add(di)
            unmatched = [i for i in range(len(dets)) if i not in used_d]

        for i in unmatched:
            self.tracks.append(KalmanTrack(dets[i]))

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks

    def confirmed(self):
        """Tracks trusted enough to count — the transient-rejection payoff."""
        return [t for t in self.tracks
                if t.confirmed(self.min_hits) and t.misses == 0]

    def rigid_ids(self, thresh, min_frames=8):
        """Confirmed tracks whose internal geometry never deforms."""
        out = []
        for t in self.tracks:
            r = t.rigidity(min_frames=min_frames)
            if r is not None and r < thresh:
                out.append(t.id)
        return out

    def reset(self):
        self.tracks = []
