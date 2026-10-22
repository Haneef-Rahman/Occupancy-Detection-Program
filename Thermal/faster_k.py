"""Experimental, conservative crowd tracking in native thermal pixels.

No performance/accuracy claim is attached to the name 'faster K'. No image
search, segmentation, synthetic boxes, hidden-person ledger or long-term re-ID.
All thresholds are uncalibrated field-test settings, not physiological limits.
"""
from dataclasses import dataclass
import math
import numpy as np
from tracker import KalmanTrack


@dataclass(frozen=True)
class FastConfig:
    recovery_seconds: float = 2.0
    max_speed: float = 30.0          # native px/s; needs distance-specific tuning
    max_accel: float = 100.0        # native px/s^2, model uncertainty
    max_radius: float = 14.0        # cap even if covariance grows
    jitter: float = 2.0
    mahalanobis_limit: float = 9.21
    size_ratio: float = 1.8
    ambiguity_margin: float = 0.15
    min_hits: int = 2
    recovery_hits: int = 2
    edge_margin: float = 5.0
    low_conf: float = 0.18
    max_rois: int = 6

    def __post_init__(self):
        for key in ('recovery_seconds', 'max_speed', 'max_accel', 'max_radius',
                    'jitter', 'mahalanobis_limit', 'size_ratio'):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.size_ratio < 1 or not 0 < self.low_conf < 1:
            raise ValueError('size_ratio >= 1 and 0 < low_conf < 1 required')
        if min(self.min_hits, self.recovery_hits, self.max_rois) < 1:
            raise ValueError('hit counts and max_rois must be positive')
        if not math.isfinite(self.ambiguity_margin) or self.ambiguity_margin < 0:
            raise ValueError('invalid ambiguity margin')


def box_iou(a, b):
    x, y, w, h = a; X, Y, W, H = b
    inter = max(0., min(x+w, X+W)-max(x, X)) * max(0., min(y+h, Y+H)-max(y, Y))
    return inter / max(1e-9, w*h+W*H-inter)


def clean_detections(dets, wh):
    """Reject malformed observations; suppress only near-identical duplicates."""
    keep = []
    W, H = wh
    for d in sorted(dets, key=lambda d: -float(d.get('conf', 0))):
        vals = np.asarray([*d['bbox'], *d['centroid'], d.get('conf', 0)], float)
        if not np.isfinite(vals).all():
            continue
        x, y, w, h = d['bbox']; cx, cy = d['centroid']
        if w <= 0 or h <= 0 or not (0 <= cx < W and 0 <= cy < H):
            continue
        if any(box_iou(d['bbox'], k['bbox']) >= .85 for k in keep):
            continue
        keep.append(d)
    return keep


def touching(dets, margin=1.0):
    for i, a in enumerate(dets):
        x, y, w, h = a['bbox']
        for b in dets[i+1:]:
            X, Y, W, H = b['bbox']
            if x <= X+W+margin and X <= x+w+margin and y <= Y+H+margin and Y <= y+h+margin:
                return True
    return False


class FastTrack:
    def __init__(self, det, now, config, identity=None):
        if identity is None:
            identity = KalmanTrack._next_id
            KalmanTrack._next_id += 1
        self.id = identity
        self.x = np.array([*det['centroid'], 0., 0.], dtype=float)
        self.P = np.diag([4., 4., 100., 100.])
        self.det = dict(det)
        self.size = np.array(det['bbox'][2:], float)
        self.last_position = self.x[:2].copy()
        self.last_seen = self.last_high = self.now = now
        self.hits = 1
        self.misses = 0
        self.low_streak = 0
        self.observed = True
        self.source = 'base'
        self.history = [tuple(self.x[:2])]
        self.config = config

    def predict(self, now):
        dt = now - self.now
        if dt <= 0:
            raise ValueError('timestamps must increase')
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        G = np.array([[dt*dt/2, 0], [0, dt*dt/2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.config.max_accel**2
        self.now = now
        self.observed = False
        self.misses += 1

    def correct(self, det, high):
        H = np.eye(4)[:2]
        R = np.eye(2) * (4. if high else 9.)
        K = np.linalg.solve(H @ self.P @ H.T + R, H @ self.P).T
        self.x += K @ (np.asarray(det['centroid']) - H @ self.x)
        I = np.eye(4) - K @ H
        self.P = I @ self.P @ I.T + K @ R @ K.T
        speed = np.linalg.norm(self.x[2:])
        if speed > self.config.max_speed:
            self.x[2:] *= self.config.max_speed / speed
        self.det = dict(det)
        self.size = np.asarray(det['bbox'][2:], float)
        self.last_position = np.asarray(det['centroid'], float)
        self.last_seen = self.now
        if high:
            self.last_high = self.now
            self.low_streak = 0
        else:
            self.low_streak += 1
        self.hits += int(high)
        self.misses = 0
        self.source = 'base' if high else 'recovery'
        self.observed = high or self.low_streak >= self.config.recovery_hits
        self.history.append(tuple(det['centroid']))
        self.history = self.history[-60:]

    @property
    def centroid(self):
        return tuple(self.x[:2])

    @property
    def bbox(self):
        # Always the measured detector box, never a Kalman-generated rectangle.
        return self.det['bbox']

    def region(self):
        cfg = self.config
        covariance_radius = math.sqrt(cfg.mahalanobis_limit * max(0., np.linalg.eigvalsh(self.P[:2, :2]).max()+9.))
        elapsed = self.now-self.last_seen
        radius = min(cfg.max_radius, covariance_radius, cfg.jitter+cfg.max_speed*elapsed)
        return {'id': self.id, 'cx': float(self.x[0]), 'cy': float(self.x[1]),
                'radius': float(radius), 'remaining_s': max(0., cfg.recovery_seconds-(self.now-self.last_high))}


class FasterK:
    def __init__(self, frame_wh=(160, 120), config=None):
        self.config = config or FastConfig()
        self.frame_wh = frame_wh
        self.tracks = []
        self.now = None
        self.high = []
        self.regions = []
        self.stats = {}

    def adopt(self, precise_tracks, now, previous_dt):
        """Carry currently observed IDs; do not import legacy ghosts/ledger."""
        self.tracks = []
        for old in precise_tracks:
            if old.misses:
                continue
            t = FastTrack(old.det, now, self.config, old.id)
            t.x[:2] = old.x[:2]
            t.x[2:] = old.x[2:4] / max(.001, previous_dt)
            speed = np.linalg.norm(t.x[2:])
            if speed > self.config.max_speed:
                t.x[2:] *= self.config.max_speed/speed
            t.hits = old.hits
            self.tracks.append(t)
        self.now = now

    def cost(self, t, d):
        cfg = self.config
        pos = np.asarray(d['centroid'], float)
        size = np.asarray(d['bbox'][2:], float)
        ratio = np.maximum(size/t.size, t.size/size)
        if ratio.max() > cfg.size_ratio:
            return math.inf
        elapsed = self.now-t.last_seen
        if np.linalg.norm(pos-t.last_position) > cfg.jitter+cfg.max_speed*elapsed:
            return math.inf
        delta = pos-t.x[:2]
        if np.linalg.norm(delta) > t.region()['radius']:
            return math.inf
        m2 = float(delta @ np.linalg.solve(t.P[:2, :2]+np.eye(2)*9., delta))
        if m2 > cfg.mahalanobis_limit:
            return math.inf
        return m2/cfg.mahalanobis_limit + .25 * float(np.abs(np.log(size/t.size)).sum())

    def associate(self, tracks, dets, high):
        """Mutual best match with a uniqueness margin; ambiguous IDs stay lost.

        Deliberately conservative instead of forcing a global assignment.
        Gated-but-ambiguous high observations cannot create duplicate identities.
        """
        if not tracks or not dets:
            return set(), set()
        cost = np.array([[self.cost(t, d) for d in dets] for t in tracks])
        contested = set(np.flatnonzero(np.isfinite(cost).any(axis=0)))
        used = set()
        for ti, t in enumerate(tracks):
            di = int(np.argmin(cost[ti]))
            best = cost[ti, di]
            if not math.isfinite(best) or int(np.argmin(cost[:, di])) != ti:
                continue
            row = np.delete(cost[ti], di)
            col = np.delete(cost[:, di], ti)
            alternative = min(row.min(initial=math.inf), col.min(initial=math.inf))
            if alternative-best < self.config.ambiguity_margin:
                self.stats['ambiguous'] += 1
                continue
            t.correct(dets[di], high)
            used.add(di)
        return used, contested

    def begin(self, high, now):
        if not math.isfinite(now) or (self.now is not None and now <= self.now):
            raise ValueError('finite, strictly increasing timestamps required')
        self.now = now
        cfg = self.config
        self.stats = {'expired': 0, 'ambiguous': 0, 'births': 0, 'recovered': 0}
        alive = []
        for t in self.tracks:
            if now-t.last_high > cfg.recovery_seconds:
                self.stats['expired'] += 1
                continue
            t.predict(now)
            alive.append(t)
        self.tracks = alive
        self.high = clean_detections(high, self.frame_wh)
        used, contested = self.associate(self.tracks, self.high, True)
        self.regions = []
        W, H = self.frame_wh
        for t in self.tracks:
            x, y = t.last_position
            if (t.misses and t.hits >= cfg.min_hits and
                    cfg.edge_margin < x < W-cfg.edge_margin and
                    cfg.edge_margin < y < H-cfg.edge_margin):
                self.regions.append(t.region())
        # Bounded compute, deterministic priority rotates if more tracks are lost.
        self.regions.sort(key=lambda r: (getattr(next(t for t in self.tracks if t.id == r['id']), 'last_roi', -math.inf), r['id']))
        self.regions = self.regions[:cfg.max_rois]
        for t in self.tracks:
            if any(r['id'] == t.id for r in self.regions):
                t.last_roi = now
        for i, d in enumerate(self.high):
            if i not in used and i not in contested:
                self.tracks.append(FastTrack(d, now, cfg))
                self.stats['births'] += 1
        return self.regions

    def finish(self, low):
        # Only ROIs issued this frame are eligible; no weak detection births.
        allowed = {r['id'] for r in self.regions}
        candidates = [t for t in self.tracks if t.misses and t.id in allowed]
        low = clean_detections(low, self.frame_wh)
        low = [d for d in low if self.config.low_conf <= d.get('conf', 0) <= 1.]
        low = [d for d in low if not any(box_iou(d['bbox'], h['bbox']) > .3 for h in self.high)]
        # Provenance ties a crop to its requested track. A candidate inside two
        # circles still must pass unambiguous association.
        low = [d for d in low if any(
            (d['centroid'][0]-r['cx'])**2+(d['centroid'][1]-r['cy'])**2 <= r['radius']**2
            for r in self.regions)]
        used, _ = self.associate(candidates, low, False)
        self.stats['recovered'] = len(used)
        for t in self.tracks:
            if t.misses:
                t.low_streak = 0
        return self.visible()

    def visible(self):
        return [t for t in self.tracks if t.observed and t.hits >= self.config.min_hits]

    def unresolved(self):
        return [t for t in self.tracks if not t.observed and t.hits >= self.config.min_hits]
