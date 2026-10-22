"""Mode routing and explicit handover for the new experimental launcher."""
import numpy as np
from tracker import KalmanTrack
from precise_k import PreciseK, dedup
from faster_k import FasterK, FastConfig, clean_detections, touching


class TrackingRouter:
    def __init__(self, frame_wh=(160, 120), config=None, mode='auto', settle_seconds=1.0):
        if mode not in ('auto', 'precise', 'faster'):
            raise ValueError('unknown tracking mode')
        self.policy = mode
        self.config = config or FastConfig()
        self.frame_wh = frame_wh
        self.settle_seconds = settle_seconds
        self.mode = 'faster' if mode == 'faster' else 'precise'
        self.precise = self.new_precise()
        self.fast = FasterK(frame_wh, self.config)
        self.previous_time = None
        self.previous_dt = 1/8.7  # conversion fallback only; not a timing estimate
        self.calm_since = None
        self.events = []
        self.regions = []

    def new_precise(self):
        return PreciseK(max_misses=8, min_hits=self.config.min_hits, gate_px=30.,
                        ghost_frames=45, frame_wh=self.frame_wh)

    def switch(self, target, now):
        if target == 'faster':
            self.fast.adopt(self.precise.tracks, self.previous_time, self.previous_dt)
            discarded = len(self.precise.occluded) + sum(t.misses > 0 for t in self.precise.tracks)
            note = f'{discarded} legacy hidden/coasting entries not carried into crowd mode'
        else:
            self.precise = self.new_precise()
            for old in self.fast.tracks:
                if not old.observed:
                    continue
                t = KalmanTrack(old.det)
                t.id = old.id
                t.x[:2] = old.x[:2]
                t.x[2:4] = old.x[2:] * self.previous_dt
                t.hits = old.hits
                # P units change from px/s to px/frame; preserve relevant block.
                J = np.diag([1., 1., self.previous_dt, self.previous_dt])
                t.P[:4, :4] = J @ old.P @ J.T
                t.history = old.history.copy()
                self.precise.tracks.append(t)
            self.precise.ledger = len(self.precise.confirmed())
            self.precise.n_entries = self.precise.ledger
            note = 'observed IDs retained; legacy hidden memory starts empty'
        self.events.append({'time': now, 'from': self.mode, 'to': target, 'note': note})
        self.mode = target
        self.calm_since = None

    def begin(self, high, now):
        if self.previous_time is not None and now <= self.previous_time:
            raise ValueError('timestamps must increase')
        high = clean_detections(high, self.frame_wh)
        crowded = len(high) > 4 or touching(high)
        if self.policy == 'auto':
            if crowded and self.mode == 'precise':
                # First frame has no state to hand over.
                if self.previous_time is None:
                    self.mode = 'faster'
                else:
                    self.switch('faster', now)
            elif self.mode == 'faster':
                # A count drop caused by missing people must not trigger fallback.
                calm = not crowded and not self.fast.unresolved() and len(self.fast.tracks) <= 4
                self.calm_since = (now if self.calm_since is None else self.calm_since) if calm else None
                if self.calm_since is not None and now-self.calm_since >= self.settle_seconds:
                    self.switch('precise', now)
        if self.mode == 'precise':
            high = dedup(high)
            self.precise.update(high)
            self.precise.resolve_occluded(high, True)
            self.regions = []
        else:
            self.regions = self.fast.begin(high, now)
        if self.previous_time is not None:
            self.previous_dt = now-self.previous_time
        self.previous_time = now
        return self.regions

    def finish(self, low):
        if self.mode == 'faster':
            self.fast.finish(low)

    def observed(self):
        return self.fast.visible() if self.mode == 'faster' else self.precise.confirmed()

    def unresolved_count(self):
        if self.mode == 'faster':
            return len(self.fast.unresolved())
        return len(self.precise.occluded) + sum(
            t.misses > 0 and t.hits >= self.config.min_hits for t in self.precise.tracks)

    def estimate(self):
        # Crowd unresolved is exposed separately, not silently added to occupancy.
        return len(self.fast.visible()) if self.mode == 'faster' else self.precise.present()
