"""Browser-independent processing, built from shared tracker/detector modules.

Does not import either integrated launcher or the desktop simulator.
"""
from dataclasses import replace
import time
import numpy as np
from crowd_detector import thermal_image
from tracking_router import TrackingRouter
from faster_k import FastConfig


class Processor:
    def __init__(self, detector, config=None, mode='auto'):
        self.detector = detector
        self.config = config or FastConfig()
        self.mode = mode
        self.reset()

    def reset(self):
        self.router = TrackingRouter((160, 120), self.config, self.mode)
        self.frame = 0

    def set_mode(self, mode):
        if mode not in ('auto', 'precise', 'faster'):
            raise ValueError('Unknown tracking policy.')
        self.mode = mode
        self.router.policy = mode

    def set_confidence(self, conf, low_conf):
        if not 0 < low_conf < conf < 1:
            raise ValueError('Require 0 < search confidence < detection confidence < 1.')
        self.detector.conf, self.detector.low_conf = conf, low_conf
        self.config = replace(self.config, low_conf=low_conf)
        self.router.config = self.router.fast.config = self.config
        for track in self.router.fast.tracks:
            track.config = self.config

    def process(self, data, timestamp, origin):
        if data.shape != (120, 160) or data.dtype.kind != 'f' or not np.isfinite(data).all():
            raise ValueError('Expected finite native 160×120 Celsius data.')
        t0 = time.perf_counter()
        image = thermal_image(data)
        high = [dict(d, detection_index=i, track_id=None) for i,d in enumerate(self.detector.base(image))]
        t1 = time.perf_counter()
        if self.mode != 'auto' and self.router.mode != self.mode:
            if self.router.previous_time is None:self.router.mode = self.mode
            else:self.router.switch(self.mode, timestamp)
        self.router.begin(high, timestamp)
        t2 = time.perf_counter()
        low = self.detector.recover(image, self.router.regions)
        t3 = time.perf_counter()
        self.router.finish(low)
        t4 = time.perf_counter()
        self.frame += 1
        for track in self.router.observed():
            if track.det.get('source','base') == 'base' and track.misses == 0:
                index = track.det.get('detection_index')
                if index is not None and 0 <= index < len(high):high[index]['track_id'] = track.id
        row = dict(frame=self.frame,time=float(timestamp),origin=origin,mode=self.router.mode,
            normal_confidence=self.detector.conf,recovery_confidence=self.detector.low_conf,tracker_policy=self.mode,
            base_detections=high,recovery_detections=low,regions=self.router.regions,
            observed=[dict(id=t.id,bbox=list(t.det['bbox']),conf=t.det.get('conf'),source=t.det.get('source','base')) for t in self.router.observed()],
            unresolved=self.router.unresolved_count(),estimate=self.router.estimate(),
            base_ms=(t1-t0)*1000,recovery_ms=(t3-t2)*1000,tracking_ms=(t2-t1+t4-t3)*1000,
            processing_ms=(t4-t0)*1000,events=list(self.router.events),
            fast_stats=dict(self.router.fast.stats) if self.router.mode == 'faster' else {})
        self.router.events.clear()
        return row
