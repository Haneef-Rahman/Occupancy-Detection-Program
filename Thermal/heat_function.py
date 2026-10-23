"""Shared thermal-area graph diagnostic. No detector/tracker feedback."""
from dataclasses import dataclass,asdict
from collections import deque
import math
import numpy as np

@dataclass(frozen=True)
class HeatConfig:
    low_c:float=28.
    high_c:float=37.
    near_weight:float=1.
    far_weight:float=3.
    power:float=1.
    area:float=400.
    a:float=.3
    b:float=.7
    warmup_s:float=5.
    switch_window_s:float=30.
    max_precise_entries:int=1
    def __post_init__(self):
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):raise ValueError('All settings must be finite')
        if not self.low_c<self.high_c:raise ValueError('Minimum temperature must be below maximum')
        if not 0<self.near_weight<=self.far_weight or self.power<=0 or self.area<=0:raise ValueError('Require 0 < near <= far; positive power and area')
        if not 0<=self.a<.5<self.b<=1 or not math.isclose(self.a+self.b,1.,abs_tol=1e-6):raise ValueError('Require a < 0.5, b > 0.5 and a + b = 1')
        if self.warmup_s<0 or self.switch_window_s<=0 or self.max_precise_entries<0 or int(self.max_precise_entries)!=self.max_precise_entries:raise ValueError('Invalid gating settings')

class PlausibleDetected:
    """Hold a detection peak; accumulated whole heat units adjust its count.

    Current Detected is always a floor. A detection-driven increase or a
    blocked decrease reanchors the heat reference, avoiding deferred changes.
    This diagnostic never feeds the detector or tracker.
    """
    def __init__(self):self.reset()
    def reset(self):self.value=None;self.reference=None
    def update(self,detected,raw_heat):
        if self.value is None:
            self.value=int(detected);self.reference=float(raw_heat)
        else:
            change=float(raw_heat)-self.reference
            nearest=round(change)
            if math.isclose(change,nearest,rel_tol=0.,abs_tol=1e-9):change=nearest
            # Truncate toward zero: sub-unit decreases must not round to -1.
            steps=math.trunc(change)
            candidate=max(0,self.value+steps)
            if detected>candidate:
                self.value=int(detected);self.reference=float(raw_heat)
            else:
                self.value=candidate;self.reference+=steps
        return self.value

class HeatFunction:
    def __init__(self,config=None):
        self.config=config or HeatConfig();self.reset()
    def reset(self):
        self.previous=None;self.previous_t=None;self.faster_since=None;self.entries=deque()
        self.plausible=PlausibleDetected()
    def update(self,data,row):
        c=self.config;t=float(row['time']);mode=row['mode']
        if self.previous_t is not None and t<=self.previous_t:self.reset()
        if mode=='precise' and self.previous!='precise':self.entries.append(t)
        while self.entries and self.entries[0]<t-c.switch_window_s:self.entries.popleft()
        if mode!='faster':self.faster_since=None
        elif self.previous!='faster' or self.faster_since is None:self.faster_since=t
        active=mode=='faster' and t-self.faster_since>=c.warmup_s and len(self.entries)<=c.max_precise_entries
        self.previous=mode;self.previous_t=t
        result=dict(heat_active=active,raw_heat=None,heat=None,detected=None,plausible_detected=None,blend_detected=None,untracked_raw=None,heat_band_pixels=None)
        if active:
            y=np.linspace(1.,0.,data.shape[0])[:,None]
            weights=c.near_weight+(c.far_weight-c.near_weight)*y**c.power
            mask=np.isfinite(data)&(data>=c.low_c)&(data<=c.high_c)
            raw=float(np.sum(mask*weights))/c.area
            raw_detections=len(row['base_detections'])
            # Preserve the original blend input; the revised displayed Detected
            # is an output and must not recursively feed back into this frame.
            baseline=raw_detections+max(len(row['observed']),row['estimate'])
            heat=math.floor(c.a*raw+c.b*baseline)
            untracked=sum(d.get('track_id') is None for d in row['base_detections'])
            result.update(raw_heat=raw,blend_detected=baseline,untracked_raw=untracked,detected=heat+untracked,
                          heat=heat,heat_band_pixels=int(mask.sum()))
            result['plausible_detected']=self.plausible.update(result['detected'],raw)
        else:self.plausible.reset()
        row.update(result);return result
