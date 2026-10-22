"""Unchanged global confidence; lower-threshold recovery inside bounded ROIs."""
import math
import numpy as np
from faster_k import clean_detections


def thermal_image(data):
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError('expected a finite 2D Celsius array')
    grey = (np.clip((data.astype(np.float32)-15.)/30., 0, 1)*255).astype(np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


def decode(result, offset=(0, 0), source='base'):
    out = []
    if result.boxes is None:
        return out
    for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(),
                              result.boxes.conf.cpu().numpy(),
                              result.boxes.cls.cpu().numpy()):
        if int(cls) != 1:
            continue
        x0, y0, x1, y1 = map(float, box)
        x0 += offset[0]; x1 += offset[0]
        y0 += offset[1]; y1 += offset[1]
        out.append({'bbox': (x0, y0, x1-x0, y1-y0),
                    'centroid': ((x0+x1)/2, (y0+y1)/2), 'conf': float(conf),
                    'src': 'cnn', 'source': source, 'body_h': (y1-y0)/.30})
    return out


class CrowdDetector:
    def __init__(self, model, conf=.374, low_conf=.18, imgsz=640, device=None):
        if not 0 < low_conf < conf < 1:
            raise ValueError('require 0 < recovery confidence < base confidence < 1')
        names = model.names
        name = names.get(1, '') if isinstance(names, dict) else (names[1] if len(names) > 1 else '')
        if str(name).lower().replace('-', '_').replace(' ', '_') not in (
                'omega', 'head_shoulder', 'head_shoulders', 'head_and_shoulders'):
            raise ValueError(f'class 1 must be omega/head-shoulders; got {name!r}')
        self.model = model
        self.conf = conf
        self.low_conf = low_conf
        self.kw = dict(verbose=False, imgsz=imgsz, classes=[1], save=False)
        if device is not None:
            self.kw['device'] = device

    def base(self, image):
        result = self.model.predict(image, conf=self.conf, **self.kw)[0]
        return clean_detections(decode(result), (image.shape[1], image.shape[0]))

    def recover(self, image, regions):
        if not regions:
            return []
        H, W = image.shape[:2]
        crops, offsets, used_regions = [], [], []
        for r in regions:
            # Context outside the circle helps the CNN; only centroids inside
            # the actual circle are eligible. Original global pass is untouched.
            extent = r['radius'] + 8.
            x0 = max(0, math.floor(r['cx']-extent)); y0 = max(0, math.floor(r['cy']-extent))
            x1 = min(W, math.ceil(r['cx']+extent)); y1 = min(H, math.ceil(r['cy']+extent))
            if x1-x0 < 2 or y1-y0 < 2:
                continue
            crops.append(image[y0:y1, x0:x1].copy())
            offsets.append((x0, y0)); used_regions.append(r)
        if not crops:
            return []
        results = self.model.predict(crops, conf=self.low_conf, **self.kw)
        if len(results) != len(crops):
            raise RuntimeError('ROI detector returned wrong batch size')
        out = []
        for result, offset, r in zip(results, offsets, used_regions):
            for d in decode(result, offset, 'recovery'):
                dx = d['centroid'][0]-r['cx']; dy = d['centroid'][1]-r['cy']
                if d['conf'] >= self.low_conf and dx*dx+dy*dy <= r['radius']**2:
                    # Even high scores in crops count as recovery only, so
                    # crop feedback cannot extend the 2 s baseline-confirmed TTL.
                    out.append(d)
        return clean_detections(out, (W, H))
