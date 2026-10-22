#!/usr/bin/env python3
"""Experimental thermal-only launcher: precise K / faster K, live or replay.

No screen recording. JSON logs contain observations, IDs, regions and timing.
Legacy launcher remains available for radar fusion and hybrid-blob comparisons.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from crowd_detector import CrowdDetector, thermal_image
from faster_k import FastConfig
from tracking_router import TrackingRouter


def draw(data, router, regions, scale=5, show_regions=False):
    grey = thermal_image(data)[:, :, 0]
    image = cv2.resize(cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO), None,
                       fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for t in router.observed():
        # Actual current omega detector rectangle in BOTH modes. Never body
        # extrapolation, and no predicted/coasting rectangles in faster K.
        x, y, w, h = t.det['bbox']
        color = (90, 230, 120) if t.det.get('source', 'base') == 'base' else (0, 210, 255)
        p = (round(x*scale), round(y*scale))
        cv2.rectangle(image, p, (round((x+w)*scale), round((y+h)*scale)), color, 1)
        cv2.putText(image, f"#{t.id} {t.det.get('conf', 0):.2f}",
                    (p[0], max(12, p[1]-4)), cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1)
    if show_regions and router.mode == 'faster':
        for r in regions:
            p = (round(r['cx']*scale), round(r['cy']*scale))
            cv2.circle(image, p, round(r['radius']*scale), (180, 160, 110), 1)
            cv2.putText(image, f"search #{r['id']}", p, cv2.FONT_HERSHEY_SIMPLEX,
                        .35, (180, 160, 110), 1)
    panel = np.zeros((94, image.shape[1], 3), np.uint8)
    texts = [f"{router.mode.upper()} K (experimental)   observed {len(router.observed())}",
             f"unresolved {router.unresolved_count()}   " +
             (f"legacy occupancy estimate {router.estimate()}" if router.mode == 'precise' else 'unresolved excluded from observed count'),
             'q quit | space pause | r reset | c search circles']
    for i, text in enumerate(texts):
        cv2.putText(panel, text, (12, 24+i*28), cv2.FONT_HERSHEY_SIMPLEX, .45, (220, 220, 220), 1)
    return np.concatenate([panel, image])


def replay_frames(folder, fps=None, limit=0):
    paths = sorted(Path(folder).glob('*.npy'))
    if not paths:
        raise ValueError(f'no .npy Celsius frames in {folder}')
    times = None
    if fps is None:
        stamp_file = Path(folder).parent/'timestamps.json'
        if stamp_file.is_file():
            pairs = json.loads(stamp_file.read_text())
            times = {p['stem']: float(p['relative_time_s']) for p in pairs}
        else:
            manifest = Path(folder).parent/'pairs.json'
            pairs = json.loads(manifest.read_text())['pairs']
            times = {p['stem']: float(p['thermal_host_receipt_epoch_us'])/1e6 for p in pairs}
        t0 = times[paths[0].stem]
    previous = None
    for i, path in enumerate(paths[:limit or None]):
        data = np.load(path, allow_pickle=False)
        if data.dtype.kind != 'f':
            raise ValueError(f'{path}: use converted Celsius npy/, not thermal_raw/')
        timestamp = times[path.stem]-t0 if times is not None else i/fps
        if not np.isfinite(timestamp) or (previous is not None and timestamp <= previous):
            raise ValueError('replay timestamps must be finite and strictly increasing')
        previous = timestamp
        yield data, timestamp, str(path)


def live_frames(scale, limit=0):
    from lepton_libuvc import LeptonUVC
    cam = LeptonUVC()
    cam.scale = scale
    count = 0
    last_frame = time.monotonic()
    try:
        while not limit or count < limit:
            # No historical queue backlog after slow inference or a pause.
            while cam.q.qsize() > 1:
                cam.q.get_nowait()
            data, radiometric = cam.read()
            now = time.monotonic()
            if data is None:
                if now-last_frame > 15:
                    raise TimeoutError('No thermal frames for 15 seconds')
                continue
            if not radiometric:
                raise ValueError('radiometric thermal input required')
            last_frame = now
            count += 1
            yield data, now, 'live-host-receipt'
    finally:
        cam.release()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tracker', choices=['auto', 'precise', 'faster'], default='auto')
    ap.add_argument('--weights')
    ap.add_argument('--conf', type=float, default=.374)
    ap.add_argument('--recovery-conf', type=float, default=.18)
    ap.add_argument('--recovery-seconds', type=float, default=2.)
    ap.add_argument('--max-speed-px-s', type=float, default=30.)
    ap.add_argument('--max-accel-px-s2', type=float, default=100.)
    ap.add_argument('--max-radius-px', type=float, default=14.)
    ap.add_argument('--size-ratio', type=float, default=1.8)
    ap.add_argument('--max-recovery-rois', type=int, default=6)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', help='YOLO inference device, e.g. cpu or mps')
    ap.add_argument('--replay', help='folder containing Celsius float .npy frames')
    ap.add_argument('--replay-fps', type=float, help='override timestamps with an explicit rate; otherwise use adjacent timestamps.json or pairs.json')
    ap.add_argument('--kelvin-per-count', type=float, choices=[.01, .1], help='live: supply verified fixed TLinear scale')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--show-search', action='store_true')
    ap.add_argument('--save-thermal', action='store_true', help='save native Celsius camera arrays for later replay; not screen recording')
    ap.add_argument('--scale', type=int, default=5)
    ap.add_argument('--output', type=Path, help='new log directory; never overwrite')
    args = ap.parse_args(argv)
    if args.replay_fps is not None and (not np.isfinite(args.replay_fps) or args.replay_fps <= 0):
        ap.error('--replay-fps must be finite and positive')
    if args.replay and args.replay_fps is None and not any(
            (Path(args.replay).parent/name).is_file() for name in ['pairs.json', 'timestamps.json']):
        ap.error('--replay needs adjacent timestamps.json / pairs.json or explicit --replay-fps')
    if not args.replay and args.kelvin_per_count is None:
        ap.error('live input requires --kelvin-per-count from verified TLinear settings')
    if args.limit < 0 or args.scale < 1 or args.imgsz < 32:
        ap.error('invalid limit, scale or image size')
    config = FastConfig(recovery_seconds=args.recovery_seconds, max_speed=args.max_speed_px_s,
                        max_accel=args.max_accel_px_s2, max_radius=args.max_radius_px,
                        size_ratio=args.size_ratio, max_rois=args.max_recovery_rois, low_conf=args.recovery_conf)
    from model_registry import resolve_weights
    from ultralytics import YOLO
    weights = Path(resolve_weights(args.weights)).resolve()
    detector = CrowdDetector(YOLO(str(weights)), args.conf, args.recovery_conf, args.imgsz, args.device)
    out = args.output or Path(__file__).resolve().parent/'logs'/f'crowd_test_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True, exist_ok=False)
    if args.save_thermal:
        (out/'npy').mkdir()
    meta = {'status': 'running', 'experimental': True, 'config': asdict(config),
            'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'weights': str(weights), 'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
            'timestamp_kind': ('explicit-replay-rate' if args.replay_fps else 'recorded-host-receipt') if args.replay else 'host-receipt-not-exposure',
            'started': datetime.now().astimezone().isoformat(), 'frames': 0}
    (out/'session.json').write_text(json.dumps(meta, indent=2))
    print(f'Experimental tracking; logs: {out}\nNo screen recording. Press q to finish.')
    stream = replay_frames(args.replay, args.replay_fps, args.limit) if args.replay else live_frames(args.kelvin_per_count, args.limit)
    router = None
    show_regions = args.show_search
    log = (out/'frames.jsonl').open('w')
    started = time.perf_counter()
    recorded_pairs = []
    try:
        for data, timestamp, origin in stream:
            if data.shape != (120, 160):
                raise ValueError(f'expected native 160x120 Celsius thermal data; got {data.shape}')
            if router is None:
                router = TrackingRouter((160, 120), config, args.tracker)
            t0 = time.perf_counter()
            image = thermal_image(data)
            high = detector.base(image)
            t1 = time.perf_counter()
            regions = router.begin(high, timestamp)
            t2 = time.perf_counter()
            low = detector.recover(image, regions)
            t3 = time.perf_counter()
            router.finish(low)
            t4 = time.perf_counter()
            meta['frames'] += 1
            if args.save_thermal:
                stem = f"cap_{meta['frames']-1:06d}"
                np.save(out/'npy'/f'{stem}.npy', data.astype(np.float32))
                recorded_pairs.append({'stem': stem, 'relative_time_s': timestamp})
            row = {'frame': meta['frames'], 'time': timestamp, 'origin': origin, 'mode': router.mode,
                   'base_detections': high, 'recovery_detections': low, 'regions': regions,
                   'observed': [{'id': t.id, 'bbox': list(t.det['bbox']), 'conf': t.det.get('conf'),
                                 'source': t.det.get('source', 'base')} for t in router.observed()],
                   'unresolved': router.unresolved_count(), 'estimate': router.estimate(),
                   'base_ms': (t1-t0)*1000, 'recovery_ms': (t3-t2)*1000,
                   'tracking_ms': (t2-t1+t4-t3)*1000, 'processing_ms': (t4-t0)*1000,
                   'events': router.events, 'fast_stats': router.fast.stats if router.mode == 'faster' else {}}
            log.write(json.dumps(row)+'\n'); log.flush(); router.events.clear()
            if not args.headless:
                cv2.imshow('FLUXNET experimental tracking v2', draw(data, router, regions, args.scale, show_regions))
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), 27):
                    break
                if key == ord('c'):
                    show_regions = not show_regions
                if key == ord('r'):
                    router = None
                if key == ord(' '):
                    while True:
                        key = cv2.waitKey(100) & 0xff
                        if key in (ord(' '), ord('q'), 27):
                            break
                    if key in (ord('q'), 27):
                        break
                    # Physical movement during a pause is unobserved. Reset the
                    # state, rather than bridge that gap with stale identities.
                    if not args.replay:
                        router = None
        meta['status'] = 'completed'
    except BaseException as error:
        meta['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        meta['error'] = str(error)
        raise
    finally:
        stream.close(); log.close()
        if not args.headless:
            cv2.destroyAllWindows()
        meta['elapsed_seconds'] = time.perf_counter()-started
        if args.save_thermal:
            (out/'timestamps.json').write_text(json.dumps(recorded_pairs, indent=2))
        (out/'session.json').write_text(json.dumps(meta, indent=2))
        # Return only this session's generated logs to the invoking user.
        if hasattr(os, 'geteuid') and os.geteuid() == 0 and os.environ.get('SUDO_UID'):
            uid, gid = int(os.environ['SUDO_UID']), int(os.environ['SUDO_GID'])
            for path in [*out.rglob('*'), out]:
                os.chown(path, uid, gid)
        print(f"{meta['frames']} frames; {meta['status']}. Logs: {out}")


if __name__ == '__main__':
    main()
