#!/usr/bin/env python3
"""Experimental thermal-only launcher: precise K / faster K, live or replay.

No screen recording. JSON logs contain observations, IDs, regions and timing.
Legacy launcher remains available for radar fusion and hybrid-blob comparisons.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import os
import queue
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
             'q quit | space pause | x reset | r record | c circles']
    for i, text in enumerate(texts):
        cv2.putText(panel, text, (12, 24+i*28), cv2.FONT_HERSHEY_SIMPLEX, .45, (220, 220, 220), 1)
    return np.concatenate([panel, image])


class TrackingSession:
    """Shared processing path for live launcher and offline simulator.

    No camera, clock or UI owned here. Callers supply original frame timestamps.
    """
    def __init__(self, detector, config=None, mode='auto'):
        self.detector = detector
        self.config = config or FastConfig()
        self.mode = mode
        self.reset()

    def reset(self):
        self.router = TrackingRouter((160, 120), self.config, self.mode)
        self.frame = 0

    def set_confidence(self, conf, low_conf):
        if not 0 < low_conf < conf < 1:
            raise ValueError('require 0 < search confidence < detection confidence < 1')
        self.detector.conf=conf;self.detector.low_conf=low_conf
        self.config=replace(self.config,low_conf=low_conf)
        self.router.config=self.config;self.router.fast.config=self.config
        for track in self.router.fast.tracks:track.config=self.config

    def set_mode(self, mode):
        if mode not in ('auto','precise','faster'):raise ValueError('unknown tracker mode')
        # Defer handover until the next frame's timestamp is available.
        self.mode=mode;self.router.policy=mode

    def process(self, data, timestamp, origin='replay'):
        if data.shape != (120, 160) or data.dtype.kind != 'f':
            raise ValueError('expected native 160x120 float Celsius frame')
        t0 = time.perf_counter()
        image = thermal_image(data)
        high = [dict(d,detection_index=i,track_id=None) for i,d in enumerate(self.detector.base(image))]
        t1 = time.perf_counter()
        if self.mode != 'auto' and self.router.mode != self.mode:
            if self.router.previous_time is None:self.router.mode=self.mode
            else:self.router.switch(self.mode,timestamp)
        self.router.begin(high, timestamp)
        t2 = time.perf_counter()
        low = self.detector.recover(image, self.router.regions)
        t3 = time.perf_counter()
        self.router.finish(low)
        t4 = time.perf_counter()
        self.frame += 1
        router = self.router
        # Exact per-frame detection provenance survives the track payload copy.
        # Only currently displayed/confirmed base tracks give raw boxes an ID;
        # recovery observations must not remove an unrelated full-frame box.
        for track in router.observed():
            if track.det.get('source','base')=='base' and track.misses==0:
                index=track.det.get('detection_index')
                if index is not None and 0<=index<len(high):high[index]['track_id']=track.id
        row = {'frame': self.frame, 'time': timestamp, 'origin': origin, 'mode': router.mode,
               'normal_confidence': getattr(self.detector,'conf',None),
               'recovery_confidence': getattr(self.detector,'low_conf',None),
               'tracker_policy': self.mode, 'base_detections': high, 'recovery_detections': low, 'regions': router.regions,
               'observed': [{'id': t.id, 'bbox': list(t.det['bbox']), 'conf': t.det.get('conf'),
                             'source': t.det.get('source', 'base')} for t in router.observed()],
               'unresolved': router.unresolved_count(), 'estimate': router.estimate(),
               'base_ms': (t1-t0)*1000, 'recovery_ms': (t3-t2)*1000,
               'tracking_ms': (t2-t1+t4-t3)*1000, 'processing_ms': (t4-t0)*1000,
               'events': list(router.events),
               'fast_stats': dict(router.fast.stats) if router.mode == 'faster' else {}}
        router.events.clear()
        return row


def replay_frames(folder, fps=None, limit=0):
    from capture_replay import CaptureSequence
    sequence=CaptureSequence(folder,fps)
    for i in range(min(limit or len(sequence),len(sequence))):
        yield sequence.load(i),float(sequence.times[i]),str(sequence.paths[i])


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
    ap.add_argument('--replay', help='capture folder or its npy directory, with recorded timestamps')
    ap.add_argument('--replay-fps', type=float, help='override timestamps with an explicit rate; otherwise use capture timestamps or legacy manifest')
    ap.add_argument('--kelvin-per-count', type=float, choices=[.01, .1], help='live: supply verified fixed TLinear scale')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--show-search', action='store_true')
    ap.add_argument('--save-thermal', action='store_true', help='save native Celsius camera arrays for later replay; not screen recording')
    ap.add_argument('--record',action='store_true',help='start legacy-format thermal dataset recording immediately')
    ap.add_argument('--capture-every',type=int,default=1,help='save one in N processed frames while recording')
    ap.add_argument('--note',default='',help='note saved with dataset captures')
    ap.add_argument('--capture-root',type=Path,default=Path(__file__).resolve().parent/'logs')
    ap.add_argument('--scale', type=int, default=5)
    ap.add_argument('--output', type=Path, help='new log directory; never overwrite')
    args = ap.parse_args(argv)
    if args.replay_fps is not None and (not np.isfinite(args.replay_fps) or args.replay_fps <= 0):
        ap.error('--replay-fps must be finite and positive')
    if not args.replay and args.kelvin_per_count is None:
        ap.error('live input requires --kelvin-per-count from verified TLinear settings')
    if args.limit < 0 or args.scale < 1 or args.imgsz < 32 or args.capture_every < 1:
        ap.error('invalid limit, scale or image size')
    if not 0 < args.recovery_conf < args.conf < 1:
        ap.error('require 0 < recovery confidence < detection confidence < 1')
    config = FastConfig(recovery_seconds=args.recovery_seconds, max_speed=args.max_speed_px_s,
                        max_accel=args.max_accel_px_s2, max_radius=args.max_radius_px,
                        size_ratio=args.size_ratio, max_rois=args.max_recovery_rois, low_conf=args.recovery_conf)
    if args.headless:
        run_capture(args, config)
    else:
        from launcher_desktop import run_desktop
        run_desktop(lambda bridge: run_capture(args, config, bridge), args)


def run_capture(args, config, bridge=None):
    """Own inference, camera, tracker and recording on one worker thread.

    The native UI receives immutable snapshots and queues changes between frames.
    Headless callers use the same processing and recording path synchronously.
    """
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
    print(f'Experimental tracking; logs: {out}\nNo screen recording. R record | S single frame | C circles | H help | Q quit.')
    stream = replay_frames(args.replay, args.replay_fps, args.limit) if args.replay else live_frames(args.kelvin_per_count, args.limit)
    from launcher_controls import DatasetCapture, LauncherControls
    session = TrackingSession(detector, config, args.tracker)
    recorder = DatasetCapture(args.capture_root,weights,args.note)
    ui = LauncherControls(session,recorder,args.show_search,live=not bool(args.replay))
    from heat_function import HeatFunction,HeatConfig
    heat_function=HeatFunction();ui.heat_config=asdict(heat_function.config)
    meta['heat_config']=ui.heat_config
    log = (out/'frames.jsonl').open('w')
    actions = (out/'controls.jsonl').open('w')
    started = time.perf_counter()
    recorded_pairs = []
    data=row=None;record_pending=args.record;record_index=0
    try:
        if bridge:bridge.publish(ui=ui)
        while not ui.quit and not (bridge and bridge.stop.is_set()):
            if bridge:
                changed=False
                while True:
                    try:request_id,action=bridge.commands.get_nowait()
                    except queue.Empty:break
                    if isinstance(action,dict) and action.get('action')=='heat_config':
                        heat_function=HeatFunction(HeatConfig(**action['values']))
                        ui.heat_config=asdict(heat_function.config);meta['heat_config']=ui.heat_config
                        ui.events.append(dict(action='heat_config',frame=meta['frames'],values=ui.heat_config))
                        ui.message='Heat settings updated; gate restarted'
                    else:ui.command(action,data,row)
                    if action==ord('r'):record_index=0
                    changed=True
                if changed:bridge.publish(ui=ui,ack=request_id)
                if ui.quit or bridge.stop.is_set():break
            if not ui.paused or ui.step or row is None:
                ui.step=False
                try:data,timestamp,origin=next(stream)
                except StopIteration:break
                received=datetime.now().isoformat(timespec='milliseconds')
                row=session.process(data,timestamp,origin)
                meta['frames']+=1;row['frame']=meta['frames'];row['received_at']=received
                if session.frame==1:heat_function.reset()
                heat_function.update(data,row)
                ui.observe(row)
                if args.save_thermal:
                    stem=f"cap_{meta['frames']-1:06d}"
                    np.save(out/'npy'/f'{stem}.npy',data.astype(np.float32))
                    recorded_pairs.append(dict(stem=stem,relative_time_s=timestamp))
                log.write(json.dumps(row)+'\n');log.flush()
                if record_pending:
                    ui.command(ord('r'),data,row);record_pending=False
                if recorder.continuous:
                    if record_index % args.capture_every==0:recorder.write(recorder.continuous,data,row)
                    record_index+=1
                else:record_index=0
                if bridge:bridge.publish(ui=ui,row=row,data=data)
            elif bridge:
                # The UI remains interactive while this worker waits or infers.
                bridge.stop.wait(.02)
            for event in ui.events:
                actions.write(json.dumps(event)+'\n')
            if ui.events:actions.flush();ui.events.clear()
        meta['status'] = 'completed'
    except BaseException as error:
        meta['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        meta['error'] = str(error)
        raise
    finally:
        stream.close(); log.close(); actions.close(); recorder.close()
        meta['dataset_captures']=[str(p) for p in recorder.folders]
        meta['dataset_frames']=recorder.total
        if bridge:bridge.publish(ui=ui)
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
        for folder in recorder.folders:print(f'Dataset capture: {folder}')


if __name__ == '__main__':
    main()
