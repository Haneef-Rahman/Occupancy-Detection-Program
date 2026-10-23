"""One serialized processing worker per workspace; HTTP never owns a camera."""
from dataclasses import asdict,replace
from datetime import datetime
from pathlib import Path
import base64, copy, hashlib, json, math, queue, threading, time, uuid
import cv2
import numpy as np
from capture_replay import CaptureSequence
from crowd_detector import CrowdDetector
from faster_k import FastConfig
from heat_function import HeatConfig, HeatFunction
from occupancy_chart import OccupancyHistory
from launcher_controls import DatasetCapture
from .processor import Processor
from .camera import CameraStream


def png(data, palette, contrast, heat_config, heat_band=False):
    if heat_band:
        c=heat_config
        gray=(np.clip((data-c.low_c)/(c.high_c-c.low_c),0,1)*255).astype(np.uint8)
        image=cv2.applyColorMap(gray,cv2.COLORMAP_INFERNO)
        image[(data<c.low_c)|(data>c.high_c)]=0
    elif palette=='inferno':
        from dataset_pipeline import colorize_view
        image=colorize_view(data,contrast)
    else:
        gray=(np.clip((data-15)/30,0,1)*255).astype(np.uint8)
        image=cv2.applyColorMap(gray,cv2.COLORMAP_VIRIDIS) if palette=='viridis' else cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    ok,encoded=cv2.imencode('.png',image)
    if not ok:raise RuntimeError('Could not encode thermal preview.')
    return base64.b64encode(encoded).decode('ascii')


class Recorder(DatasetCapture):
    def create(self,kind,row):
        cap=super().create(kind,row)
        source=cap['path']/'source.txt'
        source.write_text(source.read_text().replace('integrated_launcher_v2.py','web_workspace.py'))
        return cap


class Workspace:
    def __init__(self, thermal, kind, weights=None, detector_factory=None, camera_factory=None):
        self.thermal=Path(thermal);self.kind=kind;self.weights=weights
        self.detector_factory=detector_factory;self.camera_factory=camera_factory or CameraStream
        self.commands=queue.Queue(maxsize=64);self.lock=threading.Lock();self.io_lock=threading.Lock()
        self.stop=threading.Event();self.camera=None;self.sequence=None;self.processor=None;self.model=None
        self.heat=HeatFunction();self.history=OccupancyHistory();self.data=None;self.row=None
        self.conf=.374;self.low_conf=.18;self.mode='auto';self.device='auto';self.config=FastConfig()
        self.palette='inferno';self.contrast=0;self.heat_band=False
        self.paused=True;self.steps=0;self.index=-1;self.speed=1.;self.seek_target=None;self.next_due=0.
        self.recorder=None;self.report=None;self.log=None;self.start_time=None
        self.state=dict(version=0,frame_version=0,status='Ready',error='',loaded=False,paused=True,
            row=None,image=None,history=[],overview=[],index=-1,total=0,speed=1.,recording=False,saved=0,
            source=None,conf=self.conf,low_conf=self.low_conf,policy=self.mode,heat_config=asdict(self.heat.config),tracker_config=asdict(self.config),
            palette=self.palette,contrast=0,heat_band=False,seeking=False,device='auto',report=None)
        self.thread=threading.Thread(target=self.run,daemon=True,name='web-'+kind)
        self.thread.start()

    def publish(self,**values):
        # The processing worker keeps mutating its history. Published snapshots
        # must not share those lists with the HTTP threads.
        with self.lock:self.state.update(copy.deepcopy(values));self.state['version']+=1

    def snapshot(self):
        with self.lock:return copy.deepcopy(self.state)

    def submit(self,command):
        if self.stop.is_set():raise ValueError('Workspace is stopped.')
        try:self.commands.put_nowait(command)
        except queue.Full:raise ValueError('Controls are busy; please try again.')

    def close_source(self):
        if self.camera:self.camera.close()
        self.camera=None;self.sequence=None
        if self.recorder:self.recorder.close()
        self.recorder=None
        with self.io_lock:
            if self.log:self.log.close()
            self.log=None

    def new_processor(self):
        self.publish(status='Loading thermal model…')
        if self.detector_factory:
            detector=self.detector_factory(self.conf,self.low_conf)
            self.weight_path='test detector'
        else:
            from model_registry import resolve_weights
            from ultralytics import YOLO
            self.weight_path=resolve_weights(self.weights,quiet=True)
            if self.model is None:self.model=YOLO(self.weight_path)
            device=self.device
            if device=='auto':
                import torch
                device='0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            detector=CrowdDetector(self.model,self.conf,self.low_conf,device=device)
        self.processor=Processor(detector,self.config,self.mode)
        self.heat.reset();self.history.clear();self.index=-1;self.data=None;self.row=None;self.seek_target=None
        self.recorder=Recorder(self.thermal/'logs',self.weight_path,'Browser '+self.kind+' workspace')
        self.report=self.thermal/'logs'/'web_workspace'/('session_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        self.report.mkdir(parents=True,exist_ok=False)
        metadata=dict(workspace=self.kind,source=self.state['source'],weights=self.weight_path,
            weights_sha256=hashlib.sha256(Path(self.weight_path).read_bytes()).hexdigest() if Path(self.weight_path).is_file() else None,
            heat_config=asdict(self.heat.config),tracker_config=asdict(self.config),device=self.device,
            note='Host receipt/replay timestamps; not exposure synchronization. Heat/pD are diagnostics, not ground truth.')
        (self.report/'session.json').write_text(json.dumps(metadata,indent=2))
        self.log=(self.report/'frames.jsonl').open('w')
        self.publish(report=str(self.report),index=-1,row=None,image=None,history=[],overview=[],saved=0,recording=False)

    def command(self,c):
        action=c.get('action')
        if action=='load':
            if self.kind!='simulator':raise ValueError('Use the Simulator tab for saved captures.')
            path=Path(c['path']).expanduser()
            if not path.is_absolute():path=self.thermal/path
            fps=c.get('fps');sequence=CaptureSequence(path,float(fps) if fps else None)
            if (self.thermal/'logs'/'web_workspace').resolve().is_relative_to(sequence.root):
                raise ValueError('Choose a capture folder, not the whole logs directory.')
            self.close_source();self.sequence=sequence;self.paused=True;self.steps=0
            self.publish(source=dict(kind='replay',**sequence.info()),loaded=True,total=len(sequence),error='',paused=True)
            self.new_processor();self.process_next()
            self.publish(status='Paused — ready to replay')
        elif action=='connect':
            if self.kind!='launcher':raise ValueError('Connect Lepton in the Launcher tab.')
            scale=float(c.get('scale',.01));index=int(c.get('camera_index',0))
            if scale not in (.01,.1) or not 0<=index<=15:raise ValueError('Invalid camera index or TLinear scale.')
            self.close_source();self.publish(status='Opening Lepton…',error='')
            self.device=c.get('device','auto')
            if self.device not in ('auto','cpu','mps','0'):raise ValueError('Invalid inference device.')
            self.camera=self.camera_factory(index=index,scale=scale)
            self.paused=False;self.start_time=None
            self.publish(source=dict(kind='live',backend=self.camera.backend,kelvin_per_count=scale,
                timestamp_kind='host-receipt-not-exposure'),loaded=True,total=0,paused=False,device=self.device)
            self.new_processor();self.publish(status='Lepton connected')
        elif action=='disconnect':
            self.close_source();self.paused=True;self.processor=None;self.seek_target=None
            self.publish(loaded=False,paused=True,recording=False,seeking=False,status='Disconnected')
        elif action=='pause':
            self.paused=bool(c.get('value',not self.paused));self.steps=0;self.seek_target=None
            self.publish(paused=self.paused,seeking=False,status='Paused' if self.paused else 'Running')
        elif action=='step':
            self.paused=True;self.steps=1;self.publish(paused=True,status='Stepping')
        elif action=='speed':
            speed=float(c['value'])
            if speed not in (.25,.5,1,2,4,16):raise ValueError('Invalid replay speed.')
            self.speed=speed;self.next_due=0.;self.publish(speed=speed)
        elif action in ('reset','seek'):
            if not self.processor:raise ValueError('Open a source first.')
            if self.sequence:
                target=max(0,min(len(self.sequence)-1,int(c.get('index',0))))
                if action=='reset' or target<=self.index:
                    if self.recorder:self.recorder.close()
                    with self.io_lock:
                        if self.log:self.log.close();self.log=None
                    self.new_processor()
                self.seek_target=target;self.paused=True;self.steps=0
                self.publish(seeking=True,paused=True,status='Seeking — replaying intermediate frames')
            else:
                self.processor.reset();self.heat.reset();self.history.clear();self.index=-1
                self.publish(history=[],overview=[],status='Tracker and heat history reset')
        elif action=='confidence':
            conf=float(c['conf']);low=float(c['low_conf'])
            if not 0<low<conf<1:raise ValueError('Search confidence must be below detection confidence.')
            self.conf,self.low_conf=conf,low
            self.config=replace(self.config,low_conf=low)
            if self.processor:self.processor.set_confidence(conf,low)
            self.publish(conf=conf,low_conf=low,status='Confidence updated')
        elif action=='mode':
            mode=c['value']
            if mode not in ('auto','precise','faster'):raise ValueError('Invalid tracking policy.')
            self.mode=mode
            if self.processor:self.processor.set_mode(mode)
            self.publish(policy=mode,status='Tracking policy updated')
        elif action=='heat':
            self.heat=HeatFunction(HeatConfig(**c['values']))
            self.publish(heat_config=asdict(self.heat.config),status='Heat settings updated; gate restarted')
            self.refresh_image()
        elif action=='tracker_options':
            values=c['values']
            if set(values)!={'recovery_seconds','max_radius','max_speed'}:raise ValueError('Invalid tracker options.')
            values={key:float(value) for key,value in values.items()}
            if not all(math.isfinite(value) for value in values.values()):raise ValueError('Tracker options must be finite.')
            if not .1<=values['recovery_seconds']<=10 or not 1<=values['max_radius']<=80 or not 1<=values['max_speed']<=1000:raise ValueError('Tracker options are out of range.')
            self.config=replace(self.config,**values,low_conf=self.low_conf)
            if self.processor:
                self.recorder.close()
                with self.io_lock:
                    if self.log:self.log.close();self.log=None
                self.paused=True;self.new_processor()
                if self.sequence:self.process_next()
            self.publish(tracker_config=asdict(self.config),paused=self.paused,status='Tracker options applied; history restarted')
        elif action=='display':
            palette=c.get('palette',self.palette);contrast=int(c.get('contrast',self.contrast))
            if palette not in ('inferno','viridis','gray') or contrast not in (0,1,2):raise ValueError('Invalid display option.')
            self.palette,self.contrast=palette,contrast;self.heat_band=bool(c.get('heat_band',self.heat_band))
            self.publish(palette=palette,contrast=contrast,heat_band=self.heat_band)
            self.refresh_image()
        elif action in ('record','shot'):
            if self.row is None or self.recorder is None:raise ValueError('Process a frame first.')
            self.row['received_at']=self.row.get('received_at',datetime.now().isoformat(timespec='milliseconds'))
            if action=='record':message=self.recorder.toggle(self.row)
            else:self.recorder.shot(self.data,self.row);message='Frame saved'
            self.publish(recording=bool(self.recorder.continuous),saved=self.recorder.total,status=message)
        else:raise ValueError('Unknown command.')

    def refresh_image(self):
        if self.data is not None:
            self.publish(image=png(self.data,self.palette,self.contrast,self.heat.config,self.heat_band))

    def process_next(self):
        if not self.processor:return
        if self.sequence:
            index=self.index+1
            if index>=len(self.sequence):
                self.paused=True;self.steps=0;self.seek_target=None
                self.publish(paused=True,seeking=False,status='Replay finished');return
            data=self.sequence.load(index);stamp=float(self.sequence.times[index]);origin=str(self.sequence.paths[index])
        elif self.camera:
            frame=self.camera.read()
            if frame is None:return
            data,received=frame
            if self.start_time is None:self.start_time=received
            stamp=received-self.start_time;origin='live-host-receipt';index=self.index+1
        else:return
        row=self.processor.process(data,stamp,origin)
        row['received_at']=datetime.now().isoformat(timespec='milliseconds')
        row['temperature']=dict(min=float(data.min()),max=float(data.max()))
        row['heat_config']=asdict(self.heat.config)
        self.heat.update(data,row);self.history.append(row)
        self.data,self.row,self.index=data,row,index
        with self.io_lock:
            if self.log:self.log.write(json.dumps(row)+'\n');self.log.flush()
        if self.recorder.continuous:self.recorder.write(self.recorder.continuous,data,row)
        if self.steps:self.steps-=1
        if self.seek_target is not None and self.index>=self.seek_target:
            self.seek_target=None;self.publish(status='Paused at selected frame',seeking=False)
        if self.sequence and self.index==len(self.sequence)-1:
            self.paused=True;self.seek_target=None;self.publish(paused=True,seeking=False,status='Replay finished')
        delay=0.
        if self.sequence and self.index+1<len(self.sequence):delay=(self.sequence.times[self.index+1]-stamp)/self.speed
        self.next_due=time.monotonic()+delay
        with self.lock:version=self.state['frame_version']+1
        self.publish(row=row,image=png(data,self.palette,self.contrast,self.heat.config,self.heat_band),index=index,
            frame_version=version,history=list(self.history.samples),overview=self.history.overview,
            recording=bool(self.recorder.continuous),saved=self.recorder.total,paused=self.paused)

    def run(self):
        while not self.stop.is_set():
            try:
                try:command=self.commands.get(timeout=.015)
                except queue.Empty:command=None
                if command is not None:
                    try:
                        self.publish(error='')
                        self.command(command)
                    except BaseException as error:
                        # Model loaders historically use SystemExit; surface it in-page.
                        if isinstance(error,KeyboardInterrupt):raise
                        if command.get('action') in ('connect','load'):
                            self.close_source();self.processor=None;self.paused=True;self.publish(loaded=False,paused=True,recording=False)
                        self.publish(error=str(error),status='Action failed')
                active=self.processor and (self.camera or self.sequence)
                if active and (self.seek_target is not None or self.steps or not self.paused):
                    if self.seek_target is not None or self.steps or self.camera or time.monotonic()>=self.next_due:
                        self.process_next()
            except Exception as error:
                self.paused=True;self.close_source();self.processor=None
                self.publish(error=str(error),status='Stopped — check the source',loaded=False,paused=True,recording=False)
        self.close_source()

    def export(self):
        with self.io_lock:
            if not self.report:raise ValueError('No processed session to export.')
            metadata=json.loads((self.report/'session.json').read_text())
            frames=[json.loads(line) for line in (self.report/'frames.jsonl').read_text().splitlines() if line.strip()]
        return dict(metadata=metadata,current_heat_config=asdict(self.heat.config),frames=frames)

    def close(self):
        self.stop.set();self.thread.join(timeout=20)
        if self.thread.is_alive():raise RuntimeError('A camera/model operation is still finishing.')
