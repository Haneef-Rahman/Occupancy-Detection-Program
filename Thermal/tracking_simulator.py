#!/usr/bin/env python3
"""Local capture replay simulator, using integrated_launcher_v2 as its engine.

Never opens a camera, reads a desktop screen, or edits the source capture.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import uuid

import cv2
import numpy as np
import integrated_launcher_v2 as engine
from capture_replay import CaptureSequence, discover

HERE = Path(__file__).resolve().parent


class RunReport:
    def __init__(self, root, sequence, config, mode, weights):
        self.path = Path(root)/f'simulation_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
        self.path.mkdir(parents=True, exist_ok=False)
        self.rows = (self.path/'frames.jsonl').open('w')
        self.meta = dict(capture=sequence.info(), config=asdict(config), tracker=mode,
                         weights=str(weights), weights_sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
                         started=datetime.now().astimezone().isoformat(), status='running',
                         source_unchanged=True, accuracy_measured=False,
                         engine_files={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                       for name in ['integrated_launcher_v2.py','tracking_router.py','faster_k.py','precise_k.py','crowd_detector.py']
                                       if (HERE/name).is_file()})
        self.modes = Counter(); self.ids = set(); self.observed = []; self.processing = []; self.recovery = []
        self.tracking = []; self.unresolved = []; self.model_base_ms = []
        self.switches = 0
        self.write_summary()

    def append(self, row):
        self.rows.write(json.dumps(row)+'\n'); self.rows.flush()
        self.modes[row['mode']] += 1
        self.ids.update(t['id'] for t in row['observed'])
        self.observed.append(len(row['observed']))
        self.unresolved.append(row['unresolved'])
        self.processing.append(row['processing_ms'])
        self.recovery.append(row['recovery_ms'])
        self.tracking.append(row['tracking_ms'])
        self.model_base_ms.append(row['base_ms'])
        self.switches += len(row['events'])

    def write_summary(self, status=None):
        if status:
            self.meta['status'] = status
        result = dict(self.meta, processed_frames=len(self.processing), modes=dict(self.modes),
                      unique_observed_ids=len(self.ids), mode_switches=self.switches,
                      interpretation='Detection/tracking diagnostics only. Existing labels are not treated as identity/count ground truth.')
        if self.processing:
            result.update(observed_min=min(self.observed), observed_max=max(self.observed),
                          observed_mean=float(np.mean(self.observed)), maximum_unresolved=max(self.unresolved),
                          timing_ms={name: {'median':float(np.median(values)), 'p95':float(np.percentile(values,95))}
                                     for name,values in [('processing',self.processing), ('base_detector',self.model_base_ms),
                                                         ('recovery_detector',self.recovery), ('tracking',self.tracking)]},
                          timing_note='Measured on this computer; includes first-frame warm-up; excludes playback waiting, rendering and log writes.')
        tmp = self.path/'summary.tmp'
        tmp.write_text(json.dumps(result, indent=2)); tmp.replace(self.path/'summary.json')
        return result

    def close(self, status):
        if not self.rows.closed:
            self.write_summary(status)
            self.rows.close()


class Simulator:
    """All model/tracker mutations belong to one worker, including reset/seek."""
    def __init__(self, logs, output, weights, device='cpu', model_factory=None):
        from heat_function import HeatFunction
        self.heat_function=HeatFunction()
        self.reset_global_ids=True
        self.native_controls=None
        self.logs = Path(logs); self.output = Path(output); self.weights = str(weights)
        self.device = device; self.model_factory = model_factory
        self.lock = threading.Lock(); self.commands = queue.Queue()
        self.state = dict(status='Choose a capture', playing=False, loaded=False, index=-1,
                          total=0, version=0, speed=1., image_ready=False, error='', run=None, seek_target=None, seek_request=0)
        self.image = None; self.model = None; self.sequence = None; self.session = None; self.report = None
        self.settings = {}; self.index = -1; self.speed = 1.; self.playing = False; self.steps = 0
        self.seek_target = None; self.next_due = 0.; self.closed = False
        self.thread = threading.Thread(target=self.worker, daemon=True); self.thread.start()

    def publish(self, **values):
        with self.lock:
            self.state.update(values)
            self.state['native_version']=self.state.get('native_version',0)+1

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def picture(self):
        with self.lock:
            return self.image

    def set_picture(self, image):
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError('Could not render frame')
        with self.lock:
            self.image = encoded.tobytes()
            self.state['version'] += 1
            self.state['image_ready'] = True

    def submit(self, command):
        self.commands.put(command)

    def close(self):
        self.submit({'action':'shutdown'})
        self.thread.join(timeout=15)

    def reset(self, status='restarted'):
        if self.report:
            self.report.close(status)
        self.heat_function.reset()
        self.report = None; self.session = None
        self.index = -1; self.steps = 0; self.seek_target = None; self.playing = False
        # This worker is the only tracking session in this process. Reset ID
        # numbering so repeated runs can be compared reproducibly.
        from tracker import KalmanTrack
        if self.reset_global_ids:KalmanTrack._next_id = 1
        if self.native_controls:self.native_controls.recorder.close()
        self.native_controls=None
        self.publish(index=-1, playing=False, seek_target=None, row=None, data=None, history=[], controls=None, run=None, error='', status='Ready — press Play or Step')
        self.set_picture(self.preview(self.sequence.load(0)))

    @staticmethod
    def preview(data):
        image = cv2.applyColorMap(engine.thermal_image(data)[:,:,0], cv2.COLORMAP_INFERNO)
        return cv2.resize(image, (800,600), interpolation=cv2.INTER_NEAREST)

    def handle(self, cmd):
        action = cmd['action']
        if action == 'shutdown':
            self.closed = True
            if self.native_controls:self.native_controls.recorder.close()
            if self.report:
                self.report.close('stopped')
            return
        if action == 'load':
            self.publish(status='Reading capture…', error='')
            sequence = CaptureSequence(cmd['folder'], cmd.get('fps'))
            if self.output.resolve().is_relative_to(sequence.root):
                raise ValueError('Results directory must be outside the source capture.')
            mode = cmd.get('mode','auto')
            if mode not in ('auto','precise','faster'):
                raise ValueError('Unknown tracking mode')
            conf = float(cmd.get('conf', .374)); low = float(cmd.get('low_conf', .18))
            if not 0 < low < conf < 1:
                raise ValueError('Require 0 < recovery confidence < normal confidence < 1.')
            config = engine.FastConfig(recovery_seconds=float(cmd.get('window',2)),
                                       max_speed=float(cmd.get('max_speed',30)),
                                       max_radius=float(cmd.get('radius',14)), low_conf=low)
            self.settings = dict(config=config, mode=mode, conf=conf, low_conf=low)
            self.sequence = sequence
            self.reset('replaced')
            self.publish(loaded=True, total=len(sequence), capture=sequence.info(), settings={
                **{k:v for k,v in self.settings.items() if k!='config'}, 'config':asdict(config)}, weights=self.weights)
            return
        if action=='heat_config':
            from heat_function import HeatFunction,HeatConfig
            self.heat_function=HeatFunction(HeatConfig(**cmd['values']))
            if self.report:self.report.meta.setdefault('heat_configuration_changes',[]).append(dict(frame=self.index+1,values=cmd['values']))
            self.publish(status='Heat settings updated; gate restarted',heat_config=cmd['values']);return
        if not self.sequence:
            raise ValueError('Load a capture first.')
        if action == 'control':
            command=cmd['command']
            if self.native_controls:
                self.native_controls.command(command,self.state.get('data'),self.state.get('row'))
                self.settings['conf']=self.detector.conf;self.settings['low_conf']=self.detector.low_conf;self.settings['mode']=self.session.mode;self.settings['config']=self.session.config
                if self.report:
                    self.report.meta.setdefault('control_events',[]).extend(self.native_controls.events)
                    self.native_controls.events.clear()
                self.publish_native()
            elif isinstance(command,dict):
                conf=self.settings['conf'];low=self.settings['low_conf'];value=float(command['value'])
                if command['slider']=='conf':conf=min(.99,max(low+.01,value))
                else:low=min(conf-.01,max(.01,value))
                self.settings.update(conf=conf,low_conf=low,config=replace(self.settings['config'],low_conf=low))
            elif chr(command) in '123t':
                self.settings['mode']={'1':'auto','2':'precise','3':'faster'}.get(chr(command),('auto','precise','faster')[(('auto','precise','faster').index(self.settings['mode'])+1)%3])
            return
        if action == 'speed':
            speed = float(cmd['value'])
            if speed not in (.25,.5,1.,2.,4.,16.):
                raise ValueError('Unsupported playback speed')
            self.speed = speed; self.next_due = time.monotonic()
            self.publish(speed=speed)
        elif action == 'play':
            if self.index >= len(self.sequence)-1 or (self.report and self.report.rows.closed):
                self.reset()
            self.playing = True; self.steps = 0; self.seek_target = None
            self.next_due = time.monotonic()
            self.publish(playing=True, seek_target=None, status='Playing', error='')
        elif action == 'pause':
            self.playing = False; self.seek_target = None; self.steps = 0
            if self.report and not self.report.rows.closed:
                self.report.write_summary('paused')
            self.publish(playing=False, seek_target=None, status='Paused')
        elif action == 'step':
            self.playing = False; self.seek_target = None
            self.steps += 1; self.next_due = 0
            self.publish(playing=False, seek_target=None, status='Stepping')
        elif action == 'restart':
            self.reset()
        elif action == 'seek':
            target = int(cmd['index'])
            if not 0 <= target < len(self.sequence):
                raise ValueError('Frame outside capture')
            if target <= self.index:
                self.reset('seek-replay')
            self.seek_target = target; self.playing = False; self.steps = 0
            self.publish(playing=False, seek_target=target, seek_request=cmd.get('request', 0),
                         status=f'Replaying context to frame {target+1}…')
        elif action == 'export':
            if self.report:
                if not self.report.rows.closed:
                    self.report.write_summary('running' if self.playing else 'paused')
                self.publish(status=f'Results saved: {self.report.path}')
        else:
            raise ValueError('Unknown action')

    def process_next(self):
        if self.index+1 >= len(self.sequence):
            self.playing = False; self.steps = 0; self.seek_target = None
            if self.report:
                self.report.close('completed')
            self.publish(playing=False, seek_target=None, status='Finished — results saved')
            return
        if self.session is None:
            self.publish(status='Loading YOLO and preparing test…')
            if self.model is None:
                if self.model_factory is None:
                    from ultralytics import YOLO
                    self.model = YOLO(self.weights)
                else:
                    self.model = self.model_factory(self.weights)
            self.detector = engine.CrowdDetector(self.model, self.settings['conf'],
                                                 self.settings['low_conf'], device=self.device)
            self.session = engine.TrackingSession(self.detector, self.settings['config'], self.settings['mode'])
            self.report = RunReport(self.output, self.sequence, self.settings['config'], self.settings['mode'], self.weights)
            self.report.meta['normal_confidence'] = self.settings['conf']
            from launcher_controls import LauncherControls,DatasetCapture
            self.native_controls=LauncherControls(self.session,DatasetCapture(self.output/'captures',self.weights,'Simulator replay'),live=False)
            self.publish(run=str(self.report.path))
        frame_start = time.monotonic()
        i = self.index+1
        data = self.sequence.load(i)
        row = self.session.process(data, float(self.sequence.times[i]), str(self.sequence.paths[i]))
        row['input_array_sha256'] = hashlib.sha256(data.tobytes()).hexdigest()
        self.heat_function.update(data,row)
        self.report.meta['heat_config']=asdict(self.heat_function.config)
        self.report.append(row)
        self.native_controls.observe(row)
        if self.native_controls.recorder.continuous:self.native_controls.recorder.write(self.native_controls.recorder.continuous,data,row)
        self.publish_native(row,data)
        self.index = i
        # Render through the same launcher; remove its native keyboard footer,
        # since native controls belong to this simulator.
        image = engine.draw(data, self.session.router, self.session.router.regions, scale=5, show_regions=True)
        image[63:94] = 0
        cv2.putText(image, f"Frame {i+1}/{len(self.sequence)} | recorded t={row['time']:.3f}s",
                    (12,84), cv2.FONT_HERSHEY_SIMPLEX, .45, (220,220,220), 1)
        self.set_picture(image)
        status = 'Playing' if self.playing else 'Paused'
        if self.seek_target is not None:
            status = f'Replaying context: {i+1} / {self.seek_target+1}'
            if i >= self.seek_target:
                self.seek_target = None; status = 'Paused at selected frame'
        if self.steps:
            self.steps -= 1
        self.publish(index=i, seek_target=self.seek_target, row=row, status=status, playing=self.playing, error='')
        if (i+1) % 25 == 0:
            self.report.write_summary('running')
        if i == len(self.sequence)-1:
            self.report.close('completed'); self.playing=False; self.seek_target=None; self.steps=0
            self.publish(playing=False, seek_target=None, status='Finished — results saved')
        elif self.playing:
            gap = float(self.sequence.times[i+1]-self.sequence.times[i])/self.speed
            # Never skip frames to catch up. Slow inference slows the viewer;
            # timestamps passed to the tracker still come from the recording.
            self.next_due = max(time.monotonic(), frame_start+gap)

    def publish_native(self,row=None,data=None):
        ui=self.native_controls
        if ui is None:return
        d=ui.session.detector
        values=dict(controls=dict(conf=d.conf,low_conf=d.low_conf,mode=ui.session.mode,paused=not self.playing,recording=bool(ui.recorder.continuous),saved=ui.recorder.total),history=[dict(s) for s in ui.history.samples],overview=list(ui.history.overview),heat_config=asdict(self.heat_function.config))
        if row is not None:values.update(row=row,data=data.copy(),native_frame_version=self.state.get('native_frame_version',0)+1)
        self.publish(**values)

    def worker(self):
        while not self.closed:
            try:
                try:
                    command = self.commands.get(timeout=.015)
                except queue.Empty:
                    command = None
                if command:
                    self.handle(command)
                    self.publish(native_ack=command.get('native_request',self.state.get('native_ack',0)))
                if self.closed:
                    break
                active = self.sequence is not None and (self.steps or self.seek_target is not None or
                         (self.playing and time.monotonic() >= self.next_due))
                if active:
                    self.process_next()
            except Exception as error:
                self.playing=False; self.steps=0; self.seek_target=None
                if self.report:
                    self.report.close('failed')
                self.publish(playing=False, seek_target=None, error=str(error), status='Stopped — check error')


class SimulatorWindow:
    """Native Python/Tk UI; model work remains off the UI thread."""
    def __init__(self, root, simulator):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk; self.ttk = ttk; self.root = root; self.sim = simulator
        self.pending_seek = None; self.seek_request = 0
        self.current = {}; self.version = -1; self.dragging = False; self.photo = None
        self.closed = False; self.last_image_size = None; self.image_bytes = None; self.local_error = ''
        root.title('FLUXNET — Tracking simulator')
        root.geometry(f'{min(1320,root.winfo_screenwidth()-80)}x{min(900,root.winfo_screenheight()-100)}')
        root.minsize(960,650); root.configure(bg='#11171d')
        style = ttk.Style(root); style.theme_use('clam')
        style.configure('.',background='#1b242d',foreground='#e5edf3',font=('Helvetica',12))
        style.configure('TFrame',background='#1b242d')
        style.configure('TLabel',background='#1b242d',foreground='#e5edf3')
        style.configure('Muted.TLabel',foreground='#a6b7c4',font=('Helvetica',10))
        style.configure('Title.TLabel',font=('Helvetica',20,'bold'))
        style.configure('Metric.TLabel',font=('Helvetica',18,'bold'),foreground='#a1e1c7')
        style.configure('TButton',padding=(10,7),background='#32434f')
        style.map('TButton',background=[('active','#455e6e')],foreground=[('disabled','#70808c')])
        style.configure('TEntry',fieldbackground='#263744',foreground='#f0f4f7',insertcolor='white')
        style.configure('TCombobox',fieldbackground='#263744',foreground='#f0f4f7')
        style.map('TCombobox',fieldbackground=[('readonly','#263744')],foreground=[('readonly','#f0f4f7')])
        style.configure('TScale',background='#1b242d',troughcolor='#34434e')
        root.columnconfigure(1,weight=1); root.rowconfigure(1,weight=1)
        header=ttk.Frame(root,padding=(22,16)); header.grid(row=0,column=0,columnspan=2,sticky='ew',padx=12,pady=(12,0))
        ttk.Label(header,text='FLUXNET · Tracking simulator',style='Title.TLabel').pack(anchor='w')
        ttk.Label(header,text='Saved thermal frames → integrated launcher v2 · no camera or screen recording',style='Muted.TLabel').pack(anchor='w',pady=(6,0))
        left=ttk.Frame(root,padding=16);left.grid(row=1,column=0,sticky='ns',padx=(12,8),pady=12)
        left.columnconfigure(0,weight=1)
        self.folder=tk.StringVar();self.selected=tk.StringVar();self.mode=tk.StringVar(value='Automatic')
        self.inputs={k:tk.StringVar(value=v) for k,v in [('conf','0.374'),('low_conf','0.18'),('window','2'),('radius','14'),('max_speed','30'),('fps','')]}
        self.catalog=discover(simulator.logs)
        choices=[f"{r['name']}  ({r['frames']} frames)" for r in self.catalog]
        ttk.Label(left,text='Capture folder',font=('Helvetica',13,'bold')).grid(row=0,column=0,sticky='w')
        self.folders=ttk.Combobox(left,textvariable=self.selected,values=choices,state='readonly',width=34)
        self.folders.grid(row=1,column=0,sticky='ew',pady=(7,5));self.folders.bind('<<ComboboxSelected>>',self.choose_saved)
        ttk.Entry(left,textvariable=self.folder,width=34).grid(row=2,column=0,sticky='ew',pady=4)
        ttk.Button(left,text='Browse…',command=self.browse).grid(row=3,column=0,sticky='ew',pady=(3,14))
        ttk.Label(left,text='Tracking mode').grid(row=4,column=0,sticky='w')
        ttk.Combobox(left,textvariable=self.mode,values=['Automatic','Precise K','Faster K'],state='readonly').grid(row=5,column=0,sticky='ew',pady=(5,12))
        settings=ttk.Frame(left);settings.grid(row=6,column=0,sticky='ew')
        for j,(label,key) in enumerate([('Normal confidence','conf'),('Recovery confidence','low_conf'),('Recovery window (s)','window'),('Radius cap (px)','radius'),('Speed gate (px/s)','max_speed'),('FPS override (optional)','fps')]):
            ttk.Label(settings,text=label,style='Muted.TLabel').grid(row=j,column=0,sticky='w',pady=4)
            ttk.Entry(settings,textvariable=self.inputs[key],width=8).grid(row=j,column=1,sticky='e',padx=(10,0),pady=4)
        ttk.Label(left,text='Leave FPS blank to use recorded times.\nSettings apply when loading a new test.',style='Muted.TLabel').grid(row=7,column=0,sticky='w',pady=10)
        ttk.Button(left,text='Load / new test',command=self.load).grid(row=8,column=0,sticky='ew',pady=5)
        ttk.Separator(left).grid(row=9,column=0,sticky='ew',pady=15)
        self.save_button=ttk.Button(left,text='Save summary',command=lambda:self.send('export'))
        self.save_button.grid(row=10,column=0,sticky='ew',pady=3)
        self.open_button=ttk.Button(left,text='Open results folder',command=self.open_results)
        self.open_button.grid(row=11,column=0,sticky='ew',pady=3)
        self.result_path=tk.StringVar(value='Each test saves a separate results folder.')
        ttk.Label(left,textvariable=self.result_path,style='Muted.TLabel',wraplength=290).grid(row=12,column=0,sticky='w',pady=12)
        ttk.Label(left,text='Source captures are read-only.\nExisting annotations are not treated\nas ground-truth tracking accuracy.',style='Muted.TLabel').grid(row=13,column=0,sticky='w',pady=6)
        right=ttk.Frame(root,padding=14);right.grid(row=1,column=1,sticky='nsew',padx=(0,12),pady=12)
        right.columnconfigure(0,weight=1);right.rowconfigure(2,weight=1)
        self.status=tk.StringVar(value='Choose a capture to begin.');self.error=tk.StringVar()
        ttk.Label(right,textvariable=self.status,wraplength=720).grid(row=0,column=0,sticky='ew',pady=(0,5))
        ttk.Label(right,textvariable=self.error,foreground='#ffb6a5',wraplength=720).grid(row=1,column=0,sticky='ew')
        self.canvas=tk.Canvas(right,bg='#080d11',highlightthickness=0)
        self.canvas.grid(row=2,column=0,sticky='nsew',pady=8)
        self.canvas.create_text(320,170,text='Load a capture folder to inspect its frames',fill='#9eafbc',tags='placeholder')
        controls=ttk.Frame(right);controls.grid(row=3,column=0,sticky='ew',pady=5)
        self.play=ttk.Button(controls,text='Play',command=self.toggle)
        self.play.pack(side='left',padx=(0,5))
        self.back=ttk.Button(controls,text='← Back',command=lambda:self.send('seek',index=max(0,self.current.get('index',0)-1)))
        self.back.pack(side='left',padx=4)
        self.step=ttk.Button(controls,text='Step →',command=lambda:self.send('step'));self.step.pack(side='left',padx=4)
        self.restart=ttk.Button(controls,text='Restart',command=lambda:self.send('restart'));self.restart.pack(side='left',padx=4)
        self.speed=tk.StringVar(value='1×')
        self.speed_widget=ttk.Combobox(controls,textvariable=self.speed,values=['0.25×','0.5×','1×','2×','4×','16×'],state='readonly',width=5)
        self.speed_widget.pack(side='right');self.speed_widget.bind('<<ComboboxSelected>>',lambda _:self.send('speed',value=float(self.speed.get().rstrip('×'))))
        timeline=ttk.Frame(right);timeline.grid(row=4,column=0,sticky='ew',pady=6);timeline.columnconfigure(0,weight=1)
        self.seek=tk.Scale(timeline,from_=0,to=1,orient='horizontal',showvalue=False,
                           resolution=1,takefocus=True,bg='#1b242d',fg='#e5edf3',
                           troughcolor='#34434e',highlightthickness=1,sliderlength=20)
        self.seek.grid(row=0,column=0,sticky='ew')
        self.seek.bind('<ButtonPress-1>',self.seek_pressed)
        self.seek.bind('<B1-Motion>',self.seek_motion)
        self.seek.bind('<ButtonRelease-1>',self.seek_released)
        for key,delta in [('Left',-1),('Right',1),('Home','first'),('End','last')]:
            self.seek.bind(f'<{key}>',lambda event,d=delta:self.seek_key(d))
        self.position=tk.StringVar(value='0 / 0');ttk.Label(timeline,textvariable=self.position,width=13,anchor='e').grid(row=0,column=1,padx=8)
        self.metrics=tk.StringVar(value='Tracker —     Observed —     Unresolved —')
        ttk.Label(right,textvariable=self.metrics,style='Metric.TLabel').grid(row=5,column=0,sticky='w',pady=8)
        self.timing=tk.StringVar();self.diagnostics=tk.StringVar()
        ttk.Label(right,textvariable=self.timing,style='Muted.TLabel',wraplength=720).grid(row=6,column=0,sticky='w',pady=4)
        ttk.Label(right,textvariable=self.diagnostics,style='Muted.TLabel',wraplength=720).grid(row=7,column=0,sticky='w',pady=4)
        ttk.Label(right,text='Playback speed changes viewing pace only. Backward seeking rebuilds history from the start.\nNo skipped frames: slow inference can slow playback. Space = play/pause; → = step.',style='Muted.TLabel',wraplength=720).grid(row=8,column=0,sticky='w',pady=(8,0))
        root.bind('<space>',self.key_play);root.bind('<Right>',self.key_step)
        root.protocol('WM_DELETE_WINDOW',self.close)
        self.poll()

    def choose_saved(self, _=None):
        i=self.folders.current()
        if i>=0:self.folder.set(self.catalog[i]['path'])

    def browse(self):
        from tkinter import filedialog
        path=filedialog.askdirectory(parent=self.root,title='Choose a capture folder',initialdir=str(self.sim.logs))
        if path:self.folder.set(path)

    def send(self, action, **values):
        if action == 'seek':
            self.seek_request += 1
            values['request'] = self.seek_request
            self.pending_seek = (self.seek_request, values['index'])
        elif action in ('load','play','pause','step','restart'):
            self.pending_seek = None
        self.sim.submit(dict(action=action,**values))

    def load(self):
        try:
            folder=self.folder.get().strip()
            if not folder:raise ValueError('Choose a capture folder first.')
            values={key:float(var.get()) for key,var in self.inputs.items() if key!='fps'}
            values['fps']=float(self.inputs['fps'].get()) if self.inputs['fps'].get().strip() else None
            mode={'Automatic':'auto','Precise K':'precise','Faster K':'faster'}[self.mode.get()]
            self.send('load',folder=folder,mode=mode,**values)
            self.local_error='';self.error.set('');self.status.set('Loading…')
        except ValueError as error:
            self.local_error=str(error);self.error.set(self.local_error)

    def toggle(self):
        self.send('pause' if self.current.get('playing') else 'play')

    def key_play(self,event):
        if event.widget.winfo_class() not in ('TEntry','TCombobox','Entry') and self.current.get('loaded'):
            self.toggle();return 'break'

    def key_step(self,event):
        if event.widget.winfo_class() not in ('TEntry','TCombobox','Entry') and self.current.get('loaded'):
            self.send('step');return 'break'

    def seek_pressed(self,event):
        if not self.current.get('loaded'):return 'break'
        self.dragging=True
        self.seek.focus_set()
        self.send('pause')
        return self.seek_motion(event)

    def seek_motion(self,event):
        if self.dragging:
            # Use actual thumb-centre endpoints, including theme padding.
            last=max(0,self.current['total']-1)
            left=self.seek.coords(0)[0];right=self.seek.coords(max(1,last))[0]
            value=round((event.x-left)*last/max(1,right-left))
            self.seek.set(min(last,max(0,value)))
            self.position.set(f"{self.seek.get()+1} / {self.current['total']}")
        return 'break'

    def seek_released(self,event):
        if self.current.get('loaded'):
            if event is not None:self.seek_motion(event)
            self.send('seek',index=min(self.current['total']-1,max(0,round(self.seek.get()))))
        self.dragging=False
        return 'break'

    def seek_key(self,delta):
        if self.current.get('loaded'):
            last=self.current['total']-1
            value=0 if delta=='first' else last if delta=='last' else self.seek.get()+delta
            self.seek.set(min(last,max(0,value)))
            self.seek_released(None)
        return 'break'

    def open_results(self):
        path=self.current.get('run')
        if path:
            import os, subprocess, sys
            if sys.platform=='win32':os.startfile(path)
            else:subprocess.Popen(['open' if sys.platform=='darwin' else 'xdg-open',path])

    def paint(self):
        import io
        from PIL import Image, ImageTk
        if not self.image_bytes:return
        width,height=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        image=Image.open(io.BytesIO(self.image_bytes))
        image.thumbnail((width,height),Image.Resampling.LANCZOS)
        self.photo=ImageTk.PhotoImage(image,master=self.root)
        self.canvas.delete('all');self.canvas.create_image(width/2,height/2,image=self.photo)
        self.last_image_size=(width,height)

    def poll(self):
        if self.closed:return
        s=self.sim.snapshot();self.current=s
        self.status.set(s['status']);self.error.set(self.local_error or s.get('error',''))
        ready=s['loaded']
        for widget in [self.play,self.step,self.restart,self.speed_widget]:
            widget.state(['!disabled'] if ready else ['disabled'])
        self.seek.configure(state='normal' if ready else 'disabled')
        self.back.state(['!disabled'] if ready and s['index']>0 else ['disabled'])
        for widget in [self.save_button,self.open_button]:
            widget.state(['!disabled'] if s.get('run') else ['disabled'])
        self.play.configure(text='Pause' if s['playing'] else 'Play')
        if ready:
            if not self.folder.get():self.folder.set(s['capture']['folder'])
            self.seek.configure(to=max(1,s['total']-1))
            if self.pending_seek:
                request,target=self.pending_seek
                if s.get('error') or (s.get('seek_request')==request and s.get('seek_target') is None):
                    self.pending_seek=None
            if not self.dragging:
                selected=self.pending_seek[1] if self.pending_seek else max(0,s['index'])
                self.seek.set(selected)
                self.position.set(f"{selected+1 if self.pending_seek else s['index']+1} / {s['total']}")
            c=s['capture'];gap=c['median_interval_s'];largest=c['maximum_gap_s']
            self.timing.set(c['timing']+(f" · median gap {gap:.3f}s · maximum gap {largest:.3f}s" if gap is not None else ''))
        if s['image_ready'] and self.version!=s['version']:
            self.image_bytes=self.sim.picture();self.version=s['version'];self.paint()
        elif self.image_bytes and self.last_image_size!=(self.canvas.winfo_width(),self.canvas.winfo_height()):self.paint()
        row=s.get('row')
        if row:
            self.metrics.set(f"{row['mode'].capitalize()} K    ·    Observed {len(row['observed'])}    ·    Unresolved {row['unresolved']}")
            self.diagnostics.set(f"t={row['time']:.3f}s  ·  Processing {row['processing_ms']:.1f}ms  ·  Tracker {row['tracking_ms']:.2f}ms\nNormal detections {len(row['base_detections'])}  ·  Recovery candidates {len(row['recovery_detections'])}  ·  Search regions {len(row['regions'])}")
        else:
            self.metrics.set('Tracker —    ·    Observed —    ·    Unresolved —');self.diagnostics.set('No frames processed yet.')
        self.result_path.set(s.get('run') or 'Each test saves a separate results folder.')
        self.root.after(100,self.poll)

    def close(self):
        self.closed=True;self.sim.close();self.root.destroy()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--logs',type=Path,default=HERE/'logs')
    ap.add_argument('--output',type=Path,default=HERE/'logs'/'tracking_simulations')
    ap.add_argument('--weights')
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--capture',help='preload a capture; native UI starts paused')
    ap.add_argument('--headless',action='store_true',help='test --capture to completion without a UI')
    ap.add_argument('--tracker',choices=['auto','precise','faster'],default='auto')
    ap.add_argument('--fps',type=float,help='explicit assumed FPS override; default uses recorded timestamps')
    args=ap.parse_args()
    if args.headless and not args.capture:ap.error('--headless requires --capture')
    from model_registry import resolve_weights
    sim=Simulator(args.logs,args.output,resolve_weights(args.weights),args.device)
    if args.capture:sim.submit(dict(action='load',folder=args.capture,mode=args.tracker,fps=args.fps))
    if args.headless:
        sim.submit(dict(action='play'))
        try:
            while True:
                s=sim.snapshot()
                if s['error']:raise RuntimeError(s['error'])
                if s['status'].startswith('Finished'):
                    print(json.dumps({'frames':s['index']+1,'results':s['run']}));break
                time.sleep(.1)
        finally:sim.close()
    else:
        from launcher_desktop import run_simulator_desktop
        run_simulator_desktop(sim)


if __name__=='__main__':main()

