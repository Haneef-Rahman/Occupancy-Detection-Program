"""Native launcher controls and legacy-compatible thermal dataset capture."""
from collections import deque
from datetime import datetime
from pathlib import Path
import csv
import json
import os
import time
import cv2
import numpy as np


class DatasetCapture:
    """Use the original recorder's frame encoder; never save tracker guesses."""
    def __init__(self, root, weights, note=''):
        self.root=Path(root); self.weights=str(weights); self.note=note
        self.continuous=None; self.shots=None; self.folders=[]; self.total=0

    def create(self, kind, row):
        from dataset_recording import MANIFEST_COLS
        path=self.root/f'capture_{datetime.now():%Y%m%d_%H%M%S_%f}'
        path.mkdir(parents=True,exist_ok=False); self.folders.append(path)
        for sub in ('npy','png','labels','review'):(path/sub).mkdir()
        (path/'classes.txt').write_text('person\nhead_shoulder\n')
        (path/'source.txt').write_text(
            f'recorder   integrated_launcher_v2.py\nweights    {self.weights}\n'
            f'conf       {row["normal_confidence"]}\nclasses    omega only (class 1)\n'
            f'labels     full-frame YOLO only; no recovery or predicted boxes\n'
            f'kind       {kind}\nnote       {self.note}\n')
        fh=(path/'manifest.csv').open('w',newline=''); writer=csv.writer(fh);writer.writerow(MANIFEST_COLS)
        details=(path/'capture_frames.jsonl').open('w')
        return dict(path=path,fh=fh,writer=writer,details=details,n=0,last=None,times=[])

    @staticmethod
    def finish(cap):
        if cap:
            cap['fh'].close(); cap['details'].close()

    def toggle(self,row):
        if self.continuous:
            path=self.continuous['path'];self.finish(self.continuous);self.continuous=None
            return f'Saved {path.name}'
        self.continuous=self.create('continuous',row)
        return f'Recording {self.continuous["path"].name}'

    def write(self,cap,data,row):
        token=(row['origin'],row['time'])
        if cap['last']==token:return False
        from dataset_recording import write_frame
        omegas=[(*d['bbox'],d['conf']) for d in row['base_detections']]
        # Preserve the original encoder/label schema. Manifest time represents
        # frame arrival, not the time a paused user presses S.
        class ManifestWriter:
            def writerow(_, values):
                values[1]=row.get('received_at',datetime.now().isoformat(timespec='milliseconds'))
                cap['writer'].writerow(values)
        write_frame(str(cap['path']),ManifestWriter(),cap['n'],data,omegas,self.note,
                    False,conf=row['normal_confidence'])
        stem=f'cap_{cap["n"]:06d}'
        cap['times'].append(dict(stem=stem,relative_time_s=row['time']))
        tmp=cap['path']/'timestamps.tmp';tmp.write_text(json.dumps(cap['times']));tmp.replace(cap['path']/'timestamps.json')
        cap['details'].write(json.dumps(dict(stem=stem,origin=row['origin'],time=row['time'],
            normal_confidence=row['normal_confidence'],recovery_confidence=row['recovery_confidence'],
            tracker_policy=row['tracker_policy'],active_tracker=row['mode'],label_source='base YOLO only'))+'\n')
        cap['details'].flush();cap['fh'].flush()
        cap['n']+=1;cap['last']=token;self.total+=1
        return True

    def shot(self,data,row):
        if self.continuous:return self.write(self.continuous,data,row)
        if self.shots is None:self.shots=self.create('manual shots',row)
        return self.write(self.shots,data,row)

    def close(self):
        self.finish(self.continuous);self.finish(self.shots)
        self.continuous=None;self.shots=None
        if hasattr(os,'geteuid') and os.geteuid()==0 and os.environ.get('SUDO_UID'):
            for folder in self.folders:
                for p in [*folder.rglob('*'),folder]:
                    os.chown(p,int(os.environ['SUDO_UID']),int(os.environ['SUDO_GID']))


class LauncherControls:
    def __init__(self,session,recorder,show_search=False,live=True):
        self.session=session;self.recorder=recorder;self.show_search=show_search;self.live=live
        self.boxes=True;self.labels=True;self.raw=False;self.help=False;self.palette=0;self.contrast=0
        self.paused=False;self.step=False;self.quit=False;self.message='Ready. R records thermal dataset frames.'
        self.actions=deque();self.buttons=[];self.events=[]
        self.pointer=None;self.sliders={};self.dragging=None;self.drag_value=None
        self.bubble_name=None;self.bubble_until=0
        from occupancy_chart import OccupancyHistory
        self.history=OccupancyHistory();self.show_history=True
        self.chart_bounds=None;self.chart_hover=None

    def command(self,key,data=None,row=None):
        if isinstance(key,dict):
            name=key['slider'];value=self.constrain_slider(name,key['value'])
            d=self.session.detector
            self.session.set_confidence(value if name=='conf' else d.conf,
                                        value if name=='low_conf' else d.low_conf)
            self.message=f'Next frame: detection {d.conf:.3f} / search {d.low_conf:.3f}'
            self.events.append(dict(action='slider/'+name,frame=None if row is None else row['frame'],
                conf=d.conf,search_conf=d.low_conf,policy=self.session.mode,palette=self.palette,contrast=self.contrast))
            return
        if key in (27,ord('q')):self.quit=True;return
        modes={'1':'auto','2':'precise','3':'faster'}
        char=chr(key) if 0<=key<256 else ''
        if char in ('[',']',';',"'"):
            d=self.session.detector;conf=d.conf;low=d.low_conf
            if char in ('[',']'):conf=min(.99,max(low+.01,round(conf+(.01 if char==']' else -.01),3)))
            else:low=min(conf-.01,max(.01,round(low+(.01 if char=="'" else -.01),3)))
            self.session.set_confidence(conf,low)
            self.message=f'Next frame: detection {conf:.3f} / search {low:.3f}'
            self.bubble_name='conf' if char in ('[',']') else 'low_conf'
            self.bubble_until=time.monotonic()+1
        elif char in modes or char=='t':
            target=modes[char] if char in modes else ('auto','precise','faster')[(('auto','precise','faster').index(self.session.mode)+1)%3]
            self.session.set_mode(target);self.message=f'Tracker policy: {target} (applies next frame)'
        elif char=='x':
            self.history.clear()
            self.session.reset();self.message='Tracker reset; IDs restart their histories on next frame'
        elif char=='c':self.show_search=not self.show_search
        elif char=='b':self.boxes=not self.boxes
        elif char=='i':self.labels=not self.labels
        elif char=='g':self.raw=not self.raw
        elif char=='h':self.help=not self.help
        elif char=='o':self.show_history=not self.show_history
        elif char=='p':self.palette=(self.palette+1)%3
        elif char=='v':
            from dataset_pipeline import VIEWS
            self.palette=0;self.contrast=(self.contrast+1)%len(VIEWS)
            self.message=f'Inferno contrast: {VIEWS[self.contrast][0]} (display only)'
        elif char==' ':
            self.paused=not self.paused
            if not self.paused and self.live:self.session.reset()
        elif char=='n':
            self.paused=True;self.step=True
            if self.live:self.session.reset()
        elif char=='r' and row is not None:self.message=self.recorder.toggle(row)
        elif char=='s' and row is not None:
            self.message='Frame saved' if self.recorder.shot(data,row) else 'This frame is already saved'
        else:return
        self.events.append(dict(action=char,frame=None if row is None else row['frame'],
            conf=self.session.detector.conf,search_conf=self.session.detector.low_conf,policy=self.session.mode,palette=self.palette,contrast=self.contrast))

    def constrain_slider(self,name,value):
        if name not in ('conf','low_conf') or not np.isfinite(value):
            raise ValueError('invalid confidence slider value')
        d=self.session.detector
        if name=='conf':return round(min(.99,max(d.low_conf+.01,value)),3)
        return round(min(d.conf-.01,max(.01,value)),3)

    def drag_to(self,x):
        x0,x1,_=self.sliders[self.dragging]['rail']
        self.drag_value=self.constrain_slider(self.dragging,round(.01+.98*(x-x0)/max(1,x1-x0),3))

    def mouse(self,event,x,y,flags,param):
        self.pointer=(x,y)
        if event==cv2.EVENT_LBUTTONDOWN:
            for name,slider in self.sliders.items():
                x0,y0,x1,y1=slider['bounds']
                if x0<=x<=x1 and y0<=y<=y1:
                    self.dragging=name;self.drag_to(x);return
        elif event==cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_to(x)
        elif event==cv2.EVENT_LBUTTONUP:
            if self.dragging:
                self.drag_to(x)
                self.actions.append(dict(slider=self.dragging,value=self.drag_value))
                self.dragging=None;self.drag_value=None;return
            for (x0,y0,x1,y1),key in self.buttons:
                if x0<=x<x1 and y0<=y<y1:self.actions.append(ord(key));break

    def observe(self,row):
        self.history.append(row)

    def render(self,data,row,scale):
        from launcher_theme import render_dashboard
        return render_dashboard(self,data,row,scale)
