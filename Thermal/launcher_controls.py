"""Native launcher controls and legacy-compatible thermal dataset capture."""
from collections import deque
from datetime import datetime
from pathlib import Path
import csv
import json
import os
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
        self.boxes=True;self.labels=True;self.raw=False;self.help=True;self.palette=0
        self.paused=False;self.step=False;self.quit=False;self.message='Ready. R records thermal dataset frames.'
        self.actions=deque();self.buttons=[];self.events=[]

    def command(self,key,data=None,row=None):
        if key in (27,ord('q')):self.quit=True;return
        modes={'1':'auto','2':'precise','3':'faster'}
        char=chr(key) if 0<=key<256 else ''
        if char in ('[',']',';',"'"):
            d=self.session.detector;conf=d.conf;low=d.low_conf
            if char in ('[',']'):conf=min(.99,max(low+.01,round(conf+(.01 if char==']' else -.01),3)))
            else:low=min(conf-.01,max(.01,round(low+(.01 if char=="'" else -.01),3)))
            self.session.set_confidence(conf,low)
            self.message=f'Next frame: detection {conf:.3f} / search {low:.3f}'
        elif char in modes or char=='t':
            target=modes[char] if char in modes else ('auto','precise','faster')[(('auto','precise','faster').index(self.session.mode)+1)%3]
            self.session.set_mode(target);self.message=f'Tracker policy: {target} (applies next frame)'
        elif char=='x':
            self.session.reset();self.message='Tracker reset; IDs restart their histories on next frame'
        elif char=='c':self.show_search=not self.show_search
        elif char=='b':self.boxes=not self.boxes
        elif char=='i':self.labels=not self.labels
        elif char=='g':self.raw=not self.raw
        elif char=='h':self.help=not self.help
        elif char=='p':self.palette=(self.palette+1)%3
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
            conf=self.session.detector.conf,search_conf=self.session.detector.low_conf,policy=self.session.mode))

    def mouse(self,event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONUP:
            for (x0,y0,x1,y1),key in self.buttons:
                if x0<=x<x1 and y0<=y<y1:self.actions.append(ord(key));break

    def render(self,data,row,scale):
        from crowd_detector import thermal_image
        scale=max(3,scale);W=160*scale;H=max(740,120*scale+112);side=350
        out=np.full((H,W+side,3),(24,20,17),np.uint8)
        grey=thermal_image(data)[:,:,0]
        im=(cv2.applyColorMap(grey,[cv2.COLORMAP_INFERNO,cv2.COLORMAP_VIRIDIS][self.palette])
            if self.palette<2 else cv2.cvtColor(grey,cv2.COLOR_GRAY2BGR))
        im=cv2.resize(im,(W,120*scale),interpolation=cv2.INTER_NEAREST)
        # Draw the frozen row, so paused settings changes cannot relabel an old frame.
        if self.raw:
            for d in row['base_detections']:
                x,y,w,h=d['bbox'];cv2.rectangle(im,(round(x*scale),round(y*scale)),(round((x+w)*scale),round((y+h)*scale)),(230,170,70),1)
        if self.boxes:
            for t in row['observed']:
                x,y,w,h=t['bbox'];col=(110,230,135) if t['source']=='base' else (50,210,255)
                cv2.rectangle(im,(round(x*scale),round(y*scale)),(round((x+w)*scale),round((y+h)*scale)),col,1)
                if self.labels:cv2.putText(im,f'#{t["id"]} {t["conf"]:.2f}',(round(x*scale),max(13,round(y*scale)-4)),0,.4,col,1,cv2.LINE_AA)
        if self.show_search and row['mode']=='faster':
            for r in row['regions']:
                center=(round(r['cx']*scale),round(r['cy']*scale))
                cv2.circle(im,center,round(r['radius']*scale),(220,175,100),1)
                cv2.putText(im,f'search #{r["id"]}',center,0,.35,(220,175,100),1,cv2.LINE_AA)
        out[100:100+im.shape[0],:W]=im
        def text(s,x,y,size=.48,col=(220,225,230)):
            cv2.putText(out,s,(x,y),0,size,col,1,cv2.LINE_AA)
        text('FLUXNET / TRACKING V2 (EXPERIMENTAL)',18,28,.65)
        text(f'{row["mode"].upper()} K    OBSERVED {len(row["observed"])}    UNRESOLVED {row["unresolved"]}',18,56,.55)
        text(f'Frame {row["frame"]}   inference + tracking {row["processing_ms"]:.0f} ms   '+('PAUSED' if self.paused else 'LIVE' if self.live else 'REPLAY')+(f'   legacy estimate {row["estimate"]}' if row['mode']=='precise' else ''),18,82)
        x=W+16;out[:,W:]=(37,31,26);self.buttons=[]
        def button(label,key,y,width=314,dx=0,active=False):
            xx=x+dx;cv2.rectangle(out,(xx,y),(xx+width,y+30),(82,88,45) if active else (65,55,45),-1)
            
            if key=='r' and active:cv2.rectangle(out,(xx,y),(xx+width,y+30),(55,55,155),-1)
            text(label,xx+8,y+20,.43);self.buttons.append(((xx,y,xx+width,y+30),key))
        d=self.session.detector
        text('CONTROLS',x,28,.6)
        text(f'Detection confidence: {d.conf:.3f}',x,57)
        button('[  lower','[',68,150);button(']  higher',']',68,150,164)
        text(f'Search confidence: {d.low_conf:.3f}',x,122)
        button(';  lower',';',133,150);button("'  higher","'",133,150,164)
        text('Search applies only inside Faster K regions',x,185,.37)
        text(f'Tracker policy: {self.session.mode.upper()}',x,213)
        for j,(name,key) in enumerate([('Auto','1'),('Precise','2'),('Faster','3')]):button(f'{key} {name}',key,225,98,j*108,self.session.mode==('auto','precise','faster')[j])
        button('C  Search circles: '+('ON' if self.show_search else 'OFF'),'c',266)
        button('R  '+('STOP RECORDING' if self.recorder.continuous else 'Start recording'),'r',307,active=bool(self.recorder.continuous))
        button('S  Save single thermal frame','s',348)
        text(f'Dataset frames saved: {self.recorder.total}',x,405)
        text('Labels: full-frame YOLO omega only',x,427,.4)
        button('Space  '+('Resume' if self.paused else 'Pause'),' ',443,190)
        button('N Step','n',443,114,200)
        button('X Reset tracker','x',484)
        button('B Boxes','b',525,150,active=self.boxes);button('I IDs / scores','i',525,150,164,self.labels)
        button('G Raw detections','g',566,150,active=self.raw);button('P Palette','p',566,150,164)
        button('H Help','h',607,150,active=self.help);button('Q Quit','q',607,150,164)
        if self.help:
            text('T cycles tracker; 1/2/3 select directly.',x,666,.39)
            text('Conf changes apply to the next frame.',x,687,.39)
            text('Pause suspends recording. S saves once.',x,708,.39)
        text(self.message[:max(1,(W-20)//8)],12,H-12,.43,(135,215,230))
        return out
