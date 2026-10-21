"""Mac paired acquisition. No annotation model runs during recording.
Run beside dataset_recording.py. r toggles capture; s captures once; q finishes
and downloads RGB JPEGs from SD. Pending sessions live outside logs/.
Timing is software pairing, NOT a validated exposure synchronization guarantee.
"""
import argparse
from collections import deque
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import zlib
import csv
from alignment_preset import DEFAULT_ALIGNMENT,load_alignment,sha256
import cv2
import numpy as np
import serial
import dataset_recording as base
from lepton_libuvc import LeptonUVC

class CameraSerial:
    def __init__(self, port):
        self.port = serial.Serial(port,115200,timeout=.2,write_timeout=5)
        self.pending_line=b''
        try:
            self.port.dtr=False; self.port.rts=False
            time.sleep(1)
            self.send('STATUS')
            status=self.until('STATUS ',40)
            if 'firmware=occupancy-camera-sd-v1' not in status or 'camera=1 sd=1' not in status:raise RuntimeError(status)
        except BaseException:
            self.close()
            raise
    def send(self, text):
        self.port.write((text+'\n').encode()); self.port.flush()
    def until(self,prefix,timeout=30):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            part=self.port.readline()
            self.pending_line+=part
            if len(self.pending_line)>65536:raise ValueError('Oversized serial response')
            if not part.endswith(b'\n'):continue
            line=self.pending_line.decode(errors='replace').strip();self.pending_line=b''
            if line.startswith('ERROR '): raise RuntimeError(line)
            if line.startswith(prefix): return line
        raise TimeoutError(prefix)
    def capture(self,ident):
        sent=time.time_ns()//1000
        self.send(f'CAPTURE {ident} {sent}')
        item=json.loads(self.until('SAVED ',40)[6:])
        if item['id']!=ident or item['pc_sent_epoch_us']!=sent: raise ValueError('Capture ID mismatch')
        item['pc_ack_epoch_us']=time.time_ns()//1000
        return item
    def download(self,item,dest):
        self.send('GET '+item['id'])
        h=self.until('DATA ').split()
        if len(h)!=4 or h[1]!=item['id']: raise ValueError('Transfer header mismatch')
        size,crc=int(h[2]),int(h[3],16)
        if size!=item['bytes'] or crc!=item['crc32'] or not 4<=size<=4*1024*1024: raise ValueError('Invalid transfer size/CRC')
        data=bytearray();deadline=time.monotonic()+size/11520*2+20
        while len(data)<size and time.monotonic()<deadline:
            data.extend(self.port.read(min(8192,size-len(data))))
        if len(data)!=size or zlib.crc32(data)!=crc: raise ValueError('Transfer checksum failed')
        if self.until('END ',5)!='END '+item['id']: raise ValueError('Transfer trailer failed')
        decoded=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape[:2]!=(item['height'],item['width']): raise ValueError('JPEG decode failed')
        dest.write_bytes(data)
        item['rgb_sha256']=hashlib.sha256(data).hexdigest()
    def close(self): self.port.close()

class ThermalStream(LeptonUVC):
    def __init__(self):
        self.frames=deque(maxlen=180);self.lock=threading.Lock()
        super().__init__()
    def _on_frame(self,p,user):
        ts=time.time_ns()//1000;f=p.contents
        if f.data and f.data_bytes==f.width*f.height*2:
            raw=np.frombuffer(ctypes.string_at(f.data,f.data_bytes),np.uint16).copy().reshape(f.height,f.width)
            with self.lock: self.frames.append((ts,int(f.sequence),raw))
        super()._on_frame(p,user)
    def snapshot(self):
        with self.lock: return list(self.frames)

def persist(root,items,status,args):
    data={'schema':1,'status':status,'temperature_kelvin_per_count':args.kelvin_per_count,
          'timing_validated':False,'reviewed':False,'pairs':items,
          'alignment_file':'alignment.json','alignment_sha256':sha256(root/'alignment.json'),
          'timing_method':'nearest thermal host receipt to midpoint of RGB command-send and saved-ack; not exposure synchronization'}
    tmp=root/'pairs.json.tmp';tmp.write_text(json.dumps(data,indent=2));tmp.replace(root/'pairs.json')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',default='/dev/cu.usbserial-310')
    p.add_argument('--interval',type=float,default=1.0)
    p.add_argument('--count',type=int,default=0,help='Auto capture this many pairs; 0 opens interactive recording')
    p.add_argument('--kelvin-per-count',type=float,choices=[.01,.1],required=True,help='Confirmed sensor TLinear scale; do not guess from palette')
    p.add_argument('--note',default='')
    p.add_argument('--out',type=Path)
    p.add_argument('--annotate-on-windows',action='store_true')
    p.add_argument('--alignment',type=Path,default=DEFAULT_ALIGNMENT,help='Mount preset; a copy is saved inside each recording')
    p.add_argument('--experimental-annotator',action='store_true')
    p.add_argument('--experimental-alignment',action='store_true')
    a=p.parse_args()
    if a.annotate_on_windows and not a.experimental_annotator:p.error('The configured annotator is experimental; pass --experimental-annotator for a trial')
    if a.annotate_on_windows and not a.alignment:p.error('--alignment is required for automatic annotation')
    load_alignment(a.alignment)
    if a.interval<.5 or a.count<0: p.error('interval must be >=0.5; count >=0')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root=a.out or Path(__file__).resolve().parent/'paired_recordings'/('capture_'+stamp)
    root.mkdir(parents=True,exist_ok=False)
    import shutil
    shutil.copy2(a.alignment,root/'alignment.json')
    for sub in ['npy','png','labels','review','rgb','thermal_raw']: (root/sub).mkdir()
    (root/'classes.txt').write_text('person\nhead_shoulder\n')
    (root/'source.txt').write_text('recorder Faster_dataset_recording.py\nRGB/thermal pending annotation; no human labels\n'+a.note+'\n')
    rows=[];cam=None;esp=None;fh=None;complete=False
    try:
        esp=CameraSerial(a.port);cam=ThermalStream()
        fh=(root/'manifest.csv').open('w',newline='');writer=csv.writer(fh);writer.writerow(base.MANIFEST_COLS)
        persist(root,rows,'recording',a)
        recording=a.count>0;next_capture=0;waiting_since=time.monotonic()
        if not a.count: cv2.namedWindow('Faster paired recorder',cv2.WINDOW_NORMAL)
        while True:
            frames=cam.snapshot()
            if not frames:
                if time.monotonic()-waiting_since>15:raise TimeoutError('No thermal frames received within 15 seconds')
                time.sleep(.02);continue
            temp=frames[-1][2].astype(np.float32)*a.kelvin_per_count-273.15
            key=-1
            if not a.count:
                preview=base.draw(temp,[],5,recording,len(rows),0,root.name)
                cv2.imshow('Faster paired recorder',preview);key=cv2.waitKey(20)&255
            if key in [27,ord('q')]: break
            if key==ord('r'): recording=not recording
            if key==ord('s') or (recording and time.monotonic()>=next_capture):
                ident='p'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
                m=esp.capture(ident)
                target=(m['pc_sent_epoch_us']+m['pc_ack_epoch_us'])//2
                frames=cam.snapshot();ts,seq,raw=min(frames,key=lambda f:abs(f[0]-target))
                if abs(ts-target)>1_000_000: raise RuntimeError('No recent thermal frame; capture rejected')
                i=len(rows);stem=f'cap_{i:06d}'
                temp=raw.astype(np.float32)*a.kelvin_per_count-273.15
                base.write_frame(str(root),writer,i,temp,[],a.note,False,conf=None);fh.flush()
                np.save(root/'thermal_raw'/(stem+'.npy'),raw)
                m.update(stem=stem,thermal_host_receipt_epoch_us=ts,thermal_sequence=seq,
                         pairing_receipt_residual_ms=(ts-target)/1000,download_verified=False)
                rows.append(m);persist(root,rows,'recording',a)
                print('Saved pair',len(rows),flush=True)
                next_capture=time.monotonic()+a.interval
                if a.count and len(rows)>=a.count: break
        cam.release();cam=None
        cv2.destroyAllWindows()
        persist(root,rows,'downloading',a)
        for m in rows:
            esp.download(m,root/'rgb'/(m['stem']+'.jpg'));m['download_verified']=True
            persist(root,rows,'downloading',a);print('Downloaded',m['stem'],flush=True)
        complete=bool(rows)
    finally:
        if cam: cam.release()
        if esp: esp.close()
        if fh: fh.close()
        cv2.destroyAllWindows()
        persist(root,rows,'ready_for_annotation' if complete else 'incomplete',a)
        # Match run.sh: return files created by sudo to the invoking Mac user.
        import os
        if hasattr(os,'geteuid') and os.geteuid()==0 and 'SUDO_UID' in os.environ:
            uid,gid=int(os.environ['SUDO_UID']),int(os.environ['SUDO_GID'])
            for file in [root,*root.rglob('*')]: os.chown(file,uid,gid)
    print(root)
    if complete and a.annotate_on_windows:
        import subprocess,sys,os,pwd
        script=Path(__file__).with_name('windows_pipeline.py')
        command=[sys.executable,str(script),'annotate',str(root),'--alignment',str((root/'alignment.json').resolve())]
        if a.experimental_alignment:command.append('--experimental')
        if a.experimental_annotator:command.append('--experimental-annotator')
        if os.geteuid()==0 and 'SUDO_UID' in os.environ:
            user=pwd.getpwuid(int(os.environ['SUDO_UID'])).pw_name
            command=['sudo','-H','-u',user,*command]
        subprocess.run(command,check=True)
if __name__=='__main__': main()
