"""Windows offline RGB proposals -> class-1 thermal capture labels.
YOLOE prompts are experimental proposals, not a proven omega specialist.
Original thermal arrays/encoding are preserved. No human labels are generated.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from morph_field import MorphField,DEFAULT_MORPH
from alignment_preset import resolve_session_alignment,load_alignment,crop_bounds,sha256,cropped_box_to_full

def rgb_to_ir_matrix(alignment):
    s=alignment['state']
    angle=np.deg2rad(s['angle'])
    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    linear=rot@np.diag([s['scale']*(-1 if s.get('flipX') else 1),s['scale']*(-1 if s.get('flipY') else 1)])
    forward=np.eye(3);forward[:2,:2]=linear
    forward[:2,2]=np.array([s['x'],s['y']])-linear@np.array([80,60])
    # Edge coordinates, matching YOLO xyxy and canvas calibration.
    return np.linalg.inv(forward)

def transfer_box(box,matrix,width=160,height=120,morph=None,audit=None):
    """Map four corners, translate uniformly at centre, then clip once."""
    detail={} if audit is None else audit
    if len(box)!=4 or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
        detail['rejection']='invalid_rgb_box';return None
    x0,y0,x1,y1=box
    points=np.array([[x0,y0,1],[x1,y0,1],[x1,y1,1],[x0,y1,1]])@matrix.T
    if not np.isfinite(points).all() or np.any(np.abs(points[:,2])<1e-9):
        detail['rejection']='invalid_projection';return None
    xy=points[:,:2]/points[:,2:];lo=xy.min(axis=0);hi=xy.max(axis=0)
    detail['provisional_ir_xyxy']=[*lo.tolist(),*hi.tolist()]
    if morph is not None:
        delta,info=morph.displacement((lo+hi)/2);lo=lo+delta;hi=hi+delta;detail.update(info)
    detail['translated_ir_xyxy']=[*lo.tolist(),*hi.tolist()]
    clipped_lo=np.maximum(lo,[0,0]);clipped_hi=np.minimum(hi,[width,height])
    detail['clipped']=bool(np.any(clipped_lo!=lo) or np.any(clipped_hi!=hi))
    if np.any(clipped_hi<=clipped_lo):detail['rejection']='outside_ir_after_correction';return None
    return [*clipped_lo.astype(float).tolist(),*clipped_hi.astype(float).tolist()]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('session',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--alignment',type=Path)
    p.add_argument('--morph',type=Path,default=DEFAULT_MORPH,help='Saved displacement field; alignment hash must match')
    p.add_argument('--weights',default='yoloe-26x-seg.pt')
    p.add_argument('--mode',choices=['yoloe','specialist'],default='yoloe')
    p.add_argument('--experimental-annotator',action='store_true',help='Acknowledge unvalidated YOLOE omega proposals; initial positive test failed')
    p.add_argument('--prompt',default='human head and shoulders')
    p.add_argument('--source-class',type=int,default=0)
    p.add_argument('--conf',type=float,default=.25);p.add_argument('--imgsz',type=int,default=1280)
    p.add_argument('--device',default='0')
    p.add_argument('--allow-experimental-alignment',action='store_true')
    a=p.parse_args()
    if a.mode=='yoloe' and not a.experimental_annotator:p.error('YOLOE omega annotation failed initial positive-image evaluation. Use --experimental-annotator only for trials, or provide a validated specialist.')
    if not 0<a.conf<1: p.error('confidence must be between 0 and 1')
    a.alignment=resolve_session_alignment(a.session,a.alignment)
    alignment=load_alignment(a.alignment);pairs=json.loads((a.session/'pairs.json').read_text())
    if not pairs.get('pairs'):raise ValueError('Empty session')
    if pairs['status']!='ready_for_annotation' or not all(m['download_verified'] for m in pairs['pairs']):
        raise ValueError('Session incomplete; do not annotate missing RGB pairs')
    if not (alignment.get('validated',False) or alignment.get('workflow_approved',False)) and not a.allow_experimental_alignment:
        raise ValueError('Alignment is not validated. Use experimental override only for a review trial.')
    morph=MorphField(a.morph,sha256(a.alignment))
    matrix=rgb_to_ir_matrix(alignment)
    annotation_started=time.perf_counter()
    left,top,right,bottom=crop_bounds(alignment)
    from ultralytics import YOLO,YOLOE
    model=YOLOE(a.weights) if a.mode=='yoloe' else YOLO(a.weights)
    if a.mode=='yoloe': model.set_classes([a.prompt])
    else:
        label=str(model.names[a.source_class]).lower().replace('-','_').replace(' ','_')
        if label not in ['omega','head_shoulder','head_shoulders','head_and_shoulders']:
            raise ValueError('Specialist class must explicitly mean omega/head-and-shoulders, not person/head/face')
    a.out.mkdir(parents=True,exist_ok=False)
    for sub in ['npy','png','labels','review']:(a.out/sub).mkdir()
    (a.out/'classes.txt').write_text('person\nhead_shoulder\n')
    provenance={'reviewed':False,'weights':a.weights,'model_names':model.names,'mode':a.mode,'prompt':a.prompt,
                'conf':a.conf,'imgsz':a.imgsz,'morph':morph.provenance(),'alignment':alignment,'alignment_sha256':sha256(a.alignment),
                'inference_coordinates':'saved RGB crop; predictions restored to original RGB coordinates before IR transfer','timing_validated':pairs['timing_validated'],'integration_test_only':pairs.get('integration_test_only',False),'frames':[]}
    with (a.session/'manifest.csv').open() as f: rows={r['file']:r for r in csv.DictReader(f)}
    for m in pairs['pairs']:
        stem=m['stem']
        if Path(stem).name!=stem or not stem.startswith('cap_'): raise ValueError('Unsafe frame ID')
        image=a.session/'rgb'/(stem+'.jpg')
        if hashlib.sha256(image.read_bytes()).hexdigest()!=m['rgb_sha256']: raise ValueError('RGB checksum mismatch')
        rgb=cv2.imread(str(image))
        if rgb is None or rgb.shape[:2]!=(1200,1600): raise ValueError('Alignment requires original 1600x1200 RGB')
        # Infer on the approved crop; restore original pixel coordinates before mapping.
        started=time.perf_counter()
        result=model.predict(rgb[top:bottom,left:right],conf=a.conf,imgsz=a.imgsz,device=a.device,max_det=1000,verbose=False)[0]
        elapsed=time.perf_counter()-started
        labels=[];audit=[];transfer_seconds=0.
        for b,c,score in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.cls.cpu().numpy(),result.boxes.conf.cpu().numpy()):
            if int(c)!=(0 if a.mode=='yoloe' else a.source_class):continue
            b=np.array(cropped_box_to_full(b,alignment))
            detail={};transfer_started=time.perf_counter()
            mapped=transfer_box(b,matrix,morph=morph,audit=detail)
            transfer_seconds+=time.perf_counter()-transfer_started
            audit.append({'rgb_xyxy':b.tolist(),'thermal_xyxy':mapped,'confidence':float(score),'morph':detail})
            if mapped:
                x0,y0,x1,y1=mapped
                labels.append(f'1 {(x0+x1)/320:.6f} {(y0+y1)/240:.6f} {(x1-x0)/160:.6f} {(y1-y0)/120:.6f}')
        for sub,ext in [('npy','.npy'),('png','.png')]:shutil.copy2(a.session/sub/(stem+ext),a.out/sub/(stem+ext))
        (a.out/'labels'/(stem+'.txt')).write_text('\n'.join(labels)+ ('\n' if labels else ''))
        # Keep the thermal encoding fixed; colour is only for review display.
        temp=np.load(a.out/'npy'/(stem+'.npy'))
        if temp.shape!=(120,160):raise ValueError('Unexpected thermal dimensions')
        gray=(np.clip((temp-15)/30,0,1)*255).astype(np.uint8)
        vis=cv2.resize(cv2.applyColorMap(gray,cv2.COLORMAP_INFERNO),(640,480),interpolation=cv2.INTER_NEAREST)
        for item in audit:
            if item['thermal_xyxy']:
                x0,y0,x1,y1=item['thermal_xyxy'];cv2.rectangle(vis,(round(x0*4),round(y0*4)),(round(x1*4),round(y1*4)),(0,255,255),1)
        cv2.imwrite(str(a.out/'review'/(stem+'.png')),vis)
        rows[stem].update(n_det='0',omega_count=str(len(labels)),flags='RGB-PROPOSAL-UNREVIEWED'+ (';MORPH-EXTRAPOLATED' if any(x['morph'].get('extrapolated') for x in audit) else ''),conf=str(a.conf))
        provenance['frames'].append({'stem':stem,'proposals':audit,'wall_seconds':elapsed,'annotation_transfer_seconds':transfer_seconds,'model_speed_ms':result.speed})
    with (a.out/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(next(iter(rows.values())).keys()));writer.writeheader();writer.writerows(rows[m['stem']] for m in pairs['pairs'])
    provenance['annotation_wall_seconds']=time.perf_counter()-annotation_started
    provenance['annotation_transfer_seconds']=sum(frame['annotation_transfer_seconds'] for frame in provenance['frames'])
    shutil.copy2(a.morph,a.out/'morph.json')
    (a.out/'rgb_annotation.json').write_text(json.dumps(provenance,indent=2))
    shutil.copy2(a.session/'pairs.json',a.out/'pairs.json')
    shutil.copy2(a.alignment,a.out/'alignment.json')
    (a.out/'source.txt').write_text('RGB-derived class-1 proposals; manual thermal review required.\n'+a.weights+'\n')
    (a.out/'READY_FOR_REVIEW').write_text('Annotation completed; NOT human-reviewed.\n')
    print(a.out)
if __name__=='__main__':main()
