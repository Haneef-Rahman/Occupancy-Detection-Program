"""Versioned manual mount preset and per-recording calibration snapshots."""
from pathlib import Path
import hashlib,json,math
DEFAULT_ALIGNMENT=Path(__file__).resolve().parent/'calibration'/'mounted-20261021.json'

def load_alignment(path):
    value=json.loads(Path(path).read_text());s=value['state']
    for key in ['x','y','scale','angle','opacity','left','right','top','bottom']:
        if isinstance(s[key],bool) or not math.isfinite(s[key]):raise ValueError('Invalid alignment '+key)
    if not 0<s['scale']<=25 or not 0<=s['opacity']<=1:raise ValueError('Invalid alignment scale/opacity')
    for k in ['flipX','flipY']:
        if k in s and not isinstance(s[k],bool):raise ValueError('Invalid mirror setting')
    crop_bounds(value)
    return value

def crop_bounds(alignment):
    s=alignment['state'];v=[s[k] for k in ['left','top','right','bottom']]
    if any(isinstance(x,bool) or int(x)!=x or x<0 for x in v):raise ValueError('Crop must use nonnegative integer pixels')
    l,t,r,b=map(int,v)
    if l+r>=1600 or t+b>=1200:raise ValueError('Crop is empty')
    return l,t,1600-r,1200-b

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def resolve_session_alignment(session,explicit=None):
    path=Path(explicit) if explicit else Path(session)/'alignment.json'
    if not path.is_file():raise ValueError('No alignment snapshot in this recording. Supply --alignment explicitly for older captures.')
    load_alignment(path)
    manifest=json.loads((Path(session)/'pairs.json').read_text())
    recorded=manifest.get('alignment_sha256')
    if recorded and sha256(path)!=recorded:raise ValueError('Alignment differs from the recording snapshot')
    return path


def cropped_box_to_full(box,alignment):
    left,top,_,_=crop_bounds(alignment)
    x0,y0,x1,y1=box
    return [x0+left,y0+top,x1+left,y1+top]
