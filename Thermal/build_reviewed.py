"""Build only human-reviewed captures, with whole sessions held out for validation."""
import argparse,json,subprocess,sys
from pathlib import Path
from remote_job import digest
REPO=Path('/Users/haneefrahman2650/Developer/Occupancy-Detection-Program/Thermal')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('captures',nargs='+',type=Path);p.add_argument('--val',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--train-on-windows',action='store_true');p.add_argument('--version',type=int);a=p.parse_args()
    if a.train_on_windows and a.version is None:p.error('--version is required with --train-on-windows')
    caps=[x.resolve() for x in a.captures];val={x.resolve() for x in a.val}
    if len(set(caps))!=len(caps) or not val or not val<set(caps):p.error('Choose distinct captures and hold out some, not all, complete sessions')
    if a.out.exists():raise FileExistsError(a.out)
    audit=[];seen={}
    for cap in caps:
        if (cap/'rgb_annotation.json').exists() and json.loads((cap/'rgb_annotation.json').read_text()).get('integration_test_only'):raise ValueError('Integration test cannot enter training')
        frames=sorted((cap/'npy').glob('*.npy'))
        if not frames:raise ValueError('Empty capture: '+str(cap))
        for frame in frames:
            lab=cap/'labels_human'/(frame.stem+'.txt')
            if not lab.is_file():raise ValueError('Frame not human-reviewed: '+str(frame))
            for line in lab.read_text().splitlines():
                if not line.strip():continue
                v=line.split()
                if len(v)!=5 or v[0]!='1':raise ValueError('Only class-1 omega labels accepted: '+str(lab))
                coords=list(map(float,v[1:]))
                if not all(0<=x<=1 for x in coords) or min(coords[2:])<=0:raise ValueError('Invalid box: '+str(lab))
            split='val' if cap in val else 'train';sha=digest(frame)
            if sha in seen and seen[sha]!=split:raise ValueError('Duplicate thermal frame crosses train/val split')
            seen[sha]=split
            audit.append({'capture':str(cap),'stem':frame.stem,'split':split,'thermal_sha256':sha,'human_label_sha256':digest(lab)})
    subprocess.run([sys.executable,str(REPO/'make_dataset.py'),*[str(c) for c in caps],'--out',str(a.out),'--train-source','human','--val-sessions',','.join(str(i) for i,c in enumerate(caps) if c in val)],check=True)
    files={p.relative_to(a.out).as_posix():digest(p) for p in a.out.rglob('*') if p.is_file()}
    (a.out/'review_audit.json').write_text(json.dumps({'schema':1,'human_only':True,'split_policy':'whole capture sessions','frames':audit,'files':files},indent=2))
    print('Reviewed dataset:',a.out)
    if a.train_on_windows:subprocess.run([sys.executable,str(Path(__file__).with_name('windows_pipeline.py')),'train',str(a.out),'--version',str(a.version)],check=True)
if __name__=='__main__':main()
