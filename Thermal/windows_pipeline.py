"""Mac -> Windows annotation -> Mac review; reviewed dataset -> Windows training -> Mac model.
All exchanges are checksum verified. Interrupted jobs remain separate and can be inspected.
"""
import argparse, base64, hashlib, json, os, shutil, subprocess, sys, tempfile, uuid
from pathlib import Path
from remote_job import pack, unpack, digest
from alignment_preset import resolve_session_alignment,sha256
from morph_field import MorphField,DEFAULT_MORPH
SSH_CONFIG=Path.home()/'.ssh/occupancy_windows.conf'
REMOTE='D:/fluxnet/automation'
REPO=Path('/Users/haneefrahman2650/Developer/Occupancy-Detection-Program/Thermal')
PY=REMOTE+'/.venv/Scripts/python.exe'

def psquote(s):return "'"+str(s).replace("'","''")+"'"
def remote(script):
    encoded=base64.b64encode(("$ErrorActionPreference='Stop'; "+script).encode('utf-16le')).decode()
    subprocess.run(['ssh','-F',str(SSH_CONFIG),'occupancy-windows','powershell -NoProfile -NonInteractive -EncodedCommand '+encoded],check=True)
def runpy(script,*args):
    remote('Set-Location '+psquote(REMOTE)+'; & '+psquote(PY)+' -X utf8 '+psquote(REMOTE+'/'+script)+' '+' '.join(psquote(a) for a in args)+'; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }')
def copy(src,dst):
    subprocess.run(['scp','-F',str(SSH_CONFIG),str(src),str(dst)],check=True)
def upload(root,job,kind,work):
    archive=work/(kind+'.zip');pack(root,archive)
    copy(archive,'occupancy-windows:'+job+'/'+kind+'.zip')
    runpy('remote_job.py','unpack',job+'/'+kind+'.zip',job+'/'+kind)
def download(root,job,kind,work):
    runpy('remote_job.py','pack',root,job+'/'+kind+'.zip')
    archive=work/(kind+'.zip');copy('occupancy-windows:'+job+'/'+kind+'.zip',archive)
    out=work/kind;unpack(archive,out);return out

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('annotate');a.add_argument('session',type=Path);a.add_argument('--alignment',type=Path,help='Override only for older sessions without a saved preset');a.add_argument('--experimental',action='store_true');a.add_argument('--experimental-annotator',action='store_true');a.add_argument('--logs',type=Path,default=REPO/'logs');a.add_argument('--morph',type=Path,default=DEFAULT_MORPH)
    t=sub.add_parser('train');t.add_argument('dataset',type=Path);t.add_argument('--models',type=Path,default=REPO/'models');t.add_argument('--version',type=int,required=True)
    args=p.parse_args()
    if os.geteuid()==0:raise SystemExit('Run transfers as your normal Mac user, not sudo.')
    if args.command=='train':
        if not (args.dataset/'review_audit.json').is_file():raise SystemExit('Use build-reviewed first; missing review audit.')
        if (args.models/f'v{args.version}').exists():raise FileExistsError('Model version already exists')
    else:
        args.alignment=resolve_session_alignment(args.session,args.alignment)
        MorphField(args.morph,sha256(args.alignment))
        if not args.session.name.startswith('capture_'):raise ValueError('Session name must start capture_')
        if (args.logs/args.session.name).exists():raise FileExistsError('Capture already exists in logs')
    job=REMOTE+'/jobs/'+uuid.uuid4().hex
    remote('New-Item -ItemType Directory -Path '+psquote(job)+' | Out-Null')
    with tempfile.TemporaryDirectory(prefix='occupancy-transfer-') as tmp:
        work=Path(tmp)
        if args.command=='annotate':
            if not args.experimental_annotator:raise ValueError('Current YOLOE candidate requires --experimental-annotator; it failed initial omega evaluation')
            upload(args.session,job,'input',work);copy(args.alignment,'occupancy-windows:'+job+'/alignment.json');copy(args.morph,'occupancy-windows:'+job+'/morph.json')
            extra=['--experimental-annotator']+(['--allow-experimental-alignment'] if args.experimental else [])
            runpy('rgb_annotation_worker.py',job+'/input','--out',job+'/output','--alignment',job+'/alignment.json','--weights',REMOTE+'/yoloe-26x-seg.pt','--morph',job+'/morph.json',*extra)
            out=download(job+'/output',job,'returned',work)
            if not (out/'READY_FOR_REVIEW').is_file():raise ValueError('Annotation incomplete')
            if (out/'rgb_annotation.json').is_file() and json.loads((out/'rgb_annotation.json').read_text()).get('integration_test_only') and args.logs.resolve()==(REPO/'logs').resolve():raise ValueError('Integration tests must stay outside production logs')
            dest=args.logs/args.session.name
        else:
            upload(args.dataset,job,'dataset',work)
            runpy('train_next.py','--dataset',job+'/dataset','--project','D:/fluxnet/runs','--version',str(args.version),'--review-confirmed')
            out=download(f'D:/fluxnet/runs/v{args.version}/weights',job,'model',work)
            meta=json.loads((out/'handoff.json').read_text())
            if digest(out/'best.pt')!=meta['sha256']:raise ValueError('Model checksum mismatch')
            dest=args.models/f'v{args.version}'
        dest.parent.mkdir(parents=True,exist_ok=True)
        staging=dest.parent/('.incoming-'+uuid.uuid4().hex)
        shutil.copytree(out,staging)
        (staging/'remote_job.json').write_text(json.dumps({'job':job,'command':args.command},indent=2))
        if dest.exists():raise FileExistsError(dest)
        staging.rename(dest)
        print('Published:',dest)
if __name__=='__main__':main()
