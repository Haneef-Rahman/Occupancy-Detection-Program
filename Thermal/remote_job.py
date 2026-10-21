"""Verified archive exchange. Only project-owned job directories are accepted."""
import argparse, hashlib, json, zipfile
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def pack(root, dest):
    files=sorted(p for p in root.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Symlinks not allowed')
    manifest={p.relative_to(root).as_posix():digest(p) for p in files}
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(root).as_posix())
        z.writestr('_transfer_manifest.json',json.dumps(manifest))

def unpack(archive, dest):
    if dest.exists():raise FileExistsError(dest)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('_transfer_manifest.json'))
        if set(z.namelist())!=set(manifest)|{'_transfer_manifest.json'}:raise ValueError('Archive manifest mismatch')
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate paths')
        for name in manifest:
            p=Path(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe path')
        dest.mkdir(parents=True)
        for name,sha in manifest.items():
            data=z.read(name)
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Transfer checksum mismatch: '+name)
            p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pack','unpack']);p.add_argument('source',type=Path);p.add_argument('dest',type=Path);a=p.parse_args()
    if a.mode=='pack':pack(a.source,a.dest)
    else:unpack(a.source,a.dest)
    print('VERIFIED',a.dest)
