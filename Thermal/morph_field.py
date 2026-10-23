"""Reproducible NumPy thin-plate RBF displacement for annotation boxes only."""
from pathlib import Path
import hashlib,json
import numpy as np

DEFAULT_MORPH=Path(__file__).resolve().parent/'calibration'/'morph-20261023.json'

def thin_plate(a,b):
    r2=np.sum((a[:,None,:]-b[None,:,:])**2,axis=2)
    k=np.zeros_like(r2);positive=r2>0
    k[positive]=.5*r2[positive]*np.log(r2[positive])
    return k

def fit_coefficients(centres,residuals,normalization,smoothing):
    x=np.asarray(centres,float)/normalization;y=np.asarray(residuals,float)
    p=np.column_stack([np.ones(len(x)),x])
    system=np.block([[thin_plate(x,x)+smoothing*np.eye(len(x)),p],[p.T,np.zeros((3,3))]])
    coefficients=np.linalg.solve(system,np.vstack([y,np.zeros((3,2))]))
    return coefficients[:-3],coefficients[-3:]

class MorphField:
    def __init__(self,path,alignment_sha256):
        self.path=Path(path);raw=self.path.read_bytes();self.sha256=hashlib.sha256(raw).hexdigest();m=json.loads(raw)
        if m.get('schema')!=1 or m.get('method')!='thin_plate_rbf_affine':raise ValueError('Unsupported morph field format')
        if m.get('alignment_sha256')!=alignment_sha256:raise ValueError('Morph calibration does not match the recording alignment')
        if m.get('frame_wh')!=[160,120] or m.get('coordinate_space')!='aligned_native_ir_edge_pixels':raise ValueError('Unsupported morph coordinates')
        self.normalization=float(m['normalization_divisor']);self.strength=float(m['strength'])
        self.outside_policy=m['outside_policy']
        if not np.isfinite([self.normalization,self.strength]).all() or self.normalization<=0 or not 0<=self.strength<=1:raise ValueError('Invalid morph scaling')
        if self.outside_policy not in ('extrapolate','identity'):raise ValueError('Unknown outside-support policy')
        self.centres=np.asarray(m['centres'],float);self.weights=np.asarray(m['rbf_weights'],float);self.polynomial=np.asarray(m['affine_coefficients'],float);self.hull=np.asarray(m['support_hull'],float)
        if self.centres.ndim!=2 or self.centres.shape[1]!=2 or len(self.centres)<3 or self.weights.shape!=self.centres.shape or self.polynomial.shape!=(3,2) or self.hull.ndim!=2 or self.hull.shape[1]!=2 or len(self.hull)<3:raise ValueError('Invalid morph coefficient shapes')
        if not all(np.isfinite(v).all() for v in [self.centres,self.weights,self.polynomial,self.hull]):raise ValueError('Nonfinite morph coefficients')
        self.metadata=m
    def inside_support(self,point):
        point=np.asarray(point,float);edges=np.roll(self.hull,-1,axis=0)-self.hull;v=point-self.hull
        cross=edges[:,0]*v[:,1]-edges[:,1]*v[:,0]
        return bool(np.all(cross>=-1e-7) or np.all(cross<=1e-7))
    def evaluate(self,points):
        p=np.asarray(points,float)
        if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():raise ValueError('Expected finite Nx2 morph query')
        x=p/self.normalization
        values=thin_plate(x,self.centres/self.normalization)@self.weights+np.column_stack([np.ones(len(x)),x])@self.polynomial
        if self.outside_policy=='identity':values=np.array([v if self.inside_support(q) else np.zeros(2) for q,v in zip(p,values)])
        values*=self.strength
        if not np.isfinite(values).all():raise ValueError('Nonfinite morph displacement')
        return values
    def displacement(self,centre):
        value=self.evaluate(np.asarray(centre,float).reshape(1,2))[0]
        return value,dict(centre_ir=list(map(float,centre)),displacement_ir=value.tolist(),inside_support=self.inside_support(centre),extrapolated=not self.inside_support(centre) and self.outside_policy=='extrapolate')
    def provenance(self):
        return dict(file=self.path.name,sha256=self.sha256,alignment_sha256=self.metadata['alignment_sha256'],coverage_percent=self.metadata['coverage_percent'],strength=self.strength,outside_policy=self.outside_policy,fit_points=len(self.centres),units='native thermal pixels',application='uniform translation at provisional box centre; final clipping only')
