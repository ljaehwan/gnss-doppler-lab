"""CLIF-IP R3 leakage-safe cross-layer primitives.

All estimators in this module are fitted on normal cleanStatic data only.  PRN
identity is deliberately absent: one shared predictor is applied to every PRN.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, json
import numpy as np
from sklearn.covariance import LedoitWolf

EPS=1e-9
VALID_GRADES={"verified","reconstructed","provisional"}


def finite(x): return np.nan_to_num(np.asarray(x,dtype=float),nan=0.,posinf=0.,neginf=0.)
def robust(x):
    x=finite(x); med=np.median(x,axis=0); mad=1.4826*np.median(np.abs(x-med),axis=0); sd=np.std(x,axis=0)
    return med,np.where(mad>EPS,mad,np.where(sd>EPS,sd,1.))

def _hash_arrays(meta, arrays):
    h=hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(",",":")).encode())
    for a in arrays: h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

@dataclass
class M1FitAudit:
    fit_count:int=0
    fit_recordings:list[str]=field(default_factory=list)
    transform_recordings:list[str]=field(default_factory=list)

@dataclass(frozen=True)
class FrozenM1State:
    mean:np.ndarray; scale:np.ndarray; components:np.ndarray; ar_coef:np.ndarray
    residual_center:np.ndarray; residual_scale:np.ndarray; residual_covariance:np.ndarray
    pca_dim:int; lag:int; train_end:float; fit_rows:int; sha256:str


def fit_m1(X,t,train_end=240.,pca_dim=8,lag=6,audit=None,recording="cleanStatic"):
    """Fit normalization/PCA/AR/residual statistics exactly once on t<=train_end."""
    if audit is not None:
        if audit.fit_count: raise RuntimeError("M1 state may be fitted exactly once")
        audit.fit_count+=1; audit.fit_recordings.append(recording)
    X=finite(X); t=finite(t); mask=t<=train_end
    if mask.sum()<=lag+pca_dim: raise ValueError("insufficient clean training rows")
    mu=X[mask].mean(0); scale=X[mask].std(0); scale=np.where(scale>EPS,scale,1.)
    Z=(X-mu)/scale; _,_,vt=np.linalg.svd(Z[mask],full_matrices=False); comp=vt[:min(pca_dim,vt.shape[0])].T
    P=Z@comp; ids=np.flatnonzero(mask)
    valid=ids[ids>=lag]
    valid=np.array([i for i in valid if mask[i-lag:i+1].all()],int)
    A=np.stack([P[i-lag:i].reshape(-1) for i in valid]); Y=P[valid]
    coef=np.linalg.lstsq(A,Y,rcond=None)[0]; residual=Y-A@coef
    center,rscale=robust(residual); cov=LedoitWolf().fit(residual-center).covariance_
    meta={"pca_dim":comp.shape[1],"lag":lag,"train_end":float(train_end),"fit_rows":int(mask.sum())}
    digest=_hash_arrays(meta,[mu,scale,comp,coef,center,rscale,cov])
    return FrozenM1State(mu,scale,comp,coef,center,rscale,cov,comp.shape[1],lag,float(train_end),int(mask.sum()),digest)


def transform_m1(X,t,state,recording="unknown",audit=None):
    """Transform one recording with a reset AR history; never mutates/fits state."""
    if audit is not None: audit.transform_recordings.append(recording)
    before=state.sha256; X=finite(X); P=((X-state.mean)/state.scale)@state.components
    innovation=np.full_like(P,np.nan); score=np.full(len(P),np.nan)
    inv=np.linalg.pinv(state.residual_covariance)
    for i in range(state.lag,len(P)):
        pred=P[i-state.lag:i].reshape(-1)@state.ar_coef
        innovation[i]=P[i]-pred
        d=innovation[i]-state.residual_center; score[i]=float(np.sqrt(max(0,d@inv@d)))
    if state.sha256!=before: raise RuntimeError("frozen state mutated")
    return {"pca":P,"innovation":innovation,"score":score,"t":finite(t)}


def make_history(b0,m1,lag,include_b0=True,include_m1=True):
    """Causal designs: B0 strictly past, M1 past through current, target current B0."""
    b0=finite(b0); m1=finite(m1); rows=[]; ys=[]; idx=[]
    for i in range(lag,len(b0)):
        z=[]
        if include_b0: z.extend(b0[i-lag:i].reshape(-1))
        if include_m1: z.extend(m1[i-lag:i+1].reshape(-1))
        rows.append(z); ys.append(b0[i]); idx.append(i)
    return np.asarray(rows,float),np.asarray(ys,float),np.asarray(idx,int)

def true_exceed_fraction(scores,threshold):
    scores=finite(scores); return float(np.count_nonzero(scores>threshold)/len(scores)) if len(scores) else 0.
def aggregate_prns(scores):
    a=finite(scores); row=np.sqrt(np.sum(a*a,axis=1)) if a.ndim>1 else a
    k=max(1,min(3,len(row)))
    return {"median":float(np.median(row)),"q90":float(np.quantile(row,.9)),"topk_mean":float(np.sort(row)[-k:].mean()),"tracked_count":int(len(row))}
def fit_threshold(normal_scores,q=.99): return float(np.quantile(finite(normal_scores),q))
def independent_fpr(test_scores,threshold): return float(np.mean(finite(test_scores)>threshold))
def shuffle_pairing(b0,m1,seed=0,block=8):
    b0=np.asarray(b0).copy(); m1=np.asarray(m1).copy(); n=len(m1); starts=list(range(0,n,block)); rng=np.random.default_rng(seed); order=rng.permutation(len(starts));
    shuffled=np.concatenate([m1[starts[j]:min(starts[j]+block,n)] for j in order],axis=0)
    return b0,shuffled
def validate_provenance_grade(grade):
    if grade not in VALID_GRADES: raise ValueError(f"grade must be one of {sorted(VALID_GRADES)}")
    return grade
