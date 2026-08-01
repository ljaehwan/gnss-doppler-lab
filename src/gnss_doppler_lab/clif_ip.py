"""CLIF-IP R3 leakage-safe cross-layer primitives.

All estimators are fitted on normal cleanStatic only. PRN identity is absent.
Frozen M1 arrays are defensive, read-only copies and are content-verified at
both entry and exit of every transform.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

EPS=1e-9
VALID_GRADES={"verified","reconstructed","provisional"}


def finite(x): return np.nan_to_num(np.asarray(x,dtype=float),nan=0.,posinf=0.,neginf=0.)
def robust(x):
    x=finite(x); med=np.median(x,axis=0); mad=1.4826*np.median(np.abs(x-med),axis=0); sd=np.std(x,axis=0)
    return med,np.where(mad>EPS,mad,np.where(sd>EPS,sd,1.))


def _hash_arrays(meta,arrays):
    h=hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(",",":")).encode())
    for a in arrays: h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _readonly(a):
    z=np.array(a,dtype=float,copy=True); z.setflags(write=False); return z


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


def _state_meta(s):
    return {"pca_dim":s.pca_dim,"lag":s.lag,"train_end":float(s.train_end),"fit_rows":s.fit_rows}


def state_content_hash(s):
    return _hash_arrays(_state_meta(s),[s.mean,s.scale,s.components,s.ar_coef,s.residual_center,s.residual_scale,s.residual_covariance])


def fit_m1(X,window_start,window_end,train_end=240.,pca_dim=8,lag=6,audit=None,recording="cleanStatic"):
    """Fit once using only windows with start >= 0 and end <= train_end."""
    if audit is not None:
        if audit.fit_count: raise RuntimeError("M1 state may be fitted exactly once")
        audit.fit_count+=1; audit.fit_recordings.append(recording)
    X=finite(X); start=finite(window_start); end=finite(window_end)
    if not (len(X)==len(start)==len(end)): raise ValueError("window arrays differ in length")
    mask=(start>=0)&(end<=train_end)
    if mask.sum()<=lag+pca_dim: raise ValueError("insufficient clean training rows")
    train=X[mask]; mu=train.mean(0); scale=train.std(0); scale=np.where(scale>EPS,scale,1.)
    ztrain=(train-mu)/scale; _,_,vt=np.linalg.svd(ztrain,full_matrices=False); comp=vt[:min(pca_dim,vt.shape[0])].T
    P=ztrain@comp
    A=np.stack([P[i-lag:i].reshape(-1) for i in range(lag,len(P))]); Y=P[lag:]
    coef=np.linalg.lstsq(A,Y,rcond=None)[0]; residual=Y-A@coef
    center=residual.mean(0); _,rscale=robust(residual); cov=LedoitWolf().fit(residual-center).covariance_
    meta={"pca_dim":comp.shape[1],"lag":lag,"train_end":float(train_end),"fit_rows":int(mask.sum())}
    arrays=[_readonly(a) for a in (mu,scale,comp,coef,center,rscale,cov)]
    digest=_hash_arrays(meta,arrays)
    return FrozenM1State(*arrays,comp.shape[1],lag,float(train_end),int(mask.sum()),digest)


def transform_m1(X,t,state,recording="unknown",audit=None,reset=True):
    """Transform one complete recording/split from an empty AR history."""
    if not reset: raise ValueError("CLIF-IP requires reset=True for every recording/split")
    if state_content_hash(state)!=state.sha256: raise RuntimeError("frozen state mutated before transform")
    if audit is not None: audit.transform_recordings.append(recording)
    X=finite(X); P=((X-state.mean)/state.scale)@state.components
    innovation=np.full_like(P,np.nan); score=np.full(len(P),np.nan); inv=np.linalg.pinv(state.residual_covariance)
    for i in range(state.lag,len(P)):
        pred=P[i-state.lag:i].reshape(-1)@state.ar_coef
        innovation[i]=P[i]-pred; d=innovation[i]-state.residual_center
        score[i]=float(np.sqrt(max(0,d@inv@d)))
    if state_content_hash(state)!=state.sha256: raise RuntimeError("frozen state mutated during transform")
    return {"pca":P,"innovation":innovation,"score":score,"t":finite(t)}


def slice_whole_windows(frame,lo,hi,start_col="window_start_s",end_col="window_end_s"):
    """Select raw windows by whole-window containment, preserving original indices."""
    return frame[(frame[start_col]>=lo)&(frame[end_col]<=hi)].copy()


def make_history(b0,m1,lag,include_b0=True,include_m1=True):
    """Causal split-local designs: B0 strictly past; M1 through current."""
    b0=finite(b0); m1=finite(m1); rows=[]; ys=[]; idx=[]
    for i in range(lag,len(b0)):
        z=[]
        if include_b0:z.extend(b0[i-lag:i].reshape(-1))
        if include_m1:z.extend(m1[i-lag:i+1].reshape(-1))
        rows.append(z);ys.append(b0[i]);idx.append(i)
    return np.asarray(rows,float),np.asarray(ys,float),np.asarray(idx,int)


@dataclass(frozen=True)
class WhiteningState:
    center:np.ndarray
    covariance:np.ndarray
    columns:tuple[str,...]=()
    scale:np.ndarray|None=None
    @property
    def dimension(self): return len(self.center)


def fit_whitener(residuals):
    """Validation-only residual mean and Ledoit-Wolf covariance."""
    r=finite(residuals); center=r.mean(0); cov=LedoitWolf().fit(r-center).covariance_
    return WhiteningState(_readonly(center),_readonly(cov))


def mahalanobis_score(x,state):
    z=finite(x)-state.center
    if state.scale is not None:z=z/state.scale
    inv=np.linalg.pinv(state.covariance)
    return np.sqrt(np.maximum(0,np.einsum("ij,jk,ik->i",z,inv,z)))


def fit_component_calibrations(frame,specs):
    """Refit robust scale and shrinkage covariance for each exact component set."""
    out={}
    for name,columns in specs.items():
        x=finite(frame[list(columns)].to_numpy()); center,scale=robust(x); z=(x-center)/scale
        cov=LedoitWolf().fit(z).covariance_
        out[name]=WhiteningState(_readonly(center),_readonly(cov),tuple(columns),_readonly(scale))
    return out


def alarm_times(df,col,threshold,onset,persistence=3):
    """Alarm delays using score availability time (not target/window start time)."""
    avail=df["available_s"].to_numpy(float); flags=df[col].to_numpy(float)>threshold
    first=next((a for a,f in zip(avail,flags) if a>=onset and f),None)
    persistent=next((avail[i] for i in range(len(avail)-persistence+1) if avail[i]>=onset and flags[i:i+persistence].all()),None)
    return {"first_alarm_delay_s":-1. if first is None else float(first-onset),"persistent_delay_s":-1. if persistent is None else float(persistent-onset)}


def true_exceed_fraction(scores,threshold):
    scores=finite(scores);return float(np.count_nonzero(scores>threshold)/len(scores)) if len(scores) else 0.
def aggregate_prns(scores):
    a=finite(scores);row=np.sqrt(np.sum(a*a,axis=1)) if a.ndim>1 else a;k=max(1,min(3,len(row)))
    return {"median":float(np.median(row)),"q90":float(np.quantile(row,.9)),"topk_mean":float(np.sort(row)[-k:].mean()),"tracked_count":int(len(row))}
def fit_threshold(normal_scores,q=.99):return float(np.quantile(finite(normal_scores),q))
def independent_fpr(test_scores,threshold):return float(np.mean(finite(test_scores)>threshold))
def shuffle_pairing(b0,m1,seed=0,block=8):
    b0=np.asarray(b0).copy();m1=np.asarray(m1).copy();n=len(m1);starts=list(range(0,n,block));rng=np.random.default_rng(seed);order=rng.permutation(len(starts))
    shuffled=np.concatenate([m1[starts[j]:min(starts[j]+block,n)] for j in order],axis=0)
    return b0,shuffled
def validate_provenance_grade(grade):
    if grade not in VALID_GRADES:raise ValueError(f"grade must be one of {sorted(VALID_GRADES)}")
    return grade
