"""Shared, frozen numerical implementation for PF-DIC schema v3."""
from __future__ import annotations
import math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from scipy.special import ndtri
from torch import nn
from torch.utils.data import DataLoader, Dataset

DISTRIBUTIONS={"gaussian","student_t"}

@dataclass
class ModelConfig:
    peak_input_dim:int
    floor_input_dim:int
    hidden_dim:int=64
    layers:int=1
    dropout:float=0.
    distribution:str="gaussian"
    nu:float=5.
    context_len:int=12
    def __post_init__(self):
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {sorted(DISTRIBUTIONS)}, got {self.distribution!r}")
        if not np.isfinite(self.nu) or self.nu<=2:
            raise ValueError("nu must be finite and > 2")
        if min(self.peak_input_dim,self.floor_input_dim,self.hidden_dim,self.layers,self.context_len)<1:
            raise ValueError("model dimensions, layers, and context_len must be positive")
        if not np.isfinite(self.dropout) or not 0<=self.dropout<1:
            raise ValueError("dropout must be in [0, 1)")

class DynamicInnovationModel(nn.Module):
    def __init__(self,cfg:ModelConfig):
        super().__init__(); self.cfg=cfg; drop=cfg.dropout if cfg.layers>1 else 0.
        self.peak_gru=nn.GRU(cfg.peak_input_dim,cfg.hidden_dim,cfg.layers,batch_first=True,dropout=drop)
        self.floor_gru=nn.GRU(cfg.floor_input_dim,cfg.hidden_dim,cfg.layers,batch_first=True,dropout=drop)
        self.peak_head=nn.Linear(cfg.hidden_dim,2*cfg.peak_input_dim)
        self.floor_head=nn.Linear(cfg.hidden_dim,2*cfg.floor_input_dim)
    def forward(self,peak_context,floor_context):
        p=self.peak_head(self.peak_gru(peak_context)[0][:,-1]); f=self.floor_head(self.floor_gru(floor_context)[0][:,-1])
        pm,ps=p.chunk(2,-1); fm,fs=f.chunk(2,-1)
        return {"peak_mean":pm,"peak_log_scale":ps.clamp(-7,5),"floor_mean":fm,"floor_log_scale":fs.clamp(-7,5)}

def masked_peak_summary(x,mask):
    w=mask.to(x.dtype).unsqueeze(-1); n=w.sum(-2).clamp_min(1.); mean=(x*w).sum(-2)/n
    var=((x-mean.unsqueeze(-2)).square()*w).sum(-2)/n
    return torch.cat((mean,torch.sqrt(var.clamp_min(1e-8))),-1)

def gaussian_nll(y,mu,log_scale,reduction="mean"):
    ls=log_scale.clamp(-7,5); value=.5*((y-mu)*torch.exp(-ls)).square()+ls+.5*math.log(2*math.pi)
    return value.mean() if reduction=="mean" else value.sum(-1)

def student_t_nll(y,mu,log_scale,nu=5.,reduction="mean"):
    if not np.isfinite(nu) or nu<=2: raise ValueError("student_t nu must be finite and > 2")
    ls=log_scale.clamp(-7,5); z=(y-mu)*torch.exp(-ls); n=torch.as_tensor(nu,device=y.device,dtype=y.dtype)
    value=torch.lgamma(n/2)-torch.lgamma((n+1)/2)+.5*torch.log(n*math.pi)+ls+.5*(n+1)*torch.log1p(z.square()/n)
    return value.mean() if reduction=="mean" else value.sum(-1)

def innovation_nll(y,mu,ls,cfg,reduction="mean"):
    return student_t_nll(y,mu,ls,cfg.nu,reduction) if cfg.distribution=="student_t" else gaussian_nll(y,mu,ls,reduction)

class PairDataset(Dataset):
    def __init__(self,pairs): self.p=pairs
    def __len__(self): return len(self.p)
    def __getitem__(self,i):
        p=self.p
        return (masked_peak_summary(torch.from_numpy(p.context_morph[i]),torch.from_numpy(p.context_mask[i])),
                torch.from_numpy(p.context_floor[i]),
                masked_peak_summary(torch.from_numpy(p.target_morph[i]),torch.from_numpy(p.target_mask[i])),
                torch.from_numpy(p.target_floor[i]))

def seed_all(seed:int,deterministic:bool=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    try:
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception as exc: raise RuntimeError(f"CUDA initialization failed while seeding: {exc}") from exc
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends,"cudnn"):
            torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def resolve_device(requested:str):
    requested=str(requested)
    if requested!="auto" and requested!="cpu" and not requested.startswith("cuda"):
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
    try: available=bool(torch.cuda.is_available())
    except Exception as exc:
        if requested.startswith("cuda"): raise RuntimeError(f"CUDA initialization failed: {exc}") from exc
        available=False
    resolved="cuda" if requested=="auto" and available else ("cpu" if requested=="auto" else requested)
    if resolved.startswith("cuda") and not available: raise RuntimeError(f"CUDA device {requested!r} requested but CUDA is unavailable")
    try:
        device=torch.device(resolved)
        name=torch.cuda.get_device_name(device) if device.type=="cuda" else None
    except Exception as exc: raise RuntimeError(f"CUDA device initialization failed for {resolved!r}: {exc}") from exc
    return device,{"requested":requested,"resolved":str(device),"cuda_name":name}

def validate_training_options(*,epochs,batch_size,context_len,hidden_dim,pca_dim,rank,lr,shrinkage):
    vals={"epochs":epochs,"batch_size":batch_size,"context_len":context_len,"hidden_dim":hidden_dim,"pca_dim":pca_dim,"rank":rank}
    for name,value in vals.items():
        if not isinstance(value,(int,np.integer)) or value<1: raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(lr) or not 0<lr<=1: raise ValueError("lr must be finite and in (0, 1]")
    if not np.isfinite(shrinkage) or not 0<=shrinkage<1: raise ValueError("shrinkage must be finite and in [0, 1)")

def innovations(model,pairs,batch_size,device):
    model.eval(); peak=[]; floor=[]; ps=[]; fs=[]
    with torch.no_grad():
        for batch in DataLoader(PairDataset(pairs),batch_size=batch_size):
            pc,fc,pt,ft=[x.to(device) for x in batch]; o=model(pc,fc)
            peak.append(((pt-o["peak_mean"])*torch.exp(-o["peak_log_scale"])).cpu().numpy())
            floor.append(((ft-o["floor_mean"])*torch.exp(-o["floor_log_scale"])).cpu().numpy())
            ps.extend(innovation_nll(pt,o["peak_mean"],o["peak_log_scale"],model.cfg,"none").cpu().numpy())
            fs.extend(innovation_nll(ft,o["floor_mean"],o["floor_log_scale"],model.cfg,"none").cpu().numpy())
    return np.concatenate(peak),np.concatenate(floor),np.asarray(ps),np.asarray(fs)

def fit_pca_projection(x,max_dim):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all(): raise ValueError("PCA input must be finite 2-D with at least two rows")
    mean=x.mean(0); _,s,vt=np.linalg.svd(x-mean,full_matrices=False)
    tol=np.finfo(s.dtype).eps*max(x.shape)*(s[0] if len(s) else 0.); numerical_rank=int(np.sum(s>tol))
    dim=min(int(max_dim),numerical_rank)
    if dim<1: raise ValueError("PCA input has zero numerical rank")
    components=vt[:dim].copy(); projected=(x-mean)@components.T
    references=np.sort(projected,axis=0)
    return {"mean":mean,"components":components,"references":references,"numerical_rank":numerical_rank}

def empirical_gaussianize(values,references):
    values=np.asarray(values,dtype=np.float64); refs=np.asarray(references,dtype=np.float64)
    if values.ndim!=2 or refs.ndim!=2 or values.shape[1]!=refs.shape[1] or len(refs)<2: raise ValueError("Gaussianization shape mismatch")
    out=np.empty_like(values)
    n=len(refs)
    for j in range(values.shape[1]):
        left=np.searchsorted(refs[:,j],values[:,j],side="left"); right=np.searchsorted(refs[:,j],values[:,j],side="right")
        probability=(left+right+1)/(2*(n+1)); out[:,j]=ndtri(np.clip(probability,1/(2*(n+1)),1-1/(2*(n+1))))
    if not np.isfinite(out).all(): raise ValueError("Gaussianization produced non-finite values")
    return out

def transform_and_gaussianize(x,projection):
    raw=(np.asarray(x)-projection["mean"])@projection["components"].T
    return empirical_gaussianize(raw,projection["references"])

def fit_relation(peak,floor,rank=4,shrinkage=.1,clip=.95):
    peak=np.asarray(peak); floor=np.asarray(floor); n=max(1,len(peak)-1)
    C=(peak-peak.mean(0)).T@(floor-floor.mean(0))/n; u,s,vt=np.linalg.svd(C,full_matrices=False); r=min(rank,len(s))
    s=np.clip((1-shrinkage)*s[:r],0,clip); C=(u[:,:r]*s)@vt[:r]
    norm=np.linalg.svd(C,compute_uv=False)[0] if C.size else 0
    if norm>=clip: C*=clip/norm
    R=np.block([[np.eye(C.shape[0]),C],[C.T,np.eye(C.shape[1])]])
    sign,ld=np.linalg.slogdet(R)
    if sign<=0: raise ValueError("relation covariance is not positive definite")
    return {"C":C,"R":R,"inverse":np.linalg.inv(R),"logdet":float(ld),"rank":r,"shrinkage":shrinkage}

def relation_nll(peak,floor,relation):
    z=np.concatenate((peak,floor),1)
    return .5*(np.einsum("ni,ij,nj->n",z,relation["inverse"],z)+relation["logdet"]+z.shape[1]*math.log(2*math.pi))
def relation_scores(peak,floor,relation):
    nr=relation_nll(peak,floor,relation); d=peak.shape[1]+floor.shape[1]
    return nr-.5*((peak*peak).sum(1)+(floor*floor).sum(1)+d*math.log(2*math.pi))
def relation_deviation_scores(scores,center): return np.abs(np.asarray(scores)-float(center))
def tail_pvalues(x,cal):
    x=np.asarray(x); cal=np.asarray(cal); return np.asarray([(1+np.sum(cal>=v))/(len(cal)+1) for v in x])

def projection_from_npz(z,prefix):
    return {"mean":z[f"{prefix}_mean"],"components":z[f"{prefix}_components"],"references":z[f"{prefix}_references"]}
def load_relation_projection(path:Path):
    with np.load(path,allow_pickle=False) as z:
        peak=projection_from_npz(z,"peak"); floor=projection_from_npz(z,"floor")
        relation={k:z[k] for k in ("C","R","inverse")}; relation["logdet"]=float(z["logdet"])
        center=float(z["relation_center"])
    return {"peak":peak,"floor":floor,"relation":relation,"relation_center":center}
