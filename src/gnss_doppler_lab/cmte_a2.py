"""Preregistered CMTE-A2 detector, chronological B0, comparators and metrics.

This module intentionally contains no e-value, capital, CUSUM, restart, online
normalisation, or sequential-detector implementation.  A2 is the symmetric
per-epoch mean of negative log finite-sample conformal p-values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import warnings
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

TAP_ORDER=("E4","E3","E2","E","P","L","L2","L3","L4")
FEATURE_COLUMNS=tuple(f"tap_{x}_rel_prompt_mean" for x in TAP_ORDER)
RESIDUAL_COLUMNS=tuple(f"residual_{i:03d}" for i in range(9))
PREREG_COMMIT="e7cb2e5822923a129d72c475706f87721ddd8104"
EPSILON=1e-8
A2_PRN_COLUMNS=("physical_recording_id","history_id","role","split","prn","segment","channel",
                "window_start_s","window_end_s","window_bin_s",*RESIDUAL_COLUMNS,"q","p","rmse")
A2_EPOCH_COLUMNS=("physical_recording_id","window_end_s","tracked_prn_count","min_p","median_p",
                  "mean_q","max_q","mean_neg_log_p","median_neg_log_p","score_A2")
_FORBIDDEN_SCHEMA_TOKENS=("mean_e","s1","s2","cusum","capital","sequential")


def file_sha256(path: str|Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=_json_default).encode()).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


@dataclass(frozen=True)
class B0Config:
    feature_dim:int=9
    hidden_dim:int=128
    seq_len:int=12
    dropout:float=.05
    learning_rate:float=1e-3
    weight_decay:float=1e-4
    adamw_betas:tuple[float,float]=(.9,.999)
    adamw_epsilon:float=1e-8
    epochs:int=25
    batch_size:int=256
    seed:int=11
    gradient_clip:float=1.


class B0Predictor(nn.Module):
    """Shared PRN-local causal B0 architecture with no PRN identity input."""
    def __init__(self,config:B0Config=B0Config()):
        super().__init__(); h=config.hidden_dim
        self.config=config
        self.encoder=nn.Sequential(nn.Linear(9,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(.05),nn.Linear(h,h),nn.GELU())
        self.gru=nn.GRU(h,h,num_layers=1,batch_first=True,bidirectional=False)
        self.head=nn.Sequential(nn.Linear(h,h),nn.GELU(),nn.Linear(h,9))
    def forward(self,x:torch.Tensor)->torch.Tensor:
        encoded=self.encoder(x); output,_=self.gru(encoded)
        return self.head(output[:,-1,:])


class _Examples(Dataset):
    def __init__(self,frame:pd.DataFrame):
        self.x=np.stack(frame.history).astype(np.float32); self.y=np.stack(frame.target).astype(np.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self,index): return torch.from_numpy(self.x[index]),torch.from_numpy(self.y[index])


def deterministic_controls(seed:int=11)->dict[str,Any]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=False)
    if hasattr(torch.backends,"cudnn"):
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    return {"seed":seed,"torch_deterministic_algorithms":True,"cudnn_deterministic":True,"cudnn_benchmark":False,
            "num_workers":0,"scheduler":None,"early_stopping":False}


def fit_standardizer(values:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    x=np.asarray(values,float)
    finite=np.where(np.isfinite(x),x,np.nan)
    with np.errstate(invalid="ignore",divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        mean=np.nanmean(finite,axis=0); std=np.nanstd(finite,axis=0,ddof=0)
    mean=np.where(np.isfinite(mean),mean,0.)
    std=np.where(np.isfinite(std)&(std>=1e-6),std,1.)
    return mean.astype(np.float64),std.astype(np.float64)


def standardize(values:np.ndarray,mean:np.ndarray,std:np.ndarray)->np.ndarray:
    out=(np.asarray(values,float)-np.asarray(mean,float))/np.asarray(std,float)
    return np.where(np.isfinite(out),out,0.)


def assign_normal_role(window_start_s:float,window_end_s:float)->str|None:
    start,end=float(window_start_s),float(window_end_s)
    if start>=0 and end<=240: return "prefix"
    if start>=250 and end<=290: return "qcal"
    if start>=300 and end<=330: return "threshold"
    if start>=340: return "clean_test"
    return None


def partition_normal_roles(nodes:pd.DataFrame)->dict[str,pd.DataFrame]:
    local=nodes.copy(); local["role"]=[assign_normal_role(a,b) for a,b in zip(local.window_start_s,local.window_end_s)]
    return {role:local[local.role==role].copy() for role in ("prefix","qcal","threshold","clean_test")}


def role_frame_audit(frame:pd.DataFrame,role:str)->dict[str,Any]:
    """Machine-readable role membership and content audit without serializing data."""
    if frame.empty: raise ValueError(f"{role} role must be nonempty")
    required={"window_start_s","window_end_s","prn"}
    missing=sorted(required-set(frame))
    if missing: raise ValueError(f"{role} audit missing {missing}")
    ordered=frame.sort_values([c for c in ("physical_recording_id","prn","segment","channel","window_end_s") if c in frame],kind="mergesort")
    digest=hashlib.sha256(pd.util.hash_pandas_object(ordered,index=False).values.tobytes()).hexdigest()
    return {"role":role,"rows":int(len(frame)),"prn_count":int(frame.prn.astype(str).nunique()),
            "prns":sorted(frame.prn.astype(str).unique()),"window_start_min_s":float(frame.window_start_s.min()),
            "window_start_max_s":float(frame.window_start_s.max()),"window_end_min_s":float(frame.window_end_s.min()),
            "window_end_max_s":float(frame.window_end_s.max()),"content_sha256":digest}


def audit_normal_roles(roles:Mapping[str,pd.DataFrame])->dict[str,Any]:
    expected=("prefix","qcal","threshold","clean_test")
    if set(roles)!=set(expected): raise ValueError("exact prefix/qcal/threshold/clean_test roles required")
    return {"roles":{name:role_frame_audit(roles[name],name) for name in expected},
            "role_overlap":False,"hash_method":"pandas_hash_rows_sha256"}


def select_prn_holdout(prefix:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    prns=sorted(prefix.prn.astype(str).unique())
    if len(prns)<2: raise ValueError("prefix requires at least two PRNs for sorted-PRN holdout")
    n=max(1,len(prns)//5); held=set(prns[-n:])
    return prefix[~prefix.prn.astype(str).isin(held)].copy(),prefix[prefix.prn.astype(str).isin(held)].copy()


def _validate_node_frame(frame:pd.DataFrame,features:Sequence[str])->pd.DataFrame:
    local=frame.copy()
    if "physical_recording_id" not in local:
        if "recording_id" in local: local["physical_recording_id"]=local.recording_id.astype(str)
        else: local["physical_recording_id"]=local.run_id.astype(str)
    for col,default in (("role","evaluation"),("split","evaluation"),("segment","0"),("channel","0")):
        if col not in local: local[col]=default
    required={"physical_recording_id","role","split","prn","segment","channel","window_start_s","window_end_s","window_bin_s",*features}
    missing=sorted(required-set(local))
    if missing: raise ValueError(f"node input missing {missing}")
    if not np.isfinite(local[["window_start_s","window_end_s","window_bin_s",*features]].to_numpy(float)).all():
        raise ValueError("node timing/features must be finite")
    return local


def build_history_examples(frame:pd.DataFrame,features:Sequence[str]=FEATURE_COLUMNS,*,seq_len:int=12,cadence_s:float=.5,
                           mean:np.ndarray|None=None,std:np.ndarray|None=None)->tuple[pd.DataFrame,dict[str,Any]]:
    local=_validate_node_frame(frame,features)
    keys=["role","split","physical_recording_id","segment","channel","prn"]
    rows=[]; gaps=0; chunk_count=0
    for key,group in local.sort_values([*keys,"window_end_s"],kind="mergesort").groupby(keys,sort=True,dropna=False):
        g=group.reset_index(drop=True); times=g.window_end_s.to_numpy(float)
        if len(times)>1 and np.any(np.diff(times)<=0): raise ValueError("duplicate or noncausal history timing")
        breaks=np.flatnonzero(~np.isclose(np.diff(times),cadence_s,atol=1e-7,rtol=0))+1
        gaps+=len(breaks); cuts=np.r_[0,breaks,len(g)]
        for chunk_index,(lo,hi) in enumerate(zip(cuts[:-1],cuts[1:])):
            chunk=g.iloc[int(lo):int(hi)].reset_index(drop=True); chunk_count+=1
            values=chunk.loc[:,features].to_numpy(float)
            if mean is not None and std is not None: values=standardize(values,mean,std)
            identity="|".join(map(str,(*key[:-1],key[-1],chunk_index)))
            hid="a2-history-"+hashlib.sha256(identity.encode()).hexdigest()[:20]
            for target_index in range(seq_len,len(chunk)):
                target=chunk.iloc[target_index]
                rows.append({"physical_recording_id":str(target.physical_recording_id),"history_id":hid,"role":str(target.role),
                             "split":str(target.split),"prn":str(target.prn),"segment":target.segment,"channel":target.channel,
                             "window_start_s":float(target.window_start_s),"window_end_s":float(target.window_end_s),
                             "window_bin_s":float(target.window_bin_s),"target_window_index":target_index,
                             "history":values[target_index-seq_len:target_index].copy(),"target":values[target_index].copy()})
    columns=["physical_recording_id","history_id","role","split","prn","segment","channel","window_start_s","window_end_s",
             "window_bin_s","target_window_index","history","target"]
    out=pd.DataFrame(rows,columns=columns)
    audit={"reset_dimensions":["role","split","physical_recording_id","segment","channel","cadence_gap"],"prn_local":True,
           "seq_len":seq_len,"cadence_s":cadence_s,"input_rows":len(local),"examples":len(out),"chunks":chunk_count,
           "gaps_detected":int(gaps),"bridged":False,"filled_or_interpolated":False}
    return out,audit


def make_fit_fingerprint(frame:pd.DataFrame,mean:np.ndarray,std:np.ndarray,*,extra:Mapping[str,Any]|None=None)->str:
    ordered=frame.sort_values([c for c in ("physical_recording_id","prn","segment","channel","window_end_s") if c in frame],kind="mergesort")
    digest=hashlib.sha256(pd.util.hash_pandas_object(ordered,index=True).values.tobytes()).hexdigest()
    return canonical_hash({"rows":len(frame),"frame_hash":digest,"mean":mean,"std":std,"extra":dict(extra or {})})


def train_b0(prefix:pd.DataFrame,*,config:B0Config=B0Config(),device:str="cpu")->tuple[B0Predictor,dict[str,Any],pd.DataFrame,pd.DataFrame]:
    """Run all 25 frozen epochs and retain the strict-minimum held-PRN state."""
    if config!=B0Config(): raise ValueError("CMTE-A2 requires exact preregistered B0Config")
    if prefix.empty: raise ValueError("prefix training rows must be nonempty")
    if "role" not in prefix or not prefix.role.astype(str).eq("prefix").all():
        raise ValueError("direct B0 training requires role=prefix for every row")
    starts=pd.to_numeric(prefix.window_start_s,errors="coerce").to_numpy(float)
    ends=pd.to_numeric(prefix.window_end_s,errors="coerce").to_numpy(float)
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or np.any(starts<0) or np.any(ends>240):
        raise ValueError("prefix training rows must be fully contained in [0,240]")
    deterministic_controls(config.seed)
    gradient,held=select_prn_holdout(prefix)
    mean,std=fit_standardizer(gradient.loc[:,FEATURE_COLUMNS].to_numpy(float))
    train_examples,train_audit=build_history_examples(gradient,mean=mean,std=std)
    val_examples,val_audit=build_history_examples(held,mean=mean,std=std)
    if train_examples.empty or val_examples.empty: raise ValueError("training and held-PRN examples must be nonempty")
    generator=torch.Generator().manual_seed(config.seed)
    train_loader=DataLoader(_Examples(train_examples),batch_size=256,shuffle=True,generator=generator,num_workers=0)
    val_loader=DataLoader(_Examples(val_examples),batch_size=256,shuffle=False,num_workers=0)
    dev=torch.device(device); model=B0Predictor(config).to(dev)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4,betas=(.9,.999),eps=1e-8,amsgrad=False)
    best=float("inf"); best_epoch=0; best_state=None; history=[]
    for epoch in range(1,26):
        model.train(); train_sum=0.; train_n=0
        for x,y in train_loader:
            x=x.to(dev); y=y.to(dev); optimizer.zero_grad(set_to_none=True); pred=model(x)
            loss=torch.mean((pred-y)**2); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            train_sum+=float(loss.detach())*len(x); train_n+=len(x)
        model.eval(); val_sum=0.; val_n=0
        with torch.no_grad():
            for x,y in val_loader:
                x=x.to(dev); y=y.to(dev); loss=torch.mean((model(x)-y)**2)
                val_sum+=float(loss)*len(x); val_n+=len(x)
        val_loss=val_sum/val_n
        history.append({"epoch":epoch,"train_mse":train_sum/train_n,"validation_mse":val_loss})
        if val_loss<best:
            best=val_loss; best_epoch=epoch; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    assert best_state is not None; model.load_state_dict(best_state); model.to(dev).eval()
    audit={"config":asdict(config),"epochs_run":25,"best_epoch":best_epoch,"best_validation_mse":best,"history":history,
           "gradient_prns":sorted(gradient.prn.astype(str).unique()),"held_prns":sorted(held.prn.astype(str).unique()),
           "scaler_mean":mean.tolist(),"scaler_std":std.tolist(),"train_history_audit":train_audit,"validation_history_audit":val_audit,
           "fit_fingerprint":make_fit_fingerprint(gradient,mean,std,extra={"config":asdict(config)}),"all_rows_before_240":True,
           "all_25_epochs":True,"strict_minimum":True,"scheduler":None,"early_stopping":False}
    return model,audit,train_examples,val_examples


def predict_residuals(frame:pd.DataFrame,model:B0Predictor,mean:np.ndarray,std:np.ndarray,*,role:str,device:str="cpu")->pd.DataFrame:
    local=frame.copy(); local["role"]=role
    examples,audit=build_history_examples(local,mean=mean,std=std)
    if examples.empty:
        out=pd.DataFrame(columns=A2_PRN_COLUMNS[:-3]); out.attrs["history_audit"]=audit; return out
    x=torch.from_numpy(np.stack(examples.history).astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad(): prediction=model(x).cpu().numpy()
    residual=np.stack(examples.target)-prediction
    out=examples.drop(columns=["history","target"]).copy()
    for index,column in enumerate(RESIDUAL_COLUMNS): out[column]=residual[:,index]
    out.attrs["history_audit"]=audit
    return out


@dataclass
class FitState:
    mean:np.ndarray
    covariance:np.ndarray
    qcal:np.ndarray=field(default_factory=lambda:np.empty(0,float))
    epsilon:float=EPSILON
    shrinkage:float=0.
    metadata:dict[str,Any]=field(default_factory=dict)
    @property
    def state_hash(self)->str:
        return canonical_hash({"mean":self.mean,"covariance":self.covariance,"qcal":self.qcal,
                               "epsilon":self.epsilon,"shrinkage":self.shrinkage,"metadata":self.metadata})


def _residual_array(frame:pd.DataFrame)->np.ndarray:
    missing=sorted(set(RESIDUAL_COLUMNS)-set(frame))
    if missing: raise ValueError(f"residual input missing {missing}")
    x=frame.loc[:,RESIDUAL_COLUMNS].to_numpy(float)
    if not len(x) or not np.isfinite(x).all(): raise ValueError("finite nonempty residual input required")
    return x


def fit_distribution(frame:pd.DataFrame,*,epsilon:float=EPSILON)->FitState:
    x=_residual_array(frame); n=len(x); mean=x.mean(axis=0); d=x-mean
    empirical=d.T@d/max(1,n-1); shrinkage=min(1.,10/max(10,n)); diagonal=np.diag(np.maximum(np.diag(empirical),epsilon))
    covariance=(1-shrinkage)*empirical+shrinkage*diagonal+epsilon*np.eye(9)
    return FitState(mean,covariance,np.empty(0),epsilon,shrinkage,{"fit_n":n,"source":"cleanStatic_prefix_gradient_PRNs_only",
                    "residual":"signed_standardized_x_minus_xhat","tap_order":list(TAP_ORDER)})


def nonconformity(residuals:np.ndarray,state:FitState)->np.ndarray:
    d=np.asarray(residuals,float)-state.mean
    return np.maximum(0.,np.einsum("ni,ij,nj->n",d,np.linalg.inv(state.covariance),d))


def conformal_pvalues(calibration:Sequence[float],query:Sequence[float])->np.ndarray:
    cal=np.sort(np.asarray(calibration,float)); q=np.asarray(query,float)
    if not len(cal) or not np.isfinite(cal).all() or not np.isfinite(q).all(): raise ValueError("finite nonempty Qcal required")
    return (1+len(cal)-np.searchsorted(cal,q,side="left"))/(len(cal)+1.)


def score_residuals(frame:pd.DataFrame,state:FitState)->pd.DataFrame:
    if not len(state.qcal): raise ValueError("frozen Qcal required")
    out=frame.copy(); residual=_residual_array(out); out["q"]=nonconformity(residual,state)
    out["p"]=conformal_pvalues(state.qcal,out.q); out["rmse"]=np.sqrt(np.mean(residual**2,axis=1))
    forbidden=[c for c in out if any(t in c.lower() for t in _FORBIDDEN_SCHEMA_TOKENS)]
    if forbidden: raise ValueError(f"forbidden A2 columns: {forbidden}")
    return out


def aggregate_epochs(scored:pd.DataFrame)->pd.DataFrame:
    identity="physical_recording_id" if "physical_recording_id" in scored else "recording_id"
    key="window_end_s"
    if scored.duplicated([identity,key,"prn"]).any(): raise ValueError("duplicate PRN in physical recording epoch")
    rows=[]
    for (recording,t),g in scored.groupby([identity,key],sort=True):
        p=g.p.to_numpy(float); q=g.q.to_numpy(float); neg=-np.log(np.maximum(p,np.finfo(float).tiny))
        rows.append({"physical_recording_id":str(recording),"window_end_s":float(t),"tracked_prn_count":len(g),
                     "min_p":float(np.min(p)),"median_p":float(np.median(p)),"mean_q":float(np.mean(q)),"max_q":float(np.max(q)),
                     "mean_neg_log_p":float(np.mean(neg)),"median_neg_log_p":float(np.median(neg)),"score_A2":float(np.mean(neg))})
    return pd.DataFrame(rows,columns=A2_EPOCH_COLUMNS).sort_values(["physical_recording_id","window_end_s"],kind="mergesort").reset_index(drop=True)


def higher_quantile(values:Sequence[float],probability:float)->float:
    x=np.sort(np.asarray(values,float))
    if not len(x) or not 0<=probability<=1 or not np.isfinite(x).all(): raise ValueError("finite values and probability in [0,1] required")
    return float(x[int(math.ceil(probability*(len(x)-1)))])


def empirical_target1(values:Sequence[float])->float:
    x=np.sort(np.asarray(values,float));
    if not len(x): raise ValueError("threshold scores must be nonempty")
    candidates=[]
    for value in np.unique(x):
        if np.mean(x>value)<=.01+1e-15: candidates.append(float(value))
    if not candidates: return float(x[-1])
    minimum=min(candidates)
    return max(v for v in candidates if v==minimum)


def threshold_operating_points(scores:Sequence[float])->dict[str,float]:
    return {"q995":higher_quantile(scores,.995),"q99":higher_quantile(scores,.99),"target1":empirical_target1(scores)}


def alarm_flags(scores:Sequence[float],threshold:float)->np.ndarray:
    return np.asarray(scores,float)>float(threshold)


def _binomial_tail(k:int,n:int,p:float)->float:
    if k<=0:return 1.
    return float(sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k,n+1)))


@dataclass
class B0ExactState:
    previous_by_recording:dict[str,float]=field(default_factory=dict)


def b0_exact_scores(epochs:pd.DataFrame,node_thresholds:Mapping[str,float],state:B0ExactState|None=None)->pd.DataFrame:
    if set(node_thresholds)!={"q50","q70","q80"}: raise ValueError("exact q50/q70/q80 node thresholds required")
    state=state or B0ExactState(); rows=[]; nominal={"q50":.5,"q70":.3,"q80":.2}
    ordered=epochs.sort_values(["physical_recording_id","window_end_s"],kind="mergesort")
    for row in ordered.itertuples(index=False):
        values=np.asarray(row.rmse_values,float); n=len(values)
        if n<1: raise ValueError("tracked PRN count must be positive")
        result={"physical_recording_id":str(row.physical_recording_id),"window_end_s":float(row.window_end_s)}; surprises=[]
        for name,p in nominal.items():
            k=int(np.sum(values>node_thresholds[name])); tail=_binomial_tail(k,n,p)
            result.update({f"k_{name}":k,f"n_{name}":n,f"tail_{name}":tail}); surprises.append(-math.log(max(tail,1e-300)))
        raw=max(surprises); previous=state.previous_by_recording.get(result["physical_recording_id"],0.)
        score=.75*previous+.25*raw; state.previous_by_recording[result["physical_recording_id"]]=score
        result.update({"raw":raw,"score":score}); rows.append(result)
    return pd.DataFrame(rows)


def b0_enhanced_scores(epochs:pd.DataFrame,node_thresholds:Mapping[str,float],rates:Mapping[str,float])->pd.DataFrame:
    rows=[]
    for row in epochs.sort_values(["physical_recording_id","window_end_s"],kind="mergesort").itertuples(index=False):
        values=np.asarray(row.rmse_values,float); surprises=[]
        for name in ("q50","q70","q80"):
            k=int(np.sum(values>node_thresholds[name])); surprises.append(-math.log(max(_binomial_tail(k,len(values),float(rates[name])),1e-300)))
        rows.append({"physical_recording_id":str(row.physical_recording_id),"window_end_s":float(row.window_end_s),"raw":max(surprises)})
    out=pd.DataFrame(rows); out["score"]=out.groupby("physical_recording_id",sort=False).raw.transform(lambda x:x.ewm(alpha=.75,adjust=False).mean())
    return out


def baseline_epoch_inputs(scored:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for (recording,t),g in scored.groupby(["physical_recording_id","window_end_s"],sort=True):
        rows.append({"physical_recording_id":recording,"window_end_s":t,"rmse_values":g.rmse.to_numpy(float),"A0":float(g.rmse.max())})
    return pd.DataFrame(rows)


def calibrate_comparators(threshold_prn:pd.DataFrame)->dict[str,Any]:
    """Calibrate every comparator component from the short normal threshold role only."""
    required={"physical_recording_id","window_start_s","window_end_s","prn","rmse"}
    missing=sorted(required-set(threshold_prn))
    if missing: raise ValueError(f"threshold comparator calibration missing {missing}")
    if threshold_prn.empty: raise ValueError("threshold comparator calibration must be nonempty")
    start=threshold_prn.window_start_s.to_numpy(float); end=threshold_prn.window_end_s.to_numpy(float)
    if not np.isfinite(start).all() or not np.isfinite(end).all() or np.any(start<300) or np.any(end>330):
        raise ValueError("comparator calibration rows must be fully contained in threshold role [300,330)")
    if "role" in threshold_prn and not threshold_prn.role.astype(str).eq("threshold").all():
        raise ValueError("comparator calibration rejects non-threshold roles")
    rmse=threshold_prn.rmse.to_numpy(float)
    if not np.isfinite(rmse).all(): raise ValueError("finite threshold-role RMSE required")
    node={name:higher_quantile(rmse,q) for name,q in (("q50",.5),("q70",.7),("q80",.8))}
    rates={name:float(np.mean(rmse>value)) for name,value in node.items()}
    inp=baseline_epoch_inputs(threshold_prn)
    exact=b0_exact_scores(inp,node); enhanced=b0_enhanced_scores(inp,node,rates)
    final={"A0":threshold_operating_points(inp.A0),"B0-Exact":threshold_operating_points(exact.score),
           "B0-Enhanced":threshold_operating_points(enhanced.score)}
    return {"source_role":"threshold","source_interval_s":[300.,330.],"source_rows":int(len(threshold_prn)),
            "source_epochs":int(len(inp)),"source_sha256":role_frame_audit(threshold_prn,"threshold")["content_sha256"],
            "node_thresholds":node,"enhanced_empirical_rates":rates,"final_thresholds":final,
            "node_quantile_method":"higher","alarm_comparison":"strict_greater","attack_or_test_fit":False,
            "limitation":"Short 30-second threshold-role self-calibration jointly estimates comparator node and final thresholds."}


def phase_masks(frame:pd.DataFrame,onset_s:float)->dict[str,np.ndarray]:
    start=frame.window_start_s.to_numpy(float); end=frame.window_end_s.to_numpy(float); onset=float(onset_s)
    contained=lambda a,b:(start>=a)&(end<=b)
    return {"stable_pre":contained(30.,onset-20.),"transition":contained(onset-20.,onset),"post":start>=onset,
            "ramp":contained(onset,onset+20),"takeover":contained(onset+20,onset+40),"persistent":start>=onset+40}


def _auc(labels:np.ndarray,scores:np.ndarray)->tuple[float,float]:
    labels=np.asarray(labels,int); scores=np.asarray(scores,float)
    if len(np.unique(labels))<2:return float("nan"),float("nan")
    from sklearn.metrics import average_precision_score,roc_auc_score
    return float(roc_auc_score(labels,scores)),float(average_precision_score(labels,scores))


def _summary(values:np.ndarray)->dict[str,float|None]:
    if not len(values):return {"median":None,"q90":None,"q99":None}
    return {"median":float(np.median(values)),"q90":float(np.quantile(values,.9)),"q99":float(np.quantile(values,.99))}


def epoch_metrics(frame:pd.DataFrame,score_column:str,threshold:float,*,onset_s:float,clean_fpr:float|None=None,cadence_s:float=.5)->dict[str,Any]:
    local=frame.sort_values(["physical_recording_id","window_end_s"],kind="mergesort").reset_index(drop=True)
    score=local[score_column].to_numpy(float); alarm=alarm_flags(score,threshold); masks=phase_masks(local,onset_s)
    keep=masks["stable_pre"]|masks["post"]; labels=masks["post"][keep].astype(int); roc,pr=_auc(labels,score[keep])
    rising=0
    # Rising edges reset at every physical recording and cadence discontinuity.
    for _,index in local.groupby("physical_recording_id",sort=False).groups.items():
        idx=np.asarray(index); idx=idx[masks["stable_pre"][idx]]
        if not len(idx): continue
        times=local.window_end_s.iloc[idx].to_numpy(float)
        breaks=np.r_[True,~np.isclose(np.diff(times),cadence_s,atol=1e-7,rtol=0)]
        a=alarm[idx]; rising+=int(np.sum(a&(breaks|~np.r_[False,a[:-1]])))
    stable_n=int(masks["stable_pre"].sum()); duration_min=max(stable_n*cadence_s/60,1e-12)
    post_hits=np.flatnonzero(alarm&masks["post"]); first=None if not len(post_hits) else float(local.window_end_s.iloc[post_hits[0]]-onset_s)
    persistent3=None
    for _,index in local.groupby("physical_recording_id",sort=False).groups.items():
        idx=np.asarray(index); eligible=idx[masks["post"][idx]]
        for i in range(max(0,len(eligible)-2)):
            triple=eligible[i:i+3]
            if alarm[triple].all() and np.allclose(np.diff(local.window_end_s.iloc[triple]),cadence_s,atol=1e-7):
                persistent3=float(local.window_end_s.iloc[triple[0]]-onset_s); break
        if persistent3 is not None:break
    stable_summary=_summary(score[masks["stable_pre"]]); post_summary=_summary(score[masks["post"]])
    return {"threshold":float(threshold),"roc_auc":roc,"pr_auc":pr,"independent_clean_fpr":clean_fpr,
            "stable_pre_fpr":float(alarm[masks["stable_pre"]].mean()) if stable_n else None,
            "pre_onset_alarm":bool(alarm[local.window_end_s.to_numpy(float)<=float(onset_s)].any()),
            "rising_edge_false_alarm_events":int(rising),
            "rising_edge_false_alarm_events_per_min":rising/duration_min,"alarm_occupancy":float(alarm.mean()),
            "post_detection_rate":float(alarm[masks["post"]].mean()) if masks["post"].any() else None,
            "persistent_detection_rate":float(alarm[masks["persistent"]].mean()) if masks["persistent"].any() else None,
            "first_alarm_delay_s":first,"first_alarm_censored":first is None,"persistent_3_epoch_delay_s":persistent3,
            "persistent_3_epoch_censored":persistent3 is None,"stable_median":stable_summary["median"],"stable_q90":stable_summary["q90"],
            "stable_q99":stable_summary["q99"],"post_median":post_summary["median"],"post_q90":post_summary["q90"],
            "post_q99":post_summary["q99"],"tracked_prn_count":int(local.tracked_prn_count.sum()),
            "tracked_prn_count_median":float(local.tracked_prn_count.median()),
            "ramp_detection_rate":float(alarm[masks["ramp"]].mean()) if masks["ramp"].any() else None,
            "takeover_detection_rate":float(alarm[masks["takeover"]].mean()) if masks["takeover"].any() else None}


def _phase_selector(frame:pd.DataFrame,phase:str)->np.ndarray:
    if phase in frame and pd.api.types.is_bool_dtype(frame[phase]): return frame[phase].to_numpy(bool)
    if "phase" not in frame: raise ValueError(f"bootstrap input missing {phase} submask")
    labels=frame.phase.astype(str).to_numpy()
    # Categorical legacy input cannot represent the overlapping post/persistent
    # masks; callers requiring all established epochs provide explicit booleans.
    return labels==phase


def _complete_blocks(frame:pd.DataFrame,phase:str,block_epochs:int=20,cadence_s:float=.5)->list[np.ndarray]:
    mask=_phase_selector(frame,phase); subset=frame.loc[mask].sort_values(["physical_recording_id","window_end_s"],kind="mergesort")
    if subset.empty:return []
    candidates=("scenario","physical_recording_id","producer_chain_id","cadence_chain_id","history_chain_id","segment","channel")
    chains=[c for c in candidates if c in subset]
    if "physical_recording_id" not in chains: raise ValueError("bootstrap requires physical_recording_id")
    blocks=[]
    for _,g in subset.groupby(chains,sort=True,dropna=False):
        times=g.window_end_s.to_numpy(float); breaks=np.flatnonzero(~np.isclose(np.diff(times),cadence_s,atol=1e-7,rtol=0))+1
        cuts=np.r_[0,breaks,len(g)]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            idx=g.index.to_numpy()[lo:hi]
            blocks.extend([idx[i:i+block_epochs] for i in range(0,(len(idx)//block_epochs)*block_epochs,block_epochs)])
    return blocks


def bootstrap_metrics(frame:pd.DataFrame,*,reps:int=2000,seed:int=20260802,block_epochs:int=20,cadence_s:float=.5)->dict[str,dict[str,Any]]:
    if block_epochs<20 or not math.isclose(cadence_s,.5,abs_tol=1e-12) or block_epochs*cadence_s<10-1e-12:
        raise ValueError("bootstrap requires at least 20 epochs at the fixed 0.5-second cadence")
    requirements={"roc_auc":("stable_pre","post"),"stable_pre_fpr":("stable_pre",),
                  "persistent_detection_rate":("persistent",),"post_detection_rate":("post",)}
    masks={phase:_phase_selector(frame,phase) for phases in requirements.values() for phase in phases}
    pools={phase:_complete_blocks(frame,phase,block_epochs,cadence_s) for phase in masks}
    rng=np.random.default_rng(seed); result={}
    common={"method":"moving_block","iid_fallback":False,"block_epochs":block_epochs,"cadence_s":cadence_s,
            "phase_anchor":"fixed_chunk_start","resampled_original_block_count":True,"boundary_or_gap_crossing":False,
            "point_estimate_uses_all_eligible_epochs":True,
            "block_boundary_dimensions":["scenario","phase","physical_recording_id","producer_chain_id","cadence_chain_id","cadence_gap"]}
    for statistic,phases in requirements.items():
        metadata={**common,"stratum_epoch_counts":{p:int(masks[p].sum()) for p in phases},
                  "stratum_complete_block_counts":{p:int(len(pools[p])) for p in phases}}
        insufficient=[phase for phase in phases if len(pools[phase])<2]
        if insufficient:
            result[statistic]={"low":None,"high":None,"reason":"fewer_than_2_complete_blocks:"+",".join(insufficient),**metadata}; continue
        values=[]
        for _ in range(reps):
            sampled={phase:np.concatenate([pools[phase][i] for i in rng.integers(0,len(pools[phase]),len(pools[phase]))]) for phase in phases}
            if statistic=="roc_auc":
                neg=frame.loc[sampled["stable_pre"],"score"].to_numpy(); pos=frame.loc[sampled["post"],"score"].to_numpy()
                value=_auc(np.r_[np.zeros(len(neg)),np.ones(len(pos))],np.r_[neg,pos])[0]
            else:value=float(frame.loc[sampled[phases[0]],"alarm"].mean())
            values.append(value)
        result[statistic]={"low":float(np.quantile(values,.025)),"high":float(np.quantile(values,.975)),"reason":None,
                           "reps":reps,"seed":seed,"limitation":"conditional temporal CI; one recording" if frame.physical_recording_id.nunique()==1 else None,**metadata}
    return result


def save_fit_state(state:FitState,path:str|Path)->None:
    doc={"schema":"gnss-doppler-lab.cmte-a2-state.v1","mean":state.mean.tolist(),"covariance":state.covariance.tolist(),
         "qcal":state.qcal.tolist(),"epsilon":state.epsilon,"shrinkage":state.shrinkage,"metadata":state.metadata}
    doc["state_hash"]=canonical_hash(doc); Path(path).write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")


def load_fit_state(path:str|Path)->FitState:
    doc=json.loads(Path(path).read_text()); expected=doc.pop("state_hash")
    if canonical_hash(doc)!=expected: raise ValueError("A2 state hash mismatch")
    return FitState(np.asarray(doc["mean"]),np.asarray(doc["covariance"]),np.asarray(doc["qcal"]),doc["epsilon"],doc["shrinkage"],doc["metadata"])


def git_output(repo:Path,*args:str)->str:
    return subprocess.check_output(["git",*args],cwd=repo,text=True,stderr=subprocess.STDOUT).strip()


def create_freeze_manifest(repo:str|Path,state_dir:str|Path,paths:Iterable[str|Path],path:str|Path)->dict[str,Any]:
    root=Path(repo).resolve(); state_root=Path(state_dir).resolve(); files={}
    for raw in paths:
        candidate=Path(raw).resolve()
        try: key="STATE/"+str(candidate.relative_to(state_root))
        except ValueError:
            try: key="REPO/"+str(candidate.relative_to(root))
            except ValueError: raise ValueError(f"frozen file must be under repository or state directory: {candidate}")
        files[key]=file_sha256(candidate)
    doc={"schema":"gnss-doppler-lab.cmte-a2-freeze.v1","prereg_commit":PREREG_COMMIT,"source_commit":git_output(root,"rev-parse","HEAD"),
         "state_dir":".","files":files}
    Path(path).write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n"); return doc


def verify_confirmatory_freeze(manifest_path:str|Path,*,repo:str|Path)->dict[str,Any]:
    root=Path(repo).resolve(); doc=json.loads(Path(manifest_path).read_text())
    if doc.get("prereg_commit")!=PREREG_COMMIT: raise ValueError("wrong preregistration commit")
    try: git_output(root,"merge-base","--is-ancestor",PREREG_COMMIT,doc["source_commit"])
    except Exception as exc: raise ValueError("preregistration commit is not an ancestor") from exc
    if git_output(root,"status","--porcelain"): raise ValueError("confirmatory evaluation requires clean git tree")
    if git_output(root,"rev-parse","HEAD")!=doc.get("source_commit"): raise ValueError("execution source commit differs from freeze")
    for relative,digest in doc.get("files",{}).items():
        if relative.startswith("STATE/"): candidate=Path(manifest_path).resolve().parent/relative.removeprefix("STATE/")
        elif relative.startswith("REPO/"): candidate=root/relative.removeprefix("REPO/")
        else: raise ValueError(f"invalid frozen file scope: {relative}")
        if not candidate.is_file() or file_sha256(candidate)!=digest: raise ValueError(f"frozen file checksum mismatch: {relative}")
    return doc


def write_checksums(root:str|Path)->dict[str,str]:
    directory=Path(root); files={str(p.relative_to(directory)):file_sha256(p) for p in sorted(directory.rglob("*")) if p.is_file() and p.name!="checksums.json"}
    (directory/"checksums.json").write_text(json.dumps(files,indent=2,sort_keys=True)+"\n"); return files



def exact_n_diagnostics(frame:pd.DataFrame,*,min_epochs:int=20)->tuple[pd.DataFrame,dict[str,Any]]:
    """Exact-N score/alarm diagnostics; sparse strata remain separate and explicit."""
    required={"tier","scenario","model","phase","tracked_prn_count","score","alarm"}
    missing=sorted(required-set(frame))
    if missing: raise ValueError(f"exact-N diagnostic missing {missing}")
    rows=[]
    for key,g in frame.groupby(["tier","scenario","model","phase","tracked_prn_count"],sort=True,dropna=False):
        tier,scenario,model,phase,n=key; scores=g.score.to_numpy(float); count=len(g); sparse=count<min_epochs
        rows.append({"tier":tier,"scenario":scenario,"model":model,"phase":phase,"N":int(n),"epoch_count":int(count),
                     "score_median":float(np.median(scores)),"score_q90":float(np.quantile(scores,.9)),
                     "score_q99":float(np.quantile(scores,.99)),"alarm_occupancy":float(g.alarm.astype(bool).mean()),
                     "sparse":bool(sparse),"na_reason":f"sparse_exact_N:fewer_than_{min_epochs}_epochs" if sparse else None})
    out=pd.DataFrame(rows)
    pairs=frame[["tracked_prn_count","score"]].dropna()
    corr=None if len(pairs)<2 or pairs.tracked_prn_count.nunique()<2 else float(pairs.corr(method="spearman").iloc[0,1])
    dependence={"diagnostic":"prn_count_dependence","method":"spearman_exact_epoch_N_vs_score","coefficient":corr,
                "reason":"insufficient_N_variation" if corr is None else None,"aggregation_changed":False,"sparse_strata_pooled":False}
    return out,dependence


def matched_fpr_diagnostic(threshold_role_scores:Mapping[str,Sequence[float]],*,primary_model:str="CMTE-A2",
                           primary_threshold:float,percentiles:Sequence[float]|None=None)->dict[str,Any]:
    """Freeze diagnostic comparator cutoffs on a threshold-role percentile grid only."""
    if primary_model not in threshold_role_scores: raise ValueError("primary threshold-role scores required")
    grid=np.asarray(np.linspace(0,1,201) if percentiles is None else percentiles,float)
    if not len(grid) or np.any(~np.isfinite(grid)) or np.any((grid<0)|(grid>1)): raise ValueError("finite percentile grid in [0,1] required")
    primary=np.asarray(threshold_role_scores[primary_model],float); target=float(np.mean(primary>float(primary_threshold)))
    models={}
    for model,raw in threshold_role_scores.items():
        if model==primary_model:continue
        values=np.asarray(raw,float); candidates=np.unique([higher_quantile(values,float(q)) for q in grid])
        occupancy=np.asarray([np.mean(values>t) for t in candidates],float); distance=np.abs(occupancy-target)
        best=np.flatnonzero(np.isclose(distance,distance.min(),atol=1e-15,rtol=0)); index=best[np.argmax(candidates[best])]
        models[model]={"threshold":float(candidates[index]),"threshold_role_strict_exceedance":float(occupancy[index]),
                       "target_occupancy":target,"absolute_difference":float(distance[index]),"tie_break":"higher_threshold"}
    return {"fit_role":"threshold","source_interval_s":[300.,330.],"grid":"threshold_role_percentiles",
            "percentiles":[float(x) for x in grid],"primary_model":primary_model,"primary_threshold":float(primary_threshold),
            "models":models,"attack_or_test_fit":False,"diagnostic_only":True,"replaces_primary":False}


def _is_censored(row:Mapping[str,Any],delay_name:str)->bool:
    explicit=delay_name.replace("delay_s","censored")
    value=row.get(explicit,row.get(delay_name+"_censored",False))
    return bool(value) or row.get(delay_name) is None or pd.isna(row.get(delay_name))


def _delay_worse(a:Mapping[str,Any],b:Mapping[str,Any],name:str)->bool:
    ac,bc=_is_censored(a,name),_is_censored(b,name); av,bv=a.get(name),b.get(name)
    if ac and not bc:return True
    if ac and bc:return False
    if not ac and bc:return False
    return float(av)>float(bv)+1e-12


def success_criteria_audit(metrics:pd.DataFrame,prn_diagnostics:pd.DataFrame)->dict[str,Any]:
    """Evaluate preregistered criteria 1-6 and emit a machine-readable GO/NO-GO."""
    def row(model,scenario):
        g=metrics[(metrics.model==model)&(metrics.scenario.astype(str).str.upper()==scenario)]
        return None if g.empty else g.iloc[0].to_dict()
    a={x:row("CMTE-A2",x) for x in ("DS7","DS8")}; b={x:row("B0-Exact",x) for x in ("DS7","DS8")}
    clean_values=pd.to_numeric(metrics.loc[metrics.model=="CMTE-A2","independent_clean_fpr"],errors="coerce").dropna()
    c1=bool(len(clean_values) and clean_values.max()<=.015+1e-15)
    c2=bool(all(a[x] is not None and a[x].get("stable_pre_fpr") is not None and float(a[x]["stable_pre_fpr"])<=.05+1e-15 for x in a))
    c3=bool(all(a[x] is not None and a[x].get("stable_pre_fpr") is not None and float(a[x]["stable_pre_fpr"])<.20-1e-15 for x in a))
    improvements=[]; evidence={}
    for scenario in a:
        aa,bb=a[scenario],b[scenario]; reasons=[]
        if aa is not None and bb is not None:
            similar=abs(float(aa.get("independent_clean_fpr",math.inf))-float(bb.get("independent_clean_fpr",-math.inf)))<=.005+1e-15
            pre=bool(aa.get("pre_onset_alarm",False))
            if similar:
                for name in ("first_alarm_delay_s","persistent_3_epoch_delay_s"):
                    av,bv=aa.get(name),bb.get(name)
                    if not pre and av is not None and bv is not None and not pd.isna(av) and not pd.isna(bv) and float(bv)-float(av)>=.5-1e-12:reasons.append(name+"_gain")
                for name in ("post_detection_rate","persistent_detection_rate"):
                    av,bv=aa.get(name),bb.get(name)
                    if av is not None and bv is not None and not pd.isna(av) and not pd.isna(bv) and float(av)>float(bv)+1e-12:reasons.append(name+"_gain")
            evidence[scenario]={"similar_clean_fpr":similar,"pre_alarm_delay_invalid":pre,"improvements":reasons}
        if reasons:improvements.append(scenario)
    c4=bool(improvements); other=[x for x in a if x not in improvements]; no_degradation={}
    for scenario in other:
        aa,bb=a[scenario],b[scenario]
        if aa is None or bb is None:no_degradation[scenario]=False;continue
        worse=[_delay_worse(aa,bb,n) for n in ("first_alarm_delay_s","persistent_3_epoch_delay_s")]
        worse += [aa.get(n) is None or pd.isna(aa.get(n)) or (bb.get(n) is not None and not pd.isna(bb.get(n)) and float(aa[n])<float(bb[n])-1e-12) for n in ("post_detection_rate","persistent_detection_rate")]
        no_degradation[scenario]=not all(worse)
    c5=bool(c4 and all(no_degradation.values()))
    scenarios=set(prn_diagnostics.scenario.astype(str).str.upper()) if "scenario" in prn_diagnostics else set(); required={"CLEAN_TEST","DS7","DS8"}
    complete=required.issubset(scenarios)
    if complete and "complete" in prn_diagnostics:
        complete=bool(prn_diagnostics[prn_diagnostics.scenario.astype(str).str.upper().isin(required)].complete.all())
    if complete and "sparse" in prn_diagnostics:
        sparse=prn_diagnostics[prn_diagnostics["sparse"].astype(bool)]
        complete=bool(sparse.empty or ("na_reason" in sparse and sparse.na_reason.notna().all()))
    c6=bool(complete); values=[c1,c2,c3,c4,c5,c6]
    names=("independent_clean_fpr","per_scenario_stable_pre","catastrophic_pre_alarm_guard","improvement_over_b0_exact",
           "no_catastrophic_degradation_on_other_scenario","prn_count_diagnostic_complete")
    detail=({"clean_fpr_values":clean_values.tolist()},{"scenario_values":{x:None if a[x] is None else a[x].get("stable_pre_fpr") for x in a}},
            {"guard":"stable_pre_fpr_strictly_below_0.20"},{"scenario_evidence":evidence,"improved_scenarios":improvements},
            {"other_scenario_not_simultaneously_worse":no_degradation},{"required":sorted(required),"observed":sorted(scenarios)})
    criteria=[{"id":i+1,"name":names[i],"passed":bool(values[i]),"required":True,**detail[i]} for i in range(6)]
    passed=all(values)
    return {"schema":"gnss-doppler-lab.cmte-a2-success-audit.v1","criteria":criteria,"all_passed":passed,
            "decision":"GO" if passed else "NO-GO","na_required_fails":True}


def write_runtime_json_evidence(document:Mapping[str,Any],path:str|Path)->None:
    Path(path).write_text(json.dumps(dict(document),indent=2,sort_keys=True,default=_json_default)+chr(10))


def historical_gate_equivalence(prn_scores:pd.DataFrame,node_thresholds:Mapping[str,float],*,alarm_threshold:float,
                                evaluator_path:str|Path|None=None,evidence_path:str|Path|None=None,tolerance:float=1e-12)->dict[str,Any]:
    """Import the actual historical evaluator and record gate-only runtime evidence."""
    path=(Path(__file__).resolve().parents[2]/"scripts/eval_btail_support_gate.py" if evaluator_path is None else Path(evaluator_path)).resolve(strict=True)
    spec=importlib.util.spec_from_file_location("cmte_a2_historical_gate_runtime",path)
    if spec is None or spec.loader is None:raise ValueError("cannot import historical gate evaluator")
    module=importlib.util.module_from_spec(spec); import sys; sys.modules[spec.name]=module; spec.loader.exec_module(module)
    old=module.build_event_scores(prn_scores,node_thresholds,alpha=.75); grouped=[]
    for (run_id,_),g in prn_scores.groupby(["run_id","window_bin_s"],sort=True):
        grouped.append({"physical_recording_id":str(run_id),"window_end_s":float(g.window_start_s.min()+1),"rmse_values":g.prn_node_rmse.to_numpy(float)})
    new=b0_exact_scores(pd.DataFrame(grouped),node_thresholds); errors=[]
    for q in ("q50","q70","q80"):
        errors.append(float(np.max(np.abs(new[f"n_{q}"].to_numpy(float)-old.tracked_prn_count.to_numpy(float)))))
        errors.append(float(np.max(np.abs(new[f"k_{q}"].to_numpy(float)-old[f"k_{q}"].to_numpy(float)))))
        surprise=-np.log(np.maximum(new[f"tail_{q}"].to_numpy(float),1e-300))
        errors.append(float(np.max(np.abs(surprise-old[f"btail_{q}"].to_numpy(float)))))
    errors += [float(np.max(np.abs(new.raw-old.btail_max_507080))),float(np.max(np.abs(new.score-old[module.FINAL_SCORE])))]
    alarms_equal=bool(np.array_equal(new.score.to_numpy()>alarm_threshold,old[module.FINAL_SCORE].to_numpy()>alarm_threshold))
    maximum=max(errors,default=0.); passed=maximum<=tolerance and alarms_equal
    evidence={"schema":"gnss-doppler-lab.cmte-a2-historical-gate-equivalence.v1","actual_evaluator_imported":True,
              "evaluator_path":str(path),"evaluator_sha256":file_sha256(path),"compared":["N","K","tails","raw","retention_ewma","strict_alarm"],
              "tolerance":float(tolerance),"max_absolute_error":maximum,"strict_alarm_equal":alarms_equal,
              "strict_alarm_threshold":float(alarm_threshold),"passed":bool(passed),"new_checkpoint_score_equivalence_claim":False}
    if evidence_path is not None:write_runtime_json_evidence(evidence,evidence_path)
    return evidence
