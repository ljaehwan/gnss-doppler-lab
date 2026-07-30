"""Independent Peak--Floor data contract used by PF-DIC.

No script modules are imported: feature order, alignment, scaling, splitting and
causal support semantics are defined here so frozen artifacts are reproducible.
"""
from __future__ import annotations
import hashlib, math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import numpy as np
import pandas as pd

MAX_PRNS=32; CADENCE_S=0.5
TAP_FEATURES=["tap_E4_rel_prompt_mean","tap_E3_rel_prompt_mean","tap_E2_rel_prompt_mean","tap_E_rel_prompt_mean","tap_P_rel_prompt_mean","tap_L_rel_prompt_mean","tap_L2_rel_prompt_mean","tap_L3_rel_prompt_mean","tap_L4_rel_prompt_mean"]
MORPH_FEATURES=TAP_FEATURES+["left_right_imbalance_mean","left_right_imbalance_std","peak_index_mean","peak_index_std","peak_width_mean","peak_width_std","peak_sharpness_mean","peak_sharpness_std","prompt_mag_cv","dmcpd_prompt_dominance_mean","dmcpd_prompt_to_max_side_mean","dmcpd_max_side_to_prompt_mean","dmcpd_second_side_to_prompt_mean","dmcpd_centroid_shift_mean","dmcpd_centroid_shift_std","dmcpd_width_variance_mean","dmcpd_left_right_energy_abs_mean","dmcpd_curvature_e1l1_mean","dmcpd_pair1_signed_asym_mean","dmcpd_pair1_abs_asym_mean","dmcpd_pair2_signed_asym_mean","dmcpd_pair2_abs_asym_mean","dmcpd_pair3_signed_asym_mean","dmcpd_pair3_abs_asym_mean","dmcpd_pair4_signed_asym_mean","dmcpd_pair4_abs_asym_mean"]
FLOOR_FEATURES=["i_mean","q_mean","i_std","q_std","iq_corr","power_mean","power_std","phase_inc_mean","phase_inc_std","phase_coh","psd_entropy","psd_flatness","amp_mean","amp_std","amp_skew","amp_kurt","damp_mean","damp_std","damp_skew","damp_kurt"]+[f"ac_{i:02d}" for i in range(21)]+[f"psd_band_{i:02d}" for i in range(16)]
DEFAULT_MORPH_FEATURES=MORPH_FEATURES; DEFAULT_FLOOR_FEATURES=FLOOR_FEATURES
DEFAULT_SPLIT_RULES={"train":(None,240.),"validation":(250.,330.),"calibration":(340.,410.),"held_clean":(420.,None)}

@dataclass
class AlignedData:
    times:np.ndarray; available_times:np.ndarray; support_start_times:np.ndarray; morph:np.ndarray; floor:np.ndarray; prn_mask:np.ndarray; morph_features:list[str]; floor_features:list[str]
@dataclass
class CausalPairs:
    context_times:np.ndarray; target_times:np.ndarray; available_times:np.ndarray; support_start_times:np.ndarray
    context_morph:np.ndarray; context_floor:np.ndarray; context_mask:np.ndarray
    target_morph:np.ndarray; target_floor:np.ndarray; target_mask:np.ndarray
    def __len__(self): return len(self.target_times)

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def validate_normal_only_inputs(morph:pd.DataFrame,floor:pd.DataFrame)->dict:
    required_morph={"label","run_id","source_fingerprint","tap_count"}
    missing=sorted(required_morph-set(morph.columns))
    if missing: raise ValueError(f"cleanStatic source identity/tap_count columns missing: {missing}")
    if "scenario" not in floor: raise ValueError("cleanStatic contract requires floor scenario")
    for frame,column in ((morph,"label"),(morph,"run_id"),(morph,"source_fingerprint"),(morph,"tap_count"),(floor,"scenario")):
        if frame[column].isna().any(): raise ValueError(f"required field has missing/null/NaN values: {column}")
    labels=sorted(morph.label.astype(str).str.lower().unique()); scenarios=sorted(floor.scenario.astype(str).str.lower().unique())
    allowed={"clean","cleanstatic","oakbatcleanstatic","oakbatcleanstatic9tap","texbatcleanstatic","texbatcleanstatic9tap"}
    def clean_static(value):
        token="".join(ch for ch in value.lower() if ch.isalnum())
        return token in allowed
    if not labels or any(not clean_static(x) for x in labels): raise ValueError(f"morphology is not cleanStatic/clean static: {labels}")
    if not scenarios or any(not clean_static(x) for x in scenarios): raise ValueError(f"floor is not cleanStatic/clean static: {scenarios}")
    runs=sorted(morph.run_id.dropna().astype(str).unique());fps=sorted(morph.source_fingerprint.dropna().astype(str).unique())
    if len(runs)!=1 or len(fps)!=1 or not runs[0].strip() or not fps[0].strip(): raise ValueError("cleanStatic training requires exactly one non-empty source identity")
    if not (pd.to_numeric(morph.tap_count,errors="raise")==9).all(): raise ValueError("tap_count violates nine-tap contract")
    return {"labels":labels,"scenarios":scenarios,"run_ids":runs,"source_fingerprints":fps}

def _numeric(df,cols,kind):
    missing=[x for x in cols if x not in df]
    if missing: raise ValueError(f"{kind} missing features: {missing}")
    x=df[cols].apply(pd.to_numeric,errors="raise").to_numpy(np.float32)
    if not np.isfinite(x).all(): raise ValueError(f"{kind} contains non-finite values")
    return x

def _slot(v):
    s=str(v).strip().upper(); s=s[1:] if s.startswith("G") else s
    try: n=int(s)
    except ValueError as e: raise ValueError(f"unsupported GPS PRN: {v}") from e
    if not 1<=n<=32: raise ValueError(f"GPS PRN outside 1..32: {v}")
    return n-1

def align_modalities(morph_frame,floor_frame,morph_features=None,floor_features=None,tolerance_s=1e-4,validate_clean=False):
    if validate_clean: validate_normal_only_inputs(morph_frame,floor_frame)
    mf=list(morph_features or MORPH_FEATURES); ff=list(floor_features or FLOOR_FEATURES)
    if "window_bin_s" not in morph_frame or "prn" not in morph_frame or "window_start_s" not in floor_frame: raise ValueError("missing alignment columns")
    m=morph_frame.copy(); f=floor_frame.copy(); _numeric(m,mf,"morphology");_numeric(f,ff,"floor")
    time_columns=[(m,"window_bin_s"),(f,"window_start_s")]
    for frame,column in time_columns:
        values=pd.to_numeric(frame[column],errors="raise").to_numpy(float)
        if not np.isfinite(values).all(): raise ValueError(f"time column {column} contains non-finite values")
    for frame,column in ((m,"window_start_s"),(m,"window_end_s"),(f,"window_end_s")):
        if column in frame:
            values=pd.to_numeric(frame[column],errors="raise").to_numpy(float)
            if not np.isfinite(values).all(): raise ValueError(f"time column {column} contains non-finite values")
    m["_t"]=pd.to_numeric(m.window_bin_s);f["_t"]=pd.to_numeric(f.window_start_s)
    if f._t.duplicated().any(): raise ValueError("duplicate floor epoch")
    ft=f._t.to_numpy(float); rows=[]
    for t in sorted(m._t.unique()):
        j=int(np.argmin(abs(ft-t)))
        if abs(ft[j]-t)<=tolerance_s: rows.append((float(t),j))
    if not rows: raise ValueError("no aligned epochs")
    n=len(rows); mt=np.zeros((n,32,len(mf)),np.float32); mask=np.zeros((n,32),bool); floor=np.zeros((n,len(ff)),np.float32); avail=np.zeros(n); starts=np.zeros(n)
    groups={float(t):g for t,g in m.groupby("_t")}
    for i,(t,j) in enumerate(rows):
        g=groups[t]; slots=[_slot(x) for x in g.prn]
        if len(set(slots))!=len(slots): raise ValueError(f"duplicate PRN at {t}")
        for k,v in zip(slots,_numeric(g,mf,"morphology")): mt[i,k]=v;mask[i,k]=True
        floor[i]=_numeric(f.iloc[[j]],ff,"floor")[0]
        ma=float(pd.to_numeric(g.window_end_s).max()) if "window_end_s" in g else t
        fa=float(f.iloc[j].window_end_s) if "window_end_s" in f else t
        ms=float(pd.to_numeric(g.window_start_s).min()) if "window_start_s" in g else t
        fs=float(f.iloc[j].window_start_s) if "window_start_s" in f else t
        avail[i]=max(t,ma,fa);starts[i]=min(t,ms,fs)
    return AlignedData(np.array([x[0] for x in rows]),avail,starts,mt,floor,mask,mf,ff)

def _take(d,k): return AlignedData(d.times[k],d.available_times[k],d.support_start_times[k],d.morph[k],d.floor[k],d.prn_mask[k],d.morph_features,d.floor_features)
def partition_aligned(data,rules:Mapping[str,tuple[float|None,float|None]]):
    canonical=["train","validation","calibration","held_clean"]
    if set(rules)==set(canonical):
        previous=None
        for name in canonical:
            a,b=rules[name]; lo=-math.inf if a is None else float(a); hi=math.inf if b is None else float(b)
            if lo>hi or (previous is not None and lo<=previous):
                raise ValueError("split overlap or order violation")
            previous=hi
    ranges=[]; out={}; used=np.zeros(len(data.times),bool)
    for name,(a,b) in rules.items():
        lo=-math.inf if a is None else float(a); hi=math.inf if b is None else float(b)
        if lo>hi or any(not (hi<x or lo>y) for x,y in ranges): raise ValueError("split overlap or order violation")
        ranges.append((lo,hi)); k=(data.times>=lo)&(data.times<=hi)
        if not k.any(): raise ValueError(f"partition {name} is empty")
        if np.any(k&used):
            raise ValueError("split overlap")
        used |= k
        out[name] = _take(data,k)
    return out

def fit_robust_scalers(data):
    def fit(x):
        med=np.median(x,0).astype(np.float32); mad=(1.4826*np.median(abs(x-med),0)).astype(np.float32); std=np.std(x,0).astype(np.float32)
        scale=np.where(mad>1e-6,mad,np.where(std>1e-6,std,1.)).astype(np.float32); return med,scale
    mm,ms=fit(data.morph[data.prn_mask]);fm,fs=fit(data.floor)
    return {"morph_median":mm,"morph_scale":ms,"floor_median":fm,"floor_scale":fs,"fit_scope":"train_only"}
def apply_scalers(data,s,clip=12.):
    m=np.clip((data.morph-np.asarray(s["morph_median"]))/np.asarray(s["morph_scale"]),-clip,clip).astype(np.float32);m[~data.prn_mask]=0
    f=np.clip((data.floor-np.asarray(s["floor_median"]))/np.asarray(s["floor_scale"]),-clip,clip).astype(np.float32)
    return AlignedData(data.times.copy(),data.available_times.copy(),data.support_start_times.copy(),m,f,data.prn_mask.copy(),data.morph_features,data.floor_features)
def make_causal_pairs(data,context_len,horizon=1,stride=1):
    if min(context_len,horizon,stride)<1: raise ValueError("positive pair parameters required")
    idx=[]
    for end in range(context_len-1,len(data.times)-horizon,stride):
        start=end-context_len+1; target=end+horizon; support=np.arange(start,target+1)
        if np.allclose(np.diff(data.times[support]),CADENCE_S,atol=1e-6,rtol=0): idx.append((np.arange(start,end+1),target,support))
    if not idx: raise ValueError("no contiguous causal pairs")
    c=np.stack([x[0] for x in idx]);t=np.array([x[1] for x in idx]);av=np.array([data.available_times[x[2]].max() for x in idx]);st=np.array([data.support_start_times[x[2]].min() for x in idx])
    return CausalPairs(data.times[c],data.times[t],av,st,data.morph[c],data.floor[c],data.prn_mask[c],data.morph[t],data.floor[t],data.prn_mask[t])
