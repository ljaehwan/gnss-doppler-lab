"""AMCF-R1 causal temporal masked correlation-field model.

Prompt is a normalization reference/current quality context only.  PRN is used
only for grouping.  All selectors receive observed values and candidate IDs,
never a full hidden row.  Metric helpers are pure and non-mutating.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import copy, hashlib, json, math
from pathlib import Path
from typing import Any
import numpy as np
import torch
from scipy.special import digamma, gammaln
from torch import nn

TAP_NAMES=("E4","E3","E2","E","P","L","L2","L3","L4")
TAP_COORDS=np.arange(-4,5,dtype=np.float32)*.125
PROMPT_INDEX=4
SIDE_INDICES=(0,1,2,3,5,6,7,8)
SEED_SIDES=(3,5)
FIXED_EXTRAS={5:(2,6),7:(2,6,1,7)}
HISTORY_DIM=18
MAD_SCALE=1.4826

@dataclass(frozen=True)
class PromptGate:
 min_prompt_magnitude: float
 eps: float=1e-12
 quantile: float=.005
 fit_rows: int=0
 def __post_init__(self):
  if not np.isfinite(self.min_prompt_magnitude) or self.min_prompt_magnitude<0 or self.eps<=0: raise ValueError("finite nonnegative gate and positive eps required")

@dataclass
class WindowRecord:
 recording_id:str; prn:Any; end_s:float; role:str; source_indices:np.ndarray
 features:np.ndarray; prompt_context:np.ndarray; raw_count:int; valid_count:int; rejected_count:int

def _to_complex(iq):
 x=np.asarray(iq)
 if x.ndim!=3 or x.shape[1:]!=(9,2) or not np.issubdtype(x.dtype,np.number): raise ValueError("complex_iq must be numeric [N,9,2]")
 return x[...,0].astype(np.float64)+1j*x[...,1].astype(np.float64)

def fit_prompt_gate(iq,time_s,quantile=.005,train_end_s=240.):
 z=_to_complex(iq); t=np.asarray(time_s,float); use=(t>=0)&(t<train_end_s); p=np.abs(z[use,4])
 if not len(p) or not np.isfinite(p).all(): raise ValueError("clean train Prompt rows required")
 return PromptGate(float(np.quantile(p,quantile,method="higher")),1e-12,quantile,int(len(p)))

def normalize_prompt(iq,gate:PromptGate):
 z=_to_complex(iq); p=z[:,4]; mag=np.abs(p); valid=np.isfinite(mag)&(mag>=gate.min_prompt_magnitude)&(mag>0)
 out=np.full(z.shape,np.nan+1j*np.nan,np.complex128)
 # eps is declared but only guards a zero denominator; valid rows retain exact
 # global phase/nav-sign invariance without low-P blow-ups.
 den=mag[valid]**2+gate.eps
 out[valid]=z[valid]*np.conjugate(p[valid,None])/den[:,None]
 return out,valid

def assign_clean_role(t):
 t=float(t)
 if 0<=t<240:return "train"
 if 250<=t<330:return "validation"
 if 340<=t<410:return "calibration"
 if t>=420:return "clean_test"
 return None

def complex_summary(z,valid_fraction=1.):
 z=np.asarray(z,np.complex128)
 if z.ndim!=2: raise ValueError("rows x taps required")
 medr=np.median(z.real,axis=0); medi=np.median(z.imag,axis=0)
 madr=MAD_SCALE*np.median(np.abs(z.real-medr),axis=0); madi=MAD_SCALE*np.median(np.abs(z.imag-medi),axis=0)
 mag=np.abs(z); magmed=np.median(mag,axis=0)
 unit=np.divide(z,mag,out=np.ones_like(z),where=mag>np.finfo(float).tiny)
 concentration=np.abs(np.mean(unit,axis=0))
 return np.stack([medr,medi,madr,madi,magmed,concentration,np.full(z.shape[1],valid_fraction)],axis=1)

def magnitude_summary(z,valid_fraction=1.):
 c=complex_summary(z,valid_fraction); out=np.zeros_like(c); mag=np.abs(z); med=np.median(mag,axis=0)
 out[:,0]=med; out[:,2]=MAD_SCALE*np.median(np.abs(mag-med),axis=0); out[:,4]=med; out[:,6]=valid_fraction
 return out

def _role_for_rows(recording,t):
 if recording=="cleanStatic" or recording in {"r","test"}: return assign_clean_role(t)
 return "scenario"

def build_causal_windows(iq,time_s,prn,*,recording_id,gate,stride_s=.5,representation="complex",cn0=None):
 zraw=_to_complex(iq); t=np.asarray(time_s,float); p=np.asarray(prn); n=len(t)
 if p.shape!=(n,) or not np.isfinite(t).all() or stride_s<=0: raise ValueError("aligned finite metadata required")
 norm,valid=normalize_prompt(iq,gate); records=[]; used=set(); future=0; crossings=0
 for ident in sorted(np.unique(p),key=str):
  ids=np.flatnonzero(p==ident); ids=ids[np.argsort(t[ids],kind="mergesort")]
  if not len(ids): continue
  ends=np.arange(math.ceil(max(0.,t[ids].min())/stride_s)*stride_s, t[ids].max()+stride_s*.500001,stride_s)
  for end in ends:
   raw=ids[(t[ids]>end-1.-1e-12)&(t[ids]<=end+1e-12)]
   if not len(raw): continue
   endpoint_role=_role_for_rows(recording_id,end)
   roles={_role_for_rows(recording_id,t[j]) for j in raw}
   if endpoint_role is None or roles!={endpoint_role}: crossings+=1; continue
   good=raw[valid[raw]]
   if not len(good): continue
   frac=len(good)/len(raw); summary=complex_summary(norm[good],frac) if representation=="complex" else magnitude_summary(norm[good],frac)
   pmag=np.abs(zraw[good,4]); context=[float(np.median(np.log(pmag+gate.eps))),float(frac)]
   if cn0 is not None: context.append(float(np.nanmedian(np.asarray(cn0)[good])))
   records.append(WindowRecord(str(recording_id),ident,float(end),str(endpoint_role),raw.copy(),summary,np.asarray(context,np.float32),len(raw),len(good),len(raw)-len(good)))
   used.update(map(int,good)); future+=int(np.sum(t[good]>end+1e-9))
 records.sort(key=lambda r:(r.end_s,str(r.prn)))
 accepted=set(np.flatnonzero(valid).tolist()); utilization=len(used)/max(1,len(accepted))
 qa={"raw_rows":n,"valid_raw_rows":int(valid.sum()),"rejected_raw_rows":int((~valid).sum()),"window_count":len(records),"unique_valid_rows_used":len(used),"unique_valid_utilization":utilization,"effectively_98pct_discarded":bool(utilization<.02),"future_rows":future,"split_boundary_crossings":0,"candidate_boundary_windows_rejected":crossings,"unique_used_source_indices":sorted(used)}
 return records,qa

def history_indices(records,index,max_history=12):
 r=records[index]; candidates=[j for j,x in enumerate(records[:index]) if x.recording_id==r.recording_id and x.prn==r.prn and x.role==r.role and x.end_s<r.end_s]
 return candidates[-max_history:]

def history_vector(record):
 f=record.features
 # E/L temporal signal summaries plus current-valid QA, fixed 18-D.
 v=np.r_[f[3,:7],f[5,:7],record.prompt_context[:2],record.valid_count,record.rejected_count]
 return np.asarray(v,np.float32)

def make_history(records,index,max_history=12):
 out=np.zeros((max_history,HISTORY_DIM),np.float32); ids=history_indices(records,index,max_history)
 if ids: out[-len(ids):]=np.stack([history_vector(records[j]) for j in ids])
 return out

class AMCFModel(nn.Module):
 def __init__(self,feature_dim=7,hidden=32,df=4.):
  super().__init__(); self.feature_dim=int(feature_dim); self.hidden=int(hidden); self.df=float(df); self.target_indices=SIDE_INDICES
  self.temporal=nn.GRU(HISTORY_DIM,hidden,batch_first=True)
  self.token=nn.Sequential(nn.Linear(feature_dim+2,hidden),nn.GELU())
  self.attn=nn.MultiheadAttention(hidden,2,batch_first=True)
  self.prompt=nn.Linear(3,hidden); self.target=nn.Linear(2,hidden)
  self.decoder=nn.Sequential(nn.Linear(hidden*3,hidden),nn.GELU(),nn.Linear(hidden,feature_dim*2))
 def forward(self,history,current,observed_mask,target,prompt_context=None):
  if history.ndim!=3 or current.ndim!=3: raise ValueError("batched history/current required")
  b=current.shape[0]; device=current.device; mask=observed_mask.bool().clone(); mask[:,PROMPT_INDEX]=False
  safe=torch.where(mask[...,None],current,torch.zeros_like(current))
  coords=torch.as_tensor(TAP_COORDS,device=device,dtype=current.dtype)[None,:,None].expand(b,-1,-1)
  tok=self.token(torch.cat([safe,coords,mask[...,None].to(current.dtype)],-1)); keymask=~mask
  # avoid all-masked attention NaN using a zero E token only as padding anchor.
  empty=~mask.any(1)
  if empty.any(): keymask[empty,3]=False
  att,_=self.attn(tok,tok,tok,key_padding_mask=keymask,need_weights=False)
  weights=mask.to(current.dtype); pooled=(att*weights[...,None]).sum(1)/weights.sum(1,keepdim=True).clamp_min(1)
  _,state=self.temporal(history); temporal=state[-1]
  if prompt_context is None: pc=torch.zeros((b,3),device=device,dtype=current.dtype)
  else:
   pc=prompt_context
   if pc.shape[1]<3: pc=torch.cat([pc,torch.zeros((b,3-pc.shape[1]),device=device,dtype=current.dtype)],1)
   pc=pc[:,:3]
  ti=target.long(); tc=torch.stack([torch.as_tensor(TAP_COORDS,device=device,dtype=current.dtype)[ti],(ti>4).to(current.dtype)],1)
  q=self.target(tc)+self.prompt(pc); raw=self.decoder(torch.cat([pooled,temporal,q],1)); loc,logs=raw.chunk(2,1)
  return loc,torch.nn.functional.softplus(logs)+1e-4
 def predict_one(self,history,current,observed,target,prompt_context=None):
  if int(target)==4: raise ValueError("Prompt is context only and cannot be a target")
  mask=torch.zeros(9,dtype=torch.bool,device=current.device); mask[list(observed)]=True
  if mask[4]: raise ValueError("Prompt cannot be queried/observed token")
  pc=None if prompt_context is None else torch.as_tensor(prompt_context,dtype=current.dtype,device=current.device)[None]
  return self(history[None],current[None],mask[None],torch.tensor([target],device=current.device),pc)
 def distribution(self,history,prompt_context,observed,target):
  device=next(self.parameters()).device; cur=np.zeros((9,self.feature_dim),np.float32)
  for k,v in observed.items():
   if int(k)==4: raise ValueError("Prompt cannot be observed token")
   cur[int(k)]=np.asarray(v,np.float32)
  with torch.no_grad(): loc,scale=self.predict_one(torch.as_tensor(history,dtype=torch.float32,device=device),torch.as_tensor(cur,device=device),list(observed),int(target),prompt_context)
  return loc[0].cpu().numpy(),scale[0].cpu().numpy()

def student_t_nll(y,location,scale,df=4.):
 if torch.is_tensor(y) or torch.is_tensor(location) or torch.is_tensor(scale):
  y=torch.as_tensor(y); mu=torch.as_tensor(location,device=y.device); s=torch.as_tensor(scale,device=y.device)
  return -torch.lgamma(torch.tensor((df+1)/2,device=y.device))+torch.lgamma(torch.tensor(df/2,device=y.device))+.5*math.log(df*math.pi)+torch.log(s)+(df+1)/2*torch.log1p(((y-mu)/s)**2/df)
 y,mu,s=np.broadcast_arrays(np.asarray(y,float),np.asarray(location,float),np.asarray(scale,float)); return -gammaln((df+1)/2)+gammaln(df/2)+.5*math.log(df*math.pi)+np.log(s)+(df+1)/2*np.log1p(((y-mu)/s)**2/df)

def student_t_entropy(scale,df=4.):
 s=np.asarray(scale,float); return np.log(s)+.5*math.log(df*math.pi)+gammaln(df/2)-gammaln((df+1)/2)+(df+1)/2*(digamma((df+1)/2)-digamma(df/2))

class HiddenAccessGuard(Mapping):
 def __init__(self,full,allowed): self.full=full; self.allowed=set(allowed)
 def __getitem__(self,k):
  if k not in self.allowed: raise RuntimeError("hidden oracle access forbidden")
  return self.full[k]
 def __iter__(self): return iter(sorted(self.allowed))
 def __len__(self): return len(self.allowed)

def expected_information_gain(model,history,prompt_context,observed,candidates,mc_samples=8,seed=0,return_all=False):
 candidates=[int(x) for x in candidates]
 if 4 in candidates: raise ValueError("Prompt cannot be candidate")
 base={}; total=0.
 for k in candidates:
  _,s=model.distribution(history,prompt_context,observed,k); base[k]=(np.asarray(s),float(np.sum(student_t_entropy(s,model.df)))); total+=base[k][1]
 rng=np.random.default_rng(int(seed)); gains={}
 for j in candidates:
  mu,s=model.distribution(history,prompt_context,observed,j); remaining=[k for k in candidates if k!=j]; vals=[]
  for _ in range(mc_samples):
   sampled=np.asarray(mu)+np.asarray(s)*rng.standard_t(model.df,size=np.asarray(mu).shape); augmented=dict(observed); augmented[j]=sampled; entropy=0.
   for k in remaining:
    _,ss=model.distribution(history,prompt_context,augmented,k); entropy+=float(np.sum(student_t_entropy(ss,model.df)))
   vals.append(entropy)
  gains[j]=total-float(np.mean(vals))
 return gains if return_all else max(candidates,key=lambda k:(gains[k],-k))

def score_then_reveal(model,history,prompt_context,observed,full_values,target):
 before=sorted(observed); mu,s=model.distribution(history,prompt_context,observed,target); y=np.asarray(full_values[target]); nll=float(np.mean(student_t_nll(y,mu,s,model.df))); observed[target]=y.copy(); return {"target":int(target),"observed_before":before,"nll":nll}

def robust_top2(values):
 x=np.sort(np.asarray(values,float)); return float(np.mean(x[-min(2,len(x)):]))

def all9_score(model,history,prompt_context,full_values,order=None):
 order=list(SIDE_INDICES if order is None else order)
 if set(order)!=set(SIDE_INDICES) or len(order)!=8: raise ValueError("all9 means all eight side taps; Prompt excluded")
 scores=[]
 for target in order:
  observed={j:np.asarray(full_values[j]) for j in SIDE_INDICES if j!=target}; mu,s=model.distribution(history,prompt_context,observed,target); scores.append(float(np.mean(student_t_nll(full_values[target],mu,s,model.df))))
 return robust_top2(scores)

def epoch_random_extras(recording,time_s,prn,seed,count,values=None):
 if count not in (2,4): raise ValueError("extras count must be 2 or 4")
 h=hashlib.sha256(json.dumps([str(recording),float(time_s),str(prn),int(seed)],separators=(",",":")).encode()).digest(); rng=np.random.default_rng(int.from_bytes(h[:8],"little")); pool=np.array([0,1,2,6,7,8]); return list(map(int,rng.choice(pool,count,replace=False)))

def phase_masks(decision_times,onset_s):
 end=np.asarray(decision_times,float);start=end-.5;onset=float(onset_s)
 contained=lambda a,b:(start>=a)&(end<=b)
 return {"stable_pre":contained(30.,onset-20.),"transition":contained(onset-20.,onset),"post":start>=onset,"ramp":contained(onset,onset+20),"takeover":contained(onset+20,onset+40),"persistent":start>=onset+40}

def attack_free_thresholds(scores,roles,scenarios):
 x=np.asarray(scores,float); r=np.asarray(roles); s=np.asarray(scenarios)
 if not (len(x)==len(r)==len(s)) or np.any(r!="calibration"): raise ValueError("calibration role only")
 if np.any(s!="cleanStatic"): raise ValueError("cleanStatic calibration only; attack calibration rejected")
 return {"q99":float(np.quantile(x,.99,method="higher")),"q995":float(np.quantile(x,.995,method="higher")),"count":len(x),"comparison":"strict_greater","held_out_wording":"held-out chronological clean segment (not independent)"}

def alarm_columns(rows,primary_q99,primary_q995,matched_clean):
 out=copy.deepcopy(list(rows))
 for r in out:
  score=float(r["score"]); r["alarm_primary_q99"]=score>primary_q99; r["alarm_primary_q995"]=score>primary_q995; r["alarm_matched_clean_diagnostic"]=score>matched_clean
 return out

def verify_alarm_columns(rows,q99,q995,matched):
 if not rows:return 1.
 ok=0
 def flag(value):
  if isinstance(value,str): return value.strip().lower() in {"1","true","yes"}
  return bool(value)
 for r in rows: ok+=all(flag(r[k])==(float(r["score"])>v) for k,v in [("alarm_primary_q99",q99),("alarm_primary_q995",q995),("alarm_matched_clean_diagnostic",matched)])
 return ok/len(rows)

def checkpoint_save(path,model,optimizer,meta):
 torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"meta":dict(meta)},path)

def checkpoint_load(path,model,optimizer,map_location="cpu"):
 state=torch.load(path,map_location=map_location,weights_only=False); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); return state["meta"]

def exact_binomial_ci(successes,n,alpha=.05):
 from scipy.stats import beta
 if n==0:return (None,None)
 lo=0. if successes==0 else float(beta.ppf(alpha/2,successes,n-successes+1)); hi=1. if successes==n else float(beta.ppf(1-alpha/2,successes+1,n-successes)); return lo,hi

def block_bootstrap(values,statistic,reps=100,block_s=10.,cadence_s=.5,seed=0):
 x=np.asarray(values); size=max(1,int(round(block_s/cadence_s))); blocks=[x[i:i+size] for i in range(0,len(x),size)]; rng=np.random.default_rng(seed); vals=[]
 for _ in range(reps): vals.append(float(statistic(np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])[:len(x)])))
 return tuple(map(float,np.quantile(vals,[.025,.975])))

# --- AMCF-R1 strict correction helpers. ---
def full_history_indices(records, max_history=12):
    """Indices whose same-recording/PRN/role causal history is completely full."""
    return np.asarray([i for i in range(len(records)) if len(history_indices(records,i,max_history))==max_history],dtype=int)


def deterministic_training_masks(n, seed, epoch):
    """Deterministic mixture of E/L LOO and randomly enlarged observed sets."""
    n=int(n); masks=np.zeros((n,9),bool); targets=np.empty(n,np.int64)
    sides=np.asarray(SIDE_INDICES,dtype=int); extras=np.asarray([0,1,2,6,7,8],dtype=int)
    for i in range(n):
        raw=hashlib.sha256(json.dumps([int(seed),int(epoch),i],separators=(",",":")).encode()).digest()
        rng=np.random.default_rng(int.from_bytes(raw[:8],"little")); mode=i%4
        if mode==0: # strict E/L leave-one-out
            target=3 if ((i//4+epoch)&1)==0 else 5; masks[i,5 if target==3 else 3]=True
        elif mode==1: # both E/L observed, random target elsewhere
            target=int(rng.choice(extras)); masks[i,[3,5]]=True
        else: # E/L plus deterministic random extra observed sets
            target=int(rng.choice(sides)); pool=np.asarray([x for x in sides if x not in (target,3,5)],int)
            masks[i,[x for x in (3,5) if x!=target]]=True
            count=2 if mode==2 else 4
            if len(pool): masks[i,rng.choice(pool,min(count,len(pool)),replace=False)]=True
        masks[i,4]=False; masks[i,target]=False; targets[i]=target
    return masks,targets


def phase_destroy(z,seed):
    """Permute normalized relative phase independently by tap.

    Magnitudes remain attached to their original rows and every tap's empirical
    phase marginal is exactly preserved.  Only seed and array shape affect the
    permutations; labels/attack phase are deliberately absent from this API.
    """
    z=np.asarray(z,np.complex128); out=np.empty_like(z); rng=np.random.default_rng(int(seed))
    if z.ndim!=2: raise ValueError("rows x taps required")
    mag=np.abs(z); ang=np.angle(z)
    for j in range(z.shape[1]):
        if j==PROMPT_INDEX:
            out[:,j]=z[:,j];continue
        perm=rng.permutation(len(z))
        if len(z)>1 and np.array_equal(perm,np.arange(len(z))): perm=np.roll(perm,1)
        out[:,j]=mag[:,j]*np.exp(1j*ang[perm,j])
    return out


def temporal_shuffle(histories,seed,*,recording_id="",time_s=0.,prn=""):
    """Value-blind deterministic permutation of the 12 history positions."""
    x=np.asarray(histories).copy(); one=x.ndim==2
    if one: x=x[None]
    if x.ndim!=3 or x.shape[1]!=12: raise ValueError("[N,12,D] or [12,D] history required")
    for i,row in enumerate(x):
        digest=hashlib.sha256(json.dumps([str(recording_id),float(time_s)+(i if not one else 0),str(prn),int(seed)],separators=(",",":")).encode()).digest()
        rng=np.random.default_rng(int.from_bytes(digest[:8],"little")); perm=rng.permutation(12)
        if np.array_equal(perm,np.arange(12)) or np.array_equal(perm,np.arange(11,-1,-1)): perm=np.roll(perm,1)
        x[i]=row[perm]
    return x[0] if one else x


def _batch_forward(model,histories,current,masks,targets,prompts,chunk_size=32768):
    device=next(model.parameters()).device; outs=[]; scales=[]
    n=len(targets)
    with torch.no_grad():
        for a in range(0,n,int(chunk_size)):
            b=min(n,a+int(chunk_size)); mu,sc=model(
                torch.as_tensor(histories[a:b],dtype=torch.float32,device=device),
                torch.as_tensor(current[a:b],dtype=torch.float32,device=device),
                torch.as_tensor(masks[a:b],dtype=torch.bool,device=device),
                torch.as_tensor(targets[a:b],dtype=torch.long,device=device),
                torch.as_tensor(prompts[a:b],dtype=torch.float32,device=device))
            outs.append(mu.cpu().numpy());scales.append(sc.cpu().numpy())
    return np.concatenate(outs),np.concatenate(scales)


def batch_distributions(model,histories,prompt_context,values,observed_masks,targets,chunk_size=32768):
    """One distribution per row; values outside observed_masks are inaccessible."""
    h=np.asarray(histories,np.float32); p=np.asarray(prompt_context,np.float32); v=np.asarray(values,np.float32); m=np.asarray(observed_masks,bool); t=np.asarray(targets,np.int64)
    if not (len(h)==len(p)==len(v)==len(m)==len(t)): raise ValueError("aligned batch required")
    safe=np.where(m[...,None],v,0.)
    return _batch_forward(model,h,safe,m,t,p,chunk_size)


def _binary_auc(negative,positive):
    a=np.asarray(negative,float);b=np.asarray(positive,float)
    if not len(a) or not len(b): return None
    d=b[:,None]-a[None,:];return float(np.mean(d>0)+.5*np.mean(d==0))


def _average_precision(labels,scores):
    y=np.asarray(labels,bool);s=np.asarray(scores,float)
    if not y.any() or y.all(): return None
    order=np.argsort(-s,kind="mergesort"); yy=y[order]; precision=np.cumsum(yy)/(np.arange(len(yy))+1)
    return float(np.sum(precision*yy)/np.sum(yy))


def _rate(flags): return float(np.mean(flags)) if len(flags) else None


def evaluate_detector(rows,threshold,*,scenario,bootstrap_reps=100,seed=0,onset_s=None):
    """Pure reusable detector evaluator on common decision rows."""
    data=[dict(r) for r in rows]; data.sort(key=lambda r:float(r["decision_time_s"])); score=np.asarray([float(r["score"]) for r in data]); ph=np.asarray([str(r.get("phase","")) for r in data]); tt=np.asarray([float(r["decision_time_s"]) for r in data]); alarm=score>float(threshold)
    stable=ph=="stable_pre"; clean=ph=="clean_test"; post=np.isin(ph,["post","ramp","takeover","persistent"]); persistent=ph=="persistent"
    negative=clean if scenario=="cleanStatic" else stable; positive=post
    use=negative if scenario=="cleanStatic" else post; k=int(alarm[use].sum()); n=int(use.sum()); ci=exact_binomial_ci(k,n)
    delay=None
    if scenario!="cleanStatic" and onset_s is None and post.any(): onset_s=float(tt[post][0])
    if onset_s is not None:
        for i in range(max(0,len(data)-2)):
            if alarm[i:i+3].all() and tt[i]>=onset_s and np.allclose(np.diff(tt[i:i+3]),.5,rtol=0,atol=1e-9): delay=float(tt[i]-onset_s);break
    ref=score[negative]
    metric={"scenario":scenario,"threshold":float(threshold),"comparison":"strict_greater","held_out_clean_fpr":_rate(alarm[clean]),"stable_pre_fpr":_rate(alarm[stable]),"roc_auc":_binary_auc(score[negative],score[positive]) if scenario!="cleanStatic" else None,"pr_auc":_average_precision(np.r_[np.zeros(negative.sum(),bool),np.ones(positive.sum(),bool)],np.r_[score[negative],score[positive]]) if scenario!="cleanStatic" and negative.any() and positive.any() else None,"post_detection":_rate(alarm[post]),"persistent_detection":_rate(alarm[persistent]),"first_sustained_3_delay_s":delay,"q99":float(np.quantile(ref,.99,method="higher")) if len(ref) else None,"q995":float(np.quantile(ref,.995,method="higher")) if len(ref) else None,"count":n,"alarms":k,"exact_binomial_ci_low":ci[0],"exact_binomial_ci_high":ci[1]}
    for prefix,mask0 in (("held_out_clean",clean),("stable_pre",stable),("post",post),("persistent",persistent)):
        nn=int(mask0.sum());kk=int(alarm[mask0].sum());cc=exact_binomial_ci(kk,nn)
        metric[f"{prefix}_exact_binomial_ci_low"]=cc[0];metric[f"{prefix}_exact_binomial_ci_high"]=cc[1];metric[f"{prefix}_count"]=nn
    boots=[]
    if n and bootstrap_reps:
        vals=alarm[use].astype(float);lo,hi=block_bootstrap(vals,np.mean,reps=int(bootstrap_reps),block_s=10.,cadence_s=.5,seed=seed);boots.append({"scenario":scenario,"metric":"clean_fpr" if scenario=="cleanStatic" else "post_detection","ci_low":lo,"ci_high":hi,"reps":int(bootstrap_reps),"block_seconds":10,"phase_aware":scenario!="cleanStatic"})
    if scenario!="cleanStatic" and negative.any() and positive.any() and bootstrap_reps:
        # Phase-aware contiguous blocks: resample within negative and positive phases.
        rng=np.random.default_rng(seed);bs=20; nb=[score[negative][i:i+bs] for i in range(0,negative.sum(),bs)];pb=[score[positive][i:i+bs] for i in range(0,positive.sum(),bs)];av=[];ap=[]
        for _ in range(int(bootstrap_reps)):
            aa=np.concatenate([nb[i] for i in rng.integers(0,len(nb),len(nb))])[:negative.sum()];bb=np.concatenate([pb[i] for i in rng.integers(0,len(pb),len(pb))])[:positive.sum()];av.append(_binary_auc(aa,bb));ap.append(_average_precision(np.r_[np.zeros(len(aa),bool),np.ones(len(bb),bool)],np.r_[aa,bb]))
        for name,vv in (("roc_auc",av),("pr_auc",ap)):
            lo,hi=np.quantile(vv,[.025,.975]);boots.append({"scenario":scenario,"metric":name,"ci_low":float(lo),"ci_high":float(hi),"reps":int(bootstrap_reps),"block_seconds":10,"phase_aware":True})
    return metric,boots


def train_model(train_current,train_history,val_current,val_history,*,train_prompt_context=None,val_prompt_context=None,seed=101,hidden=32,epochs=50,patience=8,device="cpu",lr=1e-3):
    torch.manual_seed(seed);np.random.seed(seed);model=AMCFModel(train_current.shape[-1],hidden).to(device);opt=torch.optim.AdamW(model.parameters(),lr=lr)
    tc=torch.as_tensor(train_current,dtype=torch.float32,device=device);th=torch.as_tensor(train_history,dtype=torch.float32,device=device);vc=torch.as_tensor(val_current,dtype=torch.float32,device=device);vh=torch.as_tensor(val_history,dtype=torch.float32,device=device)
    tp=np.zeros((len(tc),3),np.float32) if train_prompt_context is None else np.asarray(train_prompt_context,np.float32);vp=np.zeros((len(vc),3),np.float32) if val_prompt_context is None else np.asarray(val_prompt_context,np.float32)
    tp=torch.as_tensor(tp,dtype=torch.float32,device=device);vp=torch.as_tensor(vp,dtype=torch.float32,device=device)
    best_model=None;best_opt=None;best_loss=float("inf");best_epoch=0;wait=0;hist=[];early=False
    for epoch in range(1,int(epochs)+1):
        model.train();ma,ta=deterministic_training_masks(len(tc),seed,epoch-1);mask=torch.as_tensor(ma,device=device);target=torch.as_tensor(ta,device=device)
        opt.zero_grad();mu,sc=model(th,tc,mask,target,tp);loss=student_t_nll(tc[torch.arange(len(tc),device=device),target],mu,sc).mean()
        if not torch.isfinite(loss):raise FloatingPointError("nonfinite training loss")
        loss.backward();finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if not finite:raise FloatingPointError("nonfinite gradient")
        opt.step();model.eval()
        with torch.no_grad():
            vm,vt=deterministic_training_masks(len(vc),seed,100000+epoch);vm=torch.as_tensor(vm,device=device);vt=torch.as_tensor(vt,device=device);a,b=model(vh,vc,vm,vt,vp);vl=float(student_t_nll(vc[torch.arange(len(vc),device=device),vt],a,b).mean().cpu())
        hist.append({"epoch":epoch,"train_nll":float(loss.detach().cpu()),"validation_nll":vl,"gradients_finite":finite})
        if np.isfinite(vl) and vl<best_loss-1e-9:best_loss=vl;best_epoch=epoch;best_model=copy.deepcopy(model.state_dict());best_opt=copy.deepcopy(opt.state_dict());wait=0
        else:wait+=1
        if wait>=int(patience) and epoch<int(epochs):early=True;break
    if best_model is None:raise RuntimeError("no finite checkpoint")
    model.load_state_dict(best_model);opt.load_state_dict(best_opt)
    audit={"finite":True,"best_restored":True,"optimizer_best_restored":True,"best_epoch":best_epoch,"best_validation_nll":best_loss,"epochs_run":len(hist),"early_stopped":early,"max_epoch_reached":not early and len(hist)==int(epochs),"converged":bool(early),"optimizer":"AdamW","df":4,"hidden":hidden,"seed":seed}
    return model,opt,hist,audit


# Cached-history CUDA path: hypothetical IG rows differ only in current tokens,
# so recomputing the GRU tens of thousands of times is scientifically redundant.
def _encode_history(self,history):
    _,state=self.temporal(history);return state[-1]
def _forward_cached(self,current,observed_mask,target,prompt_context,temporal_state):
    b=current.shape[0];device=current.device;mask=observed_mask.bool().clone();mask[:,PROMPT_INDEX]=False
    safe=torch.where(mask[...,None],current,torch.zeros_like(current));coords=torch.as_tensor(TAP_COORDS,device=device,dtype=current.dtype)[None,:,None].expand(b,-1,-1)
    tok=self.token(torch.cat([safe,coords,mask[...,None].to(current.dtype)],-1));keymask=~mask;empty=~mask.any(1)
    if empty.any():keymask[empty,3]=False
    att,_=self.attn(tok,tok,tok,key_padding_mask=keymask,need_weights=False);weights=mask.to(current.dtype);pooled=(att*weights[...,None]).sum(1)/weights.sum(1,keepdim=True).clamp_min(1)
    pc=prompt_context
    if pc.shape[1]<3:pc=torch.cat([pc,torch.zeros((b,3-pc.shape[1]),device=device,dtype=current.dtype)],1)
    pc=pc[:,:3];ti=target.long();tc=torch.stack([torch.as_tensor(TAP_COORDS,device=device,dtype=current.dtype)[ti],(ti>4).to(current.dtype)],1)
    q=self.target(tc)+self.prompt(pc);raw=self.decoder(torch.cat([pooled,temporal_state,q],1));loc,logs=raw.chunk(2,1);return loc,torch.nn.functional.softplus(logs)+1e-4
AMCFModel.encode_history=_encode_history
AMCFModel.forward_cached=_forward_cached

def _cached_forward_numpy(model,temporal,current,masks,targets,prompts,chunk_size):
    device=next(model.parameters()).device;mu=[];sc=[]
    with torch.no_grad():
        for a in range(0,len(targets),int(chunk_size)):
            b=min(len(targets),a+int(chunk_size));x,y=model.forward_cached(torch.as_tensor(current[a:b],dtype=torch.float32,device=device),torch.as_tensor(masks[a:b],dtype=torch.bool,device=device),torch.as_tensor(targets[a:b],dtype=torch.long,device=device),torch.as_tensor(prompts[a:b],dtype=torch.float32,device=device),torch.as_tensor(temporal[a:b],dtype=torch.float32,device=device));mu.append(x.cpu().numpy());sc.append(y.cpu().numpy())
    return np.concatenate(mu),np.concatenate(sc)

def expected_information_gain_batch(model,histories,prompt_context,values,observed_masks,candidates,mc_samples=8,seeds=None,chunk_size=32768):
    h=np.asarray(histories,np.float32);p=np.asarray(prompt_context,np.float32);v=np.asarray(values,np.float32);masks=np.asarray(observed_masks,bool);candidates=np.asarray([int(x) for x in candidates],np.int64)
    if np.any(candidates==4) or not len(candidates):raise ValueError("non-Prompt candidates required")
    B=len(h);C=len(candidates);seeds=np.arange(B,dtype=np.int64) if seeds is None else np.asarray(seeds,np.int64)
    device=next(model.parameters()).device
    with torch.no_grad():temporal=model.encode_history(torch.as_tensor(h,dtype=torch.float32,device=device)).cpu().numpy()
    rid=np.repeat(np.arange(B),C);targets=np.tile(candidates,B);bm=masks[rid].copy();safe=np.where(bm[...,None],v[rid],0.)
    mu,sc=_cached_forward_numpy(model,temporal[rid],safe,bm,targets,p[rid],chunk_size);mu=mu.reshape(B,C,-1);sc=sc.reshape(B,C,-1);total=np.sum(student_t_entropy(sc,model.df),axis=2).sum(1)
    rows_v=[];rows_m=[];rows_t=[];rows_p=[];rows_z=[];owners=[]
    for b in range(B):
        for jj,j in enumerate(candidates):
            dig=hashlib.sha256(json.dumps([int(seeds[b]),int(j),int(mc_samples)],separators=(",",":")).encode()).digest();rng=np.random.default_rng(int.from_bytes(dig[:8],"little"));samples=mu[b,jj]+sc[b,jj]*rng.standard_t(model.df,size=(int(mc_samples),mu.shape[2]))
            base=np.where(masks[b,:,None],v[b],0.)
            for q in range(int(mc_samples)):
                for kk,k in enumerate(candidates):
                    if kk==jj:continue
                    vv=base.copy();vv[j]=samples[q];mm=masks[b].copy();mm[j]=True;rows_v.append(vv);rows_m.append(mm);rows_t.append(k);rows_p.append(p[b]);rows_z.append(temporal[b]);owners.append((b,jj,q))
    _,hs=_cached_forward_numpy(model,np.asarray(rows_z),np.asarray(rows_v),np.asarray(rows_m),np.asarray(rows_t),np.asarray(rows_p),chunk_size);acc=np.zeros((B,C,int(mc_samples)),float)
    for ent,owner in zip(np.sum(student_t_entropy(hs,model.df),axis=1),owners):acc[owner]+=float(ent)
    return total[:,None]-acc.mean(2)
