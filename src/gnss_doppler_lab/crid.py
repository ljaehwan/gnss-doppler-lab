"""CRID-GNSS counterfactual receiver-invariance statistics.

This module contains no attack labels. C/N0 and lock are validity masks only;
neither enters the response vector or detector score.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json
from pathlib import Path
from typing import Iterable, Mapping
import numpy as np
from .trace_native_1ms import complex_taps, read_records

CONFIG_ORDER=("C0","C1","C2","C3")
FEATURE_NAMES=("dll_discriminator_chips","pll_phase_error_cycles","fll_frequency_error_hz",
 "delta_code_filter_output_chips_s","delta_carrier_filter_output_hz",
 "prompt_phase_increment_rad","early_minus_prompt_i","early_minus_prompt_q",
 "late_minus_prompt_i","late_minus_prompt_q")

@dataclass(frozen=True)
class ResponseTable:
 config:np.ndarray;prn:np.ndarray;sample:np.ndarray;time_s:np.ndarray
 session:np.ndarray;response:np.ndarray;cn0:np.ndarray;lock:np.ndarray

@dataclass(frozen=True)
class NormalModel:
 order:int;ridge:float;shrinkage:float
 coefficients:dict[str,np.ndarray];means:dict[str,np.ndarray]
 whiteners:dict[str,np.ndarray];h_matrices:dict[str,np.ndarray];latent_dimension:int

def _native_cadence(table:ResponseTable)->int:
 values=[]
 for prn in np.unique(table.prn):
  delta=np.diff(np.sort(table.sample[table.prn==prn]));delta=delta[delta>0]
  if len(delta):values.append(float(np.median(delta)))
 if not values:raise ValueError("no native cadence support")
 return int(np.rint(np.median(values)))

def canonical_json(value:object)->str:
 return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
def sha256_bytes(value:bytes)->str:return hashlib.sha256(value).hexdigest()

def receiver_configurations()->dict[str,dict[str,object]]:
 common={"implementation":"GPS_L1_CA_DLL_PLL_Tracking","item_type":"gr_complex",
  "coherent_integration_ms":1,"loop_order":3,"tap_count":9,
  "extend_correlation_symbols":1,
  "fll_enabled":"receiver-native pull-in behavior; no explicit override",
  "discriminators":{"dll":"GNSS-SDR native GPS L1 C/A DLL",
   "pll":"GNSS-SDR native GPS L1 C/A PLL","fll":"GNSS-SDR native pull-in FLL"}}
 vals={"C0":(.125,1.5,20.,"native_standard"),"C1":(.10,1.5,20.,"narrow_correlator"),
  "C2":(.125,.5,5.,"slow_narrow_loop"),"C3":(.125,5.,25.,"fast_wide_loop")}
 return {n:{**common,"name":n,"description":d,"Tracking_1C.tap_spacing_chips":s,
  "Tracking_1C.dll_bw_hz":dll,"Tracking_1C.pll_bw_hz":pll}
  for n,(s,dll,pll,d) in vals.items()}

def render_receiver_config(base:str,config:Mapping[str,object],values:Mapping[str,object])->str:
 repl={"Tracking_1C.tap_count":config["tap_count"],
  "Tracking_1C.tap_spacing_chips":config["Tracking_1C.tap_spacing_chips"],
  "Tracking_1C.dll_bw_hz":config["Tracking_1C.dll_bw_hz"],
  "Tracking_1C.pll_bw_hz":config["Tracking_1C.pll_bw_hz"],
  "Tracking_1C.extend_correlation_symbols":config["extend_correlation_symbols"],**values}
 found=set();lines=[]
 for line in base.splitlines():
  stripped=line.strip();key=stripped.split("=",1)[0] if "=" in stripped and not stripped.startswith("#") else ""
  if key in repl:lines.append(f"{key}={repl[key]}");found.add(key)
  else:lines.append(line)
 lines.extend(f"{k}={v}" for k,v in repl.items() if k not in found)
 return "\n".join(lines)+"\n"

def load_response(config:str,paths:Iterable[Path],cn0_min:float=28.,lock_min:float=.85)->ResponseTable:
 parts={k:[] for k in ("prn","sample","time","session","response","cn0","lock")}
 for path in sorted(paths):
  _,rows=read_records(path)
  if len(rows)<2:continue
  taps=complex_taps(rows);prompt=taps[:,4];vp=np.abs(prompt)>1e-12
  norm=taps/np.where(vp,prompt,1.)[:,None]
  phase=np.angle(prompt[1:]*np.conj(prompt[:-1]))
  same=((rows["prn"][1:]==rows["prn"][:-1])&
   (rows["tracking_session_id"][1:]==rows["tracking_session_id"][:-1])&
   (rows["loop_sequence"][1:]==rows["loop_sequence"][:-1]+1)&
   (rows["raw_interval_start_sample"][1:]==rows["raw_interval_end_sample"][:-1]))
  quality=(same&vp[1:]&vp[:-1]&(rows["valid_tracking"][1:]==1)&
   (rows["valid_lock"][1:]==1)&(rows["cn0_db_hz"][1:]>=cn0_min)&
   (rows["carrier_lock_test"][1:]>=lock_min))
  idx=np.flatnonzero(quality)+1
  if not len(idx):continue
  prev=idx-1
  feat=np.column_stack([rows["dll_discriminator_chips"][idx],
   rows["pll_phase_error_cycles"][idx],rows["fll_frequency_error_hz"][idx],
   rows["code_filter_output_chips_s"][idx]-rows["code_filter_output_chips_s"][prev],
   rows["carrier_filter_output_hz"][idx]-rows["carrier_filter_output_hz"][prev],
   phase[prev],(norm[idx,3]-1).real,(norm[idx,3]-1).imag,
   (norm[idx,5]-1).real,(norm[idx,5]-1).imag]).astype(float)
  finite=np.isfinite(feat).all(1);idx=idx[finite];feat=feat[finite]
  parts["prn"].append(rows["prn"][idx].astype(np.int16))
  parts["sample"].append(rows["raw_interval_start_sample"][idx].astype(np.int64))
  parts["time"].append(rows["receiver_timestamp_s"][idx].astype(float))
  parts["session"].append(rows["tracking_session_id"][idx].astype(np.uint64))
  parts["response"].append(feat);parts["cn0"].append(rows["cn0_db_hz"][idx].astype(float))
  parts["lock"].append(rows["carrier_lock_test"][idx].astype(float))
 if not parts["response"]:raise ValueError(f"no valid response rows for {config}")
 n=sum(map(len,parts["prn"]))
 return ResponseTable(np.full(n,config),np.concatenate(parts["prn"]),
  np.concatenate(parts["sample"]),np.concatenate(parts["time"]),
  np.concatenate(parts["session"]),np.concatenate(parts["response"]),
  np.concatenate(parts["cn0"]),np.concatenate(parts["lock"]))

def estimate_causal_delays(tables:Mapping[str,ResponseTable],max_lag:int=20)->dict[str,int]:
 ref=tables["C0"];refmap={(int(p),int(s)):r for p,s,r in zip(ref.prn,ref.sample,ref.response,strict=True)}
 out={"C0":0}
 for config in CONFIG_ORDER[1:]:
  tab=tables[config];cadence=_native_cadence(tab);best=(np.inf,0)
  for lag in range(max_lag+1):
   err=[]
   for p,s,row in zip(tab.prn,tab.sample,tab.response,strict=True):
    other=refmap.get((int(p),int(s-lag*cadence)))
    if other is not None:err.append(np.mean((row[:6]-other[:6])**2))
   value=float(np.median(err)) if err else np.inf
   if value<best[0]:best=(value,lag)
  if not np.isfinite(best[0]):raise ValueError(f"no exact-sample clean overlap for {config}")
  out[config]=best[1]
 return out

def chronological_split(samples:np.ndarray)->dict[str,np.ndarray]:
 u=np.unique(samples);n=len(u);cuts=[int(n*x) for x in (.45,.47,.70,.72)]
 names=("train","guard1","calibration","guard2","holdout")
 bounds=((0,cuts[0]),(cuts[0],cuts[1]),(cuts[1],cuts[2]),(cuts[2],cuts[3]),(cuts[3],n))
 out={name:u[a:b] for name,(a,b) in zip(names,bounds,strict=True)}
 if any(not len(out[n]) for n in names):raise ValueError("insufficient epochs for guarded split")
 return out

def _design(tab:ResponseTable,order:int):
 x=[];y=[];target=[]
 for prn in np.unique(tab.prn):
  idx=np.flatnonzero(tab.prn==prn);idx=idx[np.argsort(tab.sample[idx])]
  cadence=float(np.median(np.diff(tab.sample[idx]))) if len(idx)>1 else 0.
  for j in range(order,len(idx)):
   history=idx[j-order:j]
   delta=np.diff(tab.sample[np.r_[history,idx[j]]])
   if cadence<=0 or np.any(delta<.9*cadence) or np.any(delta>1.1*cadence):continue
   x.append(tab.response[history].reshape(-1));y.append(tab.response[idx[j]]);target.append(idx[j])
 return np.asarray(x),np.asarray(y),np.asarray(target,int)

def fit_normal_model(tables:Mapping[str,ResponseTable],train_samples:np.ndarray,
 calibration_samples:np.ndarray,order:int=4,ridge:float=1e-3,
 shrinkage:float=.2,latent_dimension:int=2)->NormalModel:
 coefs={};means={};whiteners={};blocks=[]
 for config in CONFIG_ORDER:
  tab=tables[config];x,y,target=_design(tab,order)
  train=np.isin(tab.sample[target],train_samples);cal=np.isin(tab.sample[target],calibration_samples)
  if train.sum()<x.shape[1]+1 or cal.sum()<20:raise ValueError(f"insufficient normal fit: {config}")
  xt=x[train];yt=y[train];beta=np.linalg.solve(xt.T@xt+ridge*np.eye(xt.shape[1]),xt.T@yt)
  residual=y[cal]-x[cal]@beta;mean=residual.mean(0);centered=residual-mean
  cov=np.cov(centered,rowvar=False);diag=np.diag(np.diag(cov))
  cov=(1-shrinkage)*cov+shrinkage*diag+1e-8*np.eye(cov.shape[0])
  values,vectors=np.linalg.eigh(cov);white=(vectors*np.maximum(values,1e-8)**-.5)@vectors.T
  coefs[config]=beta;means[config]=mean;whiteners[config]=white;blocks.append(centered@white.T)
 _,_,vt=np.linalg.svd(np.concatenate(blocks),full_matrices=False)
 h={c:vt[:latent_dimension].T for c in CONFIG_ORDER}
 return NormalModel(order,ridge,shrinkage,coefs,means,whiteners,h,latent_dimension)

def residual_table(tab:ResponseTable,model:NormalModel)->dict[tuple[int,int],np.ndarray]:
 x,y,target=_design(tab,model.order);config=str(tab.config[0])
 z=(y-x@model.coefficients[config]-model.means[config])@model.whiteners[config].T
 return {(int(tab.prn[i]),int(tab.sample[i])):row for i,row in zip(target,z,strict=True)}

def score_epoch(z_by_prn:Mapping[int,Mapping[str,np.ndarray]],
 hmat:Mapping[str,np.ndarray],configs:tuple[str,...]=CONFIG_ORDER)->dict[str,object]:
 per=[];h0=h1=penalty=disagreement=0.
 for prn in sorted(z_by_prn):
  z=np.concatenate([z_by_prn[prn][c] for c in configs]);h=np.vstack([hmat[c] for c in configs])
  x=np.linalg.lstsq(h,z,rcond=None)[0];rss0=float(np.sum((z-h@x)**2));fits=[];states=[]
  for c in configs:
   hc=hmat[c];zc=z_by_prn[prn][c]
   xc=np.linalg.solve(hc.T@hc+.1*np.eye(hc.shape[1]),hc.T@zc)
   fits.append(hc@xc);states.append(xc)
  rss1=float(np.sum((z-np.concatenate(fits))**2));df=(len(configs)-1)*h.shape[1]
  p=float(df*np.log(max(len(z),2)));per.append(rss0-rss1-p)
  h0+=-.5*rss0;h1+=-.5*rss1;penalty+=p
  disagreement+=float(np.mean(np.std(np.vstack(states),axis=0)))
 values=np.asarray(per)
 return {"score":float(np.median(values)),"per_prn_scores":values,
  "h0_loglike":h0,"h1_loglike":h1,"penalty":penalty,
  "configuration_disagreement":disagreement/max(len(per),1),
  "prn_count":len(per),"config_count":len(configs)}

def score_aligned(tables:Mapping[str,ResponseTable],model:NormalModel,
 delays:Mapping[str,int],minimum_prns:int=4)->list[dict[str,object]]:
 residuals={c:residual_table(tables[c],model) for c in CONFIG_ORDER};by_prn={}
 for c in CONFIG_ORDER:
  cadence=_native_cadence(tables[c]);groups={}
  for (p,s),z in residuals[c].items():groups.setdefault(p,[]).append((s-int(delays[c])*cadence,z))
  by_prn[c]={p:(np.array([x[0] for x in sorted(v)]),[x[1] for x in sorted(v)]) for p,v in groups.items()}
 epochs={};epoch_samples={};reference_cadence=_native_cadence(tables["C0"])
 for (p,s),z0 in residuals["C0"].items():
  matched={"C0":z0};ok=True
  for c in CONFIG_ORDER[1:]:
   if p not in by_prn[c]:ok=False;break
   samples,values=by_prn[c][p];j=int(np.searchsorted(samples,s));candidates=[k for k in (j-1,j) if 0<=k<len(samples)]
   if not candidates:ok=False;break
   k=min(candidates,key=lambda q:abs(int(samples[q])-s))
   if abs(int(samples[k])-s)>1:ok=False;break
   matched[c]=values[k]
  if ok:
   epoch=int(np.rint(s/reference_cadence));epochs.setdefault(epoch,{})[p]=matched;epoch_samples.setdefault(epoch,[]).append(s)
 return [{"sample":int(np.rint(np.median(epoch_samples[epoch]))),"epoch":epoch,**score_epoch(prns,model.h_matrices)} for epoch,prns in sorted(epochs.items())
  if len(prns)>=minimum_prns]

def empirical_threshold(scores:np.ndarray,quantile:float)->float:
 if not len(scores):raise ValueError("empty calibration scores")
 return float(np.quantile(np.asarray(scores,float),quantile,method="higher"))

def verify_permutations(z_by_prn:Mapping[int,Mapping[str,np.ndarray]],h)->dict[str,object]:
 base=float(score_epoch(z_by_prn,h)["score"]);pr={p:z_by_prn[p] for p in reversed(list(z_by_prn))}
 ps=float(score_epoch(pr,h)["score"]);cs=float(score_epoch(z_by_prn,h,tuple(reversed(CONFIG_ORDER)))["score"])
 return {"base":base,"prn_permuted":ps,"config_permuted":cs,
  "pass":bool(np.isclose(base,ps,atol=1e-10) and np.isclose(base,cs,atol=1e-10))}
