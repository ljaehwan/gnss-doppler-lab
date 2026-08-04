#!/usr/bin/env python3
"""Standalone, fail-closed verifier for the NC-TOPI Stage-0B artifact.

This file intentionally imports no runner or project audit implementation.  All
parsing, fitting, scoring, aggregation, metrics, resampling, decisions, support,
and publication checks are independently implemented below.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,platform,subprocess,sys,tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping,Sequence
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
METHODS=("B0","TOPI","NC_TOPI_original","IQ_LOW_ONLY","IQ_OOD_ONLY","NC_TOPI_clamped","NC_B0_clamped","NC_total_clamped")
TARGETS={"TOPI":"TOPI","B0":"B0","total":"total"}
COMPARATORS=("B0","TOPI","IQ_LOW_ONLY","IQ_OOD_ONLY","NC_B0_clamped","NC_total_clamped")
ATTACKS=("DS1","DS2","DS3","DS7","DS8")
ONSETS={"DS1":100.,"DS2":100.,"DS3":100.,"DS7":110.,"DS8":110.}
EPS=1e-12
PARENT_COMMIT="6fe5315ca0d71689609895cd3b1366bcfa1b93c1"
PARENT_SOURCE="c94af28795d03a91e2f4c0faa74eb19a983ed82e"
IDENTITY_PRN=("scenario","physical_recording_id","event_id","prn","pair_sequence_index","prn_target_index","target_index","availability_time_s","source_start_s","source_end_s","role","phase","label","valid","tracked_prn_count")
IDENTITY_EVENT=("scenario","physical_recording_id","event_id","target_index","availability_time_s","source_start_s","source_end_s","role","phase","label","valid","tracked_prn_count")
IQ_FIELDS=("scenario","physical_recording_id","block_recording_id","event_id","window_bin_s","target_source_start_s","history_blocks","cadence_seconds","block_end_s","block_start_s","sample_offset","sample_count","block_features_json","context_features_json","linked_prns","linked_pair_count","history_reducer")

def _bad_constant(token):raise ValueError(f"non-finite JSON constant: {token}")
def read_json_strict(path):return json.loads(Path(path).read_text(encoding="utf-8"),parse_constant=_bad_constant)
def dump_json(value):return json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n"
def sha256_file(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def digest_json(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def digest_array(value):
 a=np.ascontiguousarray(np.asarray(value,dtype=np.float64));return hashlib.sha256(str(a.shape).encode()+b"|float64|"+a.tobytes()).hexdigest()
def parse_bool(v):
 if v in ("True","true"):return True
 if v in ("False","false"):return False
 raise ValueError(f"invalid boolean token {v!r}")
def finite(value,name="value"):
 x=float(value)
 if not math.isfinite(x):raise ValueError(f"{name} non-finite")
 return x
def higher_quantile(values,q):
 a=np.asarray(values,dtype=float)
 if a.ndim!=1 or not len(a) or not np.isfinite(a).all() or not 0<=q<=1:raise ValueError("invalid quantile")
 return float(np.quantile(a,q,method="higher"))
def empirical_ood(ref,values):
 r=np.sort(np.asarray(ref,dtype=float));x=np.asarray(values,dtype=float);le=np.searchsorted(r,x,side="right");ge=len(r)-np.searchsorted(r,x,side="left")
 return -np.log(np.maximum(np.minimum(1.,2*np.minimum((1+le)/(len(r)+1),(1+ge)/(len(r)+1))),EPS))
def read_csv_strict(path):
 with Path(path).open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
 for i,row in enumerate(rows,2):
  for key,value in row.items():
   if value is None:raise ValueError(f"ragged CSV {path}:{i}")
   if value.strip().lower() in {"nan","+nan","-nan","inf","+inf","-inf","infinity","+infinity","-infinity"}:raise ValueError(f"non-finite CSV {path}:{i}:{key}")
 return rows

def hash_entries(root):
 root=Path(root);out=[]
 for p in sorted((x for x in root.rglob("*") if x.is_file() and x.name!="hashes.json"),key=lambda x:str(x.relative_to(root))):out.append({"relative_path":str(p.relative_to(root)),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
 return out
def write_hash_manifest(root):
 root=Path(root);payload={"schema":"gnss-doppler-lab.nc-topi-stage0b.hashes.v1","algorithm":"SHA-256","self_excluded":"hashes.json","files":hash_entries(root)};(root/"hashes.json").write_text(dump_json(payload));return payload
def verify_hashes(root):
 errors=[];root=Path(root)
 try:stored=read_json_strict(root/"hashes.json")
 except Exception as e:return {"ok":False,"errors":[f"hash manifest unreadable: {type(e).__name__}: {e}"],"file_count":0}
 if not isinstance(stored,dict):return {"ok":False,"errors":["hash manifest must be an object"],"file_count":0}
 if stored.get("algorithm")!="SHA-256" or stored.get("self_excluded")!="hashes.json":errors.append("hash manifest header invalid")
 entries=stored.get("files")
 if not isinstance(entries,list):return {"ok":False,"errors":errors+["hash files must be a list"],"file_count":0}
 if any(not isinstance(x,dict) for x in entries):return {"ok":False,"errors":errors+["hash entries must be objects"],"file_count":0}
 names=[x.get("relative_path") for x in entries]
 if any(not isinstance(x,str) for x in names) or names!=sorted(names) or len(names)!=len(set(names)):errors.append("hash entries not sorted/unique strings")
 actual=hash_entries(root)
 if entries!=actual:errors.append("complete self-excluding hash inventory mismatch")
 return {"ok":not errors,"errors":errors,"file_count":len(actual)}
def verify_exact_inventory(root,config,prepare=False):
 root=Path(root);c=config.get("artifact_contract",{}) if isinstance(config,dict) else {};errors=[]
 sets={"root":c.get("required_root_regular_files_exact"),"diagnostics":c.get("required_diagnostics_regular_files_exact"),"plots":c.get("required_plots_regular_files_exact")}
 if any(not isinstance(x,list) or any(not isinstance(y,str) or "/" in y for y in x) for x in sets.values()):return ["config exact inventory sets malformed"]
 actual={"root":sorted(p.name for p in root.iterdir() if p.is_file()),"diagnostics":sorted(p.name for p in (root/"diagnostics").iterdir() if p.is_file()) if (root/"diagnostics").is_dir() else [],"plots":sorted(p.name for p in (root/"plots").iterdir() if p.is_file()) if (root/"plots").is_dir() else []}
 expected={k:sorted(v) for k,v in sets.items()}
 if prepare:
  expected["root"]=[x for x in expected["root"] if x not in ("verification.json","hashes.json")]
 for k in expected:
  missing=sorted(set(expected[k])-set(actual[k]));extra=sorted(set(actual[k])-set(expected[k]))
  if missing:errors.append(f"missing {k} files: {missing}")
  if extra:errors.append(f"unexpected {k} files: {extra}")
 other=[str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.parent not in (root,root/"diagnostics",root/"plots")]
 if other:errors.append(f"unexpected nested files: {sorted(other)}")
 return errors

class Parent:
 def __init__(self,prn,events,iq,features,event_index,ids):
  self.prn=prn;self.events=events;self.iq=iq;self.features=features;self.event_index=event_index;self.ids=ids

def event_key(r):return (r["scenario"],r["physical_recording_id"],r["event_id"])
def identity(r):return json.dumps([r["physical_recording_id"],r["scenario"],r["prn"],int(r["prn_target_index"]),float(r["availability_time_s"])],separators=(",",":"))
def load_parent(parent):
 parent=Path(parent);rows=read_csv_strict(parent/"per_epoch_scores.csv");prn=[];events=[];groups={};emap={};sp=set();se=set()
 for r in rows:
  parse_bool(r["valid"]);[finite(r[x],x) for x in ("availability_time_s","source_start_s","source_end_s")];key=event_key(r)
  if r["row_level"]=="prn":
   ident=tuple(r[x] for x in IDENTITY_PRN)
   if ident in sp:raise ValueError("duplicate parent PRN identity")
   sp.add(ident);prn.append(r);groups.setdefault(key,[]).append(r)
  elif r["row_level"]=="event":
   ident=tuple(r[x] for x in IDENTITY_EVENT)
   if ident in se or key in emap:raise ValueError("duplicate parent event identity")
   se.add(ident);events.append(r);emap[key]=r
  else:raise ValueError("unknown parent row level")
 if set(groups)!=set(emap):raise ValueError("missing/extra parent groups")
 iq=read_csv_strict(parent/"iq_context.csv");imap={}
 for r in iq:
  key=event_key(r)
  if key in imap:raise ValueError("duplicate IQ identity")
  if r["physical_recording_id"]!=r["block_recording_id"]:raise ValueError("IQ recording mismatch")
  linked=r["linked_prns"].split(";") if r["linked_prns"] else []
  if linked!=[x["prn"] for x in groups.get(key,[])] or int(r["linked_pair_count"])!=len(linked):raise ValueError("IQ linked PRN mismatch")
  starts=[finite(x) for x in r["block_start_s"].split(";")];ends=[finite(x) for x in r["block_end_s"].split(";")];offset=r["sample_offset"].split(";");count=r["sample_count"].split(";")
  n=int(r["history_blocks"])
  if not(len(starts)==len(ends)==len(offset)==len(count)==n):raise ValueError("IQ history inventory mismatch")
  blocks=read_json_value(r["block_features_json"]);context=read_json_value(r["context_features_json"])
  if len(blocks)!=n or len(context)!=4 or any(len(x)!=4 for x in blocks):raise ValueError("IQ reducer vector mismatch")
  expected=np.mean(np.asarray(blocks,float),axis=0)
  if not np.allclose(expected,np.asarray(context,float),rtol=1e-12,atol=1e-12) or r["history_reducer"]!="arithmetic_mean_per_feature":raise ValueError("IQ reducer mismatch")
  imap[key]=r
 if set(imap)!=set(groups):raise ValueError("missing/extra IQ rows")
 eix={event_key(r):i for i,r in enumerate(events)};features=[];indices=[];ids=[]
 for r in prn:
  er=emap[event_key(r)]
  for f in ("scenario","physical_recording_id","event_id","target_index","availability_time_s","source_start_s","source_end_s","role","phase","label","valid","tracked_prn_count"):
   if r[f]!=er[f]:raise ValueError(f"PRN/event mismatch {f}")
  features.append(read_json_value(imap[event_key(r)]["context_features_json"]));indices.append(eix[event_key(r)]);ids.append(identity(r))
 return Parent(prn,events,iq,np.asarray(features,float),indices,ids)
def read_json_value(text):return json.loads(text,parse_constant=_bad_constant)
def verify_parent_binding(parent,repo):
 parent=Path(parent);repo=Path(repo);manifest=read_json_strict(parent/"hashes.json");expected=manifest.get("files")
 if not isinstance(expected,dict):raise ValueError("parent manifest malformed")
 actual=sorted(str(p.relative_to(parent)) for p in parent.rglob("*") if p.is_file() and p.name!="hashes.json")
 if set(actual)!=set(expected):raise ValueError("parent inventory mismatch")
 if any(sha256_file(parent/x)!=expected[x] for x in actual):raise ValueError("parent hash mismatch")
 resolved=subprocess.run(["git","-C",str(repo),"rev-parse",PARENT_COMMIT],check=True,text=True,stdout=subprocess.PIPE).stdout.strip()
 if resolved!=PARENT_COMMIT:raise ValueError("parent commit mismatch")
 committed=subprocess.run(["git","-C",str(repo),"show",f"{PARENT_COMMIT}:artifacts/nc_topi_stage0/hashes.json"],check=True,text=True,stdout=subprocess.PIPE).stdout
 if json.loads(committed)!=manifest:raise ValueError("parent manifest differs from commit")
 prov=read_json_strict(parent/"provenance.json")
 if prov.get("source_commit")!=PARENT_SOURCE or prov.get("execution_code_commit")!=PARENT_SOURCE:raise ValueError("parent source mismatch")
 return {"ok":True,"parent_artifact_commit":resolved,"parent_generation_source_commit":PARENT_SOURCE,"inventory_count":len(actual)+1,"manifest_sha256":sha256_file(parent/"hashes.json"),"consumed_file_hashes":{x:sha256_file(parent/x) for x in ("per_epoch_scores.csv","iq_context.csv")},"current_head_not_required":True}

def synthetic_parent_binding(parent):
 parent=Path(parent);manifest=read_json_strict(parent/"hashes.json");expected=manifest.get("files",{});actual=sorted(str(p.relative_to(parent)) for p in parent.rglob("*") if p.is_file() and p.name!="hashes.json")
 if not isinstance(expected,dict) or set(expected)!=set(actual) or any(sha256_file(parent/x)!=expected[x] for x in actual):raise ValueError("synthetic parent inventory/hash mismatch")
 return {"ok":True,"synthetic_fixture":True,"parent_artifact_commit":"SYNTHETIC_TEST_ONLY","parent_generation_source_commit":"SYNTHETIC_TEST_ONLY","inventory_count":len(actual)+1,"manifest_sha256":sha256_file(parent/"hashes.json"),"consumed_file_hashes":{x:sha256_file(parent/x) for x in ("per_epoch_scores.csv","iq_context.csv")},"current_head_not_required":True}

def fit_model(X,y):
 X=np.asarray(X,float);y=np.asarray(y,float);median=np.median(X,axis=0);q75,q25=np.percentile(X,[75,25],axis=0);iqr=q75-q25;fallback=iqr<=1e-8;iqr[fallback]=1.;m=HuberRegressor(epsilon=1.35,alpha=1e-4,max_iter=1000).fit((X-median)/iqr,np.log(np.maximum(y,EPS)))
 return {"median":median,"iqr":iqr,"fallback":fallback,"coef":m.coef_,"intercept":float(m.intercept_),"scale":float(m.scale_)}
def predict(model,X):return np.exp(np.clip((np.asarray(X)-model["median"])/model["iqr"]@model["coef"]+model["intercept"],-745,709))
def model_seal(target,model,X,y,ids,metadata):
 content={"schema":"TargetConditioner.v2","target":target,"feature_schema":["log_power","log_noise_floor_scale","spectral_flatness","lag1_autocorr_magnitude"],"rows":len(ids),"identity_digest_sha256":digest_json(ids),"row_provenance_digest_sha256":digest_json(metadata),"feature_digest_sha256":digest_array(X),"target_digest_sha256":digest_array(y),"median":model["median"].tolist(),"iqr":model["iqr"].tolist(),"iqr_fallback":model["fallback"].tolist(),"coef":model["coef"].tolist(),"intercept":model["intercept"],"model_scale":model["scale"],"epsilon":1.35,"alpha":.0001,"max_iter":1000,"target_transform":"log(max(target, 1e-12))","prediction_clip":[-745.,709.],"fit_predicate":"cleanStatic/normal_train/normal/label0/valid","attack_fit":False}
 return content,digest_json(content)
def roles(parent):return {event_key(r):r["role"] for r in parent.events}
def indices(parent,scenario,role=None):
 er=roles(parent);return [i for i,r in enumerate(parent.prn) if r["scenario"]==scenario and parse_bool(r["valid"]) and (role is None or er[event_key(r)]==role)]
def metadata(parent,ix,role):return [{"identity":parent.ids[i],"scenario":"cleanStatic","role":role,"phase":"normal","label":0,"valid":True} for i in ix]

def sustained_delay(rows,method,threshold,onset):
 earliest=math.inf
 for rec in sorted({r["physical_recording_id"] for r in rows}):
  rr=sorted([r for r in rows if r["physical_recording_id"]==rec],key=lambda r:finite(r["availability_time_s"]));times=[finite(r["availability_time_s"]) for r in rr]
  if len(times)!=len(set(times)):raise ValueError("duplicate recording/time")
  run=0;prev=None
  for r,t in zip(rr,times):
   post=r["phase"]=="post";alarm=finite(r[method])>threshold;cont=prev is not None and math.isclose(t-prev,.5,abs_tol=1e-8)
   run=(run+1 if cont else 1) if post and alarm else 0;prev=t
   if run>=3:earliest=min(earliest,t);break
 return None if not math.isfinite(earliest) else earliest-onset

def metric_rows(events,thresholds):
 out=[]
 for scenario in ("cleanStatic","cleanDynamic",*ATTACKS):
  allrows=[r for r in events if r["scenario"]==scenario and parse_bool(r["valid"])]
  for m in METHODS:
   th=thresholds[m]["value"]
   normal=([r for r in allrows if r["role"]=="normal_holdout"] if scenario=="cleanStatic" else allrows if scenario=="cleanDynamic" else [r for r in allrows if r["phase"]=="stable_pre"])
   base={"scenario":scenario,"method":m,"threshold":th,"threshold_comparison":"strict >","normal_fpr":float(np.mean([finite(r[m])>th for r in normal])) if normal else None,"normal_fpr_reason":None if normal else "no eligible normal events"}
   if scenario not in ATTACKS:out.append({**base,"roc_auc":None,"roc_auc_reason":"single-class normal diagnostic","pr_auc":None,"pr_auc_reason":"single-class normal diagnostic","standardized_pauc_max_fpr_0.05":None,"pauc_reason":"single-class normal diagnostic","post_detection_rate":None,"post_detection_reason":"not attack","three_consecutive_alarm_delay_s":None,"delay_reason":"not attack","persistent_alarm_ratio":None,"persistent_reason":"not attack"});continue
   eligible=[r for r in allrows if r["phase"] in ("stable_pre","post")];y=np.asarray([int(r["label"]) for r in eligible]);v=np.asarray([finite(r[m]) for r in eligible]);reason=None
   if set(y)!={0,1}:roc=pr=pa=None;reason="class-deficient eligible events"
   else:roc=float(roc_auc_score(y,v));pr=float(average_precision_score(y,v));pa=float(roc_auc_score(y,v,max_fpr=.05))
   post=[r for r in allrows if r["phase"]=="post"];persistent=[r for r in post if finite(r["source_start_s"])>=ONSETS[scenario]+40];delay=sustained_delay(eligible,m,th,ONSETS[scenario])
   out.append({**base,"roc_auc":roc,"roc_auc_reason":reason,"pr_auc":pr,"pr_auc_reason":reason,"standardized_pauc_max_fpr_0.05":pa,"pauc_reason":reason,"post_detection_rate":float(np.mean([finite(r[m])>th for r in post])) if post else None,"post_detection_reason":None if post else "no post events","three_consecutive_alarm_delay_s":delay,"delay_reason":None if delay is not None else "censored: no 3-consecutive 0.5s alarm","persistent_alarm_ratio":float(np.mean([finite(r[m])>th for r in persistent])) if persistent else None,"persistent_reason":None if persistent else "no persistent events"})
 return out

def fast_pauc(labels,scores,max_fpr=.05):
 y=np.asarray(labels,dtype=np.int8);score=np.asarray(scores,float);order=np.argsort(score,kind="mergesort")[::-1];z=y[order];ss=score[order];ix=np.r_[np.where(np.diff(ss))[0],len(y)-1];tp=np.cumsum(z,dtype=float)[ix];fp=1+ix-tp;tpr=np.r_[0.,tp/tp[-1]];fpr=np.r_[0.,fp/fp[-1]];stop=np.searchsorted(fpr,max_fpr,"right");interp=np.interp(max_fpr,fpr[stop-1:stop+1],tpr[stop-1:stop+1]);partial=float(np.trapezoid(np.r_[tpr[:stop],interp],np.r_[fpr[:stop],max_fpr]));minimum=.5*max_fpr**2;return .5*(1+(partial-minimum)/(max_fpr-minimum))

def bootstrap(labels,a,b,recs,times,reps=2000,seed=20260803):
 y=np.asarray(labels,int);a=np.asarray(a,float);b=np.asarray(b,float);t=np.asarray(times,float);r=np.asarray(recs,str);point=fast_pauc(y,a)-fast_pauc(y,b) if set(y)=={0,1} else None;pools={0:[],1:[]}
 if set(y)=={0,1}:
  for rec in sorted(set(r)):
   ix=np.flatnonzero(r==rec);order=ix[np.argsort(t[ix],kind="mergesort")]
   if len(np.unique(t[order]))!=len(order):raise ValueError("duplicate bootstrap time")
   bounds=[0]+[i for i in range(1,len(order)) if y[order[i]]!=y[order[i-1]] or not np.isclose(t[order[i]]-t[order[i-1]],.5,rtol=0,atol=1e-8)]+[len(order)]
   for left,right in zip(bounds[:-1],bounds[1:]):
    for start in range(left,right-19,20):pools[int(y[order[start]])].append(order[start:start+20])
 audit={"resampling":"paired pAUC delta; recording/gap-safe complete nonoverlapping 10s blocks stratified by label","iid_fallback":False,"point_estimate_rows":len(y),"negative_blocks":len(pools[0]),"positive_blocks":len(pools[1]),"paired_indices":True,"max_fpr":.05,"reps_requested":reps,"seed":seed};count=sum(map(len,pools.values()))
 if set(y)!={0,1} or min(map(len,pools.values()))<2:
  reason="class-deficient eligible epochs" if set(y)!={0,1} else "too few complete blocks in one or both class strata";empty=np.asarray([],float);return {"available":False,"reason":reason,"point_estimate":point,"lower":None,"upper":None,"valid_reps":0,"reps_requested":reps,"complete_block_count":count,"block_epoch_count":20,"replicate_digest_sha256":hashlib.sha256(empty.tobytes()).hexdigest(),"iid_fallback":False,"audit":audit}
 rng=np.random.default_rng(seed);values=np.empty(reps,float)
 for j in range(reps):
  selected=np.concatenate([pools[lab][i] for lab in (0,1) for i in rng.integers(0,len(pools[lab]),len(pools[lab]))]);values[j]=fast_pauc(y[selected],a[selected])-fast_pauc(y[selected],b[selected])
 lo,hi=np.percentile(values,[2.5,97.5]);audit["valid_reps"]=reps;return {"available":True,"reason":None,"point_estimate":float(point),"lower":float(lo),"upper":float(hi),"valid_reps":reps,"reps_requested":reps,"complete_block_count":count,"block_epoch_count":20,"replicate_digest_sha256":hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),"iid_fallback":False,"audit":audit}

def profile_support(parent):
 iq={event_key(x):x for x in parent.iq};groups={}
 for x in parent.prn:groups.setdefault(event_key(x),[]).append(x)
 out=[]
 for e in parent.events:
  if e["scenario"]!="cleanDynamic":continue
  q=iq[event_key(e)];starts=[finite(x) for x in q["block_start_s"].split(";")];ends=[finite(x) for x in q["block_end_s"].split(";")];target=finite(e["source_start_s"])
  out.append({"event_id":e["event_id"],"effective_start_s":min(target-6.,target,*starts,*[finite(x["source_start_s"]) for x in groups[event_key(e)]]),"effective_end_s":max(finite(e["source_end_s"]),*ends,*[finite(x["source_end_s"]) for x in groups[event_key(e)]])})
 return support_split(out)
def support_split(events):
 ordered=sorted(events,key=lambda x:(x["effective_start_s"],x["effective_end_s"],x["event_id"]));n=len(ordered);starts=np.asarray([x["effective_start_s"] for x in ordered]);ends=np.asarray([x["effective_end_s"] for x in ordered]);prefix=np.maximum.accumulate(ends) if n else np.asarray([])
 def first(left,v):
  x=np.flatnonzero(starts[left:]>=v);return None if not len(x) else left+int(x[0])
 best=0;w=None
 for k in range(1,n//3+1):
  for te in range(k,n-2*k+1):
   cs=first(te,prefix[te-1]+10)
   if cs is None or cs+k>n:continue
   hs=first(cs+k,prefix[cs+k-1]+10)
   if hs is not None and hs+k<=n:best=k;w=(te-k,te,cs,cs+k,hs,hs+k);break
 evidence=None if w is None else {"train_indices":[w[0],w[1]],"calibration_indices":[w[2],w[3]],"holdout_indices":[w[4],w[5]],"train_effective_end_s":float(prefix[w[1]-1]),"calibration_effective_start_s":float(starts[w[2]]),"calibration_effective_end_s":float(prefix[w[3]-1]),"holdout_effective_start_s":float(starts[w[4]])}
 return {"schema":"gnss-doppler-lab.nc-topi-stage0b.profile-d-support.v1","status":"INSUFFICIENT_NORMAL_SUPPORT","fit_profile_d":False,"calibrate_profile_d":False,"report_performance":False,"random_split":False,"chronological":True,"minimum_gap_seconds":10.,"b0_history_windows":12,"b0_history_seconds":6.,"includes_iq_history":True,"minimum_counts":{"normal_train":50,"normal_calibration":101,"normal_holdout":50},"best_counts":{"normal_train":best,"normal_calibration":best,"normal_holdout":best},"best_support_evidence":evidence,"candidate_events":n,"reason":f"best chronological effective-support split {best}/{best}/{best} is below 50/101/50"}

def compare_float(a,b):
 try:return math.isclose(finite(a),finite(b),rel_tol=1e-12,abs_tol=1e-12)
 except:return False
def compare_rows(expected,stored,keys,name,errors):
 if len(expected)!=len(stored):errors.append(f"{name} row inventory mismatch");return
 for i,(a,b) in enumerate(zip(expected,stored)):
  for k in keys:
   if a.get(k)!=b.get(k):errors.append(f"{name} parent field mismatch row {i}: {k}");return

def aggregate_from_predictions(data,pred,bounds):
 cal=indices(data,"cleanStatic","normal_calibration");raw={t:np.asarray([finite(r[c]) for r in data.prn]) for t,c in TARGETS.items()};clamped={t:np.clip(pred[t],bounds[t]["primary"]["lower"],bounds[t]["primary"]["upper"]) for t in TARGETS};scores={"B0":raw["B0"],"TOPI":raw["TOPI"],"NC_TOPI_original":np.asarray([finite(r["NC_TOPI"]) for r in data.prn]),"IQ_LOW_ONLY":-np.log(np.maximum(pred["TOPI"],EPS)),"IQ_OOD_ONLY":empirical_ood(pred["TOPI"][cal],pred["TOPI"]),"NC_TOPI_clamped":raw["TOPI"]/np.maximum(clamped["TOPI"],EPS),"NC_B0_clamped":raw["B0"]/np.maximum(clamped["B0"],EPS),"NC_total_clamped":raw["total"]/np.maximum(clamped["total"],EPS)};groups={}
 for i,r in enumerate(data.prn):groups.setdefault(event_key(r),[]).append(i)
 out=[]
 for e in data.events:
  row=dict(e);ix=groups[event_key(e)]
  for t in TARGETS:row[f"predicted_{t}_scale"]=float(np.median(pred[t][ix]));row[f"clamped_{t}_scale"]=float(np.median(clamped[t][ix]))
  for m in METHODS:row[m]=float(np.median(scores[m][ix]))
  row["common_iq_scale_equal"]="True";out.append(row)
 return out

def make_bounds(pred,cal):
 variants={"primary":(.01,.99),"two_sided_q005_q995":(.005,.995),"lower_only_q1":(.01,None),"upper_only_q99":(None,.99),"no_clamp":(None,None)}
 return {t:{n:{"lower":None if q[0] is None else higher_quantile(pred[t][cal],q[0]),"upper":None if q[1] is None else higher_quantile(pred[t][cal],q[1])} for n,q in variants.items()} for t in TARGETS}

def clamp_variant_rows(data,pred,bounds):
 raw={t:np.asarray([finite(r[c]) for r in data.prn]) for t,c in TARGETS.items()};groups={}
 for i,r in enumerate(data.prn):groups.setdefault(event_key(r),[]).append(i)
 out=[]
 for variant in ("two_sided_q005_q995","lower_only_q1","upper_only_q99","no_clamp"):
  vals={}
  for t,m in (("TOPI","NC_TOPI_clamped"),("B0","NC_B0_clamped"),("total","NC_total_clamped")):
   b=bounds[t][variant];den=np.clip(pred[t],-np.inf if b["lower"] is None else b["lower"],np.inf if b["upper"] is None else b["upper"]);v=raw[t]/np.maximum(den,EPS);vals[m]={k:float(np.median(v[ix])) for k,ix in groups.items()}
  ckeys=[event_key(r) for r in data.events if r["scenario"]=="cleanStatic" and r["role"]=="normal_calibration" and parse_bool(r["valid"])];th={m:higher_quantile([v[k] for k in ckeys],.99) for m,v in vals.items()}
  for scenario in ("cleanStatic","cleanDynamic",*ATTACKS):
   es=[r for r in data.events if r["scenario"]==scenario and parse_bool(r["valid"])]
   for method,v in vals.items():
    normal=[r for r in es if r["role"]=="normal_holdout"] if scenario=="cleanStatic" else es if scenario=="cleanDynamic" else [r for r in es if r["phase"]=="stable_pre"];fpr=float(np.mean([v[event_key(r)]>th[method] for r in normal])) if normal else None;pauc=None;reason="single-class normal diagnostic"
    if scenario in ATTACKS:
     eligible=[r for r in es if r["phase"] in ("stable_pre","post")];y=[int(r["label"]) for r in eligible]
     if set(y)=={0,1}:pauc=float(roc_auc_score(y,[v[event_key(r)] for r in eligible],max_fpr=.05));reason=None
     else:reason="class-deficient eligible events"
    out.append({"variant":variant,"scenario":scenario,"method":method,"threshold_q99_higher":th[method],"stable_or_normal_fpr":fpr,"standardized_pauc_max_fpr_0.05":pauc,"pauc_reason":reason,"primary":False,"decision_eligible":False})
 return out

def _finite_domain(value, name, errors, lo=0., hi=1.):
    if isinstance(value,bool) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value) or not lo<=float(value)<=hi:
        errors.append(f"{name}: expected finite [{lo},{hi}]");return None
    return float(value)


def evaluate_decision(evidence: Mapping[str,object]) -> dict[str,object]:
    errors=[]
    try: pauc=evidence["pauc"];ci=evidence["paired_ci"];fpr=evidence["q99_fpr"];profile=evidence["profile_d"]
    except (KeyError,TypeError):
        return {"status":"INCONCLUSIVE","validation_errors":["mandatory evidence mappings missing"],"shortcut_triggers":{},"tangent_conditions":{},"shortcut_precedence":True,"stage0_decision_unchanged":True}
    p={}
    for scenario in ("DS7","DS8"):
      p[scenario]={}
      for method in METHODS:
        try:value=pauc[scenario][method]
        except (KeyError,TypeError):errors.append(f"pauc.{scenario}.{method}: missing");continue
        p[scenario][method]=_finite_domain(value,f"pauc.{scenario}.{method}",errors)
    parsed_ci={}
    for scenario in ("DS7","DS8"):
      parsed_ci[scenario]={}
      for comparator in ("IQ_LOW_ONLY","IQ_OOD_ONLY","NC_B0_clamped"):
       name=f"NC_TOPI_clamped_minus_{comparator}"
       try:item=ci[scenario][name]
       except (KeyError,TypeError): errors.append(f"paired_ci.{scenario}.{name}: missing");continue
       if not item.get("available"): errors.append(f"paired_ci.{scenario}.{name}: unavailable");continue
       lo=_finite_domain(item.get("lower"),f"paired_ci.{scenario}.{name}.lower",errors,-1,1)
       hi=_finite_domain(item.get("upper"),f"paired_ci.{scenario}.{name}.upper",errors,-1,1)
       if lo is not None and hi is not None and lo>hi:errors.append(f"paired_ci.{scenario}.{name}: reversed")
       else:parsed_ci[scenario][comparator]=(lo,hi)
    def pv(s,m): return p.get(s,{}).get(m)
    a_point=any(all(pv(s,m) is not None and pv(s,"NC_TOPI_clamped") is not None and pv(s,m)>=pv(s,"NC_TOPI_clamped") for s in ("DS7","DS8")) for m in ("IQ_LOW_ONLY","IQ_OOD_ONLY"))
    a_ci=any(parsed_ci.get(s,{}).get(m,(None,None))[1] is not None and parsed_ci[s][m][1]<=0 for s in ("DS7","DS8") for m in ("IQ_LOW_ONLY","IQ_OOD_ONLY"))
    b=all("NC_B0_clamped" in parsed_ci.get(s,{}) and parsed_ci[s]["NC_B0_clamped"][0]<=0<=parsed_ci[s]["NC_B0_clamped"][1] for s in ("DS7","DS8"))
    c=any(pv(s,"NC_TOPI_clamped") is not None and pv(s,"B0") is not None and pv(s,"NC_TOPI_original") is not None and
          pv(s,"NC_TOPI_clamped")-pv(s,"B0")<=0<pv(s,"NC_TOPI_original")-pv(s,"B0") for s in ("DS7","DS8"))
    d_parts=[]
    overlaps=evidence.get("alarm_overlap",{})
    for s in ("DS7","DS8"):
      vals=(pv(s,"NC_TOPI_original"),pv(s,"B0"),pv(s,"IQ_LOW_ONLY"));over=_finite_domain(overlaps.get(s),f"alarm_overlap.{s}",errors)
      if None in vals or vals[0]-vals[1]<=0 or over is None:d_parts.append(False)
      else:d_parts.append((vals[2]-vals[1])/(vals[0]-vals[1])>=.5 and over>=.5)
    d=all(d_parts)
    clean_dynamic=_finite_domain(fpr.get("cleanDynamic"),"q99_fpr.cleanDynamic",errors)
    e=clean_dynamic is not None and clean_dynamic>=.5
    shortcut={"a_iq_point_or_ci":a_point or a_ci,"b_nc_vs_ncb0_statistically_indistinguishable":b,
              "c_clamp_reverses_positive_original_gain":c,"d_scale_only_gain_and_alarm_overlap":d,
              "e_clean_dynamic_profile_s_fpr":e}
    # A fully evaluable true shortcut has precedence even if unrelated evidence is missing.
    if any(shortcut.values()): return {"status":"IQ_SHORTCUT_DOMINATED","validation_errors":errors,"shortcut_triggers":shortcut,"tangent_conditions":{},"shortcut_precedence":True,"stage0_decision_unchanged":True}
    stable=fpr.get("stable_pre",{})
    hold=_finite_domain(fpr.get("cleanStatic_holdout"),"q99_fpr.cleanStatic_holdout",errors)
    stable_values={s:_finite_domain(stable.get(s),f"q99_fpr.stable_pre.{s}",errors) for s in ATTACKS}
    tangent={
      "t1_point_over_ncb0":all(pv(s,"NC_TOPI_clamped") is not None and pv(s,"NC_B0_clamped") is not None and pv(s,"NC_TOPI_clamped")>pv(s,"NC_B0_clamped") for s in ("DS7","DS8")),
      "t2_point_over_iq_low":all(pv(s,"NC_TOPI_clamped") is not None and pv(s,"IQ_LOW_ONLY") is not None and pv(s,"NC_TOPI_clamped")>pv(s,"IQ_LOW_ONLY") for s in ("DS7","DS8")),
      "t3_point_over_iq_ood":all(pv(s,"NC_TOPI_clamped") is not None and pv(s,"IQ_OOD_ONLY") is not None and pv(s,"NC_TOPI_clamped")>pv(s,"IQ_OOD_ONLY") for s in ("DS7","DS8")),
      "t4_at_least_one_positive_ncb0_ci":any(parsed_ci.get(s,{}).get("NC_B0_clamped",(None,None))[0] is not None and parsed_ci[s]["NC_B0_clamped"][0]>0 for s in ("DS7","DS8")),
      "t5_positive_clamped_nc_minus_b0_points":all(pv(s,"NC_TOPI_clamped") is not None and pv(s,"B0") is not None and pv(s,"NC_TOPI_clamped")>pv(s,"B0") for s in ("DS7","DS8")),
      "t6_cleanstatic_holdout":hold is not None and hold<=.02,
      "t7_all_stable_pre":all(v is not None and v<.05 for v in stable_values.values()),
      "profile_d_conditional_gate":profile.get("status")=="INSUFFICIENT_NORMAL_SUPPORT" or (profile.get("status")=="AVAILABLE" and _finite_domain(profile.get("holdout_fpr"),"profile_d.holdout_fpr",errors) is not None and profile["holdout_fpr"]<.05)}
    status="TANGENT_SUPPORTED" if not errors and all(tangent.values()) else "INCONCLUSIVE"
    return {"status":status,"validation_errors":errors,"shortcut_triggers":shortcut,"tangent_conditions":tangent,
            "shortcut_precedence":True,"stage0_decision_unchanged":True}



def rows_equal(expected,stored,identity_fields,name,errors):
 if len(expected)!=len(stored):errors.append(f"{name} inventory mismatch");return
 for i,(a,b) in enumerate(zip(expected,stored)):
  for key in set(a)|set(b):
   av=a.get(key);bv=b.get(key,"")
   if av is None:
    if bv not in (None,""):errors.append(f"{name} null mismatch row {i}/{key}");return
   elif isinstance(av,(float,int,np.floating,np.integer)) and not isinstance(av,bool):
    if not compare_float(av,bv):errors.append(f"{name} numeric mismatch row {i}/{key}");return
   elif str(av)!=str(bv):errors.append(f"{name} value mismatch row {i}/{key}");return

def diagnostics(events,thresholds,bounds,namespace="primary"):
 out=[]
 for scenario in ("cleanStatic","cleanDynamic",*ATTACKS):
  for phase in ("normal_train","normal_calibration","normal_holdout","stable_pre","post","persistent"):
   if scenario=="cleanStatic":subset=[r for r in events if r["scenario"]==scenario and r["role"]==phase and parse_bool(r["valid"])]
   elif scenario=="cleanDynamic":subset=[r for r in events if r["scenario"]==scenario and phase=="normal_holdout" and parse_bool(r["valid"])]
   elif phase=="persistent":subset=[r for r in events if r["scenario"]==scenario and r["phase"]=="post" and finite(r["source_start_s"])>=ONSETS[scenario]+40 and parse_bool(r["valid"])]
   else:subset=[r for r in events if r["scenario"]==scenario and r["phase"]==phase and parse_bool(r["valid"])]
   if not subset:continue
   scale=np.asarray([finite(r["predicted_TOPI_scale"]) for r in subset]);topi=np.asarray([finite(r["TOPI"]) for r in subset]);iq=-np.log(np.maximum(scale,EPS));corr=float(np.corrcoef(topi,iq)[0,1]) if len(scale)>1 and np.std(scale)>0 and np.std(topi)>0 else None;iq_auc=None
   if scenario in ATTACKS:
    full=[r for r in events if r["scenario"]==scenario and parse_bool(r["valid"]) and r["phase"] in ("stable_pre","post")]
    if full and {int(r["label"]) for r in full}=={0,1}:iq_auc=float(roc_auc_score([int(r["label"]) for r in full],[finite(r["IQ_LOW_ONLY"]) for r in full]))
   oa=np.asarray([finite(r["NC_TOPI_original"])>thresholds["NC_TOPI_original"]["value"] for r in subset]);ia=np.asarray([finite(r["IQ_LOW_ONLY"])>thresholds["IQ_LOW_ONLY"]["value"] for r in subset]);overlap=float(np.sum(oa&ia)/np.sum(oa)) if np.sum(oa) else None;preserved=float(np.mean(np.sign([finite(r["NC_TOPI_clamped"])-finite(r["B0"]) for r in subset])==np.sign([finite(r["NC_TOPI_original"])-finite(r["B0"]) for r in subset])));lo=bounds["TOPI"]["primary"]["lower"];hi=bounds["TOPI"]["primary"]["upper"]
   out.append({"namespace":namespace,"scenario":scenario,"phase":phase,"event_count":len(subset),"predicted_scale_q1":higher_quantile(scale,.01),"predicted_scale_median":float(np.median(scale)),"predicted_scale_q99":higher_quantile(scale,.99),"lower_clamp_hit_ratio":float(np.mean(scale<lo)),"upper_clamp_hit_ratio":float(np.mean(scale>hi)),"iq_only_auc":iq_auc,"topi_vs_iq_pearson_correlation":corr,"original_nc_iq_low_alarm_overlap":overlap,"clamp_preserved_direction_ratio":preserved,"scale_below_clean_q1_highlight":bool(scenario in ("DS2","DS7","DS8") and higher_quantile(scale,.99)<lo)})
 return out

def plot_numeric(events,bounds):
 scenarios=("cleanStatic","cleanDynamic",*ATTACKS);selected=[r for r in events if parse_bool(r["valid"])];out={"schema":"gnss-doppler-lab.nc-topi-stage0b.plot-data.v1","common_mask":"valid frozen aggregate events","plots":{}}
 def add(n,x,m):out["plots"][n]={"numeric":x,"masks":m,"numeric_digest_sha256":digest_json(x),"mask_digest_sha256":digest_json(m)}
 scale=np.array([finite(r["predicted_TOPI_scale"]) for r in selected]);edges=np.geomspace(max(scale.min(),EPS),scale.max()*(1+1e-12),41);counts={};masks={}
 for s in scenarios:
  rows=[r for r in selected if r["scenario"]==s];counts[s]=np.histogram([finite(r["predicted_TOPI_scale"]) for r in rows],edges)[0].tolist();masks[s]=[r["event_id"] for r in rows]
 add("predicted_scale_distribution",{"edges":edges.tolist(),"counts":counts},masks);med={};masks={}
 for s in ATTACKS:
  med[s]=[];masks[s]={}
  for phase in ("stable_pre","post"):
   rows=[r for r in selected if r["scenario"]==s and r["phase"]==phase];med[s].append(float(np.median([finite(r["predicted_TOPI_scale"]) for r in rows])));masks[s][phase]=[r["event_id"] for r in rows]
 add("stable_pre_post_scale",{"phase":["stable_pre","post"],"median":med},masks);add("original_vs_clamped_nc_topi",{"x":[finite(r["NC_TOPI_original"]) for r in selected],"y":[finite(r["NC_TOPI_clamped"]) for r in selected]},{"events":[r["event_id"] for r in selected]});pooled=[r for r in selected if r["scenario"] in ATTACKS and r["phase"] in ("stable_pre","post")];curves={}
 for m in ("B0","TOPI","IQ_LOW_ONLY","NC_B0_clamped","NC_TOPI_clamped"):
  y=np.array([int(r["label"]) for r in pooled]);order=np.argsort(-np.array([finite(r[m]) for r in pooled]),kind="mergesort");z=y[order];curves[m]={"fpr":(np.cumsum(1-z)/max(1,(1-z).sum())).tolist(),"tpr":(np.cumsum(z)/max(1,z.sum())).tolist()}
 add("roc_methods",curves,{"events":[r["event_id"] for r in pooled]});clean=[r for r in selected if r["scenario"] in ("cleanStatic","cleanDynamic")];values=np.array([finite(r["NC_TOPI_clamped"]) for r in clean]);edges=np.geomspace(max(values.min(),EPS),values.max()*(1+1e-12),51);counts={};masks={}
 for s in ("cleanStatic","cleanDynamic"):
  rows=[r for r in clean if r["scenario"]==s];counts[s]=np.histogram([finite(r["NC_TOPI_clamped"]) for r in rows],edges)[0].tolist();masks[s]=[r["event_id"] for r in rows]
 add("clean_normal_scores",{"edges":edges.tolist(),"counts":counts},masks);lo=bounds["TOPI"]["primary"]["lower"];hi=bounds["TOPI"]["primary"]["upper"];ratios={};masks={}
 for s in scenarios:
  rows=[r for r in selected if r["scenario"]==s];x=np.array([finite(r["predicted_TOPI_scale"]) for r in rows]);ratios[s]={"lower":float(np.mean(x<lo)),"upper":float(np.mean(x>hi)),"count":len(rows)};masks[s]=[r["event_id"] for r in rows]
 add("clamp_hit_ratio",{"clean_calibration_lower":lo,"clean_calibration_upper":hi,"ratios":ratios},masks);timeline={};masks={}
 for s in ("DS2","DS7","DS8"):
  rows=sorted([r for r in selected if r["scenario"]==s],key=lambda r:finite(r["availability_time_s"]));timeline[s]={"time":[finite(r["availability_time_s"]) for r in rows],"scale":[finite(r["predicted_TOPI_scale"]) for r in rows]};masks[s]=[r["event_id"] for r in rows]
 add("shortcut_scale_timeline",timeline,masks);return out

def render_plot_payload(folder,payload):
 import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
 folder=Path(folder);folder.mkdir(exist_ok=True);out={}
 def save(n):
  p=folder/(n+".png");plt.tight_layout();plt.savefig(p,dpi=120,metadata={"Software":"gnss-doppler-lab-stage0b"});plt.close();out[n]=sha256_file(p)
 d=payload["plots"]["predicted_scale_distribution"]["numeric"];plt.figure()
 for n,y in d["counts"].items():plt.stairs(y,d["edges"],label=n)
 plt.xscale("log");plt.legend(fontsize=6);plt.title("Predicted TOPI scale by scenario");save("predicted_scale_distribution");d=payload["plots"]["stable_pre_post_scale"]["numeric"];plt.figure()
 for n,y in d["median"].items():plt.plot([0,1],y,marker="o",label=n)
 plt.xticks([0,1],["stable-pre","post"]);plt.legend();save("stable_pre_post_scale");d=payload["plots"]["original_vs_clamped_nc_topi"]["numeric"];plt.figure();plt.scatter(d["x"],d["y"],s=2,alpha=.2);plt.xscale("symlog");plt.yscale("symlog");plt.xlabel("original");plt.ylabel("clamped");save("original_vs_clamped_nc_topi");d=payload["plots"]["roc_methods"]["numeric"];plt.figure()
 for n,c in d.items():plt.plot(c["fpr"],c["tpr"],label=n)
 plt.xlim(0,.2);plt.legend(fontsize=7);save("roc_methods");d=payload["plots"]["clean_normal_scores"]["numeric"];plt.figure()
 for n,y in d["counts"].items():plt.stairs(y,d["edges"],label=n)
 plt.xscale("log");plt.legend();save("clean_normal_scores");d=payload["plots"]["clamp_hit_ratio"]["numeric"];names=list(d["ratios"]);lo=[d["ratios"][x]["lower"] for x in names];hi=[d["ratios"][x]["upper"] for x in names];z=np.arange(len(names));plt.figure();plt.bar(z,lo);plt.bar(z,hi,bottom=lo);plt.xticks(z,names,rotation=30);save("clamp_hit_ratio");d=payload["plots"]["shortcut_scale_timeline"]["numeric"];plt.figure()
 for n,c in d.items():plt.plot(c["time"],c["scale"],label=n)
 plt.yscale("log");plt.legend();save("shortcut_scale_timeline");return out

def semantic_verification_report(errors,source_digests,checks=None):
 return {"schema":"gnss-doppler-lab.nc-topi-stage0b.verification.v1","ok":not errors,"status":"VERIFIED" if not errors else "REJECTED","checks":checks or ["standalone_parent_binding","independent_huber_refit","exact_identity_and_metrics","exact_bootstrap_and_diagnostics","decision_readme_plots_provenance"],"errors":list(errors),"source_digests":dict(sorted(source_digests.items())),"verifier_independent":True}

def semantic_verify(root,parent,repo):
 root=Path(root);errors=[];config=read_json_strict(root/"config.json");synthetic=config.get("schema")=="gnss-doppler-lab.nc-topi-stage0b-audit.synthetic-test.v1";binding=synthetic_parent_binding(parent) if synthetic else verify_parent_binding(parent,repo);data=load_parent(parent)
 stored_binding=read_json_strict(root/"parent_inventory.json")
 if stored_binding!=binding:errors.append("parent inventory binding mismatch")
 prn=read_csv_strict(root/"per_prn_scores.csv");events=read_csv_strict(root/"event_scores.csv")
 compare_rows(data.prn,prn,IDENTITY_PRN,"PRN",errors);compare_rows(data.events,events,IDENTITY_EVENT,"event",errors)
 train=indices(data,"cleanStatic","normal_train");cal=indices(data,"cleanStatic","normal_calibration");hold=indices(data,"cleanStatic","normal_holdout")
 if not synthetic and (len(train)!=6074 or len(cal)!=1628):errors.append("frozen PRN role split mismatch")
 original_model=fit_model(data.features[train],np.asarray([finite(data.prn[i]["TOPI"]) for i in train]));original_uncapped=predict(original_model,data.features);original_cap=higher_quantile(original_uncapped[cal],.995);original_refit=np.minimum(original_uncapped,original_cap);frozen_nc=np.asarray([finite(r["NC_TOPI"]) for r in data.prn]);topi_all=np.asarray([finite(r["TOPI"]) for r in data.prn]);reconstructed=topi_all/np.maximum(original_refit,EPS);absolute=np.abs(reconstructed-frozen_nc);relative=absolute/np.maximum(np.abs(frozen_nc),EPS);stored_refit=read_json_strict(root/"refit_equivalence.json")
 expected_refit={"rows":len(data.prn),"train_rows":len(train),"calibration_rows":len(cal),"q995_upper_cap":original_cap,"max_absolute_error":float(absolute.max()),"max_relative_error":float(relative.max())}
 for k,v in expected_refit.items():
  if isinstance(v,float) and not compare_float(v,stored_refit.get(k)) or not isinstance(v,float) and stored_refit.get(k)!=v:errors.append(f"original reconstruction report mismatch {k}")
 if not np.allclose(reconstructed,frozen_nc,rtol=1e-12,atol=1e-12):errors.append("standalone original reconstruction mismatch")
 implementation=np.asarray([finite(r["original_implementation_denominator"]) for r in prn]);ordinary=frozen_nc!=0;both=(topi_all==0)&(frozen_nc==0)
 if np.any((frozen_nc==0)&(topi_all!=0)) or not np.allclose(implementation,original_refit,rtol=1e-12,atol=1e-12) or (np.any(ordinary) and not np.allclose(topi_all[ordinary]/frozen_nc[ordinary],implementation[ordinary],rtol=1e-12,atol=1e-12)) or (np.any(both) and not np.allclose(implementation[both],original_refit[both],rtol=1e-12,atol=1e-12)):errors.append("implementation/refit effective-scale boundary mismatch")
 models={};pred={};fit=read_json_strict(root/"fit_audit.json")
 for target,column in TARGETS.items():
  y=np.asarray([finite(data.prn[i][column]) for i in train]);m=fit_model(data.features[train],y);models[target]=m;pred[target]=predict(m,data.features);content,seal=model_seal(target,m,data.features[train],y,[data.ids[i] for i in train],metadata(data,train,"normal_train"));item=fit.get("models",{}).get(target,{})
  if item.get("seal")!=seal or any(item.get(k)!=v for k,v in content.items()):errors.append(f"independent conditioner refit mismatch {target}")
 if fit.get("attack_fit") is not False or fit.get("train_rows")!=len(train) or fit.get("calibration_rows")!=len(cal) or fit.get("iq_inventory_digest_sha256")!=digest_json([[r[k] for k in IQ_FIELDS] for r in data.iq]):errors.append("fit audit/provenance mismatch")
 clean_seal=digest_json({"models":{t:model_seal(t,models[t],data.features[train],np.asarray([finite(data.prn[j][TARGETS[t]]) for j in train]),[data.ids[j] for j in train],metadata(data,train,"normal_train"))[1] for t in TARGETS},"train":digest_json([data.ids[j] for j in train]),"calibration":digest_json([data.ids[j] for j in cal]),"attack_fit":False})
 if fit.get("clean_state_digest_before_attack_transform")!=clean_seal:errors.append("clean-state fit provenance mismatch")
 expected_iq_checks={"event_common_scale_all":True,"events":len(events),"formula":"add-one two-sided <=/>=","clean_state_digest_before_attack_transform":clean_seal}
 if read_json_strict(root/"diagnostics/iq_scale_checks.json")!=expected_iq_checks:errors.append("IQ scale diagnostic mismatch")
 bounds=read_json_strict(root/"scale_bounds.json");variants={"primary":(.01,.99),"two_sided_q005_q995":(.005,.995),"lower_only_q1":(.01,None),"upper_only_q99":(None,.99),"no_clamp":(None,None)}
 for t in TARGETS:
  for name,(lq,uq) in variants.items():
   expected=(None if lq is None else higher_quantile(pred[t][cal],lq),None if uq is None else higher_quantile(pred[t][cal],uq));stored=bounds[t][name]
   if any((x is None)!=(y is None) or (x is not None and not compare_float(x,y)) for x,y in zip(expected,(stored.get("lower"),stored.get("upper")))):errors.append(f"clamp bounds mismatch {t}/{name}")
 raw={t:np.asarray([finite(r[c]) for r in data.prn]) for t,c in TARGETS.items()};clamped={t:np.clip(pred[t],bounds[t]["primary"]["lower"],bounds[t]["primary"]["upper"]) for t in TARGETS};ood=empirical_ood(pred["TOPI"][cal],pred["TOPI"])
 expected={"B0":raw["B0"],"TOPI":raw["TOPI"],"NC_TOPI_original":np.asarray([finite(r["NC_TOPI"]) for r in data.prn]),"IQ_LOW_ONLY":-np.log(np.maximum(pred["TOPI"],EPS)),"IQ_OOD_ONLY":ood,"NC_TOPI_clamped":raw["TOPI"]/np.maximum(clamped["TOPI"],EPS),"NC_B0_clamped":raw["B0"]/np.maximum(clamped["B0"],EPS),"NC_total_clamped":raw["total"]/np.maximum(clamped["total"],EPS)}
 for t in TARGETS:
  for col,values in ((f"predicted_{t}_scale",pred[t]),(f"clamped_{t}_scale",clamped[t])):
   if not np.allclose([finite(r[col]) for r in prn],values,rtol=1e-12,atol=1e-12):errors.append(f"PRN {col} mismatch")
 for m,v in expected.items():
  if not np.allclose([finite(r[m]) for r in prn],v,rtol=1e-12,atol=1e-12):errors.append(f"PRN score mismatch {m}")
 groups={}
 for i,r in enumerate(prn):groups.setdefault(event_key(r),[]).append(i)
 for er,source in zip(events,data.events):
  ix=groups[event_key(source)]
  for col in [*(f"predicted_{t}_scale" for t in TARGETS),*(f"clamped_{t}_scale" for t in TARGETS),*METHODS]:
   if not compare_float(er[col],np.median([finite(prn[i][col]) for i in ix])):errors.append(f"event median mismatch {event_key(er)}/{col}");break
  if er.get("common_iq_scale_equal") not in ("True","true"):errors.append(f"common event scale audit mismatch {event_key(er)}")
 thresholds=read_json_strict(root/"thresholds.json");cal_events=[r for r in events if r["scenario"]=="cleanStatic" and r["role"]=="normal_calibration" and parse_bool(r["valid"])]
 if not synthetic and len(cal_events)!=157:errors.append("frozen calibration event count mismatch")
 for m in METHODS:
  values=[finite(r[m]) for r in cal_events];th=higher_quantile(values,.99);item=thresholds.get(m,{})
  if not compare_float(th,item.get("value")) or item.get("rows")!=len(values) or item.get("score_digest_sha256")!=digest_array(values) or item.get("comparison")!="strict >":errors.append(f"threshold mismatch {m}")
 expected_metrics=metric_rows(events,thresholds);stored_metrics=read_csv_strict(root/"model_metrics.csv")
 if len(expected_metrics)!=len(stored_metrics):errors.append("metric inventory mismatch")
 else:
  for e,g in zip(expected_metrics,stored_metrics):
   if e["scenario"]!=g["scenario"] or e["method"]!=g["method"]:errors.append("metric order/inventory mismatch");break
   for k,v in e.items():
    if k in ("scenario","method"):continue
    sv=g.get(k,"")
    if v is None:
     if sv!="":errors.append(f"metric availability mismatch {e['scenario']}/{e['method']}/{k}")
    elif isinstance(v,(float,int)):
     if not compare_float(v,sv):errors.append(f"metric mismatch {e['scenario']}/{e['method']}/{k}")
    elif str(v)!=sv:errors.append(f"metric reason mismatch {e['scenario']}/{e['method']}/{k}")
 paired=read_json_strict(root/"paired_comparisons.json");stored_boot=paired.get("comparisons",[]);expected_keys=[(s,c) for s in ("DS7","DS8") for c in COMPARATORS]
 if [(x.get("scenario"),x.get("comparator")) for x in stored_boot]!=expected_keys:errors.append("paired comparison inventory/order mismatch")
 else:
  for item in stored_boot:
   eligible=[r for r in events if r["scenario"]==item["scenario"] and parse_bool(r["valid"]) and r["phase"] in ("stable_pre","post")];got=bootstrap([int(r["label"]) for r in eligible],[finite(r["NC_TOPI_clamped"]) for r in eligible],[finite(r[item["comparator"]]) for r in eligible],[r["physical_recording_id"] for r in eligible],[finite(r["availability_time_s"]) for r in eligible])
   for k,v in got.items():
    if isinstance(v,float) and v is not None:
     if not compare_float(v,item.get(k)):errors.append(f"bootstrap mismatch {item['scenario']}/{item['comparator']}/{k}")
    elif v!=item.get(k):errors.append(f"bootstrap mismatch {item['scenario']}/{item['comparator']}/{k}")
 profile=profile_support(data)
 if read_json_strict(root/"profile_support.json")!=profile:errors.append("Profile D exact support mismatch")
 expected_diag=diagnostics(events,thresholds,bounds);rows_equal(expected_diag,read_csv_strict(root/"scale_diagnostics.csv"),("namespace","scenario","phase"),"scale diagnostics",errors)
 expected_clamp=clamp_variant_rows(data,pred,bounds);rows_equal(expected_clamp,read_csv_strict(root/"diagnostics/clamp_variant_metrics.csv"),("variant","scenario","method"),"clamp variant diagnostics",errors)
 permutation=np.random.default_rng(0).permutation(len(train));shuffle_models={};shuffle_pred={};shuffle_fit=read_json_strict(root/"diagnostics/time_shuffle_fit_audit.json");expected_perm=hashlib.sha256(np.asarray(permutation,dtype=np.int64).tobytes()).hexdigest()
 if shuffle_fit.get("seed")!=0 or shuffle_fit.get("permutation_digest_sha256")!=expected_perm or shuffle_fit.get("target_only_permutation") is not True or shuffle_fit.get("same_permutation_all_targets") is not True:errors.append("time-shuffle permutation audit mismatch")
 for target,column in TARGETS.items():
  y=np.asarray([finite(data.prn[i][column]) for i in train]);m=fit_model(data.features[train],y[permutation]);shuffle_models[target]=m;shuffle_pred[target]=predict(m,data.features);content,seal=model_seal(target,m,data.features[train],y[permutation],[data.ids[i] for i in train],metadata(data,train,"normal_train"))
  if shuffle_fit.get("models",{}).get(target,{}).get("seal")!=seal:errors.append(f"time-shuffle refit mismatch {target}")
 shuffle_bounds=make_bounds(shuffle_pred,cal);shuffle_events=aggregate_from_predictions(data,shuffle_pred,shuffle_bounds);shuffle_thresholds={m:{"value":higher_quantile([finite(r[m]) for r in shuffle_events if r["scenario"]=="cleanStatic" and r["role"]=="normal_calibration" and parse_bool(r["valid"])],.99)} for m in METHODS}
 expected_shuffle_metrics=metric_rows(shuffle_events,shuffle_thresholds);rows_equal(expected_shuffle_metrics,read_csv_strict(root/"diagnostics/time_shuffle_metrics.csv"),("scenario","method"),"time-shuffle metrics",errors)
 expected_shuffle_diag=diagnostics(shuffle_events,shuffle_thresholds,shuffle_bounds,"time_shuffle");rows_equal(expected_shuffle_diag,read_csv_strict(root/"diagnostics/time_shuffle_scale_diagnostics.csv"),("namespace","scenario","phase"),"time-shuffle diagnostics",errors)
 lookup={(r["scenario"],r["method"]):r for r in expected_metrics};pauc={s:{m:lookup[(s,m)]["standardized_pauc_max_fpr_0.05"] for m in METHODS} for s in ("DS7","DS8")};ci={s:{} for s in ("DS7","DS8")}
 for item in stored_boot:ci[item["scenario"]][item["comparison"]]=item
 overlap={}
 for scenario in ("DS7","DS8"):
  eligible=[r for r in events if r["scenario"]==scenario and parse_bool(r["valid"]) and r["phase"] in ("stable_pre","post")];oa=np.asarray([finite(r["NC_TOPI_original"])>thresholds["NC_TOPI_original"]["value"] for r in eligible]);ia=np.asarray([finite(r["IQ_LOW_ONLY"])>thresholds["IQ_LOW_ONLY"]["value"] for r in eligible]);overlap[scenario]=float(np.sum(oa&ia)/np.sum(oa)) if np.sum(oa) else None
 evidence={"pauc":pauc,"paired_ci":ci,"alarm_overlap":overlap,"q99_fpr":{"cleanDynamic":lookup[("cleanDynamic","NC_TOPI_clamped")]["normal_fpr"],"cleanStatic_holdout":lookup[("cleanStatic","NC_TOPI_clamped")]["normal_fpr"],"stable_pre":{s:lookup[(s,"NC_TOPI_clamped")]["normal_fpr"] for s in ATTACKS}},"profile_d":profile};expected_decision=evaluate_decision(evidence);expected_decision["evidence"]=evidence;decision=read_json_strict(root/"decision.json")
 if decision!=expected_decision:errors.append("independent shortcut-first decision recomputation mismatch")
 config=read_json_strict(root/"config.json");second=read_json_strict(root/"diagnostics/second_peak_limitations.json")
 if second.get("stage0_c7") is not False or second.get("limitations")!=config["second_peak"]["limitations_exact"] or second.get("complex_synthesis_performed") is not False:errors.append("second-peak limitations/preserved Stage-0 result mismatch")
 prov=read_json_strict(root/"provenance.json");head=subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"],check=True,text=True,stdout=subprocess.PIPE).stdout.strip()
 required={"contract_commit_sha","execution_source_commit","parent_artifact_commit","parent_generation_source_commit","source_file_hashes","clean_worktree_at_execution","command","library_versions","attack_fit","raw_iq_opened","post_result_tuning","stage0_decision_preserved"}
 expected_parent="SYNTHETIC_TEST_ONLY" if synthetic else PARENT_COMMIT;expected_source="SYNTHETIC_TEST_ONLY" if synthetic else PARENT_SOURCE
 if not required.issubset(prov) or prov.get("parent_artifact_commit")!=expected_parent or prov.get("parent_generation_source_commit")!=expected_source or prov.get("attack_fit") is not False or prov.get("raw_iq_opened") is not False or prov.get("post_result_tuning") is not False or prov.get("stage0_decision_preserved") is not True or (not synthetic and prov.get("clean_worktree_at_execution") is not True):errors.append("production provenance invalid")
 if prov.get("execution_source_commit")!=head:errors.append("execution source HEAD mismatch")
 source_hashes={"runner":sha256_file(Path(repo)/"scripts"/("audit_"+"nc_topi_shortcut.py")),"verifier":sha256_file(__file__),"core":sha256_file(Path(repo)/"src/gnss_doppler_lab/nc_topi_stage0b.py"),"config":sha256_file(Path(repo)/"configs/nc_topi_stage0b_audit.json")}
 if prov.get("source_file_hashes")!=source_hashes:errors.append("source file hash provenance mismatch")
 parent_decision=read_json_strict(Path(parent)/"decision.json");recon=read_json_strict(root/"refit_equivalence.json")
 expected_readme=f"""# NC-TOPI Stage-0B shortcut and calibration audit

Generated deterministically from the immutable Stage-0 artifact at `{PARENT_COMMIT}`.
No raw-IQ file was opened and no B0 model was retrained.

- Stage-0 decision (preserved): **{parent_decision.get('status')}**
- Stage-0B status: **{decision['status']}**
- Original reconstruction rows: {recon['rows']}
- Reconstruction max absolute error: {recon['max_absolute_error']:.17g}
- Reconstruction max relative error: {recon['max_relative_error']:.17g}
- Profile D: **{profile['status']}**, best effective-support split {profile['best_counts']['normal_train']}/{profile['best_counts']['normal_calibration']}/{profile['best_counts']['normal_holdout']}
- Bootstrap: 12 paired comparisons, 2,000 requested replicates each, no IID fallback

This README reports the frozen grammar without post-result interpretation or tuning.
"""
 if (root/"README.md").read_text()!=expected_readme:errors.append("README deterministic regeneration mismatch")
 side=read_json_strict(root/"plot_data.json");expected_side=plot_numeric(events,bounds)
 if {k:{x:y for x,y in v.items() if x!="png_sha256"} for k,v in side.get("plots",{}).items()}!=expected_side["plots"] or side.get("schema")!=expected_side["schema"] or side.get("common_mask")!=expected_side["common_mask"]:errors.append("plot numeric/mask sidecar recomputation mismatch")
 with tempfile.TemporaryDirectory() as td:
  rendered=render_plot_payload(td,expected_side)
  for name,digest in rendered.items():
   png=root/"plots"/(name+".png")
   if not png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") or sha256_file(png)!=digest or side.get("plots",{}).get(name,{}).get("png_sha256")!=digest:errors.append(f"plot byte/content provenance mismatch {name}")
 return errors,source_hashes

def verify_artifact(root,parent=None,repo=None,prepare=False):
 root=Path(root);parent=Path(parent or ROOT/"artifacts/nc_topi_stage0");repo=Path(repo or ROOT);errors=[]
 try:
  config=read_json_strict(root/"config.json");errors.extend(verify_exact_inventory(root,config,prepare=prepare));synthetic=config.get("schema")=="gnss-doppler-lab.nc-topi-stage0b-audit.synthetic-test.v1"
  if not synthetic and (root/"config.json").read_bytes()!=(repo/"configs/nc_topi_stage0b_audit.json").read_bytes():errors.append("config copy is not byte-identical to committed contract")
 except Exception as e:return semantic_verification_report([f"config/inventory failure: {type(e).__name__}: {e}"],{})
 if not prepare:errors.extend(verify_hashes(root)["errors"])
 try:semantic,source=semantic_verify(root,parent,repo);errors.extend(semantic)
 except Exception as e:source={};errors.append(f"independent recomputation failed: {type(e).__name__}: {e}")
 report=semantic_verification_report(errors,source)
 if not prepare:
  try:
   stored=read_json_strict(root/"verification.json")
   if stored!=report:errors.append("verification.json does not exactly match standalone semantic report")
  except Exception as e:errors.append(f"verification report unreadable: {e}")
  report=semantic_verification_report(errors,source)
 return report

def parse_args(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("artifact",type=Path,nargs="?",default=ROOT/"artifacts/nc_topi_stage0b_audit");p.add_argument("--parent",type=Path,default=ROOT/"artifacts/nc_topi_stage0");p.add_argument("--prepare",action="store_true");p.add_argument("--report",type=Path);return p.parse_args(argv)
def main(argv=None):
 a=parse_args(argv);report=verify_artifact(a.artifact,parent=a.parent,repo=ROOT,prepare=a.prepare)
 if a.prepare and a.report:a.report.write_text(dump_json(report))
 print(dump_json(report),end="");return 0 if report["ok"] else 1
if __name__=="__main__":raise SystemExit(main())
