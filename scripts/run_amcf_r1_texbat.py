#!/usr/bin/env python3
"""Run the preregistered AMCF-R1 campaign from pinned nine-tap NPZ exports."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,os,platform,shutil,subprocess,sys,time
from collections import defaultdict,deque,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import scipy
import torch
from gnss_doppler_lab.amcf_r1 import (SIDE_INDICES,SEED_SIDES,FIXED_EXTRAS,TAP_NAMES,PromptGate,alarm_columns,assign_clean_role,attack_free_thresholds,batch_distributions,build_causal_windows,checkpoint_save,complex_summary,deterministic_training_masks,epoch_random_extras,evaluate_detector,expected_information_gain_batch,fit_prompt_gate,full_history_indices,history_vector,normalize_prompt,phase_destroy,phase_masks,robust_top2,student_t_nll,temporal_shuffle,train_model,verify_alarm_columns)
CANONICAL={
 "cleanStatic":(Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/exports/cleanStatic.npz"),"fcd1d378c28e79fe4a550b65fc1208cde3c8fb334db11406a07fed4d90fba237"),
 "DS1":(Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/exports/ds1.npz"),"b24d947c83890dbfa1c801bfbcb72e1fd192dd66509e927eb5afb8118902b072"),
 "DS2":(Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/exports/ds2.npz"),"dae0f245cbb107febd220c6de33b9a279a2bad356cb0ba772daf9418bc75d7c9"),
 "DS3":(Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/exports/ds3.npz"),"38eb5842dfec306d99bf0c5d61df6cffcb6faa25ed63721cafa8e3c3776f9b3e"),
 "DS7":(Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/ds7-sealed-input/exports/ds7.npz"),"d0e6da4e27d51e3e96abf2ef7786501124072f28667671e4e40da756eb35f3c8"),
 "DS8":(Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/cmte-a2-ds8-complex-d67f813/exports/ds8.npz"),"d1973fa150b7b4e7359df4827f36ce60289f206e9db11c1ac2bc1fd33a0df533")}
BASE_AMCF_PROVENANCE=ROOT/"artifacts/amcf_lite_texbat/provenance.json"
_base=json.loads(BASE_AMCF_PROVENANCE.read_text());_pins={n:(Path(v["path"]),v["sha256"]) for n,v in _base["input_npz"].items() if n in CANONICAL}
if _pins!=CANONICAL:raise RuntimeError("canonical pins disagree with base AMCF provenance")
CANONICAL=_pins
ONSETS={"DS1":100.,"DS2":100.,"DS3":100.,"DS7":110.,"DS8":110.};POLICY_SEEDS=(11,23,37)
DS4_STATUS="NA: canonical producer mismatch; excluded by preregistered design"
REQUIRED_OUTPUTS=("config.json","provenance.json","input_hashes.json","environment.json","window_qa.json","prompt_rejection_by_phase.csv","training_history.csv","model_audit.json","thresholds.json","metrics.csv","seed_metrics.csv","ablation_metrics.csv","query_policy_metrics.csv","query_path_histogram.csv","per_epoch","bootstrap_confidence_intervals.csv","inference_runtime.json","decision.json","README.md","plots","models","hashes.json")

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def jsonable(x):
 if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def write_json(p,x):Path(p).write_text(json.dumps(jsonable(x),indent=2,sort_keys=True)+"\n",newline="\n")
def write_csv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);rows=list(rows)
 if not rows:p.write_text("");return
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with p.open("w",newline="") as f:
  w=csv.DictWriter(f,fields,lineterminator="\n");w.writeheader();w.writerows([{k:json.dumps(v,separators=(",",":")) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
def write_hash_manifest(out):
 out=Path(out);write_json(out/"hashes.json",{"algorithm":"sha256","files":{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="hashes.json"}})
def verify_hash_manifest(out):
 out=Path(out);d=json.loads((out/"hashes.json").read_text())["files"];return sum((out/k).is_file() and sha256(out/k)==v for k,v in d.items())/max(1,len(d))
def attach_alarm_columns(rows,q99,q995,matched):return alarm_columns(rows,q99,q995,matched)
def recompute_alarm_fraction(rows,q99,q995,matched):return verify_alarm_columns(rows,q99,q995,matched)

def load_npz(name):
 p,expected=CANONICAL[name];actual=sha256(p)
 if actual!=expected:raise ValueError(f"{name} SHA-256 mismatch {actual}")
 with np.load(p,allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
 if d["complex_iq"].shape!=(len(d["time_s"]),9,2):raise ValueError("canonical shape mismatch")
 return d,{"scenario":name,"path":str(p),"sha256":actual,"rows":len(d["time_s"]),"shape":list(d["complex_iq"].shape),"fields":sorted(d)}
def load_b0_exact(path,scenario):
 """Read only timestamp and score_B0_Exact; all saved alarms are ignored."""
 with Path(path).open(newline="") as f:raw=list(csv.DictReader(f))
 out=[]
 for r in raw:
  if r.get("score_B0_Exact") not in (None,"") and r.get("decision_time_s") not in (None,""):
   out.append({"scenario":str(scenario),"decision_time_s":float(r["decision_time_s"]),"score":float(r["score_B0_Exact"])})
 return out

def phase(name,t):
 if name=="cleanStatic":return assign_clean_role(t) or "excluded"
 masks=phase_masks([t],ONSETS[name])
 for label in ("persistent","takeover","ramp","transition","stable_pre"):
  if masks[label][0]:return label
 return "excluded"
def histories(records):
 out=np.zeros((len(records),12,18),np.float32);groups=defaultdict(lambda:deque(maxlen=12))
 for i,r in enumerate(records):
  key=(r.recording_id,str(r.prn),r.role);prev=list(groups[key])
  if prev:out[i,-len(prev):]=np.stack(prev)
  groups[key].append(history_vector(r))
 return out
def choose_indices(records,role,maximum):
 # Only samples with all 12 previous same-role windows are scientifically valid.
 eligible=set(full_history_indices(records,12).tolist());idx=np.asarray([i for i,r in enumerate(records) if i in eligible and r.role==role],int)
 if maximum and len(idx)>maximum:idx=idx[np.linspace(0,len(idx)-1,int(maximum),dtype=int)]
 return idx

def histories_from_feature_matrix(records,features):
 out=np.zeros((len(records),12,18),np.float32);groups=defaultdict(lambda:deque(maxlen=12))
 for i,(r,f) in enumerate(zip(records,features)):
  key=(r.recording_id,str(r.prn),r.role);prev=list(groups[key])
  if prev:out[i,-len(prev):]=np.stack(prev)
  groups[key].append(np.asarray(np.r_[f[3,:7],f[5,:7],r.prompt_context[:2],r.valid_count,r.rejected_count],np.float32))
 return out
def destroyed_feature_matrix(records,iq,gate,seed):
 norm,valid=normalize_prompt(iq,gate);out=norm.copy();out[valid]=phase_destroy(norm[valid],seed);features=[]
 for r in records:
  good=r.source_indices[valid[r.source_indices]];features.append(complex_summary(out[good],len(good)/len(r.source_indices)))
 features=np.asarray(features,np.float32);return features,histories_from_feature_matrix(records,features)

def _distribution(values):
 vals=np.asarray(values,float)
 if not len(vals):return {"count":0,"min":None,"median":None,"q90":None,"max":None}
 return {"count":int(len(vals)),"min":float(np.min(vals)),"median":float(np.median(vals)),"q90":float(np.quantile(vals,.9)),"max":float(np.max(vals))}

def window_diagnostics(iq,time_s,prn,gate,records,*,recording_id,cn0=None,max_invariance_rows=2048):
 """Persist causal-window utilization and invariance diagnostics."""
 iq=np.asarray(iq);time_s=np.asarray(time_s);prn=np.asarray(prn);n=len(time_s)
 take=np.linspace(0,n-1,min(n,int(max_invariance_rows)),dtype=int) if n else np.asarray([],int)
 sample=iq[take]
 base,base_valid=normalize_prompt(sample,gate)
 z=sample[...,0]+1j*sample[...,1]
 def normalized_error(transformed):
  tiq=np.stack([transformed.real,transformed.imag],-1)
  got,valid=normalize_prompt(tiq,gate);mask=base_valid&valid
  return float(np.max(np.abs(got[mask]-base[mask]))) if mask.any() else None
 phase_error=normalized_error(z*np.exp(1j*.731))
 sign_error=normalized_error(-z)
 keys=[(str(r.prn),round(float(r.end_s),9),str(r.role)) for r in records]
 duplicates=len(keys)-len(set(keys))
 by_end=defaultdict(set)
 for r in records:by_end[round(float(r.end_s),9)].add(str(r.prn))
 reverse=np.arange(n-1,-1,-1)
 permuted,_=build_causal_windows(iq[reverse],time_s[reverse],prn[reverse],recording_id=recording_id,gate=gate,representation="complex",cn0=np.asarray(cn0)[reverse] if cn0 is not None else None)
 original={(str(r.prn),round(float(r.end_s),9),str(r.role)):r.features for r in records}
 replay={(str(r.prn),round(float(r.end_s),9),str(r.role)):r.features for r in permuted}
 if set(original)!=set(replay): permutation_error=None;permutation_key_match=False
 else:
  permutation_error=float(max([np.max(np.abs(original[k]-replay[k])) for k in original] or [0.]));permutation_key_match=True
 return {"global_phase_invariance_error_max":phase_error,"navigation_bit_sign_invariance_error_max":sign_error,"duplicate_epoch_prn_count":int(duplicates),"window_valid_sample_count":_distribution([r.valid_count for r in records]),"window_raw_sample_count":_distribution([r.raw_count for r in records]),"tracked_prn_count":_distribution([len(x) for x in by_end.values()]),"prn_input_permutation_invariance_error_max":permutation_error,"prn_input_permutation_key_match":permutation_key_match}

def _sync(model):
 if next(model.parameters()).is_cuda:torch.cuda.synchronize()
def _nll_batch(model,h,p,v,m,t,chunk):
 mu,sc=batch_distributions(model,h,p,v,m,t,chunk);y=v[np.arange(len(v)),np.asarray(t,int)];return np.mean(student_t_nll(y,mu,sc,model.df),axis=1)
def _score_policy(model,h,p,v,records,kind,k,model_seed,policy_seed,chunk):
 n=len(v);m=np.zeros((n,9),bool)
 # E and L are independently leave-one-out before either is revealed.
 m3=m.copy();m3[:,5]=True;m5=m.copy();m5[:,3]=True
 nll=[_nll_batch(model,h,p,v,m3,np.full(n,3),chunk),_nll_batch(model,h,p,v,m5,np.full(n,5),chunk)]
 m[:,3]=m[:,5]=True;paths=[[3,5] for _ in range(n)]
 for step in range(k-3):
  if kind=="fixed":targets=np.full(n,FIXED_EXTRAS[k][step],int)
  elif kind=="random":targets=np.asarray([epoch_random_extras(r.recording_id,r.end_s,r.prn,policy_seed,k-3)[step] for r in records],int)
  elif kind=="ig":
   targets=np.empty(n,int);groups=defaultdict(list)
   for i in range(n):groups[tuple(x for x in SIDE_INDICES if not m[i,x])].append(i)
   for cand,ids0 in groups.items():
    ids=np.asarray(ids0,int);seeds=[int.from_bytes(hashlib.sha256(json.dumps([records[i].recording_id,records[i].end_s,str(records[i].prn),model_seed,step],separators=(",",":")).encode()).digest()[:8],"little")%(2**63-1) for i in ids]
    gains=expected_information_gain_batch(model,h[ids],p[ids],v[ids],m[ids],cand,8,seeds,chunk);targets[ids]=np.asarray(cand)[np.argmax(gains,axis=1)]
  else:raise ValueError(kind)
  nll.append(_nll_batch(model,h,p,v,m,targets,chunk))
  for i,t in enumerate(targets):m[i,t]=True;paths[i].append(int(t))
 scores=np.sort(np.stack(nll,1),axis=1)[:,-2:].mean(1)
 return scores,paths

def _score_all9(model,h,p,v,chunk):
 n=len(v);rid=np.repeat(np.arange(n),8);targets=np.tile(np.asarray(SIDE_INDICES),n);m=np.ones((n*8,9),bool);m[:,4]=False;m[np.arange(n*8),targets]=False
 vals=v[rid];nll=_nll_batch(model,h[rid],p[rid],vals,m,targets,chunk).reshape(n,8);return np.sort(nll,axis=1)[:,-2:].mean(1),[list(SIDE_INDICES) for _ in range(n)]
def model_scores(name,records,hist,model,representation,model_seed,max_eval,gate=None,raw_iq=None,batch_size=256):
 eligible=full_history_indices(records,12)
 if max_eval and len(eligible)>max_eval:eligible=eligible[np.linspace(0,len(eligible)-1,int(max_eval),dtype=int)]
 specs=[]
 if representation=="magnitude":specs=[("magnitude K3","fixed",3,None),("magnitude all9","all9",9,None)]
 else:
  specs=[("complex K3","fixed",3,None),("complex fixed K5","fixed",5,None),("complex IG K5","ig",5,None),("complex fixed K7","fixed",7,None),("complex IG K7","ig",7,None),("complex all9","all9",9,None),("complex all9 phase-destroyed","destroy",9,None),("complex all9 temporal-shuffled","shuffle",9,None)]
  for ps in POLICY_SEEDS:
   specs += [(f"complex random K5 policy{ps}","random",5,ps),(f"complex random K7 policy{ps}","random",7,ps)]
 destroyed,destroyed_hist=destroyed_feature_matrix(records,raw_iq,gate,model_seed) if representation=="complex" else (None,None)
 rows=[]
 for a in range(0,len(eligible),int(batch_size)):
  ids=eligible[a:a+int(batch_size)];rr=[records[i] for i in ids];h=hist[ids];p=np.stack([r.prompt_context for r in rr]);base=np.stack([r.features for r in rr]).astype(np.float32)
  for label,kind,k,ps in specs:
   vv=destroyed[ids] if kind=="destroy" else base
   hh=np.stack([temporal_shuffle(x,model_seed,recording_id=r.recording_id,time_s=r.end_s,prn=r.prn) for x,r in zip(h,rr)]) if kind=="shuffle" else destroyed_hist[ids] if kind=="destroy" else h
   _sync(model);start=time.perf_counter()
   if kind in ("all9","destroy","shuffle"):scores,paths=_score_all9(model,hh,p,vv,batch_size*128)
   else:scores,paths=_score_policy(model,hh,p,vv,rr,kind,k,model_seed,ps,batch_size*128)
   _sync(model);elapsed=time.perf_counter()-start
   for r,score,path0 in zip(rr,scores,paths):rows.append({"scenario":name,"decision_time_s":r.end_s,"phase":phase(name,r.end_s),"prn":str(r.prn),"model":label,"model_seed":model_seed,"policy_seed":ps,"score":float(score),"query_path":path0,"inference_ms":elapsed*1000/len(rr),"queried_side_count":len(path0),"offline_replay_not_sdr_savings":True})
 return rows

def _group(rows,*keys):
 d=defaultdict(list)
 for r in rows:d[tuple(r[k] for k in keys)].append(r)
 return d
def aggregate_scored(items):
 out=[]
 for (s,t,m,seed,ps),rows in sorted(_group(items,"scenario","decision_time_s","model","model_seed","policy_seed").items(),key=str):
  out.append({"scenario":s,"decision_time_s":t,"phase":phase(s,t),"model":m,"model_seed":seed,"policy_seed":ps,"score":float(np.median([r["score"] for r in rows])),"tracked_prn_count":len(rows)})
 return out
def ensemble_rows(epochs):
 out=[]
 for (s,t,ph,m,ps),x in _group(epochs,"scenario","decision_time_s","phase","model","policy_seed").items():
  out.append({"scenario":s,"decision_time_s":t,"phase":ph,"model":m,"model_seed":"mean","policy_seed":ps,"score":float(np.mean([r["score"] for r in x])),"tracked_prn_count":int(round(np.mean([r["tracked_prn_count"] for r in x])))})
 return out
def primary_mean(epochs):
 return [dict(r,model="primary 3-seed mean complex IG K7") for r in ensemble_rows([r for r in epochs if r["model"]=="complex IG K7"])]
def common_rows(a,b):
 aa={round(float(r["decision_time_s"]),6):r for r in a};bb={round(float(r["decision_time_s"]),6):r for r in b};keys=sorted(set(aa)&set(bb));return [aa[k] for k in keys],[bb[k] for k in keys]
def threshold_for_target_fpr(scores,target):
 x=np.sort(np.asarray(scores,float));cands=np.r_[-np.inf,x,np.inf];rates=np.asarray([np.mean(x>q) for q in cands]);return float(cands[np.argmin(np.abs(rates-float(target)))])

def performance_rows(rows_by_model,bootstrap_reps,threshold_overrides=None):
 metrics=[];boots=[]
 for source,rows in rows_by_model.items():
  clean=[r for r in rows if r["scenario"]=="cleanStatic"];cal=[r for r in clean if r["phase"]=="calibration"]
  if not cal:continue
  th=attack_free_thresholds([r["score"] for r in cal],["calibration"]*len(cal),["cleanStatic"]*len(cal))
  if threshold_overrides and source in threshold_overrides: th=threshold_overrides[source]
  for scenario,part0 in _group(rows,"scenario").items():
   part=part0
   for operating_point,threshold_key in (("q99","q99"),("q995","q995")):
    m,b=evaluate_detector(part,th[threshold_key],scenario=scenario[0],bootstrap_reps=bootstrap_reps,seed=101,onset_s=ONSETS.get(scenario[0]));m.update({"model":source,"operating_point":operating_point,"calibration_q99":th["q99"],"calibration_q995":th["q995"],"calibration_count":th["count"]})
    if source.startswith("seed::"):
     m["model_seed"]=source.split("::model",1)[1].split("::",1)[0];m["policy_seed"]=source.rsplit("::policy",1)[1]
    metrics.append(m);boots.extend(dict(x,model=source,operating_point=operating_point) for x in b)
 return metrics,boots

def path_diagnostics(raw):
 out=[]
 for (scenario,model,ms,ps),rows in sorted(_group(raw,"scenario","model","model_seed","policy_seed").items(),key=str):
  paths=["-".join(map(str,r["query_path"])) for r in rows];count=Counter(paths);modal=max(count.values())/len(paths);taps=Counter(t for r in rows for t in r["query_path"]);probs=np.asarray(list(count.values()))/len(paths);phase_ent={}
  for (ph,),part in _group(rows,"phase").items():
   c=Counter("-".join(map(str,r["query_path"])) for r in part);q=np.asarray(list(c.values()))/len(part);phase_ent[ph]=float(-np.sum(q*np.log(q)))
  out.append({"scenario":scenario,"model":model,"model_seed":ms,"policy_seed":ps,"unique_ordered_paths":len(count),"modal_path":max(count,key=count.get),"modal_fraction":modal,"collapsed_ge_95pct":modal>=.95,"path_entropy":float(-np.sum(probs*np.log(probs))),"scenario_phase_entropy_json":json.dumps(phase_ent,sort_keys=True),"tap_frequency_json":json.dumps(taps,sort_keys=True),"mean_inference_ms":float(np.mean([r["inference_ms"] for r in rows])),"mean_queried_side_count":float(np.mean([r["queried_side_count"] for r in rows])),"correlation_fraction":float(np.mean([r["queried_side_count"] for r in rows])/8),"interpretation":"offline replay only; not measured SDR savings"})
 return out

def normalize_args(a):
 a.max_val_samples=a.max_val_samples or a.max_train_samples
 # Evaluation is never silently coupled to a training smoke limit.
 if a.max_eval_samples is not None and a.epochs!=1:raise ValueError("--max-eval-samples is smoke-only (requires --epochs 1); full campaigns never subsample evaluation")
 return a

def run(args):
 normalize_args(args);out=args.out.resolve()
 if out.exists():raise FileExistsError(out)
 stage=out.with_name(out.name+f".tmp-{os.getpid()}");stage.mkdir(parents=True);(stage/"models").mkdir();(stage/"plots").mkdir();(stage/"per_epoch").mkdir()
 wall0=time.perf_counter()
 try:
  names=[x.strip().upper() for x in args.scenarios.split(",") if x.strip()]
  if any(n not in ONSETS for n in names):raise ValueError("scenarios must be DS1,DS2,DS3,DS7,DS8")
  clean,ci=load_npz("cleanStatic");gate=fit_prompt_gate(clean["complex_iq"],clean["time_s"],args.prompt_quantile);datasets={"cleanStatic":clean};inputs={"cleanStatic":ci}
  for n in names:datasets[n],inputs[n]=load_npz(n)
  recs={};hists={};qa_scenarios=[];rejection=[]
  for name,d in datasets.items():
   for rep in ("complex","magnitude"):
    rr,qa=build_causal_windows(d["complex_iq"],d["time_s"],d["prn"],recording_id=name,gate=gate,representation=rep,cn0=d.get("cn0_db_hz"));recs[name,rep]=rr;hists[name,rep]=histories(rr)
    if qa["unique_valid_utilization"]<=.02:raise RuntimeError(f"{name}: unique raw utilization <=2%")
   z=d["complex_iq"][...,0]+1j*d["complex_iq"][...,1];valid=np.isfinite(abs(z[:,4]))&(abs(z[:,4])>=gate.min_prompt_magnitude)&(abs(z[:,4])>0);phase_rows=[]
   for ph in sorted(set(phase(name,t) for t in d["time_s"])):
    mask=np.asarray([phase(name,t)==ph for t in d["time_s"]]);wr=[r for r in recs[name,"complex"] if phase(name,r.end_s)==ph];tracked=[sum(r.end_s==end for r in wr) for end in sorted(set(r.end_s for r in wr))]
    row={"scenario":name,"phase":ph,"raw_rows":int(mask.sum()),"valid_rows":int(np.sum(mask&valid)),"rejected_rows":int(np.sum(mask&~valid)),"valid_rate":float(np.mean(valid[mask])) if mask.any() else None,"rejection_rate":float(np.mean(~valid[mask])) if mask.any() else None,"rejected_prn_count":int(len(np.unique(d["prn"][mask&~valid]))),"valid_prn_count":int(len(np.unique(d["prn"][mask&valid]))),"valid_window_count":len(wr),"window_valid_samples":int(sum(r.valid_count for r in wr)),"window_rejected_samples":int(sum(r.rejected_count for r in wr)),"tracked_N_min":min(tracked) if tracked else 0,"tracked_N_median":float(np.median(tracked)) if tracked else 0,"tracked_N_max":max(tracked) if tracked else 0};phase_rows.append(row);rejection.append(row)
   q0={k:v for k,v in qa.items() if k!="unique_used_source_indices"};q0.update({"scenario":name,"full_history_windows":int(len(full_history_indices(recs[name,"complex"])))})
   q0.update(window_diagnostics(d["complex_iq"],d["time_s"],d["prn"],gate,recs[name,"complex"],recording_id=name,cn0=d.get("cn0_db_hz")))
   if name=="DS1":
    vf=np.asarray([r.valid_count/max(1,r.raw_count) for r in recs[name,"complex"]]);scale=np.asarray([np.median(r.features[:,4]) for r in recs[name,"complex"]]);q0["scale_vs_missingness_diagnostic"]={"pearson_valid_fraction_vs_median_magnitude":float(np.corrcoef(vf,scale)[0,1]) if np.std(vf)>0 and np.std(scale)>0 else None,"interpretation":"diagnostic only; distinguishes amplitude scale from Prompt-gate missingness"}
   qa_scenarios.append({"summary":q0,"phase":phase_rows})
  seeds=[int(x) for x in args.seeds.split(",")];device="cuda" if torch.cuda.is_available() else "cpu";models={};audits={};training=[]
  for rep in ("complex","magnitude"):
   rr=recs["cleanStatic",rep];hh=hists["cleanStatic",rep];ti=choose_indices(rr,"train",args.max_train_samples);vi=choose_indices(rr,"validation",args.max_val_samples)
   if len(ti)<2 or len(vi)<2:raise ValueError("insufficient full-history chronological train/validation windows")
   cur=np.stack([r.features for r in rr]);pc=np.stack([r.prompt_context for r in rr])
   for seed in seeds:
    model,opt,h,a=train_model(cur[ti],hh[ti],cur[vi],hh[vi],train_prompt_context=pc[ti],val_prompt_context=pc[vi],seed=seed,hidden=args.hidden,epochs=args.epochs,patience=min(8,args.epochs),device=device);key=f"{rep}_seed{seed}";models[key]=model;audits[key]=a
    training.extend({"representation":rep,"seed":seed,**x} for x in h);checkpoint_save(stage/"models"/f"{key}.pt",model,opt,a)
  raw=[];score0=time.perf_counter()
  for name,d in datasets.items():
   for rep in ("complex","magnitude"):
    for seed in seeds:raw.extend(model_scores(name,recs[name,rep],hists[name,rep],models[f"{rep}_seed{seed}"],rep,seed,args.max_eval_samples,gate,d["complex_iq"],args.batch_size))
  scoring_s=time.perf_counter()-score0;epochs=aggregate_scored(raw);ensembles=ensemble_rows(epochs);primary=primary_mean(epochs)
  # B0 exact: calibration [340,410), all alarms ignored and recomputed.
  b0dir=ROOT/"artifacts/cmte_a2_texbat_epochfix/per_epoch";b0=[]
  for name in datasets:
   bp=b0dir/("cleanStatic_test.csv" if name=="cleanStatic" else f"{name}.csv")
   if bp.is_file():
    part=load_b0_exact(bp,name)
    for r in part:r["phase"]=phase(name,r["decision_time_s"]);r["model"]="B0 Exact";r["model_seed"]="NA";r["policy_seed"]=None
    b0.extend(part)
  b0cal=[r for r in b0 if r["scenario"]=="cleanStatic" and 340<=r["decision_time_s"]<410]
  if not b0cal:raise ValueError("B0 exact clean calibration [340,410) missing")
  b0th=attack_free_thresholds([r["score"] for r in b0cal],["calibration"]*len(b0cal),["cleanStatic"]*len(b0cal))
  pcal=[r for r in primary if r["scenario"]=="cleanStatic" and r["phase"]=="calibration"]
  pth=attack_free_thresholds([r["score"] for r in pcal],["calibration"]*len(pcal),["cleanStatic"]*len(pcal))
  pc,bc=common_rows([r for r in primary if r["scenario"]=="cleanStatic" and r["phase"]=="clean_test"],[r for r in b0 if r["scenario"]=="cleanStatic" and r["phase"]=="clean_test"])
  pfpr=np.mean([r["score"]>pth["q99"] for r in pc]);bfpr=np.mean([r["score"]>b0th["q99"] for r in bc]);target=min(pfpr,bfpr);pmatch=threshold_for_target_fpr([r["score"] for r in pc],target);bmatch=threshold_for_target_fpr([r["score"] for r in bc],target)
  pmatched_actual=float(np.mean([r["score"]>pmatch for r in pc]));bmatched_actual=float(np.mean([r["score"]>bmatch for r in bc]))
  thresholds={"primary":{**pth,"matched_clean_diagnostic":pmatch,"matched_target_fpr":target,"matched_actual_clean_fpr":pmatched_actual,"primary_model":"3-seed mean complex IG K7"},"B0_Exact":{**b0th,"matched_clean_diagnostic":bmatch,"matched_target_fpr":target,"matched_actual_clean_fpr":bmatched_actual},"calibration_count_caution":"q99/q99.5 are order statistics; q99.5 may be the second maximum or maximum when calibration N is small","primary_threshold_never_overwritten":True}
  # Common-timestamp primary and B0 rows, each with three independently recomputed alarms.
  saved=[]
  for name in datasets:
   pa,ba=common_rows([r for r in primary if r["scenario"]==name],[r for r in b0 if r["scenario"]==name])
   p_alarm=alarm_columns(pa,pth["q99"],pth["q995"],pmatch);b_alarm=alarm_columns(ba,b0th["q99"],b0th["q995"],bmatch)
   for r in p_alarm:r["threshold_source"]="primary clean calibration"
   for r in b_alarm:r["threshold_source"]="B0 Exact clean calibration"
   rows=sorted(p_alarm+b_alarm,key=lambda r:(r["decision_time_s"],r["model"]));write_csv(stage/"per_epoch"/f"{name}.csv",rows);saved.extend(rows)
  # Actual performance for every seed, ensemble/ablation, primary, and B0.
  source={}
  for (m,ms,ps),x in _group(epochs,"model","model_seed","policy_seed").items():source[f"seed::{m}::model{ms}::policy{ps}"]=x
  for (m,ps),x in _group(ensembles,"model","policy_seed").items():source[f"ensemble::{m}::policy{ps}"]=x
  source["primary 3-seed mean complex IG K7"]=primary;source["B0 Exact"]=b0
  # Every source is evaluated on timestamps shared with B0 Exact, scenario by scenario.
  source_common={}
  for label,rows0 in source.items():
   joined=[]
   for name in datasets:
    aa,_=common_rows([r for r in rows0 if r["scenario"]==name],[r for r in b0 if r["scenario"]==name]);joined.extend(aa)
   source_common[label]=joined
  metrics,boots=performance_rows(source_common,args.bootstrap_reps,{"primary 3-seed mean complex IG K7":pth,"B0 Exact":b0th})
  # Add matched common-timestamp primary/B0 metrics without changing q99 rows.
  for name in datasets:
   pa,ba=common_rows([r for r in primary if r["scenario"]==name],[r for r in b0 if r["scenario"]==name])
   for label,part,thx in (("primary 3-seed mean complex IG K7",pa,pmatch),("B0 Exact",ba,bmatch)):
    m,b=evaluate_detector(part,thx,scenario=name,bootstrap_reps=args.bootstrap_reps,seed=303,onset_s=ONSETS.get(name));m.update({"model":label,"operating_point":"matched_clean_diagnostic","calibration_q99":pth["q99"] if label.startswith("primary") else b0th["q99"],"matched_target_fpr":target,"common_timestamp_count":len(part)});metrics.append(m);boots.extend(dict(x,model=label,operating_point="matched_clean_diagnostic") for x in b)
  seedmetrics=[r for r in metrics if r["model"].startswith("seed::")]
  # Explicit mean/std rows across actual seed performance; never select a best seed.
  agg=[]
  for (scenario,op),part in _group(seedmetrics,"scenario","operating_point").items():
   bases=defaultdict(list)
   for r in part:
    base=r["model"].split("::model",1)[0].replace("seed::","");bases[base].append(r)
   for base,xx in bases.items():
    row={"scenario":scenario,"model":f"mean_std::{base}","model_seed":"mean/std","operating_point":op,"seed_count":len(set(r["model"].split("::model",1)[1].split("::",1)[0] for r in xx))}
    for key in ("held_out_clean_fpr","stable_pre_fpr","roc_auc","pr_auc","post_detection","persistent_detection"):
     vals=[v for v in (r.get(key) for r in xx) if v is not None];row[f"{key}_mean"]=float(np.mean(vals)) if vals else None;row[f"{key}_std"]=float(np.std(vals,ddof=1)) if len(vals)>1 else 0. if vals else None
    agg.append(row)
  seedmetrics.extend(agg)
  ablation=[r for r in metrics if r["model"].startswith("ensemble::") or r["model"] in ("primary 3-seed mean complex IG K7","B0 Exact")]
  primary_perf={(r["scenario"],r["operating_point"]):r for r in metrics if r["model"]=="primary 3-seed mean complex IG K7"}
  for r in ablation:
   ref=primary_perf.get((r["scenario"],r["operating_point"]));r["delta_roc_auc_vs_primary"]=(r["roc_auc"]-ref["roc_auc"]) if ref and r.get("roc_auc") is not None and ref.get("roc_auc") is not None else None;r["delta_post_detection_vs_primary"]=(r["post_detection"]-ref["post_detection"]) if ref and r.get("post_detection") is not None and ref.get("post_detection") is not None else None
  qpolicy=path_diagnostics(raw)
  # Join policy actual ROC delta against same-budget fixed ensemble diagnostic.
  em={(r["model"],r["scenario"]):r for r in metrics if r["model"].startswith("ensemble::")}
  for r in qpolicy:
   perf=em.get((f"ensemble::{r['model']}::policy{r['policy_seed']}",r["scenario"]));r["roc_auc"]=perf.get("roc_auc") if perf else None
   budget=5 if "K5" in r["model"] else 7 if "K7" in r["model"] else None
   fixed=em.get((f"ensemble::complex fixed K{budget}::policyNone",r["scenario"])) if budget else None
   random_rocs=[x.get("roc_auc") for (key,sc),x in em.items() if budget and sc==r["scenario"] and f"ensemble::complex random K{budget} " in key and x.get("roc_auc") is not None]
   r["delta_roc_auc_vs_fixed"]=(r["roc_auc"]-fixed["roc_auc"]) if r["roc_auc"] is not None and fixed and fixed.get("roc_auc") is not None else None
   r["delta_roc_auc_vs_random_mean"]=(r["roc_auc"]-float(np.mean(random_rocs))) if r["roc_auc"] is not None and random_rocs else None
  histrows=[]
  for (scenario,model,ms,ps),x in _group(raw,"scenario","model","model_seed","policy_seed").items():
   c=Counter("-".join(map(str,r["query_path"])) for r in x)
   histrows.extend({"scenario":scenario,"model":model,"model_seed":ms,"policy_seed":ps,"ordered_path":p,"count":n,"fraction":n/len(x)} for p,n in sorted(c.items()))
  total_s=time.perf_counter()-wall0;bench_n=len(raw);full_records=sum(len(full_history_indices(recs[n,"complex"])) for n in datasets)
  loaded_input_rows=sum(v["rows"] for v in inputs.values());canonical_input_rows=0
  for pp,_ in CANONICAL.values():
   with np.load(pp,allow_pickle=False) as ff:canonical_input_rows+=len(ff["time_s"])
  data_ratio=canonical_input_rows/loaded_input_rows;estimated_records=full_records*data_ratio;full_policy_rows=estimated_records*16*3
  scoring_est=scoring_s*full_policy_rows/max(1,bench_n);window_artifact_est=max(0,total_s-scoring_s)*data_ratio;training_est=2.23*2*3*50;full_est=scoring_est+window_artifact_est+training_est
  unique_benchmark_records=len({(r["scenario"],r["decision_time_s"],r["prn"]) for r in raw})
  runtime={"device":device,"batched_cuda":device=="cuda","mc_samples":8,"batch_size":args.batch_size,"scoring_seconds":scoring_s,"wall_seconds":total_s,"scored_record_policy_rows":bench_n,"unique_benchmark_records":unique_benchmark_records,"benchmark_min_200":unique_benchmark_records>=200,"full_data_records_loaded":full_records,"canonical_to_loaded_input_ratio":data_ratio,"estimated_full_record_policy_rows":full_policy_rows,"evaluation_subsampled":args.max_eval_samples is not None,"estimated_full_campaign_hours":full_est/3600,"estimated_full_scoring_hours":scoring_est/3600,"estimated_full_window_artifact_hours":window_artifact_est/3600,"estimated_full_training_hours":training_est/3600,"training_benchmark":"2.23 s for one full complex model epoch, Ntrain=4803/Nval=1552; scaled to 2 reps x 3 seeds x 50 epochs","estimate_method":"measured 200-record batched CUDA scoring scaled by canonical input-row ratio and declared 16 policies x 3 seeds; measured full-epoch training added","under_6h_target":full_est/3600<6,"inference_ms_mean":float(np.mean([r["inference_ms"] for r in raw]))}
  config={"schema":"gnss-doppler-lab.amcf-r1-config.v2","tap_order":TAP_NAMES,"side_indices":SIDE_INDICES,"prompt_index":4,"model_seeds":seeds,"policy_seeds":POLICY_SEEDS,"epochs":args.epochs,"hidden":args.hidden,"patience":8,"same_model_class_and_dims":True,"magnitude_projection":"same 7-D slots; imaginary/phase slots deterministic zero","primary":"3-seed mean complex IG K7","history_windows_required":12,"mc_samples":8,"device":device,"scenarios":names,"DS4":DS4_STATUS,"bootstrap_reps":args.bootstrap_reps,"smoke_limits":{"train":args.max_train_samples,"validation":args.max_val_samples,"evaluation":args.max_eval_samples}}
  source_files=[ROOT/"src/gnss_doppler_lab/amcf_r1.py",ROOT/"scripts/run_amcf_r1_texbat.py",ROOT/"scripts/summarize_amcf_r1.py"]
  provenance={"schema":"gnss-doppler-lab.amcf-r1-provenance.v2","source_commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),"source_hashes":{str(p.relative_to(ROOT)):sha256(p) for p in source_files},"inputs":inputs,"base_model_checkpoint_hashes":{p.name:sha256(p) for p in (stage/"models").glob("*.pt")},"attack_fit":False,"fit_selection_calibration":"clean-only","held_out_wording":"held-out chronological clean segments; not independent","raw_iq_or_receiver_rerun":False,"DS4":DS4_STATUS}
  env={"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,"pytorch":torch.__version__,"cuda_runtime":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"os":platform.platform(),"git_source_commit":provenance["source_commit"],"nvml_note":"NVML driver/library mismatch is known; successful PyTorch CUDA tensors/forwards are authoritative"}
  write_json(stage/"config.json",config);write_json(stage/"provenance.json",provenance);write_json(stage/"input_hashes.json",inputs);write_json(stage/"environment.json",env);write_json(stage/"window_qa.json",{"schema":"gnss-doppler-lab.amcf-r1-window-qa.v2","scenarios":qa_scenarios,"effectively_98pct_discarded_definition":"unique_valid_utilization < 0.02; hard failure uses <= 0.02"});write_json(stage/"model_audit.json",audits);write_json(stage/"thresholds.json",thresholds);write_json(stage/"inference_runtime.json",runtime)
  write_csv(stage/"prompt_rejection_by_phase.csv",rejection);write_csv(stage/"training_history.csv",training);write_csv(stage/"metrics.csv",metrics);write_csv(stage/"seed_metrics.csv",seedmetrics);write_csv(stage/"ablation_metrics.csv",ablation);write_csv(stage/"query_policy_metrics.csv",qpolicy);write_csv(stage/"query_path_histogram.csv",histrows);write_csv(stage/"bootstrap_confidence_intervals.csv",boots)
  try:
   import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
   for name in datasets:
    x=sorted([r for r in primary if r["scenario"]==name],key=lambda r:r["decision_time_s"]);fig,ax=plt.subplots(figsize=(8,3));ax.plot([r["decision_time_s"] for r in x],[r["score"] for r in x]);ax.axhline(pth["q99"],ls="--",color="r");ax.set(title=f"AMCF-R1 {name}",xlabel="s",ylabel="score");fig.tight_layout();fig.savefig(stage/"plots"/f"{name}.png");plt.close(fig)
  except Exception as e:(stage/"plots"/"ERROR.txt").write_text(str(e)+"\n")
  spec=importlib.util.spec_from_file_location("amcf_summary",ROOT/"scripts/summarize_amcf_r1.py");sm=importlib.util.module_from_spec(spec);spec.loader.exec_module(sm);sm.summarize(stage)
  write_hash_manifest(stage);missing=[x for x in REQUIRED_OUTPUTS if not (stage/x).exists()]
  if missing:raise RuntimeError(f"missing outputs {missing}")
  os.replace(stage,out);return out
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise

def parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,required=True);p.add_argument("--scenarios",default="DS1,DS2,DS3,DS7,DS8");p.add_argument("--epochs",type=int,default=50);p.add_argument("--seeds",default="101,202,303");p.add_argument("--hidden",type=int,default=32);p.add_argument("--max-train-samples",type=int);p.add_argument("--max-val-samples",type=int);p.add_argument("--max-eval-samples",type=int);p.add_argument("--bootstrap-reps",type=int,default=100);p.add_argument("--prompt-quantile",type=float,default=.005);p.add_argument("--batch-size",type=int,default=256);return p
def main(argv=None):
 a=normalize_args(parser().parse_args(argv));out=run(a);print(json.dumps({"out":str(out),"device":"cuda" if torch.cuda.is_available() else "cpu"},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
