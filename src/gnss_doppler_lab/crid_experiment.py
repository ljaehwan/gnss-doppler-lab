"""Frozen CRID Stage-0 evaluation and compact artifact construction."""
from __future__ import annotations
import csv,gzip,json
from pathlib import Path
from typing import Mapping
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from .crid import CONFIG_ORDER,chronological_split,empirical_threshold,estimate_causal_delays,fit_normal_model,score_aligned

def _metric(y,s,a):
 y=np.asarray(y,int);s=np.asarray(s,float);a=np.asarray(a,bool)
 if len(np.unique(y))<2:return {"roc_auc":None,"pauc_0_05":None,"pr_auc":None,"preonset_fpr":float(a.mean()),"attack_detection_rate":None}
 return {"roc_auc":float(roc_auc_score(y,s)),"pauc_0_05":float(roc_auc_score(y,s,max_fpr=.05)),
  "pr_auc":float(average_precision_score(y,s)),"preonset_fpr":float(a[y==0].mean()),"attack_detection_rate":float(a[y==1].mean())}

def fit_domain(clean_tables):
 delays=estimate_causal_delays(clean_tables);samples=np.concatenate([v.sample for v in clean_tables.values()]);split=chronological_split(samples)
 model=fit_normal_model(clean_tables,split["train"],split["calibration"]);scores=score_aligned(clean_tables,model,delays)
 cal=np.array([r["score"] for r in scores if split["calibration"][0]<=r["sample"]<=split["calibration"][-1]])
 thresholds={"q99":empirical_threshold(cal,.99),"q99_5":empirical_threshold(cal,.995),"target_1pct":empirical_threshold(cal,.99)}
 hold=np.array([r["score"] for r in scores if split["holdout"][0]<=r["sample"]<=split["holdout"][-1]])
 return model,delays,split,thresholds,scores,{"holdout_count":len(hold),"holdout_fpr_q99":float(np.mean(hold>thresholds["q99"]))}

def score_scenario(name,tables,model,delays,fs,onset,threshold):
 rows=score_aligned(tables,model,delays);out=[]
 for r in rows:
  time=float(r["sample"])/fs;label=int(time>=onset);out.append({"dataset":name,"sample":r["sample"],"time_s":time,
   "score":r["score"],"alarm":int(r["score"]>threshold),"label":label,"prn_count":r["prn_count"],"config_count":r["config_count"],
   "h0_loglike":r["h0_loglike"],"h1_loglike":r["h1_loglike"],"penalty":r["penalty"],"configuration_disagreement":r["configuration_disagreement"]})
 return out

def configuration_collapse(rows):
 # H0 replacement makes H1 improvement exactly zero, leaving its BIC penalty.
 original=np.array([r["score"] for r in rows],float);collapsed=np.array([-float(r["penalty"]) for r in rows])
 return {"original_mean":float(original.mean()) if len(original) else None,"collapsed_mean":float(collapsed.mean()) if len(collapsed) else None,
  "mean_drop":float((original-collapsed).mean()) if len(original) else None,"pass":bool(len(original) and np.mean(original-collapsed)>0)}

def shortcut_audit(rows,nuisance:Mapping[str,np.ndarray]):
 score=np.asarray([r["score"] for r in rows],float);out=[]
 for name,value in nuisance.items():
  value=np.asarray(value,float)[:len(score)];corr=float(np.corrcoef(score,value)[0,1]) if len(score)>2 and np.std(value)>0 else 0.
  out.append({"scalar":name,"correlation":corr,"absolute_correlation":abs(corr),"status":"FAIL" if abs(corr)>=.95 else "PASS"})
 return out

def leave_one_out(stacked,hmat):
 from .crid import score_epoch
 base=float(score_epoch(stacked,hmat)["score"]);configs=[];prns=[]
 for c in CONFIG_ORDER:configs.append({"omitted":c,"score":float(score_epoch(stacked,hmat,tuple(x for x in CONFIG_ORDER if x!=c))["score"]),"base":base})
 for p in stacked:prns.append({"omitted":int(p),"score":float(score_epoch({k:v for k,v in stacked.items() if k!=p},hmat)["score"]),"base":base})
 return {"configurations":configs,"prns":prns}

def write_epoch_artifacts(artifact:Path,rows):
 fields=("dataset","sample","time_s","score","alarm","label","prn_count","config_count","h0_loglike","h1_loglike","penalty","configuration_disagreement")
 with gzip.open(artifact/"per_epoch_scores.csv.gz","wt",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 # State estimates are emitted by the full receiver-state adapter. Keep a valid
 # empty compact table if evaluation stopped before that adapter completed.
 with gzip.open(artifact/"per_config_state_estimates.csv.gz","wt",newline="") as f:csv.writer(f).writerow(["dataset","sample","prn","config","delay_state","carrier_state"])

def scenario_metrics(rows,families):
 out=[]
 for dataset in sorted({r["dataset"] for r in rows}):
  rr=[r for r in rows if r["dataset"]==dataset];m=_metric([r["label"] for r in rr],[r["score"] for r in rr],[r["alarm"] for r in rr])
  out.append({"dataset":dataset,"family":families[dataset],"status":"COMPUTED",**m,"epoch_count":len(rr)})
 return out
