#!/usr/bin/env python3
"""Evaluate R0/S0/S1, domain gaps, and 199 alignment permutations."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.clif_ip_synthetic import DOMAINS,domain_gap,permutation_test

STATIC={"OAKBAT":{"scenarios":["os1","os2","os3","os4"],"stable_pre":[30,90],"exclude":[90,110],"post_start":110,"nominal_onset":120},
        "TEXBAT":{"scenarios":["DS1","DS2","DS3","DS4"],"stable_pre":[30,90],"exclude":[90,110],"post_start":110,"nominal_onset":100}}

def threshold_from_normal_validation(scores,q=.99):
 x=np.asarray(scores,float)
 if not len(x) or not np.isfinite(x).all():raise ValueError("finite normal validation scores required")
 return float(np.quantile(x,q))
def prediction_metrics(y,p,p1=None):
 y=np.asarray(y,float);p=np.asarray(p,float);mse=float(mean_squared_error(y,p));out={"mse":mse,"mae":float(mean_absolute_error(y,p)),"samples":len(y)}
 if p1 is not None:out.update({"improvement_pct":100*(p1-mse)/p1,"incremental_r2":1-mse/p1})
 return out
def scenario_metric(frame,score,threshold,contract):
 lo,hi=contract["stable_pre"];pre=frame[(frame.available_s>=lo)&(frame.available_s<hi)];post=frame[frame.available_s>=contract["post_start"]]
 if pre.empty or post.empty:return {"roc_auc":"NA","na_reason":"insufficient stable-pre/post support"}
 y=np.r_[np.zeros(len(pre)),np.ones(len(post))];x=np.r_[pre[score],post[score]]
 return {"roc_auc":float(roc_auc_score(y,x)),"stable_pre_fpr":float((pre[score]>threshold).mean()),"post_detection_rate":float((post[score]>threshold).mean()),"threshold":threshold,"transition_excluded":"90--110 s"}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("artifacts/clif_ip_synthetic_normal_r4"));ap.add_argument("--permutations",type=int,default=199);ap.add_argument("--scores",type=Path);ap.add_argument("--smoke",action="store_true");a=ap.parse_args()
 reps=3 if a.smoke else a.permutations
 if not a.smoke and reps!=199:raise SystemExit("final alignment test requires exactly 199 repetitions")
 pred=[];scenarios=[];gaps=[];destruction={"schema":"clif-ip.synthetic-normal.r4.alignment.v1","repetitions":reps,"p_value_resolution":1/(reps+1),"results":{}}
 if a.scores and a.scores.exists():
  d=pd.read_csv(a.scores)
  fit=d.query("role=='normal_validation'")
  if fit.empty or not fit.label.astype(str).str.lower().eq("normal").all():raise ValueError("threshold/calibration fit must be normal validation only")
  for (regime,domain),g in d.groupby(["regime","target_domain"]):
   v=fit.query("regime==@regime and target_domain==@domain")
   for score in ("B0","M1","mean_fusion","max_fusion","P3","Full"):
    th=threshold_from_normal_validation(v[score]);clean=g.query("role=='independent_clean_test'")
    independent_fpr=float((clean[score]>th).mean()) if len(clean) else "NA"
    for scenario,sg in g[g.role.eq("attack")].groupby("scenario"):
     contract=STATIC["OAKBAT" if domain=="SYN-OAK" else "TEXBAT"];scenarios.append({"regime":regime,"target_domain":domain,"scenario":scenario,"model":score,"independent_clean_fpr":independent_fpr,**scenario_metric(sg,score,th,contract)})
   for region,rg in [("clean_test",g.query("role=='independent_clean_test'")),("attack_pre",g.query("role=='attack' and available_s<90")),("attack_established",g.query("role=='attack' and available_s>=110"))]:
    if len(rg)>=16:destruction["results"][f"{regime}:{domain}:{region}"]=permutation_test(rg.B0.to_numpy(),rg[[c for c in rg if c.startswith("m1_innov_")]].to_numpy(),repetitions=reps,seed=73,block=8,region=region)
 else:
  destruction["provisional_reason"]="no --scores supplied; no metrics fabricated"
 pd.DataFrame(pred,columns=["regime","target_domain","split","model","mse","mae","samples","improvement_pct","incremental_r2"]).to_csv(a.root/"predictor_comparison.csv",index=False)
 pd.DataFrame(scenarios,columns=["regime","target_domain","scenario","model","roc_auc","stable_pre_fpr","independent_clean_fpr","post_detection_rate","threshold","transition_excluded","na_reason"]).to_csv(a.root/"scenario_metrics.csv",index=False)
 pd.DataFrame(gaps,columns=["regime","target_domain","feature_group","smd_mean","wasserstein_mean","mmd_rbf","rmse_ratio_5p5x","synthetic_threshold_real_fpr"]).to_csv(a.root/"domain_gap_metrics.csv",index=False)
 (a.root/"alignment_destruction_metrics.json").write_text(json.dumps(destruction,indent=2)+"\n")
 raw=[]
 for region,x in destruction["results"].items():raw.extend({"region_key":region,**r} for r in x["raw_metrics"])
 pd.DataFrame(raw,columns=["region_key","replicate","seed","aligned_mse","shuffled_mse","delta"]).to_csv(a.root/"alignment_destruction_raw_metrics.csv",index=False)
if __name__=="__main__":main()
