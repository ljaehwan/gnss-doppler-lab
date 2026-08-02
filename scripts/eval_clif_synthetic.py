#!/usr/bin/env python3
"""Execute leakage-safe R4 OAKBAT/TEXBAT evaluation and diagnostics."""
from __future__ import annotations
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.clif_ip_synthetic import TAP_ORDER,domain_gap,permutation_test,extract_m1_features
from scripts.train_clif_synthetic import SharedPrnGRU,tap_columns,load_domain,real_clean_frames
LAB=Path("/home/ubuntu/projects/gnss-doppler-lab");SSD=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts")
OAK_B=LAB/"artifacts/oakbat_cleanstatic_detector_eval_v1/preprocessed"
OAK_M={"cleanStatic":SSD/"oakbat-cleanStatic-raw-iq-noise-continuity-m1-v0-fs5m-full/oakbat_cleanStatic_raw_iq_noise_features.csv","os1":None,**{s:SSD/f"oakbat-{s}-raw-iq-noise-continuity-m1-v0-fs5m-full"/f"oakbat_{s}_raw_iq_noise_features.csv" for s in ("os2","os3","os4")}}
TEX_SCORE=LAB/"artifacts/ai_morph_gru_cleanStatic_q70_frame/scored"
TEX_RAW=Path("/home/ubuntu/unraid/gnss-datasets/texbat/raw")
ALPHAS=(.1,1.,10.,100.);LAG=6

def infer_b0(d,ck):
 cols=tap_columns(d);model=SharedPrnGRU();model.load_state_dict(ck["state_dict"]);model.eval();mu=np.asarray(ck["mean"]);sd=np.asarray(ck["std"]);rows=[]
 with torch.no_grad():
  for (run,prn),g in d.groupby(["run_id","prn"],sort=False):
   g=g.sort_values("window_bin_s");z=g[cols].to_numpy(np.float32)
   if len(z)<=12:continue
   X=np.asarray([z[i-12:i] for i in range(12,len(z))]);pred=model(torch.tensor((X-mu)/sd,dtype=torch.float32)).numpy()*sd+mu
   res=z[12:]-pred
   for i,r in enumerate(res):rows.append({"run_id":run,"prn":str(prn),"t":float(g.window_bin_s.iloc[i+12]),"available_s":float(g.window_end_s.iloc[i+12]),**{f"r{j}":r[j] for j in range(9)},"B0":float(np.sqrt(np.mean(r*r)))})
 return pd.DataFrame(rows)

def m1_innov(d,ck,run="recording"):
 cols=ck["m1_feature_columns"];st=ck["m1_state"];lag=int(st["lag"]);x=d[cols].to_numpy(float);p=((x-st["mean"])/st["scale"])@st["components"];rows=[]
 order=np.argsort(d.window_start_s.to_numpy());p=p[order];dd=d.iloc[order].reset_index(drop=True)
 for i in range(lag,len(p)):
  pred=p[i-lag:i].reshape(-1)@st["ar_coef"];r=p[i]-pred;t=float(np.floor(float(dd.window_start_s.iloc[i])*2+.5)/2)
  rows.append({"run_id":run,"t":t,"m1_available_s":float(dd.window_end_s.iloc[i]),**{f"m{j}":r[j] for j in range(len(r))},"M1":float(np.sqrt(np.mean(r*r)))})
 return pd.DataFrame(rows)

def aligned(b,m):
 x=b.merge(m,on=["run_id","t"],how="inner");return x.sort_values(["run_id","prn","t"]).reset_index(drop=True)

def synth_frame(root,domain,split,ck):
 b,m=load_domain(root,domain,split);bf=infer_b0(b,ck);parts=[]
 for run,g in m.groupby("run_id",sort=False):parts.append(m1_innov(g,ck,run))
 return aligned(bf,pd.concat(parts,ignore_index=True))

def design(d,kind):
 rc=[f"r{i}" for i in range(9)];mc=[c for c in d if c.startswith("m") and c[1:].isdigit()];X=[];Y=[];meta=[]
 for _,g in d.groupby(["run_id","prn"],sort=False):
  g=g.sort_values("t").reset_index(drop=True)
  for i in range(LAG,len(g)):
   if not np.allclose(np.diff(g.t.iloc[i-LAG:i+1]),.5):continue
   z=[]
   if kind in ("P1","P3"):z.extend(g[rc].iloc[i-LAG:i].to_numpy().ravel())
   if kind in ("P2","P3"):z.extend(g[mc].iloc[i-LAG:i+1].to_numpy().ravel())
   X.append(z);Y.append(g[rc].iloc[i].to_numpy());meta.append(g.iloc[i].to_dict())
 return np.asarray(X),np.asarray(Y),pd.DataFrame(meta)

def fit_predictors(train,val):
 out={};details=[]
 for kind in ("P1","P2","P3"):
  X,y,_=design(train,kind);Xv,yv,_=design(val,kind);mu=X.mean(0);sd=np.where(X.std(0)>1e-9,X.std(0),1.);best=None
  for a in ALPHAS:
   model=Ridge(alpha=a).fit((X-mu)/sd,y);mse=float(np.mean((yv-model.predict((Xv-mu)/sd))**2))
   if best is None or mse<best[0]:best=(mse,a,model)
  out[kind]=(best[2],mu,sd);details.append({"model":kind,"alpha":best[1],"validation_mse":best[0],"parameter_count":int(best[2].coef_.size+best[2].intercept_.size),"train_support":len(y),"validation_support":len(yv)})
 return out,details

def score_predictors(d,models):
 supports=[];frames={}
 for kind in ("P1","P2","P3"):
  X,y,m=design(d,kind);model,mu,sd=models[kind];p=model.predict((X-mu)/sd);m[kind]=np.mean((y-p)**2,axis=1);m[f"{kind}_mse_target"]=m[kind];frames[kind]=m;supports.append(list(zip(m.run_id,m.prn,m.t)))
 if supports.count(supports[0])!=3:raise RuntimeError("P1-P3 support mismatch")
 z=frames["P3"].copy();z["P0"]=np.mean(z[[f"r{i}" for i in range(9)]].to_numpy()**2,axis=1)
 for k in ("P1","P2"):z[k]=frames[k][k].to_numpy()
 return z

def calibrate(train_scores,d):
 comps=["B0","M1","P3"];cal={c:(float(train_scores[c].median()),max(float(train_scores[c].quantile(.75)-train_scores[c].quantile(.25)),1e-9)) for c in comps}
 z=d.copy()
 for c in comps:z[c+"z"]=np.maximum(0,(z[c]-cal[c][0])/cal[c][1])
 z["cal_mean"]=z[[c+"z" for c in comps]].mean(axis=1);z["cal_max"]=z[[c+"z" for c in comps]].max(axis=1);z["concordance"]=np.abs(z.B0z-z.M1z);z["Full"]=z[["B0z","M1z","P3z","concordance"]].mean(axis=1)
 return z,cal

def epoch_scores(d,models,caltrain=None):
 p=score_predictors(d,models);e=p.groupby("t",as_index=False).agg({"available_s":"max","B0":"median","M1":"first","P0":"median","P1":"median","P2":"median","P3":"median"})
 if caltrain is None:return e
 return calibrate(caltrain,e)[0]

def metrics(val,clean,attacks,regime,domain):
 rows=[]
 for model in ("B0","M1","cal_mean","cal_max","P3","Full"):
  th=float(val[model].quantile(.99));fpr=float((clean[model]>th).mean())
  for sc,d in attacks.items():
   pre=d[(d.available_s>=30)&(d.available_s<90)];post=d[d.available_s>=110];y=np.r_[np.zeros(len(pre)),np.ones(len(post))];x=np.r_[pre[model],post[model]]
   alarm=d[(d.available_s>=110)&(d[model]>th)].available_s
   rows.append({"regime":regime,"target_domain":domain,"scenario":sc,"model":model,"roc_auc":float(roc_auc_score(y,x)) if len(np.unique(y))==2 else "NA","pr_auc":float(average_precision_score(y,x)) if len(np.unique(y))==2 else "NA","stable_pre_fpr":float((pre[model]>th).mean()),"independent_clean_fpr":fpr,"post_detection_rate":float((post[model]>th).mean()),"first_alarm_delay_s":float(alarm.iloc[0]-110) if len(alarm) else "NA","persistent_detection_rate":float((post[model].rolling(3).min()>th).mean()),"threshold":th,"threshold_source":"synthetic_validation" if regime=="S0" else "real_clean_validation_250_330","transition_excluded":"90--110 s","na_reason":""})
 return rows

def real_paths(root,domain,scenario,ck):
 if domain=="SYN-OAK":
  bp=(LAB/"artifacts/oakbat_9tap_frozen_champion/cleanStatic/multi_prn_method_a_9tap_w1.0_s0.5_normalized_dmcpd/normal_prn_node_windows.csv") if scenario=="cleanStatic" else OAK_B/scenario/"multi_prn_method_a_9tap_w1.0_s0.5_normalized_dmcpd/normal_prn_node_windows.csv"
  mp=OAK_M[scenario] if scenario!="os1" else root/"../clif_ip_cross_layer_r3/input_cache/oakbat_os1_raw_iq_noise_features.csv"
 else:
  if scenario=="cleanStatic":b,_,_=real_clean_frames(root,domain);bp=None
  elif scenario=="DS4":bp=LAB/"artifacts/ai_morph_gru_window_ablation_ds4_20260723/w1.0_s0.5/ds4/multi_prn_method_a_9tap_normalized_dmcpd_w1.0_s0.5/normal_prn_node_windows.csv"
  else:return None
  cache=root/"real_inputs";cache.mkdir(exist_ok=True);mp=cache/f"texbat_{scenario}_m1_features.csv"
  if not mp.exists():
   raw=TEX_RAW/("cleanStatic.bin" if scenario=="cleanStatic" else scenario.lower()+".bin");duration=raw.stat().st_size/(4*25_000_000);extract_m1_features(raw,scenario,25_000_000,duration,block_ms=10,stride_s=.5).to_csv(mp,index=False)
 if domain=="SYN-TEX" and scenario=="cleanStatic":bd=b
 else:bd=pd.read_csv(bp)
 bd["run_id"]=scenario;md=pd.read_csv(mp);md["run_id"]=scenario
 return aligned(infer_b0(bd,ck),m1_innov(md,ck,scenario))

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("artifacts/clif_ip_synthetic_normal_r4"));ap.add_argument("--permutations",type=int,default=199);a=ap.parse_args()
 if a.permutations!=199:raise SystemExit("final requires 199 permutations")
 scenarios=[];predrows=[];gaps=[];rawperm=[];permout={"schema":"clif-ip.synthetic-normal.r4.alignment.v2","repetitions":199,"p_value_resolution":.005,"results":{},"unavailable":{}}
 for domain in ("SYN-OAK","SYN-TEX"):
  for regime in ("S0","S1"):
   ck=torch.load(a.root/"models"/f"{regime}_{domain}.pt",map_location="cpu",weights_only=False)
   if regime=="S0":train=synth_frame(a.root,domain,"train",ck);valbase=synth_frame(a.root,domain,"validation",ck)
   else:
    allclean=real_paths(a.root,domain,"cleanStatic",ck);train=allclean[(allclean.t>=0)&(allclean.t<240)];valbase=allclean[(allclean.t>=250)&(allclean.t<330)]
   models,details=fit_predictors(train,valbase)
   for x in details:predrows.append({"regime":regime,"target_domain":domain,"split":"validation","mse":x["validation_mse"],"samples":x["validation_support"],**x})
   vals=epoch_scores(valbase,models);valcal,_=calibrate(vals,vals)
   cleanbase=real_paths(a.root,domain,"cleanStatic",ck);cleanbase=cleanbase[cleanbase.t>=340];clean=epoch_scores(cleanbase,models);clean,_=calibrate(vals,clean)
   names=("os1","os2","os3","os4") if domain=="SYN-OAK" else ("DS1","DS2","DS3","DS4")
   attacks={};
   for sc in names:
    x=real_paths(a.root,domain,sc,ck)
    if x is None:
     for model in ("B0","M1","cal_mean","cal_max","P3","Full"):scenarios.append({"regime":regime,"target_domain":domain,"scenario":sc,"model":model,"na_reason":"Method-A nine-tap node rows unavailable; no values fabricated"})
     continue
    e=epoch_scores(x,models);e,_=calibrate(vals,e);attacks[sc]=e
   scenarios.extend(metrics(valcal,clean,attacks,regime,domain))
   # actual/synthetic domain gaps on canonical nine-tap residuals and M1 innovations
   syn=synth_frame(a.root,domain,"synthetic_test",ck);real=cleanbase
   for group,cols in (("B0_residual_9D",[f"r{i}" for i in range(9)]),("M1_innovation",[c for c in syn if c.startswith("m") and c[1:].isdigit()])):
    sa=syn[cols].to_numpy();ra=real[cols].to_numpy();rng=np.random.default_rng(91)
    if len(sa)>1000:sa=sa[rng.choice(len(sa),1000,replace=False)]
    if len(ra)>1000:ra=ra[rng.choice(len(ra),1000,replace=False)]
    dg=domain_gap(sa,ra);gaps.append({"regime":regime,"target_domain":domain,"feature_group":group,**dg,"synthetic_threshold_real_fpr":float((real.B0>syn.B0.quantile(.99)).mean()) if group.startswith("B0") else "NA"})
   for region,d in [("clean_test",cleanbase)]+[(f"{k}_pre",v[v.t<90]) for k,v in attacks.items()]+[(f"{k}_post",v[v.t>=110]) for k,v in attacks.items()]:
    if len(d)<16:continue
    mcols=[c for c in d if c.startswith("m") and c[1:].isdigit()]
    if all(f"r{i}" in d for i in range(9)) and mcols:bmat=d[[f"r{i}" for i in range(9)]].to_numpy();mmat=d[mcols].to_numpy()
    else:bmat=d[["B0"]].to_numpy();mmat=d[["M1"]].to_numpy()
    res=permutation_test(bmat,mmat,repetitions=199,seed=73,block=8,region=region);key=f"{regime}:{domain}:{region}";permout["results"][key]={k:v for k,v in res.items() if k!="raw_metrics"};rawperm.extend({"region_key":key,**r} for r in res["raw_metrics"])
 # exact read-only R0 OAK rows
 r3=pd.read_csv(ROOT/"artifacts/clif_ip_cross_layer_r3/scenario_metrics.csv")
 for _,r in r3.iterrows():scenarios.append({"regime":"R0","target_domain":"SYN-OAK",**r.to_dict(),"na_reason":"exact immutable R3 result"})
 # R0-TEX is explicitly unavailable as a complete CLIF comparator: residual exports do not expose trainable Method-A nodes.
 for sc in ("DS1","DS2","DS3","DS4"):
  for model in ("B0","M1","cal_mean","cal_max","P3","Full"):scenarios.append({"regime":"R0-TEX-real-only","target_domain":"SYN-TEX","scenario":sc,"model":model,"na_reason":"full chronological Method-A training nodes unavailable; existing residual exports are not silently treated as trainable targets"})
 pd.DataFrame(predrows).to_csv(a.root/"predictor_comparison.csv",index=False);pd.DataFrame(scenarios).to_csv(a.root/"scenario_metrics.csv",index=False);pd.DataFrame(gaps).to_csv(a.root/"domain_gap_metrics.csv",index=False);pd.DataFrame(rawperm).to_csv(a.root/"alignment_destruction_raw_metrics.csv",index=False);(a.root/"alignment_destruction_metrics.json").write_text(json.dumps(permout,indent=2)+"\n")
if __name__=="__main__":main()
