#!/usr/bin/env python3
"""Scientifically complete R4 evaluation with frozen calibration and raw evidence."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score,average_precision_score,r2_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.clif_ip_synthetic import TAP_ORDER,domain_gap,extract_m1_features,evaluation_masks,tex_npz_magnitude_nodes,array_hash
from scripts.train_clif_synthetic import SharedPrnGRU,tap_columns,load_domain,real_clean_frames,sha
LAB=Path("/home/ubuntu/projects/gnss-doppler-lab");SSD=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts")
OAK_B=LAB/"artifacts/oakbat_cleanstatic_detector_eval_v1/preprocessed"
OAK_M={"cleanStatic":SSD/"oakbat-cleanStatic-raw-iq-noise-continuity-m1-v0-fs5m-full/oakbat_cleanStatic_raw_iq_noise_features.csv","os1":None,**{s:SSD/f"oakbat-{s}-raw-iq-noise-continuity-m1-v0-fs5m-full"/f"oakbat_{s}_raw_iq_noise_features.csv" for s in ("os2","os3","os4")}}
TEX_NPZ=SSD/"texbat-ds123-graph-input/exports"
TEX_RAW=Path("/home/ubuntu/unraid/gnss-datasets/texbat/raw")
TEX_M1={s:SSD/f"{s.lower()}-raw-iq-noise-continuity-20260729-v0"/f"{s.lower()}_raw_iq_noise_features.csv" for s in ("DS2","DS3","DS4")}
TEX_DS4_NODE=LAB/"artifacts/ai_morph_gru_window_ablation_ds4_20260723/w1.0_s0.5/ds4/multi_prn_method_a_9tap_normalized_dmcpd_w1.0_s0.5/normal_prn_node_windows.csv"
ALPHAS=(.1,1.,10.,100.);LAG=6;KINDS=("P1","P2","P3")
INPUT_PROVENANCE={}

def digest_obj(*xs):
 h=hashlib.sha256()
 for x in xs:
  a=np.ascontiguousarray(x);h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()

def infer_b0(d,ck):
 cols=tap_columns(d);model=SharedPrnGRU();model.load_state_dict(ck["state_dict"]);model.eval();mu=np.asarray(ck["mean"]);sd=np.asarray(ck["std"]);rows=[]
 with torch.no_grad():
  for (run,prn),g in d.groupby(["run_id","prn"],sort=False):
   g=g.sort_values("window_bin_s");z=g[cols].to_numpy(np.float32)
   if len(z)<=12:continue
   X=np.asarray([z[i-12:i] for i in range(12,len(z))]);pred=model(torch.tensor((X-mu)/sd,dtype=torch.float32)).numpy()*sd+mu;res=z[12:]-pred
   for i,r in enumerate(res):
    row={"run_id":run,"prn":str(prn),"t":float(g.window_bin_s.iloc[i+12]),"available_s":float(g.window_end_s.iloc[i+12]),"B0":float(np.sqrt(np.mean(r*r)))}
    row.update({f"r{j}":float(r[j]) for j in range(9)});rows.append(row)
 return pd.DataFrame(rows)

def m1_innov(d,ck,run="recording"):
 cols=ck["m1_feature_columns"];missing=set(cols)-set(d)
 if missing:raise ValueError(f"M1 input lacks frozen columns: {sorted(missing)}")
 st=ck["m1_state"];lag=int(st["lag"]);x=d[cols].to_numpy(float);p=((x-st["mean"])/st["scale"])@st["components"];rows=[]
 order=np.argsort(d.window_start_s.to_numpy());p=p[order];dd=d.iloc[order].reset_index(drop=True);center=np.asarray(st.get("residual_center",np.zeros(p.shape[1])));cov=np.asarray(st.get("residual_cov",np.eye(p.shape[1])));inv=np.linalg.pinv(cov)
 for i in range(lag,len(p)):
  pred=p[i-lag:i].reshape(-1)@st["ar_coef"];r=p[i]-pred;t=float(np.floor(float(dd.window_start_s.iloc[i])*2+.5)/2);rc=r-center
  row={"run_id":run,"t":t,"m1_available_s":float(dd.window_end_s.iloc[i]),"M1":float(np.sqrt(max(0,rc@inv@rc)/len(rc)))}
  row.update({f"m{j}":float(r[j]) for j in range(len(r))});rows.append(row)
 return pd.DataFrame(rows)

def aligned(b,m):
 x=b.merge(m,on=["run_id","t"],how="inner");return x.sort_values(["run_id","prn","t"]).reset_index(drop=True)

def synth_frame(root,domain,split,ck):
 b,m=load_domain(root,domain,split);bf=infer_b0(b,ck);parts=[m1_innov(g,ck,run) for run,g in m.groupby("run_id",sort=False)]
 return aligned(bf,pd.concat(parts,ignore_index=True))

def design(d,kind):
 rc=[f"r{i}" for i in range(9)];mc=[c for c in d if c.startswith("m") and c[1:].isdigit()];X=[];Y=[];meta=[]
 for _,g in d.groupby(["run_id","prn"],sort=False):
  g=g.sort_values("t").reset_index(drop=True)
  for i in range(LAG,len(g)):
   if not np.allclose(np.diff(g.t.iloc[i-LAG:i+1]),.5,atol=1e-6):continue
   z=[]
   if kind in ("P1","P3"):z.extend(g[rc].iloc[i-LAG:i].to_numpy().ravel())
   if kind in ("P2","P3"):z.extend(g[mc].iloc[i-LAG:i+1].to_numpy().ravel())
   X.append(z);Y.append(g[rc].iloc[i].to_numpy());meta.append(g.iloc[i].to_dict())
 return np.asarray(X,float),np.asarray(Y,float).reshape(-1,9),pd.DataFrame(meta)

def predict_matrix(spec,X):
 z=(X-spec["mean"])/spec["scale"];base=spec["base_model"].predict(z) if spec.get("base_model") is not None else 0.
 return base+spec["model"].predict(z)

def fit_predictors(train,val,base_models=None):
 out={};details=[]
 for kind in KINDS:
  X,y,_=design(train,kind);Xv,yv,_=design(val,kind);mu=X.mean(0);sd=np.where(X.std(0)>1e-9,X.std(0),1.);base=None
  if base_models is not None:
   old=base_models[kind];base=old["model"];base_train=predict_matrix(old,X);base_val=predict_matrix(old,Xv);target=y-base_train
  else:base_val=0.;target=y
  best=None
  for a in ALPHAS:
   model=Ridge(alpha=a).fit((X-mu)/sd,target);pred=(base_val+model.predict((Xv-mu)/sd));mse=float(np.mean((yv-pred)**2))
   if best is None or mse<best[0]:best=(mse,a,model,pred)
  spec={"model":best[2],"base_model":base,"mean":mu,"scale":sd,"alpha":best[1],"correction":base_models is not None};out[kind]=spec
  basehash="NA" if base is None else digest_obj(base.coef_,base.intercept_);corrhash=digest_obj(best[2].coef_,best[2].intercept_)
  details.append({"model":kind,"alpha":best[1],"validation_mse":best[0],"parameter_count":int(best[2].coef_.size+best[2].intercept_.size),"train_support":len(y),"validation_support":len(yv),"s0_base_coef_hash":basehash,"correction_coef_hash":corrhash,"correction_nonzero":bool(np.any(np.abs(best[2].coef_)>0))})
 return out,details

def score_predictors(d,models,include_predictions=False):
 supports=[];frames={}
 for kind in KINDS:
  X,y,m=design(d,kind);p=predict_matrix(models[kind],X);m[kind]=np.mean((y-p)**2,axis=1);m[f"{kind}_mae"]=np.mean(np.abs(y-p),axis=1)
  if include_predictions:
   for j,tap in enumerate(TAP_ORDER):m[f"target_{tap}"]=y[:,j];m[f"pred_{kind}_{tap}"]=p[:,j]
  frames[kind]=m;supports.append(list(zip(m.run_id,m.prn,m.t)))
 if not supports[0] or any(x!=supports[0] for x in supports[1:]):raise RuntimeError("P1-P3 nonempty support mismatch")
 z=frames["P3"].copy();z["P0"]=np.mean(z[[f"r{i}" for i in range(9)]].to_numpy()**2,axis=1);z["P0_mae"]=np.mean(np.abs(z[[f"r{i}" for i in range(9)]].to_numpy()),axis=1)
 for k in ("P1","P2"):
  z[k]=frames[k][k].to_numpy();z[f"{k}_mae"]=frames[k][f"{k}_mae"].to_numpy()
  if include_predictions:
   for tap in TAP_ORDER:z[f"pred_{k}_{tap}"]=frames[k][f"pred_{k}_{tap}"].to_numpy()
 return z

def epoch_base(pred):
 agg={"available_s":"max","m1_available_s":"max","B0":"median","M1":"first",**{k:"median" for k in ("P0","P1","P2","P3")}}
 return pred.groupby(["run_id","t"],as_index=False).agg(agg)

def _fit_mahal(x):
 x=np.asarray(x,float);center=x.mean(0);raw=np.cov(x-center,rowvar=False)
 if np.ndim(raw)==0:raw=np.asarray([[float(raw)]])
 cov=.9*raw+.1*np.diag(np.diag(raw))+1e-9*np.eye(raw.shape[0]);return center,np.linalg.pinv(cov)

def fit_fusion(validation):
 base=validation.copy();mu={c:float(base[c].mean()) for c in ("B0","M1","P3")};sd={c:max(float(base[c].std(ddof=1)),1e-9) for c in mu}
 z=np.c_[(base.B0-mu["B0"])/sd["B0"],(base.M1-mu["M1"])/sd["M1"],(base.P3-mu["P3"])/sd["P3"]];con=np.maximum(z[:,0],0)*np.maximum(z[:,1],0)
 q=np.c_[z,con];sets={"B0_M1":[0,1],"B0_P3":[0,2],"M1_P3":[1,2],"Full":[0,1,2,3]};fits={k:_fit_mahal(q[:,v]) for k,v in sets.items()}
 return {"component_mean":mu,"component_scale":sd,"sets":sets,"fits":fits,"fit_rows":len(base),"source":"validation_only_centered_mean_shrinkage_cov","concordance":"positive_tail(B0z)*positive_tail(M1z)"}

def apply_fusion(d,cal):
 z=d.copy();arr=np.column_stack([(z[c].to_numpy(float)-cal["component_mean"][c])/cal["component_scale"][c] for c in ("B0","M1","P3")])
 z["B0z"],z["M1z"],z["P3z"]=arr[:,0],arr[:,1],arr[:,2];z["concordance"]=np.maximum(arr[:,0],0)*np.maximum(arr[:,1],0);q=np.c_[arr,z.concordance]
 for name,ix in cal["sets"].items():
  center,inv=cal["fits"][name];v=q[:,ix]-center;z[name]=np.einsum("ij,jk,ik->i",v,inv,v)
 return z

def split_prediction_metrics(pred,regime,domain,split):
 support=hashlib.sha256("\n".join(f"{r}|{p}|{t:.6f}" for r,p,t in zip(pred.run_id,pred.prn,pred.t)).encode()).hexdigest();rows=[]
 for k in ("P0","P1","P2","P3"):
  row={"regime":regime,"target_domain":domain,"split":split,"model":k,"mse":float(pred[k].mean()),"mae":float(pred[f"{k}_mae"].mean()),"samples":len(pred),"support_sha256":support,"target_dimensions":9,"na_reason":"P3-vs-P1 applies only to P3; incremental R2 applies only to P2/P3" if k in ("P0","P1","P2") else ""}
  row["improvement_vs_P1"]=(float(pred.P1.mean()-pred[k].mean()) if k=="P3" else "NA");row["incremental_r2_vs_P1"]=(1-float(pred[k].mean())/max(float(pred.P1.mean()),1e-15) if k in ("P2","P3") else "NA")
  for j,tap in enumerate(TAP_ORDER):row[f"tap_{tap}_mse"]=float(np.mean((pred[f"target_{tap}"]-pred.get(f"pred_{k}_{tap}",0))**2))
  rows.append(row)
 return rows

def scenario_metrics(val,clean,attacks,regime,domain):
 rows=[];models=("B0","M1","P0","P1","P2","P3","B0_M1","B0_P3","M1_P3","Full")
 for model in models:
  th=float(val[model].quantile(.99));fpr=float((clean[model]>th).mean())
  for sc,d in attacks.items():
   masks=evaluation_masks(d.available_s,domain);pre=d[masks["stable_pre"]];post=d[masks["established_post"]];y=np.r_[np.zeros(len(pre)),np.ones(len(post))];x=np.r_[pre[model],post[model]];alarm=post[post[model]>th].available_s
   rows.append({"regime":regime,"target_domain":domain,"comparison_scope":"r4_same_protocol","scenario":sc,"model":model,"roc_auc":float(roc_auc_score(y,x)),"pr_auc":float(average_precision_score(y,x)),"stable_pre_fpr":float((pre[model]>th).mean()),"independent_clean_fpr":fpr,"post_detection_rate":float((post[model]>th).mean()),"first_alarm_delay_s":float(alarm.iloc[0]-masks["nominal_onset_s"]) if len(alarm) else "NA","persistent_detection_rate":float((post[model].rolling(3).min()>th).mean()),"threshold":th,"threshold_source":"validation_only","nominal_onset_s":masks["nominal_onset_s"],"score_time_field":"available_s","transition_excluded":("110<=t<130" if domain=="SYN-OAK" else "90<=t<110"),"stable_pre_samples":len(pre),"established_post_samples":len(post),"na_reason":"no established-post threshold crossing; delay NA" if not len(alarm) else ""})
 return rows

def _record_input(p,role):INPUT_PROVENANCE[role]={"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size}

def real_paths(root,domain,scenario,ck):
 if domain=="SYN-OAK":
  bp=(LAB/"artifacts/oakbat_9tap_frozen_champion/cleanStatic/multi_prn_method_a_9tap_w1.0_s0.5_normalized_dmcpd/normal_prn_node_windows.csv") if scenario=="cleanStatic" else OAK_B/scenario/"multi_prn_method_a_9tap_w1.0_s0.5_normalized_dmcpd/normal_prn_node_windows.csv"
  mp=OAK_M[scenario] if scenario!="os1" else root/"../clif_ip_cross_layer_r3/input_cache/oakbat_os1_raw_iq_noise_features.csv"
  bd=pd.read_csv(bp);md=pd.read_csv(mp)
 else:
  cache=root/"real_inputs";cache.mkdir(exist_ok=True)
  if scenario=="cleanStatic":bd,md,mp=real_clean_frames(root,domain);bp=cache/"texbat_cleanStatic_method_a_magnitude_nodes.csv"
  elif scenario in ("DS1","DS2","DS3"):
   bp=cache/f"texbat_{scenario}_method_a_magnitude_nodes.csv";src=TEX_NPZ/f"{scenario.lower()}.npz"
   if not bp.exists():tex_npz_magnitude_nodes(src,bp,scenario)
   bd=pd.read_csv(bp)
   if scenario=="DS1":
    mp=cache/"texbat_DS1_m1_features.csv"
    if not mp.exists():
     raw=TEX_RAW/"ds1.bin";duration=raw.stat().st_size/(4*25_000_000);extract_m1_features(raw,"DS1",25_000_000,duration,block_ms=10,stride_s=.5).to_csv(mp,index=False)
   else:mp=TEX_M1[scenario]
   md=pd.read_csv(mp)
  elif scenario=="DS4":bp=TEX_DS4_NODE;mp=TEX_M1[scenario];bd=pd.read_csv(bp);md=pd.read_csv(mp)
  else:raise ValueError(scenario)
 _record_input(bp,f"{domain}:{scenario}:B0");_record_input(mp,f"{domain}:{scenario}:M1")
 bd["run_id"]=scenario;md["run_id"]=scenario
 return aligned(infer_b0(bd,ck),m1_innov(md,ck,scenario))

def alignment_destruction(d,models,cal,repetitions,seed,region_key):
 """Vectorized block destruction followed by frozen 9-D predictor inference."""
 d=d.reset_index(drop=True);mcols=[c for c in d if c.startswith("m") and c[1:].isdigit()];rc=[f"r{i}" for i in range(9)]
 targets=[];hist=[];bhist=[]
 for _,idx in d.groupby(["run_id","prn"],sort=False).groups.items():
  ids=np.asarray(list(idx));times=d.loc[ids,"t"].to_numpy()
  for pos in range(LAG,len(ids)):
   if np.allclose(np.diff(times[pos-LAG:pos+1]),.5,atol=1e-6):
    targets.append(ids[pos]);hist.append(ids[pos-LAG:pos+1]);bhist.append(d.loc[ids[pos-LAG:pos],rc].to_numpy().ravel())
 targets=np.asarray(targets);hist=np.asarray(hist);bhist=np.asarray(bhist);y=d.loc[targets,rc].to_numpy();mv=d[mcols].to_numpy();meta=d.loc[targets].reset_index(drop=True)
 if len(targets)==0:raise ValueError("empty destruction support")
 def score_with_map(mapping):
  mx=mv[mapping[hist]].reshape(len(hist),-1);p2=predict_matrix(models["P2"],mx);p3=predict_matrix(models["P3"],np.c_[bhist,mx]);s2=np.mean((y-p2)**2,axis=1);s3=np.mean((y-p3)**2,axis=1)
  ep=pd.DataFrame({"run_id":meta.run_id,"t":meta.t,"B0":meta.B0,"M1":d.M1.to_numpy()[mapping[targets]],"P3":s3}).groupby(["run_id","t"],as_index=False).agg({"B0":"median","M1":"first","P3":"median"})
  full=apply_fusion(ep,cal).Full.mean();return float(s2.mean()),float(s3.mean()),float(full)
 identity=np.arange(len(d));basevals=score_with_map(identity);raw=[]
 for rep in range(repetitions):
  mapping=identity.copy();rng=np.random.default_rng(seed+rep)
  for _,idx in d.groupby(["run_id","prn"],sort=False).groups.items():
   ids=np.asarray(list(idx));blocks=[ids[i:i+8] for i in range(0,len(ids),8)];order=rng.permutation(len(blocks));mapping[ids]=np.concatenate([blocks[j] for j in order])
  vals=score_with_map(mapping)
  for j,k in enumerate(("P2","P3","Full")):raw.append({"region_key":region_key,"model":k,"replicate":rep,"seed":seed+rep,"aligned":basevals[j],"shuffled":vals[j],"delta":basevals[j]-vals[j],"target_dimensions":9 if k!="Full" else "fusion","predictor_scoring_called":True,"predictor_calls_per_replicate":2,"support_preserved":True,"marginals_preserved":True})
 return raw

def run_regime(root,regime,domain,ck,base_models=None,repetitions=199):
 if regime=="S0":train=synth_frame(root,domain,"train",ck);valbase=synth_frame(root,domain,"validation",ck)
 else:
  allclean=real_paths(root,domain,"cleanStatic",ck);train=allclean[(allclean.t>=0)&(allclean.t<240)];valbase=allclean[(allclean.t>=250)&(allclean.t<330)]
 models,details=fit_predictors(train,valbase,base_models if regime=="S1" else None)
 valpred=score_predictors(valbase,models,True);valepoch=epoch_base(valpred);cal=fit_fusion(valepoch);valepoch=apply_fusion(valepoch,cal)
 cleanbase=real_paths(root,domain,"cleanStatic",ck);testbase=cleanbase[cleanbase.t>=340];testpred=score_predictors(testbase,models,True);testepoch=apply_fusion(epoch_base(testpred),cal)
 return models,details,cal,valbase,valpred,valepoch,testbase,testpred,testepoch

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("artifacts/clif_ip_synthetic_normal_r4"));ap.add_argument("--permutations",type=int,default=199);a=ap.parse_args()
 if a.permutations!=199:raise SystemExit("final requires 199 destruction replicates")
 (a.root/"per_epoch").mkdir(exist_ok=True);(a.root/"predictions").mkdir(exist_ok=True)
 scenarios=[];predrows=[];gaps=[];groupsummaries=[];rawperm=[];model_cache={};calibration_summary={};epoch_files=[]
 plans=[("S0","SYN-OAK"),("S1","SYN-OAK"),("S0","SYN-TEX"),("S1","SYN-TEX"),("R0","SYN-TEX")]
 results={}
 for regime,domain in plans:
  ck=torch.load(a.root/"models"/f"{regime}_{domain}.pt",map_location="cpu",weights_only=False)
  base=model_cache.get(("S0",domain));models,details,cal,valbase,valpred,vale,testbase,testpred,teste=run_regime(a.root,regime,domain,ck,base,a.permutations);model_cache[(regime,domain)]=models
  calibration_summary[f"{regime}:{domain}"]={"fit_rows":cal["fit_rows"],"source":cal["source"],"concordance":cal["concordance"],"component_mean":cal["component_mean"],"component_scale":cal["component_scale"]}
  for x in details:predrows.append({"regime":regime,"target_domain":domain,"split":"fit_audit","na_reason":"clean split metrics/support hash/per-tap errors not applicable to fit-audit row",**x})
  predrows.extend(split_prediction_metrics(valpred,regime,domain,"clean_validation"));predrows.extend(split_prediction_metrics(testpred,regime,domain,"independent_clean_test"))
  for split,p,e in (("clean_validation",valpred,vale),("clean_test",testpred,teste)):
   pf=a.root/"predictions"/f"{regime}_{domain}_{split}.csv";ef=a.root/"per_epoch"/f"{regime}_{domain}_{split}.csv";p.to_csv(pf,index=False);e.to_csv(ef,index=False);epoch_files.append(ef)
  names=("os1","os2","os3","os4") if domain=="SYN-OAK" else ("DS1","DS2","DS3","DS4");attacks={};rawattacks={}
  for sc in names:
   x=real_paths(a.root,domain,sc,ck);p=score_predictors(x,models,True);e=apply_fusion(epoch_base(p),cal);attacks[sc]=e;rawattacks[sc]=x
   pf=a.root/"predictions"/f"{regime}_{domain}_{sc}.csv";ef=a.root/"per_epoch"/f"{regime}_{domain}_{sc}.csv";p.to_csv(pf,index=False);e.to_csv(ef,index=False);epoch_files.append(ef)
  scenarios.extend(scenario_metrics(vale,teste,attacks,regime,domain))
  syn=synth_frame(a.root,domain,"synthetic_test",ck);real=testbase
  for group,cols in (("B0_residual_9D",[f"r{i}" for i in range(9)]),("M1_innovation",[c for c in syn if c.startswith("m") and c[1:].isdigit()])):
   for source,frame in (("synthetic",syn),("real",real)):
    gf=frame.drop_duplicates(["run_id","t"]).copy() if group.startswith("M1") else frame.copy();gf["block_id"]=(gf.t/4).astype(int)
    for (run,block_id),block_frame in gf.groupby(["run_id","block_id"]):groupsummaries.append({"regime":regime,"target_domain":domain,"feature_group":group,"source":source,"run_id":run,"block_id":block_id,"samples":len(block_frame),"rmse":float(np.sqrt(np.mean(block_frame[cols].to_numpy()**2))),"independence_note":"8-epoch block summary; blocks within one recording are not independent recording replicates"})
   # M1 must be one row per recording/time, never duplicated over PRNs.
   sa=syn.drop_duplicates(["run_id","t"])[cols].to_numpy() if group.startswith("M1") else syn[cols].to_numpy();ra=real.drop_duplicates(["run_id","t"])[cols].to_numpy() if group.startswith("M1") else real[cols].to_numpy();rng=np.random.default_rng(91)
   if len(sa)>1200:sa=sa[rng.choice(len(sa),1200,replace=False)]
   if len(ra)>1200:ra=ra[rng.choice(len(ra),1200,replace=False)]
   dg=domain_gap(sa,ra);gaps.append({"regime":regime,"target_domain":domain,"feature_group":group,"unit":"unique_(run_id,t)" if group.startswith("M1") else "PRN residual row","synthetic_rows":len(sa),"real_rows":len(ra),"uncertainty":"NA","uncertainty_na_reason":"only one real cleanStatic recording; block summaries retained but recording-level CI not estimable","mmd_label":"exploratory unbiased U-statistic; negative values permitted","agc":"NA","agc_na_reason":"no receiver-independent AGC telemetry in Method-A/M1 exports","old_comparator":5.5,**dg})
  regions=[("clean_test",testbase)]
  for sc,x in rawattacks.items():
   mask=evaluation_masks(x.available_s,domain);regions.extend([(f"{sc}_pre",x[mask["stable_pre"]]),(f"{sc}_post",x[mask["established_post"]])])
  for region,x in regions:
   if len(x)>LAG+8:rawperm.extend(alignment_destruction(x,models,cal,199,73,f"{regime}:{domain}:{region}"))
 # Immutable R3 OAK rows are explicitly segregated as historical/noncomparable.
 r3=pd.read_csv(ROOT/"artifacts/clif_ip_cross_layer_r3/scenario_metrics.csv")
 for _,r in r3.iterrows():scenarios.append({"regime":"R0","target_domain":"SYN-OAK",**r.to_dict(),"comparison_scope":"historical_r3_noncomparable","na_reason":"exact immutable R3 result; not R4 timing/training protocol"})
 raw=pd.DataFrame(rawperm);summary={"schema":"clif-ip.synthetic-normal.r4.alignment.v3","repetitions":199,"p_value_resolution":.005,"actual_predictor_rescoring":True,"target_dimensions":9,"results":{}}
 for key,g in raw.groupby(["region_key","model"]):
  delta=g.delta.to_numpy();summary["results"][":".join(key)]={"aligned":float(g.aligned.iloc[0]),"shuffled_mean":float(g.shuffled.mean()),"delta_mean":float(delta.mean()),"delta_ci":[float(np.quantile(delta,.025)),float(np.quantile(delta,.975))],"p_value":float((1+np.sum(delta>=0))/200),"replicates":199,"support_preserved":bool(g.support_preserved.all()),"marginals_preserved":bool(g.marginals_preserved.all())}
 pd.DataFrame(predrows).to_csv(a.root/"predictor_comparison.csv",index=False);pd.DataFrame(scenarios).to_csv(a.root/"scenario_metrics.csv",index=False);pd.DataFrame(gaps).to_csv(a.root/"domain_gap_metrics.csv",index=False);pd.DataFrame(groupsummaries).to_csv(a.root/"domain_gap_group_summaries.csv",index=False);raw.to_csv(a.root/"alignment_destruction_raw_metrics.csv",index=False)
 (a.root/"alignment_destruction_metrics.json").write_text(json.dumps(summary,indent=2)+"\n");(a.root/"evaluation_provenance.json").write_text(json.dumps({"inputs":INPUT_PROVENANCE,"calibration":calibration_summary,"per_epoch_files":{str(p.relative_to(a.root)):sha(p) for p in epoch_files}},indent=2)+"\n")
if __name__=="__main__":main()
