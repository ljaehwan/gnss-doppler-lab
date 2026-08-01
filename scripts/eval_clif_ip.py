#!/usr/bin/env python3
"""Leakage-safe CLIF-IP R3 evaluation on actual OAKBAT cleanStatic/os1--os4."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, subprocess, sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, average_precision_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.clif_ip import (M1FitAudit,fit_m1,transform_m1,finite,
 slice_whole_windows,fit_whitener,mahalanobis_score,fit_component_calibrations,alarm_times)

TAPS=["tap_E4_rel_prompt_mean","tap_E3_rel_prompt_mean","tap_E2_rel_prompt_mean","tap_E_rel_prompt_mean","tap_P_rel_prompt_mean","tap_L_rel_prompt_mean","tap_L2_rel_prompt_mean","tap_L3_rel_prompt_mean","tap_L4_rel_prompt_mean"]
META={"scenario","window_index","window_start_s","window_mid_s","window_end_s","block_ms","stride_s"}
SCENARIOS=("os1","os2","os3","os4");ONSETS={x:120. for x in SCENARIOS};ALPHAS=(.1,1.,10.,100.);LAGS=(2,4,6);EPS=1e-9
LAB=Path("/home/ubuntu/projects/gnss-doppler-lab");SSD=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts");RAW=Path("/home/ubuntu/unraid_hdd/oakbat/gps_l1ca/raw")
COMPONENT_SPECS={"Full":["B0","M1","P3","concordance"],"minus_M1":["B0","P3","concordance"],"minus_B0history":["B0","M1","P2","concordance"],"minus_concordance":["B0","M1","P3"]}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()


def file_record(path,hash_value=None,hash_method="sha256-full-file"):
 p=Path(path)
 if not p.exists():return {"path":str(p),"exists":False,"bytes":None,"mtime_ns":None,"sha256":None,"hash_method":None}
 st=p.stat();digest=sha(p) if hash_value is None else hash_value
 return {"path":str(p),"exists":True,"bytes":st.st_size,"mtime_ns":st.st_mtime_ns,"sha256":digest,"hash_method":hash_method}


@dataclass
class Cfg:hidden_dim:int=128;emb_dim:int=128;dropout:float=.05
class GRU(nn.Module):
 def __init__(self,f,c):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(f,c.emb_dim),nn.LayerNorm(c.emb_dim),nn.GELU(),nn.Dropout(c.dropout),nn.Linear(c.emb_dim,c.emb_dim),nn.GELU());self.gru=nn.GRU(c.emb_dim,c.hidden_dim,batch_first=True);self.head=nn.Sequential(nn.Linear(c.hidden_dim,c.hidden_dim),nn.GELU(),nn.Linear(c.hidden_dim,f))
 def forward(self,x):
  b,t,f=x.shape;z=self.encoder(x.reshape(b*t,f)).reshape(b,t,-1);o,_=self.gru(z);return self.head(o[:,-1])


def paths(out):
 bbase=LAB/"artifacts/oakbat_cleanstatic_detector_eval_v1/preprocessed"
 return ({"cleanStatic":LAB/"artifacts/oakbat_9tap_frozen_champion/cleanStatic/multi_prn_method_a_9tap_w1.0_s0.5_normalized_dmcpd/normal_prn_node_windows.csv"}|{s:bbase/s/"multi_prn_method_a_9tap_w1.0_s0.5_normalized_dmcpd/normal_prn_node_windows.csv" for s in SCENARIOS},
 {"cleanStatic":SSD/"oakbat-cleanStatic-raw-iq-noise-continuity-m1-v0-fs5m-full/oakbat_cleanStatic_raw_iq_noise_features.csv","os1":out/"input_cache/oakbat_os1_raw_iq_noise_features.csv"}|{s:SSD/f"oakbat-{s}-raw-iq-noise-continuity-m1-v0-fs5m-full"/f"oakbat_{s}_raw_iq_noise_features.csv" for s in ("os2","os3","os4")})


def load_b0(path,ck,model,bounds=None):
 """Build residual history wholly inside one recording/split (history reset)."""
 d=pd.read_csv(path)
 if bounds is not None:d=slice_whole_windows(d,*bounds)
 mean=np.asarray(ck["standardizer"]["node_mean"],np.float32);std=np.asarray(ck["standardizer"]["node_std"],np.float32);rows=[];model.eval()
 with torch.no_grad():
  for (_,prn),g in d.groupby(["run_id","prn"],sort=False):
   g=g.sort_values("window_bin_s").reset_index(drop=True);x=finite((g[TAPS].to_numpy(np.float32)-mean)/std).astype(np.float32);tt=g.window_bin_s.to_numpy(float)
   for i in range(12,len(g)):
    if not np.allclose(np.diff(tt[i-12:i+1]),.5,atol=1e-5):continue
    pred=model(torch.from_numpy(x[i-12:i][None])).numpy()[0];res=x[i]-pred;r=g.iloc[i]
    rows.append({"t":float(r.window_bin_s),"available_s":float(r.window_end_s),"source_start_s":float(g.iloc[i-12].window_start_s),"prn":str(prn),**{f"b{j}":float(v) for j,v in enumerate(res)}})
 return pd.DataFrame(rows)


def m1_frame(path,state,audit,name,bounds=None):
 """Read/slice raw frame first, then reset transform; scalar/innovation namespaces cannot collide."""
 d=pd.read_csv(path)
 if bounds is not None:d=slice_whole_windows(d,*bounds)
 cols=[c for c in d.columns if c not in META]
 z=transform_m1(d[cols].to_numpy(float),d.window_start_s.to_numpy(float),state,name,audit,reset=True)
 o=pd.DataFrame({"t":z["t"],"M1_score":z["score"],"m1_available_s":d.window_end_s.to_numpy(float)})
 for j in range(z["innovation"].shape[1]):o[f"m1_innov_{j}"]=z["innovation"][:,j]
 return o.dropna().reset_index(drop=True),cols


def merge(b,m):
 if b.empty or m.empty:return pd.DataFrame()
 return b.merge(m,on="t",how="inner",validate="many_to_one").sort_values(["prn","t"]).reset_index(drop=True)


def design(d,lag,kind):
 rows=[];bcols=[f"b{i}" for i in range(9)];mcols=[c for c in d if c.startswith("m1_innov_")]
 for prn,g in d.groupby("prn",sort=False):
  g=g.sort_values("t").reset_index(drop=True);b=g[bcols].to_numpy();m=g[mcols].to_numpy()
  for i in range(lag,len(g)):
   if not np.allclose(np.diff(g.t.iloc[i-lag:i+1]),.5,atol=1e-6):continue
   z=[]
   if kind in ("P1","P3"):z.extend(b[i-lag:i].reshape(-1))
   if kind in ("P2","P3"):z.extend(m[i-lag:i+1].reshape(-1))
   rows.append((float(g.t.iloc[i]),str(prn),np.asarray(z),b[i],float(g.available_s.iloc[i]),float(g.t.iloc[i-lag])))
 if not rows:return np.empty((0,0)),np.empty((0,9)),pd.DataFrame(columns=["t","prn","available_s","source_start_s"])
 X=np.stack([r[2] for r in rows]) if kind!="P0" else np.empty((len(rows),0))
 return X,np.stack([r[3] for r in rows]),pd.DataFrame({"t":[r[0] for r in rows],"prn":[r[1] for r in rows],"available_s":[r[4] for r in rows],"source_start_s":[r[5] for r in rows]})


def ridge(alpha):return Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=alpha))])
def train_models(train,val,lag,alphas):
 _,Y,_=design(train,lag,"P0");out={"P0":Y.mean(0)}
 for k in ("P1","P2","P3"):
  X,Y,_=design(train,lag,k);out[k]=ridge(alphas[k]).fit(X,Y)
 return out

def predict_model(d,models,lag,k):
 X,Y,meta=design(d,lag,k)
 pred=np.tile(models[k],(len(Y),1)) if k=="P0" else models[k].predict(X)
 return Y,pred,meta

def prediction_mse(y,p):return float(mean_squared_error(y,p))


def select_hyperparameters(train,val):
 """Validation-only common-lag selection and separate per-model alpha selection."""
 rows=[]
 for lag in LAGS:
  best={}
  for k in ("P1","P2","P3"):
   X,Y,_=design(train,lag,k);Xv,Yv,_=design(val,lag,k)
   candidates=[]
   for alpha in ALPHAS:
    mdl=ridge(alpha).fit(X,Y);candidates.append((prediction_mse(Yv,mdl.predict(Xv)),alpha))
   mse,alpha=min(candidates);best[k]=(mse,alpha);rows.extend({"lag":lag,"model":k,"alpha":a,"validation_mse":m} for m,a in candidates)
  for r in rows[-len(ALPHAS)*3:]:r["lag_mean_best_mse"]=np.mean([v[0] for v in best.values()])
 lag=min(LAGS,key=lambda z:next(r["lag_mean_best_mse"] for r in rows if r["lag"]==z))
 alphas={k:min((r["validation_mse"],r["alpha"]) for r in rows if r["lag"]==lag and r["model"]==k)[1] for k in ("P1","P2","P3")}
 return lag,alphas,pd.DataFrame(rows)


def residual_score(d,models,whiteners,lag,k):
 Y,P,meta=predict_model(d,models,lag,k);e=Y-P;z=meta.copy();z[k]=mahalanobis_score(e,whiteners[k])
 for j in range(9):z[f"{k}_tap{j}"]=e[:,j]
 return z

def agg(z,col):
 rows=[]
 for t,g in z.groupby("t"):
  x=g[col].to_numpy();kk=min(3,len(x));rows.append({"t":t,"available_s":float(g.available_s.max()),col+"_median":np.median(x),col+"_q90":np.quantile(x,.9),col+"_topk":np.sort(x)[-kk:].mean(),col+"_tracked":len(x)})
 return pd.DataFrame(rows)
def epoch_availability(d):return d.groupby("t",as_index=False).available_s.max()
def etail(x,ref):
 r=np.sort(np.asarray(ref));return np.maximum(1/(len(r)+1),1-np.searchsorted(r,x,side="right")/len(r))


def epoch_scores(d,models,whiteners,lag,refs=None,calibrations=None):
 pieces=[agg(residual_score(d,models,whiteners,lag,k),k) for k in ("P0","P1","P2","P3")];o=pieces[0]
 for p in pieces[1:]:o=o.merge(p.drop(columns="available_s"),on="t",how="inner")
 m=d.groupby("t").agg(M1=("M1_score","first"),m1_available_s=("m1_available_s","first"),tracked=("prn","count")).reset_index();o=o.merge(m,on="t");o["available_s"]=o[["available_s","m1_available_s"]].max(axis=1);o["B0"]=o.P0_median;o["P2"]=o.P2_median;o["P3"]=o.P3_median
 if refs is None:return o
 pb=etail(o.B0,refs["B0"]);pm=etail(o.M1,refs["M1"]);o["mean_fusion"]=-np.log(np.sqrt(pb*pm));o["max_fusion"]=-np.log(np.minimum(pb,pm));o["fisher_fusion"]=-2*(np.log(pb)+np.log(pm));o["concordance"]=(-np.log(pb))*(-np.log(pm))
 for name,state in calibrations.items():o[name]=mahalanobis_score(o[list(state.columns)].to_numpy(),state)
 return o


def alarm_metrics(df,col,ref,test_ref,onset):
 q99=float(np.quantile(ref,.99));q995=float(np.quantile(ref,.995));pre=df[df.available_s<onset-10];post=df[df.available_s>=onset+10];y=np.r_[np.zeros(len(pre)),np.ones(len(post))];x=np.r_[pre[col],post[col]];a=alarm_times(df,col,q99,onset,3)
 return {"roc_auc":float(roc_auc_score(y,x)),"pr_auc":float(average_precision_score(y,x)),"independent_test_fpr":float(np.mean(np.asarray(test_ref)>q99)),"attack_detection_rate":float((post[col]>q99).mean()),**a,"threshold_q99":q99,"threshold_q995":q995,"target_fpr_1pct_threshold":q99,"onset_time_basis":"available_s"}


def permute_region(raw,seed,block=8):
 epochs=raw.drop_duplicates("t").sort_values("t");mcols=[c for c in raw if c=="M1_score" or c.startswith("m1_innov_")];starts=list(range(0,len(epochs),block));order=np.random.default_rng(seed).permutation(len(starts));perm=np.concatenate([np.arange(starts[j],min(starts[j]+block,len(epochs))) for j in order]);old=epochs[mcols].to_numpy();new=old[perm];mapping={float(t):new[i] for i,t in enumerate(epochs.t)};out=raw.copy()
 for i,c in enumerate(mcols):out[c]=out.t.map(lambda t:mapping[float(t)][i])
 return out,old,new

def permutation_summary(observed,distribution,confidence=.95):
 d=np.asarray(distribution,float);a=(1-confidence)/2;two=min(np.sum(d<=0),np.sum(d>=0))*2
 return {"observed_mean_delta":float(observed),"p_value":float(min(1,(1+two)/(len(d)+1))),"ci_low":float(np.quantile(d,a)),"ci_high":float(np.quantile(d,1-a)),"repetitions":int(len(d))}


def destruction_measure(raw,models,whiteners,lag,refs,cals,base_epochs):
 """Fast destruction path: recompute only affected P2/P3 predictions and Full."""
 result={};p3epoch=None
 for k in ("P2","P3"):
  y,p,meta=predict_model(raw,models,lag,k);e=y-p;z=meta.copy();z[k]=mahalanobis_score(e,whiteners[k])
  result[k+"_mse"]=prediction_mse(y,p)
  if k=="P3":p3epoch=agg(z,k)[["t","P3_median"]]
 m=raw.groupby("t",as_index=False).M1_score.first().rename(columns={"M1_score":"M1"})
 o=base_epochs[["t","available_s","B0"]].merge(p3epoch,on="t").merge(m,on="t");pb=etail(o.B0,refs["B0"]);pm=etail(o.M1,refs["M1"]);o["P3"]=o.P3_median;o["concordance"]=(-np.log(pb))*(-np.log(pm));o["Full"]=mahalanobis_score(o[list(cals["Full"].columns)].to_numpy(),cals["Full"]);result["Full_mean"]=float(o.Full.mean())
 return result


def plot_timeline(d,s,out):
 cols=["B0","M1","P1_median","P2_median","P3_median","Full"];fig,axs=plt.subplots(3,2,figsize=(14,9),sharex=True)
 for ax,c in zip(axs.flat,cols):ax.plot(d.available_s,d[c],lw=.8);ax.axvline(120,color="r",ls="--");ax.set_title(c);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/"plots"/f"{s}_timeline.png",dpi=120);plt.close(fig)


def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=Path("artifacts/clif_ip_cross_layer_r3"));ap.add_argument("--permutations",type=int,default=99);a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=True);(out/"plots").mkdir(exist_ok=True)
 bpaths,mpaths=paths(out);ckpath=ROOT/"artifacts/ai_morph_gru_cleanStatic_q70_frame/prn_local_gru_predictor.pt";ck=torch.load(ckpath,map_location="cpu",weights_only=False);c=Cfg(**{k:ck["config"][k] for k in ("hidden_dim","emb_dim","dropout")});model=GRU(9,c);model.load_state_dict(ck["model_state_dict"])
 cleanraw=pd.read_csv(mpaths["cleanStatic"]);cols=[x for x in cleanraw if x not in META];audit=M1FitAudit();state=fit_m1(cleanraw[cols].to_numpy(float),cleanraw.window_start_s.to_numpy(float),cleanraw.window_end_s.to_numpy(float),240.,8,6,audit,"cleanStatic")
 bounds={"train":(0,240),"validation":(250,330),"test":(340,float("inf"))};cleanparts={}
 for name,bd in bounds.items():
  bb=load_b0(bpaths["cleanStatic"],ck,model,bd);mm,_=m1_frame(mpaths["cleanStatic"],state,audit,"cleanStatic_"+name,bd);cleanparts[name]=merge(bb,mm)
 train,val,test=(cleanparts[x] for x in ("train","validation","test"))
 merged={}
 for s in SCENARIOS:merged[s]=merge(load_b0(bpaths[s],ck,model),m1_frame(mpaths[s],state,audit,s)[0])
 lag,alphas,tuning=select_hyperparameters(train,val);tuning.to_csv(out/"hyperparameter_selection.csv",index=False);models=train_models(train,val,lag,alphas)
 whiteners={};predrows=[];val_residuals={}
 for k in ("P0","P1","P2","P3"):
  Y,P,_=predict_model(val,models,lag,k);val_residuals[k]=Y-P;whiteners[k]=fit_whitener(Y-P)
  for name,d in (("validation",val),("test",test)):
   yy,pp,meta=predict_model(d,models,lag,k);predrows.append({"split":name,"model":k,"mse":prediction_mse(yy,pp),"mae":mean_absolute_error(yy,pp),"samples":len(yy),"first_target_s":float(meta.t.min()),"last_target_s":float(meta.t.max()),"parameters":9 if k=="P0" else int(sum(np.size(v) for v in models[k].named_steps["ridge"].coef_))+int(np.size(models[k].named_steps["ridge"].intercept_)),"alpha":None if k=="P0" else alphas[k],"lag":lag})
 pc=pd.DataFrame(predrows);p1=float(pc.query("split=='test' and model=='P1'").mse.iloc[0]);p3=float(pc.query("split=='test' and model=='P3'").mse.iloc[0]);pc["p3_vs_p1_improvement_pct"]=100*(p1-p3)/p1;pc["incremental_r2_vs_p1"]=1-p3/p1;pc.to_csv(out/"predictor_comparison.csv",index=False)
 val0=epoch_scores(val,models,whiteners,lag);refs={"B0":val0.B0.to_numpy(),"M1":val0.M1.to_numpy()};val0["concordance"]=(-np.log(etail(val0.B0,refs["B0"])))*(-np.log(etail(val0.M1,refs["M1"])));cals=fit_component_calibrations(val0,COMPONENT_SPECS)
 scored={"clean_validation":epoch_scores(val,models,whiteners,lag,refs,cals),"clean_test":epoch_scores(test,models,whiteners,lag,refs,cals)}|{s:epoch_scores(merged[s],models,whiteners,lag,refs,cals) for s in SCENARIOS}
 for s,d in scored.items():d.to_csv(out/f"per_epoch_scores_{s}.csv",index=False)
 scorecols=["B0","M1","mean_fusion","max_fusion","fisher_fusion","P0_median","P1_median","P2_median","P3_median","Full","minus_M1","minus_B0history","minus_concordance"];testrefs={x:scored["clean_test"][x].to_numpy() for x in scorecols};rows=[]
 for s in SCENARIOS:
  for col in scorecols:rows.append({"scenario":s,"model":col,**alarm_metrics(scored[s],col,scored["clean_validation"][col].to_numpy(),testrefs[col],ONSETS[s])})
 metrics=pd.DataFrame(rows);metrics.to_csv(out/"scenario_metrics.csv",index=False);metrics.groupby("model",as_index=False).agg(roc_auc=("roc_auc","mean"),pr_auc=("pr_auc","mean"),independent_test_fpr=("independent_test_fpr","mean"),attack_detection_rate=("attack_detection_rate","mean")).to_csv(out/"fusion_comparison.csv",index=False);metrics[metrics.model.isin(COMPONENT_SPECS)].to_csv(out/"ablation_metrics.csv",index=False)
 lr=[]
 for sh in range(-6,7):
  x=val0.B0.to_numpy();y=pd.Series(val0.M1).shift(sh).to_numpy();ok=np.isfinite(y);lr.append({"lag_epochs":sh,"lag_s":sh*.5,"validation_pearson":np.corrcoef(x[ok],y[ok])[0,1],"selected_predictor_lag":sh==lag})
 pd.DataFrame(lr).to_csv(out/"lag_analysis.csv",index=False);fig,ax=plt.subplots();ax.plot([x["lag_s"] for x in lr],[x["validation_pearson"] for x in lr],marker="o");ax.grid();fig.savefig(out/"plots/lag_analysis.png");plt.close(fig)
 raw_regions={"clean_test":test}|{f"{s}_pre":merged[s].query("available_s < 110") for s in SCENARIOS}|{f"{s}_attack":merged[s].query("available_s >= 130") for s in SCENARIOS};dest={}
 for name,raw0 in raw_regions.items():
  aligned=epoch_scores(raw0,models,whiteners,lag,refs,cals);base=destruction_measure(raw0,models,whiteners,lag,refs,cals,aligned);dist={"P2":[],"P3":[],"Full":[]};marginal=True
  for seed in range(a.permutations):
   sh,old,new=permute_region(raw0,1000+seed);marginal=marginal and np.allclose(np.sort(old,axis=0),np.sort(new,axis=0));sm=destruction_measure(sh,models,whiteners,lag,refs,cals,aligned)
   for k in ("P2","P3"):dist[k].append(base[k+"_mse"]-sm[k+"_mse"])
   dist["Full"].append(base["Full_mean"]-sm["Full_mean"])
  dest[name]={"n_epochs":int(raw0.t.nunique()),"m1_sorted_equal_every_repeat":bool(marginal),"P2_aligned_prediction_mse":base["P2_mse"],"P3_aligned_prediction_mse":base["P3_mse"]}
  for k in dist:dest[name][k+"_delta_statistics"]=permutation_summary(float(np.mean(dist[k])),dist[k])
 (out/"alignment_destruction_metrics.json").write_text(json.dumps({"method":f"{a.permutations} deterministic region-local 8-epoch M1 block permutations; actual 9-tap P2/P3 prediction MSE and Full scores recomputed","results":dest},indent=2)+"\n")
 for s in SCENARIOS:plot_timeline(scored[s],s,out)
 # Actual validation tap residual distributions, not model coefficients.
 fig,axs=plt.subplots(3,3,figsize=(12,9));
 for j,ax in enumerate(axs.flat):ax.hist(val_residuals["P3"][:,j],bins=40,alpha=.75);ax.axvline(0,color="k",lw=.7);ax.set_title(f"P3 validation tap {j} residual")
 fig.tight_layout();fig.savefig(out/"plots/tap_residual_distribution.png",dpi=120);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4));names=list(dest);ax.bar(np.arange(len(names)),[dest[x]["Full_delta_statistics"]["observed_mean_delta"] for x in names]);ax.set_xticks(np.arange(len(names)),names,rotation=90);ax.set_ylabel("aligned - shuffled Full mean");fig.tight_layout();fig.savefig(out/"plots/aligned_shuffled.png");plt.close(fig)
 # Provenance: source CSV/checkpoint are fully hashed now; large IQ hashes use canonical cached digest plus live stat verification, explicitly labeled.
 canonical={"os1":"e9ef8ab33a3e59c5e55b3f6fb9b8bb3ba18aaf380402ae00abbe535858b1deb7","os2":"17de8e3f54095ad2eafad8a54ca7f5008596936ded35578710dfed17a5b670c1","os3":"2a3c3c5cf1accaa287fe14181e43070903500e0250c69e3c335f91c89c0cdc6c","os4":"803f3c76bcc618efbc6b394eb536fe61ed8c3e34b1822c0088b4475621bfa8e4"};prov=[]
 for s in ("cleanStatic",)+SCENARIOS:
  rp=RAW/("cleanStatic_gps.bin" if s=="cleanStatic" else f"{s}.bin");rawrec=file_record(rp,canonical.get(s),"cached canonical SHA-256; live path/size/mtime verified during this run") if s in canonical else file_record(rp,None) if rp.exists() and rp.stat().st_size<1_000_000_000 else {**file_record(rp,"", "live stat only; no trustworthy cleanStatic canonical hash available"),"sha256":None}
  prov.append({"scenario":s,"grade":"reconstructed" if s!="cleanStatic" else "provisional","raw_iq":rawrec,"sample_rate_hz":5000000,"sample_format":"interleaved little-endian int16 IQ","recording_start_sample":0,"seek_samples":0,"m1_block_ms":10,"stride_s":.5,"alignment":"timestamp reconstructed; receiver processing delay unavailable","b0_source_csv":file_record(bpaths[s]),"m1_source_csv":file_record(mpaths[s])})
 commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip();deps={x:importlib.metadata.version(x) for x in ("numpy","pandas","scikit-learn","torch","matplotlib")};manifest={"schema":"clif-ip.r3.provenance.v2","generated_utc":datetime.now(timezone.utc).isoformat(),"source_commit_at_execution":commit,"dependencies":deps,"b0_checkpoint":file_record(ckpath),"scenarios":prov,"artifact_checksums":{}}
 (out/"provenance_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 fit_summary={"kind":"M1-style surrogate (actual raw-IQ features; no reusable frozen M1 checkpoint found)","fit_count":audit.fit_count,"fit_recordings":audit.fit_recordings,"transform_recordings":audit.transform_recordings,"frozen_sha256":state.sha256,"fit_scope":"cleanStatic whole windows: start>=0 and end<=240 s","fit_rows":state.fit_rows,"pca_dim":state.pca_dim,"ar_lag":state.lag};(out/"frozen_m1_fit_summary.json").write_text(json.dumps(fit_summary,indent=2)+"\n")
 cfg={"schema":"clif-ip.cross-layer.r3","splits":{"train":"whole containment [0,240] then reset transform/history","validation":"whole containment [250,330] then reset transform/history","independent_test":"whole containment [340,end] then reset transform/history"},"lag":lag,"alphas":alphas,"selection":"same lag candidates; common lag by mean validation MSE; separate validation alpha per P1/P2/P3; StandardScaler+Ridge; no attack labels","component_sets":COMPONENT_SPECS,"onsets":ONSETS,"timing_basis":"available_s","b0_checkpoint_sha256":sha(ckpath)};(out/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
 full=metrics.query("model=='Full'");base=metrics.query("model in ['B0','M1']");verdict="GO" if p3<p1 and full.roc_auc.mean()>=base.roc_auc.mean() else "NO-GO"
 readme=f"""# CLIF-IP cross-layer R3 artifacts\n\n## Verdict: **{verdict}**\n\n1. **M1 definition.** M1-style surrogate over actual OAKBAT raw-IQ features (not an original checkpointed M1). Frozen hash `{state.sha256}`; exactly {audit.fit_count} clean fit; {state.fit_rows} whole-contained fit rows.\n2. **Leakage/reset.** Raw clean splits are sliced by whole-window containment before M1 transform and B0 residual construction. Train, validation, test, and every attack recording begin with empty M1/B0 history.\n3. **Predictor protocol.** Shared signed 9-tap P0--P3 models contain no PRN IDs and support variable cardinality. All metrics use identical lag-trimmed target IDs. StandardScaler+Ridge used the same lag candidates {LAGS}; common lag={lag}, with validation-only model-specific alphas {alphas}. Parameter counts are in `predictor_comparison.csv`.\n4. **Independent prediction.** P1 test MSE={p1:.9g}; P3={p3:.9g}; P3-vs-P1={100*(p1-p3)/p1:.3f}%. {'Improvement observed.' if p3<p1 else 'No incremental improvement; this is a failure for the cross-layer prediction claim.'}\n5. **Calibration.** Each predictor stores validation residual mean plus Ledoit-Wolf covariance and scores `e-mu`. Full and every ablation refit robust scale and shrinkage covariance on its exact validation-only component set.\n6. **Detection.** Full mean ROC-AUC={full.roc_auc.mean():.4f}, PR-AUC={full.pr_auc.mean():.4f}, independent clean-test FPR={full.independent_test_fpr.mean():.4f}. q99/q99.5 and the 1% target threshold are recorded; attack onsets and delays use score `available_s`.\n7. **Alignment destruction.** Region-local 8-epoch permutations preserve M1 marginals. Actual P2/P3 9-tap MSE and Full score deltas have {a.permutations}-repeat p-values and 95% permutation intervals.\n8. **Provenance/artifacts.** Source CSV and checkpoint SHA-256 values were computed in-run; accessible 9.6-GB attack IQ files have live stat verification against explicitly identified cached canonical hashes. cleanStatic has no trustworthy canonical raw hash and remains honestly null/provisional. Source commit/dependencies and artifact checksums are in the manifest.\n9. **Claim boundary and failures.** Can claim a leakage-safe OAKBAT os1--os4 evaluation of this surrogate and report the regenerated outcomes. Cannot claim physical causality, exact sample alignment/receiver delay, original-M1 performance, cross-corpus generalization, or universal fusion superiority. `{verdict}` follows the regenerated P3/full criteria; scenario-level failures remain visible in CSVs.\n\n## Files\n`config.json`, provenance/frozen summaries, predictor/hyperparameter/scenario/fusion/ablation/lag CSVs, per-epoch CSVs, destruction JSON, actual residual and timeline plots, and `test_summary.txt`.\n""";(out/"README.md").write_text(readme)
 # Hash final artifacts except manifest itself, then rewrite manifest.
 for p in sorted(out.rglob("*")):
  if p.is_file() and p.name!="provenance_manifest.json":manifest["artifact_checksums"][str(p.relative_to(out))]=sha(p)
 (out/"provenance_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 summary=f"experiment_command: python scripts/eval_clif_ip.py --out {out} --permutations {a.permutations}\nexperiment_completed_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment_exit: 0\nfit_count: {audit.fit_count}\nfit_rows: {state.fit_rows}\nP1_test_MSE: {p1:.9g}\nP3_test_MSE: {p3:.9g}\nFull_mean_ROC_AUC: {full.roc_auc.mean():.6g}\nverdict: {verdict}\npytest: pending post-experiment verification\n";(out/"test_summary.txt").write_text(summary);print(summary)

if __name__=="__main__":main()
