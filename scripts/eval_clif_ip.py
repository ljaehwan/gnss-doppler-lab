#!/usr/bin/env python3
"""Leakage-safe CLIF-IP R3 evaluation on actual OAKBAT cleanStatic/os1--os4."""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd
import torch
from torch import nn
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, average_precision_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.clif_ip import M1FitAudit,fit_m1,transform_m1,finite,robust,shuffle_pairing
TAPS=["tap_E4_rel_prompt_mean","tap_E3_rel_prompt_mean","tap_E2_rel_prompt_mean","tap_E_rel_prompt_mean","tap_P_rel_prompt_mean","tap_L_rel_prompt_mean","tap_L2_rel_prompt_mean","tap_L3_rel_prompt_mean","tap_L4_rel_prompt_mean"]
META={"scenario","window_index","window_start_s","window_mid_s","window_end_s","block_ms","stride_s"}
SCENARIOS=("os1","os2","os3","os4"); ONSETS={x:120. for x in SCENARIOS}; EPS=1e-9
LAB=Path("/home/ubuntu/projects/gnss-doppler-lab")
SSD=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts")
RAW=Path("/home/ubuntu/unraid_hdd/oakbat/gps_l1ca/raw")

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
@dataclass
class Cfg: hidden_dim:int=128; emb_dim:int=128; dropout:float=.05
class GRU(nn.Module):
 def __init__(self,f,c):
  super().__init__(); self.encoder=nn.Sequential(nn.Linear(f,c.emb_dim),nn.LayerNorm(c.emb_dim),nn.GELU(),nn.Dropout(c.dropout),nn.Linear(c.emb_dim,c.emb_dim),nn.GELU()); self.gru=nn.GRU(c.emb_dim,c.hidden_dim,batch_first=True); self.head=nn.Sequential(nn.Linear(c.hidden_dim,c.hidden_dim),nn.GELU(),nn.Linear(c.hidden_dim,f))
 def forward(self,x):
  b,t,f=x.shape; z=self.encoder(x.reshape(b*t,f)).reshape(b,t,-1); o,_=self.gru(z); return self.head(o[:,-1])

def paths(out):
 bbase=LAB/"artifacts/oakbat_cleanstatic_detector_eval_v1/preprocessed"
 return ({"cleanStatic":LAB/"artifacts/oakbat_9tap_frozen_champion/cleanStatic/multi_prn_method_a_9tap_w1.0_s0.5_normalized_dmcpd/normal_prn_node_windows.csv"}|{s:bbase/s/"multi_prn_method_a_9tap_w1.0_s0.5_normalized_dmcpd/normal_prn_node_windows.csv" for s in SCENARIOS},
 {"cleanStatic":SSD/"oakbat-cleanStatic-raw-iq-noise-continuity-m1-v0-fs5m-full/oakbat_cleanStatic_raw_iq_noise_features.csv","os1":out/"input_cache/oakbat_os1_raw_iq_noise_features.csv"}|{s:SSD/f"oakbat-{s}-raw-iq-noise-continuity-m1-v0-fs5m-full"/f"oakbat_{s}_raw_iq_noise_features.csv" for s in ("os2","os3","os4")})

def load_b0(path,ck,model):
 d=pd.read_csv(path); mean=np.asarray(ck["standardizer"]["node_mean"],np.float32); std=np.asarray(ck["standardizer"]["node_std"],np.float32); rows=[]; model.eval()
 with torch.no_grad():
  for (_,prn),g in d.groupby(["run_id","prn"],sort=False):
   g=g.sort_values("window_bin_s").reset_index(drop=True); x=finite((g[TAPS].to_numpy(np.float32)-mean)/std).astype(np.float32); tt=g.window_bin_s.to_numpy(float)
   for i in range(12,len(g)):
    if not np.allclose(np.diff(tt[i-12:i+1]),.5,atol=1e-5): continue
    pred=model(torch.from_numpy(x[i-12:i][None])).numpy()[0]; res=x[i]-pred; r=g.iloc[i]
    rows.append({"t":float(r.window_bin_s),"available_s":float(r.window_end_s),"prn":str(prn),**{f"b{j}":float(v) for j,v in enumerate(res)}})
 return pd.DataFrame(rows)

def m1_frame(path,state,audit,name):
 d=pd.read_csv(path); cols=[c for c in d.columns if c not in META]; z=transform_m1(d[cols].to_numpy(float),d.window_start_s.to_numpy(float),state,name,audit); o=pd.DataFrame({"t":z["t"],"m1":z["score"],"m1_available_s":d.window_end_s})
 for j in range(z["innovation"].shape[1]): o[f"m{j}"]=z["innovation"][:,j]
 return o.dropna().reset_index(drop=True),cols

def merge(b,m): return b.merge(m,on="t",how="inner",validate="many_to_one").sort_values(["prn","t"]).reset_index(drop=True)
def split(d,lo,hi):
 # whole support: B0 target window starts t and ends available_s; M1 block starts t and ends m1_available_s
 return d[(d.t>=lo)&(d.available_s<=hi)&(d.m1_available_s<=hi)].copy()
def design(d,lag,kind):
 rows=[]
 for prn,g in d.groupby("prn",sort=False):
  g=g.sort_values("t").reset_index(drop=True); b=g[[f"b{i}" for i in range(9)]].to_numpy(); m=g[[c for c in g if c.startswith("m") and c[1:].isdigit()]].to_numpy()
  for i in range(lag,len(g)):
   if not np.allclose(np.diff(g.t.iloc[i-lag:i+1]),.5,atol=1e-6): continue
   z=[]
   if kind in ("P1","P3"): z.extend(b[i-lag:i].reshape(-1))
   if kind in ("P2","P3"): z.extend(m[i-lag:i+1].reshape(-1))
   rows.append((float(g.t.iloc[i]),str(prn),np.asarray(z),b[i]))
 if not rows:return np.empty((0,0)),np.empty((0,9)),pd.DataFrame(columns=["t","prn"])
 return np.stack([r[2] for r in rows]),np.stack([r[3] for r in rows]),pd.DataFrame({"t":[r[0] for r in rows],"prn":[r[1] for r in rows]})
def train_models(train,val,lag,alpha):
 out={}; ymean=train[[f"b{i}" for i in range(9)]].mean().to_numpy()
 out["P0"]=ymean
 for k in ("P1","P2","P3"):
  X,Y,_=design(train,lag,k); out[k]=Ridge(alpha=alpha).fit(X,Y)
 return out
def predict_model(d,models,lag,k):
 if k=="P0":
  Y=d[[f"b{i}" for i in range(9)]].to_numpy(); meta=d[["t","prn"]].reset_index(drop=True); pred=np.tile(models[k],(len(Y),1))
 else:X,Y,meta=design(d,lag,k); pred=models[k].predict(X)
 return Y,pred,meta

def residual_score(d,models,covs,lag,k):
 Y,P,meta=predict_model(d,models,lag,k); e=Y-P; score=np.sqrt(np.maximum(0,np.einsum("ij,jk,ik->i",e,np.linalg.pinv(covs[k]),e)))
 z=meta.copy(); z[k]=score
 for j in range(9):z[f"{k}_tap{j}"]=e[:,j]
 return z

def agg(z,col):
 rows=[]
 for t,g in z.groupby("t"):
  x=g[col].to_numpy(); kk=min(3,len(x)); rows.append({"t":t,col+"_median":np.median(x),col+"_q90":np.quantile(x,.9),col+"_topk":np.sort(x)[-kk:].mean(),col+"_tracked":len(x)})
 return pd.DataFrame(rows)
def etail(x,ref):
 r=np.sort(np.asarray(ref)); return np.maximum(1/(len(r)+1),1-np.searchsorted(r,x,side="right")/len(r))
def calibrate_components(base,val):
 cols=["B0","M1","P3","concordance"]
 mu,sc=robust(val[cols].to_numpy()); cov=LedoitWolf().fit((val[cols].to_numpy()-mu)/sc).covariance_
 return mu,sc,cov

def epoch_scores(d,models,covs,lag,refs=None):
 pieces=[]
 for k in ("P0","P1","P2","P3"):pieces.append(agg(residual_score(d,models,covs,lag,k),k))
 o=pieces[0]
 for p in pieces[1:]:o=o.merge(p,on="t",how="inner")
 # B0 is P0 per-PRN marginal; M1 frozen marginal. Concordance is causal product of robust marginals.
 m=d.groupby("t").agg(M1=("m1","first"),tracked=("prn","count")).reset_index(); o=o.merge(m,on="t"); o["B0"]=o.P0_median
 if refs is None: return o
 b_ref,m_ref=refs["B0"],refs["M1"]; pb=etail(o.B0,b_ref); pm=etail(o.M1,m_ref); o["mean_fusion"]=-np.log(np.sqrt(pb*pm));o["max_fusion"]=-np.log(np.minimum(pb,pm));o["fisher_fusion"]=-2*(np.log(pb)+np.log(pm));o["concordance"]=(-np.log(pb))*(-np.log(pm));
 z=(o[["B0","M1","P3_median","concordance"]].to_numpy()-refs["mu"])/refs["scale"]; inv=np.linalg.pinv(refs["cov"]); o["Full"]=np.sqrt(np.maximum(0,np.einsum("ij,jk,ik->i",z,inv,z)))
 # requested ablations
 o["minus_M1"]=o[["B0","P3_median","concordance"]].mean(axis=1);o["minus_B0history"]=o[["B0","M1","P2_median","concordance"]].mean(axis=1);o["minus_concordance"]=o[["B0","M1","P3_median"]].mean(axis=1)
 return o

def alarm_metrics(df,col,ref,onset):
 q99=float(np.quantile(ref,.99)); q995=float(np.quantile(ref,.995)); pre=df[df.t<onset-10]; post=df[df.t>=onset+10]; y=np.r_[np.zeros(len(pre)),np.ones(len(post))]; x=np.r_[pre[col],post[col]]; flags=df[col].to_numpy()>q99; tt=df.t.to_numpy(); postflag=flags[tt>=onset+10]; first=next((t for t,f in zip(tt,flags) if t>=onset and f),None); pers=next((tt[i] for i in range(len(tt)-2) if tt[i]>=onset and flags[i:i+3].all()),None)
 return {"roc_auc":float(roc_auc_score(y,x)),"pr_auc":float(average_precision_score(y,x)),"independent_test_fpr":float(np.mean(np.asarray(refs_test[col])>q99)),"attack_detection_rate":float(postflag.mean()),"first_alarm_delay_s":-1.0 if first is None else float(first-onset),"persistent_delay_s":-1.0 if pers is None else float(pers-onset),"threshold_q99":q99,"threshold_q995":q995,"target_fpr_1pct_threshold":q99}

def plot_timeline(d,s,out):
 cols=["B0","M1","P1_median","P2_median","P3_median","Full"]; fig,axs=plt.subplots(3,2,figsize=(14,9),sharex=True)
 for ax,c in zip(axs.flat,cols):ax.plot(d.t,d[c],lw=.8);ax.axvline(120,color="r",ls="--");ax.set_title(c);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/"plots"/f"{s}_timeline.png",dpi=120);plt.close(fig)

def main():
 global refs_test
 ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=Path("artifacts/clif_ip_cross_layer_r3"));a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=True);(out/"plots").mkdir(exist_ok=True)
 bpaths,mpaths=paths(out); ckpath=ROOT/"artifacts/ai_morph_gru_cleanStatic_q70_frame/prn_local_gru_predictor.pt";ck=torch.load(ckpath,map_location="cpu",weights_only=False);c=Cfg(**{k:ck["config"][k] for k in ("hidden_dim","emb_dim","dropout")});model=GRU(9,c);model.load_state_dict(ck["model_state_dict"])
 b={s:load_b0(p,ck,model) for s,p in bpaths.items()}; cleanfloor=pd.read_csv(mpaths["cleanStatic"]); cols=[x for x in cleanfloor if x not in META];audit=M1FitAudit();state=fit_m1(cleanfloor[cols].to_numpy(float),cleanfloor.window_start_s.to_numpy(float),240.,8,6,audit,"cleanStatic")
 m={}; mc={}
 for s,p in mpaths.items():m[s],mc[s]=m1_frame(p,state,audit,s)
 merged={s:merge(b[s],m[s]) for s in b}; clean=merged["cleanStatic"];train=split(clean,0,240);val=split(clean,250,330);test=clean[(clean.t>=340)&(clean.available_s>=340)].copy()
 # validation-only hyperparameter selection
 tune=[]
 for lag in (2,4,6):
  for alpha in (.1,1.,10.):
   md=train_models(train,val,lag,alpha);Y,P,_=predict_model(val,md,lag,"P3");tune.append((mean_squared_error(Y,P),lag,alpha))
 _,lag,alpha=min(tune);models=train_models(train,val,lag,alpha)
 predrows=[];covs={}
 for k in ("P0","P1","P2","P3"):
  Y,P,_=predict_model(val,models,lag,k);covs[k]=LedoitWolf().fit(Y-P).covariance_
  for name,d in (("validation",val),("test",test)):
   yy,pp,_=predict_model(d,models,lag,k);predrows.append({"split":name,"model":k,"mse":mean_squared_error(yy,pp),"mae":mean_absolute_error(yy,pp),"samples":len(yy)})
 pc=pd.DataFrame(predrows);p1=float(pc.query("split=='test' and model=='P1'").mse.iloc[0]);p3=float(pc.query("split=='test' and model=='P3'").mse.iloc[0]);pc["p3_vs_p1_improvement_pct"]=100*(p1-p3)/p1;pc["incremental_r2_vs_p1"]=1-p3/p1;pc.to_csv(out/"predictor_comparison.csv",index=False)
 val0=epoch_scores(val,models,covs,lag);test0=epoch_scores(test,models,covs,lag); refs={"B0":val0.B0.to_numpy(),"M1":val0.M1.to_numpy()};val0["concordance"]=(-np.log(etail(val0.B0,refs["B0"])))*(-np.log(etail(val0.M1,refs["M1"])));mu,scale,cov=calibrate_components(None,val0.rename(columns={"P3_median":"P3"}));refs|={"mu":mu,"scale":scale,"cov":cov}
 scored={"clean_validation":epoch_scores(val,models,covs,lag,refs),"clean_test":epoch_scores(test,models,covs,lag,refs)}|{s:epoch_scores(merged[s],models,covs,lag,refs) for s in SCENARIOS}
 for s,d in scored.items():d.to_csv(out/f"per_epoch_scores_{s}.csv",index=False)
 scorecols=["B0","M1","mean_fusion","max_fusion","fisher_fusion","P0_median","P1_median","P2_median","P3_median","Full","minus_M1","minus_B0history","minus_concordance"]
 refs_test={c:scored["clean_test"][c].dropna().to_numpy() for c in scorecols};rows=[]
 for s in SCENARIOS:
  for c0 in scorecols:rows.append({"scenario":s,"model":c0,**alarm_metrics(scored[s].dropna(subset=[c0]),c0,scored["clean_validation"][c0].dropna().to_numpy(),ONSETS[s])})
 pd.DataFrame(rows).to_csv(out/"scenario_metrics.csv",index=False);pd.DataFrame(rows).groupby("model",as_index=False).agg(roc_auc=("roc_auc","mean"),pr_auc=("pr_auc","mean"),independent_test_fpr=("independent_test_fpr","mean"),attack_detection_rate=("attack_detection_rate","mean")).to_csv(out/"fusion_comparison.csv",index=False)
 pd.DataFrame(rows)[lambda x:x.model.isin(["Full","minus_M1","minus_B0history","minus_concordance"])].to_csv(out/"ablation_metrics.csv",index=False)
 # clean-validation lag analysis only
 lr=[]
 for sh in range(-6,7):
  x=val0.B0.to_numpy();y=pd.Series(val0.M1).shift(sh).to_numpy();ok=np.isfinite(y);lr.append({"lag_epochs":sh,"lag_s":sh*.5,"validation_pearson":np.corrcoef(x[ok],y[ok])[0,1],"selected_predictor_lag":sh==lag})
 pd.DataFrame(lr).to_csv(out/"lag_analysis.csv",index=False);fig,ax=plt.subplots();ax.plot([x["lag_s"] for x in lr],[x["validation_pearson"] for x in lr],marker="o");ax.grid();fig.savefig(out/"plots/lag_analysis.png");plt.close(fig)
 # Alignment destruction only inside clean test / pre-onset / established attack.
 # Permute whole M1 epoch blocks, retain every marginal, and recompute P2/P3/Full.
 dest={}
 raw_regions={"clean_test":test}|{f"{s}_pre":merged[s].query("t < 110") for s in SCENARIOS}|{f"{s}_attack":merged[s].query("t >= 130") for s in SCENARIOS}
 for name,raw0 in raw_regions.items():
  raw0=raw0.copy(); epochs=raw0.drop_duplicates("t").sort_values("t"); mcols=[c for c in raw0 if c=="m1" or (c.startswith("m") and c[1:].isdigit())]
  starts=list(range(0,len(epochs),8)); order=np.random.default_rng(31).permutation(len(starts)); perm=np.concatenate([np.arange(starts[j],min(starts[j]+8,len(epochs))) for j in order]); old=epochs[mcols].to_numpy(); new=old[perm]
  mapping={float(t):new[i] for i,t in enumerate(epochs.t)}
  shuffled=raw0.copy()
  for i,cname in enumerate(mcols): shuffled[cname]=shuffled.t.map(lambda t:mapping[float(t)][i])
  aligned_scores=epoch_scores(raw0,models,covs,lag,refs); shuffled_scores=epoch_scores(shuffled,models,covs,lag,refs); pair=aligned_scores.merge(shuffled_scores,on="t",suffixes=("_aligned","_shuffled"))
  ds=[]; b=aligned_scores.B0.to_numpy(); mm=aligned_scores.M1.to_numpy()
  for seed in range(200):_,q=shuffle_pairing(b,mm,seed=seed,block=8);ds.append(np.mean(b*q))
  observed=float(np.mean(b*mm)); center=float(np.mean(ds))
  dest[name]={"n_epochs":len(epochs),"m1_mean_before":float(old.mean()),"m1_mean_after":float(new.mean()),"m1_sorted_equal":bool(np.allclose(np.sort(old,axis=0),np.sort(new,axis=0))),"P2_score_delta_mean":float((pair.P2_median_aligned-pair.P2_median_shuffled).mean()),"P3_score_delta_mean":float((pair.P3_median_aligned-pair.P3_median_shuffled).mean()),"Full_score_delta_mean":float((pair.Full_aligned-pair.Full_shuffled).mean()),"P2_mse_proxy_delta":float(np.mean(pair.P2_median_aligned**2)-np.mean(pair.P2_median_shuffled**2)),"P3_mse_proxy_delta":float(np.mean(pair.P3_median_aligned**2)-np.mean(pair.P3_median_shuffled**2)),"aligned_concordance":observed,"shuffled_concordance":float(np.mean(aligned_scores.B0.to_numpy()*shuffled_scores.M1.to_numpy())),"permutation_p_value":float((1+np.sum(np.abs(np.asarray(ds)-center)>=abs(observed-center)))/201)}
 (out/"alignment_destruction_metrics.json").write_text(json.dumps({"method":"within-region 8-epoch M1 block permutation; P2/P3/Full recomputed; no global circular shift","results":dest},indent=2)+"\n")
 for s in SCENARIOS:plot_timeline(scored[s],s,out)
 # tap and aligned-shuffled required plots
 fig,ax=plt.subplots(figsize=(9,4));
 for j in range(9):ax.plot(np.arange(len(models["P3"].coef_[j])),models["P3"].coef_[j],alpha=.45,label=f"tap{j}")
 ax.set_title("P3 ridge coefficient / tap residual diagnostic");fig.savefig(out/"plots/tap_residual_distribution.png");plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4));names=list(dest);ax.bar(np.arange(len(names)),[dest[x]["Full_score_delta_mean"] for x in names]);ax.set_xticks(np.arange(len(names)),names,rotation=90);ax.set_ylabel("aligned - shuffled Full mean");fig.tight_layout();fig.savefig(out/"plots/aligned_shuffled.png");plt.close(fig)
 prov=[]
 rawhash={"os1":"e9ef8ab33a3e59c5e55b3f6fb9b8bb3ba18aaf380402ae00abbe535858b1deb7","os2":"17de8e3f54095ad2eafad8a54ca7f5008596936ded35578710dfed17a5b670c1","os3":"2a3c3c5cf1accaa287fe14181e43070903500e0250c69e3c335f91c89c0cdc6c","os4":"803f3c76bcc618efbc6b394eb536fe61ed8c3e34b1822c0088b4475621bfa8e4"}
 for s in ("cleanStatic",)+SCENARIOS:prov.append({"scenario":s,"grade":"reconstructed" if s!="cleanStatic" else "provisional","raw_iq_path":str(RAW/f"{s}.bin"),"canonical_raw_iq_sha256":rawhash.get(s),"sample_rate_hz":5000000,"sample_format":"interleaved little-endian int16 IQ","recording_start_sample":0,"seek_samples":0,"m1_block_ms":10,"stride_s":.5,"b0_window_s":1.,"alignment":"timestamp reconstructed; receiver processing delay unavailable","b0_sha256":sha(bpaths[s]),"m1_sha256":sha(mpaths[s])})
 (out/"provenance_manifest.json").write_text(json.dumps({"schema":"clif-ip.r3.provenance.v1","grades_are_not_equivalent":True,"scenarios":prov},indent=2)+"\n")
 fit_summary={"kind":"M1-style surrogate (actual raw-IQ features; no reusable frozen M1 checkpoint found)","fit_count":audit.fit_count,"fit_recordings":audit.fit_recordings,"transform_recordings":audit.transform_recordings,"frozen_sha256":state.sha256,"fit_scope":"cleanStatic whole windows 0-240 s","fit_rows":state.fit_rows,"pca_dim":state.pca_dim,"ar_lag":state.lag}
 (out/"frozen_m1_fit_summary.json").write_text(json.dumps(fit_summary,indent=2)+"\n")
 cfg={"schema":"clif-ip.cross-layer.r3","device":"CPU Ridge (GPU not required)","splits":{"train":"0-240 whole containment","guard1":"240-250","validation_calibration":"250-330","guard2":"330-340","independent_test":">=340"},"lag":lag,"alpha":alpha,"selection":"validation only","onsets":ONSETS,"b0_checkpoint_sha256":sha(ckpath)};(out/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
 # machine-readable summary and non-finite/artifact checks
 full=pd.DataFrame(rows).query("model=='Full'"); summary=f"M1: {fit_summary['kind']}\nM1 hash: {state.sha256}\nP3 test MSE: {p3:.9g}\nP1 test MSE: {p1:.9g}\nP3 improvement %: {100*(p1-p3)/p1:.6g}\nFull mean ROC-AUC: {full.roc_auc.mean():.6g}\nFull mean PR-AUC: {full.pr_auc.mean():.6g}\nGo/No-Go: {'GO' if p3<p1 and full.roc_auc.mean()>=pd.DataFrame(rows).query("model in [\'B0\',\'M1\']").roc_auc.mean() else 'NO-GO'}\n";(out/"test_summary.txt").write_text(summary)
 print(summary)
if __name__=="__main__":main()
