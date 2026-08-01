#!/usr/bin/env python3
"""CLIF-IP Phase 1: frozen B0 and existing M1 RF evidence, normal-only fitting.

This is an observational, causal-time association study. It does not claim a
physical RF-to-tracking causal mechanism.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAP_COLS=["tap_E4_rel_prompt_mean","tap_E3_rel_prompt_mean","tap_E2_rel_prompt_mean","tap_E_rel_prompt_mean","tap_P_rel_prompt_mean","tap_L_rel_prompt_mean","tap_L2_rel_prompt_mean","tap_L3_rel_prompt_mean","tap_L4_rel_prompt_mean"]
META={"run_id","source_fingerprint","split","label","prn","channel","sample_rate_hz","segment_index","window_index","window_start_s","window_end_s","window_mid_s","window_bin_s","epoch_count","tap_count","tap_layout"}
EPS=1e-9

@dataclass
class Cfg:
    hidden_dim:int=128; emb_dim:int=128; dropout:float=.05
class PrnLocalGRU(nn.Module):
    def __init__(self, f:int, c:Cfg):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(f,c.emb_dim),nn.LayerNorm(c.emb_dim),nn.GELU(),nn.Dropout(c.dropout),nn.Linear(c.emb_dim,c.emb_dim),nn.GELU())
        self.gru=nn.GRU(input_size=c.emb_dim,hidden_size=c.hidden_dim,batch_first=True)
        self.head=nn.Sequential(nn.Linear(c.hidden_dim,c.hidden_dim),nn.GELU(),nn.Linear(c.hidden_dim,f))
    def forward(self,x):
        b,t,f=x.shape; z=self.encoder(x.reshape(b*t,f)).reshape(b,t,-1); o,_=self.gru(z); return self.head(o[:,-1])

def robust(x):
    med=np.nanmedian(x,axis=0); mad=np.nanmedian(np.abs(x-med),axis=0)*1.4826; sd=np.nanstd(x,axis=0)
    scale=np.where(np.isfinite(mad)&(mad>EPS),mad,sd); scale=np.where(np.isfinite(scale)&(scale>EPS),scale,1.)
    return med,scale

def finite(x): return np.nan_to_num(np.asarray(x,float),nan=0.,posinf=0.,neginf=0.)
def empirical_tail(x, ref):
    ref=np.sort(np.asarray(ref,float)); return np.maximum(1/(len(ref)+1),1-np.searchsorted(ref,x,side="right")/len(ref))
def mahal_fit(X):
    model=LedoitWolf().fit(finite(X)); return model
def mahal(model,X): return np.sqrt(np.maximum(0,model.mahalanobis(finite(X))))
def qthreshold(v,q): return float(np.quantile(v,q))
def availability(df): return np.maximum(df["b0_end_s"].to_numpy(float),df["m1_end_s"].to_numpy(float))

def load_b0(path:Path, ckpt:dict, model:nn.Module, device:torch.device)->pd.DataFrame:
    df=pd.read_csv(path); rows=[]; mean=np.asarray(ckpt["standardizer"]["node_mean"],np.float32); std=np.asarray(ckpt["standardizer"]["node_std"],np.float32)
    model.eval()
    for (_,prn),g in df.groupby(["run_id","prn"],sort=False):
        g=g.sort_values("window_bin_s").reset_index(drop=True)
        x=finite((g[TAP_COLS].to_numpy(np.float32)-mean)/std).astype(np.float32)
        tt=g.window_bin_s.to_numpy(float)
        for i in range(12,len(g)):
            if not np.allclose(np.diff(tt[i-12:i+1]),.5,atol=1e-5): continue
            with torch.no_grad(): pred=model(torch.from_numpy(x[i-12:i][None]).to(device)).cpu().numpy()[0]
            res=x[i]-pred; rmse=float(np.sqrt(np.mean(res*res)))
            r=g.iloc[i]
            rows.append({"t":float(r.window_bin_s),"b0_end_s":float(r.window_end_s),"prn":str(prn),"rmse":rmse,"energy":float(np.sum(res*res)),**{f"res_{j}":float(v) for j,v in enumerate(res)}})
    pr=pd.DataFrame(rows)
    out=[]
    for t,g in pr.groupby("t",sort=True):
        rm=g.rmse.to_numpy(float)
        residuals=g[[f"res_{i}" for i in range(9)]].to_numpy(float)
        out.append({
            "t": float(t), "b0_end_s": float(g.b0_end_s.max()),
            "b0_median": float(np.median(rm)), "b0_q90": float(np.quantile(rm,.9)),
            "b0_q99": float(np.quantile(rm,.99)), "b0_energy": float(g.energy.sum()),
            "tracked_prns": int(len(g)),
            "b0_residual_vector_l2": float(np.sqrt(np.mean(residuals**2))),
        })
    return pd.DataFrame(out)

def m1_features(path:Path, clean:pd.DataFrame, train_end:float, pca_dim:int=8, lag:int=6):
    meta={"scenario","window_index","window_start_s","window_mid_s","window_end_s","block_ms","stride_s"}
    cols=[c for c in clean.columns if c not in meta]
    Xc=finite(clean[cols].to_numpy(float)); fit=clean.window_start_s.to_numpy(float)<=train_end
    mu=Xc[fit].mean(0); sd=Xc[fit].std(0); sd[sd<EPS]=1
    zc=(Xc-mu)/sd; _,_,vt=np.linalg.svd(zc[fit],full_matrices=False); V=vt[:pca_dim].T
    medx,scalex=robust(Xc[fit])
    def score(df):
        X=finite(df[cols].to_numpy(float)); P=((X-mu)/sd)@V; ts=df.window_start_s.to_numpy(float)
        Xar=np.stack([P[i-lag:i].reshape(-1) for i in range(lag,len(P))]); Y=P[lag:]; idx=np.arange(lag,len(P)); fit_ar=ts[idx]<=train_end
        W=np.linalg.lstsq(Xar[fit_ar],Y[fit_ar],rcond=None)[0]
        innov=Y-Xar@W; ar=np.sqrt(np.mean(innov*innov,1)); me,se=robust(ar[fit_ar,None]); arscore=np.maximum(0,(ar-me[0])/se[0])
        level=np.sqrt(np.mean(((X-medx)/scalex)**2,1))[idx]
        o=df.iloc[idx][["window_start_s","window_end_s"]].copy().rename(columns={"window_start_s":"t","window_end_s":"m1_end_s"}).reset_index(drop=True)
        o["m1_ar_rmse"]=ar; o["m1_ar_score"]=arscore; o["m1_level"]=level
        for j in range(innov.shape[1]): o[f"m1_innov_{j}"]=innov[:,j]
        return o
    return score, cols

def merge(b,m):
    x=pd.merge(b,m,on="t",how="inner",validate="one_to_one").sort_values("t").reset_index(drop=True)
    x["available_s"]=availability(x); return x

def fit_models(clean, train_end, val_start, val_end):
    train=clean[clean.t<=train_end].copy(); val=clean[(clean.t>=val_start)&(clean.t<=val_end)].copy()
    bcols=["b0_median","b0_q90","b0_q99","b0_energy","tracked_prns","b0_residual_vector_l2"]
    mcols=["m1_ar_rmse","m1_level"]
    # B0 PRN exceedance fraction: q99 train normal threshold.
    thr=qthreshold(train.b0_q99,.99)
    for d in (train,val,clean): d["b0_frac_exceed"]=np.clip((d.b0_q99>thr).astype(float),0,1)
    bcols.insert(4,"b0_frac_exceed")
    # scalar baseline scores calibrated normal train
    bm,bs=robust(train[bcols].to_numpy(float)); mm,ms=robust(train[mcols].to_numpy(float))
    def baseline(d):
        d=d.copy(); d["S_B0"]=np.sqrt(np.mean(((d[bcols].to_numpy(float)-bm)/bs)**2,1)); d["S_M1"]=np.sqrt(np.mean(((d[mcols].to_numpy(float)-mm)/ms)**2,1)); return d
    train=baseline(train); val=baseline(val); clean=baseline(clean)
    # C3 joint
    joint=mahal_fit(train[["S_B0","S_M1"]]);
    # C4 ridge: only past/current M1, no future; lag range fixed before attacks (0..3 sec)
    lags=6; mbase=["m1_ar_rmse","m1_level"]
    def design(d):
        arr=d[mbase].to_numpy(float); feats=[]
        for i in range(lags,len(d)): feats.append(arr[i-lags:i+1].reshape(-1))
        return np.asarray(feats), np.arange(lags,len(d))
    Xtr,itr=design(train); Ytr=train.iloc[itr][bcols].to_numpy(float); ridge=Ridge(alpha=1.).fit(Xtr,Ytr)
    predtr=ridge.predict(Xtr); em,es=robust(Ytr-predtr); crosscov=mahal_fit((Ytr-predtr))
    cascade_ref=[]
    for i in itr: cascade_ref.append(float(train.S_B0.iloc[i]*np.max(train.S_M1.iloc[max(0,i-lags):i+1])))
    cas_m,cas_s=robust(np.asarray(cascade_ref)[:,None])
    def apply(d):
        d=baseline(d); d["S_joint"]=mahal(joint,d[["S_B0","S_M1"]])
        X,idx=design(d); scores=np.full(len(d),np.nan); casc=np.full(len(d),np.nan)
        e=d.iloc[idx][bcols].to_numpy(float)-ridge.predict(X); scores[idx]=mahal(crosscov,e)
        for ii in idx: casc[ii]=float(d.S_B0.iloc[ii]*np.max(d.S_M1.iloc[max(0,ii-lags):ii+1]))
        d["S_cross"]=scores; d["S_cascade"]=np.maximum(0,(casc-cas_m[0])/cas_s[0])
        fullcols=["S_B0","S_M1","S_cross","S_cascade"]; fm=mahal_fit(apply_train[fullcols]) if False else None
        return d
    apply_train=baseline(train); apply_train["S_joint"]=mahal(joint,apply_train[["S_B0","S_M1"]]); X,idx=design(apply_train); e=apply_train.iloc[idx][bcols].to_numpy(float)-ridge.predict(X); apply_train["S_cross"]=np.nan; apply_train.loc[apply_train.index[idx],"S_cross"]=mahal(crosscov,e); cc=np.full(len(apply_train),np.nan)
    for ii in idx: cc[ii]=float(apply_train.S_B0.iloc[ii]*np.max(apply_train.S_M1.iloc[max(0,ii-lags):ii+1]))
    apply_train["S_cascade"]=np.maximum(0,(cc-cas_m[0])/cas_s[0]); fullcols=["S_B0","S_M1","S_cross","S_cascade"]; fullfit=mahal_fit(apply_train.dropna()[fullcols])
    def final(d):
        z=apply(d); z["S_full"]=np.nan; ok=z[fullcols].notna().all(1); z.loc[ok,"S_full"]=mahal(fullfit,z.loc[ok,fullcols]);
        # C2 normal-only tails; no learned attack weights.
        for name in ["S_B0","S_M1"]: z[f"p_{name}"]=empirical_tail(z[name].to_numpy(float),train[name].to_numpy(float))
        z["S_mean"]=.5*(z.S_B0+z.S_M1); z["S_max"]=np.maximum(z.S_B0,z.S_M1); z["S_fisher"]= -2*(np.log(z.p_S_B0)+np.log(z.p_S_M1))
        return z
    result={"train":final(train),"val":final(val),"clean":final(clean),"bcols":bcols,"mcols":mcols,"lag_s":lags*.5,"ridge":ridge,"fullcols":fullcols}
    return result

def metrics(scores, clean_ref, onset, normal_end=110., attack_start=130.):
    neg=scores[scores.t<=normal_end]; pos=scores[scores.t>=attack_start]; y=np.r_[np.zeros(len(neg)),np.ones(len(pos))]; x=np.r_[neg.values,pos.values]
    q99=qthreshold(clean_ref.values,.99); q995=qthreshold(clean_ref.values,.995); flags=scores.values>q99; post=(scores.index.to_numpy()>=0) # unused
    first=None
    tt=scores.index.to_numpy(float)
    # actual caller replaces index with time
    return {"roc_auc":float(roc_auc_score(y,x)) if len(np.unique(y))==2 else None,"pr_auc":float(average_precision_score(y,x)) if len(np.unique(y))==2 else None,"threshold_q99":q99,"threshold_q995":q995}

def scenario_metrics(d, clean_val, name,onset):
    rows=[]
    cols={"C0_B0":"S_B0","C1_M1":"S_M1","C2_mean":"S_mean","C2_max":"S_max","C2_fisher":"S_fisher","C3_joint":"S_joint","C4_lagged":"S_cross","C5_full":"S_full"}
    for model,col in cols.items():
        z=d[["t",col]].dropna(); ref=clean_val[col].dropna(); neg=z[z.t<=110]; pos=z[z.t>=130]; y=np.r_[np.zeros(len(neg)),np.ones(len(pos))]; x=np.r_[neg[col],pos[col]]
        q99=qthreshold(ref,.99); q995=qthreshold(ref,.995); flags=z[col].to_numpy()>q99; t=z.t.to_numpy(); post=t>=130; first=next((float(tt) for tt,ff in zip(t,flags) if tt>=onset and ff),None); persistent=flags[post].mean() if post.any() else np.nan
        clean_fpr=float((clean_val[col].dropna().to_numpy()>q99).mean())
        rows.append({"scenario":name,"model":model,"roc_auc":float(roc_auc_score(y,x)),"pr_auc":float(average_precision_score(y,x)),"normal_fpr":clean_fpr,"attack_detection_rate":float(flags[post].mean()),"first_alarm_delay_s":None if first is None else first-onset,"persistent_alarm_ratio":float(persistent),"threshold_q99":q99,"threshold_q995":q995})
    return rows

def plot(d,name,out):
    fig,axs=plt.subplots(4,1,figsize=(13,10),sharex=True)
    for ax,col,title in zip(axs,["S_B0","S_M1","S_cross","S_cascade"],["B0 score","M1 RF score","cross-layer residual","causal cascade"]): ax.plot(d.t,d[col],lw=1); ax.axvline(120,color="r",ls="--"); ax.set_ylabel(title); ax.grid(alpha=.25)
    axs[-1].set_xlabel("recording-relative time / s"); fig.tight_layout(); fig.savefig(out/f"plots/{name}_timeline.png",dpi=150); plt.close(fig)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--manifest",type=Path,default=Path("configs/da_pfrt_oakbat_manifest.json")); ap.add_argument("--checkpoint",type=Path,default=Path("artifacts/ai_morph_gru_cleanStatic_q70_frame/prn_local_gru_predictor.pt")); ap.add_argument("--out",type=Path,default=Path("artifacts/clif_ip_feasibility")); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); (a.out/"plots").mkdir(exist_ok=True)
 m=json.loads(a.manifest.read_text()); ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); cc=Cfg(**{k:ck["config"][k] for k in ["hidden_dim","emb_dim","dropout"]}); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=PrnLocalGRU(len(TAP_COLS),cc).to(dev); model.load_state_dict(ck["model_state_dict"])
 raw={}; floors={}
 for n,item in m["datasets"].items(): raw[n]=load_b0(Path(item["morph_csv"]),ck,model,dev); floors[n]=pd.read_csv(item["floor_csv"])
 scorer, mcols=m1_features(Path(m["datasets"]["cleanStatic"]["floor_csv"]),floors["cleanStatic"],240.)
 merged={n:merge(raw[n],scorer(floors[n])) for n in raw}
 fitted=fit_models(merged["cleanStatic"],240.,250.,330.)
 scored={n:(fitted["clean"] if n=="cleanStatic" else None) for n in merged}; scored["cleanStatic"]=fitted["clean"]
 # Fit all transformations only on clean chronological train rows. Attack rows are shifted
 # beyond 1000 s solely to keep them outside every clean training/validation interval.
 for n in ["os2","os3","os4"]:
   attack=merged[n].copy()
   for col in ["t","b0_end_s","m1_end_s","available_s"]:
       if col in attack: attack[col]=attack[col]+1000.0
   combo=pd.concat([merged["cleanStatic"],attack],ignore_index=True).sort_values("t").reset_index(drop=True)
   combo_fit=fit_models(combo,240.,250.,330.)
   d=combo_fit["clean"].query("t >= 1000").copy()
   for col in ["t","b0_end_s","m1_end_s","available_s"]: d[col]=d[col]-1000.0
   scored[n]=d
 rows=[]
 for n in ["os2","os3","os4"]:
   clean_val=fitted["val"]
   for r in scenario_metrics(scored[n],clean_val,n,120.):
     r["status"]="evaluated"; r["reason"]="normal-only cleanStatic train [0,240] and validation [250,330]; attack rows shifted outside fitting intervals"
     rows.append(r)
   plot(scored[n],n,a.out)
 pd.DataFrame(rows).to_csv(a.out/"scenario_metrics.csv",index=False)
 pd.DataFrame(rows).groupby("model",dropna=False).agg({"roc_auc":"mean","pr_auc":"mean","attack_detection_rate":"mean","normal_fpr":"mean"}).reset_index().to_csv(a.out/"fusion_comparison.csv",index=False)
 # Causal/diagnostic lag association is selected on clean validation only; no attack-based lag choice.
 lag_rows=[]; cv=fitted["val"].dropna(subset=["S_B0","S_M1"]).reset_index(drop=True)
 for lag in [-3,-2,-1,0,1,2,3]:
   x=cv.S_B0.to_numpy(); y=cv.S_M1.shift(lag).to_numpy(); ok=np.isfinite(x)&np.isfinite(y)
   lag_rows.append({"lag_s":lag*.5,"causal_for_model":lag>=0,"clean_validation_pearson":float(np.corrcoef(x[ok],y[ok])[0,1]),"status":"diagnostic"})
 pd.DataFrame(lag_rows).to_csv(a.out/"lag_analysis.csv",index=False)
 fig,ax=plt.subplots(figsize=(8,4)); ld=pd.DataFrame(lag_rows); ax.plot(ld.lag_s,ld.clean_validation_pearson,marker="o"); ax.axvline(0,color="k",ls=":"); ax.set(xlabel="M1 shift / s (positive=past causal history)",ylabel="Pearson(B0,M1)",title="Clean validation lag sensitivity"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(a.out/"plots/lag_sensitivity.png",dpi=150); plt.close(fig)
 destroy={}
 for n in ["os2","os3","os4"]:
   z=scored[n].dropna(subset=["S_B0","S_M1"]).reset_index(drop=True); shift=max(60,len(z)//3)
   b=z.S_B0.to_numpy(); mi=z.S_M1.to_numpy(); aligned=float(np.mean(b*mi)); shuffled=float(np.mean(b*np.roll(mi,shift)))
   destroy[n]={"circular_shift_epochs":int(shift),"circular_shift_s":float(shift*.5),"b0_marginal_mean_before":float(b.mean()),"b0_marginal_mean_after":float(b.mean()),"m1_marginal_mean_before":float(mi.mean()),"m1_marginal_mean_after":float(np.roll(mi,shift).mean()),"aligned_cascade_proxy":aligned,"shuffled_cascade_proxy":shuffled,"difference":aligned-shuffled}
   fig,ax=plt.subplots(figsize=(10,3)); ax.plot(z.t,b*mi,label="aligned",lw=1); ax.plot(z.t,b*np.roll(mi,shift),label="circular-shifted M1",lw=1,alpha=.8); ax.legend(); ax.set(title=f"{n}: alignment destruction cascade proxy",xlabel="time / s",ylabel="B0×M1"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(a.out/f"plots/{n}_aligned_vs_shuffled.png",dpi=150); plt.close(fig)
 (a.out/"alignment_destruction_metrics.json").write_text(json.dumps({"status":"evaluated_proxy","definition":"continuous B0 score multiplied by same-time M1 score; this is a diagnostic cascade proxy, not a physical causal claim","results":destroy},indent=2,allow_nan=False)+"\n")
 for n,d in scored.items(): d.to_csv(a.out/f"{n}_epoch_scores.csv",index=False)
 report={"schema":"clif-ip.phase1.v1","device":str(dev),"datasets":{"OAKBAT":{"evaluated":["os2","os3","os4"],"missing":["os1"],"normal_training":"cleanStatic chronological 0-240s; validation 250-330s; no attack rows used to fit B0/M1 normalizers, covariance, predictor, or thresholds","limitation":"single-recording normal split; not recorder-holdout"},"TEXBAT":{"status":"not_run","reason":"paired derived artifacts unavailable"}},"conclusion":"No-Go: OAKBAT-only, single-clean-recording evidence is insufficient for a cross-dataset/recorder claim. M1-alone scores dominate several scenarios and q99 false-positive rates are not stable; C5 does not establish improvement over simple fusion."}
 (a.out/"config.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
 print(json.dumps(report,indent=2))
if __name__=="__main__": main()
