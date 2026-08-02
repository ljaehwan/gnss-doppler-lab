#!/usr/bin/env python3
"""Leakage-safe R4 R0/S0/S1 training and clean-only adaptation."""
from __future__ import annotations
import argparse,hashlib,json,sys,copy
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.clif_ip_synthetic import (DOMAINS,TAP_ORDER,fit_multirun_ar,extract_m1_features,
    tex_npz_magnitude_nodes as convert_tex_nodes,adapt_m1_residual_state,array_hash)

REAL_B0={
 "SYN-OAK":Path("/home/ubuntu/projects/gnss-doppler-lab/artifacts/oakbat_9tap_frozen_champion/cleanStatic/multi_prn_method_a_9tap_w1.0_s0.5_normalized_dmcpd/normal_prn_node_windows.csv"),
 "SYN-TEX":Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/exports/cleanStatic.npz")}
REAL_M1={
 "SYN-OAK":Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/oakbat-cleanStatic-raw-iq-noise-continuity-m1-v0-fs5m-full/oakbat_cleanStatic_raw_iq_noise_features.csv"),
 "SYN-TEX":Path("/home/ubuntu/unraid/gnss-datasets/texbat/raw/cleanStatic.bin")}
META={"scenario","run_id","window_index","t","window_start_s","window_mid_s","window_end_s","start_sample","end_sample","block_ms","stride_s","split"}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()

class SharedPrnGRU(nn.Module):
 """One shared predictor for variable PRNs; PRN identity is deliberately absent."""
 def __init__(self,features=9,hidden=32):super().__init__();self.gru=nn.GRU(features,hidden,batch_first=True);self.head=nn.Linear(hidden,9)
 def forward(self,x):return self.head(self.gru(x)[0][:,-1])

def clean_only(d):
 if "label" in d:
  labels=d.label.astype(str).str.lower()
  if labels.str.contains("attack|spoof|os[1-4]|texbat_ds",regex=True).any() or not labels.str.contains("normal|cleanstatic").all():
   raise ValueError("attack/spoof rows may not enter fitting")
 return d

def load_domain(root,domain,split):
 idx=pd.read_csv(root/"synthetic_run_manifest.csv");ids=idx.query("domain==@domain and split==@split").run_id
 bs=[];ms=[]
 for rid in ids:
  rd=root/"runs"/rid;b=pd.read_csv(rd/"b0_nodes.csv");m=pd.read_csv(rd/"m1_features.csv")
  b["run_id"]=rid;m["run_id"]=rid;bs.append(b);ms.append(m)
 if not bs:raise ValueError(f"no {domain}/{split} runs")
 return clean_only(pd.concat(bs,ignore_index=True)),clean_only(pd.concat(ms,ignore_index=True))

def tap_columns(d):
 """Exact producer-layout mapping; raw taps are nine magnitudes, not signed IQ."""
 expected=tuple(TAP_ORDER)
 if "tap_layout" not in d:raise ValueError("tap_layout is required for exact canonical mapping")
 layouts={tuple(str(x).split(",")) for x in d.tap_layout.dropna().unique()}
 if layouts!={expected}:raise ValueError(f"tap_layout must be exactly {','.join(expected)}; got {sorted(layouts)}")
 cols=[f"tap_{x}_rel_prompt_mean" for x in expected]
 if not set(cols)<=set(d):raise ValueError("all nine relative-prompt magnitude columns are required exactly; absolute/scalar compression is forbidden")
 return cols

def sequences(d,cols,lag=12):
 X=[];Y=[]
 for _,g in d.groupby(["run_id","prn"],sort=False):
  g=g.sort_values("window_bin_s");z=g[cols].to_numpy(np.float32)
  for i in range(lag,len(z)):X.append(z[i-lag:i]);Y.append(z[i])
 return np.asarray(X,np.float32),np.asarray(Y,np.float32)

def fit_epochs(model,X,Y,Xv,Yv,mu,sd,epochs,device,lr):
 if not len(X) or not len(Xv):raise ValueError("insufficient run-local B0 history")
 xt=torch.tensor((X-mu)/sd,device=device);yt=torch.tensor((Y-mu)/sd,device=device)
 xv=torch.tensor((Xv-mu)/sd,device=device);yv=torch.tensor((Yv-mu)/sd,device=device)
 opt=torch.optim.AdamW(model.parameters(),lr=lr);best=None;hist=[]
 for e in range(epochs):
  model.train();opt.zero_grad();loss=((model(xt)-yt)**2).mean();loss.backward();opt.step();model.eval()
  with torch.no_grad():vl=float(((model(xv)-yv)**2).mean())
  hist.append({"epoch":e+1,"train_mse":float(loss),"validation_mse":vl})
  if best is None or vl<best[0]:best=(vl,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
 model.load_state_dict(best[1]);return hist

def state_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def tex_npz_magnitude_nodes(source,out):
 """Compatibility wrapper around the single validated converter."""
 if out.exists() and out.with_suffix(".provenance.json").exists():return pd.read_csv(out)
 return convert_tex_nodes(source,out,"cleanStatic")

def real_clean_frames(root,domain):
 if domain=="SYN-TEX":
  cache=root/"real_inputs";cache.mkdir(exist_ok=True);bp=cache/"texbat_cleanStatic_method_a_magnitude_nodes.csv";b=tex_npz_magnitude_nodes(REAL_B0[domain],bp)
 else:b=clean_only(pd.read_csv(REAL_B0[domain]))
 b["run_id"]="real-cleanStatic"
 if domain=="SYN-TEX":
  cache=root/"real_inputs";cache.mkdir(exist_ok=True);p=cache/"texbat_cleanStatic_m1_features.csv"
  if not p.exists():
   duration=REAL_M1[domain].stat().st_size/(4*25_000_000)
   extract_m1_features(REAL_M1[domain],"cleanStatic",25_000_000,duration,block_ms=10.,stride_s=.5).to_csv(p,index=False)
  m=pd.read_csv(p);mp=p
 else:m=pd.read_csv(REAL_M1[domain]);mp=REAL_M1[domain]
 m["run_id"]="real-cleanStatic"
 return b,m,mp

def mcols(d):return [c for c in d if c not in META and pd.api.types.is_numeric_dtype(d[c])]

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("artifacts/clif_ip_synthetic_normal_r4"));ap.add_argument("--regimes",nargs="*",default=["R0","S0","S1"],choices=["R0","S0","S1"]);ap.add_argument("--epochs",type=int,default=8);ap.add_argument("--adapt-epochs",type=int,default=6);ap.add_argument("--seed",type=int,default=41);a=ap.parse_args();torch.manual_seed(a.seed);np.random.seed(a.seed)
 device="cuda" if torch.cuda.is_available() else "cpu";out=a.root/"models";out.mkdir(parents=True,exist_ok=True)
 summary={"schema":"clif-ip.synthetic-normal.r4.training.v2","device":device,"seed":a.seed,"tap_layout":list(TAP_ORDER),"raw_tap_semantics":"Method-A magnitudes; signed 9D values are x-xhat innovations","regimes":{}}
 if "R0" in a.regimes:
  summary["regimes"]["R0"]={"OAKBAT":{"mode":"read-only exact R3 OAK baseline","path":str(ROOT/"artifacts/clif_ip_cross_layer_r3"),"modified":False,"comparison_scope":"historical_r3_noncomparable"},"TEXBAT":{"mode":"R0-TEX-real-only","fit_source":"cleanStatic train 0-240; validation 250-330; independent test >=340","comparison_scope":"r4_real_only"}}
  rb,rm,rmp=real_clean_frames(a.root,"SYN-TEX");cols=tap_columns(rb);rtr=rb[(rb.window_start_s>=0)&(rb.window_end_s<=240)];rva=rb[(rb.window_start_s>=250)&(rb.window_end_s<=330)]
  X,Y=sequences(rtr,cols);Xv,Yv=sequences(rva,cols);mu=X.reshape(-1,9).mean(0);sd=np.where(X.reshape(-1,9).std(0)>1e-6,X.reshape(-1,9).std(0),1.)
  rmodel=SharedPrnGRU().to(device);rh=fit_epochs(rmodel,X,Y,Xv,Yv,mu,sd,a.epochs,device,1e-3)
  rmtr=rm[(rm.window_start_s>=0)&(rm.window_end_s<=240)].copy().assign(split="train");rmc=mcols(rmtr);rst,raud=fit_multirun_ar(rmtr,rmc,pca_dim=min(8,len(rmc)),lag=6)
  rck={"architecture":"shared PRN-local GRU; no PRN ID","target_order":TAP_ORDER,"state_dict":rmodel.state_dict(),"mean":mu,"std":sd,"m1_state":rst,"m1_feature_columns":rmc,"domain":"SYN-TEX","regime":"R0","fit_roles":["real_clean_train_0_240"],"attack_rows":0,"real_rows":int(len(rtr)),"fit_source_hashes":{"b0":sha(REAL_B0["SYN-TEX"]),"m1":sha(rmp)}}
  rp=out/"R0_SYN-TEX.pt";torch.save(rck,rp);pd.DataFrame(rh).to_csv(out/"R0_SYN-TEX_history.csv",index=False)
  summary["regimes"]["R0"]["TEXBAT"].update({"checkpoint":str(rp),"checkpoint_sha256":sha(rp),"real_b0_rows":len(rtr),"real_m1_rows":len(rmtr),"attack_rows":0,"m1_fit_audit":raud})
 for domain in DOMAINS:
  tr,mtr=load_domain(a.root,domain,"train");va,mva=load_domain(a.root,domain,"validation");cols=tap_columns(tr)
  X,Y=sequences(tr,cols);Xv,Yv=sequences(va,cols);mu=X.reshape(-1,9).mean(0);sd=X.reshape(-1,9).std(0);sd=np.where(sd>1e-6,sd,1.)
  model=SharedPrnGRU().to(device);hist=fit_epochs(model,X,Y,Xv,Yv,mu,sd,a.epochs,device,1e-3)
  sm=mtr.assign(split="train");mc=mcols(sm);mstate,maudit=fit_multirun_ar(sm,mc,pca_dim=min(8,len(mc)),lag=6)
  ck={"architecture":"shared PRN-local GRU; no PRN ID","target_order":TAP_ORDER,"state_dict":copy.deepcopy(model.state_dict()),"mean":mu,"std":sd,"m1_state":mstate,"m1_feature_columns":mc,"fit_audit":maudit,"domain":domain,"regime":"S0","fit_roles":["synthetic_train"],"fit_source_hashes":sorted({sha(a.root/'runs'/r/'b0_nodes.csv') for r in tr.run_id.unique()}),"attack_rows":0,"real_rows":0}
  s0=out/f"S0_{domain}.pt";torch.save(ck,s0);s0hash=sha(s0);pd.DataFrame(hist).to_csv(out/f"S0_{domain}_history.csv",index=False)
  summary["regimes"].setdefault("S0",{})[domain]={"checkpoint":str(s0),"checkpoint_sha256":s0hash,"train_runs":int(tr.run_id.nunique()),"validation_runs":int(va.run_id.nunique()),"m1_fit_audit":maudit,"real_rows":0,"attack_rows":0}
  rb,rm,rmp=real_clean_frames(a.root,domain);rcols=tap_columns(rb);rtrain=rb[(rb.window_start_s>=0)&(rb.window_end_s<=240)];rval=rb[(rb.window_start_s>=250)&(rb.window_end_s<=330)]
  RX,RY=sequences(rtrain,rcols);RVX,RVY=sequences(rval,rcols);before=state_hash(model.state_dict());ah=fit_epochs(model,RX,RY,RVX,RVY,mu,sd,a.adapt_epochs,device,2e-4);after=state_hash(model.state_dict())
  if before==after:raise RuntimeError("S1 clean adaptation did not change weights")
  rmtrain=rm[(rm.window_start_s>=0)&(rm.window_end_s<=240)].copy().assign(split="train");rmc=mcols(rmtrain);common=[c for c in mc if c in rmc]
  # Freeze synthetic normalization/PCA; adapt only AR and innovation distribution.
  if common!=mc:raise RuntimeError("real clean input lacks one or more frozen synthetic M1 features")
  rmstate,rmaudit=adapt_m1_residual_state(mstate,rmtrain,common)
  real_hashes={"b0_cleanStatic":sha(REAL_B0[domain]),"m1_cleanStatic":sha(rmp)}
  ck1={"architecture":ck["architecture"],"target_order":TAP_ORDER,"state_dict":model.state_dict(),"mean":mu,"std":sd,"m1_synthetic_pretrain_state":mstate,"m1_state":rmstate,"m1_feature_columns":common,"fit_audit":{"synthetic_pretrain":maudit,"real_adaptation":rmaudit},"domain":domain,"regime":"S1","initialized_from":str(s0),"s0_checkpoint_sha256":s0hash,"before_weight_sha256":before,"after_weight_sha256":after,"fit_roles":["synthetic_train","real_clean_train_0_240"],"allowed_source_hashes":real_hashes,"real_b0_rows":len(rtrain),"real_m1_rows":len(rmtrain),"attack_rows":0,"threshold_role":"real_clean_validation_250_330_only"}
  s1=out/f"S1_{domain}.pt";torch.save(ck1,s1);pd.DataFrame(ah).to_csv(out/f"S1_{domain}_adaptation_history.csv",index=False)
  summary["regimes"].setdefault("S1",{})[domain]={"checkpoint":str(s1),"checkpoint_sha256":sha(s1),"initialized_from_sha256":s0hash,"before_weight_sha256":before,"after_weight_sha256":after,"weights_changed":before!=after,"real_clean_b0_rows":len(rtrain),"real_clean_m1_rows":len(rmtrain),"real_validation_b0_rows":len(rval),"allowed_source_hashes":real_hashes,"fit_roles":ck1["fit_roles"],"attack_rows":0,"m1_basis_frozen":rmaudit["basis_hash_before"]==rmaudit["basis_hash_after"],"m1_ar_changed":rmaudit["ar_hash_before"]!=rmaudit["ar_hash_after"],"m1_fit_audit":ck1["fit_audit"]}
 (a.root/"training_summary.json").write_text(json.dumps(summary,indent=2,default=lambda x:x.tolist() if hasattr(x,"tolist") else str(x))+"\n")
if __name__=="__main__":main()
