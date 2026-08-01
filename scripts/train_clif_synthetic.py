#!/usr/bin/env python3
"""Train R4 R0/S0/S1 without run leakage (shared PRN-local GRU, no PRN ID)."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.clif_ip_synthetic import DOMAINS,TAP_ORDER,fit_multirun_ar,history_design

class SharedPrnGRU(nn.Module):
 """One shared predictor for variable PRNs; PRN identity is deliberately absent."""
 def __init__(self,features=9,hidden=32):super().__init__();self.gru=nn.GRU(features,hidden,batch_first=True);self.head=nn.Linear(hidden,9)
 def forward(self,x):return self.head(self.gru(x)[0][:,-1])

def clean_only(d):
 if "label" in d and not d.label.astype(str).str.lower().eq("normal").all():raise ValueError("attack/spoof rows may not enter fitting")
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
 cols=[f"tap_{x}_rel_prompt_mean" for x in TAP_ORDER]
 if not set(cols)<=set(d):raise ValueError("signed target source must preserve canonical 9-tap order")
 return cols

def sequences(d,cols,lag=12):
 X=[];Y=[]
 for (_, _),g in d.groupby(["run_id","prn"],sort=False):
  g=g.sort_values("window_bin_s");z=g[cols].to_numpy(np.float32)
  for i in range(lag,len(z)):X.append(z[i-lag:i]);Y.append(z[i])
 return np.asarray(X,np.float32),np.asarray(Y,np.float32)

def train_gru(train,val,cols,epochs,device):
 X,Y=sequences(train,cols);Xv,Yv=sequences(val,cols)
 if not len(X) or not len(Xv):raise ValueError("insufficient run-local B0 history")
 mu=X.reshape(-1,9).mean(0);sd=X.reshape(-1,9).std(0);sd=np.where(sd>1e-6,sd,1.)
 model=SharedPrnGRU().to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-3);best=None;history=[]
 xt=torch.tensor((X-mu)/sd,device=device);yt=torch.tensor((Y-mu)/sd,device=device);xv=torch.tensor((Xv-mu)/sd,device=device);yv=torch.tensor((Yv-mu)/sd,device=device)
 for e in range(epochs):
  model.train();opt.zero_grad();loss=((model(xt)-yt)**2).mean();loss.backward();opt.step();model.eval()
  with torch.no_grad():vl=float(((model(xv)-yv)**2).mean())
  history.append({"epoch":e+1,"train_mse":float(loss),"validation_mse":vl})
  if best is None or vl<best[0]:best=(vl,{k:v.detach().cpu() for k,v in model.state_dict().items()})
 model.load_state_dict(best[1]);return model,mu,sd,history

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("artifacts/clif_ip_synthetic_normal_r4"));ap.add_argument("--regimes",nargs="*",default=["S0","S1"],choices=["R0","S0","S1"]);ap.add_argument("--epochs",type=int,default=20);ap.add_argument("--seed",type=int,default=41);a=ap.parse_args();torch.manual_seed(a.seed)
 cuda_ok=torch.cuda.is_available()
 try:
  if cuda_ok:torch.zeros(1,device="cuda");device="cuda"
  else:device="cpu"
 except Exception as e:device="cpu";cuda_error=str(e)
 out=a.root/"models";out.mkdir(parents=True,exist_ok=True);summary={"schema":"clif-ip.synthetic-normal.r4.training.v1","device":device,"cuda_probe":cuda_ok,"seed":a.seed,"regimes":{}}
 for regime in a.regimes:
  if regime=="R0":
   frozen=ROOT/"artifacts/clif_ip_cross_layer_r3";summary["regimes"][regime]={"mode":"read-only exact frozen baseline","path":str(frozen),"modified":False};continue
  for domain in DOMAINS:
   tr,mtr=load_domain(a.root,domain,"train");va,mva=load_domain(a.root,domain,"validation");cols=tap_columns(tr);model,mu,sd,h=train_gru(tr,va,cols,a.epochs,device)
   mcols=[c for c in mtr if c not in {"run_id","window_index","t","window_start_s","window_end_s","start_sample","end_sample","block_ms","stride_s","split"}]
   mtr=mtr.assign(split="train");state,audit=fit_multirun_ar(mtr,mcols,pca_dim=8,lag=6)
   ck={"architecture":"shared PRN-local GRU; no PRN ID","target_order":TAP_ORDER,"state_dict":model.state_dict(),"mean":mu,"std":sd,"m1_state":state,"fit_audit":audit,"domain":domain,"regime":regime,
    "adaptation_contract":"S1 starts from S0 then real cleanStatic chronological adaptation only; no attack/pre-onset calibration" if regime=="S1" else "synthetic train only; validation only for selection/calibration"}
   path=out/f"{regime}_{domain}.pt";torch.save(ck,path);pd.DataFrame(h).to_csv(out/f"{regime}_{domain}_history.csv",index=False)
   summary["regimes"].setdefault(regime,{})[domain]={"checkpoint":str(path),"train_runs":int(tr.run_id.nunique()),"validation_runs":int(va.run_id.nunique()),"m1_fit_audit":audit,"note":"S1 real adaptation is required before final evaluation and is not silently substituted"}
 (a.root/"training_summary.json").write_text(json.dumps(summary,indent=2,default=lambda x:x.tolist() if hasattr(x,"tolist") else str(x))+"\n")
if __name__=="__main__":main()
