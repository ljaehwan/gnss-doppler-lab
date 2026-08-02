#!/usr/bin/env python3
"""Fit CMTE from an explicitly named cleanStatic frozen-B0 residual CSV."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte import (RESIDUAL_COLUMNS, aggregate_epochs, audit_roles,
    fit_shared_state, save_state, score_residuals, sequential_scores, validate_residual_frame)

DEFAULT_SHA="f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b"
DRIFT_GRID=(0.,.01,.025,.05,.1)

def sha256(path:Path)->str:
    h=hashlib.sha256();
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def quantile(x,q):
    a=np.asarray(x,float)
    return float(np.quantile(a,q,method="higher")) if len(a) else float("nan")

def normalize(frame):
    rename={f"residual_{i:03d}":f"residual_{i:03d}" for i in range(9)}
    if "prn_node_rmse" in frame and "b0_prn_node_rmse" not in frame: rename["prn_node_rmse"]="b0_prn_node_rmse"
    frame=frame.rename(columns=rename)
    validate_residual_frame(frame)
    return frame

def binomial_tail(k:int,n:int,rate:float)->float:
    from math import comb
    return float(sum(comb(n,j)*rate**j*(1-rate)**(n-j) for j in range(k,n+1)))

def a1_epoch_scores(nodes, train_rmse):
    rows=[]; qs={str(q):quantile(train_rmse,q/100) for q in (50,70,80)}
    rates={q:float(np.mean(train_rmse>thr)) for q,thr in qs.items()}
    for (run,t),g in nodes.groupby(["run_id","window_bin_s"],sort=True):
        row={"run_id":run,"window_bin_s":t}
        vals=g.b0_prn_node_rmse.to_numpy(float)
        for q,thr in qs.items(): row[f"q{q}"]=binomial_tail(int(np.sum(vals>thr)),len(vals),rates[q])
        rows.append(row)
    out=pd.DataFrame(rows)
    for q in qs:
        # alpha=.75 means 75% current score; reset by run.
        out[f"q{q}_ewma"]=out.groupby("run_id",sort=False)[f"q{q}"].transform(lambda x:x.ewm(alpha=.75,adjust=False).mean())
    return out,qs,rates

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-prn-csv",required=True); p.add_argument("--checkpoint",required=True)
    p.add_argument("--expected-sha",default=DEFAULT_SHA); p.add_argument("--out",required=True)
    a=p.parse_args(argv); csv=Path(a.clean_prn_csv).resolve(strict=True); ckpt=Path(a.checkpoint).resolve(strict=True)
    actual=sha256(ckpt)
    if actual.lower()!=a.expected_sha.lower(): raise ValueError(f"checkpoint SHA-256 mismatch: {actual}")
    frame=normalize(pd.read_csv(csv)); t=frame.window_end_s.astype(float)
    roles={"train":frame[(t>=0)&(t<240)].copy(),"validation":frame[(t>=250)&(t<330)].copy(),"test":frame[t>=340].copy()}
    if any(x.empty for x in roles.values()): raise ValueError("all clean split roles must be nonempty: train [0,240), validation [250,330), test >=340")
    audit=audit_roles(roles); state=fit_shared_state(roles["train"],checkpoint_sha256=actual)
    val_nodes=score_residuals(roles["validation"],state); val=aggregate_epochs(val_nodes)
    candidates=[]
    loge=np.log(np.maximum(val.mean_e.to_numpy(float),1e-300))
    for drift in DRIFT_GRID:
        seq=sequential_scores(loge,val.run_id.astype(str),drift=drift)
        score=seq.s2_e_cusum.to_numpy(); candidates.append({"drift":drift,"normal_mean":float(score.mean()),"normal_q99":quantile(score,.99)})
    # Predeclared normal-only stability: smallest q99, then smallest mean, then smallest drift.
    chosen=min(candidates,key=lambda x:(x["normal_q99"],x["normal_mean"],x["drift"]))
    val_seq=sequential_scores(loge,val.run_id.astype(str),drift=chosen["drift"])
    s1_blocks=[]; s2_blocks=[]
    for (run,b),g in val.assign(block=np.floor(val.availability_time_s/20).astype(int)).groupby(["run_id","block"]):
        idx=g.index.to_numpy(); s1_blocks.append(float(val_seq.iloc[idx].s1_log_capital.max())); s2_blocks.append(float(val_seq.iloc[idx].s2_e_cusum.max()))
    s1_thr=quantile(s1_blocks,.99); s2_thr=quantile(s2_blocks,.99)
    s1_fpr=float(np.mean(np.asarray(s1_blocks)>s1_thr)); s2_fpr=float(np.mean(np.asarray(s2_blocks)>s2_thr))
    full_choice="S2_e_CUSUM" if s2_fpr<=s1_fpr else "S1_parallel_restart"
    full_thr=s2_thr if full_choice=="S2_e_CUSUM" else s1_thr
    train_rmse=roles["train"].b0_prn_node_rmse.to_numpy(float)
    a1_val,a1_qs,a1_rates=a1_epoch_scores(val_nodes,train_rmse)
    rmse_epoch=val_nodes.groupby(["run_id","window_bin_s"],sort=True).b0_prn_node_rmse.mean().to_numpy()
    a3_drift=float(np.mean(rmse_epoch)); a3=[]; g=0.
    for z in rmse_epoch: g=max(0.,g+float(z)-a3_drift); a3.append(g)
    thresholds={"source":"validation_only","attack_labels_used":False,"attack_prefix_fitted":False,
      "epoch":{"q99":quantile(val.mean_e,.99),"q99_5":quantile(val.mean_e,.995),"target_epoch_fpr_1pct":quantile(val.mean_e,.99)},
      "sequence":{"block_seconds":20,"finite_sample_block_max_q99":full_thr,"drift":chosen["drift"],"full_choice":full_choice,"tie_rule":"S2",
        "validation_comparison":{"S1_parallel_restart":{"threshold":s1_thr,"blocks_any_alarm_rate":s1_fpr},"S2_e_CUSUM":{"threshold":s2_thr,"blocks_any_alarm_rate":s2_fpr}}},
      "baselines":{"A0":{"score":"epoch mean B0 RMSE","validation_q99":quantile(rmse_epoch,.99)},
        "A1":{"train_node_thresholds":a1_qs,"train_exceedance_rates":a1_rates,"alpha":.75,"default":"q70","validation_q99":quantile(-np.log(np.maximum(a1_val.q70_ewma,1e-300)),.99)},
        "A2":{"score":"epoch mean full-Mahalanobis e","validation_q99":quantile(val.mean_e,.99)},
        "A3":{"score":"epoch mean RMSE CUSUM","validation_drift":a3_drift,"validation_q99":quantile(a3,.99)},
        "A4":{"score":"epoch mean mixture e","validation_q99":quantile(val.mean_e,.99)}},
      "selection_audit":{"grid":list(DRIFT_GRID),"criterion":"minimum normal validation q99; mean then drift ties","candidates":candidates}}
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    save_state(state,out/"model_state.json")
    config={"default_method":"full_shrinkage_mahalanobis","ablations":["rmse","diag_mahalanobis","full_shrinkage_mahalanobis","max_standardized_tap"],"kappas":[.25,.5,.75],"splits":{"train":"[0,240)","validation":"[250,330)","test":">=340"},"state_reset":"each split and run","checkpoint_sha256":actual}
    training={"source":str(csv),"source_sha256":sha256(csv),"rows":{k:len(v) for k,v in roles.items()},"audit":audit,"historical_caveat":"independent clean test is CMTE-calibration-independent but not B0-training-independent"}
    calibration={"reference":"clean train residuals only","n":len(roles["train"]),"finite_sample_p":"(1 + count(cal >= q))/(n + 1)","inclusive_ties":True}
    for name,obj in (("config.json",config),("training.json",training),("calibration.json",calibration),("thresholds.json",thresholds)):
        (out/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    pd.concat([score_residuals(v,state).assign(split=k) for k,v in roles.items()],ignore_index=True).to_csv(out/"clean_per_prn.csv",index=False)
    pd.concat([aggregate_epochs(score_residuals(v,state)).assign(split=k) for k,v in roles.items()],ignore_index=True).to_csv(out/"clean_per_epoch.csv",index=False)
    print(json.dumps({"out":str(out),"checkpoint_sha256":actual,"rows":training["rows"]},sort_keys=True))
if __name__=="__main__": main()
