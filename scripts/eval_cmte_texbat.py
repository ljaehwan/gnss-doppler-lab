#!/usr/bin/env python3
"""Evaluate a frozen CMTE state on explicitly supplied TEXBAT scenario CSVs."""
from __future__ import annotations
import argparse, hashlib, json, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte import (SCORE_METHODS, aggregate_epochs, epoch_masks, label_epochs,
    load_state, score_residuals, sequential_scores, validate_residual_frame)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def binom_tail(k,n,p):
    return sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k,n+1))
def normalize(f):
    if "prn_node_rmse" in f and "b0_prn_node_rmse" not in f:f=f.rename(columns={"prn_node_rmse":"b0_prn_node_rmse"})
    validate_residual_frame(f); return f
def metrics(score,times,threshold):
    score=np.asarray(score,float); times=np.asarray(times,float); masks=epoch_masks(times); stable=masks["stable"]; alarm=score>threshold
    duration=max(1e-9,(times[stable].max()-times[stable].min())/60) if stable.any() else float("nan")
    established=np.flatnonzero(alarm&masks["established"])
    return {"threshold":threshold,"epoch_fpr_stable":float(alarm[stable].mean()) if stable.any() else float("nan"),
      "false_alarms_per_min":float(alarm[stable].sum()/duration) if stable.any() else float("nan"),
      "normal_arl_epochs":float(1/max(alarm[stable].mean(),1/max(1,stable.sum()))) if stable.any() else float("nan"),
      "blocks_any_alarm":int(len(set(np.floor(times[stable&alarm]/20)))) if stable.any() else 0,
      "first_established_alarm_available_s":None if len(established)==0 else float(times[established[0]])}
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--state-dir",required=True)
    p.add_argument("--scenario",action="append",required=True,help="name=/explicit/path.csv (repeat)")
    p.add_argument("--out",required=True); a=p.parse_args(argv); state_dir=Path(a.state_dir).resolve(strict=True)
    config=json.loads((state_dir/"config.json").read_text()); expected=config["checkpoint_sha256"]
    # Verify both deterministic state checksum and frozen checkpoint pin before any scenario is opened.
    state=load_state(state_dir/"model_state.json",expected_checkpoint_sha256=expected)
    thresholds=json.loads((state_dir/"thresholds.json").read_text())
    specs=[]
    for spec in a.scenario:
        if "=" not in spec:raise ValueError("scenario must be name=path")
        name,path=spec.split("=",1)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*",name):raise ValueError("unsafe scenario name")
        specs.append((name,Path(path).resolve(strict=True)))
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    for d in ("scenario","ablation","per_epoch","per_prn","diagnostics","plots","provenance"): (out/d).mkdir()
    all_summary={}; provenance={"state_dir":str(state_dir),"state_sha256":sha(state_dir/"model_state.json"),"checkpoint_sha256":expected,"attack_tuning":False,"scenario_inputs":{}}
    for name,path in specs:
        raw=normalize(pd.read_csv(path)); provenance["scenario_inputs"][name]={"path":str(path),"sha256":sha(path),"rows":len(raw)}
        nodes=score_residuals(raw,state); nodes.to_csv(out/"per_prn"/f"{name}.csv",index=False)
        epoch=aggregate_epochs(nodes); seq=sequential_scores(np.log(np.maximum(epoch.mean_e,1e-300)),epoch.run_id,drift=thresholds["sequence"]["drift"])
        epoch=pd.concat([epoch,seq],axis=1); epoch["texbat_phase"]=label_epochs(epoch.availability_time_s); epoch.to_csv(out/"per_epoch"/f"{name}.csv",index=False)
        summaries={}
        # A0 scalar epoch-mean frozen B0 RMSE.
        rmse=nodes.groupby(["run_id","window_bin_s"],sort=True).b0_prn_node_rmse.mean().to_numpy()
        summaries["A0"]=metrics(rmse,epoch.availability_time_s,thresholds["baselines"]["A0"]["validation_q99"])
        # A1 q70 exact binomial tail then causal EWMA alpha=.75, reset by run.
        a1=thresholds["baselines"]["A1"]; vals=[]
        for _,g in nodes.groupby(["run_id","window_bin_s"],sort=True):
            k=int(np.sum(g.b0_prn_node_rmse.to_numpy()>a1["train_node_thresholds"]["70"])); vals.append(binom_tail(k,len(g),a1["train_exceedance_rates"]["70"]))
        a1raw=pd.Series(vals,index=epoch.index); a1ew=a1raw.groupby(epoch.run_id,sort=False).transform(lambda x:x.ewm(alpha=.75,adjust=False).mean()); a1score=-np.log(np.maximum(a1ew,1e-300))
        summaries["A1"]=metrics(a1score,epoch.availability_time_s,a1["validation_q99"])
        summaries["A2"]=metrics(epoch.mean_e,epoch.availability_time_s,thresholds["baselines"]["A2"]["validation_q99"])
        drift=thresholds["baselines"]["A3"]["validation_drift"]; g=0.; a3=[]
        for z in rmse:g=max(0.,g+float(z)-drift);a3.append(g)
        summaries["A3"]=metrics(a3,epoch.availability_time_s,thresholds["baselines"]["A3"]["validation_q99"])
        summaries["A4"]=metrics(epoch.mean_e,epoch.availability_time_s,thresholds["baselines"]["A4"]["validation_q99"])
        full_score=epoch.s2_e_cusum if thresholds["sequence"]["full_choice"]=="S2_e_CUSUM" else epoch.s1_log_capital
        summaries["Full"]=metrics(full_score,epoch.availability_time_s,thresholds["sequence"]["finite_sample_block_max_q99"])
        (out/"scenario"/f"{name}.json").write_text(json.dumps(summaries,indent=2,sort_keys=True)+"\n"); all_summary[name]=summaries
        abl=[]
        for method in SCORE_METHODS:
            temp=nodes.copy();temp["p"]=temp[f"p_{method}"];temp["e"]=temp[f"e_{method}"]
            z=aggregate_epochs(temp)
            for _,r in z.iterrows():abl.append({"method":method,**r.to_dict()})
        pd.DataFrame(abl).to_csv(out/"ablation"/f"{name}.csv",index=False)
        perm=epoch.sample(frac=1,random_state=2026); alt=sequential_scores(np.log(np.maximum(perm.mean_e,1e-300)),[name]*len(perm),drift=thresholds["sequence"]["drift"])
        diagnostic={"diagnostic_only":True,"epoch_multiset_unchanged":sorted(epoch.mean_e.tolist())==sorted(perm.mean_e.tolist()),"sequential_trajectory_changed":not np.allclose(epoch.s2_e_cusum,alt.s2_e_cusum),"seed":2026}
        (out/"diagnostics"/f"{name}_order_shuffle.json").write_text(json.dumps(diagnostic,indent=2)+"\n")
    (out/"provenance"/"provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
    (out/"scenario"/"summary.json").write_text(json.dumps(all_summary,indent=2,sort_keys=True)+"\n")
    (out/"plots"/"README.md").write_text("Plot inputs are the immutable per_epoch and ablation CSV outputs; no plots are fabricated by the synthetic smoke.\n")
    (out/"README.md").write_text("CMTE evaluation outputs. Thresholds and drift are frozen from clean validation only; scenario data are evaluation-only. Historical frozen-B0 reports, if cited externally, are historical_noncomparable.\n")
    checks={str(x.relative_to(out)):sha(x) for x in sorted(out.rglob("*")) if x.is_file()}
    (out/"checksums.json").write_text(json.dumps(checks,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"out":str(out),"scenarios":[x[0] for x in specs],"checkpoint_sha256":expected},sort_keys=True))
if __name__=="__main__":main()
