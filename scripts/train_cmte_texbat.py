#!/usr/bin/env python3
"""Train CMTE from a canonical cleanStatic node table and frozen B0 checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte import *
from gnss_doppler_lab.cmte_inputs import load_checkpoint, extract_role_innovations
DEFAULT_SHA="f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b"
DRIFT_GRID=(0.,.01,.025,.05,.1)

def quantile(x,q):
    a=np.asarray(x,float); return float(np.quantile(a,q,method="higher")) if len(a) else float("nan")
def binomial_tail(k,n,rate): return float(sum(math.comb(n,j)*rate**j*(1-rate)**(n-j) for j in range(k,n+1)))
def blocks(frame,score,threshold):
    x=frame.assign(_score=np.asarray(score),_block=np.floor(frame.availability_time_s/20).astype(int)); alarms=x._score>threshold
    maxima=x.groupby(["run_id","_block"])._score.max(); any_alarm=x.assign(_alarm=alarms).groupby(["run_id","_block"])._alarm.any()
    duration=max(1e-9,len(x)*.5/60); first=np.flatnonzero(alarms)
    return {"epoch_fpr":float(alarms.mean()),"false_alarms_per_min":float(alarms.sum()/duration),"censored_arl_epochs":float(first[0]+1 if len(first) else len(x)),
      "block_any_alarm_fraction":float(any_alarm.mean()),"blocks":int(len(maxima)),"block_maxima":maxima.tolist()}
def a1_scores(nodes,thresholds,rates):
    rows=[]
    for (run,t),g in nodes.groupby(["run_id","window_bin_s"],sort=True):
        vals=g.b0_prn_node_rmse.to_numpy(); surprises=[-math.log(max(binomial_tail(int(np.sum(vals>thresholds[q])),len(vals),rates[q]),1e-300)) for q in thresholds]
        rows.append({"run_id":run,"window_bin_s":t,"raw":max(surprises)})
    out=pd.DataFrame(rows); out["score"]=out.groupby("run_id",sort=False).raw.transform(lambda x:x.ewm(alpha=.75,adjust=False).mean()); return out.score.to_numpy()
def a3_scores(values,runs,drift):
    out=[]; prev=None; g=0.
    for z,r in zip(values,runs):
        if r!=prev:g=0.;prev=r
        g=max(0.,g+float(z)-drift);out.append(g)
    return np.asarray(out)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint",required=True); p.add_argument("--expected-sha",required=True)
    p.add_argument("--clean-node-csv",required=True); p.add_argument("--clean-manifest",required=True); p.add_argument("--out",required=True); p.add_argument("--device",default="cpu")
    a=p.parse_args(argv); ck=Path(a.checkpoint).resolve(strict=True); node=Path(a.clean_node_csv).resolve(strict=True); man=Path(a.clean_manifest).resolve(strict=True)
    actual=file_sha256(ck)
    if actual.lower()!=a.expected_sha.lower(): raise ValueError(f"checkpoint SHA-256 mismatch: {actual}")
    raw=pd.read_csv(node); provenance=validate_clean_provenance(man,raw,checkpoint_sha256=actual,node_path=node)
    model,features,mean,std,config=load_checkpoint(ck,actual,a.device,ROOT)
    roles=extract_role_innovations(raw,model,features,mean,std,seq_len=config.seq_len,device=a.device)
    if any(x.empty for x in roles.values()): raise ValueError("all split-reset roles must be nonempty")
    audit=audit_roles(roles); state=attach_calibration(fit_distribution(roles["train"],checkpoint_sha256=actual),roles["validation"])
    scored={k:score_residuals(v,state) for k,v in roles.items()}; val_nodes=scored["validation"]; val=aggregate_epochs(val_nodes)
    loge=np.log(np.maximum(val.mean_e,1e-300)); candidates=[]
    # Thresholds are 20 s validation block-max finite quantiles; detector/drift selection is normal-only.
    for detector in ("S1","S2"):
      for drift in ((0.,) if detector=="S1" else DRIFT_GRID):
        seq=sequential_scores(loge,val.run_id,drift=drift); score=seq.s1_log_capital if detector=="S1" else seq.s2_e_cusum
        maxima=val.assign(score=np.asarray(score),block=np.floor(val.availability_time_s/20).astype(int)).groupby(["run_id","block"]).score.max()
        threshold=quantile(maxima,.99); row={"detector":detector,"drift":drift,"threshold":threshold}; row.update(blocks(val,score,threshold)); candidates.append(row)
    # prereg ties: lower block alarm, FA/min; higher censored ARL; fixed S2; smaller drift.
    chosen=min(candidates,key=lambda r:(r["block_any_alarm_fraction"],r["false_alarms_per_min"],-r["censored_arl_epochs"],0 if r["detector"]=="S2" else 1,r["drift"]))
    train_rmse=roles["train"].b0_prn_node_rmse.to_numpy(); a1_thr={str(q):quantile(train_rmse,q/100) for q in (50,70,80)}; a1_rates={q:float(np.mean(train_rmse>v)) for q,v in a1_thr.items()}
    base=baseline_epoch_scores(val_nodes); a1=a1_scores(val_nodes,a1_thr,a1_rates)
    rmse_mean=val_nodes.groupby(["run_id","window_bin_s"],sort=True).b0_prn_node_rmse.mean().to_numpy(); a3_drift=float(np.median(rmse_mean)); a3=a3_scores(rmse_mean,val.run_id,a3_drift)
    thresholds={"source":"validation_only","attack_labels_used":False,"sequence":{"choice":chosen["detector"],"drift":chosen["drift"],"threshold":chosen["threshold"],"block_seconds":20,"finite_quantile":"higher q99","limited_blocks":chosen["blocks"]},
      "baselines":{"A0":{"definition":"max PRN scalar RMSE; node-level validation q99 threshold","threshold":quantile(val_nodes.b0_prn_node_rmse,.99)},"A1":{"definition":"max q50/q70/q80 exact-binomial surprise EWMA alpha=.75","train_node_thresholds":a1_thr,"train_exceedance_rates":a1_rates,"threshold":quantile(a1,.99)},
      "A2":{"definition":"epoch mean -log conformal p","threshold":quantile(base.A2,.99)},"A3":{"definition":"epoch mean scalar RMSE resettable CUSUM","drift":a3_drift,"threshold":quantile(a3,.99)},"A4":{"definition":"epoch mean mixture e","threshold":quantile(base.A4,.99)}}}
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False); save_state(state,out/"model_state.json")
    pd.DataFrame([{k:v for k,v in r.items() if k!="block_maxima"} for r in candidates]).to_csv(out/"candidate_table.csv",index=False)
    pd.concat([v.assign(split=k) for k,v in scored.items()],ignore_index=True).to_csv(out/"clean_per_prn.csv",index=False)
    pd.concat([aggregate_epochs(v).assign(split=k) for k,v in scored.items()],ignore_index=True).to_csv(out/"clean_per_epoch.csv",index=False)
    config_doc={"schema":"cmte-config-v2","default_method":"full_shrinkage_mahalanobis","q_semantics":"squared quadratic","splits":{"train":"windows fully inside [0,240)","validation":"windows fully inside [250,330)","test":"windows start >=340"},"split_reset":True,"seq_len":12,"checkpoint_sha256":actual}
    cadence_audits={k:v.attrs["cadence_chunk_audit"] for k,v in roles.items()}
    training={"clean_node":str(node),"clean_manifest":str(man),"source_node_sha256":file_sha256(node),"checkpoint":str(ck),"checkpoint_sha256":actual,"rows":{k:len(v) for k,v in roles.items()},"audit":audit,"cadence_chunk_audit":cadence_audits,"input_provenance":provenance,"historical_caveat":"CMTE-calibration-independent only; B0 trained across cleanStatic with PRN holdout"}
    shuffled_nodes=val_nodes.sample(frac=1,random_state=2026); shuffled_epoch=aggregate_epochs(shuffled_nodes)
    original_alarm=np.asarray(val.mean_e)>quantile(val.mean_e,.99); shuffled_alarm=np.asarray(shuffled_epoch.mean_e)>quantile(val.mean_e,.99)
    calibration={"reference":"validation residual Q_cal only","n":len(roles["validation"]),"fit_n":len(roles["train"]),"inclusive_plus_one":True,"validation_p_summary":{k:score_residuals(roles["validation"],state)[f"p_{k}"].describe().to_dict() for k in SCORE_METHODS},
      "validation_residual_order_shuffle":{"diagnostic_only":True,"seed":2026,"epoch_score_max_abs_delta":float(np.max(np.abs(val.mean_e.to_numpy()-shuffled_epoch.mean_e.to_numpy()))),"original_epoch_fpr":float(original_alarm.mean()),"shuffled_epoch_fpr":float(shuffled_alarm.mean()),"long_false_alarm_metrics_equal":bool(np.array_equal(original_alarm,shuffled_alarm))}}
    code_paths=[ROOT/"src/gnss_doppler_lab/cmte.py",ROOT/"src/gnss_doppler_lab/cmte_inputs.py",ROOT/"src/gnss_doppler_lab/tap_residual_common_drive.py",ROOT/"scripts/score_tap_residual_common_drive.py",ROOT/"scripts/prepare_cmte_texbat_inputs.py",ROOT/"scripts/eval_cmte_texbat.py",Path(__file__).resolve()]
    code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_paths}
    residual_manifest={"schema":"cmte-residual-manifest-v1","checkpoint":{"path":str(ck),"sha256":actual},"source_node":{"path":str(node),"sha256":file_sha256(node)},"source_manifest":{"path":str(man),"sha256":file_sha256(man)},"code_sha256":code,"semantics":{"features":list(RESIDUAL_COLUMNS),"tap_order":list(TAP_ORDER),"residual":"signed standardized target-minus-frozen-B0-prediction","availability":"window_end_s","q_default":"full shrinkage squared Mahalanobis"},"history":"partition nodes before extraction; reset each split/run/PRN and contiguous 0.5 s cadence chunk; first target index 12; never fill/interpolate/bridge","split_reset":True,"cadence_chunk_audit":cadence_audits,"roles":{k:{"rows":len(v),"run_ids":sorted(v.run_id.astype(str).unique())} for k,v in roles.items()}}
    for name,doc in (("config.json",config_doc),("training_summary.json",training),("calibration_summary.json",calibration),("thresholds.json",thresholds),("code_hashes.json",code),("residual_manifest.json",residual_manifest)):(out/name).write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"out":str(out),"checkpoint_sha256":actual,"rows":training["rows"]},sort_keys=True))
if __name__=="__main__": main()
