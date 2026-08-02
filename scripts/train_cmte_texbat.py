#!/usr/bin/env python3
"""Train CMTE from a canonical cleanStatic node table and frozen B0 checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte import *
from gnss_doppler_lab.cmte_inputs import load_checkpoint, extract_role_innovations
DEFAULT_SHA="f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b"
DRIFT_GRID=(0.,.01,.025,.05,.1)

def quantile(x,q):
    a=np.asarray(x,float); return float(np.quantile(a,q,method="higher")) if len(a) else float("nan")
def operating_points(x):
    """Validation-only epoch operating points; target1 is finite-sample q99."""
    return {"q99":quantile(x,.99),"q995":quantile(x,.995),"target1":quantile(x,.99)}
def binomial_tail(k,n,rate): return float(sum(math.comb(n,j)*rate**j*(1-rate)**(n-j) for j in range(k,n+1)))
def blocks(frame,score,threshold):
    x=frame.assign(_score=np.asarray(score),_block=np.floor(frame.availability_time_s/20).astype(int)); alarms=x._score>threshold
    run_col="recording_id" if "recording_id" in x else "run_id"; rising=np.zeros(len(x),dtype=bool); first_lengths=[]; sequence_any=[]
    for _,indices in x.groupby(run_col,sort=False).indices.items():
        idx=np.asarray(indices); a=alarms.iloc[idx].to_numpy(); edges=a & ~np.r_[False,a[:-1]]; rising[idx]=edges
        hits=np.flatnonzero(edges); first_lengths.append(int(hits[0]+1) if len(hits) else len(a)); sequence_any.append(bool(a.any()))
    maxima=x.groupby([run_col,"_block"])._score.max(); any_alarm=x.assign(_alarm=alarms).groupby([run_col,"_block"])._alarm.any()
    duration=max(1e-9,len(x)*.5/60)
    return {"epoch_fpr":float(alarms.mean()),"alarm_epoch_count":int(alarms.sum()),"alarm_epoch_occupancy_per_min":float(alarms.sum()/duration),
      "false_alarm_events":int(rising.sum()),"false_alarms_per_min":float(rising.sum()/duration),"sequence_any_alarm_fraction":float(np.mean(sequence_any)),
      "first_crossing_epoch":int(len(x) if not rising.any() else np.flatnonzero(rising)[0]+1),"first_crossing_censored":bool(not rising.any()),"censored_run_length_epochs":float(np.mean(first_lengths)),"censored_arl_epochs":float(np.mean(first_lengths)),
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
        ops=operating_points(maxima); threshold=ops["target1"]
        row={"detector":detector,"drift":drift,"threshold":threshold,"operating_points":ops,"block_maxima_n":len(maxima)}; row.update(blocks(val,score,threshold)); candidates.append(row)
    # prereg ties: lower block alarm, FA/min; higher censored ARL; fixed S2; smaller drift.
    chosen=min(candidates,key=lambda r:(r["block_any_alarm_fraction"],r["false_alarms_per_min"],-r["censored_arl_epochs"],0 if r["detector"]=="S2" else 1,r["drift"]))
    train_rmse=roles["train"].b0_prn_node_rmse.to_numpy(); a1_thr={str(q):quantile(train_rmse,q/100) for q in (50,70,80)}; a1_rates={q:float(np.mean(train_rmse>v)) for q,v in a1_thr.items()}
    base=baseline_epoch_scores(val_nodes); a1=a1_scores(val_nodes,a1_thr,a1_rates)
    rmse_mean=val_nodes.groupby(["run_id","window_bin_s"],sort=True).b0_prn_node_rmse.mean().to_numpy(); a3_drift=float(np.median(rmse_mean)); a3=a3_scores(rmse_mean,val.run_id,a3_drift)
    baseline_values={"A0":base.A0,"A1":a1,"A2":base.A2,"A3":a3,"A4":base.A4}
    definitions={"A0":"epoch max PRN scalar RMSE","A1":"max q50/q70/q80 exact-binomial surprise EWMA alpha=.75","A2":"epoch mean -log conformal p","A3":"epoch mean scalar RMSE resettable CUSUM","A4":"epoch mean mixture e"}
    thresholds={"source":"validation_only","attack_labels_used":False,"operating_point_names":["q99","q995","target1"],"epoch_target":"empirical FPR <=1% using higher q99","sequence":{"choice":chosen["detector"],"drift":chosen["drift"],"threshold":chosen["threshold"],"operating_points":chosen["operating_points"],"block_seconds":20,"criterion":"validation 20 s block-max empirical target; no attack selection","finite_sample_limitation":f"only {chosen['blocks']} validation blocks; q99/q99.5 collapse to observed order statistics"},"baselines":{}}
    for name,values in baseline_values.items():
        thresholds["baselines"][name]={"definition":definitions[name],"threshold":operating_points(values)["target1"],"operating_points":operating_points(values)}
    thresholds["baselines"]["A1"].update({"train_node_thresholds":a1_thr,"train_exceedance_rates":a1_rates})
    thresholds["baselines"]["A3"]["drift"]=a3_drift
    source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    state.metadata.update({"source_commit":source_commit,"scoring_semantics":"cmte-v3-recording-identity-log-domain-s1"})
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False); save_state(state,out/"model_state.json")
    pd.DataFrame([{k:v for k,v in r.items() if k!="block_maxima"} for r in candidates]).to_csv(out/"candidate_table.csv",index=False)
    pd.concat([v.assign(split=k) for k,v in scored.items()],ignore_index=True).to_csv(out/"clean_per_prn.csv",index=False)
    pd.concat([aggregate_epochs(v).assign(split=k) for k,v in scored.items()],ignore_index=True).to_csv(out/"clean_per_epoch.csv",index=False)
    config_doc={"schema":"cmte-config-v3","source_commit":source_commit,"timing_policy":{"authority":"user-requested TEX policy","nominal_onset_s":100,"stable_s":[30,90],"transition_s":[90,110],"established_from_s":110,"clock":"availability_time_s","DS4_caveat":"short post-onset coverage"},"default_method":"full_shrinkage_mahalanobis","q_semantics":"squared quadratic","splits":{"train":"windows fully inside [0,240)","validation":"windows fully inside [250,330)","test":"windows start >=340"},"split_reset":True,"seq_len":12,"checkpoint_sha256":actual}
    cadence_audits={k:v.attrs["cadence_chunk_audit"] for k,v in roles.items()}
    training={"source_commit":source_commit,"clean_node":str(node),"clean_manifest":str(man),"source_node_sha256":file_sha256(node),"checkpoint":str(ck),"checkpoint_sha256":actual,"rows":{k:len(v) for k,v in roles.items()},"audit":audit,"cadence_chunk_audit":cadence_audits,"input_provenance":provenance,"historical_caveat":"CMTE-calibration-independent only; B0 trained across cleanStatic with PRN holdout"}
    rng=np.random.default_rng(2026); shuffled_epoch=pd.concat([g.iloc[rng.permutation(len(g))].reset_index(drop=True) for _,g in val.groupby("recording_id",sort=False)],ignore_index=True)
    val_seq=sequential_scores(np.log(val.mean_e),val.recording_id,drift=chosen["drift"]); shuffled_seq=sequential_scores(np.log(shuffled_epoch.mean_e),shuffled_epoch.recording_id,drift=chosen["drift"])
    seq_column="s1_log_capital" if chosen["detector"]=="S1" else "s2_e_cusum"
    original_long=blocks(val,val_seq[seq_column],chosen["threshold"]); shuffled_long=blocks(shuffled_epoch,shuffled_seq[seq_column],chosen["threshold"])
    calibration={"reference":"validation residual Q_cal only","n":len(roles["validation"]),"fit_n":len(roles["train"]),"inclusive_plus_one":True,"validation_p_summary":{k:score_residuals(roles["validation"],state)[f"p_{k}"].describe().to_dict() for k in SCORE_METHODS},
      "validation_normal_order_shuffle":{"diagnostic_only":True,"interpretation":"order sensitivity only; equality is not a success condition","seed":2026,"aggregated_before_permutation":True,"order_actually_changed":bool(not np.array_equal(val.mean_e.to_numpy(),shuffled_epoch.mean_e.to_numpy())),"epoch_multiset_identical":bool(np.allclose(np.sort(val.mean_e),np.sort(shuffled_epoch.mean_e))),"original_reset_count":val_seq.attrs["reset_count"],"shuffled_reset_count":shuffled_seq.attrs["reset_count"],"original_final_state":float(val_seq[seq_column].iloc[-1]),"shuffled_final_state":float(shuffled_seq[seq_column].iloc[-1]),"original_max_state":float(val_seq[seq_column].max()),"shuffled_max_state":float(shuffled_seq[seq_column].max()),"original_sequential_long_fa":original_long,"shuffled_sequential_long_fa":shuffled_long}}
    code_paths=[ROOT/"src/gnss_doppler_lab/cmte.py",ROOT/"src/gnss_doppler_lab/cmte_inputs.py",ROOT/"src/gnss_doppler_lab/tap_residual_common_drive.py",ROOT/"scripts/score_tap_residual_common_drive.py",ROOT/"scripts/prepare_cmte_texbat_inputs.py",ROOT/"scripts/eval_cmte_texbat.py",Path(__file__).resolve()]
    code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_paths}
    residual_manifest={"schema":"cmte-residual-manifest-v1","source_commit":source_commit,"checkpoint":{"path":str(ck),"sha256":actual},"source_node":{"path":str(node),"sha256":file_sha256(node)},"source_manifest":{"path":str(man),"sha256":file_sha256(man)},"code_sha256":code,"semantics":{"features":list(RESIDUAL_COLUMNS),"tap_order":list(TAP_ORDER),"residual":"signed standardized target-minus-frozen-B0-prediction","availability":"window_end_s","q_default":"full shrinkage squared Mahalanobis"},"history":"partition nodes before extraction; reset each split/run/PRN and contiguous 0.5 s cadence chunk; first target index 12; never fill/interpolate/bridge","split_reset":True,"cadence_chunk_audit":cadence_audits,"roles":{k:{"rows":len(v),"run_ids":sorted(v.run_id.astype(str).unique())} for k,v in roles.items()}}
    for name,doc in (("config.json",config_doc),("training_summary.json",training),("calibration_summary.json",calibration),("thresholds.json",thresholds),("code_hashes.json",code),("residual_manifest.json",residual_manifest)):(out/name).write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"out":str(out),"checkpoint_sha256":actual,"rows":training["rows"]},sort_keys=True))
if __name__=="__main__": main()
