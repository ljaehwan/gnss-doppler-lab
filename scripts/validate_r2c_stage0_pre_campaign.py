#!/usr/bin/env python3
"""Read-only pre-campaign replay and synthetic/clean-only smoke.

This command deliberately has no attack-label or metric code and refuses every
path inside either campaign artifact directory.
"""
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.r2c_stage0_fix import (ComplexWhitener, TemplateProvider,
 historical_b0_status, joint_profile_glrt, run_full_controls,score_b0_nodes,replay_b0_events)

PRESERVED=ROOT/"artifacts/r2c_gnss_stage0"
CAMPAIGN=ROOT/"artifacts/r2c_gnss_stage0_fix"

def safe_report(path: Path):
    resolved=path.resolve()
    if resolved==PRESERVED.resolve() or PRESERVED.resolve() in resolved.parents or resolved==CAMPAIGN.resolve() or CAMPAIGN.resolve() in resolved.parents:
        raise ValueError("pre-campaign report cannot be written in artifact trees")
    return resolved

def synthetic_smoke():
    import importlib.util
    spec=importlib.util.spec_from_file_location("r2c_fix_operational_runner",ROOT/"scripts/run_r2c_gnss_stage0_fix.py");runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    config=json.loads((ROOT/"configs/r2c_gnss_stage0_fix.json").read_text());rows,los=runner.synthetic_inputs(config["seed"]);taps=np.asarray(config["tap_offsets_chips"]);grid=np.asarray(config["delay_grid_chips"]);provider=TemplateProvider.analytic()
    data={"y":np.asarray([r[2] for r in rows]),"time":np.asarray([r[0] for r in rows]),"prn":np.asarray([r[1] for r in rows]),"cn0":np.asarray([r[3] for r in rows])};models=runner.fit_frozen_models(data,config,provider,taps,grid,require_gpu=False)
    obs={p:data["y"][data["prn"]==p][:2] for p in sorted(los)};raw=runner.h0_residuals(np.concatenate(list(obs.values())),provider,taps,grid);q=np.mean(np.abs(raw)**2,axis=1);energy=np.mean(np.abs(np.concatenate(list(obs.values())))**2,axis=1);conditions={};cursor=0
    for p,y in obs.items():
        n=len(y);conditions[p]=(np.column_stack((np.full(n,40.),q[cursor:cursor+n])),np.column_stack((np.full(n,40.),q[cursor:cursor+n],energy[cursor:cursor+n])));cursor+=n
    smoke={**config,"optimizer_starts_m":[config["optimizer_starts_m"][0]]};scores,fits,_=runner.score_bin(obs,los,provider,taps,grid,models,smoke,conditions)
    # Calibrate the operational Full graph itself; controls use that exact q99.
    calibration=[]
    for offset in range(12):
        shifted={p:data["y"][data["prn"]==p][offset:offset+2] for p in sorted(los)}
        if any(len(v)!=2 for v in shifted.values()):continue
        shifted_scores,_,_=runner.score_bin(shifted,los,provider,taps,grid,models,smoke,conditions)
        if shifted_scores["Full"] is not None:calibration.append(shifted_scores["Full"])
    if not calibration:raise RuntimeError("no valid operational Full calibration support")
    threshold=runner.calibration_thresholds(calibration,["normal_calibration"]*len(calibration))["q99"]
    controls=run_full_controls(fits["FullScorer"],obs,los,threshold,provider,taps)
    if controls["baseline_score"]!=scores["Full"]:raise RuntimeError("pre-campaign control baseline differs from operational Full")
    if controls["threshold"]!=threshold:raise RuntimeError("control threshold differs from calibrated Full q99")
    return {"status":"PASS","template_mode":"analytic_gps_ca_acf","analytic_approximation":True,"paper_comparison_ready":False,
      "scores":scores,"fit_statuses":fits["statuses"],"controls":controls,"device":"cuda_available_checked_separately","gpu_required_for_campaign_training":True}

def b0_replay(paths,node_paths,checkpoint_path,expected,expected_checkpoint):
    import pandas as pd,torch
    base=historical_b0_status(paths,expected);checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    for scenario,path in paths.items():
        item=base["scenarios"][scenario]
        try:
            saved_for_events=pd.read_csv(path);events=replay_b0_events(saved_for_events);event_rows=events.to_dict("records")
            native_keys=saved_for_events[["run_id","prn","window_bin_s"]].astype(str).agg("|".join,axis=1).sort_values().tolist();native_timing=saved_for_events[["window_start_s","window_mid_s"]].copy();native_timing["window_end_s"]=saved_for_events["window_end_s"] if "window_end_s" in saved_for_events else saved_for_events.window_start_s+1.
            item.update({"saved_score_path":str(path.resolve()),"saved_score_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"event_rows":event_rows,
              "saved_key_set_sha256":hashlib.sha256("\n".join(native_keys).encode()).hexdigest(),"saved_timing_sha256":hashlib.sha256(native_timing.to_numpy(float).astype("<f8").tobytes()).hexdigest(),
              "event_rows_sha256":hashlib.sha256(json.dumps(event_rows,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()})
        except Exception as exc:item.update({"event_rows":[],"event_validation_reason":str(exc)})
        node_path=node_paths.get(scenario)
        if node_path is None or not node_path.is_file():
            if item["status"]!="UNAVAILABLE_AUTHENTIC_INTERFACE":item["status"]="AVAILABLE_SAVED_NATIVE_SCORE_REPLAY_WITH_NODE_LINEAGE_GAP"
            continue
        try:
            nodes=pd.read_csv(node_path);generated=score_b0_nodes(nodes,checkpoint,device="cpu",checkpoint_path=checkpoint_path,expected_checkpoint_sha256=expected_checkpoint);saved=pd.read_csv(path)
            lineage=("channel","segment_index")
            if any(x in generated.columns and x not in saved.columns for x in lineage):raise ValueError("saved scores omit generated channel/segment lineage")
            keys=["run_id","prn","window_bin_s"]+[x for x in lineage if x in generated.columns and x in saved.columns]
            for frame in (generated,saved):
                frame["prn"]=frame.prn.astype(str).str.lstrip("Gg").astype(int)
            generated_keys=set(map(tuple,generated[keys].itertuples(index=False,name=None)));saved_keys=set(map(tuple,saved[keys].itertuples(index=False,name=None)))
            if generated_keys!=saved_keys:raise ValueError(f"native score key-set mismatch generated_only={len(generated_keys-saved_keys)} saved_only={len(saved_keys-generated_keys)}")
            timing=[x for x in ("window_start_s","window_end_s","window_mid_s") if x in saved.columns]
            joined=generated.merge(saved[keys+timing+["prn_node_rmse"]],on=keys,suffixes=("_generated","_saved"),validate="one_to_one")
            for field in timing:
                if not np.allclose(joined[f"{field}_generated"],joined[f"{field}_saved"],rtol=0,atol=1e-9):raise ValueError(f"native {field} parity mismatch")
            if not np.allclose(joined.availability_time_s,joined.window_end_s_generated,rtol=0,atol=1e-9):raise ValueError("score availability parity mismatch")
            if not np.allclose(joined.prn_node_rmse_generated,joined.prn_node_rmse_saved,rtol=2e-6,atol=2e-7):raise ValueError("native node-to-score parity mismatch")
            key_hash=hashlib.sha256("\n".join(map(str,sorted(generated_keys))).encode()).hexdigest();timing_hash=hashlib.sha256(joined[[f"{x}_generated" for x in timing]+["availability_time_s"]].to_numpy(float).astype("<f8").tobytes()).hexdigest()
            item.update({"status":"AVAILABLE_AUTHENTIC_NODE_TO_SCORE_REPLAY","node_csv_path":str(node_path.resolve()),"node_csv_sha256":hashlib.sha256(node_path.read_bytes()).hexdigest(),"matched_scores":len(joined),"parity_key_set_sha256":key_hash,"parity_timing_sha256":timing_hash,"maximum_abs_score_error":float(np.max(np.abs(joined.prn_node_rmse_generated-joined.prn_node_rmse_saved)))})
        except Exception as exc:
            fallback=item.get("status")=="AVAILABLE_SAVED_NATIVE_SCORE_REPLAY_WITH_NODE_LINEAGE_GAP"
            item.update({"status":"AVAILABLE_SAVED_NATIVE_SCORE_REPLAY_WITH_NODE_LINEAGE_GAP" if fallback else "UNAVAILABLE_AUTHENTIC_INTERFACE","node_replay_reason":str(exc)})
    base["status"]="AVAILABLE_AUTHENTIC_NODE_TO_SCORE_REPLAY" if all(x["status"]=="AVAILABLE_AUTHENTIC_NODE_TO_SCORE_REPLAY" for x in base["scenarios"].values()) else "RECONSTRUCTABLE_WITH_LINEAGE_GAPS" if any(x["status"]=="AVAILABLE_SAVED_NATIVE_SCORE_REPLAY_WITH_NODE_LINEAGE_GAP" for x in base["scenarios"].values()) else "UNAVAILABLE_AUTHENTIC_INTERFACE"
    base["paper_comparison_eligible"]=base["status"]=="AVAILABLE_AUTHENTIC_NODE_TO_SCORE_REPLAY"
    for name in ("DS1","DS2"):base["scenarios"][name]={"status":"UNAVAILABLE_AUTHENTIC_INTERFACE","reason":"explicit diagnostic scenario without canonical native source","event_rows":[]}
    return base

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--b0-root",type=Path,required=True);ap.add_argument("--checkpoint",type=Path,required=True);ap.add_argument("--node",action="append",default=[],help="NAME=CSV"); ap.add_argument("--report",type=Path,required=True)
    args=ap.parse_args(); root=args.b0_root
    names={"cleanStatic":"cleanStatic","cleanDynamic":"cleanDynamic","DS3":"ds3","DS7":"ds7","DS8":"ds8"}
    paths={key:root/value/f"texbat_{value}_prn_local_scores.csv" for key,value in names.items()}
    expected=json.loads((ROOT/"configs/r2c_gnss_stage0_fix.json").read_text())["b0"]["saved_score_sha256"]
    node_paths={k:Path(v) for k,v in (x.split("=",1) for x in args.node)}
    checkpoint_hash=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest();b0=b0_replay(paths,node_paths,args.checkpoint,expected,json.loads((ROOT/"configs/r2c_gnss_stage0_fix.json").read_text())["b0"]["checkpoint_sha256"])
    report={"schema":"gnss-doppler-lab.r2c-b0-validation.v2","attack_scores_computed":False,"aggregate_status":b0["status"],"paper_comparison_eligible":b0["paper_comparison_eligible"],
      "checkpoint":{"path":str(args.checkpoint.resolve()),"sha256":checkpoint_hash},"scenarios":b0["scenarios"],"synthetic_and_clean_only_smoke":synthetic_smoke()}
    target=safe_report(args.report); target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,default=lambda x:x.tolist() if hasattr(x,"tolist") else x)+"\n")
    print(json.dumps({"report":str(target),"b0_status":report["aggregate_status"],"attack_campaign_run":False}))
if __name__=="__main__": main()
