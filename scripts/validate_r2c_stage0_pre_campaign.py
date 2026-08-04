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
    rng=np.random.default_rng(20260803); taps=np.arange(-.5,.5001,.125); provider=TemplateProvider.analytic()
    residual=(rng.normal(size=(64,9))+1j*rng.normal(size=(64,9)))*.01
    whitener=ComplexWhitener().fit(residual,["normal_train"]*64,[f"clean:{i}" for i in range(64)])
    los=np.array([[1,0,0],[0,1,0],[0,0,1],[-.6,-.5,-.6245],[.5,-.7,.5099]],float); los/=np.linalg.norm(los,axis=1)[:,None]
    beta=np.array([20,-15,7,80.]); observations={}; los_map={}
    for i,u in enumerate(los,1):
        delay=(-u@beta[:3]+beta[3])/299792458*1023000
        observations[i]=np.array([provider.evaluate(taps)+.3j*provider.evaluate(taps-delay)+residual[i],
                                  provider.evaluate(taps)-.2*provider.evaluate(taps-delay)+residual[i+8]])
        los_map[i]=u
    h0=joint_profile_glrt(observations,los_map,provider,taps,np.arange(-.5,.5001,.125),hypothesis="H0",whitener=whitener)
    shared=joint_profile_glrt(observations,los_map,provider,taps,np.arange(-.5,.5001,.125),hypothesis="H1-shared",whitener=whitener)
    controls=run_full_controls(lambda obs,l:float(sum(np.linalg.norm(whitener.transform(y)) for y in obs.values())+sum(np.asarray(v)[0] for v in l.values())),observations,los_map,10.,provider,taps)
    return {"status":"PASS","template_mode":"analytic_gps_ca_acf","analytic_approximation":True,
      "paper_comparison_ready":False,"h0":h0.__dict__,"shared":shared.__dict__,"controls":controls,
      "device":"cuda_available_checked_separately","gpu_required_for_campaign_training":True}

def b0_replay(paths,node_paths,checkpoint_path,expected,expected_checkpoint):
    import pandas as pd,torch
    base=historical_b0_status(paths,expected);checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    for scenario,path in paths.items():
        item=base["scenarios"][scenario]
        node_path=node_paths.get(scenario)
        if node_path is None or not node_path.is_file():
            if item["status"]!="UNAVAILABLE_AUTHENTIC_INTERFACE":item["status"]="AVAILABLE_SAVED_NATIVE_SCORE_REPLAY_WITH_NODE_LINEAGE_GAP"
            continue
        try:
            nodes=pd.read_csv(node_path);generated=score_b0_nodes(nodes,checkpoint,device="cpu",checkpoint_path=checkpoint_path,expected_checkpoint_sha256=expected_checkpoint);saved=pd.read_csv(path)
            keys=["run_id","prn","window_bin_s"]+[x for x in ("channel","segment_index") if x in generated.columns and x in saved.columns]
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
            item.update({"status":"AVAILABLE_AUTHENTIC_NODE_TO_SCORE_REPLAY","node_csv":str(node_path),"node_csv_sha256":hashlib.sha256(node_path.read_bytes()).hexdigest(),"matched_scores":len(joined),"maximum_abs_score_error":float(np.max(np.abs(joined.prn_node_rmse_generated-joined.prn_node_rmse_saved)))})
        except Exception as exc:
            fallback=item.get("status")=="AVAILABLE_SAVED_NATIVE_SCORE_REPLAY_WITH_NODE_LINEAGE_GAP"
            item.update({"status":"AVAILABLE_SAVED_NATIVE_SCORE_REPLAY_WITH_NODE_LINEAGE_GAP" if fallback else "UNAVAILABLE_AUTHENTIC_INTERFACE","node_replay_reason":str(exc)})
    base["status"]="AVAILABLE_AUTHENTIC_NODE_TO_SCORE_REPLAY" if all(x["status"]=="AVAILABLE_AUTHENTIC_NODE_TO_SCORE_REPLAY" for x in base["scenarios"].values()) else "RECONSTRUCTABLE_WITH_LINEAGE_GAPS" if any(x["status"]=="AVAILABLE_SAVED_NATIVE_SCORE_REPLAY_WITH_NODE_LINEAGE_GAP" for x in base["scenarios"].values()) else "UNAVAILABLE_AUTHENTIC_INTERFACE"
    base["paper_comparison_eligible"]=base["status"]=="AVAILABLE_AUTHENTIC_NODE_TO_SCORE_REPLAY"
    return base

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--b0-root",type=Path,required=True);ap.add_argument("--checkpoint",type=Path,required=True);ap.add_argument("--node",action="append",default=[],help="NAME=CSV"); ap.add_argument("--report",type=Path,required=True)
    args=ap.parse_args(); root=args.b0_root
    names={"cleanStatic":"cleanStatic","cleanDynamic":"cleanDynamic","DS3":"ds3","DS7":"ds7","DS8":"ds8"}
    paths={key:root/value/f"texbat_{value}_prn_local_scores.csv" for key,value in names.items()}
    expected=json.loads((ROOT/"configs/r2c_gnss_stage0_fix.json").read_text())["b0"]["saved_score_sha256"]
    node_paths={k:Path(v) for k,v in (x.split("=",1) for x in args.node)}
    report={"schema":"gnss-doppler-lab.r2c-stage0-pre-campaign.v1","attack_campaign_run":False,
            "b0":b0_replay(paths,node_paths,args.checkpoint,expected,json.loads((ROOT/"configs/r2c_gnss_stage0_fix.json").read_text())["b0"]["checkpoint_sha256"]),"synthetic_and_clean_only_smoke":synthetic_smoke()}
    target=safe_report(args.report); target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,default=lambda x:x.tolist() if hasattr(x,"tolist") else x)+"\n")
    print(json.dumps({"report":str(target),"b0_status":report["b0"]["status"],"attack_campaign_run":False}))
if __name__=="__main__": main()
