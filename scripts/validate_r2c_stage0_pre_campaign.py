#!/usr/bin/env python3
"""Read-only pre-campaign replay and synthetic/clean-only smoke.

This command deliberately has no attack-label or metric code and refuses every
path inside either campaign artifact directory.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.r2c_stage0_fix import (ComplexWhitener, TemplateProvider,
 historical_b0_status, joint_profile_glrt, run_full_controls)

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
    h0=joint_profile_glrt(observations,los_map,provider,taps,np.arange(-.5,.5001,.125),hypothesis="H0")
    shared=joint_profile_glrt(observations,los_map,provider,taps,np.arange(-.5,.5001,.125),hypothesis="H1-shared",beta_candidates_m=[(0,0,0,0),tuple(beta)])
    flat=np.concatenate(list(observations.values()))
    controls=run_full_controls(lambda y,l:float(np.linalg.norm(whitener.transform(np.asarray(y).reshape(-1,9)))),flat,los_map)
    return {"status":"PASS","template_mode":"analytic_gps_ca_acf","analytic_approximation":True,
      "paper_comparison_ready":False,"h0":h0.__dict__,"shared":shared.__dict__,"controls":controls,
      "device":"cpu","gpu_required_for_campaign_training":True}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--b0-root",type=Path,required=True); ap.add_argument("--report",type=Path,required=True)
    args=ap.parse_args(); root=args.b0_root
    names={"cleanStatic":"cleanStatic","cleanDynamic":"cleanDynamic","DS3":"ds3","DS7":"ds7","DS8":"ds8"}
    paths={key:root/value/f"texbat_{value}_prn_local_scores.csv" for key,value in names.items()}
    expected={"cleanStatic":"9a6bc537bd8f1bc16a17257a5f7ae2e47f327c10e215c63d7ebd82ca0b80c36a",
              "cleanDynamic":"855c5ad2b2ea355136f027c49cc22e7234fab2147a6b812f832213b0c7ab082c"}
    report={"schema":"gnss-doppler-lab.r2c-stage0-pre-campaign.v1","attack_campaign_run":False,
            "b0":historical_b0_status(paths,expected),"synthetic_and_clean_only_smoke":synthetic_smoke()}
    target=safe_report(args.report); target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,default=lambda x:x.tolist() if hasattr(x,"tolist") else x)+"\n")
    print(json.dumps({"report":str(target),"b0_status":report["b0"]["status"],"attack_campaign_run":False}))
if __name__=="__main__": main()
