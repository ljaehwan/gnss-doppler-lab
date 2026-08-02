#!/usr/bin/env python3
"""Evaluate a frozen CMTE-A2 state in strictly separated development/confirmatory tiers."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte_a2 import (
    B0Predictor, PREREG_COMMIT, aggregate_epochs, baseline_epoch_inputs,
    b0_enhanced_scores, b0_exact_scores, bootstrap_metrics, epoch_metrics,
    file_sha256, load_fit_state, predict_residuals, score_residuals,
    verify_confirmatory_freeze, write_checksums,
)
from gnss_doppler_lab.cmte_a2_inputs import parse_scenario_mappings, validate_input_manifest


def _specs(items,tier):
    pairs=[]; paths=[]
    for item in items:
        parts=item.split("=",2)
        if len(parts)!=3: raise ValueError("scenario must be NAME=/node.csv=/manifest.json")
        name,node,manifest=parts; pairs.append((name.upper(),Path(node),Path(manifest))); paths.append(f"{name}={node}")
    parsed=parse_scenario_mappings(paths,tier=tier)
    return [(name,parsed[name],manifest.resolve(strict=True)) for name,_,manifest in pairs]


def _onset(name,manifest):
    if name in {"DS7","DS8"}: return 110.
    candidates=(manifest.get("onset_s"),manifest.get("source_metadata",{}).get("onset_s"),
                manifest.get("metadata",{}).get("onset_s"))
    for value in candidates:
        if value is not None:return float(value)
    raise ValueError(f"{name} onset must come from existing authoritative metadata")


def _add_window_start(epoch,node_epoch):
    timing=node_epoch.groupby(["physical_recording_id","window_end_s"],sort=True).window_start_s.min().reset_index()
    return epoch.merge(timing,on=["physical_recording_id","window_end_s"],validate="one_to_one")


def _clean_fpr(state_dir,thresholds):
    per_prn=pd.read_csv(state_dir/"clean_per_prn.csv"); clean=per_prn[per_prn.role=="clean_test"]
    a2=aggregate_epochs(clean); base=pd.read_csv(state_dir/"baseline_threshold_epoch.csv")
    lookup={"CMTE-A2":float((a2.score_A2>thresholds["CMTE-A2"]["q995"]).mean())}
    # Independent clean baseline scores are recomputed from frozen per-PRN residual rows.
    inp=baseline_epoch_inputs(clean); exact=b0_exact_scores(inp,thresholds["node_thresholds"])
    enhanced=b0_enhanced_scores(inp,thresholds["node_thresholds"],thresholds["enhanced_empirical_rates"])
    lookup.update({"A0":float((inp.A0>thresholds["baselines"]["A0"]["q995"]).mean()),
                   "B0-Exact":float((exact.score>thresholds["baselines"]["B0-Exact"]["q995"]).mean()),
                   "B0-Enhanced":float((enhanced.score>thresholds["baselines"]["B0-Enhanced"]["q995"]).mean())})
    return lookup


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier",choices=("development","confirmatory"),required=True)
    parser.add_argument("--state-dir",required=True); parser.add_argument("--scenario",action="append",required=True)
    parser.add_argument("--out",required=True); parser.add_argument("--freeze-manifest"); parser.add_argument("--device",default="cpu")
    args=parser.parse_args(argv)
    state_dir=Path(args.state_dir).resolve(strict=True); specs=_specs(args.scenario,args.tier)
    if args.tier=="confirmatory":
        if not args.freeze_manifest: raise ValueError("confirmatory tier requires --freeze-manifest")
        supplied=Path(args.freeze_manifest).resolve(strict=True)
        if supplied!=(state_dir/"freeze_manifest.json").resolve(): raise ValueError("confirmatory freeze manifest must be the state directory manifest")
        freeze_doc=verify_confirmatory_freeze(supplied,repo=ROOT)
        required_state={"STATE/b0_model.pt","STATE/a2_state.json","STATE/thresholds.json","STATE/config.json"}
        if not required_state.issubset(freeze_doc.get("files",{})): raise ValueError("freeze manifest omits required model/state/threshold files")
    checkpoint_path=state_dir/"b0_model.pt"; state_path=state_dir/"a2_state.json"; thresholds_path=state_dir/"thresholds.json"
    for path in (checkpoint_path,state_path,thresholds_path,state_dir/"freeze_manifest.json"):
        if not path.is_file(): raise ValueError(f"missing frozen state artifact {path}")
    checkpoint=torch.load(checkpoint_path,map_location=args.device,weights_only=False); model=B0Predictor(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"]); model.to(args.device).eval()
    mean=np.asarray(checkpoint["scaler_mean"]); std=np.asarray(checkpoint["scaler_std"])
    state=load_fit_state(state_path); thresholds=json.loads(thresholds_path.read_text()); clean_fpr=_clean_fpr(state_dir,thresholds)
    out=Path(args.out).resolve()
    if out.exists(): raise FileExistsError("atomic non-overwrite output required")
    staging=out.with_name(out.name+f".tmp-{os.getpid()}"); staging.mkdir(parents=True)
    for directory in ("per_epoch","per_prn","plots","provenance"): (staging/directory).mkdir()
    try:
        all_metrics=[]; all_boot=[]; baseline_metrics=[]; input_provenance={}
        for name,node,manifest_path in specs:
            validated=validate_input_manifest(manifest_path,node,name,confirmatory=args.tier=="confirmatory")
            manifest=validated["manifest"]; onset=_onset(name,manifest)
            residual=predict_residuals(validated["frame"],model,mean,std,role=f"{args.tier}:{name}",device=args.device)
            scored=score_residuals(residual,state); scored.to_csv(staging/"per_prn"/f"{name}.csv",index=False)
            epoch=_add_window_start(aggregate_epochs(scored),scored); inp=baseline_epoch_inputs(scored)
            exact=b0_exact_scores(inp,thresholds["node_thresholds"])
            enhanced=b0_enhanced_scores(inp,thresholds["node_thresholds"],thresholds["enhanced_empirical_rates"])
            epoch["score_A0"]=inp.A0.to_numpy(); epoch["score_B0_Exact"]=exact.score.to_numpy(); epoch["score_B0_Enhanced"]=enhanced.score.to_numpy()
            methods={"CMTE-A2":("score_A2",thresholds["CMTE-A2"]["q995"]),
                     "A0":("score_A0",thresholds["baselines"]["A0"]["q995"]),
                     "B0-Exact":("score_B0_Exact",thresholds["baselines"]["B0-Exact"]["q995"]),
                     "B0-Enhanced":("score_B0_Enhanced",thresholds["baselines"]["B0-Enhanced"]["q995"])}
            masks=None
            for method,(column,threshold) in methods.items():
                metric=epoch_metrics(epoch,column,threshold,onset_s=onset,clean_fpr=clean_fpr[method])
                row={"tier":args.tier,"scenario":name,"model":method,"operating_point":"q995_higher","onset_s":onset,**metric}
                (all_metrics if method=="CMTE-A2" else baseline_metrics).append(row)
            from gnss_doppler_lab.cmte_a2 import phase_masks
            masks=phase_masks(epoch,onset); epoch["phase"]="excluded"
            for phase in ("stable_pre","post","persistent"):
                epoch.loc[masks[phase],"phase"]=phase
            boot_input=epoch.rename(columns={"score_A2":"score"}).copy(); boot_input["alarm"]=boot_input.score>thresholds["CMTE-A2"]["q995"]
            ci=bootstrap_metrics(boot_input,reps=2000,seed=20260802)
            for metric,values in ci.items(): all_boot.append({"tier":args.tier,"scenario":name,"model":"CMTE-A2","metric":metric,**values})
            epoch.to_csv(staging/"per_epoch"/f"{name}.csv",index=False)
            fig,ax=plt.subplots(figsize=(10,4)); ax.plot(epoch.window_end_s,epoch.score_A2,label="CMTE-A2")
            ax.axhline(thresholds["CMTE-A2"]["q995"],color="r",ls="--",label="q99.5 higher")
            ax.axvline(onset,color="k",ls=":",label="metadata onset"); ax.set(xlabel="availability window_end_s",ylabel="mean -log(p)",title=name)
            ax.legend(); fig.tight_layout(); fig.savefig(staging/"plots"/f"{name}_a2.png",dpi=120); plt.close(fig)
            input_provenance[name]={"node":str(node),"node_sha256":file_sha256(node),"manifest":str(manifest_path),
                                    "manifest_sha256":file_sha256(manifest_path),"onset_s":onset,"producer_grade":manifest.get("producer_grade")}
        pd.DataFrame(all_metrics).to_csv(staging/("development_metrics.csv" if args.tier=="development" else "confirmatory_metrics.csv"),index=False)
        pd.DataFrame(baseline_metrics).to_csv(staging/"baseline_metrics.csv",index=False)
        pd.DataFrame(all_boot).to_csv(staging/"bootstrap.csv",index=False)
        other="confirmatory_metrics.csv" if args.tier=="development" else "development_metrics.csv"
        pd.DataFrame(columns=["tier","scenario","status"]).to_csv(staging/other,index=False)
        shutil.copyfile(ROOT/"configs/cmte_a2_preregistration.json",staging/"preregistration.json")
        shutil.copyfile(state_dir/"config.json",staging/"config.json"); shutil.copyfile(state_dir/"training.json",staging/"training.json")
        shutil.copyfile(state_dir/"calibration.json",staging/"calibration.json"); shutil.copyfile(thresholds_path,staging/"thresholds.json")
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        provenance={"schema":"gnss-doppler-lab.cmte-a2-evaluation-provenance.v1","tier":args.tier,"execution_source_commit":commit,
                    "prereg_commit":PREREG_COMMIT,"prereg_ancestor_verified":args.tier=="confirmatory","inputs":input_provenance,
                    "state":{"directory":str(state_dir),"checkpoint_sha256":file_sha256(checkpoint_path),"state_sha256":file_sha256(state_path)},
                    "threshold_refit":False,"attack_or_test_fit":False,"tiers_mixed":False}
        (staging/"provenance"/"provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
        summary={"tier":args.tier,"scenarios":[x[0] for x in specs],"rows":len(all_metrics),"bootstrap_reps":2000,
                 "DS7_DS8_scored":args.tier=="confirmatory","result_status":"generated"}
        (staging/"test_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
        (staging/"README.md").write_text(f"# CMTE-A2 {args.tier} evaluation\n\nFrozen q99.5-higher thresholds, strict exceedance alarms, and gap-safe 20-epoch moving-block bootstrap. Tiers are not mixed.\n")
        (staging/"plots"/"README.md").write_text("Plots are rendered directly from immutable per_epoch values and frozen q99.5 thresholds.\n")
        write_checksums(staging); os.replace(staging,out)
        print(json.dumps({"out":str(out),"tier":args.tier,"scenarios":[x[0] for x in specs],"source_commit":commit},sort_keys=True))
    except Exception:
        shutil.rmtree(staging,ignore_errors=True); raise
if __name__=="__main__": main()
