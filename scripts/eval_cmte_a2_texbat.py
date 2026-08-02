#!/usr/bin/env python3
"""Evaluate a frozen CMTE-A2 state in strictly separated development/confirmatory tiers."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte_a2 import (
    B0Predictor, PREREG_COMMIT, aggregate_epochs, baseline_epoch_inputs,
    b0_enhanced_scores, b0_exact_scores, bootstrap_metrics, epoch_metrics,
    exact_n_diagnostics, file_sha256, load_fit_state, predict_residuals,
    score_residuals, success_criteria_audit, verify_confirmatory_freeze, write_checksums,
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
    candidates=(manifest.get("onset_s"),manifest.get("source_metadata",{}).get("onset_s"),manifest.get("metadata",{}).get("onset_s"))
    for value in candidates:
        if value is not None:return float(value)
    raise ValueError(f"{name} onset must come from existing authoritative metadata")


def _add_epoch_audit_columns(epoch,node_epoch):
    rows=[]
    for (recording,t),g in node_epoch.groupby(["physical_recording_id","window_end_s"],sort=True):
        identities=sorted(set("|".join(map(str,(x.history_id,x.segment,x.channel))) for x in g.itertuples()))
        rows.append({"physical_recording_id":str(recording),"window_end_s":float(t),"window_start_s":float(g.window_start_s.min()),
                     "producer_chain_id":"producer-"+hashlib.sha256(";".join(identities).encode()).hexdigest()[:20]})
    return epoch.merge(pd.DataFrame(rows),on=["physical_recording_id","window_end_s"],validate="one_to_one")


def _score_columns(scored,thresholds):
    epoch=_add_epoch_audit_columns(aggregate_epochs(scored),scored); inp=baseline_epoch_inputs(scored)
    exact=b0_exact_scores(inp,thresholds["node_thresholds"])
    enhanced=b0_enhanced_scores(inp,thresholds["node_thresholds"],thresholds["enhanced_empirical_rates"])
    epoch["score_A0"]=inp.A0.to_numpy(); epoch["score_B0_Exact"]=exact.score.to_numpy(); epoch["score_B0_Enhanced"]=enhanced.score.to_numpy()
    return epoch


def _methods(thresholds):
    return {"CMTE-A2":("score_A2",float(thresholds["CMTE-A2"]["q995"])),
            "A0":("score_A0",float(thresholds["baselines"]["A0"]["q995"])),
            "B0-Exact":("score_B0_Exact",float(thresholds["baselines"]["B0-Exact"]["q995"])),
            "B0-Enhanced":("score_B0_Enhanced",float(thresholds["baselines"]["B0-Enhanced"]["q995"]))}


def _clean_scores(state_dir,thresholds):
    per_prn=pd.read_csv(state_dir/"clean_per_prn.csv"); clean=per_prn[per_prn.role=="clean_test"].copy()
    epoch=_score_columns(clean,thresholds)
    return clean,epoch


def _diagnostic_parts(epoch,tier,scenario,methods,masks):
    pieces=[]
    for model,(column,threshold) in methods.items():
        for phase in ("stable_pre","post","persistent"):
            part=epoch.loc[masks[phase],["tracked_prn_count",column]].copy()
            if part.empty:continue
            part=part.rename(columns={column:"score"}); part["alarm"]=part.score>threshold
            part["tier"]=tier; part["scenario"]=scenario; part["model"]=model; part["phase"]=phase; pieces.append(part)
    return pieces


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier",choices=("development","confirmatory"),required=True)
    parser.add_argument("--state-dir",required=True); parser.add_argument("--scenario",action="append",required=True)
    parser.add_argument("--out",required=True); parser.add_argument("--freeze-manifest"); parser.add_argument("--device",default="cpu")
    args=parser.parse_args(argv)
    state_dir=Path(args.state_dir).resolve(strict=True); specs=_specs(args.scenario,args.tier)
    freeze_doc=None
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
    checkpoint=torch.load(checkpoint_path,map_location=args.device,weights_only=False); predictor=B0Predictor(checkpoint["config"])
    predictor.load_state_dict(checkpoint["model_state"]); predictor.to(args.device).eval()
    mean=np.asarray(checkpoint["scaler_mean"]); std=np.asarray(checkpoint["scaler_std"])
    state=load_fit_state(state_path); thresholds=json.loads(thresholds_path.read_text()); methods=_methods(thresholds)
    clean_prn,clean_epoch=_clean_scores(state_dir,thresholds)
    clean_fpr={model:float((clean_epoch[column]>threshold).mean()) for model,(column,threshold) in methods.items()}
    out=Path(args.out).resolve()
    if out.exists(): raise FileExistsError("atomic non-overwrite output required")
    staging=out.with_name(out.name+f".tmp-{os.getpid()}"); staging.mkdir(parents=True)
    for directory in ("per_epoch","per_prn","plots","provenance"): (staging/directory).mkdir()
    try:
        all_metrics=[]; all_boot=[]; baseline_metrics=[]; diagnostic_parts=[]; matched_rows=[]; input_provenance={}
        clean_parts=[]
        for model,(column,threshold) in methods.items():
            x=clean_epoch[["tracked_prn_count",column]].rename(columns={column:"score"}).copy(); x["alarm"]=x.score>threshold
            x["tier"]="normal"; x["scenario"]="clean_test"; x["model"]=model; x["phase"]="clean"; clean_parts.append(x)
        diagnostic_parts.extend(clean_parts)
        matched_doc=thresholds.get("matched_fpr_diagnostic",{})
        matched_clean={name:float((clean_epoch[{"A0":"score_A0","B0-Exact":"score_B0_Exact","B0-Enhanced":"score_B0_Enhanced"}[name]]>item["threshold"]).mean())
                       for name,item in matched_doc.get("models",{}).items()}
        for name,node,manifest_path in specs:
            validated=validate_input_manifest(manifest_path,node,name,confirmatory=args.tier=="confirmatory")
            manifest=validated["manifest"]; onset=_onset(name,manifest)
            residual=predict_residuals(validated["frame"],predictor,mean,std,role=f"{args.tier}:{name}",device=args.device)
            scored=score_residuals(residual,state); scored.to_csv(staging/"per_prn"/f"{name}.csv",index=False)
            epoch=_score_columns(scored,thresholds)
            from gnss_doppler_lab.cmte_a2 import phase_masks
            masks=phase_masks(epoch,onset); epoch["phase"]="excluded"
            epoch.loc[masks["stable_pre"],"phase"]="stable_pre"; epoch.loc[masks["ramp"],"phase"]="ramp"
            epoch.loc[masks["takeover"],"phase"]="takeover"; epoch.loc[masks["persistent"],"phase"]="persistent"
            epoch["scenario"]=name; epoch["stable_pre"]=masks["stable_pre"]; epoch["post"]=masks["post"]; epoch["persistent"]=masks["persistent"]
            diagnostic_parts.extend(_diagnostic_parts(epoch,args.tier,name,methods,masks))
            for method,(column,threshold) in methods.items():
                metric=epoch_metrics(epoch,column,threshold,onset_s=onset,clean_fpr=clean_fpr[method])
                row={"tier":args.tier,"scenario":name,"model":method,"operating_point":"q995_higher","onset_s":onset,**metric}
                (all_metrics if method=="CMTE-A2" else baseline_metrics).append(row)
                boot=epoch[["scenario","physical_recording_id","producer_chain_id","window_end_s","stable_pre","post","persistent",column]].rename(columns={column:"score"}).copy()
                boot["alarm"]=boot.score>threshold
                for statistic,values in bootstrap_metrics(boot,reps=2000,seed=20260802).items():
                    all_boot.append({"tier":args.tier,"scenario":name,"model":method,"metric":statistic,**values})
            for method,item in matched_doc.get("models",{}).items():
                column=methods[method][0]; threshold=float(item["threshold"])
                metric=epoch_metrics(epoch,column,threshold,onset_s=onset,clean_fpr=matched_clean[method])
                matched_rows.append({"tier":args.tier,"scenario":name,"model":method,"operating_point":"matched_clean_fpr_diagnostic",
                                     "threshold_role_fit_only":True,"diagnostic_only":True,**metric})
            epoch.to_csv(staging/"per_epoch"/f"{name}.csv",index=False)
            fig,ax=plt.subplots(figsize=(10,4)); ax.plot(epoch.window_end_s,epoch.score_A2,label="CMTE-A2")
            ax.axhline(methods["CMTE-A2"][1],color="r",ls="--",label="q99.5 higher")
            ax.axvline(onset,color="k",ls=":",label="metadata onset"); ax.set(xlabel="availability window_end_s",ylabel="mean -log(p)",title=name)
            ax.legend(); fig.tight_layout(); fig.savefig(staging/"plots"/f"{name}_a2.png",dpi=120); plt.close(fig)
            input_provenance[name]={"node":str(node),"node_sha256":file_sha256(node),"manifest":str(manifest_path),
                                    "manifest_sha256":file_sha256(manifest_path),"onset_s":onset,"producer_grade":manifest.get("producer_grade"),
                                    "history_audit":residual.attrs.get("history_audit",{})}
        primary_name="development_metrics.csv" if args.tier=="development" else "confirmatory_metrics.csv"
        pd.DataFrame(all_metrics).to_csv(staging/primary_name,index=False)
        pd.DataFrame(baseline_metrics).to_csv(staging/"baseline_metrics.csv",index=False)
        pd.DataFrame(all_boot).to_csv(staging/"bootstrap.csv",index=False)
        pd.DataFrame(matched_rows).to_csv(staging/"matched_fpr.csv",index=False)
        exact_input=pd.concat(diagnostic_parts,ignore_index=True); exact_rows,dependence=exact_n_diagnostics(exact_input)
        exact_rows.to_csv(staging/"exact_n_diagnostics.csv",index=False)
        dependence_rows=[]
        for (scenario,method),group in exact_input.groupby(["scenario","model"],sort=True):
            _,item=exact_n_diagnostics(group); dependence_rows.append({"scenario":scenario,"model":method,**item})
        (staging/"prn_dependence.json").write_text(json.dumps(dependence_rows,indent=2,sort_keys=True)+"\n")
        combined_metrics=pd.concat([pd.DataFrame(all_metrics),pd.DataFrame(baseline_metrics)],ignore_index=True)
        success=success_criteria_audit(combined_metrics,exact_rows)
        (staging/"success_audit.json").write_text(json.dumps(success,indent=2,sort_keys=True)+"\n")
        shutil.copyfile(ROOT/"configs/cmte_a2_preregistration.json",staging/"preregistration.json")
        for name in ("config.json","training.json","calibration.json","thresholds.json"): shutil.copyfile(state_dir/name,staging/name)
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        provenance={"schema":"gnss-doppler-lab.cmte-a2-evaluation-provenance.v1","tier":args.tier,"execution_source_commit":commit,
                    "prereg_commit":PREREG_COMMIT,"prereg_ancestor_verified":args.tier=="confirmatory","inputs":input_provenance,
                    "state":{"directory":str(state_dir),"checkpoint_sha256":file_sha256(checkpoint_path),"state_sha256":file_sha256(state_path),
                             "thresholds_sha256":file_sha256(thresholds_path),"scaler_sha256":hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest(),
                             "qcal_sha256":hashlib.sha256(np.asarray(state.qcal).tobytes()).hexdigest()},
                    "threshold_refit":False,"attack_or_test_fit":False,"tiers_mixed":False}
        (staging/"provenance"/"provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
        training_doc=json.loads((state_dir/"training.json").read_text())
        audit={"schema":"gnss-doppler-lab.cmte-a2-evaluation-audit.v1","tier":args.tier,"role_audit":training_doc.get("role_audit"),
               "evaluation_history_audit":{k:v["history_audit"] for k,v in input_provenance.items()},
               "bootstrap":{"phase_recording_producer_cadence_safe":True,"iid_fallback":False,"post_includes_persistent":True},
               "exact_n":{"aggregation_changed":False,"sparse_pooled":False},"matched_fpr":{"threshold_role_fit_only":True,"diagnostic_only":True},
               "confirmatory_placeholder_created":False,"primary_metric_file":primary_name}
        (staging/"audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
        summary={"tier":args.tier,"scenarios":[x[0] for x in specs],"rows":len(all_metrics),"bootstrap_reps":2000,
                 "bootstrap_models":sorted(methods),"DS7_DS8_scored":args.tier=="confirmatory","result_status":"generated",
                 "confirmatory_placeholder_created":False}
        (staging/"test_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
        (staging/"README.md").write_text(f"# CMTE-A2 {args.tier} evaluation\n\nActual {args.tier} metrics, all-model moving-block CIs, exact-N and matched-FPR diagnostics. No opposite-tier placeholder is created.\n")
        (staging/"plots"/"README.md").write_text("Plots are rendered directly from immutable per_epoch values and frozen q99.5 thresholds.\n")
        write_checksums(staging); os.replace(staging,out)
        print(json.dumps({"out":str(out),"tier":args.tier,"scenarios":[x[0] for x in specs],"source_commit":commit},sort_keys=True))
    except Exception:
        shutil.rmtree(staging,ignore_errors=True); raise
if __name__=="__main__": main()
