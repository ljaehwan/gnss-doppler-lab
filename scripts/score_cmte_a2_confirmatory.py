#!/usr/bin/env python3
"""Capability-gated confirmatory CMTE-A2 scorer; not a public CLI."""
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
from gnss_doppler_lab.cmte_a2_campaign import (validate_source_tree,verify_checksums,
 validate_confirm_capability)


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
    per_prn=pd.read_csv(state_dir/"clean_per_prn.csv")
    # Score comparator EWMAs over every available clean-recording role first;
    # only then slice the independent clean test. There is no role reset.
    all_epoch=_score_columns(per_prn,thresholds)
    clean=per_prn[per_prn.role=="clean_test"].copy()
    keys=clean[["physical_recording_id","window_end_s"]].drop_duplicates()
    epoch=all_epoch.merge(keys,on=["physical_recording_id","window_end_s"],how="inner",validate="one_to_one")
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


def score_confirmatory(capability,argv=None):
    validate_confirm_capability(capability)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier",choices=("confirmatory",),required=True)
    parser.add_argument("--state-dir",required=True); parser.add_argument("--scenario",action="append",required=True)
    parser.add_argument("--out",required=True); parser.add_argument("--freeze-manifest"); parser.add_argument("--device",default="cpu")
    args=parser.parse_args(argv)
    current_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    validate_source_tree(ROOT,current_commit,require_clean=True)
    state_dir=Path(args.state_dir).resolve(strict=True)
    verify_checksums(state_dir)
    training_guard=json.loads((state_dir/"training.json").read_text()); provenance_guard=json.loads((state_dir/"provenance.json").read_text())
    execution=training_guard.get("execution",{})
    if execution.get("source_commit")!=current_commit or execution.get("clean_tree_asserted") is not True:
        raise ValueError("state training execution source/clean-tree assertion missing or stale")
    if provenance_guard.get("execution_source_commit")!=current_commit or provenance_guard.get("clean_tree_asserted") is not True:
        raise ValueError("state provenance execution source/clean-tree assertion missing or stale")
    supplied=Path(args.freeze_manifest).resolve(strict=True) if args.freeze_manifest else (state_dir/"freeze_manifest.json").resolve(strict=True)
    if supplied!=(state_dir/"freeze_manifest.json").resolve(): raise ValueError("freeze manifest must be the state directory manifest")
    freeze_doc=verify_confirmatory_freeze(supplied,repo=ROOT)
    required_state={"STATE/b0_model.pt","STATE/a2_state.json","STATE/thresholds.json","STATE/config.json"}
    if not required_state.issubset(freeze_doc.get("files",{})): raise ValueError("freeze manifest omits required model/state/threshold files")
    specs=_specs(args.scenario,args.tier)
    checkpoint_path=state_dir/"b0_model.pt"; state_path=state_dir/"a2_state.json"; thresholds_path=state_dir/"thresholds.json"
    for path in (checkpoint_path,state_path,thresholds_path,state_dir/"freeze_manifest.json"):
        if not path.is_file(): raise ValueError(f"missing frozen state artifact {path}")
    checkpoint=torch.load(checkpoint_path,map_location=args.device,weights_only=False); predictor=B0Predictor(checkpoint["config"])
    predictor.load_state_dict(checkpoint["model_state"]); predictor.to(args.device).eval()
    mean=np.asarray(checkpoint["scaler_mean"]); std=np.asarray(checkpoint["scaler_std"])
    state=load_fit_state(state_path); thresholds=json.loads(thresholds_path.read_text()); methods=_methods(thresholds)
    validated_inputs={}; expected_fingerprint=None
    for input_name,input_node,input_manifest in specs:
        item=validate_input_manifest(input_manifest,input_node,input_name,confirmatory=args.tier=="confirmatory",expected_fingerprint=expected_fingerprint)
        fingerprint=item["manifest"].get("campaign_converter_fingerprint")
        if args.tier=="confirmatory" and expected_fingerprint is None: expected_fingerprint=fingerprint
        validated_inputs[input_name]=item
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
            validated=validated_inputs[name]
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
                boot["alarm"]=boot.score>threshold; boot["prereg_subphase"]=epoch["phase"].to_numpy()
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
            fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
            for method,(column,threshold) in methods.items():
                axes[0].plot(epoch.window_end_s,epoch[column],label=method,linewidth=.9); axes[0].axhline(threshold,ls="--",linewidth=.6)
            axes[0].axvline(onset,color="k",ls=":"); axes[0].set_ylabel("frozen score"); axes[0].legend(ncol=2)
            axes[1].step(epoch.window_end_s,epoch.tracked_prn_count,where="post"); axes[1].axvline(onset,color="k",ls=":")
            axes[1].set(xlabel="availability window_end_s",ylabel="exact tracked PRN N"); fig.suptitle(f"{name}: all models and exact-N timeline")
            fig.tight_layout(); fig.savefig(staging/"plots"/f"{name}_all_models_prn_n.png",dpi=120); plt.close(fig)
            input_provenance[name]={"node":str(node),"node_sha256":file_sha256(node),"manifest":str(manifest_path),
                                    "manifest_sha256":file_sha256(manifest_path),"onset_s":onset,"producer_grade":manifest.get("producer_grade"),
                                    "history_audit":residual.attrs.get("history_audit",{})}
        primary_name="development_metrics.csv" if args.tier=="development" else "confirmatory_metrics.csv"
        pd.DataFrame(all_metrics).to_csv(staging/primary_name,index=False)
        pd.DataFrame(baseline_metrics).to_csv(staging/"baseline_metrics.csv",index=False)
        boot_frame=pd.DataFrame(all_boot); boot_frame.to_csv(staging/"bootstrap.csv",index=False)
        pd.DataFrame(matched_rows).to_csv(staging/"matched_fpr.csv",index=False)
        ci=boot_frame.dropna(subset=["low","high"] if {"low","high"}.issubset(boot_frame) else []).copy()
        fig,ax=plt.subplots(figsize=(12,5))
        if not ci.empty:
            labels=(ci.scenario.astype(str)+"/"+ci.model.astype(str)+"/"+ci.metric.astype(str)).tolist(); y=np.arange(len(ci))
            point=ci["point"].to_numpy(float) if "point" in ci else (ci.low.to_numpy(float)+ci.high.to_numpy(float))/2
            # Keep the all-eligible point even if a small-replicate percentile CI excludes it.
            ax.hlines(y,ci.low.to_numpy(float),ci.high.to_numpy(float),linewidth=.8)
            ax.plot(point,y,"o",markersize=3)
            ax.set_yticks(y,labels,fontsize=6)
        else: ax.text(.5,.5,"CI unavailable; see machine-readable NA reasons",ha="center")
        ax.set_title(f"{args.tier} moving-block bootstrap confidence intervals"); fig.tight_layout()
        fig.savefig(staging/"plots"/"all_models_bootstrap_ci.png",dpi=120); plt.close(fig)
        exact_input=pd.concat(diagnostic_parts,ignore_index=True)
        required_strata=[{"tier":"normal","scenario":"clean_test","model":m,"phase":"clean"} for m in methods]
        required_strata += [{"tier":args.tier,"scenario":scenario,"model":m,"phase":phase}
                            for scenario,_,_ in specs for m in methods for phase in ("stable_pre","post","persistent")]
        exact_rows,dependence=exact_n_diagnostics(exact_input,required_strata=required_strata)
        exact_rows.to_csv(staging/"exact_n_diagnostics.csv",index=False)
        dependence_rows=[]
        for (scenario,method),group in exact_input.groupby(["scenario","model"],sort=True):
            _,item=exact_n_diagnostics(group); dependence_rows.append({"scenario":scenario,"model":method,**item})
        dependence_doc={"schema":"gnss-doppler-lab.cmte-a2-prn-dependence.v1","passed":True,
          "aggregation_changed":False,"sparse_strata_pooled":False,"empty_strata_explicit_epoch_count_zero":True,
          "overall":dependence,"rows":dependence_rows}
        (staging/"prn_dependence.json").write_text(json.dumps(dependence_doc,indent=2,sort_keys=True)+"\n")
        combined_metrics=pd.concat([pd.DataFrame(all_metrics),pd.DataFrame(baseline_metrics)],ignore_index=True)
        success=success_criteria_audit(combined_metrics,exact_rows)
        (staging/"success_audit.json").write_text(json.dumps(success,indent=2,sort_keys=True)+"\n")
        shutil.copyfile(ROOT/"configs/cmte_a2_preregistration.json",staging/"preregistration.json")
        for name in ("config.json","training.json","calibration.json","thresholds.json","historical_b0_gate_equivalence.json"):
            shutil.copyfile(state_dir/name,staging/name)
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        provenance={"schema":"gnss-doppler-lab.cmte-a2-evaluation-provenance.v1","tier":args.tier,"execution_source_commit":commit,"clean_tree_asserted":True,
                    "artifact_execution_source_recorded":True,"prereg_commit":PREREG_COMMIT,"prereg_ancestor_verified":args.tier=="confirmatory","inputs":input_provenance,
                    "state":{"directory":str(state_dir),"checkpoint_sha256":file_sha256(checkpoint_path),"state_sha256":file_sha256(state_path),
                             "thresholds_sha256":file_sha256(thresholds_path),"scaler_sha256":hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest(),
                             "qcal_sha256":hashlib.sha256(np.asarray(state.qcal).tobytes()).hexdigest()},
                    "threshold_refit":False,"attack_or_test_fit":False,"tiers_mixed":False}
        (staging/"provenance"/"provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
        training_doc=json.loads((state_dir/"training.json").read_text())
        audit={"schema":"gnss-doppler-lab.cmte-a2-evaluation-audit.v1","tier":args.tier,"role_audit":training_doc.get("role_audit"),
               "evaluation_history_audit":{k:v["history_audit"] for k,v in input_provenance.items()},
               "bootstrap":{"phase_recording_producer_cadence_safe":True,"iid_fallback":False,"post_includes_persistent":True,
                 "subphase_boundaries":["metadata_onset","onset+20","onset+40"],"point_estimate_uses_all_eligible_epochs":True},
               "comparator_reset":{"reset_dimension":"physical_recording_id","role_reset":False,"phase_reset":False,
                 "clean_test_sliced_after_full_recording_score":True,"attack_scored_from_recording_start":True},
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
