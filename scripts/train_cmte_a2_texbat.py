#!/usr/bin/env python3
"""Train/freeze the exact chronological CMTE-A2 B0 and normal-only state."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte_a2_campaign import validate_source_tree
from gnss_doppler_lab.cmte_a2 import (
    B0Config, FEATURE_COLUMNS, PREREG_COMMIT, aggregate_epochs, baseline_epoch_inputs,
    audit_normal_roles, b0_enhanced_scores, b0_exact_scores, calibrate_comparators,
    create_freeze_manifest, file_sha256, fit_distribution, matched_fpr_diagnostic,
    partition_normal_roles, predict_residuals, role_frame_audit, save_fit_state,
    score_residuals, select_prn_holdout, threshold_operating_points, train_b0, write_checksums,
)


def _copy_prereg(out):
    shutil.copyfile(ROOT/"configs/cmte_a2_preregistration.json",out/"preregistration.json")
    return {"source":"configs/cmte_a2_preregistration.json","sha256":file_sha256(out/"preregistration.json"),"commit":PREREG_COMMIT}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-node-csv",required=True); parser.add_argument("--clean-manifest",required=True)
    parser.add_argument("--out",required=True); parser.add_argument("--device",default="cpu")
    args=parser.parse_args(argv)
    source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    validate_source_tree(ROOT,source_commit,require_clean=True)
    source=Path(args.clean_node_csv).resolve(strict=True); manifest=Path(args.clean_manifest).resolve(strict=True)
    manifest_doc=json.loads(manifest.read_text())
    if manifest_doc.get("scenario")!="cleanStatic" or manifest_doc.get("role")!="normal_clean": raise ValueError("exact cleanStatic normal_clean source required")
    if manifest_doc.get("node_sha256")!=file_sha256(source): raise ValueError("clean node hash mismatch")
    nodes=pd.read_csv(source); roles=partition_normal_roles(nodes)
    if any(roles[name].empty for name in roles): raise ValueError("prefix, qcal, threshold and independent clean roles must all be nonempty")
    out=Path(args.out).resolve()
    if out.exists(): raise FileExistsError("atomic non-overwrite output required")
    staging=out.with_name(out.name+f".tmp-{os.getpid()}"); staging.mkdir(parents=True)
    try:
        model,audit,_,_=train_b0(roles["prefix"],config=B0Config(),device=args.device)
        mean=np.asarray(audit["scaler_mean"]); std=np.asarray(audit["scaler_std"])
        gradient,_=select_prn_holdout(roles["prefix"])
        residuals={"fit":predict_residuals(gradient,model,mean,std,role="fit",device=args.device)}
        for name in ("qcal","threshold","clean_test"):
            residuals[name]=predict_residuals(roles[name],model,mean,std,role=name,device=args.device)
        state=fit_distribution(residuals["fit"])
        state.qcal=np.sort(np.asarray([] if residuals["qcal"].empty else
            np.maximum(0,np.einsum("ni,ij,nj->n",residuals["qcal"].loc[:,[f"residual_{i:03d}" for i in range(9)]].to_numpy()-state.mean,
                                   np.linalg.inv(state.covariance),residuals["qcal"].loc[:,[f"residual_{i:03d}" for i in range(9)]].to_numpy()-state.mean))))
        state.metadata.update({"prereg_commit":PREREG_COMMIT,"fit_fingerprint":audit["fit_fingerprint"],"qcal_n":len(state.qcal),
                               "all_250_plus_forbidden_from_model_scaler_selection":True})
        scored={name:score_residuals(frame,state) for name,frame in residuals.items() if name!="fit"}
        epochs={name:aggregate_epochs(frame) for name,frame in scored.items()}
        ops=threshold_operating_points(epochs["threshold"].score_A2)
        baseline_input=baseline_epoch_inputs(scored["threshold"])
        comparator_calibration=calibrate_comparators(scored["threshold"])
        node_thresholds=comparator_calibration["node_thresholds"]
        rates=comparator_calibration["enhanced_empirical_rates"]
        exact=b0_exact_scores(baseline_input,node_thresholds); enhanced=b0_enhanced_scores(baseline_input,node_thresholds,rates)
        baseline_ops=comparator_calibration["final_thresholds"]
        threshold_role_scores={"CMTE-A2":epochs["threshold"].score_A2.to_numpy(),"A0":baseline_input.A0.to_numpy(),
                               "B0-Exact":exact.score.to_numpy(),"B0-Enhanced":enhanced.score.to_numpy()}
        matched=matched_fpr_diagnostic(threshold_role_scores,primary_threshold=ops["q995"])
        role_audit=audit_normal_roles(roles)
        role_audit["distribution_fit_gradient"] = role_frame_audit(gradient,"prefix-gradient")
        role_audit["train"] = role_frame_audit(gradient,"train")
        role_audit["qcal"] = role_audit["roles"]["qcal"]
        role_audit["threshold"] = role_audit["roles"]["threshold"]
        role_audit["test"] = role_audit["roles"]["clean_test"]
        role_audit["history_resets"]={name:frame.attrs.get("history_audit",{}) for name,frame in residuals.items()}
        checkpoint={"schema":"gnss-doppler-lab.cmte-a2-b0.v1","model_state":model.cpu().state_dict(),"config":B0Config(),
                    "scaler_mean":mean,"scaler_std":std,"training_audit":audit}
        torch.save(checkpoint,staging/"b0_model.pt")
        save_fit_state(state,staging/"a2_state.json")
        prereg=_copy_prereg(staging)
        config={"schema":"gnss-doppler-lab.cmte-a2-config.v1","tap_order":["E4","E3","E2","E","P","L","L2","L3","L4"],
                "history":12,"roles":{"prefix":[0,240],"qcal":[250,290],"threshold":[300,330],"clean_test_start":340},
                "score":"mean_i[-ln(p_i)]","alarm":"strict score > threshold","primary_operating_point":"q995_higher",
                "forbidden":["mixture e-values","e-CUSUM","restart capital","sequential score","online normalization"]}
        training={"schema":"gnss-doppler-lab.cmte-a2-training.v1","source_node":str(source),"source_node_sha256":file_sha256(source),
                  "source_manifest":str(manifest),"source_manifest_sha256":file_sha256(manifest),"audit":audit,
                  "role_rows":{name:len(frame) for name,frame in roles.items()},"residual_rows":{name:len(frame) for name,frame in residuals.items()},
                  "role_audit":role_audit,"checkpoint_sha256":file_sha256(staging/"b0_model.pt"),
                  "execution":{"source_commit":source_commit,"clean_tree_asserted":True,"prereg_unchanged_asserted":True}}
        calibration={"schema":"gnss-doppler-lab.cmte-a2-calibration.v1","distribution_fit_n":len(residuals["fit"]),
                     "distribution_state_hash":state.state_hash,"shrinkage":state.shrinkage,"epsilon":state.epsilon,"qcal_n":len(state.qcal),
                     "qcal_source":"cleanStatic fully-contained [250,290)","inclusive_ties":True,"plus_one":True}
        thresholds={"schema":"gnss-doppler-lab.cmte-a2-thresholds.v1","source":"cleanStatic fully-contained [300,330)",
                    "CMTE-A2":ops,"primary":"q995","method":"higher","alarm_strict_greater":True,"baselines":baseline_ops,
                    "baseline_labels":{"A0":"A0","B0-Exact":"chronological B0 with exact gate semantics","B0-Enhanced":"B0-Enhanced"},
                    "node_threshold_source":"cleanStatic fully-contained threshold role [300,330) only","node_thresholds":node_thresholds,
                    "enhanced_rate_source":"same threshold role; empirical strict RMSE exceedance","enhanced_empirical_rates":rates,
                    "comparator_calibration":comparator_calibration,"matched_fpr_diagnostic":matched,
                    "short_self_calibration_limitation":comparator_calibration["limitation"]}
        for name,doc in (("config.json",config),("training.json",training),("calibration.json",calibration),("thresholds.json",thresholds)):
            (staging/name).write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
        pd.concat([frame.assign(role=name) for name,frame in scored.items()],ignore_index=True).to_csv(staging/"clean_per_prn.csv",index=False)
        pd.concat([frame.assign(role=name) for name,frame in epochs.items()],ignore_index=True).to_csv(staging/"clean_per_epoch.csv",index=False)
        pd.DataFrame({"physical_recording_id":exact.physical_recording_id,"window_end_s":exact.window_end_s,"A0":baseline_input.A0,
                      "B0_Exact":exact.score,"B0_Enhanced":enhanced.score}).to_csv(staging/"baseline_threshold_epoch.csv",index=False)
        provenance={"schema":"gnss-doppler-lab.cmte-a2-provenance.v1","execution_source_commit":source_commit,"clean_tree_asserted":True,"prereg_unchanged_asserted":True,
                    "preregistration":prereg,"training_tier":"normal_only","DS7_DS8_accessed":False}
        (staging/"provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
        (staging/"test_summary.txt").write_text("Generator code tests must be executed and recorded before a campaign. No DS7/DS8 scoring was run.\n")
        (staging/"README.md").write_text("# CMTE-A2 frozen training state\n\nNormal-only chronological B0, distribution, Qcal, and threshold artifacts. No DS7/DS8 inference or scoring.\n")
        freeze_paths=[staging/p for p in ("b0_model.pt","a2_state.json","config.json","training.json","calibration.json","thresholds.json","preregistration.json")]
        freeze_paths += [ROOT/p for p in ("src/gnss_doppler_lab/cmte_a2.py","src/gnss_doppler_lab/cmte_a2_inputs.py",
                                          "src/gnss_doppler_lab/cmte_a2_campaign.py","scripts/train_cmte_a2_texbat.py",
                                          "scripts/eval_cmte_a2_texbat.py","scripts/prepare_cmte_a2_inputs.py",
                                          "scripts/prepare_cmte_a2_ds8_complex.py","scripts/build_cmte_a2_confirm_input_manifest.py",
                                          "scripts/freeze_cmte_a2_campaign.py","scripts/confirm_cmte_a2_texbat.py",
                                          "scripts/finalize_cmte_a2_campaign.py","configs/cmte_a2_ds8_receiver.conf")]
        create_freeze_manifest(ROOT,staging,freeze_paths,staging/"freeze_manifest.json")
        write_checksums(staging); os.replace(staging,out)
        print(json.dumps({"out":str(out),"checkpoint_sha256":training["checkpoint_sha256"],"qcal_n":len(state.qcal),"threshold":ops["q995"]},sort_keys=True))
    except Exception:
        shutil.rmtree(staging,ignore_errors=True); raise
if __name__=="__main__": main()
