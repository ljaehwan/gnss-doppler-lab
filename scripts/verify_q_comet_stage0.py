#!/usr/bin/env python3
"""Strict science/admin verifier for the Q-COMET Stage-0 artifact bundle."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
REQUIRED={"README.md","config.json","pre_evaluation_freeze.json","source_commit.json","data_inventory.json",
          "source_binding.json","timeline_inventory.json","normal_split_audit.json","normal_model_summary.json",
          "nuisance_projection_validation.json","thresholds.json","scenario_metrics.csv","ablation_metrics.csv",
          "per_epoch_scores.csv.gz","common_onset_estimates.csv","participation_posteriors.csv.gz","external_static_fpr.csv",
          "relation_destruction_metrics.json","physical_controls.json","bootstrap_intervals.csv",
          "cross_dataset_confirmation.json","final_verdict.json","artifact_manifest_sha256.json",
          "ds78_byte_identity_audit.json","implementation_repairs.json","runner_runs.json",
          "test_results.json",
          "texbat_prn_bayes_factors.csv.gz","oakbat_prn_bayes_factors.csv.gz",
          "texbat_score_diagnostics.csv.gz","oakbat_score_diagnostics.csv.gz"}
PLOTS={"normal_manifold_quotient_residual.png","nuisance_projection_validation.png","official_estimated_onset.png",
       "prn_bf_heatmap.png","participation_posterior.png","original_vs_desync_score.png","ablation_pauc_delay.png",
       "external_static_fpr.png","oakbat_confirmation.png","score_vs_residual_power.png"}
EXCLUDE={"artifact_manifest_sha256.json","verifier_report.json","fresh_clone_verifier_report.json"}
VERDICTS={"GO_FOR_QCOMET_NEURAL_STAGE1","NO_GO_SHARED_ONSET_HYPOTHESIS","INCONCLUSIVE_DATA_OR_PROVENANCE"}


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def manifest_entries(directory):
    return {str(path.relative_to(directory)):sha(path) for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name not in EXCLUDE and not path.name.endswith(".tmp")}


def write_manifest(directory):
    entries=manifest_entries(directory)
    payload={"schema":"gnss-doppler-lab.q-comet-artifact-manifest.v1","algorithm":"sha256",
             "excluded":sorted(EXCLUDE),"files":entries,
             "manifest_content_sha256":hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(",",":")).encode()).hexdigest()}
    (directory/"artifact_manifest_sha256.json").write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")
    return payload


def _read_csv(path):
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"rt",encoding="utf8",newline="") as handle:return list(csv.DictReader(handle))


def verify(directory):
    science=[];admin=[]
    missing=sorted(name for name in REQUIRED if not (directory/name).is_file())
    missing_plots=sorted(name for name in PLOTS if not (directory/"plots"/name).is_file())
    admin.append({"id":"required_files","status":"PASS" if not missing else "FAIL","missing":missing})
    admin.append({"id":"required_plots","status":"PASS" if not missing_plots else "FAIL","missing":missing_plots})
    if missing:return science,admin
    freeze=json.loads((directory/"pre_evaluation_freeze.json").read_text())
    source=json.loads((directory/"source_commit.json").read_text())
    verdict=json.loads((directory/"final_verdict.json").read_text())
    thresholds=json.loads((directory/"thresholds.json").read_text())
    split=json.loads((directory/"normal_split_audit.json").read_text())
    binding=json.loads((directory/"source_binding.json").read_text())
    cross=json.loads((directory/"cross_dataset_confirmation.json").read_text())
    prefix=json.loads((directory/"ds78_byte_identity_audit.json").read_text())
    checks=[
      ("freeze_type",freeze.get("freeze_type")=="PRE_EVALUATION_CONFIGURATION_FREEZE"),
      ("no_attack_configuration",freeze.get("attack_results_used") is False),
      ("normal_only_threshold",thresholds.get("source")=="cleanStatic calibration-B only"),
      ("split_disjoint",split.get("all_disjoint") is True and split.get("calibration_reuse") is False),
      ("source_binding",str(binding.get("status","")).startswith("PASS")),
      ("frozen_cross_dataset",cross.get("status")=="FROZEN_CROSS_DATASET_CONFIRMATION"),
      ("ds78_prefix_audit",prefix.get("status")=="PASS" and prefix.get("independent_normal_confirmation_counted") is False),
      ("valid_verdict",verdict.get("verdict") in VERDICTS),
      ("no_neural_stage1",verdict.get("neural_stage1_implemented") is False),
    ]
    science.extend({"id":name,"status":"PASS" if ok else "FAIL"} for name,ok in checks)
    freeze_sha=source.get("freeze_commit_sha");result_sha=source.get("result_commit_sha")
    freeze_ok=isinstance(freeze_sha,str) and len(freeze_sha)==40 and subprocess.run(["git","merge-base","--is-ancestor",freeze_sha,"HEAD"],cwd=ROOT).returncode==0
    science.append({"id":"freeze_commit_ancestor","status":"PASS" if freeze_ok else "FAIL","freeze_commit_sha":freeze_sha})
    # Observation/support audit.
    scenario=_read_csv(directory/"scenario_metrics.csv");ablations=_read_csv(directory/"ablation_metrics.csv")
    scenarios={row.get("scenario") for row in scenario};methods={row.get("method") for row in ablations}
    science.append({"id":"required_scenarios","status":"PASS" if {"DS3","DS4","DS7","DS8","OS1","OS2","OS3","OS4"}<=scenarios else "FAIL"})
    science.append({"id":"required_ablations","status":"PASS" if {"A0","A1","A2","A3","A4","A5","A6","A7","Full","EPL3","No-quotient"}<=methods else "FAIL"})
    full_rows=[row for row in scenario if row.get("method")=="Full"]
    corr_fields={"score_total_residual_energy_pearson_r","score_total_prompt_power_pearson_r"}
    science.append({"id":"score_energy_power_correlations","status":"PASS" if full_rows and all(corr_fields<=row.keys() for row in full_rows) else "FAIL"})
    per_epoch=_read_csv(directory/"per_epoch_scores.csv.gz")
    common_support=True
    compared_methods={"A1","A2","A3","A4","A5","A6","A7","Full","EPL3","No-quotient"}
    for scenario_id in {row["scenario"] for row in per_epoch}:
        support=[]
        for method in compared_methods:
            support.append({(row["epoch"],row["tracked_prns"]) for row in per_epoch
                            if row["scenario"]==scenario_id and row["method"]==method})
        common_support &= bool(support[0]) and all(item==support[0] for item in support[1:])
    science.append({"id":"ablation_common_epoch_prn_support","status":"PASS" if common_support else "FAIL"})
    for name in ("per_epoch_scores.csv.gz","participation_posteriors.csv.gz","texbat_prn_bayes_factors.csv.gz",
                 "oakbat_prn_bayes_factors.csv.gz","texbat_score_diagnostics.csv.gz","oakbat_score_diagnostics.csv.gz"):
        try: rows=_read_csv(directory/name);ok=len(rows)>0
        except Exception:ok=False
        admin.append({"id":f"read_{name}","status":"PASS" if ok else "FAIL"})
    plot_sizes={name:(directory/"plots"/name).stat().st_size for name in PLOTS if (directory/"plots"/name).is_file()}
    admin.append({"id":"plots_nontrivial","status":"PASS" if len(plot_sizes)==len(PLOTS) and min(plot_sizes.values())>5000 else "FAIL",
                  "sizes":plot_sizes})
    manifest=json.loads((directory/"artifact_manifest_sha256.json").read_text());actual=manifest_entries(directory)
    ok=manifest.get("files")==actual
    admin.append({"id":"artifact_checksums","status":"PASS" if ok else "FAIL",
                  "declared_count":len(manifest.get("files",{})),"actual_count":len(actual)})
    forbidden=False
    forbidden_term="".join(("bli","nd pre","registration"))
    for path in [ROOT/"docs/Q_COMET_STAGE0.md",directory/"README.md",directory/"pre_evaluation_freeze.json"]:
        content=path.read_text(errors="replace").lower()
        forbidden |= forbidden_term in content
    admin.append({"id":"terminology","status":"PASS" if not forbidden else "FAIL"})
    return science,admin


def main():
    p=argparse.ArgumentParser();p.add_argument("artifact_dir",type=Path);p.add_argument("--create-manifest",action="store_true");p.add_argument("--fresh-clone",action="store_true");args=p.parse_args()
    directory=args.artifact_dir.resolve()
    if args.create_manifest:write_manifest(directory)
    science,admin=verify(directory)
    report={"schema":"gnss-doppler-lab.q-comet-verifier.v1","science_checks":science,"administrative_checks":admin,
            "science_status":"PASS" if science and all(x["status"]=="PASS" for x in science) else "FAIL",
            "administrative_status":"PASS" if admin and all(x["status"]=="PASS" for x in admin) else "FAIL"}
    report["overall_status"]="PASS" if report["science_status"]==report["administrative_status"]=="PASS" else "FAIL"
    (directory/"verifier_report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(report,sort_keys=True))
    if args.fresh_clone and report["overall_status"]=="PASS":
        with tempfile.TemporaryDirectory(prefix="q-comet-fresh-clone-") as temporary:
            clone=Path(temporary)/"repo";subprocess.run(["git","clone","--quiet","--no-hardlinks",str(ROOT),str(clone)],check=True)
            subprocess.run(["git","-C",str(clone),"checkout","--quiet",subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()],check=True)
            command=[sys.executable,str(clone/"scripts/verify_q_comet_stage0.py"),str(clone/"artifacts/q_comet_stage0_static")]
            child=subprocess.run(command,text=True,capture_output=True)
            fresh={"schema":"gnss-doppler-lab.q-comet-fresh-clone-verifier.v1","exit_code":child.returncode,"stdout":child.stdout,"stderr":child.stderr,
                   "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"status":"PASS" if child.returncode==0 else "FAIL"}
            (directory/"fresh_clone_verifier_report.json").write_text(json.dumps(fresh,sort_keys=True,indent=2)+"\n")
            if child.returncode:return child.returncode
    return 0 if report["overall_status"]=="PASS" else 2


if __name__=="__main__":raise SystemExit(main())
