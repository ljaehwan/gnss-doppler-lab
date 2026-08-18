#!/usr/bin/env python3
"""Verify MIRAGE Stage-0A artifact checksums and fail-closed semantics."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts/mirage_stage0a_complex_minor_feasibility"
REQUIRED={"README.md","config.json","preregistration.json","source_commit.json","data_inventory.json",
"clean_split_audit.json","caf_grid.json","injection_design.json","injection_design_sha256.json",
"algebraic_minor_tests.json","thresholds.json","clean_metrics.csv","per_case_scores.csv.gz",
"injection_metrics.csv","control_metrics.csv","scale_ablation.csv","relation_destruction_metrics.json",
"prn_dominance.json","bootstrap_intervals.csv","final_verdict.json","artifact_manifest_sha256.json"}
PLOTS={"clean_vs_two_source_minor_distribution.png","collapsed_vs_resolvable_source.png","scale_effect.png",
"gain_awgn_control.png","four_prn_detection.png","temporal_desynchronization.png","prn_contribution.png",
"rms_cn0_shortcut_audit.png","threshold_stability.png","example_complex_caf_and_minor_field.png"}
VERDICTS={"GO_FOR_FROZEN_STAGE0B_REAL_STATIC_EVALUATION","NO_GO_MIRAGE_PHYSICAL_HYPOTHESIS","INCONCLUSIVE_INPUT_OR_SUPPORT"}


def sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def actual_manifest(root:Path)->dict[str,object]:
    files=[{"path":str(p.relative_to(root)),"size_bytes":p.stat().st_size,"sha256":sha(p)}
           for p in sorted(root.rglob("*")) if p.is_file() and p.name!="artifact_manifest_sha256.json"]
    return {"schema":"gnss-doppler-lab.artifact-manifest-sha256.v1","files":files}


def verify(root:Path=ART)->dict[str,object]:
    files={str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} if root.exists() else set()
    missing=sorted(REQUIRED-files); plot_files={Path(x).name for x in files if x.startswith("plots/")}
    recorded=json.loads((root/"artifact_manifest_sha256.json").read_text()) if (root/"artifact_manifest_sha256.json").exists() else {}
    actual=actual_manifest(root) if root.exists() else {}
    final=json.loads((root/"final_verdict.json").read_text()) if (root/"final_verdict.json").exists() else {}
    config=json.loads((root/"config.json").read_text()) if (root/"config.json").exists() else {}
    split=json.loads((root/"clean_split_audit.json").read_text()) if (root/"clean_split_audit.json").exists() else {}
    algebra=json.loads((root/"algebraic_minor_tests.json").read_text()) if (root/"algebraic_minor_tests.json").exists() else {}
    inconclusive=final.get("verdict")=="INCONCLUSIVE_INPUT_OR_SUPPORT"
    fail_closed=(not inconclusive or (final.get("controlled_injection_executed") is False
        and final.get("receiver_replay_executed") is False and final.get("clean_scoring_executed") is False
        and split.get("status")=="FAIL"))
    clean_scope=(config.get("attack_data_accessed") is False and config.get("real_attack_evaluation") is False
                 and final.get("real_attack_data_accessed") is False)
    checks={"required_files":not missing,"required_plots":PLOTS<=plot_files,"manifest":recorded==actual,
        "allowed_verdict":final.get("verdict") in VERDICTS,"clean_scope":clean_scope,
        "algebraic_tests":algebra.get("status")=="PASS","fail_closed":fail_closed,
        "frozen_base":config.get("base_sha")=="3db0e12976b6ff98452096e921cf298be459d0e8"}
    return {"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"missing":missing,
            "manifest_file_count":len(actual.get("files",[]))}


def main()->int:
    if "--write-manifest" in sys.argv:
        (ART/"artifact_manifest_sha256.json").write_text(json.dumps(actual_manifest(ART),indent=2,sort_keys=True)+"\n")
    result=verify();print(json.dumps(result,indent=2,sort_keys=True));return 0 if result["status"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
