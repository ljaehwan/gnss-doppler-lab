#!/usr/bin/env python3
"""Independent structural/checksum verifier for CHORD Stage-0A."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts/chord_stage0a_clean_identifiability"
REQUIRED={"README.md","preregistration.json","source_commit.json","source_binding.json","extraction_schedule.json",
"data_inventory.json","split_audit.json","profile_availability.csv","pair_metrics.csv","lag_metrics.csv",
"per_prn_metrics.csv","baseline_metrics.csv","control_metrics.json","bootstrap_intervals.csv","final_verdict.json",
"artifact_manifest_sha256.json"}
PLOTS=("similarity_distribution","roc","lag_auc","prn_heatmap","prn_effect","baselines","norm_vs_similarity","cn0_vs_similarity","lopo","controls")
VERDICTS={"CHORD_CLEAN_IDENTIFIABILITY_PASS_WORTH_ATTACK_STAGE0B","NO_GO_CHORD_CLEAN_IDENTIFIABILITY","INCONCLUSIVE_INPUT_OR_ALIGNMENT"}

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def actual_manifest(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name!="artifact_manifest_sha256.json"}

def verify(root=ART):
    files={str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} if root.exists() else set()
    missing=sorted(REQUIRED-files)
    actual=actual_manifest(root) if root.exists() else {}
    recorded=json.loads((root/"artifact_manifest_sha256.json").read_text()) if (root/"artifact_manifest_sha256.json").exists() else {}
    final=json.loads((root/"final_verdict.json").read_text()) if (root/"final_verdict.json").exists() else {}
    prereg=json.loads((root/"preregistration.json").read_text()) if (root/"preregistration.json").exists() else {}
    source=json.loads((root/"source_binding.json").read_text()) if (root/"source_binding.json").exists() else {}
    split=json.loads((root/"split_audit.json").read_text()) if (root/"split_audit.json").exists() else {}
    plots_ok=all(any(name in p for p in files) for name in PLOTS)
    clean=(prereg.get("attack_data_used") is False and prereg.get("attack_paths_accessed") is False
           and final.get("attack_data_used") is False and final.get("attack_paths_accessed") is False
           and final.get("stage0b_run") is False and final.get("ai_model_used") is False)
    forbidden=any(any(token in str(v).lower() for token in ("/ds1","/ds2","/ds3","/ds4","/ds5","/ds6","/ds7","/ds8","/os1","/os2","/os3","/os4")) for v in actual.values())
    split_ok=bool(split) and all(not v["raw_sample_overlap"] and not v["ten_second_block_overlap"] for v in split.values())
    csv_ok=True
    for name in ("profile_availability.csv","pair_metrics.csv","lag_metrics.csv","per_prn_metrics.csv","baseline_metrics.csv","bootstrap_intervals.csv"):
        path=root/name
        csv_ok &= path.exists() and len(list(csv.DictReader(path.open())))>0
    checks={"required":not missing,"manifest":recorded==actual,"verdict":final.get("verdict") in VERDICTS,
            "plots":plots_ok,"clean_scope":clean and not forbidden,"source_bound":source.get("status") in ("PASS","EXPECTED_NOT_HASHED_UNTIL_EVALUATION"),
            "split":split_ok,"csv_nonempty":csv_ok}
    return {"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"missing":missing,"file_count":len(actual)}

def main():
    if "--write-manifest" in sys.argv:
        (ART/"artifact_manifest_sha256.json").write_text(json.dumps(actual_manifest(ART),indent=2,sort_keys=True)+"\n")
    result=verify(); print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["status"]=="PASS" else 1

if __name__=="__main__": raise SystemExit(main())
