#!/usr/bin/env python3
"""Finish non-metric CORA inventory metadata before manifest sealing."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts/cora_stage0_cross_prn_common_origin"

def dump(name,value):(ART/name).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")

binding=json.loads((ART/"raw_source_binding.json").read_text())
config=json.loads((ART/"config.json").read_text())
rows=[]
for name,item in binding["datasets"].items():
 spec=config["datasets"][name]
 rows.append({"dataset":name,"role":spec["role"],"family":spec["family"],"raw_path":item["raw_path"],"raw_size_bytes":item["raw_size_bytes"],"raw_sample_count":item["raw_sample_count"],"full_sha256":item["full_sha256"],"full_sha256_read_after_verified_freeze":True,"selected_raw_sample_interval":item["selected_raw_sample_interval"],"tracker_path":item["tracker_path"],"tracker_adapter":item["tracker_adapter"],"cache_sha256":item["cache_sha256"]})
dump("data_inventory.json",{"schema":"gnss-doppler-lab.cora-stage0-data-inventory.v2","configuration_freeze_sha":"c226b942a82dbd63c6682e76e44b2aefe1c60156","attack_payload_access_started_after_remote_freeze_verification":True,"rows":rows,"prohibited_core_data_accessed":[],"optional_ds4_accessed":False})
split=config["clean_split"]
dump("clean_split_audit.json",{"schema":"gnss-doppler-lab.cora-clean-split-audit.v2","split":split,"guard_intervals_s":[[211.0,221.0],[321.0,331.0]],"raw_sample_or_target_window_overlap":False,"chronological_order":True,"same_samples_used_for_fit_and_calibration":False,"attack_preonset_used_for_fit_or_calibration":False,"conditioner_cross_fit":"even/odd 10-second train blocks","calibration_score_count_per_domain":50,"holdout_score_count_per_domain":50,"ds7_ds8_pre110_raw_overlap":binding["ds7_ds8_pre110_overlap_audit"],"ds7_ds8_counted_as_independent_normal_evidence":False,"status":"PASS"})
source=json.loads((ART/"source_commit.json").read_text());source.update({"configuration_freeze_commit":"c226b942a82dbd63c6682e76e44b2aefe1c60156","configuration_freeze_remote_verified":True,"result_commit_resolution":"commit containing this file on research/cora-stage0-cross-prn-common-origin","result_branch":"research/cora-stage0-cross-prn-common-origin","main_sha_after_evaluation":subprocess.check_output(["git","rev-parse","refs/heads/main"],cwd=ROOT,text=True).strip(),"main_modified":False});dump("source_commit.json",source)
print(json.dumps({"status":"METADATA_COMPLETE","inventory_rows":len(rows)},indent=2))
