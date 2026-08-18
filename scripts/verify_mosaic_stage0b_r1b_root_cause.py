#!/usr/bin/env python3
"""Fresh-clone compact verifier for the MOSAIC R1b diagnostic artifact."""
from __future__ import annotations
import csv, gzip, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts/mosaic_stage0b_r1b_multiprn_root_cause"
PRIOR=ROOT/"artifacts/mosaic_stage0b_r1a_frozen_analysis/final_verdict.json"
REQUIRED={"README.md","root_cause_preregistration.json","source_binding.json","retained_evidence_inventory.json","failure_case_metrics.csv","comparator_case_metrics.csv","receiver_frame_trajectories.csv.gz","tracking_action_differences.csv.gz","template_projection_metrics.csv","oracle_diagnostic_metrics.csv","phase_cancellation_metrics.csv","lock_channel_stability.json","prn_baseline_dominance.csv","temporal_window_diagnostics.csv","factorial_identifiability.json","root_cause_decision_table.csv","final_root_cause_verdict.json"}

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def verify():
    manifest=json.loads((ART/"artifact_manifest_sha256.json").read_text()); listed=set()
    for item in manifest["files"]:
        p=ART/item["path"]; listed.add(item["path"])
        if not p.is_file() or p.stat().st_size!=item["size_bytes"] or sha(p)!=item["sha256"]: raise ValueError(f"checksum mismatch: {p}")
    if not REQUIRED<=listed: raise ValueError(f"required artifacts missing: {REQUIRED-listed}")
    actual={str(p.relative_to(ART)) for p in ART.rglob("*") if p.is_file() and p.name!="artifact_manifest_sha256.json"}
    if actual!=listed: raise ValueError("manifest coverage mismatch")
    prior=json.loads(PRIOR.read_text()); final=json.loads((ART/"final_root_cause_verdict.json").read_text())
    if prior["verdict"]!="NO_GO_MOSAIC_MULTI_PRN_RECOVERY" or final["prior_r1a_verdict"]!=prior["verdict"]: raise ValueError("prior verdict not preserved")
    if final["recommendation"] not in ("Frozen corrected observer confirmation","Terminate MOSAIC"): raise ValueError("invalid recommendation")
    with (ART/"failure_case_metrics.csv").open(newline="") as f: failures=list(csv.DictReader(f))
    with (ART/"comparator_case_metrics.csv").open(newline="") as f: comparators=list(csv.DictReader(f))
    if len(failures)!=8 or len(comparators)!=20: raise ValueError("target accounting mismatch")
    print(json.dumps({"status":"PASS","checksums":len(listed),"failure_targets":len(failures),"comparator_targets":len(comparators),"prior_r1a_verdict":prior["verdict"],"iq_injection_rerun":False,"receiver_replay_rerun":False},indent=2))
    return len(listed)

if __name__=="__main__":verify()
