#!/usr/bin/env python3
"""Independent R1.3 artifact verifier; trusts no runner summary fields."""
from __future__ import annotations

import argparse, csv, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REQUIRED=("README.md","config.json","environment.json","receiver_source_binding.json","gnss_sdr_source_binding.json","gnss_sdr_tracking_semantics.md","ca_code_validation.json","ca_code_correlation.csv","candidate_application_audit.csv","candidate_fingerprints.json","alignment_hypotheses.csv","selected_alignment.json","center_validation.csv","center_validation_summary.json","center_metrics_by_prn.csv","center_metrics_by_channel.csv","center_metrics_by_time_block.csv","global_offset_sensitivity.csv","global_offset_application_audit.json","prn_sampling_summary.csv","raw_overlap_audit.json","execution_validity.json","go_no_go.json","test_report.txt","verification_report.json","checksums.json","plots")

def load_json(path): return json.loads(path.read_text())
def load_csv(path):
    with path.open(newline="") as handle: return list(csv.DictReader(handle))
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()
def recursive_checksums(root):
    return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name not in {"checksums.json","verification_report.json"}}
def f(row,key): return float(row[key])

def candidate_fingerprints_valid(document):
    values=list(document.get("fingerprints",{}).values())
    return bool(values) and len(values)==int(document.get("expected_unique",-1)) and len(set(values))==len(values)

def global_offset_calls_valid(calls):
    by_epoch=defaultdict(set)
    for call in calls: by_epoch[str(call.get("invocation"))].add((int(call.get("global_offset_samples")),int(call.get("start_byte")),int(call.get("end_byte"))))
    required={-1000,-500,0,500,1000}
    return bool(by_epoch) and all({x[0] for x in values}==required and len({(x[1],x[2]) for x in values})==5 for values in by_epoch.values())

def verify(root: Path):
    errors=[]
    for name in REQUIRED:
        if not (root/name).exists(): errors.append(f"missing:{name}")
    if errors: return {"status":"FAIL","errors":errors}
    rows=load_csv(root/"center_validation.csv")
    prns=Counter(int(r["prn"]) for r in rows); blocks=defaultdict(list)
    for row in rows: blocks[row["role"]].append(row)
    n=len(rows); within=sum(abs(f(r,"peak_delay_offset_chips"))<=.125 and abs(f(r,"peak_doppler_offset_hz"))<=50 for r in rows)/n if n else 0
    exact=sum(f(r,"peak_delay_offset_chips")==0 and f(r,"peak_doppler_offset_hz")==0 for r in rows)/n if n else 0
    boundary=sum(str(r["grid_boundary"]).lower() in ("1","true") for r in rows)/n if n else 1
    rho=float(spearmanr([f(r,"center_magnitude") for r in rows],[f(r,"mat_prompt_magnitude") for r in rows]).statistic) if n>2 else 0
    prn_rhos=[]
    for prn in prns:
        group=[r for r in rows if int(r["prn"])==prn]
        if len(group)>2: prn_rhos.append(float(spearmanr([f(r,"center_magnitude") for r in group],[f(r,"mat_prompt_magnitude") for r in group]).statistic))
    median_rho=float(np.nanmedian(prn_rhos)) if prn_rhos else 0
    actual={"n":n,"prn_count":len(prns),"min_per_prn":min(prns.values()) if prns else 0,
            "dominant_fraction":max(prns.values())/n if n else 1,"within_tolerance_fraction":within,
            "exact_center_fraction":exact,"boundary_fraction":boundary,"pooled_spearman":rho,
            "median_prn_spearman":median_rho,"block_counts":{k:len(v) for k,v in blocks.items()},
            "block_prns":{k:len({int(x['prn']) for x in v}) for k,v in blocks.items()}}
    binding=load_json(root/"receiver_source_binding.json"); execution=load_json(root/"execution_validity.json")
    ca=load_json(root/"ca_code_validation.json"); fingerprints=load_json(root/"candidate_fingerprints.json")
    offset=load_json(root/"global_offset_application_audit.json")
    unique=set(fingerprints.get("fingerprints",{}).values()); expected=fingerprints.get("expected_unique",0)
    calls=offset.get("calls",[]); by_epoch=defaultdict(set)
    for call in calls: by_epoch[str(call.get("invocation"))].add((call.get("global_offset_samples"),call.get("start_byte"),call.get("end_byte")))
    offset_ok=global_offset_calls_valid(calls)
    a1=binding.get("A1_SOURCE_BINDING")=="PASS" and binding.get("format")=="ishort" and float(binding.get("fs",0))==25_000_000 and "skip_samples" in binding and "resampling" in binding
    a2=(ca.get("canonical_prns_passed")==32 and ca.get("local_generator_absent") is True and execution.get("source_interval_semantics_confirmed") is True and execution.get("candidate_fields_applied") is True and execution.get("stable_filter_applied") is True and candidate_fingerprints_valid(fingerprints) and offset_ok)
    a3=(n>=800 and len(prns)>=8 and actual["min_per_prn"]>=50 and actual["dominant_fraction"]<=.2 and set(blocks)=={"train","calibration","holdout"} and all(len(v)>=200 and actual["block_prns"][k]>=8 for k,v in blocks.items()) and within>=.95 and rho>=.9 and median_rho>=.8 and boundary<=.05)
    verdict="SOURCE_BINDING_INVALID" if not a1 else "RECONSTRUCTION_IMPLEMENTATION_INVALID" if not a2 else "TRACKER_RAW_ALIGNMENT_UNRESOLVED" if not a3 else "PHYSICAL_CENTER_VALID"
    summary=load_json(root/"center_validation_summary.json")
    for key in ("n","prn_count","min_per_prn","dominant_fraction","within_tolerance_fraction","exact_center_fraction","boundary_fraction","pooled_spearman","median_prn_spearman"):
        if key not in summary or not math.isclose(float(summary[key]),float(actual[key]),rel_tol=1e-9,abs_tol=1e-9): errors.append(f"summary_mismatch:{key}")
    go=load_json(root/"go_no_go.json")
    if go.get("verdict")!=verdict or go.get("physics_no_go_claim") is not False: errors.append("verdict_mismatch")
    saved=load_json(root/"checksums.json"); recomputed=recursive_checksums(root)
    if saved.get("files")!=recomputed: errors.append("checksum_mismatch")
    return {"status":"PASS" if not errors else "FAIL","errors":errors,"recomputed":actual,
            "gates":{"A1_SOURCE_BINDING":"PASS" if a1 else "FAIL","A2_IMPLEMENTATION_AND_INTERVAL_VALIDITY":"PASS" if a2 else "FAIL","A3_MULTI_PRN_CENTER_RECOVERY":"PASS" if a3 else "FAIL"},"verdict":verdict,
            "candidate_fingerprint_unique":len(unique),"global_offsets_independent":offset_ok,"recursive_checksums":recomputed}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("artifact_dir",type=Path); args=ap.parse_args()
    report=verify(args.artifact_dir); (args.artifact_dir/"verification_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    raise SystemExit(0 if report["status"]=="PASS" else 1)
if __name__=="__main__": main()
