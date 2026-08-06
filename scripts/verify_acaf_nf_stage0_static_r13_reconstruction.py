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

PHYSICAL_HASH_FIELDS=("raw_interval_content_sha256","raw_interval_range_sha256",
 "replica_chip_indices_sha256","carrier_wipeoff_sha256","aux_indices_sha256",
 "nco_indices_sha256","prompt_indices_sha256","result_field_sha256")

def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)

def physical_applications_valid(rows):
    """Reject opaque candidate labels unless persisted physical applications differ."""
    if not rows or any(not all(_sha(row.get(k)) for k in PHYSICAL_HASH_FIELDS) for row in rows): return False
    by_candidate=defaultdict(set)
    for row in rows:
        by_candidate[row.get("candidate")].add(tuple(row[k] for k in PHYSICAL_HASH_FIELDS))
    return len(by_candidate)>1 and all(by_candidate.values()) and len({tuple(sorted(v)) for v in by_candidate.values()})==len(by_candidate)

def global_offset_calls_valid(calls):
    by_epoch=defaultdict(set)
    for call in calls: by_epoch[str(call.get("invocation"))].add((int(call.get("global_offset_samples")),int(call.get("start_byte")),int(call.get("end_byte"))))
    required={-1000,-500,0,500,1000}
    return bool(by_epoch) and all({x[0] for x in values}==required and len({(x[1],x[2]) for x in values})==5 for values in by_epoch.values())

def roles_nonoverlap(rows):
    for i,left in enumerate(rows):
        for right in rows[i+1:]:
            if left.get("role")!=right.get("role") and int(left["start"])<int(right["end"]) and int(right["start"])<int(left["end"]): return False
    return True

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
    binding=load_json(root/"receiver_source_binding.json"); source=load_json(root/"gnss_sdr_source_binding.json")
    ca=load_json(root/"ca_code_validation.json"); audit=load_csv(root/"candidate_application_audit.csv")
    offset=load_json(root/"global_offset_application_audit.json")
    calls=offset.get("calls",[]); by_epoch=defaultdict(set)
    for call in calls: by_epoch[str(call.get("invocation"))].add((call.get("global_offset_samples"),call.get("start_byte"),call.get("end_byte")))
    offset_ok=global_offset_calls_valid(calls)
    binding_checks=binding.get("checks",{})
    a1=(bool(binding_checks) and all(v is True for v in binding_checks.values()) and
        binding.get("recording_id")=="cleanStatic" and _sha(binding.get("raw_sha256")) and
        binding.get("format")=="ishort" and float(binding.get("fs",0))==25_000_000 and
        int(binding.get("skip_samples",-1))==0 and binding.get("resampling")=="none")
    source_required=("git_base_sha","modified_tracked_files","build_evidence","compiler_evidence",
                     "executable_sha256","receiver_config_values","vector_length","tap_count",
                     "tap_spacing_chips","extended_integration_symbols","track_pilot",
                     "prompt_support_mapping_authenticated")
    source_ok=all(k in source and source[k] not in (None,"",[]) for k in source_required)
    overlap=load_json(root/"raw_overlap_audit.json"); overlap_ok=roles_nonoverlap(overlap.get("rows",[]))
    row_evidence=all(all(k in r for k in ("tracker_row","mat_sha256","raw_start_sample","raw_end_sample",
                 "support_start_sample","support_end_sample","center_magnitude","mat_prompt_magnitude")) for r in rows)
    a2=(source_ok and source.get("prompt_support_mapping_authenticated") is True and
        ca.get("canonical_prns_passed")==32 and ca.get("local_generator_absent") is True and
        physical_applications_valid(audit) and offset_ok and overlap_ok and row_evidence)
    a3=(n>=800 and len(prns)>=8 and actual["min_per_prn"]>=50 and actual["dominant_fraction"]<=.2 and set(blocks)=={"train","calibration","holdout"} and all(len(v)>=200 and actual["block_prns"][k]>=8 for k,v in blocks.items()) and within>=.95 and rho>=.9 and median_rho>=.8 and boundary<=.05)
    verdict="SOURCE_BINDING_INVALID" if not a1 else "RECONSTRUCTION_IMPLEMENTATION_INVALID" if not a2 else "TRACKER_RAW_ALIGNMENT_UNRESOLVED" if not a3 else "PHYSICAL_CENTER_VALID"
    summary=load_json(root/"center_validation_summary.json")
    for key in ("n","prn_count","min_per_prn","dominant_fraction","within_tolerance_fraction","exact_center_fraction","boundary_fraction","pooled_spearman","median_prn_spearman"):
        if key not in summary or not math.isclose(float(summary[key]),float(actual[key]),rel_tol=1e-9,abs_tol=1e-9): errors.append(f"summary_mismatch:{key}")
    saved_prn={int(r["prn"]):int(r["n"]) for r in load_csv(root/"center_metrics_by_prn.csv")}
    if saved_prn!={prn:count for prn,count in prns.items()}: errors.append("per_prn_count_mismatch")
    saved_sampling={int(r["prn"]):int(r["n"]) for r in load_csv(root/"prn_sampling_summary.csv")}
    if saved_sampling!={prn:count for prn,count in prns.items()}: errors.append("prn_sampling_mismatch")
    saved_blocks={r["time_block"]:(int(r["n"]),int(r["prn_count"])) for r in load_csv(root/"center_metrics_by_time_block.csv")}
    actual_blocks={role:(len(group),len({int(x["prn"]) for x in group})) for role,group in blocks.items()}
    if saved_blocks!=actual_blocks: errors.append("block_metric_mismatch")
    go=load_json(root/"go_no_go.json")
    if go.get("verdict")!=verdict or go.get("physics_no_go_claim") is not False: errors.append("verdict_mismatch")
    selected=load_json(root/"selected_alignment.json")
    if selected!=go or ((not (a1 and a2 and a3)) and selected.get("selected_alignment") is not None): errors.append("selected_alignment_mismatch")
    expected_gates={"A1_SOURCE_BINDING":"PASS" if a1 else "FAIL","A2_IMPLEMENTATION_AND_INTERVAL_VALIDITY":"PASS" if a2 else "FAIL","A3_MULTI_PRN_CENTER_RECOVERY":"PASS" if a3 else "FAIL"}
    if any(go.get(k)!=v for k,v in expected_gates.items()): errors.append("gate_predicate_mismatch")
    saved=load_json(root/"checksums.json"); recomputed=recursive_checksums(root)
    if saved.get("files")!=recomputed: errors.append("checksum_mismatch")
    return {"status":"PASS" if not errors else "FAIL","errors":errors,"recomputed":actual,
            "gates":{"A1_SOURCE_BINDING":"PASS" if a1 else "FAIL","A2_IMPLEMENTATION_AND_INTERVAL_VALIDITY":"PASS" if a2 else "FAIL","A3_MULTI_PRN_CENTER_RECOVERY":"PASS" if a3 else "FAIL"},"verdict":verdict,
            "candidate_physical_applications":physical_applications_valid(audit),"global_offsets_independent":offset_ok,
            "cross_role_nonoverlap":overlap_ok,"recursive_checksums":recomputed}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("artifact_dir",type=Path); args=ap.parse_args()
    report=verify(args.artifact_dir); (args.artifact_dir/"verification_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    raise SystemExit(0 if report["status"]=="PASS" else 1)
if __name__=="__main__": main()
