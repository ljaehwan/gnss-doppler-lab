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
    by_epoch=defaultdict(list)
    try:
        for call in calls:
            start=int(call.get("start")); end=int(call.get("end")); offset=int(call.get("global_offset_samples"))
            if int(call.get("start_byte"))!=4*start or int(call.get("end_byte"))!=4*end: return False
            if end-start != int(call.get("support_length_samples",call.get("n_samples",end-start))): return False
            by_epoch[str(call.get("invocation"))].append((offset,start,end))
    except (TypeError,ValueError): return False
    required={-1000,-500,0,500,1000}
    return bool(by_epoch) and all({x[0] for x in values}==required and len(values)==5 and
      all(start-offset==values[0][1]-values[0][0] and end-offset==values[0][2]-values[0][0]
          for offset,start,end in values) for values in by_epoch.values())

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
    offset_csv=load_csv(root/"global_offset_sensitivity.csv")
    recomputed_offsets=[]
    for offset_value in (-1000,-500,0,500,1000):
        group=[row for row in calls if int(row.get("global_offset_samples"))==offset_value]
        count=len(group)
        exact_fraction=sum(float(r["peak_delay_offset_chips"])==0 and float(r["peak_doppler_offset_hz"])==0 for r in group)/count if count else 0
        within_fraction=sum(abs(float(r["peak_delay_offset_chips"]))<=.125 and abs(float(r["peak_doppler_offset_hz"]))<=50 for r in group)/count if count else 0
        boundary_fraction=sum(str(r["grid_boundary"]).lower() in ("1","true") for r in group)/count if count else 1
        pooled=float(spearmanr([float(r["center_magnitude"]) for r in group],[float(r["mat_prompt_magnitude"]) for r in group]).statistic) if count>2 else 0
        per_prn=[]
        for prn in sorted({int(r["prn"]) for r in group}):
            pr=[r for r in group if int(r["prn"])==prn]
            if len(pr)>2: per_prn.append(float(spearmanr([float(r["center_magnitude"]) for r in pr],[float(r["mat_prompt_magnitude"]) for r in pr]).statistic))
        recomputed_offsets.append({"global_offset_samples":offset_value,"valid_raw_epochs":count,
          "exact_center_fraction":exact_fraction,"within_tolerance_fraction":within_fraction,
          "boundary_fraction":boundary_fraction,"pooled_spearman":pooled,
          "median_prn_spearman":float(np.nanmedian(per_prn)) if per_prn else 0,
          "peak_delay_median":float(np.median([float(r["peak_delay_offset_chips"]) for r in group])) if group else 0,
          "peak_doppler_median":float(np.median([float(r["peak_doppler_offset_hz"]) for r in group])) if group else 0})
    saved_offsets={int(row["global_offset_samples"]):row for row in offset_csv}
    offset_csv_ok=len(saved_offsets)==5
    for expected in recomputed_offsets:
        saved_row=saved_offsets.get(expected["global_offset_samples"])
        if saved_row is None: offset_csv_ok=False; continue
        for key,value in expected.items():
            if key=="global_offset_samples": continue
            if not math.isclose(float(saved_row[key]),float(value),rel_tol=1e-9,abs_tol=1e-9): offset_csv_ok=False
    binding_checks=binding.get("checks",{})
    a1=(bool(binding_checks) and all(v is True for v in binding_checks.values()) and
        binding.get("recording_id")=="cleanStatic" and _sha(binding.get("raw_sha256")) and
        _sha(binding.get("manifest_sha256")) and bool(binding.get("manifest_path")) and
        bool(binding.get("mat_inventory")) and all(_sha(item.get("sha256")) for item in binding.get("mat_inventory",[])) and
        isinstance(binding.get("config_values"),dict) and binding.get("format")=="ishort" and
        float(binding.get("fs",0))==25_000_000 and int(binding.get("skip_samples",-1))==0 and binding.get("resampling")=="none")
    source_required=("git_base_sha","modified_tracked_files","build_evidence","compiler_evidence",
                     "executable_sha256","receiver_config_values","vector_length","tap_count",
                     "tap_spacing_chips","extended_integration_symbols","track_pilot",
                     "prompt_support_mapping_authenticated")
    source_ok=(all(k in source and source[k] not in (None,"",[]) for k in source_required) and
      source.get("base_head_exact") is True and source.get("modified_file_set_exact") is True and
      source.get("A2_source_semantics_sufficient") is True and int(source.get("vector_length",0))==25000)
    overlap=load_json(root/"raw_overlap_audit.json"); overlap_ok=roles_nonoverlap(overlap.get("rows",[]))
    schema=("recording","prn","channel","previous_mat_row","current_mat_row","next_mat_row","mat_path","mat_sha256",
      "consumed_start_sample","consumed_end_sample","consumed_length_samples","support_start_sample","support_end_sample",
      "support_length_samples","raw_start_sample","raw_end_sample","raw_start_byte","raw_end_byte","vector_length",
      "aux_row_index","nco_row_index","prompt_row_index","center_magnitude","mat_prompt_magnitude","role")
    row_evidence=all(all(k in r for k in schema) for r in rows)
    mapping_ok=all(int(r["previous_mat_row"])+1==int(r["current_mat_row"]) and int(r["current_mat_row"])+1==int(r["next_mat_row"]) and
      int(r["aux_row_index"])==int(r["previous_mat_row"]) and int(r["nco_row_index"])==int(r["previous_mat_row"]) and
      int(r["prompt_row_index"])==int(r["current_mat_row"]) and int(r["support_start_sample"])==int(r["consumed_start_sample"]) and
      int(r["support_end_sample"])-int(r["support_start_sample"])==int(r["vector_length"])==int(r["support_length_samples"]) and
      int(r["consumed_end_sample"])-int(r["consumed_start_sample"])==int(r["consumed_length_samples"]) and
      24999<=int(r["consumed_length_samples"])<=25001 for r in rows)
    ca_rows=load_csv(root/"ca_code_correlation.csv")
    ca_ok=ca.get("canonical_prns_passed")==32 and ca.get("local_generator_absent") is True and len(ca_rows)==32 and {int(x["prn"]) for x in ca_rows}==set(range(1,33))
    fingerprints=load_json(root/"candidate_fingerprints.json")
    expected_fp={r.get("key"):{field:r.get(field) for field in PHYSICAL_HASH_FIELDS} for r in audit}
    fingerprint_ok=fingerprints.get("fingerprints")==expected_fp and fingerprints.get("expected_unique")==len(expected_fp)
    audit_schema=all(all(k in r for k in schema) for r in audit)
    a2=(source_ok and source.get("prompt_support_mapping_authenticated") is True and ca_ok and
        physical_applications_valid(audit) and fingerprint_ok and offset_ok and offset_csv_ok and overlap_ok and row_evidence and audit_schema and mapping_ok)
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
