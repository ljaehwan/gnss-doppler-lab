#!/usr/bin/env python3
"""Independent R1.4 verifier.  It imports no producer metric or gate helper."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
REQUIRED=("README.md config.json environment.json r13_frozen_lineage.json frozen_reconstruction_config.json prompt_reproduction_metrics.json prompt_reproduction_by_prn.csv prompt_reproduction_by_time_block.csv delay_recovery_metrics.json delay_recovery_by_prn.csv delay_recovery_by_time_block.csv doppler_1ms_metrics.json aggregation_metrics.csv aggregation_by_prn.csv aggregation_by_time_block.csv paired_improvement.csv bootstrap_results.json doppler_mainlobe_diagnostics.csv residual_doppler_diagnostics.json per_block_scores.csv execution_validity.json go_no_go.json test_report.txt verification_report.json checksums.json plots").split()
CANDIDATE="nco_row=previous_aux_row=previous_remnant_sign=-1_carrier_sign=-1_global_offset=0"
VERDICTS={"RECONSTRUCTION_IMPLEMENTATION_INVALID","TRACKER_RAW_RECONSTRUCTION_UNRESOLVED","PHYSICAL_RECONSTRUCTION_VALID_DOPPLER_RESOLUTION_LIMITED","PHYSICAL_CENTER_VALID"}
R13={"n":969,"prn_count":8,"pooled_spearman":0.9999965049269979,"median_prn_spearman":0.9999652753663446,"boundary_fraction":0.006191950464396285,"within_tolerance_fraction":0.8565531475748194,"exact_center_fraction":0.42105263157894735}
def loadj(p): return json.loads(p.read_text())
def loadc(p):
    with p.open(newline="") as f:return list(csv.DictReader(f))
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def checksums(root):return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name not in {"checksums.json","verification_report.json"}}
def b(v): return str(v).lower() in {"true","1"}
def rho(rows):
    if len(rows)<3:return 0.0
    x=float(spearmanr([float(r["center_magnitude"]) for r in rows],[float(r["mat_prompt_magnitude"]) for r in rows]).statistic)
    return x if math.isfinite(x) else 0.0
def prompt(rows):
    e=np.asarray([abs(float(r["center_magnitude"])/float(r["mat_prompt_magnitude"])-1) for r in rows])
    ps=[rho([r for r in rows if int(r["prn"])==p]) for p in sorted({int(r["prn"]) for r in rows})]
    return {"n":len(rows),"pooled_spearman":rho(rows),"median_prn_spearman":float(np.median(ps)),"median_relative_error":float(np.median(e)),"p95_relative_error":float(np.quantile(e,.95)),"p99_relative_error":float(np.quantile(e,.99)),"max_relative_error":float(np.max(e))}
def delay(rows):
    x=np.asarray([float(r["peak_delay_offset_chips"]) for r in rows]); bd=np.asarray([b(r["delay_boundary"]) for r in rows])
    return {"n":len(rows),"exact_center_fraction":float(np.mean(x==0)),"within_0_125_fraction":float(np.mean(abs(x)<=.125)),"boundary_fraction":float(np.mean(bd)),"histogram":{str(v):int(np.sum(x==v)) for v in sorted(set(x))}}
def doppler(rows):
    x=np.asarray([float(r["peak_doppler_offset_hz"]) for r in rows]); ratios=np.asarray([float(r["peak_center_ratio"]) for r in rows])
    out={"n":len(rows),"exact_center_fraction":float(np.mean(x==0)),"boundary_fraction":float(np.mean([b(r["doppler_boundary"]) for r in rows])),"median_abs_offset_hz":float(np.median(abs(x))),"p95_abs_offset_hz":float(np.quantile(abs(x),.95)),"median_peak_center_ratio":float(np.median(ratios)),"histogram":{str(v):int(np.sum(x==v)) for v in sorted(set(x))}}
    for hz in (50,100,150):out[f"within_{hz}_fraction"]=float(np.mean(abs(x)<=hz))
    return out
def close(a,b,tol=1e-12):
    if set(a)!=set(b):return False
    for k,v in a.items():
        if isinstance(v,dict):
            if not close(v,b[k],tol):return False
        elif isinstance(v,(int,float)) and not isinstance(v,bool):
            if not math.isfinite(float(b[k])) or abs(float(v)-float(b[k]))>tol:return False
        elif v!=b[k]:return False
    return True
def gate(a1,a2,a3a,a3b,a3c):
    if not a1 or not a2:v="RECONSTRUCTION_IMPLEMENTATION_INVALID"
    elif not a3a or not a3b:v="TRACKER_RAW_RECONSTRUCTION_UNRESOLVED"
    elif not a3c:v="PHYSICAL_RECONSTRUCTION_VALID_DOPPLER_RESOLUTION_LIMITED"
    else:v="PHYSICAL_CENTER_VALID"
    return {"A1_SOURCE_BINDING":"PASS" if a1 else "FAIL","A2_RECONSTRUCTION_IMPLEMENTATION":"PASS" if a2 else "FAIL","A3a_PROMPT_REPRODUCTION":"PASS" if a3a else "FAIL","A3b_CODE_DELAY":"PASS" if a3b else "FAIL","A3c_DOPPLER_AGGREGATION":"PASS" if a3c else "FAIL","verdict":v}

def verify(root:Path):
    errors=[]
    missing=[x for x in REQUIRED if not (root/x).exists()]
    if missing:return {"status":"FAIL","errors":[f"missing:{x}" for x in missing]}
    try:
        if loadj(root/"checksums.json").get("files")!=checksums(root):errors.append("recursive_checksums")
        cfg=loadj(root/"frozen_reconstruction_config.json"); lineage=loadj(root/"r13_frozen_lineage.json")
        expected={"signal":"gps_l1ca_code","fs_hz":25000000.0,"raw_format":"interleaved_signed_int16_iq","global_offset_samples":0,"nco_row":"previous","aux_row":"previous","remnant_sign":-1,"carrier_sign":-1,"replica_direction":"forward","prompt_row":"current","support_samples":25000,"candidate_string":CANDIDATE}
        if cfg!=expected:errors.append("frozen_config")
        if lineage.get("reference_metrics")!=R13 or lineage.get("metrics_tolerance")!=1e-6:errors.append("r13_metrics")
        source=ROOT/lineage.get("source_path",""); artifact=ROOT/lineage.get("artifact_path","")
        if not source.is_file() or digest(source)!=lineage.get("r13_source_sha256"):errors.append("r13_source_sha")
        if not (artifact/"checksums.json").is_file() or digest(artifact/"checksums.json")!=lineage.get("r13_artifact_checksums_sha256"):errors.append("r13_artifact_sha")
        rows=loadc(root/"per_block_scores.csv")
        if len(rows)!=969 or len({(r["channel"],r["prn"],r["tracker_row"]) for r in rows})!=len(rows):errors.append("row_identities")
        if {r["recording"] for r in rows}!={"cleanStatic"} or any("attack" in str(v).lower() for r in rows for v in r.values()):errors.append("clean_only")
        if {r["role"] for r in rows}!={"train","calibration","holdout"} or len({int(r["prn"]) for r in rows})!=8:errors.append("role_prn_schema")
        numeric=("center_magnitude","mat_prompt_magnitude","prompt_ratio","prompt_abs_relative_error","peak_delay_offset_chips","peak_doppler_offset_hz","peak_center_ratio")
        if any(not math.isfinite(float(r[k])) for r in rows for k in numeric):errors.append("nonfinite")
        actualp=prompt(rows); actuald=delay(rows); actualdop=doppler(rows)
        if not close(actualp,loadj(root/"prompt_reproduction_metrics.json")):errors.append("prompt_summary")
        if not close(actuald,loadj(root/"delay_recovery_metrics.json")):errors.append("delay_summary")
        if not close(actualdop,loadj(root/"doppler_1ms_metrics.json")):errors.append("doppler_summary")
        ag=loadc(root/"aggregation_metrics.csv"); pairs=loadc(root/"paired_improvement.csv"); boots=loadj(root/"bootstrap_results.json")
        if {int(r["L"]) for r in ag}!={1,5,10,20}:errors.append("aggregation_lengths")
        counts={int(r["L"]):int(r["block_count"]) for r in ag}
        if len(set(counts.values()))!=1:errors.append("common_anchor_counts")
        for L in (5,10,20):
            group=[r for r in pairs if int(r["L"])==L]
            if len(group)!=counts[L] or len({(r["channel"],r["prn"],r["anchor_tracker_row"],r["role"]) for r in group})!=len(group):errors.append(f"paired_{L}")
            by=defaultdict(list)
            for r in group:by[int(r["prn"])].append(float(r["difference"]))
            rng=np.random.default_rng(1401); vals=np.empty(10000); prns=sorted(by)
            for i in range(10000):vals[i]=np.mean([x for p in rng.choice(prns,len(prns),replace=True) for x in by[int(p)]])
            expected_boot={"seed":1401,"replicates":10000,"observed_difference":float(np.mean([float(r["difference"]) for r in group])),"ci95_low":float(np.quantile(vals,.025)),"ci95_high":float(np.quantile(vals,.975)),"sign_consistent":bool(np.quantile(vals,.025)>0)}
            if not close(expected_boot,boots[str(L)]):errors.append(f"bootstrap_{L}")
        pm=actualp; a3a=pm["pooled_spearman"]>=.999 and pm["median_prn_spearman"]>=.99 and pm["median_relative_error"]<=.001 and pm["p99_relative_error"]<=.01
        byprn=[]
        for p in sorted({int(r["prn"]) for r in rows}):byprn.append(delay([r for r in rows if int(r["prn"])==p]))
        byrole=[]
        for role in ("train","calibration","holdout"):byrole.append(delay([r for r in rows if r["role"]==role]))
        a3b=actuald["within_0_125_fraction"]>=.95 and actuald["boundary_fraction"]<=.01 and sum(x["within_0_125_fraction"]>=.95 for x in byprn)>=7 and all(x["within_0_125_fraction"]>=.95 for x in byrole)
        l20=next(r for r in ag if int(r["L"])==20); pair20=[r for r in pairs if int(r["L"])==20]
        pd={p:np.mean([float(r["difference"]) for r in pair20 if int(r["prn"])==p]) for p in {int(r["prn"]) for r in pair20}}; rd={p:np.mean([float(r["difference"]) for r in pair20 if r["role"]==p]) for p in {r["role"] for r in pair20}}
        a3c=float(l20["within_50_fraction"])>=.95 and float(l20["boundary_fraction"])<=.01 and boots["20"]["ci95_low"]>0 and sum(x>0 for x in pd.values())>=7 and len(pd)==8 and set(rd)=={"train","calibration","holdout"} and all(x>0 for x in rd.values())
        expected_gate=gate(not any(x.startswith("r13_") or x=="frozen_config" for x in errors),True,a3a,a3b,a3c)
        saved=loadj(root/"go_no_go.json")
        if saved.get("verdict") not in VERDICTS or saved!=expected_gate:errors.append("gate_verdict")
    except Exception as exc:errors.append(f"exception:{type(exc).__name__}:{exc}")
    return {"status":"PASS" if not errors else "FAIL","errors":errors}
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("artifact",type=Path);p.add_argument("--write-report",action="store_true");a=p.parse_args(argv);report=verify(a.artifact)
    if a.write_report:(a.artifact/"verification_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True));raise SystemExit(0 if report["status"]=="PASS" else 1)
if __name__=="__main__":main()
