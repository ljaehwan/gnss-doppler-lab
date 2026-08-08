#!/usr/bin/env python3
"""Independent ACAF-NF Stage-1 R2 verifier (does not import the producer)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

CHECKPOINT4_REQUIRED = (
    "README.md", "go_no_go.json", "execution_validity.json", "foundation_evidence.json",
    "normal_split.json", "ensemble_h1.json", "thresholds.json", "normal_model_summary.json",
    "bootstrap_results.json", "b0_results.json", "scenario_metrics.csv", "phase_metrics.csv",
    "baseline_metrics.csv", "control_metrics.csv", "per_window_scores.csv",
    "secondary_component_metrics.csv", "positive_control_sweep.csv", "complex_awgn_controls.csv",
    "attack_tracker_manifest.json", "scenario_timeline.json", "source_binding.json",
    "execution_manifest.json", "checkpoint4_manifest.json", "checkpoint2_verification_report.json",
    "plots/README.md", "config.json", "checksums.json", "test_report.txt", "verification_report.json",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""): value.update(chunk)
    return value.hexdigest()


def verify_checkpoint2(root: Path) -> tuple[bool, dict]:
    errors = []
    required = ("continuous_tracker_alignment_audit.json", "full_cleanstatic_validation.json",
                "full_cleanstatic_validation_epochs.csv", "full_cleanstatic_caf_surfaces.npz",
                "full_cleanstatic_l20_windows.json", "r1_artifact_inventory.json", "checksums.json",
                "fresh_replay_binding.json", "fresh_continuous_tracker_manifest.json",
                "foundation_status.json", "checkpoint2_correction.json", "continuous_tracker_cleanStatic.csv")
    for name in required:
        if not (root / name).is_file(): errors.append(f"missing:{name}")
    if errors: return False, {"status": "FAIL", "checkpoint": 2, "errors": errors}
    alignment = json.loads((root / required[0]).read_text(encoding="utf-8"))
    if (alignment.get("selected_row_shift") != 0
            or alignment.get("selected_candidate") != "prompt_current_state_shift_+0"
            or alignment.get("selection_data") != "fresh_cleanStatic_replay_only"
            or alignment.get("attacks_examined_for_alignment")):
        errors.append("alignment_freeze")
    if not all(alignment.get("gates", {}).values()): errors.append("alignment_gates")
    report = json.loads((root / required[1]).read_text(encoding="utf-8"))
    with (root / required[2]).open(newline="", encoding="utf-8") as handle: epochs = list(csv.DictReader(handle))
    archive = np.load(root / required[3], allow_pickle=False)
    surfaces = archive["surfaces"]; delays = archive["delay_grid"]; dopplers = archive["doppler_grid_hz"]
    if surfaces.shape != (len(epochs), len(dopplers), len(delays)): errors.append("surface_shape")
    center=[]; prompt=[]; relative=[]; peaks=[]; boundary=[]; by_pair=defaultdict(lambda: ([], []))
    for index, row in enumerate(epochs):
        magnitude=np.abs(surfaces[index]); peak=np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
        c=float(magnitude[list(dopplers).index(0), list(delays).index(0)]); p=float(row["mat_prompt_magnitude"])
        center.append(c); prompt.append(p); relative.append(abs(c/p-1)); peaks.append(float(delays[peak[1]]))
        boundary.append(peak[0] in (0,len(dopplers)-1) or peak[1] in (0,len(delays)-1))
        pair=(int(row["channel"]),int(row["prn"])); by_pair[pair][0].append(c);by_pair[pair][1].append(p)
        if abs(c-float(row["center_magnitude"]))>1e-6: errors.append("center_metric")
        if hashlib.sha256(np.ascontiguousarray(surfaces[index]).view(np.uint8)).hexdigest()!=row["surface_sha256"]: errors.append("surface_hash")
    rho=float(spearmanr(center,prompt).statistic)
    median_rho=float(np.median([spearmanr(x,y).statistic for x,y in by_pair.values()]))
    windows=json.loads((root/required[4]).read_text(encoding="utf-8")); offsets=[]; window_boundary=[]
    for window in windows:
        indices=[int(x) for x in window["surface_indices"]]
        power=np.abs(surfaces[indices])**2; normalized=power/(np.sum(power,axis=(1,2),keepdims=True)+1e-15)
        aggregate=np.sqrt(np.mean(normalized,axis=0)); peak=np.unravel_index(int(np.argmax(aggregate)),aggregate.shape)
        offsets.append(float(dopplers[peak[0]]));window_boundary.append(peak[0] in (0,len(dopplers)-1) or peak[1] in (0,len(delays)-1))
    r14_path = root.parent / "acaf_nf_stage0_static_r14_doppler_validation" / "per_block_scores.csv"
    r14 = {}
    with r14_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["record_type"] == "epoch":
                r14[(int(row["channel"]), int(row["prn"]), int(row["support_start_sample"]))] = row
    r14_n=0; r14_max=0.0; r14_hash=True
    for row in epochs:
        frozen=r14.get((int(row["channel"]),int(row["prn"]),int(row["support_start_sample"])))
        if frozen is None: continue
        r14_n += 1
        for field in ("center_magnitude","peak_magnitude","peak_delay_offset_chips","peak_doppler_offset_hz"):
            r14_max=max(r14_max,abs(float(row[field])-float(frozen[field])))
        r14_hash = r14_hash and row["surface_sha256"] == frozen["surface_sha256"]
    tracker_manifest=json.loads((root/"fresh_continuous_tracker_manifest.json").read_text(encoding="utf-8"))
    binding=json.loads((root/"fresh_replay_binding.json").read_text(encoding="utf-8"))
    receiver_manifest_path=Path(binding["manifest_path"])
    receiver_manifest=json.loads(receiver_manifest_path.read_text(encoding="utf-8"))
    if digest(receiver_manifest_path)!=binding["manifest_sha256"]: errors.append("fresh_manifest_hash")
    if receiver_manifest_path.parent.name!="cleanStatic" or receiver_manifest_path.name!="manifest.json": errors.append("fresh_manifest_path")
    authenticated=True; exporter_rows=0
    for kind in ("mat","dat"):
        for name,expected in receiver_manifest["tracking"][f"{kind}_inventory"].items():
            path=receiver_manifest_path.parent/"raw"/name
            actual=digest(path); authenticated=authenticated and actual==expected
            item=binding["files"].get(f"raw/{name}",{})
            if item.get("sha256")!=actual or item.get("manifest_sha256")!=expected or item.get("sha256_match") is not True:
                errors.append(f"fresh_inventory:{name}")
            if kind=="mat":
                with h5py.File(path,"r") as handle: rows=int(handle["PRN"].size)
                exporter_rows += rows
                if item.get("exporter_rows")!=rows: errors.append(f"fresh_rows:{name}")
    if not authenticated or binding.get("status")!="PASS" or exporter_rows!=binding.get("exporter_rows"):
        errors.append("fresh_replay_authentication")
    raw_path=Path(binding["source_path"])
    if raw_path.stat().st_size!=binding["source_size_bytes"] or digest(raw_path)!=binding["source_sha256"]:
        errors.append("raw_source_authentication")
    tracker_csv=root/"continuous_tracker_cleanStatic.csv"
    csv_hash=digest(tracker_csv); tracker_rows=0; pairs=set(); valid_pairs=set(); stream_unique=True; l20_total=0
    tracker_state_semantics=True
    previous_pair=None; previous_start=None; run=0
    with tracker_csv.open(newline="",encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tracker_rows += 1; pair=(int(row["channel"]),int(row["prn"]));start=int(row["raw_start_sample"]);pairs.add(pair)
            tracker_state_semantics = tracker_state_semantics and int(row["state_mat_row"])==int(row["mat_row"])-1
            if pair!=previous_pair:
                run=1
            else:
                delta=start-int(previous_start)
                if delta<25000: stream_unique=False
                run=run+1 if delta==25000 else 1
            if run>=20: l20_total+=1;valid_pairs.add(pair)
            previous_pair=pair;previous_start=start
    if not tracker_state_semantics: errors.append("tracker_state_semantics")
    if (csv_hash!=tracker_manifest["csv_sha256"] or tracker_rows!=tracker_manifest["rows"]
            or tracker_manifest["csv_size_bytes"]!=tracker_csv.stat().st_size): errors.append("fresh_tracker_csv_binding")
    cadence=report["fresh_tracker_cadence"]
    if (tracker_rows!=cadence["rows"] or len(pairs)!=cadence["prn_channels"]
            or len(valid_pairs)!=cadence["valid_prn_channels"] or l20_total!=cadence["l20_total_windows"]
            or stream_unique!=cadence["unique_intervals"]): errors.append("fresh_tracker_cadence")
    gates={
        "raw_support_continuous_unique": stream_unique, "valid_prn_channels_ge_4": len(by_pair)>=4,
        "target_prn_channels_ge_8": len(by_pair)>=8, "l20_windows_ge_100": len(windows)>=100,
        "pooled_spearman_ge_0_999": rho>=.999, "median_prn_spearman_ge_0_99": median_rho>=.99,
        "prompt_p99_relative_error_le_0_01": float(np.quantile(relative,.99))<=.01,
        "delay_within_0_125_ge_0_95": float(np.mean(np.abs(peaks)<=.125))>=.95,
        "l20_doppler_within_50_ge_0_95": float(np.mean(np.abs(offsets)<=50))>=.95,
        "grid_boundary_le_0_01": float(np.mean(boundary))<=.01 and float(np.mean(window_boundary))<=.01,
        "r14_common_reproduced_1e_6": r14_n>0 and r14_max<=1e-6 and r14_hash,
    }
    if gates != report.get("gates"): errors.append("gate_recompute")
    r14_report=report.get("r14_common_epochs",{})
    if (r14_report.get("match_mode")!="raw_support" or r14_report.get("n")!=r14_n
            or abs(float(r14_report.get("max_numeric_delta",-1))-r14_max)>1e-12
            or r14_report.get("surface_sha256_all_match")!=r14_hash): errors.append("r14_common_recompute")
    expected_tracker_status="CONTINUOUS_TRACKER_VALID" if all(gates.values()) else "CONTINUOUS_TRACKER_INVALID"
    if report.get("status")!=expected_tracker_status: errors.append("tracker_verdict")
    clean_semantics=report.get("scenario_semantics_checks",{}).get("cleanStatic",{})
    attacks=report.get("scenario_semantics_checks",{}).get("attacks",{})
    if clean_semantics.get("status")!="PASS" or attacks.get("status")!="NOT_EVALUATED": errors.append("scenario_semantics")
    foundation=json.loads((root/"foundation_status.json").read_text(encoding="utf-8"))
    expected_foundation="FOUNDATION_PASS" if expected_tracker_status=="CONTINUOUS_TRACKER_VALID" else "FOUNDATION_INVALID"
    if (foundation.get("status")!=expected_foundation or foundation.get("checkpoint3_physics_authorized")!=(expected_foundation=="FOUNDATION_PASS")
            or foundation.get("attack_iq_scoring_performed") is not False): errors.append("foundation_verdict")
    correction=json.loads((root/"checkpoint2_correction.json").read_text(encoding="utf-8"))
    if correction.get("supersedes_commit")!="b5d53fa" or correction.get("attack_data_used_for_selection") is not False:
        errors.append("correction_lineage")
    checks=json.loads((root/"checksums.json").read_text(encoding="utf-8"))["files"]
    for name in required:
        if name == "checksums.json": continue
        actual = csv_hash if name == "continuous_tracker_cleanStatic.csv" else digest(root/name)
        if checks.get(name,{}).get("sha256")!=actual: errors.append(f"checksum:{name}")
    recomputed={"selected_epochs":len(epochs),"selected_prn_channels":len(by_pair),"l20_windows":len(windows),
                "pooled_spearman":rho,"median_prn_spearman":median_rho,"p99_relative_error":float(np.quantile(relative,.99)),
                "r14_common_epochs":r14_n,"r14_max_numeric_delta":r14_max,"r14_surface_hash_all_match":r14_hash,
                "fresh_tracker_rows":tracker_rows,"fresh_tracker_sha256":csv_hash,"fresh_exporter_rows":exporter_rows,
                "derived_tracker_status":expected_tracker_status,"derived_foundation_status":expected_foundation,"gates":gates}
    return not errors,{"schema":"acaf_nf_stage1_r2_verification.v1","status":"PASS" if not errors else "FAIL",
                       "checkpoint":2,"errors":sorted(set(errors)),"recomputed":recomputed}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_checkpoint4(root: Path) -> tuple[bool, dict]:
    errors: list[str] = []
    for name in CHECKPOINT4_REQUIRED:
        if not (root / name).is_file(): errors.append(f"missing:{name}")
    if errors:
        return False, {"schema":"acaf_nf_stage1_r2_verification.v1","status":"FAIL","checkpoint":4,"errors":errors}
    foundation=json.loads((root/"foundation_status.json").read_text(encoding="utf-8"))
    clean=json.loads((root/"full_cleanstatic_validation.json").read_text(encoding="utf-8"))
    tracker=json.loads((root/"fresh_continuous_tracker_manifest.json").read_text(encoding="utf-8"))
    cp2=json.loads((root/"checkpoint2_verification_report.json").read_text(encoding="utf-8"))
    evidence=json.loads((root/"foundation_evidence.json").read_text(encoding="utf-8"))
    if (foundation.get("status")!="FOUNDATION_INVALID" or foundation.get("checkpoint3_physics_authorized") is not False
            or foundation.get("attack_iq_scoring_performed") is not False): errors.append("foundation_status")
    if (clean.get("status")!="CONTINUOUS_TRACKER_INVALID" or all(clean.get("gates",{}).values())):
        errors.append("clean_foundation_not_invalid")
    if (cp2.get("status")!="PASS" or cp2.get("checkpoint")!=2
            or cp2.get("recomputed",{}).get("derived_tracker_status")!="CONTINUOUS_TRACKER_INVALID"
            or cp2.get("recomputed",{}).get("derived_foundation_status")!="FOUNDATION_INVALID"):
        errors.append("checkpoint2_independent_verification")
    failed=sorted(key for key,value in clean["gates"].items() if not value)
    if failed!=sorted(foundation.get("failed_clean_gates",[])): errors.append("failed_gate_lineage")
    expected_evidence={
        "rows":tracker["rows"],"sha256":tracker["csv_sha256"],"size_bytes":tracker["csv_size_bytes"],
        "exporter_rows":tracker["exporter_rows"],"status":tracker["status"],
    }
    if evidence.get("fresh_tracker")!=expected_evidence: errors.append("tracker_evidence")
    ce=evidence.get("clean_validation",{})
    for key in ("selected_epochs","selected_prn_channels","gates","prompt_reproduction","delay_recovery",
                "l20_doppler","grid_boundary_fraction","r14_common_epochs"):
        if ce.get(key)!=clean.get(key): errors.append(f"clean_evidence:{key}")
    if evidence.get("attack_iq_bytes_read_for_scoring")!=0 or evidence.get("attack_evidence")!="NOT_EVALUATED":
        errors.append("attack_evidence")
    go=json.loads((root/"go_no_go.json").read_text(encoding="utf-8"))
    if (go.get("verdict")!="FOUNDATION_INVALID" or go.get("physics_feasibility_status")!="NOT_EVALUATED"
            or go.get("paper_candidate_status")!="NOT_EVALUATED" or go.get("PHYSICS_FEASIBILITY_GO") is not False
            or go.get("PAPER_CANDIDATE_GO") is not False or go.get("stage2_justified") is not False
            or go.get("B0",{}).get("status")!="NOT_EVALUATED"): errors.append("go_no_go_semantics")
    execution=json.loads((root/"execution_validity.json").read_text(encoding="utf-8"))
    false_fields=("checkpoint3_physics_executed","caf_attack_scoring_executed","h0_h1_fit_executed",
                  "threshold_calibration_executed","bootstrap_executed","B0_evaluator_executed")
    if (execution.get("status")!="FOUNDATION_INVALID" or execution.get("science_status")!="NOT_EVALUATED"
            or any(execution.get(key) is not False for key in false_fields)
            or execution.get("attack_rows_in_fit_or_calibration")!=0
            or execution.get("attack_iq_bytes_read_for_scoring")!=0
            or execution.get("science_csv_semantics")!="explicit_NOT_EVALUATED_rows"
            or execution.get("physics_plots",{}).get("count")!=0): errors.append("execution_semantics")
    split=json.loads((root/"normal_split.json").read_text(encoding="utf-8"))
    fractions=[item.get("fraction") for item in split.get("fractions",[])]
    roles=[item.get("role") for item in split.get("fractions",[])]
    if (split.get("status")!="NOT_EVALUATED" or split.get("applied") is not False
            or split.get("chronological") is not True or split.get("no_overlap") is not True
            or fractions!=[.4,.15,.25,.2] or roles!=["train","h1_selection","calibration","holdout"]
            or split.get("boundaries") is not None): errors.append("normal_split_semantics")
    unavailable_json=("ensemble_h1.json","thresholds.json","normal_model_summary.json","bootstrap_results.json","b0_results.json")
    for name in unavailable_json:
        value=json.loads((root/name).read_text(encoding="utf-8"))
        if value.get("status")!="NOT_EVALUATED": errors.append(f"not_evaluated:{name}")
    if json.loads((root/"b0_results.json").read_text(encoding="utf-8")).get("historic_scores_reused") is not False:
        errors.append("b0_reuse")
    csv_contracts={
        "scenario_metrics.csv":5,"phase_metrics.csv":17,"baseline_metrics.csv":5,"control_metrics.csv":4,
        "per_window_scores.csv":17,"secondary_component_metrics.csv":5,"positive_control_sweep.csv":1,
        "complex_awgn_controls.csv":5,
    }
    rows_by_file={}
    for name,count in csv_contracts.items():
        rows=_csv_rows(root/name);rows_by_file[name]=len(rows)
        if len(rows)!=count or any(row.get("status")!="NOT_EVALUATED" for row in rows):
            errors.append(f"csv_status:{name}")
        for row in rows:
            for field,value in row.items():
                if field not in {"scenario","phase","role","status","reason","baseline","lineage","control","kind","noise_reference"} and value:
                    errors.append(f"unavailable_numeric_value:{name}:{field}")
    attack=json.loads((root/"attack_tracker_manifest.json").read_text(encoding="utf-8"))
    if (attack.get("status")!="NOT_EVALUATED" or attack.get("attack_replays_read") is not False
            or attack.get("attack_iq_bytes_read_for_scoring")!=0
            or attack.get("mapping_selected_from_attack_data") is not False
            or any(value.get("status")!="NOT_EVALUATED" for value in attack.get("scenarios",{}).values())):
        errors.append("attack_manifest")
    source=json.loads((root/"source_binding.json").read_text(encoding="utf-8"))
    if (source.get("status")!="FOUNDATION_INVALID"
            or any(value.get("status")!="NOT_EVALUATED" for value in source.get("attacks",{}).values())):
        errors.append("source_binding")
    manifest=json.loads((root/"checkpoint4_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("checkpoint")!=4 or manifest.get("status")!="FOUNDATION_INVALID"
            or manifest.get("science_status")!="NOT_EVALUATED"
            or manifest.get("required_artifacts")!=list(CHECKPOINT4_REQUIRED)
            or manifest.get("attack_results_used_for_selection") is not False): errors.append("checkpoint4_manifest")
    config=json.loads((root/"config.json").read_text(encoding="utf-8"))
    if (config.get("checkpoint")!=4 or config.get("status")!="FOUNDATION_INVALID"
            or config.get("science_status")!="NOT_EVALUATED" or config.get("physics_executed") is not False
            or config.get("attack_data_read") is not False or config.get("B0")!="NOT_EVALUATED"):
        errors.append("config_semantics")
    readme=(root/"README.md").read_text(encoding="utf-8")
    if not all(text in readme for text in ("FOUNDATION_INVALID","NOT_EVALUATED","No attack replay","prohibited")):
        errors.append("readme_semantics")
    checks=json.loads((root/"checksums.json").read_text(encoding="utf-8"))["files"]
    for name in CHECKPOINT4_REQUIRED:
        if name in {"checksums.json","verification_report.json"}: continue
        if checks.get(name,{}).get("sha256")!=digest(root/name): errors.append(f"checksum:{name}")
    recomputed={
        "derived_tracker_status":"CONTINUOUS_TRACKER_INVALID","derived_foundation_status":"FOUNDATION_INVALID",
        "science_status":"NOT_EVALUATED","failed_clean_gates":failed,
        "fresh_tracker_rows":tracker["rows"],"fresh_tracker_sha256":tracker["csv_sha256"],
        "attack_iq_bytes_read_for_scoring":0,"checkpoint3_physics_executed":False,
        "csv_rows":rows_by_file,"required_artifact_count":len(CHECKPOINT4_REQUIRED),
    }
    return not errors,{"schema":"acaf_nf_stage1_r2_verification.v1","status":"PASS" if not errors else "FAIL",
                       "checkpoint":4,"errors":sorted(set(errors)),"recomputed":recomputed}


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("artifact",type=Path);parser.add_argument("--checkpoint",choices=("2","4"),default="2")
    parser.add_argument("--write-report",action="store_true");args=parser.parse_args()
    ok,report=verify_checkpoint2(args.artifact) if args.checkpoint=="2" else verify_checkpoint4(args.artifact)
    if args.write_report: (args.artifact/"verification_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True));raise SystemExit(0 if ok else 2)


if __name__ == "__main__": main()
