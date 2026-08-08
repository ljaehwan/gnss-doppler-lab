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


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("artifact",type=Path);parser.add_argument("--checkpoint",choices=("2",),default="2")
    parser.add_argument("--write-report",action="store_true");args=parser.parse_args();ok,report=verify_checkpoint2(args.artifact)
    if args.write_report: (args.artifact/"verification_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True));raise SystemExit(0 if ok else 2)


if __name__ == "__main__": main()
