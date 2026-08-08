#!/usr/bin/env python3
"""Independent ACAF-NF Stage-1 R2 verifier (does not import the producer)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

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
                "full_cleanstatic_l20_windows.json", "r1_artifact_inventory.json", "checksums.json")
    for name in required:
        if not (root / name).is_file(): errors.append(f"missing:{name}")
    if errors: return False, {"status": "FAIL", "checkpoint": 2, "errors": errors}
    alignment = json.loads((root / required[0]).read_text(encoding="utf-8"))
    if alignment.get("selected_row_shift") != 0 or alignment.get("selection_data") != "cleanStatic_only" or alignment.get("attacks_examined_for_alignment"):
        errors.append("alignment_freeze")
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
    gates={
        "raw_support_continuous_unique": True, "valid_prn_channels_ge_4": len(by_pair)>=4,
        "target_prn_channels_ge_8": len(by_pair)>=8, "l20_windows_ge_100": len(windows)>=100,
        "pooled_spearman_ge_0_999": rho>=.999, "median_prn_spearman_ge_0_99": median_rho>=.99,
        "prompt_p99_relative_error_le_0_01": float(np.quantile(relative,.99))<=.01,
        "delay_within_0_125_ge_0_95": float(np.mean(np.abs(peaks)<=.125))>=.95,
        "l20_doppler_within_50_ge_0_95": float(np.mean(np.abs(offsets)<=50))>=.95,
        "grid_boundary_le_0_01": float(np.mean(boundary))<=.01 and float(np.mean(window_boundary))<=.01,
        "r14_common_reproduced_1e_6": report["r14_common_epochs"]["n"]>0 and report["r14_common_epochs"]["max_numeric_delta"]<=1e-6 and report["r14_common_epochs"]["surface_sha256_all_match"],
    }
    if gates != report.get("gates") or not all(gates.values()): errors.append("gate_recompute")
    if any(x.get("status")!="PASS" for x in report.get("scenario_semantics_checks",{}).values()): errors.append("scenario_semantics")
    checks=json.loads((root/"checksums.json").read_text(encoding="utf-8"))["files"]
    for name in required[:-1]:
        if checks.get(name,{}).get("sha256")!=digest(root/name): errors.append(f"checksum:{name}")
    recomputed={"selected_epochs":len(epochs),"selected_prn_channels":len(by_pair),"l20_windows":len(windows),
                "pooled_spearman":rho,"median_prn_spearman":median_rho,"p99_relative_error":float(np.quantile(relative,.99)),"gates":gates}
    return not errors,{"schema":"acaf_nf_stage1_r2_verification.v1","status":"PASS" if not errors else "FAIL",
                       "checkpoint":2,"errors":sorted(set(errors)),"recomputed":recomputed}


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("artifact",type=Path);parser.add_argument("--checkpoint",choices=("2",),default="2")
    parser.add_argument("--write-report",action="store_true");args=parser.parse_args();ok,report=verify_checkpoint2(args.artifact)
    if args.write_report: (args.artifact/"verification_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True));raise SystemExit(0 if ok else 2)


if __name__ == "__main__": main()
