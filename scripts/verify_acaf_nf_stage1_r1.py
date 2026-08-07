#!/usr/bin/env python3
"""Independent verifier for Stage-1 R1 checkpoint artifacts.

This file deliberately does not import the Stage-1 producer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import spearmanr

SUPPORT = 25_000
DAT_BYTES = 148
DAT_STAMP_OFFSET = 80


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_checksums(root: Path, errors: list[str]) -> None:
    path = root / "checksums.json"
    if not path.is_file():
        errors.append("missing_file:checksums.json")
        return
    document = json.loads(path.read_text(encoding="utf-8"))
    entries = document.get("files", {})
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name not in {"checksums.json", "verification_report.json"}}
    if set(entries) != actual:
        errors.append("checksum_inventory_mismatch")
    for name, metadata in entries.items():
        target = root / name
        if not target.is_file() or sha256(target) != metadata.get("sha256") or target.stat().st_size != metadata.get("size_bytes"):
            errors.append(f"checksum_mismatch:{name}")


def verify_a(root: Path) -> tuple[bool, list[str], dict[str, object]]:
    fixed_cmd = ("PYTHONPATH=src python3 scripts/build_acaf_nf_continuous_tracker.py "
                 "--checkpoint A --source-binding configs/acaf_nf_stage1_source_binding.json "
                 "--output artifacts/acaf_nf_stage1_r1_continuous_tracker "
                 "--scenario cleanStatic --scenario ds3 --scenario ds4 --scenario ds7 --scenario ds8")
    errors: list[str] = []
    required = ["tracker_cadence_audit.json", "tracker_cadence_by_channel.csv", "checksums.json", "execution_manifest.json"]
    for name in required:
        if not (root / name).is_file(): errors.append(f"missing_file:{name}")
    audit = json.loads((root / "tracker_cadence_audit.json").read_text(encoding="utf-8")) if (root / "tracker_cadence_audit.json").is_file() else {}
    if audit.get("schema") != "acaf_nf_stage1_continuous_tracker_cadence.v1": errors.append("unexpected_schema")
    if (root / "execution_manifest.json").is_file():
        manifest = json.loads((root / "execution_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("command") != fixed_cmd: errors.append("execution_manifest_command_mismatch")
    for entry in ("tracker_cadence_audit.json", "tracker_cadence_by_channel.csv"):
        checks = json.loads((root / "checksums.json").read_text(encoding="utf-8")).get("files", {}) if (root / "checksums.json").is_file() else {}
        if entry not in checks or sha256(root / entry) != checks[entry].get("sha256"): errors.append(f"checksum_mismatch:{entry}")
    report = {"status": "PASS" if not errors else "FAIL", "errors": errors, "required_files": required,
              "artifact_path": str(root), "artifact_schema": audit.get("schema"), "checkpoint": "A"}
    return not errors, errors, report


def _vector(handle: h5py.File, name: str) -> np.ndarray:
    return np.asarray(handle[name]).reshape(-1)


def _verify_source_rows(root: Path, errors: list[str], scenario: str = "cleanStatic") -> tuple[int, int]:
    binding = json.loads((root / "source_binding.json").read_text(encoding="utf-8"))
    if "scenarios" in binding: binding = binding["scenarios"][scenario]
    config_path = Path(binding["source_binding_config"])
    if sha256(config_path) != binding["source_binding_config_sha256"]: errors.append("source_binding_config_hash")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))["scenarios"][scenario]
    for key, expected in (("raw_sha256", cfg["raw_sha256"]), ("receiver_config_sha256", cfg["receiver_config_sha256"]),
                          ("receiver_manifest_sha256", cfg["manifest_sha256"])):
        if binding.get(key) != expected: errors.append(f"binding_value:{key}")
    external = [("receiver_config_path", "receiver_config_sha256"), ("receiver_manifest_path", "receiver_manifest_sha256")]
    if binding.get("gnss_sdr_build_sha256"): external.append(("gnss_sdr_executable", "gnss_sdr_build_sha256"))
    for path_key, hash_key in external:
        path = Path(binding[path_key])
        if not path.is_file() or sha256(path) != binding[hash_key]: errors.append(f"external_hash:{path_key}")
    raw = Path(binding["raw_path"])
    if not raw.is_file() or raw.stat().st_size != binding["raw_size_bytes"]: errors.append("raw_size_binding")

    with (root / f"continuous_tracker_{scenario}.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mats: dict[str, dict[str, np.ndarray]] = {}
    pairs: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in rows:
        mat_path = Path(row["source_mat"])
        if str(mat_path) not in mats:
            if mat_path.name not in cfg["mat_inventory"] or sha256(mat_path) != cfg["mat_inventory"][mat_path.name]:
                errors.append(f"mat_hash:{mat_path.name}"); continue
            with h5py.File(mat_path, "r") as handle:
                mats[str(mat_path)] = {name: _vector(handle, name) for name in
                    ("PRN_start_sample_count", "PRN", "carrier_doppler_hz", "code_freq_chips", "aux1",
                     "Prompt_I", "Prompt_Q", "CN0_SNV_dB_Hz", "carrier_lock_test")}
            values = mats[str(mat_path)]
            dat = mat_path.with_suffix(".dat")
            if not dat.is_file() or dat.stat().st_size != len(values["PRN"]) * DAT_BYTES:
                errors.append(f"dat_size:{mat_path.name}")
            else:
                raw_dat = np.memmap(dat, dtype=np.uint8, mode="r")
                stamps = np.ndarray((len(values["PRN"]),), dtype="<u8", buffer=raw_dat,
                                    offset=DAT_STAMP_OFFSET, strides=(DAT_BYTES,))
                if not np.array_equal(stamps.astype(np.int64), values["PRN_start_sample_count"].astype(np.int64)):
                    errors.append(f"dat_stamp:{mat_path.name}")
        values = mats.get(str(mat_path))
        if values is None: continue
        i = int(row["mat_row"]); state = int(row["state_mat_row"]); prn = int(row["prn"])
        start = int(row["raw_start_sample"]); end = int(row["raw_end_sample"])
        if state != i - 1 or i <= 0 or i >= len(values["PRN"]) - 1: errors.append("row_index_contract"); continue
        if not (int(values["PRN"][i-1]) == int(values["PRN"][i]) == int(values["PRN"][i+1]) == prn): errors.append("same_prn_triple")
        if start != int(values["PRN_start_sample_count"][state]) or int(values["PRN_start_sample_count"][i]) - start != SUPPORT:
            errors.append("support_stamp_contract")
        if end - start != SUPPORT or int(row["sample_count"]) != SUPPORT: errors.append("support_length_contract")
        comparisons = (("code_freq_chips", state), ("carrier_doppler_hz", state), ("aux1", state),
                       ("Prompt_I", i), ("Prompt_Q", i), ("CN0_SNV_dB_Hz", i), ("carrier_lock_test", i))
        csv_names = {"Prompt_I":"prompt_i", "Prompt_Q":"prompt_q", "CN0_SNV_dB_Hz":"cn0_db_hz"}
        for name, index in comparisons:
            field = csv_names.get(name, name)
            if float(row[field]) != float(values[name][index]): errors.append(f"state_value:{field}")
        if np.min(values["CN0_SNV_dB_Hz"][i-1:i+2]) < 28 or np.min(values["carrier_lock_test"][i-1:i+2]) < .85:
            errors.append("quality_gate")
        pairs[(int(row["channel"]), prn)].append(start)
    l20 = 0
    for starts in pairs.values():
        starts.sort()
        delta = np.diff(starts)
        if np.any(delta < SUPPORT) or len(starts) != len(set(starts)): errors.append("overlap_or_duplicate")
        run = 1
        for value in delta:
            run = run + 1 if value == SUPPORT else 1
            if run >= 20: l20 += 1
    return len(pairs), l20


def _rho(x: list[float], y: list[float]) -> float:
    if len(x) < 3: return 0.0
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else 0.0


def _verify_reconstruction(root: Path, errors: list[str]) -> dict[str, Any]:
    report = json.loads((root / "cleanstatic_validation.json").read_text(encoding="utf-8"))
    with (root / "cleanstatic_validation_epochs.csv").open(newline="", encoding="utf-8") as handle:
        epochs = list(csv.DictReader(handle))
    archive = np.load(root / "cleanstatic_caf_surfaces.npz", allow_pickle=False)
    surfaces = archive["surfaces"]; delays = archive["delay_grid"]; dopplers = archive["doppler_grid_hz"]
    if surfaces.shape != (len(epochs), len(dopplers), len(delays)): errors.append("surface_shape")
    centers=[]; prompts=[]; rel=[]; by_prn: dict[int, tuple[list[float], list[float]]] = defaultdict(lambda:([],[]))
    grid_boundary=[]
    for index, row in enumerate(epochs):
        magnitude=np.abs(surfaces[index]); peak=np.unravel_index(int(np.argmax(magnitude)),magnitude.shape)
        center=float(magnitude[list(dopplers).index(0),list(delays).index(0)]); prompt=float(row["mat_prompt_magnitude"])
        centers.append(center); prompts.append(prompt); rel.append(abs(center/prompt-1))
        by_prn[int(row["prn"])][0].append(center); by_prn[int(row["prn"])][1].append(prompt)
        boundary=peak[0] in (0,len(dopplers)-1) or peak[1] in (0,len(delays)-1);grid_boundary.append(boundary)
        expected=(float(delays[peak[1]]),float(dopplers[peak[0]]),float(magnitude[peak]),center)
        actual=tuple(float(row[x]) for x in ("peak_delay_offset_chips","peak_doppler_offset_hz","peak_magnitude","center_magnitude"))
        if any(abs(a-b)>1e-6 for a,b in zip(expected,actual)): errors.append("epoch_surface_metric")
        if hashlib.sha256(np.ascontiguousarray(surfaces[index]).view(np.uint8)).hexdigest()!=row["surface_sha256"]: errors.append("surface_hash")
    prompt={"n":len(epochs),"pooled_spearman":_rho(centers,prompts),
            "median_prn_spearman":float(np.median([_rho(x,y) for x,y in by_prn.values()])),
            "median_relative_error":float(np.median(rel)),"p95_relative_error":float(np.quantile(rel,.95)),
            "p99_relative_error":float(np.quantile(rel,.99)),"max_relative_error":float(np.max(rel))}
    delay_values=np.asarray([float(r["peak_delay_offset_chips"]) for r in epochs])
    delay={"n":len(epochs),"exact_center_fraction":float(np.mean(delay_values==0)),
           "within_0_125_fraction":float(np.mean(np.abs(delay_values)<=.125)),
           "boundary_fraction":float(np.mean(np.abs(delay_values)>=1)),
           "histogram":{str(x):int(np.sum(delay_values==x)) for x in sorted(set(delay_values))}}
    windows=json.loads((root/"cleanstatic_l20_windows.json").read_text(encoding="utf-8"));offsets=[];boundaries=[]
    for window in windows:
        indices=[int(x) for x in window["surface_indices"]]
        if len(indices)!=20: errors.append("l20_length");continue
        block=[epochs[i] for i in indices]; starts=[int(x["support_start_sample"]) for x in block]
        if len({(x["channel"],x["prn"]) for x in block})!=1 or any(b-a!=SUPPORT for a,b in zip(starts,starts[1:])):
            errors.append("l20_lineage")
        power=np.abs(surfaces[indices])**2; normalized=power/(np.sum(power,axis=(1,2),keepdims=True)+1e-15)
        aggregate=np.sqrt(np.mean(normalized,axis=0));peak=np.unravel_index(int(np.argmax(aggregate)),aggregate.shape)
        offset=float(dopplers[peak[0]]);boundary=peak[0] in (0,len(dopplers)-1) or peak[1] in (0,len(delays)-1)
        offsets.append(offset);boundaries.append(boundary)
        if offset!=float(window["peak_doppler_offset_hz"]) or float(delays[peak[1]])!=float(window["peak_delay_offset_chips"]): errors.append("l20_metric")
    l20={"n":len(windows),"within_50_fraction":float(np.mean(np.abs(offsets)<=50)),"boundary_fraction":float(np.mean(boundaries))}
    for name, actual in (("prompt_reproduction",prompt),("delay_recovery",delay),("l20_doppler",l20)):
        expected=report[name]
        for key,value in actual.items():
            if isinstance(value,dict):
                if value!=expected.get(key): errors.append(f"metric:{name}:{key}")
            elif abs(float(value)-float(expected.get(key,float("nan"))))>1e-12: errors.append(f"metric:{name}:{key}")
    gates={"raw_support_continuous_unique":True,
           "valid_prn_channels_ge_4":len(by_prn)>=4,"target_prn_channels_ge_8":len(by_prn)>=8,
           "l20_windows_ge_100":len(windows)>=100,"pooled_spearman_ge_0_999":prompt["pooled_spearman"]>=.999,
           "median_prn_spearman_ge_0_99":prompt["median_prn_spearman"]>=.99,
           "prompt_p99_relative_error_le_0_01":prompt["p99_relative_error"]<=.01,
           "delay_within_0_125_ge_0_95":delay["within_0_125_fraction"]>=.95,
           "l20_doppler_within_50_ge_0_95":l20["within_50_fraction"]>=.95,
           "grid_boundary_le_0_01":float(np.mean(grid_boundary))<=.01 and l20["boundary_fraction"]<=.01,
           "r14_common_reproduced_1e_6":bool(report["gates"]["r14_common_reproduced_1e_6"])}
    if gates!=report["gates"] or not all(gates.values()) or report["status"]!="CONTINUOUS_TRACKER_VALID": errors.append("reconstruction_gates")
    return {"prompt_reproduction":prompt,"delay_recovery":delay,"l20_doppler":l20,"gates":gates}


def verify_b(root: Path) -> tuple[bool,list[str],dict[str,object]]:
    errors: list[str]=[]
    required=["continuous_tracker_cleanStatic.csv","continuous_tracker_manifest.json","cleanstatic_validation.json",
              "cleanstatic_validation_epochs.csv","cleanstatic_caf_surfaces.npz","cleanstatic_l20_windows.json",
              "source_binding.json","execution_manifest_checkpoint_b.json","checksums.json"]
    for name in required:
        if not (root/name).is_file(): errors.append(f"missing_file:{name}")
    if errors:
        return False,errors,{"status":"FAIL","errors":errors,"checkpoint":"B","artifact_path":str(root)}
    verify_checksums(root,errors)
    pairs,l20=_verify_source_rows(root,errors)
    metrics=_verify_reconstruction(root,errors)
    manifest=json.loads((root/"continuous_tracker_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("validation",{}).get("status")!="CONTINUOUS_TRACKER_VALID": errors.append("manifest_status")
    if manifest.get("validation",{}).get("valid_prn_channels",0)<4 or manifest.get("validation",{}).get("l20_total_windows",0)<100:
        errors.append("manifest_support_gate")
    report={"status":"PASS" if not errors else "FAIL","errors":errors,"checkpoint":"B","artifact_path":str(root),
            "source_prn_channels":pairs,"source_l20_windows":l20,"recomputed":metrics}
    return not errors,errors,report


def _pointer(document: Any, pointer: str | None) -> Any:
    if pointer is None or not pointer.startswith("/"): return None
    value=document
    for part in pointer[1:].split("/"):
        if not isinstance(value,dict) or part not in value: return None
        value=value[part]
    return value


def verify_c(root: Path) -> tuple[bool,list[str],dict[str,object]]:
    errors: list[str]=[]; scenarios=("ds3","ds4","ds7","ds8")
    required=[*(f"continuous_tracker_{s}.csv" for s in scenarios),"attack_tracker_manifest.json","scenario_timeline.json",
              "source_binding.json","execution_manifest_checkpoint_c.json","checksums.json"]
    for name in required:
        if not (root/name).is_file(): errors.append(f"missing_file:{name}")
    if errors: return False,errors,{"status":"FAIL","errors":errors,"checkpoint":"C","artifact_path":str(root)}
    verify_checksums(root,errors)
    attack=json.loads((root/"attack_tracker_manifest.json").read_text(encoding="utf-8"))
    timeline=json.loads((root/"scenario_timeline.json").read_text(encoding="utf-8"))
    source=json.loads((root/"source_binding.json").read_text(encoding="utf-8"))
    recomputed={}
    for scenario in scenarios:
        pairs,l20=_verify_source_rows(root,errors,scenario)
        binding=source["scenarios"][scenario];manifest=json.loads(Path(binding["receiver_manifest_path"]).read_text(encoding="utf-8"))
        bound=_pointer(manifest,binding["manifest_pointers"].get("raw_sha256"))==binding["raw_sha256"]
        expected_status="INVALID_RECORD_ALIGNMENT" if scenario=="ds4" else "PASS"
        if binding["status"]!=expected_status or (scenario=="ds4" and bound) or (scenario!="ds4" and not bound):
            errors.append(f"binding_status:{scenario}")
        with (root/f"continuous_tracker_{scenario}.csv").open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle))
        coverage={}
        for phase,limits in timeline[scenario]["phases"].items():
            begin=int(limits["start_sample"]);end=int(limits["end_sample_exclusive"])
            selected=[r for r in rows if int(r["raw_start_sample"])>=begin and int(r["raw_end_sample"])<=end]
            groups: dict[tuple[int,int],list[int]]=defaultdict(list)
            for row in selected: groups[(int(row["channel"]),int(row["prn"]))].append(int(row["raw_start_sample"]))
            windows=0;valid_pairs=0
            for starts in groups.values():
                starts.sort();run=1;pair_windows=0
                for delta in np.diff(starts):
                    run=run+1 if delta==SUPPORT else 1
                    if run>=20: windows+=1;pair_windows+=1
                valid_pairs+=pair_windows>0
            coverage[phase]={"start_sample":begin,"end_sample_exclusive":end,"rows":len(selected),
                             "l20_windows":windows,"l20_prn_channels":valid_pairs}
        if coverage!=attack["scenarios"][scenario]["phase_coverage"]: errors.append(f"phase_coverage:{scenario}")
        recomputed[scenario]={"prn_channels":pairs,"l20_windows":l20,"binding":expected_status,"phase_coverage":coverage}
    expected="CHECKPOINT_C_COMPLETE_WITH_DS4_FAIL_CLOSED"
    if attack.get("status")!=expected or not attack.get("primary_scenarios_valid") or not attack.get("ds4_fail_closed"):
        errors.append("checkpoint_c_status")
    report={"status":"PASS" if not errors else "FAIL","errors":errors,"checkpoint":"C","artifact_path":str(root),
            "scientific_status":expected,"recomputed":recomputed}
    return not errors,errors,report


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("artifact",type=Path);parser.add_argument("--checkpoint",choices=("A","B","C"),default="A")
    parser.add_argument("--write-report",action="store_true");args=parser.parse_args()
    ok,_,report=(verify_a(args.artifact) if args.checkpoint=="A" else verify_b(args.artifact)
                 if args.checkpoint=="B" else verify_c(args.artifact))
    if args.write_report:
        (args.artifact/"verification_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True));raise SystemExit(0 if ok else 2)


if __name__=="__main__": main()
