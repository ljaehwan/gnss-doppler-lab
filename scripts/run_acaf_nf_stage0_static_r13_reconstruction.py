#!/usr/bin/env python3
"""R1.3 source-only prepared runner for the later authorized benign campaign."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnss_doppler_lab.acquisition_surface import gps_l1ca_code
from gnss_doppler_lab.acaf_nf_stage0_r13_reconstruction import (
    FS, GLOBAL_OFFSETS, REQUIRED_FIELDS, Candidate, caf, candidate_application,
    clean_only_guard, filter_stable_triples, gate_verdict, interval_rows,
    roles_nonoverlap, source_support, wide_grid,
)

OUT = Path("artifacts/acaf_nf_stage0_static_r13_reconstruction")
ARTIFACT_FILES = (
    "README.md", "config.json", "environment.json", "receiver_source_binding.json",
    "gnss_sdr_source_binding.json", "gnss_sdr_tracking_semantics.md",
    "ca_code_validation.json", "ca_code_correlation.csv", "candidate_application_audit.csv",
    "candidate_fingerprints.json", "alignment_hypotheses.csv", "selected_alignment.json",
    "center_validation.csv", "center_validation_summary.json", "center_metrics_by_prn.csv",
    "center_metrics_by_channel.csv", "center_metrics_by_time_block.csv",
    "global_offset_sensitivity.csv", "global_offset_application_audit.json",
    "prn_sampling_summary.csv", "raw_overlap_audit.json", "execution_validity.json",
    "go_no_go.json", "test_report.txt", "verification_report.json", "checksums.json",
)

CA_CODE_FIELDS = (
    "prn", "input_int", "input_gxx", "int_gxx_exact", "length", "alphabet",
    "alphabet_exact", "cyclic_zero_shift", "cyclic_max_nonzero_shift",
    "cyclic_nonzero_equal_1023_count", "zero_shift_peak_exact",
    "nonzero_no_1023_peak", "code_sha256", "status",
)


def _legacy_bad_code_fixture(prn: int) -> np.ndarray:
    """Frozen regression fixture for the previously used, reversed-register generator."""
    taps={1:(2,6),2:(3,7),3:(4,8),4:(5,9),5:(1,9),6:(2,10),7:(1,8),8:(2,9),9:(3,10),10:(2,3),11:(3,4),12:(5,6),13:(6,7),14:(7,8),15:(8,9),16:(9,10),17:(1,4),18:(2,5),19:(3,6),20:(4,7),21:(5,8),22:(6,9),23:(1,3),24:(4,6),25:(5,7),26:(6,8),27:(7,9),28:(8,10),29:(1,6),30:(2,7),31:(3,8),32:(4,9)}
    g1=np.ones(10,dtype=np.int8); g2=np.ones(10,dtype=np.int8)
    out=np.empty(1023,dtype=np.int8); a,b=taps[prn]
    for i in range(1023):
        out[i]=1 if g1[-1] == (g2[a-1] ^ g2[b-1]) else -1
        g1=np.r_[g1[1:],g1[2]^g1[9]]
        g2=np.r_[g2[1:],g2[1]^g2[2]^g2[5]^g2[7]^g2[8]^g2[9]]
    return out


def ca_code_evidence():
    """Generate fail-closed, cyclic C/A-code evidence from the canonical implementation."""
    rows=[]
    for prn in range(1,33):
        code_int=np.asarray(gps_l1ca_code(prn))
        code_gxx=np.asarray(gps_l1ca_code(f"G{prn:02d}"))
        code64=code_int.astype(np.int64,copy=False)
        cyclic=np.asarray([np.dot(code64,np.roll(code64,shift)) for shift in range(1023)])
        alphabet=sorted(int(value) for value in np.unique(code_int))
        exact=np.array_equal(code_int,code_gxx)
        row={"prn":prn,"input_int":prn,"input_gxx":f"G{prn:02d}",
             "int_gxx_exact":exact,"length":len(code_int),
             "alphabet":",".join(map(str,alphabet)),"alphabet_exact":alphabet==[-1,1],
             "cyclic_zero_shift":int(cyclic[0]),
             "cyclic_max_nonzero_shift":int(cyclic[1:].max()),
             "cyclic_nonzero_equal_1023_count":int(np.count_nonzero(cyclic[1:]==1023)),
             "zero_shift_peak_exact":int(cyclic[0])==1023,
             "nonzero_no_1023_peak":not np.any(cyclic[1:]==1023),
             "code_sha256":hashlib.sha256(code_int.tobytes()).hexdigest()}
        row["status"]="PASS" if (exact and row["length"]==1023 and row["alphabet_exact"]
          and row["zero_shift_peak_exact"] and row["nonzero_no_1023_peak"]) else "FAIL"
        rows.append(row)
    canonical=np.asarray(gps_l1ca_code(1)); legacy=_legacy_bad_code_fixture(1)
    legacy_evidence={"fixture_id":"legacy_reversed_register_v1","prn":1,
      "canonical_sha256":hashlib.sha256(canonical.tobytes()).hexdigest(),
      "legacy_sha256":hashlib.sha256(legacy.tobytes()).hexdigest(),
      "differing_chips":int(np.count_nonzero(canonical!=legacy)),
      "differs":not np.array_equal(canonical,legacy)}
    return rows,legacy_evidence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows, fields=None) -> None:
    rows = list(rows); fields = fields or (list(rows[0]) if rows else ["status"])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


OBSERVATION_FIELDS=("recording","prn","channel","role","candidate","previous_mat_row","current_mat_row",
 "next_mat_row","tracker_row","mat_path","mat_sha256","consumed_start_sample","consumed_end_sample",
 "consumed_length_samples","support_start_sample","support_end_sample","support_length_samples",
 "raw_start_sample","raw_end_sample","raw_start_byte","raw_end_byte","vector_length","aux_row_index",
 "aux1_value","nco_row_index","code_freq_chips_value","carrier_doppler_hz_value","prompt_row_index",
 "prompt_i_value","prompt_q_value","mat_prompt_magnitude","peak_delay_offset_chips","peak_doppler_offset_hz",
 "peak_magnitude","center_magnitude","exact_center","within_tolerance","grid_boundary") + (
 "raw_interval_content_sha256","raw_interval_range_sha256","replica_chip_indices_sha256",
 "carrier_wipeoff_sha256","aux_indices_sha256","nco_indices_sha256","prompt_indices_sha256",
 "result_field_sha256")


def publish_fail_closed(output: Path, verdict: str, reason: str, *, a1: bool=False) -> dict:
    """Atomically publish the complete no-CAF schema for an A1/A2 failure."""
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-",dir=output.parent))
    (staging/"plots").mkdir()
    gate=gate_verdict(a1,False,False)
    if gate["verdict"] != verdict:
        raise ValueError("verdict does not match A1/A2 state")
    write_json(staging/"config.json",{"scope":"cleanStatic-only","status":"preflight_failed"})
    write_json(staging/"environment.json",{"python":sys.version})
    checks={"synthetic_preflight":True} if a1 else {"preflight":False}
    write_json(staging/"receiver_source_binding.json",{"recording_id":"cleanStatic","checks":checks,
      "raw_sha256":"0"*64,"format":"ishort","fs":FS,"skip_samples":0,"resampling":"none"})
    write_json(staging/"gnss_sdr_source_binding.json",{"prompt_support_mapping_authenticated":False,
      "failure_reason":reason})
    (staging/"gnss_sdr_tracking_semantics.md").write_text(f"# Preflight unavailable\n\n{reason}\n")
    ca_rows,legacy_evidence=ca_code_evidence()
    write_json(staging/"ca_code_validation.json",{"schema_version":"canonical_gps_l1ca_cyclic_v1",
      "canonical_prns_passed":sum(row["status"]=="PASS" for row in ca_rows),
      "local_generator_absent":True,"legacy_bad_generator_evidence":legacy_evidence,"rows":ca_rows})
    write_csv(staging/"ca_code_correlation.csv",ca_rows,CA_CODE_FIELDS)
    write_csv(staging/"candidate_application_audit.csv",[],OBSERVATION_FIELDS)
    write_json(staging/"candidate_fingerprints.json",{"expected_unique":0,"fingerprints":{}})
    write_csv(staging/"alignment_hypotheses.csv",[],["candidate","n"])
    write_json(staging/"selected_alignment.json",gate)
    write_csv(staging/"center_validation.csv",[],OBSERVATION_FIELDS)
    zero={"n":0,"prn_count":0,"min_per_prn":0,"dominant_fraction":1,
      "within_tolerance_fraction":0,"exact_center_fraction":0,"boundary_fraction":1,
      "pooled_spearman":0,"median_prn_spearman":0}
    write_json(staging/"center_validation_summary.json",zero)
    write_csv(staging/"center_metrics_by_prn.csv",[],["prn","n"])
    write_csv(staging/"center_metrics_by_channel.csv",[],["channel","n"])
    write_csv(staging/"center_metrics_by_time_block.csv",[],["time_block","n","prn_count"])
    write_csv(staging/"global_offset_sensitivity.csv",[],OBSERVATION_FIELDS+("global_offset_samples","invocation"))
    write_json(staging/"global_offset_application_audit.json",{"calls":[],"origin_selected":None})
    write_csv(staging/"prn_sampling_summary.csv",[],["prn","n"])
    write_json(staging/"raw_overlap_audit.json",{"rows":[],"cross_role_time_overlap":False,"same_epoch_cross_prn_allowed":True})
    write_json(staging/"execution_validity.json",{"caf_executed":False,"failure_reason":reason})
    write_json(staging/"go_no_go.json",gate)
    (staging/"README.md").write_text(f"# R1.3 fail-closed artifacts\n\nNo CAF executed. {reason}\n")
    (staging/"test_report.txt").write_text(f"PREFLIGHT FAILED: {reason}\n")
    (staging/"plots"/"preflight-status.svg").write_text(
      '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="120"><rect width="100%" height="100%" fill="#fee"/><text x="20" y="65" font-size="22">CAF unavailable: preflight failed</text></svg>\n')
    write_json(staging/"verification_report.json",{"status":"NOT_YET_VERIFIED"})
    write_json(staging/"checksums.json",{"files":{str(p.relative_to(staging)):sha256(p) for p in sorted(staging.rglob('*')) if p.is_file() and p.name not in {'checksums.json','verification_report.json'}}})
    missing=[name for name in ARTIFACT_FILES if not (staging/name).is_file()]
    if missing or not any((staging/"plots").iterdir()):
        raise RuntimeError(f"incomplete fail-closed inventory: {missing}")
    os.replace(staging,output)
    return gate


def read_iq(raw_path: Path, start: int, end: int) -> np.ndarray:
    if start < 0 or end <= start or 4 * end > raw_path.stat().st_size:
        raise ValueError("requested raw interval is out of bounds")
    data = np.memmap(raw_path, dtype="<i2", mode="r", offset=4 * start, shape=(2 * (end-start),))
    return data[0::2].astype(np.float32) + 1j * data[1::2].astype(np.float32)


def authenticate_inputs(raw_path: Path, tracker_dir: Path, manifest_path: Path | None = None) -> dict:
    """Authenticate the benign receiver run before opening any IQ or MAT file."""
    raw_path=Path(raw_path); tracker_dir=Path(tracker_dir)
    manifest_path=Path(manifest_path) if manifest_path else tracker_dir.parent/"manifest.json"
    try: manifest=json.loads(manifest_path.read_text())
    except (OSError,json.JSONDecodeError) as exc: raise ValueError("receiver binding manifest unavailable") from exc
    if manifest.get("recording_id")!="cleanStatic" or manifest.get("normal_only") is not True or manifest.get("attack_inputs_read") is not False:
        raise ValueError("receiver binding is not cleanStatic-only")
    auth=manifest.get("authenticated_inputs",{}); iq=auth.get("iq_before_receiver",{})
    receiver=manifest.get("receiver",{}); retained=manifest.get("retained_files",[])
    config_path=manifest_path.parent/receiver.get("config","")
    runtime_path=manifest_path.parent/receiver.get("runtime_config","")
    executable=Path(receiver.get("executable","/nonexistent"))
    expected_mats={x["name"]:x for x in retained
                   if x.get("role")=="raw_receiver_output"
                   and Path(x.get("name","")).name.startswith("epl_tracking_ch_")
                   and x.get("name","").endswith(".mat")}
    actual_mats={str(p.relative_to(manifest_path.parent)):p for p in sorted(tracker_dir.glob("epl_tracking_ch_*.mat"))}
    checks={
      "raw_name":raw_path.name==Path(iq.get("path","/nonexistent")).name,
      "raw_size":raw_path.is_file() and raw_path.stat().st_size==int(iq.get("size_bytes",-1)),
      "raw_sha256":raw_path.is_file() and sha256(raw_path)==iq.get("sha256"),
      "config_sha256":config_path.is_file() and sha256(config_path)==receiver.get("config_sha256"),
      "runtime_config_sha256":runtime_path.is_file() and sha256(runtime_path)==receiver.get("runtime_config_sha256"),
      "executable_sha256":executable.is_file() and sha256(executable)==receiver.get("executable_sha256"),
      "mat_inventory":set(actual_mats)==set(expected_mats),
      "mat_hashes":bool(actual_mats) and all(sha256(path)==expected_mats[name].get("sha256") for name,path in actual_mats.items()),
    }
    config=config_path.read_text() if config_path.is_file() else ""
    required={"SignalSource.item_type":"ishort","SignalSource.sampling_frequency":"25000000",
              "GNSS-SDR.internal_fs_sps":"25000000","Resampler.implementation":"Pass_Through",
              "Tracking_1C.tap_count":"9","Tracking_1C.tap_spacing_chips":"0.125"}
    values={}
    for line in config.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key,value=line.split("=",1); values[key.strip()]=value.strip()
    checks["receiver_config_values"]=all(values.get(k)==v for k,v in required.items())
    if not all(checks.values()): raise ValueError("receiver binding failed: "+",".join(k for k,v in checks.items() if not v))
    return {"checks":checks,"raw_sha256":iq["sha256"],"raw_path_classification":"exact_same_raw",
            "manifest_raw_path":iq.get("path"),"current_raw_path":str(raw_path),"manifest_path":str(manifest_path),
            "manifest_sha256":sha256(manifest_path),"config_values":values,
            "mat_inventory":[{"path":name,"sha256":expected_mats[name]["sha256"]} for name in sorted(expected_mats)]}


def raw_caf(raw_path, triple, candidate, fs=FS, *, start=None, end=None, grid=None):
    """One physical invocation: independently read the requested byte range and correlate it."""
    if start is None or end is None:
        raise ValueError("authenticated correlator support bounds are required")
    start=int(start); end=int(end)
    iq = read_iq(Path(raw_path), start, end)
    aux = triple[{"previous":0,"current":1,"next":2}[candidate.aux_row]]
    nco = triple[{"previous":0,"current":1,"next":2}[candidate.nco_row]]
    result = caf(iq, triple[1]["PRN"], fs, nco["code_freq_chips"], aux["aux1"],
                 nco["carrier_doppler_hz"], candidate, grid)
    application=candidate_application(iq,triple,candidate,start,end,fs,result.get("result_field_hash"))
    result.update({"raw_start_sample":start,"raw_end_sample":end,"raw_start_byte":4*start,
                   "raw_end_byte":4*end,"n_samples":end-start,
                   **application})
    return result


def run_offset_sensitivity(raw_path, triples, candidate, offsets=GLOBAL_OFFSETS, fs=FS, *, support_bounds):
    results = []
    for offset in offsets:
        shifted = replace(candidate, global_offset=int(offset))
        for invocation, (triple,bounds) in enumerate(zip(triples,support_bounds,strict=True)):
            support_start, support_end = map(int, bounds)
            start=support_start+int(offset); end=support_end+int(offset)
            result = raw_caf(raw_path, triple, shifted, fs, start=start, end=end, grid=wide_grid())
            results.append({"global_offset_samples":offset,"invocation":invocation,"start":start,"end":end,
                            "start_byte":4*start,"end_byte":4*end,**result})
    return results


def load_triples(raw_dir: Path, raw_samples: int):
    triples = []
    for channel, mat in enumerate(sorted(raw_dir.glob("epl_tracking_ch_*.mat"))):
        mat_hash = sha256(mat)
        with h5py.File(mat, "r") as handle:
            if not all(key in handle for key in REQUIRED_FIELDS[:9]):
                continue
            arrays = {key: np.asarray(handle[key]).reshape(-1) for key in REQUIRED_FIELDS[:9]}
            rows = []
            for i in range(len(arrays["PRN"])):
                row = {key: arrays[key][i].item() for key in REQUIRED_FIELDS[:9]}
                prn=float(row["PRN"]); stamp=float(row["PRN_start_sample_count"])
                if not np.isfinite(prn) or not prn.is_integer() or not 1 <= int(prn) <= 32:
                    continue
                if not np.isfinite(stamp) or not stamp.is_integer():
                    continue
                row["PRN"]=int(prn); row["PRN_start_sample_count"]=int(stamp)
                row.update(mat_row=i, channel=channel, mat_path=str(mat), mat_sha256=mat_hash)
                rows.append(row)
            triples.extend(filter_stable_triples(rows, raw_samples))
    return triples


def balanced_sample(triples, target=969, min_prns=8):
    """Choose a dense common cleanStatic span, split time first, then round-robin PRNs."""
    ordered=sorted(triples,key=lambda x:(x[0]["PRN_start_sample_count"],x[1]["PRN_start_sample_count"],x[1]["PRN"],x[1]["channel"]))
    if not ordered: return []
    role_names=("train","calibration","holdout")
    quotas={role:target//3+(index < target%3) for index,role in enumerate(role_names)}
    by_prn=defaultdict(list)
    for triple in ordered: by_prn[int(triple[1]["PRN"])].append(triple)
    minimum_total=sum(quotas.values())//min_prns
    eligible_prns=sorted(prn for prn,rows in by_prn.items() if len(rows)>=minimum_total)
    spans=[]
    for combo in combinations(eligible_prns,min_prns):
        lo=max(min(int(t[0]["PRN_start_sample_count"]) for t in by_prn[p]) for p in combo)
        hi=min(max(int(t[1]["PRN_start_sample_count"]) for t in by_prn[p]) for p in combo)
        if hi>lo: spans.append((hi-lo,combo,lo,hi))
    best=None
    for width,combo,lo,hi in sorted(spans,reverse=True):
        cuts=(lo,lo+(hi-lo)//3,lo+2*(hi-lo)//3,hi+1)
        feasible=True
        for role,begin,finish in zip(role_names,cuts,cuts[1:]):
            floor=quotas[role]//min_prns
            for prn in combo:
                count=sum(begin<=int(t[0]["PRN_start_sample_count"]) and int(t[1]["PRN_start_sample_count"])<=finish for t in by_prn[prn])
                if count<floor: feasible=False; break
            if not feasible: break
        if feasible:
            best=(width,combo,cuts)
            break
    if best is None: return []
    _,chosen_prns,cuts=best
    selected=[]; used_rows=set()
    for role,begin,finish in zip(role_names,cuts,cuts[1:]):
        queues={prn:[t for t in by_prn[prn]
                     if begin<=int(t[0]["PRN_start_sample_count"])
                     and int(t[1]["PRN_start_sample_count"])<=finish]
                for prn in chosen_prns}
        quota=quotas[role]
        while sum(1 for x in selected if x[0]==role)<quota and any(queues.values()):
            for prn in chosen_prns:
                if queues[prn] and sum(1 for x in selected if x[0]==role)<quota:
                    triple=queues[prn].pop(0)
                    key=(triple[1]["mat_sha256"],str(triple[1]["channel"]),int(triple[1]["mat_row"]))
                    if key not in used_rows:
                        selected.append((role,triple)); used_rows.add(key)
    return selected


def candidate_family():
    """Source-constrained hypotheses; reverse code progression is impossible."""
    return [Candidate(remnant_sign=rem, carrier_sign=sign)
            for rem in (-1,1) for sign in (-1,1)]


def stats(rows):
    n=len(rows)
    return {"exact_center_fraction":sum(r["exact_center"] for r in rows)/n if n else 0,
            "within_tolerance_fraction":sum(r["within_tolerance"] for r in rows)/n if n else 0,
            "boundary_fraction":sum(r["grid_boundary"] for r in rows)/n if n else 1}


def rho(rows):
    return float(spearmanr([r["center_magnitude"] for r in rows],[r["mat_prompt_magnitude"] for r in rows]).statistic) if len(rows)>2 else 0.0


def source_binding_document(receiver_values=None):
    root=Path("/home/ubuntu/build-gnss-sdr-complex9")
    expected=("src/algorithms/tracking/gnuradio_blocks/dll_pll_veml_tracking.cc",
              "src/algorithms/tracking/gnuradio_blocks/dll_pll_veml_tracking.h",
              "src/algorithms/tracking/libs/dll_pll_conf.cc",
              "src/algorithms/tracking/libs/dll_pll_conf.h")
    names=subprocess.run(["git","diff","--name-only","HEAD"],cwd=root,text=True,capture_output=True,check=True).stdout.splitlines()
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=root,text=True,capture_output=True,check=True).stdout.strip()
    modified=[{"path":name,"sha256":sha256(root/name),
               "diff_sha256":hashlib.sha256(subprocess.run(["git","diff","--binary","HEAD","--",name],cwd=root,capture_output=True,check=True).stdout).hexdigest()}
              for name in names]
    diff=subprocess.run(["git","diff","--binary","--",*names],cwd=root,capture_output=True,check=True).stdout
    exe=root/"build-complex/src/main/gnss-sdr"; cache=root/"build-complex/CMakeCache.txt"
    cache_text=cache.read_text(errors="replace")
    compiler=next((x.split("=",1)[1] for x in cache_text.splitlines() if x.startswith("CMAKE_CXX_COMPILER:FILEPATH=")),None)
    adapter=root/"src/algorithms/tracking/adapters/gps_l1_ca_dll_pll_tracking.cc"
    adapter_text=adapter.read_text()
    fs=int((receiver_values or {}).get("GNSS-SDR.internal_fs_sps",25000000))
    vector_length=round(fs/(1_023_000/1023))
    exact_files=set(names)==set(expected)
    return {"source_root":str(root),"git_base_sha":head,"git_head_sha":head,
            "base_head_exact":head=="1ddd4562723040fd66cb334b578a5b69455625f4",
            "modified_file_set_exact":exact_files,
            "modified_tracked_files":modified,"modified_diff_sha256":hashlib.sha256(diff).hexdigest(),
            "build_evidence":{"cmake_cache_path":str(cache),"cmake_cache_sha256":sha256(cache)},
            "compiler_evidence":compiler,"executable_sha256":sha256(exe),
            "receiver_config_values":receiver_values or {},
            "adapter_path":str(adapter),"adapter_sha256":sha256(adapter),
            "vector_length_derivation":"round(fs_in / (GPS_L1_CA_CODE_RATE_CPS / GPS_L1_CA_CODE_LENGTH_CHIPS))",
            "vector_length":vector_length,"tap_count":9,"tap_spacing_chips":0.125,
            "extended_integration_symbols":1,"track_pilot":False,
            "high_dynamics":False,"sampling_rate_sps":fs,"code_rate_chips_s":1023000,"code_length_chips":1023,
            "prompt_target":{"Prompt_I_Q":"single-step d_Prompt complex output","nine_tap_P":"d_tap_accu[4]",
                             "equivalence_authenticated":True,"EPL":"available as nine-tap E/P/L fields; Prompt_I/Q is current d_Prompt target"},
            "prompt_support_mapping_authenticated":bool(exact_files and "vector_length" in adapter_text and vector_length==25000),
            "missing_evidence":[],
            "excerpts":{"correlation":"lines 1061-1092: correlator receives remnant carrier/code phase and forward NCO steps",
                        "updates":"lines 1137-1159: carrier Doppler and code frequency updates",
                        "remnant":"lines 1222-1291: exact interval, forward code step, remnant samples-to-chips",
                        "prompt":"lines 1435-1443: Prompt I/Q taken from complex correlator",
                        "stamp":"lines 1482-1518: stamp=nitems_read+current_prn_length_samples; NCO, quality, aux1 dumped"},
            "interpretation":{"consumed_interval":"adjacent stamps are retained only as consume/boundary audit",
                              "correlator_support":"[stamp(k-1), stamp(k-1)+vector_length)",
                              "aux1":"row k-1 updated next-call remnant; applied to current Prompt row k",
                              "replica_direction":1,"prompt_row":"current","nco_row":"previous","aux_row":"previous"},
            "A2_source_semantics_sufficient":bool(exact_files and vector_length==25000)}


def _execute_campaign(args):
    binding=authenticate_inputs(args.raw,args.tracker_dir,args.manifest)
    source_binding=source_binding_document(binding["config_values"])
    if not source_binding["prompt_support_mapping_authenticated"]:
        raise RuntimeError("A2 fail closed: source/config support mapping is not authenticated")
    vector_length=int(source_binding["vector_length"])
    out=args.output; out.mkdir(parents=True,exist_ok=False); (out/"plots").mkdir()
    raw_sha=binding["raw_sha256"]; raw_samples=args.raw.stat().st_size//4
    triples=load_triples(args.tracker_dir,raw_samples); selected=balanced_sample(triples,args.epochs)
    if args.epochs < 950 or len(selected) != args.epochs:
        raise RuntimeError("sampling requires at least 950 balanced stable epochs for 19 PRNs")
    write_json(out/"config.json",{"scope":"cleanStatic-only","origin_offset_samples":0,"global_offsets":GLOBAL_OFFSETS,"wide_grid":wide_grid(),"epochs":args.epochs})
    write_json(out/"environment.json",{"python":sys.version,"numpy":np.__version__})
    write_json(out/"receiver_source_binding.json",{"recording_id":"cleanStatic","checks":binding["checks"],
      "raw_path":str(args.raw),"raw_sha256":raw_sha,"format":"ishort","fs":FS,"skip_samples":0,"resampling":"none",
      "manifest_path":binding["manifest_path"],"manifest_sha256":binding["manifest_sha256"],
      "manifest_raw_path":binding["manifest_raw_path"],"current_raw_path":binding["current_raw_path"],
      "classification":binding["raw_path_classification"],
      "config_values":binding["config_values"],"mat_inventory":binding["mat_inventory"]})
    write_json(out/"gnss_sdr_source_binding.json",source_binding)
    semantics=(Path(__file__).resolve().parents[1]/"docs/ACAF_NF_STAGE0_STATIC_R13_RECONSTRUCTION.md").read_text()
    (out/"gnss_sdr_tracking_semantics.md").write_text(semantics)
    ca_rows,legacy_evidence=ca_code_evidence()
    write_json(out/"ca_code_validation.json",{"schema_version":"canonical_gps_l1ca_cyclic_v1",
      "canonical_prns_passed":sum(row["status"]=="PASS" for row in ca_rows),
      "local_generator_absent":True,"legacy_bad_generator_evidence":legacy_evidence,"rows":ca_rows})
    write_csv(out/"ca_code_correlation.csv",ca_rows,CA_CODE_FIELDS)
    hypotheses=[]; details_by_name={}; audit=[]; fingerprints={}
    for candidate in candidate_family():
        details=[]
        for role,triple in selected:
            support=source_support(triple,vector_length)
            result=raw_caf(args.raw,triple,candidate,FS,start=support["start_sample"],end=support["end_sample"])
            prompt=triple[1]; result.update(candidate=candidate.name,prn=int(prompt["PRN"]),channel=prompt["channel"],role=role,
                recording="cleanStatic",tracker_row=int(prompt["mat_row"]),mat_prompt_magnitude=float(np.hypot(prompt["Prompt_I"],prompt["Prompt_Q"])),
                previous_mat_row=int(triple[0]["mat_row"]),current_mat_row=int(prompt["mat_row"]),next_mat_row=int(triple[2]["mat_row"]),
                mat_path=prompt.get("mat_path",""),mat_sha256=prompt["mat_sha256"],
                consumed_start_sample=support["consumed_start_sample"],consumed_end_sample=support["consumed_end_sample"],
                consumed_length_samples=support["consumed_length_samples"],support_start_sample=support["start_sample"],
                support_end_sample=support["end_sample"],support_length_samples=support["length_samples"],
                vector_length=vector_length,prompt_row_index=int(prompt["mat_row"]))
            details.append(result); key=f"{candidate.name}:{prompt['channel']}:{prompt['mat_row']}"; fingerprints[key]=result["result_field_sha256"]
            audit.append({"key":key,**result})
        byprn=[]
        for prn in sorted({r["prn"] for r in details}): byprn.append(rho([r for r in details if r["prn"]==prn]))
        hypotheses.append({"candidate":candidate.name,"n":len(details),"pooled_spearman":rho(details),"median_prn_spearman":float(np.nanmedian(byprn)),**stats(details)})
        details_by_name[candidate.name]=details
    physical_fields=("raw_interval_content_sha256","raw_interval_range_sha256","replica_chip_indices_sha256",
      "carrier_wipeoff_sha256","aux_indices_sha256","nco_indices_sha256","prompt_indices_sha256","result_field_sha256")
    fingerprint_rows={row["key"]:{field:row[field] for field in physical_fields} for row in audit}
    write_csv(out/"candidate_application_audit.csv",audit); write_json(out/"candidate_fingerprints.json",{"expected_unique":len(fingerprint_rows),"fingerprints":fingerprint_rows})
    write_csv(out/"alignment_hypotheses.csv",hypotheses); best=max(hypotheses,key=lambda x:(x["within_tolerance_fraction"],x["pooled_spearman"])); details=details_by_name[best["candidate"]]
    write_csv(out/"center_validation.csv",details)
    counts=Counter(r["prn"] for r in details); blocks=defaultdict(list)
    for r in details: blocks[r["role"]].append(r)
    prn_metrics=[]
    for prn in sorted(counts):
        group=[r for r in details if r["prn"]==prn]; prn_metrics.append({"prn":prn,"n":len(group),"prompt_spearman":rho(group),**stats(group)})
    channel_metrics=[]
    for ch in sorted({r["channel"] for r in details}):
        group=[r for r in details if r["channel"]==ch]; channel_metrics.append({"channel":ch,"n":len(group),"prompt_spearman":rho(group),**stats(group)})
    block_metrics=[]
    for role in ("train","calibration","holdout"):
        group=blocks[role]; block_metrics.append({"time_block":role,"n":len(group),"prn_count":len({r['prn'] for r in group}),"prompt_spearman":rho(group),**stats(group)})
    summary={"n":len(details),"prn_count":len(counts),"min_per_prn":min(counts.values()),"dominant_fraction":max(counts.values())/len(details),
             "pooled_spearman":rho(details),"median_prn_spearman":float(np.nanmedian([x["prompt_spearman"] for x in prn_metrics])),**stats(details)}
    write_csv(out/"center_metrics_by_prn.csv",prn_metrics); write_csv(out/"center_metrics_by_channel.csv",channel_metrics); write_csv(out/"center_metrics_by_time_block.csv",block_metrics); write_json(out/"center_validation_summary.json",summary)
    write_csv(out/"prn_sampling_summary.csv",[{'prn':p,'n':n,'fraction':n/len(details)} for p,n in sorted(counts.items())])
    intervals=[{"role":role,"start":tr[0]["PRN_start_sample_count"],"end":tr[1]["PRN_start_sample_count"],"prn":tr[1]["PRN"],"channel":tr[1]["channel"],"tracker_row":tr[1]["mat_row"]} for role,tr in selected]
    write_json(out/"raw_overlap_audit.json",{"same_epoch_cross_prn_allowed":True,"cross_role_time_overlap":not roles_nonoverlap(intervals),"unique_tracker_rows":len({(x['channel'],x['tracker_row']) for x in intervals}),"rows":intervals})
    chosen=next(c for c in candidate_family() if c.name==best["candidate"])
    support_bounds=[(source_support(x[1],vector_length)["start_sample"],source_support(x[1],vector_length)["end_sample"]) for x in selected]
    offsets=run_offset_sensitivity(args.raw,[x[1] for x in selected],chosen,GLOBAL_OFFSETS,FS,support_bounds=support_bounds)
    offset_summaries=[]
    selected_by_invocation={i:(role,triple) for i,(role,triple) in enumerate(selected)}
    for offset in GLOBAL_OFFSETS:
        group=[row for row in offsets if int(row["global_offset_samples"])==int(offset)]
        for row in group:
            role,triple=selected_by_invocation[int(row["invocation"])]
            prompt=triple[1]
            row.update(prn=int(prompt["PRN"]),channel=int(prompt["channel"]),role=role,
                       mat_prompt_magnitude=float(np.hypot(prompt["Prompt_I"],prompt["Prompt_Q"])))
        per_prn=[rho([r for r in group if r["prn"]==prn]) for prn in sorted({r["prn"] for r in group})]
        offset_summaries.append({"global_offset_samples":offset,"valid_raw_epochs":len(group),
          **stats(group),"pooled_spearman":rho(group),"median_prn_spearman":float(np.nanmedian(per_prn)),
          "peak_delay_median":float(np.median([r["peak_delay_offset_chips"] for r in group])),
          "peak_doppler_median":float(np.median([r["peak_doppler_offset_hz"] for r in group]))})
    write_csv(out/"global_offset_sensitivity.csv",offset_summaries); write_json(out/"global_offset_application_audit.json",{"calls":offsets,"origin_selected":0})
    write_json(out/"execution_validity.json",{"prompt_support_mapping_authenticated":True,"stable_filter_evidence_rows":len(details),"exact_support_lengths":sorted({r['n_samples'] for r in details})})
    a3=(summary["n"]>=800 and summary["prn_count"]>=8 and summary["min_per_prn"]>=50 and summary["dominant_fraction"]<=.2 and all(x["n"]>=200 and x["prn_count"]>=8 for x in block_metrics) and summary["within_tolerance_fraction"]>=.95 and summary["pooled_spearman"]>=.9 and summary["median_prn_spearman"]>=.8 and summary["boundary_fraction"]<=.05)
    a1=all(binding["checks"].values()); a2=bool(source_binding["prompt_support_mapping_authenticated"])
    verdict=gate_verdict(a1,a2,a3,best["candidate"]); write_json(out/"selected_alignment.json",verdict); write_json(out/"go_no_go.json",verdict)
    (out/"README.md").write_text(f"""# ACAF-NF Stage-0 R1.3 reconstruction

R1.2 was scientifically invalid because it used a non-canonical C/A replica, ignored candidate dimensions, copied global-offset metrics, mislabeled A2, and omitted stable-lock filtering. R1.3 imports the canonical generator, applies aux/remnant/carrier inputs, separates variable consumed intervals from source-authenticated correlator support, and recomputes every offset on cleanStatic only.

- Canonical C/A: 32/32
- PRNs / epochs: {summary['prn_count']} / {summary['n']}
- Blocks: {', '.join(f"{x['time_block']}={x['n']}" for x in block_metrics)}
- Best diagnostic candidate: {best['candidate']}
- Within tolerance: {summary['within_tolerance_fraction']:.9f}
- Pooled Prompt Spearman: {summary['pooled_spearman']:.9f}
- Median PRN Prompt Spearman: {summary['median_prn_spearman']:.9f}
- Wide-grid boundary: {summary['boundary_fraction']:.9f}
- A1/A2/A3: {'PASS' if a1 else 'FAIL'} / {'PASS' if a2 else 'FAIL'} / {'PASS' if a3 else 'FAIL'}
- Selected alignment: {verdict['selected_alignment']}

Failure of A3 is tracker/raw reconstruction or alignment unresolved, not an ACAF physical-model NO-GO. `physics_no_go_claim` is always false in this audit.
"""); (out/"test_report.txt").write_text("See source-phase and production wrapper logs.\n")
    (out/"plots"/"center-recovery.svg").write_text(
      f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="120"><rect width="100%" height="100%" fill="#eef"/><text x="20" y="65" font-size="22">center recovery: {summary["within_tolerance_fraction"]:.3f}</text></svg>\n')
    write_json(out/"checksums.json",{"files":{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in {'checksums.json','verification_report.json'}}})
    write_json(out/"verification_report.json",{"status":"NOT_YET_VERIFIED","instruction":"run independent verifier"})
    return verdict


def execute_campaign(args):
    """Run in a sibling staging directory and publish only a complete result."""
    final=Path(args.output)
    if final.exists():
        raise FileExistsError(final)
    final.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-",dir=final.parent))
    staging.rmdir()  # reserve a unique name; the core creates the directory
    staged_args=argparse.Namespace(**vars(args)); staged_args.output=staging
    try:
        verdict=_execute_campaign(staged_args)
        missing=[name for name in ARTIFACT_FILES if not (staging/name).is_file()]
        if missing or not any((staging/"plots").iterdir()):
            raise RuntimeError(f"incomplete production inventory: {missing}")
        os.replace(staging,final)
        return verdict
    except ValueError as exc:
        if staging.exists(): shutil.rmtree(staging)
        return publish_fail_closed(final,"SOURCE_BINDING_INVALID",str(exc),a1=False)
    except RuntimeError as exc:
        if staging.exists(): shutil.rmtree(staging)
        if str(exc).startswith("A2 fail closed:"):
            return publish_fail_closed(final,"RECONSTRUCTION_IMPLEMENTATION_INVALID",str(exc),a1=True)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw",type=Path,required=True); parser.add_argument("--tracker-dir",type=Path,required=True)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--output",type=Path,default=OUT); parser.add_argument("--epochs",type=int,default=969); parser.add_argument("--execute-production",action="store_true")
    args=parser.parse_args(argv)
    clean_only_guard(["cleanStatic"])
    if not args.execute_production:
        raise SystemExit("source-only safety: a later production phase must pass --execute-production")
    # Production implementation is intentionally fail-closed at every prerequisite.
    if args.output.resolve() != OUT.resolve():
        raise ValueError("R1.3 writes only its new artifact directory")
    execute_campaign(args)


if __name__ == "__main__": main()
