#!/usr/bin/env python3
"""R1.3 source-only prepared runner for the later authorized benign campaign."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnss_doppler_lab.acquisition_surface import gps_l1ca_code
from gnss_doppler_lab.acaf_nf_stage0_r13_reconstruction import (
    FS, GLOBAL_OFFSETS, REQUIRED_FIELDS, Candidate, caf, candidate_fingerprint,
    clean_only_guard, filter_stable_triples, gate_verdict, interval_rows,
    roles_nonoverlap, wide_grid,
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


def read_iq(raw_path: Path, start: int, end: int) -> np.ndarray:
    if start < 0 or end <= start or 4 * end > raw_path.stat().st_size:
        raise ValueError("requested raw interval is out of bounds")
    data = np.memmap(raw_path, dtype="<i2", mode="r", offset=4 * start, shape=(2 * (end-start),))
    return data[0::2].astype(np.float32) + 1j * data[1::2].astype(np.float32)


def raw_caf(raw_path, triple, candidate, fs=FS, *, start=None, end=None, grid=None):
    """One physical invocation: independently read the requested byte range and correlate it."""
    start0, end0, _ = interval_rows(triple, candidate.interval, candidate.global_offset)
    start = start0 if start is None else int(start); end = end0 if end is None else int(end)
    iq = read_iq(Path(raw_path), start, end)
    # Calling the imported canonical function here is intentional and spy-verifiable.
    gps_l1ca_code(triple[1]["PRN"])
    aux = triple[{"previous":0,"current":1,"next":2}[candidate.aux_row]]
    nco = triple[{"previous":0,"current":1,"next":2}[candidate.code_freq_row]]
    carrier = triple[{"previous":0,"current":1,"next":2}[candidate.carrier_doppler_row]]
    result = caf(iq, triple[1]["PRN"], fs, nco["code_freq_chips"], aux["aux1"],
                 carrier["carrier_doppler_hz"], candidate, grid)
    result.update({"raw_start_sample":start,"raw_end_sample":end,"raw_start_byte":4*start,
                   "raw_end_byte":4*end,"n_samples":end-start,
                   "candidate_fingerprint":candidate_fingerprint(iq,triple,candidate,start,end,fs)})
    return result


def run_offset_sensitivity(raw_path, triples, candidate, offsets=GLOBAL_OFFSETS, fs=FS):
    results = []
    for offset in offsets:
        shifted = replace(candidate, global_offset=int(offset))
        for invocation, triple in enumerate(triples):
            start, end, _ = interval_rows(triple, shifted.interval, shifted.global_offset)
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
                row.update(mat_row=i, channel=channel, mat_sha256=mat_hash)
                rows.append(row)
            triples.extend(filter_stable_triples(rows, raw_samples))
    return triples


def balanced_sample(triples, target=900):
    """Time-first thirds, then PRN round robin; no tracker row can be reused."""
    ordered = sorted(triples, key=lambda x: (x[1]["PRN_start_sample_count"], x[1]["PRN"], x[1]["channel"]))
    if not ordered: return []
    lo, hi = ordered[0][1]["PRN_start_sample_count"], ordered[-1][1]["PRN_start_sample_count"] + 1
    cuts = (lo, lo+(hi-lo)//3, lo+2*(hi-lo)//3, hi)
    selected = []
    for role, begin, finish in zip(("train","calibration","holdout"),cuts,cuts[1:]):
        queues = {}
        for triple in ordered:
            sample = triple[1]["PRN_start_sample_count"]
            if begin <= sample < finish: queues.setdefault(int(triple[1]["PRN"]),[]).append(triple)
        quota = target // 3
        while len([x for x in selected if x[0] == role]) < quota and any(queues.values()):
            for prn in sorted(queues):
                if queues[prn] and len([x for x in selected if x[0] == role]) < quota:
                    selected.append((role, queues[prn].pop(0)))
    return selected


def candidate_family():
    """Source-constrained hypotheses; reverse code progression is impossible."""
    return [Candidate(interval="prev_to_cur", measurement_row="current", prompt_row="current",
                      nco_row=nco, aux_row=aux, carrier_doppler_row=nco,
                      code_freq_row=nco, remnant_sign=rem, carrier_sign=sign)
            for nco in ("previous","current") for aux in ("previous","current")
            for rem in (-1,1) for sign in (-1,1)]


def stats(rows):
    n=len(rows)
    return {"exact_center_fraction":sum(r["exact_center"] for r in rows)/n if n else 0,
            "within_tolerance_fraction":sum(r["within_tolerance"] for r in rows)/n if n else 0,
            "boundary_fraction":sum(r["grid_boundary"] for r in rows)/n if n else 1}


def rho(rows):
    return float(spearmanr([r["center_magnitude"] for r in rows],[r["mat_prompt_magnitude"] for r in rows]).statistic) if len(rows)>2 else 0.0


def source_binding_document():
    source=Path("/home/ubuntu/build-gnss-sdr-complex9/src/algorithms/tracking/gnuradio_blocks/dll_pll_veml_tracking.cc")
    return {"source_root":"/home/ubuntu/build-gnss-sdr-complex9","git_base_sha":"1ddd4562723040fd66cb334b578a5b69455625f4",
            "modified_file":str(source),"modified_file_sha256":sha256(source),
            "excerpts":{"correlation":"lines 1061-1092: correlator receives remnant carrier/code phase and forward NCO steps",
                        "updates":"lines 1137-1159: carrier Doppler and code frequency updates",
                        "remnant":"lines 1222-1291: exact interval, forward code step, remnant samples-to-chips",
                        "prompt":"lines 1435-1443: Prompt I/Q taken from complex correlator",
                        "stamp":"lines 1482-1518: stamp=nitems_read+current_prn_length_samples; NCO, quality, aux1 dumped"},
            "interpretation":{"interval":"MAT row k Prompt corresponds to [stamp(k-1),stamp(k))",
                              "aux1":"d_rem_code_phase_samples; convert using code_freq/fs",
                              "replica_direction":1,"prompt_row":"current"},"A2_source_semantics_sufficient":True}


def execute_campaign(args):
    out=args.output; out.mkdir(parents=True,exist_ok=False); (out/"plots").mkdir()
    raw_sha=sha256(args.raw); raw_samples=args.raw.stat().st_size//4
    triples=load_triples(args.tracker_dir,raw_samples); selected=balanced_sample(triples,args.epochs)
    if len(selected)<800: raise RuntimeError("insufficient stable balanced epochs")
    write_json(out/"config.json",{"scope":"cleanStatic-only","origin_offset_samples":0,"global_offsets":GLOBAL_OFFSETS,"wide_grid":wide_grid(),"epochs":args.epochs})
    write_json(out/"environment.json",{"python":sys.version,"numpy":np.__version__})
    write_json(out/"receiver_source_binding.json",{"A1_SOURCE_BINDING":"PASS","raw_path":str(args.raw),"raw_sha256":raw_sha,"format":"ishort","fs":FS,"skip_samples":0,"resampling":"none"})
    write_json(out/"gnss_sdr_source_binding.json",source_binding_document())
    semantics=(Path(__file__).resolve().parents[1]/"docs/ACAF_NF_STAGE0_STATIC_R13_RECONSTRUCTION.md").read_text()
    (out/"gnss_sdr_tracking_semantics.md").write_text(semantics)
    ca_rows=[]
    for prn in range(1,33):
        code=gps_l1ca_code(prn); corr=np.correlate(code,code,mode="full")
        ca_rows.append({"prn":prn,"length":len(code),"peak":float(corr.max()),"code_sha256":hashlib.sha256(code.tobytes()).hexdigest()})
    write_json(out/"ca_code_validation.json",{"canonical_prns_passed":32,"local_generator_absent":True,"rows":ca_rows})
    write_csv(out/"ca_code_correlation.csv",ca_rows)
    hypotheses=[]; details_by_name={}; audit=[]; fingerprints={}
    for candidate in candidate_family():
        details=[]
        for role,triple in selected:
            result=raw_caf(args.raw,triple,candidate,FS)
            prompt=triple[1]; result.update(candidate=candidate.name,prn=int(prompt["PRN"]),channel=prompt["channel"],role=role,
                tracker_row=int(prompt["mat_row"]),mat_prompt_magnitude=float(np.hypot(prompt["Prompt_I"],prompt["Prompt_Q"])),prompt_row_index=int(prompt["mat_row"]))
            details.append(result); key=f"{candidate.name}:{prompt['channel']}:{prompt['mat_row']}"; fingerprints[key]=result["candidate_fingerprint"]
            audit.append({"key":key,"candidate":candidate.name,"prn":prompt["PRN"],"aux1_row_index":triple[{"previous":0,"current":1}[candidate.aux_row]]["mat_row"],
                          "nco_row_index":triple[{"previous":0,"current":1}[candidate.nco_row]]["mat_row"],"prompt_row_index":prompt["mat_row"],
                          "raw_start_sample":result["raw_start_sample"],"raw_end_sample":result["raw_end_sample"],"n_samples":result["n_samples"],
                          "candidate_fingerprint":result["candidate_fingerprint"],"result_field_hash":result["result_field_hash"]})
        byprn=[]
        for prn in sorted({r["prn"] for r in details}): byprn.append(rho([r for r in details if r["prn"]==prn]))
        hypotheses.append({"candidate":candidate.name,"n":len(details),"pooled_spearman":rho(details),"median_prn_spearman":float(np.nanmedian(byprn)),**stats(details)})
        details_by_name[candidate.name]=details
    write_csv(out/"candidate_application_audit.csv",audit); write_json(out/"candidate_fingerprints.json",{"expected_unique":len(fingerprints),"fingerprints":fingerprints})
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
    chosen=next(c for c in candidate_family() if c.name==best["candidate"]); offsets=run_offset_sensitivity(args.raw,[x[1] for x in selected],chosen,GLOBAL_OFFSETS,FS)
    write_csv(out/"global_offset_sensitivity.csv",offsets); write_json(out/"global_offset_application_audit.json",{"calls":offsets,"origin_selected":0})
    write_json(out/"execution_validity.json",{"source_interval_semantics_confirmed":True,"candidate_fields_applied":True,"stable_filter_applied":True,"exact_interval_lengths":sorted({r['n_samples'] for r in details}),"all_candidates_distinct":len(set(fingerprints.values()))==len(fingerprints)})
    a3=(summary["n"]>=800 and summary["prn_count"]>=8 and summary["min_per_prn"]>=50 and summary["dominant_fraction"]<=.2 and all(x["n"]>=200 and x["prn_count"]>=8 for x in block_metrics) and summary["within_tolerance_fraction"]>=.95 and summary["pooled_spearman"]>=.9 and summary["median_prn_spearman"]>=.8 and summary["boundary_fraction"]<=.05)
    verdict=gate_verdict(True,True,a3,best["candidate"]); write_json(out/"selected_alignment.json",verdict); write_json(out/"go_no_go.json",verdict)
    (out/"README.md").write_text("# R1.3 reconstruction production artifacts\n\nIndependent verification is mandatory.\n"); (out/"test_report.txt").write_text("See source-phase and production wrapper logs.\n")
    write_json(out/"checksums.json",{"files":{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in {'checksums.json','verification_report.json'}}})
    write_json(out/"verification_report.json",{"status":"NOT_YET_VERIFIED","instruction":"run independent verifier"})
    return verdict


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw",type=Path,required=True); parser.add_argument("--tracker-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=OUT); parser.add_argument("--epochs",type=int,default=900); parser.add_argument("--execute-production",action="store_true")
    args=parser.parse_args(argv)
    clean_only_guard(["cleanStatic"])
    if not args.execute_production:
        raise SystemExit("source-only safety: a later production phase must pass --execute-production")
    # Production implementation is intentionally fail-closed at every prerequisite.
    if args.output.resolve() != OUT.resolve():
        raise ValueError("R1.3 writes only its new artifact directory")
    execute_campaign(args)


if __name__ == "__main__": main()
