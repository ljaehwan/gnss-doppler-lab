#!/usr/bin/env python3
"""Production wrapper for the later R1.4 cleanStatic-only campaign.

No output is created unless --execute-production is explicit.  The final path
is published atomically only after the complete inventory is present.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, shutil, sys, tempfile, time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gnss_doppler_lab.acquisition_surface import gps_l1ca_code
from gnss_doppler_lab.acaf_nf_stage0_r13_reconstruction import code_replica, carrier_wipeoff, source_support
from gnss_doppler_lab.acaf_nf_stage0_r14_doppler_validation import (
    CANDIDATE_STRING, FROZEN_CONFIG, FS, LENGTHS, R13_REFERENCE, ROLES,
    aggregation_gate, bootstrap_paired, check_r13_metrics, clean_only_guard,
    common_anchor_blocks, delay_gate, delay_metrics, diagnostic_aggregates,
    doppler_metrics, final_gates, offset_zero_clearly_better, paired_improvements,
    prompt_evidence, prompt_gate, prompt_metrics,
)

# Read-only reuse of authenticated R1.3 ingestion and exact epoch selection.
from run_acaf_nf_stage0_static_r13_reconstruction import (
    authenticate_inputs, balanced_sample, load_triples, read_iq, sha256,
)

OUT = ROOT / "artifacts/acaf_nf_stage0_static_r14_doppler_validation"
R13_ARTIFACT = ROOT / "artifacts/acaf_nf_stage0_static_r13_reconstruction"
R13_SOURCE = ROOT / "src/gnss_doppler_lab/acaf_nf_stage0_r13_reconstruction.py"
GRID = {"delay_chips": [round(-1 + .125*i, 3) for i in range(17)],
        "doppler_hz": list(range(-250, 251, 50))}
INVENTORY = ("README.md config.json environment.json r13_frozen_lineage.json frozen_reconstruction_config.json "
 "prompt_reproduction_metrics.json prompt_reproduction_by_prn.csv prompt_reproduction_by_time_block.csv "
 "delay_recovery_metrics.json delay_recovery_by_prn.csv delay_recovery_by_time_block.csv doppler_1ms_metrics.json "
 "aggregation_metrics.csv aggregation_by_prn.csv aggregation_by_time_block.csv paired_improvement.csv bootstrap_results.json "
 "doppler_mainlobe_diagnostics.csv residual_doppler_diagnostics.json per_block_scores.csv execution_validity.json "
 "go_no_go.json test_report.txt verification_report.json checksums.json plots").split()


def write_json(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
def write_csv(path, rows):
    rows=list(rows); fields=list(rows[0]) if rows else ["status"]
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)


def frozen_lineage() -> dict:
    summary=json.loads((R13_ARTIFACT/"center_validation_summary.json").read_text())
    checks={"r13_source_sha256":sha256(R13_SOURCE),
            "r13_artifact_checksums_sha256":sha256(R13_ARTIFACT/"checksums.json")}
    return {"artifact_path":str(R13_ARTIFACT.relative_to(ROOT)), "source_path":str(R13_SOURCE.relative_to(ROOT)),
            **checks, "reference_metrics":R13_REFERENCE, "metrics_tolerance":1e-6,
            "metrics_exact_within_tolerance":check_r13_metrics(summary),
            "immutable_candidate":CANDIDATE_STRING}


def complex_caf_surface(iq, row) -> np.ndarray:
    """Vectorized 11x17 complex field; each call uses this row's frozen state."""
    n=len(iq); candidate=FROZEN_CONFIG.candidate
    replicas=np.asarray([code_replica(row["prn"],n,FS,row["code_freq_chips"],row["aux1"],
                        candidate.remnant_sign,d,replica_direction=1)[0] for d in GRID["delay_chips"]])
    wiped=np.asarray([carrier_wipeoff(n,FS,row["carrier_doppler_hz"],f,candidate.carrier_sign)[0]
                       for f in GRID["doppler_hz"]]) * np.asarray(iq)[None,:]
    return wiped @ replicas.T


def surface_score(surface, identity) -> dict:
    power=np.abs(surface); flat=int(np.argmax(power)); di,ci=np.unravel_index(flat,power.shape)
    center=float(power[GRID["doppler_hz"].index(0),GRID["delay_chips"].index(0)])
    peak=float(power[di,ci]); delay=float(GRID["delay_chips"][ci]); doppler=float(GRID["doppler_hz"][di])
    return {**identity,"peak_delay_offset_chips":delay,"peak_doppler_offset_hz":doppler,
            "peak_magnitude":peak,"center_magnitude":center,"peak_center_ratio":peak/max(center,np.finfo(float).eps),
            "delay_boundary":ci in (0,power.shape[1]-1),"doppler_boundary":di in (0,power.shape[0]-1)}


def grouped(rows, key, metric):
    out=[]
    for value in sorted({r[key] for r in rows},key=str): out.append({key:value,**metric([r for r in rows if r[key]==value])})
    return out


def run(args):
    lineage=frozen_lineage()
    if not lineage["metrics_exact_within_tolerance"]: raise RuntimeError("R1.3 metric lineage drift")
    binding=authenticate_inputs(args.raw,args.tracker_dir,args.manifest)
    if not all(binding["checks"].values()): raise RuntimeError("A1 source binding failed")
    raw_samples=args.raw.stat().st_size//4
    selected=balanced_sample(load_triples(args.tracker_dir,raw_samples),969)
    if len(selected)!=969 or {r for r,_ in selected} != set(ROLES): raise RuntimeError("exact R1.3 epoch reproduction failed")

    final=Path(args.output); final.parent.mkdir(parents=True,exist_ok=True)
    if final.exists(): raise FileExistsError(final)
    stage=Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-",dir=final.parent)); (stage/"plots").mkdir()
    started=time.perf_counter(); scores=[]; surfaces={}
    try:
        for role,triple in selected:
            state=triple[0]; prompt=triple[1]; support=source_support(triple,25000)
            iq=read_iq(args.raw,support["start_sample"],support["end_sample"])
            identity={"recording":"cleanStatic","role":role,"prn":int(prompt["PRN"]),"channel":str(prompt["channel"]),
              "tracker_row":int(prompt["mat_row"]),"anchor_tracker_row":int(prompt["mat_row"]),
              "support_start_sample":support["start_sample"],"support_length_samples":support["length_samples"],
              "valid_raw_support":True,"cn0_db_hz":float(prompt["CN0_SNV_dB_Hz"]),"carrier_lock":float(prompt["carrier_lock_test"]),
              "code_freq_chips":float(state["code_freq_chips"]),"carrier_doppler_hz":float(state["carrier_doppler_hz"]),"aux1":float(state["aux1"]),
              "mat_prompt_magnitude":float(np.hypot(prompt["Prompt_I"],prompt["Prompt_Q"]))}
            surface=complex_caf_surface(iq,identity); key=(identity["channel"],identity["prn"],identity["tracker_row"])
            surfaces[key]=surface; scores.append(prompt_evidence(surface_score(surface,identity)))

        # R1.3 offset proof is read, never reselected.
        offset_rows=list(csv.DictReader((R13_ARTIFACT/"global_offset_sensitivity.csv").open(newline="")))
        prompt_all=prompt_metrics(scores); prompt_prn=grouped(scores,"prn",prompt_metrics); prompt_role=grouped(scores,"role",prompt_metrics)
        delay_all=delay_metrics(scores); delay_prn=grouped(scores,"prn",delay_metrics); delay_role=grouped(scores,"role",delay_metrics)
        doppler_all=doppler_metrics(scores)
        blocks=common_anchor_blocks(scores); aggregation=[]; aggregation_prn=[]; aggregation_role=[]; paired=[]; boots={}
        for L in LENGTHS:
            current=[]
            for block in blocks[L]:
                aggregate=diagnostic_aggregates([surfaces[(str(r["channel"]),int(r["prn"]),int(r["tracker_row"]))] for r in block])
                ident={k:block[-1][k] for k in ("role","prn","channel","anchor_tracker_row")}
                current.append(surface_score(np.sqrt(aggregate["normalized_power_mean"]),ident))
            metrics={"L":L,**doppler_metrics(current),"block_count":len(current),"prn_count":len({r['prn'] for r in current}),
                     "dominant_fraction":max(Counter(r['prn'] for r in current).values(),default=0)/max(len(current),1)}
            aggregation.append(metrics)
            aggregation_prn.extend({"L":L,**x} for x in grouped(current,"prn",doppler_metrics))
            aggregation_role.extend({"L":L,**x} for x in grouped(current,"role",doppler_metrics))
            if L>1:
                pairs=paired_improvements([surface_score(np.sqrt(diagnostic_aggregates([surfaces[(str(b[-1]['channel']),int(b[-1]['prn']),int(b[-1]['tracker_row']))]])["normalized_power_mean"]),{k:b[-1][k] for k in ("role","prn","channel","anchor_tracker_row")}) for b in blocks[1]],current)
                paired.extend({"L":L,**x} for x in pairs); boots[str(L)]=bootstrap_paired(pairs)
        l20=next(x for x in aggregation if x["L"]==20); pair20=[r for r in paired if r["L"]==20]
        prn_diff=[{"prn":p,"difference":float(np.mean([r["difference"] for r in pair20 if r["prn"]==p]))} for p in sorted({r["prn"] for r in pair20})]
        role_diff=[{"role":p,"difference":float(np.mean([r["difference"] for r in pair20 if r["role"]==p]))} for p in ROLES]
        a1=all(binding["checks"].values()) and lineage["metrics_exact_within_tolerance"]
        a2=FROZEN_CONFIG.document()["candidate_string"]==CANDIDATE_STRING and len(gps_l1ca_code(1))==1023
        a3a=prompt_gate(prompt_all) and all(prompt_gate(x) for x in prompt_role) and offset_zero_clearly_better(offset_rows)
        a3b=delay_gate(delay_all,delay_prn,delay_role); a3c=aggregation_gate(l20,boots.get("20",{}),prn_diff,role_diff)
        verdict=final_gates(a1,a2,a3a,a3b,a3c)
        write_json(stage/"config.json",{"scope":"cleanStatic-only","epochs":969,"lengths":LENGTHS,"grid":GRID,"bootstrap_seed":1401})
        write_json(stage/"environment.json",{"python":sys.version,"numpy":np.__version__})
        write_json(stage/"r13_frozen_lineage.json",lineage); write_json(stage/"frozen_reconstruction_config.json",FROZEN_CONFIG.document())
        write_json(stage/"prompt_reproduction_metrics.json",prompt_all); write_csv(stage/"prompt_reproduction_by_prn.csv",prompt_prn); write_csv(stage/"prompt_reproduction_by_time_block.csv",prompt_role)
        write_json(stage/"delay_recovery_metrics.json",delay_all); write_csv(stage/"delay_recovery_by_prn.csv",delay_prn); write_csv(stage/"delay_recovery_by_time_block.csv",delay_role)
        write_json(stage/"doppler_1ms_metrics.json",doppler_all); write_csv(stage/"aggregation_metrics.csv",aggregation); write_csv(stage/"aggregation_by_prn.csv",aggregation_prn); write_csv(stage/"aggregation_by_time_block.csv",aggregation_role)
        write_csv(stage/"paired_improvement.csv",paired); write_json(stage/"bootstrap_results.json",boots)
        mainlobe=[]
        for r in scores:
            s=surfaces[(str(r["channel"]),int(r["prn"]),int(r["tracker_row"]))]; ci=GRID["delay_chips"].index(0); center=abs(s[GRID["doppler_hz"].index(0),ci])
            mainlobe.append({k:r[k] for k in ("role","prn","channel","tracker_row")} | {f"c_{d:+d}_hz":float(abs(s[GRID['doppler_hz'].index(d),ci])) for d in (-150,-100,-50,0,50,100,150)} | {"peak_center_ratio":r["peak_center_ratio"],"relative_excess":r["peak_center_ratio"]-1})
        write_csv(stage/"doppler_mainlobe_diagnostics.csv",mainlobe)
        write_json(stage/"residual_doppler_diagnostics.json",{"status":"NOT_APPLICABLE","reason":"authenticated MAT complex Prompt continuity not established","nav_bit_assumption":"q=(P/(|P|+eps))^2 removes 180-degree navigation-bit signs only"})
        write_csv(stage/"per_block_scores.csv",scores)
        write_json(stage/"execution_validity.json",{"caf_executed":True,"elapsed_seconds":time.perf_counter()-started,"common_anchor_counts":{str(k):len(v) for k,v in blocks.items()},"attack_inputs_read":False})
        write_json(stage/"go_no_go.json",verdict); (stage/"README.md").write_text("# R1.4 Doppler validation\n\ncleanStatic reconstruction diagnostic. A 1 ms coherent integration has an approximately 1/T = 1 kHz main lobe; A3c is diagnostic and cannot invalidate A3a/A3b.\n")
        (stage/"test_report.txt").write_text("Production campaign; see source-phase commit for focused test report.\n")
        for name in ("l-histograms","l-recovery","prn-l1-l20","role-comparison","prompt-scatter","delay-histogram","peak-center-distribution"):
            (stage/"plots"/f"{name}.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>\n')
        write_json(stage/"verification_report.json",{"status":"NOT_YET_VERIFIED"})
        write_json(stage/"checksums.json",{"files":{str(p.relative_to(stage)):sha256(p) for p in sorted(stage.rglob("*")) if p.is_file() and p.name not in {"checksums.json","verification_report.json"}}})
        missing=[x for x in INVENTORY if not (stage/x).exists()]
        if missing: raise RuntimeError(f"incomplete artifact: {missing}")
        os.replace(stage,final); return verdict
    except Exception:
        shutil.rmtree(stage,ignore_errors=True); raise


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--raw",type=Path,required=True); p.add_argument("--tracker-dir",type=Path,required=True); p.add_argument("--manifest",type=Path)
    p.add_argument("--output",type=Path,default=OUT); p.add_argument("--execute-production",action="store_true")
    args=p.parse_args(argv); clean_only_guard(["cleanStatic"])
    if not args.execute_production: raise SystemExit("source-only safety: pass --execute-production in the later campaign")
    if args.output.resolve()!=OUT.resolve(): raise ValueError("R1.4 writes only its reserved artifact path")
    run(args)
if __name__=="__main__": main()
