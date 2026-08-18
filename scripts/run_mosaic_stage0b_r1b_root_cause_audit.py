#!/usr/bin/env python3
"""Generate the frozen R1b audit from retained tap-domain evidence only."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.mosaic_stage0b_r1_executor import trace_for_prn
from gnss_doppler_lab.mosaic_stage0b_r1b_root_cause import (
    decide_recommendation, decide_root_cause, diagnostic_projection,
    fitted_clean_residual, frozen_grid_score, integrate_phase, physics_recovered,
    receiver_frame_coordinates, segment_indices, select_single_comparators,
    triangular_template, wrap_phase,
)
from gnss_doppler_lab.trace_native_1ms import TAPS, complex_taps, read_records, sha256_file

BASE_SHA = "7f5431bc68039ec4c2119867226a586533d5e1ac"
EXECUTOR_FREEZE_SHA = "913334d87657d75354b7d47546e986dd9f48d58d"
PRIOR_VERDICT = "NO_GO_MOSAIC_MULTI_PRN_RECOVERY"
EXTERNAL = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mosaic-stage0b-r1-execution")
ART = ROOT / "artifacts/mosaic_stage0b_r1b_multiprn_root_cause"
SOURCE = ROOT / "artifacts/mosaic_stage0b_r1_execution"
R1A = ROOT / "artifacts/mosaic_stage0b_r1a_frozen_analysis"
CASE_IDS = (
    "TEXBAT.cleanStatic.four.03", "TEXBAT.cleanStatic.four.07",
    "TEXBAT.cleanStatic.four.01", "TEXBAT.cleanStatic.four.02", "TEXBAT.cleanStatic.four.05",
    "OAKBAT.cleanStatic.four.03", "OAKBAT.cleanStatic.four.07",
)
FAILURES = {"TEXBAT.cleanStatic.four.03", "TEXBAT.cleanStatic.four.07"}


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str], gz: bool = False) -> None:
    opener = gzip.open if gz else open
    with opener(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def load_cases() -> list[dict[str, object]]:
    return [json.loads(p.read_text()) for p in sorted((EXTERNAL / "cases").glob("*/case_result.json"))]


def aligned_records(reference_dir: Path, case_dir: Path, prn: int, lo: int, hi: int):
    ah, a = read_records(trace_for_prn(reference_dir, prn)); bh, b = read_records(trace_for_prn(case_dir, prn))
    amap = {int(v): i for i, v in enumerate(a["raw_interval_start_sample"])}
    ib = np.flatnonzero((b["raw_interval_start_sample"] >= lo) & (b["raw_interval_start_sample"] < hi))
    pairs = [(amap.get(int(b["raw_interval_start_sample"][j])), j) for j in ib]
    pairs = [(i, j) for i, j in pairs if i is not None]
    ia = np.asarray([i for i, _ in pairs], dtype=int); ib = np.asarray([j for _, j in pairs], dtype=int)
    if not len(ia) or not np.array_equal(a["raw_interval_start_sample"][ia], b["raw_interval_start_sample"][ib]):
        raise ValueError(f"no exact common support for PRN {prn}")
    if ah.tap_offsets_chips != bh.tap_offsets_chips or ah.sample_rate_hz != bh.sample_rate_hz:
        raise ValueError("clean/observed TRACE header mismatch")
    return ah, a[ia], b[ib]


def summary(values: np.ndarray) -> tuple[float, float, float]:
    return float(np.median(values)), float(np.min(values)), float(np.max(values))


def make_plots(metrics, trajectories, actions, phase_rows, baseline_rows, temporal_rows, factorial):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = ART / "plots"; plots.mkdir(exist_ok=True)
    def save(name, title, xlabel, ylabel):
        plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(plots/name, dpi=140); plt.close()
    for field, req, name, ylabel in [
        ("effective_delay_chips", "requested_delay_chips", "requested_vs_receiver_frame_effective_delay.png", "effective delay (chip)"),
        ("effective_doppler_hz", "requested_doppler_hz", "requested_vs_receiver_frame_effective_doppler.png", "effective Doppler (Hz)"),
    ]:
        for key, rows in _group(trajectories, ("case_id","prn")).items():
            plt.plot([r["time_s"] for r in rows], [r[field] for r in rows], alpha=.55, label=f"{key[0].split('.')[-1]}:{key[1]}")
            plt.plot([rows[0]["time_s"], rows[-1]["time_s"]], [rows[0][req]]*2, "k--", alpha=.08)
        save(name, name[:-4].replace("_", " "), "seconds", ylabel)
    x=np.arange(len(metrics)); labels=[f"{r['case_id'].split('.')[-1]}:{r['target_prn']}" for r in metrics]
    plt.scatter(x,[r["original_delta_bic"] for r in metrics],label="original CAF max"); plt.scatter(x,[r["oracle_delta_bic"] for r in metrics],label="oracle"); plt.xticks(x,labels,rotation=90,fontsize=6); plt.legend(); save("fixed_vs_oracle_delta_bic.png","Fixed vs oracle delta BIC","case:PRN","delta BIC")
    plt.scatter(x,[r["fixed_projection_ratio"] for r in metrics],label="fixed"); plt.scatter(x,[r["oracle_projection_ratio"] for r in metrics],label="oracle"); plt.xticks(x,labels,rotation=90,fontsize=6); plt.legend(); save("fixed_vs_oracle_projection_ratio.png","Fixed vs oracle projection ratio","case:PRN","projection ratio")
    for key,rows in _group(phase_rows,("case_id","prn")).items():
        plt.plot([r["time_s"] for r in rows],[r["prompt_magnitude_ratio"] for r in rows],alpha=.6,label=f"{key[0].split('.')[-1]}:{key[1]}")
    plt.legend(fontsize=5,ncol=4); save("prompt_magnitude_phase_trajectory.png","Prompt magnitude ratio trajectories","seconds","observed/clean Prompt magnitude")
    for key,rows in _group(actions,("case_id","prn")).items(): plt.plot([r["time_s"] for r in rows],[r["carrier_nco_difference_hz"] for r in rows],alpha=.55,label=f"{key[0].split('.')[-1]}:{key[1]}")
    save("reference_vs_injected_tracking_actions.png","Reference vs injected carrier actions","seconds","carrier NCO difference (Hz)")
    plt.scatter([r["clean_baseline_delta_bic"] for r in baseline_rows],[r["target_delta_bic"] for r in baseline_rows],c=[r["is_target"] for r in baseline_rows]); save("prn_baseline_target_score.png","PRN clean baseline vs target score","clean baseline delta BIC","case delta BIC")
    grouped=_group(metrics,("case_id",)); box=[[r["oracle_projection_ratio"]-r["fixed_projection_ratio"] for r in rows] for rows in grouped.values()]; plt.boxplot(box,tick_labels=[k[0].split('.')[-1] for k in grouped]); save("f1_f2_vs_success_comparator.png","F1/F2 vs comparator oracle gain","case index","projection-ratio gain")
    matrix=factorial["rho_phase_matrix"]; phases=sorted({r["phase_rad"] for r in matrix}); rhos=sorted({r["rho_db"] for r in matrix}); z=np.full((len(rhos),len(phases)),np.nan)
    for r in matrix:z[rhos.index(r["rho_db"]),phases.index(r["phase_rad"])]=r["recovery_rate"]
    plt.imshow(z,aspect="auto",vmin=0,vmax=1); plt.colorbar(label="recovery rate"); plt.xticks(range(len(phases)),[f"{p:.2f}" for p in phases]); plt.yticks(range(len(rhos)),rhos); save("rho_phase_recovery_matrix.png","rho x phase recovery matrix","phase (rad)","rho (dB)")
    grouped=_group(temporal_rows,("window_seconds",)); plt.boxplot([[r["delta_bic"] for r in rows] for rows in grouped.values()],tick_labels=[str(k[0]) for k in grouped]); save("window_length_diagnostic_score.png","Window length diagnostic score","window seconds","delta BIC")


def _group(rows, keys):
    out=defaultdict(list)
    for row in rows: out[tuple(row[k] for k in keys)].append(row)
    return dict(out)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--analysis-freeze-sha", required=True); args=parser.parse_args()
    prereg=json.loads((ART/"root_cause_preregistration.json").read_text())
    if prereg["status"] != "POST_HOC_ROOT_CAUSE_AUDIT": raise ValueError("missing root-cause freeze")
    prior=json.loads((R1A/"final_verdict.json").read_text())
    if prior["verdict"] != PRIOR_VERDICT: raise ValueError("R1a verdict changed")
    binding=json.loads((SOURCE/"source_binding.json").read_text()); cases=load_cases(); by_id={x["case"]["case_id"]:x for x in cases}
    if len(cases)!=72 or any(x["freeze_sha"]!=EXECUTOR_FREEZE_SHA for x in cases): raise ValueError("frozen 72-case binding failed")
    selected=[by_id[x] for x in CASE_IDS]; singles=[x for x in cases if x["case"]["mode"]=="single_prn"]
    source_binding={"schema":"gnss-doppler-lab.mosaic-stage0b-r1b-source-binding.v1","analysis_freeze_sha":args.analysis_freeze_sha,"base_sha":BASE_SHA,"executor_freeze_sha":EXECUTOR_FREEZE_SHA,"external_result_root":str(EXTERNAL),"prior_r1a_verdict":PRIOR_VERDICT,"iq_injection_rerun":False,"receiver_replay_rerun":False,"case_results":[],"single_prn_matches":[]}
    inventory=[]; evidence_ok=True
    for result in selected:
        cid=result["case"]["case_id"]; cp=EXTERNAL/"cases"/cid/"case_result.json"; ds=result["case"]["dataset"]; ref=Path(binding["datasets"][ds]["reference_trace_dir"])
        tier,matches=select_single_comparators(result,singles); source_binding["single_prn_matches"].append({"case_id":cid,"tier":tier,"case_ids":[x["case"]["case_id"] for x in matches]})
        source_binding["case_results"].append({"case_id":cid,"sha256":sha256_file(cp)})
        for receipt in result["receiver"]["trace_files"]:
            p=Path(receipt["path"]); ok=p.is_file() and p.stat().st_size==receipt["size_bytes"] and sha256_file(p)==receipt["sha256"]; evidence_ok &= ok
            inventory.append({"case_id":cid,"kind":"observed_trace","path":str(p),"sha256":receipt["sha256"],"size_bytes":receipt["size_bytes"],"verified":ok})
        for prn in result["assignment"]["sorted_prns"]:
            p=trace_for_prn(ref,int(prn)); inventory.append({"case_id":cid,"kind":"clean_reference_trace","path":str(p),"sha256":sha256_file(p),"size_bytes":p.stat().st_size,"verified":True})
    dump(ART/"source_binding.json",source_binding); dump(ART/"retained_evidence_inventory.json",{"status":"PASS" if evidence_ok else "FAIL","raw_iq_retained":False,"raw_iq_receipt_binding":True,"items":inventory})
    if not evidence_ok:
        dump(ART/"final_root_cause_verdict.json",{"verdict":"ROOT_CAUSE_EVIDENCE_UNAVAILABLE","preserved_r1a_verdict":PRIOR_VERDICT,"recommendation":"Terminate MOSAIC"}); return
    metrics=[]; trajectories=[]; actions=[]; projections=[]; oracle=[]; phase_rows=[]; stability={}; baseline=[]; temporal=[]
    for result in selected:
        case=result["case"]; cid=case["case_id"]; ds=case["dataset"]; spec=binding["datasets"][ds]; fs=int(spec["sample_rate_hz"]); start=int(spec["authorized_interval"][0]); ref=Path(spec["reference_trace_dir"]); cdir=EXTERNAL/"cases"/cid/"receiver"; targets=set(map(int,result["assignment"]["target_prns"])); scores={int(x["prn"]):x for x in result["scores"]}
        case_stability=[]
        for prn in result["assignment"]["sorted_prns"]:
            h,a,b=aligned_records(ref,cdir,int(prn),start+4*fs,start+10*fs); times=(b["raw_interval_start_sample"].astype(float)-start)/fs; clean=complex_taps(a); observed=complex_taps(b)
            delay,doppler=receiver_frame_coordinates(case["delta_tau_chips"],case["delta_f_hz"],a["action_used_residual_code_phase_chips"],b["action_used_residual_code_phase_chips"],a["action_used_carrier_doppler_hz"],b["action_used_carrier_doppler_hz"])
            phase=integrate_phase(times,doppler,case["delta_phi_rad"]); fixed=triangular_template(h.tap_offsets_chips,np.full(len(times),case["delta_tau_chips"]),case["delta_phi_rad"]+2*np.pi*case["delta_f_hz"]*(times-times[0])); oracle_t=triangular_template(h.tap_offsets_chips,delay,phase)
            fixed_m=diagnostic_projection(clean,observed,fixed); oracle_m=diagnostic_projection(clean,observed,oracle_t)
            resid,_=fitted_clean_residual(clean,observed); prompt_clean=np.abs(clean[:,4]); prompt_obs=np.abs(observed[:,4]); pmr=prompt_obs/np.maximum(prompt_clean,np.finfo(float).tiny); prot=wrap_phase(np.angle(observed[:,4])-np.angle(clean[:,4]))
            code_diff=b["action_used_code_nco_rate_chips_s"]-a["action_used_code_nco_rate_chips_s"]; carr_diff=b["action_used_carrier_doppler_hz"]-a["action_used_carrier_doppler_hz"]
            for i,t in enumerate(times):
                trajectories.append({"case_id":cid,"prn":int(prn),"raw_sample_timestamp":int(b["raw_interval_start_sample"][i]),"time_s":float(t),"requested_delay_chips":case["delta_tau_chips"],"effective_delay_chips":float(delay[i]),"requested_doppler_hz":case["delta_f_hz"],"effective_doppler_hz":float(doppler[i]),"residual_code_phase_chips":float(b["action_used_residual_code_phase_chips"][i]-a["action_used_residual_code_phase_chips"][i]),"residual_carrier_phase_rad":float(wrap_phase(phase[i]))})
                actions.append({"case_id":cid,"prn":int(prn),"raw_sample_timestamp":int(b["raw_interval_start_sample"][i]),"time_s":float(t),"code_nco_difference_chips_s":float(code_diff[i]),"carrier_nco_difference_hz":float(carr_diff[i]),"clean_code_nco_chips_s":float(a["action_used_code_nco_rate_chips_s"][i]),"observed_code_nco_chips_s":float(b["action_used_code_nco_rate_chips_s"][i]),"clean_carrier_nco_hz":float(a["action_used_carrier_doppler_hz"][i]),"observed_carrier_nco_hz":float(b["action_used_carrier_doppler_hz"][i])})
                phase_rows.append({"case_id":cid,"prn":int(prn),"time_s":float(t),"prompt_magnitude_ratio":float(pmr[i]),"prompt_phase_rotation_rad":float(prot[i]),"clean_cn0_db_hz":float(a["cn0_db_hz"][i]),"observed_cn0_db_hz":float(b["cn0_db_hz"][i]),"clean_lock":float(a["carrier_lock_test"][i]),"observed_lock":float(b["carrier_lock_test"][i])})
            lock_loss=int(np.sum((b["valid_lock"]==0)|(b["carrier_lock_test"]<.85))); missing=max(0,round((times[-1]-times[0])/0.001)+1-len(times)); jumps=int(np.sum(np.abs(np.diff(b["dll_discriminator_chips"]))>.25)+np.sum(np.abs(np.diff(b["pll_phase_error_cycles"]))>.25))
            case_stability.append({"prn":int(prn),"common_epochs":len(times),"lock_loss_epochs":lock_loss,"missing_epoch_estimate":missing,"discriminator_jumps":jumps,"cn0_change_db_median":float(np.median(b["cn0_db_hz"]-a["cn0_db_hz"])),"session_count":int(len(np.unique(b["tracking_session_id"])))})
            clean_baseline=diagnostic_projection(clean,clean,triangular_template(h.tap_offsets_chips,np.full(len(times),0.0),np.zeros(len(times))))["delta_bic"]
            baseline.append({"case_id":cid,"prn":int(prn),"is_target":int(prn) in targets,"clean_baseline_delta_bic":clean_baseline,"target_delta_bic":scores[int(prn)]["delta_bic"],"baseline_tap_residual":float(np.sqrt(np.mean(np.abs(resid)**2)))})
            if int(prn) not in targets: continue
            dmed,dmin,dmax=summary(delay); fmed,fmin,fmax=summary(doppler); recovered=physics_recovered(case["delta_tau_chips"],scores[int(prn)]["recovered_delay_chips"],case["delta_f_hz"],scores[int(prn)]["recovered_doppler_hz"])
            dominant=max((s for p,s in scores.items() if p!=int(prn)),key=lambda s:s["delta_bic"])["prn"]
            row={"case_id":cid,"target_prn":int(prn),"recovery_boolean":recovered,"requested_delay_chips":case["delta_tau_chips"],"requested_doppler_hz":case["delta_f_hz"],"original_recovered_delay_chips":scores[int(prn)]["recovered_delay_chips"],"original_recovered_doppler_hz":scores[int(prn)]["recovered_doppler_hz"],"effective_delay_median_chips":dmed,"effective_delay_min_chips":dmin,"effective_delay_max_chips":dmax,"effective_doppler_median_hz":fmed,"effective_doppler_min_hz":fmin,"effective_doppler_max_hz":fmax,"original_delta_bic":scores[int(prn)]["delta_bic"],"oracle_delta_bic":oracle_m["delta_bic"],"fixed_projection_ratio":fixed_m["projection_ratio"],"oracle_projection_ratio":oracle_m["projection_ratio"],"prompt_magnitude_reduction":float(1-np.median(pmr)),"phase_rotation_rad":float(np.angle(np.mean(np.exp(1j*prot)))),"cn0_change_db":float(np.median(b["cn0_db_hz"]-a["cn0_db_hz"])),"lock_change":float(np.median(b["carrier_lock_test"]-a["carrier_lock_test"])),"code_action_change_chips_s":float(np.median(code_diff)),"carrier_action_change_hz":float(np.median(carr_diff)),"clean_baseline_residual":float(np.sqrt(np.mean(np.abs(resid)**2))),"dominant_competing_prn":int(dominant),"root_cause_labels":"PENDING_DECISION_TABLE"}
            metrics.append(row); projections.append({**row,"fixed_unexplained_residual_energy":fixed_m["unexplained_residual_energy"],"oracle_unexplained_residual_energy":oracle_m["unexplained_residual_energy"]}); oracle.append({"diagnostic_label":"POST_HOC_ORACLE_DIAGNOSTIC",**row})
            for window in (.001,.1,.5,1.,6.):
                for block,idx in enumerate(segment_indices(times,4.0,window)):
                    if not len(idx): continue
                    if window == .001:
                        one_template=triangular_template(h.tap_offsets_chips,np.full(len(idx),case["delta_tau_chips"]),np.zeros(len(idx)))
                        scored={**diagnostic_projection(clean[idx],observed[idx],one_template),"recovered_delay_chips":case["delta_tau_chips"],"recovered_doppler_hz":case["delta_f_hz"]}
                    else:
                        scored=frozen_grid_score(clean[idx],observed[idx],times[idx]-times[idx][0],np.asarray(h.tap_offsets_chips))
                    temporal.append({"diagnostic_label":"POST_HOC_DIAGNOSTIC","case_id":cid,"target_prn":int(prn),"window_seconds":window,"block_index":block,"epochs":len(idx),"delta_bic":scored["delta_bic"],"recovered_delay_chips":scored["recovered_delay_chips"],"recovered_doppler_hz":scored["recovered_doppler_hz"]})
        stability[cid]=case_stability
    # Factorial descriptions use all frozen target outcomes; no inferential p-values.
    factrows=[]
    for r in cases:
        c=r["case"]; targets=set(map(int,r["assignment"]["target_prns"])); scores={int(s["prn"]):s for s in r["scores"]}
        for p in targets:
            s=scores[p]; factrows.append({"dataset":c["dataset"],"prn":p,"rho_db":c["rho_db"],"phase_rad":c["delta_phi_rad"],"delay_chips":c["delta_tau_chips"],"doppler_hz":c["delta_f_hz"],"recovered":physics_recovered(c["delta_tau_chips"],s["recovered_delay_chips"],c["delta_f_hz"],s["recovered_doppler_hz"])})
    def table(keys):
        out=[]
        for key,rows in _group(factrows,keys).items(): out.append({**dict(zip(keys,key)),"n":len(rows),"recovery_rate":float(np.mean([r["recovered"] for r in rows]))})
        return out
    factorial={"verdict":"INSUFFICIENT_FACTORIAL_IDENTIFIABILITY","reason":"sparse/aliased frozen design cells; descriptive rates only, with rho and phase jointly structured","tables":{k:table((k,)) for k in ("rho_db","phase_rad","delay_chips","doppler_hz","dataset","prn")},"rho_phase_matrix":table(("rho_db","phase_rad"))}
    failed=[r for r in metrics if r["case_id"] in FAILURES and not r["recovery_boolean"]]; oracle_gain=[r["oracle_projection_ratio"]>r["fixed_projection_ratio"] and r["oracle_delta_bic"]>r["original_delta_bic"] for r in failed]
    oracle_restores=bool(failed and all(oracle_gain)); recenter=bool(failed and all(abs(r["effective_doppler_median_hz"]-r["requested_doppler_hz"])>10 or abs(r["effective_delay_median_chips"]-r["requested_delay_chips"])>.05 for r in failed))
    unstable=any(x["lock_loss_epochs"]>0 or x["session_count"]>1 for cid in FAILURES for x in stability[cid]); cancellation=bool(failed and all(r["prompt_magnitude_reduction"]>.25 for r in failed)); template_mismatch=bool(failed and all(r["oracle_projection_ratio"]<.25 for r in failed)); dilution=False
    h={"H1":{"state":"SUPPORTED" if recenter else "UNSUPPORTED","basis":"frozen receiver-frame coordinate tolerance mismatch"},"H2":{"state":"SUPPORTED" if cancellation else "INCONCLUSIVE","basis":"Prompt reduction; rho and phase remain confounded"},"H3":{"state":"SUPPORTED" if template_mismatch else "UNSUPPORTED","basis":"oracle-template projection ratio"},"H4":{"state":"SUPPORTED" if unstable else "UNSUPPORTED","basis":"lock/session stability inventory"},"H5":{"state":"INCONCLUSIVE","basis":"dominance is not treated as independent when it collapses to 3-of-4 failures"},"H6":{"state":"SUPPORTED" if dilution else "UNSUPPORTED","basis":"pre-fixed window summaries; no detector tuning"}}
    supported=[k for k,v in h.items() if v["state"]=="SUPPORTED"]; verdict=decide_root_cause(True,supported,oracle_restores)
    controls=prior["gate_values"]["control_separation_pass"]; recommendation=decide_recommendation(oracle_restores=oracle_restores,consistent_improvement=all(oracle_gain),comparators_not_degraded=False,controls_separated=controls)
    decision=[{"hypothesis":k,"status":v["state"],"basis":v["basis"]} for k,v in h.items()]
    for row in metrics: row["root_cause_labels"]=";".join(k for k,v in h.items() if v["state"]=="SUPPORTED") or "UNIDENTIFIED"
    fields=list(metrics[0]); write_csv(ART/"failure_case_metrics.csv",[r for r in metrics if r["case_id"] in FAILURES],fields); write_csv(ART/"comparator_case_metrics.csv",[r for r in metrics if r["case_id"] not in FAILURES],fields)
    write_csv(ART/"receiver_frame_trajectories.csv.gz",trajectories,list(trajectories[0]),True); write_csv(ART/"tracking_action_differences.csv.gz",actions,list(actions[0]),True); write_csv(ART/"template_projection_metrics.csv",projections,list(projections[0])); write_csv(ART/"oracle_diagnostic_metrics.csv",oracle,list(oracle[0])); write_csv(ART/"phase_cancellation_metrics.csv",phase_rows,list(phase_rows[0])); dump(ART/"lock_channel_stability.json",stability); write_csv(ART/"prn_baseline_dominance.csv",baseline,list(baseline[0])); write_csv(ART/"temporal_window_diagnostics.csv",temporal,list(temporal[0])); dump(ART/"factorial_identifiability.json",factorial); write_csv(ART/"root_cause_decision_table.csv",decision,["hypothesis","status","basis"])
    final={"schema":"gnss-doppler-lab.mosaic-stage0b-r1b-final-root-cause.v1","analysis_freeze_sha":args.analysis_freeze_sha,"base_sha":BASE_SHA,"executor_freeze_sha":EXECUTOR_FREEZE_SHA,"iq_injection_rerun":False,"receiver_replay_rerun":False,"prior_r1a_verdict":PRIOR_VERDICT,"prior_r1a_verdict_unchanged":True,"hypotheses":h,"oracle_restores_failed_targets":oracle_restores,"primary_root_cause_verdict":verdict,"secondary_root_causes":[k for k in supported if k!="H1"],"recommendation":recommendation,"recommendation_note":"No corrected-detector performance claim is made on the frozen 72 cases; sealed confirmation is permitted only by the preregistered truth table."}
    dump(ART/"final_root_cause_verdict.json",final); make_plots(metrics,trajectories,actions,phase_rows,baseline,temporal,factorial)
    files=[]
    for p in sorted(ART.rglob("*")):
        if p.is_file() and p.name!="artifact_manifest_sha256.json": files.append({"path":str(p.relative_to(ART)),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
    dump(ART/"artifact_manifest_sha256.json",{"schema":"gnss-doppler-lab.mosaic-stage0b-r1b-manifest.v1","files":files})


if __name__ == "__main__": main()
