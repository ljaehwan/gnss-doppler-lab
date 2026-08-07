"""Bounded, outcome-blind Stage-1 R1 evaluation on authenticated L20 supports."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnss_doppler_lab.acaf_nf_stage1_static_feasibility import (
    CENTER, DELAYS, DOPPLERS, H1_COORDINATES, H1_GRID_SHA256, FS_HZ,
    BASELINE_SELECTORS, binary_metrics, build_h1_template_from_raw_recorrelation,
    calibrate_scores, choose_pooling, dense_complex_caf, learn_diagonal_variance,
    normalize_caf, pool_prns, standardized_score, synthesize_same_prn_second_source,
    two_source_wls,
)

SUPPORT = 25_000
LENGTH = 20
PRNS_PER_BIN = 3
CLEAN_BINS_PER_ROLE = 3
ATTACK_BINS_PER_PHASE = 6
ROLES = ("train", "selection", "calibration", "holdout")
BASELINES = ("power_only", "prompt_magnitude", "epl_3point_complex",
             "fixed_9_delay_tap_complex", "dense_one_source_residual", "dense_two_source_score")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _phase(start: int, phases: dict[str, dict[str, Any]]) -> str | None:
    for name, limits in phases.items():
        if int(limits["start_sample"]) <= start < int(limits["end_sample_exclusive"]):
            return name
    return None


def _evenly(values: list[Any], count: int) -> list[Any]:
    if len(values) <= count: return values
    return [values[i] for i in np.linspace(0, len(values) - 1, count, dtype=int)]


def collect_l20_bins(path: Path, phases: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Stream a tracker CSV and retain one exact L20 per PRN/channel/second."""
    queues: dict[tuple[int, int], deque[dict[str, str]]] = defaultdict(lambda: deque(maxlen=LENGTH))
    previous: dict[tuple[int, int], int] = {}
    seen_second: dict[tuple[int, int], int] = {}
    bins: dict[tuple[str, int], dict[tuple[int, int], list[dict[str, str]]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pair = (int(row["channel"]), int(row["prn"])); start = int(row["raw_start_sample"])
            if pair in previous and start - previous[pair] != SUPPORT: queues[pair].clear()
            previous[pair] = start; queues[pair].append(row)
            if len(queues[pair]) != LENGTH: continue
            window = list(queues[pair]); phase = _phase(int(window[0]["raw_start_sample"]), phases)
            if phase is None or _phase(int(window[-1]["raw_end_sample"]) - 1, phases) != phase: continue
            second = int(window[-1]["raw_end_sample"]) // int(FS_HZ)
            if seen_second.get(pair) == second: continue
            seen_second[pair] = second; bins[(phase, second)][pair] = window
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (phase, second), pairs in bins.items():
        if len(pairs) >= PRNS_PER_BIN:
            selected = dict(sorted(pairs.items())[:PRNS_PER_BIN])
            result[phase].append({"second": second, "pairs": selected})
    for values in result.values(): values.sort(key=lambda x: x["second"])
    return dict(result)


def select_clean_roles(bins: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if len(bins) < len(ROLES) * CLEAN_BINS_PER_ROLE:
        raise ValueError("insufficient cleanStatic L20 time bins")
    chunks = np.array_split(np.asarray(bins, dtype=object), len(ROLES))
    return {role: _evenly(list(chunk), CLEAN_BINS_PER_ROLE) for role, chunk in zip(ROLES, chunks)}


def read_iq(raw: Path, start: int) -> np.ndarray:
    scalars = np.memmap(raw, dtype="<i2", mode="r", offset=start * 4, shape=(SUPPORT * 2,))
    return np.asarray(scalars[0::2], dtype=np.float64) + 1j * np.asarray(scalars[1::2], dtype=np.float64)


def support_surface(raw: Path, row: dict[str, str], transform: Callable[[np.ndarray, dict[str, str]], np.ndarray] | None = None) -> tuple[np.ndarray, float]:
    iq = read_iq(raw, int(row["raw_start_sample"])); iq = transform(iq, row) if transform else iq
    caf = dense_complex_caf(iq, int(row["prn"]), float(row["code_freq_chips"]),
                            float(row["aux1"]), float(row["carrier_doppler_hz"]))
    normalized, _ = normalize_caf(caf)
    return normalized, float(np.mean(np.abs(iq) ** 2))


def aggregate_window(raw: Path, window: list[dict[str, str]], transform: Callable[[np.ndarray, dict[str, str]], np.ndarray] | None = None) -> dict[str, Any]:
    surfaces=[]; powers=[]
    for row in window:
        surface, power = support_surface(raw, row, transform); surfaces.append(surface); powers.append(power)
    prompt = [np.hypot(float(row["prompt_i"]), float(row["prompt_q"])) for row in window]
    return {"surface": np.mean(np.stack(surfaces), axis=0), "power": float(np.mean(powers)),
            "prompt": float(np.mean(prompt)), "row": window[-1]}


def _simple_calibration(values: list[float]) -> dict[str, float]:
    x=np.asarray(values,float); center=float(np.median(x)); scale=max(float(np.quantile(x,.75)-np.quantile(x,.25)),np.finfo(float).eps)
    return {"center":center,"scale":scale}


def _apply_simple(value: float, calibration: dict[str, float]) -> float:
    return (float(value)-calibration["center"])/calibration["scale"]


def model_score(item: dict[str, Any], template0: np.ndarray, templates: dict[Any, Any], variance: np.ndarray) -> dict[str, Any]:
    fit=two_source_wls(item["surface"],template0,templates,variance)
    di,dj=fit["selected_delta"]; ratio=float(abs(fit["h1_beta"])/max(abs(fit["h1_alpha"]),np.finfo(float).eps))
    residual=np.abs(item["surface"]-template0)
    epl=float(np.mean(residual[CENTER[0],[4,8,12]])); nine=float(np.mean(residual[CENTER[0],4:13]))
    return {**fit,"power_only":item["power"],"prompt_magnitude":item["prompt"],
            "epl_3point_complex":epl,"fixed_9_delay_tap_complex":nine,
            "dense_one_source_residual":fit["h0_rss"],"dense_two_source_score":fit["raw_s2src"],
            "delay_chips":float(DELAYS[dj]),"doppler_hz":float(DOPPLERS[di]),"beta_alpha_ratio":ratio,
            "grid_boundary":bool(di in (0,len(DOPPLERS)-1) or dj in (0,len(DELAYS)-1))}


def independent_block_effect(pre: list[dict[str, Any]], post: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    def blocks(rows: list[dict[str, Any]]) -> np.ndarray:
        grouped: dict[int,list[float]]=defaultdict(list)
        for row in rows: grouped[int(float(row["time_s"])//10)].append(float(row["score"]))
        return np.asarray([np.mean(grouped[key]) for key in sorted(grouped)],float)
    a=blocks(pre);b=blocks(post)
    if not len(a) or not len(b): return {"status":"NOT_EVALUATED","reason":"missing_pre_or_post_blocks"}
    rng=np.random.default_rng(seed); estimates=[]
    for _ in range(1000): estimates.append(float(np.mean(rng.choice(b,len(b),True))-np.mean(rng.choice(a,len(a),True))))
    return {"status":"PASS","effect":float(np.mean(b)-np.mean(a)),"ci95":[float(np.quantile(estimates,.025)),float(np.quantile(estimates,.975))],
            "block_seconds":10,"replicates":1000,"seed":seed,"pre_blocks":len(a),"post_blocks":len(b)}


def _refresh_checksums(root: Path) -> None:
    files={p.relative_to(root).as_posix():{"sha256":sha256(p),"size_bytes":p.stat().st_size}
           for p in root.rglob("*") if p.is_file() and p.name not in {"checksums.json","verification_report.json"}}
    dump(root/"checksums.json",{"algorithm":"sha256","files":files})


def run_stage1_r1(root: Path, source_binding: Path) -> Path:
    root=root.resolve(); source_binding=source_binding.resolve(); cfg=json.loads(source_binding.read_text(encoding="utf-8"))
    c_report=json.loads((root/"verification_report.json").read_text(encoding="utf-8"))
    attack=json.loads((root/"attack_tracker_manifest.json").read_text(encoding="utf-8"))
    clean=json.loads((root/"continuous_tracker_manifest.json").read_text(encoding="utf-8"))
    dump(root/"checkpoint_c_verification_report.json",c_report)
    foundation=(c_report.get("checkpoint")=="C" and c_report.get("status")=="PASS"
                and c_report.get("scientific_status")=="CHECKPOINT_C_COMPLETE"
                and attack.get("status")=="CHECKPOINT_C_COMPLETE"
                and clean.get("validation",{}).get("status")=="CONTINUOUS_TRACKER_VALID")
    if not foundation: raise RuntimeError("FOUNDATION_INVALID: checkpoint B/C independent gates are not PASS")
    timeline=json.loads((root/"scenario_timeline.json").read_text(encoding="utf-8"))
    clean_phases={"all":{"start_sample":0,"end_sample_exclusive":Path(cfg["scenarios"]["cleanStatic"]["raw_path"]).stat().st_size//4}}
    clean_bins=collect_l20_bins(root/"continuous_tracker_cleanStatic.csv",clean_phases)["all"]
    clean_roles=select_clean_roles(clean_bins)
    attack_bins={}
    for scenario in ("ds3","ds4","ds7","ds8"):
        found=collect_l20_bins(root/f"continuous_tracker_{scenario}.csv",timeline[scenario]["phases"])
        attack_bins[scenario]={phase:_evenly(values,ATTACK_BINS_PER_PHASE) for phase,values in found.items()}
        if any(len(attack_bins[scenario].get(phase,[]))<ATTACK_BINS_PER_PHASE for phase in timeline[scenario]["phases"]):
            raise RuntimeError(f"FOUNDATION_INVALID: insufficient sampled phase bins for {scenario}")
    raw={name:Path(spec["raw_path"]) for name,spec in cfg["scenarios"].items()}

    clean_items=[]
    for role,bins in clean_roles.items():
        for item in bins:
            for pair,window in item["pairs"].items():
                clean_items.append({"scenario":"cleanStatic","phase":"all","role":role,"second":item["second"],"pair":pair,
                                    "window":window,"aggregate":aggregate_window(raw["cleanStatic"],window)})
    train=[x["aggregate"]["surface"] for x in clean_items if x["role"]=="train"]
    template0=np.mean(np.stack(train),axis=0);variance=learn_diagonal_variance(np.stack(train))
    template_row=next(x for x in clean_items if x["role"]=="train")["window"][0]
    template_iq=read_iq(raw["cleanStatic"],int(template_row["raw_start_sample"])); interval_sha=hashlib.sha256(np.ascontiguousarray(template_iq).view(np.uint8)).hexdigest()
    interval={"start":int(template_row["raw_start_sample"]),"end":int(template_row["raw_end_sample"]),"sha256":interval_sha,
              "recording_sha256":cfg["scenarios"]["cleanStatic"]["raw_sha256"]}
    templates={}
    for coordinate in H1_COORDINATES:
        lineage={"recording_sha256":cfg["scenarios"]["cleanStatic"]["raw_sha256"],"scenario":"cleanStatic","role":"normal_train",
                 "raw_intervals":[interval],"construction_method":"raw_iq_periodic_recorrelation",
                 "algorithm":"dense_complex_caf_periodic_l1ca","version":"1","grid_sha256":H1_GRID_SHA256}
        templates[coordinate]=build_h1_template_from_raw_recorrelation(coordinate,[template_iq],lineage=lineage,
            prn=int(template_row["prn"]),code_freq_chips=float(template_row["code_freq_chips"]),aux1_samples=float(template_row["aux1"]),
            tracker_doppler_hz=float(template_row["carrier_doppler_hz"]))
    for item in clean_items: item["model"]=model_score(item["aggregate"],template0,templates,variance)
    calibration_scores=[x["model"]["raw_s2src"] for x in clean_items if x["role"]=="calibration"]
    calibration=calibrate_scores(calibration_scores)
    for item in clean_items: item["model"]["dense_two_source_score"]=standardized_score(item["model"]["raw_s2src"],calibration)
    selection=[]
    for second in sorted({x["second"] for x in clean_items if x["role"]=="selection"}):
        values={x["pair"][1]:x["model"]["dense_two_source_score"] for x in clean_items if x["role"]=="selection" and x["second"]==second}
        selection.append({"role":"selection","scenario":"cleanStatic","recording_sha256":cfg["scenarios"]["cleanStatic"]["raw_sha256"],"scores":values})
    pooling=choose_pooling(selection,cleanstatic_sha256=cfg["scenarios"]["cleanStatic"]["raw_sha256"])
    baseline_cal={name:_simple_calibration([x["model"][name] for x in clean_items if x["role"]=="calibration"]) for name in BASELINES[:-1]}

    all_items=list(clean_items)
    for scenario,phases in attack_bins.items():
        for phase,bins in phases.items():
            for item in bins:
                for pair,window in item["pairs"].items():
                    aggregate=aggregate_window(raw[scenario],window);model=model_score(aggregate,template0,templates,variance)
                    model["dense_two_source_score"]=standardized_score(model["raw_s2src"],calibration)
                    all_items.append({"scenario":scenario,"phase":phase,"role":"external_evaluation","second":item["second"],
                                      "pair":pair,"window":window,"aggregate":aggregate,"model":model})
    for item in all_items:
        for name in BASELINES[:-1]: item["model"][name]=_apply_simple(item["model"][name],baseline_cal[name])

    window_rows=[]
    for scenario in ("cleanStatic","ds3","ds4","ds7","ds8"):
        scenario_items=[x for x in all_items if x["scenario"]==scenario]
        for second in sorted({x["second"] for x in scenario_items}):
            group=[x for x in scenario_items if x["second"]==second]; phase=group[0]["phase"];role=group[0]["role"]
            pooled={name:pool_prns({x["pair"][1]:x["model"][name] for x in group},pooling)[0] for name in BASELINES}
            diagnostics=pool_prns({x["pair"][1]:x["model"]["dense_two_source_score"] for x in group},pooling)[1]
            representative=max(group,key=lambda x:x["model"]["dense_two_source_score"])["model"]
            window_rows.append({"scenario":scenario,"phase":phase,"role":role,"time_s":float(second),"score":pooled["dense_two_source_score"],
                **pooled,"selected_delay_chips":representative["delay_chips"],"selected_doppler_hz":representative["doppler_hz"],
                "beta_alpha_ratio":representative["beta_alpha_ratio"],"grid_boundary":int(representative["grid_boundary"]),
                "prn_count":diagnostics["prn_count"],"dominant_fraction":diagnostics["dominant_fraction"]})
    calibration_pooled=[r["score"] for r in window_rows if r["scenario"]=="cleanStatic" and r["role"]=="calibration"]
    q99=float(np.quantile(calibration_pooled,.99,method="higher"));q995=float(np.quantile(calibration_pooled,.995,method="higher"))

    scenario_rows=[];phase_rows=[];boot={"schema":"acaf_nf_stage1_r1_bootstrap.v1","scenarios":{}}
    pre_names={"ds3":"pre_onset","ds4":"pre_onset","ds7":"pre_onset","ds8":"pre_onset"}
    for scenario in ("cleanStatic","ds3","ds4","ds7","ds8"):
        rows=[r for r in window_rows if r["scenario"]==scenario]
        phases=sorted({r["phase"] for r in rows})
        for phase in phases:
            selected=[r for r in rows if r["phase"]==phase]
            phase_rows.append({"scenario":scenario,"phase":phase,"n":len(selected),"mean_score":float(np.mean([r["score"] for r in selected])),
                               "alarm_fraction":float(np.mean([r["score"]>=q99 for r in selected])),"prn_coverage_min":min(r["prn_count"] for r in selected)})
        if scenario=="cleanStatic":
            hold=[r for r in rows if r["role"]=="holdout"]
            scenario_rows.append({"scenario":scenario,"pre_onset_fpr":"","attack_detection_rate":"","first_alarm_delay_s":"",
                                  "holdout_fpr":float(np.mean([r["score"]>=q99 for r in hold])),"sustained_alarm_fraction":""})
        else:
            pre=[r for r in rows if r["phase"]==pre_names[scenario]];post=[r for r in rows if r["phase"]!=pre_names[scenario]]
            alarm=[r for r in post if r["score"]>=q99];onset=min(r["time_s"] for r in post)
            scenario_rows.append({"scenario":scenario,"pre_onset_fpr":float(np.mean([r["score"]>=q99 for r in pre])),
                "attack_detection_rate":float(np.mean([r["score"]>=q99 for r in post])),
                "first_alarm_delay_s":float(min(r["time_s"] for r in alarm)-onset) if alarm else "",
                "holdout_fpr":"","sustained_alarm_fraction":float(np.mean([r["score"]>=q99 for r in post]))})
            boot["scenarios"][scenario]=independent_block_effect(pre,post,100+len(boot["scenarios"]))

    labels=[];baseline_values={name:[] for name in BASELINES}
    for row in window_rows:
        if row["scenario"]=="cleanStatic": continue
        labels.append(0 if row["phase"]==pre_names[row["scenario"]] else 1)
        for name in BASELINES: baseline_values[name].append(float(row[name]))
    baseline_rows=[]
    for name in BASELINES:
        metrics=binary_metrics(labels,baseline_values[name],max_fpr=.05)
        baseline_rows.append({"baseline":name,"status":"PASS",**metrics,"lineage":str(BASELINE_SELECTORS.get(name))})

    reference=next(x for x in clean_items if x["role"]=="holdout")
    controls=[]
    def control(name: str, transform: Callable[[np.ndarray,dict[str,str]],np.ndarray], kind: str) -> None:
        aggregate=aggregate_window(raw["cleanStatic"],reference["window"],transform);model=model_score(aggregate,template0,templates,variance)
        controls.append({"control":name,"kind":kind,"score":standardized_score(model["raw_s2src"],calibration),
                         "delay_chips":model["delay_chips"],"doppler_hz":model["doppler_hz"]})
    control("identity",lambda x,r:x,"negative");control("gain_1_7",lambda x,r:x*1.7,"negative")
    control("global_phase_1_2",lambda x,r:x*np.exp(1.2j),"negative")
    control("awgn_5pct",lambda x,r:x+np.random.default_rng(int(r["raw_start_sample"])%2**32).normal(0,.05*np.std(x),len(x)),"negative")
    control("amplitude_only",lambda x,r:x*1.5,"negative")
    for delay,doppler,ratio,phase in ((-.25,-100,.25,.3),(.25,100,.25,1.1),(-.5,150,.5,2.0),(.5,-150,.5,2.7)):
        control(f"second_source_d{delay}_f{doppler}_r{ratio}",lambda x,r,d=delay,f=doppler,a=ratio,p=phase:
                x+synthesize_same_prn_second_source(int(r["prn"]),len(x),d,f,p,a*np.sqrt(np.mean(np.abs(x)**2))),"positive")
    identity=next(r["score"] for r in controls if r["control"]=="identity")
    invariant=max(abs(r["score"]-identity) for r in controls if r["control"] in {"gain_1_7","global_phase_1_2","amplitude_only"}) <= 1e-6
    positive=[r["score"] for r in controls if r["kind"]=="positive"]
    holdout_fpr=float(next(r["holdout_fpr"] for r in scenario_rows if r["scenario"]=="cleanStatic"))
    static_pre=max(float(r["pre_onset_fpr"]) for r in scenario_rows if r["scenario"] in {"ds3","ds4"})
    significant=sum(boot["scenarios"][s].get("ci95",[-1])[0]>0 for s in ("ds3","ds7","ds8"))>=2
    dense_auc=next(r["roc_auc"] for r in baseline_rows if r["baseline"]=="dense_two_source_score")
    power_auc=next(r["roc_auc"] for r in baseline_rows if r["baseline"]=="power_only")
    physics={"clean_holdout_fpr_le_0_02":holdout_fpr<=.02,"ds3_ds4_pre_fpr_le_0_05":static_pre<=.05,
             "gain_phase_invariant":invariant,"positive_control_sensitive":float(np.median(positive))>identity,
             "two_of_three_bootstrap_positive":significant,"not_power_only":dense_auc>power_auc}
    paper={"all_primary_bootstrap_positive":all(boot["scenarios"][s].get("ci95",[-1])[0]>0 for s in ("ds3","ds7","ds8")),
           "static_pre_fpr_le_0_05":static_pre<=.05,"dense_low_fpr_ge_fixed":next(r["partial_auc"] for r in baseline_rows if r["baseline"]=="dense_two_source_score")>=max(r["partial_auc"] for r in baseline_rows if r["baseline"] in {"epl_3point_complex","fixed_9_delay_tap_complex"}),
           "prn_not_dominated":max(r["dominant_fraction"] for r in window_rows)<.8}
    go={"verdict":"PHYSICS_FEASIBILITY_GO" if all(physics.values()) else "PHYSICS_FEASIBILITY_NO_GO",
        "foundation":"FOUNDATION_PASS","PHYSICS_FEASIBILITY_GO":all(physics.values()),"PAPER_CANDIDATE_GO":all(physics.values()) and all(paper.values()),
        "stage2_justified":all(physics.values()),"physics_criteria":physics,"paper_criteria":paper,"B0":"PROVISIONAL_UNAVAILABLE"}

    fields=["scenario","phase","role","time_s","score",*BASELINES,"selected_delay_chips","selected_doppler_hz","beta_alpha_ratio","grid_boundary","prn_count","dominant_fraction"]
    write_csv(root/"per_window_scores.csv",window_rows,fields)
    write_csv(root/"scenario_metrics.csv",scenario_rows,["scenario","pre_onset_fpr","attack_detection_rate","first_alarm_delay_s","holdout_fpr","sustained_alarm_fraction"])
    write_csv(root/"phase_metrics.csv",phase_rows,["scenario","phase","n","mean_score","alarm_fraction","prn_coverage_min"])
    write_csv(root/"baseline_metrics.csv",baseline_rows,["baseline","status","roc_auc","average_precision","partial_auc","max_fpr","lineage"])
    write_csv(root/"control_metrics.csv",controls,["control","kind","score","delay_chips","doppler_hz"])
    secondary=[{"scenario":r["scenario"],"phase":r["phase"],"delay_chips":r["selected_delay_chips"],"doppler_hz":r["selected_doppler_hz"],
                "beta_alpha_ratio":r["beta_alpha_ratio"],"grid_boundary":r["grid_boundary"],"prn_count":r["prn_count"]} for r in window_rows]
    write_csv(root/"secondary_component_metrics.csv",secondary,["scenario","phase","delay_chips","doppler_hz","beta_alpha_ratio","grid_boundary","prn_count"])
    dump(root/"normal_model_summary.json",{"status":"PASS","source":"cleanStatic_only","train_windows":sum(x["role"]=="train" for x in clean_items),
        "selection_windows":sum(x["role"]=="selection" for x in clean_items),"calibration_windows":sum(x["role"]=="calibration" for x in clean_items),
        "holdout_windows":sum(x["role"]=="holdout" for x in clean_items),"h1_coordinates":len(templates),"template0_sha256":hashlib.sha256(np.ascontiguousarray(template0).view(np.uint8)).hexdigest(),
        "variance_sha256":hashlib.sha256(np.ascontiguousarray(variance).view(np.uint8)).hexdigest(),"pooling":pooling})
    dump(root/"thresholds.json",{"status":"PASS","source":"cleanStatic_calibration_only","q99":q99,"q99_5":q995,"calibration":calibration,
        "pooling":pooling,"calibration_bins":len(calibration_pooled)})
    dump(root/"bootstrap_results.json",boot);dump(root/"go_no_go.json",go)
    dump(root/"config.json",{"schema":"acaf_nf_stage1_r1_config.v1","source_binding":str(source_binding),"source_binding_sha256":sha256(source_binding),
        "clean_bins_per_role":CLEAN_BINS_PER_ROLE,"attack_bins_per_phase":ATTACK_BINS_PER_PHASE,"prns_per_bin":PRNS_PER_BIN,"window_length":LENGTH,
        "delay_grid":list(DELAYS),"doppler_grid_hz":list(DOPPLERS),"attack_data_used_for_fit":False,"B0":"PROVISIONAL_UNAVAILABLE"})
    dump(root/"execution_validity.json",{"status":"FOUNDATION_PASS_SCIENCE_EXECUTED","foundation_inputs":{"checkpoint_b":"PASS","checkpoint_c":"PASS"},
        "attack_rows_in_fit_or_calibration":0,"caf_executed":True,"h0_h1_wls_executed":True,"full_noncenter_grid":len(templates),
        "raw_supports_scored":len(all_items)*LENGTH,"science_csv_semantics":"numeric","plots":{"count":1,"path":"plots/stage1_r1_scores.png"}})
    plots=root/"plots";plots.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(11,5))
    for scenario in ("cleanStatic","ds3","ds4","ds7","ds8"):
        rows=[r for r in window_rows if r["scenario"]==scenario];ax.plot([r["time_s"] for r in rows],[r["score"] for r in rows],"o-",label=scenario,alpha=.75)
    ax.axhline(q99,color="black",linestyle="--",label="clean q99");ax.set(xlabel="receiver time (s)",ylabel="pooled calibrated S_2src",title="ACAF-NF Stage-1 R1 actual L20 scores")
    ax.legend(ncol=3);fig.tight_layout();fig.savefig(plots/"stage1_r1_scores.png",dpi=150);plt.close(fig)
    readme=("# ACAF-NF Stage-1 R1\n\nActual dense complex CAF/GLRT evaluation on authenticated L=20 continuous tracker supports. "
            "Only cleanStatic train/selection/calibration data selected T0, variance, pooling, and thresholds. Attack pre-onset is external FPR only. "
            f"Final verdict: `{go['verdict']}`. This is a bounded Stage-1 experiment, not a Stage-2 or paper claim. B0 remains `PROVISIONAL_UNAVAILABLE`.\n")
    (root/"README.md").write_text(readme,encoding="utf-8")
    command=[sys.executable,"-m","pytest","-q","tests/test_acaf_nf_stage1_continuous_tracker.py","tests/test_acaf_nf_stage1_static_feasibility.py","tests/test_acaf_nf_stage1_r1.py"]
    result=subprocess.run(command,cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True,env={**__import__("os").environ,"PYTHONPATH":"src","OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
    (root/"test_report.txt").write_text("command: "+" ".join(command)+f"\nexit_code: {result.returncode}\n"+result.stdout+result.stderr,encoding="utf-8")
    if result.returncode: raise RuntimeError("D artifact pytest failed")
    dump(root/"verification_report.json",{"status":"PENDING_INDEPENDENT_VERIFICATION","checkpoint":"D"})
    _refresh_checksums(root)
    return root
