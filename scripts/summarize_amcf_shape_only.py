#!/usr/bin/env python3
"""Verify and deterministically recompute a finalized AMCF Shape-Only artifact."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location("amcf_shape_runner_contract",ROOT/"scripts/run_amcf_shape_only.py")
assert _spec and _spec.loader
runner=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(runner)


def _rows(path: Path) -> list[dict[str,str]]:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip(): return []
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def _float(value: Any) -> float | None:
    if value in (None,""):return None
    x=float(value); return x if math.isfinite(x) else None


def _bool(value: Any) -> bool:
    if isinstance(value,bool):return value
    return str(value).strip().lower() in {"1","true","yes"}


def _equal(a: Any,b: Any,tol: float=1e-12) -> bool:
    if a is None or b is None:return a is None and b is None
    if isinstance(a,str) or isinstance(b,str):return str(a)==str(b)
    try:return math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol)
    except (ValueError,TypeError):return a==b


def _metric_match(saved: dict[str,str], actual: dict[str,Any]) -> None:
    for key in ("clean_test_fpr","stable_pre_fpr","post_detection","persistent_detection","roc_auc","pr_auc","sustained3_delay_s"):
        expected=saved.get(key)
        if expected in (None,""): expected=None
        if not _equal(expected,actual.get(key)):
            raise ValueError(f"metric recomputation mismatch {saved.get('scenario')} {key}: {expected!r} != {actual.get(key)!r}")


def _verify_decision_digests(out:Path,decision:dict[str,Any]) -> None:
    paths={"metrics":"scenario_metrics.csv","paired":"paired_comparisons.csv","convergence":"convergence_audit.json","schema":"feature_schema.json","feature_audit":"feature_audit.json"}
    if set(decision.get("source_digests",{})) != set(paths):raise ValueError("GO decision lacks exact source digests")
    for key,rel in paths.items():
        if decision["source_digests"][key] != runner.sha256(out/rel):raise ValueError(f"GO source digest mismatch: {key}")


def _verify_schema(out:Path) -> dict[str,Any]:
    schema=json.loads((out/"feature_schema.json").read_text(encoding="utf-8"))
    reps=schema.get("representations",{})
    if reps.get("complex",{}).get("dimensions") != list(runner.COMPLEX_SCHEMA):raise ValueError("Complex feature schema mismatch")
    if reps.get("magnitude",{}).get("dimensions") != list(runner.MAGNITUDE_SCHEMA):raise ValueError("Magnitude feature schema mismatch")
    forbidden={"cn0","c_n0","context","valid_fraction","prompt_magnitude","log_prompt"}
    declared={str(x).lower() for r in reps.values() for x in r.get("dimensions",[])}
    if declared&forbidden:raise ValueError("forbidden field in feature schema")
    if schema.get("tap_order") != list(runner.TAP_NAMES) or schema.get("prompt_index")!=4 or schema.get("tensor_side_count")!=8:raise ValueError("tap order/schema metadata mismatch")
    return schema


def _verify_alarms(rows:list[dict[str,str]],thresholds:dict[str,Any],mode:str) -> None:
    variants={}
    if mode=="synthetic-smoke":
        variants={"complex":("score_complex_ensemble","alarm_complex_q99","alarm_complex_q995",thresholds["Complex all9"]),
                  "magnitude":("score_magnitude_ensemble","alarm_magnitude_q99","alarm_magnitude_q995",thresholds["Magnitude all9"]),
                  "B0":("score_B0_Exact","alarm_B0_q99","alarm_B0_q995",thresholds["B0 Exact"])}
    else:
        for v in ("complex_all9","magnitude_all9","complex_EPL","magnitude_EPL"):
            variants[v]=(f"score_{v}_ensemble",f"alarm_{v}_q99",f"alarm_{v}_q995",thresholds[v])
        variants["B0"]=("score_B0_Exact","alarm_B0_q99","alarm_B0_q995",thresholds["B0 Exact"])
    for name,(score,a99,a995,th) in variants.items():
        for row in rows:
            if score not in row:continue
            value=float(row[score])
            if _bool(row[a99]) != (value>float(th["q99"])) or _bool(row[a995]) != (value>float(th["q995"])):
                raise ValueError(f"saved alarm recomputation mismatch: {name}")


def _primary_criteria(metrics:list[dict[str,str]],seeds:list[dict[str,str]],paired:list[dict[str,str]],feature_audit:dict[str,Any],convergence:dict[str,Any]) -> dict[str,bool]:
    names=tuple(runner.ONSETS)
    def pick(s,m):
        rows=[r for r in metrics if r.get("scenario")==s and r.get("model")==m and r.get("operating_point")=="q99"]
        if len(rows)!=1:raise ValueError(f"exact primary metric row required: {s} {m}")
        return rows[0]
    c=[pick(s,"Complex all9") for s in names]; m=[pick(s,"Magnitude all9") for s in names]; b=[pick(s,"B0 Exact") for s in names]
    ci=[]
    for s in names:
        rows=[r for r in paired if r.get("scenario")==s and r.get("comparator")=="Magnitude all9" and r.get("metric")=="roc_auc"]
        if len(rows)!=1:raise ValueError("exact Complex-Magnitude ROC paired evidence required")
        ci.append(float(rows[0]["ci_low"]))
    directions={}
    for s in names:
        n=0
        for seed in runner.SEEDS:
            cx=[r for r in seeds if r.get("scenario")==s and r.get("model")=="Complex all9" and int(r["seed"])==seed]
            mg=[r for r in seeds if r.get("scenario")==s and r.get("model")=="Magnitude all9" and int(r["seed"])==seed]
            if len(cx)!=1 or len(mg)!=1:raise ValueError("exact same-seed metric evidence required")
            n+=float(cx[0]["roc_auc"])>float(mg[0]["roc_auc"])
        directions[s]=n
    beats=0
    for x,z in zip(c,b):
        beats+=((float(x["roc_auc"])-float(z["roc_auc"])>=.02 or float(x["post_detection"])-float(z["post_detection"])>=.05) and float(x["stable_pre_fpr"])-float(z["stable_pre_fpr"])<=.01)
    return {"stable_pre_fpr_all_below_0.05":all(float(x["stable_pre_fpr"])<.05 for x in c),
            "complex_auc_gt_magnitude_4_of_5":sum(float(x["roc_auc"])>float(y["roc_auc"]) for x,y in zip(c,m))>=4,
            "auc_bootstrap_ci_lower_gt_zero_3_of_5":sum(x>0 for x in ci)>=3,
            "same_seed_direction_each_scenario":all(x>=2 for x in directions.values()),
            "beats_b0_with_fpr_guard_3_of_5":beats>=3,
            "all_required_seeds_converged":convergence.get("exact_three_converged_per_variant") is True,
            "ds7_ds8_no_collapse":feature_audit.get("pass") is True and set(feature_audit.get("scenarios",{}))=={"DS7","DS8"}}


def verify_and_summarize(out:Path|str) -> dict[str,Any]:
    out=Path(out)
    # Hash verification is deliberately first: no parser sees unauthenticated
    # score, schema, audit, checkpoint, or alarm evidence.
    runner.verify_hashes(out)
    missing=[x for x in runner.REQUIRED_INVENTORY if not (out/x).exists()]
    if missing:raise ValueError(f"required artifact inventory missing: {missing}")
    config=json.loads((out/"config.json").read_text(encoding="utf-8")); mode=config.get("mode")
    schema=_verify_schema(out); thresholds=json.loads((out/"thresholds.json").read_text(encoding="utf-8")); decision=json.loads((out/"decision.json").read_text(encoding="utf-8")); convergence=json.loads((out/"convergence_audit.json").read_text(encoding="utf-8")); feature_audit=json.loads((out/"feature_audit.json").read_text(encoding="utf-8"))
    _verify_decision_digests(out,decision)
    if thresholds.get("primary")!="q99" or thresholds.get("q995_role")!="diagnostic_only" or thresholds.get("comparison")!="strict_greater":raise ValueError("threshold primary/diagnostic contract mismatch")
    metrics=_rows(out/"scenario_metrics.csv"); seeds=_rows(out/"seed_metrics.csv"); paired=_rows(out/"paired_comparisons.csv")
    metric_index={(r.get("scenario"),r.get("model"),r.get("operating_point")):r for r in metrics}
    checked=0
    for path in sorted((out/"per_epoch").glob("*.csv")):
        if path.name=="cleanStatic_calibration.csv":continue
        scenario=path.stem; rows=_rows(path); _verify_alarms(rows,thresholds,mode)
        if not rows:continue
        score_col="score_complex_ensemble" if mode=="synthetic-smoke" else "score_complex_all9_ensemble"
        alarm99="alarm_complex_q99" if mode=="synthetic-smoke" else "alarm_complex_all9_q99"; alarm995="alarm_complex_q995" if mode=="synthetic-smoke" else "alarm_complex_all9_q995"
        th=thresholds["Complex all9"] if mode=="synthetic-smoke" else thresholds["complex_all9"]
        generic=[{"decision_time_s":r["decision_time_s"],"source_start":r["source_start"],"source_end":r["source_end"],"score_ensemble":r[score_col],"alarm_q99":r[alarm99],"alarm_q995":r[alarm995]} for r in rows]
        actual=runner.recompute_scenario_metrics(scenario,generic,float(th["q99"]),float(th["q995"]),onset_s=runner.ONSETS.get(scenario))
        saved=metric_index.get((scenario,"Complex all9","q99"))
        if saved is None:raise ValueError(f"missing recomputable metric row: {scenario}")
        _metric_match(saved,actual); checked+=1
    if checked!=6:raise ValueError("all six scenario metric/alarm files must independently recompute")
    if mode=="synthetic-smoke":criteria=runner._smoke_criteria(); status="SMOKE-NO-GO"
    else:criteria=_primary_criteria(metrics,seeds,paired,feature_audit,convergence); status="PRIMARY COMPLETE"
    final="GO" if all(criteria.values()) else "NO-GO"
    if decision.get("primary_quantile")!="q99" or decision.get("primary_decision")!=final or decision.get("criteria")!=criteria:raise ValueError("deterministic q99 GO recomputation mismatch")
    expected=runner.render_readme(final,status,criteria).encode(); actual=(out/"README.md").read_bytes()
    if expected!=actual:raise ValueError("README is not byte-identical to deterministic regeneration")
    # Checkpoint presence and checkpoint digests are independently covered by
    # hashes.json; convergence evidence must name every primary variant/seed.
    audits=convergence.get("audits",{})
    expected_audits={f"{rep}_{obj}_seed{seed}" for rep in ("complex","magnitude") for obj in ("all9","EPL") for seed in runner.SEEDS}
    if set(audits)!=expected_audits:raise ValueError("convergence audit model inventory mismatch")
    for key,row in audits.items():
        model=out/"models"/f"{key}.pt"
        if not model.is_file() or row.get("checkpoint_sha256")!=runner.sha256(model):raise ValueError(f"checkpoint audit hash mismatch: {key}")
        if not row.get("converged",False) and not row.get("excluded_from_ensemble",False):raise ValueError("nonconverged model was not excluded")
    return {"schema":"gnss-doppler-lab.amcf-shape-only-summary-audit.v1","hashes_verified":True,"byte_identical":True,"metrics_recomputed":checked,"alarms_recomputed":True,"primary_quantile":"q99","primary_decision":final,"amcf_wcl":"GO candidate" if final=="GO" else "AMCF WCL no-go","criteria":criteria,"mode":mode}


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("artifact",type=Path);p.add_argument("--audit-out",type=Path);a=p.parse_args(argv); report=verify_and_summarize(a.artifact)
    text=json.dumps(report,sort_keys=True,indent=2)+"\n"
    if a.audit_out:a.audit_out.write_text(text,encoding="utf-8",newline="\n")
    print(json.dumps(report,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
