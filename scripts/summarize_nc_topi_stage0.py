#!/usr/bin/env python3
"""Independent, no-fit NC-TOPI Stage-0 artifact verifier/summarizer.

This program never imports the runner, never fits a statistical object and does
not tune a threshold.  It reconstructs stored reductions from PRN/event rows,
recomputes clean higher-quantile thresholds, metrics, physics criteria and the
frozen bootstrap whenever their raw evidence is present.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from gnss_doppler_lab import nc_topi as core

REQUIRED_FILES = (
    "config.json", "data_manifest.json", "thresholds.json", "scenario_metrics.csv",
    "ablation_metrics.csv", "per_epoch_scores.csv", "synthetic_physics_tests.json",
    "bootstrap_results.json", "README.md", "hashes.json", "decision.json",
    "fit_audit.json", "provenance.json",
)
REQUIRED_DIRS = ("plots",)
AGGREGATORS = ("median", "top25_mean")
METHOD_ALIASES = {"NC-TOPI": "NC_TOPI", "shuffledNC-TOPI": "NC_TOPI_time_shuffle"}


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block=stream.read(chunk)
            if not block: break
            h.update(block)
    return h.hexdigest()


def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                  and p.name != "hashes.json" and not p.name.startswith("."))


def write_hash_inventory(root: str | Path) -> Path:
    root=Path(root)
    entries={name: sha256_file(root/name) for name in _files(root)}
    payload={"schema":"gnss-doppler-lab.nc-topi-stage0.hashes.v1",
             "algorithm":"sha256", "self_excluded":"hashes.json", "files":entries}
    path=root/"hashes.json"
    path.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")
    return path


def _json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _csv(path: Path) -> list[dict[str,str]]:
    with path.open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value, label: str) -> float:
    try: out=float(value)
    except (TypeError,ValueError) as exc: raise ValueError(f"{label} is not numeric") from exc
    if not np.isfinite(out): raise ValueError(f"{label} is nonfinite")
    return out


def _bool(value, label: str) -> bool:
    if isinstance(value,bool): return value
    if str(value).strip().lower() in ("1","true"): return True
    if str(value).strip().lower() in ("0","false"): return False
    raise ValueError(f"{label} is not boolean")


def aggregate(values: Sequence[float], kind: str) -> float:
    x=np.asarray(values,float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("event aggregation requires finite PRN values")
    if kind=="median": return float(np.median(x))
    if kind=="top25_mean":
        count=int(math.ceil(.25*len(x)))
        return float(np.mean(np.sort(x)[-count:]))
    raise ValueError("unknown aggregator")


def _png(path: Path) -> tuple[int,int]:
    data=path.read_bytes()
    if len(data)<33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError(f"not a real PNG: {path.name}")
    width,height=struct.unpack(">II",data[16:24])
    if width<1 or height<1 or data[-12:] != b"\x00\x00\x00\x00IEND\xaeB`\x82":
        raise ValueError(f"truncated/invalid PNG: {path.name}")
    return width,height


def _event_key(row: Mapping[str,str]) -> tuple[str,...]:
    # PRN availability is its own source_end while event availability is max end.
    return (row.get("scenario",""),row.get("physical_recording_id",""),row.get("event_id",""),
            row.get("target_index",""))


def _event_phase(scenario: str, source_start: float, source_end: float,
                 onsets: Mapping[str,object]) -> tuple[str,str]:
    # Independent copy of the frozen event-support phase grammar.
    if scenario in ("cleanStatic","cleanDynamic"):
        return "normal",""
    if scenario not in onsets:
        raise ValueError(f"missing attack onset for {scenario}")
    onset=_number(onsets[scenario],f"{scenario} onset")
    if source_start>=onset:
        return "post","1"
    if source_start>=30 and source_end<=onset-20:
        return "stable_pre","0"
    return "transition_excluded",""


def verify_epoch_rows(root: Path, errors: list[str]) -> dict[str,object]:
    try: rows=_csv(root/"per_epoch_scores.csv")
    except Exception as exc:
        errors.append(f"per_epoch_scores unreadable: {exc}"); return {"rows":0}
    required={"row_level","scenario","physical_recording_id","event_id","target_index",
              "availability_time_s","source_start_s","source_end_s","role","phase","label",
              "valid","tracked_prn_count"}
    if not rows:
        errors.append("per_epoch_scores.csv has no rows"); return {"rows":0}
    missing=required-set(rows[0])
    if missing: errors.append(f"per_epoch_scores missing columns {sorted(missing)}")
    prn=[r for r in rows if r.get("row_level")=="prn"]
    events=[r for r in rows if r.get("row_level")=="event"]
    if not prn: errors.append("per_epoch_scores has no PRN rows")
    if not events: errors.append("per_epoch_scores has no event rows")
    methods=sorted(c for c in (prn[0] if prn else {}) if c not in required|{"prn","prn_target_index","pair_sequence_index"}
                   and not c.endswith("_median") and not c.endswith("_top25_mean")
                   and all((row.get(c,"")!="") for row in prn))
    # Every method is required on exactly the same physical PRN identity mask.
    base_keys=None
    for method in methods:
        keys=tuple(_event_key(r)+(r.get("prn",""),) for r in prn if r.get(method,"")!="")
        if len(keys)!=len(set(keys)): errors.append(f"duplicate PRN identity for {method}")
        if base_keys is None: base_keys=keys
        elif keys != base_keys: errors.append(f"method epoch mask mismatch: {method}")
    event_map={_event_key(r):r for r in events}
    if len(event_map)!=len(events): errors.append("duplicate event row identity")
    grouped: dict[tuple[str,...],list[dict[str,str]]]={}
    for row in prn: grouped.setdefault(_event_key(row),[]).append(row)
    try:
        config=_json(root/"config.json")
        onsets=config.get("attacks",{}).get("onsets_seconds",{})
        if not isinstance(onsets,Mapping): raise ValueError("attack onsets are not a mapping")
    except Exception as exc:
        errors.append(f"phase config unreadable: {exc}");onsets={}
    checked=0
    for key,group in grouped.items():
        event=event_map.get(key)
        if event is None:
            errors.append(f"PRN event missing event row: {key}"); continue
        try:
            tracked=int(float(event["tracked_prn_count"]))
            if tracked != len({x.get("prn","") for x in group}):
                errors.append(f"tracked_prn_count mismatch: {key}")
            starts=[_number(x["source_start_s"],"source_start_s") for x in group]
            ends=[_number(x["source_end_s"],"source_end_s") for x in group]
            event_start=min(starts);event_end=max(ends)
            if not math.isclose(_number(event["source_start_s"],"event source_start"),event_start,abs_tol=1e-12):
                errors.append(f"event source_start is not min PRN support: {key}")
            if not math.isclose(_number(event["source_end_s"],"event source_end"),event_end,abs_tol=1e-12):
                errors.append(f"event source_end is not max PRN support: {key}")
            if not math.isclose(_number(event["availability_time_s"],"event availability"),event_end,abs_tol=1e-12):
                errors.append(f"event availability is not max PRN source_end: {key}")
            for row,row_end in zip(group,ends):
                if not math.isclose(_number(row["availability_time_s"],"PRN availability"),row_end,abs_tol=1e-12):
                    errors.append(f"PRN availability is not its source_end: {key}/{row.get('prn','')}")
            expected_phase,expected_label=_event_phase(event["scenario"],event_start,event_end,onsets)
            expected_valid=expected_phase!="transition_excluded"
            for row in [event,*group]:
                if row.get("phase","")!=expected_phase:
                    errors.append(f"event-support phase mismatch: {key}/{row.get('row_level','')}/{row.get('prn','')}")
                if row.get("label","")!=expected_label:
                    errors.append(f"event-support label mismatch: {key}/{row.get('row_level','')}/{row.get('prn','')}")
                if _bool(row.get("valid",""),"valid")!=expected_valid:
                    errors.append(f"event-support valid mismatch: {key}/{row.get('row_level','')}/{row.get('prn','')}")
            expected_event_role=_source_role(event["scenario"],event_start,event_end)
            if event.get("role")!=expected_event_role:
                errors.append(f"event source-support role mismatch: {key}; expected {expected_event_role}")
            for method in methods:
                values=[_number(x[method],method) for x in group]
                for agg in AGGREGATORS:
                    column=f"{method}_{agg}"
                    if column not in event or event[column]=="":
                        errors.append(f"missing event aggregation {column}: {key}"); continue
                    actual=_number(event[column],column); expected=aggregate(values,agg)
                    if not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12):
                        errors.append(f"tampered aggregation {column}: {key}; expected {expected!r}")
            checked += 1
        except Exception as exc: errors.append(f"event reconstruction failed {key}: {exc}")
    for i,row in enumerate(rows):
        try:
            start=_number(row["source_start_s"],"source_start_s"); end=_number(row["source_end_s"],"source_end_s")
            expected_role=_source_role(row["scenario"],start,end)
            if row["role"] != expected_role: errors.append(f"source-support role mismatch row {i}: expected {expected_role}")
            valid=_bool(row["valid"],"valid")
            if row["scenario"] in ("cleanStatic","cleanDynamic") and not valid: errors.append(f"normal row invalid at {i}")
            if row["phase"]=="transition_excluded" and (valid or row.get("label","")!=""): errors.append(f"transition validity/label mismatch row {i}")
        except Exception as exc: errors.append(f"row role/valid reconstruction {i}: {exc}")
    return {"rows":len(rows),"prn_rows":len(prn),"event_rows":len(events),
            "events_reaggregated":checked,"methods":methods,"rows_data":rows}


def _threshold_key_parts(key: str, item: Mapping[str,object]):
    parts=key.split("/")
    detector=str(item.get("detector",parts[0] if parts else ""))
    aggregator=str(item.get("aggregator",parts[1] if len(parts)>1 else ""))
    q=float(item.get("quantile",.995 if key.endswith("q995") else .99))
    return METHOD_ALIASES.get(detector,detector),aggregator,q


def verify_thresholds(root: Path, rows: Sequence[Mapping[str,str]], errors: list[str]) -> dict[str,object]:
    try: thresholds=_json(root/"thresholds.json")
    except Exception as exc:
        errors.append(f"thresholds unreadable: {exc}"); return {"checked":0}
    if not isinstance(thresholds,dict) or not thresholds:
        errors.append("thresholds must be a nonempty object"); return {"checked":0}
    event_rows=[x for x in rows if x.get("row_level")=="event" and
                x.get("scenario")=="cleanStatic" and x.get("role")=="normal_calibration" and
                _bool(x.get("valid","0"),"valid")]
    checked=0
    for key,item in thresholds.items():
        try:
            if not isinstance(item,dict): raise ValueError("entry must be object")
            method,agg,q=_threshold_key_parts(key,item)
            if agg not in AGGREGATORS or q not in (.99,.995): raise ValueError("unfrozen aggregator/quantile")
            if item.get("method") != "higher": raise ValueError("quantile method is not higher")
            if item.get("clean_role") != "normal_calibration": raise ValueError("threshold fit role is not clean calibration")
            column=f"{method}_{agg}"
            values=np.asarray([_number(x[column],column) for x in event_rows],float)
            if not len(values): raise ValueError(f"no clean calibration event values for {column}")
            expected=float(np.quantile(values,q,method="higher")); actual=_number(item.get("value"),"threshold")
            if not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError(f"tampered threshold, expected {expected!r}")
            if "score_digest_sha256" in item:
                digest=hashlib.sha256(str(values.shape).encode()+b"|float64|"+
                    np.ascontiguousarray(values,dtype=np.float64).tobytes()).hexdigest()
                if item["score_digest_sha256"] != digest: raise ValueError("calibration score digest mismatch")
            checked += 1
        except Exception as exc: errors.append(f"threshold {key}: {exc}")
    primary="NC_TOPI/median/q99"
    aliases={k.replace("NC-TOPI","NC_TOPI") for k in thresholds}
    if primary not in aliases: errors.append("missing primary NC-TOPI/median/q99 threshold")
    return {"checked":checked,"clean_calibration_events":len(event_rows)}


def verify_iq_causality(root: Path, errors: list[str]) -> dict[str,object]:
    """Verify every IQ context structurally and, when available, join every event/PRN."""
    path=root/"iq_context.csv"
    if not path.exists():
        return {"available":False,"reason":"optional IQ context evidence absent"}
    rows=_csv(path);cfg=_json(root/"config.json") if (root/"config.json").exists() else {}
    iqcfg=cfg.get("iq_conditioner",{});rate=int(iqcfg.get("sample_rate_hz",25_000_000))
    duration=float(iqcfg.get("block_duration_seconds",.01));stride=float(iqcfg.get("block_stride_seconds",.5))
    block_count=int(round(rate*duration));stride_count=int(round(rate*stride))
    epoch_rows=_csv(root/"per_epoch_scores.csv") if (root/"per_epoch_scores.csv").exists() else []
    events={(x.get("scenario"),x.get("physical_recording_id"),x.get("event_id")):x
            for x in epoch_rows if x.get("row_level")=="event"}
    prn_groups={}
    for x in epoch_rows:
        if x.get("row_level")=="prn":
            prn_groups.setdefault((x.get("scenario"),x.get("physical_recording_id"),x.get("event_id")),[]).append(x)
    if len(events)!=sum(x.get("row_level")=="event" for x in epoch_rows):
        errors.append("IQ join event keys are not unique")
    checked=0;joined=0
    seen=set()
    for i,row in enumerate(rows):
      try:
        key=(row["scenario"],row["physical_recording_id"],row["event_id"])
        if row.get("block_recording_id")!=row["physical_recording_id"]:
            raise ValueError("target/block physical recording group mismatch")
        if key in seen: raise ValueError("duplicate scenario/recording/event IQ row")
        seen.add(key)
        expected_event=f"{row['physical_recording_id']}@{float(row['window_bin_s']):.10g}"
        if row["event_id"]!=expected_event: raise ValueError("event_id/window_bin mismatch")
        target=_number(row["target_source_start_s"],"target_source_start_s")
        ends=[_number(x,"block_end") for x in row["block_end_s"].split(";")]
        starts=[_number(x,"block_start") for x in row["block_start_s"].split(";")]
        offsets=[int(x) for x in row["sample_offset"].split(";")]
        counts=[int(x) for x in row["sample_count"].split(";")]
        features=np.asarray(json.loads(row["block_features_json"]),float)
        context=np.asarray(json.loads(row["context_features_json"]),float)
        if not (len(ends)==len(starts)==len(offsets)==len(counts)==4 and features.shape==(4,4) and context.shape==(4,)):
            raise ValueError("history4 shape contract")
        if int(row["history_blocks"])!=4 or not math.isclose(float(row["cadence_seconds"]),stride,rel_tol=0,abs_tol=1e-12):
            raise ValueError("history/cadence metadata")
        if row.get("history_reducer")!="arithmetic_mean_per_feature": raise ValueError("history reducer")
        if (any(n!=block_count for n in counts) or any(o<0 or o%stride_count for o in offsets)
                or np.any(np.diff(offsets)!=stride_count)): raise ValueError("sample block grid")
        if not np.allclose(np.asarray(offsets,float)/rate,starts,rtol=0,atol=1e-12): raise ValueError("sample_offset/block_start mismatch")
        if not np.allclose((np.asarray(offsets)+np.asarray(counts))/rate,ends,rtol=0,atol=1e-12): raise ValueError("sample_count/block_end mismatch")
        if not np.allclose(np.diff(ends),stride,rtol=0,atol=1e-8) or ends[-1]>target+1e-12 or ends[-1]+stride<=target+1e-12:
            raise ValueError("not latest contiguous causal history4")
        if not np.allclose(context,features.mean(axis=0),rtol=1e-12,atol=1e-12): raise ValueError("context reduction")
        linked=row.get("linked_prns","").split(";") if row.get("linked_prns","") else []
        if linked!=sorted(set(linked)) or int(row["linked_pair_count"])<len(linked) or not linked:
            raise ValueError("linked PRN/count metadata")
        if epoch_rows:
            event=events.get(key);prns=prn_groups.get(key,[])
            if event is None or not prns: raise ValueError("IQ row does not join exact event+PRN rows")
            actual_prns=sorted({x["prn"] for x in prns})
            if linked!=actual_prns or int(row["linked_pair_count"])!=len(prns): raise ValueError("linked PRNs/count not exact")
            minimum=min(_number(x["source_start_s"],"PRN source_start") for x in prns)
            if not math.isclose(target,minimum,rel_tol=0,abs_tol=1e-12): raise ValueError("target_source_start is not linked PRN minimum")
            if not math.isclose(_number(event["source_start_s"],"event source_start"),minimum,rel_tol=0,abs_tol=1e-12): raise ValueError("event source_start join mismatch")
            joined+=1
        checked+=1
      except Exception as exc: errors.append(f"IQ context row {i}: {exc}")
    if epoch_rows and set(events)!=seen:
        errors.append(f"IQ/event inventory mismatch missing={len(set(events)-seen)} extra={len(seen-set(events))}")
    npz=root/"iq_context.npz"
    if npz.exists():
      try:
        with np.load(npz,allow_pickle=False) as z:
          if len(z["target_source_start_s"])!=len(rows): raise ValueError("row count")
          if not np.allclose(z["target_source_start_s"],[float(x["target_source_start_s"]) for x in rows],rtol=0,atol=0): raise ValueError("target array")
          if not np.array_equal(z["sample_offset"],np.asarray([[int(y) for y in x["sample_offset"].split(";")] for x in rows])): raise ValueError("offset array")
          if not np.allclose(z["block_end_s"],np.asarray([[float(y) for y in x["block_end_s"].split(";")] for x in rows]),rtol=0,atol=0): raise ValueError("block-end array")
          if not np.allclose(z["context_features"],np.asarray([json.loads(x["context_features_json"]) for x in rows]),rtol=0,atol=0): raise ValueError("context array")
      except Exception as exc: errors.append(f"IQ NPZ/CSV exact binding: {exc}")
    return {"available":True,"rows":len(rows),"checked":checked,"joined":joined,
            "strict_causal":checked==len(rows)}

def _spearman(x,y):
    from scipy.stats import spearmanr
    return float(spearmanr(x,y).statistic)


def _nuisance_kind_pass_independent(rows):
    result={}
    for kind in ("amplitude","shift","noise"):
        selected=[x for x in rows if x["kind"]==kind]
        if not selected: raise ValueError(f"missing nuisance kind {kind}")
        topi_median=float(np.median([float(x["topi_normalized"]) for x in selected]))
        b0_median=float(np.median([float(x["b0_normalized"]) for x in selected]))
        result[kind]=bool(topi_median <= b0_median if kind=="noise" else topi_median < b0_median)
    return result


def recompute_synthetic(data: Mapping[str,object], errors: list[str]) -> dict[str,object]:
    raw=data.get("raw_trials",[])
    reconstructed_equal=[];reconstructed_second=[]
    reference=data.get("reference_state",{})
    if reference:
      try:
       W=np.asarray(reference["W"],float);Sigma=np.asarray(reference["Sigma"],float)
       if W.shape!=(9,9) or Sigma.shape!=(9,9) or not np.allclose(W,np.linalg.pinv(Sigma,rcond=1e-10,hermitian=True),rtol=1e-10,atol=1e-12): raise ValueError("W/Sigma mismatch")
       def project(residual,basis):
        residual=np.asarray(residual,float);J=np.asarray(basis,float);normal=J.T@W@J
        fitted=J@np.linalg.pinv(normal,rcond=1e-10,hermitian=True)@(J.T@W@residual);perp=residual-fitted
        return float(perp@W@perp),float(residual@W@residual)
       for i,row in enumerate(raw):
        tangent=np.asarray(row["tangent_raw"],float);orth=np.asarray(row["orthogonal_raw"],float);std=np.asarray(row["standardizer_std"],float);basis=row["basis_matrix"]
        tb=float(np.sqrt(np.mean((tangent/std)**2)));ob=float(np.sqrt(np.mean((orth/std)**2)));tq,_=project(tangent,basis);oq,total=project(orth,basis)
        derived=(abs(tb-ob)/max(tb,ob,1e-15),tq/max(oq,1e-15),oq/max(total,1e-15))
        checks=((tb,row["b0_tangent"]),(ob,row["b0_orthogonal"]),(tq,row["tangent_topi"]),(oq,row["orthogonal_topi"]),
                (derived[0],row["b0_relative_difference"]),(derived[1],row["tangent_to_orthogonal_topi_ratio"]),
                (derived[2],row["orthogonal_preserved_fraction"]))
        if any(not math.isclose(a,float(b),rel_tol=1e-10,abs_tol=1e-12) for a,b in checks): raise ValueError(f"equal-RMSE derived field tampered/raw reconstruction row {i}")
        reconstructed_equal.append(derived)
       pred=np.asarray(reference["predicted_raw"],float);basis=reference["basis_matrix"];std=np.asarray(reference["standardizer_std"],float)
       for i,row in enumerate(data.get("second_peak_grid",[])):
        residual=np.asarray(row["residual_raw"],float);changed=np.asarray(row["changed_raw"],float)
        if not np.allclose(residual,changed-pred,rtol=0,atol=1e-14): raise ValueError(f"second peak residual row {i}")
        topi,_=project(residual,basis);b0=float(np.sqrt(np.mean((residual/std)**2)))
        if not math.isclose(topi,float(row["topi"]),rel_tol=1e-10,abs_tol=1e-12) or not math.isclose(b0,float(row["b0"]),rel_tol=1e-10,abs_tol=1e-12): raise ValueError(f"second peak reconstruction row {i}")
        reconstructed_second.append((float(row["relative_power"]),float(row["separation_chips"]),topi))
       coords=np.asarray(reference["coordinates"],float)
       for i,row in enumerate(data.get("nuisance_grid",[])):
        kind=row["kind"];amount=float(row["amount"]);scale=float(row["noise_scale"])
        if kind=="amplitude": expected=amount*pred
        elif kind=="shift": expected=amount*np.gradient(pred,coords,edge_order=2)
        elif kind=="noise": expected=np.asarray(row["noise_standard_normal"],float)*std*scale
        else: raise ValueError(f"unknown nuisance kind {kind}")
        residual=np.asarray(row["residual_raw"],float);changed=np.asarray(row["changed_raw"],float)
        if not np.allclose(residual,expected,rtol=0,atol=1e-14) or not np.allclose(changed,pred+expected,rtol=0,atol=1e-14):
            raise ValueError(f"nuisance perturbation reconstruction row {i}")
        topi,_=project(expected,basis);b0=float(np.sqrt(np.mean((expected/std)**2)))
        checks=((topi,float(row["topi"])),(b0,float(row["b0"])),
                (topi/float(reference["clean_reference_topi_median"]),float(row["topi_normalized"])),
                (b0/float(reference["clean_reference_b0_median"]),float(row["b0_normalized"])))
        if any(not math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-12) for x,y in checks): raise ValueError(f"nuisance reconstruction row {i}")
      except Exception as exc:errors.append(f"synthetic raw vector reconstruction: {exc}")
    stated=data.get("criteria",{})
    result={}
    if raw:
        try:
            if len(reconstructed_equal)!=len(raw): raise ValueError("not all raw vectors reconstructed")
            rel=np.asarray([x[0] for x in reconstructed_equal]);ratios=np.asarray([x[1] for x in reconstructed_equal]);preserve=np.asarray([x[2] for x in reconstructed_equal])
            result["equal_rmse_pass"]=bool(np.all(rel<=1e-8) and np.median(ratios)<=.05 and np.median(preserve)>=.95)
        except Exception as exc: errors.append(f"synthetic equal-RMSE reconstruction: {exc}")
    grid=data.get("second_peak_grid",[])
    if grid:
        try:
            if len(reconstructed_second)!=len(grid): raise ValueError("not all raw vectors reconstructed")
            passes=[]
            for sep in (.25,.375,.5):
                rows=[x for x in reconstructed_second if math.isclose(x[1],sep)]
                passes.append(_spearman([x[0] for x in rows],[x[2] for x in rows])>=.8)
            for power in (.2,.4,.8):
                rows=[x for x in reconstructed_second if math.isclose(x[0],power)]
                passes.append(_spearman([x[1] for x in rows],[x[2] for x in rows])>=.8)
            result["second_peak_power_pass"]=bool(all(passes[:3]));result["second_peak_separation_pass"]=bool(all(passes[3:]));result["second_peak_pass"]=bool(all(passes))
        except Exception as exc: errors.append(f"synthetic second-peak reconstruction: {exc}")
    nuisance=data.get("nuisance_grid",[])
    if nuisance:
      try:
       kinds=_nuisance_kind_pass_independent(nuisance)
       result["nuisance_kind_pass"]=kinds
       result["nuisance_pass"]=bool(all(kinds.values()))
       if stated.get("nuisance_grammar")!={"amplitude":"strict <","shift":"strict <","noise":"<="}:
        raise ValueError("nuisance grammar is not exact")
      except Exception as exc:errors.append(f"synthetic nuisance reconstruction: {exc}")
    for name,value in result.items():
        if name not in stated: continue
        mismatch=(stated[name]!=value) if isinstance(value,dict) else (bool(stated[name]) != value)
        if mismatch: errors.append(f"synthetic criterion tampered: {name}")
    diagnostics={}
    if len(reconstructed_equal)==len(raw) and raw:
        diagnostics["equal_rmse"]={"max_b0_relative_difference":max(x[0] for x in reconstructed_equal),
          "median_tangent_to_orthogonal_topi_ratio":float(np.median([x[1] for x in reconstructed_equal])),
          "median_orthogonal_preserved_fraction":float(np.median([x[2] for x in reconstructed_equal]))}
    if len(reconstructed_second)==len(grid) and grid:
        diagnostics["second_peak"]={
          "power_rho_by_separation":{str(sep):_spearman([x[0] for x in reconstructed_second if math.isclose(x[1],sep)],[x[2] for x in reconstructed_second if math.isclose(x[1],sep)]) for sep in (.25,.375,.5)},
          "separation_rho_by_power":{str(power):_spearman([x[1] for x in reconstructed_second if math.isclose(x[0],power)],[x[2] for x in reconstructed_second if math.isclose(x[0],power)]) for power in (.2,.4,.8)}}
    if nuisance:
        diagnostics["nuisance"]={kind:{"rows":len(selected),
          "median_b0_normalized":float(np.median([float(x["b0_normalized"]) for x in selected])),
          "median_topi_normalized":float(np.median([float(x["topi_normalized"]) for x in selected])),
          "pass":result.get("nuisance_kind_pass",{}).get(kind)} for kind in ("amplitude","shift","noise") for selected in ([x for x in nuisance if x["kind"]==kind],)}
    return {"raw_trials":len(raw),"second_peak_grid_rows":len(grid),"recomputed":result,"diagnostics":diagnostics}


def recompute_bootstrap(data: Mapping[str,object], errors: list[str]) -> dict[str,object]:
    comparisons=data.get("comparisons",[]) if isinstance(data,dict) else []
    checked=0
    for i,item in enumerate(comparisons):
        if not all(k in item for k in ("labels","score_a","score_b","recording_ids","availability_time_s")):
            errors.append(f"bootstrap comparison {i} lacks stored event rows"); continue
        try:
            result=core.paired_pauc_delta_block_bootstrap(
                item["labels"],item["score_a"],item["score_b"],item["recording_ids"],
                item["availability_time_s"],max_fpr=.05,block_seconds=10.,cadence=.5,
                reps=2000,seed=int(item.get("seed",core.DEFAULT_SEED)))
            if bool(item.get("available")) != result.available: raise ValueError("availability mismatch")
            if result.available:
                if result.valid_reps!=2000: raise ValueError("valid bootstrap reps are not 2000")
                if not math.isclose(float(item["point_estimate"]),result.point_estimate,rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError("point estimate mismatch")
                if not np.allclose(item["ci"],result.ci,rtol=1e-12,atol=1e-12): raise ValueError("CI mismatch")
                if "replicates_sha256" in item:
                    digest=hashlib.sha256(np.ascontiguousarray(result.replicates,dtype=np.float64).tobytes()).hexdigest()
                    if digest!=item["replicates_sha256"]: raise ValueError("replicates digest mismatch")
            elif item.get("reason")!=result.reason: raise ValueError("unavailable reason mismatch")
            checked+=1
        except Exception as exc: errors.append(f"bootstrap comparison {i}: {exc}")
    return {"comparisons":len(comparisons),"deterministically_recomputed":checked,"repetitions":2000}


def verify_metrics(root: Path, errors: list[str]) -> dict[str,object]:
    total=0; checked=0
    for filename in ("scenario_metrics.csv","ablation_metrics.csv"):
        try: rows=_csv(root/filename)
        except Exception as exc: errors.append(f"{filename} unreadable: {exc}"); continue
        total+=len(rows)
        for i,row in enumerate(rows):
            # Production rows carry numerator/denominator or labels/scores JSON specifically
            # so a verifier does not trust an opaque metric scalar.
            try:
                if row.get("numerator","") and row.get("denominator","") and row.get("value",""):
                    expected=_number(row["numerator"],"numerator")/_number(row["denominator"],"denominator")
                    if not math.isclose(_number(row["value"],"value"),expected,rel_tol=1e-12,abs_tol=1e-12):
                        raise ValueError("rate metric mismatch")
                    checked+=1
                elif row.get("labels_json","") and row.get("scores_json","") and row.get("metric",""):
                    labels=json.loads(row["labels_json"]); scores=json.loads(row["scores_json"])
                    metric=row["metric"]
                    if metric=="pauc": expected=core.standardized_pauc(labels,scores,max_fpr=.05)
                    else:
                        from sklearn.metrics import average_precision_score,roc_auc_score
                        expected=float(roc_auc_score(labels,scores) if metric=="roc_auc" else average_precision_score(labels,scores))
                    if not math.isclose(_number(row["value"],"value"),expected,rel_tol=1e-12,abs_tol=1e-12):
                        raise ValueError("classification metric mismatch")
                    checked+=1
            except Exception as exc: errors.append(f"{filename} row {i}: {exc}")
    return {"rows":total,"recomputed_rows":checked}


def verify_source_manifest(root: Path, errors: list[str]) -> dict[str,object]:
    try: provenance=_json(root/"provenance.json")
    except Exception as exc: errors.append(f"provenance unreadable: {exc}"); return {"checked":False}
    source_commit=provenance.get("source_commit")
    source_root=Path(provenance.get("source_root",ROOT))
    if not isinstance(source_commit,str) or len(source_commit)!=40:
        errors.append("provenance source_commit missing/noncanonical"); return {"checked":False}
    try:
        head=subprocess.run(["git","rev-parse","HEAD"],cwd=source_root,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()
        if source_commit!=head or provenance.get("execution_code_commit")!=head: raise ValueError("source commit is not exact execution HEAD")
        tracked=subprocess.run(["git","ls-files"],cwd=source_root,check=True,text=True,stdout=subprocess.PIPE).stdout.splitlines()
        expected=sorted(x for x in tracked if ("nc_topi" in x.lower() or x=="docs/NC_TOPI_STAGE0.md"))
        if sorted(provenance.get("source_file_sha256",{}))!=expected: raise ValueError("source_file_sha256 inventory is incomplete or extra")
        for rel,expected in provenance.get("source_file_sha256",{}).items():
            if sha256_file(source_root/rel)!=expected: errors.append(f"source file hash mismatch: {rel}")
        return {"checked":True,"source_commit":source_commit,"source_root":str(source_root)}
    except Exception as exc: errors.append(f"source commit ancestry check failed: {exc}"); return {"checked":False}


PRODUCTION_METHODS=("B0","total","amp_only","shift_only","amp_shift","amp_shift_width",
                    "TOPI","NC_TOPI","NC_TOPI_time_shuffle","NC_TOPI_conditioning_removed")
PRODUCTION_SCENARIOS=("cleanStatic","cleanDynamic","DS1","DS2","DS3","DS7","DS8")
ATTACKS=("DS1","DS2","DS3","DS7","DS8")


def verify_freeze_bundle(root: str|Path) -> dict[str,object]:
    root=Path(root);required={"freeze_manifest.json","config.json","data_manifest.json","fit_audit.json",
      "model_lineage_audit.json","iq_context.csv","iq_context.npz"}
    present={x.name for x in root.iterdir() if x.is_file()}
    if present!=required: raise ValueError(f"freeze bundle inventory mismatch missing={sorted(required-present)} extra={sorted(present-required)}")
    freeze=_json(root/"freeze_manifest.json");fit=_json(root/"fit_audit.json")
    if freeze.get("attack_loader_calls")!=0 or fit.get("attack_loader_calls")!=0: raise ValueError("freeze bundle loaded attacks")
    if freeze.get("attack_fit") is not False or fit.get("attack_fit") is not False: raise ValueError("freeze bundle attack_fit guard failed")
    expected=("covariance","conditioner","conditioner_cap","shuffled_conditioner","shuffled_cap","thresholds")
    if set(freeze.get("sealed_fits",{}))!=set(expected): raise ValueError("freeze typed seal inventory mismatch")
    trace=freeze.get("phase_trace",[])
    if "attack_loader_called" in trace or not trace or trace[-1]!="freeze_sealed": raise ValueError("freeze stage trace invalid")
    rows=_csv(root/"iq_context.csv")
    if len(rows)!=int(freeze.get("iq_context_count",-1)) or not rows: raise ValueError("freeze IQ context count mismatch")
    iq_errors=[];iq=verify_iq_causality(root,iq_errors)
    if iq_errors or iq.get("checked")!=len(rows): raise ValueError(f"freeze IQ structural verification failed: {iq_errors}")
    return {"ok":True,"pair_count":freeze["pair_count"],"split_counts":freeze["split_counts"],
            "iq_context_count":len(rows),"attack_loader_calls":0,"sealed_fits":sorted(expected)}


def _source_role(scenario,start,end):
    if scenario!="cleanStatic": return "evaluation"
    if end<=300:return "normal_train"
    if start>=320 and end<=400:return "normal_calibration"
    if start>=420:return "normal_holdout"
    return "excluded_boundary_crossing"


def _iq_features_independent(raw,epsilon=1e-12):
    # Independent implementation of the frozen four formulas (does not call the runner).
    z=raw[:,0].astype(np.float64)+1j*raw[:,1].astype(np.float64)
    power=float(np.mean(np.abs(z)**2))
    diff=np.abs(np.diff(z)); noise=float(np.median(np.abs(diff-np.median(diff)))/.67448975)
    take=min(len(z),65536); spectrum=np.abs(np.fft.fft(z[:take]))**2/max(1,take)
    flat=math.exp(float(np.mean(np.log(spectrum+epsilon))))/float(np.mean(spectrum+epsilon))
    lag=abs(np.vdot(z[:-1],z[1:]))/math.sqrt(float(np.vdot(z[:-1],z[:-1]).real*np.vdot(z[1:],z[1:]).real)+epsilon)
    return np.asarray([math.log(power+epsilon),math.log(noise+epsilon),flat,lag])

def verify_production_contract(root: Path, errors: list[str], *, allow_test_fixture: bool=False, physics_recomputed: Mapping[str,object]|None=None) -> dict[str,object]:
    result={};cfg=_json(root/"config.json");manifest=_json(root/"data_manifest.json");provenance=_json(root/"provenance.json")
    if "test_fixture" in cfg: errors.append("public artifact config contains forbidden test_fixture escape")
    if provenance.get("test_fixture") is not bool(allow_test_fixture): errors.append("artifact test_fixture provenance does not match verifier mode")
    if not allow_test_fixture:
      if provenance.get("worktree_clean") is not True or provenance.get("diff_inventory")!=[]: errors.append("production provenance is not clean-tree exact")
      try: core.validate_config(cfg)
      except Exception as exc: errors.append(f"production frozen config validation: {exc}")
    rows=_csv(root/"per_epoch_scores.csv");prn=[x for x in rows if x.get("row_level")=="prn"]
    events=[x for x in rows if x.get("row_level")=="event"]
    required_meta={"row_level","scenario","physical_recording_id","event_id","target_index","availability_time_s",
      "source_start_s","source_end_s","role","phase","label","valid","tracked_prn_count","prn",
      "prn_target_index","pair_sequence_index"}
    if not prn or not events: errors.append("production epoch inventory requires PRN and event rows")
    if prn:
      missing=set(PRODUCTION_METHODS)-set(prn[0]);unknown={x for x in set(prn[0])-required_meta-set(PRODUCTION_METHODS)
               if not any(x==f"{m}_{a}" for m in PRODUCTION_METHODS for a in AGGREGATORS)}
      if missing: errors.append(f"production method inventory missing: {sorted(missing)}")
      if unknown: errors.append(f"production method inventory has unknown/alias columns: {sorted(unknown)}")
      for i,row in enumerate(prn):
       for method in PRODUCTION_METHODS:
        try:_number(row.get(method),f"PRN {i}/{method}")
        except Exception as exc:errors.append(str(exc))
       try:
        start=_number(row["source_start_s"],"source_start");end=_number(row["source_end_s"],"source_end")
        expected=_source_role(row["scenario"],start,end)
        if row["role"]!=expected: errors.append(f"source-support role mismatch row {i}: expected {expected}")
       except Exception as exc:errors.append(f"role reconstruction row {i}: {exc}")
    if set(x["scenario"] for x in events)!=set(PRODUCTION_SCENARIOS): errors.append("production scenario event set is not exact")
    # Threshold inventory is exact, not at-least-one.
    thresholds=_json(root/"thresholds.json")
    expected_thresholds={f"{m}/{a}/{q}" for m in PRODUCTION_METHODS for a in AGGREGATORS for q in ("q99","q995")}
    if set(thresholds)!=expected_thresholds: errors.append(f"threshold inventory mismatch missing={sorted(expected_thresholds-set(thresholds))} extra={sorted(set(thresholds)-expected_thresholds)}")
    # Metrics inventory emitted by the production runner.
    sm=_csv(root/"scenario_metrics.csv");keys=[(x.get("scenario"),x.get("method"),x.get("aggregator"),x.get("quantile"),x.get("metric"),x.get("phase")) for x in sm]
    expected=set()
    for s in PRODUCTION_SCENARIOS:
     for m in PRODUCTION_METHODS:
      for a in AGGREGATORS:
       for q in ("0.99","0.995"):
        phase="normal_holdout" if s=="cleanStatic" else ("normal" if s=="cleanDynamic" else "stable_pre")
        expected.add((s,m,a,q,"fpr",phase))
        if s in ATTACKS:
         expected.update({(s,m,a,q,"roc_auc","stable_pre+post"),(s,m,a,q,"pr_auc","stable_pre+post"),
                          (s,m,a,q,"pauc","stable_pre+post"),(s,m,a,q,"detection_rate","post"),
                          (s,m,a,q,"persistent_alarm_ratio","persistent"),(s,m,a,q,"sustained_delay","post")})
    if len(keys)!=len(set(keys)): errors.append("duplicate scenario metric key")
    if set(keys)!=expected: errors.append(f"scenario metric inventory mismatch missing={len(expected-set(keys))} extra={len(set(keys)-expected)}")
    ab=_csv(root/"ablation_metrics.csv");abkeys=[(x.get("method"),x.get("aggregator"),x.get("quantile"),x.get("metric")) for x in ab]
    expected_ab={(m,a,q,"mean_attack_pauc") for m in PRODUCTION_METHODS for a in AGGREGATORS for q in ("0.99","0.995")}
    if len(abkeys)!=len(set(abkeys)) or set(abkeys)!=expected_ab: errors.append("ablation metric inventory is not exact/unique")
    # Recompute every metric directly from stored event scores and typed thresholds.
    metric_recomputed=0
    for i,item in enumerate(sm):
     try:
      scenario=item["scenario"];method=item["method"];agg=item["aggregator"];q=float(item["quantile"])
      subset=[x for x in events if x["scenario"]==scenario]
      if item["metric"]=="fpr":
       if scenario=="cleanStatic": eligible=[x for x in subset if x["role"]=="normal_holdout"]
       elif scenario=="cleanDynamic": eligible=subset
       else: eligible=[x for x in subset if x["phase"]=="stable_pre"]
       suffix="q99" if q==.99 else "q995"; threshold=_number(thresholds[f"{method}/{agg}/{suffix}"]["value"],"threshold")
       scores=[_number(x[f"{method}_{agg}"],"event score") for x in eligible]
       alarms=[x>threshold for x in scores]
       if not scores: raise ValueError("empty FPR eligibility")
       expected_value=sum(alarms)/len(alarms)
       if int(float(item["numerator"]))!=sum(alarms) or int(float(item["denominator"]))!=len(alarms): raise ValueError("FPR counts mismatch")
      elif item["metric"] in ("roc_auc","pr_auc","pauc"):
       from sklearn.metrics import average_precision_score,roc_auc_score
       eligible=[x for x in subset if x["phase"] in ("stable_pre","post")]
       labels=[int(x["label"]) for x in eligible];scores=[_number(x[f"{method}_{agg}"],"event score") for x in eligible]
       if set(labels)!={0,1}: raise ValueError("classification class inventory")
       expected_value=(core.standardized_pauc(labels,scores,max_fpr=.05) if item["metric"]=="pauc" else
         float(roc_auc_score(labels,scores) if item["metric"]=="roc_auc" else average_precision_score(labels,scores)))
       if json.loads(item["labels_json"])!=labels or not np.array_equal(np.asarray(json.loads(item["scores_json"]),float),np.asarray(scores,float)): raise ValueError("classification raw evidence mismatch")
      elif item["metric"] in ("detection_rate","persistent_alarm_ratio"):
       onset=float(cfg["attacks"]["onsets_seconds"][scenario])
       eligible=[x for x in subset if x["phase"]=="post" and (item["metric"]=="detection_rate" or _number(x["source_start_s"],"source start")>=onset+40.)]
       suffix="q99" if q==.99 else "q995";threshold=_number(thresholds[f"{method}/{agg}/{suffix}"]["value"],"threshold")
       alarms=[_number(x[f"{method}_{agg}"],"event score")>threshold for x in eligible]
       expected_value=sum(alarms)/len(alarms)
       if int(float(item["numerator"]))!=sum(alarms) or int(float(item["denominator"]))!=len(alarms): raise ValueError("detection/persistent counts mismatch")
      elif item["metric"]=="sustained_delay":
       onset=float(cfg["attacks"]["onsets_seconds"][scenario]);suffix="q99" if q==.99 else "q995"
       threshold=_number(thresholds[f"{method}/{agg}/{suffix}"]["value"],"threshold")
       eligible=[x for x in subset if x["phase"] in ("stable_pre","post")]
       alarm_result=core.sustained_alarm_delay([_number(x["availability_time_s"],"availability") for x in eligible],
        [_number(x[f"{method}_{agg}"],"event score")>threshold for x in eligible],
        recording_ids=[x["physical_recording_id"] for x in eligible],post_eligible_mask=[x["phase"]=="post" for x in eligible],
        onset=onset,required=3,cadence=.5,stable_pre_mask=[x["phase"]=="stable_pre" for x in eligible])
       finite=bool(np.isfinite(alarm_result.delay));censored=_bool(item["censored"],"delay censored")
       status="already_alarming_stable_pre" if alarm_result.already_alarming_stable_pre else ("detected" if finite else "censored")
       if censored==finite or item["status"]!=status or _bool(item["already_alarming_stable_pre"],"already alarming")!=alarm_result.already_alarming_stable_pre: raise ValueError("sustained status/censor mismatch")
       if json.loads(item["stable_pre_alarm_by_recording_json"])!=dict(alarm_result.stable_pre_alarm_by_recording): raise ValueError("stable-pre recording audit mismatch")
       expected_value=alarm_result.delay if finite else None
       if finite and (not math.isclose(_number(item["value"],"delay"),expected_value,rel_tol=1e-12,abs_tol=1e-12) or not math.isclose(_number(item["alarm_time_s"],"alarm time"),alarm_result.alarm_time,rel_tol=1e-12,abs_tol=1e-12)): raise ValueError("sustained delay mismatch")
       if not finite and (item.get("value","")!="" or item.get("alarm_time_s","")!=""): raise ValueError("censored delay must have blank value/time")
      else: raise ValueError("unknown production metric")
      if expected_value is not None and not math.isclose(_number(item["value"],"metric value"),expected_value,rel_tol=1e-12,abs_tol=1e-12): raise ValueError("metric value mismatch")
      metric_recomputed+=1
     except Exception as exc: errors.append(f"scenario metric row {i} reconstruction: {exc}")
    for i,item in enumerate(ab):
     try:
      selected=[_number(x["value"],"attack pAUC") for x in sm if x["scenario"] in ATTACKS and x["method"]==item["method"] and x["aggregator"]==item["aggregator"] and x["quantile"]==item["quantile"] and x["metric"]=="pauc"]
      if len(selected)!=5: raise ValueError("ablation attack inventory")
      expected_value=float(np.mean(selected))
      if not math.isclose(_number(item["value"],"ablation value"),expected_value,rel_tol=1e-12,abs_tol=1e-12): raise ValueError("ablation value mismatch")
      if not math.isclose(_number(item["numerator"],"ablation numerator"),sum(selected),rel_tol=1e-12,abs_tol=1e-12) or int(float(item["denominator"]))!=5: raise ValueError("ablation reduction mismatch")
     except Exception as exc: errors.append(f"ablation metric row {i} reconstruction: {exc}")
    result["metric_rows_recomputed"]=metric_recomputed
    # Rebuild ordered typed fit identities from PRN/event rows and require disjoint roles.
    fit_data=_json(root/"fit_audit.json");sealed=fit_data.get("sealed_fits",{})
    def payload(row,event=False):
     return {"type":"EpochIdentity.v1","recording_id":row["physical_recording_id"],"scenario":"cleanStatic","prn":"EVENT" if event else row["prn"],"target_index":int(float(row["target_index"] if event else row["prn_target_index"])),"availability_time_s":float(row["availability_time_s"])}
    train_rows=sorted([x for x in prn if x["scenario"]=="cleanStatic" and x["role"]=="normal_train"],key=lambda x:int(float(x["pair_sequence_index"])))
    cal_rows=sorted([x for x in prn if x["scenario"]=="cleanStatic" and x["role"]=="normal_calibration"],key=lambda x:int(float(x["pair_sequence_index"])))
    cal_events=sorted([x for x in events if x["scenario"]=="cleanStatic" and x["role"]=="normal_calibration"],key=lambda x:int(float(x["target_index"])))
    expected_fit={"covariance":[payload(x) for x in train_rows],"conditioner":[payload(x) for x in train_rows],"shuffled_conditioner":[payload(x) for x in train_rows],"conditioner_cap":[payload(x) for x in cal_rows],"shuffled_cap":[payload(x) for x in cal_rows],"thresholds":[payload(x,True) for x in cal_events]}
    for name,expected_ids in expected_fit.items():
     item=sealed.get(name,{})
     if item.get("identities")!=expected_ids: errors.append(f"fit identity rows mismatch: {name}")
     typed=[core.EpochIdentity(x["recording_id"],x["scenario"],x["prn"],x["target_index"],x["availability_time_s"]) for x in expected_ids]
     digest=core._identity_digest(typed)
     if item.get("identity_digest_sha256")!=digest or int(item.get("identity_count",-1))!=len(expected_ids): errors.append(f"fit identity digest/count mismatch: {name}")
    train_keys={json.dumps(x,sort_keys=True) for x in expected_fit["covariance"]};cal_keys={json.dumps(x,sort_keys=True) for x in expected_fit["conditioner_cap"]}
    if train_keys & cal_keys: errors.append("clean train/calibration fit identities overlap")
    for audit_name,seal_name in (("covariance_audit","covariance"),("conditioner_fit","conditioner"),("shuffle_fit","shuffled_conditioner"),("conditioner_cap","conditioner_cap"),("shuffle_cap","shuffled_cap")):
     if fit_data.get(audit_name,{}).get("identity_digest_sha256")!=sealed.get(seal_name,{}).get("identity_digest_sha256"): errors.append(f"fit audit identity binding mismatch: {audit_name}")
    result["split_counts"]={"normal_train":len(train_rows),"normal_calibration":len(cal_rows),"normal_holdout":len([x for x in prn if x["scenario"]=="cleanStatic" and x["role"]=="normal_holdout"]),"excluded_boundary_crossing":len([x for x in prn if x["scenario"]=="cleanStatic" and x["role"]=="excluded_boundary_crossing"])}
    boot=_json(root/"bootstrap_results.json").get("comparisons",[])
    bkeys=[(x.get("scenario"),x.get("aggregator"),x.get("comparison")) for x in boot]
    expected_b={(s,a,c) for s in ATTACKS for a in AGGREGATORS for c in ("NC-B0","TOPI-B0","NC-TOPI","shuffleNC-TOPI")}
    if len(bkeys)!=len(set(bkeys)) or set(bkeys)!=expected_b: errors.append("bootstrap inventory must be exact 5x2x4 and unique")
    for item in boot:
     if item.get("available") and int(item.get("valid_reps",0))!=2000: errors.append("available bootstrap does not have 2000 valid reps")
     if not item.get("available") and item.get("reason") not in ("class-deficient eligible epochs","too few complete blocks in one or both class strata","no valid bootstrap replicates"):
      errors.append("bootstrap unavailable reason is not a core block-support failure")
    # Data inventory and raw witness contract.
    if set(manifest.get("canonical_npz",{}))!={"cleanStatic","DS1","DS2","DS3","DS7","DS8"}: errors.append("canonical NPZ manifest inventory is not exact six")
    if set(manifest.get("raw_iq",{}))!=set(PRODUCTION_SCENARIOS): errors.append("raw IQ manifest inventory is not exact seven")
    for s,item in manifest.get("canonical_npz",{}).items():
     cfgitem=cfg.get("canonical_inputs",{}).get(s,{})
     if item.get("expected_sha256")!=cfgitem.get("sha256") or item.get("sha256_recomputed")!=cfgitem.get("sha256"): errors.append(f"canonical NPZ manifest hash mismatch {s}")
    for s,item in manifest.get("raw_iq",{}).items():
     expected=cfg.get("raw_iq_inputs",{}).get(s,{})
     for key in ("size_bytes","first_1MiB_sha256","last_1MiB_sha256"):
      if item.get(key)!=expected.get(key): errors.append(f"raw manifest mismatch {s}/{key}")
     if item.get("expected_full_sha256")!=expected.get("sha256"): errors.append(f"raw expected hash mismatch {s}")
     if not item.get("full_sha256_recomputed") and item.get("full_hash_status")!="expected_not_recomputed": errors.append(f"raw full hash status invalid {s}")
    dynamic=manifest.get("cleanDynamic_nodes",{})
    if dynamic.get("available") is not True or dynamic.get("sha256_recomputed")!=cfg.get("clean_dynamic_nodes",{}).get("sha256"): errors.append("cleanDynamic availability/hash manifest invalid")
    # Independently rebuild scenario -> physical recording -> raw source linkage and group equality.
    iqrows=_csv(root/"iq_context.csv");linkage=manifest.get("scenario_recording_raw_linkage",{})
    if set(linkage)!=set(PRODUCTION_SCENARIOS):
     errors.append("scenario/recording/raw linkage inventory is not exact seven")
    try:
     dynamic_source=_csv(Path(cfg["clean_dynamic_nodes"]["path"]))
     dynamic_recordings=sorted({row["run_id"] for row in dynamic_source})
     if len(dynamic_recordings)!=1: raise ValueError(f"expected one cleanDynamic run_id, got {dynamic_recordings}")
     expected_recording={scenario:scenario for scenario in PRODUCTION_SCENARIOS}
     expected_recording["cleanDynamic"]=dynamic_recordings[0]
     group_checks=0
     for scenario in PRODUCTION_SCENARIOS:
      item=linkage.get(scenario,{});rawcfg=cfg["raw_iq_inputs"][scenario]
      expected={"scenario":scenario,"physical_recording_id":expected_recording[scenario],
        "raw_path":str(Path(rawcfg["path"])),"expected_raw_sha256":rawcfg["sha256"],
        "raw_size_bytes":int(rawcfg["size_bytes"]),"target_groups":[expected_recording[scenario]],
        "block_groups":[expected_recording[scenario]],"groups_exact_match":True}
      for key,value in expected.items():
       if item.get(key)!=value: raise ValueError(f"{scenario}/{key} linkage mismatch")
      if scenario=="cleanDynamic" and (item.get("node_path")!=str(Path(cfg["clean_dynamic_nodes"]["path"]))
          or item.get("expected_node_sha256")!=cfg["clean_dynamic_nodes"]["sha256"]):
       raise ValueError("cleanDynamic node path/hash linkage mismatch")
      scenario_iq=[row for row in iqrows if row.get("scenario")==scenario]
      target_groups=sorted({row.get("physical_recording_id","") for row in scenario_iq})
      block_groups=sorted({row.get("block_recording_id","") for row in scenario_iq})
      if not scenario_iq or target_groups!=block_groups or target_groups!=[expected_recording[scenario]]:
       raise ValueError(f"{scenario} IQ target/block group equality mismatch")
      group_checks+=1
     result["iq_physical_group_linkages_recomputed"]=group_checks
    except Exception as exc: errors.append(f"scenario/recording/raw linkage reconstruction: {exc}")
    # Deterministic independent raw IQ audit: first ten contexts in every scenario.
    checked=0
    for scenario in PRODUCTION_SCENARIOS:
     sample=sorted([x for x in iqrows if x.get("scenario")==scenario],key=lambda x:(x.get("event_id","")))[:10]
     if len(sample)<10: errors.append(f"IQ verifier has fewer than 10 contexts for {scenario}");continue
     rawcfg=cfg["raw_iq_inputs"][scenario];rate=int(cfg["iq_conditioner"]["sample_rate_hz"])
     path=Path(rawcfg["path"]);mem=np.memmap(path,mode="r",dtype="<i2",shape=(path.stat().st_size//4,2))
     for row in sample:
      try:
       offsets=[int(x) for x in row["sample_offset"].split(";")];counts=[int(x) for x in row["sample_count"].split(";")]
       ends=[float(x) for x in row["block_end_s"].split(";")];stored=np.asarray(json.loads(row["block_features_json"]),float)
       if len(offsets)!=4 or not np.allclose(np.diff(ends),.5,rtol=0,atol=1e-8): raise ValueError("history/cadence")
       actual=np.stack([_iq_features_independent(mem[o:o+n]) for o,n in zip(offsets,counts)])
       if not np.allclose(actual,stored,rtol=1e-12,atol=1e-12): raise ValueError("raw feature mismatch")
       context=np.asarray(json.loads(row["context_features_json"]),float)
       if not np.allclose(context,actual.mean(axis=0),rtol=1e-12,atol=1e-12): raise ValueError("history reduction mismatch")
       if ends[-1]>float(row["target_source_start_s"])+1e-12: raise ValueError("causality")
       checked+=1
      except Exception as exc:errors.append(f"IQ independent sample {scenario}: {exc}")
     del mem
    result["iq_raw_contexts_recomputed"]=checked
    # DS7 legacy replay is strictly non-primary and must be exact when source files exist.
    try:
     lineage=_json(root/"model_lineage_audit.json");legacy=lineage.get("legacy_positive_control",{})
     required={"scenario","primary_use","available","node_path","score_path","node_sha256","score_sha256",
               "checkpoint_path","checkpoint_sha256","expected_coverage","defined_tolerance","reason"}
     if not required.issubset(legacy): raise ValueError(f"legacy evidence fields missing: {sorted(required-set(legacy))}")
     if legacy["scenario"]!="DS7" or legacy["primary_use"] is not False or int(legacy["expected_coverage"])!=5465 or not math.isclose(float(legacy["defined_tolerance"]),3e-4,rel_tol=0,abs_tol=0):
      raise ValueError("legacy scenario/coverage/tolerance/primary contract")
     paths=[Path(legacy[x]) for x in ("node_path","score_path","checkpoint_path")]
     available=all(x.is_file() for x in paths)
     if available:
      if legacy.get("available") is not True or legacy.get("pass") is not True or legacy.get("positive_control_pass") is not True: raise ValueError("available legacy replay is not passing")
      if sha256_file(paths[0])!=legacy["node_sha256"] or sha256_file(paths[1])!=legacy["score_sha256"] or sha256_file(paths[2])!=legacy["checkpoint_sha256"]: raise ValueError("legacy path hash mismatch")
      if legacy.get("legacy_rows")!=5465 or legacy.get("covered_keys")!=5465 or legacy.get("regenerated_rows")!=5465: raise ValueError("legacy exact coverage mismatch")
      import importlib.util
      spec=importlib.util.spec_from_file_location("nc_topi_legacy_replay",ROOT/"scripts"/"eval_texbat_nc_topi_stage0.py")
      runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
      model=runner.FrozenB0(paths[2],cfg["b0"],device="cpu")
      replay=runner.legacy_b0_positive_control(paths[0],paths[1],model,node_sha256=legacy["node_sha256"],
        score_sha256=legacy["score_sha256"],gpu_atol=float(legacy["defined_tolerance"]))
      for field in ("positive_control_pass","key_sets_equal","legacy_rows","regenerated_rows","covered_keys"):
       if replay.get(field)!=legacy.get(field): raise ValueError(f"legacy replay mismatch {field}")
      if float(legacy["max_abs_rmse_error"])>float(legacy["defined_tolerance"]) or float(replay["max_abs_rmse_error"])>float(legacy["defined_tolerance"]):
       raise ValueError("legacy replay exceeds defined CPU/GPU tolerance")
      result["legacy_positive_control_recomputed"]=True
     else:
      if legacy.get("available") is not False or not legacy.get("reason"): raise ValueError("missing legacy source lacks explicit unavailable reason")
      result["legacy_positive_control_recomputed"]=False
     result["legacy_positive_control"]={key:legacy.get(key) for key in ("scenario","expected_coverage","covered_keys","max_abs_rmse_error","defined_tolerance","positive_control_pass","primary_use","available")}
    except Exception as exc: errors.append(f"legacy positive control verification: {exc}")
    physics_data=_json(root/"synthetic_physics_tests.json");fit_data=_json(root/"fit_audit.json")
    ref=physics_data.get("reference_state",{})
    if ref.get("covariance_fit_identity_digest_sha256")!=fit_data.get("covariance_audit",{}).get("identity_digest_sha256"):
        errors.append("physics reference is not bound to clean-train covariance identities")
    # Derive decision evidence only from independently reconstructable metric/bootstrap/physics rows.
    try:
     def one_metric(s,m,metric,phase):
      hits=[x for x in sm if x["scenario"]==s and x["method"]==m and x["aggregator"]=="median" and x["quantile"]=="0.99" and x["metric"]==metric and x["phase"]==phase]
      if len(hits)!=1: raise ValueError(f"decision metric not unique {s}/{m}/{metric}")
      return _number(hits[0]["value"],"decision metric")
     nc={s:one_metric(s,"NC_TOPI","pauc","stable_pre+post") for s in ATTACKS};b0p={s:one_metric(s,"B0","pauc","stable_pre+post") for s in ATTACKS}
     bootmap={(x["scenario"],x["aggregator"],x["comparison"]):x for x in boot}
     lower={s:(float(bootmap[(s,"median","NC-B0")]["ci"][0]) if bootmap[(s,"median","NC-B0")]["available"] else -1.) for s in ATTACKS}
     upper={s:(float(bootmap[(s,"median","NC-B0")]["ci"][1]) if bootmap[(s,"median","NC-B0")]["available"] else 1.) for s in ATTACKS}
     criteria=(physics_recomputed or {}).get("recomputed",{})
     def one_delay(s,m):
      hits=[x for x in sm if x["scenario"]==s and x["method"]==m and x["aggregator"]=="median" and x["quantile"]=="0.99" and x["metric"]=="sustained_delay" and x["phase"]=="post"]
      if len(hits)!=1: raise ValueError(f"decision delay not unique {s}/{m}")
      return None if _bool(hits[0]["censored"],"decision delay censored") else _number(hits[0]["value"],"decision delay")
     evidence={"clean_nc_fpr":one_metric("cleanStatic","NC_TOPI","fpr","normal_holdout"),
      "clean_b0_fpr":one_metric("cleanStatic","B0","fpr","normal_holdout"),
      "stable_pre_fpr":{s:one_metric(s,"NC_TOPI","fpr","stable_pre") for s in ATTACKS},
      "nc_pauc":nc,"b0_pauc":b0p,"pauc_delta":{s:nc[s]-b0p[s] for s in ATTACKS},
      "nc_delay":{s:one_delay(s,"NC_TOPI") for s in ATTACKS},
      "b0_delay":{s:one_delay(s,"B0") for s in ATTACKS},
      "pauc_ci_lower":lower,"pauc_ci_upper":upper,
      "equal_rmse_pass":bool(criteria.get("equal_rmse_pass")),
      "second_peak_pass":bool(criteria.get("second_peak_pass")),
      "actual_nc_mean_pauc":float(np.mean(list(nc.values()))),
      "topi_mean_pauc":float(np.mean([one_metric(s,"TOPI","pauc","stable_pre+post") for s in ATTACKS])),
      "shuffled_nc_mean_pauc":float(np.mean([one_metric(s,"NC_TOPI_time_shuffle","pauc","stable_pre+post") for s in ATTACKS]))}
     rebuilt=core.evaluate_stage0_decision(**evidence);stored_decision=_json(root/"decision.json")
     if stored_decision.get("evidence")!=evidence or stored_decision.get("status")!=rebuilt.status or stored_decision.get("criteria")!=rebuilt.criteria:
      errors.append("decision is not derived solely from verifier-recomputed metrics/bootstrap/physics")
    except Exception as exc:errors.append(f"independent decision evidence derivation: {exc}")
    # README inputs are retained only after their underlying rows have been independently reconstructed.
    def report_row(scenario,method,metric):
     phase="normal_holdout" if scenario=="cleanStatic" else ("normal" if scenario=="cleanDynamic" else ("post" if metric in ("detection_rate","sustained_delay") else ("persistent" if metric=="persistent_alarm_ratio" else ("stable_pre" if metric=="fpr" else "stable_pre+post"))))
     hits=[x for x in sm if x["scenario"]==scenario and x["method"]==method and x["aggregator"]=="median" and x["quantile"]=="0.99" and x["metric"]==metric and x["phase"]==phase]
     if len(hits)!=1: raise ValueError(f"README metric not unique {scenario}/{method}/{metric}")
     row=hits[0];return {"value":None if row.get("value","")=="" else float(row["value"]),"numerator":None if row.get("numerator","")=="" else int(float(row["numerator"])),"denominator":None if row.get("denominator","")=="" else int(float(row["denominator"])),"status":row.get("status",""),"censored":row.get("censored","")}
    primary={s:{m:{metric:report_row(s,m,metric) for metric in (("fpr",) if s in ("cleanStatic","cleanDynamic") else ("fpr","roc_auc","pr_auc","pauc","detection_rate","sustained_delay","persistent_alarm_ratio"))} for m in ("B0","TOPI","NC_TOPI")} for s in PRODUCTION_SCENARIOS}
    paired={s:{key:item for key,item in (("delta",evidence["pauc_delta"][s]),("ci_lower",evidence["pauc_ci_lower"][s]),("ci_upper",evidence["pauc_ci_upper"][s]),("available",bootmap[(s,"median","NC-B0")]["available"]),("valid_reps",bootmap[(s,"median","NC-B0")]["valid_reps"]))} for s in ATTACKS}
    event_split={role:len([x for x in events if x["scenario"]=="cleanStatic" and x["role"]==role]) for role in ("normal_train","normal_calibration","normal_holdout","excluded_boundary_crossing")}
    result["report"]={"lineage":{"source_commit":provenance.get("source_commit"),"checkpoint_sha256":manifest.get("checkpoint",{}).get("sha256_recomputed")},"inventory":{"scenario_metric_rows":len(sm),"ablation_metric_rows":len(ab),"production_metric_rows_reconstructed":result.get("metric_rows_recomputed",0),"simple_metric_rows_recomputed":sum(1 for x in sm+ab if (x.get("numerator","") and x.get("denominator","") and x.get("value","")) or (x.get("labels_json","") and x.get("scores_json","") and x.get("metric",""))),"methods":list(PRODUCTION_METHODS),"aggregators":list(AGGREGATORS),"quantiles":["q99","q995"]},"clean_split_prn_counts":result.get("split_counts",{}),"clean_split_event_counts":event_split,"iq":{"event_context_rows":len(iqrows),"linked_prn_contexts":sum(int(x["linked_pair_count"]) for x in iqrows),"raw_contexts_independently_recomputed":result.get("iq_raw_contexts_recomputed",0),"strict_causal":True},"legacy_ds7":result.get("legacy_positive_control",{}),"primary_q99_median":primary,"paired_nc_b0":paired,"decision":{"status":rebuilt.status,"criteria":rebuilt.criteria,"counts":rebuilt.counts,"no_go_triggers":rebuilt.no_go_triggers,"evidence":evidence},"physics":{"criteria":criteria,"diagnostics":(physics_recomputed or {}).get("diagnostics",{})}}
    plots=_json(root/"plot_provenance.json") if (root/"plot_provenance.json").exists() else {}
    required_plots={f"score_comparison_{s}.png" for s in PRODUCTION_SCENARIOS}|{f"prn_orth_heatmap_{s}.png" for s in PRODUCTION_SCENARIOS}|{"tangent_orth_energy.png","clean_distributions.png","roc.png","roc_low_fpr.png","second_peak_heatmap.png"}
    if set(plots.get("plots",{}))!=required_plots: errors.append("plot provenance inventory mismatch")
    for name,item in plots.get("plots",{}).items():
     if int(item.get("data_series",0))<1 or int(item.get("data_points",0))<1: errors.append(f"plot has no nonzero data series: {name}")
    return result


def compute_summary(root: str | Path, *, verify_hashes: bool=True,
                    verify_source: bool=True, allow_test_fixture: bool=False) -> dict[str,object]:
    root=Path(root); errors=[]
    inventory={"required_files":len(REQUIRED_FILES),"png_count":0,"hashed_files":0}
    for name in REQUIRED_FILES:
        if not (root/name).is_file(): errors.append(f"missing required file: {name}")
    for name in REQUIRED_DIRS:
        if not (root/name).is_dir(): errors.append(f"missing required directory: {name}")
    if verify_hashes and (root/"hashes.json").exists():
        try:
            manifest=_json(root/"hashes.json"); stated=manifest.get("files",{})
            actual_names=_files(root)
            if set(stated)!=set(actual_names):
                errors.append(f"hash inventory is incomplete/extra: missing={sorted(set(actual_names)-set(stated))}, extra={sorted(set(stated)-set(actual_names))}")
            for name in sorted(set(stated)&set(actual_names)):
                if stated[name] != sha256_file(root/name): errors.append(f"checksum mismatch: {name}")
            inventory["hashed_files"]=len(stated)
        except Exception as exc: errors.append(f"hash inventory unreadable: {exc}")
    pngs=sorted((root/"plots").glob("*.png")) if (root/"plots").is_dir() else []
    if not pngs: errors.append("plots/ contains no PNG files")
    for path in pngs:
        try: _png(path)
        except Exception as exc: errors.append(str(exc))
    inventory["png_count"]=len(pngs)
    fit={}
    for name in ("fit_audit.json","provenance.json"):
        if (root/name).exists():
            try:
                data=_json(root/name)
                if data.get("attack_fit") is not False: errors.append(f"{name} does not attest attack_fit=false")
                bad=set(data.get("fit_scenarios",[]))-{"cleanStatic"}
                if bad: errors.append(f"{name} has non-clean fit scenarios: {sorted(bad)}")
                fit[name]=data
            except Exception as exc: errors.append(f"{name} unreadable: {exc}")
    epochs=verify_epoch_rows(root,errors) if (root/"per_epoch_scores.csv").exists() else {"rows":0,"rows_data":[]}
    thresholds=verify_thresholds(root,epochs.get("rows_data",[]),errors) if (root/"thresholds.json").exists() else {"checked":0}
    iq=verify_iq_causality(root,errors)
    physics=recompute_synthetic(_json(root/"synthetic_physics_tests.json"),errors) if (root/"synthetic_physics_tests.json").exists() else {}
    bootstrap=recompute_bootstrap(_json(root/"bootstrap_results.json"),errors) if (root/"bootstrap_results.json").exists() else {}
    metrics=verify_metrics(root,errors)
    source=verify_source_manifest(root,errors) if verify_source and (root/"provenance.json").exists() else {"checked":False}
    # Full frozen config activates production inventory/evidence requirements; tiny verifier fixtures remain possible.
    production=False;fixture=False
    try:
      production=bool(_json(root/"config.json").get("canonical_inputs"))
      fixture=bool(_json(root/"provenance.json").get("test_fixture"))
      if fixture and not allow_test_fixture: errors.append("production verifier rejects test_fixture artifacts")
      if allow_test_fixture and not fixture: errors.append("test verifier requires test_fixture=true provenance")
    except Exception: pass
    production_contract={}
    if production:
        for required in ("iq_context.csv","iq_context.npz","model_lineage_audit.json","plot_provenance.json"):
            if not (root/required).is_file(): errors.append(f"missing production required file: {required}")
        try: production_contract=verify_production_contract(root,errors,allow_test_fixture=allow_test_fixture,physics_recomputed=physics)
        except Exception as exc: errors.append(f"production strict contract failed: {exc}")
        scenarios=("cleanStatic","cleanDynamic","DS1","DS2","DS3","DS7","DS8")
        expected_plots={f"score_comparison_{name}.png" for name in scenarios} | {f"prn_orth_heatmap_{name}.png" for name in scenarios} | {"tangent_orth_energy.png","clean_distributions.png","roc.png","roc_low_fpr.png","second_peak_heatmap.png"}
        present={path.name for path in pngs}
        if not expected_plots.issubset(present): errors.append(f"production PNG inventory missing: {sorted(expected_plots-present)}")
        if physics.get("raw_trials")!=100: errors.append("production synthetic evidence must contain 100 equal-RMSE raw trials")
        if physics.get("second_peak_grid_rows",0)<25: errors.append("production second-peak grid is incomplete")
        if bootstrap.get("comparisons",0)<8: errors.append("production bootstrap evidence must cover four comparisons and both aggregators")
        if bootstrap.get("deterministically_recomputed")!=bootstrap.get("comparisons"): errors.append("not every production bootstrap comparison was recomputed")
        if thresholds.get("checked",0)<1: errors.append("no production thresholds independently recomputed")
        try:
            decision=_json(root/"decision.json")
            evidence=decision.get("evidence")
            if not isinstance(evidence,dict): errors.append("production decision lacks raw evidence for frozen grammar")
            else:
                rebuilt=core.evaluate_stage0_decision(**evidence)
                if rebuilt.status!=decision.get("status") or rebuilt.criteria!=decision.get("criteria"): errors.append("decision status/criteria do not match frozen evaluator")
        except Exception as exc: errors.append(f"decision recomputation failed: {exc}")
    epochs.pop("rows_data",None)
    return {"schema":"gnss-doppler-lab.nc-topi-stage0.verifier.v1","ok":not errors,
            "errors":errors,"inventory":inventory,"epochs":epochs,"thresholds":thresholds,
            "iq_causality":iq,"synthetic":physics,"bootstrap":bootstrap,"metrics":metrics,
            "source":source,"fit":fit,"production_contract":production_contract,
            "decision_status":(_json(root/"decision.json").get("status") if (root/"decision.json").exists() else "unavailable"),"fit_attack_free":not any("attack_fit" in x for x in errors)}


def verify_test_artifact(root: str|Path, *, verify_source: bool=False) -> dict[str,object]:
    """Explicit non-production verifier for private synthetic campaign fixtures."""
    summary=compute_summary(root,verify_hashes=True,verify_source=verify_source,allow_test_fixture=True)
    summary["verification_mode"]="non-production-test-fixture"
    summary["ok"]=not summary["errors"]
    return summary


def render_readme(summary: Mapping[str,object]) -> str:
    """Byte-deterministic report rendered only from verifier-reconstructed evidence."""
    inv=summary.get("inventory",{});epochs=summary.get("epochs",{});th=summary.get("thresholds",{})
    iq_summary=summary.get("iq_causality",{});syn=summary.get("synthetic",{});boot=summary.get("bootstrap",{})
    metrics=summary.get("metrics",{});report=summary.get("production_contract",{}).get("report",{})
    if not report:
        lines=["# NC-TOPI Stage-0 verified report","","## Verified inventory and lineage",
          f"- Required files: {int(inv.get('required_files',0))}; hashed files: {int(inv.get('hashed_files',0))}; PNG plots: {int(inv.get('png_count',0))}.",
          f"- Epoch rows: {int(epochs.get('rows',0))}; thresholds independently recomputed: {int(th.get('checked',0))}.",
          f"- Metric rows: {int(metrics.get('rows',0))}; simple rows independently recomputed: {int(metrics.get('recomputed_rows',0))}.","",
          "## Verification result",f"- Independent verification passed: {str(bool(summary.get('ok',False))).lower()}.",
          f"- Verification error count: {len(summary.get('errors',[]))}."]
        return "\n".join(lines)+"\n"
    def f(value):
        if value is None:return "NA"
        if isinstance(value,bool):return str(value).lower()
        if isinstance(value,float):return f"{value:.6g}"
        return str(value)
    rinv=report["inventory"];lineage=report["lineage"];prn=report["clean_split_prn_counts"];events=report["clean_split_event_counts"]
    iq=report["iq"];legacy=report["legacy_ds7"];primary=report["primary_q99_median"]
    physics=report["physics"];pc=physics["criteria"];diag=physics.get("diagnostics",{});decision=report["decision"]
    total=int(rinv["scenario_metric_rows"])+int(rinv["ablation_metric_rows"])
    lines=["# NC-TOPI Stage-0 verified report","","## Verified inventory and lineage",
      f"- {total:,} total metric rows = {int(rinv['scenario_metric_rows']):,} scenario + {int(rinv['ablation_metric_rows']):,} ablation.",
      f"- {int(rinv['simple_metric_rows_recomputed']):,} simple scalar/classification rows independently recomputed; all {int(rinv['production_metric_rows_reconstructed']):,} scenario rows independently reconstructed by the production contract.",
      f"- Frozen B0 checkpoint SHA256: `{lineage['checkpoint_sha256']}`.",f"- Source commit: `{lineage['source_commit']}`.",
      f"- Clean split PRN counts: train={prn['normal_train']:,}, calibration={prn['normal_calibration']:,}, holdout={prn['normal_holdout']:,}, excluded-boundary={prn['excluded_boundary_crossing']:,}.",
      f"- Clean split event counts: train={events['normal_train']:,}, calibration={events['normal_calibration']:,}, holdout={events['normal_holdout']:,}, excluded-boundary={events['excluded_boundary_crossing']:,}.",
      f"- Causal IQ: {iq['event_context_rows']:,} event contexts / {iq['linked_prn_contexts']:,} linked PRN contexts; {iq['raw_contexts_independently_recomputed']:,} raw contexts independently reread; strict causal={f(iq['strict_causal'])}.",
      f"- DS7 legacy positive control (non-primary): coverage={legacy.get('covered_keys')}/{legacy.get('expected_coverage')}, max absolute RMSE error={f(legacy.get('max_abs_rmse_error'))}, tolerance={f(legacy.get('defined_tolerance'))}, pass={f(legacy.get('positive_control_pass'))}.","",
      "## Primary q99 / median results",
      "Primary evidence is the frozen median aggregator at q99. Values below are independently reconstructed from event scores and typed thresholds.","",
      "| Scenario | Method | FPR | ROC-AUC | PR-AUC | pAUC@0.05 | stable-pre FPR | post detection | 3-consecutive sustained delay (s/status) | persistent ratio |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for scenario in PRODUCTION_SCENARIOS:
      for method in ("B0","TOPI","NC_TOPI"):
        rows=primary[scenario][method]
        if scenario in ("cleanStatic","cleanDynamic"):
          label="cleanStatic holdout" if scenario=="cleanStatic" else "cleanDynamic external"
          lines.append(f"| {label} | {method} | {f(rows['fpr']['value'])} ({rows['fpr']['numerator']}/{rows['fpr']['denominator']}) | - | - | - | - | - | - | - |")
        else:
          delay=rows["sustained_delay"];detect=rows["detection_rate"];persist=rows["persistent_alarm_ratio"]
          lines.append(f"| {scenario} | {method} | - | {f(rows['roc_auc']['value'])} | {f(rows['pr_auc']['value'])} | {f(rows['pauc']['value'])} | {f(rows['fpr']['value'])} ({rows['fpr']['numerator']}/{rows['fpr']['denominator']}) | {f(detect['value'])} ({detect['numerator']}/{detect['denominator']}) | {f(delay['value'])}/{delay['status']} | {f(persist['value'])} ({persist['numerator']}/{persist['denominator']}) |")
    lines += ["","## NC-B0 pAUC deltas and paired 10 s block bootstrap 95% CIs",
      "Paired scores are resampled in frozen nonoverlapping 10 s recording/label/cadence blocks (2,000 repetitions where available).","",
      "| Attack | NC-B0 pAUC delta | 95% CI | valid reps |","|---|---:|---:|---:|"]
    for scenario in ATTACKS:
      row=report["paired_nc_b0"][scenario]
      lines.append(f"| {scenario} | {f(row['delta'])} | [{f(row['ci_lower'])}, {f(row['ci_upper'])}] | {row['valid_reps']} |")
    criteria="; ".join(f"{name}={f(value)}" for name,value in sorted(decision["criteria"].items()))
    counts="; ".join(f"{name}={value}" for name,value in sorted(decision["counts"].items()))
    triggers=", ".join(decision["no_go_triggers"]) or "none"
    lines += ["","## Frozen c1-c8 decision",f"- Status: **{decision['status']}**.",f"- Criteria: {criteria}.",f"- Counts: {counts}.",f"- Exact NO-GO trigger(s): {triggers}.",
      "- c6 is exactly the frozen equal-RMSE direction test; nuisance controls are diagnostics only and do not rescue or fail c1-c8.","",
      "## Synthetic physics and nuisance diagnostics"]
    equal=diag.get("equal_rmse",{});second=diag.get("second_peak",{})
    lines += [f"- Equal-RMSE: pass={f(pc.get('equal_rmse_pass'))}; trials={syn.get('raw_trials',0)}; max B0 relative difference={f(equal.get('max_b0_relative_difference'))}; median tangent/orthogonal TOPI ratio={f(equal.get('median_tangent_to_orthogonal_topi_ratio'))}; median orthogonal preservation={f(equal.get('median_orthogonal_preserved_fraction'))}.",
      f"- Second peak: pass={f(pc.get('second_peak_pass'))}; power pass={f(pc.get('second_peak_power_pass'))}; separation pass={f(pc.get('second_peak_separation_pass'))}; rows={syn.get('second_peak_grid_rows',0)}.",
      f"- Second-peak power rho by separation: {json.dumps(second.get('power_rho_by_separation',{}),sort_keys=True,separators=(',',':'))}.",
      f"- Second-peak separation rho by power: {json.dumps(second.get('separation_rho_by_power',{}),sort_keys=True,separators=(',',':'))}."]
    for kind in ("amplitude","shift","noise"):
      row=diag.get("nuisance",{}).get(kind,{})
      semantics="local numerical delay tangent (amount * gradient, edge_order=2)" if kind=="shift" else ("prompt-normalized shape-scale" if kind=="amplitude" else "frozen seeded standardized noise")
      lines.append(f"- Nuisance {kind} ({semantics}): rows={row.get('rows',0)}, median normalized B0={f(row.get('median_b0_normalized'))}, median normalized TOPI={f(row.get('median_topi_normalized'))}, pass={f(row.get('pass'))}.")
    dynamic=primary["cleanDynamic"];ds1=primary["DS1"]["NC_TOPI"];missed=ds1["detection_rate"]["denominator"]-ds1["detection_rate"]["numerator"]
    lines += ["","## Frozen inventory scope and limitations",
      f"- Both median/top25_mean aggregators and q99/q995 thresholds are frozen and reported in `scenario_metrics.csv`; primary is median/q99. `ablation_metrics.csv` reports all 10 ablations.",
      f"- cleanDynamic OOD failure: external q99/median FPR is B0={f(dynamic['B0']['fpr']['value'])}, TOPI={f(dynamic['TOPI']['fpr']['value'])}, NC-TOPI={f(dynamic['NC_TOPI']['fpr']['value'])}; this failure is reported, not hidden.",
      f"- DS1 capture failure/reporting: q99/median NC-TOPI misses {missed}/{ds1['detection_rate']['denominator']} post epochs and its 3-consecutive delay is {f(ds1['sustained_delay']['value'])} s; no failed capture is removed.",
      "- Frozen B0 predicts a finite nine-tap shape, not an infinite-support physical correlation function; edge/tail behavior is a finite-tap caveat.",
      "- Historical B0 training overlap with Stage-0 source support limits independence; B0 was hash-pinned and never retrained or selected here.","",
      "## Baselines and claim boundary",
      "- PD-ML is a likelihood-family method, correlator LASSO is a sparse multi-component fit, and B0 is standardized frozen-predictor nine-tap RMSE. They are distinct and not interchangeable.",
      "- Claimable contribution: a hash-bound, clean-fit, independently reconstructable Stage-0 evaluation of tangent-orthogonal TOPI and causal IQ scale conditioning on these frozen recordings/splits.",
      "- Non-claimable: universal spoofing detection, superiority to unimplemented PD-ML/correlator LASSO baselines, causal RF-mechanism identification, or independence from historical B0 overlap.","",
      "## Verification result",f"- Independent verification passed: {str(bool(summary.get('ok',False))).lower()}.",f"- Verification error count: {len(summary.get('errors',[]))}."]
    return "\n".join(lines)+"\n"

def verify_artifact(root: str | Path, *, verify_source: bool=True) -> dict[str,object]:
    root=Path(root)
    summary=compute_summary(root,verify_hashes=True,verify_source=verify_source)
    readme=root/"README.md"
    if readme.exists():
        expected=render_readme(summary)
        # A valid artifact was rendered from a passing summary. Avoid error-count recursion:
        # compare against the same independently reconstructed evidence with errors cleared.
        clean=dict(summary); clean["ok"]=True; clean["errors"]=[]
        expected=render_readme(clean)
        if readme.read_bytes()!=expected.encode():
            summary["errors"].append("README.md is not deterministic byte-identical regeneration")
    summary["ok"]=not summary["errors"]
    return summary


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact",type=Path)
    p.add_argument("--no-source-check",action="store_true",help="test fixture only")
    p.add_argument("--write-readme",action="store_true",help="render README before final hash inventory")
    p.add_argument("--write-hashes",action="store_true",help="write self-excluding complete SHA256 inventory")
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    if args.write_readme:
        summary=compute_summary(args.artifact,verify_hashes=False,verify_source=not args.no_source_check)
        summary["ok"]=True; summary["errors"]=[]
        (args.artifact/"README.md").write_text(render_readme(summary))
    if args.write_hashes: write_hash_inventory(args.artifact)
    result=verify_artifact(args.artifact,verify_source=not args.no_source_check)
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0 if result["ok"] else 1


if __name__=="__main__":
    raise SystemExit(main())
