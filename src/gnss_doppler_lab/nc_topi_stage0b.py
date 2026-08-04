"""NC-TOPI Stage-0B shortcut/calibration audit primitives.

The module consumes only the frozen Stage-0 CSV evidence.  It has no raw-IQ
loader and deliberately keeps fit, calibration, attack scoring, and publication
as separate fail-closed operations.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from . import nc_topi as stage0

FEATURE_SCHEMA = ("log_power", "log_noise_floor_scale", "spectral_flatness", "lag1_autocorr_magnitude")
TARGETS = MappingProxyType({"TOPI": "TOPI", "B0": "B0", "total": "total"})
METHODS = ("B0", "TOPI", "NC_TOPI_original", "IQ_LOW_ONLY", "IQ_OOD_ONLY",
           "NC_TOPI_clamped", "NC_B0_clamped", "NC_total_clamped")
COMPARATORS = ("B0", "TOPI", "IQ_LOW_ONLY", "IQ_OOD_ONLY", "NC_B0_clamped", "NC_total_clamped")
ATTACKS = ("DS1", "DS2", "DS3", "DS7", "DS8")
ONSETS = MappingProxyType({"DS1":100., "DS2":100., "DS3":100., "DS7":110., "DS8":110.})
PARENT_ARTIFACT_COMMIT = "6fe5315ca0d71689609895cd3b1366bcfa1b93c1"
PARENT_GENERATION_SOURCE = "c94af28795d03a91e2f4c0faa74eb19a983ed82e"
EPSILON = 1e-12


def _finite_array(value, name: str, ndim: int = 1) -> np.ndarray:
    try:
        out=np.asarray(value,dtype=np.float64)
    except (TypeError,ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if out.ndim != ndim or not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite and {ndim}-dimensional")
    return out


def _digest_array(value) -> str:
    a=np.ascontiguousarray(np.asarray(value,dtype=np.float64))
    return hashlib.sha256(str(a.shape).encode()+b"|float64|"+a.tobytes()).hexdigest()


def _digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def parse_bool(value) -> bool:
    """Parse only canonical CSV boolean spellings; reject truthy aliases."""
    if not isinstance(value,str): raise ValueError("boolean must be a string")
    if value in ("True","true"): return True
    if value in ("False","false"): return False
    raise ValueError(f"invalid boolean token: {value!r}")


def higher_quantile(values, q: float) -> float:
    a=_finite_array(values,"quantile values")
    if len(a)==0 or isinstance(q,bool) or not 0 <= float(q) <= 1: raise ValueError("invalid quantile")
    return float(np.quantile(a,float(q),method="higher"))


def strict_alarms(scores, threshold: float) -> np.ndarray:
    values=_finite_array(scores,"scores")
    if isinstance(threshold,bool) or not np.isfinite(threshold): raise ValueError("threshold must be finite")
    return values > float(threshold)


def empirical_iq_ood_score(reference, evaluated) -> np.ndarray:
    ref=np.sort(_finite_array(reference,"calibration reference"))
    values=_finite_array(evaluated,"evaluated scales")
    if len(ref)==0: raise ValueError("calibration reference cannot be empty")
    # searchsorted exactly realizes <= and >=, including ties in both tails.
    le=np.searchsorted(ref,values,side="right")
    ge=len(ref)-np.searchsorted(ref,values,side="left")
    lower=(1.+le)/(len(ref)+1.);upper=(1.+ge)/(len(ref)+1.)
    p=np.minimum(1.,2.*np.minimum(lower,upper))
    return -np.log(np.maximum(p,EPSILON))


def reconstruct_original_nc(topi, predicted_scale, upper_cap: float, *, epsilon: float=EPSILON):
    target=_finite_array(topi,"TOPI")
    scale=_finite_array(predicted_scale,"predicted scale")
    if target.shape != scale.shape or np.any(target<0) or np.any(scale<0): raise ValueError("invalid reconstruction inputs")
    if not np.isfinite(upper_cap) or upper_cap < 0 or epsilon <= 0: raise ValueError("invalid cap/epsilon")
    denominator=np.maximum(np.minimum(scale,float(upper_cap)),float(epsilon))
    return target/denominator,denominator


def check_effective_scale(topi, original_nc, materialized_denominator, *, rtol=1e-12, atol=1e-12):
    top=_finite_array(topi,"TOPI");nc=_finite_array(original_nc,"original NC");den=_finite_array(materialized_denominator,"denominator")
    if not (top.shape==nc.shape==den.shape): raise ValueError("effective scale vectors mismatch")
    if np.any((nc==0)&(top!=0)): raise ValueError("illegal zero division: nonzero TOPI with zero NC")
    ordinary=nc!=0
    if np.any(ordinary) and not np.allclose(top[ordinary]/nc[ordinary],den[ordinary],rtol=rtol,atol=atol):
        raise ValueError("inferred original scale mismatch")
    # both-zero rows intentionally do not divide; their directly materialized denominator is authoritative.
    if np.any(den <= 0) or not np.isfinite(den).all(): raise ValueError("materialized denominator invalid")
    return {"ordinary_rows":int(ordinary.sum()),"both_zero_rows":int(((top==0)&(nc==0)).sum())}


def _readonly(value) -> np.ndarray:
    raw=np.ascontiguousarray(np.asarray(value,dtype=np.float64))
    out=np.frombuffer(raw.tobytes(),dtype=np.float64).reshape(raw.shape);out.setflags(write=False);return out


@dataclass(frozen=True)
class ScaleBounds:
    lower: float | None
    upper: float | None
    lower_quantile: float | None
    upper_quantile: float | None
    calibration_digest_sha256: str


@dataclass(frozen=True)
class TargetConditioner:
    """Immutable, target-tagged normal-only Huber conditioner."""
    target: str
    median: np.ndarray
    iqr: np.ndarray
    iqr_fallback: tuple[bool,...]
    coef: np.ndarray
    intercept: float
    model_scale: float
    train_identity_digest: str
    train_identity_set: frozenset[str]
    audit: Mapping[str,object]
    seal: str

    @staticmethod
    def fit(target: str, X, y, identities: Sequence[object]) -> "TargetConditioner":
        if target not in TARGETS: raise ValueError("conditioner target must be exactly TOPI, B0, or total")
        x=_finite_array(X,"conditioner features",2);response=_finite_array(y,"conditioner target")
        ids=tuple(str(i) for i in identities)
        if x.shape[1]!=4 or len(x)!=len(response) or len(ids)!=len(response) or not len(ids): raise ValueError("conditioner dimensions invalid")
        if len(set(ids))!=len(ids): raise ValueError("duplicate fit identities")
        if np.any(response<0): raise ValueError("conditioner target must be nonnegative")
        median=np.median(x,axis=0);q75,q25=np.percentile(x,[75,25],axis=0);raw_iqr=q75-q25
        fallback=raw_iqr<=1e-8;iqr=raw_iqr.copy();iqr[fallback]=1.
        transformed=(x-median)/iqr;log_target=np.log(np.maximum(response,EPSILON))
        model=HuberRegressor(epsilon=1.35,alpha=1e-4,max_iter=1000).fit(transformed,log_target)
        identity_digest=_digest_json(list(ids));feature_digest=_digest_array(x);target_digest=_digest_array(response)
        content={"schema":"TargetConditioner.v1","target":target,"feature_schema":list(FEATURE_SCHEMA),
                 "rows":len(ids),"identity_digest_sha256":identity_digest,"feature_digest_sha256":feature_digest,
                 "target_digest_sha256":target_digest,"median":median.tolist(),"iqr":iqr.tolist(),
                 "iqr_fallback":fallback.tolist(),"coef":model.coef_.tolist(),"intercept":float(model.intercept_),
                 "model_scale":float(model.scale_),"epsilon":1.35,"alpha":1e-4,"max_iter":1000,
                 "target_transform":"log(max(target, 1e-12))","prediction_clip":[-745.,709.],
                 "fit_scenario":"cleanStatic","fit_role":"normal_train",
                 "forbidden_inputs":{"attack":False,"label":False,"scenario":False,"onset":False,"prn":False}}
        seal=_digest_json(content)
        return TargetConditioner(target,_readonly(median),_readonly(iqr),tuple(bool(x) for x in fallback),
          _readonly(model.coef_),float(model.intercept_),float(model.scale_),identity_digest,frozenset(ids),
          MappingProxyType(content),seal)

    def _validate(self):
        content=dict(self.audit)
        if (self.target not in TARGETS or content.get("target")!=self.target or tuple(content.get("feature_schema",()))!=FEATURE_SCHEMA
            or _digest_json(content)!=self.seal or not np.array_equal(self.median,np.asarray(content["median"]))
            or not np.array_equal(self.iqr,np.asarray(content["iqr"])) or not np.array_equal(self.coef,np.asarray(content["coef"]))):
            raise ValueError("conditioner immutable state seal failed")

    def predict_scale(self, X) -> np.ndarray:
        self._validate();x=_finite_array(X,"conditioner features",2)
        if x.shape[1]!=4: raise ValueError("feature schema width must be 4")
        log=(x-self.median)/self.iqr@self.coef+self.intercept
        return np.exp(np.clip(log,-745.,709.))

    def calibration_bounds(self, X, identities: Sequence[object], *, lower_q=.01, upper_q=.99) -> ScaleBounds:
        ids=tuple(str(i) for i in identities)
        if len(ids)!=len(X) or len(set(ids))!=len(ids): raise ValueError("calibration identities invalid")
        if self.train_identity_set.intersection(ids): raise ValueError("train/calibration identities must be disjoint")
        values=self.predict_scale(X)
        lo=None if lower_q is None else higher_quantile(values,lower_q)
        hi=None if upper_q is None else higher_quantile(values,upper_q)
        return ScaleBounds(lo,hi,lower_q,upper_q,_digest_array(values))


@dataclass(frozen=True)
class ParentEvidence:
    prn_rows: tuple[Mapping[str,str],...]
    event_rows: tuple[Mapping[str,str],...]
    iq_rows: tuple[Mapping[str,str],...]
    features: np.ndarray
    event_index: tuple[int,...]
    event_keys: tuple[tuple[str,str,str],...]
    prn_identities: tuple[str,...]


def _event_key(row): return (row["scenario"],row["physical_recording_id"],row["event_id"])


def _identity_string(row) -> str:
    return json.dumps([row["physical_recording_id"],row["scenario"],row["prn"],int(row["prn_target_index"]),
                       float(row["availability_time_s"])],separators=(",",":"))


def verify_parent_binding(parent: str | Path, *, repo: str | Path) -> dict[str,object]:
    parent=Path(parent);repo=Path(repo)
    if not parent.is_dir(): raise FileNotFoundError(parent)
    def git(*args):
        return subprocess.run(["git","-C",str(repo),*args],check=True,text=True,stdout=subprocess.PIPE).stdout
    resolved=git("rev-parse",PARENT_ARTIFACT_COMMIT).strip()
    if resolved!=PARENT_ARTIFACT_COMMIT: raise ValueError("parent artifact commit cannot be resolved exactly")
    manifest=json.loads((parent/"hashes.json").read_text())
    expected=manifest.get("files",{})
    actual_files=sorted(str(p.relative_to(parent)) for p in parent.rglob("*") if p.is_file() and p.name!="hashes.json")
    if set(actual_files)!=set(expected): raise ValueError("parent artifact inventory mismatch")
    bad=[name for name in actual_files if sha256_file(parent/name)!=expected[name]]
    if bad: raise ValueError(f"parent artifact hash mismatch: {bad}")
    committed=git("show",f"{PARENT_ARTIFACT_COMMIT}:artifacts/nc_topi_stage0/hashes.json")
    if json.loads(committed)!=manifest: raise ValueError("working parent manifest differs from parent artifact commit")
    provenance=json.loads((parent/"provenance.json").read_text())
    if provenance.get("source_commit")!=PARENT_GENERATION_SOURCE or provenance.get("execution_code_commit")!=PARENT_GENERATION_SOURCE:
        raise ValueError("parent generation source binding mismatch")
    consumed={name:sha256_file(parent/name) for name in ("per_epoch_scores.csv","iq_context.csv")}
    return {"ok":True,"parent_artifact_commit":resolved,"parent_generation_source_commit":PARENT_GENERATION_SOURCE,
            "inventory_count":len(actual_files)+1,"manifest_sha256":sha256_file(parent/"hashes.json"),
            "consumed_file_hashes":consumed,"current_head_not_required":True}


def load_parent_evidence(parent: str | Path, *, verify_binding=True, repo: str | Path | None=None) -> ParentEvidence:
    parent=Path(parent)
    if verify_binding: verify_parent_binding(parent,repo=repo or parent.parents[1])
    with (parent/"per_epoch_scores.csv").open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError("parent score inventory is empty")
    prn=[];events=[];groups={};event_map={}
    seen_prn=set();seen_event=set()
    for row in rows:
        if row.get("row_level") not in ("prn","event"): raise ValueError("unknown parent row level")
        row_valid=parse_bool(row.get("valid"))
        for field in ("availability_time_s","source_start_s","source_end_s"):
            if not np.isfinite(float(row[field])): raise ValueError(f"nonfinite parent {field}")
        key=_event_key(row)
        if row["row_level"]=="event":
            identity=(row["physical_recording_id"],row["scenario"],row["event_id"],int(row["target_index"]),float(row["availability_time_s"]))
            if identity in seen_event: raise ValueError("duplicate event identity")
            seen_event.add(identity);event_map[key]=row;events.append(MappingProxyType(dict(row)))
        else:
            if not row.get("prn") or row.get("prn_target_index","")=="": raise ValueError("invalid PRN identity")
            identity=_identity_string(row)
            if identity in seen_prn: raise ValueError("duplicate PRN identity")
            seen_prn.add(identity);groups.setdefault(key,[]).append(row);prn.append(row)
    if set(groups)!=set(event_map): raise ValueError("missing or extra parent PRN/event groups")
    with (parent/"iq_context.csv").open(newline="",encoding="utf-8") as f: iq=list(csv.DictReader(f))
    iq_map={};
    for row in iq:
        key=_event_key(row)
        if key in iq_map: raise ValueError("duplicate IQ event identity")
        if row["block_recording_id"]!=row["physical_recording_id"]: raise ValueError("IQ recording linkage mismatch")
        vector=json.loads(row["context_features_json"])
        if not isinstance(vector,list) or len(vector)!=4 or not np.isfinite(np.asarray(vector,dtype=float)).all():
            raise ValueError("context_features_json must use exact finite 4-feature schema")
        linked=tuple(row["linked_prns"].split(";")) if row["linked_prns"] else ()
        expected=tuple(x["prn"] for x in groups.get(key,()))
        if linked!=expected or int(row["linked_pair_count"])!=len(expected): raise ValueError("linked PRN order/inventory mismatch")
        iq_map[key]=(row,np.asarray(vector,dtype=np.float64))
    if set(iq_map)!=set(groups): raise ValueError("missing or extra IQ event rows")
    features=[];event_indices=[];keys=[];identities=[]
    ordered_event_index={_event_key(row):i for i,row in enumerate(events)}
    for row in prn:
        key=_event_key(row);event=event_map[key]
        # Event-level authority is broadcast exactly for these fields.
        for field in ("scenario","physical_recording_id","event_id","target_index","phase","label","valid","tracked_prn_count"):
            if row[field]!=event[field]: raise ValueError(f"PRN/event metadata mismatch: {field}")
        features.append(iq_map[key][1]);event_indices.append(ordered_event_index[key]);keys.append(key);identities.append(_identity_string(row))
    x=_readonly(np.asarray(features,dtype=np.float64))
    return ParentEvidence(tuple(MappingProxyType(dict(x)) for x in prn),tuple(events),
      tuple(MappingProxyType(dict(x)) for x in iq),x,tuple(event_indices),tuple(keys),tuple(identities))


def run_original_reconstruction_gate(parent: str | Path, *, repo: str | Path) -> dict[str,object]:
    data=load_parent_evidence(parent,verify_binding=True,repo=repo)
    train=[i for i,r in enumerate(data.prn_rows) if r["scenario"]=="cleanStatic" and r["role"]=="normal_train" and parse_bool(r["valid"])]
    cal=[i for i,r in enumerate(data.prn_rows) if r["scenario"]=="cleanStatic" and r["role"]=="normal_calibration" and parse_bool(r["valid"])]
    identities=[]
    for r in data.prn_rows:
        identities.append(stage0.EpochIdentity(r["physical_recording_id"],r["scenario"],r["prn"],int(r["prn_target_index"]),float(r["availability_time_s"])))
    fit=stage0.FitProvenance("cleanStatic","normal_train",tuple(identities[i] for i in train))
    model=stage0.RobustConditioner().fit(data.features[train],[float(data.prn_rows[i]["TOPI"]) for i in train],provenance=fit)
    cap_prov=stage0.FitProvenance("cleanStatic","normal_calibration",tuple(identities[i] for i in cal))
    cap=model.calibrate_cap(data.features[cal],provenance=cap_prov,q=.995)
    predicted=model.predict_scale(data.features)
    reconstructed=np.asarray([float(r["TOPI"]) for r in data.prn_rows])/np.maximum(predicted,EPSILON)
    frozen=np.asarray([float(r["NC_TOPI"]) for r in data.prn_rows])
    abs_error=np.abs(reconstructed-frozen);rel_error=abs_error/np.maximum(np.abs(frozen),EPSILON)
    if not np.allclose(reconstructed,frozen,rtol=1e-12,atol=1e-12): raise ValueError("original NC-TOPI reconstruction failed")
    boundary=check_effective_scale(np.asarray([float(r["TOPI"]) for r in data.prn_rows]),frozen,predicted)
    return {"ok":True,"rows":len(frozen),"train_rows":len(train),"calibration_rows":len(cal),"q995_upper_cap":float(cap),
            "max_absolute_error":float(abs_error.max()),"max_relative_error":float(rel_error.max()),
            "all_rows_within_rel_abs_1e12":True,
            "effective_scale":boundary,"relative_tolerance":1e-12,"absolute_tolerance":1e-12}


def paired_block_bootstrap(labels,score_a,score_b,recording_ids,times,*,reps=2000,seed=20260803):
    result=stage0.paired_pauc_delta_block_bootstrap(labels,score_a,score_b,recording_ids,times,
      max_fpr=.05,block_seconds=10.,cadence=.5,reps=reps,seed=seed)
    digest=hashlib.sha256(np.ascontiguousarray(result.replicates,dtype=np.float64).tobytes()).hexdigest()
    return {"available":bool(result.available),"reason":result.reason,"point_estimate":None if not np.isfinite(result.point_estimate) else float(result.point_estimate),
            "lower":float(result.ci[0]) if result.available else None,"upper":float(result.ci[1]) if result.available else None,
            "valid_reps":int(result.valid_reps),"reps_requested":int(reps),"complete_block_count":int(result.complete_block_count),
            "block_epoch_count":int(result.block_epoch_count),"replicate_digest_sha256":digest,"iid_fallback":False,
            "audit":dict(result.audit)}


def check_profile_d_support(events: Sequence[Mapping[str,object]], *, fit_callback: Callable | None=None) -> dict[str,object]:
    """Evaluate chronological effective support before any Profile-D fit.

    Inputs already contain the union support of target, 12-window B0 history,
    and IQ history.  The search is deterministic and never randomizes rows.
    """
    ordered=sorted(events,key=lambda x:(float(x["effective_start_s"]),float(x["effective_end_s"]),str(x["event_id"])))
    if any(float(x["effective_end_s"])<float(x["effective_start_s"]) for x in ordered):raise ValueError("reversed effective support")
    n=len(ordered);gap=10.
    starts=np.asarray([float(x["effective_start_s"]) for x in ordered]);ends=np.asarray([float(x["effective_end_s"]) for x in ordered])
    def first_after(left,value):
        hits=np.flatnonzero(starts[left:]>=value);return None if not len(hits) else left+int(hits[0])
    best=0;best_witness=None
    for k in range(1,n//3+1):
        cal_start=first_after(k,ends[k-1]+gap)
        if cal_start is None or cal_start+k>n:continue
        hold_start=first_after(cal_start+k,ends[cal_start+k-1]+gap)
        if hold_start is not None and hold_start+k<=n:best=k;best_witness=(0,k,cal_start,cal_start+k,hold_start,hold_start+k)
    best_counts={"normal_train":best,"normal_calibration":best,"normal_holdout":best}
    minimum={"normal_train":50,"normal_calibration":101,"normal_holdout":50}
    sufficient_witness=None
    if n>=sum(minimum.values()):
      train_end=minimum["normal_train"]
      cal_start=first_after(train_end,ends[train_end-1]+gap)
      if cal_start is not None:
       cal_end=cal_start+minimum["normal_calibration"]
       if cal_end<=n:
        hold_start=first_after(cal_end,ends[cal_end-1]+gap)
        if hold_start is not None and hold_start+minimum["normal_holdout"]<=n:sufficient_witness=(0,train_end,cal_start,cal_end,hold_start,hold_start+minimum["normal_holdout"])
    sufficient=sufficient_witness is not None
    if sufficient and fit_callback is not None:fit_callback(ordered,sufficient_witness)
    witness=best_witness
    support_evidence=None if witness is None else {"train_indices":[witness[0],witness[1]],"calibration_indices":[witness[2],witness[3]],"holdout_indices":[witness[4],witness[5]],
      "train_effective_end_s":float(ends[witness[1]-1]),"calibration_effective_start_s":float(starts[witness[2]]),
      "calibration_effective_end_s":float(ends[witness[3]-1]),"holdout_effective_start_s":float(starts[witness[4]])}
    return {"schema":"gnss-doppler-lab.nc-topi-stage0b.profile-d-support.v1",
      "status":"AVAILABLE" if sufficient else "INSUFFICIENT_NORMAL_SUPPORT","fit_profile_d":bool(sufficient),
      "calibrate_profile_d":bool(sufficient),"report_performance":bool(sufficient),"random_split":False,
      "chronological":True,"minimum_gap_seconds":gap,"b0_history_windows":12,"includes_iq_history":True,
      "minimum_counts":minimum,"best_counts":best_counts,"best_support_evidence":support_evidence,
      "candidate_events":n,"reason":None if sufficient else f"best chronological effective-support split {best}/{best}/{best} is below 50/101/50"}

def _finite_domain(value, name, errors, lo=0., hi=1.):
    if isinstance(value,bool) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value) or not lo<=float(value)<=hi:
        errors.append(f"{name}: expected finite [{lo},{hi}]");return None
    return float(value)


def evaluate_decision(evidence: Mapping[str,object]) -> dict[str,object]:
    errors=[]
    try: pauc=evidence["pauc"];ci=evidence["paired_ci"];fpr=evidence["q99_fpr"];profile=evidence["profile_d"]
    except (KeyError,TypeError):
        return {"status":"INCONCLUSIVE","validation_errors":["mandatory evidence mappings missing"],"shortcut_triggers":{},"tangent_conditions":{},"shortcut_precedence":True,"stage0_decision_unchanged":True}
    p={}
    for scenario in ("DS7","DS8"):
      p[scenario]={}
      for method in METHODS:
        try:value=pauc[scenario][method]
        except (KeyError,TypeError):errors.append(f"pauc.{scenario}.{method}: missing");continue
        p[scenario][method]=_finite_domain(value,f"pauc.{scenario}.{method}",errors)
    parsed_ci={}
    for scenario in ("DS7","DS8"):
      parsed_ci[scenario]={}
      for comparator in ("IQ_LOW_ONLY","IQ_OOD_ONLY","NC_B0_clamped"):
       name=f"NC_TOPI_clamped_minus_{comparator}"
       try:item=ci[scenario][name]
       except (KeyError,TypeError): errors.append(f"paired_ci.{scenario}.{name}: missing");continue
       if not item.get("available"): errors.append(f"paired_ci.{scenario}.{name}: unavailable");continue
       lo=_finite_domain(item.get("lower"),f"paired_ci.{scenario}.{name}.lower",errors,-1,1)
       hi=_finite_domain(item.get("upper"),f"paired_ci.{scenario}.{name}.upper",errors,-1,1)
       if lo is not None and hi is not None and lo>hi:errors.append(f"paired_ci.{scenario}.{name}: reversed")
       else:parsed_ci[scenario][comparator]=(lo,hi)
    def pv(s,m): return p.get(s,{}).get(m)
    a_point=any(all(pv(s,m) is not None and pv(s,"NC_TOPI_clamped") is not None and pv(s,m)>=pv(s,"NC_TOPI_clamped") for s in ("DS7","DS8")) for m in ("IQ_LOW_ONLY","IQ_OOD_ONLY"))
    a_ci=any(parsed_ci.get(s,{}).get(m,(None,None))[1] is not None and parsed_ci[s][m][1]<=0 for s in ("DS7","DS8") for m in ("IQ_LOW_ONLY","IQ_OOD_ONLY"))
    b=all("NC_B0_clamped" in parsed_ci.get(s,{}) and parsed_ci[s]["NC_B0_clamped"][0]<=0<=parsed_ci[s]["NC_B0_clamped"][1] for s in ("DS7","DS8"))
    c=any(pv(s,"NC_TOPI_clamped") is not None and pv(s,"B0") is not None and pv(s,"NC_TOPI_original") is not None and
          pv(s,"NC_TOPI_clamped")-pv(s,"B0")<=0<pv(s,"NC_TOPI_original")-pv(s,"B0") for s in ("DS7","DS8"))
    d_parts=[]
    overlaps=evidence.get("alarm_overlap",{})
    for s in ("DS7","DS8"):
      vals=(pv(s,"NC_TOPI_original"),pv(s,"B0"),pv(s,"IQ_LOW_ONLY"));over=_finite_domain(overlaps.get(s),f"alarm_overlap.{s}",errors)
      if None in vals or vals[0]-vals[1]<=0 or over is None:d_parts.append(False)
      else:d_parts.append((vals[2]-vals[1])/(vals[0]-vals[1])>=.5 and over>=.5)
    d=all(d_parts)
    clean_dynamic=_finite_domain(fpr.get("cleanDynamic"),"q99_fpr.cleanDynamic",errors)
    e=clean_dynamic is not None and clean_dynamic>=.5
    shortcut={"a_iq_point_or_ci":a_point or a_ci,"b_nc_vs_ncb0_statistically_indistinguishable":b,
              "c_clamp_reverses_positive_original_gain":c,"d_scale_only_gain_and_alarm_overlap":d,
              "e_clean_dynamic_profile_s_fpr":e}
    # A fully evaluable true shortcut has precedence even if unrelated evidence is missing.
    if any(shortcut.values()): return {"status":"IQ_SHORTCUT_DOMINATED","validation_errors":errors,"shortcut_triggers":shortcut,"tangent_conditions":{},"shortcut_precedence":True,"stage0_decision_unchanged":True}
    stable=fpr.get("stable_pre",{})
    hold=_finite_domain(fpr.get("cleanStatic_holdout"),"q99_fpr.cleanStatic_holdout",errors)
    stable_values={s:_finite_domain(stable.get(s),f"q99_fpr.stable_pre.{s}",errors) for s in ATTACKS}
    tangent={
      "t1_point_over_ncb0":all(pv(s,"NC_TOPI_clamped") is not None and pv(s,"NC_B0_clamped") is not None and pv(s,"NC_TOPI_clamped")>pv(s,"NC_B0_clamped") for s in ("DS7","DS8")),
      "t2_point_over_iq_low":all(pv(s,"NC_TOPI_clamped") is not None and pv(s,"IQ_LOW_ONLY") is not None and pv(s,"NC_TOPI_clamped")>pv(s,"IQ_LOW_ONLY") for s in ("DS7","DS8")),
      "t3_point_over_iq_ood":all(pv(s,"NC_TOPI_clamped") is not None and pv(s,"IQ_OOD_ONLY") is not None and pv(s,"NC_TOPI_clamped")>pv(s,"IQ_OOD_ONLY") for s in ("DS7","DS8")),
      "t4_at_least_one_positive_ncb0_ci":any(parsed_ci.get(s,{}).get("NC_B0_clamped",(None,None))[0] is not None and parsed_ci[s]["NC_B0_clamped"][0]>0 for s in ("DS7","DS8")),
      "t5_positive_clamped_nc_minus_b0_points":all(pv(s,"NC_TOPI_clamped") is not None and pv(s,"B0") is not None and pv(s,"NC_TOPI_clamped")>pv(s,"B0") for s in ("DS7","DS8")),
      "t6_cleanstatic_holdout":hold is not None and hold<=.02,
      "t7_all_stable_pre":all(v is not None and v<.05 for v in stable_values.values()),
      "profile_d_conditional_gate":profile.get("status")=="INSUFFICIENT_NORMAL_SUPPORT" or (profile.get("status")=="AVAILABLE" and _finite_domain(profile.get("holdout_fpr"),"profile_d.holdout_fpr",errors) is not None and profile["holdout_fpr"]<.05)}
    status="TANGENT_SUPPORTED" if not errors and all(tangent.values()) else "INCONCLUSIVE"
    return {"status":status,"validation_errors":errors,"shortcut_triggers":shortcut,"tangent_conditions":tangent,
            "shortcut_precedence":True,"stage0_decision_unchanged":True}


class ArtifactStage:
    def __init__(self,final_path: str|Path):
        self.final=Path(final_path);self.path=self.final.parent/f".{self.final.name}.tmp.{os.getpid()}";self._published=False
    def __enter__(self):
        if self.final.exists():raise FileExistsError(f"artifact output already exists: {self.final}")
        if self.path.exists():raise FileExistsError(f"artifact stage already exists: {self.path}")
        self.path.mkdir(parents=True);return self
    def publish(self,verifier:Callable[[Path],Mapping[str,object]]):
        result=verifier(self.path)
        if not result.get("ok"):raise RuntimeError(f"independent artifact verifier failed: {result.get('errors')}")
        if self.final.exists():raise FileExistsError(self.final)
        os.replace(self.path,self.final);self._published=True
    def __exit__(self,kind,value,traceback):
        if kind is not None and self.path.exists():
            (self.path/"FAILED.json").write_text(json.dumps({"exception_type":kind.__name__,"message":str(value),"published":False,"unix_time":time.time()},sort_keys=True,indent=2)+"\n")
        return False


def hash_entries(root: str|Path):
    root=Path(root);entries=[]
    for p in sorted((x for x in root.rglob("*") if x.is_file() and x.name!="hashes.json"),key=lambda x:str(x.relative_to(root))):
        entries.append({"relative_path":str(p.relative_to(root)),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
    return entries


def write_hash_manifest(root: str|Path):
    root=Path(root);payload={"schema":"gnss-doppler-lab.nc-topi-stage0b.hashes.v1","algorithm":"SHA-256","self_excluded":"hashes.json","files":hash_entries(root)}
    (root/"hashes.json").write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n");return payload
