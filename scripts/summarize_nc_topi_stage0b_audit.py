#!/usr/bin/env python3
"""Independent verifier for the NC-TOPI Stage-0B audit artifact."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path:sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab import nc_topi_stage0b as core

REQUIRED=("README.md","config.json","provenance.json","data_manifest.json","parent_inventory.json","profile_support.json",
 "fit_audit.json","refit_equivalence.json","scale_bounds.json","thresholds.json","per_prn_scores.csv","event_scores.csv",
 "scenario_metrics.csv","bootstrap_results.json","decision.json","verification.json","hashes.json","scale_diagnostics.csv",
 "model_metrics.csv","paired_comparisons.json","profile_d_support.json","per_epoch_audit_scores.csv","plot_provenance.json")
DIAGNOSTIC=("diagnostics/clamp_variant_metrics.csv","diagnostics/time_shuffle_metrics.csv","diagnostics/time_shuffle_fit_audit.json",
 "diagnostics/iq_scale_checks.json","diagnostics/second_peak_limitations.json")
PLOTS=("plots/predicted_scale_distribution.png","plots/stable_pre_post_scale.png","plots/original_vs_clamped_nc_topi.png",
 "plots/roc_methods.png","plots/clean_normal_scores.png","plots/clamp_hit_ratio.png","plots/shortcut_scale_timeline.png")

def _read_csv(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def _close(a,b):return np.isclose(float(a),float(b),rtol=1e-12,atol=1e-12)

def verify_hashes(root):
    root=Path(root);errors=[]
    try:stored=json.loads((root/"hashes.json").read_text())
    except Exception as exc:return {"ok":False,"errors":[f"hash manifest unreadable: {exc}"]}
    if stored.get("algorithm")!="SHA-256" or stored.get("self_excluded")!="hashes.json":errors.append("hash manifest header invalid")
    entries=stored.get("files")
    if not isinstance(entries,list):return {"ok":False,"errors":errors+["hash files must be a list"]}
    names=[x.get("relative_path") for x in entries]
    if names!=sorted(names) or len(names)!=len(set(names)):errors.append("hash entries not sorted/unique")
    actual=core.hash_entries(root)
    if entries!=actual:errors.append("complete self-excluding hash inventory mismatch")
    return {"ok":not errors,"errors":errors,"file_count":len(actual)}

def _fit_independent(data,shuffle=False):
    roles={core._event_key(r):r["role"] for r in data.event_rows}
    train=[i for i,r in enumerate(data.prn_rows) if r["scenario"]=="cleanStatic" and roles[core._event_key(r)]=="normal_train" and core.parse_bool(r["valid"])]
    ids=[data.prn_identities[i] for i in train];perm=np.random.default_rng(0).permutation(len(train)) if shuffle else np.arange(len(train));models={}
    for target,column in core.TARGETS.items():
      y=np.asarray([float(data.prn_rows[i][column]) for i in train]);models[target]=core.TargetConditioner.fit(target,data.features[train],y[perm],ids)
    return models,train,perm

def _semantic_verify(root,parent,repo,errors):
    binding=core.verify_parent_binding(parent,repo=repo);data=core.load_parent_evidence(parent,verify_binding=False)
    refit=core.run_original_reconstruction_gate(parent,repo=repo);stored_refit=json.loads((root/"refit_equivalence.json").read_text())
    for key in ("rows","train_rows","calibration_rows","q995_upper_cap","max_absolute_error","max_relative_error"):
      if key not in stored_refit or (isinstance(refit[key],float) and not _close(refit[key],stored_refit[key])) or (not isinstance(refit[key],float) and refit[key]!=stored_refit[key]):errors.append(f"refit equivalence tampered: {key}")
    parent_stored=json.loads((root/"parent_inventory.json").read_text())
    if parent_stored!=binding:errors.append("parent inventory/binding tampered")
    models,train,perm=_fit_independent(data);fit=json.loads((root/"fit_audit.json").read_text())
    if fit.get("train_rows")!=len(train):errors.append("fit train inventory mismatch")
    for target,model in models.items():
      item=fit.get("models",{}).get(target,{})
      if item.get("seal")!=model.seal or item.get("target")!=target:errors.append(f"independent {target} conditioner refit mismatch")
    shuffled,_,shuffle_perm=_fit_independent(data,shuffle=True)
    shuffle_fit=json.loads((root/"diagnostics/time_shuffle_fit_audit.json").read_text())
    expected_perm_digest=hashlib.sha256(np.asarray(shuffle_perm,dtype=np.int64).tobytes()).hexdigest()
    if shuffle_fit.get("seed")!=0 or shuffle_fit.get("permutation_digest_sha256")!=expected_perm_digest:errors.append("time-shuffle permutation mismatch")
    for target,model in shuffled.items():
      if shuffle_fit.get("models",{}).get(target,{}).get("seal")!=model.seal:errors.append(f"time-shuffle {target} refit mismatch")
    bounds=json.loads((root/"scale_bounds.json").read_text());roles={core._event_key(r):r["role"] for r in data.event_rows}
    cal=[i for i,r in enumerate(data.prn_rows) if r["scenario"]=="cleanStatic" and roles[core._event_key(r)]=="normal_calibration" and core.parse_bool(r["valid"])]
    ids=[data.prn_identities[i] for i in cal];pred={t:m.predict_scale(data.features) for t,m in models.items()}
    for target,m in models.items():
      for name,q in {"primary":(.01,.99),"two_sided_q005_q995":(.005,.995),"lower_only_q1":(.01,None),"upper_only_q99":(None,.99),"no_clamp":(None,None)}.items():
       b=m.calibration_bounds(data.features[cal],ids,lower_q=q[0],upper_q=q[1]);stored=bounds[target][name]
       if (b.lower is None)!=(stored["lower"] is None) or (b.upper is None)!=(stored["upper"] is None):errors.append(f"{target}/{name} clamp null mismatch")
       if b.lower is not None and not _close(b.lower,stored["lower"]):errors.append(f"{target}/{name} lower clamp mismatch")
       if b.upper is not None and not _close(b.upper,stored["upper"]):errors.append(f"{target}/{name} upper clamp mismatch")
    prn=_read_csv(root/"per_prn_scores.csv")
    if len(prn)!=len(data.prn_rows):errors.append("PRN row inventory mismatch");return
    reference=pred["TOPI"][cal];ood=core.empirical_iq_ood_score(reference,pred["TOPI"])
    raw={t:np.asarray([float(r[c]) for r in data.prn_rows]) for t,c in core.TARGETS.items()};clamped={}
    for t in core.TARGETS:
      b=bounds[t]["primary"];clamped[t]=np.clip(pred[t],b["lower"],b["upper"])
    expected={"B0":raw["B0"],"TOPI":raw["TOPI"],"NC_TOPI_original":np.asarray([float(r["NC_TOPI"]) for r in data.prn_rows]),
      "IQ_LOW_ONLY":-np.log(np.maximum(pred["TOPI"],core.EPSILON)),"IQ_OOD_ONLY":ood,
      "NC_TOPI_clamped":raw["TOPI"]/np.maximum(clamped["TOPI"],core.EPSILON),
      "NC_B0_clamped":raw["B0"]/np.maximum(clamped["B0"],core.EPSILON),"NC_total_clamped":raw["total"]/np.maximum(clamped["total"],core.EPSILON)}
    for i,(source,row) in enumerate(zip(data.prn_rows,prn)):
      if any(row[k]!=source[k] for k in ("scenario","physical_recording_id","event_id","prn","prn_target_index","valid","label","phase")):errors.append(f"PRN identity/mask mismatch at {i}");break
    for method,value in expected.items():
      stored=np.asarray([float(r[method]) for r in prn])
      if not np.allclose(stored,value,rtol=1e-12,atol=1e-12):errors.append(f"PRN method recomputation mismatch: {method}")
    events=_read_csv(root/"event_scores.csv");groups={}
    for i,r in enumerate(prn):groups.setdefault((r["scenario"],r["physical_recording_id"],r["event_id"]),[]).append(i)
    if len(events)!=len(data.event_rows):errors.append("event inventory mismatch")
    else:
      for er,source in zip(events,data.event_rows):
       key=core._event_key(source);ix=groups[key]
       if core._event_key(er)!=key or any(not _close(er[m],np.median([float(prn[i][m]) for i in ix])) for m in core.METHODS):errors.append(f"event median/identity mismatch: {key}");break
    thresholds=json.loads((root/"thresholds.json").read_text());cal_events=[r for r in events if r["scenario"]=="cleanStatic" and r["role"]=="normal_calibration" and core.parse_bool(r["valid"])]
    for m in core.METHODS:
      value=core.higher_quantile([float(r[m]) for r in cal_events],.99)
      if not _close(value,thresholds[m]["value"]) or thresholds[m].get("comparison")!="strict >":errors.append(f"threshold mismatch: {m}")
    metrics=_read_csv(root/"scenario_metrics.csv");lookup={(r["scenario"],r["method"]):r for r in metrics}
    for scenario in core.ATTACKS:
      eligible=[r for r in events if r["scenario"]==scenario and core.parse_bool(r["valid"]) and r["phase"] in ("stable_pre","post")];labels=[int(r["label"]) for r in eligible]
      for m in core.METHODS:
       row=lookup.get((scenario,m));values=[float(r[m]) for r in eligible]
       if row is None or not _close(row["roc_auc"],roc_auc_score(labels,values)) or not _close(row["pr_auc"],average_precision_score(labels,values)) or not _close(row["standardized_pauc_max_fpr_0.05"],roc_auc_score(labels,values,max_fpr=.05)):errors.append(f"point metrics mismatch: {scenario}/{m}")
    boot=json.loads((root/"bootstrap_results.json").read_text()).get("comparisons",[])
    if len(boot)!=12:errors.append("paired comparison inventory must be exactly 12")
    for item in boot:
      scenario=item["scenario"];comp=item["comparator"];eligible=[r for r in events if r["scenario"]==scenario and core.parse_bool(r["valid"]) and r["phase"] in ("stable_pre","post")]
      got=core.paired_block_bootstrap([int(r["label"]) for r in eligible],[float(r["NC_TOPI_clamped"]) for r in eligible],[float(r[comp]) for r in eligible],[r["physical_recording_id"] for r in eligible],[float(r["availability_time_s"]) for r in eligible])
      for key in ("available","reason","valid_reps","reps_requested","replicate_digest_sha256","iid_fallback"):
       if got[key]!=item.get(key):errors.append(f"bootstrap mismatch {scenario}/{comp}/{key}")
      if got["available"] and (not _close(got["lower"],item["lower"]) or not _close(got["upper"],item["upper"])):errors.append(f"bootstrap CI mismatch {scenario}/{comp}")
    profile=json.loads((root/"profile_d_support.json").read_text());dynamic=[{"event_id":r["event_id"],"effective_start_s":float(r["source_start_s"])-5.5,"effective_end_s":float(r["source_end_s"])} for r in data.event_rows if r["scenario"]=="cleanDynamic"]
    if profile!=core.check_profile_d_support(dynamic):errors.append("Profile D support pre-rule mismatch")
    stored_decision=json.loads((root/"decision.json").read_text());pauc={s:{m:float(lookup[(s,m)]["standardized_pauc_max_fpr_0.05"]) for m in core.METHODS} for s in ("DS7","DS8")}
    ci={s:{} for s in ("DS7","DS8")}
    for item in boot:ci[item["scenario"]][item["comparison"]]=item
    overlap={}
    for scenario in ("DS7","DS8"):
      eligible=[r for r in events if r["scenario"]==scenario and core.parse_bool(r["valid"]) and r["phase"] in ("stable_pre","post")]
      oa=np.asarray([float(r["NC_TOPI_original"])>float(thresholds["NC_TOPI_original"]["value"]) for r in eligible]);ia=np.asarray([float(r["IQ_LOW_ONLY"])>float(thresholds["IQ_LOW_ONLY"]["value"]) for r in eligible])
      overlap[scenario]=float(np.sum(oa&ia)/np.sum(oa)) if np.sum(oa) else None
    evidence={"pauc":pauc,"paired_ci":ci,"alarm_overlap":overlap,"q99_fpr":{"cleanDynamic":float(lookup[("cleanDynamic","NC_TOPI_clamped")]["normal_fpr"]),"cleanStatic_holdout":float(lookup[("cleanStatic","NC_TOPI_clamped")]["normal_fpr"]),"stable_pre":{x:float(lookup[(x,"NC_TOPI_clamped")]["normal_fpr"]) for x in core.ATTACKS}},"profile_d":profile}
    regenerated=core.evaluate_decision(evidence)
    for key in ("status","validation_errors","shortcut_triggers","tangent_conditions","shortcut_precedence","stage0_decision_unchanged"):
      if stored_decision.get(key)!=regenerated.get(key):errors.append(f"shortcut-first decision mismatch: {key}")
    second=json.loads((root/"diagnostics/second_peak_limitations.json").read_text());cfg=json.loads((root/"config.json").read_text())
    if second.get("stage0_c7") is not False or second.get("limitations")!=cfg["second_peak"]["limitations_exact"] or second.get("complex_synthesis_performed") is not False:errors.append("Stage-0 c7/limitations not preserved")
    parent_decision=json.loads((parent/"decision.json").read_text());expected_readme=f"""# NC-TOPI Stage-0B shortcut and calibration audit

Generated deterministically from the immutable Stage-0 artifact at `{core.PARENT_ARTIFACT_COMMIT}`.
No raw-IQ file was opened and no B0 model was retrained.

- Stage-0 decision (preserved): **{parent_decision.get('status')}**
- Stage-0B status: **{stored_decision['status']}**
- Original reconstruction rows: {refit['rows']}
- Reconstruction max absolute error: {refit['max_absolute_error']:.17g}
- Reconstruction max relative error: {refit['max_relative_error']:.17g}
- Profile D: **{profile['status']}**, best effective-support split 33/33/33
- Bootstrap: 12 paired comparisons, 2,000 requested replicates each, no IID fallback

This README reports the frozen grammar without post-result interpretation or tuning.
"""
    if (root/"README.md").read_text()!=expected_readme:errors.append("README not deterministically regenerated from verified evidence")
    plots=json.loads((root/"plot_provenance.json").read_text()).get("plots",[])
    if plots!=list(PLOTS):errors.append("plot provenance/inventory mismatch")

def verify_artifact(root,parent=None,repo=None):
    root=Path(root);parent=Path(parent or ROOT/"artifacts/nc_topi_stage0");repo=Path(repo or ROOT);errors=[]
    for name in (*REQUIRED,*DIAGNOSTIC,*PLOTS):
      p=root/name
      if not p.is_file() or p.stat().st_size==0:errors.append(f"missing/empty required file: {name}")
    h=verify_hashes(root);errors.extend(h["errors"])
    try:
      if (root/"config.json").read_bytes()!=(repo/"configs/nc_topi_stage0b_audit.json").read_bytes():errors.append("config copy is not byte-identical")
      for name in PLOTS:
       if not (root/name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):errors.append(f"invalid PNG: {name}")
      prov=json.loads((root/"provenance.json").read_text())
      if prov.get("raw_iq_opened") is not False or prov.get("post_result_tuning") is not False or prov.get("parent_generation_source_commit")!=core.PARENT_GENERATION_SOURCE:errors.append("forbidden-operation/provenance attestation mismatch")
      _semantic_verify(root,parent,repo,errors)
      decision=json.loads((root/"decision.json").read_text())
      if decision.get("status") not in ("IQ_SHORTCUT_DOMINATED","TANGENT_SUPPORTED","INCONCLUSIVE") or decision.get("stage0_decision_unchanged") is not True:errors.append("decision status/preservation invalid")
    except Exception as exc:errors.append(f"independent recomputation failed: {type(exc).__name__}: {exc}")
    return {"ok":not errors,"errors":errors,"hash_file_count":h.get("file_count",0),"independent_refit":not errors,"raw_iq_opened":False}

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("artifact",type=Path,nargs="?",default=ROOT/"artifacts/nc_topi_stage0b_audit");p.add_argument("--parent",type=Path,default=ROOT/"artifacts/nc_topi_stage0");return p.parse_args(argv)
def main(argv=None):
    a=parse_args(argv);report=verify_artifact(a.artifact,parent=a.parent,repo=ROOT);print(json.dumps(report,sort_keys=True,indent=2));return 0 if report["ok"] else 1
if __name__=="__main__":raise SystemExit(main())
