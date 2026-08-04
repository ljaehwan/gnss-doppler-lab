#!/usr/bin/env python3
"""Build the frozen NC-TOPI Stage-0B shortcut/calibration audit artifact."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path:sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab import nc_topi_stage0b as core
from gnss_doppler_lab import nc_topi as stage0


def _json(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
def _csv(path,rows,fields=None):
    rows=list(rows);fields=fields or (list(rows[0]) if rows else [])
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)

def _indices(data,scenario,role=None):
    return [i for i,r in enumerate(data.prn_rows) if r["scenario"]==scenario and core.parse_bool(r["valid"]) and (role is None or r["role"]==role)]

def _fit_models(data,shuffle=False):
    train=_indices(data,"cleanStatic","normal_train");meta=[{"identity":data.prn_identities[i],"scenario":"cleanStatic","role":"normal_train","phase":"normal","label":0,"valid":True} for i in train]
    perm=np.random.default_rng(0).permutation(len(train)) if shuffle else np.arange(len(train))
    models={}
    for target,column in core.TARGETS.items():
        y=np.asarray([float(data.prn_rows[i][column]) for i in train]);models[target]=core.TargetConditioner.fit(target,data.features[train],y[perm],meta)
    return models,train,perm

def _original_predictions(data):
    train=[i for i,r in enumerate(data.prn_rows) if r["scenario"]=="cleanStatic" and r["role"]=="normal_train" and core.parse_bool(r["valid"])]
    cal=[i for i,r in enumerate(data.prn_rows) if r["scenario"]=="cleanStatic" and r["role"]=="normal_calibration" and core.parse_bool(r["valid"])]
    identities=[stage0.EpochIdentity(r["physical_recording_id"],r["scenario"],r["prn"],int(r["prn_target_index"]),float(r["availability_time_s"])) for r in data.prn_rows]
    m=stage0.RobustConditioner().fit(data.features[train],[float(data.prn_rows[i]["TOPI"]) for i in train],
      provenance=stage0.FitProvenance("cleanStatic","normal_train",tuple(identities[i] for i in train)))
    cap=m.calibrate_cap(data.features[cal],provenance=stage0.FitProvenance("cleanStatic","normal_calibration",tuple(identities[i] for i in cal)))
    return m.predict_scale(data.features),cap

def _bounds(models,data):
    cal=_indices(data,"cleanStatic","normal_calibration");meta=[{"identity":data.prn_identities[i],"scenario":"cleanStatic","role":"normal_calibration","phase":"normal","label":0,"valid":True} for i in cal]
    variants={"primary":(.01,.99),"two_sided_q005_q995":(.005,.995),"lower_only_q1":(.01,None),"upper_only_q99":(None,.99),"no_clamp":(None,None)}
    out={}
    for target,m in models.items():
        out[target]={name:m.calibration_bounds(data.features[cal],meta,lower_q=q[0],upper_q=q[1]) for name,q in variants.items()}
    return out,cal

def _clamp(values,bounds):
    lo=-np.inf if bounds.lower is None else bounds.lower;hi=np.inf if bounds.upper is None else bounds.upper
    return np.clip(values,lo,hi)

def _score_prns(data,models,bounds,original_scale):
    predicted={target:model.predict_scale(data.features) for target,model in models.items()}
    cal=_indices(data,"cleanStatic","normal_calibration");reference=predicted["TOPI"][cal]
    base={name:np.asarray([float(r[col]) for r in data.prn_rows]) for name,col in core.TARGETS.items()}
    primary={t:_clamp(predicted[t],bounds[t]["primary"]) for t in core.TARGETS}
    scores={"B0":base["B0"],"TOPI":base["TOPI"],
      "NC_TOPI_original":np.asarray([float(r["NC_TOPI"]) for r in data.prn_rows]),
      "IQ_LOW_ONLY":-np.log(np.maximum(predicted["TOPI"],core.EPSILON)),
      "IQ_OOD_ONLY":core.empirical_iq_ood_score(reference,predicted["TOPI"]),
      "NC_TOPI_clamped":base["TOPI"]/np.maximum(primary["TOPI"],core.EPSILON),
      "NC_B0_clamped":base["B0"]/np.maximum(primary["B0"],core.EPSILON),
      "NC_total_clamped":base["total"]/np.maximum(primary["total"],core.EPSILON)}
    reconstructed,_=core.reconstruct_original_nc(base["TOPI"],original_scale,float(np.max(original_scale)))
    if not np.allclose(reconstructed,scores["NC_TOPI_original"],rtol=1e-12,atol=1e-12):raise RuntimeError("frozen original NC changed")
    return predicted,primary,scores

def _aggregate(data,predicted,scores,primary,bounds):
    groups={}
    for i,r in enumerate(data.prn_rows):groups.setdefault(core._event_key(r),[]).append(i)
    event_rows=[];prn_rows=[]
    for i,r in enumerate(data.prn_rows):
        row={k:r[k] for k in ("scenario","physical_recording_id","event_id","target_index","availability_time_s","source_start_s","source_end_s","role","phase","label","valid","tracked_prn_count","prn","prn_target_index","pair_sequence_index")}
        row["prn_identity_sha256"]=hashlib.sha256(data.prn_identities[i].encode()).hexdigest()
        for t in core.TARGETS:row[f"predicted_{t}_scale"]=predicted[t][i];row[f"clamped_{t}_scale"]=primary[t][i]
        for m in core.METHODS:row[m]=scores[m][i]
        row["TOPI_lower_clamp_hit"]=predicted["TOPI"][i]<bounds["TOPI"]["primary"].lower
        row["TOPI_upper_clamp_hit"]=predicted["TOPI"][i]>bounds["TOPI"]["primary"].upper
        prn_rows.append(row)
    for er in data.event_rows:
        key=core._event_key(er);ix=groups[key]
        row={k:er[k] for k in ("scenario","physical_recording_id","event_id","target_index","availability_time_s","source_start_s","source_end_s","role","phase","label","valid","tracked_prn_count")}
        for t in core.TARGETS:row[f"predicted_{t}_scale"]=float(np.median(predicted[t][ix]));row[f"clamped_{t}_scale"]=float(np.median(primary[t][ix]))
        for m in core.METHODS:row[m]=float(np.median(scores[m][ix]))
        row["common_iq_scale_equal"]=all(np.ptp(predicted[t][ix])==0 for t in core.TARGETS)
        if not row["common_iq_scale_equal"]:raise RuntimeError("event broadcast IQ scale mismatch")
        event_rows.append(row)
    return prn_rows,event_rows

def _thresholds(events):
    cal=[r for r in events if r["scenario"]=="cleanStatic" and r["role"]=="normal_calibration" and core.parse_bool(r["valid"])]
    return {m:{"value":core.higher_quantile([float(r[m]) for r in cal],.99),"quantile":.99,"method":"higher","comparison":"strict >","rows":len(cal),
               "score_digest_sha256":core._digest_array([float(r[m]) for r in cal])} for m in core.METHODS}

def _metric(value=None,reason=None):return {"value":None if value is None else float(value),"available":value is not None,"reason":reason}
def _metrics(events,thresholds):
    rows=[]
    for scenario in ("cleanStatic","cleanDynamic",*core.ATTACKS):
      all_s=[r for r in events if r["scenario"]==scenario and core.parse_bool(r["valid"])]
      for method in core.METHODS:
        threshold=thresholds[method]["value"]
        if scenario=="cleanStatic":normal=[r for r in all_s if r["role"]=="normal_holdout"]
        elif scenario=="cleanDynamic":normal=all_s
        else:normal=[r for r in all_s if r["phase"]=="stable_pre"]
        fpr=float(np.mean([float(r[method])>threshold for r in normal])) if normal else None
        base={"scenario":scenario,"method":method,"threshold":threshold,"threshold_comparison":"strict >","normal_fpr":fpr,"normal_fpr_reason":None if normal else "no eligible normal events"}
        if scenario not in core.ATTACKS:
          rows.append({**base,"roc_auc":None,"roc_auc_reason":"single-class normal diagnostic","pr_auc":None,"pr_auc_reason":"single-class normal diagnostic",
            "standardized_pauc_max_fpr_0.05":None,"pauc_reason":"single-class normal diagnostic","post_detection_rate":None,"post_detection_reason":"not attack",
            "three_consecutive_alarm_delay_s":None,"delay_reason":"not attack","persistent_alarm_ratio":None,"persistent_reason":"not attack"});continue
        eligible=[r for r in all_s if r["phase"] in ("stable_pre","post")];labels=np.asarray([int(r["label"]) for r in eligible]);values=np.asarray([float(r[method]) for r in eligible])
        if set(labels)!={0,1}:roc=pr=pauc=None;reason="class-deficient eligible events"
        else:roc=float(roc_auc_score(labels,values));pr=float(average_precision_score(labels,values));pauc=float(roc_auc_score(labels,values,max_fpr=.05));reason=None
        post=[r for r in all_s if r["phase"]=="post"];persistent=[r for r in post if float(r["source_start_s"])>=core.ONSETS[scenario]+40]
        alarm=[float(r[method])>threshold for r in eligible]
        delay=stage0.sustained_alarm_delay([float(r["availability_time_s"]) for r in eligible],alarm,
          recording_ids=[r["physical_recording_id"] for r in eligible],post_eligible_mask=[r["phase"]=="post" for r in eligible],
          onset=core.ONSETS[scenario],required=3,cadence=.5,stable_pre_mask=[r["phase"]=="stable_pre" for r in eligible])
        finite=np.isfinite(delay.delay)
        rows.append({**base,"roc_auc":roc,"roc_auc_reason":reason,"pr_auc":pr,"pr_auc_reason":reason,
          "standardized_pauc_max_fpr_0.05":pauc,"pauc_reason":reason,
          "post_detection_rate":float(np.mean([float(r[method])>threshold for r in post])) if post else None,"post_detection_reason":None if post else "no post events",
          "three_consecutive_alarm_delay_s":float(delay.delay) if finite else None,"delay_reason":None if finite else "censored: no 3-consecutive 0.5s alarm",
          "persistent_alarm_ratio":float(np.mean([float(r[method])>threshold for r in persistent])) if persistent else None,"persistent_reason":None if persistent else "no persistent events"})
    return rows

def _bootstraps(events,metrics):
    look={(r["scenario"],r["method"]):r for r in metrics};out=[]
    for scenario in ("DS7","DS8"):
      eligible=[r for r in events if r["scenario"]==scenario and core.parse_bool(r["valid"]) and r["phase"] in ("stable_pre","post")]
      labels=[int(r["label"]) for r in eligible];rec=[r["physical_recording_id"] for r in eligible];times=[float(r["availability_time_s"]) for r in eligible]
      for comparator in core.COMPARATORS:
        item=core.paired_block_bootstrap(labels,[float(r["NC_TOPI_clamped"]) for r in eligible],[float(r[comparator]) for r in eligible],rec,times)
        item.update({"scenario":scenario,"primary":"NC_TOPI_clamped","comparator":comparator,"comparison":f"NC_TOPI_clamped_minus_{comparator}","eligible_event_count":len(eligible)})
        out.append(item)
    return out

def _diagnostics(events,thresholds,bounds,namespace="primary"):
    rows=[]
    phases=("normal_train","normal_calibration","normal_holdout","stable_pre","post","persistent")
    for scenario in ("cleanStatic","cleanDynamic",*core.ATTACKS):
      for phase in phases:
        if scenario=="cleanStatic":subset=[r for r in events if r["scenario"]==scenario and r["role"]==phase and core.parse_bool(r["valid"])]
        elif scenario=="cleanDynamic":subset=[r for r in events if r["scenario"]==scenario and phase=="normal_holdout" and core.parse_bool(r["valid"])]
        elif phase=="persistent":subset=[r for r in events if r["scenario"]==scenario and r["phase"]=="post" and float(r["source_start_s"])>=core.ONSETS[scenario]+40 and core.parse_bool(r["valid"])]
        else:subset=[r for r in events if r["scenario"]==scenario and r["phase"]==phase and core.parse_bool(r["valid"])]
        if not subset:continue
        scale=np.asarray([float(r["predicted_TOPI_scale"]) for r in subset]);topi=np.asarray([float(r["TOPI"]) for r in subset]);iq=-np.log(np.maximum(scale,core.EPSILON))
        corr=float(np.corrcoef(topi,iq)[0,1]) if len(scale)>1 and np.std(scale)>0 and np.std(topi)>0 else None
        iq_auc=None
        if scenario in core.ATTACKS:
          full=[r for r in events if r["scenario"]==scenario and core.parse_bool(r["valid"]) and r["phase"] in ("stable_pre","post")]
          if full and {int(r["label"]) for r in full}=={0,1}:iq_auc=float(roc_auc_score([int(r["label"]) for r in full],[float(r["IQ_LOW_ONLY"]) for r in full]))
        original_alarm=np.asarray([float(r["NC_TOPI_original"])>thresholds["NC_TOPI_original"]["value"] for r in subset]);iq_alarm=np.asarray([float(r["IQ_LOW_ONLY"])>thresholds["IQ_LOW_ONLY"]["value"] for r in subset])
        overlap=float(np.sum(original_alarm&iq_alarm)/np.sum(original_alarm)) if np.sum(original_alarm) else None
        preserved=float(np.mean(np.sign([float(r["NC_TOPI_clamped"])-float(r["B0"]) for r in subset])==np.sign([float(r["NC_TOPI_original"])-float(r["B0"]) for r in subset])))
        lo=bounds["TOPI"]["primary"].lower;hi=bounds["TOPI"]["primary"].upper
        rows.append({"namespace":namespace,"scenario":scenario,"phase":phase,"event_count":len(subset),
          "predicted_scale_q1":core.higher_quantile(scale,.01),"predicted_scale_median":float(np.median(scale)),"predicted_scale_q99":core.higher_quantile(scale,.99),
          "lower_clamp_hit_ratio":float(np.mean(scale<lo)),"upper_clamp_hit_ratio":float(np.mean(scale>hi)),
          "iq_only_auc":iq_auc,"topi_vs_iq_pearson_correlation":corr,"original_nc_iq_low_alarm_overlap":overlap,
          "clamp_preserved_direction_ratio":preserved,"scale_below_clean_q1_highlight":bool(scenario in ("DS2","DS7","DS8") and core.higher_quantile(scale,.99)<lo)})
    return rows

def _clamp_variant_metrics(data,models,bounds):
    """Score every declared clamp without selecting any variant from attacks."""
    predicted={t:m.predict_scale(data.features) for t,m in models.items()};raw={t:np.asarray([float(r[c]) for r in data.prn_rows]) for t,c in core.TARGETS.items()}
    groups={}
    for i,r in enumerate(data.prn_rows):groups.setdefault(core._event_key(r),[]).append(i)
    rows=[]
    for variant in ("two_sided_q005_q995","lower_only_q1","upper_only_q99","no_clamp"):
      event_values={name:{} for name in ("NC_TOPI_clamped","NC_B0_clamped","NC_total_clamped")}
      for target,name in (("TOPI","NC_TOPI_clamped"),("B0","NC_B0_clamped"),("total","NC_total_clamped")):
        denom=np.maximum(_clamp(predicted[target],bounds[target][variant]),core.EPSILON);values=raw[target]/denom
        for key,ix in groups.items():event_values[name][key]=float(np.median(values[ix]))
      cal_keys=[core._event_key(r) for r in data.event_rows if r["scenario"]=="cleanStatic" and r["role"]=="normal_calibration" and core.parse_bool(r["valid"])]
      thresholds={name:core.higher_quantile([values[k] for k in cal_keys],.99) for name,values in event_values.items()}
      for scenario in ("cleanStatic","cleanDynamic",*core.ATTACKS):
       events=[r for r in data.event_rows if r["scenario"]==scenario and core.parse_bool(r["valid"])]
       for method,values in event_values.items():
        if scenario=="cleanStatic":normal=[r for r in events if r["role"]=="normal_holdout"]
        elif scenario=="cleanDynamic":normal=events
        else:normal=[r for r in events if r["phase"]=="stable_pre"]
        fpr=float(np.mean([values[core._event_key(r)]>thresholds[method] for r in normal])) if normal else None
        pauc=None;reason="single-class normal diagnostic"
        if scenario in core.ATTACKS:
          eligible=[r for r in events if r["phase"] in ("stable_pre","post")];labels=[int(r["label"]) for r in eligible]
          if set(labels)=={0,1}:pauc=float(roc_auc_score(labels,[values[core._event_key(r)] for r in eligible],max_fpr=.05));reason=None
          else:reason="class-deficient eligible events"
        rows.append({"variant":variant,"scenario":scenario,"method":method,"threshold_q99_higher":thresholds[method],"stable_or_normal_fpr":fpr,
          "standardized_pauc_max_fpr_0.05":pauc,"pauc_reason":reason,"primary":False,"decision_eligible":False})
    return rows


def _decision(metrics,boot,events,profile):
    look={(r["scenario"],r["method"]):r for r in metrics};pauc={s:{m:look[(s,m)]["standardized_pauc_max_fpr_0.05"] for m in core.METHODS} for s in ("DS7","DS8")}
    ci={s:{} for s in ("DS7","DS8")}
    for b in boot:ci[b["scenario"]][b["comparison"]]=b
    overlap={}
    for s in ("DS7","DS8"):
      subset=[r for r in events if r["scenario"]==s and core.parse_bool(r["valid"]) and r["phase"] in ("stable_pre","post")]
      oa=np.asarray([float(r["NC_TOPI_original"])>look[("cleanStatic","NC_TOPI_original")]["threshold"] for r in subset]);ia=np.asarray([float(r["IQ_LOW_ONLY"])>look[("cleanStatic","IQ_LOW_ONLY")]["threshold"] for r in subset])
      overlap[s]=float(np.sum(oa&ia)/np.sum(oa)) if np.sum(oa) else None
    evidence={"pauc":pauc,"paired_ci":ci,"alarm_overlap":overlap,"q99_fpr":{"cleanDynamic":look[("cleanDynamic","NC_TOPI_clamped")]["normal_fpr"],
      "cleanStatic_holdout":look[("cleanStatic","NC_TOPI_clamped")]["normal_fpr"],"stable_pre":{s:look[(s,"NC_TOPI_clamped")]["normal_fpr"] for s in core.ATTACKS}},"profile_d":profile}
    result=core.evaluate_decision(evidence);result["evidence"]=evidence;return result

def _plot_numeric(events,bounds):
    scenarios=("cleanStatic","cleanDynamic",*core.ATTACKS);selected=[r for r in events if core.parse_bool(r["valid"])];out={"schema":"gnss-doppler-lab.nc-topi-stage0b.plot-data.v1","common_mask":"valid frozen aggregate events","plots":{}}
    def add(n,x,m):out["plots"][n]={"numeric":x,"masks":m,"numeric_digest_sha256":core._digest_json(x),"mask_digest_sha256":core._digest_json(m)}
    scale=np.array([float(r["predicted_TOPI_scale"]) for r in selected]);edges=np.geomspace(max(scale.min(),core.EPSILON),scale.max()*(1+1e-12),41);counts={};masks={}
    for s in scenarios:
      rows=[r for r in selected if r["scenario"]==s];counts[s]=np.histogram([float(r["predicted_TOPI_scale"]) for r in rows],edges)[0].tolist();masks[s]=[r["event_id"] for r in rows]
    add("predicted_scale_distribution",{"edges":edges.tolist(),"counts":counts},masks);med={};masks={}
    for s in core.ATTACKS:
      med[s]=[];masks[s]={}
      for phase in ("stable_pre","post"):
       rows=[r for r in selected if r["scenario"]==s and r["phase"]==phase];med[s].append(float(np.median([float(r["predicted_TOPI_scale"]) for r in rows])));masks[s][phase]=[r["event_id"] for r in rows]
    add("stable_pre_post_scale",{"phase":["stable_pre","post"],"median":med},masks);add("original_vs_clamped_nc_topi",{"x":[float(r["NC_TOPI_original"]) for r in selected],"y":[float(r["NC_TOPI_clamped"]) for r in selected]},{"events":[r["event_id"] for r in selected]})
    pooled=[r for r in selected if r["scenario"] in core.ATTACKS and r["phase"] in ("stable_pre","post")];curves={}
    for m in ("B0","TOPI","IQ_LOW_ONLY","NC_B0_clamped","NC_TOPI_clamped"):
      y=np.array([int(r["label"]) for r in pooled]);order=np.argsort(-np.array([float(r[m]) for r in pooled]),kind="mergesort");z=y[order];curves[m]={"fpr":(np.cumsum(1-z)/max(1,(1-z).sum())).tolist(),"tpr":(np.cumsum(z)/max(1,z.sum())).tolist()}
    add("roc_methods",curves,{"events":[r["event_id"] for r in pooled]});clean=[r for r in selected if r["scenario"] in ("cleanStatic","cleanDynamic")];values=np.array([float(r["NC_TOPI_clamped"]) for r in clean]);edges=np.geomspace(max(values.min(),core.EPSILON),values.max()*(1+1e-12),51);counts={};masks={}
    for s in ("cleanStatic","cleanDynamic"):
      rows=[r for r in clean if r["scenario"]==s];counts[s]=np.histogram([float(r["NC_TOPI_clamped"]) for r in rows],edges)[0].tolist();masks[s]=[r["event_id"] for r in rows]
    add("clean_normal_scores",{"edges":edges.tolist(),"counts":counts},masks);lo=bounds["TOPI"]["primary"].lower;hi=bounds["TOPI"]["primary"].upper;ratios={};masks={}
    for s in scenarios:
      rows=[r for r in selected if r["scenario"]==s];x=np.array([float(r["predicted_TOPI_scale"]) for r in rows]);ratios[s]={"lower":float(np.mean(x<lo)),"upper":float(np.mean(x>hi)),"count":len(rows)};masks[s]=[r["event_id"] for r in rows]
    add("clamp_hit_ratio",{"clean_calibration_lower":lo,"clean_calibration_upper":hi,"ratios":ratios},masks);timeline={};masks={}
    for s in ("DS2","DS7","DS8"):
      rows=sorted([r for r in selected if r["scenario"]==s],key=lambda r:float(r["availability_time_s"]));timeline[s]={"time":[float(r["availability_time_s"]) for r in rows],"scale":[float(r["predicted_TOPI_scale"]) for r in rows]};masks[s]=[r["event_id"] for r in rows]
    add("shortcut_scale_timeline",timeline,masks);return out

def _plots(root,events,bounds):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    out=_plot_numeric(events,bounds);folder=root/"plots";folder.mkdir()
    def save(n):
      p=folder/(n+".png");plt.tight_layout();plt.savefig(p,dpi=120,metadata={"Software":"gnss-doppler-lab-stage0b"});plt.close();out["plots"][n]["png_sha256"]=core.sha256_file(p)
    d=out["plots"]["predicted_scale_distribution"]["numeric"];plt.figure()
    for n,y in d["counts"].items():plt.stairs(y,d["edges"],label=n)
    plt.xscale("log");plt.legend(fontsize=6);plt.title("Predicted TOPI scale by scenario");save("predicted_scale_distribution");d=out["plots"]["stable_pre_post_scale"]["numeric"];plt.figure()
    for n,y in d["median"].items():plt.plot([0,1],y,marker="o",label=n)
    plt.xticks([0,1],["stable-pre","post"]);plt.legend();save("stable_pre_post_scale");d=out["plots"]["original_vs_clamped_nc_topi"]["numeric"];plt.figure();plt.scatter(d["x"],d["y"],s=2,alpha=.2);plt.xscale("symlog");plt.yscale("symlog");plt.xlabel("original");plt.ylabel("clamped");save("original_vs_clamped_nc_topi");d=out["plots"]["roc_methods"]["numeric"];plt.figure()
    for n,c in d.items():plt.plot(c["fpr"],c["tpr"],label=n)
    plt.xlim(0,.2);plt.legend(fontsize=7);save("roc_methods");d=out["plots"]["clean_normal_scores"]["numeric"];plt.figure()
    for n,y in d["counts"].items():plt.stairs(y,d["edges"],label=n)
    plt.xscale("log");plt.legend();save("clean_normal_scores");d=out["plots"]["clamp_hit_ratio"]["numeric"];names=list(d["ratios"]);lo=[d["ratios"][x]["lower"] for x in names];hi=[d["ratios"][x]["upper"] for x in names];z=np.arange(len(names));plt.figure();plt.bar(z,lo);plt.bar(z,hi,bottom=lo);plt.xticks(z,names,rotation=30);save("clamp_hit_ratio");d=out["plots"]["shortcut_scale_timeline"]["numeric"];plt.figure()
    for n,c in d.items():plt.plot(c["time"],c["scale"],label=n)
    plt.yscale("log");plt.legend();save("shortcut_scale_timeline");return out

def _synthetic_parent_binding(parent):
    manifest=json.loads((parent/"hashes.json").read_text());expected=manifest.get("files",{});actual=sorted(str(p.relative_to(parent)) for p in parent.rglob("*") if p.is_file() and p.name!="hashes.json")
    if not isinstance(expected,dict) or set(expected)!=set(actual) or any(core.sha256_file(parent/x)!=expected[x] for x in actual):raise ValueError("synthetic parent inventory/hash mismatch")
    return {"ok":True,"synthetic_fixture":True,"parent_artifact_commit":"SYNTHETIC_TEST_ONLY","parent_generation_source_commit":"SYNTHETIC_TEST_ONLY","inventory_count":len(actual)+1,"manifest_sha256":core.sha256_file(parent/"hashes.json"),"consumed_file_hashes":{x:core.sha256_file(parent/x) for x in ("per_epoch_scores.csv","iq_context.csv")},"current_head_not_required":True}

def run(config_path,out,parent,execution_source_commit=None,synthetic_fixture_mode=False):
    config_path=Path(config_path);out=Path(out);parent=Path(parent);dirty=subprocess.run(["git","-C",str(ROOT),"status","--porcelain"],check=True,text=True,stdout=subprocess.PIPE).stdout
    if dirty and not synthetic_fixture_mode:raise RuntimeError("production audit requires a clean source worktree")
    head=subprocess.run(["git","-C",str(ROOT),"rev-parse","HEAD"],check=True,text=True,stdout=subprocess.PIPE).stdout.strip()
    if execution_source_commit is not None and execution_source_commit!=head:raise ValueError("injected execution source commit must equal clean execution HEAD")
    cfg=json.loads(config_path.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(f"non-finite config {x}")))
    expected_schema="gnss-doppler-lab.nc-topi-stage0b-audit.synthetic-test.v1" if synthetic_fixture_mode else "gnss-doppler-lab.nc-topi-stage0b-audit.v1"
    if cfg.get("schema")!=expected_schema:raise ValueError("unexpected audit config/schema for execution mode")
    binding=_synthetic_parent_binding(parent) if synthetic_fixture_mode else core.verify_parent_binding(parent,repo=ROOT);data=core.load_parent_evidence(parent,verify_binding=False);original_scale,original_cap=_original_predictions(data)
    if synthetic_fixture_mode:
      frozen=np.asarray([float(r["NC_TOPI"]) for r in data.prn_rows]);top=np.asarray([float(r["TOPI"]) for r in data.prn_rows]);got=top/np.maximum(original_scale,core.EPSILON);ae=np.abs(got-frozen);re=np.divide(ae,np.maximum(np.abs(frozen),core.EPSILON));reconstruction={"ok":True,"rows":len(frozen),"train_rows":len(_indices(data,"cleanStatic","normal_train")),"calibration_rows":len(_indices(data,"cleanStatic","normal_calibration")),"q995_upper_cap":float(original_cap),"max_absolute_error":float(ae.max()),"max_relative_error":float(re.max()),"all_rows_within_rel_abs_1e12":bool(np.allclose(got,frozen,rtol=1e-12,atol=1e-12)),"effective_scale":core.check_effective_scale(top,frozen,original_scale,original_scale),"relative_tolerance":1e-12,"absolute_tolerance":1e-12}
      if not reconstruction["all_rows_within_rel_abs_1e12"]:raise ValueError("synthetic original reconstruction mismatch")
    else:reconstruction=core.run_original_reconstruction_gate(parent,repo=ROOT)
    models,train,_=_fit_models(data);bounds,cal=_bounds(models,data);holdout=_indices(data,"cleanStatic","normal_holdout")
    if not synthetic_fixture_mode and (len(train),len(cal))!=(6074,1628):raise RuntimeError("frozen conditioner PRN split changed")
    if (set(train)&set(cal))|(set(train)&set(holdout))|(set(cal)&set(holdout)):raise RuntimeError("Profile-S roles overlap")
    clean_seal=core._digest_json({"models":{t:m.seal for t,m in models.items()},"train":core._digest_json([data.prn_identities[i] for i in train]),"calibration":core._digest_json([data.prn_identities[i] for i in cal]),"attack_fit":False});predicted,primary,scores=_score_prns(data,models,bounds,original_scale);prn,events=_aggregate(data,predicted,scores,primary,bounds)
    groups={}
    for i,row in enumerate(prn):row["original_implementation_denominator"]=float(original_scale[i]);groups.setdefault(core._event_key(row),[]).append(i)
    for row in events:row["original_implementation_denominator"]=float(np.median(original_scale[groups[core._event_key(row)]]))
    thresholds=_thresholds(events);metrics=_metrics(events,thresholds);boot=_bootstraps(events,metrics)
    if core.validate_comparison_inventory(boot):raise RuntimeError("paired comparison inventory invalid")
    shuffled,_,perm=_fit_models(data,shuffle=True);sbounds,_=_bounds(shuffled,data);sp,scp,ss=_score_prns(data,shuffled,sbounds,original_scale);_,sevents=_aggregate(data,sp,ss,scp,sbounds);diagnostics=_diagnostics(events,thresholds,bounds);clamp_metrics=_clamp_variant_metrics(data,models,bounds);sthresholds=_thresholds(sevents);smetrics=_metrics(sevents,sthresholds);sdiag=_diagnostics(sevents,sthresholds,sbounds,"time_shuffle");profile=core.check_profile_d_support(core.profile_d_support_from_parent(data));decision=_decision(metrics,boot,events,profile);parent_decision=json.loads((parent/"decision.json").read_text())
    with core.ArtifactStage(out) as stage:
      root=stage.path;(root/"diagnostics").mkdir();(root/"config.json").write_bytes(config_path.read_bytes());_json(root/"parent_inventory.json",binding);source_hashes={"runner":core.sha256_file(ROOT/"scripts/audit_nc_topi_shortcut.py"),"verifier":core.sha256_file(ROOT/"scripts/summarize_nc_topi_stage0b_audit.py"),"core":core.sha256_file(ROOT/"src/gnss_doppler_lab/nc_topi_stage0b.py"),"config":core.sha256_file(ROOT/"configs/nc_topi_stage0b_audit.json")}
      _json(root/"provenance.json",{"contract_commit_sha":"10a9be4b0c278e53278b914a7cd368175d0b4c41","parent_artifact_commit":binding["parent_artifact_commit"],"parent_generation_source_commit":binding["parent_generation_source_commit"],"execution_source_commit":head,"source_file_hashes":source_hashes,"clean_worktree_at_execution":not bool(dirty),"synthetic_fixture_mode":synthetic_fixture_mode,"worktree_status":"clean","command":" ".join(sys.argv),"library_versions":{"python":platform.python_version(),"numpy":np.__version__,"sklearn":sklearn_version},"attack_loader_opened_after_freeze":True,"attack_fit":False,"post_result_tuning":False,"raw_iq_opened":False,"stage0_decision":parent_decision.get("status"),"stage0_decision_preserved":True});_json(root/"profile_support.json",profile)
      iqf=("scenario","physical_recording_id","block_recording_id","event_id","window_bin_s","target_source_start_s","history_blocks","cadence_seconds","block_end_s","block_start_s","sample_offset","sample_count","block_features_json","context_features_json","linked_prns","linked_pair_count","history_reducer");pdigest=hashlib.sha256(np.asarray(perm,dtype=np.int64).tobytes()).hexdigest();fit={"schema":"gnss-doppler-lab.nc-topi-stage0b.fit-audit.v2","train_rows":len(train),"calibration_rows":len(cal),"holdout_rows":len(holdout),"train_calibration_holdout_disjoint":True,"role_identity_digests":{"normal_train":core._digest_json([data.prn_identities[i] for i in train]),"normal_calibration":core._digest_json([data.prn_identities[i] for i in cal]),"normal_holdout":core._digest_json([data.prn_identities[i] for i in holdout])},"iq_inventory_digest_sha256":core._digest_json([[r[k] for k in iqf] for r in data.iq_rows]),"clean_state_digest_before_attack_transform":clean_seal,"models":{t:dict(m.audit)|{"seal":m.seal,"content_digest_sha256":m.seal} for t,m in models.items()},"attack_fit":False,"time_shuffle_permutation_digest_sha256":pdigest};_json(root/"fit_audit.json",fit);_json(root/"refit_equivalence.json",reconstruction)
      bj={t:{n:{"lower":b.lower,"upper":b.upper,"lower_quantile":b.lower_quantile,"upper_quantile":b.upper_quantile,"method":"higher","calibration_digest_sha256":b.calibration_digest_sha256} for n,b in v.items()} for t,v in bounds.items()};_json(root/"scale_bounds.json",bj);_json(root/"thresholds.json",thresholds);_csv(root/"per_prn_scores.csv",prn);_csv(root/"event_scores.csv",events);_csv(root/"model_metrics.csv",metrics);_csv(root/"scale_diagnostics.csv",diagnostics);_json(root/"paired_comparisons.json",{"schema":"gnss-doppler-lab.nc-topi-stage0b.bootstrap.v1","repetitions":2000,"seed":20260803,"comparisons":boot});_json(root/"decision.json",decision)
      _csv(root/"diagnostics/clamp_variant_metrics.csv",clamp_metrics);_csv(root/"diagnostics/time_shuffle_metrics.csv",smetrics);_json(root/"diagnostics/time_shuffle_fit_audit.json",{"seed":0,"target_only_permutation":True,"same_permutation_all_targets":True,"permutation_digest_sha256":pdigest,"models":{t:dict(m.audit)|{"seal":m.seal,"content_digest_sha256":m.seal} for t,m in shuffled.items()}});_csv(root/"diagnostics/time_shuffle_scale_diagnostics.csv",sdiag);_json(root/"diagnostics/iq_scale_checks.json",{"event_common_scale_all":True,"events":len(events),"formula":"add-one two-sided <=/>=","clean_state_digest_before_attack_transform":clean_seal});_json(root/"diagnostics/second_peak_limitations.json",{"stage0_c7":False,"preserved_status":"failure","limitations":cfg["second_peak"]["limitations_exact"],"complex_synthesis_performed":False});_json(root/"plot_data.json",_plots(root,events,bounds))
      b=profile["best_counts"];readme=f"# NC-TOPI Stage-0B shortcut and calibration audit\n\nGenerated deterministically from the immutable Stage-0 artifact at `{core.PARENT_ARTIFACT_COMMIT}`.\nNo raw-IQ file was opened and no B0 model was retrained.\n\n- Stage-0 decision (preserved): **{parent_decision.get('status')}**\n- Stage-0B status: **{decision['status']}**\n- Original reconstruction rows: {reconstruction['rows']}\n- Reconstruction max absolute error: {reconstruction['max_absolute_error']:.17g}\n- Reconstruction max relative error: {reconstruction['max_relative_error']:.17g}\n- Profile D: **{profile['status']}**, best effective-support split {b['normal_train']}/{b['normal_calibration']}/{b['normal_holdout']}\n- Bootstrap: 12 paired comparisons, 2,000 requested replicates each, no IID fallback\n\nThis README reports the frozen grammar without post-result interpretation or tuning.\n";(root/"README.md").write_text(readme);cmd=[sys.executable,str(ROOT/"scripts/summarize_nc_topi_stage0b_audit.py"),str(root),"--parent",str(parent)];prepared=subprocess.run([*cmd,"--prepare","--report",str(root/"verification.json")],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
      if prepared.returncode:raise RuntimeError(f"standalone verifier prepare failed: {prepared.stdout} {prepared.stderr}")
      core.write_hash_manifest(root);final=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
      try:report=json.loads(final.stdout)
      except Exception as exc:raise RuntimeError(f"standalone verifier malformed final report: {final.stdout} {final.stderr}") from exc
      if final.returncode or not report.get("ok"):raise RuntimeError(f"standalone verifier final failed: {report}")
      stage.publish(lambda _p:report)
    return out

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,default=ROOT/"configs/nc_topi_stage0b_audit.json");p.add_argument("--parent",type=Path,default=ROOT/"artifacts/nc_topi_stage0");p.add_argument("--out",type=Path,default=ROOT/"artifacts/nc_topi_stage0b_audit");p.add_argument("--verify-after-run",action="store_true");p.add_argument("--reconstruction-only",action="store_true");p.add_argument("--execution-source-commit");p.add_argument("--synthetic-fixture-mode",action="store_true",help=argparse.SUPPRESS);return p.parse_args(argv)
def main(argv=None):
    args=parse_args(argv)
    if args.reconstruction_only:print(json.dumps(core.run_original_reconstruction_gate(args.parent,repo=ROOT),sort_keys=True,indent=2));return 0
    result=run(args.config,args.out,args.parent,args.execution_source_commit,args.synthetic_fixture_mode)
    if args.verify_after_run:
      checked=subprocess.run([sys.executable,str(ROOT/"scripts/summarize_nc_topi_stage0b_audit.py"),str(result),"--parent",str(args.parent)],text=True,stdout=subprocess.PIPE)
      if checked.returncode:raise SystemExit(checked.stdout)
    print(json.dumps({"ok":True,"output":str(result)},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
