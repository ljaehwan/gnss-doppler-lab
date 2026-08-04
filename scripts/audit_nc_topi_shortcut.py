#!/usr/bin/env python3
"""Build the frozen NC-TOPI Stage-0B shortcut/calibration audit artifact."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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


def _json(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
def _csv(path,rows,fields=None):
    rows=list(rows);fields=fields or (list(rows[0]) if rows else [])
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)

def _event_roles(data):return {core._event_key(r):r["role"] for r in data.event_rows}
def _indices(data,scenario,role=None):
    er=_event_roles(data)
    return [i for i,r in enumerate(data.prn_rows) if r["scenario"]==scenario and core.parse_bool(r["valid"]) and (role is None or er[core._event_key(r)]==role)]

def _fit_models(data,shuffle=False):
    train=_indices(data,"cleanStatic","normal_train");ids=[data.prn_identities[i] for i in train]
    perm=np.random.default_rng(0).permutation(len(train)) if shuffle else np.arange(len(train))
    models={}
    for target,column in core.TARGETS.items():
        y=np.asarray([float(data.prn_rows[i][column]) for i in train]);models[target]=core.TargetConditioner.fit(target,data.features[train],y[perm],ids)
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
    cal=_indices(data,"cleanStatic","normal_calibration");ids=[data.prn_identities[i] for i in cal]
    variants={"primary":(.01,.99),"two_sided_q005_q995":(.005,.995),"lower_only_q1":(.01,None),"upper_only_q99":(None,.99),"no_clamp":(None,None)}
    out={}
    for target,m in models.items():
        out[target]={name:m.calibration_bounds(data.features[cal],ids,lower_q=q[0],upper_q=q[1]) for name,q in variants.items()}
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
        row={k:r[k] for k in ("scenario","physical_recording_id","event_id","target_index","availability_time_s","source_start_s","source_end_s","role","phase","label","valid","tracked_prn_count","prn","prn_target_index")}
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

def _plots(root,events,metrics):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    plot=root/"plots";plot.mkdir();made=[]
    def save(name):
      p=plot/name;plt.tight_layout();plt.savefig(p,dpi=120);plt.close();
      if p.stat().st_size<100:raise RuntimeError(f"empty plot {name}")
      made.append(str(Path("plots")/name))
    scenarios=("cleanStatic","cleanDynamic",*core.ATTACKS)
    plt.figure();
    for s in scenarios:
      x=[float(r["predicted_TOPI_scale"]) for r in events if r["scenario"]==s];plt.hist(x,bins=40,histtype="step",label=s,density=True)
    plt.xscale("log");plt.legend(fontsize=6);plt.title("Predicted TOPI scale by scenario");save("predicted_scale_distribution.png")
    plt.figure();
    for s in core.ATTACKS:
      vals=[[float(r["predicted_TOPI_scale"]) for r in events if r["scenario"]==s and r["phase"]==p] for p in ("stable_pre","post")];plt.plot([0,1],[np.median(v) for v in vals],marker="o",label=s)
    plt.xticks([0,1],["stable-pre","post"]);plt.legend();save("stable_pre_post_scale.png")
    plt.figure();x=[float(r["NC_TOPI_original"]) for r in events];y=[float(r["NC_TOPI_clamped"]) for r in events];plt.scatter(x,y,s=2,alpha=.2);plt.xscale("symlog");plt.yscale("symlog");plt.xlabel("original");plt.ylabel("clamped");save("original_vs_clamped_nc_topi.png")
    plt.figure();
    for method in ("B0","TOPI","IQ_LOW_ONLY","NC_B0_clamped","NC_TOPI_clamped"):
      pooled=[r for r in events if r["scenario"] in core.ATTACKS and core.parse_bool(r["valid"]) and r["phase"] in ("stable_pre","post")];y=np.asarray([int(r["label"]) for r in pooled]);score=np.asarray([float(r[method]) for r in pooled]);order=np.argsort(-score);ys=y[order];tp=np.cumsum(ys)/max(1,ys.sum());fp=np.cumsum(1-ys)/max(1,(1-ys).sum());plt.plot(fp,tp,label=method)
    plt.xlim(0,.2);plt.legend(fontsize=7);save("roc_methods.png")
    plt.figure();
    for s in ("cleanStatic","cleanDynamic"):
      x=[float(r["NC_TOPI_clamped"]) for r in events if r["scenario"]==s and core.parse_bool(r["valid"])];plt.hist(x,bins=50,histtype="step",label=s,density=True)
    plt.xscale("log");plt.legend();save("clean_normal_scores.png")
    plt.figure();names=[];lo=[];hi=[]
    for s in scenarios:
      x=np.asarray([float(r["predicted_TOPI_scale"]) for r in events if r["scenario"]==s]);names.append(s);lo.append(np.mean(x<np.quantile(x,.01)));hi.append(np.mean(x>np.quantile(x,.99)))
    z=np.arange(len(names));plt.bar(z,lo,label="lower");plt.bar(z,hi,bottom=lo,label="upper");plt.xticks(z,names,rotation=30);plt.legend();save("clamp_hit_ratio.png")
    plt.figure();
    for s in ("DS2","DS7","DS8"):
      x=[float(r["availability_time_s"]) for r in events if r["scenario"]==s];y=[float(r["predicted_TOPI_scale"]) for r in events if r["scenario"]==s];plt.plot(x,y,label=s,alpha=.7)
    plt.yscale("log");plt.legend();save("shortcut_scale_timeline.png")
    return made

def run(config_path,out,parent):
    started=time.monotonic();config_path=Path(config_path);out=Path(out);parent=Path(parent)
    source_status=subprocess.run(["git","-C",str(ROOT),"status","--porcelain"],check=True,text=True,stdout=subprocess.PIPE).stdout
    if source_status:raise RuntimeError("production audit requires a clean source worktree")
    source_head=subprocess.run(["git","-C",str(ROOT),"rev-parse","HEAD"],check=True,text=True,stdout=subprocess.PIPE).stdout.strip()
    cfg=json.loads(config_path.read_text());
    if cfg.get("schema")!="gnss-doppler-lab.nc-topi-stage0b-audit.v1":raise ValueError("unexpected audit config")
    binding=core.verify_parent_binding(parent,repo=ROOT);data=core.load_parent_evidence(parent,verify_binding=False)
    reconstruction=core.run_original_reconstruction_gate(parent,repo=ROOT);original_scale,original_cap=_original_predictions(data)
    models,train,perm=_fit_models(data);bounds,cal=_bounds(models,data);holdout=_indices(data,"cleanStatic","normal_holdout")
    if set(train)&set(cal) or set(train)&set(holdout) or set(cal)&set(holdout):raise RuntimeError("Profile-S train/calibration/holdout roles overlap")
    predicted,primary,scores=_score_prns(data,models,bounds,original_scale)
    prn,events=_aggregate(data,predicted,scores,primary,bounds);thresholds=_thresholds(events);metrics=_metrics(events,thresholds);boot=_bootstraps(events,metrics)
    shuffled,_,shuffle_perm=_fit_models(data,shuffle=True);shuffle_bounds,_=_bounds(shuffled,data);sp,scp,ss=_score_prns(data,shuffled,shuffle_bounds,original_scale);_,shuffle_events=_aggregate(data,sp,ss,scp,shuffle_bounds)
    diagnostics=_diagnostics(events,thresholds,bounds);clamp_variant_metrics=_clamp_variant_metrics(data,models,bounds);shuffle_thresholds=_thresholds(shuffle_events);shuffle_metrics=_metrics(shuffle_events,shuffle_thresholds);shuffle_diag=_diagnostics(shuffle_events,shuffle_thresholds,shuffle_bounds,"time_shuffle")
    dynamic=[{"event_id":r["event_id"],"effective_start_s":float(r["source_start_s"])-5.5,"effective_end_s":float(r["source_end_s"])} for r in data.event_rows if r["scenario"]=="cleanDynamic"]
    profile=core.check_profile_d_support(dynamic);decision=_decision(metrics,boot,events,profile)
    parent_decision=json.loads((parent/"decision.json").read_text())
    with core.ArtifactStage(out) as stage:
      root=stage.path;(root/"diagnostics").mkdir()
      (root/"config.json").write_bytes(config_path.read_bytes());_json(root/"parent_inventory.json",binding)
      _json(root/"data_manifest.json",{"parent_consumed":binding["consumed_file_hashes"],"raw_iq_opened":False,"raw_iq_extraction_invoked":False,"prn_rows":len(prn),"event_rows":len(events)})
      git_head=source_head
      provenance={"contract_commit_sha":"b8ed4a2895700c90ce2cebfa3ad1a011652ea388","parent_artifact_commit":core.PARENT_ARTIFACT_COMMIT,"parent_generation_source_commit":core.PARENT_GENERATION_SOURCE,
        "execution_source_commit":git_head,"worktree_status":"clean","command":" ".join(sys.argv),"library_versions":{"python":platform.python_version(),"numpy":np.__version__,"sklearn":sklearn_version},
        "attack_loader_opened_after_freeze":True,"attack_fit":False,"post_result_tuning":False,"raw_iq_opened":False,"stage0_decision":parent_decision.get("status"),"stage0_decision_preserved":True}
      _json(root/"provenance.json",provenance);_json(root/"profile_support.json",profile);_json(root/"profile_d_support.json",profile)
      fit_audit={"schema":"gnss-doppler-lab.nc-topi-stage0b.fit-audit.v1","train_rows":len(train),"calibration_rows":len(cal),"holdout_rows":len(holdout),"train_calibration_holdout_disjoint":not bool((set(train)&set(cal))|(set(train)&set(holdout))|(set(cal)&set(holdout))),
        "role_identity_digests":{"normal_train":core._digest_json([data.prn_identities[i] for i in train]),"normal_calibration":core._digest_json([data.prn_identities[i] for i in cal]),"normal_holdout":core._digest_json([data.prn_identities[i] for i in holdout])},
        "models":{t:dict(m.audit)|{"seal":m.seal,"content_digest_sha256":m.seal} for t,m in models.items()},"attack_fit":False,"time_shuffle_permutation_digest_sha256":hashlib.sha256(np.asarray(shuffle_perm,dtype=np.int64).tobytes()).hexdigest()}
      _json(root/"fit_audit.json",fit_audit);_json(root/"refit_equivalence.json",reconstruction)
      bounds_json={t:{n:{"lower":b.lower,"upper":b.upper,"lower_quantile":b.lower_quantile,"upper_quantile":b.upper_quantile,"method":"higher","calibration_digest_sha256":b.calibration_digest_sha256} for n,b in v.items()} for t,v in bounds.items()}
      _json(root/"scale_bounds.json",bounds_json);_json(root/"thresholds.json",thresholds)
      _csv(root/"per_prn_scores.csv",prn);_csv(root/"per_epoch_audit_scores.csv",prn);_csv(root/"event_scores.csv",events);_csv(root/"scenario_metrics.csv",metrics);_csv(root/"model_metrics.csv",metrics);_csv(root/"scale_diagnostics.csv",diagnostics)
      bootstrap={"schema":"gnss-doppler-lab.nc-topi-stage0b.bootstrap.v1","repetitions":2000,"seed":20260803,"comparisons":boot};_json(root/"bootstrap_results.json",bootstrap);_json(root/"paired_comparisons.json",bootstrap);_json(root/"decision.json",decision)
      _csv(root/"diagnostics/clamp_variant_metrics.csv",clamp_variant_metrics);_csv(root/"diagnostics/time_shuffle_metrics.csv",shuffle_metrics);_json(root/"diagnostics/time_shuffle_fit_audit.json",{"seed":0,"target_only_permutation":True,"same_permutation_all_targets":True,"permutation_digest_sha256":fit_audit["time_shuffle_permutation_digest_sha256"],"models":{t:dict(m.audit)|{"seal":m.seal,"content_digest_sha256":m.seal} for t,m in shuffled.items()}})
      _json(root/"diagnostics/iq_scale_checks.json",{"event_common_scale_all":True,"events":len(events),"formula":"add-one two-sided <=/>="});_json(root/"diagnostics/second_peak_limitations.json",{"stage0_c7":False,"preserved_status":"failure","limitations":cfg["second_peak"]["limitations_exact"],"complex_synthesis_performed":False})
      _csv(root/"diagnostics/time_shuffle_scale_diagnostics.csv",shuffle_diag)
      plots=_plots(root,events,metrics);_json(root/"plot_provenance.json",{"plots":plots,"source":"event_scores.csv","deterministic":True})
      readme=f"""# NC-TOPI Stage-0B shortcut and calibration audit\n\nGenerated deterministically from the immutable Stage-0 artifact at `{core.PARENT_ARTIFACT_COMMIT}`.\nNo raw-IQ file was opened and no B0 model was retrained.\n\n- Stage-0 decision (preserved): **{parent_decision.get('status')}**\n- Stage-0B status: **{decision['status']}**\n- Original reconstruction rows: {reconstruction['rows']}\n- Reconstruction max absolute error: {reconstruction['max_absolute_error']:.17g}\n- Reconstruction max relative error: {reconstruction['max_relative_error']:.17g}\n- Profile D: **{profile['status']}**, best effective-support split 33/33/33\n- Bootstrap: 12 paired comparisons, 2,000 requested replicates each, no IID fallback\n\nThis README reports the frozen grammar without post-result interpretation or tuning.\n"""
      (root/"README.md").write_text(readme);_json(root/"verification.json",{"ok":True,"status":"pending independent publication gate","checks":"recomputed by scripts/summarize_nc_topi_stage0b_audit.py"});core.write_hash_manifest(root)
      spec=importlib.util.spec_from_file_location("stage0b_verify",ROOT/"scripts/summarize_nc_topi_stage0b_audit.py");v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
      stage.publish(lambda p:v.verify_artifact(p,parent=parent,repo=ROOT))
    return out

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,default=ROOT/"configs/nc_topi_stage0b_audit.json");p.add_argument("--parent",type=Path,default=ROOT/"artifacts/nc_topi_stage0");p.add_argument("--out",type=Path,default=ROOT/"artifacts/nc_topi_stage0b_audit");p.add_argument("--verify-after-run",action="store_true");p.add_argument("--reconstruction-only",action="store_true");return p.parse_args(argv)
def main(argv=None):
    args=parse_args(argv)
    if args.reconstruction_only:print(json.dumps(core.run_original_reconstruction_gate(args.parent,repo=ROOT),sort_keys=True,indent=2));return 0
    result=run(args.config,args.out,args.parent)
    if args.verify_after_run:
      spec=importlib.util.spec_from_file_location("stage0b_verify",ROOT/"scripts/summarize_nc_topi_stage0b_audit.py");v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);report=v.verify_artifact(result,parent=args.parent,repo=ROOT)
      if not report["ok"]:raise SystemExit(json.dumps(report,sort_keys=True))
    print(json.dumps({"ok":True,"output":str(result)},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
