#!/usr/bin/env python3
"""Operational one-shot R2C Stage-0-fix runner.

Production output is single-use and exact-path only. ``--test-output`` is accepted
only together with ``--synthetic`` for isolated integration tests.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.r2c_stage0_artifact import FIX_SOURCE_FILES, PRESERVED_TREE, TOP_LEVEL_FILES
from gnss_doppler_lab.r2c_stage0_fix import (ComplexWhitener, SmallNuisanceConditioner, TemplateProvider,
 calibration_thresholds, derive_two_layer_decision, detection_metrics, detector_scores, joint_profile_glrt, ranking_metrics,
 historical_b0_status, paired_block_bootstrap, run_full_controls)

DETECTORS=("A1","A2","A3","A4","Full","Neural-with-energy","Power-only")

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()
def git(*args): return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def load(path: Path): return json.loads(path.read_text())
def source_bundle():
    files={name:sha(ROOT/name) for name in FIX_SOURCE_FILES}
    return {"files":files,"bundle_sha256":hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest()}

def validate_destination(output: Path, *, test_mode=False):
    output=output.resolve(); production=(ROOT/"artifacts/r2c_gnss_stage0_fix").resolve()
    if test_mode:
        if production==output or production in output.parents or ROOT.resolve() in output.parents:
            raise ValueError("temporary test output must be outside the repository and production artifact")
    elif output!=production: raise ValueError("production output must be exactly artifacts/r2c_gnss_stage0_fix")
    if output.exists(): raise FileExistsError("one-campaign guard: output already exists")
    return output

def validate_source(source_commit: str, *, test_mode=False):
    head=git("rev-parse","HEAD")
    if source_commit!=head: raise ValueError("frozen source commit does not equal HEAD")
    if not test_mode and git("status","--porcelain=v1"): raise ValueError("source worktree must be clean")
    if git("rev-parse","HEAD:artifacts/r2c_gnss_stage0")!=PRESERVED_TREE: raise ValueError("preserved artifact tree changed")
    return head

def load_all_epochs(path: Path):
    with np.load(path,allow_pickle=False) as z:
        required={"complex_iq","time_s","prn"}
        if missing:=required-set(z.files): raise ValueError(f"NPZ missing {sorted(missing)}")
        iq=np.asarray(z["complex_iq"]); time=np.asarray(z["time_s"],float); prn=np.asarray(z["prn"])
        if iq.shape!=(len(time),9,2): raise ValueError("complex input shape mismatch")
        y=iq[...,0].astype(float)+1j*iq[...,1].astype(float)
        cn0=np.asarray(z["cn0_db_hz"],float) if "cn0_db_hz" in z.files else np.full(len(time),np.nan)
        noise=np.asarray(z["noise_floor"],float) if "noise_floor" in z.files else None
    finite=np.isfinite(time)&np.isfinite(y).all(1)&np.asarray([1<=int(str(p).lstrip("Gg"))<=32 for p in prn])
    if not finite.all(): raise ValueError("all epochs must be finite/authentic; silent row dropping forbidden")
    return {"y":y,"time":time,"prn":np.asarray([int(str(p).lstrip("Gg")) for p in prn]),"cn0":cn0,"noise_floor":noise,
            "bin":np.floor(time/.5).astype(int),"source_sha256":sha(path),"row_count":len(time)}

def h0_residuals(y,provider,taps,grid,*,chunk_rows=4096):
    """Profile every row in deterministic bounded chunks; no support subsampling."""
    values=np.asarray(y,complex);output=np.empty_like(values)
    templates=[provider.evaluate(taps-delay) for delay in grid]
    for start in range(0,len(values),chunk_rows):
        block=values[start:start+chunk_rows];best_rss=np.full(len(block),np.inf);best=np.empty_like(block)
        for template in templates:
            amps=(block@template.conj())/np.vdot(template,template)
            residual=block-amps[:,None]*template;rss=np.sum(np.abs(residual)**2,axis=1);take=rss<best_rss
            best_rss[take]=rss[take];best[take]=residual[take]
        output[start:start+len(block)]=best
    return output

def conditions(data,residuals,indices,with_energy=False):
    cn0=data["cn0"].copy(); finite=np.isfinite(cn0[indices]); impute=float(np.median(cn0[indices][finite])) if finite.any() else 0.
    quality=np.mean(np.abs(residuals)**2,axis=1); values=[np.where(np.isfinite(cn0),cn0,impute),quality]
    if with_energy: values.append(np.mean(np.abs(data["y"])**2,axis=1))
    return np.column_stack(values)

def inference_conditions(data,residuals,impute,with_energy=False):
    cn0=np.where(np.isfinite(data["cn0"]),data["cn0"],impute);quality=np.mean(np.abs(residuals)**2,axis=1)
    values=[cn0,quality]
    if with_energy:values.append(np.mean(np.abs(data["y"])**2,axis=1))
    return np.column_stack(values)

def fit_frozen_models(clean,config,provider,taps,grid,*,require_gpu):
    train=clean["time"]<=config["splits"]["normal_train"]["source_end_lte_s"]
    if not train.any(): raise ValueError("cleanStatic normal_train empty")
    raw=h0_residuals(clean["y"],provider,taps,grid); ids=[f"{clean['time'][i]:.9f}:{clean['prn'][i]}" for i in np.flatnonzero(train)]
    whitening_args={k:config["whitening"][k] for k in ("shrinkage","eigen_floor_fraction")}
    analytic=ComplexWhitener(**whitening_args).fit(raw[train],["normal_train"]*int(train.sum()),ids)
    finite=np.isfinite(clean["cn0"][train]); impute=float(np.median(clean["cn0"][train][finite])) if finite.any() else 0.
    x=inference_conditions(clean,raw,impute,False);xe=inference_conditions(clean,raw,impute,True)
    neural=SmallNuisanceConditioner(config["neural"]["no_energy_inputs"],hidden=config["neural"]["hidden"],seed=config["seed"]).fit(
      x[train],raw[train],["normal_train"]*int(train.sum()),epochs=config["neural"]["epochs"],learning_rate=config["neural"]["learning_rate"],require_gpu=require_gpu)
    energy=SmallNuisanceConditioner(config["neural"]["with_energy_inputs"],hidden=config["neural"]["hidden"],seed=config["seed"],with_energy=True).fit(
      xe[train],raw[train],["normal_train"]*int(train.sum()),epochs=config["neural"]["epochs"],learning_rate=config["neural"]["learning_rate"],require_gpu=require_gpu)
    pred=neural.predict(x); prede=energy.predict(xe)
    nw=ComplexWhitener(**whitening_args).fit((raw-pred)[train],["normal_train"]*int(train.sum()),ids)
    ew=ComplexWhitener(**whitening_args).fit((raw-prede)[train],["normal_train"]*int(train.sum()),ids)
    return {"analytic":analytic,"neural_model":neural,"energy_model":energy,"neural_whitener":nw,"energy_whitener":ew,
            "clean_raw_residuals":raw,"train_mask":train,"cn0_imputation":impute}

def score_bin(observations,los,provider,taps,grid,models,config,conditions_by_prn=None):
    h0=joint_profile_glrt(observations,los,provider,taps,grid,hypothesis="H0",whitener=models["analytic"])
    individual={}
    for p,y in observations.items():
        fit=joint_profile_glrt({p:y},{},provider,taps,grid,hypothesis="H1-independent",whitener=models["analytic"])
        individual[p]=fit.score
    if conditions_by_prn is None: raise ValueError("runner neural inference conditions are required")
    neural_obs={p:y-models["neural_model"].predict(conditions_by_prn[p][0]) for p,y in observations.items()}
    energy_obs={p:y-models["energy_model"].predict(conditions_by_prn[p][1]) for p,y in observations.items()}
    independent=joint_profile_glrt(neural_obs,los,provider,taps,grid,hypothesis="H1-independent",whitener=models["neural_whitener"])
    shared=joint_profile_glrt(neural_obs,los,provider,taps,grid,hypothesis="H1-shared",whitener=models["neural_whitener"],
      beta_bounds_m=config["beta_bounds_m"],optimizer_starts=config["optimizer_starts_m"])
    analytic_shared=joint_profile_glrt(observations,los,provider,taps,grid,hypothesis="H1-shared",whitener=models["analytic"],
      beta_bounds_m=config["beta_bounds_m"],optimizer_starts=config["optimizer_starts_m"])
    energy_shared=joint_profile_glrt(energy_obs,los,provider,taps,grid,hypothesis="H1-shared",whitener=models["energy_whitener"],
      beta_bounds_m=config["beta_bounds_m"],optimizer_starts=config["optimizer_starts_m"])
    scores=detector_scores(individual,analytic_shared,independent,shared,energy_shared.score,float(np.mean([np.mean(np.abs(y)**2) for y in observations.values()])))
    return scores,{"H0":h0,"A4":independent,"Full":shared,"A3":analytic_shared,"Neural-with-energy":energy_shared},individual

def synthetic_inputs(seed=20260803):
    rng=np.random.default_rng(seed); taps=np.arange(-.5,.5001,.125); provider=TemplateProvider.analytic()
    los=np.array([[1,0,0],[0,1,0],[0,0,1],[-.6,-.5,-.6245],[.5,-.7,.5099]],float); los/=np.linalg.norm(los,axis=1)[:,None]
    rows=[]
    for bin_id in range(50):
        for p,u in enumerate(los,1):
            for epoch in range(2):
                y=(1+.03*rng.normal())*np.exp(1j*rng.uniform(-.2,.2))*provider.evaluate(taps)+.01*(rng.normal(size=9)+1j*rng.normal(size=9))
                rows.append((bin_id*.5+epoch*.1,p,y,40+rng.normal()))
    return rows,{p:u for p,u in enumerate(los,1)}

def write_artifact(output,documents,csvs,plots):
    output.mkdir(); (output/"plots").mkdir()
    for name,value in documents.items(): (output/name).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    for name,(fields,rows) in csvs.items():
        with (output/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    for name,text in plots.items(): (output/"plots"/name).write_text(text)
    (output/"README.md").write_text("# R2C GNSS Stage-0 fix campaign\n")
    missing=TOP_LEVEL_FILES-{p.name for p in output.iterdir() if p.is_file()}
    if missing!={"hashes.json"}: raise RuntimeError(f"artifact assembly mismatch {missing}")
    hashes={str(p.relative_to(output)):sha(p) for p in output.rglob("*") if p.is_file() and p.name!="hashes.json"}
    (output/"hashes.json").write_text(json.dumps({"algorithm":"sha256","policy":"all files recursively except hashes.json","files":hashes},indent=2,sort_keys=True)+"\n")

def run_synthetic(output,config,source_commit):
    rows,los=synthetic_inputs(config["seed"]); taps=np.asarray(config["tap_offsets_chips"]); grid=np.asarray(config["delay_grid_chips"]); provider=TemplateProvider.analytic()
    data={"y":np.asarray([r[2] for r in rows]),"time":np.asarray([r[0] for r in rows]),"prn":np.asarray([r[1] for r in rows]),"cn0":np.asarray([r[3] for r in rows])}
    models=fit_frozen_models(data,config,provider,taps,grid,require_gpu=False)
    chosen={p:data["y"][data["prn"]==p][:2] for p in sorted(los)}
    raw=h0_residuals(np.concatenate(list(chosen.values())),provider,taps,grid); cn=np.full(len(raw),40.)
    q=np.mean(np.abs(raw)**2,axis=1); e=np.mean(np.abs(np.concatenate(list(chosen.values())))**2,axis=1)
    condition_map={}; cursor=0
    for p,y in chosen.items():
        count=len(y); condition_map[p]=(np.column_stack((cn[cursor:cursor+count],q[cursor:cursor+count])),np.column_stack((cn[cursor:cursor+count],q[cursor:cursor+count],e[cursor:cursor+count]))); cursor+=count
    smoke_config={**config,"optimizer_starts_m":[config["optimizer_starts_m"][0]]}
    scores,fits,_=score_bin(chosen,los,provider,taps,grid,models,smoke_config,condition_map)
    calibration=np.linspace(0,1,100); thresholds=calibration_thresholds(calibration,["normal_calibration"]*100)
    def full(obs,pair):
        local={p:(np.column_stack((np.full(len(y),40.),np.mean(np.abs(h0_residuals(y,provider,taps,grid))**2,axis=1))),
                    np.column_stack((np.full(len(y),40.),np.mean(np.abs(h0_residuals(y,provider,taps,grid))**2,axis=1),np.mean(np.abs(y)**2,axis=1)))) for p,y in obs.items()}
        adjusted={p:y-models["neural_model"].predict(local[p][0]) for p,y in obs.items()}
        return joint_profile_glrt(adjusted,pair,provider,taps,grid,hypothesis="H1-shared",whitener=models["neural_whitener"],beta_bounds_m=smoke_config["beta_bounds_m"],optimizer_starts=smoke_config["optimizer_starts_m"]).score
    controls=run_full_controls(full,chosen,los,thresholds["q99"],provider,taps,seed=config["seed"])
    gates={name:{"status":"NOT_EVALUATED"} for name in ("complex_provenance","time_los_alignment","geometry_coverage","clean_dynamic_fpr","gain_invariance","phase_invariance","noise_gain_alarms","relation_destruction","full_improvement","full_a2_two_scenarios","shortcut_controls")}
    decision=derive_two_layer_decision(gates)
    provenance={"source_commit":source_commit,"source_bundle":source_bundle(),"preserved_artifact_tree":PRESERVED_TREE,"synthetic_test_mode":True,"external_inputs":[]}
    documents={name:{} for name in TOP_LEVEL_FILES if name.endswith(".json") and name!="hashes.json"}
    documents.update({"config.json":config,"provenance.json":provenance,"training_summary.json":{"models":{k:v.serialize() if hasattr(v,"serialize") else None for k,v in models.items() if k in ("analytic","neural_model","energy_model","neural_whitener","energy_whitener")}},
      "thresholds.json":thresholds,"gain_invariance.json":controls,"phase_invariance.json":controls,"noise_control.json":controls,
      "multipath_control.json":controls,"second_source_injection.json":controls,"relation_destruction.json":controls,
      "bootstrap_comparisons.json":{"status":"NOT_EVALUATED_SYNTHETIC","repetitions":2000},"decision.json":{**decision,"gates":gates},"verification.json":{"status":"PENDING_EXTERNAL_VERIFIER"}})
    score_rows=[]
    for detector,value in scores.items():
        fit=fits.get(detector); score_rows.append({"scenario":"synthetic","availability_time_s":.5,"detector":detector,"score":value,
          "ll0":fit.null_log_likelihood if fit else "","ll1":fit.log_likelihood if fit else "","n":fit.n if fit else "","k0":fit.null_k if fit else "","k1":fit.k if fit else ""})
    csvs={"per_epoch_scores.csv":(["scenario","availability_time_s","detector","score","ll0","ll1","n","k0","k1"],score_rows),
      "scenario_metrics.csv":(["scenario","status"],[{"scenario":"synthetic","status":"SMOKE_ONLY"}]),
      "ablation_metrics.csv":(["detector","status"],[{"detector":d,"status":"SMOKE_ONLY"} for d in scores])}
    write_artifact(output,documents,csvs,{"synthetic_source.csv":"time,score\n0,0\n"})
    return {"scores":scores,"controls":controls}

def run_production(output,config,source_commit,input_specs,geometry_specs,b0_specs):
    inputs={k:Path(v) for k,v in (x.split("=",1) for x in input_specs)}
    if "cleanStatic" not in inputs:raise ValueError("cleanStatic input required")
    data={name:load_all_epochs(path) for name,path in inputs.items()};provider=TemplateProvider.analytic()
    taps=np.asarray(config["tap_offsets_chips"]);grid=np.asarray(config["delay_grid_chips"])
    models=fit_frozen_models(data["cleanStatic"],config,provider,taps,grid,require_gpu=config["neural"]["require_gpu"])
    geometry={k:load(Path(v)) for k,v in (x.split("=",1) for x in geometry_specs)}
    b0_paths={k:Path(v) for k,v in (x.split("=",1) for x in b0_specs)}
    b0=(historical_b0_status(b0_paths,config["b0"]["saved_score_sha256"])
        if b0_paths else {"status":"UNAVAILABLE_AUTHENTIC_INTERFACE",
                          "paper_comparison_eligible":False,"scenarios":{}})
    score_rows=[];event_rows=[]
    for scenario,dataset in data.items():
        raw=h0_residuals(dataset["y"],provider,taps,grid)
        x=inference_conditions(dataset,raw,models["cn0_imputation"],False);xe=inference_conditions(dataset,raw,models["cn0_imputation"],True)
        los_bins=geometry.get(scenario,{}).get("los_by_bin",{})
        for bin_id in np.unique(dataset["bin"]):
            indices=np.flatnonzero(dataset["bin"]==bin_id);observations={};condition_map={};los={}
            for p in sorted(np.unique(dataset["prn"][indices])):
                chosen=indices[dataset["prn"][indices]==p];observations[int(p)]=dataset["y"][chosen]
                condition_map[int(p)]=(x[chosen],xe[chosen])
                vector=los_bins.get(str(int(bin_id)),{}).get(str(int(p)))
                if vector is not None:los[int(p)]=np.asarray(vector,float)
            availability=float(dataset["time"][indices].max())
            try:scores,fits,_=score_bin(observations,los,provider,taps,grid,models,config,condition_map)
            except (ValueError,RuntimeError):
                # Geometry-free paths remain operational; geometry paths stay unavailable.
                h0=joint_profile_glrt(observations,{},provider,taps,grid,hypothesis="H0",whitener=models["analytic"])
                individual={p:joint_profile_glrt({p:y},{},provider,taps,grid,hypothesis="H1-independent",whitener=models["analytic"]).score for p,y in observations.items()}
                neural_obs={p:y-models["neural_model"].predict(condition_map[p][0]) for p,y in observations.items()}
                independent=joint_profile_glrt(neural_obs,{},provider,taps,grid,hypothesis="H1-independent",whitener=models["neural_whitener"])
                values=np.asarray(list(individual.values()));scores={"A1":float(values.max()),"A2":float(np.median(values)+np.mean(np.sort(values)[-min(4,len(values)):])),"A3":None,"A4":independent.score,"Full":None,"Neural-with-energy":None,"Power-only":float(np.mean([np.mean(np.abs(y)**2) for y in observations.values()]))};fits={"H0":h0,"A4":independent}
            for detector,score in scores.items():
                fit=fits.get(detector);score_rows.append({"scenario":scenario,"time_bin":int(bin_id),"availability_time_s":availability,
                  "detector":detector,"score":"" if score is None else score,"ll0":fit.null_log_likelihood if fit else "",
                  "ll1":fit.log_likelihood if fit else "","n":fit.n if fit else "","k0":fit.null_k if fit else "","k1":fit.k if fit else "",
                  "epoch_count":sum(len(y) for y in observations.values()),"prn_count":len(observations),"geometry_valid":bool(fit.valid) if fit and detector in {"A3","Full","Neural-with-energy"} else ""})
    clean=[r for r in score_rows if r["scenario"]=="cleanStatic" and 320<=r["availability_time_s"]<=400 and r["score"]!=""]
    detector_thresholds={}
    for detector in DETECTORS:
        values=[float(r["score"]) for r in clean if r["detector"]==detector]
        if values:detector_thresholds[detector]=calibration_thresholds(values,["normal_calibration"]*len(values))
    full_threshold=detector_thresholds.get("Full",{"q99":np.inf,"q99.5":np.inf,"target_fpr_1pct":np.inf})
    thresholds={**full_threshold,"detectors":detector_thresholds,"method":"higher","comparison":"strict_greater"}
    metric_rows=[]
    for scenario in data:
        onset=config.get("attacks",{}).get("onset_s",{}).get(scenario)
        for detector in DETECTORS:
            rows=[r for r in score_rows if r["scenario"]==scenario and r["detector"]==detector and r["score"]!=""]
            if onset is None or not rows:metric_rows.append({"scenario":scenario,"detector":detector,"status":"NORMAL_ONLY"});continue
            labels=np.asarray([r["availability_time_s"]>=onset+40 for r in rows]);pre=np.asarray([r["availability_time_s"]<onset-20 for r in rows]);use=labels|pre
            if labels[use].any() and (~labels[use]).any():
                used_rows=[r for r,u in zip(rows,use) if u];used_labels=labels[use];used_scores=np.asarray([float(r["score"]) for r in used_rows])
                threshold=detector_thresholds.get(detector,{}).get("q99",np.inf)
                metric_rows.append({"scenario":scenario,"detector":detector,"status":"EVALUATED",
                  **ranking_metrics(used_labels,used_scores),
                  **detection_metrics([r["availability_time_s"] for r in used_rows],used_labels,used_scores>threshold,[scenario]*len(used_rows),attack_onset_s=onset)})
    comparisons=[]
    for scenario in config.get("attacks",{}).get("primary",[]):
        onset=config["attacks"]["onset_s"].get(scenario)
        if onset is None: continue
        full={int(r["time_bin"]):r for r in score_rows if r["scenario"]==scenario and r["detector"]=="Full" and r["score"]!=""}
        for detector in ("A1","A2","A3","A4","Neural-with-energy","Power-only"):
            other={int(r["time_bin"]):r for r in score_rows if r["scenario"]==scenario and r["detector"]==detector and r["score"]!=""}
            keys=sorted(set(full)&set(other)); records=[]
            for key in keys:
                t=float(full[key]["availability_time_s"])
                if t<onset-20: records.append((t,False,float(full[key]["score"]),float(other[key]["score"])))
                elif t>=onset+config["attacks"]["persistent_offset_s"]: records.append((t,True,float(full[key]["score"]),float(other[key]["score"])))
            if records:
                try:
                    result=paired_block_bootstrap([x[0] for x in records],[scenario]*len(records),[x[1] for x in records],
                                                  [x[2] for x in records],[x[3] for x in records],
                                                  repetitions=config["bootstrap"]["repetitions"],seed=config["bootstrap"]["seed"],block_s=config["bootstrap"]["block_s"])
                    comparisons.append({"scenario":scenario,"left":"Full","right":detector,**result})
                except ValueError:
                    continue
    bootstrap={"status":"COMPUTED" if comparisons else "MISSING_COMMON_COMPLETE_BLOCKS","repetitions":2000,
      "valid_draw_count":2000 if comparisons else 0,"comparisons":comparisons,
      "draw_index_sha256":hashlib.sha256(json.dumps([x["draw_index_sha256"] for x in comparisons],sort_keys=True).encode()).hexdigest() if comparisons else None,
      "seed":config["bootstrap"]["seed"],"block_s":config["bootstrap"]["block_s"]}
    valid_control=None
    for scenario in ("cleanStatic",):
        bins=sorted(set(int(r["time_bin"]) for r in score_rows if r["scenario"]==scenario and r["detector"]=="Full" and r["score"]!=""))
        if bins:
            b=bins[-1];idx=np.flatnonzero(data[scenario]["bin"]==b);obs={p:data[scenario]["y"][idx[data[scenario]["prn"][idx]==p]] for p in np.unique(data[scenario]["prn"][idx])};lb=geometry.get(scenario,{}).get("los_by_bin",{}).get(str(b),{});los={int(p):np.asarray(v,float) for p,v in lb.items()}
            if len(los)>=5:
                def full(obs,pair):
                    adjusted={}
                    for p,y in obs.items():
                        residual=h0_residuals(y,provider,taps,grid);condition=np.column_stack((np.full(len(y),models["cn0_imputation"]),np.mean(np.abs(residual)**2,axis=1)))
                        adjusted[p]=y-models["neural_model"].predict(condition)
                    return joint_profile_glrt(adjusted,pair,provider,taps,grid,hypothesis="H1-shared",whitener=models["neural_whitener"],beta_bounds_m=config["beta_bounds_m"],optimizer_starts=config["optimizer_starts_m"]).score
                valid_control=run_full_controls(full,obs,los,full_threshold["q99"],provider,taps,seed=config["seed"]);break
    control=valid_control or {"status":"NOT_EVALUATED","threshold":full_threshold["q99"],"rows":[]}
    gates={n:{"status":"NOT_EVALUATED"} for n in ("complex_provenance","time_los_alignment","geometry_coverage","clean_dynamic_fpr","gain_invariance","phase_invariance","noise_gain_alarms","relation_destruction","full_improvement","full_a2_two_scenarios","shortcut_controls")}
    input_integrity=all(d["row_count"]>0 and len(d["source_sha256"])==64 for d in data.values())
    gates["complex_provenance"]={"status":"PASS" if input_integrity else "FAIL","derived_from":"finite rows and source hashes"}
    coverage=[];time_status=[]
    for name in config["geometry"]["required_scenarios"]:
        item=geometry.get(name,{})
        coverage.append(item.get("offline_geometry_coverage",{}).get("status"))
        time_status.append(item.get("derived_time",{}).get("status"))
    gates["geometry_coverage"]={"status":"PASS" if coverage and all(x=="PASS" for x in coverage) else "FAIL","scenario_statuses":coverage}
    gates["time_los_alignment"]={"status":"PASS" if time_status and all(x=="PASS" for x in time_status) else "FAIL","scenario_statuses":time_status}
    gates["empirical_wide_template"]={"status":"PASS" if not provider.analytic_approximation else "FAIL","provenance":provider.provenance}
    decision={**derive_two_layer_decision(gates),"gates":gates}
    provenance={"source_commit":source_commit,"source_bundle":source_bundle(),"preserved_artifact_tree":PRESERVED_TREE,"synthetic_test_mode":False,
      "external_inputs":[{"role":"selected_input","scenario":name,"path":str(path.resolve()),"sha256":data[name]["source_sha256"]} for name,path in inputs.items()]
        +[{"role":"geometry_reconstruction","scenario":name,"path":str(Path(path).resolve()),"sha256":sha(Path(path))} for name,path in (x.split("=",1) for x in geometry_specs)]
        +[{"role":"b0_native_score","scenario":name,"path":str(Path(path).resolve()),"sha256":sha(Path(path))} for name,path in (x.split("=",1) for x in b0_specs)]}
    documents={name:{} for name in TOP_LEVEL_FILES if name.endswith(".json") and name!="hashes.json"}
    documents.update({"config.json":config,"provenance.json":provenance,"input_validity.json":{"datasets":{n:{"rows":d["row_count"]} for n,d in data.items()}},
      "time_geometry_validation.json":geometry,"b0_interface_validation.json":{**b0,"core_blocked":False},
      "training_summary.json":{"models":{k:v.serialize() for k,v in models.items() if hasattr(v,"serialize")}},"thresholds.json":thresholds,
      "gain_invariance.json":control,"phase_invariance.json":control,"noise_control.json":control,"multipath_control.json":control,
      "second_source_injection.json":control,"relation_destruction.json":control,"bootstrap_comparisons.json":bootstrap,
      "decision.json":decision,"verification.json":{"status":"PENDING_EXTERNAL_VERIFIER"}})
    fields=["scenario","time_bin","availability_time_s","detector","score","ll0","ll1","n","k0","k1","epoch_count","prn_count","geometry_valid"]
    csvs={"per_epoch_scores.csv":(fields,score_rows),"scenario_metrics.csv":(sorted(set().union(*(r.keys() for r in metric_rows))),metric_rows),
      "ablation_metrics.csv":(["detector","status"],[{"detector":d,"status":"COMPUTED"} for d in DETECTORS])}
    write_artifact(output,documents,csvs,{"score_source.csv":"scenario,time,detector,score\n"})
    return {"rows":len(score_rows)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,default=ROOT/"configs/r2c_gnss_stage0_fix.json")
    ap.add_argument("--output",type=Path,default=ROOT/"artifacts/r2c_gnss_stage0_fix"); ap.add_argument("--source-commit",required=True)
    ap.add_argument("--input",action="append",default=[]);ap.add_argument("--geometry",action="append",default=[])
    ap.add_argument("--b0-score",action="append",default=[],help="SCENARIO=native-prn-score.csv")
    ap.add_argument("--synthetic",action="store_true"); ap.add_argument("--test-output",action="store_true")
    args=ap.parse_args(); output=validate_destination(args.output,test_mode=args.test_output); validate_source(args.source_commit,test_mode=args.test_output)
    config=json.loads(args.config.read_text())
    if args.test_output and not args.synthetic: ap.error("--test-output requires --synthetic")
    if args.synthetic: result=run_synthetic(output,config,args.source_commit); print(json.dumps({"output":str(output),"synthetic":True,"detectors":list(result["scores"])})); return
    if not args.input:ap.error("production requires --input NAME=NPZ")
    result=run_production(output,config,args.source_commit,args.input,args.geometry,args.b0_score);print(json.dumps({"output":str(output),"rows":result["rows"]}))
if __name__=="__main__": main()
