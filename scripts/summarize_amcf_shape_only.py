#!/usr/bin/env python3
"""Independently verify and recompute a finalized AMCF Shape-Only artifact."""
from __future__ import annotations
import argparse,csv,importlib.util,json,math,subprocess
from pathlib import Path
from typing import Any,Mapping
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location("amcf_shape_runner_contract",ROOT/"scripts/run_amcf_shape_only.py")
assert _spec and _spec.loader
runner=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(runner)


def _rows(path:Path)->list[dict[str,str]]:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():return []
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))

def _bool(v:Any)->bool:return v is True or str(v).strip().lower() in {"1","true","yes"}
def _equal(a:Any,b:Any,tol:float=1e-12)->bool:
    if a in (None,"") or b in (None,""):return a in (None,"") and b in (None,"")
    try:return math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol)
    except (ValueError,TypeError):return str(a)==str(b)

def _metric_match(saved:Mapping[str,Any],actual:Mapping[str,Any])->None:
    for key in ("clean_test_fpr","stable_pre_fpr","post_detection","persistent_detection","roc_auc","pr_auc","sustained3_delay_s","threshold","rows"):
        if not _equal(saved.get(key),actual.get(key)):
            raise ValueError(f"metric recomputation mismatch {saved.get('scenario')} {saved.get('model')} {saved.get('operating_point')} {key}")

def _verify_schema(out:Path)->dict[str,Any]:
    schema=json.loads((out/"feature_schema.json").read_text())
    reps=schema.get("representations",{})
    if reps.get("complex",{}).get("dimensions")!=list(runner.COMPLEX_SCHEMA) or reps.get("magnitude",{}).get("dimensions")!=list(runner.MAGNITUDE_SCHEMA):raise ValueError("feature schema mismatch")
    if schema.get("tap_order")!=list(runner.TAP_NAMES) or schema.get("prompt_index")!=4 or schema.get("tensor_side_count")!=8:raise ValueError("tap order/schema metadata mismatch")
    forbidden=("cn0","c_n0","context","valid_fraction","prompt_magnitude","log_prompt")
    if any(x in json.dumps(schema).lower() for x in forbidden):raise ValueError("forbidden field in feature schema")
    return schema

def _verify_decision_digests(out:Path,decision:Mapping[str,Any],mode:str)->None:
    got=decision.get("source_digests",{})
    if mode=="synthetic-smoke":
        paths={"metrics":"scenario_metrics.csv","paired":"paired_comparisons.csv","convergence":"convergence_audit.json","schema":"feature_schema.json","feature_audit":"feature_audit.json"}
        if set(got)!=set(paths):raise ValueError("GO decision lacks exact source digests")
        for k,v in paths.items():
            if got[k]!=runner.sha256(out/v):raise ValueError(f"GO source digest mismatch: {k}")
        return
    paths={"thresholds":"thresholds.json","seed_metrics":"seed_metrics.csv","paired_comparisons":"paired_comparisons.csv","scenario_metrics":"scenario_metrics.csv","provenance":"provenance.json","feature_audit":"feature_audit.json","feature_schema":"feature_schema.json","fit_checkpoint_audits":"convergence_audit.json","training_history":"training_history.csv","calibration_evidence":"calibration_evidence.json"}
    if set(got)!=(set(paths)|{"per_epoch","checkpoints"}):raise ValueError("primary GO decision lacks exact full source digests")
    for k,v in paths.items():
        if got[k]!=runner.sha256(out/v):raise ValueError(f"GO source digest mismatch: {k}")
    for key,folder,pattern in (("per_epoch","per_epoch","*.csv"),("checkpoints","models","*.pt")):
        actual={str(p.relative_to(out)):runner.sha256(p) for p in sorted((out/folder).glob(pattern))}
        if got[key]!=actual:raise ValueError(f"GO source digest mismatch: {key}")

def _verify_primary_config(config:Mapping[str,Any],provenance:Mapping[str,Any])->None:
    if config.get("base_sha")!=runner.PRIMARY_BASE_SHA or config.get("minimum_valid_rows")!=runner.PRIMARY_MIN_VALID_ROWS:raise ValueError("primary hard-freeze config mismatch")
    if config.get("seeds")!=list(runner.SEEDS) or int(config.get("bootstrap_reps",0))<2000:raise ValueError("primary seeds/bootstrap freeze mismatch")
    for k,v in runner.PRIMARY_FIT_CONFIG.items():
        if config.get(k)!=v:raise ValueError(f"primary fit config mismatch: {k}")
    if config.get("DS4")!={"status":"NA","included_in_attack_go":False} or "DS4:NA" not in config.get("scenarios",[]):raise ValueError("DS4 NA contract missing")
    commit=str(provenance.get("source_commit",""))
    if len(commit)!=40 or provenance.get("clean_tree") is not True or provenance.get("execution_tree_clean") is not True:raise ValueError("primary execution source clean-tree provenance missing")
    if provenance.get("baseline")!=runner.PRIMARY_BASE_SHA:raise ValueError("primary execution baseline mismatch")
    if provenance.get("fit_config")!=dict(runner.PRIMARY_FIT_CONFIG):raise ValueError("primary fit provenance mismatch")
    if "[340,410)" not in provenance.get("threshold_protocol_limitation","") or "[300,330)" not in provenance.get("threshold_protocol_limitation",""):raise ValueError("B0/AMCF protocol limitation missing")

def _verify_calibration(out:Path,thresholds:Mapping[str,Any],provenance:Mapping[str,Any])->None:
    evidence=json.loads((out/"calibration_evidence.json").read_text())
    if set(evidence)!={"complex_all9","magnitude_all9","complex_EPL","magnitude_EPL"}:raise ValueError("calibration variant inventory mismatch")
    for variant,row in evidence.items():
        checked=runner.recompute_calibration_evidence(row); th=thresholds.get(variant,{})
        if checked["q99"]!=float(th.get("q99",math.nan)) or checked["q995"]!=float(th.get("q995",math.nan)) or checked["threshold_digest"]!=th.get("threshold_digest"):raise ValueError("calibration threshold binding mismatch")
        if row.get("source_commit")!=provenance.get("source_commit"):raise ValueError("calibration source commit mismatch")
    b0=thresholds.get("B0 Exact",{}); contract=provenance.get("B0_frozen_contract",{})
    if float(b0.get("q99",math.nan))!=runner.B0_FROZEN_Q99 or float(b0.get("q995",math.nan))!=runner.B0_FROZEN_Q995:raise ValueError("B0 frozen threshold mismatch")
    if b0.get("source_role")!="cleanStatic [300,330)" or contract.get("fit_interval")!=[300.0,330.0] or b0.get("contract_digest")!=contract.get("contract_digest"):raise ValueError("B0 protected provenance mismatch")

def _alarm_specs(mode:str,thresholds:Mapping[str,Any]):
    if mode=="synthetic-smoke":return {"Complex all9":("complex","score_complex_ensemble","alarm_complex_q99","alarm_complex_q995",thresholds["Complex all9"]),"Magnitude all9":("magnitude","score_magnitude_ensemble","alarm_magnitude_q99","alarm_magnitude_q995",thresholds["Magnitude all9"]),"B0 Exact":("B0","score_B0_Exact","alarm_B0_q99","alarm_B0_q995",thresholds["B0 Exact"])}
    out={}
    for variant,label in (("complex_all9","Complex all9"),("magnitude_all9","Magnitude all9"),("complex_EPL","Complex EPL"),("magnitude_EPL","Magnitude EPL")):
        out[label]=(variant,f"score_{variant}_ensemble",f"alarm_{variant}_q99",f"alarm_{variant}_q995",thresholds[variant])
    out["B0 Exact"]=("B0_Exact","score_B0_Exact","alarm_B0_q99","alarm_B0_q995",thresholds["B0 Exact"]);return out

def _verify_metrics(out:Path,mode:str,thresholds:Mapping[str,Any],metrics:list[dict[str,str]])->int:
    index=runner.index_metric_rows([r for r in metrics if r.get("scenario")!="DS4"])
    specs=_alarm_specs(mode,thresholds); checked=0
    for path in sorted((out/"per_epoch").glob("*.csv")):
        if path.name=="cleanStatic_calibration.csv":continue
        scenario=path.stem; rows=_rows(path)
        for label,(variant,score,a99,a995,th) in specs.items():
            if not rows or score not in rows[0]:continue
            for r in rows:
                value=float(r[score])
                if _bool(r[a99])!=(value>float(th["q99"])) or _bool(r[a995])!=(value>float(th["q995"])):raise ValueError(f"saved alarm recomputation mismatch: {scenario} {label}")
            generic=[{"decision_time_s":r["decision_time_s"],"source_start":r["source_start"],"source_end":r["source_end"],"score_ensemble":r[score],"alarm_q99":r[a99],"alarm_q995":r[a995]} for r in rows]
            for op in ("q99","q995"):
                if mode=="synthetic-smoke" and (scenario,label,op) not in index:continue
                q=float(th[op]); adjusted=[dict(r,alarm_q99=float(r["score_ensemble"])>q,alarm_q995=float(r["score_ensemble"])>q) for r in generic]
                actual=runner.recompute_scenario_metrics(scenario,adjusted,q,q,onset_s=runner.ONSETS.get(scenario))
                _metric_match(index[(scenario,label,op)],actual);checked+=1
    expected=12 if mode=="synthetic-smoke" else 6*5*2
    if checked!=expected:raise ValueError(f"all model/scenario/operating-point metrics must independently recompute: {checked}/{expected}")
    ds4=[r for r in metrics if r.get("scenario")=="DS4"]
    if len(ds4)!=1 or ds4[0].get("status")!="NA" or _bool(ds4[0].get("included_in_attack_go")):raise ValueError("DS4 exact NA row missing")
    return checked

def _verify_conformal_and_phase_rows(out:Path,thresholds:Mapping[str,Any])->None:
    calibration=json.loads((out/"calibration_evidence.json").read_text())
    variants=("complex_all9","magnitude_all9","complex_EPL","magnitude_EPL")
    for scenario in runner.CANONICAL:
        rows=_rows(out/"per_epoch"/f"{scenario}.csv")
        times=[float(r["decision_time_s"]) for r in rows]
        if times!=sorted(times) or len(times)!=len(set(times)):raise ValueError("per_epoch timestamp identity/order mismatch")
        for row in rows:
            start=float(row["source_start"]);end=float(row["source_end"])
            if end!=float(row["decision_time_s"]) or not start<end:raise ValueError("per_epoch actual source interval mismatch")
            if scenario=="cleanStatic":expected="clean_test"
            else:
                onset=runner.ONSETS[scenario];expected="persistent" if start>=onset+40 else "post" if start>=onset else "stable_pre" if start>=30 and end<=onset-20 else "transition"
            if row.get("phase")!=expected:raise ValueError("per_epoch stored phase differs from actual source interval")
        for variant in variants:
            member=[]
            for seed in runner.SEEDS:
                cal=np.asarray(calibration[variant]["per_seed_raw_scores"][str(seed)],float);raw=np.asarray([float(r[f"score_{variant}_seed{seed}"]) for r in rows]);p,e=runner._conformal(cal,raw)
                saved_p=np.asarray([float(r[f"p_{variant}_seed{seed}"]) for r in rows]);saved_e=np.asarray([float(r[f"e_{variant}_seed{seed}"]) for r in rows])
                if not np.array_equal(p,saved_p) or not np.array_equal(e,saved_e):raise ValueError("per_epoch conformal p/e recomputation mismatch")
                member.append(e)
            ensemble=np.mean(member,axis=0);saved=np.asarray([float(r[f"score_{variant}_ensemble"]) for r in rows])
            if not np.array_equal(ensemble,saved):raise ValueError("per_epoch conformal ensemble recomputation mismatch")
            for op in ("q99","q995"):
                alarms=ensemble>float(thresholds[variant][op]);stored=np.asarray([_bool(r[f"alarm_{variant}_{op}"]) for r in rows])
                if not np.array_equal(alarms,stored):raise ValueError("per_epoch full-precision threshold/alarm mismatch")


def _verify_seed_metrics(out:Path,seeds:list[dict[str,str]])->None:
    raw=[r for r in seeds if r.get("seed") not in ("mean/std","")]
    expected={(scenario,label,str(seed),op) for scenario in runner.CANONICAL for label in ("Complex all9","Magnitude all9","Complex EPL","Magnitude EPL") for seed in runner.SEEDS for op in ("q99","q995")}
    index={(r["scenario"],r["model"],r["seed"],r["operating_point"]):r for r in raw}
    if set(index)!=expected:raise ValueError("seed metric exact inventory mismatch")
    for (scenario,label,seed,op),saved in index.items():
        rows=_rows(out/"per_epoch"/f"{scenario}.csv"); variant={"Complex all9":"complex_all9","Magnitude all9":"magnitude_all9","Complex EPL":"complex_EPL","Magnitude EPL":"magnitude_EPL"}[label]
        if scenario in runner.ONSETS:
            start=np.asarray([float(r["source_start"]) for r in rows]);end=np.asarray([float(r["source_end"]) for r in rows]);m=runner.phase_labels(start,end,runner.ONSETS[scenario]);use=m["stable_pre"]|m["post"];score=np.asarray([float(r[f"e_{variant}_seed{seed}"]) for r in rows]);auc=runner._roc_auc(m["post"][use],score[use])
        else:auc=None
        if not _equal(saved.get("roc_auc"),auc):raise ValueError("per-seed AUC recomputation mismatch")

def _verify_paired(out:Path,paired:list[dict[str,str]],thresholds:Mapping[str,Any])->None:
    expected={(s,c,m) for s in runner.ONSETS for c in ("Magnitude all9","B0 Exact") for m in ("roc_auc","post_detection","stable_pre_fpr")}
    index={(r["scenario"],r["comparator"],r["metric"]):r for r in paired}
    if set(index)!=expected:raise ValueError("paired comparison exact inventory mismatch")
    for key,saved in index.items():
        scenario,comparator,metric=key;rows=_rows(out/"per_epoch"/f"{scenario}.csv");t=np.asarray([float(r["decision_time_s"]) for r in rows]);start=np.asarray([float(r["source_start"]) for r in rows]);end=np.asarray([float(r["source_end"]) for r in rows]);m=runner.phase_labels(start,end,runner.ONSETS[scenario]);y=m["post"]
        a=np.asarray([float(r["score_complex_all9_ensemble"]) for r in rows]);col="score_magnitude_all9_ensemble" if comparator=="Magnitude all9" else "score_B0_Exact";b=np.asarray([float(r[col]) for r in rows]);thkey="magnitude_all9" if comparator=="Magnitude all9" else "B0 Exact";mask={"roc_auc":m["stable_pre"]|m["post"],"post_detection":m["post"],"stable_pre_fpr":m["stable_pre"]}[metric]
        actual=runner._bootstrap_delta(t,y,a,b,metric,float(thresholds["complex_all9"]["q99"]),float(thresholds[thkey]["q99"]),mask,int(saved["reps"]),int(saved["bootstrap_seed"]))
        for field in ("estimate","ci_low","ci_high","reps","block_s","bootstrap_seed","phase_population_hash"):
            if not _equal(saved.get(field),actual.get(field)):raise ValueError(f"paired bootstrap recomputation mismatch: {key} {field}")

def _verify_execution_source(provenance:Mapping[str,Any])->None:
    commit=str(provenance["source_commit"])
    try:
        subprocess.check_call(["git","-C",str(ROOT),"cat-file","-e",f"{commit}^{{commit}}"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        changed=set(subprocess.check_output(["git","-C",str(ROOT),"diff","--name-only",f"{runner.PRIMARY_BASE_SHA}..{commit}"],text=True).splitlines())
    except subprocess.CalledProcessError as exc:raise ValueError("execution source commit is not present") from exc
    if changed-runner.SHAPE_ONLY_ALLOWED_FILES:raise ValueError("execution source commit changed protected baseline files")
    expected=provenance.get("source_hashes",{})
    if set(expected)!={p for p in runner.SOURCE_FILES if (ROOT/p).is_file()}:raise ValueError("execution source hash inventory mismatch")
    for rel,digest in expected.items():
        data=subprocess.check_output(["git","-C",str(ROOT),"show",f"{commit}:{rel}"])
        import hashlib
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError("execution source commit/file hash mismatch")


def _derive_primary_fit_bindings(out:Path,schema:Mapping[str,Any],provenance:Mapping[str,Any])->dict[str,dict[str,Any]]:
    """Rebuild canonical features/examples, preventing regenerated outer hashes from blessing edits."""
    inputs=json.loads((out/"input_hashes.json").read_text());qas=json.loads((out/"window_qa.json").read_text())
    if set(inputs)!=set(runner.CANONICAL) or set(qas)!=set(runner.CANONICAL):raise ValueError("canonical input/QA inventory mismatch")
    datasets={};bundles={}
    for name,(path,digest) in runner.CANONICAL.items():
        if inputs[name].get("sha256")!=digest or inputs[name].get("loaded_fields")!=list(runner.ALLOWED_NPZ_FIELDS):raise ValueError("canonical input audit mismatch")
        datasets[name],audit=runner.load_canonical_npz(path,digest,name,tap_order=runner.TAP_NAMES)
        if audit!=inputs[name]:raise ValueError("canonical input audit does not independently regenerate")
    gate=runner.fit_gate_from_clean_train(datasets["cleanStatic"])
    if dict(gate._asdict())!=provenance.get("gate"):raise ValueError("prompt gate independent derivation mismatch")
    for name in runner.CANONICAL:
        bundle,qa=runner.build_feature_bundle(datasets[name],name,gate,min_valid_rows=runner.PRIMARY_MIN_VALID_ROWS)
        if qa!=qas[name]:raise ValueError("feature window QA independent derivation mismatch")
        stored={}
        for rep in ("complex","magnitude"):
            path=out/"feature_cache"/f"{name}_{rep}.npz"
            with np.load(path,allow_pickle=False) as f:stored[rep]={k:np.array(f[k],copy=True) for k in f.files}
        runner.verify_feature_provenance(provenance["feature_provenance"][name],inputs[name],gate,schema,stored,qa)
        if runner._digest_value(bundle)!=runner._digest_value(stored):raise ValueError("canonical feature cache differs from independent rebuild")
        bundles[name]=stored
    scalers={}
    for rep in ("complex","magnitude"):
        part=bundles["cleanStatic"][rep];scalers[rep]=runner._fit_scaler(part["features"][part["role"]=="train"])
        saved_scaler=provenance["scalers"][rep]
        if scalers[rep]["hash"]!=saved_scaler["hash"] or not np.array_equal(scalers[rep]["median"],np.asarray(saved_scaler["median"])) or not np.array_equal(scalers[rep]["iqr"],np.asarray(saved_scaler["iqr"])):raise ValueError("scaler independent derivation mismatch")
    transformed={name:runner._transform_bundle(bundle,scalers) for name,bundle in bundles.items()};examples={}
    for name in runner.CANONICAL:
        for rep in ("complex","magnitude"):examples[name,rep]=runner.build_examples(transformed[name][rep])
    expected={}
    for rep in ("complex","magnitude"):
        ex=examples["cleanStatic",rep];train={k:v[ex["role"]=="train"] for k,v in ex.items()};validation={k:v[ex["role"]=="validation"] for k,v in ex.items()};tensor_hash=runner._digest_value(transformed["cleanStatic"][rep]["features"])
        for objective in ("all9","EPL"):
            for seed in runner.SEEDS:
                key=f"{rep}_{objective}_seed{seed}";fit=runner.make_audited_fit_input(train,validation,canonical_input_hash=inputs["cleanStatic"]["sha256"],prompt_gate_hash=runner._digest_value(dict(gate._asdict())),scaler_hash=scalers[rep]["hash"],transformed_feature_tensor_hash=tensor_hash,objective=objective,seed=seed)
                expected[key]={"fit_manifest_digest":fit.fit_manifest_digest,"upstream_digests":{"train_manifest":fit.train_manifest.manifest_digest,"validation_manifest":fit.validation_manifest.manifest_digest,**dict(fit.train_manifest.upstream_digests)}}
    if provenance.get("fit_audits")!=expected:raise ValueError("fit audit differs from independently derived examples/features")
    return expected


def _verify_convergence(out:Path,convergence:Mapping[str,Any],history:list[dict[str,str]],mode:str,derived:Mapping[str,Any]|None=None)->None:
    audits=convergence.get("audits",{});expected={f"{r}_{o}_seed{s}" for r in ("complex","magnitude") for o in ("all9","EPL") for s in runner.SEEDS}
    if set(audits)!=expected:raise ValueError("convergence audit model inventory mismatch")
    if mode=="synthetic-smoke":
        for key,row in audits.items():
            if row.get("checkpoint_sha256")!=runner.sha256(out/"models"/f"{key}.pt") or row.get("converged") is not False:raise ValueError("smoke convergence/checkpoint mismatch")
        return
    import torch
    for key,audit in audits.items():
        if derived is None or audit.get("fit_manifest_digest")!=derived[key]["fit_manifest_digest"] or audit.get("upstream_digests")!=derived[key]["upstream_digests"]:raise ValueError("checkpoint audit is not bound to independently derived fit input")
        path=out/"models"/f"{key}.pt"
        try:obj=torch.load(path,map_location="cpu",weights_only=False)
        except Exception as exc:raise ValueError("primary-full requires real torch checkpoints") from exc
        if obj.get("checkpoint_role")!="amcf-shape-only-primary-real-torch" or not obj.get("state_dict") or not obj.get("optimizer",{}).get("state"):raise ValueError("smoke/placeholder checkpoint cannot be primary-full")
        sys_path_added=False
        import sys
        if str(ROOT/"src") not in sys.path:sys.path.insert(0,str(ROOT/"src"));sys_path_added=True
        from gnss_doppler_lab.amcf_shape_only import ShapeOnlyModel
        try:
            model=ShapeOnlyModel(int(obj.get("feature_dim",-1)),hidden=32,df=4.0);model.load_state_dict(obj["state_dict"],strict=True)
        except Exception as exc:raise ValueError("primary checkpoint is not the exact audited ShapeOnlyModel architecture") from exc
        if audit.get("checkpoint_sha256")!=runner.sha256(path) or obj.get("fit_manifest_digest")!=audit.get("fit_manifest_digest") or obj.get("upstream_digests")!=audit.get("upstream_digests"):raise ValueError("checkpoint metadata/audit binding mismatch")
        if obj.get("audit",{}).get("optimizer_updates")!=audit.get("optimizer_updates") or obj.get("fit_config")!=dict(runner.PRIMARY_FIT_CONFIG):raise ValueError("checkpoint actual training audit mismatch")
        part=[r for r in history if f"{r['representation']}_{r['objective']}_seed{r['seed']}"==key]
        if len(part)!=int(audit.get("epochs_run",-1)) or not part:raise ValueError("training history/convergence actual-field mismatch")
        updates=sum(int(r["optimizer_updates"]) for r in part)
        derived=(_bool(audit.get("finite")) and _bool(audit.get("patience_early_stop")) and updates==int(audit.get("optimizer_updates",-1)) and int(audit.get("gradient_audited_updates",-2))==updates and _bool(audit.get("every_trainable_parameter_finite_gradient_each_update")))
        if derived!=_bool(audit.get("converged")):raise ValueError("convergence boolean does not follow actual checkpoint/history fields")
    if not all(_bool(x.get("converged")) for x in audits.values()) or convergence.get("exact_three_converged_per_variant") is not True:raise ValueError("primary full has nonconverged checkpoint")

def _verify_feature_audit(out:Path,audit:Mapping[str,Any],schema:Mapping[str,Any],provenance:Mapping[str,Any])->dict[str,Any]:
    if audit.get("canonical_fields")!=list(runner.ALLOWED_NPZ_FIELDS) or any(x in audit.get("canonical_fields",[]) for x in ("cn0_db_hz","context")):raise ValueError("feature audit canonical fields mismatch")
    scenarios=audit.get("scenarios",{});
    if set(scenarios)!={"cleanStatic","DS7","DS8"}:raise ValueError("clean/DS7/DS8 feature audit inventory mismatch")
    recomputed={}
    for name,row in scenarios.items():
        checks={}
        for rep,d in (("complex",4),("magnitude",2)):
            saved=row["representation_checks"][rep];path=out/saved["feature_cache"]
            with np.load(path,allow_pickle=False) as f:bundle={k:np.array(f[k],copy=True) for k in f.files}
            x=bundle["features"];iq=np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0);passed=x.shape[1:]==(8,d) and np.isfinite(x).all() and np.all(iq>1e-8)
            if saved.get("source_feature_digest")!=runner._digest_value(bundle) or saved.get("feature_tensor_digest")!=runner._digest_value(x) or saved.get("schema_hash")!=runner._digest_value(schema) or saved.get("scaler_hash")!=provenance["scalers"][rep]["hash"] or saved.get("iqr")!=iq.tolist() or saved.get("pass") is not bool(passed):raise ValueError("feature sufficient-stat/source digest audit mismatch")
            checks[rep]=saved
        rows=_rows(out/"per_epoch"/f"{name}.csv");stable=rows if name=="cleanStatic" else [r for r in rows if r["phase"]=="stable_pre"]
        threshold=json.loads((out/"thresholds.json").read_text())["complex_all9"]["q99"]
        evidence=[(float(r["decision_time_s"]),float(r["source_start"]),float(r["source_end"]),float(r["score_complex_all9_ensemble"]),threshold,_bool(r["alarm_complex_all9_q99"])) for r in stable]
        rate=float(np.mean([x[-1] for x in evidence])) if evidence else 0.
        if row.get("stable_pre_score_threshold_alarm_digest")!=runner._digest_value(evidence) or not _equal(row.get("stable_pre_alarm_rate"),rate):raise ValueError("feature audit per-epoch alarm binding mismatch")
        recomputed[name]={**row,"representation_checks":checks}
    binding={"canonical_fields":list(runner.ALLOWED_NPZ_FIELDS),"forbidden_fields":["cn0_db_hz","context"],"scenarios":recomputed}
    if audit.get("audit_digest")!=runner._digest_value(binding):raise ValueError("feature audit digest mismatch")
    passed=all(_bool(recomputed[n]["pass"]) for n in ("DS7","DS8"))
    if audit.get("pass") is not passed:raise ValueError("feature audit pass must be derived")
    return {"pass":passed,"scenarios":recomputed}

def _verify_smoke_feature_audit(out:Path,audit:Mapping[str,Any],schema:Mapping[str,Any])->None:
    if audit.get("canonical_fields")!=list(runner.ALLOWED_NPZ_FIELDS) or audit.get("schema_hash")!=runner._digest_value(schema):raise ValueError("smoke feature audit schema mismatch")
    checks=audit.get("scenarios",{})
    if set(checks)!={"cleanStatic","DS7","DS8"}:raise ValueError("smoke feature audit inventory mismatch")
    recomputed={}
    for scenario,row in checks.items():
        reps={}
        for rep,d in (("complex",4),("magnitude",2)):
            saved=row["representations"][rep]
            with np.load(out/saved["file"],allow_pickle=False) as f:part={k:np.array(f[k],copy=True) for k in f.files}
            x=part["features"];iq=np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0);passed=x.shape[1:]==(8,d) and np.isfinite(x).all() and np.all(iq>1e-8)
            if saved.get("bundle_digest")!=runner._digest_value(part) or saved.get("tensor_digest")!=runner._digest_value(x) or saved.get("iqr")!=iq.tolist() or saved.get("pass") is not bool(passed):raise ValueError("smoke feature cache sufficient-stat mismatch")
            reps[rep]=saved
        rows=_rows(out/"per_epoch"/f"{scenario}.csv");stable=rows if scenario=="cleanStatic" else [r for r in rows if r["phase"]=="stable_pre"]
        evidence=[(float(r["decision_time_s"]),float(r["source_start"]),float(r["source_end"]),float(r["score_complex_ensemble"]),1.0,_bool(r["alarm_complex_q99"])) for r in stable];rate=float(np.mean([x[-1] for x in evidence])) if evidence else 0.
        if row.get("stable_alarm_evidence_digest")!=runner._digest_value(evidence) or not _equal(row.get("stable_pre_alarm_rate"),rate):raise ValueError("smoke feature/per-epoch alarm audit mismatch")
        recomputed[scenario]={"representations":reps,"stable_alarm_evidence_digest":row["stable_alarm_evidence_digest"],"stable_pre_alarm_rate":rate,"pass":all(x["pass"] for x in reps.values())}
    binding={"canonical_fields":list(runner.ALLOWED_NPZ_FIELDS),"schema_hash":runner._digest_value(schema),"scenarios":recomputed}
    if audit.get("audit_digest")!=runner._digest_value(binding) or audit.get("pass") is not all(recomputed[n]["pass"] for n in ("DS7","DS8")):raise ValueError("smoke feature audit digest/pass mismatch")


def _primary_criteria(metrics,seeds,paired,feature_verified,convergence):
    index=runner.index_metric_rows([r for r in metrics if r.get("scenario")!="DS4"]);names=tuple(runner.ONSETS)
    c=[runner.select_primary_metric(index,s,"Complex all9") for s in names];m=[runner.select_primary_metric(index,s,"Magnitude all9") for s in names];b=[runner.select_primary_metric(index,s,"B0 Exact") for s in names]
    pindex={(r["scenario"],r["comparator"],r["metric"]):r for r in paired};directions={}
    raw=[r for r in seeds if r.get("operating_point")=="q99" and r.get("seed")!="mean/std"]
    for scenario in names:
        directions[scenario]=sum(float(next(r for r in raw if r["scenario"]==scenario and r["model"]=="Complex all9" and int(r["seed"])==seed)["roc_auc"])>float(next(r for r in raw if r["scenario"]==scenario and r["model"]=="Magnitude all9" and int(r["seed"])==seed)["roc_auc"]) for seed in runner.SEEDS)
    beats=sum(((float(x["roc_auc"])-float(z["roc_auc"])>=.02 or float(x["post_detection"])-float(z["post_detection"])>=.05) and float(x["stable_pre_fpr"])-float(z["stable_pre_fpr"])<=.01) for x,z in zip(c,b))
    return {"stable_pre_fpr_all_below_0.05":all(float(x["stable_pre_fpr"])<.05 for x in c),"complex_auc_gt_magnitude_4_of_5":sum(float(x["roc_auc"])>float(y["roc_auc"]) for x,y in zip(c,m))>=4,"auc_bootstrap_ci_lower_gt_zero_3_of_5":sum(float(pindex[s,"Magnitude all9","roc_auc"]["ci_low"])>0 for s in names)>=3,"same_seed_direction_each_scenario":all(x>=2 for x in directions.values()),"beats_b0_with_fpr_guard_3_of_5":beats>=3,"all_required_seeds_converged":all(_bool(x.get("converged")) for x in convergence.get("audits",{}).values()),"ds7_ds8_no_collapse":feature_verified["pass"]}

def verify_and_summarize(out:Path|str)->dict[str,Any]:
    out=Path(out);runner.verify_hashes(out)
    missing=[x for x in runner.REQUIRED_INVENTORY if not (out/x).exists()]
    if missing:raise ValueError(f"required artifact inventory missing: {missing}")
    config=json.loads((out/"config.json").read_text());mode=config.get("mode")
    if mode not in {"synthetic-smoke","primary-full"}:raise ValueError("unknown artifact mode")
    if config.get("DS4")!={"status":"NA","included_in_attack_go":False} or "DS4:NA" not in config.get("scenarios",[]):raise ValueError("DS4 NA config contract missing")
    schema=_verify_schema(out);thresholds=json.loads((out/"thresholds.json").read_text());decision=json.loads((out/"decision.json").read_text());convergence=json.loads((out/"convergence_audit.json").read_text());provenance=json.loads((out/"provenance.json").read_text());feature_audit=json.loads((out/"feature_audit.json").read_text());history=_rows(out/"training_history.csv")
    _verify_decision_digests(out,decision,mode)
    if thresholds.get("primary")!="q99" or thresholds.get("q995_role")!="diagnostic_only" or thresholds.get("comparison")!="strict_greater":raise ValueError("threshold primary/diagnostic contract mismatch")
    metrics=_rows(out/"scenario_metrics.csv");seeds=_rows(out/"seed_metrics.csv");paired=_rows(out/"paired_comparisons.csv")
    derived=None
    if mode=="primary-full":_verify_primary_config(config,provenance);_verify_execution_source(provenance);_verify_calibration(out,thresholds,provenance);derived=_derive_primary_fit_bindings(out,schema,provenance)
    checked=_verify_metrics(out,mode,thresholds,metrics);_verify_convergence(out,convergence,history,mode,derived)
    if mode=="primary-full":
        _verify_conformal_and_phase_rows(out,thresholds);_verify_seed_metrics(out,seeds);_verify_paired(out,paired,thresholds);verified_feature=_verify_feature_audit(out,feature_audit,schema,provenance);criteria=_primary_criteria(metrics,seeds,paired,verified_feature,convergence);status="PRIMARY COMPLETE"
    else:_verify_smoke_feature_audit(out,feature_audit,schema);criteria=runner._smoke_criteria();status="SMOKE-NO-GO"
    final="GO" if all(criteria.values()) else "NO-GO"
    if decision.get("primary_quantile")!="q99" or decision.get("primary_decision")!=final or decision.get("criteria")!=criteria:raise ValueError("deterministic q99 GO recomputation mismatch")
    expected=runner.render_readme(final,status,criteria).encode();actual=(out/"README.md").read_bytes()
    if expected!=actual:raise ValueError("README is not byte-identical to deterministic regeneration")
    return {"schema":"gnss-doppler-lab.amcf-shape-only-summary-audit.v2","hashes_verified":True,"byte_identical":True,"metrics_recomputed":checked,"alarms_recomputed":True,"thresholds_recomputed":mode=="primary-full","paired_recomputed":mode=="primary-full","checkpoints_loaded":mode=="primary-full","primary_quantile":"q99","primary_decision":final,"amcf_wcl":"GO candidate" if final=="GO" else "AMCF WCL no-go","criteria":criteria,"mode":mode}

def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("artifact",type=Path);p.add_argument("--audit-out",type=Path);a=p.parse_args(argv);report=verify_and_summarize(a.artifact);text=json.dumps(report,sort_keys=True,indent=2)+"\n";
    if a.audit_out:a.audit_out.write_text(text,encoding="utf-8",newline="\n")
    print(json.dumps(report,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
