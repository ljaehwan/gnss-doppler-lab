#!/usr/bin/env python3
"""Production-shape cleanStatic benchmark; attack inputs are metadata-counted only."""
from __future__ import annotations
import argparse,contextlib,hashlib,importlib.util,io,json,os,platform,random,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
import scipy

os.environ.setdefault("OPENBLAS_NUM_THREADS","1");os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1")
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
from gnss_doppler_lab.r2c_stage0_fix import ComplexWhitener,TemplateProvider,compile_profile_plan,joint_profile_glrt
from supervise_r2c_stage0 import hardware_key
SCENARIOS=("cleanStatic","cleanDynamic","DS1","DS2","DS3","DS7","DS8")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):h.update(block)
    return h.hexdigest()
def load_runner():
    path=ROOT/"scripts/run_r2c_gnss_stage0_fix.py";spec=importlib.util.spec_from_file_location("r2c_benchmark_runner",path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module
def specs(values):
    result=dict(item.split("=",1) for item in values)
    if tuple(result)!=SCENARIOS:raise ValueError(f"inputs must be in exact canonical order {SCENARIOS}")
    return {name:Path(value) for name,value in result.items()}
def histogram(path):
    with np.load(path,allow_pickle=False) as z:
        time_s=np.asarray(z["time_s"],float);prn=np.asarray(z["prn"],int)
    bins=np.floor(time_s/.5).astype(np.int64);records=[]
    for bin_id in np.unique(bins):
        selected=bins==bin_id;ps=prn[selected];counts=[int(np.sum(ps==p)) for p in np.unique(ps)]
        records.append((int(bin_id),int(selected.sum()),int(len(counts)),counts))
    digest=hashlib.sha256(json.dumps(records,separators=(",",":")).encode()).hexdigest()
    return {"rows":len(time_s),"bins":len(records),"prns":len(np.unique(prn)),"workload_digest":digest,
            "bin_records":records,"epochs_per_prn":{"min":min(min(x[3]) for x in records),"max":max(max(x[3]) for x in records)}}
def percentile(values,q):return float(np.percentile(np.asarray(values,float),q))
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--input",action="append",required=True)
    parser.add_argument("--geometry",type=Path,required=True);parser.add_argument("--b0-validation",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/r2c_gnss_stage0_fix.json");parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();inputs=specs(args.input);config=json.loads(args.config.read_text());runner=load_runner()
    geometry_doc=json.loads(args.geometry.read_text());b0_doc=json.loads(args.b0_validation.read_text())
    if set(geometry_doc.get("scenarios",{}))!=set(SCENARIOS) or set(b0_doc.get("scenarios",{}))!=set(SCENARIOS):raise ValueError("wrapper roster mismatch")
    workloads={name:histogram(path) for name,path in inputs.items()}
    input_records={name:{"path":str(path.resolve()),"sha256":sha(path),**{k:v for k,v in workloads[name].items() if k!="bin_records"}} for name,path in inputs.items()}
    provider=TemplateProvider.analytic();taps=np.asarray(config["tap_offsets_chips"]);grid=np.asarray(config["delay_grid_chips"])
    geometry=geometry_doc["scenarios"]["cleanStatic"];causal=geometry.get("event_time_causal_ephemeris_availability",{}).get("status")=="PASS"
    def measure_clean():
      stage={};started=time.perf_counter();clean=runner.load_all_epochs(inputs["cleanStatic"]);stage["load_cleanstatic_s"]=time.perf_counter()-started
      started=time.perf_counter();raw=runner.h0_residuals(clean["y"],provider,taps,grid,chunk_rows=config["epoch_policy"]["chunk_rows"]);stage["h0_residual_s"]=time.perf_counter()-started
      started=time.perf_counter();models=runner.fit_frozen_models(clean,config,provider,taps,grid,require_gpu=config["neural"]["require_gpu"],raw_residuals=raw);stage["frozen_model_fit_s"]=time.perf_counter()-started
      started=time.perf_counter();x=runner.inference_conditions(clean,raw,models["cn0_imputation"],False);xe=runner.inference_conditions(clean,raw,models["cn0_imputation"],True);stage["inference_conditions_s"]=time.perf_counter()-started
      stage["pre_scoring_total_s"]=sum(stage.values());los_bins=runner.causal_core_los(geometry) if causal else {};costs=[];scoring_started=time.perf_counter()
      for bin_id in np.unique(clean["bin"]):
        indices=np.flatnonzero(clean["bin"]==bin_id);observations={};conditions={};los={}
        for p in sorted(np.unique(clean["prn"][indices])):
          chosen=indices[clean["prn"][indices]==p];observations[int(p)]=clean["y"][chosen];conditions[int(p)]=(x[chosen],xe[chosen])
          if str(int(p)) in los_bins.get(str(int(bin_id)),{}):los[int(p)]=np.asarray(los_bins[str(int(bin_id))][str(int(p))],float)
        tick=time.perf_counter();runner.score_bin(observations,los,provider,taps,grid,models,config,conditions)
        costs.append({"seconds":time.perf_counter()-tick,"epochs":sum(map(len,observations.values())),"prns":len(observations)})
      stage["cleanstatic_score_bin_all_s"]=time.perf_counter()-scoring_started;stage["cleanstatic_end_to_end_s"]=stage["pre_scoring_total_s"]+stage["cleanstatic_score_bin_all_s"]
      return stage,models,costs,clean
    runs=[]
    for _ in range(3):stage,models,costs,clean=measure_clean();runs.append(stage)
    stage=max(runs,key=lambda item:item["cleanstatic_end_to_end_s"]);cleanstatic_runs=[item["cleanstatic_end_to_end_s"] for item in runs]
    sequential_preprocess={}
    for name in SCENARIOS[1:]:
      tick=time.perf_counter();dataset=runner.load_all_epochs(inputs[name]);raw_scenario=runner.h0_residuals(dataset["y"],provider,taps,grid,chunk_rows=config["epoch_policy"]["chunk_rows"])
      runner.inference_conditions(dataset,raw_scenario,models["cn0_imputation"],False);runner.inference_conditions(dataset,raw_scenario,models["cn0_imputation"],True)
      sequential_preprocess[name]=time.perf_counter()-tick;del dataset,raw_scenario
    design=np.asarray([[1,c["epochs"],c["prns"]] for c in costs],float);target=np.asarray([c["seconds"] for c in costs]);coef=np.linalg.lstsq(design,target,rcond=None)[0]
    residual=target-design@coef;residual_p95=max(0.,percentile(residual,95));projected={}
    for name,workload in workloads.items():
        estimates=[max(0.,float(np.dot([1,row[1],row[2]],coef))+residual_p95) for row in workload["bin_records"]]
        projected[name]={"seconds":sum(estimates),"bins":len(estimates)}
    all_projected=sum(item["seconds"] for item in projected.values());non_scoring_upper=60.
    projected_p95=all_projected+stage["pre_scoring_total_s"]+sum(sequential_preprocess.values())+non_scoring_upper
    # Warmed, randomized/interleaved paired kernel diagnostics; relative speedups are diagnostic only.
    w=ComplexWhitener(shrinkage=0);w.mean=np.zeros(9,complex);w.covariance=np.eye(9);w.inverse_sqrt=np.eye(9);w.pseudo_covariance=np.zeros((9,9),complex);w.diagnostics={}
    plan=compile_profile_plan(provider,taps,grid,w);sample={1:clean["y"][:17]};joint_profile_glrt(sample,{},provider,taps,grid,hypothesis="H1-independent",whitener=w,profile_plan=plan)
    paired=[];orders=["scalar","vector"]*5;random.Random(config["seed"]).shuffle(orders)
    for first in orders:
        row={}
        for kind in (first,"vector" if first=="scalar" else "scalar"):
            tick=time.perf_counter();joint_profile_glrt(sample,{},provider,taps,grid,hypothesis="H1-independent",whitener=w,
              profile_plan=plan if kind=="vector" else None,scalar_reference=kind=="scalar");row[kind]=time.perf_counter()-tick
        paired.append(row)
    counters={key:sum(p.counters.snapshot()[key] for p in models["profile_plans"].values())+getattr(plan.counters,key) for key in plan.counters.snapshot()}
    counters["near_tie_rate"]=counters["near_tie_events"]/max(counters["bank_evaluations"],1)
    peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024;upper=1.25*projected_p95
    gates={"cleanstatic_repeated_max":max(cleanstatic_runs)<=900,"all_scenario_safety_upper":upper<=7200,
           "peak_rss":peak<=1879048192,"instrumentation":all(np.isfinite(counters[key]) and counters[key]>=0 for key in counters)}
    blas=io.StringIO()
    with contextlib.redirect_stdout(blas):np.show_config()
    report={"schema":"gnss-doppler-lab.r2c-stage0-profile-benchmark.v3","status":"GO" if all(gates.values()) else "NO_GO","gates":gates,
      "source_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"config":{"path":str(args.config.resolve()),"sha256":sha(args.config)},
      "inputs":input_records,"workload_count_digest":hashlib.sha256(json.dumps({n:w["workload_digest"] for n,w in workloads.items()},sort_keys=True).encode()).hexdigest(),
      "geometry":{"path":str(args.geometry.resolve()),"sha256":sha(args.geometry),"cleanstatic_causal_geometry":causal},"b0":{"path":str(args.b0_validation.resolve()),"sha256":sha(args.b0_validation),"aggregate_status":b0_doc.get("aggregate_status")},
      "hardware":hardware_key(),"identity":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"blas":blas.getvalue(),"cpu":platform.processor(),
        "ram_bytes":os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES"),"gpu":subprocess.run(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(),
        "threads":{key:os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}},
      "cleanstatic_runs_s":cleanstatic_runs,"cleanstatic_median_s":percentile(cleanstatic_runs,50),"cleanstatic_max_s":max(cleanstatic_runs),
      "stage_times":stage,"clean_bin_cost":{"median_s":percentile([c["seconds"] for c in costs],50),"p95_s":percentile([c["seconds"] for c in costs],95),"regression_coefficients":coef.tolist(),"residual_p95_s":residual_p95},
      "sequential_preprocess_s":sequential_preprocess,
      "formula":{"preprocessing_and_model_fit_s":stage["pre_scoring_total_s"]+sum(sequential_preprocess.values()),"projected_scenario_scoring_s":all_projected,"metrics_bootstrap_controls_assembly_s":non_scoring_upper,"safety_factor":1.25},
      "projection":{"scenarios":projected,"all_scenario_estimate_s":projected_p95,"all_scenario_safety_upper_s":upper},"peak_rss_bytes":peak,"runtime_counters":counters,
      "kernel_diagnostics":{"repetitions":len(paired),"scalar_median_s":percentile([x["scalar"] for x in paired],50),"scalar_p95_s":percentile([x["scalar"] for x in paired],95),
        "vector_median_s":percentile([x["vector"] for x in paired],50),"vector_p95_s":percentile([x["vector"] for x in paired],95)}}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    if report["status"]!="GO":print("benchmark NO_GO: "+", ".join(k for k,v in gates.items() if not v),file=sys.stderr);return 1
    print(json.dumps({"status":"GO","manifest":str(args.output)}));return 0
if __name__=="__main__":raise SystemExit(main())
