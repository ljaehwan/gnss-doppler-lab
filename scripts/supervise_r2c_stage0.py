#!/usr/bin/env python3
"""No-retry, one-worker durable supervisor for the Stage-0 production command."""
from __future__ import annotations
import argparse, contextlib,hashlib,io,json, os, platform, signal, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.r2c_stage0_observer import atomic_json


def git(*args): return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()


def hardware_key():
    model=""
    try:model=next(line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    except (OSError,StopIteration):pass
    return {"machine":platform.machine(),"processor":platform.processor(),"cpu_model":model,"logical_cores":os.cpu_count(),
            "affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,"python":platform.python_version()}


def campaign_runtime_gate(safety_upper_seconds):
    return float(safety_upper_seconds)<=8*3600


def sha(path):
    h=hashlib.sha256();h.update(Path(path).read_bytes());return h.hexdigest()
def workload_digest(path):
    with np.load(path,allow_pickle=False) as z:times=np.asarray(z["time_s"],float);prns=np.asarray(z["prn"],int)
    bins=np.floor(times/.5).astype(np.int64);records=[]
    for value in np.unique(bins):
        selected=bins==value;p=prns[selected];records.append((int(value),int(selected.sum()),int(len(np.unique(p))),[int(np.sum(p==x)) for x in np.unique(p)]))
    return hashlib.sha256(json.dumps(records,separators=(",",":")).encode()).hexdigest()
def validate_go(path: Path, source_sha: str, config:Path,inputs,geometry:Path,b0:Path):
    value=json.loads(path.read_text())
    if value.get("schema")!="gnss-doppler-lab.r2c-stage0-profile-benchmark.v3" or value.get("status")!="GO" or value.get("source_sha")!=source_sha:
        raise ValueError("benchmark GO manifest does not match exact source/config")
    if value.get("config",{}).get("sha256")!=sha(config):raise ValueError("benchmark config hash mismatch")
    if value.get("geometry",{}).get("sha256")!=sha(geometry) or value.get("b0",{}).get("sha256")!=sha(b0):raise ValueError("benchmark wrapper hash mismatch")
    records=value.get("inputs",{});digests={}
    for name,input_path in inputs.items():
        if records.get(name,{}).get("path")!=str(input_path.resolve()) or records.get(name,{}).get("sha256")!=sha(input_path):raise ValueError("benchmark input binding mismatch")
        digest=workload_digest(input_path);digests[name]=digest
        if records[name].get("workload_digest")!=digest:raise ValueError("benchmark workload count mismatch")
    aggregate=hashlib.sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest()
    if value.get("workload_count_digest")!=aggregate:raise ValueError("benchmark workload digest mismatch")
    if value.get("hardware")!=hardware_key(): raise ValueError("benchmark GO manifest hardware mismatch")
    blas=io.StringIO()
    with contextlib.redirect_stdout(blas):np.show_config()
    if value.get("identity",{}).get("blas")!=blas.getvalue():raise ValueError("benchmark BLAS identity mismatch")
    identity=value.get("identity",{});import scipy
    current_gpu=subprocess.run(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
    expected={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"gpu":current_gpu,
      "ram_bytes":os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES"),"threads":{key:os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}}
    if any(identity.get(key)!=item for key,item in expected.items()):raise ValueError("benchmark software/hardware identity mismatch")
    execution=value.get("execution",{})
    expected_backend=os.environ.get("R2C_STAGE0_SCORE_BACKEND","fork").strip().lower()
    expected_workers=int(os.environ.get("R2C_STAGE0_SCORE_WORKERS",min(os.cpu_count() or 1,16)))
    if execution!={"score_backend":expected_backend,"score_workers":expected_workers}:raise ValueError("benchmark execution backend/workers mismatch")
    projection=value.get("projection",{});formula=value.get("formula",{});counters=value.get("runtime_counters",{});gates=value.get("gates",{})
    numeric=list(counters.values())
    if not numeric or any(not isinstance(x,(int,float)) or not np.isfinite(x) or x<0 for x in numeric):raise ValueError("benchmark counters malformed")
    if counters.get("near_tie_rate")!=counters.get("near_tie_events",0)/max(counters.get("bank_evaluations",0),1):raise ValueError("counter rate inconsistent")
    estimate=formula.get("preprocessing_and_model_fit_s",np.nan)+formula.get("projected_scenario_scoring_s",np.nan)+formula.get("metrics_bootstrap_controls_assembly_s",np.nan)
    if not np.isclose(estimate,projection.get("all_scenario_estimate_s",np.nan),rtol=0,atol=1e-9):raise ValueError("estimate formula inconsistent")
    if not np.isclose(estimate*formula.get("safety_factor",np.nan),projection.get("all_scenario_safety_upper_s",np.nan),rtol=0,atol=1e-9):raise ValueError("safety formula inconsistent")
    stages=value.get("stage_times",{})
    if not np.isclose(stages.get("pre_scoring_total_s",np.nan)+stages.get("cleanstatic_score_bin_all_s",np.nan),stages.get("cleanstatic_end_to_end_s",np.nan),atol=1e-9):raise ValueError("stage formula inconsistent")
    recomputed={"cleanstatic_repeated_max":value.get("cleanstatic_max_s",float("inf"))<=900,
      "all_scenario_safety_upper_8h":campaign_runtime_gate(projection.get("all_scenario_safety_upper_s",float("inf"))),
      "peak_rss":value.get("peak_rss_bytes",float("inf"))<=1879048192,"instrumentation":True}
    if gates!=recomputed or not all(recomputed.values()):raise ValueError("benchmark numeric gates do not independently pass")
    return value


def supervise(command, attempt_dir: Path, *, stale_s=45., initial_heartbeat_s=45.,poll_s=.25,campaign_identity="test"):
    process=None;pending=False;forwarded=False
    def terminate(*_):
        nonlocal pending,forwarded
        pending=True;forwarded=True
        if process is not None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    old_term=signal.signal(signal.SIGTERM,terminate);old_int=signal.signal(signal.SIGINT,terminate)
    reservation=attempt_dir.parent/f".production-{campaign_identity}.campaign-reservation"
    try:
      reservation.mkdir(parents=False);atomic_json(reservation/"reservation.json",{"attempt_id":attempt_dir.name,"reserved":time.time(),"pid":os.getpid()})
      attempt_dir.mkdir(parents=False)
      atomic_json(attempt_dir/"supervisor.json",{"status":"STARTING","started":time.time(),"command":command})
      stdout=(attempt_dir/"stdout.log").open("wb");stderr=(attempt_dir/"stderr.log").open("wb")
      environment=os.environ.copy();environment["R2C_ATTEMPT_ID"]=attempt_dir.name;environment["R2C_ATTEMPT_DIR"]=str(attempt_dir)
      process=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True,env=environment)
      if pending:
          try:os.killpg(process.pid,signal.SIGTERM)
          except ProcessLookupError:pass
    except Exception:
      signal.signal(signal.SIGTERM,old_term);signal.signal(signal.SIGINT,old_int);raise
    atomic_json(attempt_dir/"worker.json",{"pid":process.pid,"started":time.time()})
    stale_recorded=False;missing_recorded=False;worker_started=time.time()
    try:
        while process.poll() is None:
            heartbeat=attempt_dir/"heartbeat.json"
            if not heartbeat.exists() and time.time()-worker_started>initial_heartbeat_s and not missing_recorded:
                atomic_json(attempt_dir/"missing-initial-heartbeat-warning.json",{"warning":"missing_initial_heartbeat","time":time.time()})
                missing_recorded=True
            if heartbeat.exists() and time.time()-heartbeat.stat().st_mtime>stale_s and not stale_recorded:
                atomic_json(attempt_dir/"stale-heartbeat-warning.json",{"warning":"stale_heartbeat","time":time.time()})
                stale_recorded=True
            time.sleep(poll_s)
        code=int(process.returncode)
    finally:
        # If the runner is OOM-killed, its forked score children can otherwise be
        # re-parented and retain GBs of memory.  The worker owns a new session, so
        # reap the entire process group on every terminal path.
        if process is not None:
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            time.sleep(.2)
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError: pass
        signal.signal(signal.SIGTERM,old_term);signal.signal(signal.SIGINT,old_int);stdout.close();stderr.close()
    atomic_json(attempt_dir/"supervisor.json",{"status":"FINISHED","finished":time.time(),"exit_code":code,
                                               "sigterm_forwarded":forwarded,"auto_retries":0})
    return code


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--go-manifest",type=Path,required=True);parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--input",action="append",required=True);parser.add_argument("--geometry",type=Path,required=True);parser.add_argument("--b0-validation",type=Path,required=True);parser.add_argument("--attempt-root",type=Path,required=True)
    parser.add_argument("--attempt-id",required=True);parser.add_argument("--source-commit",required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();source=git("rev-parse","HEAD")
    if source!=args.source_commit:parser.error("source commit mismatch")
    if git("status","--porcelain=v1"):parser.error("production preflight requires clean worktree")
    inputs=dict(item.split("=",1) for item in args.input)
    if tuple(inputs)!=("cleanStatic","cleanDynamic","DS1","DS2","DS3","DS7","DS8"):parser.error("exact canonical input order required")
    validate_go(args.go_manifest,source,args.config,{name:Path(path) for name,path in inputs.items()},args.geometry,args.b0_validation)
    canonical=(ROOT/"artifacts/r2c_gnss_stage0_fix").resolve()
    if args.output.resolve()!=canonical:parser.error("canonical output path required")
    command=[sys.executable,str(ROOT/"scripts/run_r2c_gnss_stage0_fix.py"),"--config",str(args.config),"--output",str(canonical),"--source-commit",source]
    for name,path in inputs.items():command += ["--input",f"{name}={path}"]
    command += ["--geometry",str(args.geometry),"--b0-validation",str(args.b0_validation)]
    identity=hashlib.sha256(json.dumps({"output":str(canonical),"source":source,"config":sha(args.config),"inputs":{n:sha(Path(p)) for n,p in inputs.items()}},sort_keys=True).encode()).hexdigest()[:24]
    raise SystemExit(supervise(command,args.attempt_root/args.attempt_id,campaign_identity=identity))
if __name__=="__main__":main()
