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
    return {"machine":platform.machine(),"processor":platform.processor(),"python":platform.python_version()}


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
    if value.get("schema")!="gnss-doppler-lab.r2c-stage0-profile-benchmark.v2" or value.get("status")!="GO" or value.get("source_sha")!=source_sha:
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
    projection=value.get("projection",{});counters=value.get("runtime_counters",{});gates=value.get("gates",{})
    recomputed={"cleanstatic_end_to_end_p95":value.get("stage_times",{}).get("cleanstatic_end_to_end_s",float("inf"))<=900,
      "all_scenario_projected_p95":projection.get("all_scenario_p95_s",float("inf"))<=5400,"safety_upper":projection.get("upper_bound_s",float("inf"))<=7200,
      "peak_rss":value.get("peak_rss_bytes",float("inf"))<=2*1024**3,"instrumentation":all(counters.get(k) is not None for k in ("bank_evaluations","near_tie_events","refined_candidate_count","decomposition_calls","refinement_elapsed_s","near_tie_rate"))}
    if gates!=recomputed or not all(recomputed.values()):raise ValueError("benchmark numeric gates do not independently pass")
    return value


def supervise(command, attempt_dir: Path, *, stale_s=45., initial_heartbeat_s=45.,poll_s=.25,reservation_namespace="production"):
    reservation=attempt_dir.parent/f".{reservation_namespace}.campaign-reservation"
    reservation.mkdir(parents=False)
    atomic_json(reservation/"reservation.json",{"attempt_id":attempt_dir.name,"reserved":time.time(),"pid":os.getpid()})
    attempt_dir.mkdir(parents=False)  # atomic reservation; duplicate launch fails
    atomic_json(attempt_dir/"supervisor.json",{"status":"STARTING","started":time.time(),"command":command})
    stdout=(attempt_dir/"stdout.log").open("wb");stderr=(attempt_dir/"stderr.log").open("wb")
    environment=os.environ.copy();environment["R2C_ATTEMPT_ID"]=attempt_dir.name;environment["R2C_ATTEMPT_DIR"]=str(attempt_dir)
    process=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True,env=environment)
    atomic_json(attempt_dir/"worker.json",{"pid":process.pid,"started":time.time()})
    forwarded=False; stale_recorded=False;missing_recorded=False;worker_started=time.time()
    def terminate(*_):
        nonlocal forwarded
        forwarded=True
        try: os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError: pass
    old=signal.signal(signal.SIGTERM,terminate)
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
        signal.signal(signal.SIGTERM,old);stdout.close();stderr.close()
    atomic_json(attempt_dir/"supervisor.json",{"status":"FINISHED","finished":time.time(),"exit_code":code,
                                               "sigterm_forwarded":forwarded,"auto_retries":0})
    return code


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--go-manifest",type=Path,required=True);parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--input",action="append",required=True);parser.add_argument("--geometry",type=Path,required=True);parser.add_argument("--b0-validation",type=Path,required=True);parser.add_argument("--attempt-root",type=Path,required=True)
    parser.add_argument("--attempt-id",required=True);parser.add_argument("command",nargs=argparse.REMAINDER)
    parser.add_argument("--reservation-namespace",default="production")
    args=parser.parse_args();source=git("rev-parse","HEAD")
    if git("status","--porcelain=v1"):parser.error("production preflight requires clean worktree")
    inputs=dict(item.split("=",1) for item in args.input)
    if tuple(inputs)!=("cleanStatic","cleanDynamic","DS1","DS2","DS3","DS7","DS8"):parser.error("exact canonical input order required")
    validate_go(args.go_manifest,source,args.config,{name:Path(path) for name,path in inputs.items()},args.geometry,args.b0_validation)
    if not args.command:parser.error("worker command required after --")
    command=args.command[1:] if args.command[0]=="--" else args.command
    raise SystemExit(supervise(command,args.attempt_root/args.attempt_id,reservation_namespace=args.reservation_namespace))
if __name__=="__main__":main()
