#!/usr/bin/env python3
"""No-retry, one-worker durable supervisor for the Stage-0 production command."""
from __future__ import annotations
import argparse, json, os, platform, signal, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.r2c_stage0_observer import atomic_json


def git(*args): return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()


def hardware_key():
    return {"machine":platform.machine(),"processor":platform.processor(),"python":platform.python_version()}


def validate_go(path: Path, source_sha: str, config_hash: str):
    value=json.loads(path.read_text())
    if value.get("status")!="GO" or value.get("source_sha")!=source_sha or value.get("config_hash")!=config_hash:
        raise ValueError("benchmark GO manifest does not match exact source/config")
    if value.get("hardware")!=hardware_key(): raise ValueError("benchmark GO manifest hardware mismatch")
    return value


def supervise(command, attempt_dir: Path, *, stale_s=45., poll_s=.25):
    attempt_dir.mkdir(parents=False)  # atomic reservation; duplicate launch fails
    atomic_json(attempt_dir/"supervisor.json",{"status":"STARTING","started":time.time(),"command":command})
    stdout=(attempt_dir/"stdout.log").open("wb");stderr=(attempt_dir/"stderr.log").open("wb")
    environment=os.environ.copy();environment["R2C_ATTEMPT_ID"]=attempt_dir.name;environment["R2C_ATTEMPT_DIR"]=str(attempt_dir)
    process=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True,env=environment)
    atomic_json(attempt_dir/"worker.json",{"pid":process.pid,"started":time.time()})
    forwarded=False; stale_recorded=False
    def terminate(*_):
        nonlocal forwarded
        forwarded=True
        try: os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError: pass
    old=signal.signal(signal.SIGTERM,terminate)
    try:
        while process.poll() is None:
            heartbeat=attempt_dir/"heartbeat.json"
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
    parser=argparse.ArgumentParser();parser.add_argument("--go-manifest",type=Path,required=True)
    parser.add_argument("--config-hash",required=True);parser.add_argument("--attempt-root",type=Path,required=True)
    parser.add_argument("--attempt-id",required=True);parser.add_argument("command",nargs=argparse.REMAINDER)
    args=parser.parse_args();source=git("rev-parse","HEAD")
    if git("status","--porcelain=v1"):parser.error("production preflight requires clean worktree")
    validate_go(args.go_manifest,source,args.config_hash)
    if not args.command:parser.error("worker command required after --")
    command=args.command[1:] if args.command[0]=="--" else args.command
    raise SystemExit(supervise(command,args.attempt_root/args.attempt_id))
if __name__=="__main__":main()
