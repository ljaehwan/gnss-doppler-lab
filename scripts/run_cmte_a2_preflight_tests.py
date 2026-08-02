#!/usr/bin/env python3
"""Run the fixed CMTE-A2 pre-campaign test suite and seal subprocess evidence."""
from __future__ import annotations
import argparse, json, os, platform, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte_a2_campaign import (TEST_ATTESTATION_SCHEMA,atomic_json,canonical_preflight_argv,file_sha256,validate_source_tree)

COUNT_RE=re.compile(r"(?P<count>\d+)\s+(?P<kind>passed|failed|error|errors|skipped|xfailed|xpassed)")

def _utc(): return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")

def _counts(text):
 counts={"passed":0,"failed":0,"skipped":0}
 for match in COUNT_RE.finditer(text):
  kind=match.group("kind"); count=int(match.group("count"))
  if kind=="passed": counts["passed"]+=count
  elif kind in {"failed","error","errors"}: counts["failed"]+=count
  elif kind=="skipped": counts["skipped"]+=count
 return counts

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--source-commit",required=True); p.add_argument("--out",required=True); a=p.parse_args(argv)
 validate_source_tree(ROOT,a.source_commit,require_clean=True)
 out=Path(a.out).absolute(); log=out.with_suffix(out.suffix+".log")
 if out.exists() or log.exists(): raise FileExistsError("preflight attestation and log are atomic non-overwriting outputs")
 out.parent.mkdir(parents=True,exist_ok=True); tmp_log=log.with_name(log.name+f".tmp-{os.getpid()}")
 started=_utc(); commands=[]; aggregate_exit=0
 try:
  with tmp_log.open("w",encoding="utf-8") as stream:
   for index,canonical in enumerate(canonical_preflight_argv(ROOT),1):
    command=list(canonical); command_started=_utc()
    stream.write(f"## command {index}\n$ {json.dumps(command)}\nstarted_utc={command_started}\n"); stream.flush()
    result=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
    stream.write(result.stdout); command_completed=_utc(); stream.write(f"\ncompleted_utc={command_completed}\nexit_code={result.returncode}\n"); stream.flush(); os.fsync(stream.fileno())
    counts=_counts(result.stdout); commands.append({"argv":command,"cwd":str(ROOT),"started_utc":command_started,
      "completed_utc":command_completed,"exit_code":result.returncode,**counts})
    if result.returncode!=0: aggregate_exit=result.returncode or 1
  os.replace(tmp_log,log)
  summary={k:sum(x[k] for x in commands) for k in ("passed","failed","skipped")}; summary["tests"]=summary["passed"]+summary["failed"]
  doc={"schema":TEST_ATTESTATION_SCHEMA,"source_commit":a.source_commit,"clean_tree_asserted":True,
    "started_utc":started,"completed_utc":_utc(),"exit_code":aggregate_exit,"commands":commands,"summary":summary,
    "log":{"path":str(log.resolve()),"sha256":file_sha256(log),"bytes":log.stat().st_size},
    "python":{"executable":commands[0]["argv"][0],"version":sys.version,"platform":platform.platform()},
    "environment":{k:os.environ.get(k,"") for k in ("PYTHONHASHSEED","PYTHONPATH","VIRTUAL_ENV","PATH")},
    "fixed_suite":True,"holdout_accessed":False,"subprocess_e2e_attested":False}
  atomic_json(out,doc,exclusive=True); print(json.dumps({"attestation":str(out),"log":str(log),"source_commit":a.source_commit,**summary},sort_keys=True))
  return aggregate_exit
 finally:
  if tmp_log.exists(): tmp_log.unlink()
if __name__=="__main__": raise SystemExit(main())
