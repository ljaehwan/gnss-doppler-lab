"""GNSS-SDR replay orchestration for CRID."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, os, resource, signal, subprocess, time
from pathlib import Path
from typing import Mapping
from .crid import receiver_configurations, render_receiver_config

def sha256_file(path:Path)->str:
 d=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b""):d.update(chunk)
 return d.hexdigest()

def _proc_io(pid:int)->dict[str,int]:
 path=Path(f"/proc/{pid}/io")
 try:
  return {key:int(value) for key,value in (line.split(":",1) for line in path.read_text().splitlines())}
 except (FileNotFoundError,PermissionError,ValueError):return {}

def _raw_fd_state(pid:int,raw:Path)->tuple[int|None,int|None]:
 """Return the receiver's raw-IQ fd and its byte offset, if still open."""
 target=str(raw.resolve());fd_root=Path(f"/proc/{pid}/fd")
 try:entries=list(fd_root.iterdir())
 except (FileNotFoundError,PermissionError):return None,None
 for entry in entries:
  try:
   if os.readlink(entry)!=target:continue
   fields=dict(line.split(":",1) for line in (fd_root.parent/"fdinfo"/entry.name).read_text().splitlines())
   return int(entry.name),int(fields["pos"].strip())
  except (FileNotFoundError,PermissionError,OSError,KeyError,ValueError):continue
 return None,None

def _dump_sizes(out:Path)->dict[str,int]:
 return {p.name:p.stat().st_size for p in sorted(out.glob("trace_native_1ms_ch_*.bin"))}

def _supervised_run(command:list[str],*,cwd:Path,log,raw:Path,expected_end_byte:int,
 expected_dump_count:int=10,poll_s:float=.5,stable_s:float=5.,grace_s:float=30.)->tuple[int,dict]:
 """Run finite replay and request SIGINT only after proved EOF and dump stability."""
 process=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 began=time.monotonic();timeline=[];max_pos=None;last_sizes={};stable_since=None;sigint_at=None
 while process.poll() is None:
  now=time.monotonic();fd,pos=_raw_fd_state(process.pid,raw)
  if pos is not None:max_pos=pos if max_pos is None else max(max_pos,pos)
  sizes=_dump_sizes(cwd);io=_proc_io(process.pid)
  if sizes==last_sizes and len(sizes)==expected_dump_count and all(size>0 for size in sizes.values()):
   stable_since=stable_since if stable_since is not None else now
  else:stable_since=now if len(sizes)==expected_dump_count and all(size>0 for size in sizes.values()) else None
  last_sizes=sizes
  timeline.append({"elapsed_s":round(now-began,6),"state":"running","raw_fd":fd,
   "raw_fd_position":pos,"max_raw_fd_position":max_pos,"dump_count":len(sizes),
   "dump_bytes":sum(sizes.values()),"rchar":io.get("rchar"),"wchar":io.get("wchar"),
   "read_bytes":io.get("read_bytes"),"write_bytes":io.get("write_bytes"),"signal":None})
  eof=max_pos==expected_end_byte
  stable=stable_since is not None and now-stable_since>=stable_s
  if sigint_at is None and eof and stable:
   os.killpg(process.pid,signal.SIGINT);sigint_at=now;timeline[-1]["signal"]="SIGINT"
  if sigint_at is not None and now-sigint_at>grace_s:
   os.killpg(process.pid,signal.SIGKILL);process.wait()
   return process.returncode,{"status":"FAIL","failure":"graceful_sigint_timeout_sigkill_cleanup",
    "exit_cause":"SIGKILL_CLEANUP_FAILURE","expected_raw_fd_end_byte":expected_end_byte,
    "max_raw_fd_position":max_pos,"input_bytes_consumed_exactly":max_pos==expected_end_byte,
    "sigint_sent":True,"sigterm_sent":False,"sigkill_sent":True,"timeline":timeline}
  time.sleep(poll_s)
 rc=process.returncode;ended=time.monotonic();sizes=_dump_sizes(cwd)
 exit_cause="natural_eos" if sigint_at is None else "verified_eof_graceful_sigint"
 accepted=rc==0 and len(sizes)==expected_dump_count and (sigint_at is None or max_pos==expected_end_byte)
 termination={"status":"PASS" if accepted else "FAIL","exit_cause":exit_cause,
  "exit_code":rc,"elapsed_s":ended-began,"expected_raw_fd_end_byte":expected_end_byte,
  "max_raw_fd_position":max_pos,"input_bytes_consumed_exactly":max_pos==expected_end_byte,
  "input_completion_evidence":"natural_eos_rc0" if sigint_at is None else "exact_raw_fd_endpoint",
  "expected_dump_count":expected_dump_count,"dump_count":len(sizes),"dump_sizes":sizes,
  "pre_signal_stability_s":0. if sigint_at is None else sigint_at-(stable_since or sigint_at),
  "sigint_sent":sigint_at is not None,"sigterm_sent":False,"sigkill_sent":False,"timeline":timeline}
 return rc,termination

def run_replay(*,receiver:Path,base_config:Path,raw:Path,out:Path,scenario:str,
 config_name:str,fs:int,skip_s:float,duration_s:float,raw_sha256:str|None=None,
 terminal_drain_repair:bool=True)->dict:
 out.mkdir(parents=True,exist_ok=True);cfg=out/"receiver.conf"
 for prior in out.glob("trace_native_1ms_ch_*.bin"):
  prior.unlink()
 values={"SignalSource.filename":str(raw),"SignalSource.seconds_to_skip":skip_s,
  "SignalSource.samples":int(duration_s*fs*2),"SignalSource.repeat":"false",
  "Tracking_1C.dump":"false","Tracking_1C.dump_mat":"false",
  "Tracking_1C.trace_dump":"true","Tracking_1C.trace_dump_filename":"trace_native_1ms_ch_",
  "Tracking_1C.trace_scenario_id":scenario,"Tracking_1C.trace_raw_sample_offset":int(skip_s*fs),
  "Observables.dump":"false"}
 cfg.write_text(render_receiver_config(base_config.read_text(),receiver_configurations()[config_name],values))
 command=[str(receiver),f"--config_file={cfg}","--keyboard=false"]
 started=datetime.now(timezone.utc).isoformat();t=time.monotonic()
 with (out/"receiver.log").open("wb") as log:
  if terminal_drain_repair:
   expected_end_byte=int((skip_s+duration_s)*fs*4)
   returncode,termination=_supervised_run(command,cwd=out,log=log,raw=raw,
    expected_end_byte=expected_end_byte)
  else:
   process=subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=False)
   returncode=process.returncode;termination={"status":"NOT_APPLIED","exit_cause":"natural_eos",
    "exit_code":returncode,"sigint_sent":False,"sigterm_sent":False,"sigkill_sent":False}
 elapsed=time.monotonic()-t;ended=datetime.now(timezone.utc).isoformat()
 dumps=sorted(out.glob("trace_native_1ms_ch_*.bin"))
 manifest={"schema":"gnss-doppler-lab.crid-replay.v1","scenario":scenario,"config":config_name,
  "command":command,"start":started,"end":ended,"elapsed_s":elapsed,"exit_code":returncode,
  "peak_rss_kib":resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
  "receiver":{"path":str(receiver),"sha256":sha256_file(receiver)},
  "raw":{"path":str(raw),"sha256":raw_sha256,"sample_rate_hz":fs,
   "sample_start":int(skip_s*fs),"sample_end":int((skip_s+duration_s)*fs)},
  "config":{"path":str(cfg),"sha256":sha256_file(cfg)},"termination":termination,
  "dumps":[{"path":str(p),"size":p.stat().st_size,"sha256":sha256_file(p)} for p in dumps]}
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 if returncode or not dumps or (terminal_drain_repair and termination["status"]!="PASS"):
  raise RuntimeError(f"replay failed {scenario}/{config_name}: rc={returncode}, termination={termination['status']}")
 return manifest
