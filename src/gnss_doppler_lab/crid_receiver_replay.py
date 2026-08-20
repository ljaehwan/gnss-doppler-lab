"""GNSS-SDR replay orchestration for CRID."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, os, resource, subprocess, time
from pathlib import Path
from typing import Mapping
from .crid import receiver_configurations, render_receiver_config

def sha256_file(path:Path)->str:
 d=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b""):d.update(chunk)
 return d.hexdigest()

def run_replay(*,receiver:Path,base_config:Path,raw:Path,out:Path,scenario:str,
 config_name:str,fs:int,skip_s:float,duration_s:float,raw_sha256:str|None=None)->dict:
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
  process=subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=False)
 elapsed=time.monotonic()-t;ended=datetime.now(timezone.utc).isoformat()
 dumps=sorted(out.glob("trace_native_1ms_ch_*.bin"))
 manifest={"schema":"gnss-doppler-lab.crid-replay.v1","scenario":scenario,"config":config_name,
  "command":command,"start":started,"end":ended,"elapsed_s":elapsed,"exit_code":process.returncode,
  "peak_rss_kib":resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
  "receiver":{"path":str(receiver),"sha256":sha256_file(receiver)},
  "raw":{"path":str(raw),"sha256":raw_sha256,"sample_rate_hz":fs,
   "sample_start":int(skip_s*fs),"sample_end":int((skip_s+duration_s)*fs)},
  "config":{"path":str(cfg),"sha256":sha256_file(cfg)},
  "dumps":[{"path":str(p),"size":p.stat().st_size,"sha256":sha256_file(p)} for p in dumps]}
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 if process.returncode or not dumps:raise RuntimeError(f"replay failed {scenario}/{config_name}: rc={process.returncode}")
 return manifest
