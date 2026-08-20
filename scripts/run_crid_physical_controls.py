#!/usr/bin/env python3
"""Materialize frozen clean raw-IQ controls and replay four receivers."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
from gnss_doppler_lab.crid_physical_controls import transform_ishort
from gnss_doppler_lab.crid_receiver_replay import run_replay
from gnss_doppler_lab.crid import CONFIG_ORDER
from run_crid_stage0 import BASE_CONFIG,DATA,RECEIVER,SSD

def controls(fs,sigma):
 chip_samples=fs/1_023_000
 neg={"identical":("byte_identical",{}),"gain_0p5":("gain",{"gain":.5}),"phase_1p1":("phase",{"phase_rad":1.1}),
  "nav_minus":("nav_sign",{}),"awgn_0p5":("awgn",{"noise_sigma":.5*sigma}),"awgn_1":("awgn",{"noise_sigma":sigma}),
  "awgn_2":("awgn",{"noise_sigma":2*sigma}),"cn0_reduction":("cn0_reduction",{"noise_sigma":2*sigma}),
  "code_ramp":("single_source_code_ramp",{}),"doppler_ramp":("doppler_ramp",{"ramp_hz_per_s":.2}),
  "clock_drift":("clock_drift",{"gain":1.}),"zero_delay_duplicate":("zero_delay_duplicate",{"duplicate_db":-3.,"phase_rad":.7})}
 pos={f"second_source_d{d}_p{p}":("duplicate",{"delay_samples":d*chip_samples,"duplicate_db":p,
  "phase_rad":float(np.random.default_rng(20260820+int(d*100)+p).uniform(-np.pi,np.pi))}) for d in (.05,.15,.30) for p in (-6,-3,0)}
 return neg|pos
def main():
 p=argparse.ArgumentParser();p.add_argument("--dataset",choices=("oak_clean","tex_clean"),required=True);p.add_argument("--duration",type=float,default=45.);p.add_argument("--only");a=p.parse_args();spec=DATA[a.dataset];raw=Path(spec["raw"]);fs=spec["fs"]
 probe=np.fromfile(raw,dtype="<i2",count=2*fs);sigma=float(np.std(probe));registry=controls(fs,sigma)
 if a.only:registry={a.only:registry[a.only]}
 manifests=[]
 for name,(kind,kw) in registry.items():
  target=SSD/"controls"/a.dataset/name/"control.bin";meta=transform_ishort(raw,target,kind,sample_rate_hz=fs,max_samples=int(a.duration*fs),**kw)
  for c in CONFIG_ORDER:
   out=SSD/"control_replays"/a.dataset/name/c
   manifests.append(run_replay(receiver=RECEIVER,base_config=BASE_CONFIG[spec["domain"]],raw=target,out=out,scenario=f"{a.dataset}.{name}",config_name=c,fs=fs,skip_s=0.,duration_s=a.duration,raw_sha256=meta["sha256"]))
  (target.parent/"control_manifest.json").write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"dataset":a.dataset,"controls":list(registry),"replay_count":len(manifests)},indent=2))
if __name__=="__main__":main()
