#!/usr/bin/env python3
"""Fixed-schedule raw CAF Stage-0 campaign; attacks never select the fit."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from gnss_doppler_lab.acaf_nf_stage0 import *
RAW=Path("/home/ubuntu/unraid_hdd/texbat/raw"); NAMES=["cleanStatic","ds1","ds2","ds3","ds4","ds7","ds8"]; CAND={"ds1":125,"ds2":110.1,"ds3":118.9,"ds4":113.8,"ds7":110,"ds8":110}
def put(p,x):p.write_text(strict_manifest(x)+"\n")
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",default="artifacts/acaf_nf_stage0_static");a=ap.parse_args();o=Path(a.output);o.mkdir(parents=True,exist_ok=True)
 sched={"fixed_before_attack_evaluation":True,"native_fs_hz":25000000,"coherent_ms":1,"caf_grid":{"code_chips":CAF_CODE.tolist(),"doppler_hz":CAF_DOPPLER.tolist()},"epochs":{"cleanStatic":[30,180,330,420],**{n:[10,60,CAND[n]-2,CAND[n]+2] for n in NAMES if n!="cleanStatic"}},"candidate_times_s":CAND,"center_policy":"only matching raw-hash proven historic tracker center; unavailable center=0, INCONCLUSIVE"};put(o/"schedule.json",sched)
 src={};rows=[]
 for n in NAMES:
  p=RAW/(n+".bin");src[n]={"path":str(p),"sha256":sha256_file(p),"bytes":p.stat().st_size,"format":"25Msps interleaved signed int16 I/Q","lineage":"hash calculated; no historic CAF/taps read","historic_center":None}
  for sec in sched["epochs"][n]:
   try:
    z=caf_surface(raw_epoch(p,sec),3,25000000,0,0);r=feature(z);r.update(scenario=n,epoch_s=sec,center_status="UNVERIFIED_CENTER");rows.append(r)
   except Exception as e:rows.append({"scenario":n,"epoch_s":sec,"error":str(e),"center_status":"UNVERIFIED_CENTER"})
 base=two_source_control(3,6,1000000,7);ctl=[]
 for g in [.5,.8,1.2,2]:
  for ph in [0,np.pi/2]:
   for noise in [0,.03]:ctl.append({"gain":g,"phase_rad":ph,"awgn_sigma":noise,"caf":feature(caf_surface(augment(base,g,ph,noise,11),3,1000000,0,0))})
 ctl += [{"representation":rep,"K":k,"status":"synthetic direct C/A control"} for rep in ["fixed_EPL","fixed9","rawpower","dense"] for k in select_k([3,5,9,16])]
 split=chronological_clean_split(sched["epochs"]["cleanStatic"]);met={"two_source_diagnostics":feature(caf_surface(base,3,1000000,0,0)),"gain_phase_awgn":True,"query_orders":["fixed","random","shuffled seed=11"],"clean_chronological_roles":split,"no_attack_fit":attack_free_fit(split,{"fit":"cleanStatic"}),"bootstrap_10s":{"feasible":False,"reason":"only lightweight sampled ms epochs"},"ds7_ds8_overlap":ds78_overlap_status(None,None),"B0":"NOT_DIRECTLY_COMPARABLE: native B0 was not rerun; no CSV reused"}
 put(o/"raw_sources.json",src);put(o/"caf_features.json",rows);put(o/"controls.json",ctl);put(o/"metrics.json",met)
 import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
 vals=[r.get("peak_to_median",float("nan")) for r in rows];plt.figure(figsize=(9,3));plt.plot(vals,"o");plt.ylabel("CAF peak/median");plt.tight_layout();plt.savefig(o/"caf_summary.png",dpi=130);plt.close()
 checks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in o.iterdir() if p.is_file()};man={"schema_version":1,"study":"ACAF-NF Stage-0 static physical feasibility","status":"INCONCLUSIVE","inconclusive_reasons":["fresh raw-hash-proven tracker code/doppler centers unavailable","DS7/8 exact-time nonoverlap provenance unavailable"],"raw_caf_direct":True,"no_precomputed_caf_or_taps":True,"scenario_metadata":{"official_document":"not locally bundled; official PDF hash/page quote unavailable","candidate_times_s":CAND,"page_quote":None},"outputs":checks};put(o/"manifest.json",man);checks["manifest.json"]=hashlib.sha256((o/"manifest.json").read_bytes()).hexdigest();put(o/"checksums.json",checks);print(strict_manifest({"status":man["status"],"epochs":len(rows)}))
if __name__=="__main__":main()
