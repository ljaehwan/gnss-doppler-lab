#!/usr/bin/env python3
"""CleanStatic-only scale gate; never reads or scores attack scenarios."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, resource, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.r2c_stage0_fix import ComplexWhitener,TemplateProvider,compile_profile_plan,joint_profile_glrt
from supervise_r2c_stage0 import hardware_key

def sha(path):
    h=hashlib.sha256();h.update(path.read_bytes());return h.hexdigest()
def timed(fn,repeats=3):
    values=[]
    for _ in range(repeats):start=time.perf_counter();fn();values.append(time.perf_counter()-start)
    return min(values)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--clean",type=Path,required=True,help="canonical cleanStatic selected NPZ only")
    ap.add_argument("--config",type=Path,default=ROOT/"configs/r2c_gnss_stage0_fix.json");ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();config=json.loads(a.config.read_text());
    with np.load(a.clean,allow_pickle=False) as z:
        iq=np.asarray(z["complex_iq"],float); y=iq[...,0]+1j*iq[...,1];prn=np.asarray(z["prn"],int)
        bins=np.asarray(z["time_s"],float)//.5
    provider=TemplateProvider.analytic();taps=np.asarray(config["tap_offsets_chips"]);grid=np.asarray(config["delay_grid_chips"])
    w=ComplexWhitener(shrinkage=0);w.mean=np.zeros(9,complex);w.covariance=np.eye(9);w.inverse_sqrt=np.eye(9);w.pseudo_covariance=np.zeros((9,9),complex);w.diagnostics={}
    plan=compile_profile_plan(provider,taps,grid,w,row_chunk=config["epoch_policy"]["chunk_rows"])
    sizes=[1,2,3,4,17,min(4096,len(y))]; rows=[]
    for size in sizes:
        obs={1:y[:size]};scalar=timed(lambda:joint_profile_glrt(obs,{},provider,taps,grid,hypothesis="H1-independent",whitener=w,scalar_reference=True),1)
        vector=timed(lambda:joint_profile_glrt(obs,{},provider,taps,grid,hypothesis="H1-independent",whitener=w,profile_plan=plan))
        rows.append({"epochs":size,"scalar_s":scalar,"vector_s":vector,"speedup":scalar/max(vector,1e-12)})
    large=rows[-1];epochs_s=large["epochs"]/large["vector_s"]
    clean_projection=len(y)/epochs_s; all_rows=sum(config.get("input_contract",{}).get("known_row_counts",[len(y)*7]))
    all_projection=all_rows/epochs_s; upper=all_projection*1.25
    median=float(np.median([x["speedup"] for x in rows]));worst_small=min(x["speedup"] for x in rows if x["epochs"]<=4)
    peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    failures=[]
    for ok,name in ((median>=20,"median inner-kernel speedup >=20x"),(worst_small>=10,"E=1..4 speedup >=10x"),
                    (clean_projection<=900,"cleanStatic projection <=15 min"),(all_projection<=5400,"all-scenario projection <=90 min"),
                    (upper<=7200,"1.25x upper bound <=120 min"),(peak<=2*1024**3,"peak RSS <=2 GiB")):
        if not ok:failures.append(name)
    source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    report={"status":"GO" if not failures else "NO_GO","failures":failures,"source_sha":source,"config_hash":sha(a.config),
      "hardware":hardware_key(),"versions":{"python":platform.python_version(),"numpy":np.__version__},"clean":{"rows":len(y),"bins":len(np.unique(bins)),"prns":len(np.unique(prn)),"epochs_by_prn":{str(p):int(np.sum(prn==p)) for p in np.unique(prn)}},
      "timings":rows,"median_speedup":median,"worst_small_speedup":worst_small,"epochs_s":epochs_s,"clean_projection_s":clean_projection,
      "all_projection_s":all_projection,"upper_bound_s":upper,"peak_rss_bytes":peak,"near_tie_fallback_count":None,"near_tie_fallback_rate":None}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    if failures:
        print("benchmark NO_GO: "+"; ".join(failures),file=sys.stderr);return 1
    print(json.dumps({"status":"GO","manifest":str(a.output)}));return 0
if __name__=="__main__":raise SystemExit(main())
