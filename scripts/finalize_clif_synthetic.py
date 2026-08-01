#!/usr/bin/env python3
"""Finalize/checksum the R4 artifact bundle without inventing unavailable metrics."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
REQUIRED=("config.json","synthetic_run_manifest.csv","generation_summary.json","impairment_distribution.json","training_summary.json","predictor_comparison.csv","scenario_metrics.csv","domain_gap_metrics.csv","alignment_destruction_metrics.json","test_summary.txt","README.md")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("artifacts/clif_ip_synthetic_normal_r4"));ap.add_argument("--allow-provisional",action="store_true");a=ap.parse_args();root=a.root;root.mkdir(parents=True,exist_ok=True);(root/"plots").mkdir(exist_ok=True)
 idx=pd.read_csv(root/"synthetic_run_manifest.csv");imps=[json.loads(x) for x in idx.impairments_json]
 dist={k:{"min":min(float(x[k]) for x in imps),"max":max(float(x[k]) for x in imps)} for k in imps[0] if isinstance(imps[0][k],(int,float)) and not isinstance(imps[0][k],bool)}
 (root/"impairment_distribution.json").write_text(json.dumps({"axes":dist,"attack":False,"spoofing":False},indent=2)+"\n")
 cfg={"schema":"clif-ip.synthetic-normal.r4","final_runs":60,"per_domain":{"train":24,"validation":3,"synthetic_test":3},"duration_s":120,"targets":{"SYN-OAK":"5 Msps s16le IQ/ishort","SYN-TEX":"25 Msps s16le IQ/ishort"},"threshold_fit":"per regime/domain normal validation only","real_static":{"OAKBAT":"os1--os4","TEXBAT":"DS1--DS4 primary; stable pre 30--90, exclude 90--110, post >=110"},"permutations":199,"r0":"read-only exact R3 baseline","success":"P3<P1 and Full AUC>M1/simple fusion with controlled independent clean FPR"}
 (root/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
 missing=[x for x in REQUIRED if not (root/x).is_file()]
 if missing and not a.allow_provisional:raise SystemExit(f"missing required artifacts: {missing}")
 for x in missing:
  p=root/x
  if p.suffix==".csv":p.write_text("status,na_reason\nNA,not generated; final campaign not run\n")
  elif p.suffix==".json":p.write_text(json.dumps({"status":"NA","na_reason":"not generated; final campaign not run"},indent=2)+"\n")
  else:p.write_text("NA: not generated; final campaign not run\n")
 checks={x:sha(root/x) for x in REQUIRED};commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).decode().strip() if False else subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
 (root/"checksums.json").write_text(json.dumps({"generated_utc":datetime.now(timezone.utc).isoformat(),"source_commit":commit,"files":checks},indent=2)+"\n")
if __name__=="__main__":main()
