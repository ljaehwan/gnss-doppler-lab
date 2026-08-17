#!/usr/bin/env python3
"""Verifier and deterministic compact-artifact finalizer for R1 execution."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts/mosaic_stage0b_r1_execution"
EXTERNAL=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mosaic-stage0b-r1-execution")
REQUIRED=("README.md","executor_freeze.json","config.json","source_commit.json","source_binding.json")

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value): Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def write_csv(path,rows,fields):
 with Path(path).open("w",newline="",encoding="utf-8") as s:
  w=csv.DictWriter(s,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
def manifest():
 files=[]
 for p in sorted(ART.rglob("*")):
  if p.is_file() and p.name!="artifact_manifest_sha256.json":files.append({"path":str(p.relative_to(ART)),"size_bytes":p.stat().st_size,"sha256":sha(p)})
 dump(ART/"artifact_manifest_sha256.json",{"schema":"gnss-doppler-lab.artifact-manifest-sha256.v1","files":files})

def verify_manifest():
 value=json.loads((ART/"artifact_manifest_sha256.json").read_text())
 for x in value["files"]:
  p=ART/x["path"]
  if not p.is_file() or p.stat().st_size!=x["size_bytes"] or sha(p)!=x["sha256"]:raise ValueError(f"checksum mismatch {x['path']}")
 return len(value["files"])

def verify_freeze():
 for name in REQUIRED:
  if not (ART/name).is_file():raise ValueError(f"missing freeze file {name}")
 f=json.loads((ART/"executor_freeze.json").read_text())
 if f["status"]!="PRE_EXECUTION_FREEZE" or f["results_viewed"] or f["cases_executed"]!=0 or f["case_count"]!=72:raise ValueError("invalid freeze")
 return f

def finalize(freeze_sha):
 freeze=verify_freeze();engineering=json.loads((EXTERNAL/"engineering_identity_gate.json").read_text())
 case_files=sorted((EXTERNAL/"cases").glob("*/case_result.json")) if (EXTERNAL/"cases").exists() else []
 cases=[json.loads(p.read_text()) for p in case_files]
 case_rows=[];replay_rows=[];single=[];four=[];target_rows=[];recovery=[];controls=[];failures=[];epochs=[]
 for r in cases:
  c=r["case"];a=r["assignment"];inj=r["injected_iq"];targets=set(a["target_prns"])
  case_rows.append({"case_id":c["case_id"],"dataset":c["dataset"],"mode":c["mode"],"status":r["status"],"targets":";".join(map(str,sorted(targets))),
   "rho_db":c["rho_db"],"delay_chips":c["delta_tau_chips"],"doppler_hz":c["delta_f_hz"],"phase_rad":c["delta_phi_rad"],"iq_sha256":inj["sha256"],"iq_size_bytes":inj["size_bytes"],"clipping_ratio":inj["clipping_ratio"]})
  replay_rows.append({"case_id":c["case_id"],"exit_code":r["receiver"]["exit_code"],"trace_count":r["receiver"]["trace_count"],"runtime_seconds":r["receiver"]["runtime_seconds"],"status":r["receiver"]["status"]})
  if r["status"]!="PASS":failures.append({"case_id":c["case_id"],"stage":"receiver_replay","label":"REPLAY_FAIL"})
  for s in r["scores"]:
   row={"case_id":c["case_id"],"dataset":c["dataset"],"prn":s["prn"],"is_target":s["prn"] in targets,"delta_bic":s["delta_bic"]}
   target_rows.append(row);recovery.append({**row,"requested_delay_chips":c["delta_tau_chips"],"recovered_delay_chips":s["recovered_delay_chips"],"requested_doppler_hz":c["delta_f_hz"],"recovered_doppler_hz":s["recovered_doppler_hz"]})
   epochs.append({**row,"epochs":s["epochs"],"rss_h0":s["rss_h0"],"rss_h1":s["rss_h1"],"bic_h0":s["bic_h0"],"bic_h1":s["bic_h1"],"tap_rms":s["tap_rms"]})
  target_scores=[s for s in r["scores"] if s["prn"] in targets]
  metric={"case_id":c["case_id"],"dataset":c["dataset"],"target_count":len(targets),"targets_recovered":sum(abs(s["recovered_delay_chips"]-c["delta_tau_chips"])<=.05 and abs(s["recovered_doppler_hz"]-c["delta_f_hz"])<=10 for s in target_scores),"status":r["status"]}
  (single if c["mode"]=="single_prn" else four).append(metric)
 fields_case=["case_id","dataset","mode","status","targets","rho_db","delay_chips","doppler_hz","phase_rad","iq_sha256","iq_size_bytes","clipping_ratio"]
 write_csv(ART/"case_execution_inventory.csv",case_rows,fields_case);write_csv(ART/"receiver_replay_inventory.csv",replay_rows,["case_id","exit_code","trace_count","runtime_seconds","status"])
 write_csv(ART/"single_prn_metrics.csv",single,["case_id","dataset","target_count","targets_recovered","status"]);write_csv(ART/"four_prn_metrics.csv",four,["case_id","dataset","target_count","targets_recovered","status"])
 write_csv(ART/"target_nontarget_metrics.csv",target_rows,["case_id","dataset","prn","is_target","delta_bic"])
 write_csv(ART/"recovery_metrics.csv",recovery,["case_id","dataset","prn","is_target","delta_bic","requested_delay_chips","recovered_delay_chips","requested_doppler_hz","recovered_doppler_hz"])
 write_csv(ART/"control_metrics.csv",controls,["case_id","dataset","control","delta_bic"]);write_csv(ART/"bootstrap_intervals.csv",[],["dataset","metric","mean","ci_lower","ci_upper","pass"])
 write_csv(ART/"failure_inventory.csv",failures,["case_id","stage","label"])
 with gzip.open(ART/"per_epoch_scores.csv.gz","wt",newline="",encoding="utf-8") as s:
  fields=["case_id","dataset","prn","is_target","delta_bic","epochs","rss_h0","rss_h1","bic_h0","bic_h1","tap_rms"];w=csv.DictWriter(s,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(epochs)
 dump(ART/"engineering_identity_gate.json",engineering)
 if engineering["status"]!="PASS":verdict="INCONCLUSIVE_ENGINEERING_ALIGNMENT"
 elif len([r for r in cases if r["case"]["mode"]=="single_prn"])<56:verdict="INCONCLUSIVE_INSUFFICIENT_SUPPORT"
 else:
  sg=json.loads((EXTERNAL/"single_gate.json").read_text())
  if not sg["pass"]:verdict="NO_GO_MOSAIC_SINGLE_PRN_PHYSICS"
  elif len([r for r in cases if r["case"]["mode"]!="single_prn"])<16:verdict="INCONCLUSIVE_INSUFFICIENT_SUPPORT"
  else:verdict="INCONCLUSIVE_PREREG_GATE_UNDERSPECIFIED"
 dump(ART/"runtime_resource_summary.json",{"external_root":str(EXTERNAL),"executed_cases":len(cases),"failed_cases":len(failures),"temporary_iq_retained":False})
 dump(ART/"final_verdict.json",{"verdict":verdict,"freeze_sha":freeze_sha,"executed_cases":len(cases),"failed_cases":len(failures),"result_dependent_gate_changes":False})
 (ART/"plots").mkdir(exist_ok=True)
 for name in ("requested_vs_recovered_delay","requested_vs_recovered_doppler","requested_vs_measured_scer","delta_bic_grid","target_vs_nontarget","controls","oak_vs_tex","clipping_rms_score","four_prn_heatmap"):
  (ART/"plots"/f"{name}.txt").write_text("Plot unavailable because execution stopped before sufficient validated results.\n")
 manifest();print(verdict)

def main():
 import argparse;p=argparse.ArgumentParser();p.add_argument("--finalize",action="store_true");p.add_argument("--freeze-sha")
 a=p.parse_args()
 if a.finalize:finalize(a.freeze_sha)
 else:
  f=verify_freeze();count=verify_manifest() if (ART/"artifact_manifest_sha256.json").exists() else 0
  if (ART/"final_verdict.json").exists():print(f"PASS: R1 results artifact {json.loads((ART/'final_verdict.json').read_text())['verdict']}; {count} checksums")
  else:print(f"PASS: PRE_EXECUTION_FREEZE; 0 cases executed; {count} checksums")
if __name__=="__main__":main()
