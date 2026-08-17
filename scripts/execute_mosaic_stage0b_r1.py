#!/usr/bin/env python3
"""Freeze and execute the full-prefix MOSAIC Stage-0B R1 campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from gnss_doppler_lab.mosaic_stage0b_r1_executor import (canonical_sha, compare_identity,
    generate_injected_prefix, load_csv_gz, run_receiver, score_trace_prn)
from gnss_doppler_lab.mosaic_stage0b_r1_execution_metrics import (BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED, collapsed, median_abs_error, paired_bootstrap_ci, sign_accuracy,
    spearman_abs, strong_resolvable)
from gnss_doppler_lab.trace_native_1ms import sha256_file

BASE="3db0e12976b6ff98452096e921cf298be459d0e8"
BRANCH="research/mosaic-stage0b-r1-execution"
ART=ROOT/"artifacts/mosaic_stage0b_r1_execution"
PREREG=ROOT/"artifacts/mosaic_stage0b_r1_receiver_in_loop"
EXTERNAL=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mosaic-stage0b-r1-execution")
RECEIVER=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2c-terminal-drain-repair/receiver-build/src/main/gnss-sdr")
MCTD=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mctd-stage0-static/dumps/phase_a")
SPECS={
 "OAKBAT.cleanStatic":{"slug":"oakbat_cleanstatic","fs":5_000_000,"interval":[150275296,210202273],"samples":450_000_000,
  "raw":Path("/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/cleanStatic_gps.bin"),"prns":[10,11,21,24,27]},
 "TEXBAT.cleanStatic":{"slug":"texbat_cleanstatic","fs":25_000_000,"interval":[817815304,1117517038],"samples":2_250_000_000,
  "raw":Path("/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/cleanStatic.bin"),"prns":[3,13,16,19,30]},
}

def dump(path:Path,value): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def command(*args): return subprocess.run(args,check=True,text=True,stdout=subprocess.PIPE).stdout.strip()
def file_sha(path):
 d=hashlib.sha256()
 with path.open("rb") as s:
  for b in iter(lambda:s.read(8*1024*1024),b""): d.update(b)
 return d.hexdigest()

def write_manifest():
 files=[]
 for p in sorted(ART.rglob("*")):
  if p.is_file() and p.name!="artifact_manifest_sha256.json": files.append({"path":str(p.relative_to(ART)),"size_bytes":p.stat().st_size,"sha256":file_sha(p)})
 dump(ART/"artifact_manifest_sha256.json",{"schema":"gnss-doppler-lab.artifact-manifest-sha256.v1","files":files})

def code_hashes():
 names=["src/gnss_doppler_lab/mosaic_stage0b_r1_executor.py","src/gnss_doppler_lab/mosaic_stage0b_r1_execution_metrics.py",
  "scripts/execute_mosaic_stage0b_r1.py","scripts/verify_mosaic_stage0b_r1_results.py","tests/test_mosaic_stage0b_r1_execution.py"]
 return {name:file_sha(ROOT/name) for name in names}

def freeze():
 if command("git","rev-parse","HEAD")!=BASE: raise SystemExit("freeze must start at exact R1 preregistration base")
 if ART.exists(): raise FileExistsError(ART)
 design=json.loads((PREREG/"frozen_injection_design.json").read_text())
 if len(design)!=72 or canonical_sha(design)!="b1a06556f7cd67738274c132f80b0581b20914d971f72f4e4ab0b5efc9a7facf": raise ValueError("design mismatch")
 ART.mkdir(parents=True)
 freeze_value={"status":"PRE_EXECUTION_FREEZE","results_viewed":False,"cases_executed":0,"required_base_commit":BASE,
  "frozen_design_canonical_sha256":canonical_sha(design),"case_count":72,"executor_code_sha256":code_hashes(),
  "worker_count":1,"chunk_samples":250000,"full_prefix_replay":True,"source_prefix_byte_identity_required":True,
  "config_change_allowlist":["SignalSource.filename"],"alignment_tolerance":{"integer_fields":"exact","float_rtol":1e-6,"float_atol":1e-6},
  "bootstrap":{"unit":"paired_case_within_dataset","seed":BOOTSTRAP_SEED,"resamples":BOOTSTRAP_RESAMPLES,"ci":.95},
  "quantified_gates":{"bic_control_ci_lower_gt":0,"target_specificity_ci_lower_gt":0,"rms_spearman_abs_lt":.5,
   "clipping_ratio_median_max":1e-4,"clipping_ratio_case_max":1e-3,"permutation_absolute_tolerance":1e-9,
   "observability_definition":"finite target delta_BIC > 0","dataset_pooling":False},
  "control_policy":{"collapsed":"frozen delta_tau=0 and delta_f=0 cases","gain":"identity taps scaled to injected tap RMS",
   "awgn":"deterministic complex Gaussian at injected residual RMS; seed=20260818+PRN","no_additional_receiver_cases":True},
  "single_gate_before_four_prn":True,"result_dependent_changes_prohibited":True}
 dump(ART/"executor_freeze.json",freeze_value)
 dump(ART/"config.json",{"schema":"gnss-doppler-lab.mosaic-stage0b-r1-execution.v1","external_result_root":str(EXTERNAL),
  "worker_count":1,"attack_data_accessed":False,"source_raw_modified":False,"temporary_injected_iq_retained":False})
 dump(ART/"source_commit.json",{"required_base_branch":"origin/research/mosaic-stage0b-r1-receiver-in-loop-injection",
  "required_base_commit":BASE,"observed_generation_commit":BASE,"work_branch":BRANCH,"base_match":True})
 dump(ART/"source_binding.json",{"preregistration_manifest_sha256":file_sha(PREREG/"artifact_manifest_sha256.json"),
  "receiver_binary_sha256":file_sha(RECEIVER),"datasets":{d:{"raw_path":str(s["raw"]),"raw_size_bytes":s["raw"].stat().st_size,
  "sample_rate_hz":s["fs"],"authorized_interval":s["interval"],"full_prefix_samples":s["samples"],
  "base_config":str(MCTD/s["slug"]/"slow/rep1/receiver.conf"),"reference_trace_dir":str(MCTD/s["slug"]/"slow/rep1")} for d,s in SPECS.items()}})
 (ART/"README.md").write_text("# MOSAIC Stage-0B R1 execution\n\nPRE_EXECUTION_FREEZE: executor, controls, statistics, gates, and full-prefix replay policy are frozen before any result generation.\n")
 write_manifest()
 print("PRE_EXECUTION_FREEZE_READY")

def verify_freeze(freeze_sha):
 if command("git","rev-parse","HEAD")!=freeze_sha: raise ValueError("execution HEAD differs from pushed freeze SHA")
 if command("git","status","--porcelain"): raise ValueError("execution worktree is not clean")
 value=json.loads((ART/"executor_freeze.json").read_text())
 if value["executor_code_sha256"]!=code_hashes() or value["results_viewed"]: raise ValueError("executor freeze mismatch")

def copy_identity(source:Path,target:Path,samples:int):
 target.parent.mkdir(parents=True,exist_ok=False); ds=hashlib.sha256(); do=hashlib.sha256(); remaining=samples*4
 with source.open("rb") as a,target.open("xb") as b:
  while remaining:
   payload=a.read(min(8*1024*1024,remaining))
   if not payload: raise EOFError("identity source truncated")
   ds.update(payload);do.update(payload);b.write(payload);remaining-=len(payload)
 return {"source_prefix_sha256":ds.hexdigest(),"output_sha256":do.hexdigest(),"size_bytes":target.stat().st_size,
  "sample_count":samples,"byte_identity":ds.hexdigest()==do.hexdigest(),"clipped_sample_count":0}

def engineering(freeze_sha):
 verify_freeze(freeze_sha); EXTERNAL.mkdir(parents=True,exist_ok=False)
 results={"freeze_sha":freeze_sha,"datasets":{},"status":"PASS"}
 for dataset,spec in SPECS.items():
  root=EXTERNAL/"engineering"/spec["slug"]; raw=root/"raw"/"identity.bin"
  identity=copy_identity(spec["raw"],raw,spec["samples"])
  replay=run_receiver(RECEIVER,MCTD/spec["slug"]/"slow/rep1/receiver.conf",raw,root/"receiver")
  alignment=compare_identity(MCTD/spec["slug"]/"slow/rep1",root/"receiver",spec["prns"])
  receipt={"identity":identity,"replay":replay,"alignment":alignment,"raw_deleted_after_receipt":True}
  dump(root/"receipt.json",receipt);raw.unlink()
  ok=identity["byte_identity"] and replay["status"]=="PASS" and alignment["status"]=="PASS"
  results["datasets"][dataset]={**receipt,"status":"PASS" if ok else "FAIL"};results["status"]="PASS" if results["status"]=="PASS" and ok else "FAIL"
 dump(EXTERNAL/"engineering_identity_gate.json",results);print(json.dumps({"engineering":results["status"]}))

def execute_family(freeze_sha,family):
 verify_freeze(freeze_sha)
 engineering_result=json.loads((EXTERNAL/"engineering_identity_gate.json").read_text())
 if engineering_result["status"]!="PASS": raise SystemExit("engineering gate failed")
 design=json.loads((PREREG/"frozen_injection_design.json").read_text()); assignments=json.loads((PREREG/"case_target_assignment.json").read_text())["assignments"]
 assignment={r["case_id"]:r for r in assignments};mapping=load_csv_gz(ROOT/"artifacts/mosaic_stage0b_r0c_boundary_phase_extrapolation/corrected_bit_mapping.csv.gz")
 wanted="single_prn" if family=="single" else "four_prn_diagnostic_after_single_prn_pass_only"
 cases=[c for c in design if c["mode"]==wanted]
 if family=="four" and not json.loads((EXTERNAL/"single_gate.json").read_text()).get("pass"): raise SystemExit("single gate did not pass")
 for index,case in enumerate(cases,1):
  dataset=case["dataset"];spec=SPECS[dataset];root=EXTERNAL/"cases"/case["case_id"]
  if (root/"case_result.json").exists(): continue
  targets=assignment[case["case_id"]]["target_prns"]
  injected=generate_injected_prefix(spec["raw"],root/"raw"/"injected.bin",total_samples=spec["samples"],interval=tuple(spec["interval"]),fs=spec["fs"],
   targets=targets,rho_db=case["rho_db"],delay_chips=case["delta_tau_chips"],delta_f_hz=case["delta_f_hz"],phase0_rad=case["delta_phi_rad"],
   trace_dir=MCTD/spec["slug"]/"slow/rep1",mapping_rows=[r for r in mapping if r["dataset"]==dataset])
  replay=run_receiver(RECEIVER,MCTD/spec["slug"]/"slow/rep1/receiver.conf",Path(injected["path"]),root/"receiver")
  scores=[score_trace_prn(MCTD/spec["slug"]/"slow/rep1",root/"receiver",prn,spec["interval"][0],spec["fs"]) for prn in spec["prns"]] if replay["status"]=="PASS" else []
  result={"freeze_sha":freeze_sha,"case":case,"assignment":assignment[case["case_id"]],"injected_iq":injected,"receiver":replay,"scores":scores,
   "raw_deleted_after_receipt":True,"status":"PASS" if replay["status"]=="PASS" else "FAIL"}
  dump(root/"case_result.json",result);Path(injected["path"]).unlink();print(f"{family} {index}/{len(cases)} {case['case_id']} {result['status']}",flush=True)

def single_gate(freeze_sha):
 verify_freeze(freeze_sha);results=[json.loads(p.read_text()) for p in sorted((EXTERNAL/"cases").glob("*/case_result.json")) if ".single." in p.parent.name]
 out={"freeze_sha":freeze_sha,"datasets":{},"pass":True}
 for dataset in SPECS:
  cases=[r for r in results if r["case"]["dataset"]==dataset and r["status"]=="PASS"]
  strong=[r for r in cases if strong_resolvable(r["case"]["rho_db"],r["case"]["delta_tau_chips"],r["case"]["delta_f_hz"])]
  rows=[]
  for r in strong:
   target=r["assignment"]["target_prns"][0];score=next(x for x in r["scores"] if x["prn"]==target);non=[x["delta_bic"] for x in r["scores"] if x["prn"]!=target]
   scer=r["injected_iq"]["realized_scer_db"][str(target)]-r["case"]["rho_db"]
   rows.append({"delay":r["case"]["delta_tau_chips"],"rdelay":score["recovered_delay_chips"],"doppler":r["case"]["delta_f_hz"],"rdoppler":score["recovered_doppler_hz"],
    "dbic":score["delta_bic"],"specificity":score["delta_bic"]-float(np.median(non)),"scer_error":scer,"clip":r["injected_iq"]["clipping_ratio"],"rms":r["injected_iq"]["output_interval_rms"]})
  delay_sign=sign_accuracy(np.array([x["delay"] for x in rows]),np.array([x["rdelay"] for x in rows]));delay_mae=median_abs_error(np.array([x["delay"] for x in rows]),np.array([x["rdelay"] for x in rows]))
  dop_sign=sign_accuracy(np.array([x["doppler"] for x in rows]),np.array([x["rdoppler"] for x in rows]));dop_mae=median_abs_error(np.array([x["doppler"] for x in rows]),np.array([x["rdoppler"] for x in rows]))
  _,spec_lo,_=paired_bootstrap_ci(np.array([x["specificity"] for x in rows]));observability=float(np.mean([x["dbic"]>0 for x in rows])) if rows else 0
  passed=bool(rows and observability>=.75 and delay_sign is not None and delay_sign>=.8 and delay_mae<=.05 and dop_sign is not None and dop_sign>=.8 and dop_mae<=10 and
   np.median(np.abs([x["scer_error"] for x in rows]))<=1 and spec_lo>0 and np.median([x["clip"] for x in rows])<=1e-4 and max(x["clip"] for x in rows)<=1e-3)
  out["datasets"][dataset]={"strong_cases":len(rows),"observability":observability,"delay_sign_accuracy":delay_sign,"delay_mae":delay_mae,"doppler_sign_accuracy":dop_sign,
   "doppler_mae":dop_mae,"specificity_ci_lower":spec_lo,"pass":passed};out["pass"] &= passed
 dump(EXTERNAL/"single_gate.json",out);print(json.dumps(out,indent=2))

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest="cmd",required=True);sub.add_parser("freeze")
 for name in ("engineering","single","single-gate","four"):
  q=sub.add_parser(name);q.add_argument("--freeze-sha",required=True)
 a=p.parse_args()
 if a.cmd=="freeze":freeze()
 elif a.cmd=="engineering":engineering(a.freeze_sha)
 elif a.cmd=="single":execute_family(a.freeze_sha,"single")
 elif a.cmd=="single-gate":single_gate(a.freeze_sha)
 elif a.cmd=="four":execute_family(a.freeze_sha,"four")

if __name__=="__main__":main()
