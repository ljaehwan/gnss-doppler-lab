#!/usr/bin/env python3
"""Produce the deterministic fail-closed Stage-1 feasibility artifact."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.acaf_nf_stage1_static_feasibility import FROZEN_CONFIG

DEFAULT_OUTPUT=ROOT/"artifacts/acaf_nf_stage1_static_feasibility"
R14_ARTIFACT=ROOT/"artifacts/acaf_nf_stage0_static_r14_doppler_validation"
RAW_ROOT=Path("/home/ubuntu/unraid_hdd/texbat/raw")
TRACKERS={
 "cleanStatic":Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9/raw"),
 "ds3":Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver/ds3-complex9/raw"),
 "ds4":Path("/home/ubuntu/projects/gnss-doppler-lab/artifacts/ai_morph_gru_window_ablation_ds4_20260723/receiver_shared/ds4/receiver/texbat-ds4-method-a-9tap-external-validation/raw"),
 "ds7":Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/ds7-sealed-input/receiver/ds7-complex9/raw"),
 "ds8":Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/cmte-a2-ds8-complex-d67f813/receiver/raw"),
}
HASHES={"cleanStatic":"dd295ab46616bfe9634d1c37479520e720ebc54bcb64adf0a247315a541fb9b9","ds3":"e37e11b060bc2c675d4a60024f8b4a53e95e7cd1d304bea80cd903856075a30d","ds4":"1fff2b048a00732686bb1d77a13941da81c9fac648ca3695a9028f4ee3485285","ds7":"d5fb1430d476f68930f3bb0290b80b649f08012eb6b6981d112493528813400e","ds8":"1614d8de6fc8ebc3429def6e9505050c08a3ee8da69c11ecc27a98305f735d78"}
TIMELINES={"ds3":{"onset_s":118.9,"pull_off_s":195.0},"ds4":{"onset_s":113.8,"pull_off_s":225.0,"raw_end_approx_s":128.22},"ds7":{"injection_s":110.,"transition":[110.,130.],"held":[130.,150.],"time_push_start_s":150.},"ds8":{"injection_s":110.,"transition":[110.,130.],"held":[130.,150.],"time_push_start_s":150.}}
EXPECTED_PHASE={"ds3":{"pre":[148699,11],"onset_to_pulloff":[39762,6],"post_pulloff":[5200,3]},"ds4":{"pre":[107471,11],"post_onset":[18,1]},"ds7":{"pre":[100520,11],"post_onset":[0,0]},"ds8":{"pre":[102940,11],"110_to_150":[0,0],"ge_150":[288,1]}}
SCIENCE_CSV={"scenario_metrics.csv":["status","reason","scenario"],"phase_metrics.csv":["status","reason","scenario","phase"],"per_window_scores.csv":["status","reason","scenario","time_s","score"],"secondary_component_metrics.csv":["status","reason","component"],"baseline_metrics.csv":["status","reason","baseline"],"control_metrics.csv":["status","reason","control"]}

def digest(path, chunk=8*1024*1024):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  while block:=f.read(chunk):h.update(block)
 return h.hexdigest()

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")

def discover_manifest(tracker):
 for parent in [tracker,*tracker.parents[:5]]:
  p=parent/"manifest.json"
  if p.is_file():return p
 return None

def manifest_contains(value, needle):
 if isinstance(value,dict):return any(manifest_contains(v,needle) for v in value.values())
 if isinstance(value,list):return any(manifest_contains(v,needle) for v in value)
 return str(value).lower()==str(needle).lower()

def _field(handle,name):
 if name not in handle:return None
 return np.asarray(handle[name]).reshape(-1)

def tracker_rows(tracker:Path):
 rows=[];inventory=[]
 for mat in sorted(tracker.glob("*.mat")):
  with h5py.File(mat,"r") as f:
   starts=_field(f,"PRN_start_sample_count");prns=_field(f,"PRN");cn0=_field(f,"CN0_SNV_dB_Hz");lock=_field(f,"carrier_lock_test")
   n=0 if starts is None else len(starts);inventory.append({"path":str(mat),"sha256":digest(mat),"rows":n,"fields":sorted(f.keys())})
   if any(x is None for x in (starts,prns,cn0,lock)):continue
   channel=mat.stem
   for i in range(1,n-1):
    rows.append({"channel":channel,"row":i,"prn":int(prns[i]),"start":int(starts[i]),
     "delta_previous":int(starts[i]-starts[i-1]),"delta_next":int(starts[i+1]-starts[i]),
     "same_prn_triple":bool(int(prns[i-1])==int(prns[i])==int(prns[i+1])),
     "min_triple_cn0":float(min(cn0[i-1:i+2])),"min_triple_lock":float(min(lock[i-1:i+2]))})
 return rows,inventory

def phase_name(scenario,t):
 if scenario=="ds3":return "pre" if t<118.9 else "onset_to_pulloff" if t<195 else "post_pulloff"
 if scenario=="ds4":return "pre" if t<113.8 else "post_onset"
 if scenario=="ds7":return "pre" if t<110 else "post_onset"
 if scenario=="ds8":return "pre" if t<110 else "110_to_150" if t<150 else "ge_150"
 raise ValueError(scenario)

def support_audit(tracker,scenario):
 rows,inventory=tracker_rows(tracker);accepted=[]
 for r in rows:
  if (r["same_prn_triple"] and 24_999<=r["delta_previous"]<=25_001
      and 24_999<=r["delta_next"]<=25_001 and r["min_triple_cn0"]>=28
      and r["min_triple_lock"]>=.85):accepted.append(r)
 counts={}
 for phase in EXPECTED_PHASE[scenario]:
  selected=[r for r in accepted if phase_name(scenario,r["start"]/25e6)==phase]
  counts[phase]=[len(selected),len({r["prn"] for r in selected})]
 return {"scenario":scenario,"counts":counts,"expected":EXPECTED_PHASE[scenario],"matches_expected":counts==EXPECTED_PHASE[scenario],"mat_inventory":inventory,"accepted_rule":"adjacent delta 24999..25001, CN0>=28, lock>=.85","twenty_ms_gaps_interpolated":False}

def foundation_gate(audits, attack_score_callback=None):
 """Return before attack scoring whenever required continuous support fails."""
 failures=[]
 for name in ("ds3","ds4","ds7","ds8"):
  if name not in audits or not audits[name].get("matches_expected"):failures.append(f"{name}_support_audit_mismatch")
 if not failures:
  for name in ("ds7","ds8"):
   post=sum(v[0] for k,v in audits[name]["counts"].items() if k!="pre")
   if post < 20:failures.append(f"{name}_no_L20_post_onset_support")
 if failures:
  return {"verdict":"FOUNDATION_INVALID","reasons":failures,"no_attack_raw_scoring_performed":True}
 if attack_score_callback is None:raise RuntimeError("support valid but scoring callback unavailable")
 return attack_score_callback()

def write_csv_header(path,fields):
 with path.open("w",newline="") as f:csv.DictWriter(f,fieldnames=fields).writeheader()

def publish(args, *, audit_function=support_audit, score_callback=None):
 output=Path(args.output)
 if output.exists():raise FileExistsError(f"refusing existing output: {output}")
 output.parent.mkdir(parents=True,exist_ok=True)
 staging=Path(tempfile.mkdtemp(prefix=output.name+".staging-",dir=output.parent))
 try:
  raw_paths={s:Path(getattr(args,"raw_"+("clean" if s=="cleanStatic" else s))) for s in TRACKERS}
  tracker_paths={s:Path(getattr(args,"tracker_"+("clean" if s=="cleanStatic" else s))) for s in TRACKERS}
  bindings={};audits={}
  for scenario in TRACKERS:
   raw=raw_paths[scenario]
   if not raw.is_file():raise FileNotFoundError(raw)
   actual=digest(raw)
   if actual!=HASHES[scenario]:raise RuntimeError(f"{scenario} raw SHA-256 mismatch")
   manifest=discover_manifest(tracker_paths[scenario])
   if manifest is None:raise RuntimeError(f"{scenario} tracker manifest unavailable")
   manifest_doc=json.loads(manifest.read_text())
   manifest_bound=manifest_contains(manifest_doc,actual)
   if scenario!="ds4" and not manifest_bound:raise RuntimeError(f"{scenario} manifest does not bind raw SHA-256")
   audits[scenario]=audit_function(tracker_paths[scenario],scenario) if scenario!="cleanStatic" else {"scenario":"cleanStatic","mat_inventory":tracker_rows(tracker_paths[scenario])[1]}
   bindings[scenario]={"raw_path":str(raw),"raw_sha256":actual,"expected_raw_sha256":HASHES[scenario],"raw_bytes_read_purpose":"full_sha256_only","tracker_path":str(tracker_paths[scenario]),"manifest_path":str(manifest),"manifest_sha256":digest(manifest),"manifest_raw_hash_binding":"PASS" if manifest_bound else "DS4_CURRENT_RAW_FULL_SHA_BINDING"}
  verdict=foundation_gate(audits,score_callback)
  if verdict["verdict"]!="FOUNDATION_INVALID":raise RuntimeError("this producer is authorized only for fail-closed artifact")
  config={"frozen":FROZEN_CONFIG.document(),"delay_grid":list(np.arange(-1,1.0001,.125)),"doppler_grid_hz":list(range(-250,251,50)),"pooling_candidates":["median","top50_mean","trimmed_mean"],"baseline_B0":"PROVISIONAL_UNAVAILABLE"}
  dump(staging/"config.json",config);dump(staging/"source_binding.json",{"sources":bindings,"tracker_support_audits":audits,"ds7_ds8_pre_attack_pairing":"paired replay diagnostic only if byte identity authenticated"})
  dump(staging/"r14_frozen_lineage.json",{"artifact":str(R14_ARTIFACT),"verifier_required":"PASS","contract":FROZEN_CONFIG.document(),"status":"AUTHENTICATED_BY_PREFLIGHT"})
  dump(staging/"scenario_timeline.json",TIMELINES)
  reason="required primary scenario continuous same-PRN 1 ms tracker/raw support is absent under frozen R1.4"
  not_eval={"status":"NOT_EVALUATED","reason":reason}
  dump(staging/"normal_model_summary.json",not_eval);dump(staging/"thresholds.json",not_eval);dump(staging/"bootstrap_results.json",not_eval)
  for name,fields in SCIENCE_CSV.items():write_csv_header(staging/name,fields)
  dump(staging/"go_no_go.json",{"verdict":"FOUNDATION_INVALID","PHYSICS_FEASIBILITY_GO":False,"physics_feasibility_status":"NOT_EVALUATED","PAPER_CANDIDATE_GO":False,"paper_candidate_status":"NOT_EVALUATED","stage2_justified":False,"reason":reason})
  dump(staging/"execution_validity.json",{"status":"FOUNDATION_INVALID","no_attack_raw_scoring_performed":True,"attack_iq_bytes_read_for_scoring":0,"raw_bytes_read_purpose":"full_sha256_only","science_csv_semantics":"header_only","plots":{"count":0,"reason":reason},"B0":"PROVISIONAL_UNAVAILABLE"})
  dump(staging/"verification_report.json",{"status":"PENDING_INDEPENDENT_VERIFICATION","producer_verdict_not_authoritative":True})
  (staging/"plots").mkdir();(staging/"test_report.txt").write_text(getattr(args,"test_report","source-phase pytest captured separately")+"\n")
  (staging/"README.md").write_text("# ACAF-NF Stage-1 static feasibility\n\n`FOUNDATION_INVALID` is a source-support finding, not a physics `NO_GO`. Attack raw IQ was full-hashed only and was never scored. Science CSVs are header-only, model/threshold/bootstrap are `NOT_EVALUATED`, and `plots/` is intentionally empty. `PHYSICS_FEASIBILITY_GO=false` and `PAPER_CANDIDATE_GO=false` mean not evaluated. Stage-2 is not justified until independently validated continuous 1 ms tracker/source binding exists. B0 is `PROVISIONAL_UNAVAILABLE` because its exact evaluator interface, support, and threshold lineage were not authenticated.\n")
  files={str(p.relative_to(staging)):digest(p) for p in sorted(staging.rglob("*")) if p.is_file() and p.name not in {"checksums.json","verification_report.json"}}
  dump(staging/"checksums.json",{"algorithm":"sha256","files":files})
  os.replace(staging,output)
  return verdict
 except Exception:
  shutil.rmtree(staging,ignore_errors=True);raise

def parser():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);p.add_argument("--execute-production",action="store_true")
 for s in TRACKERS:
  suffix="clean" if s=="cleanStatic" else s;p.add_argument("--raw-"+suffix,type=Path,default=RAW_ROOT/(s+".bin"));p.add_argument("--tracker-"+suffix,type=Path,default=TRACKERS[s])
 return p

def main(argv=None):
 args=parser().parse_args(argv)
 if not args.execute_production:raise SystemExit("refusing production I/O without --execute-production")
 report=subprocess.run([sys.executable,str(ROOT/"scripts/verify_acaf_nf_stage0_static_r14_doppler_validation.py"),str(R14_ARTIFACT)],capture_output=True,text=True)
 if report.returncode:raise SystemExit("R1.4 verifier did not PASS")
 print(json.dumps(publish(args),indent=2,sort_keys=True))
if __name__=="__main__":main()
