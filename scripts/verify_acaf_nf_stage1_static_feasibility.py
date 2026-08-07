#!/usr/bin/env python3
"""Independent Stage-1 fail-closed verifier; intentionally duplicates constants."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,stat,subprocess,sys
from pathlib import Path
import h5py,numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE_BINDING_CONFIG=ROOT/"configs/acaf_nf_stage1_source_binding.json"
SOURCE_BINDING_SHA256="21a261676e095683d68a753094967d18a4f048f3c4a03b50d2596e385543913e"
NAMES={"README.md","config.json","source_binding.json","r14_frozen_lineage.json","scenario_timeline.json","normal_model_summary.json","thresholds.json","scenario_metrics.csv","phase_metrics.csv","per_window_scores.csv","secondary_component_metrics.csv","baseline_metrics.csv","control_metrics.csv","bootstrap_results.json","go_no_go.json","execution_validity.json","verification_report.json","checksums.json","test_report.txt"}
CSVS={"scenario_metrics.csv","phase_metrics.csv","per_window_scores.csv","secondary_component_metrics.csv","baseline_metrics.csv","control_metrics.csv"}
HASHES={"cleanStatic":"dd295ab46616bfe9634d1c37479520e720ebc54bcb64adf0a247315a541fb9b9","ds3":"e37e11b060bc2c675d4a60024f8b4a53e95e7cd1d304bea80cd903856075a30d","ds4":"1fff2b048a00732686bb1d77a13941da81c9fac648ca3695a9028f4ee3485285","ds7":"d5fb1430d476f68930f3bb0290b80b649f08012eb6b6981d112493528813400e","ds8":"1614d8de6fc8ebc3429def6e9505050c08a3ee8da69c11ecc27a98305f735d78"}
EXPECTED={"ds3":{"pre":{"triples":148699,"l20":148196,"prns":11},"onset_to_pulloff":{"triples":39762,"l20":39605,"prns":5},"post_pulloff":{"triples":5200,"l20":5124,"prns":3}},"ds4":{"pre":{"triples":107471,"l20":106914,"prns":11},"transition_only":{"triples":18,"l20":0,"prns":0}},"ds7":{"pre":{"triples":100520,"l20":100290,"prns":11},"transition":{"triples":0,"l20":0,"prns":0},"held":{"triples":0,"l20":0,"prns":0},"time_push":{"triples":0,"l20":0,"prns":0}},"ds8":{"pre":{"triples":102940,"l20":102662,"prns":10},"transition":{"triples":0,"l20":0,"prns":0},"held":{"triples":0,"l20":0,"prns":0},"time_push":{"triples":288,"l20":232,"prns":1}}}
TIMELINES={"ds3":{"onset_s":118.9,"pull_off_s":195.0},"ds4":{"onset_s":113.8,"pull_off_s":225.0,"raw_end_approx_s":128.22},"ds7":{"injection_s":110.,"transition":[110.,130.],"held":[130.,150.],"time_push_start_s":150.},"ds8":{"injection_s":110.,"transition":[110.,130.],"held":[130.,150.],"time_push_start_s":150.}}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def field(f,n):return np.asarray(f[n]).reshape(-1) if n in f else None
def inventory(tracker):
 out=[]
 for mat in sorted(Path(tracker).glob("*.mat")):
  with h5py.File(mat,"r") as f:
   st=field(f,"PRN_start_sample_count")
   out.append({"path":str(mat),"sha256":digest(mat),"rows":0 if st is None else len(st),"fields":sorted(f.keys())})
 return out
def phase(s,t):
 if s=="ds3":return "pre" if t<118.9 else "onset_to_pulloff" if t<195 else "post_pulloff"
 if s=="ds4":return "pre" if t<113.8 else "post_onset"
 if s=="ds7":return "pre" if t<110 else "post_onset"
 return "pre" if t<110 else "110_to_150" if t<150 else "ge_150"
def support(tracker,s):
 accepted=[]
 for mat in sorted(Path(tracker).glob("*.mat")):
  with h5py.File(mat,"r") as f:
   st=field(f,"PRN_start_sample_count");prn=field(f,"PRN");cn=field(f,"CN0_SNV_dB_Hz");lk=field(f,"carrier_lock_test")
   if any(x is None for x in (st,prn,cn,lk)):continue
   for i in range(1,len(st)-1):
    before=int(st[i]-st[i-1]);after=int(st[i+1]-st[i])
    if (int(prn[i-1])==int(prn[i])==int(prn[i+1]) and 24999<=before<=25001
        and 24999<=after<=25001 and min(cn[i-1:i+2])>=28 and min(lk[i-1:i+2])>=.85):
     accepted.append((int(st[i]),int(prn[i])))
 return {p:[len(v:=[x for x in accepted if phase(s,x[0]/25e6)==p]),len({x[1] for x in v})] for p in EXPECTED[s]}

def verify(root:Path, recompute_external=True):
 root=Path(root);errors=[]
 try:
  if digest(SOURCE_BINDING_CONFIG)!=SOURCE_BINDING_SHA256:errors.append("source_binding_config_drift")
  if root.is_symlink() or not root.is_dir():raise ValueError("artifact root is not a real directory")
  for p in root.rglob("*"):
   mode=p.lstat().st_mode
   if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):errors.append("symlink_or_special:"+p.relative_to(root).as_posix())
  files={str(p.relative_to(root)) for p in root.iterdir() if p.is_file()};dirs={p.name for p in root.iterdir() if p.is_dir()}
  if files!=NAMES or dirs!={"plots"} or any((root/"plots").iterdir()):errors.append("exact_inventory_or_nonempty_plots")
  checks=load(root/"checksums.json")
  expected_checksum_names=NAMES-{"checksums.json"}
  entries=checks.get("files",{})
  safe=all(isinstance(n,str) and n==Path(n).as_posix() and not Path(n).is_absolute() and all(x not in ("",".","..") for x in Path(n).parts) for n in entries)
  if set(checks)!={"algorithm","files"} or checks.get("algorithm")!="sha256" or set(entries)!=expected_checksum_names or not safe:errors.append("checksum_schema_or_inventory")
  elif any(set(v)!={"sha256","size_bytes"} or digest(root/n)!=v["sha256"] or (root/n).stat().st_size!=v["size_bytes"] for n,v in entries.items()):errors.append("checksum_tamper")
  for name in CSVS:
   with (root/name).open(newline="") as f:
    rows=list(csv.reader(f))
   if len(rows)!=1 or not {"status","reason"}.issubset(rows[0]):errors.append("science_csv_not_header_only:"+name)
  for name in ("normal_model_summary.json","thresholds.json","bootstrap_results.json"):
   value=load(root/name)
   if set(value)!={"status","value","reason"} or value.get("status")!="NOT_EVALUATED" or value.get("value") is not None or not isinstance(value.get("reason"),str):errors.append("evaluated:"+name)
  report=load(root/"verification_report.json")
  if recompute_external:
   if report.get("status") not in {"PENDING_INDEPENDENT_VERIFICATION","PASS"}:errors.append("verification_report_phase")
  elif report!={"status":"PENDING_INDEPENDENT_VERIFICATION","producer_verdict_not_authoritative":True}:errors.append("verification_report_pending_schema")
  if load(root/"scenario_timeline.json")!=TIMELINES:errors.append("timeline")
  execution=load(root/"execution_validity.json");go=load(root/"go_no_go.json")
  if execution.get("no_attack_raw_scoring_performed") is not True or execution.get("raw_bytes_read_purpose")!="full_sha256_only" or execution.get("attack_iq_bytes_read_for_scoring")!=0:errors.append("attack_scoring_semantics")
  if go.get("verdict")!="FOUNDATION_INVALID" or go.get("physics_feasibility_status")!="NOT_EVALUATED" or go.get("paper_candidate_status")!="NOT_EVALUATED" or go.get("stage2_justified") is not False:errors.append("verdict_semantics")
  binding=load(root/"source_binding.json");sources=binding["sources"];audits=binding["tracker_support_audits"]
  for s,h in HASHES.items():
   if sources[s]["raw_sha256"]!=h or sources[s]["raw_bytes_read_purpose"]!="full_sha256_only":errors.append("raw_binding_document:"+s)
   if recompute_external:
    if digest(sources[s]["raw_path"])!=h:errors.append("raw_hash:"+s)
    if digest(sources[s]["manifest_path"])!=sources[s]["manifest_sha256"]:errors.append("manifest_hash:"+s)
    if inventory(sources[s]["tracker_path"])!=audits[s].get("mat_inventory"):errors.append("mat_inventory_hash:"+s)
    if s!="cleanStatic" and support(sources[s]["tracker_path"],s)!=EXPECTED[s]:errors.append("support:"+s)
   if s!="cleanStatic" and (any({k:audits[s].get("counts",{}).get(p,{}).get(k) for k in ("triples","l20","prns")}!=e for p,e in EXPECTED[s].items()) or audits[s].get("matches_expected") is not True):errors.append("support_document:"+s)
  if EXPECTED["ds7"]["time_push"]["triples"] or EXPECTED["ds8"]["transition"]["triples"] or EXPECTED["ds8"]["held"]["triples"]:errors.append("foundation_derivation")
  lineage=load(root/"r14_frozen_lineage.json")
  if recompute_external:
   r=subprocess.run([sys.executable,str(ROOT/"scripts/verify_acaf_nf_stage0_static_r14_doppler_validation.py"),lineage["artifact"]],capture_output=True,text=True)
   if r.returncode:errors.append("r14_verifier")
 except Exception as exc:errors.append(f"exception:{type(exc).__name__}:{exc}")
 status=("FAIL" if errors else "PASS") if recompute_external else ("FAIL" if errors else "INCOMPLETE")
 return {"status":status,"derived_verdict":"FOUNDATION_INVALID" if not errors and recompute_external else None,"errors":errors,"independent_of_producer_verdict":True,"external_recomputation_performed":bool(recompute_external)}

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("artifact",type=Path);p.add_argument("--finalize",action="store_true");p.add_argument("--skip-external-recompute",action="store_true");a=p.parse_args(argv)
 if a.finalize and a.skip_external_recompute:raise SystemExit("diagnostic skip cannot finalize")
 result=verify(a.artifact,not a.skip_external_recompute)
 if a.finalize:
  if result["status"]!="PASS":raise SystemExit(1)
  (a.artifact/"verification_report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  files={n:{"sha256":digest(a.artifact/n),"size_bytes":(a.artifact/n).stat().st_size} for n in sorted(NAMES-{"checksums.json"})}
  (a.artifact/"checksums.json").write_text(json.dumps({"algorithm":"sha256","files":files},indent=2,sort_keys=True)+"\n")
 print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result["status"] in {"PASS","INCOMPLETE"} else 1)
if __name__=="__main__":main()
