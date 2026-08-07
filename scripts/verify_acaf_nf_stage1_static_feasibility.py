#!/usr/bin/env python3
"""Independent Stage-1 fail-closed verifier; intentionally duplicates constants."""
from __future__ import annotations
import argparse,csv,ctypes,errno,hashlib,json,os,shutil,stat,subprocess,sys,tempfile
from pathlib import Path
import h5py,numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE_BINDING_CONFIG=ROOT/"configs/acaf_nf_stage1_source_binding.json"
SOURCE_BINDING_SHA256="23e4e5846283ba4a7e4e04f69c62c994acd339d1f848bba4415f865fc252c4ae"
NAMES={"README.md","config.json","source_binding.json","r14_frozen_lineage.json","scenario_timeline.json","normal_model_summary.json","thresholds.json","scenario_metrics.csv","phase_metrics.csv","per_window_scores.csv","secondary_component_metrics.csv","baseline_metrics.csv","control_metrics.csv","bootstrap_results.json","go_no_go.json","execution_validity.json","verification_report.json","checksums.json","test_report.txt"}
CSV_HEADERS={"scenario_metrics.csv":["status","reason","scenario"],"phase_metrics.csv":["status","reason","scenario","phase"],"per_window_scores.csv":["status","reason","scenario","time_s","score"],"secondary_component_metrics.csv":["status","reason","component"],"baseline_metrics.csv":["status","reason","baseline"],"control_metrics.csv":["status","reason","control"]};CSVS=set(CSV_HEADERS)
HASHES={"cleanStatic":"dd295ab46616bfe9634d1c37479520e720ebc54bcb64adf0a247315a541fb9b9","ds3":"e37e11b060bc2c675d4a60024f8b4a53e95e7cd1d304bea80cd903856075a30d","ds4":"1fff2b048a00732686bb1d77a13941da81c9fac648ca3695a9028f4ee3485285","ds7":"d5fb1430d476f68930f3bb0290b80b649f08012eb6b6981d112493528813400e","ds8":"1614d8de6fc8ebc3429def6e9505050c08a3ee8da69c11ecc27a98305f735d78"}
EXPECTED={"cleanStatic":{"all":{"triples":101278,"l20":100864,"prns":11}},"ds3":{"pre":{"triples":148699,"l20":148196,"prns":11},"onset_to_pulloff":{"triples":39761,"l20":39604,"prns":5},"post_pulloff":{"triples":5200,"l20":5124,"prns":3}},"ds4":{"pre":{"triples":107471,"l20":106914,"prns":11},"transition_only":{"triples":18,"l20":0,"prns":0}},"ds7":{"pre":{"triples":100520,"l20":100290,"prns":11},"transition":{"triples":0,"l20":0,"prns":0},"held":{"triples":0,"l20":0,"prns":0},"time_push":{"triples":0,"l20":0,"prns":0}},"ds8":{"pre":{"triples":102940,"l20":102662,"prns":10},"transition":{"triples":0,"l20":0,"prns":0},"held":{"triples":0,"l20":0,"prns":0},"time_push":{"triples":288,"l20":232,"prns":1}}}
TIMELINES={"ds3":{"onset_s":118.9,"pull_off_s":195.0},"ds4":{"onset_s":113.8,"pull_off_s":225.0,"raw_end_approx_s":128.22},"ds7":{"injection_s":110.,"transition":[110.,130.],"held":[130.,150.],"time_push_start_s":150.},"ds8":{"injection_s":110.,"transition":[110.,130.],"held":[130.,150.],"time_push_start_s":150.}}
PHASES={"cleanStatic":{"all":[0,None]},"ds3":{"pre":[0,2972500000],"onset_to_pulloff":[2972500000,4875000000],"post_pulloff":[4875000000,None]},"ds4":{"pre":[0,2845000000],"transition_only":[2845000000,5625000000]},"ds7":{"pre":[0,2750000000],"transition":[2750000000,3250000000],"held":[3250000000,3750000000],"time_push":[3750000000,None]},"ds8":{"pre":[0,2750000000],"transition":[2750000000,3250000000],"held":[3250000000,3750000000],"time_push":[3750000000,None]}}
REQUIRED={"stamp":"PRN_start_sample_count","prn":"PRN","carrier_doppler_hz":"carrier_doppler_hz","code_freq_chips":"code_freq_chips","aux1":"aux1","prompt_i":"Prompt_I","prompt_q":"Prompt_Q","cn0":"CN0_SNV_dB_Hz","lock":"carrier_lock_test"}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def field(f,n):return np.asarray(f[n]).reshape(-1) if n in f else None
def json_pointer(document,pointer):
 value=document
 for token in pointer.split("/")[1:]:value=value[token.replace("~1","/").replace("~0","~")]
 return value
def metadata_contract(spec, scenario):
 manifest=load(spec["manifest_path"]);pointers=spec["manifest_pointers"]
 rate=json_pointer(manifest,pointers["sample_rate_hz"]);fmt=str(json_pointer(manifest,pointers["sample_format"])).lower()
 if rate!=25000000 or "ishort" not in fmt or not ("iq" in fmt or "complex" in fmt):raise ValueError("manifest rate/format contract")
 if pointers["raw_sha256"] is None:
  raw_binding={"status":"FAIL","reason":"MANIFEST_DOES_NOT_BIND_RAW_SHA","pointer":None}
 elif json_pointer(manifest,pointers["raw_sha256"])!=spec["raw_sha256"]:raise ValueError("manifest raw SHA contract")
 else:raw_binding={"status":"PASS","reason":None,"pointer":pointers["raw_sha256"]}
 values={}
 for raw_line in Path(spec["receiver_config_path"]).read_text().splitlines():
  line=raw_line.strip()
  if line and not line.startswith("#") and "=" in line:
   key,value=line.split("=",1);values[key.strip()]=value.strip()
 forbidden=[k for k in values if k.startswith("SignalSource.") and any(x in k.lower() for x in ("skip","header","offset"))]
 expected={"SignalSource.item_type":"ishort","SignalSource.sampling_frequency":"25000000","SignalSource.samples":"0"};failures=[]
 if forbidden:failures.append("FORBIDDEN_SKIP_HEADER_OFFSET_KEY")
 if any(values.get(k)!=v for k,v in expected.items()):failures.append("SOURCE_KEYS_MISMATCH")
 if not values.get("SignalSource.filename"):failures.append("MISSING_FILENAME")
 receiver={"status":"FAIL" if failures else "PASS","reasons":failures,"keys":{k:values.get(k) for k in (*expected,"SignalSource.filename")},"expected_keys":expected,"forbidden_keys":forbidden,"configured_filename_alias_classification":"HISTORICAL_RELOCATED_ALIAS","configured_filename_content_sha256":spec["raw_sha256"],"first_file_sample_is_raw_sample":0}
 return raw_binding,receiver
def phase_name(scenario,sample):
 for name,(start,end) in PHASES[scenario].items():
  if sample>=start and (end is None or sample<end):return name
 raise ValueError("sample outside exact phase contract")
def inventory(tracker, spec, observables):
 tracker=Path(tracker);entries=list(tracker.iterdir());canonical=set(spec["mat_inventory"]);aliases=set(spec.get("ignored_alias_symlinks",{}));noninputs={Path(n).with_suffix(".dat").name for n in canonical}|set(observables)
 if tracker.is_symlink() or not tracker.is_dir() or {p.name for p in entries}!=canonical|aliases|noninputs:raise ValueError("exact tracker inventory mismatch")
 out=[];ignored=[]
 for name in sorted(aliases):
  p=tracker/name
  if not p.is_symlink():raise ValueError("expected alias symlink")
  target=os.readlink(p);relative=(p.parent/target).resolve(strict=True).relative_to(tracker.resolve()).as_posix()
  if relative!=spec["ignored_alias_symlinks"][name]:raise ValueError("alias target mismatch")
  ignored.append({"filename":name,"link_target":target,"resolved_relative_target":relative,"ignored_noninput":True})
 for name in sorted(noninputs):
  p=tracker/name
  if p.is_symlink() or not p.is_file():raise ValueError("noninput must be regular")
 for name in sorted(canonical):
  mat=tracker/name
  if mat.is_symlink() or not mat.is_file() or digest(mat)!=spec["mat_inventory"][name]:raise ValueError("canonical MAT binding failure")
  with h5py.File(mat,"r") as f:
   values={k:field(f,v) for k,v in REQUIRED.items()}
   if any(v is None for v in values.values()):raise ValueError("required field missing")
   schema=[{"name":n,"dtype":str(f[n].dtype),"shape":list(f[n].shape)} for n in sorted(f.keys())]
   sd=hashlib.sha256(json.dumps(schema,separators=(",",":"),sort_keys=True).encode()).hexdigest()
   out.append({"filename":name,"sha256":digest(mat),"size_bytes":mat.stat().st_size,"rows":len(values["stamp"]),"required_field_schema":{"digest_sha256":sd,"fields":schema}})
 return {"canonical_mat_files":out,"ignored_alias_symlinks":ignored,"ignored_regular_noninputs":sorted(noninputs)}
def support(tracker,s,spec,observables,raw_sample_count):
 inv=inventory(tracker,spec,observables);accepted=[]
 for entry in inv["canonical_mat_files"]:
  mat=Path(tracker)/entry["filename"]
  with h5py.File(mat,"r") as f:
   values={k:field(f,v) for k,v in REQUIRED.items()};st=values["stamp"];prn=values["prn"];cn=values["cn0"];lk=values["lock"]
   if len(st)<3 or any(len(v)!=len(st) or v.dtype.kind not in "iuf" or not np.isfinite(v).all() for v in values.values()):raise ValueError("field schema/finite failure")
   for i in range(1,len(st)-1):
    start=int(st[i-1]);end=start+25000;before=int(st[i]-st[i-1]);after=int(st[i+1]-st[i])
    if (int(prn[i-1])==int(prn[i])==int(prn[i+1]) and 24999<=before<=25001
        and 24999<=after<=25001 and min(cn[i-1:i+2])>=28 and min(lk[i-1:i+2])>=.85
        and 1<=int(prn[i])<=32 and start>=0 and end<=raw_sample_count):accepted.append({"channel":entry["filename"],"row":i,"prn":int(prn[i]),"start":start,"end":end})
 result={}
 for phase in EXPECTED[s]:
  selected=[r for r in accepted if phase_name(s,r["start"])==phase and phase_name(s,r["end"]-1)==phase];per_prn={}
  groups={}
  for r in selected:groups.setdefault((r["channel"],r["prn"]),[]).append(r)
  for group in groups.values():
   group.sort(key=lambda r:r["start"])
   for j in range(19,len(group)):
    w=group[j-19:j+1]
    if len({r["row"] for r in w})==20 and all(24999<=b["start"]-a["start"]<=25001 and b["row"]==a["row"]+1 for a,b in zip(w,w[1:])) and phase_name(s,w[0]["start"])==phase_name(s,w[-1]["end"]-1)==phase:per_prn[w[-1]["prn"]]=per_prn.get(w[-1]["prn"],0)+1
  total=sum(per_prn.values());result[phase]={"triples":len(selected),"l20":total,"prns":len(per_prn),"dominant_fraction":max(per_prn.values(),default=0)/total if total else 0.,"crossing_excluded":sum(phase_name(s,r["start"])==phase and phase_name(s,r["end"]-1)!=phase for r in accepted)}
 return result,inv

def verify(root:Path, recompute_external=True, *, authenticated_hash_ledger=None, integration_diagnostic=False):
 root=Path(root);errors=[]
 try:
  if SOURCE_BINDING_SHA256=="TO_BE_FINALIZED" or digest(SOURCE_BINDING_CONFIG)!=SOURCE_BINDING_SHA256:errors.append("source_binding_config_drift")
  canonical=load(SOURCE_BINDING_CONFIG)
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
   if rows!=[CSV_HEADERS[name]]:errors.append("science_csv_not_header_only:"+name)
  for name in ("normal_model_summary.json","thresholds.json","bootstrap_results.json"):
   value=load(root/name)
   if set(value)!={"status","value","reason"} or value.get("status")!="NOT_EVALUATED" or value.get("value") is not None or not isinstance(value.get("reason"),str):errors.append("evaluated:"+name)
  report=load(root/"verification_report.json")
  if recompute_external:
   if report.get("status") not in {"PENDING_INDEPENDENT_VERIFICATION","PASS"}:errors.append("verification_report_phase")
   if report.get("status")=="PASS" and (set(report)!={"status","derived_verdict","errors","independent_of_producer_verdict","external_recomputation_performed","integration_diagnostic"} or report.get("errors")!=[] or report.get("derived_verdict")!="FOUNDATION_INVALID" or report.get("external_recomputation_performed") is not True):errors.append("verification_report_pass_schema")
  elif report!={"status":"PENDING_INDEPENDENT_VERIFICATION","producer_verdict_not_authoritative":True}:errors.append("verification_report_pending_schema")
  if load(root/"scenario_timeline.json")!=TIMELINES:errors.append("timeline")
  campaign=load(root/"config.json")
  if set(campaign)!={"frozen","source_binding_config_sha256","search_complexity","delay_grid","doppler_grid_hz","pooling_candidates","baseline_B0"} or campaign.get("source_binding_config_sha256")!=SOURCE_BINDING_SHA256 or campaign.get("baseline_B0")!="PROVISIONAL_UNAVAILABLE":errors.append("campaign_config_schema")
  execution=load(root/"execution_validity.json");go=load(root/"go_no_go.json")
  if set(execution)!={"status","no_attack_raw_scoring_performed","attack_iq_bytes_read_for_scoring","raw_bytes_read_purpose","science_csv_semantics","plots","B0"}:errors.append("execution_schema")
  if set(go)!={"verdict","PHYSICS_FEASIBILITY_GO","physics_feasibility_status","PAPER_CANDIDATE_GO","paper_candidate_status","stage2_justified","reason"}:errors.append("go_schema")
  if execution.get("no_attack_raw_scoring_performed") is not True or execution.get("raw_bytes_read_purpose")!="full_sha256_only" or execution.get("attack_iq_bytes_read_for_scoring")!=0:errors.append("attack_scoring_semantics")
  if go.get("verdict")!="FOUNDATION_INVALID" or go.get("physics_feasibility_status")!="NOT_EVALUATED" or go.get("paper_candidate_status")!="NOT_EVALUATED" or go.get("stage2_justified") is not False:errors.append("verdict_semantics")
  if go.get("PHYSICS_FEASIBILITY_GO") is not False or go.get("PAPER_CANDIDATE_GO") is not False or execution.get("B0")!="PROVISIONAL_UNAVAILABLE":errors.append("go_or_baseline_semantics")
  readme=(root/"README.md").read_text()
  if not all(x in readme for x in ("FOUNDATION_INVALID","not a physics `NO_GO`","never scored","NOT_EVALUATED","PROVISIONAL_UNAVAILABLE")):errors.append("readme_verdict_wording")
  test_report=(root/"test_report.txt").read_text()
  if not all(x in test_report for x in ("command:","exit_code: 0","collected:","passed:","failed:","Python","numpy","scipy","h5py","sklearn","matplotlib")):errors.append("test_report_schema")
  binding=load(root/"source_binding.json");sources=binding["sources"];audits=binding["tracker_support_audits"]
  for s,h in HASHES.items():
   if sources[s]["raw_sha256"]!=h or sources[s]["raw_bytes_read_purpose"]!="full_sha256_only":errors.append("raw_binding_document:"+s)
   spec=canonical["scenarios"][s]
   for key in ("raw_path","tracker_path","manifest_path","receiver_config_path"):
    if sources[s].get(key)!=spec[key]:errors.append("artifact_external_path:"+s+":"+key)
   if recompute_external:
    if authenticated_hash_ledger is not None:
     if not integration_diagnostic or authenticated_hash_ledger.get(s)!=h:errors.append("invalid_test_hash_ledger:"+s)
    elif digest(sources[s]["raw_path"])!=h:errors.append("raw_hash:"+s)
    if digest(spec["manifest_path"])!=spec["manifest_sha256"] or digest(spec["receiver_config_path"])!=spec["receiver_config_sha256"]:errors.append("metadata_hash:"+s)
    raw_binding,receiver_binding=metadata_contract(spec,s)
    if raw_binding!=sources[s].get("tracker_raw_binding") or receiver_binding!=sources[s].get("receiver_config_checks"):errors.append("metadata_contract:"+s)
    worker=subprocess.run([sys.executable,str(Path(__file__)),"--preflight-scenario",s,"--raw-sample-count",str(sources[s]["raw_sample_count"])],capture_output=True,text=True)
    if worker.returncode:raise RuntimeError("metadata worker failed:"+s+":"+worker.stderr)
    recomputed=json.loads(worker.stdout);observed=recomputed["counts"];inv=recomputed["inventory"]
    if inv!=audits[s].get("mat_inventory"):errors.append("mat_inventory_hash:"+s)
    if observed!=audits[s].get("counts"):errors.append("support_recompute:"+s)
    if any({k:observed[p][k] for k in ("triples","l20","prns")}!=e for p,e in EXPECTED[s].items()):errors.append("support_snapshot:"+s)
   if any({k:audits[s].get("counts",{}).get(p,{}).get(k) for k in ("triples","l20","prns")}!=e for p,e in EXPECTED[s].items()) or audits[s].get("matches_expected") is not True:errors.append("support_document:"+s)
  if sources["ds4"].get("tracker_raw_binding")!={"status":"FAIL","reason":"MANIFEST_DOES_NOT_BIND_RAW_SHA","pointer":None}:errors.append("ds4_unbound_semantics")
  if EXPECTED["ds7"]["time_push"]["triples"] or EXPECTED["ds8"]["transition"]["triples"] or EXPECTED["ds8"]["held"]["triples"]:errors.append("foundation_derivation")
  lineage=load(root/"r14_frozen_lineage.json")
  if lineage.get("artifact")!=str(ROOT/canonical["r14"]["artifact_path"]):errors.append("r14_artifact_path")
  for field,config_field in (("artifact_checksums_sha256","checksums_sha256"),("verification_report_sha256","verification_report_sha256"),("verifier_sha256","verifier_sha256"),("module_sha256","module_sha256"),("runner_sha256","runner_sha256")):
   if lineage.get(field)!=canonical["r14"][config_field]:errors.append("r14_lineage:"+field)
  source_hashes=lineage.get("stage1_source_hashes",{});expected_source_hashes={"producer_sha256":digest(ROOT/"scripts/run_acaf_nf_stage1_static_feasibility.py"),"module_sha256":digest(ROOT/"src/gnss_doppler_lab/acaf_nf_stage1_static_feasibility.py"),"verifier_sha256":digest(Path(__file__)),"config_sha256":digest(SOURCE_BINDING_CONFIG)}
  if source_hashes!=expected_source_hashes:errors.append("stage1_source_hashes")
  actual_head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();actual_dirty=bool(subprocess.run(["git","status","--porcelain"],cwd=ROOT,text=True,capture_output=True,check=True).stdout)
  if lineage.get("git_head")!=actual_head or lineage.get("git_dirty")!=actual_dirty:errors.append("git_state")
  if recompute_external:
   r=subprocess.run([sys.executable,str(ROOT/"scripts/verify_acaf_nf_stage0_static_r14_doppler_validation.py"),lineage["artifact"]],capture_output=True,text=True)
   if r.returncode:errors.append("r14_verifier")
 except Exception as exc:errors.append(f"exception:{type(exc).__name__}:{exc}")
 status=("FAIL" if errors else "PASS") if recompute_external else ("FAIL" if errors else "INCOMPLETE")
 return {"status":status,"derived_verdict":"FOUNDATION_INVALID" if not errors and recompute_external else None,"errors":errors,"independent_of_producer_verdict":True,"external_recomputation_performed":bool(recompute_external),"integration_diagnostic":bool(integration_diagnostic)}

def main(argv=None):
 argv=list(sys.argv[1:] if argv is None else argv)
 if argv and argv[0]=="--preflight-scenario":
  worker=argparse.ArgumentParser();worker.add_argument("--preflight-scenario",choices=sorted(HASHES));worker.add_argument("--raw-sample-count",type=int,required=True);wa=worker.parse_args(argv)
  canonical=load(SOURCE_BINDING_CONFIG);spec=canonical["scenarios"][wa.preflight_scenario];counts,inv=support(spec["tracker_path"],wa.preflight_scenario,spec,canonical["tracker_inventory_contract"]["observables_noninputs_by_scenario"][wa.preflight_scenario],wa.raw_sample_count)
  print(json.dumps({"counts":counts,"inventory":inv},sort_keys=True));raise SystemExit(0)
 p=argparse.ArgumentParser();p.add_argument("artifact",type=Path);p.add_argument("--finalize-to",type=Path);p.add_argument("--skip-external-recompute",action="store_true");p.add_argument("--integration-authenticated-hash-ledger",type=Path);a=p.parse_args(argv)
 if a.finalize_to and a.skip_external_recompute:raise SystemExit("diagnostic skip cannot finalize")
 if a.integration_authenticated_hash_ledger and os.environ.get("ACAF_TEST_ONLY_HASH_LEDGER")!="1":raise SystemExit("test-only hash ledger requires ACAF_TEST_ONLY_HASH_LEDGER=1")
 ledger=load(a.integration_authenticated_hash_ledger) if a.integration_authenticated_hash_ledger else None
 result=verify(a.artifact,not a.skip_external_recompute,authenticated_hash_ledger=ledger,integration_diagnostic=ledger is not None)
 if a.finalize_to:
  if result["status"]!="PASS":raise SystemExit(1)
  destination=a.finalize_to
  if destination.exists():raise FileExistsError(destination)
  stage=Path(tempfile.mkdtemp(prefix=destination.name+".verifying-",dir=destination.parent))
  shutil.rmtree(stage);shutil.copytree(a.artifact,stage,symlinks=True)
  (stage/"verification_report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  files={n:{"sha256":digest(stage/n),"size_bytes":(stage/n).stat().st_size} for n in sorted(NAMES-{"checksums.json"})}
  (stage/"checksums.json").write_text(json.dumps({"algorithm":"sha256","files":files},indent=2,sort_keys=True)+"\n")
  for item in stage.rglob("*"):
   if item.is_file():
    with item.open("rb") as handle:os.fsync(handle.fileno())
  for directory in (stage,stage/"plots"):
   fd=os.open(directory,os.O_RDONLY);os.fsync(fd);os.close(fd)
  libc=ctypes.CDLL(None,use_errno=True);rc=libc.renameat2(-100,os.fsencode(stage),-100,os.fsencode(destination),1)
  if rc:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()),destination)
  parent_fd=os.open(destination.parent,os.O_RDONLY);os.fsync(parent_fd);os.close(parent_fd)
 print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result["status"]=="PASS" else 2 if result["status"]=="INCOMPLETE" else 1)
if __name__=="__main__":main()
