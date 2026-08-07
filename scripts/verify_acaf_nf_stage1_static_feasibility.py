#!/usr/bin/env python3
"""Independent Stage-1 fail-closed verifier; intentionally duplicates constants."""
from __future__ import annotations
import argparse,csv,ctypes,errno,hashlib,json,os,re,shutil,stat,subprocess,sys,tempfile
from pathlib import Path
import h5py,numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE_BINDING_CONFIG=ROOT/"configs/acaf_nf_stage1_source_binding.json"
SOURCE_BINDING_SHA256="af59415f82c3615ebba0690fa9b7a117ea9348bf1d61ab7a95e716d981cef8ae"
NAMES={"README.md","config.json","source_binding.json","r14_frozen_lineage.json","scenario_timeline.json","normal_model_summary.json","thresholds.json","scenario_metrics.csv","phase_metrics.csv","per_window_scores.csv","secondary_component_metrics.csv","baseline_metrics.csv","control_metrics.csv","bootstrap_results.json","go_no_go.json","execution_validity.json","verification_report.json","checksums.json","test_report.txt"}
CSV_HEADERS={"scenario_metrics.csv":["status","reason","scenario"],"phase_metrics.csv":["status","reason","scenario","phase"],"per_window_scores.csv":["status","reason","scenario","time_s","score"],"secondary_component_metrics.csv":["status","reason","component"],"baseline_metrics.csv":["status","reason","baseline"],"control_metrics.csv":["status","reason","control"]};CSVS=set(CSV_HEADERS)
HASHES={"cleanStatic":"dd295ab46616bfe9634d1c37479520e720ebc54bcb64adf0a247315a541fb9b9","ds3":"e37e11b060bc2c675d4a60024f8b4a53e95e7cd1d304bea80cd903856075a30d","ds4":"1fff2b048a00732686bb1d77a13941da81c9fac648ca3695a9028f4ee3485285","ds7":"d5fb1430d476f68930f3bb0290b80b649f08012eb6b6981d112493528813400e","ds8":"1614d8de6fc8ebc3429def6e9505050c08a3ee8da69c11ecc27a98305f735d78"}
EXPECTED={"cleanStatic":{"all":{"triples":101278,"l20":100864,"prns":11}},"ds3":{"pre":{"triples":148699,"l20":148196,"prns":11},"onset_to_pulloff":{"triples":39761,"l20":39604,"prns":5},"post_pulloff":{"triples":5200,"l20":5124,"prns":3}},"ds4":{"pre":{"triples":107471,"l20":106914,"prns":11},"transition_only":{"triples":18,"l20":0,"prns":0}},"ds7":{"pre":{"triples":100520,"l20":100290,"prns":11},"transition":{"triples":0,"l20":0,"prns":0},"held":{"triples":0,"l20":0,"prns":0},"time_push":{"triples":0,"l20":0,"prns":0}},"ds8":{"pre":{"triples":102940,"l20":102662,"prns":10},"transition":{"triples":0,"l20":0,"prns":0},"held":{"triples":0,"l20":0,"prns":0},"time_push":{"triples":288,"l20":232,"prns":1}}}
TIMELINES={"ds3":{"onset_s":118.9,"pull_off_s":195.0},"ds4":{"onset_s":113.8,"pull_off_s":225.0,"raw_end_approx_s":128.22},"ds7":{"injection_s":110.,"transition":[110.,130.],"held":[130.,150.],"time_push_start_s":150.},"ds8":{"injection_s":110.,"transition":[110.,130.],"held":[130.,150.],"time_push_start_s":150.}}
PHASES={"cleanStatic":{"all":[0,None]},"ds3":{"pre":[0,2972500000],"onset_to_pulloff":[2972500000,4875000000],"post_pulloff":[4875000000,None]},"ds4":{"pre":[0,2845000000],"transition_only":[2845000000,5625000000]},"ds7":{"pre":[0,2750000000],"transition":[2750000000,3250000000],"held":[3250000000,3750000000],"time_push":[3750000000,None]},"ds8":{"pre":[0,2750000000],"transition":[2750000000,3250000000],"held":[3250000000,3750000000],"time_push":[3750000000,None]}}
REQUIRED={"stamp":"PRN_start_sample_count","prn":"PRN","carrier_doppler_hz":"carrier_doppler_hz","code_freq_chips":"code_freq_chips","aux1":"aux1","prompt_i":"Prompt_I","prompt_q":"Prompt_Q","cn0":"CN0_SNV_dB_Hz","lock":"carrier_lock_test"}
FROZEN={"signal":"canonical_gps_l1_ca","fs_hz":25000000.0,"raw_format":"signed_int16_interleaved_complex_iq","global_raw_offset_samples":0,"nco_row":"previous","aux_row":"previous","remnant_sign":-1,"carrier_sign":-1,"replica_direction":"forward","prompt_row":"current","support_samples":25000,"window_length":20,"delay_start_chip":-1.0,"delay_stop_chip":1.0,"delay_step_chip":.125,"doppler_start_hz":-250.0,"doppler_stop_hz":250.0,"doppler_step_hz":50.0,"h1_center_excluded":True}
CAMPAIGN={"frozen":FROZEN,"source_binding_config_sha256":SOURCE_BINDING_SHA256,"search_complexity":{"calibration_statistic":"full_minimized_delta_search","scalar_penalty":0.0},"delay_grid":[-1.0+i*.125 for i in range(17)],"doppler_grid_hz":list(range(-250,251,50)),"pooling_candidates":["median","top50_mean","trimmed_mean"],"baseline_B0":"PROVISIONAL_UNAVAILABLE"}
SOURCE_KEYS={"raw_path","raw_sha256","expected_raw_sha256","raw_size_bytes","raw_sample_count","raw_count_unit","raw_bytes_read_purpose","raw_checks","tracker_path","manifest_path","manifest_sha256","receiver_config_path","receiver_config_sha256","manifest_checks","receiver_config_checks","tracker_raw_binding"}
AUDIT_KEYS={"scenario","counts","expected","matches_expected","excluded","boundary_crossing_supports","mat_inventory","accepted_rule","twenty_ms_gaps_interpolated","tracker_raw_binding_status","receiver_config_status"}
TEST_TARGETS=[["tests/test_acaf_nf_stage1_static_feasibility.py"],["tests/test_acaf_nf_stage0.py","tests/test_acaf_nf_stage0_r1.py","tests/test_acaf_nf_stage0_r11_validity.py","tests/test_acaf_nf_stage0_r12_alignment.py","tests/test_acaf_nf_stage0_r13_reconstruction.py","tests/test_acaf_nf_stage0_r14_doppler_validation.py"]]

def digest(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def test_versions():
 result={"python":sys.version.split()[0],"numpy":np.__version__,"h5py":h5py.__version__}
 for package in ("scipy","sklearn","matplotlib"):result[package]=__import__(package).__version__
 return result
def pytest_counts(stdout):
 matches=re.findall(r"(?:(\d+) failed,?\s*)?(?:(\d+) passed).*? in [0-9.]+s",stdout)
 if not matches:return None
 failed,passed=matches[-1];passed=int(passed);failed=int(failed or 0)
 return {"collected":passed+failed,"passed":passed,"failed":failed}
def field(f,n):return np.asarray(f[n]).reshape(-1) if n in f else None
def json_pointer(document,pointer):
 value=document
 for token in pointer.split("/")[1:]:value=value[token.replace("~1","/").replace("~0","~")]
 return value
def metadata_contract(spec, scenario, raw_size_bytes):
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
 expected={"SignalSource.item_type":"ishort","SignalSource.sampling_frequency":"25000000"};failures=[]
 if forbidden:failures.append("FORBIDDEN_SKIP_HEADER_OFFSET_KEY")
 if any(values.get(k)!=v for k,v in expected.items()):failures.append("SOURCE_KEYS_MISMATCH")
 if not values.get("SignalSource.filename"):failures.append("MISSING_FILENAME")
 try:samples=int(values.get("SignalSource.samples", ""))
 except ValueError:samples=-1
 full_scalar_count=raw_size_bytes//2;covers=samples==0 or samples==full_scalar_count
 if samples<0 or not covers:failures.append("SAMPLES_DO_NOT_COVER_FULL_FILE")
 expected["SignalSource.samples"]="0_or_exact_scalar_int16_full_source_count"
 receiver={"status":"FAIL" if failures else "PASS","reasons":failures,"keys":{k:values.get(k) for k in ("SignalSource.item_type","SignalSource.sampling_frequency","SignalSource.samples","SignalSource.filename")},"expected_keys":expected,"forbidden_keys":forbidden,"configured_filename_alias_classification":"HISTORICAL_RELOCATED_ALIAS","configured_filename_content_sha256":spec["raw_sha256"],"first_file_sample_is_raw_sample":0,"count_unit":"scalar_int16","configured_count":samples,"full_source_count":full_scalar_count,"covers_full_file":covers}
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
  if campaign!=CAMPAIGN:errors.append("campaign_config_schema")
  execution=load(root/"execution_validity.json");go=load(root/"go_no_go.json")
  if set(execution)!={"status","no_attack_raw_scoring_performed","attack_iq_bytes_read_for_scoring","raw_bytes_read_purpose","science_csv_semantics","plots","B0"}:errors.append("execution_schema")
  if set(go)!={"verdict","PHYSICS_FEASIBILITY_GO","physics_feasibility_status","PAPER_CANDIDATE_GO","paper_candidate_status","stage2_justified","reason"}:errors.append("go_schema")
  if set(execution.get("plots",{}))!={"count","reason"}:errors.append("execution_nested_schema")
  if execution.get("no_attack_raw_scoring_performed") is not True or execution.get("raw_bytes_read_purpose")!="full_sha256_only" or execution.get("attack_iq_bytes_read_for_scoring")!=0 or execution.get("status")!="FOUNDATION_INVALID" or execution.get("science_csv_semantics")!="header_only" or execution.get("plots",{}).get("count")!=0:errors.append("attack_scoring_semantics")
  if go.get("verdict")!="FOUNDATION_INVALID" or go.get("physics_feasibility_status")!="NOT_EVALUATED" or go.get("paper_candidate_status")!="NOT_EVALUATED" or go.get("stage2_justified") is not False:errors.append("verdict_semantics")
  if go.get("PHYSICS_FEASIBILITY_GO") is not False or go.get("PAPER_CANDIDATE_GO") is not False or execution.get("B0")!="PROVISIONAL_UNAVAILABLE":errors.append("go_or_baseline_semantics")
  readme=(root/"README.md").read_text()
  if not all(x in readme for x in ("FOUNDATION_INVALID","not a physics `NO_GO`","never scored","NOT_EVALUATED","PROVISIONAL_UNAVAILABLE")):errors.append("readme_verdict_wording")
  test_report=load(root/"test_report.txt");test_head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
  commands=[[sys.executable,"-m","pytest","-q",*targets] for targets in TEST_TARGETS]
  if (set(test_report)!={"schema","source_head","versions","environment","runs"} or test_report.get("schema")!="acaf-stage1-test-report-v1" or test_report.get("source_head")!=test_head or test_report.get("versions")!=test_versions() or test_report.get("environment")!={"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"} or not isinstance(test_report.get("runs"),list) or len(test_report["runs"])!=2):errors.append("test_report_schema")
  else:
   for saved,command in zip(test_report["runs"],commands):
    if set(saved)!={"command","stdout","exit_code","collected","passed","failed"} or saved.get("command")!=command or saved.get("exit_code")!=0 or pytest_counts(saved.get("stdout",""))!={k:saved.get(k) for k in ("collected","passed","failed")}:errors.append("test_report_run_schema");continue
    if recompute_external:
     env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"};rerun=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,env=env)
     counts=pytest_counts(rerun.stdout+rerun.stderr)
     if rerun.returncode!=0 or counts!={k:saved[k] for k in ("collected","passed","failed")}:errors.append("test_report_recompute")
  binding=load(root/"source_binding.json")
  if set(binding)!={"sources","tracker_support_audits","ds7_ds8_pre_attack_pairing"} or binding.get("ds7_ds8_pre_attack_pairing")!="paired replay diagnostic only if byte identity authenticated":errors.append("source_binding_schema")
  sources=binding.get("sources",{});audits=binding.get("tracker_support_audits",{})
  if set(sources)!=set(HASHES) or set(audits)!=set(HASHES):errors.append("source_binding_scenarios")
  for s,h in HASHES.items():
   if set(sources.get(s,{}))!=SOURCE_KEYS or set(audits.get(s,{}))!=AUDIT_KEYS:errors.append("source_binding_nested_schema:"+s);continue
   if sources[s]["raw_sha256"]!=h or sources[s]["raw_bytes_read_purpose"]!="full_sha256_only":errors.append("raw_binding_document:"+s)
   spec=canonical["scenarios"][s]
   for key in ("raw_path","tracker_path","manifest_path","receiver_config_path"):
    if sources[s].get(key)!=spec[key]:errors.append("artifact_external_path:"+s+":"+key)
   if (sources[s].get("manifest_sha256")!=spec["manifest_sha256"] or sources[s].get("receiver_config_sha256")!=spec["receiver_config_sha256"] or sources[s].get("expected_raw_sha256")!=h or sources[s].get("raw_count_unit")!="complex_int16_iq" or sources[s].get("raw_size_bytes")!=sources[s].get("raw_sample_count")*4):errors.append("source_values:"+s)
   expected_raw_checks={"size_divisible_by_4":{"status":"PASS"},"format":{"status":"PASS"},"rate":{"status":"PASS"},"sample_count":sources[s]["raw_sample_count"]}
   if sources[s].get("raw_checks")!=expected_raw_checks:errors.append("raw_checks:"+s)
   manifest_checks=sources[s].get("manifest_checks",{});receiver_checks=sources[s].get("receiver_config_checks",{})
   if (set(manifest_checks)!={"tracker_raw_binding","sample_rate","sample_format"}
       or set(manifest_checks.get("sample_rate",{}))!={"status","pointer","value"}
       or set(manifest_checks.get("sample_format",{}))!={"status","pointer","value"}
       or set(receiver_checks)!={"status","reasons","keys","expected_keys","forbidden_keys","configured_filename_alias_classification","configured_filename_content_sha256","first_file_sample_is_raw_sample","count_unit","configured_count","full_source_count","covers_full_file"}
       or receiver_checks.get("count_unit")!="scalar_int16" or receiver_checks.get("covers_full_file") is not True
       or receiver_checks.get("first_file_sample_is_raw_sample")!=0
       or receiver_checks.get("full_source_count")!=sources[s]["raw_size_bytes"]//2):errors.append("metadata_nested_schema:"+s)
   for phase,expected in EXPECTED[s].items():
    count=audits[s].get("counts",{}).get(phase,{})
    if set(count)!={"triples","l20","prns","dominant_fraction","crossing_excluded"} or {k:count.get(k) for k in ("triples","l20","prns")}!=expected:errors.append("support_nested_schema:"+s+":"+phase)
   required_manifest_status="FAIL" if s=="ds4" else "PASS"
   if (sources[s].get("tracker_raw_binding",{}).get("status")!=required_manifest_status or sources[s].get("receiver_config_checks",{}).get("status")!="PASS" or audits[s].get("receiver_config_status")!="PASS" or audits[s].get("tracker_raw_binding_status")!=required_manifest_status):errors.append("mandatory_source_gate:"+s)
   if recompute_external:
    if authenticated_hash_ledger is not None:
     if not integration_diagnostic or authenticated_hash_ledger.get(s)!=h:errors.append("invalid_test_hash_ledger:"+s)
    elif digest(sources[s]["raw_path"])!=h:errors.append("raw_hash:"+s)
    if digest(spec["manifest_path"])!=spec["manifest_sha256"] or digest(spec["receiver_config_path"])!=spec["receiver_config_sha256"]:errors.append("metadata_hash:"+s)
    raw_binding,receiver_binding=metadata_contract(spec,s,sources[s]["raw_size_bytes"])
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
  lineage_keys={"artifact","artifact_checksums_sha256","verification_report_sha256","verifier_sha256","module_sha256","runner_sha256","verifier_required","contract","base_sha","git_head","git_dirty","stage1_source_hashes","versions","status"}
  if set(lineage)!=lineage_keys or lineage.get("status")!="PASS" or lineage.get("base_sha")!=canonical["base_commit"] or lineage.get("contract")!=FROZEN or lineage.get("verifier_required")!="PASS" or lineage.get("versions")!=test_versions():errors.append("r14_lineage_schema")
  if lineage.get("artifact")!=str(ROOT/canonical["r14"]["artifact_path"]):errors.append("r14_artifact_path")
  for field,config_field in (("artifact_checksums_sha256","checksums_sha256"),("verification_report_sha256","verification_report_sha256"),("verifier_sha256","verifier_sha256"),("module_sha256","module_sha256"),("runner_sha256","runner_sha256")):
   if lineage.get(field)!=canonical["r14"][config_field]:errors.append("r14_lineage:"+field)
  source_hashes=lineage.get("stage1_source_hashes",{});expected_source_hashes={"producer_sha256":digest(ROOT/"scripts/run_acaf_nf_stage1_static_feasibility.py"),"module_sha256":digest(ROOT/"src/gnss_doppler_lab/acaf_nf_stage1_static_feasibility.py"),"verifier_sha256":digest(Path(__file__)),"config_sha256":digest(SOURCE_BINDING_CONFIG)}
  if source_hashes!=expected_source_hashes:errors.append("stage1_source_hashes")
  actual_head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();actual_dirty=bool(subprocess.run(["git","status","--porcelain"],cwd=ROOT,text=True,capture_output=True,check=True).stdout)
  if lineage.get("git_head")!=actual_head or lineage.get("git_dirty")!=actual_dirty:errors.append("git_state")
  reasons=[]
  if audits.get("ds4",{}).get("tracker_raw_binding_status")=="FAIL":reasons.append("DS4_MANIFEST_DOES_NOT_BIND_RAW_SHA")
  if audits.get("ds4",{}).get("counts",{}).get("transition_only",{}).get("l20",0)==0:reasons.append("DS4_NO_POST_L20")
  if sum(audits.get("ds7",{}).get("counts",{}).get(p,{}).get("triples",0) for p in ("transition","held","time_push"))==0:reasons.append("DS7_NO_POST_SUPPORT")
  d8=audits.get("ds8",{}).get("counts",{})
  if d8.get("transition",{}).get("triples",0)==d8.get("held",{}).get("triples",0)==0:reasons.append("DS8_NO_TRANSITION_OR_HELD_SUPPORT")
  if d8.get("time_push",{}).get("prns")==1:reasons.append("DS8_TIME_PUSH_ONE_PRN_DIAGNOSTIC_ONLY")
  exact_reason=";".join(reasons)
  if not reasons or go.get("reason")!=exact_reason or execution.get("plots",{}).get("reason")!=exact_reason:errors.append("foundation_reason_derivation")
  expected_not_eval={"status":"NOT_EVALUATED","value":None,"reason":exact_reason}
  if any(load(root/name)!=expected_not_eval for name in ("normal_model_summary.json","thresholds.json","bootstrap_results.json")):errors.append("not_evaluated_exact")
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
 if a.finalize_to:
  destination=a.finalize_to
  if destination.exists():raise FileExistsError(destination)
  stage=Path(tempfile.mkdtemp(prefix=destination.name+".verifying-",dir=destination.parent))
  shutil.rmtree(stage)
  try:
   shutil.copytree(a.artifact,stage,symlinks=True)
   result=verify(stage,True,authenticated_hash_ledger=ledger,integration_diagnostic=ledger is not None)
   if result["status"]!="PASS":raise RuntimeError("copied staging tree failed external verification")
   (stage/"verification_report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
   files={n:{"sha256":digest(stage/n),"size_bytes":(stage/n).stat().st_size} for n in sorted(NAMES-{"checksums.json"})}
   (stage/"checksums.json").write_text(json.dumps({"algorithm":"sha256","files":files},indent=2,sort_keys=True)+"\n")
   closure=verify(stage,True,authenticated_hash_ledger=ledger,integration_diagnostic=ledger is not None)
   if closure["status"]!="PASS":raise RuntimeError("staging closure verification failed")
   for item in stage.rglob("*"):
    if item.is_file():
     with item.open("rb") as handle:os.fsync(handle.fileno())
   for directory in (stage,stage/"plots"):
    fd=os.open(directory,os.O_RDONLY);os.fsync(fd);os.close(fd)
   libc=ctypes.CDLL(None,use_errno=True);rc=libc.renameat2(-100,os.fsencode(stage),-100,os.fsencode(destination),1)
   if rc:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()),destination)
   parent_fd=os.open(destination.parent,os.O_RDONLY);os.fsync(parent_fd);os.close(parent_fd)
  except Exception:
   shutil.rmtree(stage,ignore_errors=True);raise
 else:
  result=verify(a.artifact,not a.skip_external_recompute,authenticated_hash_ledger=ledger,integration_diagnostic=ledger is not None)
 print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result["status"]=="PASS" else 2 if result["status"]=="INCOMPLETE" else 1)
if __name__=="__main__":main()
