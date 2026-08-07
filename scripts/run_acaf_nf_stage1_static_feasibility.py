#!/usr/bin/env python3
"""Produce the deterministic fail-closed Stage-1 feasibility artifact."""
from __future__ import annotations

import argparse, csv, ctypes, errno, hashlib, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.acaf_nf_stage1_static_feasibility import FROZEN_CONFIG,consecutive_windows

DEFAULT_OUTPUT=ROOT/"artifacts/acaf_nf_stage1_static_feasibility.pending"
SOURCE_BINDING_CONFIG=ROOT/"configs/acaf_nf_stage1_source_binding.json"
SOURCE_BINDING_SHA256="af59415f82c3615ebba0690fa9b7a117ea9348bf1d61ab7a95e716d981cef8ae"
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
MANIFEST_POINTERS={s:{"raw_sha256":"/source/iq_sha256","rate":"/source/sample_rate_hz","format":"/source/sample_format"} for s in ("cleanStatic","ds3","ds7")}
MANIFEST_POINTERS["ds8"]={"raw_sha256":"/source/sha256","rate":"/source/sample_rate_hz","format":"/source/format"}
MANIFEST_POINTERS["ds4"]={"raw_sha256":None,"rate":"/source/sample_rate_hz","format":"/source/sample_format"}
TIMELINES={"ds3":{"onset_s":118.9,"pull_off_s":195.0},"ds4":{"onset_s":113.8,"pull_off_s":225.0,"raw_end_approx_s":128.22},"ds7":{"injection_s":110.,"transition":[110.,130.],"held":[130.,150.],"time_push_start_s":150.},"ds8":{"injection_s":110.,"transition":[110.,130.],"held":[130.,150.],"time_push_start_s":150.}}
PHASES={
 "cleanStatic":{"all":[0,None]},
 "ds3":{"pre":[0,2_972_500_000],"onset_to_pulloff":[2_972_500_000,4_875_000_000],"post_pulloff":[4_875_000_000,None]},
 "ds4":{"pre":[0,2_845_000_000],"transition_only":[2_845_000_000,5_625_000_000]},
 "ds7":{"pre":[0,2_750_000_000],"transition":[2_750_000_000,3_250_000_000],"held":[3_250_000_000,3_750_000_000],"time_push":[3_750_000_000,None]},
 "ds8":{"pre":[0,2_750_000_000],"transition":[2_750_000_000,3_250_000_000],"held":[3_250_000_000,3_750_000_000],"time_push":[3_750_000_000,None]}}
EXPECTED_PHASE={
 "cleanStatic":{"all":{"triples":101278,"l20":100864,"prns":11}},
 "ds3":{"pre":{"triples":148699,"l20":148196,"prns":11},"onset_to_pulloff":{"triples":39761,"l20":39604,"prns":5},"post_pulloff":{"triples":5200,"l20":5124,"prns":3}},
 "ds4":{"pre":{"triples":107471,"l20":106914,"prns":11},"transition_only":{"triples":18,"l20":0,"prns":0}},
 "ds7":{"pre":{"triples":100520,"l20":100290,"prns":11},"transition":{"triples":0,"l20":0,"prns":0},"held":{"triples":0,"l20":0,"prns":0},"time_push":{"triples":0,"l20":0,"prns":0}},
 "ds8":{"pre":{"triples":102940,"l20":102662,"prns":10},"transition":{"triples":0,"l20":0,"prns":0},"held":{"triples":0,"l20":0,"prns":0},"time_push":{"triples":288,"l20":232,"prns":1}}}
SCIENCE_CSV={"scenario_metrics.csv":["status","reason","scenario"],"phase_metrics.csv":["status","reason","scenario","phase"],"per_window_scores.csv":["status","reason","scenario","time_s","score"],"secondary_component_metrics.csv":["status","reason","component"],"baseline_metrics.csv":["status","reason","baseline"],"control_metrics.csv":["status","reason","control"]}
TEST_TARGETS=[["tests/test_acaf_nf_stage1_static_feasibility.py"],
              ["tests/test_acaf_nf_stage0.py","tests/test_acaf_nf_stage0_r1.py","tests/test_acaf_nf_stage0_r11_validity.py","tests/test_acaf_nf_stage0_r12_alignment.py","tests/test_acaf_nf_stage0_r13_reconstruction.py","tests/test_acaf_nf_stage0_r14_doppler_validation.py"]]

def digest(path, chunk=8*1024*1024):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  while block:=f.read(chunk):h.update(block)
 return h.hexdigest()

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")

def rename_noreplace(source, destination):
 libc=ctypes.CDLL(None,use_errno=True);fn=libc.renameat2
 result=fn(-100,os.fsencode(source),-100,os.fsencode(destination),1)
 if result:
  code=ctypes.get_errno()
  if code==errno.EEXIST:raise FileExistsError(destination)
  raise OSError(code,os.strerror(code),destination)

def discover_manifest(tracker):
 for parent in [tracker,*tracker.parents[:5]]:
  p=parent/"manifest.json"
  if p.is_file():return p
 return None

def manifest_contains(value, needle):
 if isinstance(value,dict):return any(manifest_contains(v,needle) for v in value.values())
 if isinstance(value,list):return any(manifest_contains(v,needle) for v in value)
 return str(value).lower()==str(needle).lower()

def json_pointer(document,pointer):
 value=document
 for token in pointer.split("/")[1:]:value=value[token.replace("~1","/").replace("~0","~")]
 return value

def authenticate_manifest(document,scenario):
 pointers=MANIFEST_POINTERS[scenario];checks={}
 if pointers["raw_sha256"] is None:checks["tracker_raw_binding"]={"status":"FAIL","reason":"MANIFEST_DOES_NOT_BIND_RAW_SHA","pointer":None}
 else:
  value=json_pointer(document,pointers["raw_sha256"]);checks["tracker_raw_binding"]={"status":"PASS" if value==HASHES[scenario] else "FAIL","reason":None if value==HASHES[scenario] else "RAW_SHA_MISMATCH","pointer":pointers["raw_sha256"]}
 rate=json_pointer(document,pointers["rate"]);fmt=str(json_pointer(document,pointers["format"])).lower()
 checks["sample_rate"]={"status":"PASS" if rate==25_000_000 else "FAIL","pointer":pointers["rate"],"value":rate}
 checks["sample_format"]={"status":"PASS" if "ishort" in fmt and ("iq" in fmt or "complex" in fmt) else "FAIL","pointer":pointers["format"],"value":fmt}
 return checks

def authenticate_receiver_config(path, scenario_config):
 values={}
 for raw_line in Path(path).read_text().splitlines():
  line=raw_line.strip()
  if not line or line.startswith("#") or "=" not in line:continue
  key,value=line.split("=",1);values[key.strip()]=value.strip()
 forbidden=[k for k in values if k.startswith("SignalSource.") and any(x in k.lower() for x in ("skip","header","offset"))]
 expected={"SignalSource.item_type":"ishort","SignalSource.sampling_frequency":"25000000"}
 failures=[]
 if forbidden:failures.append("FORBIDDEN_SKIP_HEADER_OFFSET_KEY")
 if any(values.get(k)!=v for k,v in expected.items()):failures.append("SOURCE_KEYS_MISMATCH")
 if not values.get("SignalSource.filename"):failures.append("MISSING_FILENAME")
 raw_size=scenario_config.get("raw_size_bytes")
 if raw_size is None and scenario_config.get("raw_path") and Path(scenario_config["raw_path"]).is_file():raw_size=Path(scenario_config["raw_path"]).stat().st_size
 try:samples=int(values.get("SignalSource.samples", ""))
 except ValueError:samples=-1
 full_scalar_count=None if raw_size is None else raw_size//2
 covers_full_file=samples==0 or (full_scalar_count is not None and samples==full_scalar_count)
 if samples<0 or not covers_full_file:failures.append("SAMPLES_DO_NOT_COVER_FULL_FILE")
 expected["SignalSource.samples"]="0_or_exact_scalar_int16_full_source_count"
 return {"status":"FAIL" if failures else "PASS","reasons":failures,"keys":{k:values.get(k) for k in ("SignalSource.item_type","SignalSource.sampling_frequency","SignalSource.samples","SignalSource.filename")},"expected_keys":expected,"forbidden_keys":forbidden,
         "configured_filename_alias_classification":"HISTORICAL_RELOCATED_ALIAS",
         "configured_filename_content_sha256":scenario_config["raw_sha256"],"first_file_sample_is_raw_sample":0,
         "count_unit":"scalar_int16","configured_count":samples,"full_source_count":full_scalar_count,
         "covers_full_file":covers_full_file}

def _field(handle,name):
 if name not in handle:return None
 return np.asarray(handle[name]).reshape(-1)

REQUIRED={"stamp":"PRN_start_sample_count","prn":"PRN","carrier_doppler_hz":"carrier_doppler_hz","code_freq_chips":"code_freq_chips","aux1":"aux1","prompt_i":"Prompt_I","prompt_q":"Prompt_Q","cn0":"CN0_SNV_dB_Hz","lock":"carrier_lock_test"}
def tracker_rows(tracker:Path, raw_sample_count=None, inventory_config=None, scenario=None):
 rows=[];inventory=[]
 tracker=Path(tracker)
 if tracker.is_symlink() or not tracker.is_dir():raise ValueError("tracker root must be a real directory")
 entries=list(tracker.iterdir())
 aliases=[]
 if inventory_config is not None:
  canonical=set(inventory_config["mat_inventory"])
  ignored=set(inventory_config.get("ignored_alias_symlinks",{}))
  observations=set(inventory_config["_observables_noninputs"])
  dats={Path(n).with_suffix(".dat").name for n in canonical}
  if {p.name for p in entries} != canonical|ignored|observations|dats:raise ValueError("exact tracker inventory mismatch")
  for p in entries:
   if p.name in canonical and (p.is_symlink() or not p.is_file()):raise ValueError("canonical MAT must be regular")
   elif p.name in ignored:
    if not p.is_symlink():raise ValueError("expected ignored alias symlink")
    raw_target=os.readlink(p);resolved=(p.parent/raw_target).resolve(strict=True)
    relative=resolved.relative_to(tracker.resolve()).as_posix()
    if relative!=inventory_config["ignored_alias_symlinks"][p.name]:raise ValueError("alias target mismatch")
    aliases.append({"filename":p.name,"link_target":raw_target,"resolved_relative_target":relative,"ignored_noninput":True})
   elif p.name in observations|dats:
    if p.is_symlink() or not p.is_file():raise ValueError("noninput must be regular")
   elif p.is_symlink() or not p.is_file():raise ValueError("unexpected tracker node")
  mats=sorted(tracker/n for n in canonical)
 else:
  if any(p.is_symlink() or not p.is_file() for p in entries):raise ValueError("unexpected nested/symlink/special tracker entry")
  mats=sorted(p for p in entries if p.suffix==".mat" and p.stem not in {"observables","obse"})
 if not mats:raise ValueError("no tracker MAT files")
 for mat in mats:
  with h5py.File(mat,"r") as f:
   values={k:_field(f,v) for k,v in REQUIRED.items()}
   missing=[REQUIRED[k] for k,v in values.items() if v is None]
   if missing:raise ValueError(f"{mat.name}: missing required fields {missing}")
   n=len(values["stamp"])
   if n<3 or any(len(v)!=n for v in values.values()):raise ValueError(f"{mat.name}: required field length mismatch")
   if any(v.dtype.kind not in "iuf" or not np.isfinite(v).all() for v in values.values()):raise ValueError(f"{mat.name}: required field type/nonfinite")
   if not np.equal(values["stamp"],np.floor(values["stamp"])).all():raise ValueError(f"{mat.name}: noninteger stamp")
   schema=[{"name":name,"dtype":str(f[name].dtype),"shape":list(f[name].shape)} for name in sorted(f.keys())]
   schema_digest=hashlib.sha256(json.dumps(schema,separators=(",",":"),sort_keys=True).encode()).hexdigest()
   entry={"filename":mat.name,"sha256":digest(mat),"size_bytes":mat.stat().st_size,"rows":n,"required_field_schema":{"digest_sha256":schema_digest,"fields":schema}}
   if inventory_config is not None and entry["sha256"]!=inventory_config["mat_inventory"][mat.name]:raise ValueError("pinned canonical MAT hash mismatch")
   inventory.append(entry)
   starts=values["stamp"].astype(np.int64);prns=values["prn"]
   channel=mat.stem
   for i in range(1,n-1):
    start=int(starts[i-1]);end=start+25_000;consumed=int(starts[i])
    finite=all(np.isfinite(values[k][j]) for k in values for j in (i-1,i,i+1))
    rows.append({"channel":channel,"row":i,"prn":int(prns[i]),"start":start,"end":end,"consumed_end":consumed,
     "delta_previous":int(starts[i]-starts[i-1]),"delta_next":int(starts[i+1]-starts[i]),
     "same_prn_triple":bool(int(prns[i-1])==int(prns[i])==int(prns[i+1])),
     "prn_valid":bool(all(1<=int(prns[j])<=32 for j in (i-1,i,i+1))),"finite_required":bool(finite),
     "min_triple_cn0":float(min(values["cn0"][i-1:i+2])),"min_triple_lock":float(min(values["lock"][i-1:i+2])),
     "raw_bounds":bool(start>=0 and (raw_sample_count is None or end<=raw_sample_count))})
 return rows,{"canonical_mat_files":inventory,"ignored_alias_symlinks":sorted(aliases,key=lambda x:x["filename"]),"ignored_regular_noninputs":sorted(observations|dats) if inventory_config is not None else []}

def phase_name(scenario,sample):
 if isinstance(sample,float) and not sample.is_integer():sample=round(sample*25_000_000)
 sample=int(sample)
 for name,(a,b) in PHASES[scenario].items():
  if sample>=a and (b is None or sample<b):return name
 raise ValueError((scenario,sample))

def support_audit(tracker,scenario,raw_sample_count=None,inventory_config=None):
 rows,inventory=tracker_rows(tracker,raw_sample_count,inventory_config,scenario);accepted=[];excluded={}
 for r in rows:
  reasons=[]
  if not r["same_prn_triple"]:reasons.append("MIXED_PRN_TRIPLE")
  if not r["prn_valid"]:reasons.append("PRN_OUT_OF_RANGE")
  if not r["finite_required"]:reasons.append("NONFINITE_REQUIRED_FIELD")
  if not 24_999<=r["delta_previous"]<=25_001 or not 24_999<=r["delta_next"]<=25_001:reasons.append("STAMP_DELTA")
  if r["min_triple_cn0"]<28:reasons.append("CN0")
  if r["min_triple_lock"]<.85:reasons.append("LOCK")
  if r["consumed_end"]!=r["start"]+r["delta_previous"]:reasons.append("CONSUMED_END")
  if r["end"]-r["start"]!=25_000 or not r["raw_bounds"]:reasons.append("RAW_BOUNDS")
  if not reasons and (r["same_prn_triple"] and 24_999<=r["delta_previous"]<=25_001
      and 24_999<=r["delta_next"]<=25_001 and r["min_triple_cn0"]>=28
      and r["min_triple_lock"]>=.85):accepted.append(r)
  else:
   for reason in reasons:excluded[reason]=excluded.get(reason,0)+1
 counts={}
 crossing=sum(phase_name(scenario,r["start"])!=phase_name(scenario,r["end"]-1) for r in accepted)
 for phase in EXPECTED_PHASE[scenario]:
  selected=[r for r in accepted if phase_name(scenario,r["start"])==phase and phase_name(scenario,r["end"]-1)==phase]
  eligible=[{"channel":r["channel"],"prn":r["prn"],"tracker_row":r["row"],"phase":phase,"support_start_sample":r["start"],"support_end_sample":r["end"],"cn0_db_hz":r["min_triple_cn0"],"carrier_lock":r["min_triple_lock"]} for r in selected]
  per_prn={}
  groups={}
  for r in eligible:groups.setdefault((r["channel"],r["prn"]),[]).append(r)
  for group in groups.values():
   group.sort(key=lambda r:r["support_start_sample"])
   for j in range(19,len(group)):
    w=group[j-19:j+1]
    if (len({r["tracker_row"] for r in w})==20
        and all(b["tracker_row"]==a["tracker_row"]+1 and 24999<=b["support_start_sample"]-a["support_start_sample"]<=25001 for a,b in zip(w,w[1:]))
        and phase_name(scenario,w[0]["support_start_sample"])==phase_name(scenario,w[-1]["support_end_sample"]-1)==phase):per_prn[w[-1]["prn"]]=per_prn.get(w[-1]["prn"],0)+1
  total=sum(per_prn.values())
  counts[phase]={"triples":len(selected),"l20":total,"prns":len(per_prn),"dominant_fraction":max(per_prn.values(),default=0)/total if total else 0.,"crossing_excluded":sum(phase_name(scenario,r["start"])==phase and phase_name(scenario,r["end"]-1)!=phase for r in accepted)}
 return {"scenario":scenario,"counts":counts,"expected":EXPECTED_PHASE[scenario],"matches_expected":all({k:counts[p][k] for k in ("triples","l20","prns")}==e for p,e in EXPECTED_PHASE[scenario].items()),"excluded":excluded,"boundary_crossing_supports":crossing,"mat_inventory":inventory,"accepted_rule":"previous-row support; exact triple; deltas 24999..25001; finite full schema; quality every row; bounds; half-open phase containment","twenty_ms_gaps_interpolated":False}

def foundation_gate(audits):
 """Derive the source-only foundation verdict; scoring is not in this API."""
 failures=[]
 for name in ("cleanStatic","ds3","ds4","ds7","ds8"):
  if name not in audits or not audits[name].get("matches_expected"):failures.append(f"{name}_support_audit_mismatch")
 if audits.get("ds4",{}).get("tracker_raw_binding_status")=="FAIL":failures.insert(0,"DS4_MANIFEST_DOES_NOT_BIND_RAW_SHA")
 for name in audits:
  if audits[name].get("receiver_config_status")=="FAIL":failures.append(name+"_RECEIVER_CONFIG_TIME_ORIGIN_FAIL")
 if audits.get("ds4",{}).get("counts",{}).get("transition_only",{}).get("l20",0)==0:failures.append("DS4_NO_POST_L20")
 if sum(audits.get("ds7",{}).get("counts",{}).get(p,{}).get("triples",0) for p in ("transition","held","time_push"))==0:failures.append("DS7_NO_POST_SUPPORT")
 d8=audits.get("ds8",{}).get("counts",{})
 if d8.get("transition",{}).get("triples",0)==d8.get("held",{}).get("triples",0)==0:failures.append("DS8_NO_TRANSITION_OR_HELD_SUPPORT")
 if d8.get("time_push",{}).get("prns") == 1:failures.append("DS8_TIME_PUSH_ONE_PRN_DIAGNOSTIC_ONLY")
 return {"verdict":"FOUNDATION_INVALID" if failures else "FOUNDATION_PASS","reasons":failures,"no_attack_raw_scoring_performed":True}

def _foundation_gate_with_sentinel_for_test(audits, callback):
 result=foundation_gate(audits)
 if result["verdict"]!="FOUNDATION_PASS":return result
 return callback()

def write_csv_header(path,fields):
 with path.open("w",newline="") as f:csv.DictWriter(f,fieldnames=fields).writeheader()

def publish(args, *, audit_function=support_audit):
 output=Path(args.output)
 if output.exists():raise FileExistsError(f"refusing existing output: {output}")
 output.parent.mkdir(parents=True,exist_ok=True)
 staging=Path(tempfile.mkdtemp(prefix=output.name+".staging-",dir=output.parent))
 try:
  if SOURCE_BINDING_SHA256=="TO_BE_FINALIZED" or digest(SOURCE_BINDING_CONFIG)!=SOURCE_BINDING_SHA256:raise RuntimeError("pinned source-binding config drift")
  source_config=json.loads(SOURCE_BINDING_CONFIG.read_text())
  raw_paths={s:Path(getattr(args,"raw_"+("clean" if s=="cleanStatic" else s))) for s in TRACKERS}
  tracker_paths={s:Path(getattr(args,"tracker_"+("clean" if s=="cleanStatic" else s))) for s in TRACKERS}
  bindings={};audits={}
  for scenario in TRACKERS:
   raw=raw_paths[scenario]
   configured=source_config["scenarios"][scenario]
   if raw!=Path(configured["raw_path"]):raise RuntimeError(f"{scenario} artifact-controlled raw path rejected")
   if not raw.is_file():raise FileNotFoundError(raw)
   actual=digest(raw)
   if actual!=HASHES[scenario]:raise RuntimeError(f"{scenario} raw SHA-256 mismatch")
   if tracker_paths[scenario]!=Path(configured["tracker_path"]):raise RuntimeError(f"{scenario} artifact-controlled tracker path rejected")
   manifest=Path(configured["manifest_path"])
   receiver_config=Path(configured["receiver_config_path"])
   if not manifest.is_file() or not receiver_config.is_file():raise RuntimeError(f"{scenario} configured metadata unavailable")
   if digest(manifest)!=configured["manifest_sha256"] or digest(receiver_config)!=configured["receiver_config_sha256"]:raise RuntimeError(f"{scenario} pinned metadata hash mismatch")
   manifest_doc=json.loads(manifest.read_text())
   manifest_checks=authenticate_manifest(manifest_doc,scenario)
   if scenario!="ds4" and any(x["status"]!="PASS" for x in manifest_checks.values()):raise RuntimeError(f"{scenario} exact manifest binding failed")
   size=raw.stat().st_size
   raw_checks={"size_divisible_by_4":{"status":"PASS" if size%4==0 else "FAIL"},"format":{"status":"PASS" if manifest_checks["sample_format"]["status"]=="PASS" else "FAIL"},"rate":{"status":"PASS" if manifest_checks["sample_rate"]["status"]=="PASS" else "FAIL"},"sample_count":size//4}
   configured["_observables_noninputs"]=source_config["tracker_inventory_contract"]["observables_noninputs_by_scenario"][scenario]
   audits[scenario]=audit_function(tracker_paths[scenario],scenario,size//4,configured)
   audits[scenario]["tracker_raw_binding_status"]=manifest_checks["tracker_raw_binding"]["status"]
   receiver_checks=authenticate_receiver_config(receiver_config,{**configured,"raw_size_bytes":size})
   audits[scenario]["receiver_config_status"]=receiver_checks["status"]
   bindings[scenario]={"raw_path":str(raw),"raw_sha256":actual,"expected_raw_sha256":HASHES[scenario],"raw_size_bytes":size,"raw_sample_count":size//4,"raw_count_unit":"complex_int16_iq","raw_bytes_read_purpose":"full_sha256_only","raw_checks":raw_checks,"tracker_path":str(tracker_paths[scenario]),"manifest_path":str(manifest),"manifest_sha256":digest(manifest),"receiver_config_path":str(receiver_config),"receiver_config_sha256":digest(receiver_config),"manifest_checks":manifest_checks,"receiver_config_checks":receiver_checks,"tracker_raw_binding":manifest_checks["tracker_raw_binding"]}
  verdict=foundation_gate(audits)
  if verdict["verdict"]!="FOUNDATION_INVALID":raise RuntimeError("this producer is authorized only for fail-closed artifact")
  config={"frozen":FROZEN_CONFIG.document(),"source_binding_config_sha256":SOURCE_BINDING_SHA256,"search_complexity":{"calibration_statistic":"full_minimized_delta_search","scalar_penalty":0.0},"delay_grid":list(np.arange(-1,1.0001,.125)),"doppler_grid_hz":list(range(-250,251,50)),"pooling_candidates":["median","top50_mean","trimmed_mean"],"baseline_B0":"PROVISIONAL_UNAVAILABLE"}
  dump(staging/"config.json",config);dump(staging/"source_binding.json",{"sources":bindings,"tracker_support_audits":audits,"ds7_ds8_pre_attack_pairing":"paired replay diagnostic only if byte identity authenticated"})
  versions={"python":sys.version.split()[0],"numpy":np.__version__,"h5py":h5py.__version__}
  for package in ("scipy","sklearn","matplotlib"):
   module=__import__(package);versions[package]=module.__version__
  git_head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();git_dirty=bool(subprocess.run(["git","status","--porcelain"],cwd=ROOT,text=True,capture_output=True,check=True).stdout)
  source_hashes={"producer_sha256":digest(Path(__file__)),"module_sha256":digest(ROOT/"src/gnss_doppler_lab/acaf_nf_stage1_static_feasibility.py"),"verifier_sha256":digest(ROOT/"scripts/verify_acaf_nf_stage1_static_feasibility.py"),"config_sha256":digest(SOURCE_BINDING_CONFIG)}
  dump(staging/"r14_frozen_lineage.json",{"artifact":str(R14_ARTIFACT),"artifact_checksums_sha256":source_config["r14"]["checksums_sha256"],"verification_report_sha256":source_config["r14"]["verification_report_sha256"],"verifier_sha256":source_config["r14"]["verifier_sha256"],"module_sha256":source_config["r14"]["module_sha256"],"runner_sha256":source_config["r14"]["runner_sha256"],"verifier_required":"PASS","contract":FROZEN_CONFIG.document(),"base_sha":source_config["base_commit"],"git_head":git_head,"git_dirty":git_dirty,"stage1_source_hashes":source_hashes,"versions":versions,"status":"PASS"})
  dump(staging/"scenario_timeline.json",TIMELINES)
  reason=";".join(verdict["reasons"])
  not_eval={"status":"NOT_EVALUATED","value":None,"reason":reason}
  dump(staging/"normal_model_summary.json",not_eval);dump(staging/"thresholds.json",not_eval);dump(staging/"bootstrap_results.json",not_eval)
  for name,fields in SCIENCE_CSV.items():write_csv_header(staging/name,fields)
  dump(staging/"go_no_go.json",{"verdict":"FOUNDATION_INVALID","PHYSICS_FEASIBILITY_GO":False,"physics_feasibility_status":"NOT_EVALUATED","PAPER_CANDIDATE_GO":False,"paper_candidate_status":"NOT_EVALUATED","stage2_justified":False,"reason":reason})
  dump(staging/"execution_validity.json",{"status":"FOUNDATION_INVALID","no_attack_raw_scoring_performed":True,"attack_iq_bytes_read_for_scoring":0,"raw_bytes_read_purpose":"full_sha256_only","science_csv_semantics":"header_only","plots":{"count":0,"reason":reason},"B0":"PROVISIONAL_UNAVAILABLE"})
  dump(staging/"verification_report.json",{"status":"PENDING_INDEPENDENT_VERIFICATION","producer_verdict_not_authoritative":True})
  (staging/"plots").mkdir()
  report_path=Path(args.test_report_input)
  report=json.loads(report_path.read_text())
  report_keys={"schema","source_head","versions","environment","runs"};run_keys={"command","stdout","exit_code","collected","passed","failed"}
  expected_commands=[[sys.executable,"-m","pytest","-q",*targets] for targets in TEST_TARGETS]
  if (set(report)!=report_keys or report.get("schema")!="acaf-stage1-test-report-v1"
      or report.get("source_head")!=git_head or report.get("versions")!=versions
      or report.get("environment")!={"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
      or not isinstance(report.get("runs"),list) or len(report["runs"])!=2
      or any(set(run)!=run_keys or run.get("command")!=command or run.get("exit_code")!=0
             or type(run.get("collected")) is not int or run.get("collected")<=0
             or run.get("passed")!=run.get("collected") or run.get("failed")!=0
             or not isinstance(run.get("stdout"),str) or not run["stdout"].strip()
             for run,command in zip(report["runs"],expected_commands))):
   raise ValueError("test report is not an exact successful source-phase capture")
  (staging/"test_report.txt").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
  (staging/"README.md").write_text("# ACAF-NF Stage-1 static feasibility\n\n`FOUNDATION_INVALID` is a source-support finding, not a physics `NO_GO`. Attack raw IQ was full-hashed only and was never scored. Science CSVs are header-only, model/threshold/bootstrap are `NOT_EVALUATED`, and `plots/` is intentionally empty. `PHYSICS_FEASIBILITY_GO=false` and `PAPER_CANDIDATE_GO=false` mean not evaluated. Stage-2 is not justified until independently validated continuous 1 ms tracker/source binding exists. B0 is `PROVISIONAL_UNAVAILABLE` because its exact evaluator interface, support, and threshold lineage were not authenticated.\n")
  files={str(p.relative_to(staging)):{"sha256":digest(p),"size_bytes":p.stat().st_size} for p in sorted(staging.rglob("*")) if p.is_file() and p.name!="checksums.json"}
  dump(staging/"checksums.json",{"algorithm":"sha256","files":files})
  for p in (staging,staging/"plots"):
   fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)
  rename_noreplace(staging,output)
  return verdict
 except Exception:
  shutil.rmtree(staging,ignore_errors=True);raise

def parser():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);p.add_argument("--execute-production",action="store_true");p.add_argument("--test-report-input",type=Path,required=True)
 for s in TRACKERS:
  suffix="clean" if s=="cleanStatic" else s;p.add_argument("--raw-"+suffix,type=Path,default=RAW_ROOT/(s+".bin"));p.add_argument("--tracker-"+suffix,type=Path,default=TRACKERS[s])
 return p

def main(argv=None):
 args=parser().parse_args(argv)
 if not args.execute_production:raise SystemExit("refusing production I/O without --execute-production")
 if subprocess.run(["git","status","--porcelain"],cwd=ROOT,text=True,capture_output=True,check=True).stdout:raise SystemExit("source stage must be clean at production")
 report=subprocess.run([sys.executable,str(ROOT/"scripts/verify_acaf_nf_stage0_static_r14_doppler_validation.py"),str(R14_ARTIFACT)],capture_output=True,text=True)
 if report.returncode:raise SystemExit("R1.4 verifier did not PASS")
 print(json.dumps(publish(args),indent=2,sort_keys=True))
if __name__=="__main__":main()
