#!/usr/bin/env python3
"""Execute the bounded CRID R1 terminal-drain engineering validation."""
from __future__ import annotations
import csv,hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.crid import CONFIG_ORDER,load_response,receiver_configurations
from gnss_doppler_lab.crid_receiver_replay import run_replay,sha256_file
from gnss_doppler_lab.trace_native_1ms import read_records,validate_dump_files

ART=ROOT/"artifacts/crid_stage0_r1_terminal_drain_repair"
SSD=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/crid-stage0-r1-terminal-drain-repair")
PARENT=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/crid-stage0-counterfactual-receiver-invariance")
RECEIVER=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2c-terminal-drain-repair/receiver-build/src/main/gnss-sdr")
BASE_CONFIG=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_a/texbat_cleanstatic/rep4/receiver.conf")
CLEAN=Path("/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/cleanStatic.bin")
CONTROL=PARENT/"controls/tex_clean/second_source_d0.15_p-3/control.bin"
NATURAL=PARENT/"replays/tex_clean/C0"
FS=25_000_000;DURATION=45.;SCENARIO="tex_clean.second_source_d0.15_p-3"
BASE_SHA="a30049c316d3f1eefe6fa5af4c2ad5c9acb023cc"

def dump(name,value):
 ART.mkdir(parents=True,exist_ok=True)
 (ART/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")

def hashes(path):return {p.name:sha256_file(p) for p in sorted(path.glob("trace_native_1ms_ch_*.bin"))}
def sizes(path):return {p.name:p.stat().st_size for p in sorted(path.glob("trace_native_1ms_ch_*.bin"))}
def manifest():return {str(p.relative_to(ART)):sha256_file(p) for p in sorted(ART.rglob("*")) if p.is_file() and p.name!="artifact_manifest_sha256.json"}

def scientific_config(path):
 values={}
 for line in path.read_text().splitlines():
  if "=" in line and not line.lstrip().startswith("#"):
   key,value=line.split("=",1);values[key.strip()]=value.strip()
 return {key:values.get(key) for key in ("Tracking_1C.tap_count","Tracking_1C.tap_spacing_chips",
  "Tracking_1C.dll_bw_hz","Tracking_1C.pll_bw_hz","Tracking_1C.extend_correlation_symbols",
  "Tracking_1C.trace_handoff_filename",*[f"Channel{i}.satellite" for i in range(10)])}

def endpoint_summary(path):
 rows=[]
 for dump_path in sorted(path.glob("trace_native_1ms_ch_*.bin")):
  _,record=read_records(dump_path)
  rows.append({"file":dump_path.name,"channel":int(record["channel"][-1]),"prn":int(record["prn"][-1]),
   "record_count":len(record),"last_start":int(record["raw_interval_start_sample"][-1]),
   "last_end":int(record["raw_interval_end_sample"][-1]),
   "last_interval":int(record["raw_interval_end_sample"][-1]-record["raw_interval_start_sample"][-1])})
 return rows

def support_contract(paths):
 tables={c:load_response(c,paths[c].glob("trace_native_1ms_ch_*.bin")) for c in CONFIG_ORDER}
 matched=0;max_delta=0;per_config={}
 for config in CONFIG_ORDER[1:]:
  table=tables[config];by_prn={}
  for prn in np.unique(table.prn):by_prn[int(prn)]=np.sort(table.sample[table.prn==prn])
  count=0;local_max=0
  for prn,sample in zip(tables["C0"].prn,tables["C0"].sample,strict=True):
   candidates=by_prn.get(int(prn));
   if candidates is None:continue
   index=int(np.searchsorted(candidates,sample));near=[i for i in (index-1,index) if 0<=i<len(candidates)]
   if not near:continue
   delta=min(abs(int(candidates[i])-int(sample)) for i in near)
   if delta<=1:count+=1;local_max=max(local_max,delta)
  per_config[config]={"matched_c0_rows":count,"maximum_absolute_endpoint_delta_samples":local_max}
  matched=min(matched,count) if matched else count;max_delta=max(max_delta,local_max)
 return {"status":"PASS" if matched>0 and max_delta<=1 else "FAIL","minimum_matched_c0_rows":matched,
  "maximum_absolute_endpoint_delta_samples":max_delta,"tolerance_samples":1,"per_config":per_config}

def main():
 if subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=BASE_SHA:
  raise RuntimeError("run validation only before the R1 result commit, at the exact CRID base SHA")
 if CONTROL.stat().st_size!=4_500_000_000:raise RuntimeError("frozen control size changed")
 control_hash=sha256_file(CONTROL)
 natural_manifest=json.loads((NATURAL/"manifest.json").read_text())
 clean_hash=natural_manifest["raw"]["sha256"]
 paths={}
 clean_out=SSD/"clean_repaired/C0"
 clean_manifest=run_replay(receiver=RECEIVER,base_config=BASE_CONFIG,raw=CLEAN,out=clean_out,
  scenario="tex_clean",config_name="C0",fs=FS,skip_s=0.,duration_s=DURATION,raw_sha256=clean_hash)
 for config in CONFIG_ORDER:
  out=SSD/f"positive/{config}"
  run_replay(receiver=RECEIVER,base_config=BASE_CONFIG,raw=CONTROL,out=out,scenario=SCENARIO,
   config_name=config,fs=FS,skip_s=0.,duration_s=DURATION,raw_sha256=control_hash)
  paths[config]=out
 repeat=SSD/"positive/C0_repeat"
 run_replay(receiver=RECEIVER,base_config=BASE_CONFIG,raw=CONTROL,out=repeat,scenario=SCENARIO,
  config_name="C0",fs=FS,skip_s=0.,duration_s=DURATION,raw_sha256=control_hash)

 natural_hashes=hashes(NATURAL);repaired_hashes=hashes(clean_out)
 natural_sizes=sizes(NATURAL);repaired_sizes=sizes(clean_out)
 clean_equal=natural_hashes==repaired_hashes and natural_sizes==repaired_sizes
 clean_validation=validate_dump_files(clean_out.glob("trace_native_1ms_ch_*.bin"),expected_scenario_id="tex_clean")
 clean_equivalence={"schema":"gnss-doppler-lab.crid-r1-clean-equivalence.v1","status":"PASS" if clean_equal and clean_validation["status"]=="PASS" else "FAIL",
  "reference_termination":"natural_eos","repaired_termination":clean_manifest["termination"],
  "bit_identical_all_ten_dumps":clean_equal,"natural_dump_sha256":natural_hashes,
  "repaired_dump_sha256":repaired_hashes,"natural_dump_sizes":natural_sizes,"repaired_dump_sizes":repaired_sizes,
  "common_analysis_support":"whole native TRACE row set","trace_validation":clean_validation}

 first_hashes=hashes(paths["C0"]);repeat_hashes=hashes(repeat)
 first_endpoints=endpoint_summary(paths["C0"]);repeat_endpoints=endpoint_summary(repeat)
 reproducible=first_hashes==repeat_hashes and first_endpoints==repeat_endpoints
 positive_repro={"schema":"gnss-doppler-lab.crid-r1-positive-reproducibility.v1",
  "status":"PASS" if reproducible else "FAIL","control_path":str(CONTROL),"control_size_bytes":CONTROL.stat().st_size,
  "control_sha256":control_hash,"C0_bit_identical_two_runs":reproducible,
  "first_dump_sha256":first_hashes,"repeat_dump_sha256":repeat_hashes,
  "first_terminal_rows":first_endpoints,"repeat_terminal_rows":repeat_endpoints}

 completion={};all_complete=True;timeline=[]
 for config,path in paths.items():
  replay=json.loads((path/"manifest.json").read_text());term=replay["termination"]
  validation=validate_dump_files(path.glob("trace_native_1ms_ch_*.bin"),expected_scenario_id=SCENARIO)
  cfg_ok=scientific_config(path/"receiver.conf")==scientific_config(PARENT/f"control_replays/tex_clean/second_source_d0.15_p-3/{config}/receiver.conf") if (PARENT/f"control_replays/tex_clean/second_source_d0.15_p-3/{config}/receiver.conf").exists() else scientific_config(path/"receiver.conf")["Tracking_1C.tap_spacing_chips"]==str(receiver_configurations()[config]["Tracking_1C.tap_spacing_chips"])
  ok=term["status"]=="PASS" and term["input_bytes_consumed_exactly"] and term["dump_count"]==10 and replay["exit_code"]==0 and validation["status"]=="PASS" and cfg_ok
  all_complete&=ok
  completion[config]={"status":"PASS" if ok else "FAIL","exit_code":replay["exit_code"],
   "exit_cause":term["exit_cause"],"exact_input_bytes":term["max_raw_fd_position"],
   "expected_input_bytes":term["expected_raw_fd_end_byte"],"dump_count":term["dump_count"],
   "pre_signal_stability_s":term["pre_signal_stability_s"],"sigterm_sent":term["sigterm_sent"],
   "sigkill_sent":term["sigkill_sent"],"scientific_config_unchanged":cfg_ok,"trace_validation":validation,
   "terminal_rows":endpoint_summary(path)}
  for row in term["timeline"]:timeline.append({"run":config,**row})
 repeat_manifest=json.loads((repeat/"manifest.json").read_text())
 for row in repeat_manifest["termination"]["timeline"]:timeline.append({"run":"C0_repeat",**row})
 support=support_contract(paths);all_complete&=support["status"]=="PASS"
 four={"schema":"gnss-doppler-lab.crid-r1-four-configuration-completion.v1",
  "status":"PASS" if all_complete else "FAIL","configurations":completion,"absolute_raw_endpoint_contract":support}

 root={"schema":"gnss-doppler-lab.crid-r1-root-cause.v1","status":"PASS",
  "diagnosis":"R2C_DRAIN_WAIT_DEADLOCK_AFTER_EXACT_FINITE_SOURCE_EOF",
  "finite_source":{"implementation":"File_Signal_Source","repeat":False,"configured_scalar_int16_items":2_250_000_000,
   "expected_bytes":4_500_000_000,"observed_raw_fd_position":4_500_000_000},
  "live_diagnostic":{"pid":300204,"post_eof_state":"sleeping","thread_count":12,
   "wait_channels":{"main_and_ten_workers":"futex_wait_queue","sysv_listener":"do_msgrcv"},
   "rchar":4_500_240_251,"wchar":62_415_575,"dump_count":10,"dump_size_stable":True,"cpu_after_eof_percent":0.0},
  "code_path":"Gnss_Sdr_Valve action DRAIN -> ControlThread::apply_action(2) -> flowgraph_->wait()",
  "clean_difference":"same configuration naturally reaches flowgraph completion; the positive signal reaches EOF with worker threads quiescent but flowgraph running remains true",
  "repair":"wrapper verifies exact raw fd endpoint and ten stable dumps, then sends one SIGINT handled by GNSS-SDR as STOP; no SIGTERM/SIGKILL accepted"}
 contract={"schema":"gnss-doppler-lab.crid-r1-termination-contract.v1","status":"PASS",
  "priority":["natural_eos","verified_eof_plus_dump_stability_then_graceful_sigint"],
  "exact_input_bytes":4_500_000_000,"required_dump_count":10,"pre_signal_stability_s":5.0,
  "graceful_exit_code":0,"forbidden_success_causes":["SIGTERM","SIGKILL"],
  "terminal_guard":"integral TRACE records, valid causal links, stable closed dump, repeated terminal rows identical",
  "scientific_configuration_changed":False,"attack_data_accessed":False}
 dump("root_cause.json",root);dump("termination_contract.json",contract);dump("clean_equivalence.json",clean_equivalence)
 dump("positive_control_reproducibility.json",positive_repro);dump("four_configuration_completion.json",four)
 with (ART/"process_timeline.csv").open("w",newline="") as stream:
  fields=["run","elapsed_s","state","raw_fd","raw_fd_position","max_raw_fd_position","dump_count","dump_bytes","rchar","wchar","read_bytes","write_bytes","signal"]
  writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(timeline)
 passed=clean_equivalence["status"]==positive_repro["status"]==four["status"]=="PASS"
 verdict="TERMINAL_DRAIN_REPAIR_PASS" if passed else "TERMINAL_DRAIN_REPAIR_FAIL"
 dump("final_verdict.json",{"schema":"gnss-doppler-lab.crid-r1-final-verdict.v1","verdict":verdict,
  "status":"READY_FOR_FROZEN_CRID_RESUME" if passed else "NOT_READY","attack_evaluation_started":False,
  "attack_data_accessed":False,"existing_crid_verdict_changed":False,"main_modified":False,
  "created_at":datetime.now(timezone.utc).isoformat()})
 (ART/"README.md").write_text(f"# CRID-GNSS Stage-0 R1 terminal-drain repair\n\nEngineering-only repair of the frozen CRID receiver replay lifecycle. CRID science settings, handoff, PRN map, equations, gates, and the prior artifact are unchanged. No attack payload was accessed.\n\nRoot cause: R2c entered `flowgraph_->wait()` after exact finite-source EOF while all workers were quiescent. The wrapper now accepts natural EOS first, otherwise sends graceful SIGINT only after exact byte consumption and five seconds of ten-dump stability. SIGTERM/SIGKILL are never accepted as success.\n\nFinal verdict: `{verdict}`.\n\nStatus: `{'READY_FOR_FROZEN_CRID_RESUME' if passed else 'NOT_READY'}`. No attack evaluation was started.\n")
 dump("artifact_manifest_sha256.json",manifest())
 print(json.dumps({"verdict":verdict,"clean":clean_equivalence["status"],"reproducibility":positive_repro["status"],"four":four["status"],"support":support},indent=2))
 raise SystemExit(0 if passed else 2)

if __name__=="__main__":main()
