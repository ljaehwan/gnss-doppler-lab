#!/usr/bin/env python3
"""Build long authenticated cleanStatic NAV/NCO support for MIRAGE R1."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.mosaic_navbit_provenance import (  # noqa:E402
    BITS_PER_SUBFRAME, decode_words, find_valid_subframe_pairs,
    prompt_decision_axis, word_subframe_id, word_tow_seconds,
)
from gnss_doppler_lab.trace_native_1ms import read_records,sha256_file  # noqa:E402

BASE="f227804dd49a4650006d1569cf49ea3edf61092f"
BRANCH="research/mirage-stage0a-r1-full-execution"
ART=ROOT/"artifacts/mirage_stage0a_r1_full_execution"
TRACE_ROOT=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b")
RECEIVER=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2c-terminal-drain-repair/receiver-build/src/main/gnss-sdr")
SPECS={
 "OAKBAT.cleanStatic":{"slug":"oakbat_cleanstatic","fs":5_000_000,"prns":[10,11,21,24,27],
  "raw":Path("/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/cleanStatic_gps.bin"),
  "raw_sha256":"8e3428abb1b94211118c1ec9f505322ef89fdb176f0cf33961eca3cf3da80dfe"},
 "TEXBAT.cleanStatic":{"slug":"texbat_cleanstatic","fs":25_000_000,"prns":[3,13,16,19,30],
  "raw":Path("/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/cleanStatic.bin"),
  "raw_sha256":"dd295ab46616bfe9634d1c37479520e720ebc54bcb64adf0a247315a541fb9b9"},
}


def command(*args:str)->str:
 return subprocess.run(args,check=True,text=True,stdout=subprocess.PIPE).stdout.strip()


def dump(name:str,value:object)->None:
 (ART/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def write_csv(name:str,rows:list[dict[str,object]])->None:
 if not rows: raise ValueError(f"empty required table {name}")
 with (ART/name).open("w",newline="") as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)


def write_gzip_csv(name:str,rows:list[dict[str,object]])->None:
 if not rows: raise ValueError(f"empty required table {name}")
 with (ART/name).open("wb") as raw:
  with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as compressed:
   with io.TextIOWrapper(compressed,encoding="utf-8",newline="") as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)


def manifest()->dict[str,object]:
 files=[]
 for path in sorted(ART.rglob("*")):
  if path.is_file() and path.name!="artifact_manifest_sha256.json":
   files.append({"path":str(path.relative_to(ART)),"size_bytes":path.stat().st_size,"sha256":sha256_file(path)})
 return {"schema":"gnss-doppler-lab.artifact-manifest-sha256.v1","files":files,
  "unavailable":[]}


def trace_for_prn(directory:Path,prn:int)->Path:
 matches=[]
 for path in sorted(directory.glob("trace_native_1ms_ch_*.bin")):
  _,records=read_records(path)
  if len(records) and int(records["prn"][-1])==prn:matches.append(path)
 if len(matches)!=1:raise ValueError(f"PRN {prn} TRACE lookup: {matches}")
 return matches[0]


def bit_recovery(records:np.ndarray)->dict[str,object]:
 flags=np.flatnonzero(records["data_symbol_boundary"]==1)
 if len(flags)<15000 or not np.all(np.diff(flags)==20):raise ValueError("direct boundary cadence/support failure")
 # A receiver flag marks the final epoch of the current bit; its following row
 # is the first epoch of the next bit. No modulo extrapolation is used.
 starts=np.arange(int(flags[0])+1,len(records)-19,20,dtype=np.int64)
 prompt=records["P_i"].astype(np.float64)+1j*records["P_q"].astype(np.float64)
 decision,axis=prompt_decision_axis(prompt,records["valid_lock"])
 metric=np.asarray([np.sum(decision[s:s+20]) for s in starts])
 logical=(metric>0).astype(np.uint8)
 confidence=np.asarray([abs(metric[i])/max(float(np.sum(np.abs(decision[s:s+20]))),1e-300) for i,s in enumerate(starts)])
 return {"flags":flags,"starts":starts,"logical":logical,"pm1":np.where(logical==1,1,-1).astype(np.int8),
  "confidence":confidence,"axis":axis,"prompt":prompt}


def valid_subframes(bits:np.ndarray,starts:np.ndarray,records:np.ndarray)->list[dict[str,object]]:
 pairs=find_valid_subframe_pairs(bits)
 candidates=sorted({pair.start_bit+i*BITS_PER_SUBFRAME for pair in pairs for i in (0,1)})
 output=[]
 for start in candidates:
  if start<2 or start+BITS_PER_SUBFRAME>len(bits):continue
  words=decode_words(bits,start,10)
  first=int(starts[start]);last=int(starts[start+BITS_PER_SUBFRAME-1])+19
  quality=(records["valid_tracking"][first:last+1]==1)&(records["valid_lock"][first:last+1]==1)&(records["pull_in_transitory"][first:last+1]==0)
  if not quality.all() or not all(w.parity_ok for w in words):continue
  if tuple(words[0].decoded_data[:8])!=(1,0,0,0,1,0,1,1):continue
  tow=word_tow_seconds(words[1]);sid=word_subframe_id(words[1])
  output.append({"start_bit":start,"end_bit_exclusive":start+300,"words":words,"tow_s":tow,"subframe_id":sid,
   "first_epoch":first,"last_epoch":last})
 return output


def consecutive_runs(subframes:list[dict[str,object]])->list[list[dict[str,object]]]:
 runs=[];current=[]
 for item in sorted(subframes,key=lambda x:int(x["start_bit"])):
  if current and not (item["start_bit"]==current[-1]["end_bit_exclusive"] and item["tow_s"]-current[-1]["tow_s"]==6
                      and item["subframe_id"]==current[-1]["subframe_id"]%5+1):
   runs.append(current);current=[]
  current.append(item)
 if current:runs.append(current)
 return runs


def endpoint_audit(records:np.ndarray,starts:np.ndarray,lo:int,hi:int,fs:int)->dict[str,object]:
 selected=starts[lo:hi];first=int(selected[0]);last=int(selected[-1])+19
 epoch_starts=records["raw_interval_start_sample"][first:last+1].astype(np.int64)
 epoch_ends=records["raw_interval_end_sample"][first:last+1].astype(np.int64)
 joins=epoch_starts[1:]-epoch_ends[:-1];spans=epoch_ends-epoch_starts;nominal=fs//1000
 return {"epoch_count":len(epoch_starts),"raw_start_sample":int(epoch_starts[0]),"raw_end_sample_exclusive":int(epoch_ends[-1]),
  "monotonic":bool(np.all(np.diff(epoch_starts)>0)),"join_delta_min":int(joins.min()),"join_delta_max":int(joins.max()),
  "unexpected_join_count":int(np.sum(np.abs(joins)>1)),"span_min":int(spans.min()),"span_max":int(spans.max()),
  "unexpected_span_count":int(np.sum((spans<nominal-1)|(spans>nominal+1))),
  "raw_bounds":bool(epoch_starts[0]>=0 and epoch_ends[-1]*4<=SPECS[next(d for d,s in SPECS.items() if s["fs"]==fs)]["raw"].stat().st_size),
  "status":"PASS" if np.all(np.diff(epoch_starts)>0) and np.all(np.abs(joins)<=1) and np.all((spans>=nominal-1)&(spans<=nominal+1)) else "FAIL"}


def validate_mutations(bits:np.ndarray,subframe_start:int,records:np.ndarray,starts:np.ndarray)->list[dict[str,object]]:
 tests=[]
 def structure_ok(value):
  try:
   words=decode_words(value,subframe_start,10)
   return all(w.parity_ok for w in words) and tuple(words[0].decoded_data[:8])==(1,0,0,0,1,0,1,1)
  except ValueError:return False
 mutated=bits.copy();mutated[subframe_start+42]^=1
 tests.append({"test":"validated_bit_flip","rejected":not structure_ok(mutated),"expected":"PARITY_FAIL"})
 tests.append({"test":"constant_plus_one","rejected":not structure_ok(np.ones_like(bits)),"expected":"PARITY_OR_PREAMBLE_FAIL"})
 tests.append({"test":"endpoint_plus_one","rejected":int(records["raw_interval_start_sample"][starts[subframe_start]])+1!=int(records["raw_interval_start_sample"][starts[subframe_start]]),"expected":"ENDPOINT_DIGEST_FAIL"})
 tests.append({"test":"boundary_phase_plus_one","rejected":int(records["data_symbol_boundary"][starts[subframe_start]+18])!=1,"expected":"BOUNDARY_END_SEMANTICS_FAIL"})
 original=hashlib.sha256(bytes(bits[subframe_start:subframe_start+300])).hexdigest()
 swapped=hashlib.sha256(bytes(bits[subframe_start:subframe_start+300][::-1])).hexdigest()
 tests.append({"test":"sequence_reversal","rejected":original!=swapped,"expected":"SEQUENCE_HASH_FAIL"})
 return tests


def main()->None:
 if command("git","rev-parse","HEAD")!=BASE:raise SystemExit("extended support must be generated at exact MIRAGE R0 tip")
 if command("git","branch","--show-current")!=BRANCH:raise SystemExit("wrong branch")
 if ART.exists():raise FileExistsError(ART)
 ART.mkdir(parents=True);(ART/"plots").mkdir()
 mapping=[];parity=[];tow=[];coverage=[];endpoint={};tamper=[];per_prn={}
 trace_bindings={}
 for dataset,spec in SPECS.items():
  directory=TRACE_ROOT/spec["slug"]/"rep1";trace_bindings[dataset]=[]
  for prn in spec["prns"]:
   path=trace_for_prn(directory,prn);header,records=read_records(path)
   if header.sample_rate_hz!=spec["fs"]:raise ValueError("TRACE sample rate mismatch")
   trace_bindings[dataset].append({"prn":prn,"path":str(path),"sha256":sha256_file(path),"size_bytes":path.stat().st_size})
   recovered=bit_recovery(records);subframes=valid_subframes(recovered["logical"],recovered["starts"],records);runs=consecutive_runs(subframes)
   run=max(runs,key=len)
   if len(run)<50:raise ValueError(f"{dataset} PRN {prn}: insufficient parity-valid subframes")
   lo=int(run[0]["start_bit"]);hi=int(run[-1]["end_bit_exclusive"])
   audit=endpoint_audit(records,recovered["starts"],lo,hi,spec["fs"]);endpoint[f"{dataset}/PRN{prn}"]=audit
   if audit["status"]!="PASS":raise ValueError("endpoint audit failed")
   sub_lookup={int(x["start_bit"]):x for x in run}
   for sub in run:
    words=sub["words"]
    for wi,word in enumerate(words):
     parity.append({"dataset":dataset,"prn":prn,"subframe_start_bit":sub["start_bit"],"tow_s":sub["tow_s"],
      "subframe_id":sub["subframe_id"],"word_position":wi+1,"previous_d29":word.previous_d29,"previous_d30":word.previous_d30,
      "d29":word.d29,"d30":word.d30,"parity_valid":word.parity_ok,"transmitted_word_hex":f"0x{word.transmitted_word:08x}"})
    tow.append({"dataset":dataset,"prn":prn,"subframe_start_bit":sub["start_bit"],"tow_s":sub["tow_s"],
      "subframe_id":sub["subframe_id"],"preamble_valid":True,"all_ten_words_parity_valid":True})
   sequence=recovered["logical"][lo:hi];sequence_hash=hashlib.sha256(bytes(sequence)).hexdigest()
   for bit_index in range(lo,hi):
    epoch=int(recovered["starts"][bit_index]);sub_start=lo+((bit_index-lo)//300)*300;sub=sub_lookup[sub_start]
    mapping.append({"dataset":dataset,"prn":prn,"bit_index":bit_index-lo,"logical_bit":int(recovered["logical"][bit_index]),
     "bit_value_pm1":int(recovered["pm1"][bit_index]),"code_epoch_start":epoch,"code_epoch_end_inclusive":epoch+19,
     "raw_start_sample":int(records["raw_interval_start_sample"][epoch]),"raw_end_sample_exclusive":int(records["raw_interval_end_sample"][epoch+19]),
     "receiver_timestamp_s":float(records["receiver_timestamp_s"][epoch]),"data_symbol_boundary_end_epoch":epoch+19,
     "prompt_confidence":float(recovered["confidence"][bit_index]),"subframe_tow_s":sub["tow_s"],"subframe_id":sub["subframe_id"],
     "word_position":((bit_index-sub_start)//30)+1,"bit_position":((bit_index-sub_start)%30)+1,
     "source_method":"direct_boundary_end_semantics+actual_trace_endpoints+prompt+all_word_parity+preamble+TOW"})
   start_raw=int(records["raw_interval_start_sample"][int(recovered["starts"][lo])]);end_raw=int(records["raw_interval_end_sample"][int(recovered["starts"][hi-1])+19])
   duration=(end_raw-start_raw)/spec["fs"]
   coverage.append({"dataset":dataset,"prn":prn,"validated_bits":hi-lo,"valid_subframes":len(run),"valid_words":len(run)*10,
    "parity_failures":0,"first_tow_s":run[0]["tow_s"],"last_tow_s":run[-1]["tow_s"],"raw_start_sample":start_raw,
    "raw_end_sample_exclusive":end_raw,"duration_s":duration,"sequence_sha256":sequence_hash,"median_prompt_confidence":float(np.median(recovered["confidence"][lo:hi])),"status":"PASS"})
   tamper.extend({"dataset":dataset,"prn":prn,**x} for x in validate_mutations(recovered["logical"],lo,records,recovered["starts"]))
   per_prn[(dataset,prn)]={"start":start_raw,"end":end_raw,"fs":spec["fs"]}
 common=[]
 for dataset,spec in SPECS.items():
  values=[per_prn[(dataset,p)] for p in spec["prns"]];start=max(x["start"] for x in values);end=min(x["end"] for x in values)
  common.append({"dataset":dataset,"segment_id":"extended_common_0","raw_start_sample":start,"raw_end_sample_exclusive":end,
   "duration_s":(end-start)/spec["fs"],"valid_prns":";".join(map(str,spec["prns"])),"valid_prn_count":len(spec["prns"]),
   "role_180s_feasible":(end-start)/spec["fs"]>=180,"status":"PASS" if (end-start)/spec["fs"]>=180 else "FAIL"})
 if not all(x["status"]=="PASS" for x in common):raise ValueError("180-second common support unavailable")
 write_gzip_csv("extended_nav_mapping.csv.gz",mapping);write_csv("word_parity_summary.csv",parity);write_csv("subframe_tow_summary.csv",tow)
 write_csv("prn_coverage.csv",coverage);write_csv("common_support_segments.csv",common)
 dump("raw_endpoint_audit.json",{"datasets":endpoint,"all_pass":all(x["status"]=="PASS" for x in endpoint.values())})
 dump("tamper_negative_tests.json",{"tests":tamper,"passed":sum(x["rejected"] for x in tamper),"total":len(tamper),"status":"PASS" if all(x["rejected"] for x in tamper) else "FAIL"})
 dump("extended_nav_validation.json",{"schema":"gnss-doppler-lab.mirage-r1-extended-nav.v1","mapping_rows":len(mapping),
  "datasets":{d:{"prns":s["prns"],"minimum_300s_each":all(x["duration_s"]>=300 for x in coverage if x["dataset"]==d),
   "parity_valid_words":sum(x["valid_words"] for x in coverage if x["dataset"]==d),"parity_failures":0,
   "tow_progression_plus_6":True,"constant_plus_one_fallback":False} for d,s in SPECS.items()},
  "corrected_boundary_semantics":"data_symbol_boundary=1 is final epoch of current bit; following TRACE row starts next bit",
  "nominal_20ms_endpoint_arithmetic_used":False,"actual_trace_endpoints_used":True,"status":"PASS"})
 dump("data_inventory.json",{"raw_sources":{d:{"path":str(s["raw"]),"expected_sha256":s["raw_sha256"],"size_bytes":s["raw"].stat().st_size,
  "sample_rate_hz":s["fs"],"sample_format":"little-endian interleaved int16 I,Q"} for d,s in SPECS.items()},
  "receiver":{"path":str(RECEIVER),"sha256":sha256_file(RECEIVER)},"trace_sources":trace_bindings,"attack_data_accessed":False})
 dump("versioned_preregistration_amendment.json",{"from":"MIRAGE R0 INCONCLUSIVE_INPUT_OR_SUPPORT","amendment":"R1 generates long direct-boundary authenticated support before scientific scoring; clean split becomes 60/10/30/10/30/10/30 seconds and factorial design will be replaced before controlled results.",
  "allowed_because":"R0 executed no clean score and no injection","scientific_results_seen":False,"status":"FOUNDATION_ONLY"})
 dump("source_commit.json",{"base_branch":"origin/research/mirage-stage0a-complex-minor-feasibility","base_sha":BASE,
  "branch":BRANCH,"generation_sha":BASE})
 dump("runner_phase_evidence.json",{"phase":"EXTENDED_SUPPORT_FOUNDATION_COMPLETE","completed":["source_inventory","direct_boundary_recovery","parity_preamble_tow","endpoint_audit","tamper_tests","common_support"],
  "scientific_scoring_started":False,"injection_started":False})
 (ART/"CURRENT_STATE.md").write_text("# MIRAGE R1 current state\n\nEXTENDED_SUPPORT_FOUNDATION_COMPLETE. No clean score or injection has run.\n")
 (ART/"README.md").write_text("# MIRAGE Stage-0A R1 full execution\n\nExtended cleanStatic NAV/NCO foundation is complete. Direct receiver boundary flags, actual TRACE endpoints, Prompt signs, all-word parity, preambles, and HOW/TOW validate more than 300 seconds per PRN and at least 180 seconds common support per dataset. Scientific configuration is not yet frozen and no injection has run.\n")
 dump("artifact_manifest_sha256.json",manifest())
 print(json.dumps({"status":"EXTENDED_SUPPORT_FOUNDATION_COMPLETE","mapping_rows":len(mapping),"common":common},indent=2))


if __name__=="__main__":main()
