#!/usr/bin/env python3
"""Independent compact-artifact verifier for CRID R1 terminal-drain repair."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/"artifacts/crid_stage0_r1_terminal_drain_repair"
REQUIRED={"README.md","root_cause.json","termination_contract.json","clean_equivalence.json",
 "positive_control_reproducibility.json","four_configuration_completion.json",
 "process_timeline.csv","artifact_manifest_sha256.json","final_verdict.json"}

def digest(path):
 value=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(1024*1024),b""):value.update(block)
 return value.hexdigest()
def actual_manifest(art):return {str(path.relative_to(art)):digest(path) for path in sorted(art.rglob("*")) if path.is_file() and path.name!="artifact_manifest_sha256.json"}

def verify(art):
 failures=[];present={str(path.relative_to(art)) for path in art.rglob("*") if path.is_file()}
 missing=sorted(REQUIRED-present)
 if missing:failures.extend(f"missing:{name}" for name in missing)
 expected=json.loads((art/"artifact_manifest_sha256.json").read_text());actual=actual_manifest(art)
 if expected!=actual:failures.append("manifest")
 root=json.loads((art/"root_cause.json").read_text());contract=json.loads((art/"termination_contract.json").read_text())
 clean=json.loads((art/"clean_equivalence.json").read_text());positive=json.loads((art/"positive_control_reproducibility.json").read_text())
 four=json.loads((art/"four_configuration_completion.json").read_text());final=json.loads((art/"final_verdict.json").read_text())
 if root.get("diagnosis")!="R2C_DRAIN_WAIT_DEADLOCK_AFTER_EXACT_FINITE_SOURCE_EOF":failures.append("root_cause")
 if root.get("finite_source",{}).get("observed_raw_fd_position")!=4_500_000_000:failures.append("diagnostic_eof")
 if contract.get("exact_input_bytes")!=4_500_000_000 or contract.get("required_dump_count")!=10:failures.append("termination_contract")
 if contract.get("forbidden_success_causes")!=["SIGTERM","SIGKILL"]:failures.append("forced_exit_contract")
 if clean.get("status")!="PASS" or clean.get("bit_identical_all_ten_dumps") is not True:failures.append("clean_equivalence")
 if positive.get("status")!="PASS" or positive.get("C0_bit_identical_two_runs") is not True:failures.append("positive_reproducibility")
 if positive.get("control_size_bytes")!=4_500_000_000:failures.append("control_size")
 if four.get("status")!="PASS":failures.append("four_completion")
 configs=four.get("configurations",{})
 if set(configs)!={"C0","C1","C2","C3"}:failures.append("configuration_set")
 for name,row in configs.items():
  if not (row.get("status")=="PASS" and row.get("exit_code")==0 and row.get("exit_cause")=="verified_eof_graceful_sigint"
   and row.get("exact_input_bytes")==4_500_000_000 and row.get("dump_count")==10 and row.get("sigterm_sent") is False
   and row.get("sigkill_sent") is False and row.get("scientific_config_unchanged") is True and len(row.get("terminal_rows",[]))==10):failures.append(f"completion:{name}")
 endpoint=four.get("absolute_raw_endpoint_contract",{})
 if endpoint.get("status")!="PASS" or endpoint.get("maximum_absolute_endpoint_delta_samples",2)>1:failures.append("endpoint_contract")
 if final.get("verdict")!="TERMINAL_DRAIN_REPAIR_PASS" or final.get("status")!="READY_FOR_FROZEN_CRID_RESUME":failures.append("verdict")
 if final.get("attack_data_accessed") is not False or final.get("attack_evaluation_started") is not False:failures.append("attack_scope")
 with (art/"process_timeline.csv").open(newline="") as stream:timeline=list(csv.DictReader(stream))
 signal_runs={row["run"] for row in timeline if row["signal"]=="SIGINT"}
 if signal_runs!={"C0","C1","C2","C3","C0_repeat"}:failures.append("timeline_signals")
 return {"schema":"gnss-doppler-lab.crid-r1-verifier.v1","status":"PASS" if not failures else "FAIL",
  "failures":failures,"checks":{"required_missing":missing,"manifest_match":expected==actual,"manifest_entries":len(actual),
   "timeline_rows":len(timeline),"graceful_sigint_runs":sorted(signal_runs),"recomputed_verdict":"TERMINAL_DRAIN_REPAIR_PASS" if not failures else "TERMINAL_DRAIN_REPAIR_FAIL"}}

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--artifact",type=Path,default=DEFAULT);args=parser.parse_args()
 result=verify(args.artifact);print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(result["status"]!="PASS")
if __name__=="__main__":main()
