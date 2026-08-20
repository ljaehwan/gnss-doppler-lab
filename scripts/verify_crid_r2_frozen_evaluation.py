#!/usr/bin/env python3
"""Independent verifier for the fail-closed CRID R2 frozen evaluation."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/"artifacts/crid_stage0_r2_frozen_evaluation"
REQUIRED={"README.md","config_binding.json","source_commit.json","repair_binding.json","replay_completion.json",
 "physical_control_metrics.csv","physical_identifiability.json","thresholds.json","scenario_metrics.csv","ablation_metrics.csv",
 "per_epoch_scores.csv.gz","configuration_state_metrics.csv.gz","external_static_fpr.csv","shortcut_audit.csv",
 "relation_destruction.json","bootstrap_intervals.csv","final_verdict.json","artifact_manifest_sha256.json"}

def digest(path):
 value=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(1024*1024),b""):value.update(block)
 return value.hexdigest()
def actual_manifest(art):return {str(path.relative_to(art)):digest(path) for path in sorted(art.rglob("*")) if path.is_file() and path.name!="artifact_manifest_sha256.json"}
def read_csv(path):
 with path.open(newline="") as stream:return list(csv.DictReader(stream))
def read_gzip(path):
 with gzip.open(path,"rt",newline="") as stream:return list(csv.DictReader(stream))

def verify(art):
 failures=[];present={str(path.relative_to(art)) for path in art.rglob("*") if path.is_file()};missing=sorted(REQUIRED-present)
 if missing:failures.extend(f"missing:{name}" for name in missing)
 expected=json.loads((art/"artifact_manifest_sha256.json").read_text());actual=actual_manifest(art)
 if expected!=actual:failures.append("manifest")
 binding=json.loads((art/"config_binding.json").read_text());source=json.loads((art/"source_commit.json").read_text())
 repair=json.loads((art/"repair_binding.json").read_text());replay=json.loads((art/"replay_completion.json").read_text())
 physical=json.loads((art/"physical_identifiability.json").read_text());final=json.loads((art/"final_verdict.json").read_text())
 thresholds=json.loads((art/"thresholds.json").read_text())
 if binding.get("status")!="PASS" or binding.get("freeze_sha")!="2bd96157b00df8997b66830894912edb016c067b":failures.append("config_binding")
 if binding.get("config_sha256")!="326a1ffb178f0fc5e9937418ae5701021c661fa64ba3bb9f43ba32ec4936aa7e":failures.append("config_hash")
 if binding.get("config_preregistration_byte_identical") is not True:failures.append("preregistration_binding")
 if source.get("base_expected")!="aa78fe158c9ea8f00ddd82c46bf87fc4fd328551" or source.get("base_actual")!=source.get("base_expected"):failures.append("source_base")
 if repair.get("status")!="PASS" or repair.get("repair_verdict")!="TERMINAL_DRAIN_REPAIR_PASS":failures.append("repair_binding")
 if repair.get("all_configurations_pass") is not True or repair.get("endpoint_max_error_samples",2)>1:failures.append("repair_contract")
 if replay.get("phase_a_complete") is not False or replay.get("phase_b_started") is not False or replay.get("attack_replay_count")!=0:failures.append("replay_scope")
 for name,row in replay.get("r1_reuse",{}).get("configurations",{}).items():
  if not (row.get("status")=="PASS" and row.get("exact_input_bytes")==4_500_000_000 and row.get("dump_count")==10
   and row.get("exit_code")==0 and row.get("sigterm_sent") is False and row.get("sigkill_sent") is False):failures.append(f"r1_reuse:{name}")
 controls=read_csv(art/"physical_control_metrics.csv");required=[row for row in controls if row["frozen_required"]=="True"]
 if len(required)!=66:failures.append("required_control_count")
 for domain in ("TEX","OAK"):
  domain_rows=[row for row in required if row["domain"]==domain]
  if sum(row["kind"]=="negative" for row in domain_rows)!=15 or sum(row["kind"]=="positive" for row in domain_rows)!=18:failures.append(f"grid:{domain}")
  if sum(row["conforming"]=="True" for row in domain_rows if row["kind"]=="negative")!=9:failures.append(f"negative_binding:{domain}")
  if any(row["conforming"]=="True" for row in domain_rows if row["kind"]=="positive"):failures.append(f"positive_binding:{domain}")
 if physical.get("status")!="INCONCLUSIVE" or physical.get("grid_complete") is not False or physical.get("attack_open_authorized") is not False:failures.append("physical_gate")
 diagnostic=physical.get("r1_diagnostic",{})
 if diagnostic.get("alarm_ratio_q99")!=0.037102997922232116 or diagnostic.get("positive_gate_alarm_ratio")!=.70:failures.append("diagnostic_metric")
 if thresholds.get("status")!="BOUND_UNCHANGED" or thresholds.get("TEX",{}).get("q99")!=-21.942672917134093:failures.append("threshold_binding")
 scenarios=read_csv(art/"scenario_metrics.csv")
 if len(scenarios)!=7 or any(row["status"]!="NOT_OPENED_PHASE_A_BLOCKED" for row in scenarios):failures.append("scenario_scope")
 if read_gzip(art/"per_epoch_scores.csv.gz") or read_gzip(art/"configuration_state_metrics.csv.gz"):failures.append("unexpected_phase_b_rows")
 if final.get("verdict")!="INCONCLUSIVE_RECEIVER_REPLAY_OR_ALIGNMENT" or final.get("phase_b")!="NOT_AUTHORIZED":failures.append("verdict")
 if final.get("attack_payload_opened") is not False or final.get("attack_payload_bytes_read")!=0 or final.get("neural_stage1_implemented") is not False:failures.append("leakage_scope")
 plots=sorted((art/"plots").glob("*.png"))
 if len(plots)<2 or any(path.stat().st_size==0 for path in plots):failures.append("plots")
 return {"schema":"gnss-doppler-lab.crid-r2-verifier.v1","status":"PASS" if not failures else "FAIL","failures":failures,
  "checks":{"required_missing":missing,"manifest_match":expected==actual,"manifest_entries":len(actual),"required_control_rows":len(required),
   "attack_rows":len(read_gzip(art/"per_epoch_scores.csv.gz")),"scenario_rows":len(scenarios),"plot_count":len(plots),
   "recomputed_verdict":"INCONCLUSIVE_RECEIVER_REPLAY_OR_ALIGNMENT" if not failures else "INVALID"}}

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--artifact",type=Path,default=DEFAULT);args=parser.parse_args()
 result=verify(args.artifact);print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(result["status"]!="PASS")
if __name__=="__main__":main()
