#!/usr/bin/env python3
"""Independent compact-artifact verifier for CRID Stage-0."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DEFAULT=ROOT/"artifacts/crid_stage0_counterfactual_receiver_invariance"
REQUIRED={"README.md","config.json","preregistration.json","source_commit.json","receiver_binary_inventory.json","receiver_configurations.json","receiver_config_hashes.json","data_inventory.json","alignment_audit.json","clean_split_audit.json","normal_model_summary.json","thresholds.json","physical_control_metrics.csv","scenario_metrics.csv","ablation_metrics.csv","common_support_metrics.csv","external_static_fpr.csv","counterfactual_validity.json","shortcut_audit.csv","bootstrap_intervals.csv","per_epoch_scores.csv.gz","per_config_state_estimates.csv.gz","final_verdict.json","artifact_manifest_sha256.json"}
def digest(p):
 d=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):d.update(b)
 return d.hexdigest()
def actual_manifest(art):return {str(p.relative_to(art)):digest(p) for p in sorted(art.rglob("*")) if p.is_file() and p.name!="artifact_manifest_sha256.json"}
def rows(path,gz=False):
 op=gzip.open if gz else open
 with op(path,"rt" if gz else "r",newline="") as f:return list(csv.DictReader(f))
def verify(art):
 fail=[];checks={};present={str(p.relative_to(art)) for p in art.rglob("*") if p.is_file()};missing=sorted(REQUIRED-present)
 if missing:fail.extend(f"missing:{x}" for x in missing)
 checks["required"]={"missing":missing};expected=json.loads((art/"artifact_manifest_sha256.json").read_text());actual=actual_manifest(art)
 checks["manifest"]={"match":expected==actual,"entries":len(actual)}
 if expected!=actual:fail.append("manifest")
 final=json.loads((art/"final_verdict.json").read_text());verdict=final["verdict"]
 allowed={"GO_FOR_CRID_NEURAL_STAGE1","GO_PHYSICS_BASELINE_PENDING","NO_GO_CRID_COUNTERFACTUAL_INVARIANCE","NO_GO_CRID_CLEAN_PHYSICAL_IDENTIFIABILITY","INCONCLUSIVE_RECEIVER_REPLAY_OR_ALIGNMENT"}
 if verdict not in allowed:fail.append("verdict_enum")
 epoch=rows(art/"per_epoch_scores.csv.gz",True);scenario=rows(art/"scenario_metrics.csv")
 recomputed={}
 if epoch:
  for reported in scenario:
   rr=[r for r in epoch if r["dataset"]==reported["dataset"]]
   if not rr or reported.get("status")!="COMPUTED":continue
   labels=np.array([int(r["label"]) for r in rr]);alarm=np.array([int(r["alarm"]) for r in rr])
   calc={"preonset_fpr":float(alarm[labels==0].mean()),"attack_detection_rate":float(alarm[labels==1].mean())};recomputed[reported["dataset"]]=calc
   for k,v in calc.items():
    if not np.isclose(v,float(reported[k]),atol=1e-10):fail.append(f"scenario:{reported['dataset']}:{k}")
 checks["metric_recomputation"]=recomputed
 if verdict=="INCONCLUSIVE_RECEIVER_REPLAY_OR_ALIGNMENT":
  if json.loads((art/"alignment_audit.json").read_text()).get("status")!="FAIL":fail.append("inconclusive_without_alignment_failure")
  if final.get("attack_data_accessed") is not False:fail.append("attack_access_claim")
 elif not epoch:fail.append("computed_verdict_without_epoch_evidence")
 return {"schema":"gnss-doppler-lab.crid-verifier.v1","status":"PASS" if not fail else "FAIL","failures":fail,"checks":checks,"recomputed_verdict":verdict}
def main():
 p=argparse.ArgumentParser();p.add_argument("--artifact",type=Path,default=DEFAULT);p.add_argument("--write-manifest",action="store_true");a=p.parse_args()
 if a.write_manifest:(a.artifact/"artifact_manifest_sha256.json").write_text(json.dumps(actual_manifest(a.artifact),indent=2,sort_keys=True)+"\n")
 result=verify(a.artifact);print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(result["status"]!="PASS")
if __name__=="__main__":main()
