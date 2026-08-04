#!/usr/bin/env python3
"""Fail-closed verifier for the corrected real-data R2C artifact."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.r2c_gnss import artifact_hashes,sha256_file,write_json # noqa:E402
REQUIRED={"README.md","config.json","freeze.json","provenance.json","input_validity.json","training_summary.json","thresholds.json","scenario_metrics.csv","ablation_metrics.csv","per_epoch_scores.csv","gain_invariance.json","phase_invariance.json","noise_control.json","multipath_control.json","second_source_injection.json","relation_destruction.json","decision.json","verification.json","hashes.json","plots/relation_control.png","plots/relation_control_source.csv"}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--artifact",type=Path,default=ROOT/"artifacts/r2c_gnss_stage0"); p.add_argument("--write-result",action="store_true"); p.add_argument("--skip-external-hashes",action="store_true"); a=p.parse_args(); root=a.artifact.resolve(); err=[]
 miss=sorted(x for x in REQUIRED if not (root/x).is_file())
 if miss: err.append(f"missing files: {miss}")
 if not miss:
  cfg=json.loads((root/"config.json").read_text()); fr=json.loads((root/"freeze.json").read_text()); pv=json.loads((root/"provenance.json").read_text()); iv=json.loads((root/"input_validity.json").read_text()); th=json.loads((root/"thresholds.json").read_text()); de=json.loads((root/"decision.json").read_text())
  if json.loads((root/"hashes.json").read_text())["files"]!=artifact_hashes(root): err.append("artifact hash mismatch")
  if not fr.get("written_before_score_computation") or not fr.get("no_attack_label_tuning"): err.append("invalid freeze")
  if not iv.get("frozen_before_attack_evaluation") or iv.get("attack_outcomes_inspected_for_tuning"): err.append("input gate/tuning violation")
  if de.get("verdict") not in {"PHYSICS_SUPPORTED","NOT_SUPPORTED","DATA_INVALID","INCONCLUSIVE"}: err.append("invalid verdict")
  if de.get("old_result_status")!="SUPERSEDED_BY_EXTERNAL_DATA_DISCOVERY": err.append("superseded provenance absent")
  if pv.get("frozen_base_commit")!="461eb4dc7bb794e719295daf028f6811658ba37f": err.append("wrong frozen base")
  if pv.get("branch")!="research/r2c-gnss-stage0": err.append("wrong branch")
  current=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
  if not (current==pv.get("source_commit_at_generation") or subprocess.call(["git","merge-base","--is-ancestor",pv.get("source_commit_at_generation","x"),current],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)==0): err.append("generation commit is not current/ancestor")
  if th.get("source")!="cleanStatic normal_calibration only" or th.get("method")!="higher": err.append("threshold contamination/policy")
  if not a.skip_external_hashes:
   for n,d in iv.get("datasets",{}).items():
    q=Path(d["resolved_path"])
    if not q.is_file() or sha256_file(q)!=d["npz_sha256"]: err.append(f"external input hash mismatch: {n}")
  for tab in ("scenario_metrics.csv","ablation_metrics.csv","per_epoch_scores.csv"):
   rows=list(csv.DictReader((root/tab).open()));
   if not rows: err.append(f"empty table: {tab}")
   if any(str(v).lower() in {"nan","inf","-inf"} for r in rows for v in r.values()): err.append(f"nonfinite: {tab}")
 result={"status":"PASS" if not err else "FAIL","errors":err,"checked_required_files":len(REQUIRED),"external_hashes_checked":not a.skip_external_hashes,"hash_policy":"hashes.json and verification.json excluded; generation commit may be current or ancestor of final result commit"}
 if a.write_result:
  write_json(root/"verification.json",result); write_json(root/"hashes.json",{"algorithm":"sha256","files":artifact_hashes(root)})
 print(json.dumps(result,indent=2)); return bool(err)
if __name__=="__main__": raise SystemExit(main())
