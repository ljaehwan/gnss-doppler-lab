#!/usr/bin/env python3
"""Validate and assemble immutable CMTE-A2 train/dev/freeze/confirm evidence."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte_a2_campaign import (PREREG_HASHES,atomic_json,copy_tree_files,file_sha256,
 require_nonempty,validate_trust_anchor,verify_checksums)
from gnss_doppler_lab.cmte_a2 import write_checksums

def _csv(paths):
 frames=[pd.read_csv(require_nonempty(p)) for p in paths]
 if any(x.empty for x in frames): raise ValueError("empty placeholder table rejected")
 return pd.concat(frames,ignore_index=True)

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("state-dir","development-dir","confirmatory-dir","trust-anchor","ledger","expected-anchor-sha256","out"): p.add_argument("--"+name,required=True)
 p.add_argument("--result-commit-sha",default="PENDING_FINALIZATION"); p.add_argument("--repo",default=str(ROOT)); a=p.parse_args(argv)
 anchor=validate_trust_anchor(a.trust_anchor,a.expected_anchor_sha256,repo=a.repo)
 state=Path(a.state_dir).resolve(strict=True); dev=Path(a.development_dir).resolve(strict=True); confirm=Path(a.confirmatory_dir).resolve(strict=True)
 if str(state)!=anchor["state_dir"] or str(dev)!=anchor["development_dir"]: raise ValueError("artifact directories differ from trust anchor")
 for directory in (state,dev,confirm): verify_checksums(directory)
 ledger=json.loads(require_nonempty(a.ledger).read_text())
 if ledger.get("status")!="completed" or ledger.get("trust_anchor_sha256")!=a.expected_anchor_sha256: raise ValueError("one-shot ledger is not completed for this anchor")
 if ledger.get("result_sha256")!=file_sha256(confirm/"checksums.json"): raise ValueError("ledger confirm result checksum mismatch")
 required_state=("b0_model.pt","a2_state.json","config.json","training.json","calibration.json","thresholds.json","preregistration.json","provenance.json")
 for n in required_state: require_nonempty(state/n)
 required_eval=("baseline_metrics.csv","bootstrap.csv","exact_n_diagnostics.csv","matched_fpr.csv","audit.json","test_summary.json","provenance/provenance.json")
 for d,primary,scenarios in ((dev,"development_metrics.csv",("DS1","DS2","DS3","DS4")),(confirm,"confirmatory_metrics.csv",("DS7","DS8"))):
  require_nonempty(d/primary)
  for n in required_eval: require_nonempty(d/n)
  for s in scenarios:
   require_nonempty(d/"per_epoch"/f"{s}.csv"); require_nonempty(d/"per_prn"/f"{s}.csv")
 source=anchor["source_commit"]
 for d in (dev,confirm):
  prov=json.loads((d/"provenance/provenance.json").read_text())
  if prov.get("execution_source_commit")!=source: raise ValueError("evaluation source commit mismatch")
 prereg=ROOT/"configs/cmte_a2_preregistration.json"
 if file_sha256(prereg)!=PREREG_HASHES["configs/cmte_a2_preregistration.json"]: raise ValueError("preregistration changed")
 out=Path(a.out).absolute()
 if out.exists(): raise FileExistsError("final artifact is atomic and non-overwriting")
 staging=out.with_name(out.name+f".tmp-{os.getpid()}"); staging.mkdir(parents=True)
 try:
  shutil.copyfile(prereg,staging/"preregistration.json")
  for n in ("config.json","training.json","calibration.json","thresholds.json","a2_state.json","b0_model.pt"): shutil.copyfile(state/n,staging/n)
  _csv([dev/"development_metrics.csv",confirm/"confirmatory_metrics.csv"]).to_csv(staging/"scenario_metrics.csv",index=False)
  shutil.copyfile(dev/"development_metrics.csv",staging/"development_metrics.csv"); shutil.copyfile(confirm/"confirmatory_metrics.csv",staging/"confirmatory_metrics.csv")
  _csv([dev/"baseline_metrics.csv",confirm/"baseline_metrics.csv"]).to_csv(staging/"baseline_metrics.csv",index=False)
  _csv([dev/"bootstrap.csv",confirm/"bootstrap.csv"]).to_csv(staging/"bootstrap_cis.csv",index=False)
  _csv([dev/"exact_n_diagnostics.csv",confirm/"exact_n_diagnostics.csv"]).to_csv(staging/"exact_n_diagnostics.csv",index=False)
  _csv([dev/"matched_fpr.csv",confirm/"matched_fpr.csv"]).to_csv(staging/"matched_fpr.csv",index=False)
  for kind in ("per_epoch","per_prn"):
   (staging/kind).mkdir()
   for d in (dev,confirm):
    if (d/kind).is_dir(): copy_tree_files(d/kind,staging/kind)
  (staging/"plots").mkdir()
  copy_tree_files(dev/"plots",staging/"plots"/"development")
  copy_tree_files(confirm/"plots",staging/"plots"/"confirmatory")
  (staging/"audit").mkdir(); shutil.copyfile(dev/"audit.json",staging/"audit/development_audit.json"); shutil.copyfile(confirm/"audit.json",staging/"audit/confirmatory_audit.json")
  for name in ("success_audit.json","prn_dependence.json"):
   if (confirm/name).is_file(): shutil.copyfile(confirm/name,staging/"audit"/name)
  provenance={"schema":"gnss-doppler-lab.cmte-a2-final-provenance.v1","source_commit":source,
   "result_commit_sha":a.result_commit_sha,"result_commit_may_be_finalized_without_scientific_metric_changes":True,
   "preregistration_sha256":file_sha256(prereg),"trust_anchor_sha256":a.expected_anchor_sha256,
   "ledger_sha256":file_sha256(a.ledger),"state_checksums_sha256":file_sha256(state/"checksums.json"),
   "development_checksums_sha256":file_sha256(dev/"checksums.json"),"confirmatory_checksums_sha256":file_sha256(confirm/"checksums.json"),
   "chronology":["normal-only training","development DS1-DS4","pre-holdout freeze","O_EXCL one-shot ledger","confirmatory DS7-DS8","finalization"],
   "producer_exceptions":{"DS4":"development sensitivity; mixed producer; never confirmatory"}}
  atomic_json(staging/"provenance.json",provenance)
  tests={"state":(state/"test_summary.txt").read_text(),"development":json.loads((dev/"test_summary.json").read_text()),
         "confirmatory":json.loads((confirm/"test_summary.json").read_text())}
  atomic_json(staging/"test_summary.json",tests)
  metrics=pd.read_csv(staging/"scenario_metrics.csv"); rows=[]
  for r in metrics.itertuples(index=False):
   rows.append(f"- {getattr(r,'scenario','?')} / {getattr(r,'model','CMTE-A2')}: stable-pre FPR={getattr(r,'stable_pre_fpr','NA')}, post detection={getattr(r,'post_detection_rate','NA')}, first delay={getattr(r,'first_alarm_delay_s','NA')}")
  decision="See audit/success_audit.json for the exact preregistered criteria 1-6 decision."
  readme="# CMTE-A2 sealed campaign artifact\n\n"+f"Preregistration SHA-256: `{file_sha256(prereg)}`\n\nCode commit: `{source}`  \nResult commit: `{a.result_commit_sha}`\n\n"
  readme+="## Chronological evidence\n\n1. Normal-only training and calibration.\n2. DS1-DS4 development evaluation.\n3. External pre-holdout trust anchor.\n4. O_EXCL one-shot ledger.\n5. DS7/DS8 confirmation.\n6. Atomic finalization.\n\n"
  readme+="## Producer exception\n\nDS4 is mixed-producer development sensitivity only; DS7/DS8 share the frozen complex converter fingerprint.\n\n## Results\n\n"+"\n".join(rows)+"\n\n## Criteria and claims\n\n"+decision+" Claims are limited to the preregistered CMTE-A2-specific holdout scope.\n"
  (staging/"README.md").write_text(readme)
  # Assert all six scenario files survived and plots are nonempty.
  for s in ("DS1","DS2","DS3","DS4","DS7","DS8"):
   require_nonempty(staging/"per_epoch"/f"{s}.csv"); require_nonempty(staging/"per_prn"/f"{s}.csv")
  plots=[x for x in (staging/"plots").rglob("*") if x.is_file() and x.stat().st_size]
  if not plots: raise ValueError("nonempty campaign plots required")
  write_checksums(staging); os.replace(staging,out)
  print(json.dumps({"out":str(out),"source_commit":source,"files":len(list(out.rglob("*"))),"result_commit_sha":a.result_commit_sha},sort_keys=True))
 except Exception:
  shutil.rmtree(staging,ignore_errors=True); raise
if __name__=="__main__": main()
