#!/usr/bin/env python3
"""Validate and assemble immutable CMTE-A2 train/dev/freeze/confirm evidence."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte_a2_campaign import (CONVERTER_SHA,EXPORTER_SHA,PREREG_COMMIT,PREREG_HASHES,RECEIVER_SHA,atomic_json,copy_tree_files,file_sha256,
 require_nonempty,validate_test_attestation,validate_trust_anchor,verify_checksums)
from gnss_doppler_lab.cmte_a2 import write_checksums

def _csv(paths):
 frames=[pd.read_csv(require_nonempty(p)) for p in paths]
 if any(x.empty for x in frames): raise ValueError("empty placeholder table rejected")
 return pd.concat(frames,ignore_index=True)

def _audit_json(path,expected_schema):
 doc=json.loads(require_nonempty(path).read_text())
 if not isinstance(doc,dict) or doc.get("schema")!=expected_schema: raise ValueError(f"audit schema mismatch: {path}")
 return doc

def _validate_evaluation_audits(directory):
 success=_audit_json(directory/"success_audit.json","gnss-doppler-lab.cmte-a2-success-audit.v1")
 if [x.get("id") for x in success.get("criteria",[])]!=[1,2,3,4,5,6] or success.get("decision") not in {"GO","NO-GO"}:
  raise ValueError("success audit criteria/decision schema invalid")
 dependence=_audit_json(directory/"prn_dependence.json","gnss-doppler-lab.cmte-a2-prn-dependence.v1")
 if dependence.get("passed") is not True or dependence.get("aggregation_changed") is not False or dependence.get("sparse_strata_pooled") is not False:
  raise ValueError("PRN dependence audit failed or altered aggregation")
 historical=_audit_json(directory/"historical_b0_gate_equivalence.json","gnss-doppler-lab.cmte-a2-historical-gate-equivalence.v1")
 if historical.get("passed") is not True or float(historical.get("max_absolute_error",1))>1e-12 or historical.get("strict_alarm_equal") is not True:
  raise ValueError("historical B0 equivalence audit failed")
 return success,dependence,historical

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("state-dir","development-dir","confirmatory-dir","trust-anchor","ledger","expected-anchor-sha256","out"): p.add_argument("--"+name,required=True)
 p.add_argument("--result-commit-sha",default="PENDING_FINALIZATION"); p.add_argument("--repo",default=str(ROOT)); a=p.parse_args(argv)
 anchor=validate_trust_anchor(a.trust_anchor,a.expected_anchor_sha256,repo=a.repo)
 state=Path(a.state_dir).resolve(strict=True); dev=Path(a.development_dir).resolve(strict=True); confirm=Path(a.confirmatory_dir).resolve(strict=True)
 if str(state)!=anchor["state_dir"] or str(dev)!=anchor["development_dir"]: raise ValueError("artifact directories differ from trust anchor")
 for directory in (state,dev,confirm): verify_checksums(directory)
 ledger=json.loads(require_nonempty(a.ledger).read_text())
 if ledger.get("schema")!="gnss-doppler-lab.cmte-a2-one-shot-ledger.v1" or ledger.get("one_shot") is not True: raise ValueError("one-shot ledger schema invalid")
 if ledger.get("status")!="completed" or ledger.get("trust_anchor_sha256")!=a.expected_anchor_sha256: raise ValueError("one-shot ledger is not completed for this anchor")
 if ledger.get("result_sha256")!=file_sha256(confirm/"checksums.json"): raise ValueError("ledger confirm result checksum mismatch")
 required_state=("b0_model.pt","a2_state.json","config.json","training.json","calibration.json","thresholds.json","preregistration.json","provenance.json","historical_b0_gate_equivalence.json")
 for n in required_state: require_nonempty(state/n)
 required_eval=("baseline_metrics.csv","bootstrap.csv","exact_n_diagnostics.csv","matched_fpr.csv","audit.json","test_summary.json",
 "success_audit.json","prn_dependence.json","historical_b0_gate_equivalence.json","provenance/provenance.json")
 for d,primary,scenarios in ((dev,"development_metrics.csv",("DS1","DS2","DS3","DS4")),(confirm,"confirmatory_metrics.csv",("DS7","DS8"))):
  require_nonempty(d/primary)
  for n in required_eval: require_nonempty(d/n)
  for s in scenarios:
   require_nonempty(d/"per_epoch"/f"{s}.csv"); require_nonempty(d/"per_prn"/f"{s}.csv")
  _validate_evaluation_audits(d)
 source=anchor["source_commit"]
 attestation_path=Path(anchor["test_attestation"]["path"]); attestation=validate_test_attestation(attestation_path,source)
 confirm_manifest_path=Path(anchor["confirm_input_manifest"]["path"]); confirm_inputs=json.loads(require_nonempty(confirm_manifest_path).read_text())
 if confirm_inputs.get("schema")!="gnss-doppler-lab.cmte-a2-confirm-inputs.v1" or confirm_inputs.get("scenarios")!=["DS7","DS8"]: raise ValueError("confirm input manifest schema/scenarios invalid")
 producer_hashes=confirm_inputs.get("producer_hashes",{})
 if set(producer_hashes)!={"converter","wrapper","receiver","exporter","template"} or any(len(str(x))!=64 for x in producer_hashes.values()): raise ValueError("confirm producer hash inventory invalid")
 expected_producers={"converter":file_sha256(ROOT/"src/gnss_doppler_lab/cmte_inputs.py"),"wrapper":file_sha256(ROOT/"src/gnss_doppler_lab/cmte_a2_inputs.py"),
  "receiver":RECEIVER_SHA,"exporter":EXPORTER_SHA,"template":file_sha256(ROOT/"configs/cmte_a2_ds8_receiver.conf")}
 if producer_hashes!=expected_producers or producer_hashes["converter"]!=CONVERTER_SHA: raise ValueError("confirm producer bytes differ from frozen production producers")
 for key,item in confirm_inputs.get("files",{}).items():
  candidate=require_nonempty(item.get("path",""))
  if file_sha256(candidate)!=item.get("sha256") or candidate.stat().st_size!=item.get("bytes"): raise ValueError(f"confirm prepared input inventory mismatch: {key}")
 dev_prov=json.loads(require_nonempty(dev/"provenance/provenance.json").read_text()); confirm_prov=json.loads(require_nonempty(confirm/"provenance/provenance.json").read_text())
 if dev_prov.get("schema")!="gnss-doppler-lab.cmte-a2-evaluation-provenance.v1" or dev_prov.get("tier")!="development": raise ValueError("development provenance schema/tier invalid")
 if confirm_prov.get("schema")!="gnss-doppler-lab.cmte-a2-evaluation-provenance.v1" or confirm_prov.get("tier")!="confirmatory": raise ValueError("confirmatory provenance schema/tier invalid")
 for d in (dev,confirm):
  prov=json.loads((d/"provenance/provenance.json").read_text())
  if prov.get("execution_source_commit")!=source: raise ValueError("evaluation source commit mismatch")
 prereg=ROOT/"configs/cmte_a2_preregistration.json"
 if file_sha256(prereg)!=PREREG_HASHES["configs/cmte_a2_preregistration.json"]: raise ValueError("preregistration changed")
 out=Path(a.out).absolute()
 if out.exists(): raise FileExistsError("final artifact is atomic and non-overwriting")
 staging=out.with_name(out.name+f".tmp-{os.getpid()}"); staging.mkdir(parents=True)
 try:
  provenance_dir=staging/"provenance"; provenance_dir.mkdir()
  copies=((a.trust_anchor,"trust_anchor.json"),(confirm_manifest_path,"confirm_input_manifest.json"),(a.ledger,"one_shot_ledger.json"),
   (dev/"provenance/provenance.json","development.json"),(confirm/"provenance/provenance.json","confirmatory.json"),(attestation_path,"test_attestation.json"),
   (attestation["log"]["path"],"preflight_tests.log"))
  for source_path,name in copies: shutil.copyfile(source_path,provenance_dir/name)
  state_hashes={"checkpoint_sha256":file_sha256(state/"b0_model.pt"),"state_sha256":file_sha256(state/"a2_state.json"),
   "thresholds_sha256":file_sha256(state/"thresholds.json"),"config_sha256":file_sha256(state/"config.json"),"calibration_sha256":file_sha256(state/"calibration.json"),"training_sha256":file_sha256(state/"training.json")}
  eval_state=dev_prov.get("state",{})
  for key in ("checkpoint_sha256","state_sha256","thresholds_sha256","scaler_sha256","qcal_sha256"):
   if not eval_state.get(key) or eval_state.get(key)!=confirm_prov.get("state",{}).get(key): raise ValueError(f"evaluation state hash disagreement: {key}")
  for key in ("checkpoint_sha256","state_sha256","thresholds_sha256"):
   if eval_state[key]!=state_hashes[key]: raise ValueError(f"frozen state hash mismatch: {key}")
  hash_inventory={"schema":"gnss-doppler-lab.cmte-a2-final-hash-inventory.v1","producer_hashes":producer_hashes,
   "prepared_inputs":{key:{**item,"producer_hashes":producer_hashes} for key,item in confirm_inputs["files"].items()},
   "frozen_state":{**state_hashes,"scaler_sha256":eval_state["scaler_sha256"],"qcal_sha256":eval_state["qcal_sha256"]},
   "evidence":{name:file_sha256(provenance_dir/name) for name in ("trust_anchor.json","confirm_input_manifest.json","one_shot_ledger.json","development.json","confirmatory.json","test_attestation.json","preflight_tests.log")}}
  atomic_json(provenance_dir/"hash_inventory.json",hash_inventory)
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
  for name in ("success_audit.json","prn_dependence.json","historical_b0_gate_equivalence.json"):
   shutil.copyfile(confirm/name,staging/"audit"/name)
  provenance={"schema":"gnss-doppler-lab.cmte-a2-final-provenance.v1","source_commit":source,
   "result_commit_sha":a.result_commit_sha,"result_commit_may_be_finalized_without_scientific_metric_changes":True,
   "preregistration_sha256":file_sha256(prereg),"immutable_preregistration_hashes":dict(PREREG_HASHES),
   "immutable_preregistration_commit":PREREG_COMMIT,
   "preregistration_not_edited_after_result_exposure":all(file_sha256(ROOT/rel)==digest and
     __import__("hashlib").sha256(subprocess.check_output(["git","show",f"{PREREG_COMMIT}:{rel}"],cwd=ROOT)).hexdigest()==digest
     for rel,digest in PREREG_HASHES.items()),
   "preregistration_immutability_derivation":"working bytes and prereg commit blobs equal immutable registered SHA-256 values",
   "trust_anchor_sha256":a.expected_anchor_sha256,
   "ledger_sha256":file_sha256(a.ledger),"state_checksums_sha256":file_sha256(state/"checksums.json"),
   "development_checksums_sha256":file_sha256(dev/"checksums.json"),"confirmatory_checksums_sha256":file_sha256(confirm/"checksums.json"),
   "chronology":["normal-only training","development DS1-DS4","pre-holdout freeze","O_EXCL one-shot ledger","confirmatory DS7-DS8","finalization"],
   "producer_exceptions":{"DS4":"development sensitivity; mixed producer; never confirmatory"},
   "sealed_provenance":{name:{"path":f"provenance/{name}","sha256":file_sha256(provenance_dir/name)} for name in ("trust_anchor.json","confirm_input_manifest.json","one_shot_ledger.json","development.json","confirmatory.json","hash_inventory.json","test_attestation.json","preflight_tests.log")}}
  if provenance["preregistration_not_edited_after_result_exposure"] is not True: raise ValueError("preregistration blob equality failed")
  atomic_json(staging/"provenance.json",provenance)
  tests={"pre_campaign_attestation":attestation,"development":json.loads((dev/"test_summary.json").read_text()),
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
  for rel in ("provenance/trust_anchor.json","provenance/confirm_input_manifest.json","provenance/one_shot_ledger.json",
              "provenance/development.json","provenance/confirmatory.json","provenance/hash_inventory.json","provenance/test_attestation.json","provenance/preflight_tests.log"):
   require_nonempty(staging/rel)
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
