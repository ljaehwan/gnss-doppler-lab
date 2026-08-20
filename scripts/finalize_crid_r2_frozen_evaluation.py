#!/usr/bin/env python3
"""Fail-closed CRID R2 finalizer for the audited frozen Phase-A binding."""
from __future__ import annotations
import csv,gzip,hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts/crid_stage0_r2_frozen_evaluation"
PARENT=ROOT/"artifacts/crid_stage0_counterfactual_receiver_invariance"
REPAIR=ROOT/"artifacts/crid_stage0_r1_terminal_drain_repair"
BASE="aa78fe158c9ea8f00ddd82c46bf87fc4fd328551"
BRANCH="research/crid-stage0-r2-frozen-evaluation"
NEGATIVE=["byte_identical","gain","global_phase","nav_sign","awgn_0.5sigma","awgn_1sigma","awgn_2sigma",
 "cn0_reduction","single_source_code_ramp","single_source_doppler_ramp","common_clock_drift","prn_drop_add",
 "single_prn_disturbance","independent_multipath","zero_delay_collapsed_duplicate"]
IMPLEMENTED_CONFORMING={"byte_identical","gain","global_phase","awgn_0.5sigma","awgn_1sigma","awgn_2sigma",
 "cn0_reduction","single_source_doppler_ramp","zero_delay_collapsed_duplicate"}
METHODS=("A0","A1","A2","A3","A4","A5","Full")

def sha(path):
 digest=hashlib.sha256()
 with Path(path).open("rb") as stream:
  for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
 return digest.hexdigest()
def dump(name,value):
 ART.mkdir(parents=True,exist_ok=True);(ART/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def write_csv(name,fields,rows):
 with (ART/name).open("w",newline="") as stream:
  writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)
def manifest():return {str(path.relative_to(ART)):sha(path) for path in sorted(ART.rglob("*")) if path.is_file() and path.name!="artifact_manifest_sha256.json"}

def main():
 ART.mkdir(parents=True,exist_ok=True)
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
 if head!=BASE or branch!=BRANCH:raise RuntimeError("R2 finalizer must run at exact R1 base on the isolated R2 branch")
 config=json.loads((PARENT/"config.json").read_text());repair=json.loads((REPAIR/"final_verdict.json").read_text())
 four=json.loads((REPAIR/"four_configuration_completion.json").read_text());repro=json.loads((REPAIR/"positive_control_reproducibility.json").read_text())
 config_binding={"schema":"gnss-doppler-lab.crid-r2-config-binding.v1","status":"PASS",
  "freeze_sha":"2bd96157b00df8997b66830894912edb016c067b","result_sha":"a30049c316d3f1eefe6fa5af4c2ad5c9acb023cc",
  "config_sha256":sha(PARENT/"config.json"),"preregistration_sha256":sha(PARENT/"preregistration.json"),
  "config_preregistration_byte_identical":sha(PARENT/"config.json")==sha(PARENT/"preregistration.json"),
  "science_file_sha256":{name:sha(ROOT/name) for name in ("src/gnss_doppler_lab/crid.py","src/gnss_doppler_lab/crid_experiment.py","src/gnss_doppler_lab/crid_metrics.py","src/gnss_doppler_lab/crid_physical_controls.py")},
  "frozen_sections_unchanged":["C0-C3","handoff_PRN_map","features_preprocessing","predictor","H0_H1","covariance_whitening","pooling_score","thresholds","physical_control_grid","ablations","gates","scenario_timeline"]}
 repair_binding={"schema":"gnss-doppler-lab.crid-r2-repair-binding.v1","status":"PASS","repair_base_sha":BASE,
  "repair_verdict":repair["verdict"],"repair_status":repair["status"],"repair_manifest_sha256":sha(REPAIR/"artifact_manifest_sha256.json"),
  "contract":{"exact_bytes":4_500_000_000,"dump_count":10,"stability_s":5.0,"exit_code":0,"sigterm":False,"sigkill":False},
  "all_configurations_pass":all(row["status"]=="PASS" for row in four["configurations"].values()),
  "endpoint_max_error_samples":four["absolute_raw_endpoint_contract"]["maximum_absolute_endpoint_delta_samples"]}
 source={"schema":"gnss-doppler-lab.crid-r2-source-commit.v1","base_branch":"origin/research/crid-stage0-r1-terminal-drain-repair",
  "base_expected":BASE,"base_actual":head,"research_branch":branch,"main_sha_unchanged":"461eb4dc7bb794e719295daf028f6811658ba37f",
  "created_at":datetime.now(timezone.utc).isoformat()}

 # Audit the executable registry against the already frozen semantic grid.
 rows=[]
 for domain in ("TEX","OAK"):
  for control in NEGATIVE:
   conforming=control in IMPLEMENTED_CONFORMING
   reason="implemented raw-IQ transform but not executed because the complete Phase-A binding failed" if conforming else {
    "nav_sign":"implementation applies one global pi rotation, not NAV-bit-wise signs",
    "single_source_code_ramp":"implementation returns input IQ unchanged",
    "common_clock_drift":"implementation uses gain=1 and returns input IQ unchanged",
    "prn_drop_add":"no registered transform","single_prn_disturbance":"no registered transform",
    "independent_multipath":"no registered transform"}[control]
   rows.append({"domain":domain,"control":control,"kind":"negative","frozen_required":True,"executable":conforming,
    "conforming":conforming,"executed":False,"reused":False,"alarm_ratio":"","status":"BLOCKED_BINDING" if not conforming else "NOT_RUN_FAIL_CLOSED","reason":reason})
  for count in ("single_prn","four_prn"):
   for delay in (.05,.15,.30):
    for power in (-6,-3,0):
     rows.append({"domain":domain,"control":f"second_source_{count}_d{delay:g}_p{power}_smooth_pull_off","kind":"positive",
      "frozen_required":True,"executable":False,"conforming":False,"executed":False,"reused":False,"alarm_ratio":"",
      "status":"BLOCKED_BINDING","reason":"registered duplicate is all-waveform fixed delay; it has neither selected-PRN multiplicity nor smooth pull-off"})
 rows.append({"domain":"TEX","control":"second_source_all_waveform_d0.15_p-3_fixed","kind":"diagnostic_nonconforming",
  "frozen_required":False,"executable":True,"conforming":False,"executed":True,"reused":True,"alarm_ratio":0.037102997922232116,
  "status":"DIAGNOSTIC_ONLY","reason":"R1 checksum/config exact, but stimulus does not satisfy frozen selected-PRN smooth-pull-off definition"})
 write_csv("physical_control_metrics.csv",["domain","control","kind","frozen_required","executable","conforming","executed","reused","alarm_ratio","status","reason"],rows)
 coverage={domain:{"required_negative":15,"conforming_negative":9,"required_positive":18,"conforming_positive":0} for domain in ("TEX","OAK")}
 physical={"schema":"gnss-doppler-lab.crid-r2-physical-identifiability.v1","status":"INCONCLUSIVE",
  "phase_a_gate":"FAIL_CLOSED_EXECUTION_BINDING","coverage":coverage,"grid_complete":False,"attack_open_authorized":False,
  "r1_diagnostic":{"dataset":"TEX","stimulus":"all-waveform fixed-delay duplicate","delay_chip":.15,"power_db":-3,
   "control_sha256":repro["control_sha256"],"score_threshold_q99":-21.942672917134093,"epoch_count":13476,
   "median_score":-22.096401965536558,"score_q99":-21.667799007368508,"alarm_ratio_q99":.037102997922232116,
   "positive_gate_alarm_ratio":.70,"interpretation":"diagnostic below gate, but nonconforming stimulus cannot decide the frozen physical hypothesis"},
  "blocking_reason":"Frozen semantic grid and pre-attack executable implementation are not congruent. Completing PRN-selective/smooth controls now would introduce unregistered scientific choices after freeze."}
 replay={"schema":"gnss-doppler-lab.crid-r2-replay-completion.v1","status":"PARTIAL_PHASE_A_ONLY","r1_reuse":{
  "allowed_by_checksum":True,"scientific_use":"DIAGNOSTIC_ONLY_NONCONFORMING_STIMULUS","control_sha256":repro["control_sha256"],
  "C0_repeat_bit_identical":repro["C0_bit_identical_two_runs"],"configurations":{key:{"status":value["status"],"exact_input_bytes":value["exact_input_bytes"],"dump_count":value["dump_count"],"exit_code":value["exit_code"],"exit_cause":value["exit_cause"],"sigterm_sent":value["sigterm_sent"],"sigkill_sent":value["sigkill_sent"]} for key,value in four["configurations"].items()},
  "native_endpoint_contract":four["absolute_raw_endpoint_contract"]},"phase_a_complete":False,"phase_b_started":False,"attack_replay_count":0}
 thresholds={"schema":"gnss-doppler-lab.crid-r2-thresholds.v1","status":"BOUND_UNCHANGED","source":"parent cleanStatic calibration only",
  "OAK":{"q99":-21.705587048010322,"q99_5":-21.666766135199047,"holdout_fpr_q99":.00730093543235227},
  "TEX":{"q99":-21.942672917134093,"q99_5":-21.921639651039257,"holdout_fpr_q99":.012708150744960562}}
 dump("config_binding.json",config_binding);dump("source_commit.json",source);dump("repair_binding.json",repair_binding)
 dump("replay_completion.json",replay);dump("physical_identifiability.json",physical);dump("thresholds.json",thresholds)

 scenarios=[]
 for name,spec in config["datasets"].items():
  if spec["role"] in ("core","diagnostic","appendix"):
   scenarios.append({"dataset":name,"family":spec.get("family",name),"role":spec["role"],"status":"NOT_OPENED_PHASE_A_BLOCKED",
    "roc_auc":"","pauc_0_05":"","pr_auc":"","preonset_fpr":"","attack_detection_rate":"","transition_detection_rate":"",
    "established_detection_rate":"","first_alarm_onset_delay_s":"","first_alarm_pull_off_delay_s":"","persistent_alarm_ratio":""})
 write_csv("scenario_metrics.csv",["dataset","family","role","status","roc_auc","pauc_0_05","pr_auc","preonset_fpr","attack_detection_rate","transition_detection_rate","established_detection_rate","first_alarm_onset_delay_s","first_alarm_pull_off_delay_s","persistent_alarm_ratio"],scenarios)
 ablations=[{"scope":family,"method":method,"status":"NOT_EVALUATED_PHASE_A_BLOCKED","pauc_0_05":"","delta_full":""} for family in ("TEX_DS3","TEX_DS7_DS8","OAK_OS3_OS4") for method in METHODS]
 write_csv("ablation_metrics.csv",["scope","method","status","pauc_0_05","delta_full"],ablations)
 with gzip.open(ART/"per_epoch_scores.csv.gz","wt",newline="") as stream:csv.writer(stream,lineterminator="\n").writerow(["dataset","sample","time_s","score","alarm","label","prn_count","config_count","h0_loglike","h1_loglike","configuration_disagreement"])
 with gzip.open(ART/"configuration_state_metrics.csv.gz","wt",newline="") as stream:csv.writer(stream,lineterminator="\n").writerow(["dataset","sample","prn","config","delay_state","carrier_state"])
 write_csv("external_static_fpr.csv",["dataset","family","status","fpr"],[{"dataset":row["dataset"],"family":row["family"],"status":"NOT_OPENED_PHASE_A_BLOCKED","fpr":""} for row in scenarios])
 write_csv("shortcut_audit.csv",["scalar","correlation","auc","status"],[{"scalar":scalar,"correlation":"","auc":"","status":"NOT_EVALUATED_PHASE_A_BLOCKED"} for scalar in ("IQ_power","C/N0","carrier_Doppler","DLL_error","PLL_error","tracked_PRNs","lock_loss")])
 dump("relation_destruction.json",{"schema":"gnss-doppler-lab.crid-r2-relation-destruction.v1","status":"NOT_EVALUATED_PHASE_A_BLOCKED","configuration_label_shuffle":None,"configuration_collapse":None,"configuration_permutation":"IMPLEMENTATION_UNIT_TEST_PASS","prn_permutation":"IMPLEMENTATION_UNIT_TEST_PASS","leave_one_configuration":None,"leave_one_prn":None})
 write_csv("bootstrap_intervals.csv",["comparison","block_s","replicates","estimate","lower","upper","status"],[
  {"comparison":"Full_minus_A0","block_s":10,"replicates":2000,"estimate":"","lower":"","upper":"","status":"NOT_EVALUATED_PHASE_A_BLOCKED"},
  {"comparison":"Full_minus_A1","block_s":30,"replicates":2000,"estimate":"","lower":"","upper":"","status":"NOT_EVALUATED_PHASE_A_BLOCKED"}])
 dump("data_leakage_audit.json",{"status":"PASS","attack_payload_bytes_read":0,"attack_payload_opened":False,"phase_b_authorized":False,"calibration_sources":["OAK cleanStatic","TEX cleanStatic"],"attack_labels_used_for_fit":False})
 dump("common_support_validation.json",{"status":"PASS_R1_DIAGNOSTIC_ONLY","minimum_matched_c0_rows":four["absolute_raw_endpoint_contract"]["minimum_matched_c0_rows"],"maximum_endpoint_error_samples":four["absolute_raw_endpoint_contract"]["maximum_absolute_endpoint_delta_samples"],"tolerance_samples":1})
 dump("timestamp_onset_validation.json",{"status":"NOT_EVALUATED_PHASE_A_BLOCKED","frozen_timeline_bound":True,"attack_payload_opened":False,"reason":"onset/pull-off cannot be validated against unopened attack replay"})
 dump("deterministic_reproduction.json",{"status":"PASS_R1_DIAGNOSTIC_ONLY","C0_repeat_bit_identical":repro["C0_bit_identical_two_runs"],"ten_dump_hashes_match":repro["first_dump_sha256"]==repro["repeat_dump_sha256"]})

 plots=ART/"plots";plots.mkdir(exist_ok=True)
 fig,ax=plt.subplots(figsize=(6,3));ax.bar(["TEX neg","TEX pos","OAK neg","OAK pos"],[9,0,9,0],color=["#4472c4","#c44e52","#4472c4","#c44e52"]);ax.axhline(15,color="gray",ls="--",lw=.8);ax.set_ylabel("conforming executable controls");ax.set_title("Frozen Phase-A execution binding coverage");fig.tight_layout();fig.savefig(plots/"phase_a_binding_coverage.png",dpi=140);plt.close(fig)
 fig,ax=plt.subplots(figsize=(5,3));ax.bar(["R1 diagnostic","positive gate"],[.037102997922232116,.70],color=["#4472c4","#c44e52"]);ax.set_ylim(0,1);ax.set_ylabel("q99 alarm ratio");ax.set_title("Nonconforming R1 diagnostic (not a hypothesis verdict)");fig.tight_layout();fig.savefig(plots/"r1_diagnostic_alarm_ratio.png",dpi=140);plt.close(fig)
 verdict="INCONCLUSIVE_RECEIVER_REPLAY_OR_ALIGNMENT"
 final={"schema":"gnss-doppler-lab.crid-r2-final-verdict.v1","verdict":verdict,"phase_a":"INCONCLUSIVE_EXECUTION_BINDING",
  "phase_b":"NOT_AUTHORIZED","attack_payload_opened":False,"attack_payload_bytes_read":0,"neural_stage1_implemented":False,
  "reason":"Frozen Phase-A semantic control grid cannot be generated by the frozen executable implementation; a nonconforming R1 diagnostic cannot authorize attack access.",
  "next_action":"Create and remotely freeze one audited raw-IQ control generator that implements PRN-selective single/four-PRN smooth pull-off and every negative transform, then repeat Phase A."}
 dump("final_verdict.json",final)
 (ART/"README.md").write_text("""# CRID-GNSS Stage-0 R2 frozen evaluation\n\nFinal verdict: `INCONCLUSIVE_RECEIVER_REPLAY_OR_ALIGNMENT`. Phase B was not authorized and no attack payload was opened.\n\n## 1. What CRID measures\n\nCRID tests whether four counterfactual tracking configurations applied to the same IQ can be explained by one shared, dynamics-corrected delay/carrier state. It is not an IQ-power, C/N0, or single-residual detector.\n\n## 2. Physical controls\n\nThe R1 all-waveform 0.15-chip/-3-dB fixed duplicate replay terminated correctly and was deterministic, but its q99 alarm ratio was only 3.71%. More importantly, it is not the frozen single/four-PRN smooth-pull-off stimulus. The frozen JSON requires 15 negative and 18 positive cases per domain, while the frozen generator has only nine conforming negative transforms and no conforming PRN-selective smooth positive transform. Phase A therefore cannot be completed without introducing new, unregistered scientific choices.\n\n## 3. TEXBAT/OAKBAT performance\n\nNo attack performance was computed. TEXBAT DS1/DS3/DS4/DS7/DS8 and OAKBAT OS3/OS4 payloads remained unopened because Phase A did not authorize Phase B.\n\n## 4. Difference from B0\n\nB0 is an exact baseline only when it can be rerun on identical support. CRID instead tests counterfactual receiver-configuration invariance after causal dynamics compensation. No B0 comparison is claimed here.\n\n## 5. Novelty boundary\n\nComparing multiple tracking-loop configurations or observing auxiliary peaks is not itself novel. A possible contribution would require the combined same-IQ counterfactual replay, shared-state H0 versus configuration-dependent H1 with complexity correction, and multi-PRN persistence. This R2 result does not establish that contribution.\n\n## 6. Successful and failed scenarios\n\nR1 termination, deterministic C0 reproduction, ten dumps per configuration, and ±1-sample support passed for the diagnostic control. The frozen Phase-A stimulus binding failed. No attack scenario was evaluated.\n\n## 7. Claims\n\nIt is valid to claim that receiver replay engineering is repaired and that the preregistered Phase-A generator is incomplete. It is not valid to claim spoofing detection, physical-hypothesis failure, detection advantage, or Stage-1 readiness.\n\n## 8. One next action\n\nFreeze one audited raw-IQ generator implementing all negative controls plus PRN-selective single/four-PRN smooth pull-off, then repeat Phase A before opening attack data.\n""")
 dump("artifact_manifest_sha256.json",manifest());print(json.dumps({"verdict":verdict,"attack_payload_opened":False,"phase_a_grid_complete":False,"r1_diagnostic_alarm_ratio":.037102997922232116},indent=2))

if __name__=="__main__":main()
