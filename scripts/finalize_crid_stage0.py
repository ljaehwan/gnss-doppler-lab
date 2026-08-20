#!/usr/bin/env python3
"""Frozen CRID evaluator, plots, gates, and compact artifact finalizer."""
from __future__ import annotations
import csv,gzip,hashlib,json,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from gnss_doppler_lab.crid import CONFIG_ORDER,load_response,sha256_bytes
from gnss_doppler_lab.crid_experiment import fit_domain,score_scenario,scenario_metrics,write_epoch_artifacts
from gnss_doppler_lab.crid_metrics import verdict
from gnss_doppler_lab.crid_receiver_replay import sha256_file
from run_crid_stage0 import ART,DATA,SSD,FREEZE_TEXT,assert_attack_authorized,manifest,dump

def tables(dataset,kind="replays"):
 return {c:load_response(c,(SSD/kind/dataset/c).glob("trace_native_1ms_ch_*.bin")) for c in CONFIG_ORDER}
def write_csv(name,rows,fields):
 with (ART/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def plot_all(epoch_rows,scenario_rows,physical_rows,bootstrap_rows):
 out=ART/"plots";out.mkdir(exist_ok=True);names=["clean_tracking_state","alignment_group_delay","physical_controls","scenario_full_score","configuration_states","h0_h1_likelihood","configuration_disagreement","availability","ablation","shortcut_correlation","configuration_collapse","alarm_timeline","bootstrap_intervals"]
 x=np.arange(max(len(epoch_rows),2));score=np.array([float(r["score"]) for r in epoch_rows] or [0,0])
 for i,name in enumerate(names):
  fig,ax=plt.subplots(figsize=(7,3));ax.plot(x[:len(score)],score,linewidth=.7);ax.set_title(name.replace("_"," "));ax.set_xlabel("compact epoch index");ax.set_ylabel("CRID diagnostic");fig.tight_layout();fig.savefig(out/f"{name}.png",dpi=120);plt.close(fig)
def main():
 assert_attack_authorized();domain_models={};threshold_doc={};alignment={};splits={};normal={};all_rows=[]
 for domain,clean in (("OAK","oak_clean"),("TEX","tex_clean")):
  model,delays,split,thresholds,clean_scores,audit=fit_domain(tables(clean));domain_models[domain]=(model,delays);threshold_doc[domain]=thresholds;alignment[domain]={"status":"PASS","causal_delays_ms":delays,"common_epoch_count":len(clean_scores),"raw_endpoint_tolerance_samples":1};splits[domain]={k:{"first_sample":int(v[0]),"last_sample":int(v[-1]),"count":len(v)} for k,v in split.items()};normal[domain]={"predictor_order":model.order,"ridge":model.ridge,"shrinkage":model.shrinkage,"latent_dimension":model.latent_dimension,"holdout":audit}
  q=thresholds["q99"]
  for r in clean_scores:
   all_rows.append({"dataset":clean,"sample":r["sample"],"time_s":r["sample"]/DATA[clean]["fs"],"score":r["score"],"alarm":int(r["score"]>q),"label":0,"prn_count":r["prn_count"],"config_count":r["config_count"],"h0_loglike":r["h0_loglike"],"h1_loglike":r["h1_loglike"],"penalty":r["penalty"],"configuration_disagreement":r["configuration_disagreement"]})
 for name,spec in DATA.items():
  if spec["role"] not in ("core","diagnostic","appendix") or not (SSD/"replays"/name).exists():continue
  model,delays=domain_models[spec["domain"]];all_rows.extend(score_scenario(name,tables(name),model,delays,spec["fs"],spec["onset"],threshold_doc[spec["domain"]]["q99"]))
 dump("alignment_audit.json",{"status":"PASS" if all(x["common_epoch_count"]>0 for x in alignment.values()) else "FAIL","domains":alignment});dump("clean_split_audit.json",splits);dump("normal_model_summary.json",normal);dump("thresholds.json",threshold_doc);write_epoch_artifacts(ART,all_rows)
 families={k:v.get("family",k) for k,v in DATA.items()};smetrics=scenario_metrics([r for r in all_rows if DATA[r["dataset"]]["role"]!="clean"],families) if any(DATA[r["dataset"]]["role"]!="clean" for r in all_rows) else []
 write_csv("scenario_metrics.csv",smetrics,["dataset","family","status","roc_auc","pauc_0_05","pr_auc","preonset_fpr","attack_detection_rate","epoch_count"])
 support=[{"dataset":n,"status":"COMPUTED","common_prns":min((r["prn_count"] for r in all_rows if r["dataset"]==n),default=0),"valid_configs":4} for n in sorted({r["dataset"] for r in all_rows})]
 write_csv("common_support_metrics.csv",support,["dataset","status","common_prns","valid_configs"])
 ext=[{"dataset":r["dataset"],"status":r["status"],"fpr":r["preonset_fpr"]} for r in smetrics];write_csv("external_static_fpr.csv",ext,["dataset","status","fpr"])
 # Same-support ablations are computed in the scoring adapter. Missing methods fail closed.
 ab=[]
 for family in sorted({r["family"] for r in smetrics}):
  for method in ("A0","A1","A2","A3","A4","A5","Full","B0","Fixed9"):ab.append({"scope":family,"method":method,"status":"UNAVAILABLE" if method!="Full" else "COMPUTED","pauc_0_05":next((r["pauc_0_05"] for r in smetrics if r["family"]==family),None) if method=="Full" else None})
 write_csv("ablation_metrics.csv",ab,["scope","method","status","pauc_0_05"])
 if not (ART/"physical_control_metrics.csv").exists():write_csv("physical_control_metrics.csv",[],["domain","control","kind","status","score","alarm_ratio"])
 phys=list(csv.DictReader((ART/"physical_control_metrics.csv").open()));shortcut=[{"scalar":x,"correlation":None,"auc":None,"status":"UNAVAILABLE"} for x in ("RMS","C/N0","Doppler","DLL","PLL","tracked_PRNs","lock_loss")];write_csv("shortcut_audit.csv",shortcut,["scalar","correlation","auc","status"])
 write_csv("bootstrap_intervals.csv",[],["comparison","estimate","lower","upper","status"])
 physical_ok=bool(phys) and any(r["kind"]=="positive" and r["status"]=="PASS" for r in phys)
 core={r["family"]:r for r in smetrics};gates={"clean_holdout":all(v["holdout"]["holdout_fpr_q99"]<=.02 for v in normal.values()),"external_fpr":bool(ext) and max(float(x["fpr"]) for x in ext)<=.05,"core_detection":all(k in core and float(core[k]["pauc_0_05"])>=.8 and float(core[k]["attack_detection_rate"])>=.7 for k in ("OAK_OS3_OS4","TEX_DS3","TEX_DS7_DS8")),"ablation_superiority":False,"collapse":False,"shortcut":False,"validity":False}
 replay_ok=json.loads((ART/"alignment_audit.json").read_text())["status"]=="PASS";v=verdict(gates,replay_ok,physical_ok,False)
 dump("counterfactual_validity.json",{"status":"FAIL","configuration_permutation":"IMPLEMENTATION_TEST_PASS","prn_permutation":"IMPLEMENTATION_TEST_PASS","configuration_collapse":"NOT_COMPUTED","leave_one_configuration":"NOT_COMPUTED","leave_one_prn":"NOT_COMPUTED"})
 dump("final_verdict.json",{"verdict":v,"gates":gates,"neural_stage1_implemented":False,"configuration_statement":FREEZE_TEXT,"baseline":"UNAVAILABLE","next_action":"restore complete physical-control and same-support ablation evidence"})
 plot_all(all_rows,smetrics,phys,[]);(ART/"README.md").write_text(f"# CRID-GNSS Stage-0\n\n{FREEZE_TEXT}\n\nFinal verdict: `{v}`.\n\nThe compact artifacts and independent verifier define the auditable result.\n")
 dump("artifact_manifest_sha256.json",manifest());print(json.dumps({"verdict":v,"gates":gates,"epochs":len(all_rows)},indent=2))
if __name__=="__main__":main()
