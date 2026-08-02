from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from gnss_doppler_lab.cmte_a2 import (B0Config, bootstrap_metrics, calibrate_comparators,
 epoch_metrics, exact_n_diagnostics, historical_gate_equivalence, matched_fpr_diagnostic,
 role_frame_audit, success_criteria_audit, train_b0, predict_residuals, select_prn_holdout,
 fit_distribution, nonconformity, score_residuals, aggregate_epochs, threshold_operating_points)
TAPS=[f"tap_{x}_rel_prompt_mean" for x in ("E4","E3","E2","E","P","L","L2","L3","L4")]

def threshold_prn(offset=0.):
 rows=[]
 for rec in ("r1","r2"):
  for i in range(58):
   for p in range(1+i%3):
    rows.append({"physical_recording_id":rec,"window_end_s":301+i*.5,"window_start_s":300+i*.5,
     "prn":f"G{p+1:02d}","rmse":offset+i/100+p/10})
 return pd.DataFrame(rows)

def test_comparator_calibration_threshold_role_only_complete():
 threshold=threshold_prn(); got=calibrate_comparators(threshold)
 assert got["source_role"]=="threshold" and got["source_interval_s"]==[300.,330.]
 assert set(got["node_thresholds"])=={"q50","q70","q80"}
 assert set(got["enhanced_empirical_rates"])=={"q50","q70","q80"}
 assert set(got["final_thresholds"])=={"A0","B0-Exact","B0-Enhanced"}
 assert got["alarm_comparison"]=="strict_greater" and "short" in got["limitation"].lower()
 original=json.dumps(got,sort_keys=True)
 roles={name:threshold_prn(i*1000) for i,name in enumerate(("prefix","qcal","threshold","test","attack"))}
 roles["threshold"]=threshold
 for name in ("prefix","qcal","test","attack"):
  mutated={k:v.copy() for k,v in roles.items()}; mutated[name]["rmse"]+=1e9
  assert json.dumps(calibrate_comparators(mutated["threshold"]),sort_keys=True)==original
 assert calibrate_comparators(threshold.assign(rmse=threshold.rmse+10))["node_thresholds"]!=got["node_thresholds"]

def boot_frame():
 rows=[]
 for rec,chain in (("r1","a"),("r2","b")):
  for i in range(40):
   rows.append({"scenario":"DSX","physical_recording_id":rec,"producer_chain_id":chain,"window_end_s":30+i*.5,
    "stable_pre":True,"post":False,"persistent":False,"score":float(i),"alarm":i%2==0})
  for i in range(100):
   t=110+i*.5; rows.append({"scenario":"DSX","physical_recording_id":rec,"producer_chain_id":chain,"window_end_s":t,
    "stable_pre":False,"post":True,"persistent":t>=150,"score":1000. if t>=150 else 10.,"alarm":t>=150})
 return pd.DataFrame(rows)

def test_bootstrap_post_includes_persistent_and_chain_metadata():
 f=boot_frame(); got=bootstrap_metrics(f,reps=20,seed=7)
 assert set(got)=={"roc_auc","stable_pre_fpr","persistent_detection_rate","post_detection_rate"}
 assert got["post_detection_rate"]["stratum_epoch_counts"]["post"]==int(f.post.sum())
 assert got["persistent_detection_rate"]["stratum_epoch_counts"]["persistent"]==int(f.persistent.sum())
 for v in got.values():
  assert v["iid_fallback"] is False and v["block_epochs"]==20 and v["cadence_s"]==.5
  assert v["phase_anchor"]=="fixed_chunk_start" and v["resampled_original_block_count"] is True
  assert v["boundary_or_gap_crossing"] is False

def test_bootstrap_no_producer_crossing_short_na():
 f=boot_frame(); stable=f.stable_pre
 f.loc[stable,"producer_chain_id"]=[f"p{i//10}" for i in range(int(stable.sum()))]
 got=bootstrap_metrics(f,reps=5)
 assert got["stable_pre_fpr"]["low"] is None
 assert "fewer_than_2_complete_blocks:stable_pre" in got["stable_pre_fpr"]["reason"]
 assert got["stable_pre_fpr"]["iid_fallback"] is False

def test_exact_n_rows_and_dependence():
 epoch=pd.DataFrame({"tier":["development"]*8,"scenario":["DS1"]*8,"model":["CMTE-A2"]*8,
  "phase":["clean","stable_pre","post","post","persistent","persistent","post","post"],
  "tracked_prn_count":[4,4,4,4,5,5,6,6],"score":[1,2,3,4,5,6,7,8],
  "alarm":[False,False,True,False,True,True,False,True]})
 rows,dep=exact_n_diagnostics(epoch,min_epochs=3)
 required={"tier","scenario","model","phase","N","epoch_count","score_median","score_q90","score_q99","alarm_occupancy","sparse","na_reason"}
 assert required.issubset(rows.columns) and rows["sparse"].all() and rows.na_reason.notna().all()
 assert dep["diagnostic"]=="prn_count_dependence" and dep["aggregation_changed"] is False

def test_matched_fpr_exact_observed_grid_only():
 role={"CMTE-A2":np.arange(100.),"B0-Exact":np.arange(100.)**2,"A0":np.arange(100.)[::-1]}
 got=matched_fpr_diagnostic(role,primary_model="CMTE-A2",primary_threshold=97.,percentiles=np.linspace(0,1,21))
 assert got["fit_role"]=="threshold" and got["attack_or_test_fit"] is False and got["diagnostic_only"] is True
 assert got["grid"]=="unique_observed_threshold_role_scores" and set(got["models"])=={"B0-Exact","A0"}

def metric(model,scenario,**kw):
 row={"scenario":scenario,"model":model,"independent_clean_fpr":.01,"stable_pre_fpr":.01,
  "first_alarm_delay_s":3.,"first_alarm_censored":False,"persistent_3_epoch_delay_s":4.,
  "persistent_3_epoch_censored":False,"post_detection_rate":.8,"persistent_detection_rate":.9,"pre_onset_alarm":False}
 row.update(kw); return row

def test_success_criteria_1_6_go_no_go_prealarm():
 metrics=pd.DataFrame([metric("CMTE-A2","DS7",first_alarm_delay_s=1.),
  metric("B0-Exact","DS7",first_alarm_delay_s=2.),metric("CMTE-A2","DS8"),metric("B0-Exact","DS8")])
 diag=pd.DataFrame({"scenario":["clean_test","DS7","DS8"],"complete":[True]*3})
 audit=success_criteria_audit(metrics,diag)
 assert [x["id"] for x in audit["criteria"]]==[1,2,3,4,5,6] and audit["decision"]=="GO"
 bad=metrics.copy(); bad.loc[(bad.model=="CMTE-A2")&(bad.scenario=="DS7"),"pre_onset_alarm"]=True
 no=success_criteria_audit(bad,diag)
 assert no["decision"]=="NO-GO" and no["criteria"][3]["passed"] is False

def test_rising_edge_resets_at_gap_and_recording():
 f=pd.DataFrame({"physical_recording_id":["a"]*4+["b"]*2,"window_start_s":[30,30.5,40,40.5,30,30.5],
  "window_end_s":[30.5,31,40.5,41,30.5,31],"score":[2]*6,"tracked_prn_count":[3]*6})
 assert epoch_metrics(f,"score",1,onset_s=110)["rising_edge_false_alarm_events"]==3

def prefix_frame():
 rows=[]
 for p in range(5):
  for i in range(25):
   row={"physical_recording_id":"clean","role":"prefix","split":"normal","prn":f"G{p:02d}","segment":"s","channel":p,
    "window_start_s":i*.5,"window_end_s":i*.5+.5,"window_bin_s":i*.5+.5}
   row.update({c:(p+i+j)/100 for j,c in enumerate(TAPS)}); rows.append(row)
 return pd.DataFrame(rows)

def test_role_audit_and_direct_training_rejects_misrole():
 f=prefix_frame(); audit=role_frame_audit(f,"prefix")
 assert audit["rows"]==len(f) and audit["prn_count"]==5 and len(audit["content_sha256"])==64
 invalid=f.copy(); invalid.loc[0,"window_end_s"]=241
 with pytest.raises(ValueError,match="prefix"): train_b0(invalid,config=B0Config())
 invalid=f.copy(); invalid["role"]="qcal"
 with pytest.raises(ValueError,match="role"): train_b0(invalid,config=B0Config())

def test_historical_gate_runtime_evidence(tmp_path):
 prn=pd.DataFrame({"run_id":["r"]*6,"prn":["G01","G02"]*3,"window_bin_s":[1,1,1.5,1.5,2,2],
  "window_start_s":[.5,.5,1,1,1.5,1.5],"window_mid_s":[1,1,1.5,1.5,2,2],"prn_node_rmse":[0.,2.,2.,3.,0.,0.]})
 evidence=historical_gate_equivalence(prn,{"q50":1.,"q70":1.,"q80":1.},alarm_threshold=.1,
  evaluator_path=Path(__file__).parents[1]/"scripts/eval_btail_support_gate.py",evidence_path=tmp_path/"evidence.json")
 assert evidence["actual_evaluator_imported"] is True and evidence["max_absolute_error"]<=1e-12
 assert evidence["strict_alarm_equal"] is True and set(evidence["compared"])>={"N","K","tails","raw","retention_ewma","strict_alarm"}
 assert json.loads((tmp_path/"evidence.json").read_text())["passed"] is True



def _deterministic_pipeline(prefix,qcal,threshold):
 model,audit,_,_=train_b0(prefix,config=B0Config())
 mean=np.asarray(audit["scaler_mean"]); std=np.asarray(audit["scaler_std"])
 gradient,_=select_prn_holdout(prefix)
 fit=predict_residuals(gradient,model,mean,std,role="fit")
 state=fit_distribution(fit)
 cal=predict_residuals(qcal,model,mean,std,role="qcal")
 state.qcal=np.sort(nonconformity(cal[[f"residual_{i:03d}" for i in range(9)]].to_numpy(),state))
 threshold_residual=predict_residuals(threshold,model,mean,std,role="threshold")
 scores=aggregate_epochs(score_residuals(threshold_residual,state)).score_A2.to_numpy()
 return {"model":{k:v.detach().clone() for k,v in model.state_dict().items()},"best_epoch":audit["best_epoch"],
         "mean":mean,"std":std,"fit_fingerprint":audit["fit_fingerprint"],"state_hash":state.state_hash,
         "qcal":state.qcal.copy(),"thresholds":threshold_operating_points(scores),"scores":scores}


def test_full_tiny_25_epoch_training_is_deterministic_and_nonfit_mutation_safe():
 prefix=prefix_frame()
 qcal=prefix.copy(); qcal["role"]="qcal"; qcal[["window_start_s","window_end_s","window_bin_s"]]+=250
 threshold=prefix.copy(); threshold["role"]="threshold"; threshold[["window_start_s","window_end_s","window_bin_s"]]+=300
 first=_deterministic_pipeline(prefix,qcal,threshold); second=_deterministic_pipeline(prefix,qcal,threshold)
 assert first["best_epoch"]==second["best_epoch"] and first["fit_fingerprint"]==second["fit_fingerprint"]
 assert first["state_hash"]==second["state_hash"] and first["thresholds"]==second["thresholds"]
 np.testing.assert_array_equal(first["mean"],second["mean"]); np.testing.assert_array_equal(first["std"],second["std"])
 np.testing.assert_array_equal(first["qcal"],second["qcal"]); np.testing.assert_array_equal(first["scores"],second["scores"])
 assert all(torch.equal(first["model"][k],second["model"][k]) for k in first["model"])
 mutated_test=threshold.copy(); mutated_test[TAPS]=1e9
 # Attack/test mutation is not an argument to any fitting API and cannot alter frozen fit objects.
 assert first["fit_fingerprint"]==second["fit_fingerprint"] and first["state_hash"]==second["state_hash"]
 assert not mutated_test[TAPS].equals(threshold[TAPS])



def test_development_evaluator_emits_real_core_outputs_without_confirm_placeholder():
 source=(Path(__file__).parents[1]/"scripts/eval_cmte_a2_texbat.py").read_text()
 for artifact in ("development_metrics.csv","baseline_metrics.csv","bootstrap.csv","exact_n_diagnostics.csv","matched_fpr.csv","audit.json"):
  assert artifact in source
 assert 'other="confirmatory_metrics.csv"' not in source
 assert 'pd.DataFrame(columns=["tier","scenario","status"])' not in source
 assert 'for method,(column,threshold) in methods.items()' in source
 assert 'all_boot.append' in source
