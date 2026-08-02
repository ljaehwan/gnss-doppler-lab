from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
def load_script(name):
 path=ROOT/"scripts"/name; spec=importlib.util.spec_from_file_location("remaining_"+name.replace(".","_"),path); assert spec and spec.loader
 mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod
def event_frame(recording, raw):
 return pd.DataFrame([{"physical_recording_id":recording,"window_end_s":float(i+1),"rmse_values":np.array([x])} for i,x in enumerate(raw)])
def test_public_evaluator_is_development_only():
 mod=load_script("eval_cmte_a2_texbat.py")
 with pytest.raises(SystemExit): mod.main(["--tier","confirmatory","--state-dir","x","--scenario","DS7=x=y","--out","z"])
def test_confirm_ledger_precedes_holdout_and_failure_is_recorded(tmp_path,monkeypatch):
 mod=load_script("confirm_cmte_a2_texbat.py"); order=[]
 monkeypatch.setattr(mod,"validate_trust_anchor",lambda *a,**k: order.append("anchor") or {"state_dir":"state"})
 real_create=mod.create_ledger
 def create(path,sha): order.append("ledger"); return real_create(path,sha)
 monkeypatch.setattr(mod,"create_ledger",create)
 def resolve(anchor): order.append("resolve"); raise ValueError("synthetic holdout failure")
 monkeypatch.setattr(mod,"resolve_confirmatory_inputs",resolve)
 ledger=tmp_path/"ledger.json"
 with pytest.raises(ValueError,match="synthetic holdout"):
  mod.main(["--trust-anchor",str(tmp_path/"anchor"),"--expected-sha256","a"*64,"--ledger",str(ledger),"--out",str(tmp_path/"out")])
 assert order==["anchor","ledger","resolve"] and json.loads(ledger.read_text())["status"]=="failed"
 with pytest.raises(FileExistsError): mod.main(["--trust-anchor",str(tmp_path/"anchor"),"--expected-sha256","a"*64,"--ledger",str(ledger),"--out",str(tmp_path/"out")])
def test_internal_confirm_api_rejects_forged_capability():
 mod=load_script("eval_cmte_a2_texbat.py")
 with pytest.raises(PermissionError,match="capability"): mod._confirm_main([],object())
def test_comparator_ewma_no_role_reset_and_recording_only_reset():
 from gnss_doppler_lab.cmte_a2 import b0_exact_scores,b0_enhanced_scores
 thresholds={"q50":.5,"q70":.5,"q80":.5}; rates={"q50":.5,"q70":.3,"q80":.2}
 prior=event_frame("r",[1.,1.]); test=event_frame("r",[0.]); test["window_end_s"]+=2; whole=pd.concat([prior,test],ignore_index=True)
 exact_whole=b0_exact_scores(whole,thresholds); enhanced_whole=b0_enhanced_scores(whole,thresholds,rates)
 assert exact_whole.iloc[-1].score != b0_exact_scores(test,thresholds).iloc[-1].score
 assert enhanced_whole.iloc[-1].score != b0_enhanced_scores(test,thresholds,rates).iloc[-1].score
 two=pd.concat([event_frame("a",[1.]),event_frame("b",[1.])],ignore_index=True)
 ex=b0_exact_scores(two,thresholds); en=b0_enhanced_scores(two,thresholds,rates)
 assert ex.iloc[0].score==ex.iloc[1].score and en.iloc[0].score==en.iloc[1].score
def test_bootstrap_blocks_split_at_prereg_subphase_and_point_is_all_rows():
 from gnss_doppler_lab.cmte_a2 import _complete_blocks,bootstrap_metrics
 rows=[]
 for i,t in enumerate(np.arange(110.5,170.5,.5)):
  sub="ramp" if t<=130 else "takeover" if t<=150 else "persistent"
  rows.append({"scenario":"DS7","physical_recording_id":"r","producer_chain_id":"p","window_end_s":t,"stable_pre":False,"post":True,"persistent":sub=="persistent","prereg_subphase":sub,"score":float(i),"alarm":i%3==0})
 f=pd.DataFrame(rows); blocks=_complete_blocks(f,"post",20,.5)
 for idx in blocks: assert f.loc[idx,"prereg_subphase"].nunique()==1
 got=bootstrap_metrics(f,reps=5); assert got["post_detection_rate"]["point"]==pytest.approx(float(f.alarm.mean()))
def test_matched_fpr_uses_exact_observed_candidates():
 from gnss_doppler_lab.cmte_a2 import matched_fpr_diagnostic
 values=np.array([0.,0.,2.,5.]); got=matched_fpr_diagnostic({"CMTE-A2":[0.,1.,2.,3.],"B0-Exact":values},primary_threshold=1.5)
 item=got["models"]["B0-Exact"]; assert item["threshold"] in np.unique(values)
 assert got["grid"]=="unique_observed_threshold_role_scores" and item["threshold_role_strict_exceedance"]==float(np.mean(values>item["threshold"]))
def test_receiver_config_is_single_section_actual_syntax():
 text=(ROOT/"configs/cmte_a2_ds8_receiver.conf").read_text(); assert text.count("[GNSS-SDR]")==1 and text.count("[")==1
 for token in ("Channel.signal=1C","InputFilter.input_item_type=gr_complex","InputFilter.output_item_type=gr_complex","Resampler.item_type=gr_complex","Tracking_1C.order=3","Acquisition_1C.dump=false"): assert token in text
def test_confirm_builder_rejects_wrapper_disagreement(tmp_path):
 import gnss_doppler_lab.cmte_a2_campaign as c
 files={}
 for n in ("d7raw","d7npz","d7node","d7manifest","d8raw","d8npz","d8conf","d8prep","d8node","d8manifest"): files[n]=tmp_path/n; files[n].write_text(n)
 from gnss_doppler_lab.cmte_a2_inputs import CONVERTER_SEMANTICS,_fingerprint
 fp=_fingerprint(c.CONVERTER_SHA)
 def inp(s,npz,node,wrapper): return {"scenario":s,"campaign_converter_fingerprint":fp,"converter_content_sha256":c.CONVERTER_SHA,"converter_semantics":CONVERTER_SEMANTICS,"wrapper_content_sha256":wrapper,"source_sha256":c.file_sha256(npz),"node_sha256":c.file_sha256(node)}
 files["d7manifest"].write_text(json.dumps(inp("DS7",files["d7npz"],files["d7node"],"a"*64))); files["d8manifest"].write_text(json.dumps(inp("DS8",files["d8npz"],files["d8node"],"b"*64)))
 files["d8prep"].write_text(json.dumps({"status":"prepared","raw_sha256":c.file_sha256(files["d8raw"]),"npz":{"sha256":c.file_sha256(files["d8npz"])},"rendered_config_sha256":c.file_sha256(files["d8conf"])}))
 with pytest.raises(ValueError,match="wrapper"):
  c.build_confirm_input_manifest(tmp_path/"out",ds7_raw=files["d7raw"],ds7_npz=files["d7npz"],ds7_node=files["d7node"],ds7_input_manifest=files["d7manifest"],ds8_raw=files["d8raw"],ds8_npz=files["d8npz"],ds8_rendered_config=files["d8conf"],ds8_prep_manifest=files["d8prep"],ds8_node=files["d8node"],ds8_input_manifest=files["d8manifest"],expected_ds7_raw_sha=c.file_sha256(files["d7raw"]),expected_ds8_raw_sha=c.file_sha256(files["d8raw"]),code_hashes={"converter":c.CONVERTER_SHA,"wrapper":"a"*64,"receiver":c.RECEIVER_SHA,"exporter":c.EXPORTER_SHA,"template":"f"*64})

def test_actual_pinned_receiver_config_smoke_subprocess(tmp_path):
 import subprocess
 receiver=Path("/home/ubuntu/build-gnss-sdr-complex9/build-complex/src/main/gnss-sdr")
 if not receiver.is_file(): pytest.skip("pinned complex9 binary unavailable")
 evidence=tmp_path/"smoke.json"
 r=subprocess.run([sys.executable,str(ROOT/"scripts/smoke_cmte_a2_receiver_config.py"),"--output-json",str(evidence)],
                  cwd=ROOT,text=True,capture_output=True,check=False)
 assert r.returncode==0,r.stdout+r.stderr
 doc=json.loads(evidence.read_text()); assert doc["passed"] is True and doc["actual_binary_executed"] is True
 assert doc["controlled_exit"]=="too_short_no_signal" and doc["holdout_opened"] is False and doc["exporter_fixture_validated"] is True

def test_guarded_confirm_subprocess_anchor_failure_creates_no_ledger(tmp_path):
 import subprocess
 ledger=tmp_path/"ledger.json"
 r=subprocess.run([sys.executable,str(ROOT/"scripts/confirm_cmte_a2_texbat.py"),"--trust-anchor",str(tmp_path/"missing-anchor"),
                   "--expected-sha256","a"*64,"--ledger",str(ledger),"--out",str(tmp_path/"out")],cwd=ROOT,text=True,capture_output=True)
 assert r.returncode!=0 and not ledger.exists() and not (tmp_path/"out").exists()

def test_public_evaluator_confirm_subprocess_rejected_before_paths(tmp_path):
 import subprocess
 r=subprocess.run([sys.executable,str(ROOT/"scripts/eval_cmte_a2_texbat.py"),"--tier","confirmatory","--state-dir",str(tmp_path/"missing"),
                   "--scenario",f"DS7={tmp_path/'holdout'}={tmp_path/'manifest'}","--out",str(tmp_path/"out")],cwd=ROOT,text=True,capture_output=True)
 assert r.returncode!=0 and "invalid choice" in r.stderr and not (tmp_path/"out").exists()

def test_historical_production_helper_hashes_and_equivalence(tmp_path):
 from gnss_doppler_lab.cmte_a2 import historical_gate_equivalence_files
 out=tmp_path/"historical.json"
 doc=historical_gate_equivalence_files(ROOT/"artifacts/cmte_texbat_poc/per_prn/DS1.csv",
   ROOT/"artifacts/cmte_a2_historical_b0/ds1_golden_events.csv",evaluator_path=ROOT/"scripts/eval_btail_support_gate.py",
   calibration_path=ROOT/"configs/detectors/texbat_btail_gate_v1.json",evidence_path=out)
 assert doc["passed"] is True and doc["max_absolute_error"]<=1e-12 and doc["strict_alarm_equal"] is True
 for key in ("evaluator_sha256","implementation_sha256","input_sha256","golden_sha256","calibration_sha256"):
  assert len(doc[key])==64
 assert doc["holdout_accessed"] is False and out.stat().st_size>0

def test_freeze_and_finalizer_require_exact_machine_audits():
 campaign=(ROOT/"src/gnss_doppler_lab/cmte_a2_campaign.py").read_text()
 final=(ROOT/"scripts/finalize_cmte_a2_campaign.py").read_text()
 for name in ("success_audit.json","prn_dependence.json","historical_b0_gate_equivalence.json"):
  assert name in campaign and name in final
 for schema in ("gnss-doppler-lab.cmte-a2-success-audit.v1","gnss-doppler-lab.cmte-a2-prn-dependence.v1",
                "gnss-doppler-lab.cmte-a2-historical-gate-equivalence.v1"):
  assert schema in final
 assert "preregistration_not_edited_after_result_exposure" in final and "immutable_preregistration_hashes" in final

def test_exact_n_empty_required_stratum_is_explicit_na_and_unpooled():
 from gnss_doppler_lab.cmte_a2 import exact_n_diagnostics
 frame=pd.DataFrame({"tier":["confirmatory"],"scenario":["DS7"],"model":["CMTE-A2"],"phase":["stable_pre"],
  "tracked_prn_count":[4],"score":[1.],"alarm":[False]})
 required=[{"tier":"confirmatory","scenario":"DS7","model":"CMTE-A2","phase":"persistent"}]
 rows,audit=exact_n_diagnostics(frame,required_strata=required)
 empty=rows[rows.phase=="persistent"].iloc[0]
 assert empty.epoch_count==0 and pd.isna(empty.N) and pd.isna(empty.score_median) and empty.na_reason=="no_eligible_epochs"
 assert not bool(empty.aggregation_changed) and not bool(empty.sparse_pooled) and audit["passed"] is True
