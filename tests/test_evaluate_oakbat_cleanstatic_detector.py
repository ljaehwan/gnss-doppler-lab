import importlib.util,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
P=Path(__file__).parents[1]/"scripts/evaluate_oakbat_cleanstatic_detector.py"
def mod():
 s=importlib.util.spec_from_file_location("oe",P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def quality_score_metadata(times):
 times=np.asarray(times,float);target=np.arange(200,200+len(times))
 return {"channel":0,"segment_index":0,"prn_segment_ordinal":0,"continuity_block_index":0,"target_window_index":target,"target_sequence_position":np.arange(12,12+len(times)),"epoch_count":50,"tracking_age_s":target*.5,"continuity_age_s":times-times[0],"segment_start_s":times-target*.5,"history_start_window_index":target-12,"history_end_window_index":target-1,"history_start_s":times-6.,"history_end_s":times+.5,"history_length":12,"reacquisition_flag":0,"sequence_restart_flag":0,"history_same_segment_flag":1}
def test_auc_ties():assert mod().probability_auc([1,2,2],[2,2,3])==pytest.approx(7/9)
def test_alias_and_restriction():
 m=mod();assert m.scenario_contract("os1")["official_scenario_id"]=="os1a"
 with pytest.raises(ValueError):m.scenario_contract("cleanStatic")
def test_availability_sustained_and_pre_fp():
 m=mod();k=m.gate_lib.FINAL_SCORE;e=pd.DataFrame({"window_start_s":[109.,130.,130.5,131.],k:[9,9,9,9]});d=m.detection_report(e,5);assert d["first_detection_online_availability_s"]==131 and d["first_detection_delay_s"]==11;assert d["first_three_consecutive_online_availability_s"]==132 and d["pre_onset_false_positives"]==1
def test_frozen_load_before_attack_access(tmp_path,monkeypatch):
 m=mod();seen=[];monkeypatch.setattr(m.trainer,"load_frozen_artifacts",lambda p:(_ for _ in ()).throw(ValueError("incomplete")));monkeypatch.setattr(m,"evaluate_scenario",lambda *a:seen.append(1))
 with pytest.raises(ValueError,match="incomplete"):m.run_evaluation(tmp_path/"campaign",tmp_path/"raw",tmp_path/"out",["os1"],"x")
 assert not seen and not (tmp_path/"raw").exists()
def test_attack_does_not_calibrate(monkeypatch):
 m=mod();seen={};c={"node_thresholds":{"q50":1.,"q70":2.,"q80":3.},"alpha":.75};s=pd.DataFrame()
 monkeypatch.setattr(m.gate_lib,"build_event_scores",lambda x,t,alpha:seen.update(t=t,alpha=alpha) or pd.DataFrame({"x":[1]}));m.build_attack_events(s,c);assert seen=={"t":c["node_thresholds"],"alpha":.75}
def test_parser_scenarios():
 m=mod();assert m.build_parser().parse_args(["--campaign-root","x"]).scenarios==list(m.SCENARIOS)
 with pytest.raises(SystemExit):m.build_parser().parse_args(["--campaign-root","x","--scenarios","cleanStatic"])

def test_resume_report_tamper_rejected(tmp_path,monkeypatch):
    m=mod();k=m.gate_lib.FINAL_SCORE
    s=pd.DataFrame({"run_id":["oakbat-os2-method-a-9tap"]*4,"prn":["G01"]*4,"window_bin_s":[100,130,130.5,131],"window_start_s":[100,130,130.5,131],"window_mid_s":[100.5,130.5,131,131.5],"window_end_s":[101,131,131.5,132],"prn_node_rmse":[1,2,3,4]})
    e=pd.DataFrame({"window_start_s":[100,130,130.5,131],k:[1,6,6,6]})
    c={"node_thresholds":{"q50":1.,"q70":2.,"q80":3.},"alpha":.75,"event_q99_threshold":5.}
    monkeypatch.setattr(m,"build_attack_events",lambda scores,calibration:e.copy())
    monkeypatch.setattr(m,"validate_scores",lambda scores,scenario:None)
    m.atomic_csv(tmp_path/"attack_prn_scores.csv",s);m.atomic_csv(tmp_path/"attack_event_scores.csv",e)
    r=m.build_report("os2",s,e,5.,{"x":1},tmp_path);m.atomic_json(tmp_path/"report.json",r)
    assert m.load_valid_resume("os2",tmp_path,{"x":1},c)
    forged=m.build_report("os2",s,e,99.,{"x":1},tmp_path);m.atomic_json(tmp_path/"report.json",forged)
    assert m.load_valid_resume("os2",tmp_path,{"x":1},c) is None


def test_resume_accepts_normal_noninteger_csv_round_trip(tmp_path,monkeypatch):
    m=mod();k=m.gate_lib.FINAL_SCORE
    times=np.arange(100.0,131.5,0.5);n=len(times)
    s=pd.DataFrame({"run_id":["oakbat-os2-method-a-9tap"]*n,"prn":["G01"]*n,"window_bin_s":times,"window_start_s":times,"window_mid_s":times+0.5,"window_end_s":times+1.0,"prn_node_rmse":np.sin(times)*0.123456789+0.5}).assign(**quality_score_metadata(times))
    e=pd.DataFrame({"window_start_s":times,k:np.cos(times)*0.234567891+2.0})
    c={"node_thresholds":{"q50":1.,"q70":2.,"q80":3.},"alpha":.75,"event_q99_threshold":2.1}
    monkeypatch.setattr(m,"build_attack_events",lambda scores,calibration:e.copy())
    m.atomic_csv(tmp_path/"attack_prn_scores.csv",s);m.atomic_csv(tmp_path/"attack_event_scores.csv",e)
    persisted_s=pd.read_csv(tmp_path/"attack_prn_scores.csv");persisted_e=pd.read_csv(tmp_path/"attack_event_scores.csv")
    m.atomic_json(tmp_path/"report.json",m.build_report("os2",persisted_s,persisted_e,2.1,{"x":1},tmp_path))
    assert m.load_valid_resume("os2",tmp_path,{"x":1},c)


def test_failed_rerun_invalidates_old_complete_manifest_and_preflights(tmp_path,monkeypatch):
    m=mod();out=tmp_path/"out";out.mkdir();m.atomic_json(out/"manifest.json",{"complete":True,"stale":True})
    campaign=tmp_path/"campaign";campaign.mkdir()
    monkeypatch.setattr(m.trainer,"load_frozen_artifacts",lambda p:{"calibration":{"node_thresholds":{"q50":1.,"q70":2.,"q80":3.},"normal_only":True,"attack_inputs_read":False,"alpha":.75,"event_q99_threshold":4.}})
    monkeypatch.setattr(m,"identity",lambda p:{"path":str(Path(p)),"size_bytes":1,"sha256":"a"*64})
    calls=[]
    monkeypatch.setattr(m.pipeline,"preflight_output_space",lambda root,**kw:calls.append((Path(root),kw)) or {"free_bytes":100,"required_free_bytes":10})
    monkeypatch.setattr(m,"evaluate_scenario",lambda *a,**kw:(_ for _ in ()).throw(RuntimeError("receiver failed")))
    with pytest.raises(RuntimeError,match="receiver failed"):
        m.run_evaluation(campaign,tmp_path/"raw",out,["os1"],"exe",minimum_free_bytes=10)
    state=json.loads((out/"manifest.json").read_text())
    assert state["complete"] is False and state["status"]=="running"
    assert len(calls)==2 and all(call[1]["scenario_count"]==1 for call in calls)
