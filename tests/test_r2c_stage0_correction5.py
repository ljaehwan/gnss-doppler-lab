import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np,pandas as pd,pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.r2c_stage0_fix import replay_b0_events

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/file);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def event_digest(rows):return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()

def b0_wrapper(tmp_path,status="AVAILABLE_AUTHENTIC_NODE_TO_SCORE_REPLAY"):
    runner=module("c5_runner_b0","run_r2c_gnss_stage0_fix.py");checkpoint=tmp_path/"checkpoint";checkpoint.write_bytes(b"checkpoint")
    saved=tmp_path/"saved.csv";pd.DataFrame([{"run_id":"r","prn":1,"window_bin_s":1.,"window_start_s":.5,"window_mid_s":1.,"window_end_s":1.5,"availability_time_s":1.5,"prn_node_rmse":.2}]).to_csv(saved,index=False)
    rows=replay_b0_events(pd.read_csv(saved)).to_dict("records");scenarios={name:{"status":"UNAVAILABLE_AUTHENTIC_INTERFACE","event_rows":[]} for name in runner.SCENARIOS}
    scenarios["cleanStatic"]={"status":status,"saved_score_path":str(saved),"saved_score_sha256":digest(saved),"event_rows":rows,"event_rows_sha256":event_digest(rows)}
    forged={**scenarios["cleanStatic"],"status":"AVAILABLE_AUTHENTIC_NODE_TO_SCORE_REPLAY"}
    scenarios["DS1"]=forged.copy();scenarios["DS2"]=forged.copy()
    report=tmp_path/"b0.json";report.write_text(json.dumps({"schema":"gnss-doppler-lab.r2c-b0-validation.v2","attack_scores_computed":False,"paper_comparison_eligible":True,"checkpoint":{"path":str(checkpoint),"sha256":digest(checkpoint)},"scenarios":scenarios}))
    config={"b0":{"checkpoint_sha256":digest(checkpoint),"saved_score_sha256":{"cleanStatic":digest(saved)}}}
    return runner,report,config

def test_noncanonical_ds1_ds2_b0_is_impossible_to_forge(tmp_path):
    runner,report,config=b0_wrapper(tmp_path);doc=runner.load_b0_validation(report,config)
    for name in ("DS1","DS2"):
        assert doc["scenarios"][name]["status"]=="UNAVAILABLE_AUTHENTIC_INTERFACE"
        assert doc["scenarios"][name]["event_rows"]==[]

def test_self_asserted_node_authentication_is_downgraded(tmp_path):
    runner,report,config=b0_wrapper(tmp_path);doc=runner.load_b0_validation(report,config)
    assert doc["scenarios"]["cleanStatic"]["status"]=="AVAILABLE_SAVED_NATIVE_SCORE_REPLAY_WITH_NODE_LINEAGE_GAP"
    assert doc["paper_comparison_eligible"] is False

def test_offline_los_never_enters_shared_scoring_or_controls():
    runner=module("c5_runner_los","run_r2c_gnss_stage0_fix.py")
    item={"derived_time":{"status":"PASS"},"offline_geometry_coverage":{"status":"PASS"},"event_time_causal_ephemeris_availability":{"status":"OFFLINE_ORACLE_ONLY","decoded_history_authenticated":False,"causal_decode_history_verified_by":"UNIMPLEMENTED_STAGE0"},"los_by_bin":{"0":{"1":[1,0,0]}}}
    assert runner.offline_physics_los(item)==item["los_by_bin"]
    assert runner.causal_deployment_los(item)=={}
    assert not runner.eligible_full_control_candidate({}, {"FullScorer":object(),"Full":object()})

def test_forged_decode_history_cannot_enable_causal_pass(tmp_path):
    runner=module("c5_runner_causal","run_r2c_gnss_stage0_fix.py");history=tmp_path/"history";history.write_text("forged")
    forged={"status":"PASS","decoded_history_authenticated":True,"causal_decode_history_verified_by":"UNIMPLEMENTED_STAGE0","decode_history":{"path":str(history),"sha256":digest(history)},"event_time_coverage":{"all_events_covered":True}}
    with pytest.raises(ValueError,match="cannot authenticate"):runner.validate_causal_capability(forged)

def test_manifest_iq_mismatch_is_lineage_gap(tmp_path):
    producer=module("c5_geometry","reconstruct_r2c_time_geometry.py");exports=tmp_path/"exports";receiver=tmp_path/"receiver";exports.mkdir();receiver.mkdir()
    selected=exports/"cleanStatic.npz";np.savez(selected,time_s=np.array([0.,.5]))
    receiver_manifest=receiver/"manifest.json";receiver_manifest.write_text(json.dumps({"authenticated_inputs":{"iq_after_receiver":{"sha256":"b"*64}}}))
    manifest=selected.with_suffix(".manifest.json");manifest.write_text(json.dumps({"recording_id":"cleanStatic","source_iq_sha256":"a"*64,"output":{"path":str(selected),"sha256":digest(selected),"row_count":2},"receiver_manifest":{"path":str(receiver_manifest),"sha256":digest(receiver_manifest)}}))
    _,lineage=producer.selected_bins(selected,tmp_path,"cleanStatic")
    assert lineage["lineage_status"]=="LINEAGE_GAP"
    assert lineage["source_iq_sha256"]=="a"*64 and lineage["receiver_source_iq_sha256"]=="b"*64
