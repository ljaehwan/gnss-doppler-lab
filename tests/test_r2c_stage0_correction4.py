import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np,pandas as pd,pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.r2c_stage0_fix import JointFit,detector_scores,replay_b0_events

def load_script(name,filename=None):
    spec=importlib.util.spec_from_file_location(name,ROOT/f"scripts/{filename or name}.py");module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def fit(valid,score=7.):
    return JointFit("H1-independent",-1.,1.,18,5,2.,score,(.2,),None,valid,not valid,valid,None if valid else "boundary_or_nonconvergence",1,1,-2.,2)

def test_detector_scores_invalid_a4_fails_closed():
    values=detector_scores({1:2.},None,fit(False),None,None,3.)
    assert values["A4"] is None

def score_csv(path):
    rows=[]
    for prn,value in ((1,.05),(2,.2)):
        rows.append({"run_id":"r","prn":prn,"window_bin_s":1.,"window_start_s":.5+prn/100,
          "window_mid_s":1.+prn/100,"window_end_s":1.5+prn/100,"availability_time_s":1.5+prn/100,"prn_node_rmse":value})
    pd.DataFrame(rows).to_csv(path,index=False)

def test_b0_requires_canonical_hash_and_rejects_forged_events(tmp_path):
    runner=load_script("run_r2c_gnss_stage0_fix");saved=tmp_path/"saved.csv";score_csv(saved)
    checkpoint=tmp_path/"checkpoint";checkpoint.write_bytes(b"checkpoint")
    events=replay_b0_events(pd.read_csv(saved)).to_dict("records")
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    event_hash=lambda rows:hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
    scenarios={name:{"status":"UNAVAILABLE_AUTHENTIC_INTERFACE","event_rows":[]} for name in runner.SCENARIOS}
    scenarios["cleanStatic"]={"status":"AVAILABLE_SAVED_NATIVE_SCORE_REPLAY_WITH_NODE_LINEAGE_GAP","saved_score_path":str(saved),"saved_score_sha256":digest(saved),"event_rows":events,"event_rows_sha256":event_hash(events)}
    wrapper={"schema":"gnss-doppler-lab.r2c-b0-validation.v2","attack_scores_computed":False,"checkpoint":{"path":str(checkpoint),"sha256":digest(checkpoint)},"scenarios":scenarios}
    report=tmp_path/"b0.json";report.write_text(json.dumps(wrapper))
    config={"b0":{"checkpoint_sha256":digest(checkpoint),"saved_score_sha256":{"cleanStatic":"0"*64}}}
    assert not runner.load_b0_validation(report,config)["scenarios"]["cleanStatic"]["event_rows"]
    config["b0"]["saved_score_sha256"]["cleanStatic"]=digest(saved)
    wrapper["scenarios"]["cleanStatic"]["event_rows"][0]["availability_time_s"]+=1
    wrapper["scenarios"]["cleanStatic"]["event_rows_sha256"]=event_hash(wrapper["scenarios"]["cleanStatic"]["event_rows"]);report.write_text(json.dumps(wrapper))
    assert not runner.load_b0_validation(report,config)["scenarios"]["cleanStatic"]["event_rows"]

def test_missing_iq_hashes_are_never_bound(tmp_path):
    producer=load_script("reconstruct_r2c_time_geometry")
    assert not producer.valid_sha(None)
    assert not (producer.valid_sha(None) and producer.valid_sha(None) and None==None)

def test_verification_recompute_destination_is_verifier_only(tmp_path,monkeypatch):
    runner=load_script("run_r2c_gnss_stage0_fix_dest","run_r2c_gnss_stage0_fix")
    monkeypatch.delenv("R2C_VERIFIER_RECOMPUTE",raising=False)
    with pytest.raises(ValueError,match="verifier-only"):runner.validate_destination(tmp_path/"out",verification_recompute=True)
    monkeypatch.setenv("R2C_VERIFIER_RECOMPUTE","1")
    assert runner.validate_destination(tmp_path/"out",verification_recompute=True)==(tmp_path/"out").resolve()

def test_pre_campaign_controls_have_no_hardcoded_threshold():
    source=(ROOT/"scripts/validate_r2c_stage0_pre_campaign.py").read_text()
    assert "run_full_controls(fits[\"FullScorer\"],obs,los,10." not in source
    assert "controls[\"threshold\"]!=threshold" in source
