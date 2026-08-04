import hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd,pytest

ROOT=Path(__file__).parents[1]
from gnss_doppler_lab.r2c_stage0_fix import B0_FEATURES,ScoreResult,replay_b0_events,run_full_controls,score_b0_nodes,validate_b0_nodes

def load_runner():
    spec=importlib.util.spec_from_file_location("correction2_runner",ROOT/"scripts/run_r2c_gnss_stage0_fix.py")
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def node_frame(count=13):
    rows=[]
    for i in range(count):
        start=i*.5;rows.append({"run_id":"r","prn":3,"channel":1,"segment_index":2,"window_bin_s":start+.5,
          "window_start_s":start,"window_end_s":start+1.,"window_mid_s":start+.5,"epoch_count":4,**{x:1. for x in B0_FEATURES}})
    return pd.DataFrame(rows)

def test_b0_adversarial_node_contract_and_event_timing():
    frame=node_frame();bad=frame.copy();bad.loc[0,"epoch_count"]=1
    with pytest.raises(ValueError,match="supported"):validate_b0_nodes(bad)
    bad=frame.copy();bad.loc[0,"window_end_s"]+=.1
    with pytest.raises(ValueError,match="timing"):validate_b0_nodes(bad)
    bad=pd.concat([frame.iloc[:6],frame.iloc[7:]],ignore_index=True)
    with pytest.raises(ValueError,match="gap"):validate_b0_nodes(bad)
    scores=pd.DataFrame([{"run_id":"r","prn":p,"window_bin_s":.5,"window_start_s":0.,"window_mid_s":.5,"window_end_s":1.,"availability_time_s":1.,"prn_node_rmse":.2} for p in range(1,6)])
    event=replay_b0_events(scores);assert event.iloc[0].btail_max_507080_ewma075==pytest.approx(.25*event.iloc[0].btail_max_507080)
    scores.loc[0,"window_mid_s"]+=.1
    with pytest.raises(ValueError,match="timing"):replay_b0_events(scores)

def test_b0_checkpoint_hash_and_generated_key_set_fail_closed(tmp_path):
    checkpoint=tmp_path/"checkpoint.pt";checkpoint.write_bytes(b"wrong")
    with pytest.raises(ValueError,match="checkpoint hash"):
        score_b0_nodes(node_frame(),{"config":{"seq_len":12},"node_feature_columns":list(B0_FEATURES),"standardizer":{"node_mean":[0]*9,"node_std":[1]*9}},checkpoint_path=checkpoint,expected_checkpoint_sha256="0"*64)
    generated={("r",3,.5),("r",3,1.)};saved={("r",3,.5)}
    assert generated!=saved and generated-saved

def test_partial_geometry_and_frozen_control_fail_closed(monkeypatch):
    runner=load_runner();rows,los=runner.synthetic_inputs(4);config=json.loads((ROOT/"configs/r2c_gnss_stage0_fix.json").read_text())
    data={"y":np.asarray([r[2] for r in rows]),"time":np.asarray([r[0] for r in rows]),"prn":np.asarray([r[1] for r in rows]),"cn0":np.asarray([r[3] for r in rows])}
    provider=runner.TemplateProvider.analytic();taps=np.asarray(config["tap_offsets_chips"]);grid=np.asarray(config["delay_grid_chips"]);models=runner.fit_frozen_models(data,config,provider,taps,grid,require_gpu=False)
    observations={p:data["y"][data["prn"]==p][:2] for p in los};observations[6]=observations[1].copy()
    conditions={p:(np.column_stack((np.full(len(y),40.),np.zeros(len(y)))),np.column_stack((np.full(len(y),40.),np.zeros(len(y)),np.ones(len(y))))) for p,y in observations.items()}
    good_scores,good_fits,_=runner.score_bin({p:observations[p] for p in los},los,provider,taps,grid,models,{**config,"optimizer_starts_m":[config["optimizer_starts_m"][0]]},{p:conditions[p] for p in los})
    if good_scores["Full"] is not None:
        controls=run_full_controls(good_fits["FullScorer"],{p:observations[p] for p in los},los,1e100,provider,taps)
        assert controls["baseline_score"]==good_scores["Full"]
    # Explicit intersection is rank deficient: geometry-free paths still use all six PRNs.
    bad_los={p:np.array([1.,0.,0.]) for p in los}
    scores,fits,individual=runner.score_bin(observations,bad_los,provider,taps,grid,models,{**config,"optimizer_starts_m":[config["optimizer_starts_m"][0]]},conditions)
    assert len(individual)==6 and all(scores[x] is not None for x in ("A1","A2","A4","Power-only"))
    assert scores["A3"] is scores["Full"] is scores["Neural-with-energy"] is None
    assert fits["statuses"]["Full"].startswith("UNAVAILABLE_")
    invalid=lambda obs,pair,**kw:ScoreResult("UNAVAILABLE_INVALID_SHARED_FIT",None,"rank")
    controls=run_full_controls(invalid,observations,bad_los,0.,provider,taps)
    assert controls["baseline_status"].startswith("UNAVAILABLE_") and all(r["pre_alarm"] is None and r["post_alarm"] is None for r in controls["rows"])

def test_production_synthetic_guards_and_geometry_wrapper(tmp_path):
    runner=load_runner();production=ROOT/"artifacts/r2c_gnss_stage0_fix"
    with pytest.raises(ValueError):runner.validate_destination(production,test_mode=True)
    with pytest.raises(ValueError,match="exact scenario roster"):runner.parse_named_specs(["cleanStatic=x"],roster=runner.SCENARIOS)
    wrapper=tmp_path/"geometry.json";wrapper.write_text(json.dumps({"schema":"x","scenarios":{name:{"scenario":name} for name in runner.SCENARIOS}}))
    loaded,path=runner.load_geometry_specs([str(wrapper)]);assert set(loaded)==set(runner.SCENARIOS) and path==wrapper
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    command=[sys.executable,str(ROOT/"scripts/run_r2c_gnss_stage0_fix.py"),"--source-commit",head,"--synthetic","--output",str(production)]
    result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True);assert result.returncode and not production.exists()
