from datetime import date, datetime, timezone
import itertools
import csv, hashlib, importlib.util, json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnss_doppler_lab.r2c_stage0_fix import (
    B0_FEATURES, ComplexWhitener, SmallNuisanceConditioner, TemplateProvider, build_b0_node_windows,
    calibration_thresholds, derive_two_layer_decision, detector_scores,
    detection_metrics, gps_week_tow, joint_profile_glrt, normalized_pauc, paired_block_bootstrap, reconstruct_time_geometry,
    replay_b0_events, resolve_nmea_rollover, run_full_controls, validate_b0_nodes,
)

TAPS=np.arange(-.5,.5001,.125); GRID=np.arange(-.5,.5001,.125)


def test_b0_prompt_normalize_then_mean_inclusive_and_no_first_row_substitution():
    rows=[]
    for t,prompt,e4 in [(0.,2.,2.),(.25,2.,4.),(.5,4.,12.),(1.,8.,8.),(1.5,2.,4.)]:
        row={"time_s":t,"prn":3}
        for tap in ("E4","E3","E2","E","P","L","L2","L3","L4"): row[f"tap_{tap}"]=prompt
        row["tap_E4"]=e4; rows.append(row)
    nodes=build_b0_node_windows(pd.DataFrame(rows),run_id="r")
    assert list(nodes.columns[-9:])==list(B0_FEATURES)
    assert nodes.iloc[0].epoch_count==4 # inclusive [0,1]
    assert nodes.iloc[0].tap_E4_rel_prompt_mean==pytest.approx((1+2+3+1)/4)
    assert nodes.iloc[0].tap_E4_rel_prompt_mean != 1 # cannot be raw first row
    assert nodes.iloc[0].window_end_s==nodes.iloc[0].window_start_s+1


def test_b0_sequence_rejects_gap_duplicate_and_schema_order():
    base={"run_id":"r","prn":1,"channel":0,"segment_index":0,"window_start_s":0.,"window_end_s":1.,"window_mid_s":.5,"epoch_count":4,
          **{f:1. for f in B0_FEATURES}}
    required=["run_id","prn","channel","segment_index","window_bin_s","window_start_s","window_end_s","window_mid_s","epoch_count",*B0_FEATURES]
    frame=pd.DataFrame([{**base,"window_bin_s":.5},{**base,"window_bin_s":1.5,"window_start_s":1.,"window_end_s":2.,"window_mid_s":1.5}])[required]
    with pytest.raises(ValueError,match="gap"): validate_b0_nodes(frame)
    duplicate=pd.concat([frame.iloc[:1],frame.iloc[:1]])
    with pytest.raises(ValueError,match="duplicate"): validate_b0_nodes(duplicate)
    with pytest.raises(ValueError,match="feature order"): validate_b0_nodes(frame[list(reversed(frame.columns))])


def test_b0_replay_known_binomial_and_recording_reset():
    rows=[]
    for run in ("a","b"):
        for prn,value in enumerate([.2,.2,.01],1): rows.append({"run_id":run,"prn":prn,"window_bin_s":2.5,
            "window_start_s":2.,"window_mid_s":2.5,"prn_node_rmse":value})
    out=replay_b0_events(pd.DataFrame(rows))
    assert out.availability_time_s.tolist()==[3.,3.]
    assert out.btail_max_507080_ewma075.iloc[0]==pytest.approx(out.btail_max_507080_ewma075.iloc[1])


def test_template_fallback_full_physical_support_complex_empirical():
    analytic=TemplateProvider.analytic()
    assert analytic.evaluate([-1.2,-1,-.5,0,.5,1,1.2]).tolist()==[0,0,.5,1,.5,0,0]
    assert analytic.analytic_approximation and not analytic.paper_comparison_ready
    x=np.arange(-1,1.001,.125); y=(1-np.abs(x))*np.exp(1j*(.2+.3*x))
    empirical=TemplateProvider.empirical(x,y,{"source_sha256":"a"*64})
    assert np.iscomplexobj(empirical.evaluate(TAPS-.4))
    with pytest.raises(ValueError,match="out-of-support"): empirical.evaluate([-1.01])
    with pytest.raises(ValueError,match="requires authenticated"): TemplateProvider.empirical(TAPS,np.ones(9,dtype=complex),{})


def test_proper_complex_whitening_and_train_hash():
    rng=np.random.default_rng(4); z=rng.normal(size=(1000,9))+1j*rng.normal(size=(1000,9)); z[:,1]+=.7j*z[:,0]
    w=ComplexWhitener(shrinkage=.05).fit(z,["normal_train"]*len(z),[f"r{i}" for i in range(len(z))])
    transformed=w.transform(z-z.mean(0)); cov=transformed.T@transformed.conj()/(len(z)-1)
    assert np.max(np.abs(cov-np.eye(9))) < .2
    assert w.diagnostics["pseudo_covariance_frobenius"]>=0 and len(w.diagnostics["train_rows_sha256"])==64
    assert w.serialize()["inverse_sqrt"]["imag"]
    with pytest.raises(ValueError,match="normal_train"): ComplexWhitener().fit(z,["post"]*len(z))


def geometry():
    los=np.array([[1,0,0],[0,1,0],[0,0,1],[-.6,-.5,-.6245],[.5,-.7,.5099]])
    return {i+1:v/np.linalg.norm(v) for i,v in enumerate(los)}


def synth(beta=(20.,-15.,7.,80.), epochs=2, phase=.4):
    provider=TemplateProvider.analytic(); los=geometry(); base=provider.evaluate(TAPS); out={}
    for p,u in los.items():
        d=(-u@np.asarray(beta[:3])+beta[3])/299792458*1023000
        second=provider.evaluate(TAPS-d)
        out[p]=np.array([(1+.1*j)*np.exp(1j*(phase+.2*j))*base+.4*np.exp(-1j*(.7+j*.1))*second for j in range(epochs)])
    return provider,los,out


def test_joint_all_epoch_bic_identity_beta_recovery_and_permutations():
    provider,los,obs=synth()
    h0=joint_profile_glrt(obs,los,provider,TAPS,GRID,hypothesis="H0")
    ind=joint_profile_glrt(obs,los,provider,TAPS,GRID,hypothesis="H1-independent")
    shared=joint_profile_glrt(obs,los,provider,TAPS,GRID,hypothesis="H1-shared")
    assert h0.epoch_count==10 and h0.n==2*9*10 and h0.k==25
    assert ind.k==50 and shared.k==49
    assert shared.bic==pytest.approx(-2*shared.log_likelihood+shared.k*np.log(shared.n))
    assert shared.score==pytest.approx(2*(shared.log_likelihood-h0.log_likelihood)-(shared.k-h0.k)*np.log(shared.n))
    assert shared.beta_m==pytest.approx((20.,-15.,7.,80.),abs=2.) and shared.rss < h0.rss
    order=[5,2,4,1,3]; perm={p:obs[p] for p in order}; plos={p:los[p] for p in order}
    assert joint_profile_glrt(perm,plos,provider,TAPS,GRID,hypothesis="H1-shared").score==pytest.approx(shared.score,rel=1e-6)
    for p in obs: obs[p]=obs[p][::-1]
    assert joint_profile_glrt(obs,los,provider,TAPS,GRID,hypothesis="H1-shared").score==pytest.approx(shared.score,rel=1e-6)


def test_joint_rank_dof_condition_and_nonconvergence_fail_closed():
    provider,los,obs=synth(); four={p:obs[p] for p in list(obs)[:4]}; flos={p:los[p] for p in four}
    fit=joint_profile_glrt(four,flos,provider,TAPS,GRID,hypothesis="H1-shared")
    assert not fit.valid and fit.reason=="insufficient_prns"
    fit=joint_profile_glrt(obs,los,provider,TAPS,GRID,hypothesis="H1-shared",beta_bounds_m=((1e9,1e9+1),)*4,optimizer_starts=[(1e9,)*4])
    assert not fit.valid and fit.reason=="boundary_or_nonconvergence"


def test_detectors_are_distinct_and_geometry_invalid_is_unavailable():
    provider,los,obs=synth(); h0=joint_profile_glrt(obs,los,provider,TAPS,GRID,hypothesis="H0")
    ind=joint_profile_glrt(obs,los,provider,TAPS,GRID,hypothesis="H1-independent")
    out=detector_scores({1:1,2:2,3:7},h0,ind,None,3.2,4.1)
    assert out["A1"]!=out["A2"] and out["A3"]==0 and out["A4"]==ind.score and out["Full"] is None
    assert len(set(out))==7


def test_time_rollover_leap_causality_and_coverage():
    stamps=resolve_nmea_rollover([235959.5, .5],date(2012,9,1)); assert stamps[1].date()>stamps[0].date()
    week,tow=gps_week_tow(datetime(2012,9,1,tzinfo=timezone.utc),16); assert week==1703 # conversion itself, no filename inference
    with pytest.raises(ValueError,match="time reversal"): resolve_nmea_rollover([120001,120000],date(2012,1,1))
    los=geometry(); report=reconstruct_time_geometry(scenario="x",relative_times_s=[0,1],
      authenticated_utc_start=datetime(2012,9,1,tzinfo=timezone.utc),leap_seconds=16,
      observable_rx_time=[0,1],ephemeris_toe={p:tow-10 for p in los},pvt_times=[0,1],los_by_event={0.:los,1.:los})
    assert report["authenticated_absolute_time_binding"]["status"]=="PASS"
    assert report["pvt_coverage"]["valid_events"]==2
    future=reconstruct_time_geometry(scenario="x",relative_times_s=[0],authenticated_utc_start=datetime(2012,9,1,tzinfo=timezone.utc),
      leap_seconds=16,observable_rx_time=[0],ephemeris_toe={p:tow+1 for p in los},pvt_times=[1],los_by_event={0.:los})
    assert future["event_time_causal_ephemeris"]["status"]=="UNAVAILABLE" and future["pvt_coverage"]["valid_events"]==0


def passing_gates():
    names=("complex_provenance","time_los_alignment","geometry_coverage","clean_dynamic_fpr","gain_invariance",
      "phase_invariance","noise_gain_alarms","relation_destruction","full_improvement","full_a2_two_scenarios",
      "shortcut_controls","complex_second_source","geometry_removal","b0_authentic_common_support","full_b0_comparison",
      "empirical_wide_template","paper_gates")
    return {n:{"status":"PASS"} for n in names}


def test_two_layer_truth_table_b0_orthogonal_and_disqualifier_precedence():
    gates=passing_gates(); assert derive_two_layer_decision(gates)["verdict"]=="PHYSICS_SUPPORTED"
    gates["b0_authentic_common_support"]={"status":"UNAVAILABLE"}; result=derive_two_layer_decision(gates)
    assert result["core_physics_verdict"]=="R2C_CORE_SUPPORTED" and not result["paper_comparison_ready"]
    gates["geometry_coverage"]={"status":"UNAVAILABLE"}; assert derive_two_layer_decision(gates)["core_physics_verdict"]=="R2C_CORE_INCONCLUSIVE"
    gates["clean_dynamic_fpr"]={"status":"FAIL"}; assert derive_two_layer_decision(gates)["core_physics_verdict"]=="R2C_CORE_NOT_SUPPORTED"


def test_calibration_pauc_and_actual_full_controls():
    assert set(calibration_thresholds(range(200),["normal_calibration"]*200))=={"q99","q99.5","target_fpr_1pct"}
    assert normalized_pauc([0,0,1,1],[0,1,2,3])==pytest.approx(1.)
    y={p:np.ones((2,9),complex) for p in geometry()}; los=geometry()
    controls=run_full_controls(lambda value,pairing:float(sum(np.mean(np.abs(v/v[:,4,None])) for v in value.values())+sum((pairing or {}).keys())),y,los,20.)
    assert controls["computed_rows"]>=13 and all("pre_score" in r and "post_score" in r for r in controls["rows"])


def test_pre_campaign_report_refuses_both_artifact_trees(tmp_path):
    spec=importlib.util.spec_from_file_location("pre",Path(__file__).parents[1]/"scripts/validate_r2c_stage0_pre_campaign.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for target in (module.PRESERVED/"x.json",module.CAMPAIGN/"x.json"):
        with pytest.raises(ValueError,match="artifact trees"): module.safe_report(target)
    assert module.safe_report(tmp_path/"report.json")== (tmp_path/"report.json").resolve()


def test_verifier_rejects_data_and_regenerated_hash_tamper(tmp_path):
    spec=importlib.util.spec_from_file_location("verify",Path(__file__).parents[1]/"scripts/verify_r2c_gnss_stage0_fix.py")
    verifier=importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    root=tmp_path/"r2c_gnss_stage0_fix"; root.mkdir(); (root/"plots").mkdir()
    for name in verifier.TOP_LEVEL_FILES:
        path=root/name
        if name.endswith(".csv"): path.write_text("detector,ll0,ll1,n,k0,k1,score\nA1,0,1,18,2,4,-3.780743516\n")
        else: path.write_text("{}\n")
    (root/"per_epoch_scores.csv").write_text("detector,ll0,ll1,n,k0,k1,score\nFull,0,1,18,2,4,999\n")
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file() and p.name!="hashes.json"}
    (root/"hashes.json").write_text(json.dumps({"files":hashes}))
    assert any("BIC identity" in e for e in verifier.verify(root))


def test_neural_conditioner_schema_training_and_energy_separation():
    rng=np.random.default_rng(9); x=rng.normal(size=(20,2)); z=rng.normal(size=(20,9))+1j*rng.normal(size=(20,9))
    model=SmallNuisanceConditioner(["cn0","h0_residual_quality"],hidden=3).fit(x,z,["normal_train"]*20,epochs=2)
    assert model.summary["architecture"]==[2,3,18] and len(model.summary["weights_sha256"])==64
    with pytest.raises(ValueError,match="schema"): SmallNuisanceConditioner(["recording_id"])
    with pytest.raises(ValueError,match="schema"): SmallNuisanceConditioner(["explicit_energy"])
    energy=SmallNuisanceConditioner(["cn0","explicit_energy"],with_energy=True)
    assert energy.with_energy and not model.with_energy


def test_causal_metrics_and_exact_paired_block_bootstrap():
    metric=detection_metrics([1,1.5,2,5,5.5,6],[1]*6,[1]*6,["a"]*6)
    assert metric["sustained_detection_rate"]==pytest.approx(2/6) # reset across the gap
    times=np.arange(80)*.5; labels=np.r_[np.zeros(40,bool),np.ones(40,bool)]
    result=paired_block_bootstrap(times,["r"]*80,labels,labels.astype(float),np.zeros(80),repetitions=2000)
    assert result["repetitions"]==2000 and result["block_s"]==10 and result["estimate"]>0
    with pytest.raises(ValueError,match="exactly 2000"): paired_block_bootstrap(times,["r"]*80,labels,labels,np.zeros(80),repetitions=10)


def test_benchmark_uses_offline_oracle_los_when_causal_history_is_unavailable():
    spec=importlib.util.spec_from_file_location("benchmark",Path(__file__).parents[1]/"scripts/benchmark_r2c_stage0_profile.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    geometry={"derived_time":{"status":"PASS"},"event_time_causal_ephemeris_availability":{"status":"UNAVAILABLE"},
              "los_by_bin":{"0":{"1":[1.,0.,0.]}}}
    assert module.stage0_benchmark_los(geometry)==geometry["los_by_bin"]


def test_conditioner_cpu_inference_copy_preserves_predictions():
    rng=np.random.default_rng(321); x=rng.normal(size=(20,2)); z=rng.normal(size=(20,9))+1j*rng.normal(size=(20,9))
    model=SmallNuisanceConditioner(["cn0","h0_residual_quality"],hidden=3).fit(x,z,["normal_train"]*20,epochs=2)
    copied=model.cpu_inference_copy()
    assert next(copied.model.parameters()).device.type=="cpu"
    assert copied.predict(x[:4])==pytest.approx(model.predict(x[:4]))
    assert copied.summary["device"]==model.summary["device"]


def test_ordered_fork_map_preserves_input_order_and_worker_results():
    spec=importlib.util.spec_from_file_location("runner",Path(__file__).parents[1]/"scripts/run_r2c_gnss_stage0_fix.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    expected=[9,1,4]
    assert list(module.ordered_fork_map([3,1,2],lambda x:x*x,2))==expected
    assert list(module.ordered_score_map([3,1,2],lambda x:x*x,2,backend="thread"))==expected
    assert list(module.ordered_score_map([3,1,2],lambda x:x*x,2,backend="fork"))==expected
    class Fit: valid=True
    fits={"Full":Fit(),"FullScorer":object(),"statuses":{}}
    assert module.detach_full_scorer_for_transport({1:np.ones(3)},fits)
    assert "FullScorer" not in fits and fits["Full"].valid


def test_benchmark_bin_selector_is_deterministic_and_requires_offline_los_support():
    spec=importlib.util.spec_from_file_location("benchmark_select",Path(__file__).parents[1]/"scripts/benchmark_r2c_stage0_profile.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    dataset={"bin":np.repeat(np.arange(5),5),"prn":np.tile(np.arange(1,6),5)}
    los={str(b):{str(p):[1.,0.,0.] for p in range(1,6)} for b in range(5)}
    los["2"]={"1":[1.,0.,0.]}
    assert module.stage0_benchmark_bin_ids(dataset,los,3)==[0,1,4]


def test_parallel_projection_uses_measured_effective_parallelism():
    spec=importlib.util.spec_from_file_location("benchmark_parallel",Path(__file__).parents[1]/"scripts/benchmark_r2c_stage0_profile.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.parallel_wall_projection(100.,50.,25.)==pytest.approx(50.)


def test_campaign_runtime_gate_allows_a_bounded_multi_hour_offline_oracle_run():
    spec=importlib.util.spec_from_file_location("benchmark_runtime",Path(__file__).parents[1]/"scripts/benchmark_r2c_stage0_profile.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.campaign_runtime_gate(21218.)
    assert not module.campaign_runtime_gate(28801.)


def test_supervisor_runtime_gate_matches_offline_campaign_contract():
    spec=importlib.util.spec_from_file_location("supervisor",Path(__file__).parents[1]/"scripts/supervise_r2c_stage0.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.campaign_runtime_gate(21218.) and not module.campaign_runtime_gate(28801.)


def test_ordered_fork_map_returns_torch_tensor_without_resource_sharer_fd():
    import torch
    spec=importlib.util.spec_from_file_location("runner_tensor",Path(__file__).parents[1]/"scripts/run_r2c_gnss_stage0_fix.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result=list(module.ordered_fork_map([3],lambda x:torch.tensor([x]),1))
    assert result[0].tolist()==[3]
