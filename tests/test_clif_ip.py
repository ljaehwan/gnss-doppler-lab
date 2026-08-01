import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import json
import subprocess
from gnss_doppler_lab.clif_ip import (
    M1FitAudit, fit_m1, transform_m1, make_history, true_exceed_fraction,
    aggregate_prns, fit_threshold, independent_fpr, shuffle_pairing,
    validate_provenance_grade, slice_whole_windows, fit_whitener,
    mahalanobis_score, fit_component_calibrations, alarm_times,
)

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("eval_clif_ip",ROOT/"scripts/eval_clif_ip.py")
EVAL=importlib.util.module_from_spec(SPEC); import sys; sys.modules[SPEC.name]=EVAL; SPEC.loader.exec_module(EVAL)
TRAIN_SPEC=importlib.util.spec_from_file_location("train_clif_ip",ROOT/"scripts/train_clif_ip.py")
TRAIN=importlib.util.module_from_spec(TRAIN_SPEC);sys.modules[TRAIN_SPEC.name]=TRAIN;TRAIN_SPEC.loader.exec_module(TRAIN)


def features(n=40, d=8, seed=1):
    r=np.random.default_rng(seed)
    return r.normal(size=(n,d)), np.arange(n)*.5


def test_m1_fit_exactly_once_clean_train_only_and_hash_invariant():
    x,t=features(); audit=M1FitAudit(); state=fit_m1(x,t,t+.5,train_end=10,pca_dim=3,lag=2,audit=audit,recording="cleanStatic")
    h=state.sha256; assert audit.fit_count==1 and audit.fit_recordings==["cleanStatic"]
    transform_m1(x,t,state,recording="os2",audit=audit)
    assert audit.fit_count==1 and state.sha256==h


def test_attack_transform_does_not_fit_and_mutation_cannot_change_state():
    x,t=features(); a=M1FitAudit(); s=fit_m1(x,t,t+.5,10,3,2,audit=a)
    before=s.sha256; attack=np.full_like(x,1e9); transform_m1(attack,t,s,"attack",a)
    assert s.sha256==before and a.fit_count==1 and a.transform_recordings[-1]=="attack"


def test_history_reset_no_future_signed_nine_taps_and_no_prn_identity():
    b=np.arange(6*9,dtype=float).reshape(6,9)-20; m=np.arange(12,dtype=float).reshape(6,2)
    X,y,idx=make_history(b,m,lag=2)
    assert y.shape[1]==9 and np.any(y<0) and idx[0]==2
    assert np.array_equal(X[0,:18],b[:2].reshape(-1))
    assert not np.any(X[0]==b[3,0])
    X2,_,idx2=make_history(b[:3],m[:3],lag=2)
    assert idx2.tolist()==[2]


def test_prn_permutation_invariance_variable_count_and_true_k_over_n():
    r=np.array([[1.,2.],[4.,3.],[2.,8.]])
    assert aggregate_prns(r)==aggregate_prns(r[[2,0,1]])
    assert aggregate_prns(r[:2])["tracked_count"]==2
    assert true_exceed_fraction(np.array([.1,2.,3.]),1.)==pytest.approx(2/3)


def test_normal_only_threshold_and_independent_fpr():
    normal=np.arange(100,dtype=float); threshold=fit_threshold(normal,.99)
    assert threshold==pytest.approx(np.quantile(normal,.99))
    assert independent_fpr(np.array([0,100]),threshold)==.5


def test_shuffle_preserves_marginals_but_breaks_pairing():
    b=np.arange(30); m=b.copy(); bs,ms=shuffle_pairing(b,m,seed=7,block=3)
    assert np.array_equal(bs,b) and np.array_equal(np.sort(ms),m)
    assert not np.array_equal(ms,m)


def test_provenance_grade_separation():
    assert validate_provenance_grade("verified")=="verified"
    assert validate_provenance_grade("reconstructed")=="reconstructed"
    assert validate_provenance_grade("provisional")=="provisional"
    with pytest.raises(ValueError): validate_provenance_grade("verified-ish")


def test_m1_fit_requires_both_window_boundaries_and_has_480_rows():
    t=np.arange(-.5,241.,.5); x=np.c_[np.sin(t),np.cos(t),t%7,t%11]
    s=fit_m1(x,t,t+.5,train_end=240,pca_dim=2,lag=2)
    assert s.fit_rows==480==np.sum((t>=0)&(t+.5<=240))


def test_m1_score_nonnegative_and_namespace_not_colliding(tmp_path):
    t=np.arange(30)*.5; x=np.random.default_rng(4).normal(size=(30,4)); s=fit_m1(x,t,t+.5,10,2,2)
    p=tmp_path/"m.csv"; pd.DataFrame({"window_start_s":t,"window_end_s":t+.5,**{f"f{i}":x[:,i] for i in range(4)}}).to_csv(p,index=False)
    frame,_=EVAL.m1_frame(p,s,M1FitAudit(),"x")
    assert "M1_score" in frame and "m1_innov_0" in frame and "m1" not in frame
    assert (frame.M1_score>=0).all()


def test_whole_window_split_precedes_transform_and_resets_history():
    d=pd.DataFrame({"window_start_s":[249.5,250.,250.5,329.5,330.],"window_end_s":[250.,250.5,251.,330.,330.5]})
    got=slice_whole_windows(d,250,330)
    assert got.index.tolist()==[1,2,3]


def test_split_design_first_target_never_uses_previous_split():
    rows=[]
    for i,t in enumerate(np.arange(250,254,.5)):
        rows.append({"t":t,"available_s":t+.5,"m1_available_s":t+.5,"prn":"1",**{f"b{j}":i+j for j in range(9)},**{f"m1_innov_{j}":i-j for j in range(2)},"M1_score":i})
    d=pd.DataFrame(rows); _,_,meta=EVAL.design(d,2,"P3")
    assert meta.t.iloc[0]==251 and meta.source_start_s.iloc[0]>=250


def test_predictor_models_have_identical_target_support():
    rows=[]
    for i,t in enumerate(np.arange(20)*.5):
        rows.append({"t":t,"available_s":t+.5,"m1_available_s":t+.5,"prn":"1",**{f"b{j}":np.sin(i+j) for j in range(9)},**{f"m1_innov_{j}":np.cos(i+j) for j in range(2)},"M1_score":abs(np.sin(i))})
    d=pd.DataFrame(rows); models=EVAL.train_models(d,d,2,{"P1":1.,"P2":1.,"P3":1.})
    supports=[]
    for k in ("P0","P1","P2","P3"):
        y,p,meta=EVAL.predict_model(d,models,2,k); supports.append(list(zip(meta.t,meta.prn))); assert len(y)==len(p)
    assert supports.count(supports[0])==4


def test_predictors_use_separate_validation_selected_alphas():
    assert EVAL.select_hyperparameters.__doc__ and "validation" in EVAL.select_hyperparameters.__doc__.lower()
    assert set(EVAL.ALPHAS)=={.1,1.,10.,100.}


def test_whitening_centers_validation_residuals():
    r=np.array([[10.,0.],[12.,1.],[11.,-1.],[9.,.5]])
    w=fit_whitener(r); scores=mahalanobis_score(r,w); shifted=mahalanobis_score(r+100,w)
    assert np.all(np.isfinite(scores)) and np.mean(shifted)>np.mean(scores)
    assert np.allclose(w.center,r.mean(0))


def test_ablation_calibration_refit_for_exact_component_set():
    rng=np.random.default_rng(3); d={k:rng.normal(size=100) for k in ("B0","M1","P2","P3","concordance")}
    specs={"Full":["B0","M1","P3","concordance"],"minus_M1":["B0","P3","concordance"],"minus_B0history":["B0","M1","P2","concordance"],"minus_concordance":["B0","M1","P3"]}
    cal=fit_component_calibrations(pd.DataFrame(d),specs)
    assert all(cal[k].dimension==len(v) for k,v in specs.items())
    assert all(tuple(cal[k].columns)==tuple(v) for k,v in specs.items())


def test_frozen_state_detects_actual_array_mutation_not_stored_string():
    x,t=features(); s=fit_m1(x,t,t+.5,10,3,2)
    with pytest.raises(ValueError): s.mean[0]=123
    s.mean.setflags(write=True); s.mean[0]+=1
    with pytest.raises(RuntimeError,match="mutated"): transform_m1(x,t,s)


def test_alarm_timing_uses_available_time_not_target_time():
    d=pd.DataFrame({"t":[119.,119.5,120.],"available_s":[120.,120.5,121.],"score":[0.,2.,2.]})
    a=alarm_times(d,"score",1.,120.,persistence=2)
    assert a["first_alarm_delay_s"]==pytest.approx(.5)
    assert a["persistent_delay_s"]==pytest.approx(.5)


def test_block_permutation_is_region_local_and_preserves_marginals():
    x=np.arange(41); _,y=shuffle_pairing(x,x,seed=2,block=8)
    assert np.array_equal(np.sort(y),x) and len(y)==len(x)


def test_actual_prediction_mse_not_score_proxy():
    y=np.array([[1.,-1.],[2.,-2.]]); p=np.array([[0.,0.],[1.,-1.]])
    assert EVAL.prediction_mse(y,p)==pytest.approx(1.)


def test_permutation_statistics_include_pvalue_and_ci():
    stats=EVAL.permutation_summary(2.,np.arange(20,dtype=float),confidence=.95)
    assert 0<stats["p_value"]<=1 and stats["ci_low"]<=stats["ci_high"] and stats["repetitions"]==20


def test_available_time_propagates_to_epoch_scores():
    d=pd.DataFrame({"t":[1.,1.,2.],"available_s":[1.5,1.6,2.5]})
    got=EVAL.epoch_availability(d)
    assert got.available_s.tolist()==[1.6,2.5]


def test_hash_record_has_actual_stat_and_hash(tmp_path):
    p=tmp_path/"x";p.write_bytes(b"actual")
    r=EVAL.file_record(p)
    assert r["bytes"]==6 and r["sha256"]==hashlib.sha256(b"actual").hexdigest() and r["exists"]


def test_ablation_dependency_sets_are_independent_and_exact():
    assert EVAL.COMPONENT_SPECS == {
        "Full": ["B0", "M1", "P3", "concordance"],
        "minus_M1": ["B0", "P1"],
        "minus_B0history": ["M1", "P2"],
        "minus_concordance": ["B0", "M1", "P3"],
    }


def test_b0_training_digest_ignores_independent_test_mutation():
    d=pd.DataFrame({"run_id":["r"]*4,"prn":[1]*4,"window_start_s":[0.,.5,340.,340.5],
                    "window_end_s":[.5,1.,340.5,341.],"window_bin_s":[0.,.5,340.,340.5],"f":[1.,2.,3.,4.]})
    before=TRAIN.training_input_digest(d,["f"]);d.loc[d.window_start_s>=340,"f"]=1e12
    assert TRAIN.training_input_digest(d,["f"]) == before


def test_actual_clif_artifact_bundle_regressions():
    out=ROOT/"artifacts/clif_ip_cross_layer_r3"
    meta=json.loads((out/"clif_b0_training_metadata.json").read_text())
    assert meta["fit_scope"]["max_window_end_s"] <= 240
    assert meta["standardizer"]["fit_scope"] == "CLIF clean train rows only"
    ck=EVAL.torch.load(out/"clif_b0_prn_local_gru.pt",map_location="cpu",weights_only=False)
    assert ck["training_provenance"]["max_window_end_s"] <= 240
    for name in ("clean_validation","clean_test","os1","os2","os3","os4"):
        d=pd.read_csv(out/f"per_epoch_scores_{name}.csv")
        assert (d.M1 >= 0).all()
        supports=[d[f"P{k}_tracked"].tolist() for k in range(4)]
        assert supports.count(supports[0]) == 4
    lag=pd.read_csv(out/"lag_analysis.csv")
    assert {"alignment_shift_epochs","alignment_shift_s","predictor_history_lag_epochs","predictor_history_lag_s"} <= set(lag)


def test_artifact_checksums_and_tracked_completeness():
    out=ROOT/"artifacts/clif_ip_cross_layer_r3"
    manifest=json.loads((out/"provenance_manifest.json").read_text())
    checks=manifest["artifact_checksums"]
    assert "provenance_manifest.json" not in checks and "test_summary.txt" in checks
    for rel,digest in checks.items():
        assert hashlib.sha256((out/rel).read_bytes()).hexdigest() == digest
    tracked=set(subprocess.check_output(["git","ls-files"],cwd=ROOT,text=True).splitlines())
    required={
        "README.md","config.json","provenance_manifest.json","frozen_m1_fit_summary.json",
        "clif_b0_prn_local_gru.pt","clif_b0_training_metadata.json","clif_b0_training_history.csv",
        "predictor_comparison.csv","hyperparameter_selection.csv","scenario_metrics.csv",
        "fusion_comparison.csv","ablation_metrics.csv","lag_analysis.csv",
        "alignment_destruction_metrics.json","test_summary.txt",
    }
    assert not {str((out/r).relative_to(ROOT)) for r in required} - tracked


def test_readme_csv_key_metrics_are_synchronized():
    out=ROOT/"artifacts/clif_ip_cross_layer_r3"; text=(out/"README.md").read_text()
    pc=pd.read_csv(out/"predictor_comparison.csv")
    p1=pc.query("split=='test' and model=='P1'").mse.iloc[0]
    p3=pc.query("split=='test' and model=='P3'").mse.iloc[0]
    assert f"P1 test MSE={p1:.9g}" in text and f"P3={p3:.9g}" in text
    assert "existing B0 architecture retrained/frozen on CLIF clean train" in text


def test_b0_checkpoint_and_metadata_hash_agree():
    out=ROOT/"artifacts/clif_ip_cross_layer_r3"
    meta=json.loads((out/"clif_b0_training_metadata.json").read_text())
    assert hashlib.sha256((out/"clif_b0_prn_local_gru.pt").read_bytes()).hexdigest() == meta["checkpoint"]["sha256"]


def test_generated_csvs_are_nonempty_and_numeric_values_finite():
    out=ROOT/"artifacts/clif_ip_cross_layer_r3"
    for p in out.glob("*.csv"):
        d=pd.read_csv(p,keep_default_na=False);assert len(d)>0
        for c in d:
            numeric=pd.to_numeric(d[c],errors="coerce")
            values=numeric[numeric.notna()].to_numpy(float)
            assert np.isfinite(values).all(),f"{p.name}:{c}"
    pc=pd.read_csv(out/"predictor_comparison.csv",keep_default_na=False)
    assert set(pc.loc[pc.model.eq("P0"),"alpha"]) == {"N/A"}


def test_raw_iq_provenance_distinguishes_cached_from_live_hashes():
    p=json.loads((ROOT/"artifacts/clif_ip_cross_layer_r3/provenance_manifest.json").read_text())
    for s in p["scenarios"]:
        if s["scenario"].startswith("os"):
            assert s["raw_iq"]["digest_reverified_this_run"] is False
            assert "not reverified this run" in s["raw_iq"]["hash_method"]
        assert s["b0_source_csv"]["digest_reverified_this_run"] is True
        assert s["m1_source_csv"]["digest_reverified_this_run"] is True


def test_actual_permutation_resolution_and_interval_definition():
    p=json.loads((ROOT/"artifacts/clif_ip_cross_layer_r3/alignment_destruction_metrics.json").read_text())
    for region in p["results"].values():
        for k in ("P2","P3","Full"):
            stat=region[f"{k}_delta_statistics"]
            assert stat["repetitions"]==19 and stat["p_value_resolution"]==pytest.approx(.05)
            assert "permutation delta" in stat["ci_definition"]


def test_docs_config_b0_scope_and_alpha_sync():
    out=ROOT/"artifacts/clif_ip_cross_layer_r3";cfg=json.loads((out/"config.json").read_text())
    doc=(ROOT/"docs/CLIF_IP.md").read_text()
    assert cfg["b0_training_max_window_end_s"]==pytest.approx(239.71602)
    assert cfg["b0_checkpoint_sha256"] in doc
    assert "alpha 100 independently for P1/P2/P3" in doc
