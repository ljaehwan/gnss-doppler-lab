from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from gnss_doppler_lab.cmte_a2 import (
    A2_EPOCH_COLUMNS,
    A2_PRN_COLUMNS,
    B0Config,
    B0ExactState,
    B0Predictor,
    FitState,
    aggregate_epochs,
    alarm_flags,
    assign_normal_role,
    b0_enhanced_scores,
    b0_exact_scores,
    bootstrap_metrics,
    build_history_examples,
    conformal_pvalues,
    deterministic_controls,
    empirical_target1,
    epoch_metrics,
    fit_distribution,
    fit_standardizer,
    higher_quantile,
    make_fit_fingerprint,
    phase_masks,
    score_residuals,
    select_prn_holdout,
    verify_confirmatory_freeze,
)
from gnss_doppler_lab.cmte_a2_inputs import parse_scenario_mappings, prepare_named_complex_inputs

TAPS = [f"tap_{x}_rel_prompt_mean" for x in ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")]
RES = [f"residual_{i:03d}" for i in range(9)]


def node_frame(n=80, run="clean", prns=("G01", "G02"), start=0.0):
    rows=[]
    for prn_i,prn in enumerate(prns):
        for i in range(n):
            t=start+i*.5
            row={"run_id":run,"physical_recording_id":run,"split":"normal","prn":prn,
                 "segment":"s0","channel":prn_i,"window_start_s":t,"window_end_s":t+1,
                 "window_mid_s":t+.5,"window_bin_s":t+.5}
            row.update({c:float(i+j+prn_i)/100 for j,c in enumerate(TAPS)})
            rows.append(row)
    return pd.DataFrame(rows)


def residual_frame(n=30, run="r", seed=1):
    rng=np.random.default_rng(seed); rows=[]
    for i in range(n):
        row={"physical_recording_id":run,"history_id":f"{run}:h","role":"fit","split":"normal",
             "prn":f"G{i%5+1:02d}","segment":"s","channel":0,"window_start_s":i*.5,
             "window_end_s":i*.5+1,"window_bin_s":i*.5+1,"target_window_index":12+i}
        row.update(dict(zip(RES,rng.normal(size=9))))
        rows.append(row)
    return pd.DataFrame(rows)


def test_schema_has_exact_a2_evidence_and_no_sequential_fields():
    forbidden=("e","mean_e","S1","S2","CUSUM","capital","sequential")
    columns={x.lower() for x in A2_PRN_COLUMNS+A2_EPOCH_COLUMNS}
    assert not any(x.lower() in columns for x in forbidden)
    assert {"p","q","rmse",*RES}.issubset(A2_PRN_COLUMNS)
    assert {"tracked_prn_count","min_p","median_p","mean_q","max_q","mean_neg_log_p","median_neg_log_p","score_A2"}.issubset(A2_EPOCH_COLUMNS)


def test_b0_architecture_exact_and_prn_identity_absent():
    model=B0Predictor(B0Config())
    assert model.encoder[0].in_features==9 and model.encoder[0].out_features==128
    assert isinstance(model.encoder[1],torch.nn.LayerNorm)
    assert model.encoder[3].p==.05
    assert model.gru.input_size==model.gru.hidden_size==128 and model.gru.num_layers==1
    assert model.head[-1].out_features==9
    names=" ".join(n for n,_ in model.named_parameters()).lower()
    assert "prn" not in names and "embedding" not in names
    x=torch.randn(4,12,9); assert model(x).shape==(4,9)
    x[:,11,:]=999
    assert torch.equal(model.eval()(x[:,:12]),model(x[:,:12]))


def test_deterministic_rerun_controls_reproduce_model_initialization():
    deterministic_controls(11); first=B0Predictor(B0Config()).state_dict()
    deterministic_controls(11); second=B0Predictor(B0Config()).state_dict()
    assert all(torch.equal(first[key],second[key]) for key in first)


def test_roles_are_fully_contained_and_separate():
    f=pd.DataFrame({"window_start_s":[239,240,249,250,289,290,299,300,329,330,339,340],
                    "window_end_s":[240,241,250,251,290,291,300,301,330,331,340,341]})
    got=[assign_normal_role(r.window_start_s,r.window_end_s) for r in f.itertuples()]
    assert got==["prefix",None,None,"qcal","qcal",None,None,"threshold","threshold",None,None,"clean_test"]


def test_history_reset_role_split_recording_segment_channel_and_gap():
    base=node_frame(20,prns=("G01",))
    base["role"]="prefix"
    variants=[]
    for col,value in (("role","qcal"),("split","other"),("physical_recording_id","r2"),("segment","s2"),("channel",7)):
        x=base.iloc[:13].copy(); x[col]=value; variants.append(x)
    gap=base.iloc[:13].copy(); gap[["window_start_s","window_end_s","window_mid_s","window_bin_s"]]+=30
    all_rows=pd.concat([base,*variants,gap],ignore_index=True)
    examples,audit=build_history_examples(all_rows,TAPS,seq_len=12,cadence_s=.5)
    assert len(examples)==(20-12)+len(variants)+1  # post-gap 13-row chunk contributes one target
    assert audit["reset_dimensions"]==["role","split","physical_recording_id","segment","channel","cadence_gap"]
    assert audit["gaps_detected"]==1 and audit["bridged"] is False
    assert examples.history_id.nunique()==2+len(variants)


def test_causal_future_exclusion_and_target_index():
    f=node_frame(20,prns=("G01",)); f["role"]="prefix"
    a,_=build_history_examples(f,TAPS)
    mutated=f.copy(); mutated.loc[mutated.window_start_s>=7,TAPS]=1e9
    b,_=build_history_examples(mutated,TAPS)
    np.testing.assert_array_equal(a.iloc[0].history,b.iloc[0].history)
    np.testing.assert_array_equal(a.iloc[0].target,b.iloc[0].target)
    assert a.target_window_index.iloc[0]==12 and a.history.iloc[0].shape==(12,9)


def test_holdout_scaler_prefix_only_and_fingerprint_attack_invariant():
    f=node_frame(30,prns=("G01","G02","G03","G04","G05")); f["role"]="prefix"
    train,val=select_prn_holdout(f)
    assert set(val.prn.unique())=={"G05"} and "G05" not in set(train.prn)
    mean,std=fit_standardizer(train[TAPS].to_numpy())
    assert np.all(std>=1e-6)
    fp=make_fit_fingerprint(train,mean,std,extra={"epochs":25})
    attack=node_frame(20,run="DS7",prns=("G01",),start=100)
    attack[TAPS]=999
    assert fp==make_fit_fingerprint(train,mean,std,extra={"epochs":25})


def test_scaler_nan_rules_ddof_zero():
    x=np.array([[1,np.nan,3],[3,np.nan,np.inf]],float)
    mean,std=fit_standardizer(x)
    np.testing.assert_allclose(mean,[2,0,3]); np.testing.assert_allclose(std,[1,1,1])


def test_distribution_formula_and_attack_mutation_no_fit_effect():
    fit=residual_frame(40); state=fit_distribution(fit)
    assert state.mean.shape==(9,) and state.covariance.shape==(9,9)
    assert state.shrinkage==pytest.approx(10/40)
    attack=residual_frame(20,"DS7"); a=state.state_hash
    attack[RES]*=1e6
    assert state.state_hash==a


def test_conformal_plus_one_inclusive_ties_and_variable_n_permutation():
    p=conformal_pvalues(np.array([1,2,2,4]),np.array([0,2,3,5]))
    np.testing.assert_allclose(p,[1,4/5,2/5,1/5])
    s=fit_distribution(residual_frame(50)); s.qcal=np.array([1,2,2,4],float)
    scored=score_residuals(residual_frame(8),s)
    epoch=aggregate_epochs(scored)
    shuffled=aggregate_epochs(scored.sample(frac=1,random_state=2))
    pd.testing.assert_frame_equal(epoch,shuffled)
    assert epoch.tracked_prn_count.tolist()==[1]*8


def test_duplicate_prn_epoch_fails_closed():
    f=residual_frame(2); f.loc[1,["prn","window_end_s","window_bin_s"]]=f.loc[0,["prn","window_end_s","window_bin_s"]]
    s=fit_distribution(residual_frame(50)); s.qcal=np.arange(10.)
    with pytest.raises(ValueError,match="duplicate PRN"):
        aggregate_epochs(score_residuals(f,s))


def test_epoch_aggregation_supports_variable_current_tracked_n():
    f=residual_frame(4); f.loc[:2,["window_end_s","window_bin_s"]]=1.; f.loc[3,["window_end_s","window_bin_s"]]=2.
    f.loc[:2,"prn"]=["G01","G02","G03"]; f.loc[3,"prn"]="G01"
    s=fit_distribution(residual_frame(50)); s.qcal=np.arange(10.)
    out=aggregate_epochs(score_residuals(f,s))
    assert out.tracked_prn_count.tolist()==[3,1] and np.isfinite(out.score_A2).all()


def test_a2_exact_mean_logp_and_forbidden_cmte_functions_unused(monkeypatch):
    import gnss_doppler_lab.cmte as old
    for name in ("mixture_evalues","sequential_scores"):
        monkeypatch.setattr(old,name,lambda *a,**k: (_ for _ in ()).throw(AssertionError("forbidden")))
    s=fit_distribution(residual_frame(60)); s.qcal=np.arange(1,20,dtype=float)
    scored=score_residuals(residual_frame(10),s); out=aggregate_epochs(scored)
    expected=scored.groupby(["physical_recording_id","window_end_s"]).p.apply(lambda x:np.mean(-np.log(x)))
    np.testing.assert_allclose(out.score_A2,expected.to_numpy())


def test_higher_quantile_target_and_strict_alarm():
    x=np.array([0.,1.,2.,3.,4.])
    assert higher_quantile(x,.5)==2 and higher_quantile(x,.995)==4
    assert empirical_target1(x)==4
    np.testing.assert_array_equal(alarm_flags([3,4,5],4),[False,False,True])


def test_b0_exact_gate_golden_and_reset():
    f=pd.DataFrame({"physical_recording_id":["a"]*2+["b"],"window_end_s":[1,1.5,1],"rmse_values":[np.array([0,2,3]),np.array([0,0]),np.array([2]) ]})
    out=b0_exact_scores(f,{"q50":1.,"q70":1.,"q80":1.})
    raw0=max(-math.log(sum(math.comb(3,j)*p**j*(1-p)**(3-j) for j in range(2,4))) for p in (.5,.3,.2))
    assert out.raw.iloc[0]==pytest.approx(raw0,abs=1e-12)
    assert out.score.iloc[0]==pytest.approx(.25*raw0,abs=1e-12)
    assert out.score.iloc[2]==pytest.approx(.25*out.raw.iloc[2],abs=1e-12)
    assert {"k_q50","n_q50","tail_q50"}.issubset(out)


def test_b0_exact_gate_numerically_matches_historical_evaluator():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("historical_gate",root/"scripts/eval_btail_support_gate.py")
    assert spec is not None and spec.loader is not None
    historical=importlib.util.module_from_spec(spec); sys.modules[spec.name]=historical; spec.loader.exec_module(historical)
    prn=pd.DataFrame({"run_id":["r"]*6,"prn":["G01","G02"]*3,"window_bin_s":[1,1,1.5,1.5,2,2],
                      "window_start_s":[.5,.5,1,1,1.5,1.5],"window_mid_s":[1,1,1.5,1.5,2,2],
                      "prn_node_rmse":[0.,2.,2.,3.,0.,0.]})
    thresholds={"q50":1.,"q70":1.,"q80":1.}
    old=historical.build_event_scores(prn,thresholds,alpha=.75)
    grouped=pd.DataFrame({"physical_recording_id":old.run_id,"window_end_s":old.window_start_s+1,
                          "rmse_values":[g.prn_node_rmse.to_numpy() for _,g in prn.groupby("window_bin_s",sort=True)]})
    new=b0_exact_scores(grouped,thresholds)
    for q in ("q50","q70","q80"):
        np.testing.assert_array_equal(new[f"k_{q}"],old[f"k_{q}"])
        np.testing.assert_allclose(-np.log(new[f"tail_{q}"]),old[f"btail_{q}"],atol=1e-12,rtol=0)
    np.testing.assert_allclose(new.raw,old.btail_max_507080,atol=1e-12,rtol=0)
    np.testing.assert_allclose(new.score,old[historical.FINAL_SCORE],atol=1e-12,rtol=0)


def test_b0_enhanced_is_current_weight_pandas_semantics():
    f=pd.DataFrame({"physical_recording_id":["a"]*2,"window_end_s":[1,1.5],"rmse_values":[np.array([2]),np.array([0])]})
    out=b0_enhanced_scores(f,{"q50":1.,"q70":1.,"q80":1.},{"q50":.5,"q70":.3,"q80":.2})
    assert out.score.iloc[0]==out.raw.iloc[0]
    assert out.score.iloc[1]==pytest.approx(.25*out.raw.iloc[0]+.75*out.raw.iloc[1])


def test_phase_policy_boundaries_and_metrics():
    f=pd.DataFrame({"physical_recording_id":["r"]*8,"window_start_s":[30,89,89.5,90,109,109.5,110,150],
                    "window_end_s":[31,90,90.5,91,110,110.5,111,151],"score_A2":[0,0,9,0,0,9,2,3],"tracked_prn_count":[5]*8})
    m=phase_masks(f,110)
    assert m["stable_pre"].tolist()==[True,True,False,False,False,False,False,False]
    assert m["transition"].tolist()==[False,False,False,True,True,False,False,False]
    result=epoch_metrics(f,"score_A2",1.,onset_s=110,clean_fpr=.01)
    assert result["stable_pre_fpr"]==0 and result["post_detection_rate"]==1
    assert result["first_alarm_delay_s"]==1
    assert result["tracked_prn_count_median"]==5


def test_bootstrap_deterministic_gap_safe_and_short_na():
    rows=[]
    for phase,start,label in (("stable_pre",30,0),("post",110,1),("persistent",150,1)):
        for i in range(40): rows.append({"scenario":"DS7","physical_recording_id":"r","phase":phase,"window_end_s":start+i*.5,
                                         "score":i/40+label,"alarm":bool(i%3==0)})
    f=pd.DataFrame(rows)
    a=bootstrap_metrics(f,reps=50,seed=20260802); b=bootstrap_metrics(f,reps=50,seed=20260802)
    assert a==b and all(v["reason"] is None for v in a.values())
    short=bootstrap_metrics(f.groupby("phase",group_keys=False).head(20),reps=5)
    assert all(v["low"] is None and "fewer_than_2_complete_blocks" in v["reason"] for v in short.values())


def test_confirmatory_freeze_requires_prereg_ancestor_clean_hashes(tmp_path):
    manifest={"prereg_commit":"e7cb2e5822923a129d72c475706f87721ddd8104","source_commit":"abc","files":{}}
    p=tmp_path/"f.json"; p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): verify_confirmatory_freeze(p,repo=tmp_path)


def test_tier_mappings_exact_and_development_rejects_holdout_tokens(tmp_path):
    specs=parse_scenario_mappings([f"DS{i}={tmp_path}/x{i}.npz" for i in range(1,5)],tier="development",require_exists=False)
    assert set(specs)=={"DS1","DS2","DS3","DS4"}
    with pytest.raises(ValueError,match="DS7/DS8"):
        parse_scenario_mappings([f"DS{i}={tmp_path}/DS7/x{i}.npz" for i in range(1,5)],tier="development",require_exists=False)
    with pytest.raises(ValueError): parse_scenario_mappings([f"DS7={tmp_path}/x"],tier="confirmatory",require_exists=False)


def test_uniform_complex_producer_and_hash_manifests(tmp_path):
    paths=[]
    for name in ("DS1","DS2"):
        path=tmp_path/f"{name}.npz"; n=50
        iq=np.ones((n,9,2),np.float32); iq[:,:,1]=np.arange(9)[None,:]
        np.savez(path,complex_iq=iq,prn=np.ones(n,int),time_s=np.arange(n)*.05,
                 segment=np.zeros(n,int),channel=np.zeros(n,int)); paths.append(f"{name}={path}")
    manifests=prepare_named_complex_inputs(dict(item.split("=",1) for item in paths),tmp_path/"prepared")
    assert manifests["DS1"]["converter_semantics"]==manifests["DS2"]["converter_semantics"]
    for name in ("DS1","DS2"):
        node=tmp_path/"prepared"/f"{name}_nodes.csv"
        assert hashlib.sha256(node.read_bytes()).hexdigest()==manifests[name]["node_sha256"]


def test_ds8_source_prep_import_graph_has_no_scorer_or_detector_import():
    root=Path(__file__).resolve().parents[1]
    source=(root/"scripts/prepare_cmte_a2_ds8_complex.py").read_text()
    tree=ast.parse(source); imports=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Import): imports.extend(x.name for x in n.names)
        if isinstance(n,ast.ImportFrom): imports.append(n.module or "")
    text=" ".join(imports).lower()
    assert "cmte_a2" not in text and "score" not in text and "detector" not in text
    assert "1614d8de6fc8ebc3429def6e9505050c08a3ee8da69c11ecc27a98305f735d78" in source
    assert "6c4512adefcfe49ae7d964c0425b26bfffd8b988ad7f9a0cf6f4b2e30fc5cafb" in source
    assert "30a45f988cec15fdce84552ff30747b472c7d76df07d93f79d6ae236166d4039" in source


def test_eval_confirmatory_source_does_not_import_training_script():
    root=Path(__file__).resolve().parents[1]; source=(root/"scripts/eval_cmte_a2_texbat.py").read_text().lower()
    assert "train_cmte_a2" not in source
    assert "fit_distribution(" not in source and "fit_threshold" not in source
    assert source.index("verify_confirmatory_freeze") < source.index("torch.load")


def test_preregistration_files_unchanged():
    root=Path(__file__).resolve().parents[1]
    expected={"docs/CMTE_A2_PREREGISTRATION.md":"5bc92fb711ed85ee20f67e9c8deac7b10bbc9de01d65736eaf4b797d87b6a64f",
              "configs/cmte_a2_preregistration.json":"c2e090aba28acbbd094272aa6bd2c13edab4399d8e406811ec0417d941ebfd8f"}
    for name,digest in expected.items(): assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
