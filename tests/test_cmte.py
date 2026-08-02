import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnss_doppler_lab.cmte import (
    RESIDUAL_COLUMNS, TAP_ORDER, FitState, SequentialState, aggregate_epochs,
    audit_roles, conformal_pvalues, epoch_masks, fit_shared_state,
    label_epochs, load_state, mixture_evalues, score_residuals,
    sequential_scores, validate_residual_frame, save_state,
)


def frame(times=(10., 11., 12.), prns=("G01", "G02"), run="clean"):
    rows=[]
    for t in times:
        for j,p in enumerate(prns):
            r=np.arange(1,10,dtype=float)*(.01+j*.002+t*.0001)
            row={"run_id":run,"prn":p,"window_start_s":t-.8,"window_end_s":t,
                 "window_mid_s":t-.4,"window_bin_s":t-.5,
                 "b0_prn_node_rmse":float(np.sqrt(np.mean(r*r)))}
            row.update(dict(zip(RESIDUAL_COLUMNS,r)))
            rows.append(row)
    return pd.DataFrame(rows)


def test_finite_sample_conformal_ties_and_plus_one():
    cal=np.array([1.,2.,2.,4.])
    got=conformal_pvalues(cal,np.array([2.,3.,5.]))
    assert np.allclose(got,[4/5,2/5,1/5])  # inclusive ties and +1


def test_roles_normal_only_and_no_scenario_or_prefix_leakage():
    roles={"train":frame((10.,)),"validation":frame((20.,)),"test":frame((30.,))}
    report=audit_roles(roles)
    assert report["disjoint"] and report["fit_sources"]==["clean"]
    bad=frame((40.,),run="ds1")
    with pytest.raises(ValueError,match="normal-only"):
        audit_roles({"train":bad,"validation":roles["validation"],"test":roles["test"]})
    prefix=frame((99.,),run="ds2")
    with pytest.raises(ValueError,match="scenario"):
        audit_roles({"train":roles["train"],"validation":prefix,"test":roles["test"]})


def test_adapter_timing_is_causal_and_boundaries_reset():
    x=frame((10.,11.)); validate_residual_frame(x)
    assert (x.window_end_s>x.window_start_s).all()
    # extractor contract: first target index is 12 separately for each run/split.
    x["target_window_index"]=12
    validate_residual_frame(x,require_history_reset=True)
    bad=pd.concat([x.assign(run_id="a"),x.assign(run_id="b",target_window_index=13)])
    with pytest.raises(ValueError,match="history reset"):
        validate_residual_frame(bad,require_history_reset=True)


def test_exact_nine_residuals_order_no_prn_feature_and_rmse():
    x=frame(); meta=validate_residual_frame(x)
    assert RESIDUAL_COLUMNS==tuple(f"residual_{i:03d}" for i in range(9))
    assert TAP_ORDER==("E4","E3","E2","E","P","L","L2","L3","L4")
    assert "prn" not in meta["feature_columns"]
    y=x.copy(); y["b0_prn_node_rmse"]+=1
    with pytest.raises(ValueError,match="RMSE"):
        validate_residual_frame(y)


def test_epoch_aggregation_permutation_variable_counts_empty_and_one_prn():
    state=fit_shared_state(frame((10.,11.,12.)))
    scored=score_residuals(frame((20.,21.),("G01","G02","G03")),state)
    a=aggregate_epochs(scored); b=aggregate_epochs(scored.sample(frac=1,random_state=3))
    pd.testing.assert_frame_equal(a,b)
    one=aggregate_epochs(score_residuals(frame((22.,),("G09",)),state))
    assert len(one)==1 and np.isfinite(one.select_dtypes("number")).all().all()
    empty=aggregate_epochs(scored.iloc[:0])
    assert empty.empty and empty.attrs["contract"]=="skip_epoch; summary metrics NaN"


def test_shrinkage_spd_and_all_scores():
    x=frame(tuple(range(20)),("G01",))
    state=fit_shared_state(x,epsilon=1e-7)
    assert np.linalg.eigvalsh(state.covariance).min()>0
    s=score_residuals(x,state)
    for c in ("q_rmse","q_diag_mahalanobis","q_full_shrinkage_mahalanobis","q_max_standardized_tap"):
        assert c in s and np.isfinite(s[c]).all()


def test_mixture_fixed_kappas_log_clipping_finite():
    out=mixture_evalues(np.array([0.,1e-300,.5,1.]))
    assert out["kappas"]==[.25,.5,.75]
    assert np.isfinite(out["log_e"]).all() and out["clipped_count"]==2


def test_sequential_s1_parallel_restart_and_s2_reset_deterministic():
    loge=np.log(np.array([.5,2.,3.,.2]))
    a=sequential_scores(loge,run_ids=["a","a","b","b"],drift=.01)
    b=sequential_scores(loge,run_ids=["a","a","b","b"],drift=.01)
    assert np.allclose(a[["s1_log_capital","s2_e_cusum"]],b[["s1_log_capital","s2_e_cusum"]])
    assert a.iloc[2].s2_e_cusum==max(0.,loge[2]-.01)
    assert a.iloc[0].s1_log_capital>=0


def test_availability_and_texbat_masks_boundaries():
    x=frame((29.999,30.,89.999,90.,109.999,110.))
    e=aggregate_epochs(score_residuals(x,fit_shared_state(frame(tuple(range(10))))))
    assert np.allclose(e.availability_time_s,[29.999,30.,89.999,90.,109.999,110.])
    labels=label_epochs(e.availability_time_s,onset_s=100.)
    assert list(labels)==["outside","stable","stable","transition","transition","established"]
    masks=epoch_masks(e.availability_time_s,onset_s=100.)
    assert masks["stable"].sum()==2 and masks["transition"].sum()==2 and masks["established"].sum()==1


def test_threshold_source_validation_only_test_independent():
    report=audit_roles({"train":frame((1.,)),"validation":frame((2.,)),"test":frame((3.,))})
    assert report["threshold_source"]=="validation_only"
    assert report["test_used_for_calibration"] is False


def test_validation_shuffle_sanity_and_attack_order_diagnostic():
    state=fit_shared_state(frame(tuple(range(30))))
    scored=score_residuals(frame(tuple(range(30,50))),state)
    e=aggregate_epochs(scored)
    shuffled=aggregate_epochs(scored.sample(frac=1,random_state=8))
    assert np.allclose(e.mean_e,shuffled.mean_e)
    base=sequential_scores(np.log(e.mean_e),["x"]*len(e),drift=.01)
    perm=e.sample(frac=1,random_state=2)
    alt=sequential_scores(np.log(perm.mean_e),["x"]*len(perm),drift=.01)
    assert sorted(e.mean_e)==sorted(perm.mean_e)
    assert not np.allclose(base.s2_e_cusum,alt.s2_e_cusum)


def test_state_hash_pin_roundtrip_and_raw_semantics(tmp_path):
    sha="f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b"
    st=fit_shared_state(frame(tuple(range(15))),checkpoint_sha256=sha)
    p=tmp_path/"state.json"; save_state(st,p); loaded=load_state(p,expected_checkpoint_sha256=sha)
    assert loaded.checkpoint_sha256==sha
    assert loaded.metadata["raw_taps"]=="prompt-relative magnitudes"
    assert loaded.metadata["residuals"]=="signed standardized target-prediction"
    doc=json.loads(p.read_text()); assert doc["state_sha256"]
    with pytest.raises(ValueError,match="checkpoint"):
        load_state(p,expected_checkpoint_sha256="0"*64)
