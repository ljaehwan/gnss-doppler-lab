from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.q_comet import (
    InnovationTable, analytic_log_bf, complex_to_real, empirical_threshold,
    fit_predictor, fit_whitener, innovations, nuisance_jacobian,
    predictor_validation_nll, quotient_project, rank1_values,
    real_to_complex, score_common_onset, score_independent_changepoints,
)
from gnss_doppler_lab.q_comet_data import (
    EpochData, aggregate_epochs, audit_split_ranges, canonical_json_hash,
    desynchronize_by_prn,
)


def synthetic_data(n_epochs=90, n_prns=6, cadence=.1, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    shape = np.exp(-np.abs(np.arange(9)-4)/2.5) * np.exp(1j*np.linspace(-.1,.1,9))
    state = {p: shape*(1+.02*p)*np.exp(.1j*p) for p in range(1,n_prns+1)}
    for e in range(n_epochs):
        for p in range(1,n_prns+1):
            state[p] = .98*state[p] + .02*shape + .002*(rng.normal(size=9)+1j*rng.normal(size=9))
            rows.append((e*cadence+.09*cadence, e, p, 0, state[p].copy(), e*100+p, 45., abs(state[p][4])**2))
    columns = [np.asarray([row[k] for row in rows]) for k in range(8)]
    return EpochData(*columns, cadence, "synthetic-clean")


def fit_fixture():
    data = synthetic_data()
    model = fit_predictor(data, kind="ridge_var", lags=2, ridge=1e-3, train_range=(0,4))
    ix, y, pred = model.predict_rows(data, start_s=4, end_s=6)
    white = fit_whitener(y-pred)
    return data, model, white


def test_complex_to_real_roundtrip():
    z=np.arange(18).reshape(2,9)+1j*np.arange(18,36).reshape(2,9)
    assert np.array_equal(real_to_complex(complex_to_real(z)),z)


def test_causal_predictor_and_no_future_leakage():
    data=synthetic_data(); model=fit_predictor(data,kind="persistence",lags=1,ridge=0,train_range=(0,4))
    ix,_,pred=model.predict_rows(data,start_s=2,end_s=3)
    mutated=data.complex_taps.copy(); mutated[data.time_s>=3]+=100
    other=EpochData(data.time_s,data.epoch,data.prn,data.segment,mutated,data.sample_count,data.cn0_db_hz,data.prompt_power,data.cadence_s,data.recording_id)
    ix2,_,pred2=model.predict_rows(other,start_s=2,end_s=3)
    assert np.array_equal(ix,ix2) and np.array_equal(pred,pred2)


def test_normal_only_fit_is_range_restricted():
    data=synthetic_data(); m1=fit_predictor(data,kind="ridge_var",lags=2,ridge=.1,train_range=(0,4))
    changed=data.complex_taps.copy(); changed[data.time_s>=4]+=999
    d2=EpochData(data.time_s,data.epoch,data.prn,data.segment,changed,data.sample_count,data.cn0_db_hz,data.prompt_power,data.cadence_s,data.recording_id)
    m2=fit_predictor(d2,kind="ridge_var",lags=2,ridge=.1,train_range=(0,4))
    assert np.array_equal(m1.coefficients,m2.coefficients)


def test_calibration_a_b_separation_and_byte_audit():
    report=audit_split_ranges({"train":(20,140),"calibration_a":(150,210),"calibration_b":(220,340),"holdout":(350,470)},sample_rate_hz=25e6)
    assert report["all_disjoint"] and all(x["overlap_bytes"]==0 for x in report["pairwise_overlap"])


def test_aggregate_is_causal_and_preserves_identity():
    iq=np.zeros((8,9,2)); iq[:,4,0]=np.arange(8)+1
    time=np.array([.01,.02,.11,.12]*2); prn=np.array([1]*4+[2]*4); channel=np.array([0]*4+[1]*4)
    data=aggregate_epochs(iq=iq,time_s=time,prn=prn,segment=channel,sample_count=np.arange(8)+1,cn0=np.ones(8)*40,cadence_s=.1,recording_id="x")
    assert len(data.time_s)==4 and np.all(data.time_s >= data.epoch*.1)


def test_covariance_regularization_positive_definite():
    x=np.ones((30,18)); w=fit_whitener(x)
    assert np.linalg.eigvalsh(w.covariance).min()>0 and np.isfinite(w.inverse_sqrt).all()


def test_nuisance_projection_removes_gain_phase_delay_tangents():
    reference=np.exp(-abs(np.arange(9)-4)/2)*np.exp(1j*.1*np.arange(9)); r=complex_to_real(reference)
    inv=np.eye(18); j=nuisance_jacobian(r,inv)
    for tangent in j:
        assert np.linalg.norm(quotient_project(tangent,r,inv))<1e-9


def test_prn_permutation_invariance_and_variable_count():
    data,model,w=fit_fixture(); table=innovations(data,model,w,start_s=6,end_s=8)
    one=score_common_onset(table,memory_epochs=8,participation=.5,prior_variance=.25)
    order=np.arange(len(table.prn))[::-1]
    perm=InnovationTable(*(np.asarray(getattr(table,name))[order] for name in table.__dataclass_fields__))
    two=score_common_onset(perm,memory_epochs=8,participation=.5,prior_variance=.25)
    assert [x["score"] for x in one]==[x["score"] for x in two]
    assert [x["prn_log_bf"] for x in one]==[x["prn_log_bf"] for x in two]
    assert all(set(row["prn_log_bf"])==set(map(str,np.unique(table.prn[table.epoch==row["epoch"]]))) for row in one)


def test_less_than_four_prns_is_no_score():
    data=synthetic_data(n_prns=3); model=fit_predictor(data,kind="persistence",lags=1,ridge=0,train_range=(0,4))
    ix,y,p=model.predict_rows(data,start_s=4,end_s=5); w=fit_whitener(y-p)
    rows=score_common_onset(innovations(data,model,w,start_s=4,end_s=5),memory_epochs=5,participation=.5,prior_variance=.25)
    assert all(np.isnan(row["score"]) for row in rows)


def test_common_onset_likelihood_responds_to_shared_change():
    rng=np.random.default_rng(2); normal=rng.normal(size=(10,8))*.1; changed=normal+2
    assert analytic_log_bf(changed)>analytic_log_bf(normal)


def test_independent_changepoint_ablation_runs_same_epochs():
    data,model,w=fit_fixture(); table=innovations(data,model,w,start_s=6,end_s=8)
    full=score_common_onset(table,memory_epochs=8,participation=.5,prior_variance=.25)
    a3=score_independent_changepoints(table,memory_epochs=8,participation=.5,prior_variance=.25)
    assert [x["epoch"] for x in full]==[x["epoch"] for x in a3]


def test_timestamp_gap_and_recording_boundary_reset():
    data=synthetic_data(); epochs=data.epoch.copy(); epochs[epochs>=45]+=5
    changed=EpochData(data.time_s,epochs,data.prn,data.segment,data.complex_taps,data.sample_count,data.cn0_db_hz,data.prompt_power,data.cadence_s,data.recording_id)
    model=fit_predictor(changed,kind="persistence",lags=1,ridge=0,train_range=(0,9))
    ix,_,_=model.predict_rows(changed)
    assert not np.any(changed.epoch[ix]==50)  # first row after gap has no bridged history
    segments=data.segment.copy(); segments[data.epoch>=45]=1
    boundary=EpochData(data.time_s,data.epoch,data.prn,segments,data.complex_taps,data.sample_count,data.cn0_db_hz,data.prompt_power,data.cadence_s,"other")
    ix2,_,_=fit_predictor(boundary,kind="persistence",lags=1,ridge=0,train_range=(0,9)).predict_rows(boundary)
    assert not np.any(boundary.epoch[ix2]==45)


def test_timeline_validation_rejects_nonmonotone():
    report=audit_split_ranges({"a":(0,2),"b":(1,3)},sample_rate_hz=10)
    assert not report["all_disjoint"]


def test_common_support_rank1_shape():
    v=np.arange(48,dtype=float).reshape(6,8); prn=np.array([1,2,3]*2); epoch=np.array([1]*3+[2]*3)
    out=rank1_values(v,prn,epoch)
    assert out.shape==v.shape and np.linalg.matrix_rank(out[:3])==1


def test_relation_destruction_preservation_and_determinism():
    v=np.arange(120,dtype=float).reshape(15,8); p=np.repeat([1,2,3],5); e=np.tile(np.arange(5),3)
    a,ra=desynchronize_by_prn(v,p,e); b,rb=desynchronize_by_prn(v,p,e)
    assert np.array_equal(a,b) and ra==rb
    assert np.array_equal(np.sort(np.linalg.norm(a,axis=1)),np.sort(np.linalg.norm(v,axis=1)))


def test_predictor_selection_and_threshold_deterministic():
    data=synthetic_data(); p=fit_predictor(data,kind="persistence",lags=1,ridge=0,train_range=(0,4))
    assert predictor_validation_nll(data,p,(4,6))==predictor_validation_nll(data,p,(4,6))
    assert empirical_threshold([1,2,3,4],.75)==3


def test_artifact_checksum_canonical():
    assert canonical_json_hash({"b":1,"a":2})==hashlib.sha256(b'{"a":2,"b":1}').hexdigest()
