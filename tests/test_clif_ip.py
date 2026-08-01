import hashlib
import numpy as np
import pytest
from gnss_doppler_lab.clif_ip import (
    M1FitAudit, fit_m1, transform_m1, make_history, true_exceed_fraction,
    aggregate_prns, fit_threshold, independent_fpr, shuffle_pairing,
    validate_provenance_grade,
)


def features(n=40, d=8, seed=1):
    r=np.random.default_rng(seed)
    return r.normal(size=(n,d)), np.arange(n)*.5


def test_m1_fit_exactly_once_clean_train_only_and_hash_invariant():
    x,t=features(); audit=M1FitAudit(); state=fit_m1(x,t,train_end=10,pca_dim=3,lag=2,audit=audit,recording="cleanStatic")
    h=state.sha256; assert audit.fit_count==1 and audit.fit_recordings==["cleanStatic"]
    transform_m1(x,t,state,recording="os2",audit=audit)
    assert audit.fit_count==1 and state.sha256==h


def test_attack_transform_does_not_fit_and_mutation_cannot_change_state():
    x,t=features(); a=M1FitAudit(); s=fit_m1(x,t,10,3,2,audit=a)
    before=s.sha256; attack=np.full_like(x,1e9); transform_m1(attack,t,s,"attack",a)
    assert s.sha256==before and a.fit_count==1 and a.transform_recordings[-1]=="attack"


def test_history_reset_no_future_signed_nine_taps_and_no_prn_identity():
    b=np.arange(6*9,dtype=float).reshape(6,9)-20; m=np.arange(12,dtype=float).reshape(6,2)
    X,y,idx=make_history(b,m,lag=2)
    assert y.shape[1]==9 and np.any(y<0) and idx[0]==2
    assert np.array_equal(X[0,:18],b[:2].reshape(-1))
    assert not np.any(X[0]==b[3,0])
    X2,_,idx2=make_history(b[:3],m[:3],lag=2)
    assert idx2.tolist()==[2]  # a new recording/split starts from an empty history


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
