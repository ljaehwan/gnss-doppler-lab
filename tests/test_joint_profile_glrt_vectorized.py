"""Differential contracts for the compiled proper-complex profile kernel."""
from __future__ import annotations

from dataclasses import fields
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.r2c_stage0_fix import (
    ComplexWhitener,
    JointFit,
    TemplateProvider,
    compile_profile_plan,
    joint_profile_glrt,
)


TAPS = np.linspace(-.5, .5, 9)
GRID = np.linspace(-.5, .5, 9)


def whitener(*, dense=False, mean=False):
    obj = ComplexWhitener(shrinkage=0.)
    if dense:
        rng = np.random.default_rng(91)
        a = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
        covariance = a @ a.conj().T + np.eye(9)
        eig, vec = np.linalg.eigh(covariance)
        obj.inverse_sqrt = (vec * (1 / np.sqrt(eig))) @ vec.conj().T
        obj.covariance = covariance
    else:
        obj.inverse_sqrt = np.eye(9, dtype=np.complex128)
        obj.covariance = np.eye(9, dtype=np.complex128)
    obj.mean = (np.linspace(-.2, .3, 9) + 1j*np.linspace(.1, -.4, 9)) if mean else np.zeros(9, complex)
    obj.pseudo_covariance = np.zeros((9, 9), complex)
    obj.diagnostics = {}
    return obj


def observations(counts=(17, 11), seed=17):
    rng = np.random.default_rng(seed)
    provider = TemplateProvider.analytic()
    out = {}
    for prn, count in enumerate(counts, 1):
        base = provider.evaluate(TAPS - GRID[(prn + 2) % len(GRID)])
        out[prn] = ((rng.normal(size=(count, 1)) + 1j*rng.normal(size=(count, 1))) * base +
                    .02*(rng.normal(size=(count, 9)) + 1j*rng.normal(size=(count, 9))))
    return out


_PARENT = None
def scalar_reference(obs, los, provider, taps, grid, *, hypothesis, whitener, **kwargs):
    """Execute the exact parent implementation in an isolated module namespace."""
    global _PARENT
    if _PARENT is None:
        root=Path(__file__).resolve().parents[1]
        source=subprocess.check_output(["git","show","7d8500f:src/gnss_doppler_lab/r2c_stage0_fix.py"],cwd=root,text=True)
        _PARENT=types.ModuleType("frozen_r2c_parent_7d8500f");_PARENT.__dict__["__name__"]=_PARENT.__name__
        import sys
        sys.modules[_PARENT.__name__]=_PARENT
        exec(compile(source,"7d8500f:r2c_stage0_fix.py","exec"),_PARENT.__dict__)
    if provider.analytic_approximation:parent_provider=_PARENT.TemplateProvider.analytic()
    else:parent_provider=_PARENT.TemplateProvider.empirical(provider.offsets_chips,provider.values,provider.provenance)
    parent_w=_PARENT.ComplexWhitener(whitener.shrinkage,whitener.eigen_floor_fraction)
    for name in ("mean","covariance","pseudo_covariance","inverse_sqrt","diagnostics"):
        setattr(parent_w,name,getattr(whitener,name))
    return _PARENT.joint_profile_glrt(obs,los,parent_provider,taps,grid,hypothesis=hypothesis,whitener=parent_w,**kwargs)


@pytest.mark.parametrize("epochs", [1, 2, 17, 4095, 4096, 4097])
def test_h0_h1_independent_scalar_vectorized_differential(epochs):
    _assert_parent_differential(epochs,4096,8)

@pytest.mark.parametrize("row_chunk", [1,4096])
@pytest.mark.parametrize("candidate_chunk", [1,7,8,16,72])
def test_chunk_matrix_parent_differential(row_chunk,candidate_chunk):
    _assert_parent_differential(17,row_chunk,candidate_chunk)

def _assert_parent_differential(epochs, row_chunk, candidate_chunk):
    provider = TemplateProvider.analytic(); w = whitener(dense=epochs % 2 == 0, mean=epochs % 3 == 0)
    obs = observations((epochs,), seed=epochs)
    plan = compile_profile_plan(provider, TAPS, GRID, w, row_chunk=row_chunk,
                                candidate_chunk=candidate_chunk)
    for hypothesis in ("H0", "H1-independent"):
        expected = scalar_reference(obs, {}, provider, TAPS, GRID, hypothesis=hypothesis, whitener=w)
        actual = joint_profile_glrt(obs, {}, provider, TAPS, GRID, hypothesis=hypothesis,
                                    whitener=w, profile_plan=plan)
        for name in ("hypothesis", "n", "k", "epoch_count", "prn_count", "valid",
                     "boundary", "converged", "reason", "delays_chips"):
            assert getattr(actual, name) == getattr(expected, name)
        for name in ("rss", "log_likelihood", "bic", "score", "null_log_likelihood"):
            assert getattr(actual, name) == pytest.approx(getattr(expected, name), rel=1e-10, abs=1e-8)


def test_compiled_plan_is_frozen_snapshot_and_candidate_order():
    provider = TemplateProvider.analytic(); w = whitener(dense=True, mean=True)
    plan = compile_profile_plan(provider, TAPS, GRID[::-1], w)
    assert getattr(type(plan), "__dataclass_params__").frozen
    assert len(plan.h0_candidates) == 9
    assert len(plan.h1_candidates) == 72
    assert plan.h1_candidates == tuple((float(da), float(ds)) for da in GRID for ds in GRID if ds != 0)
    before = plan.whitener.tobytes()
    w.inverse_sqrt[:] = 0
    assert plan.whitener.tobytes() == before


def test_hot_path_decomposition_count_is_independent_of_epochs(monkeypatch):
    provider = TemplateProvider.analytic(); w = whitener(dense=True, mean=True)
    real = np.linalg.svd; calls = []
    monkeypatch.setattr(np.linalg, "svd", lambda *a, **k: (calls.append(np.shape(a[0])), real(*a, **k))[1])
    plan = compile_profile_plan(provider, TAPS, GRID, w)
    compile_calls = len(calls)
    joint_profile_glrt(observations((4096,)), {}, provider, TAPS, GRID,
                       hypothesis="H1-independent", whitener=w, profile_plan=plan)
    assert len(calls) == compile_calls
    assert compile_calls <= 81


def test_lstsq_calls_are_zero_for_unique_and_exact_for_ties(monkeypatch):
    provider=TemplateProvider.analytic();w=whitener(dense=True,mean=True)
    plan=compile_profile_plan(provider,TAPS,GRID,w);row=observations((1,),seed=812)[1]
    real_lstsq=np.linalg.lstsq;counts=[]
    for epochs in (1,2,17,100,4096):
        calls=[]
        monkeypatch.setattr(np.linalg,"lstsq",lambda *a,**k:(calls.append(np.shape(a[1])),real_lstsq(*a,**k))[1])
        joint_profile_glrt({1:np.repeat(row,epochs,axis=0)}, {}, provider,TAPS,GRID,
                           hypothesis="H1-independent",whitener=w,profile_plan=plan)
        counts.append(len(calls))
    assert counts==[2,4,34,200,8192]
    assert plan.counters.scalar_fallback_lstsq_calls==sum(counts)
    unique_plan=compile_profile_plan(provider,TAPS,GRID,w);calls=[]
    monkeypatch.setattr(np.linalg,"lstsq",lambda *a,**k:(calls.append(1),real_lstsq(*a,**k))[1])
    joint_profile_glrt({1:observations((17,),seed=999)[1]}, {},provider,TAPS,GRID,hypothesis="H0",whitener=w,profile_plan=unique_plan)
    assert calls==[] and unique_plan.counters.unique_winner_events==1


def test_jointfit_likelihood_and_bic_identities():
    provider = TemplateProvider.analytic(); w = whitener(dense=True, mean=True); obs = observations((17, 9))
    plan = compile_profile_plan(provider, TAPS, GRID, w, row_chunk=1, candidate_chunk=7)
    h0 = joint_profile_glrt(obs, {}, provider, TAPS, GRID, hypothesis="H0", whitener=w, profile_plan=plan)
    h1 = joint_profile_glrt(obs, {}, provider, TAPS, GRID, hypothesis="H1-independent", whitener=w, profile_plan=plan)
    assert h1.log_likelihood-h1.null_log_likelihood == pytest.approx(h0.rss-h1.rss)
    assert h1.bic == pytest.approx(-2*h1.log_likelihood+h1.k*np.log(h1.n))
    assert h1.score == pytest.approx((-2*h0.log_likelihood+h0.k*np.log(h0.n))-h1.bic)
    assert tuple(f.name for f in fields(JointFit))

def test_h1_shared_exact_parent_all_fields():
    provider=TemplateProvider.analytic();w=whitener(dense=True,mean=True)
    vectors=np.asarray([[1,0,0],[0,1,0],[0,0,1],[-.6,-.5,-.6245],[.5,-.7,.5099]],float);vectors/=np.linalg.norm(vectors,axis=1)[:,None]
    los={p:v for p,v in enumerate(vectors,1)};truth=np.asarray([20.,-15.,10.,40.]);rng=np.random.default_rng(404);obs={}
    for p,u in los.items():
        second=float((-u@truth[:3]+truth[3])/299792458.*1023000.)
        rows=[]
        for _ in range(3):rows.append((1+.1j)*provider.evaluate(TAPS)+(0.7-.2j)*provider.evaluate(TAPS-second))
        obs[p]=np.asarray(rows)
    kwargs={"optimizer_starts":[truth]}
    expected=scalar_reference(obs,los,provider,TAPS,GRID,hypothesis="H1-shared",whitener=w,**kwargs)
    plan=compile_profile_plan(provider,TAPS,GRID,w,candidate_chunk=7)
    actual=joint_profile_glrt(obs,los,provider,TAPS,GRID,hypothesis="H1-shared",whitener=w,profile_plan=plan,**kwargs)
    for name in ("hypothesis","n","k","epoch_count","prn_count","valid","boundary","converged","reason","delays_chips"):
        left=getattr(actual,name);right=getattr(expected,name)
        if name=="delays_chips":assert left==pytest.approx(right,abs=1e-8)
        else:assert left==right
    for name in ("rss","log_likelihood","bic","score","null_log_likelihood"):
        assert getattr(actual,name)==pytest.approx(getattr(expected,name),rel=1e-10,abs=1e-8)
    if actual.beta_m is not None:assert actual.beta_m==pytest.approx(expected.beta_m,abs=1e-3)

def test_authenticated_empirical_provider_parent_differential_and_binding():
    offsets=np.linspace(-2,2,33);values=np.maximum(1-np.abs(offsets),0)*np.exp(1j*.2*offsets)
    provider=TemplateProvider.empirical(offsets,values,{"source_sha256":"a"*64,"kind":"fixture"});w=whitener(dense=True,mean=True)
    obs=observations((17,),seed=505);plan=compile_profile_plan(provider,TAPS,GRID,w)
    expected=scalar_reference(obs,{},provider,TAPS,GRID,hypothesis="H1-independent",whitener=w)
    actual=joint_profile_glrt(obs,{},provider,TAPS,GRID,hypothesis="H1-independent",whitener=w,profile_plan=plan)
    assert actual.delays_chips==expected.delays_chips
    assert actual.rss==pytest.approx(expected.rss,rel=1e-10,abs=1e-8)
    mismatched=TemplateProvider.empirical(offsets,values,{"source_sha256":"b"*64,"kind":"fixture"})
    with pytest.raises(ValueError,match="does not match"):joint_profile_glrt(obs,{},mismatched,TAPS,GRID,hypothesis="H0",whitener=w,profile_plan=plan)

def test_plan_signature_binds_every_fixed_input():
    provider=TemplateProvider.analytic();w=whitener(dense=True,mean=True);base=compile_profile_plan(provider,TAPS,GRID,w)
    variants=[]
    w2=whitener(dense=True,mean=True);w2.mean=w2.mean.copy();w2.mean[0]+=1e-8;variants.append(compile_profile_plan(provider,TAPS,GRID,w2))
    variants.append(compile_profile_plan(provider,TAPS+1e-8,GRID,w))
    variants.append(compile_profile_plan(provider,TAPS,GRID+np.linspace(-1e-8,1e-8,9),w))
    empirical_offsets=np.linspace(-2,2,33);empirical=TemplateProvider.empirical(empirical_offsets,np.maximum(1-np.abs(empirical_offsets),0).astype(complex),{"source_sha256":"c"*64})
    variants.append(compile_profile_plan(empirical,TAPS,GRID,w))
    assert all(item.signature!=base.signature for item in variants)


def test_shared_compiled_objective_does_not_decompose_per_beta_iteration(monkeypatch):
    """The shared beta objective must use the compiled/batched kernel, not SVD per candidate."""
    provider=TemplateProvider.analytic(); w=whitener(dense=True,mean=True)
    vectors=np.asarray([[1,0,0],[0,1,0],[0,0,1],[-.6,-.5,-.6245],[.5,-.7,.5099]],float)
    vectors/=np.linalg.norm(vectors,axis=1)[:,None]
    los={p:v for p,v in enumerate(vectors,1)}
    truth=np.asarray([20.,-15.,10.,40.]); rng=np.random.default_rng(909)
    obs={}
    for p,u in los.items():
        second=float((-u@truth[:3]+truth[3])/299792458.*1023000.)
        base=(1+.1j)*provider.evaluate(TAPS)+(.7-.2j)*provider.evaluate(TAPS-second)
        obs[p]=np.asarray([base+.01*(rng.normal(size=9)+1j*rng.normal(size=9)) for _ in range(4)])
    real=np.linalg.svd; calls=[]
    monkeypatch.setattr(np.linalg,"svd",lambda *a,**kw:(calls.append(np.shape(a[0])),real(*a,**kw))[1])
    plan=compile_profile_plan(provider,TAPS,GRID,w,candidate_chunk=7)
    compiled=len(calls)
    fit=joint_profile_glrt(obs,los,provider,TAPS,GRID,hypothesis="H1-shared",whitener=w,
                           profile_plan=plan,optimizer_starts=[truth])
    assert fit.valid
    # The shared objective may factor a whole candidate bank, but never dispatch
    # a separate decomposition for each candidate.
    extra=calls[compiled:]
    assert extra and all(shape==(len(GRID),len(TAPS),2) for shape in extra)
