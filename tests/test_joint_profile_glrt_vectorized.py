"""Differential contracts for the compiled proper-complex profile kernel."""
from __future__ import annotations

from dataclasses import fields

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


def scalar_reference(obs, los, provider, taps, grid, *, hypothesis, whitener, **kwargs):
    """Test-local frozen access to the legacy epoch-scalar implementation."""
    return joint_profile_glrt(obs, los, provider, taps, grid, hypothesis=hypothesis,
                              whitener=whitener, profile_plan=None, scalar_reference=True, **kwargs)


@pytest.mark.parametrize("epochs", [1, 2, 17, 4095, 4096, 4097])
@pytest.mark.parametrize("row_chunk", [1, 4096])
@pytest.mark.parametrize("candidate_chunk", [1, 7, 8, 16, 72])
def test_h0_h1_independent_scalar_vectorized_differential(epochs, row_chunk, candidate_chunk):
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


def test_jointfit_likelihood_and_bic_identities():
    provider = TemplateProvider.analytic(); w = whitener(dense=True, mean=True); obs = observations((17, 9))
    plan = compile_profile_plan(provider, TAPS, GRID, w, row_chunk=1, candidate_chunk=7)
    h0 = joint_profile_glrt(obs, {}, provider, TAPS, GRID, hypothesis="H0", whitener=w, profile_plan=plan)
    h1 = joint_profile_glrt(obs, {}, provider, TAPS, GRID, hypothesis="H1-independent", whitener=w, profile_plan=plan)
    assert h1.log_likelihood-h1.null_log_likelihood == pytest.approx(h0.rss-h1.rss)
    assert h1.bic == pytest.approx(-2*h1.log_likelihood+h1.k*np.log(h1.n))
    assert h1.score == pytest.approx((-2*h0.log_likelihood+h0.k*np.log(h0.n))-h1.bic)
    assert tuple(f.name for f in fields(JointFit))
