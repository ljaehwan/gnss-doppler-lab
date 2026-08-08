from __future__ import annotations

import numpy as np
import pytest

from gnss_doppler_lab.acquisition_surface import gps_l1ca_code
from gnss_doppler_lab.pg_scc_physics import (
    CENTER, COORDINATES, N_COORDINATES, analytic_same_prn_template,
    coordinate_index, estimate_complex_covariance, inject_same_prn_second_source,
    normalize_complex, two_source_glrt,
)


def test_canonical_gps_l1_ca_code():
    code = gps_l1ca_code(1)
    assert code.shape == (1023,)
    assert set(np.unique(code)) == {-1.0, 1.0}
    assert np.array_equal(code[:10], np.asarray([-1, -1, 1, 1, -1, 1, 1, 1, 1, 1]))


def test_complex_normalization_gain_and_phase_invariance():
    rng = np.random.default_rng(8)
    value = rng.normal(size=N_COORDINATES) + 1j * rng.normal(size=N_COORDINATES)
    value[CENTER] += 9 + 2j
    reference = normalize_complex(value, "prompt_phase")
    for gain in (0.4, 0.8, 1.3, 3.0):
        for phase in (0.0, np.pi / 3, np.pi / 2, np.pi):
            assert np.allclose(reference, normalize_complex(value * gain * np.exp(1j * phase)), atol=2e-9)


def test_same_prn_second_source_is_direct_and_not_zero_padded():
    left = analytic_same_prn_template(-0.75, -150)
    right = analytic_same_prn_template(0.75, 150)
    assert left.shape == right.shape == (187,)
    assert np.count_nonzero(left) > 20 and np.count_nonzero(right) > 20
    assert np.isfinite(left).all() and np.isfinite(right).all()
    rng = np.random.default_rng(2)
    base = analytic_same_prn_template(0, 0)
    injected = inject_same_prn_second_source(
        base, delta_tau_chips=.375, delta_doppler_hz=75, relative_amplitude=.5,
        relative_phase_rad=1.2, noise_sigma=.02, rng=rng, normalization="prompt_phase",
    )
    assert injected.shape == base.shape and not np.allclose(injected, normalize_complex(base))


def test_dense_h0_h1_sanity_and_shrinkage_covariance():
    rng = np.random.default_rng(4)
    auth = analytic_same_prn_template(0, 0)
    clean = np.asarray([normalize_complex(auth + .01 * (rng.normal(size=187) + 1j * rng.normal(size=187))) for _ in range(20)])
    covariance = estimate_complex_covariance(clean, auth, shrinkage=1.0)
    h0 = two_source_glrt(clean[0], auth, covariance).score
    h1_surface = normalize_complex(auth + .8j * analytic_same_prn_template(.375, 75))
    h1 = two_source_glrt(h1_surface, auth, covariance).score
    assert h1 > h0
    assert np.all(np.real(np.diag(covariance)) > 0)
    assert np.allclose(covariance, covariance.conj().T)


def test_coordinate_accounting_and_no_out_of_grid_fallback():
    assert len(COORDINATES) == N_COORDINATES == 187
    assert coordinate_index(0, 0) == CENTER
    with pytest.raises(ValueError):
        coordinate_index(.01, 0)
