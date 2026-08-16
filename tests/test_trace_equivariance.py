import numpy as np

from gnss_doppler_lab.trace_action_warp import prompt_normalize, receiver_action, warp_complex_taps
from gnss_doppler_lab.trace_equivariance import (
    action_shuffle_indices, consecutive_alarm, persistent_alarm_ratio,
)


def triangular_peak(center: float = 0.0) -> np.ndarray:
    coords = np.arange(-0.5, 0.5001, 0.125)
    return np.maximum(0.0, 1.0 - np.abs(coords - center)).astype(complex)


def test_canonical_ca_vector_and_action_sign_units():
    current = triangular_peak(0.0)
    warped, valid = warp_complex_taps(current, 0.125, 0.0)
    assert np.allclose(warped[valid], triangular_peak(-0.125)[valid])
    code, phase = receiver_action(1_022_999.0, 0.0, 0.001)
    assert np.isclose(code, -0.001)
    assert phase == 0.0


def test_prompt_normalization_stability_and_invariances():
    base = triangular_peak() * (2.0 + 3.0j)
    expected, valid = prompt_normalize(base)
    for changed in (4.2 * base, base * np.exp(1j * 1.3), -base):
        actual, mask = prompt_normalize(changed)
        assert valid and mask
        assert np.allclose(actual, expected)
    invalid, mask = prompt_normalize(np.zeros(9, complex))
    assert not mask and np.isnan(invalid).all()


def test_no_zero_padding_and_valid_support_mask():
    warped, valid = warp_complex_taps(triangular_peak(), 0.125, 0.0)
    assert valid.sum() == 8
    assert not valid[-1]
    assert np.isnan(warped[-1])


def test_prn_permutation_invariance_and_variable_count():
    values = np.array([1.0, 9.0, 2.0, 3.0])
    assert np.median(values) == np.median(values[[2, 0, 3, 1]])
    assert np.median(values[:3]) != np.median(values)


def test_action_shuffle_preserves_prn_cn0_marginals():
    prn = np.repeat([1, 2], 12)
    cn0 = np.tile(np.repeat([30.1, 34.1], 6), 2)
    idx = action_shuffle_indices(prn, cn0, seed=7)
    assert sorted(idx) == list(range(len(idx)))
    assert np.array_equal(prn[idx], prn)
    assert np.array_equal(np.floor(cn0[idx] / 3), np.floor(cn0 / 3))
    assert not np.array_equal(idx, np.arange(len(idx)))


def test_gap_reset_and_actual_three_consecutive_alarm():
    times = np.array([0.0, 0.5, 1.0, 2.0, 2.5, 3.0])
    alarm = consecutive_alarm(times, np.ones(6), 0.5)
    assert alarm.tolist() == [False, False, True, False, False, True]


def test_persistent_alarm_ratio_is_ratio_not_boolean():
    assert persistent_alarm_ratio([False, True, True], [True, True, True]) == 2 / 3
