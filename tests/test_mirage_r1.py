import numpy as np

from gnss_doppler_lab.mirage_r1 import (
    DELAYS,
    SCALES,
    XI,
    assign_cases,
    balanced_factorial,
    design_balance,
    full_score,
    magnitude_minors,
    multiscale_cafs,
    relative_complex_minors,
)


def test_literal_rank_one_and_rank_two():
    rng = np.random.default_rng(7)
    rank_one = np.outer(rng.normal(size=9) + 1j * rng.normal(size=9),
                        rng.normal(size=5) + 1j * rng.normal(size=5))
    assert np.max(relative_complex_minors(rank_one)) < 1e-12
    rank_two = rank_one + .4 * np.outer(
        rng.normal(size=9) + 1j * rng.normal(size=9),
        rng.normal(size=5) + 1j * rng.normal(size=5),
    )
    assert np.max(relative_complex_minors(rank_two)) > .01


def test_gain_phase_relative_epsilon_and_zero():
    rng = np.random.default_rng(8)
    caf = rng.normal(size=(9, 5)) + 1j * rng.normal(size=(9, 5))
    expected = relative_complex_minors(caf)
    for gain in np.logspace(-6, 6, 13):
        for phase in np.linspace(0, 2 * np.pi, 9):
            observed = relative_complex_minors(gain * np.exp(1j * phase) * caf)
            np.testing.assert_allclose(observed, expected, rtol=2e-10, atol=2e-12)
    np.testing.assert_array_equal(relative_complex_minors(np.zeros((9, 5), complex)), np.zeros((8, 4)))


def test_equal_magnitude_different_phase_is_complex_information():
    magnitude = np.ones((9, 5))
    phase = np.outer(np.linspace(0, 1, 9) ** 2, np.linspace(0, 2, 5) ** 2)
    caf = magnitude * np.exp(1j * phase)
    assert np.max(magnitude_minors(caf)) < 1e-12
    assert np.max(relative_complex_minors(caf)) > .01


def test_multiscale_grid_and_causal_windows():
    taps = np.ones((500, 9), complex)
    cafs = multiscale_cafs(taps)
    assert tuple(cafs) == SCALES
    for scale, caf in cafs.items():
        assert caf.shape == (len(DELAYS), len(XI))
        assert abs(caf[4, 2] - scale * 1000) < 1e-10
        assert np.max(np.abs(np.delete(caf[4], 2))) < 1e-10


def test_design_determinism_seed_sensitivity_balance_and_no_confounding():
    a = balanced_factorial(20260819)
    b = balanced_factorial(20260819)
    c = balanced_factorial(20260820)
    assert a == b
    assert a != c
    audit = design_balance(a)
    assert audit["status"] == "PASS"
    assert audit["delay_power_not_one_to_one"]
    assert audit["phase_doppler_not_one_to_one"]
    anchors = list(range(12))
    cases = assign_cases(20260819, "OAKBAT.cleanStatic", [10, 11, 21, 24, 27], anchors)
    assert len(cases) == 42
    assert sum(row["mode"] == "single_prn" for row in cases) == 30
    assert sum(row["mode"] == "simultaneous_four_prn" for row in cases) == 12
    assert all(len(row["target_prns"]) in (1, 4) for row in cases)


def test_prn_permutation_and_variable_count_full_score():
    assert full_score([9, 1, 7, 3]) == full_score([3, 9, 1, 7]) == 5
    assert full_score([1, 2, 3]) is None
    assert full_score([1, 2, 3, 4, 100]) == 3
