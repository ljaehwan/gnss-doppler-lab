from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from gnss_doppler_lab import splitclock_r2_model as model
from gnss_doppler_lab.splitclock_r2_model import (
    center_fit_only,
    first_persistent_alarm_index,
    inject_clock,
    localization_record,
    matched_horizon_statistics,
    observation_wise_mixture_diagnostic,
    persistence_statistic,
    persistent_prn_mixture_loglik,
    score_window,
)


SCALES = np.asarray([0.5, 0.04, 0.04])
PROCESS = np.asarray([0.2, 0.02])


def clean_panel(epochs: int = 10, prns: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20250824)
    values = rng.normal(0.0, np.asarray([0.015, 0.003, 0.003]), (epochs, prns, 3))
    values[:, :, 0] += np.linspace(-0.2, 0.2, prns)[None, :]
    return values, np.ones_like(values, dtype=bool)


def dynamic_two_clock(epochs: int = 10) -> tuple[np.ndarray, np.ndarray]:
    values, valid = clean_panel(epochs)
    values[:, 3:, 1] += 0.5
    values[:, 3:, 2] += 0.5
    return values, valid


def test_01_dynamic_only_two_clock_retention() -> None:
    clean, valid = clean_panel()
    split, _ = dynamic_two_clock()
    centered, centers = center_fit_only(split, valid, np.ones(6, dtype=bool))
    assert np.median(centered[:7, 3:, 1]) - np.median(centered[:7, :3, 1]) == pytest.approx(0.5, abs=0.02)
    assert np.median(centered[:7, 3:, 2]) - np.median(centered[:7, :3, 2]) == pytest.approx(0.5, abs=0.02)
    assert np.ptp(centers[:, 1]) == 0.0
    assert np.ptp(centers[:, 2]) == 0.0
    assert score_window(split, valid, SCALES, PROCESS).score > score_window(clean, valid, SCALES, PROCESS).score + 5.0


def test_02_constant_drift_persistence() -> None:
    clean, valid = clean_panel(30)
    split = clean.copy()
    split[5:, 3:, 1:] += 0.5
    clean_scores = [score_window(clean[end - 9 : end + 1], valid[end - 9 : end + 1], SCALES, PROCESS).score for end in range(9, 30)]
    split_scores = [score_window(split[end - 9 : end + 1], valid[end - 9 : end + 1], SCALES, PROCESS).score for end in range(9, 30)]
    threshold = max(clean_scores)
    assert first_persistent_alarm_index(split_scores, threshold) is not None


def test_03_fixed_prn_latent_likelihood_matches_manual() -> None:
    path0 = np.asarray([-1.0, -3.0])
    path1 = np.asarray([-4.0, -2.0])
    q = np.asarray([0.25, 0.8])
    eligible = np.asarray([True, True])
    expected = math.log(0.75 * math.exp(-1.0) + 0.25 * math.exp(-4.0))
    expected += math.log(0.2 * math.exp(-3.0) + 0.8 * math.exp(-2.0))
    assert persistent_prn_mixture_loglik(path0, path1, q, eligible) == pytest.approx(expected, abs=1e-12)


def test_04_alternating_path_rejected_by_persistent_likelihood() -> None:
    path0_terms = np.asarray([0.0, -20.0, 0.0, -20.0, 0.0, -20.0]).reshape(3, 1, 2)
    path1_terms = np.asarray([-20.0, 0.0, -20.0, 0.0, -20.0, 0.0]).reshape(3, 1, 2)
    q = np.asarray([0.5])
    persistent = persistent_prn_mixture_loglik(np.asarray([path0_terms.sum()]), np.asarray([path1_terms.sum()]), q, np.asarray([True]))
    observation = observation_wise_mixture_diagnostic(path0_terms, path1_terms, q, np.ones_like(path0_terms, dtype=bool))
    assert persistent < observation - 40.0


def test_05_persistent_path_accepted() -> None:
    persistent_path = persistent_prn_mixture_loglik(np.asarray([0.0]), np.asarray([-120.0]), np.asarray([0.5]), np.asarray([True]))
    alternating_path = persistent_prn_mixture_loglik(np.asarray([-60.0]), np.asarray([-60.0]), np.asarray([0.5]), np.asarray([True]))
    assert persistent_path > alternating_path + 50.0


def test_06_same_mask() -> None:
    values, valid = clean_panel()
    result = score_window(values, valid, SCALES, PROCESS)
    assert np.array_equal(result.evaluation_mask, valid[7:])
    assert result.n_valid == int(np.sum(valid[7:]))


def test_07_missing_modality_dynamic_panel() -> None:
    values, valid = clean_panel()
    valid[2, 0, 1] = False
    valid[4, 1, 2] = False
    valid[8, 2, 1] = False
    values[~valid] = np.nan
    result = score_window(values, valid, SCALES, PROCESS)
    assert np.isfinite(result.score)
    assert result.n_valid == int(np.sum(result.evaluation_mask))


def test_08_heldout_isolation() -> None:
    values, valid = dynamic_two_clock()
    first = score_window(values, valid, SCALES, PROCESS)
    mutated = values.copy()
    mutated[7:] += 1000.0
    second = score_window(mutated, valid, SCALES, PROCESS)
    assert first.fit_digest == second.fit_digest
    assert first.selected_restart == second.selected_restart
    np.testing.assert_allclose(first.memberships, second.memberships, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(first.centering, second.centering, atol=0.0, rtol=0.0)


def test_09_prn_permutation_invariance() -> None:
    values, valid = dynamic_two_clock()
    permutation = np.asarray([2, 5, 0, 4, 1, 3])
    first = score_window(values, valid, SCALES, PROCESS)
    second = score_window(values[:, permutation], valid[:, permutation], SCALES, PROCESS)
    assert abs(first.score - second.score) <= 1e-10


def test_10_deterministic_reproduction() -> None:
    values, valid = dynamic_two_clock()
    first = score_window(values, valid, SCALES, PROCESS)
    second = score_window(values, valid, SCALES, PROCESS)
    assert first.fit_digest == second.fit_digest
    assert first.score == second.score


def test_11_all_prn_coherent_boundary() -> None:
    clean, valid = clean_panel()
    coherent = inject_clock(clean, np.arange(6), 0, 0.0, 0.5, 0.0)
    partial = inject_clock(clean, np.arange(3), 0, 0.0, 0.5, 0.0)
    assert score_window(coherent, valid, SCALES, PROCESS).score < score_window(partial, valid, SCALES, PROCESS).score


def test_12_matched_horizon_t_statistic() -> None:
    scores = [-5.0, 2.0, 3.0, 4.0, -1.0, 9.0, 8.0, 1.0]
    assert persistence_statistic(scores) == 2.0
    assert first_persistent_alarm_index(scores, 1.5) == 3


def test_13_a0_a6_identical_horizon() -> None:
    result = matched_horizon_statistics([1, 2, 3, 4], [2, 3, 4, 5])
    assert result == {"score_count": 4, "A0_T": 2.0, "A6_T": 3.0}
    with pytest.raises(ValueError, match="horizon mismatch"):
        matched_horizon_statistics([1, 2, 3], [1, 2])


def test_14_undetected_localization_zero() -> None:
    values, valid = dynamic_two_clock()
    oracle = score_window(values, valid, SCALES, PROCESS)
    record = localization_record(None, oracle, np.asarray([3, 4, 5]))
    assert record["primary_f1"] == 0.0
    assert record["detected"] is False


def test_15_oracle_localization_separate_from_primary() -> None:
    values, valid = dynamic_two_clock()
    oracle = score_window(values, valid, SCALES, PROCESS)
    record = localization_record(None, oracle, np.asarray([3, 4, 5]))
    assert record["oracle_f1"] >= record["primary_f1"]
    assert record["oracle_used_for_gate"] is False


def test_16_primary_has_no_observation_wise_mixture_shortcut() -> None:
    source = inspect.getsource(model.score_window)
    assert "k2_hold = persistent_prn_mixture_loglik" in source
    assert "observation_wise_score = observation_wise_hold" in source
    assert "k2_hold = observation_wise" not in source
