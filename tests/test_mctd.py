from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.mctd import (
    chronological_masks, consecutive_alarms, epoch_scores, mahalanobis_score,
    nominal_epoch_ms, nonoverlap_blocks, paired_bootstrap_blocks,
    permutation_invariant_score, prompt_normalize, robust_fit, unwrap_by_prn,
)


def test_prompt_complex_normalization_and_gate():
    taps = np.ones((2, 9), dtype=np.complex128) * (2 + 2j)
    taps[0, 4] = 1 + 1j
    taps[1, 4] = 0
    normalized, valid = prompt_normalize(taps, min_magnitude=1e-3)
    assert normalized[0, 4] == pytest.approx(1 + 0j)
    assert valid.tolist() == [True, False]


def test_phase_unwrap_is_prn_local():
    prn = np.array([1, 2, 1, 2])
    epoch = np.array([0, 0, 1, 1])
    phase = np.array([3.0, -3.0, -3.0, 3.0])
    out = unwrap_by_prn(prn, epoch, phase)
    assert abs(out[2] - out[0]) < 1
    assert abs(out[3] - out[1]) < 1


def test_nominal_epoch_alignment():
    assert nominal_epoch_ms(np.array([25001, 49999]), 25_000_000).tolist() == [1, 2]


def test_robust_model_and_identical_collapse():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(1000, 5))
    model = robust_fit(x)
    scores = mahalanobis_score(np.zeros((20, 5)), robust_fit(np.zeros((100, 5))))
    assert np.max(scores) == pytest.approx(0.0)
    assert np.isfinite(mahalanobis_score(x[:10], model)).all()


def test_prn_permutation_invariance_and_variable_n():
    epoch = np.repeat([1, 2], [4, 5])
    prn = np.r_[np.arange(4), np.arange(5)]
    score = np.arange(9.0)
    expected = epoch_scores(epoch, prn, score)
    order = np.array([3, 1, 0, 2, 8, 5, 7, 4, 6])
    actual = permutation_invariant_score(epoch[order], prn[order], score[order])
    for left, right in zip(expected, actual):
        np.testing.assert_allclose(left, right)


def test_fewer_than_four_prns_rejected():
    epoch, score, count = epoch_scores(np.zeros(3), np.arange(3), np.ones(3))
    assert len(epoch) == len(score) == len(count) == 0


def test_nonoverlap_100ms_blocks():
    block, score, count = nonoverlap_blocks(np.array([0, 99, 100, 199]), np.array([1, 3, 5, 7]))
    assert block.tolist() == [0, 100]
    assert score.tolist() == [2, 6]
    assert count.tolist() == [2, 2]


def test_true_consecutive_alarm_and_gap_reset():
    blocks = np.array([0, 100, 200, 400, 500, 600])
    alarms = consecutive_alarms(blocks, np.ones(6) * 2, 1)
    assert alarms.tolist() == [False, False, True, False, False, True]


def test_chronological_split_and_guard_no_overlap():
    time = np.arange(0.0, 500.0, 0.1)
    masks = chronological_masks(time)
    assert all(mask.any() for mask in masks.values())
    total = sum(mask.astype(int) for mask in masks.values())
    assert np.max(total) == 1
    assert np.all(np.diff([time[mask].min() for mask in masks.values()]) > 0)


def test_bootstrap_block_construction():
    np.testing.assert_array_equal(paired_bootstrap_blocks(np.array([0, 9.999, 10, 20])), [0, 0, 1, 2])


def test_freeze_files_and_scenario_handoffs_are_declared():
    root = Path(__file__).resolve().parents[1]
    artifact = root / "artifacts/mctd_stage0_static"
    if artifact.exists():
        assert (artifact / "preregistration.json").exists()
        configs = list((artifact / "frozen_configs").glob("**/*.conf"))
        assert configs
        assert any("texbat_ds3" in path.name for path in configs)
        assert any("texbat_ds7" in path.name for path in configs)

