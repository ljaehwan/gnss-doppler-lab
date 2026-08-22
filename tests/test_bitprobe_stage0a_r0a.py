from __future__ import annotations

import itertools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.bitprobe_stage0a_r0a import (
    BLOCK_LENGTH,
    REPLICATES,
    exact_prn_permutation,
    permutation_statistic,
    resample_half_indices,
    split_chronological,
)


class FixedRng:
    def __init__(self, values: list[int]):
        self.values = iter(values)

    def integers(self, low: int, high: int | None = None, size: int | None = None) -> int:
        assert size is None
        value = next(self.values)
        assert low <= value < int(high)
        return value


def test_first_half_source_never_enters_second_resample() -> None:
    sequence = np.arange(21)
    first, second = split_chronological(sequence)
    first_sample = first[resample_half_indices(len(first), np.random.default_rng(1))]
    second_sample = second[resample_half_indices(len(second), np.random.default_rng(2))]
    assert set(first_sample).isdisjoint(set(second))


def test_second_half_source_never_enters_first_resample() -> None:
    sequence = np.arange(21)
    first, second = split_chronological(sequence)
    first_sample = first[resample_half_indices(len(first), np.random.default_rng(3))]
    second_sample = second[resample_half_indices(len(second), np.random.default_rng(4))]
    assert set(second_sample).isdisjoint(set(first))


def test_sampled_blocks_are_contiguous_and_final_block_only_is_truncated() -> None:
    indices = resample_half_indices(25, FixedRng([1, 2, 0]), block_length=10)
    assert indices.tolist() == list(range(10, 20)) + list(range(20, 25)) + list(range(0, 10))
    assert len(indices) == 25


def test_bootstrap_replicate_count_exactly_frozen_count() -> None:
    rng = np.random.default_rng(5)
    values = [resample_half_indices(23, rng, BLOCK_LENGTH) for _ in range(REPLICATES)]
    assert len(values) == REPLICATES
    assert all(len(value) == 23 for value in values)


def test_fixed_seed_is_byte_identical() -> None:
    first = [resample_half_indices(37, np.random.default_rng(20260822 + index)).tobytes() for index in range(10)]
    second = [resample_half_indices(37, np.random.default_rng(20260822 + index)).tobytes() for index in range(10)]
    assert first == second


def test_different_seed_changes_resample_without_breaking_half_membership() -> None:
    sequence = np.arange(41)
    first_half, second_half = split_chronological(sequence)
    left = first_half[resample_half_indices(len(first_half), np.random.default_rng(10))]
    right = first_half[resample_half_indices(len(first_half), np.random.default_rng(11))]
    assert left.tobytes() != right.tobytes()
    assert set(left).issubset(set(first_half)) and set(right).issubset(set(first_half))
    assert set(left).isdisjoint(set(second_half)) and set(right).isdisjoint(set(second_half))


def test_odd_n_floor_split_assigns_extra_to_second() -> None:
    first, second = split_chronological(np.arange(9))
    assert first.tolist() == [0, 1, 2, 3]
    assert second.tolist() == [4, 5, 6, 7, 8]


def test_identity_permutation_equals_observed_statistic() -> None:
    matrix = np.asarray([[0.9, 0.2, 0.1], [0.3, 0.8, 0.2], [0.1, 0.4, 0.95]])
    reference = float(np.median(np.diag(matrix)) - np.median(matrix[~np.eye(3, dtype=bool)]))
    value, same, different = permutation_statistic(matrix, (0, 1, 2))
    assert value == pytest.approx(reference)
    assert (same, different) == (3, 6)


def test_every_permutation_has_shape_m_and_m_times_m_minus_one() -> None:
    matrix = np.arange(25, dtype=float).reshape(5, 5)
    shapes = [permutation_statistic(matrix, permutation)[1:] for permutation in itertools.permutations(range(5))]
    assert len(shapes) == 120
    assert set(shapes) == {(5, 20)}


def test_prn_order_relabeling_does_not_change_exact_p_value() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(5, 5))
    order = np.asarray([3, 0, 4, 1, 2])
    reordered = matrix[np.ix_(order, order)]
    assert exact_prn_permutation(matrix)["p_value"] == exact_prn_permutation(reordered)["p_value"]


def test_exact_permutation_matches_brute_force_reference() -> None:
    matrix = np.asarray([[0.8, 0.1, 0.3], [0.2, 0.9, 0.4], [0.5, 0.2, 0.7]])
    observed = np.median(np.diag(matrix)) - np.median(matrix[~np.eye(3, dtype=bool)])
    values = []
    for permutation in itertools.permutations(range(3)):
        same = [matrix[i, permutation[i]] for i in range(3)]
        different = [matrix[i, permutation[j]] for i in range(3) for j in range(3) if i != j]
        values.append(np.median(same) - np.median(different))
    expected = sum(value >= observed - 1e-15 for value in values) / 6
    assert exact_prn_permutation(matrix)["p_value"] == pytest.approx(expected)


def test_perfectly_separated_five_prn_kernels_have_small_exact_p() -> None:
    # Every diagonal entry is strictly larger than every off-diagonal entry.
    # Non-constant off-diagonal geometry avoids the median ties of an identity
    # matrix while retaining perfect nearest-kernel separation.
    matrix = np.asarray([
        [0.9029777641, 0.3588855204, 0.3102742761, 0.0900828760, 0.1200665140],
        [0.3494213782, 0.8932412051, 0.3284913674, 0.3188277715, 0.1871739811],
        [0.1212129707, 0.1113702448, 0.9834335546, 0.1780305224, 0.2018193036],
        [0.2213989408, 0.3982001134, 0.3170647677, 0.9258452509, 0.3955840591],
        [0.0861234793, 0.0640848135, 0.2450158417, 0.0175768032, 0.9028235293],
    ])
    off_diagonal_max = float(np.max(matrix[~np.eye(5, dtype=bool)]))
    assert np.all(np.diag(matrix) > off_diagonal_max)
    result = exact_prn_permutation(matrix)
    assert result["p_value"] == pytest.approx(1 / 120)
    assert result["p_value"] <= 0.01


def test_exchangeable_kernels_are_not_significant() -> None:
    result = exact_prn_permutation(np.ones((5, 5)))
    assert result["p_value"] == 1.0


def test_verifier_help_works_without_pythonpath() -> None:
    repo = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "scripts/verify_bitprobe_stage0a_r0a.py", "--help"],
        cwd=repo, env=environment, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0
    assert "--self-test" in completed.stdout
    assert "--freeze-only" not in completed.stdout
