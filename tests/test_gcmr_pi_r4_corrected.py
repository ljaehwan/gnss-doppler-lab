import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab import gcmr_pi_r4_corrected as r4


@dataclass(frozen=True)
class NormalRow:
    event_index: int
    row_index: int
    cn0: float
    direction: np.ndarray
    role: str = "event_calibration"


def pool():
    return [
        NormalRow(0, 0, 40.0, np.array([1.0, 0.0, 0.0])),
        NormalRow(1, 0, 40.0, np.array([0.0, 1.0, 0.0])),
        NormalRow(2, 0, 40.0, np.array([0.0, 0.0, 1.0])),
        NormalRow(3, 0, 52.0, np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)),
    ]


def test_warmup_masks_have_exact_start_guard_and_onset_boundaries():
    startup, pre, transition, post = r4.warmup_period_masks(np.array([29.999, 30.0, 109.999, 110.0, 129.999, 130.0]))
    assert startup.tolist() == [True, False, False, False, False, False]
    assert pre.tolist() == [False, True, True, False, False, False]
    assert transition.tolist() == [False, False, False, True, True, False]
    assert post.tolist() == [False, False, False, False, False, True]


def test_relation_only_threshold_is_direct_quantile_of_joint_normal_score_not_sum_of_thresholds():
    normal = {
        "s_common": np.array([0.0, 10.0, 20.0]),
        "n_eff": np.array([2.0, 1.0, 0.0]),
        "s_pair": np.array([2.0, 1.0, 0.0]),
        "energy": np.array([0.0, 1.0, 2.0]),
    }
    location = {key: 0.0 for key in normal}
    scale = {key: 1.0 for key in normal}
    scores = r4.component_scores(normal, location, scale)
    thresholds = r4.direct_normal_thresholds(scores)
    assert thresholds["RelationOnly"]["q99"] == pytest.approx(np.quantile(np.array([4.0, 12.0, 20.0]), 0.99))
    assert thresholds["RelationOnly"]["q99"] != pytest.approx(
        np.quantile(normal["s_common"] + normal["n_eff"], 0.99) + np.quantile(normal["s_pair"], 0.99)
    )


def test_direction_pool_rejects_attack_and_non_normal_roles():
    with pytest.raises(r4.FailClosedError, match="normal event_calibration"):
        r4.validate_direction_pool([NormalRow(0, 0, 40.0, np.array([1., 0., 0.]), role="attack")])


def test_destroyed_rows_use_distinct_normal_event_row_sources_and_preserve_norms():
    original = np.array([[3.0, 4.0, 0.0], [0.0, 5.0, 12.0], [8.0, 0.0, 6.0]])
    destroyed, provenance = r4.destroy_innovation_directions(original, np.array([40.0, 40.0, 40.0]), pool(), seed=44)
    assert np.allclose(np.linalg.norm(destroyed, axis=1), np.linalg.norm(original, axis=1))
    source_keys = [(item["source_event_index"], item["source_row_index"]) for item in provenance]
    assert len(source_keys) == len(set(source_keys))
    assert all(item["source_role"] == "event_calibration" for item in provenance)
    assert all(item["used_global_pool"] is False for item in provenance)


def test_seed_is_reproducible_but_a_different_seed_changes_assignments():
    original = np.array([[3.0, 4.0, 0.0], [0.0, 5.0, 12.0], [8.0, 0.0, 6.0]])
    a, pa = r4.destroy_innovation_directions(original, np.array([40.0, 40.0, 40.0]), pool(), seed=9)
    b, pb = r4.destroy_innovation_directions(original, np.array([40.0, 40.0, 40.0]), pool(), seed=9)
    c, pc = r4.destroy_innovation_directions(original, np.array([40.0, 40.0, 40.0]), pool(), seed=10)
    assert np.array_equal(a, b) and pa == pb
    assert pa != pc or not np.array_equal(a, c)


def test_empty_or_nonfinite_innovation_fails_closed():
    with pytest.raises(r4.FailClosedError, match="innovation"):
        r4.validate_reconstructable_innovations(np.array([[np.nan, 1.0, 2.0]]))
