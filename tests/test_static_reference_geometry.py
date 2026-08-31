from __future__ import annotations

import numpy as np

from gnss_doppler_lab.static_reference_geometry import (
    clean_only_thresholds,
    partial_f_score,
    persistent_alarm_by_second,
    score_static_reference_geometry,
)


def _los() -> np.ndarray:
    values = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ], dtype=np.float64)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_static_reference_geometry_recovers_displacement_and_ignores_clock() -> None:
    los = _los()
    truth = np.asarray([80.0, -30.0, 15.0])
    delays = -los @ truth + 120.0
    score = score_static_reference_geometry(los, delays)

    assert np.allclose(score.displacement, truth)
    assert np.isclose(score.clock_bias, 120.0)
    assert score.clock_centered_residual < 1e-20
    assert score.partial_f_p_value < 1e-20


def test_common_time_push_has_no_directional_improvement() -> None:
    score = score_static_reference_geometry(_los(), np.full(8, 600.0))

    assert score.displacement_norm < 1e-10
    assert score.clock_centered_residual == 1.0
    assert score.partial_f_p_value == 1.0


def test_partial_f_same_residual_is_less_significant_with_less_support() -> None:
    _, p_eight = partial_f_score(0.1, 8)
    _, p_twelve = partial_f_score(0.1, 12)
    assert p_eight > p_twelve


def test_clean_thresholds_and_wall_clock_persistence() -> None:
    thresholds = clean_only_thresholds(
        [0.9, 0.8, 0.7, 0.6], [1.0, 2.0, 3.0, 4.0]
    )
    assert thresholds["calibration_bin_count"] == 4
    assert 0.6 < thresholds["partial_f_p_alarm_threshold"] < 0.7
    assert 3.0 < thresholds["displacement_alarm_threshold_m"] < 4.0

    persistent = persistent_alarm_by_second(
        {10: True, 11: True, 13: True, 20: True},
        start_second=10,
        end_second=21,
    )
    assert persistent[12] is False
    assert persistent[13] is True
    assert persistent[14] is True
    assert persistent[20] is False
