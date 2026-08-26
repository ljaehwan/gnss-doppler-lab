import numpy as np
import pytest

from gnss_doppler_lab.clock_centered_geometry import (
    fit_clock_centered_geometry,
)


def _los() -> np.ndarray:
    los = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=float)
    los[-1] /= np.linalg.norm(los[-1])
    return los


def test_clock_centered_score_is_invariant_to_common_bias():
    independent = np.asarray([0.11, 0.42, 0.25, 0.31, 0.17, 0.38])
    original = fit_clock_centered_geometry(_los(), independent)
    shifted = fit_clock_centered_geometry(_los(), independent + 10.0)

    assert shifted.clock_centered_normalized_residual == pytest.approx(
        original.clock_centered_normalized_residual, abs=1e-12
    )
    assert shifted.normalized_residual < original.normalized_residual


def test_constant_delay_contains_no_directional_evidence():
    fit = fit_clock_centered_geometry(_los(), np.full(6, 0.3))

    assert fit.clock_only_bias_chips == pytest.approx(0.3)
    assert fit.clock_centered_normalized_residual == 1.0
    assert fit.directional_coherence == 0.0


def test_exact_los_geometry_has_unit_directional_coherence():
    theta = np.asarray([0.2, -0.1, 0.05, 0.4])
    design = np.column_stack((-_los(), np.ones(6)))
    delays = design @ theta
    fit = fit_clock_centered_geometry(_los(), delays)

    assert fit.clock_centered_normalized_residual == pytest.approx(
        0.0, abs=1e-20
    )
    assert fit.directional_coherence == pytest.approx(1.0)
