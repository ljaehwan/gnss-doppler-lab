"""Geometry score that treats common receiver clock bias as a nuisance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .correlator_geometry import EPSILON, fit_common_geometry


@dataclass(frozen=True)
class ClockCenteredGeometryFit:
    """Full LOS-plus-clock fit with both zero- and clock-referenced scores."""

    theta: np.ndarray
    predicted_delays_chips: np.ndarray
    normalized_residual: float
    coherence: float
    rank: int
    clock_only_bias_chips: float
    clock_centered_normalized_residual: float
    directional_coherence: float


def fit_clock_centered_geometry(
    los_enu: np.ndarray,
    signed_delays_chips: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> ClockCenteredGeometryFit:
    """Fit LOS geometry and compare it with an intercept-only clock null.

    directional_coherence is the partial R-squared contributed by the three
    LOS columns beyond a common clock intercept. Consequently, adding the same
    finite delay to every satellite cannot improve this score.

    The legacy zero-referenced residual is retained in the returned record so
    that the normalization failure can be audited without modifying the
    previously frozen and hash-pinned geometry implementation.
    """
    base = fit_common_geometry(los_enu, signed_delays_chips, weights=weights)
    delays = np.asarray(signed_delays_chips, dtype=np.float64)
    weight = (
        np.ones(len(delays), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    clock_only_bias = float(np.sum(weight * delays) / np.sum(weight))
    clock_centered_energy = float(
        np.sum(weight * (delays - clock_only_bias) ** 2)
    )
    residual_energy = float(
        np.sum(weight * (delays - base.predicted_delays_chips) ** 2)
    )
    centered_residual = (
        1.0
        if clock_centered_energy <= EPSILON
        else min(1.0, max(0.0, residual_energy / clock_centered_energy))
    )
    return ClockCenteredGeometryFit(
        theta=base.theta,
        predicted_delays_chips=base.predicted_delays_chips,
        normalized_residual=base.normalized_residual,
        coherence=base.coherence,
        rank=base.rank,
        clock_only_bias_chips=clock_only_bias,
        clock_centered_normalized_residual=centered_residual,
        directional_coherence=1.0 - centered_residual,
    )
