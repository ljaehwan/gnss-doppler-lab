"""Static pre-attack-reference geometry scoring for GNSS code residuals.

This module deliberately reuses the clock-centered CGC nested model.  The
only change is the signed-delay observable: a per-PRN code residual referenced
to a trusted static interval replaces a prompt-local correlator delay.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np
from scipy.stats import f as f_distribution

from .clock_centered_geometry import fit_clock_centered_geometry


@dataclass(frozen=True)
class StaticReferenceGeometryScore:
    """One support-normalized static reference-geometry score."""

    displacement: np.ndarray
    displacement_norm: float
    clock_bias: float
    clock_centered_residual: float
    directional_coherence: float
    partial_f: float
    partial_f_p_value: float
    prn_count: int
    rank: int


def partial_f_score(clock_centered_residual: float, prn_count: int) -> tuple[float, float]:
    """Return the three-parameter partial-F score and its upper-tail value.

    The returned tail value is a support-normalized ranking variable.  It is
    not interpreted as an exact false-alarm probability because GNSS code
    residuals are neither independent nor guaranteed Gaussian.
    """
    residual = float(clock_centered_residual)
    if not math.isfinite(residual) or not 0.0 <= residual <= 1.0:
        raise ValueError("clock_centered_residual must be finite and in [0, 1]")
    if isinstance(prn_count, bool) or int(prn_count) != prn_count or prn_count < 5:
        raise ValueError("partial-F scoring requires at least five PRNs")
    support = int(prn_count)
    if residual == 0.0:
        return math.inf, 0.0
    statistic = ((1.0 - residual) * (support - 4)) / (3.0 * residual)
    return float(statistic), float(f_distribution.sf(statistic, 3, support - 4))


def score_static_reference_geometry(
    los: np.ndarray,
    signed_code_residuals_m: np.ndarray,
    *,
    displacement_reference_m: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> StaticReferenceGeometryScore:
    """Fit one common displacement plus a nuisance receiver-clock term."""
    fit = fit_clock_centered_geometry(los, signed_code_residuals_m, weights=weights)
    reference = (
        np.zeros(3, dtype=np.float64)
        if displacement_reference_m is None
        else np.asarray(displacement_reference_m, dtype=np.float64)
    )
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("displacement_reference_m must contain three finite values")
    displacement = np.asarray(fit.theta[:3], dtype=np.float64) - reference
    statistic, p_value = partial_f_score(
        fit.clock_centered_normalized_residual, len(signed_code_residuals_m)
    )
    return StaticReferenceGeometryScore(
        displacement=displacement,
        displacement_norm=float(np.linalg.norm(displacement)),
        clock_bias=float(fit.theta[3]),
        clock_centered_residual=float(fit.clock_centered_normalized_residual),
        directional_coherence=float(fit.directional_coherence),
        partial_f=statistic,
        partial_f_p_value=p_value,
        prn_count=int(len(signed_code_residuals_m)),
        rank=int(fit.rank),
    )


def clean_only_thresholds(
    p_values: Iterable[float],
    displacement_norms_m: Iterable[float],
    *,
    p_quantile: float = 0.05,
    displacement_quantile: float = 0.95,
) -> dict[str, float | int]:
    """Freeze a joint lower-p/upper-displacement rule from clean bins only."""
    p = np.asarray(tuple(p_values), dtype=np.float64)
    displacement = np.asarray(tuple(displacement_norms_m), dtype=np.float64)
    if (
        p.ndim != 1
        or displacement.ndim != 1
        or len(p) != len(displacement)
        or not len(p)
        or not np.isfinite(p).all()
        or not np.isfinite(displacement).all()
        or np.any((p < 0.0) | (p > 1.0))
        or np.any(displacement < 0.0)
    ):
        raise ValueError("clean calibration arrays must be paired, finite, and nonempty")
    if not 0.0 < p_quantile < 1.0 or not 0.0 < displacement_quantile < 1.0:
        raise ValueError("calibration quantiles must lie strictly inside (0, 1)")
    return {
        "partial_f_p_alarm_threshold": float(np.quantile(p, p_quantile)),
        "displacement_alarm_threshold_m": float(
            np.quantile(displacement, displacement_quantile)
        ),
        "calibration_bin_count": int(len(p)),
        "p_quantile": float(p_quantile),
        "displacement_quantile": float(displacement_quantile),
    }


def joint_raw_alarm(
    score: StaticReferenceGeometryScore,
    thresholds: Mapping[str, float | int],
) -> bool:
    """Require both significant geometry and nontrivial apparent motion."""
    return bool(
        score.partial_f_p_value <= float(thresholds["partial_f_p_alarm_threshold"])
        and score.displacement_norm
        >= float(thresholds["displacement_alarm_threshold_m"])
    )


def persistent_alarm_by_second(
    raw_alarm_by_second: Mapping[int, bool],
    *,
    start_second: int,
    end_second: int,
    window_seconds: int = 5,
    required_seconds: int = 3,
) -> dict[int, bool]:
    """Apply a causal wall-clock persistence rule; unavailable seconds are false."""
    if (
        isinstance(start_second, bool)
        or isinstance(end_second, bool)
        or int(start_second) != start_second
        or int(end_second) != end_second
        or start_second >= end_second
    ):
        raise ValueError("persistence interval must be increasing integer seconds")
    if (
        isinstance(window_seconds, bool)
        or isinstance(required_seconds, bool)
        or int(window_seconds) != window_seconds
        or int(required_seconds) != required_seconds
        or not 1 <= required_seconds <= window_seconds
    ):
        raise ValueError("invalid persistence window or requirement")
    start, end = int(start_second), int(end_second)
    window, required = int(window_seconds), int(required_seconds)
    result: dict[int, bool] = {}
    for second in range(start, end):
        left = second - window + 1
        positives = sum(
            bool(raw_alarm_by_second.get(candidate, False))
            for candidate in range(left, second + 1)
        )
        result[second] = positives >= required
    return result
