"""MOSAIC Stage-0B analytic parameter recovery baselines."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mosaic_residual_caf import ResidualCAFResult


@dataclass(frozen=True)
class RecoveryEstimate:
    delay_chips: float
    doppler_hz: float
    relative_power_db: float | None
    relative_phase_rad: float | None
    curvature: float
    observable: bool


def estimate_from_residual_caf(result: ResidualCAFResult, *, observable_floor_fraction: float = 0.05) -> RecoveryEstimate:
    s = np.asarray(result.surface, dtype=float)
    if s.size == 0 or not np.isfinite(s).all() or float(s.max()) <= 0.0:
        return RecoveryEstimate(float("nan"), float("nan"), None, None, 0.0, False)
    peak = float(s.max())
    flat = np.sort(s.ravel())
    second = float(flat[-2]) if flat.size > 1 else 0.0
    curvature = float((peak - second) / peak) if peak else 0.0
    observable = bool(curvature >= observable_floor_fraction)
    return RecoveryEstimate(result.peak_delay_chips, result.peak_doppler_hz, None, None, curvature, observable)


def delay_direction_accuracy(injected: np.ndarray, recovered: np.ndarray) -> float | None:
    inj = np.asarray(injected, dtype=float)
    rec = np.asarray(recovered, dtype=float)
    mask = (inj != 0) & np.isfinite(inj) & np.isfinite(rec)
    if not mask.any():
        return None
    return float(np.mean(np.sign(inj[mask]) == np.sign(rec[mask])))


def median_abs_error(injected: np.ndarray, recovered: np.ndarray, *, minimum_abs: float) -> float | None:
    inj = np.asarray(injected, dtype=float)
    rec = np.asarray(recovered, dtype=float)
    mask = (np.abs(inj) >= minimum_abs) & np.isfinite(inj) & np.isfinite(rec)
    if not mask.any():
        return None
    return float(np.median(np.abs(inj[mask] - rec[mask])))
