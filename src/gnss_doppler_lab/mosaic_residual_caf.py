"""Residual projection and re-correlation for MOSAIC Stage-0B."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mosaic_iq_injector import sampled_prn_replica


@dataclass(frozen=True)
class ResidualCAFResult:
    delay_grid_chips: np.ndarray
    doppler_grid_hz: np.ndarray
    surface: np.ndarray
    peak_delay_chips: float
    peak_doppler_hz: float
    peak_value: float


def complex_least_squares(y: np.ndarray, design: np.ndarray, *, ridge: float = 1e-9) -> np.ndarray:
    yy = np.asarray(y, dtype=np.complex128)
    aa = np.asarray(design, dtype=np.complex128)
    if aa.ndim == 1:
        aa = aa[:, None]
    gram = aa.conj().T @ aa + float(ridge) * np.eye(aa.shape[1])
    return np.linalg.solve(gram, aa.conj().T @ yy)


def h0_residual(y: np.ndarray, replicas: np.ndarray, *, ridge: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    aa = np.asarray(replicas, dtype=np.complex128)
    if aa.ndim == 1:
        aa = aa[:, None]
    alpha = complex_least_squares(y, aa, ridge=ridge)
    return np.asarray(y, dtype=np.complex128) - aa @ alpha, alpha


def residual_caf(
    residual: np.ndarray,
    prn: str | int,
    sample_rate_hz: float,
    delay_grid_chips: np.ndarray,
    doppler_grid_hz: np.ndarray,
    *,
    nav_bits: np.ndarray | float = 1.0,
    base_code_phase_chips: float = 0.0,
) -> ResidualCAFResult:
    e = np.asarray(residual, dtype=np.complex128)
    surface = np.empty((len(doppler_grid_hz), len(delay_grid_chips)), dtype=np.float64)
    for i, doppler in enumerate(doppler_grid_hz):
        for j, delay in enumerate(delay_grid_chips):
            replica = sampled_prn_replica(
                prn,
                sample_rate_hz,
                e.size,
                code_phase_chips=base_code_phase_chips + float(delay),
                doppler_hz=float(doppler),
                nav_bits=nav_bits,
            )
            surface[i, j] = float(abs(np.vdot(replica, e)) ** 2)
    peak = int(np.argmax(surface)) if surface.size else 0
    pi, pj = np.unravel_index(peak, surface.shape) if surface.size else (0, 0)
    return ResidualCAFResult(
        np.asarray(delay_grid_chips, dtype=float),
        np.asarray(doppler_grid_hz, dtype=float),
        surface,
        float(delay_grid_chips[pj]) if len(delay_grid_chips) else float("nan"),
        float(doppler_grid_hz[pi]) if len(doppler_grid_hz) else float("nan"),
        float(surface[pi, pj]) if surface.size else 0.0,
    )
