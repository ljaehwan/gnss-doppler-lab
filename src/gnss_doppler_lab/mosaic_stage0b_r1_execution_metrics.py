"""Frozen result metrics for MOSAIC Stage-0B R1 execution."""
from __future__ import annotations

import math
import numpy as np


BOOTSTRAP_SEED = 20260818
BOOTSTRAP_RESAMPLES = 10_000


def raised_cosine_envelope(t: np.ndarray, duration: float) -> np.ndarray:
    x = np.asarray(t, dtype=np.float64)
    if duration <= 10:
        raise ValueError("interval must extend beyond 10 seconds")
    e = np.zeros_like(x)
    ramp = (x >= 2) & (x < 4)
    e[ramp] = .5 * (1 - np.cos(np.pi * (x[ramp] - 2) / 2))
    e[(x >= 4) & (x < 10)] = 1
    release = (x >= 10) & (x <= duration)
    e[release] = .5 * (1 + np.cos(np.pi * (x[release] - 10) / (duration - 10)))
    return e


def raised_cosine_integral(t: np.ndarray, duration: float) -> np.ndarray:
    """Integral from zero to t in seconds, used for continuous Doppler phase."""
    x = np.asarray(t, dtype=np.float64)
    out = np.zeros_like(x)
    ramp = (x >= 2) & (x < 4)
    u = x[ramp] - 2
    out[ramp] = .5 * u - np.sin(np.pi * u / 2) / np.pi
    hold = (x >= 4) & (x < 10)
    out[hold] = 1 + (x[hold] - 4)
    release = x >= 10
    u = np.minimum(x[release], duration) - 10
    width = duration - 10
    out[release] = 7 + .5 * u + width * np.sin(np.pi * u / width) / (2 * np.pi)
    return out


def bic(rss: float, observations: int, parameters: int) -> float:
    if rss <= 0 or observations <= parameters:
        raise ValueError("invalid BIC inputs")
    return float(observations * math.log(rss / observations) + parameters * math.log(observations))


def fit_complex_models(y: np.ndarray, authentic: np.ndarray, second_sources: np.ndarray) -> dict[str, float | np.ndarray]:
    yy = np.asarray(y, np.complex128).reshape(-1)
    a0 = np.asarray(authentic, np.complex128).reshape(-1, 1)
    a1 = np.asarray(second_sources, np.complex128)
    if a1.ndim == 1:
        a1 = a1[:, None]
    if len(yy) != len(a0) or len(yy) != len(a1):
        raise ValueError("model row mismatch")
    c0, *_ = np.linalg.lstsq(a0, yy, rcond=None)
    full = np.column_stack([a0, a1])
    c1, *_ = np.linalg.lstsq(full, yy, rcond=None)
    r0 = yy - a0 @ c0; r1 = yy - full @ c1
    energy_floor = float(np.finfo(np.float64).eps * max(np.vdot(yy, yy).real, 1.0))
    rss0 = max(float(np.vdot(r0, r0).real), energy_floor)
    rss1 = max(float(np.vdot(r1, r1).real), energy_floor)
    n = 2 * len(yy)
    # Each complex coefficient contributes two real parameters.
    b0 = bic(rss0, n, 2)
    b1 = bic(rss1, n, 2 * full.shape[1])
    return {"rss_h0": rss0, "rss_h1": rss1, "bic_h0": b0, "bic_h1": b1,
            "delta_bic": b0 - b1, "coefficients_h0": c0, "coefficients_h1": c1}


def paired_bootstrap_ci(differences: np.ndarray, *, seed: int = BOOTSTRAP_SEED,
                        resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        size = min(1000, resamples - start)
        draw = rng.integers(0, len(values), size=(size, len(values)))
        means[start:start + size] = values[draw].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def spearman_abs(x: np.ndarray, y: np.ndarray) -> float:
    a, b = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        out = np.empty(len(v), float); out[order] = np.arange(len(v), dtype=float)
        return out
    return float(abs(np.corrcoef(ranks(a[mask]), ranks(b[mask]))[0, 1]))


def sign_accuracy(requested: np.ndarray, recovered: np.ndarray) -> float | None:
    a, b = np.asarray(requested, float), np.asarray(recovered, float)
    mask = (a != 0) & np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.sign(a[mask]) == np.sign(b[mask]))) if mask.any() else None


def median_abs_error(requested: np.ndarray, recovered: np.ndarray) -> float | None:
    a, b = np.asarray(requested, float), np.asarray(recovered, float)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.median(np.abs(a[mask] - b[mask]))) if mask.any() else None


def strong_resolvable(rho_db: float, delay_chips: float, doppler_hz: float) -> bool:
    return bool(rho_db >= -6 and (abs(delay_chips) >= .10 or abs(doppler_hz) >= 25))


def collapsed(delay_chips: float, doppler_hz: float) -> bool:
    return delay_chips == 0 and doppler_hz == 0
