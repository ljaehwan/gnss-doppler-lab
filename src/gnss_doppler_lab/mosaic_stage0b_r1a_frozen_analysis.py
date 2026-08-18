"""Frozen-policy scientific finalization helpers for MOSAIC Stage-0B R1a.

This module operates on retained tap-domain evidence only.  It contains no IQ
injection or receiver-replay entry point.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .mosaic_stage0b_r1_execution_metrics import bic


BOOTSTRAP_SEED = 20260818
BOOTSTRAP_RESAMPLES = 10_000
DELAY_GRID = np.round(np.arange(-0.35, 0.350001, 0.025), 12)
DOPPLER_GRID = np.arange(-75.0, 75.0001, 5.0)


def strong_resolvable(rho_db: float, delay_chips: float, doppler_hz: float) -> bool:
    return bool(rho_db >= -6 and (abs(delay_chips) >= 0.10 or abs(doppler_hz) >= 25))


def physics_recovered(
    requested_delay: float,
    recovered_delay: float,
    requested_doppler: float,
    recovered_doppler: float,
) -> bool:
    return bool(
        abs(recovered_delay - requested_delay) <= 0.05
        and abs(recovered_doppler - requested_doppler) <= 10.0
    )


def is_collapsed(delay_chips: float, doppler_hz: float) -> bool:
    return float(delay_chips) == 0.0 and float(doppler_hz) == 0.0


def target_nontarget_difference(target_delta_bic: float, non_target_delta_bic: Iterable[float]) -> float:
    values = np.asarray(list(non_target_delta_bic), dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("non-target scores must be finite and non-empty")
    return float(target_delta_bic) - float(np.median(values))


def paired_bootstrap_ci(
    values: Iterable[float], *, seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float, float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    for start in range(0, resamples, 1000):
        count = min(1000, resamples - start)
        indices = rng.integers(0, len(x), size=(count, len(x)))
        means[start : start + count] = x[indices].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_abs(x: Iterable[float], y: Iterable[float]) -> float:
    a, b = np.asarray(list(x), float), np.asarray(list(y), float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return math.nan
    ra, rb = _average_ranks(a[mask]), _average_ranks(b[mask])
    if np.std(ra) == 0 or np.std(rb) == 0:
        return math.nan
    return float(abs(np.corrcoef(ra, rb)[0, 1]))


def sign_accuracy(requested: Iterable[float], recovered: Iterable[float]) -> float | None:
    a, b = np.asarray(list(requested), float), np.asarray(list(recovered), float)
    mask = (a != 0) & np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.sign(a[mask]) == np.sign(b[mask]))) if mask.any() else None


def median_abs_error(requested: Iterable[float], recovered: Iterable[float]) -> float | None:
    a, b = np.asarray(list(requested), float), np.asarray(list(recovered), float)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.median(np.abs(a[mask] - b[mask]))) if mask.any() else None


def gain_matched_control(authentic: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, complex]:
    """Scale clean taps with one scalar; shape and relative phase are unchanged."""
    a = np.asarray(authentic, np.complex128)
    y = np.asarray(observed, np.complex128)
    auth_rms = float(np.sqrt(np.mean(np.abs(a) ** 2)))
    observed_rms = float(np.sqrt(np.mean(np.abs(y) ** 2)))
    if auth_rms <= 0:
        raise ValueError("clean reference tap RMS is zero")
    scalar = complex(observed_rms / auth_rms, 0.0)
    return a * scalar, scalar


def deterministic_awgn_control(
    authentic: np.ndarray, residual_rms: float, prn: int, *, seed_base: int = BOOTSTRAP_SEED
) -> np.ndarray:
    """Add exactly RMS-normalized deterministic circular complex Gaussian noise."""
    a = np.asarray(authentic, np.complex128)
    rng = np.random.default_rng(seed_base + int(prn))
    noise = rng.standard_normal(a.shape) + 1j * rng.standard_normal(a.shape)
    noise_rms = float(np.sqrt(np.mean(np.abs(noise) ** 2)))
    if noise_rms == 0:
        raise ValueError("degenerate AWGN draw")
    return a + noise * (float(residual_rms) / noise_rms)


def score_tap_arrays(
    authentic: np.ndarray,
    observed: np.ndarray,
    starts: np.ndarray,
    tap_offsets_chips: np.ndarray,
    interval_start: int,
    sample_rate_hz: int,
) -> dict[str, float | int]:
    """Apply the frozen R1 CAF grid and complex-parameter BIC formula."""
    auth = np.asarray(authentic, np.complex128)
    obs = np.asarray(observed, np.complex128)
    starts = np.asarray(starts, np.int64)
    if auth.shape != obs.shape or auth.shape[0] != len(starts):
        raise ValueError("aligned tap shape mismatch")
    y = obs.reshape(-1)
    a = auth.reshape(-1, 1)
    c0, *_ = np.linalg.lstsq(a, y, rcond=None)
    residual = (y - a @ c0).reshape(len(starts), auth.shape[1])
    rss0 = float(np.vdot(residual, residual).real)
    energy_floor = float(np.finfo(np.float64).eps * max(np.vdot(y, y).real, 1.0))
    rss0 = max(rss0, energy_floor)
    nobs = 2 * y.size
    b0 = bic(rss0, nobs, 2)
    t = (starts - int(interval_start)) / float(sample_rate_hz)
    carrier = np.exp(1j * 2 * np.pi * np.outer(t, DOPPLER_GRID))
    gram = complex((a.conj().T @ a)[0, 0])
    best: tuple[float, float, float, float, float] | None = None
    for delay in DELAY_GRID:
        spatial = np.maximum(1 - np.abs(np.asarray(tap_offsets_chips) - delay), 0.0)
        templates = carrier[:, :, None] * spatial[None, None, :]
        flat = templates.transpose(0, 2, 1).reshape(y.size, len(DOPPLER_GRID))
        projection = a.conj().T @ flat
        orthogonal = flat - a @ (projection / gram)
        denominator = np.sum(np.abs(orthogonal) ** 2, axis=0)
        numerator = np.abs(orthogonal.conj().T @ residual.reshape(-1)) ** 2
        rss1 = np.maximum(rss0 - numerator / np.maximum(denominator, 1e-300), energy_floor)
        b1 = nobs * np.log(rss1 / nobs) + 4 * np.log(nobs)
        delta = b0 - b1
        index = int(np.argmax(delta))
        candidate = (float(delta[index]), float(delay), float(DOPPLER_GRID[index]), float(rss1[index]), float(b1[index]))
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    return {
        "epochs": len(starts),
        "rss_h0": rss0,
        "rss_h1": best[3],
        "bic_h0": b0,
        "bic_h1": best[4],
        "delta_bic": best[0],
        "recovered_delay_chips": best[1],
        "recovered_doppler_hz": best[2],
        "tap_rms": float(np.sqrt(np.mean(np.abs(obs) ** 2))),
    }


def four_prn_success(recovered_count: int) -> bool:
    return int(recovered_count) >= 3


def decide_verdict(gates: dict[str, object]) -> str:
    """Compute an allowed verdict from explicit frozen gate booleans."""
    if not gates.get("integrity_pass", False):
        return "INCONCLUSIVE_RESULT_INTEGRITY_FAILURE"
    if not gates.get("retained_evidence_complete", False):
        return "INCONCLUSIVE_MISSING_RETAINED_EVIDENCE"
    if not gates.get("four_prn_numeric_criterion_defined", False):
        return "INCONCLUSIVE_PREREG_GATE_UNDERSPECIFIED"
    if not gates.get("single_prn_physics_pass", False):
        return "NO_GO_MOSAIC_SINGLE_PRN_PHYSICS"
    if not gates.get("control_separation_pass", False):
        return "NO_GO_MOSAIC_CONTROL_SEPARATION"
    if not gates.get("multi_prn_recovery_pass", False):
        return "NO_GO_MOSAIC_MULTI_PRN_RECOVERY"
    if not gates.get("physical_hypothesis_pass", False):
        return "NO_GO_MOSAIC_PHYSICAL_HYPOTHESIS"
    return "GO_FOR_MOSAIC_STAGE1"
