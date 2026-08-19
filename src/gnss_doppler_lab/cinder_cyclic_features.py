"""Frozen chip-synchronous C4 features for CINDER Stage-0A.

The implementation deliberately exposes the physical operations separately so
that invariance, alignment, and leakage controls can exercise the same code as
the clean experiment.  No PRN identifier is returned as a feature.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .mosaic_raw_recorrelation import read_ishort_complex_window, receiver_l1ca_code


FRACTIONAL_CHIP_COORDS = np.asarray((0.125, 0.375, 0.625, 0.875), dtype=np.float64)
CYCLIC_FREQUENCIES_CYCLES_PER_CHIP = np.asarray((0.0, 0.125, 0.25), dtype=np.float64)
LAG_TUPLES = ((0, 0, 0), (1, 0, 1), (1, 1, 2), (2, 1, 3), (4, 2, 4))
VARIANCE_FLOOR = 1e-10


@dataclass(frozen=True)
class ResamplingAudit:
    source_sample_count: int
    output_chip_count: int
    output_samples_per_chip: int
    interpolation: str
    antialias_filter: str
    group_delay_source_samples: float
    out_of_bounds_queries: int
    maximum_fractional_source_error: float


def fourth_order_cyclic_cumulant(
    x: np.ndarray,
    alpha: float,
    lags: tuple[int, int, int],
    *,
    mask: np.ndarray | None = None,
) -> complex:
    """Estimate conjugate-balanced fourth cyclic cumulant.

    The pair terms are removed explicitly.  The same cyclic exponential is
    applied to the fourth moment and to the product of the stationary pair
    moments.  This convention reduces to the standard fourth cumulant at
    alpha=0 and is sufficient for the fixed Stage-0A cyclic grid.
    """
    z = np.asarray(x, dtype=np.complex128).reshape(-1)
    if z.size < 16 or not np.isfinite(z).all():
        raise ValueError("C4 input must contain at least 16 finite complex samples")
    l1, l2, l3 = (int(v) for v in lags)
    if min(l1, l2, l3) < 0:
        raise ValueError("lags must be non-negative")
    start = max(l1, l2, l3)
    t = np.arange(start, z.size)
    if mask is not None:
        supplied = np.asarray(mask).reshape(-1)
        if supplied.dtype == bool:
            t = t[supplied[t]]
        else:
            t = supplied[supplied >= start].astype(np.int64, copy=False)
    if t.size < 8:
        return 0.0j
    a, b, c, d = z[t], np.conj(z[t - l1]), z[t - l2], np.conj(z[t - l3])
    centered = a * b * c * d
    centered -= np.mean(a * b) * np.mean(c * d)
    centered -= np.mean(a * c) * np.mean(b * d)
    centered -= np.mean(a * d) * np.mean(b * c)
    phase = np.exp(-2j * np.pi * float(alpha) * t)
    return complex(np.mean(centered * phase))


def transition_classes(code: np.ndarray) -> np.ndarray:
    """Encode previous/current/next binary chip signs into eight classes."""
    c = np.asarray(code).reshape(-1)
    bits = (c > 0).astype(np.uint8)
    return (
        4 * np.roll(bits, 1) + 2 * bits + np.roll(bits, -1)
    ).astype(np.uint8)


def c4_vector(
    chip_waveforms: np.ndarray,
    code: np.ndarray,
    *,
    alphas: Iterable[float] = CYCLIC_FREQUENCIES_CYCLES_PER_CHIP,
    lag_tuples: Iterable[tuple[int, int, int]] = LAG_TUPLES,
    variance_floor: float = VARIANCE_FLOOR,
) -> np.ndarray:
    """Return the fixed complex C4 vector after code-pattern equalization."""
    wave = np.asarray(chip_waveforms, dtype=np.complex128)
    if wave.ndim != 2 or wave.shape[1] != 4:
        raise ValueError("chip waveforms must have shape (chips, 4)")
    ca = np.resize(np.asarray(code).reshape(-1), wave.shape[0])
    residual = wave - np.mean(wave, axis=1, keepdims=True)
    # Frozen edge and inner contrasts preserve fractional-chip pulse shape.
    channels = np.column_stack((residual[:, 0] - residual[:, 3], residual[:, 1] - residual[:, 2]))
    rms = float(np.sqrt(np.mean(np.abs(channels) ** 2)))
    alpha_grid = tuple(float(v) for v in alphas)
    lag_grid = tuple(tuple(int(q) for q in v) for v in lag_tuples)
    if not np.isfinite(rms) or rms <= variance_floor:
        return np.zeros(2 * len(alpha_grid) * len(lag_grid), dtype=np.complex128)
    channels = channels / rms
    classes = transition_classes(ca)
    class_indices = [np.flatnonzero(classes == cls) for cls in range(8)]
    output: list[complex] = []
    for channel in channels.T:
        for alpha in alpha_grid:
            for lags in lag_grid:
                estimates = [
                    fourth_order_cyclic_cumulant(channel, alpha, lags, mask=class_indices[cls])
                    for cls in range(8)
                ]
                output.append(complex(np.mean(estimates)))
    return np.asarray(output, dtype=np.complex128)


def hermitian_projective_compact(v: np.ndarray, *, epsilon: float = VARIANCE_FLOOR) -> np.ndarray:
    """Compact diagonal + first-upper-diagonal of vv^H/(||v||^2+eps)."""
    z = np.asarray(v, dtype=np.complex128).reshape(-1)
    denom = float(np.vdot(z, z).real) + float(epsilon)
    diagonal = np.abs(z) ** 2 / denom
    adjacent = z[:-1] * np.conj(z[1:]) / denom
    return np.concatenate((diagonal, adjacent.real, adjacent.imag)).astype(np.float64)


def second_order_feature(chip_waveforms: np.ndarray, *, epsilon: float = VARIANCE_FLOOR) -> np.ndarray:
    wave = np.asarray(chip_waveforms, dtype=np.complex128)
    residual = wave - np.mean(wave, axis=1, keepdims=True)
    cov = residual.conj().T @ residual / max(len(residual), 1)
    denom = float(np.trace(cov).real) + epsilon
    cov /= denom
    iu = np.triu_indices(4)
    vals = cov[iu]
    return np.concatenate((vals.real, vals.imag)).astype(np.float64)


def prompt_scattering_feature(prompt: np.ndarray, scales: Iterable[int] = (1, 2, 4, 8, 16, 32)) -> np.ndarray:
    """Frozen phase-innovation multiscale diagnostic (not fused into primary)."""
    p = np.asarray(prompt, dtype=np.complex128).reshape(-1)
    unit = p / np.maximum(np.abs(p), 1e-12)
    out: list[float] = []
    for scale in scales:
        if p.size <= scale:
            out.extend((0.0, 0.0))
            continue
        innovation = unit[scale:] * np.conj(unit[:-scale])
        out.extend((float(np.mean(np.abs(np.angle(innovation)))), float(np.std(np.angle(innovation)))))
    return np.asarray(out, dtype=np.float64)


def fractional_chip_resample_records(
    raw_path: str | Path,
    records: np.ndarray,
    prn: int,
    *,
    gain: float = 1.0,
    phase_rad: float = 0.0,
    nav_sign: int = 1,
    code_override: np.ndarray | None = None,
) -> tuple[np.ndarray, ResamplingAudit]:
    """Read a bounded raw span and linearly sample four points per aligned chip.

    The source rates (5 and 25 Msps) are both above the requested 4.092 Msps
    output grid, so linear band-limited local interpolation does not upsample.
    The frozen two-tap triangular kernel has zero centered group delay.
    """
    if not len(records):
        raise ValueError("empty TRACE record selection")
    starts = records["raw_interval_start_sample"].astype(np.int64)
    ends = records["raw_interval_end_sample"].astype(np.int64)
    if np.any(ends <= starts) or np.any(starts[1:] != ends[:-1]):
        raise ValueError("TRACE records are not a contiguous raw span")
    span_start, span_end = int(starts[0]), int(ends[-1])
    iq = read_ishort_complex_window(raw_path, span_start, span_end - span_start).astype(np.complex64)
    iq *= np.complex64(float(gain) * int(nav_sign) * np.exp(1j * float(phase_rad)))
    chip_grid = (np.arange(1023, dtype=np.float64)[:, None] + FRACTIONAL_CHIP_COORDS[None, :]).reshape(-1)
    code = receiver_l1ca_code(prn) if code_override is None else np.asarray(code_override, dtype=np.float64)
    if code.shape != (1023,):
        raise ValueError("code override must have 1023 chips")
    output = np.empty((len(records) * 1023, 4), dtype=np.complex64)
    out_of_bounds = 0
    maximum_fractional_error = 0.0
    for row_index, record in enumerate(records):
        step = float(record["action_used_code_phase_step_chips_per_sample"])
        residual = float(record["action_used_residual_code_phase_chips"])
        local = (chip_grid + residual) / step
        lo = np.floor(local).astype(np.int64)
        frac = local - lo
        count = int(ends[row_index] - starts[row_index])
        bad = (lo < 0) | (lo + 1 >= count)
        out_of_bounds += int(bad.sum())
        lo = np.clip(lo, 0, max(count - 2, 0))
        frac = np.where(bad, np.clip(frac, 0.0, 1.0), frac)
        offset = int(starts[row_index] - span_start)
        values = iq[offset + lo] * (1.0 - frac) + iq[offset + lo + 1] * frac
        carrier_phase = (
            float(record["action_used_residual_carrier_phase_rad"])
            + float(record["action_used_carrier_phase_step_rad_per_sample"]) * local
        )
        values *= np.exp(-1j * carrier_phase)
        values *= np.repeat(code, 4)
        output[row_index * 1023:(row_index + 1) * 1023] = values.reshape(1023, 4)
        maximum_fractional_error = max(maximum_fractional_error, float(np.max(np.abs(local - (lo + frac)))))
    return output, ResamplingAudit(
        source_sample_count=span_end - span_start,
        output_chip_count=len(output),
        output_samples_per_chip=4,
        interpolation="centered two-tap linear (triangular) interpolation",
        antialias_filter="source is already band-limited GNSS complex baseband; 5/25 Msps to 4.092 Msps query grid, no additional decimation FIR",
        group_delay_source_samples=0.0,
        out_of_bounds_queries=out_of_bounds,
        maximum_fractional_source_error=maximum_fractional_error,
    )
