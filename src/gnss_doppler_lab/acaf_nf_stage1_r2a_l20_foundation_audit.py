"""Clean-only primitives for the Stage-1 R2a L20 foundation audit.

The module deliberately owns no recording discovery.  Callers must provide the
authenticated cleanStatic paths explicitly.  Attack recordings and fitting are
outside this audit's contract.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from gnss_doppler_lab.acaf_nf_stage0_r13_reconstruction import code_replica, carrier_wipeoff
from gnss_doppler_lab.acaf_nf_stage0_r14_doppler_validation import normalized_noncoherent_power

FS_HZ = 25_000_000.0
SUPPORT_SAMPLES = 25_000
DELAY_GRID_CHIPS = np.arange(-1.0, 1.0001, 0.125)
DOPPLER_GRID_HZ = np.arange(-250.0, 250.0001, 50.0)
CENTER_INDEX = (5, 8)
AUTHORIZED_RECORDING = "cleanStatic"
PROHIBITED_TOKENS = ("ds3", "ds4", "ds7", "ds8", "attack")


@dataclass(frozen=True)
class State:
    """The state applied to one fixed 1-ms raw support."""

    channel: int
    prn: int
    tracker_row: int
    state_row: int
    raw_start_sample: int
    code_freq_chips: float
    carrier_doppler_hz: float
    aux1: float
    prompt_i: float
    prompt_q: float
    cn0_db_hz: float
    carrier_lock: float


def clean_only_guard(recording: str, paths: Iterable[str | Path] = ()) -> None:
    """Fail closed before any path is opened."""

    if recording != AUTHORIZED_RECORDING:
        raise ValueError("R2a foundation audit accepts cleanStatic only")
    for path in paths:
        lowered = str(path).lower()
        if any(token in lowered for token in PROHIBITED_TOKENS):
            raise ValueError(f"prohibited non-clean input path: {path}")


def surface_sha256(surface: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(surface).view(np.uint8)).hexdigest()


def complex_caf_surface(
    iq: np.ndarray,
    state: State,
    *,
    carrier_sign: int = -1,
    state_doppler_adjustment_hz: float = 0.0,
) -> np.ndarray:
    """R1.4-equivalent complex CAF on the frozen 11x17 grid."""

    values = np.asarray(iq, dtype=np.complex128)
    if values.ndim != 1 or values.size != SUPPORT_SAMPLES:
        raise ValueError("one complex 25,000-sample support is required")
    if carrier_sign not in (-1, 1):
        raise ValueError("carrier sign must be +/-1")
    replicas = np.asarray([
        code_replica(
            state.prn,
            values.size,
            FS_HZ,
            state.code_freq_chips,
            state.aux1,
            -1,
            float(delay),
            replica_direction=1,
        )[0]
        for delay in DELAY_GRID_CHIPS
    ], dtype=np.float64)
    wipes = np.asarray([
        carrier_wipeoff(
            values.size,
            FS_HZ,
            state.carrier_doppler_hz + state_doppler_adjustment_hz,
            float(doppler),
            carrier_sign,
        )[0]
        for doppler in DOPPLER_GRID_HZ
    ], dtype=np.complex128)
    return (wipes * values[None, :]) @ replicas.T


def r14_l20_aggregate(surfaces: Sequence[np.ndarray]) -> np.ndarray:
    """R1.4 primary surface: mean normalized noncoherent power."""

    if len(surfaces) != 20:
        raise ValueError("L20 requires exactly 20 constituent surfaces")
    return normalized_noncoherent_power(surfaces)


def r2_l20_aggregate(surfaces: Sequence[np.ndarray]) -> np.ndarray:
    """R2 equation written independently for a numerical-equivalence audit."""

    values = np.asarray(surfaces)
    if values.shape[0] != 20 or values.ndim != 3 or not np.iscomplexobj(values):
        raise ValueError("R2 L20 requires 20 same-shaped complex surfaces")
    power = values.real * values.real + values.imag * values.imag
    denominator = np.sum(power, axis=(1, 2), keepdims=True) + 1e-15
    return np.mean(power / denominator, axis=0)


def score_power_surface(power: np.ndarray) -> dict[str, float | bool | int]:
    """Score a nonnegative power surface without changing its peak."""

    values = np.asarray(power, dtype=np.float64)
    if values.shape != (len(DOPPLER_GRID_HZ), len(DELAY_GRID_CHIPS)):
        raise ValueError("unexpected CAF grid shape")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("finite nonnegative power surface required")
    di, ci = np.unravel_index(int(np.argmax(values)), values.shape)
    center = float(np.sqrt(values[CENTER_INDEX]))
    peak = float(np.sqrt(values[di, ci]))
    return {
        "peak_doppler_index": int(di),
        "peak_delay_index": int(ci),
        "peak_doppler_offset_hz": float(DOPPLER_GRID_HZ[di]),
        "peak_delay_offset_chips": float(DELAY_GRID_CHIPS[ci]),
        "center_magnitude": center,
        "peak_magnitude": peak,
        "center_peak_ratio": center / max(peak, np.finfo(float).eps),
        "delay_boundary": bool(ci in (0, len(DELAY_GRID_CHIPS) - 1)),
        "doppler_boundary": bool(di in (0, len(DOPPLER_GRID_HZ) - 1)),
        "grid_boundary": bool(
            di in (0, len(DOPPLER_GRID_HZ) - 1)
            or ci in (0, len(DELAY_GRID_CHIPS) - 1)
        ),
    }

def score_complex_surface(surface: np.ndarray) -> dict[str, float | bool | int]:
    return score_power_surface(np.abs(np.asarray(surface)) ** 2)


def support_deltas_are_causal(starts: Sequence[int], *, allow_r14_overlap: bool = False) -> bool:
    """Validate ordered L20 support without hiding 24,999/25,000 semantics."""

    if len(starts) != 20:
        return False
    allowed = {SUPPORT_SAMPLES}
    if allow_r14_overlap:
        allowed.add(SUPPORT_SAMPLES - 1)
    return all(int(b) - int(a) in allowed for a, b in zip(starts, starts[1:]))


def same_assignment(states: Sequence[State]) -> bool:
    if len(states) != 20:
        return False
    pair = (states[0].channel, states[0].prn)
    rows = [state.tracker_row for state in states]
    return (
        all((state.channel, state.prn) == pair for state in states)
        and rows == list(range(rows[0], rows[0] + 20))
        and support_deltas_are_causal([state.raw_start_sample for state in states])
    )


def numerical_equivalence(surfaces: Sequence[np.ndarray], tolerance: float = 1e-12) -> dict:
    """Compare R1.4 and R2 aggregate/peak/center semantics."""

    left = r14_l20_aggregate(surfaces)
    right = r2_l20_aggregate(surfaces)
    ls = score_power_surface(left)
    rs = score_power_surface(right)
    delta = float(np.max(np.abs(left - right)))
    numeric_keys = (
        "peak_doppler_offset_hz",
        "peak_delay_offset_chips",
        "center_magnitude",
        "peak_magnitude",
    )
    numeric_delta = max(abs(float(ls[key]) - float(rs[key])) for key in numeric_keys)
    return {
        "tolerance": tolerance,
        "aggregate_max_abs_delta": delta,
        "score_max_abs_delta": numeric_delta,
        "aggregate_equal": bool(delta <= tolerance),
        "peak_delay_equal": ls["peak_delay_offset_chips"] == rs["peak_delay_offset_chips"],
        "peak_doppler_equal": ls["peak_doppler_offset_hz"] == rs["peak_doppler_offset_hz"],
        "center_magnitude_equal": bool(
            abs(float(ls["center_magnitude"]) - float(rs["center_magnitude"])) <= tolerance
        ),
        "status": "PASS" if delta <= tolerance and numeric_delta <= tolerance else "FAIL",
        "r14": ls,
        "r2": rs,
    }
