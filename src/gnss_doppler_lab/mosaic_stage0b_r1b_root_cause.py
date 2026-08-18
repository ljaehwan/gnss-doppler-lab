"""Frozen post-hoc root-cause diagnostics for MOSAIC Stage-0B R1b.

This module has no injection, raw-IQ writing, receiver replay, subprocess, or
detector-training entry point.  All scores are diagnostics over retained TRACE.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .mosaic_stage0b_r1_execution_metrics import bic

DELAY_GRID = np.round(np.arange(-0.35, 0.350001, 0.025), 12)
DOPPLER_GRID = np.arange(-75.0, 75.0001, 5.0)
FROZEN_DELAY_TOLERANCE_CHIPS = 0.05
FROZEN_DOPPLER_TOLERANCE_HZ = 10.0
ALLOWED_HYPOTHESIS_STATES = {"SUPPORTED", "UNSUPPORTED", "INCONCLUSIVE"}
ALLOWED_RECOMMENDATIONS = {"Frozen corrected observer confirmation", "Terminate MOSAIC"}


def wrap_phase(value: np.ndarray | float) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    return (x + np.pi) % (2 * np.pi) - np.pi


def receiver_frame_coordinates(
    requested_delay_chips: float,
    requested_doppler_hz: float,
    clean_residual_code_phase_chips: Iterable[float],
    observed_residual_code_phase_chips: Iterable[float],
    clean_carrier_doppler_hz: Iterable[float],
    observed_carrier_doppler_hz: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the injector-derived receiver-frame sign and unit convention."""
    clean_code = np.asarray(list(clean_residual_code_phase_chips), float)
    observed_code = np.asarray(list(observed_residual_code_phase_chips), float)
    clean_carrier = np.asarray(list(clean_carrier_doppler_hz), float)
    observed_carrier = np.asarray(list(observed_carrier_doppler_hz), float)
    if not (clean_code.shape == observed_code.shape == clean_carrier.shape == observed_carrier.shape):
        raise ValueError("receiver action arrays must share exact common support")
    delay = float(requested_delay_chips) + observed_code - clean_code
    doppler = float(requested_doppler_hz) + clean_carrier - observed_carrier
    return delay, doppler


def integrate_phase(times_s: Iterable[float], doppler_hz: Iterable[float], phase0_rad: float) -> np.ndarray:
    times = np.asarray(list(times_s), float)
    freq = np.asarray(list(doppler_hz), float)
    if times.shape != freq.shape or times.ndim != 1 or np.any(np.diff(times) < 0):
        raise ValueError("phase integration requires ordered matching vectors")
    out = np.empty(len(times), float)
    if not len(times):
        return out
    out[0] = float(phase0_rad)
    if len(times) > 1:
        out[1:] = out[0] + 2 * np.pi * np.cumsum(0.5 * (freq[1:] + freq[:-1]) * np.diff(times))
    return out


def triangular_template(tap_offsets_chips: Iterable[float], delays_chips: Iterable[float], phase_rad: Iterable[float]) -> np.ndarray:
    taps = np.asarray(list(tap_offsets_chips), float)
    delays = np.asarray(list(delays_chips), float)
    phase = np.asarray(list(phase_rad), float)
    if delays.shape != phase.shape:
        raise ValueError("delay and phase trajectories differ")
    spatial = np.maximum(1.0 - np.abs(taps[None, :] - delays[:, None]), 0.0)
    return spatial * np.exp(1j * phase[:, None])


def fitted_clean_residual(clean: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, complex]:
    a = np.asarray(clean, np.complex128).reshape(-1, 1)
    y = np.asarray(observed, np.complex128).reshape(-1)
    if a.shape[0] != y.size or not y.size:
        raise ValueError("clean and observed taps require nonempty identical support")
    coefficient = complex(np.linalg.lstsq(a, y, rcond=None)[0][0])
    return (y - a[:, 0] * coefficient).reshape(np.asarray(observed).shape), coefficient


def diagnostic_projection(clean: np.ndarray, observed: np.ndarray, template: np.ndarray) -> dict[str, float]:
    residual, _ = fitted_clean_residual(clean, observed)
    a = np.asarray(clean, np.complex128).reshape(-1, 1)
    q = np.asarray(template, np.complex128).reshape(-1, 1)
    if q.shape != a.shape:
        raise ValueError("template support differs from aligned taps")
    gram = complex((a.conj().T @ a)[0, 0])
    orth = q[:, 0] - a[:, 0] * complex((a.conj().T @ q)[0, 0] / gram)
    r = residual.reshape(-1)
    rss0 = max(float(np.vdot(r, r).real), np.finfo(float).tiny)
    denom = max(float(np.vdot(orth, orth).real), np.finfo(float).tiny)
    explained = float(abs(np.vdot(orth, r)) ** 2 / denom)
    ratio = explained / rss0
    if ratio < -1e-10 or ratio > 1 + 1e-10:
        raise ValueError(f"projection ratio outside mathematical bounds: {ratio}")
    ratio = float(np.clip(ratio, 0.0, 1.0))
    rss1 = max(rss0 - explained, np.finfo(float).eps * max(rss0, 1.0))
    nobs = 2 * r.size
    delta = bic(rss0, nobs, 2) - bic(rss1, nobs, 4)
    return {"projection_ratio": ratio, "unexplained_residual_energy": rss1, "residual_energy": rss0, "delta_bic": float(delta)}


def frozen_grid_score(clean: np.ndarray, observed: np.ndarray, times_s: np.ndarray, tap_offsets_chips: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    for delay in DELAY_GRID:
        for doppler in DOPPLER_GRID:
            template = triangular_template(tap_offsets_chips, np.full(len(times_s), delay), 2*np.pi*doppler*times_s)
            value = diagnostic_projection(clean, observed, template)
            candidate = {**value, "recovered_delay_chips": float(delay), "recovered_doppler_hz": float(doppler)}
            if best is None or candidate["delta_bic"] > best["delta_bic"]:
                best = candidate
    if best is None:
        raise ValueError("empty frozen score support")
    return best


def segment_indices(times_s: Iterable[float], origin_s: float, window_s: float) -> list[np.ndarray]:
    times = np.asarray(list(times_s), float)
    if window_s <= 0 or np.any(times < origin_s - 1e-12):
        raise ValueError("invalid deterministic window support")
    bins = np.floor((times - origin_s) / float(window_s) + 1e-10).astype(np.int64)
    return [np.flatnonzero(bins == value) for value in np.unique(bins)]


def physics_recovered(requested_delay: float, recovered_delay: float, requested_doppler: float, recovered_doppler: float) -> bool:
    return bool(abs(recovered_delay-requested_delay) <= FROZEN_DELAY_TOLERANCE_CHIPS and abs(recovered_doppler-requested_doppler) <= FROZEN_DOPPLER_TOLERANCE_HZ)


def select_single_comparators(four_case: dict[str, object], singles: list[dict[str, object]]) -> tuple[str, list[dict[str, object]]]:
    same_dataset = [x for x in singles if x["case"]["dataset"] == four_case["case"]["dataset"]]
    keys = ("rho_db", "delta_tau_chips", "delta_f_hz", "delta_phi_rad")
    exact = [x for x in same_dataset if all(x["case"][k] == four_case["case"][k] for k in keys)]
    if exact:
        return "exact_parameter_match", sorted(exact, key=lambda x: x["case"]["case_id"])
    one = [x for x in same_dataset if sum(x["case"][k] != four_case["case"][k] for k in keys) == 1]
    if one:
        return "exactly_one_variable_different", sorted(one, key=lambda x: x["case"]["case_id"])
    index = int(four_case["case"]["case_id"].rsplit(".", 1)[1])
    matched = [x for x in same_dataset if int(x["case"]["case_id"].rsplit(".", 1)[1]) == index]
    return "same_design_index", sorted(matched, key=lambda x: x["case"]["case_id"])


def decide_root_cause(evidence_available: bool, supported: Iterable[str], oracle_restores: bool) -> str:
    if not evidence_available:
        return "ROOT_CAUSE_EVIDENCE_UNAVAILABLE"
    causes = set(supported)
    if oracle_restores and causes == {"H1"}:
        return "SCORER_RECEIVER_FRAME_MISMATCH_SUPPORTED"
    mapping = {"H2": "RELATIVE_PHASE_CANCELLATION_SUPPORTED", "H3": "ANALYTIC_TEMPLATE_MISMATCH_SUPPORTED", "H5": "PRN_SPECIFIC_BASELINE_DOMINANCE_SUPPORTED", "H6": "TEMPORAL_AGGREGATION_DILUTION_SUPPORTED"}
    if len(causes) == 1 and next(iter(causes)) in mapping:
        return mapping[next(iter(causes))]
    if causes == {"H4"}:
        return "TRACKING_LOOP_NONLINEARITY_SUPPORTED"
    return "MIXED_OR_UNIDENTIFIED_ROOT_CAUSE"


def decide_recommendation(*, oracle_restores: bool, consistent_improvement: bool, comparators_not_degraded: bool, controls_separated: bool) -> str:
    values = (oracle_restores, consistent_improvement, comparators_not_degraded, controls_separated)
    return "Frozen corrected observer confirmation" if all(values) else "Terminate MOSAIC"
