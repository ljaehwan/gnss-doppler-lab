"""Native-cadence contracts for TRACE Stage-0 R1.

This module is intentionally independent of the earlier TRACE Stage-0 loader.
It audits every consecutive per-channel row before any model is fit or any
attack score is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


TAPS = ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
CADENCE_1MS = "approximately_1_ms"
CADENCE_20MS = "approximately_20_ms"
CADENCE_GAP = "gap_or_reacquisition"
CADENCE_INVALID = "invalid_or_outlier"


@dataclass(frozen=True)
class ScenarioSpec:
    dataset: str
    scenario: str
    receiver_root: Path
    sample_rate_hz: float
    onset_s: float | None = None


@dataclass(frozen=True)
class PairRows:
    channel: np.ndarray
    prn: np.ndarray
    source_row: np.ndarray
    start_sample: np.ndarray
    target_sample: np.ndarray
    time_s: np.ndarray
    dt_s: np.ndarray
    cadence: np.ndarray
    transition_excluded: np.ndarray

    def take(self, mask: np.ndarray) -> "PairRows":
        return PairRows(**{name: getattr(self, name)[mask] for name in self.__dataclass_fields__})


def classify_cadence(dt_s: np.ndarray) -> np.ndarray:
    """Classify actual sample-count intervals without assuming a native rate."""
    dt = np.asarray(dt_s, dtype=np.float64)
    labels = np.full(dt.shape, CADENCE_INVALID, dtype="U24")
    labels[(dt >= 0.0009) & (dt <= 0.0011)] = CADENCE_1MS
    labels[(dt >= 0.019) & (dt <= 0.021)] = CADENCE_20MS
    labels[dt > 0.021] = CADENCE_GAP
    return labels


def transition_mask(cadence: np.ndarray) -> np.ndarray:
    """Exclude both sides of a 1 ms <-> 20 ms transition."""
    values = np.asarray(cadence)
    out = np.zeros(len(values), dtype=bool)
    if len(values) < 2:
        return out
    native = np.isin(values, (CADENCE_1MS, CADENCE_20MS))
    changed = native[:-1] & native[1:] & (values[:-1] != values[1:])
    where = np.flatnonzero(changed)
    out[where] = True
    out[where + 1] = True
    return out


def load_consecutive_pairs(spec: ScenarioSpec) -> PairRows:
    """Load all same-PRN consecutive row intervals for one scenario."""
    batches: dict[str, list[np.ndarray]] = {name: [] for name in PairRows.__dataclass_fields__}
    paths = sorted((spec.receiver_root / "raw").glob("epl_tracking_ch_*.mat"))
    if not paths:
        raise FileNotFoundError(f"no receiver MAT files under {spec.receiver_root}")
    for channel, path in enumerate(paths):
        with h5py.File(path, "r") as handle:
            samples = np.asarray(handle["PRN_start_sample_count"]).reshape(-1).astype(np.int64)
            prn = np.asarray(handle["PRN"]).reshape(-1).astype(np.int16)
        delta = np.diff(samples)
        indices = np.flatnonzero(prn[:-1] == prn[1:])
        dt = delta[indices].astype(np.float64) / float(spec.sample_rate_hz)
        cadence = classify_cadence(dt)
        transition = transition_mask(cadence)
        count = len(indices)
        batches["channel"].append(np.full(count, channel, dtype=np.int16))
        batches["prn"].append(prn[indices])
        batches["source_row"].append(indices.astype(np.int64))
        batches["start_sample"].append(samples[indices])
        batches["target_sample"].append(samples[indices + 1])
        batches["time_s"].append(samples[indices + 1].astype(np.float64) / spec.sample_rate_hz)
        batches["dt_s"].append(dt)
        batches["cadence"].append(cadence)
        batches["transition_excluded"].append(transition)
    return PairRows(**{name: np.concatenate(parts) for name, parts in batches.items()})


def cadence_counts(rows: PairRows, mask: np.ndarray | None = None) -> dict[str, int]:
    selected = np.ones(len(rows.prn), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    return {
        CADENCE_1MS: int(np.sum(selected & (rows.cadence == CADENCE_1MS))),
        CADENCE_20MS: int(np.sum(selected & (rows.cadence == CADENCE_20MS))),
        CADENCE_GAP: int(np.sum(selected & (rows.cadence == CADENCE_GAP))),
        CADENCE_INVALID: int(np.sum(selected & (rows.cadence == CADENCE_INVALID))),
    }


def valid_native_mask(rows: PairRows, cadence: str = CADENCE_20MS) -> np.ndarray:
    return (rows.cadence == cadence) & ~rows.transition_excluded


def block_support(
    rows: PairRows,
    *,
    block_s: float = 0.5,
    cadence: str = CADENCE_20MS,
) -> list[dict[str, int | float]]:
    mask = valid_native_mask(rows, cadence)
    if not np.any(mask):
        return []
    block_ids = np.floor(rows.time_s[mask] / block_s).astype(np.int64)
    prn = rows.prn[mask]
    result: list[dict[str, int | float]] = []
    for block_id in np.unique(block_ids):
        selected = block_ids == block_id
        result.append(
            {
                "block_start_s": float(block_id * block_s),
                "block_end_s": float((block_id + 1) * block_s),
                "valid_prn_count": int(len(np.unique(prn[selected]))),
                "pair_count": int(np.sum(selected)),
            }
        )
    return result


def gap_distribution(rows: PairRows) -> dict[str, float | int | None]:
    gaps = rows.dt_s[rows.cadence == CADENCE_GAP]
    if not len(gaps):
        return {"count": 0, "min_s": None, "median_s": None, "q95_s": None, "max_s": None}
    return {
        "count": int(len(gaps)),
        "min_s": float(np.min(gaps)),
        "median_s": float(np.median(gaps)),
        "q95_s": float(np.quantile(gaps, 0.95)),
        "max_s": float(np.max(gaps)),
    }


def consecutive_alarm(
    block_start_s: Iterable[float], scores: Iterable[float], threshold: float, run_length: int = 3
) -> np.ndarray:
    """Alarm only on actual consecutive 0.5 s blocks, resetting across gaps."""
    times = np.asarray(tuple(block_start_s), dtype=float)
    values = np.asarray(tuple(scores), dtype=float)
    alarm = np.zeros(len(values), dtype=bool)
    run = 0
    previous: float | None = None
    for index, (time_s, score) in enumerate(zip(times, values, strict=True)):
        if previous is None or not np.isclose(time_s - previous, 0.5, atol=1e-9):
            run = 0
        run = run + 1 if score > threshold else 0
        alarm[index] = run >= run_length
        previous = time_s
    return alarm


def trace_r1_verdict(
    *,
    mapping_verified: bool,
    native_dump_available: bool,
    real_scores_exist: bool,
    provenance_complete: bool = True,
    go_conditions: Iterable[bool] = (),
) -> str:
    """Fail-closed TRACE-R1 route/verdict logic."""
    if not mapping_verified and not native_dump_available:
        return "NEEDS_TRACE_SPECIFIC_RECEIVER_DUMP"
    if real_scores_exist and not provenance_complete:
        return "INCONCLUSIVE_BASELINE_OR_PROVENANCE"
    conditions = tuple(go_conditions)
    if real_scores_exist and conditions and all(conditions):
        return "GO_FOR_TRACE_STAGE1"
    return "NO_GO_ACTION_EQUIVARIANCE"
