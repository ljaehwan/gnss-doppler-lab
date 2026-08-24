"""Clean-only receiver-state conditional conformal tail detector.

The local GRU score remains frozen.  This module calibrates finite-sample
right-tail conformal p-values from receiver-quality-contract score rows and
aggregates simultaneous PRN anomalies with an exact binomial tail.  A matched
global conformal detector is emitted from the same clean reference rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from gnss_doppler_lab.quality_conditioned_tail import (
    binomial_tail_surprise,
    evaluate_attack,
    evaluate_clean,
)

CONFORMAL_ALPHAS: dict[str, float] = {"q50": 0.50, "q70": 0.30, "q80": 0.20}
GLOBAL_SCORE = "global_conformal_btail_max_507080_ewma075"
STATE_SCORE = "receiver_state_conformal_btail_max_507080_ewma075"
RAW_GLOBAL_SCORE = "global_conformal_btail_max_507080"
RAW_STATE_SCORE = "receiver_state_conformal_btail_max_507080"
DEFAULT_AGE_CUTOFFS_S = (10.0, 30.0)
DEFAULT_MIN_POOL_ROWS = 100
DEFAULT_EWMA_PREVIOUS_WEIGHT = 0.75

CONTRACT_COLUMNS = (
    "channel",
    "segment_index",
    "continuity_block_index",
    "tracking_age_s",
    "reacquisition_flag",
    "history_same_segment_flag",
)


@dataclass(frozen=True)
class CleanSplit:
    reference: pd.DataFrame
    event_calibration: pd.DataFrame
    held_clean: pd.DataFrame
    inventory: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ReceiverStateReference:
    global_scores: np.ndarray
    age_scores: Mapping[str, np.ndarray]
    exact_scores: Mapping[str, np.ndarray]
    age_cutoffs_s: tuple[float, float]
    min_pool_rows: int
    score_column: str


@dataclass(frozen=True)
class ReceiverStateCalibration:
    reference: ReceiverStateReference
    global_event_threshold: float
    state_event_threshold: float
    event_quantile: float
    reference_rows: int
    reference_events: int
    event_calibration_rows: int
    event_calibration_events: int


def _validate_age_cutoffs(values: Iterable[float]) -> tuple[float, float]:
    cutoffs = tuple(float(value) for value in values)
    if len(cutoffs) != 2 or not np.isfinite(cutoffs).all() or not 0.0 < cutoffs[0] < cutoffs[1]:
        raise ValueError("age_cutoffs_s must contain two finite increasing positive values")
    return cutoffs


def age_bin_names(age_cutoffs_s: Iterable[float] = DEFAULT_AGE_CUTOFFS_S) -> tuple[str, str, str]:
    low, high = _validate_age_cutoffs(age_cutoffs_s)
    return (f"age_lt_{low:g}s", f"age_{low:g}_to_{high:g}s", f"age_ge_{high:g}s")


def validate_contract_scores(
    scores: pd.DataFrame,
    score_column: str = "prn_node_rmse",
) -> pd.DataFrame:
    required = {
        "run_id", "prn", "window_bin_s", "window_start_s", "window_mid_s",
        score_column, *CONTRACT_COLUMNS,
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"quality score CSV missing columns: {missing}")
    if scores.empty:
        raise ValueError("quality score CSV is empty")
    frame = scores.copy()
    for column in ("run_id", "prn"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"quality score CSV contains null or empty {column}")
        frame[column] = frame[column].astype(str)

    numeric = [
        "window_bin_s", "window_start_s", "window_mid_s", score_column,
        "channel", "segment_index", "continuity_block_index", "tracking_age_s",
        "reacquisition_flag", "history_same_segment_flag",
    ]
    if "window_end_s" in frame:
        numeric.append("window_end_s")
    try:
        converted = frame[numeric].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("quality score CSV contract columns must be numeric") from exc
    if not np.isfinite(converted.to_numpy(float)).all():
        raise ValueError("quality score CSV contract columns must be finite")
    frame[numeric] = converted
    for column in ("channel", "segment_index", "continuity_block_index"):
        values = frame[column].to_numpy(float)
        if np.any(values < 0) or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain nonnegative integers")
    if np.any(frame["tracking_age_s"].to_numpy(float) < 0):
        raise ValueError("tracking_age_s must be nonnegative")
    for column in ("reacquisition_flag", "history_same_segment_flag"):
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain binary flags")
    if not frame["history_same_segment_flag"].eq(1).all():
        raise ValueError("all score histories must remain within one receiver segment")
    if frame.duplicated(["run_id", "window_bin_s", "prn"]).any():
        raise ValueError("duplicate (run_id, window_bin_s, prn) rows in quality scores")
    return frame


def annotate_receiver_state(
    scores: pd.DataFrame,
    *,
    score_column: str = "prn_node_rmse",
    age_cutoffs_s: Iterable[float] = DEFAULT_AGE_CUTOFFS_S,
) -> pd.DataFrame:
    cutoffs = _validate_age_cutoffs(age_cutoffs_s)
    frame = validate_contract_scores(scores, score_column=score_column)
    low, high = cutoffs
    labels = age_bin_names(cutoffs)
    ages = frame["tracking_age_s"].to_numpy(float)
    frame["receiver_age_bin"] = np.select(
        [ages < low, ages < high], [labels[0], labels[1]], default=labels[2]
    )
    reacquired = frame["reacquisition_flag"].eq(1).to_numpy()
    restarted = frame["continuity_block_index"].gt(0).to_numpy()
    frame["receiver_origin"] = np.select(
        [reacquired, restarted], ["reacquired", "gap_restart"], default="initial"
    )
    frame["receiver_state"] = frame["receiver_origin"] + "|" + frame["receiver_age_bin"]
    return frame.sort_values(["run_id", "window_bin_s", "prn"], kind="mergesort").reset_index(drop=True)


def chronological_clean_split(
    scores: pd.DataFrame,
    *,
    reference_fraction: float = 0.60,
    event_calibration_fraction: float = 0.20,
    score_column: str = "prn_node_rmse",
) -> CleanSplit:
    """Split each clean run by complete event bins, preserving chronology."""
    if not 0.0 < reference_fraction < 1.0:
        raise ValueError("reference_fraction must lie strictly between zero and one")
    if not 0.0 < event_calibration_fraction < 1.0:
        raise ValueError("event_calibration_fraction must lie strictly between zero and one")
    if reference_fraction + event_calibration_fraction >= 1.0:
        raise ValueError("clean split must leave a held-clean tail")
    frame = validate_contract_scores(scores, score_column=score_column)
    parts: dict[str, list[pd.DataFrame]] = {
        "reference": [], "event_calibration": [], "held_clean": [],
    }
    inventory: list[dict[str, object]] = []
    for run_id, run in frame.groupby("run_id", sort=True):
        bins = np.sort(run["window_bin_s"].unique())
        if len(bins) < 5:
            raise ValueError(f"clean run {run_id} needs at least five event bins")
        reference_end = int(np.floor(len(bins) * reference_fraction))
        calibration_end = int(np.floor(len(bins) * (reference_fraction + event_calibration_fraction)))
        if reference_end <= 0 or calibration_end <= reference_end or calibration_end >= len(bins):
            raise ValueError(f"clean run {run_id} produces an empty split")
        ranges = {
            "reference": bins[:reference_end],
            "event_calibration": bins[reference_end:calibration_end],
            "held_clean": bins[calibration_end:],
        }
        record: dict[str, object] = {"run_id": str(run_id), "total_event_bins": int(len(bins))}
        for role, selected in ranges.items():
            subset = run.loc[run["window_bin_s"].isin(selected)].copy()
            parts[role].append(subset)
            record[f"{role}_event_bins"] = int(len(selected))
            record[f"{role}_rows"] = int(len(subset))
            record[f"{role}_start_s"] = float(selected[0])
            record[f"{role}_end_s"] = float(selected[-1])
        inventory.append(record)
    return CleanSplit(
        reference=pd.concat(parts["reference"], ignore_index=True),
        event_calibration=pd.concat(parts["event_calibration"], ignore_index=True),
        held_clean=pd.concat(parts["held_clean"], ignore_index=True),
        inventory=tuple(inventory),
    )


def fit_receiver_state_reference(
    clean_reference_scores: pd.DataFrame,
    *,
    score_column: str = "prn_node_rmse",
    age_cutoffs_s: Iterable[float] = DEFAULT_AGE_CUTOFFS_S,
    min_pool_rows: int = DEFAULT_MIN_POOL_ROWS,
) -> ReceiverStateReference:
    if min_pool_rows <= 0:
        raise ValueError("min_pool_rows must be positive")
    cutoffs = _validate_age_cutoffs(age_cutoffs_s)
    annotated = annotate_receiver_state(
        clean_reference_scores, score_column=score_column, age_cutoffs_s=cutoffs
    )
    global_scores = np.sort(annotated[score_column].to_numpy(float))
    age_scores = {
        str(name): np.sort(group[score_column].to_numpy(float))
        for name, group in annotated.groupby("receiver_age_bin", sort=True)
    }
    exact_scores = {
        str(name): np.sort(group[score_column].to_numpy(float))
        for name, group in annotated.groupby("receiver_state", sort=True)
    }
    return ReceiverStateReference(
        global_scores=global_scores,
        age_scores=age_scores,
        exact_scores=exact_scores,
        age_cutoffs_s=cutoffs,
        min_pool_rows=int(min_pool_rows),
        score_column=score_column,
    )


def select_reference_pool(
    reference: ReceiverStateReference,
    receiver_state: str,
    receiver_age_bin: str,
) -> tuple[np.ndarray, str]:
    exact = reference.exact_scores.get(str(receiver_state), np.asarray([], dtype=float))
    if len(exact) >= reference.min_pool_rows:
        return exact, "exact_state"
    age = reference.age_scores.get(str(receiver_age_bin), np.asarray([], dtype=float))
    if len(age) >= reference.min_pool_rows:
        return age, "age_bin"
    return reference.global_scores, "global"


def right_tail_conformal_p(scores: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    """Finite-sample p=(1 + #{reference >= score})/(m + 1), including ties."""
    values = np.asarray(scores, dtype=float)
    reference = np.asarray(sorted_reference, dtype=float)
    if reference.ndim != 1 or len(reference) == 0 or not np.isfinite(reference).all():
        raise ValueError("conformal reference must be a nonempty finite vector")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("conformal scores must be a finite vector")
    if np.any(reference[1:] < reference[:-1]):
        raise ValueError("conformal reference must be sorted")
    greater_or_equal = len(reference) - np.searchsorted(reference, values, side="left")
    return (1.0 + greater_or_equal) / (len(reference) + 1.0)


def build_conformal_event_scores(
    scores: pd.DataFrame,
    *,
    reference: ReceiverStateReference,
    ewma_previous_weight: float = DEFAULT_EWMA_PREVIOUS_WEIGHT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= ewma_previous_weight < 1.0:
        raise ValueError("ewma_previous_weight must lie in [0, 1)")
    frame = annotate_receiver_state(
        scores,
        score_column=reference.score_column,
        age_cutoffs_s=reference.age_cutoffs_s,
    )
    values = frame[reference.score_column].to_numpy(float)
    frame["global_conformal_p"] = right_tail_conformal_p(values, reference.global_scores)
    state_p = np.empty(len(frame), dtype=float)
    pool_level = np.empty(len(frame), dtype=object)
    pool_rows = np.empty(len(frame), dtype=int)
    for (state, age_bin), positions in frame.groupby(
        ["receiver_state", "receiver_age_bin"], sort=False
    ).groups.items():
        index = np.asarray(list(positions), dtype=int)
        pool, level = select_reference_pool(reference, str(state), str(age_bin))
        state_p[index] = right_tail_conformal_p(values[index], pool)
        pool_level[index] = level
        pool_rows[index] = len(pool)
    frame["receiver_state_conformal_p"] = state_p
    frame["calibration_pool_level"] = pool_level
    frame["calibration_pool_rows"] = pool_rows

    rows: list[dict[str, object]] = []
    for (run_id, window_bin_s), group in frame.groupby(["run_id", "window_bin_s"], sort=True):
        n = int(len(group))
        row: dict[str, object] = {
            "run_id": str(run_id),
            "window_bin_s": float(window_bin_s),
            "window_start_s": float(group["window_start_s"].min()),
            "window_mid_s": float(group["window_mid_s"].min()),
            "window_end_s": (
                float(group["window_end_s"].max())
                if "window_end_s" in group else float(group["window_start_s"].max() + 1.0)
            ),
            "tracked_prn_count": n,
            "exact_state_prn_count": int(group["calibration_pool_level"].eq("exact_state").sum()),
            "age_fallback_prn_count": int(group["calibration_pool_level"].eq("age_bin").sum()),
            "global_fallback_prn_count": int(group["calibration_pool_level"].eq("global").sum()),
        }
        global_surprises: list[float] = []
        state_surprises: list[float] = []
        for name, alpha in CONFORMAL_ALPHAS.items():
            global_k = int(group["global_conformal_p"].le(alpha).sum())
            state_k = int(group["receiver_state_conformal_p"].le(alpha).sum())
            row[f"global_k_{name}"] = global_k
            row[f"receiver_state_k_{name}"] = state_k
            row[f"global_btail_{name}"] = binomial_tail_surprise(global_k, n, alpha)
            row[f"receiver_state_btail_{name}"] = binomial_tail_surprise(state_k, n, alpha)
            global_surprises.append(float(row[f"global_btail_{name}"]))
            state_surprises.append(float(row[f"receiver_state_btail_{name}"]))
        row[RAW_GLOBAL_SCORE] = max(global_surprises)
        row[RAW_STATE_SCORE] = max(state_surprises)
        rows.append(row)
    events = pd.DataFrame(rows).sort_values(["run_id", "window_bin_s"]).reset_index(drop=True)
    for raw_column, final_column in (
        (RAW_GLOBAL_SCORE, GLOBAL_SCORE), (RAW_STATE_SCORE, STATE_SCORE),
    ):
        events[final_column] = 0.0
        for _, positions in events.groupby("run_id", sort=False).groups.items():
            previous = 0.0
            for position in positions:
                current = float(events.at[position, raw_column])
                previous = ewma_previous_weight * previous + (1.0 - ewma_previous_weight) * current
                events.at[position, final_column] = previous
    return events, frame


def calibrate_receiver_state_detectors(
    clean_reference_scores: pd.DataFrame,
    clean_event_calibration_scores: pd.DataFrame,
    *,
    score_column: str = "prn_node_rmse",
    age_cutoffs_s: Iterable[float] = DEFAULT_AGE_CUTOFFS_S,
    min_pool_rows: int = DEFAULT_MIN_POOL_ROWS,
    event_quantile: float = 0.99,
) -> tuple[ReceiverStateCalibration, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < event_quantile < 1.0:
        raise ValueError("event_quantile must lie strictly between zero and one")
    reference = fit_receiver_state_reference(
        clean_reference_scores,
        score_column=score_column,
        age_cutoffs_s=age_cutoffs_s,
        min_pool_rows=min_pool_rows,
    )
    reference_events, _ = build_conformal_event_scores(
        clean_reference_scores, reference=reference
    )
    calibration_events, calibration_nodes = build_conformal_event_scores(
        clean_event_calibration_scores, reference=reference
    )
    calibration = ReceiverStateCalibration(
        reference=reference,
        global_event_threshold=float(calibration_events[GLOBAL_SCORE].quantile(event_quantile)),
        state_event_threshold=float(calibration_events[STATE_SCORE].quantile(event_quantile)),
        event_quantile=float(event_quantile),
        reference_rows=int(len(clean_reference_scores)),
        reference_events=int(len(reference_events)),
        event_calibration_rows=int(len(clean_event_calibration_scores)),
        event_calibration_events=int(len(calibration_events)),
    )
    return calibration, calibration_events, calibration_nodes


def reference_inventory(reference: ReceiverStateReference) -> dict[str, object]:
    return {
        "global_rows": int(len(reference.global_scores)),
        "age_pool_rows": {key: int(len(value)) for key, value in reference.age_scores.items()},
        "exact_pool_rows": {key: int(len(value)) for key, value in reference.exact_scores.items()},
        "age_cutoffs_s": list(reference.age_cutoffs_s),
        "min_pool_rows": reference.min_pool_rows,
        "fallback_order": ["exact_state", "age_bin", "global"],
        "score_column": reference.score_column,
    }


__all__ = [
    "CONFORMAL_ALPHAS", "GLOBAL_SCORE", "STATE_SCORE", "CleanSplit",
    "ReceiverStateReference", "ReceiverStateCalibration", "age_bin_names",
    "validate_contract_scores", "annotate_receiver_state", "chronological_clean_split",
    "fit_receiver_state_reference", "select_reference_pool", "right_tail_conformal_p",
    "build_conformal_event_scores", "calibrate_receiver_state_detectors",
    "reference_inventory", "evaluate_clean", "evaluate_attack",
]
