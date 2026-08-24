"""Quality-conditioned multi-PRN binomial-tail spoofing score.

The local detector score is intentionally treated as an input.  This module
only asks whether a local score is unusual for the current *causal tracking
quality state*, then aggregates simultaneous PRN exceedances.  The first
prototype uses observed-score continuity age because it is available in both
the frozen TEXBAT and OAKBAT score contracts without reading attack labels or
same-window RF features.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

NODE_QUANTILES: dict[str, float] = {"q50": 0.50, "q70": 0.70, "q80": 0.80}
GLOBAL_SCORE = "global_btail_max_507080_ewma075"
QUALITY_SCORE = "quality_btail_max_507080_ewma075"
RAW_GLOBAL_SCORE = "global_btail_max_507080"
RAW_QUALITY_SCORE = "quality_btail_max_507080"
EWMA_PREVIOUS_WEIGHT = 0.75
DEFAULT_AGE_CUTOFFS_S = (5.0, 20.0)
DEFAULT_MAX_GAP_S = 0.75
DEFAULT_WINDOW_AVAILABILITY_OFFSET_S = 1.0


@dataclass(frozen=True)
class TailCalibration:
    """Normal-only calibration shared by global and quality-aware detectors."""

    global_node_thresholds: dict[str, float]
    quality_node_thresholds: dict[str, dict[str, float]]
    quality_bin_counts: dict[str, int]
    quality_bin_fallbacks: dict[str, bool]
    global_event_threshold: float
    quality_event_threshold: float
    age_cutoffs_s: tuple[float, float]
    max_gap_s: float
    min_bin_rows: int
    calibration_rows: int
    calibration_events: int


def _validate_age_cutoffs(age_cutoffs_s: Iterable[float]) -> tuple[float, float]:
    values = tuple(float(value) for value in age_cutoffs_s)
    if len(values) != 2 or not np.isfinite(values).all() or not 0.0 < values[0] < values[1]:
        raise ValueError("age_cutoffs_s must contain two finite increasing positive values")
    return values


def quality_bin_names(age_cutoffs_s: Iterable[float] = DEFAULT_AGE_CUTOFFS_S) -> tuple[str, str, str]:
    low, high = _validate_age_cutoffs(age_cutoffs_s)
    return (f"age_lt_{low:g}s", f"age_{low:g}_to_{high:g}s", f"age_ge_{high:g}s")


def validate_prn_scores(prn_scores: pd.DataFrame, score_column: str = "prn_node_rmse") -> pd.DataFrame:
    required = {
        "run_id", "prn", "window_bin_s", "window_start_s", "window_mid_s", score_column,
    }
    missing = sorted(required - set(prn_scores.columns))
    if missing:
        raise ValueError(f"PRN score CSV missing columns: {missing}")
    frame = prn_scores.copy()
    for column in ("run_id", "prn"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"PRN score CSV contains null or empty {column}")
        frame[column] = frame[column].astype(str)
    numeric_columns = ["window_bin_s", "window_start_s", "window_mid_s", score_column]
    if "window_end_s" in frame.columns:
        numeric_columns.append("window_end_s")
    for column in numeric_columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"PRN score CSV contains non-numeric {column}") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"PRN score CSV contains non-finite {column}")
        frame[column] = values
    duplicate_key = ["run_id", "window_bin_s", "prn"]
    if frame.duplicated(duplicate_key, keep=False).any():
        raise ValueError("duplicate (run_id, window_bin_s, prn) rows in PRN scores")
    return frame


def annotate_quality_state(
    prn_scores: pd.DataFrame,
    *,
    score_column: str = "prn_node_rmse",
    age_cutoffs_s: Iterable[float] = DEFAULT_AGE_CUTOFFS_S,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
) -> pd.DataFrame:
    """Add causal continuity segment and age derived only from past timestamps.

    A gap larger than ``max_gap_s`` starts a new observed-score segment.  Age is
    zero at the first available score after the GRU warm-up or a gap.  It is a
    portable proxy for tracking maturity, not a claim of hardware lock age.
    """
    cutoffs = _validate_age_cutoffs(age_cutoffs_s)
    if not np.isfinite(max_gap_s) or max_gap_s <= 0.0:
        raise ValueError("max_gap_s must be finite and positive")
    frame = validate_prn_scores(prn_scores, score_column=score_column)
    frame = frame.sort_values(["run_id", "prn", "window_bin_s"]).reset_index(drop=True)
    frame["quality_segment_index"] = 0
    frame["quality_age_s"] = 0.0

    for _, positions in frame.groupby(["run_id", "prn"], sort=False).groups.items():
        index = np.asarray(list(positions), dtype=int)
        times = frame.loc[index, "window_bin_s"].to_numpy(float)
        gaps = np.diff(times, prepend=times[0])
        starts = np.zeros(len(index), dtype=bool)
        starts[0] = True
        starts[1:] = (gaps[1:] > max_gap_s) | (gaps[1:] <= 0.0)
        segments = np.cumsum(starts) - 1
        segment_start_times = pd.Series(times).groupby(segments).transform("first").to_numpy(float)
        frame.loc[index, "quality_segment_index"] = segments
        frame.loc[index, "quality_age_s"] = times - segment_start_times

    low, high = cutoffs
    labels = quality_bin_names(cutoffs)
    ages = frame["quality_age_s"].to_numpy(float)
    frame["quality_bin"] = np.select(
        [ages < low, ages < high], [labels[0], labels[1]], default=labels[2]
    )
    return frame.sort_values(["run_id", "window_bin_s", "prn"]).reset_index(drop=True)


def binomial_tail_surprise(k: int, n: int, exceedance_probability: float) -> float:
    """Return ``-ln(P[X >= k])`` for ``X ~ Binomial(n, p)``."""
    if n <= 0 or k <= 0:
        return 0.0
    if k > n:
        return -math.log(1e-300)
    probability = float(exceedance_probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("exceedance_probability must lie strictly between zero and one")
    tail = sum(
        math.comb(n, value)
        * probability**value
        * (1.0 - probability) ** (n - value)
        for value in range(k, n + 1)
    )
    return -math.log(max(tail, 1e-300))


def fit_node_thresholds(
    annotated_clean_scores: pd.DataFrame,
    *,
    score_column: str = "prn_node_rmse",
    age_cutoffs_s: Iterable[float] = DEFAULT_AGE_CUTOFFS_S,
    min_bin_rows: int = 100,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, int], dict[str, bool]]:
    if min_bin_rows <= 0:
        raise ValueError("min_bin_rows must be positive")
    if "quality_bin" not in annotated_clean_scores:
        raise ValueError("clean scores must first be annotated with quality state")
    values = annotated_clean_scores[score_column]
    global_thresholds = {
        name: float(values.quantile(quantile)) for name, quantile in NODE_QUANTILES.items()
    }
    quality_thresholds: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    fallbacks: dict[str, bool] = {}
    for quality_bin in quality_bin_names(age_cutoffs_s):
        subset = annotated_clean_scores.loc[
            annotated_clean_scores["quality_bin"] == quality_bin, score_column
        ]
        counts[quality_bin] = int(len(subset))
        fallbacks[quality_bin] = len(subset) < min_bin_rows
        quality_thresholds[quality_bin] = (
            dict(global_thresholds)
            if fallbacks[quality_bin]
            else {
                name: float(subset.quantile(quantile))
                for name, quantile in NODE_QUANTILES.items()
            }
        )
    return global_thresholds, quality_thresholds, counts, fallbacks


def build_event_scores(
    prn_scores: pd.DataFrame,
    *,
    global_node_thresholds: Mapping[str, float],
    quality_node_thresholds: Mapping[str, Mapping[str, float]],
    score_column: str = "prn_node_rmse",
    age_cutoffs_s: Iterable[float] = DEFAULT_AGE_CUTOFFS_S,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    ewma_previous_weight: float = EWMA_PREVIOUS_WEIGHT,
) -> pd.DataFrame:
    """Build matched global and quality-conditioned event scores."""
    if set(global_node_thresholds) != set(NODE_QUANTILES):
        raise ValueError(f"global thresholds must contain exactly {sorted(NODE_QUANTILES)}")
    expected_bins = set(quality_bin_names(age_cutoffs_s))
    if set(quality_node_thresholds) != expected_bins:
        raise ValueError(f"quality thresholds must contain exactly {sorted(expected_bins)}")
    for thresholds in quality_node_thresholds.values():
        if set(thresholds) != set(NODE_QUANTILES):
            raise ValueError("each quality bin must contain all node quantile thresholds")
    all_thresholds = [*global_node_thresholds.values()]
    all_thresholds.extend(value for item in quality_node_thresholds.values() for value in item.values())
    if not np.isfinite(np.asarray(all_thresholds, dtype=float)).all():
        raise ValueError("node thresholds must be finite")
    if not 0.0 <= ewma_previous_weight < 1.0:
        raise ValueError("ewma_previous_weight must lie in [0, 1)")

    frame = annotate_quality_state(
        prn_scores,
        score_column=score_column,
        age_cutoffs_s=age_cutoffs_s,
        max_gap_s=max_gap_s,
    )
    rows: list[dict[str, float | int | str]] = []
    for (run_id, window_bin_s), group in frame.groupby(["run_id", "window_bin_s"], sort=True):
        scores = group[score_column].to_numpy(float)
        quality_bins = group["quality_bin"].astype(str).to_numpy()
        n = int(len(scores))
        row: dict[str, float | int | str] = {
            "run_id": str(run_id),
            "window_bin_s": float(window_bin_s),
            "window_start_s": float(group["window_start_s"].min()),
            "window_mid_s": float(group["window_mid_s"].min()),
            "window_end_s": (
                float(group["window_end_s"].max())
                if "window_end_s" in group else float(group["window_start_s"].max() + 1.0)
            ),
            "tracked_prn_count": n,
            "young_prn_count": int((group["quality_age_s"] < _validate_age_cutoffs(age_cutoffs_s)[0]).sum()),
        }
        global_surprises: list[float] = []
        quality_surprises: list[float] = []
        for name, quantile in NODE_QUANTILES.items():
            global_k = int(np.sum(scores > float(global_node_thresholds[name])))
            per_row_threshold = np.asarray(
                [float(quality_node_thresholds[quality_bin][name]) for quality_bin in quality_bins]
            )
            quality_k = int(np.sum(scores > per_row_threshold))
            exceedance_probability = 1.0 - quantile
            global_surprise = binomial_tail_surprise(global_k, n, exceedance_probability)
            quality_surprise = binomial_tail_surprise(quality_k, n, exceedance_probability)
            row[f"global_k_{name}"] = global_k
            row[f"quality_k_{name}"] = quality_k
            row[f"global_btail_{name}"] = global_surprise
            row[f"quality_btail_{name}"] = quality_surprise
            global_surprises.append(global_surprise)
            quality_surprises.append(quality_surprise)
        row[RAW_GLOBAL_SCORE] = float(max(global_surprises))
        row[RAW_QUALITY_SCORE] = float(max(quality_surprises))
        rows.append(row)

    events = pd.DataFrame(rows).sort_values(["run_id", "window_bin_s"]).reset_index(drop=True)
    for raw_column, final_column in (
        (RAW_GLOBAL_SCORE, GLOBAL_SCORE), (RAW_QUALITY_SCORE, QUALITY_SCORE),
    ):
        smoothed = np.empty(len(events), dtype=float)
        for _, positions in events.groupby("run_id", sort=False).groups.items():
            previous = 0.0
            for position in positions:
                current = float(events.at[position, raw_column])
                previous = ewma_previous_weight * previous + (1.0 - ewma_previous_weight) * current
                smoothed[position] = previous
        events[final_column] = smoothed
    return events


def calibrate_tail_detectors(
    clean_scores: pd.DataFrame,
    *,
    score_column: str = "prn_node_rmse",
    age_cutoffs_s: Iterable[float] = DEFAULT_AGE_CUTOFFS_S,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    min_bin_rows: int = 100,
    event_quantile: float = 0.99,
) -> tuple[TailCalibration, pd.DataFrame]:
    if not 0.0 < event_quantile < 1.0:
        raise ValueError("event_quantile must lie strictly between zero and one")
    cutoffs = _validate_age_cutoffs(age_cutoffs_s)
    annotated = annotate_quality_state(
        clean_scores,
        score_column=score_column,
        age_cutoffs_s=cutoffs,
        max_gap_s=max_gap_s,
    )
    global_thresholds, quality_thresholds, counts, fallbacks = fit_node_thresholds(
        annotated,
        score_column=score_column,
        age_cutoffs_s=cutoffs,
        min_bin_rows=min_bin_rows,
    )
    events = build_event_scores(
        annotated,
        global_node_thresholds=global_thresholds,
        quality_node_thresholds=quality_thresholds,
        score_column=score_column,
        age_cutoffs_s=cutoffs,
        max_gap_s=max_gap_s,
    )
    calibration = TailCalibration(
        global_node_thresholds=global_thresholds,
        quality_node_thresholds=quality_thresholds,
        quality_bin_counts=counts,
        quality_bin_fallbacks=fallbacks,
        global_event_threshold=float(events[GLOBAL_SCORE].quantile(event_quantile)),
        quality_event_threshold=float(events[QUALITY_SCORE].quantile(event_quantile)),
        age_cutoffs_s=cutoffs,
        max_gap_s=float(max_gap_s),
        min_bin_rows=int(min_bin_rows),
        calibration_rows=int(len(annotated)),
        calibration_events=int(len(events)),
    )
    return calibration, events


def evaluate_clean(events: pd.DataFrame, score_column: str, threshold: float) -> dict[str, object]:
    if events.empty:
        raise ValueError("clean evaluation requires at least one event")
    flags = events[score_column] > threshold
    return {
        "windows": int(len(events)),
        "false_positive_flags": int(flags.sum()),
        "false_positive_rate": float(flags.mean()),
        "any_false_positive": bool(flags.any()),
        "threshold": float(threshold),
    }


def _first_consecutive_time(events: pd.DataFrame, flags: pd.Series, count: int) -> float | None:
    if count <= 0:
        raise ValueError("consecutive count must be positive")
    for _, positions in events.groupby("run_id", sort=False).groups.items():
        run = 0
        for position in positions:
            run = run + 1 if bool(flags.at[position]) else 0
            if run >= count:
                return float(events.at[position, "window_start_s"])
    return None


def evaluate_attack(
    events: pd.DataFrame,
    score_column: str,
    threshold: float,
    onset_s: float,
    *,
    guard_s: float = 10.0,
    availability_offset_s: float = DEFAULT_WINDOW_AVAILABILITY_OFFSET_S,
) -> dict[str, object]:
    if events.empty:
        raise ValueError("attack evaluation requires at least one event")
    if guard_s < 0.0 or availability_offset_s < 0.0:
        raise ValueError("guard and availability offset must be non-negative")
    flags = events[score_column] > threshold
    times = events["window_start_s"]
    pre = times < onset_s - guard_s
    post = times >= onset_s + guard_s
    after_onset = times >= onset_s
    if not pre.any() or not post.any():
        raise ValueError("attack evaluation requires buffered pre- and post-onset events")
    first_mask = flags & after_onset
    first = float(times[first_mask].min()) if first_mask.any() else None
    consecutive_flags = flags & after_onset
    third = _first_consecutive_time(events, consecutive_flags, 3)
    return {
        "onset_s": float(onset_s),
        "guard_s": float(guard_s),
        "threshold": float(threshold),
        "pre_windows": int(pre.sum()),
        "pre_false_flags": int((flags & pre).sum()),
        "pre_false_positive_rate": float((flags & pre).sum() / int(pre.sum())),
        "post_windows": int(post.sum()),
        "post_detection_flags": int((flags & post).sum()),
        "post_detection_rate": float((flags & post).sum() / int(post.sum())),
        "first_detection_score_time_s": first,
        "first_detection_delay_s": None if first is None else float(first - onset_s),
        "first_detection_available_delay_s": (
            None if first is None else float(first + availability_offset_s - onset_s)
        ),
        "first_three_consecutive_score_time_s": third,
        "first_three_consecutive_delay_s": None if third is None else float(third - onset_s),
        "first_three_consecutive_available_delay_s": (
            None if third is None else float(third + availability_offset_s - onset_s)
        ),
    }
