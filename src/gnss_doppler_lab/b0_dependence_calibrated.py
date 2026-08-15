"""B0-CS clean-only calibration and dependence-aware receiver evidence.

The module is deliberately label-free.  Scenario labels and attack timelines
belong in the stage runner; every primitive here operates on causal features,
residuals, nuisance context, and recording identities only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

TAP_ORDER = ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
FEATURE_COLUMNS = tuple(f"tap_{tap}_rel_prompt_mean" for tap in TAP_ORDER)
ROLE_ORDER = ("train", "validation", "calibration", "holdout")
COUNT_BINS = ((4, 6), (7, 9), (10, None))
EPSILON = 1e-12


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unavailable(reason: str, **details: object) -> dict[str, object]:
    return {"status": "UNAVAILABLE", "reason": str(reason), "details": details}


def safe_prompt_normalize(
    complex_iq: np.ndarray, *, epsilon: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Return prompt-normalized magnitudes and a low-power validity mask."""
    values = np.asarray(complex_iq)
    if values.ndim != 3 or values.shape[1:] != (9, 2):
        raise ValueError("complex_iq must have shape [N,9,2]")
    if not np.isfinite(values).all() or epsilon <= 0:
        raise ValueError("complex_iq must be finite and epsilon positive")
    magnitude = np.hypot(values[:, :, 0], values[:, :, 1])
    prompt = magnitude[:, 4]
    valid = np.isfinite(prompt) & (prompt > epsilon)
    normalized = np.zeros_like(magnitude, dtype=np.float64)
    normalized[valid] = magnitude[valid] / prompt[valid, None]
    if not np.isfinite(normalized).all():
        raise ValueError("prompt normalization produced non-finite values")
    return normalized, valid


def _required_npz_arrays(archive: np.lib.npyio.NpzFile) -> None:
    required = {
        "complex_iq", "prn", "time_s", "segment_index", "channel",
        "sample_count", "cn0_db_hz",
    }
    missing = sorted(required - set(archive.files))
    if missing:
        raise ValueError(f"source NPZ missing arrays: {missing}")


def build_node_windows_from_npz(
    path: str | Path,
    *,
    recording_id: str,
    window_seconds: float = 1.0,
    stride_seconds: float = 0.5,
    cn0_lag_seconds: float = 1.0,
    prompt_epsilon: float = 1e-8,
    bytes_per_complex_sample: int = 4,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build causal Method-A nodes while preserving C/N0 and sample lineage."""
    source = Path(path).resolve(strict=True)
    if window_seconds <= 0 or stride_seconds <= 0 or cn0_lag_seconds <= 0:
        raise ValueError("window, stride, and C/N0 lag must be positive")
    if bytes_per_complex_sample <= 0:
        raise ValueError("bytes_per_complex_sample must be positive")
    archive = np.load(source, allow_pickle=False)
    _required_npz_arrays(archive)
    norm, prompt_valid = safe_prompt_normalize(
        np.asarray(archive["complex_iq"]), epsilon=prompt_epsilon
    )
    n = len(norm)
    arrays = {
        "prn": np.asarray(archive["prn"]),
        "time_s": np.asarray(archive["time_s"], dtype=float),
        "segment": np.asarray(archive["segment_index"]),
        "channel": np.asarray(archive["channel"]),
        "sample_count": np.asarray(archive["sample_count"], dtype=np.uint64),
        "cn0_db_hz": np.asarray(archive["cn0_db_hz"], dtype=float),
    }
    if any(len(values) != n for values in arrays.values()):
        raise ValueError("source NPZ arrays have inconsistent lengths")
    if not np.isfinite(arrays["time_s"]).all():
        raise ValueError("source times must be finite")

    identity = pd.DataFrame({
        "prn": arrays["prn"], "segment": arrays["segment"],
        "channel": arrays["channel"],
    }).drop_duplicates()
    rows: list[dict[str, object]] = []
    low_power_epochs = int((~prompt_valid).sum())
    short_or_unlagged_windows = 0
    gaps_detected = 0
    for prn_value, segment_value, channel_value in identity.itertuples(index=False, name=None):
        selected = np.flatnonzero(
            (arrays["prn"] == prn_value)
            & (arrays["segment"] == segment_value)
            & (arrays["channel"] == channel_value)
        )
        order = selected[np.argsort(arrays["time_s"][selected], kind="mergesort")]
        times = arrays["time_s"][order]
        if len(times) < 2:
            continue
        positive_diffs = np.diff(times)
        positive_diffs = positive_diffs[positive_diffs > 0]
        if not len(positive_diffs):
            continue
        cadence = float(np.median(positive_diffs))
        breaks = np.flatnonzero(np.diff(times) > 1.5 * cadence) + 1
        gaps_detected += int(len(breaks))
        cuts = np.r_[0, breaks, len(order)]
        for chunk_index, (begin, finish) in enumerate(zip(cuts[:-1], cuts[1:])):
            local_index = order[int(begin):int(finish)]
            local_time = arrays["time_s"][local_index]
            if len(local_time) < 2:
                continue
            start = float(local_time.min())
            stop_limit = float(local_time.max())
            window_index = 0
            while start + window_seconds <= stop_limit + cadence + 1e-9:
                left = int(np.searchsorted(local_time, start, side="left"))
                right = int(np.searchsorted(local_time, start + window_seconds, side="left"))
                lag_left = int(np.searchsorted(local_time, start - cn0_lag_seconds, side="left"))
                lag_right = left
                current_index = local_index[left:right]
                lag_index = local_index[lag_left:lag_right]
                coverage = (
                    len(current_index) > 0
                    and arrays["time_s"][current_index].min() <= start + 1.5 * cadence
                    and arrays["time_s"][current_index].max()
                    >= start + window_seconds - 1.5 * cadence
                )
                valid_current = current_index[prompt_valid[current_index]]
                finite_lag = lag_index[np.isfinite(arrays["cn0_db_hz"][lag_index])]
                if not coverage or not len(valid_current) or not len(finite_lag):
                    short_or_unlagged_windows += 1
                    start += stride_seconds
                    window_index += 1
                    continue
                sample_values = arrays["sample_count"][current_index].astype(np.uint64)
                sample_diffs = np.diff(np.sort(sample_values.astype(np.int64)))
                sample_diffs = sample_diffs[sample_diffs > 0]
                sample_step = int(np.median(sample_diffs)) if len(sample_diffs) else 1
                sample_start = int(sample_values.min())
                sample_end = int(sample_values.max()) + sample_step
                feature_values = norm[valid_current].mean(axis=0)
                prn_text = (
                    f"G{int(prn_value):02d}"
                    if str(prn_value).lstrip("+-").isdigit() else str(prn_value)
                )
                row: dict[str, object] = {
                    "physical_recording_id": str(recording_id),
                    "run_id": str(recording_id),
                    "prn": prn_text,
                    "segment": str(segment_value),
                    "channel": str(channel_value),
                    "history_chunk": int(chunk_index),
                    "window_start_s": float(start),
                    "window_end_s": float(start + window_seconds),
                    "window_mid_s": float(start + window_seconds / 2),
                    "window_bin_s": float(np.floor((start + window_seconds / 2) * 2 + .5) / 2),
                    "window_index": int(window_index),
                    "raw_epoch_count": int(len(current_index)),
                    "valid_prompt_epoch_count": int(len(valid_current)),
                    "prompt_valid_fraction": float(len(valid_current) / len(current_index)),
                    "lagged_cn0_db_hz": float(np.median(arrays["cn0_db_hz"][finite_lag])),
                    "raw_sample_start": sample_start,
                    "raw_sample_end_exclusive": sample_end,
                    "raw_byte_start": sample_start * bytes_per_complex_sample,
                    "raw_byte_end_exclusive": sample_end * bytes_per_complex_sample,
                }
                row.update({name: float(feature_values[i]) for i, name in enumerate(FEATURE_COLUMNS)})
                rows.append(row)
                start += stride_seconds
                window_index += 1
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("source produced no causal node windows")
    duplicate_key = [
        "physical_recording_id", "prn", "segment", "channel", "history_chunk", "window_bin_s"
    ]
    if frame.duplicated(duplicate_key).any():
        raise ValueError("duplicate node identity/window after conversion")
    numeric = frame[[
        "window_start_s", "window_end_s", "window_bin_s", "lagged_cn0_db_hz", *FEATURE_COLUMNS
    ]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("node conversion produced non-finite values")
    audit = {
        "schema": "gnss-doppler-lab.b0-cs-node-conversion.v1",
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "recording_id": str(recording_id),
        "input_raw_epochs": int(n),
        "node_rows": int(len(frame)),
        "prn_count": int(frame.prn.nunique()),
        "low_prompt_raw_epochs": low_power_epochs,
        "dropped_short_or_unlagged_windows": int(short_or_unlagged_windows),
        "gaps_detected": int(gaps_detected),
        "feature_columns": list(FEATURE_COLUMNS),
        "prn_identity_input": False,
        "cn0_context": "strictly lagged median over preceding one second",
        "sample_and_byte_lineage_preserved": True,
    }
    return frame.sort_values(["window_bin_s", "prn", "segment", "channel"], kind="mergesort").reset_index(drop=True), audit


def receiver_epoch_counts(nodes: pd.DataFrame) -> pd.DataFrame:
    required = {"physical_recording_id", "window_bin_s", "prn"}
    missing = sorted(required - set(nodes))
    if missing:
        raise ValueError(f"node table missing receiver epoch fields: {missing}")
    return (nodes.groupby(["physical_recording_id", "window_bin_s"], as_index=False, sort=True)
            .agg(tracked_prn_count=("prn", "nunique")))


def stable_tracking_start(
    epochs: pd.DataFrame, *, minimum_prns: int = 4, consecutive_epochs: int = 20, cadence_s: float = .5
) -> float:
    if minimum_prns < 1 or consecutive_epochs < 1 or cadence_s <= 0:
        raise ValueError("invalid stable tracking rule")
    if epochs.physical_recording_id.nunique() != 1:
        raise ValueError("stable tracking start requires one recording")
    ordered = epochs.sort_values("window_bin_s", kind="mergesort")
    times = ordered.window_bin_s.to_numpy(float)
    counts = ordered.tracked_prn_count.to_numpy(int)
    for index in range(0, len(times) - consecutive_epochs + 1):
        local_times = times[index:index + consecutive_epochs]
        if (np.all(counts[index:index + consecutive_epochs] >= minimum_prns)
                and np.allclose(np.diff(local_times), cadence_s, atol=1e-9, rtol=0)):
            return float(local_times[0])
    raise ValueError("no stable tracking interval satisfies the frozen rule")


def _hamilton_counts(total: int, ratios: Sequence[float]) -> list[int]:
    weights = np.asarray(ratios, dtype=float)
    if total < len(weights) or np.any(weights <= 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("invalid Hamilton allocation")
    quotas = total * weights
    counts = np.floor(quotas).astype(int)
    remainder = int(total - counts.sum())
    order = sorted(range(len(weights)), key=lambda index: (-(quotas[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return counts.tolist()


def chronological_role_split(
    nodes: pd.DataFrame,
    *,
    ratios: Sequence[float] = (.50, .15, .20, .15),
    guard_seconds: float = 6.0,
    cadence_s: float = .5,
    minimum_prns: int = 4,
    stable_epochs: int = 20,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Assign receiver target epochs once, with whole-epoch chronological guards."""
    epochs = receiver_epoch_counts(nodes)
    start = stable_tracking_start(
        epochs, minimum_prns=minimum_prns, consecutive_epochs=stable_epochs, cadence_s=cadence_s
    )
    eligible = epochs[epochs.window_bin_s >= start].sort_values("window_bin_s", kind="mergesort")
    times = eligible.window_bin_s.to_numpy(float)
    if len(times) < 4 or not np.allclose(np.diff(times), cadence_s, atol=1e-9, rtol=0):
        raise ValueError("eligible receiver epoch grid is not contiguous")
    guard_epochs = int(math.ceil(guard_seconds / cadence_s))
    role_total = len(times) - 3 * guard_epochs
    counts = _hamilton_counts(role_total, ratios)
    cursor = 0
    roles: dict[str, pd.DataFrame] = {}
    intervals: dict[str, dict[str, object]] = {}
    target_sets: dict[str, set[float]] = {}
    for role_index, (role, count) in enumerate(zip(ROLE_ORDER, counts)):
        selected_times = times[cursor:cursor + count]
        target_sets[role] = set(float(value) for value in selected_times)
        local = nodes[nodes.window_bin_s.astype(float).isin(target_sets[role])].copy()
        local["role"] = role
        roles[role] = local
        intervals[role] = {
            "target_epochs": int(len(selected_times)),
            "node_rows": int(len(local)),
            "first_window_bin_s": float(selected_times[0]),
            "last_window_bin_s": float(selected_times[-1]),
        }
        cursor += count
        if role_index < len(ROLE_ORDER) - 1:
            cursor += guard_epochs
    if cursor != len(times):
        raise AssertionError("split allocation did not consume eligible epoch grid")
    for left, right in zip(ROLE_ORDER[:-1], ROLE_ORDER[1:]):
        if target_sets[left] & target_sets[right]:
            raise AssertionError("target role overlap")
    audit = audit_role_separation(roles)
    audit.update({
        "schema": "gnss-doppler-lab.b0-cs-split-audit.v1",
        "stable_tracking_start_window_bin_s": float(start),
        "stable_rule": {"minimum_prns": minimum_prns, "consecutive_epochs": stable_epochs},
        "ratios": dict(zip(ROLE_ORDER, map(float, ratios))),
        "guard_seconds": float(guard_seconds),
        "guard_epochs": int(guard_epochs),
        "allocation": "Hamilton largest remainder; role-order tie break",
        "roles": intervals,
    })
    return roles, audit


def audit_role_separation(roles: Mapping[str, pd.DataFrame]) -> dict[str, object]:
    if set(roles) != set(ROLE_ORDER):
        raise ValueError("roles must be train, validation, calibration, holdout")
    target_keys: dict[str, set[tuple[str, float]]] = {}
    sample_ranges: dict[str, dict[str, int | None]] = {}
    content_hashes: dict[str, str] = {}
    for role in ROLE_ORDER:
        frame = roles[role]
        keys = set(zip(frame.physical_recording_id.astype(str), frame.window_bin_s.astype(float)))
        target_keys[role] = keys
        sample_ranges[role] = {
            "raw_sample_start": None if frame.empty else int(frame.raw_sample_start.min()),
            "raw_sample_end_exclusive": None if frame.empty else int(frame.raw_sample_end_exclusive.max()),
            "raw_byte_start": None if frame.empty else int(frame.raw_byte_start.min()),
            "raw_byte_end_exclusive": None if frame.empty else int(frame.raw_byte_end_exclusive.max()),
        }
        hashed = pd.util.hash_pandas_object(
            frame.sort_values(["physical_recording_id", "window_bin_s", "prn"], kind="mergesort"),
            index=False,
        ).to_numpy(np.uint64)
        content_hashes[role] = hashlib.sha256(hashed.tobytes()).hexdigest()
    overlaps: dict[str, int] = {}
    raw_disjoint = True
    for left_index, left in enumerate(ROLE_ORDER):
        for right in ROLE_ORDER[left_index + 1:]:
            overlaps[f"{left}:{right}"] = len(target_keys[left] & target_keys[right])
    for left, right in zip(ROLE_ORDER[:-1], ROLE_ORDER[1:]):
        left_end = sample_ranges[left]["raw_sample_end_exclusive"]
        right_start = sample_ranges[right]["raw_sample_start"]
        if left_end is not None and right_start is not None and int(left_end) > int(right_start):
            raw_disjoint = False
    return {
        "target_epoch_overlap_counts": overlaps,
        "no_target_epoch_overlap": not any(overlaps.values()),
        "raw_sample_and_byte_ranges": sample_ranges,
        "no_raw_sample_or_byte_interval_overlap": raw_disjoint,
        "role_content_sha256": content_hashes,
        "predictor_fit_roles": ["train"],
        "checkpoint_selection_roles": ["validation"],
        "calibrator_threshold_roles": ["calibration"],
        "sealed_evaluation_roles": ["holdout"],
        "normal_only": True,
        "attack_labels_used": False,
    }


def fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not len(array):
        raise ValueError("standardizer requires a nonempty matrix")
    mean = np.nanmean(array, axis=0).astype(np.float32)
    stdev = np.nanstd(array, axis=0).astype(np.float32)
    mean[~np.isfinite(mean)] = 0
    stdev[~np.isfinite(stdev) | (stdev < 1e-6)] = 1
    return mean, stdev


def standardize(values: np.ndarray, mean: np.ndarray, stdev: np.ndarray) -> np.ndarray:
    result = (np.asarray(values, dtype=np.float32) - np.asarray(mean, dtype=np.float32)) / np.asarray(stdev, dtype=np.float32)
    result[~np.isfinite(result)] = 0
    return result.astype(np.float32)


def causal_examples(
    frame: pd.DataFrame,
    mean: np.ndarray,
    stdev: np.ndarray,
    *,
    seq_len: int = 12,
    cadence_s: float = .5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    """Create role-local PRN histories and reset at every identity/gap boundary."""
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    required = {
        "role", "physical_recording_id", "segment", "channel", "prn", "window_bin_s",
        "lagged_cn0_db_hz", *FEATURE_COLUMNS,
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"causal example input missing: {missing}")
    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    dropped_chunks = 0
    gaps = 0
    group_columns = ["role", "physical_recording_id", "segment", "channel", "prn"]
    for identity, group in frame.sort_values([*group_columns, "window_bin_s"], kind="mergesort").groupby(group_columns, sort=True):
        bins = group.window_bin_s.to_numpy(float)
        breaks = np.flatnonzero(~np.isclose(np.diff(bins), cadence_s, atol=1e-9, rtol=0)) + 1
        gaps += int(len(breaks))
        cuts = np.r_[0, breaks, len(group)]
        for chunk_index, (begin, finish) in enumerate(zip(cuts[:-1], cuts[1:])):
            chunk = group.iloc[int(begin):int(finish)].reset_index(drop=True)
            if len(chunk) <= seq_len:
                dropped_chunks += 1
                continue
            values = standardize(chunk.loc[:, FEATURE_COLUMNS].to_numpy(float), mean, stdev)
            for target_index in range(seq_len, len(chunk)):
                target = chunk.iloc[target_index]
                sequences.append(values[target_index - seq_len:target_index])
                targets.append(values[target_index])
                metadata.append({
                    "role": str(identity[0]),
                    "physical_recording_id": str(identity[1]),
                    "segment": str(identity[2]),
                    "channel": str(identity[3]),
                    "prn": str(identity[4]),
                    "history_chunk": int(chunk_index),
                    "target_window_index": int(target_index),
                    "window_start_s": float(target.window_start_s),
                    "window_end_s": float(target.window_end_s),
                    "window_bin_s": float(target.window_bin_s),
                    "lagged_cn0_db_hz": float(target.lagged_cn0_db_hz),
                    "raw_sample_start": int(target.raw_sample_start),
                    "raw_sample_end_exclusive": int(target.raw_sample_end_exclusive),
                })
    feature_count = len(FEATURE_COLUMNS)
    x = np.stack(sequences).astype(np.float32) if sequences else np.empty((0, seq_len, feature_count), np.float32)
    y = np.stack(targets).astype(np.float32) if targets else np.empty((0, feature_count), np.float32)
    meta = pd.DataFrame(metadata)
    audit = {
        "seq_len": int(seq_len), "cadence_seconds": float(cadence_s),
        "examples": int(len(x)), "gaps_detected": int(gaps),
        "short_chunks_dropped": int(dropped_chunks),
        "reset_dimensions": group_columns + ["cadence_gap"],
        "first_target_index_is_seq_len": bool(meta.empty or meta.groupby(
            ["role", "physical_recording_id", "segment", "channel", "prn", "history_chunk"]
        ).head(1).target_window_index.eq(seq_len).all()),
        "causal_no_lookahead": True,
    }
    return x, y, meta, audit


def residual_frame(metadata: pd.DataFrame, target: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    actual = np.asarray(target, dtype=float)
    estimate = np.asarray(prediction, dtype=float)
    if actual.shape != estimate.shape or actual.ndim != 2 or len(metadata) != len(actual):
        raise ValueError("residual inputs have inconsistent shapes")
    result = metadata.reset_index(drop=True).copy()
    residual = actual - estimate
    result["b0_residual_rmse"] = np.sqrt(np.mean(residual ** 2, axis=1))
    for index in range(residual.shape[1]):
        result[f"residual_{index:03d}"] = residual[:, index]
    return result


def attach_tracked_count(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    counts = (result.groupby(["physical_recording_id", "window_bin_s"])["prn"]
              .transform("nunique").astype(int))
    result["tracked_prn_count"] = counts
    return result


def cn0_tertile_edges(train_lagged_cn0: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(train_lagged_cn0, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("finite train C/N0 is required")
    edges = np.quantile(values, [1 / 3, 2 / 3])
    if not np.isfinite(edges).all():
        raise ValueError("C/N0 tertile edges are non-finite")
    return float(edges[0]), float(edges[1])


def cn0_bin(values: Sequence[float], edges: Sequence[float]) -> np.ndarray:
    edge = np.asarray(edges, dtype=float)
    query = np.asarray(values, dtype=float)
    if edge.shape != (2,) or not np.isfinite(edge).all() or edge[0] > edge[1]:
        raise ValueError("two ordered finite C/N0 edges required")
    if not np.isfinite(query).all():
        raise ValueError("finite lagged C/N0 required")
    return np.searchsorted(edge, query, side="right").astype(int)


def count_bin(values: Sequence[int]) -> np.ndarray:
    counts = np.asarray(values, dtype=int)
    result = np.full(counts.shape, -1, dtype=int)
    result[(counts >= 4) & (counts <= 6)] = 0
    result[(counts >= 7) & (counts <= 9)] = 1
    result[counts >= 10] = 2
    return result


def conformal_pvalues(calibration: Sequence[float], query: Sequence[float]) -> np.ndarray:
    reference = np.sort(np.asarray(calibration, dtype=float))
    scores = np.asarray(query, dtype=float)
    if reference.ndim != 1 or not len(reference) or not np.isfinite(reference).all():
        raise ValueError("finite nonempty one-dimensional calibration required")
    if not np.isfinite(scores).all():
        raise ValueError("finite conformal query required")
    upper = len(reference) - np.searchsorted(reference, scores, side="left")
    return (1 + upper) / (len(reference) + 1.0)


def power_evalues(pvalues: Sequence[float]) -> np.ndarray:
    pvalue = np.asarray(pvalues, dtype=float)
    if np.any(~np.isfinite(pvalue)) or np.any(pvalue <= 0) or np.any(pvalue > 1):
        raise ValueError("p-values must be finite in (0,1]")
    return 0.5 * np.power(pvalue, -0.5)


@dataclass(frozen=True)
class StratumEntry:
    cn0_bins: tuple[int, ...]
    count_bins: tuple[int, ...]
    calibration: np.ndarray
    block_count: int
    merge_level: str
    sufficient: bool


@dataclass(frozen=True)
class StratumCalibrator:
    cn0_edges: tuple[float, float]
    block_seconds: float
    minimum_blocks: int
    entries: Mapping[tuple[int, int], StratumEntry]


def _adjacent_candidates(target: int) -> list[tuple[int, ...]]:
    if target == 0:
        return [(0,), (0, 1), (0, 1, 2)]
    if target == 1:
        return [(1,), (0, 1), (0, 1, 2)]
    if target == 2:
        return [(2,), (1, 2), (0, 1, 2)]
    raise ValueError("C/N0 bin must be 0, 1, or 2")


def fit_stratum_calibrator(
    calibration_residuals: pd.DataFrame,
    *,
    cn0_edges: Sequence[float],
    block_seconds: float,
    minimum_blocks: int = 20,
) -> StratumCalibrator:
    if block_seconds <= 0 or minimum_blocks < 1:
        raise ValueError("invalid stratum calibration rule")
    frame = attach_tracked_count(calibration_residuals)
    frame = frame[frame.tracked_prn_count >= 4].copy()
    if frame.empty:
        raise ValueError("calibration has no receiver epochs with at least four PRNs")
    frame["cn0_bin"] = cn0_bin(frame.lagged_cn0_db_hz, cn0_edges)
    frame["count_bin"] = count_bin(frame.tracked_prn_count)
    origin = frame.groupby("physical_recording_id").window_bin_s.transform("min")
    frame["calibration_block"] = np.floor((frame.window_bin_s - origin) / block_seconds).astype(int)

    entries: dict[tuple[int, int], StratumEntry] = {}
    for target_cn0 in range(3):
        for target_count in range(3):
            selected: pd.DataFrame | None = None
            selected_cn0: tuple[int, ...] = ()
            selected_count = (target_count,)
            merge_level = "exact"
            sufficient = False
            for candidate_index, candidate_cn0 in enumerate(_adjacent_candidates(target_cn0)):
                trial = frame[
                    frame.cn0_bin.isin(candidate_cn0) & frame.count_bin.eq(target_count)
                ]
                block_count_value = trial[["physical_recording_id", "calibration_block"]].drop_duplicates().shape[0]
                selected = trial
                selected_cn0 = candidate_cn0
                merge_level = ("exact" if candidate_index == 0 else
                               "adjacent_cn0" if len(candidate_cn0) == 2 else
                               "all_cn0_preserve_count")
                if block_count_value >= minimum_blocks:
                    sufficient = True
                    break
            if selected is None or selected.empty or not sufficient:
                selected = frame
                selected_cn0 = (0, 1, 2)
                selected_count = (0, 1, 2)
                merge_level = "global"
                block_count_value = selected[["physical_recording_id", "calibration_block"]].drop_duplicates().shape[0]
                sufficient = block_count_value >= minimum_blocks
            values = np.sort(selected.b0_residual_rmse.to_numpy(float))
            entries[(target_cn0, target_count)] = StratumEntry(
                tuple(selected_cn0), tuple(selected_count), values,
                int(block_count_value), merge_level, bool(sufficient),
            )
    return StratumCalibrator(tuple(map(float, cn0_edges)), float(block_seconds), int(minimum_blocks), entries)


def calibrator_to_dict(calibrator: StratumCalibrator, *, include_values: bool = True) -> dict[str, object]:
    entries: dict[str, object] = {}
    for key, entry in sorted(calibrator.entries.items()):
        record: dict[str, object] = {
            "query_cn0_bin": key[0], "query_count_bin": key[1],
            "source_cn0_bins": list(entry.cn0_bins), "source_count_bins": list(entry.count_bins),
            "calibration_n": int(len(entry.calibration)), "nonoverlapping_block_count": entry.block_count,
            "merge_level": entry.merge_level, "minimum_block_rule_satisfied": entry.sufficient,
            "minimum_pvalue": float(1 / (len(entry.calibration) + 1)),
        }
        if include_values:
            record["calibration_values"] = entry.calibration.tolist()
        entries[f"cn0{key[0]}_n{key[1]}"] = record
    return {
        "cn0_edges": list(calibrator.cn0_edges), "block_seconds": calibrator.block_seconds,
        "minimum_blocks": calibrator.minimum_blocks, "prn_identity_stratum": False,
        "entries": entries,
    }


def calibrator_from_dict(document: Mapping[str, object]) -> StratumCalibrator:
    entries: dict[tuple[int, int], StratumEntry] = {}
    raw_entries = document["entries"]
    if not isinstance(raw_entries, Mapping):
        raise ValueError("invalid calibrator entries")
    for raw in raw_entries.values():
        if not isinstance(raw, Mapping) or "calibration_values" not in raw:
            raise ValueError("serialized calibrator lacks calibration values")
        key = (int(raw["query_cn0_bin"]), int(raw["query_count_bin"]))
        entries[key] = StratumEntry(
            tuple(map(int, raw["source_cn0_bins"])), tuple(map(int, raw["source_count_bins"])),
            np.asarray(raw["calibration_values"], dtype=float),
            int(raw["nonoverlapping_block_count"]), str(raw["merge_level"]),
            bool(raw["minimum_block_rule_satisfied"]),
        )
    return StratumCalibrator(
        tuple(map(float, document["cn0_edges"])), float(document["block_seconds"]),
        int(document["minimum_blocks"]), entries,
    )


def score_prn_evidence(
    residuals: pd.DataFrame,
    calibrator: StratumCalibrator,
    *,
    nuisance_conditioned: bool = True,
) -> pd.DataFrame:
    frame = attach_tracked_count(residuals)
    frame["cn0_bin"] = cn0_bin(frame.lagged_cn0_db_hz, calibrator.cn0_edges)
    frame["count_bin"] = count_bin(frame.tracked_prn_count)
    frame["conformal_pvalue"] = 1.0
    frame["prn_evalue"] = 0.5
    frame["calibration_key"] = "suppressed_n_lt_4"
    valid = frame.tracked_prn_count >= 4
    if nuisance_conditioned:
        for key, entry in calibrator.entries.items():
            mask = valid & frame.cn0_bin.eq(key[0]) & frame.count_bin.eq(key[1])
            if mask.any():
                pvalue = conformal_pvalues(entry.calibration, frame.loc[mask, "b0_residual_rmse"])
                frame.loc[mask, "conformal_pvalue"] = pvalue
                frame.loc[mask, "prn_evalue"] = power_evalues(pvalue)
                frame.loc[mask, "calibration_key"] = f"cn0{key[0]}_n{key[1]}:{entry.merge_level}"
    else:
        global_values = np.sort(np.concatenate([
            entry.calibration for entry in calibrator.entries.values()
            if entry.merge_level == "global"
        ]))
        if not len(global_values):
            global_values = np.sort(np.concatenate([entry.calibration for entry in calibrator.entries.values()]))
        global_values = np.unique(global_values)
        pvalue = conformal_pvalues(global_values, frame.loc[valid, "b0_residual_rmse"])
        frame.loc[valid, "conformal_pvalue"] = pvalue
        frame.loc[valid, "prn_evalue"] = power_evalues(pvalue)
        frame.loc[valid, "calibration_key"] = "global_no_nuisance"
    return frame


def robust_pool(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("finite nonempty residual set required")
    median = float(np.median(array))
    return median + 1.4826 * float(np.median(np.abs(array - median)))


def aggregate_receiver_scores(scored_prns: pd.DataFrame) -> pd.DataFrame:
    required = {
        "physical_recording_id", "window_bin_s", "window_end_s", "prn",
        "b0_residual_rmse", "conformal_pvalue", "prn_evalue",
    }
    missing = sorted(required - set(scored_prns))
    if missing:
        raise ValueError(f"scored PRNs missing receiver fields: {missing}")
    if scored_prns.duplicated(["physical_recording_id", "window_bin_s", "prn"]).any():
        raise ValueError("duplicate PRN within receiver epoch")
    rows: list[dict[str, object]] = []
    for (recording, epoch), group in scored_prns.groupby(
        ["physical_recording_id", "window_bin_s"], sort=True
    ):
        n = int(group.prn.nunique())
        valid = n >= 4
        evalues = group.prn_evalue.to_numpy(float)
        pvalues = group.conformal_pvalue.to_numpy(float)
        residuals = group.b0_residual_rmse.to_numpy(float)
        rows.append({
            "physical_recording_id": str(recording), "window_bin_s": float(epoch),
            "availability_time_s": float(group.window_end_s.max()), "tracked_prn_count": n,
            "a0_robust_pool": robust_pool(residuals),
            "residual_max": float(residuals.max()), "residual_mean": float(residuals.mean()),
            "mean_prn_evalue": float(evalues.mean()) if valid else np.nan,
            "set_score": float(np.log(evalues.mean() + EPSILON)) if valid else np.nan,
            "min_prn_pvalue": float(pvalues.min()) if valid else np.nan,
            "median_prn_pvalue": float(np.median(pvalues)) if valid else np.nan,
            "score_valid": bool(valid),
        })
    return pd.DataFrame(rows).sort_values(
        ["physical_recording_id", "window_bin_s"], kind="mergesort"
    ).reset_index(drop=True)


def integrated_autocorrelation_time(
    values: Sequence[float], *, cadence_seconds: float = .5, max_lag: int | None = None
) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 4 or cadence_seconds <= 0:
        raise ValueError("IAT requires at least four finite values and positive cadence")
    centered = array - array.mean()
    variance = float(np.dot(centered, centered) / len(centered))
    if variance <= 0:
        return {"iat_epochs": 1.0, "iat_seconds": float(cadence_seconds), "positive_lags": 0}
    limit = min(len(array) // 4, 200) if max_lag is None else min(int(max_lag), len(array) - 1)
    positive: list[float] = []
    for lag in range(1, limit + 1):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / ((len(centered) - lag) * variance))
        if not np.isfinite(rho) or rho <= 0:
            break
        positive.append(rho)
    iat_epochs = float(1 + 2 * sum(positive))
    return {"iat_epochs": iat_epochs, "iat_seconds": iat_epochs * cadence_seconds,
            "positive_lags": len(positive)}


def choose_block_seconds(iat_seconds: float, *, default_seconds: float = 2.0, cadence_seconds: float = .5) -> float:
    if not np.isfinite(iat_seconds) or iat_seconds <= 0:
        raise ValueError("positive finite IAT required")
    if iat_seconds <= default_seconds:
        return float(default_seconds)
    return float((math.floor(iat_seconds / cadence_seconds) + 1) * cadence_seconds)


def receiver_blocks(receiver_scores: pd.DataFrame, *, block_seconds: float) -> pd.DataFrame:
    if block_seconds <= 0:
        raise ValueError("block_seconds must be positive")
    valid = receiver_scores[receiver_scores.score_valid & receiver_scores.set_score.notna()].copy()
    rows: list[dict[str, object]] = []
    for recording, group in valid.groupby("physical_recording_id", sort=True):
        group = group.sort_values("window_bin_s", kind="mergesort")
        origin = float(group.window_bin_s.min())
        group["block_index"] = np.floor((group.window_bin_s - origin) / block_seconds).astype(int)
        for block_index, block in group.groupby("block_index", sort=True):
            rows.append({
                "physical_recording_id": str(recording), "block_index": int(block_index),
                "block_start_s": float(origin + block_index * block_seconds),
                "block_end_s": float(origin + (block_index + 1) * block_seconds),
                "block_score": float(block.set_score.max()), "epoch_count": int(len(block)),
                "tracked_prn_count_min": int(block.tracked_prn_count.min()),
                "tracked_prn_count_max": int(block.tracked_prn_count.max()),
            })
    return pd.DataFrame(rows)


def higher_quantile(values: Sequence[float], probability: float) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    if not len(array) or not np.isfinite(array).all() or not 0 <= probability <= 1:
        raise ValueError("finite nonempty values and probability in [0,1] required")
    index = int(math.ceil(probability * (len(array) - 1)))
    return float(array[index])


def score_block_evidence(blocks: pd.DataFrame, calibration_block_scores: Sequence[float]) -> pd.DataFrame:
    result = blocks.copy()
    result["block_pvalue"] = conformal_pvalues(calibration_block_scores, result.block_score)
    result["block_evalue"] = power_evalues(result.block_pvalue)
    capitals = np.empty(len(result), dtype=float)
    alarms = np.empty(len(result), dtype=bool)
    for _, indices in result.groupby("physical_recording_id", sort=False).groups.items():
        capital = 1.0
        for index in indices:
            capital = max(1.0, capital) * float(result.at[index, "block_evalue"])
            if not np.isfinite(capital):
                capital = float(np.finfo(np.float64).max)
            capitals[index] = capital
            alarms[index] = capital >= 100.0
    result["e_cusum"] = capitals
    result["alarm"] = alarms
    return result


def sequential_e_cusum(evalues: Sequence[float], run_ids: Sequence[str] | None = None) -> np.ndarray:
    values = np.asarray(evalues, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("finite nonnegative e-values required")
    runs = np.asarray(run_ids if run_ids is not None else ["run"] * len(values)).astype(str)
    if len(runs) != len(values):
        raise ValueError("run IDs and e-values must align")
    result = np.empty(len(values), dtype=float)
    previous_run: str | None = None
    capital = 1.0
    for index, (value, run) in enumerate(zip(values, runs)):
        if run != previous_run:
            capital = 1.0
            previous_run = run
        capital = max(1.0, capital) * float(value)
        result[index] = capital
    return result


def consecutive_alarm(flags: Sequence[bool], *, consecutive_epochs: int = 3) -> np.ndarray:
    values = np.asarray(flags, dtype=bool)
    if consecutive_epochs < 1:
        raise ValueError("consecutive_epochs must be positive")
    result = np.zeros(len(values), dtype=bool)
    run = 0
    for index, value in enumerate(values):
        run = run + 1 if value else 0
        result[index] = run >= consecutive_epochs
    return result


def scenario_family(scenario: str) -> str:
    name = str(scenario).upper()
    if name in {"DS7", "DS8"}:
        return "DS7-DS8"
    return name


def official_timeline(scenario: str) -> dict[str, float]:
    timelines = {
        "DS1": {"signal_onset": 125.0},
        "DS2": {"signal_onset": 110.1},
        "DS3": {"signal_onset": 118.9, "pull_off": 195.0},
        "DS4": {"signal_onset": 113.8, "pull_off": 225.0},
        "DS7": {"signal_onset": 110.0, "time_push": 150.0},
        "DS8": {"signal_onset": 110.0, "time_push": 150.0},
    }
    name = str(scenario).upper()
    if name not in timelines:
        raise ValueError(f"no frozen timeline for {scenario}")
    return dict(timelines[name])


def paired_block_bootstrap(
    labels: Sequence[int],
    first: Sequence[float],
    second: Sequence[float],
    times: Sequence[float],
    *,
    metric,
    block_seconds: float = 10.0,
    repetitions: int = 2000,
    seed: int = 20260816,
) -> np.ndarray:
    """Paired moving-block bootstrap; never falls back to IID resampling."""
    label = np.asarray(labels)
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    timestamp = np.asarray(times, dtype=float)
    if not (len(label) == len(a) == len(b) == len(timestamp)) or not len(label):
        raise ValueError("paired bootstrap arrays must be nonempty and aligned")
    if repetitions < 1 or block_seconds <= 0:
        raise ValueError("invalid paired block bootstrap configuration")
    block_id = np.floor((timestamp - timestamp.min()) / block_seconds).astype(int)
    blocks = [np.flatnonzero(block_id == value) for value in np.unique(block_id)]
    if not blocks:
        raise ValueError("no bootstrap blocks")
    rng = np.random.default_rng(seed)
    output = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        index = np.concatenate([blocks[value] for value in chosen])
        output[repetition] = float(metric(label[index], a[index]) - metric(label[index], b[index]))
    return output


def artifact_manifest(root: str | Path) -> dict[str, object]:
    directory = Path(root)
    manifest_path = directory / "artifact_manifest_sha256.json"
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path != manifest_path:
            records.append({
                "path": str(path.relative_to(directory)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })
    return {"schema": "gnss-doppler-lab.b0-cs-artifact-manifest.v1", "files": records}


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
