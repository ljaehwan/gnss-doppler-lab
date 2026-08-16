"""Canonical TRACE native-dump reproducibility diagnostics.

This module is deliberately independent of TRACE scoring.  It compares receiver
records on physical identity ``(dataset, PRN, raw interval start, raw interval
end)`` and keeps channel/session/file-order metadata out of the key.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .trace_native_1ms import ACTION_VALUE_FIELDS, MEASUREMENT_FIELDS, TAPS, read_records

KEY_FIELDS = ("dataset", "prn", "raw_interval_start_sample", "raw_interval_end_sample")
METADATA_FIELDS = (
    "channel",
    "tracking_session_id",
    "loop_sequence",
    "action_used_source_loop_sequence",
)
STATE_FIELDS = (
    "valid_tracking",
    "valid_lock",
    "data_symbol_boundary",
    "loop_update_boundary",
    "navigation_bit_wipeoff_applied",
    "pull_in_transitory",
    "receiver_state",
)
PHYSICAL_FIELDS = tuple(
    [value for tap in TAPS for value in (f"{tap}_i", f"{tap}_q")]
    + list(MEASUREMENT_FIELDS)
    + [
        value
        for prefix in ("action_used", "action_next")
        for value in (
            *[f"{prefix}_{field}" for field in ACTION_VALUE_FIELDS],
            f"{prefix}_interval_length_samples",
        )
    ]
    + list(STATE_FIELDS)
)


@dataclass(frozen=True)
class CanonicalReplay:
    dataset: str
    directory: Path
    records: np.ndarray
    files: np.ndarray

    @property
    def keys(self) -> pd.DataFrame:
        records = self.records
        return pd.DataFrame(
            {
                "dataset": np.repeat(self.dataset, len(records)),
                "prn": records["prn"],
                "raw_interval_start_sample": records["raw_interval_start_sample"],
                "raw_interval_end_sample": records["raw_interval_end_sample"],
                "row_index": np.arange(len(records), dtype=np.int64),
            }
        )


def load_replay(directory: Path | str, dataset: str) -> CanonicalReplay:
    directory = Path(directory)
    arrays: list[np.ndarray] = []
    files: list[str] = []
    for path in sorted(directory.glob("trace_native_1ms_ch_*.bin")):
        _, records = read_records(path)
        arrays.append(np.asarray(records))
        files.extend([path.name] * len(records))
    if not arrays:
        raise ValueError(f"{directory}: no TRACE native dumps")
    return CanonicalReplay(dataset, directory, np.concatenate(arrays), np.asarray(files))


def canonical_join(first: CanonicalReplay, second: CanonicalReplay) -> pd.DataFrame:
    left = first.keys.rename(columns={"row_index": "rep1_row_index"})
    right = second.keys.rename(columns={"row_index": "rep2_row_index"})
    if left.duplicated(list(KEY_FIELDS)).any() or right.duplicated(list(KEY_FIELDS)).any():
        raise ValueError("duplicate canonical key in replay")
    return left.merge(right, on=list(KEY_FIELDS), how="inner", validate="one_to_one").sort_values(
        list(KEY_FIELDS), kind="mergesort"
    )


def exact_equal(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.dtype != second.dtype:
        raise TypeError("exact comparison requires identical dtypes")
    if np.issubdtype(first.dtype, np.floating):
        return first.view(f"u{first.dtype.itemsize}") == second.view(f"u{second.dtype.itemsize}")
    return first == second


def canonical_semantic_hash(replay: CanonicalReplay) -> str:
    order = np.lexsort(
        (
            replay.records["raw_interval_end_sample"],
            replay.records["raw_interval_start_sample"],
            replay.records["prn"],
        )
    )
    digest = hashlib.sha256()
    dataset = replay.dataset.encode("utf-8")
    for row in order:
        digest.update(len(dataset).to_bytes(4, "little"))
        digest.update(dataset)
        for field in ("prn", "raw_interval_start_sample", "raw_interval_end_sample", *PHYSICAL_FIELDS):
            digest.update(np.asarray(replay.records[field][row]).tobytes())
    return digest.hexdigest()


def common_epoch_count(joined: pd.DataFrame, sample_rate_hz: float, minimum_prns: int = 4) -> int:
    rounded_ms = np.rint(
        joined["raw_interval_start_sample"].to_numpy(dtype=np.float64)
        / (float(sample_rate_hz) * 0.001)
    ).astype(np.int64)
    groups = joined.assign(rounded_ms_epoch=rounded_ms).groupby("rounded_ms_epoch", sort=False)["prn"].nunique()
    return int((groups >= minimum_prns).sum())


def complex_correlation(first: np.ndarray, second: np.ndarray) -> dict[str, float | None]:
    left = np.column_stack([first[f"{tap}_i"].astype(np.float64) + 1j * first[f"{tap}_q"] for tap in TAPS]).ravel()
    right = np.column_stack([second[f"{tap}_i"].astype(np.float64) + 1j * second[f"{tap}_q"] for tap in TAPS]).ravel()
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.sqrt(np.vdot(left, left).real * np.vdot(right, right).real))
    if denominator == 0.0:
        return {"magnitude": None, "real": None, "imag": None}
    value = np.vdot(left, right) / denominator
    return {"magnitude": float(abs(value)), "real": float(value.real), "imag": float(value.imag)}


def field_statistics(first: np.ndarray, second: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray]:
    exact_rows = np.ones(len(first), dtype=bool)
    output: list[dict[str, object]] = []
    for field in PHYSICAL_FIELDS:
        left = first[field]
        right = second[field]
        equal = exact_equal(left, right)
        exact_rows &= equal
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        if np.issubdtype(left.dtype, np.number) and np.std(left.astype(np.float64)) and np.std(right.astype(np.float64)):
            correlation = float(np.corrcoef(left.astype(np.float64), right.astype(np.float64))[0, 1])
        else:
            correlation = None
        output.append(
            {
                "field": field,
                "dtype": str(left.dtype),
                "exact_bit_match_count": int(equal.sum()),
                "exact_bit_match_ratio": float(equal.mean()) if len(equal) else None,
                "maximum_absolute_error": float(np.max(difference)) if len(difference) else None,
                "median_absolute_error": float(np.median(difference)) if len(difference) else None,
                "p99_9_absolute_error": float(np.percentile(difference, 99.9)) if len(difference) else None,
                "correlation": correlation,
            }
        )
    return output, exact_rows


def assignment_rows(replay: CanonicalReplay, repetition: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for channel in np.unique(replay.records["channel"]):
        channel_records = replay.records[replay.records["channel"] == channel]
        for prn in np.unique(channel_records["prn"]):
            selected = channel_records[channel_records["prn"] == prn]
            output.append(
                {
                    "repetition": repetition,
                    "dataset": replay.dataset,
                    "channel": int(channel),
                    "prn": int(prn),
                    "record_count": int(len(selected)),
                    "first_raw_interval_start_sample": int(selected["raw_interval_start_sample"].min()),
                    "last_raw_interval_end_sample": int(selected["raw_interval_end_sample"].max()),
                    "tracking_session_ids": ";".join(map(str, sorted(map(int, np.unique(selected["tracking_session_id"]))))),
                }
            )
    return output
