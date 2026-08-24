"""Receiver-state metadata contract for segment-safe PRN sequence scoring.

The receiver's ``segment_index`` is channel-local. It is preserved as source
metadata, but reacquisition is derived from the chronological segment ordinal
observed for each ``(run_id, prn)`` pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


SCORE_CONTRACT_SCHEMA = "gnss-doppler-lab.prn-local-quality-score.v1"
SOURCE_QUALITY_COLUMNS = ("channel", "segment_index", "window_index", "epoch_count")
TIMING_COLUMNS = ("window_bin_s", "window_start_s", "window_mid_s", "window_end_s")
SCORE_QUALITY_COLUMNS = (
    "channel", "segment_index", "prn_segment_ordinal", "continuity_block_index",
    "target_window_index", "target_sequence_position", "epoch_count",
    "tracking_age_s", "continuity_age_s", "segment_start_s",
    "history_start_window_index", "history_end_window_index",
    "history_start_s", "history_end_s", "history_length",
    "reacquisition_flag", "sequence_restart_flag", "history_same_segment_flag",
)


@dataclass(frozen=True)
class SegmentSafeBlock:
    """One uninterrupted sequence block within a receiver tracking segment."""

    run_id: str
    prn: object
    channel: int
    segment_index: int
    prn_segment_ordinal: int
    continuity_block_index: int
    frame: pd.DataFrame

    @property
    def reacquisition_flag(self) -> int:
        return int(self.prn_segment_ordinal > 0)

    @property
    def sequence_restart_flag(self) -> int:
        return int(self.prn_segment_ordinal > 0 or self.continuity_block_index > 0)


def _numeric_integer(frame: pd.DataFrame, column: str, *, positive: bool = False) -> np.ndarray:
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be numeric") from exc
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{column} must contain finite integers")
    lower_bound = 1 if positive else 0
    if np.any(values < lower_bound):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{column} must contain {qualifier} integers")
    return values.astype(np.int64)


def validate_quality_node_frame(
    frame: pd.DataFrame,
    feature_columns: Iterable[str] = (),
) -> None:
    """Validate source identity, receiver state, timing, and optional features."""

    features = tuple(feature_columns)
    required = ("run_id", "prn", *TIMING_COLUMNS, *SOURCE_QUALITY_COLUMNS, *features)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"node frame missing receiver-quality columns: {missing[:8]}")
    if frame.empty:
        raise ValueError("node frame is empty")
    for column in ("run_id", "prn"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"node frame contains null or empty {column}")

    _numeric_integer(frame, "channel")
    _numeric_integer(frame, "segment_index")
    _numeric_integer(frame, "window_index")
    _numeric_integer(frame, "epoch_count", positive=True)

    numeric_columns = [*TIMING_COLUMNS, *features]
    try:
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("node timing and feature inputs must be numeric") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("node timing and feature inputs must be finite")
    starts = pd.to_numeric(frame["window_start_s"]).to_numpy(float)
    mids = pd.to_numeric(frame["window_mid_s"]).to_numpy(float)
    ends = pd.to_numeric(frame["window_end_s"]).to_numpy(float)
    if np.any(ends <= starts) or np.any(mids < starts) or np.any(mids > ends):
        raise ValueError("node window timing is inconsistent")

    source_key = ["run_id", "prn", "channel", "segment_index", "window_index"]
    if frame.duplicated(source_key).any():
        raise ValueError(f"duplicate receiver window identity: {source_key}")
    event_key = ["run_id", "prn", "window_bin_s"]
    if frame.duplicated(event_key).any():
        raise ValueError(f"duplicate PRN event identity: {event_key}")


def segment_safe_blocks(
    frame: pd.DataFrame,
    feature_columns: Iterable[str] = (),
    *,
    expected_stride_s: float = 0.5,
    time_tolerance_s: float = 1e-6,
) -> list[SegmentSafeBlock]:
    """Split node windows at receiver segments and all cadence discontinuities."""

    if not np.isfinite(expected_stride_s) or expected_stride_s <= 0:
        raise ValueError("expected_stride_s must be finite and positive")
    if not np.isfinite(time_tolerance_s) or time_tolerance_s < 0:
        raise ValueError("time_tolerance_s must be finite and nonnegative")
    validate_quality_node_frame(frame, feature_columns)

    work = frame.copy()
    for column in (*SOURCE_QUALITY_COLUMNS, *TIMING_COLUMNS):
        work[column] = pd.to_numeric(work[column], errors="raise")
    segment_keys = ["run_id", "prn", "channel", "segment_index"]
    inventory = (
        work.groupby(segment_keys, as_index=False, sort=False)
        .agg(segment_start=("window_start_s", "min"))
        .sort_values(
            ["run_id", "prn", "segment_start", "channel", "segment_index"],
            kind="mergesort",
        )
    )
    inventory["prn_segment_ordinal"] = inventory.groupby(
        ["run_id", "prn"], sort=False
    ).cumcount()
    ordinal_lookup = {
        (str(row.run_id), row.prn, int(row.channel), int(row.segment_index)):
            int(row.prn_segment_ordinal)
        for row in inventory.itertuples(index=False)
    }

    blocks: list[SegmentSafeBlock] = []
    for keys, group in work.groupby(segment_keys, sort=False):
        run_id, prn, channel, segment_index = keys
        group = group.sort_values(
            ["window_index", "window_bin_s"], kind="mergesort"
        ).reset_index(drop=True)
        window_delta = group["window_index"].diff().to_numpy(float)
        time_delta = group["window_bin_s"].diff().to_numpy(float)
        breaks = np.zeros(len(group), dtype=bool)
        breaks[0] = True
        if len(group) > 1:
            breaks[1:] = (window_delta[1:] != 1) | ~np.isclose(
                time_delta[1:], expected_stride_s, rtol=0.0, atol=time_tolerance_s
            )
        block_ids = np.cumsum(breaks) - 1
        ordinal = ordinal_lookup[(str(run_id), prn, int(channel), int(segment_index))]
        for block_index in np.unique(block_ids):
            block_frame = group.loc[block_ids == block_index].reset_index(drop=True)
            blocks.append(SegmentSafeBlock(
                run_id=str(run_id),
                prn=prn,
                channel=int(channel),
                segment_index=int(segment_index),
                prn_segment_ordinal=ordinal,
                continuity_block_index=int(block_index),
                frame=block_frame,
            ))
    return sorted(
        blocks,
        key=lambda block: (
            block.run_id,
            str(block.prn),
            float(block.frame.iloc[0]["window_start_s"]),
            block.channel,
            block.segment_index,
            block.continuity_block_index,
        ),
    )


def score_quality_metadata(
    block: SegmentSafeBlock,
    target_position: int,
    history_length: int,
    *,
    expected_stride_s: float = 0.5,
) -> dict[str, int | float]:
    """Build causal receiver-quality metadata for one next-window prediction."""

    if isinstance(target_position, bool) or not isinstance(target_position, (int, np.integer)):
        raise ValueError("target_position must be an integer")
    if isinstance(history_length, bool) or not isinstance(history_length, (int, np.integer)):
        raise ValueError("history_length must be an integer")
    if history_length <= 0 or target_position < history_length or target_position >= len(block.frame):
        raise ValueError("target must follow a complete in-block history")
    if not np.isfinite(expected_stride_s) or expected_stride_s <= 0:
        raise ValueError("expected_stride_s must be finite and positive")

    target = block.frame.iloc[target_position]
    history_start = block.frame.iloc[target_position - history_length]
    history_end = block.frame.iloc[target_position - 1]
    target_window_index = int(target["window_index"])
    tracking_age_s = float(target_window_index * expected_stride_s)
    continuity_age_s = float(target["window_bin_s"] - block.frame.iloc[0]["window_bin_s"])
    return {
        "channel": block.channel,
        "segment_index": block.segment_index,
        "prn_segment_ordinal": block.prn_segment_ordinal,
        "continuity_block_index": block.continuity_block_index,
        "target_window_index": target_window_index,
        "target_sequence_position": int(target_position),
        "epoch_count": int(target["epoch_count"]),
        "tracking_age_s": tracking_age_s,
        "continuity_age_s": continuity_age_s,
        "segment_start_s": float(target["window_start_s"] - tracking_age_s),
        "history_start_window_index": int(history_start["window_index"]),
        "history_end_window_index": int(history_end["window_index"]),
        "history_start_s": float(history_start["window_start_s"]),
        "history_end_s": float(history_end["window_end_s"]),
        "history_length": int(history_length),
        "reacquisition_flag": block.reacquisition_flag,
        "sequence_restart_flag": block.sequence_restart_flag,
        "history_same_segment_flag": 1,
    }


def score_contract_document(*, expected_stride_s: float, history_length: int) -> dict[str, object]:
    """Return a serializable description for score manifests and reports."""

    return {
        "schema": SCORE_CONTRACT_SCHEMA,
        "source_quality_columns": list(SOURCE_QUALITY_COLUMNS),
        "score_quality_columns": list(SCORE_QUALITY_COLUMNS),
        "expected_stride_s": float(expected_stride_s),
        "history_length": int(history_length),
        "sequence_boundary": (
            "history and target share run_id, prn, channel, segment_index and an uninterrupted "
            "window_index/time cadence"
        ),
        "reacquisition_definition": (
            "prn_segment_ordinal > 0; raw segment_index is channel-local and is not itself a flag"
        ),
        "causality": (
            "quality fields are receiver bookkeeping available no later than the scored target window; "
            "no residual-derived quality proxy is used"
        ),
    }
