"""Causal temporal stabilization for per-PRN CGC signed-delay estimates."""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

import numpy as np


def causal_prn_median(
    rows: Iterable[Mapping[str, Any]],
    *,
    window_bins: int = 5,
    stream_fields: Sequence[str] = ("pair_id", "condition"),
    bin_field: str = "bin_index",
    prn_field: str = "prn",
    value_field: str = "estimated_delay_chips",
    output_field: str = "stabilized_delay_chips",
) -> list[dict[str, Any]]:
    """Return a causal, wall-clock median of each PRN's signed delay.

    Only samples in ``[current_bin - window_bins + 1, current_bin]`` from the
    same stream and PRN are used.  Missing seconds therefore do not silently
    extend the temporal aperture, and future observations can never enter the
    estimate.  All input columns are retained in the returned records.
    """
    if isinstance(window_bins, bool) or int(window_bins) != window_bins or window_bins < 1:
        raise ValueError("window_bins must be a positive integer")
    window = int(window_bins)
    materialized = [dict(row) for row in rows]
    if not materialized:
        return []

    def identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        try:
            return tuple(row[field] for field in stream_fields) + (row[prn_field],)
        except KeyError as exc:
            raise ValueError(f"missing grouping field: {exc.args[0]}") from exc

    ordered = sorted(
        enumerate(materialized),
        key=lambda item: (identity(item[1]), int(item[1][bin_field]), item[0]),
    )
    histories: dict[tuple[Any, ...], deque[tuple[int, float]]] = defaultdict(deque)
    seen: set[tuple[tuple[Any, ...], int]] = set()
    output: list[dict[str, Any] | None] = [None] * len(materialized)
    for original_index, row in ordered:
        key = identity(row)
        try:
            bin_index = int(row[bin_field])
            value = float(row[value_field])
        except KeyError as exc:
            raise ValueError(f"missing temporal field: {exc.args[0]}") from exc
        if isinstance(row[bin_field], bool) or float(row[bin_field]) != bin_index:
            raise ValueError("bin indices must be integers")
        if not math.isfinite(value):
            raise ValueError("signed-delay values must be finite")
        marker = (key, bin_index)
        if marker in seen:
            raise ValueError("each stream/PRN/bin must have at most one delay estimate")
        seen.add(marker)
        history = histories[key]
        left = bin_index - window + 1
        while history and history[0][0] < left:
            history.popleft()
        history.append((bin_index, value))
        stabilized = float(np.median([sample for _, sample in history]))
        output[original_index] = {
            **row,
            output_field: stabilized,
            "temporal_window_bins": window,
            "temporal_support_bins": len(history),
        }
    return [row for row in output if row is not None]
