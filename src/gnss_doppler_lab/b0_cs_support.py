"""Exact common-support alignment for B0-CS predictor comparisons."""
from __future__ import annotations

import pandas as pd


KEY_COLUMNS = ("physical_recording_id", "window_bin_s", "prn")


def common_epoch_prn_support(*frames: pd.DataFrame) -> tuple[list[pd.DataFrame], dict[str, object]]:
    if len(frames) < 2:
        raise ValueError("at least two frames are required")
    common = None
    input_rows = []
    for frame in frames:
        missing = sorted(set(KEY_COLUMNS) - set(frame))
        if missing:
            raise ValueError(f"common support frame missing: {missing}")
        if frame.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError("duplicate epoch/PRN key in common-support input")
        keys = set(zip(
            frame.physical_recording_id.astype(str),
            frame.window_bin_s.astype(float),
            frame.prn.astype(str),
        ))
        common = keys if common is None else common & keys
        input_rows.append(len(frame))
    if not common:
        raise ValueError("no common epoch/PRN support")
    aligned = []
    for frame in frames:
        keys = list(zip(
            frame.physical_recording_id.astype(str),
            frame.window_bin_s.astype(float),
            frame.prn.astype(str),
        ))
        local = frame.loc[[key in common for key in keys]].copy()
        aligned.append(local.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True))
    audit = {
        "schema": "gnss-doppler-lab.b0-cs-common-support.v1",
        "key_columns": list(KEY_COLUMNS),
        "input_rows": input_rows,
        "common_rows": len(common),
        "output_rows": [len(frame) for frame in aligned],
        "exact_common_epoch_prn_support": all(len(frame) == len(common) for frame in aligned),
    }
    return aligned, audit
