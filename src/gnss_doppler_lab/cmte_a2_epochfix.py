"""Exploratory causal multi-PRN decision-grid aggregation for CMTE-A2.

This module does not modify the sealed PRIMARY INVALID implementation.  It maps
per-PRN residual availability times to a canonical recording-relative grid and
selects, independently for every PRN, the latest causal residual whose age is
within a fixed limit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EpochPolicy:
    grid_origin_s: float = 0.0
    grid_stride_s: float = 0.5
    timestamp_tolerance_s: float = 1e-9
    max_residual_age_s: float = 0.55

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.grid_origin_s, self.grid_stride_s, self.timestamp_tolerance_s, self.max_residual_age_s],
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError("epoch policy values must be finite")
        if self.grid_stride_s <= 0 or self.timestamp_tolerance_s < 0 or self.max_residual_age_s < 0:
            raise ValueError("stride must be positive and tolerances nonnegative")
        if self.max_residual_age_s < self.grid_stride_s:
            raise ValueError("maximum residual age must cover one decision stride")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "availability_column": "window_end_s",
            "mapping": "first grid t_k >= availability, within numerical tolerance; latest causal residual per PRN is retained",
            "grid_formula": "t_k = origin + k*stride, k integer",
            "first_grid_index_formula": "ceil((availability-origin-tolerance)/stride)",
            "causality": "selected availability <= decision_time + tolerance",
            "tie_break": "maximum availability, then lexicographically maximum SHA-256 of stable row content",
            "state_boundaries": ["physical_recording_id", "history_id", "segment", "channel", "cadence gap via maximum age"],
            "prn_identity_usage": "grouping and diagnostics only; never a score feature",
            "source_index_decision": {
                "window_bin_s": "not used: it can precede residual availability",
                "target_window_index": "not used as a global key: it is history/channel local",
            },
        }


def _validate(frame: pd.DataFrame) -> None:
    required = {
        "physical_recording_id",
        "history_id",
        "prn",
        "segment",
        "channel",
        "window_start_s",
        "window_end_s",
        "p",
        "q",
        "rmse",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"multi-PRN epoch input missing columns: {missing}")
    if frame.empty:
        raise ValueError("multi-PRN epoch input must be nonempty")
    numeric = frame[["window_start_s", "window_end_s", "p", "q", "rmse"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("multi-PRN epoch input must be finite")
    if np.any(frame["p"].to_numpy(float) <= 0) or np.any(frame["p"].to_numpy(float) > 1):
        raise ValueError("conformal p-values must be in (0,1]")


def _stable_row_digest(row: pd.Series) -> str:
    fields = [
        "physical_recording_id",
        "history_id",
        "prn",
        "segment",
        "channel",
        "window_start_s",
        "window_end_s",
        "p",
        "q",
        "rmse",
    ] + sorted(c for c in row.index if c.startswith("residual_"))
    payload = {name: row[name].item() if hasattr(row[name], "item") else row[name] for name in fields if name in row}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def canonical_decision_rows(frame: pd.DataFrame, policy: EpochPolicy = EpochPolicy()) -> pd.DataFrame:
    """Return one latest causal residual per recording/decision-time/PRN.

    A source residual may remain eligible for more than one decision time only
    while its age is within ``max_residual_age_s``. Newer rows deterministically
    replace older rows for the same PRN. No future row can be selected.
    """
    _validate(frame)
    source = frame.copy().reset_index(drop=True)
    source["_stable_row_digest"] = source.apply(_stable_row_digest, axis=1)
    candidates: list[dict[str, Any]] = []
    origin = policy.grid_origin_s
    stride = policy.grid_stride_s
    tol = policy.timestamp_tolerance_s
    max_age = policy.max_residual_age_s
    for row in source.to_dict("records"):
        availability = float(row["window_end_s"])
        first = int(math.ceil((availability - origin - tol) / stride))
        last = int(math.floor((availability + max_age - origin + tol) / stride))
        for index in range(first, last + 1):
            decision = origin + index * stride
            age = decision - availability
            if availability > decision + tol or age < -tol or age > max_age + tol:
                continue
            item = dict(row)
            item["decision_time_s"] = float(decision)
            item["residual_age_s"] = float(max(0.0, age))
            candidates.append(item)
    if not candidates:
        raise ValueError("no residual is eligible on the canonical decision grid")
    out = pd.DataFrame(candidates)
    keys = ["physical_recording_id", "decision_time_s", "prn"]
    out = out.sort_values(keys + ["window_end_s", "_stable_row_digest"], kind="mergesort")
    out = out.drop_duplicates(keys, keep="last")
    out = out.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if out.duplicated(keys).any():
        raise AssertionError("duplicate PRN survived canonical decision selection")
    if np.any(out.window_end_s.to_numpy(float) > out.decision_time_s.to_numpy(float) + tol):
        raise AssertionError("future residual survived canonical decision selection")
    if np.any(out.residual_age_s.to_numpy(float) > max_age + tol):
        raise AssertionError("stale residual survived canonical decision selection")
    return out.drop(columns=["_stable_row_digest"])


def aggregate_multi_prn_epochs(frame: pd.DataFrame, policy: EpochPolicy = EpochPolicy()) -> pd.DataFrame:
    selected = canonical_decision_rows(frame, policy)
    rows: list[dict[str, Any]] = []
    for (recording, decision), group in selected.groupby(
        ["physical_recording_id", "decision_time_s"], sort=True, dropna=False
    ):
        if group.duplicated("prn").any():
            raise AssertionError("one PRN contributed more than once to a decision epoch")
        p = group.p.to_numpy(float)
        q = group.q.to_numpy(float)
        rmse = group.rmse.to_numpy(float)
        ages = group.residual_age_s.to_numpy(float)
        evidence = -np.log(np.maximum(p, np.finfo(float).tiny))
        prns = sorted(group.prn.astype(str).tolist())
        identities = sorted(
            "|".join(map(str, (x.history_id, x.segment, x.channel))) for x in group.itertuples(index=False)
        )
        rows.append(
            {
                "physical_recording_id": str(recording),
                "window_end_s": float(decision),
                "decision_time_s": float(decision),
                "window_start_s": float(group.window_start_s.min()),
                "tracked_prn_count": int(len(group)),
                "min_p": float(np.min(p)),
                "median_p": float(np.median(p)),
                "mean_q": float(np.mean(q)),
                "max_q": float(np.max(q)),
                "mean_neg_log_p": float(np.mean(evidence)),
                "median_neg_log_p": float(np.median(evidence)),
                "score_A2": float(np.mean(evidence)),
                "score_A0": float(np.max(rmse)),
                "rmse_values": rmse,
                "min_residual_age_s": float(np.min(ages)),
                "median_residual_age_s": float(np.median(ages)),
                "max_residual_age_s": float(np.max(ages)),
                "prn_set_hash": hashlib.sha256(";".join(prns).encode()).hexdigest(),
                "producer_chain_id": "epochfix-" + hashlib.sha256(";".join(identities).encode()).hexdigest()[:20],
            }
        )
    out = pd.DataFrame(rows).sort_values(["physical_recording_id", "window_end_s"], kind="mergesort").reset_index(drop=True)
    return out
