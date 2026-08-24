#!/usr/bin/env python3
"""Audit legacy PRN scoring against the segment-safe receiver-quality contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gnss_doppler_lab import receiver_quality_contract as contract


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _score_frame(path: Path, description: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"run_id", "prn", "window_bin_s", "prn_node_rmse"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"{description} missing nonempty score contract")
    if frame.duplicated(["run_id", "prn", "window_bin_s"]).any():
        raise ValueError(f"{description} contains duplicate score keys")
    numeric = frame[["window_bin_s", "prn_node_rmse"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{description} contains non-finite scores")
    return frame


def _event_frame(path: Path, description: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "window_bin_s", "window_start_s", "prn_node_rmse_max",
        "prn_node_rmse_top3_mean", "prn_node_rmse_mean",
    }
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"{description} missing nonempty event contract")
    if frame.duplicated(["window_bin_s"]).any():
        raise ValueError(f"{description} contains duplicate event keys")
    if not np.isfinite(frame[list(required)].to_numpy(float)).all():
        raise ValueError(f"{description} contains non-finite event values")
    return frame


def _count_changed(values: np.ndarray, tolerance: float = 1e-6) -> dict[str, object]:
    delta = np.abs(values)
    return {
        "changed_gt_tolerance": int(np.sum(delta > tolerance)),
        "tolerance": tolerance,
        "max_abs_delta": float(delta.max(initial=0.0)),
        "median_abs_delta": float(np.median(delta)) if len(delta) else 0.0,
    }


def build_audit(
    node_csv: Path,
    legacy_prn_scores_csv: Path,
    quality_prn_scores_csv: Path,
    legacy_event_scores_csv: Path,
    quality_event_scores_csv: Path,
    quality_summary_json: Path,
    *,
    stride_s: float,
    history_length: int,
    onset_s: float,
) -> dict[str, object]:
    node_csv = Path(node_csv)
    legacy_prn_scores_csv = Path(legacy_prn_scores_csv)
    quality_prn_scores_csv = Path(quality_prn_scores_csv)
    legacy_event_scores_csv = Path(legacy_event_scores_csv)
    quality_event_scores_csv = Path(quality_event_scores_csv)
    quality_summary_json = Path(quality_summary_json)

    node = pd.read_csv(node_csv)
    contract.validate_quality_node_frame(node)
    blocks = contract.segment_safe_blocks(node, expected_stride_s=stride_s)
    legacy = _score_frame(legacy_prn_scores_csv, "legacy PRN scores")
    quality = _score_frame(quality_prn_scores_csv, "quality PRN scores")
    missing_quality = sorted(set(contract.SCORE_QUALITY_COLUMNS) - set(quality.columns))
    if missing_quality:
        raise ValueError(f"quality PRN scores missing metadata: {missing_quality}")
    if not quality["history_same_segment_flag"].eq(1).all():
        raise ValueError("quality PRN scores contain cross-segment history")
    if not quality["history_length"].eq(history_length).all():
        raise ValueError("quality PRN score history length mismatch")

    score_keys = ["run_id", "prn", "window_bin_s"]
    shared = legacy.merge(
        quality,
        on=score_keys,
        how="inner",
        suffixes=("_legacy", "_quality"),
        validate="one_to_one",
    )
    legacy_only = legacy.merge(
        quality[score_keys], on=score_keys, how="left", indicator=True
    ).query("_merge == 'left_only'")
    quality_only = quality.merge(
        legacy[score_keys], on=score_keys, how="left", indicator=True
    ).query("_merge == 'left_only'")

    raw_segments = {
        (block.run_id, str(block.prn), block.channel, block.segment_index)
        for block in blocks
    }
    reacquired_segments = {
        (block.run_id, str(block.prn), block.channel, block.segment_index)
        for block in blocks if block.prn_segment_ordinal > 0
    }
    expected_legacy_rows = sum(
        max(0, len(group) - history_length)
        for _, group in node.groupby(["run_id", "prn"], sort=False)
    )
    expected_quality_rows = sum(max(0, len(block.frame) - history_length) for block in blocks)

    legacy_events = _event_frame(legacy_event_scores_csv, "legacy event scores")
    quality_events = _event_frame(quality_event_scores_csv, "quality event scores")
    events = legacy_events.merge(
        quality_events,
        on="window_bin_s",
        how="inner",
        suffixes=("_legacy", "_quality"),
        validate="one_to_one",
    )
    if len(events) != len(legacy_events) or len(events) != len(quality_events):
        raise ValueError("legacy and quality event grids differ")

    summary = json.loads(quality_summary_json.read_text(encoding="utf-8"))
    thresholds = summary.get("normal_prn_thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("quality summary missing normal PRN thresholds")

    event_changes = {}
    for score in (
        "prn_node_rmse_max",
        "prn_node_rmse_top3_mean",
        "prn_node_rmse_mean",
    ):
        event_changes[score] = _count_changed(
            events[f"{score}_legacy"].to_numpy(float)
            - events[f"{score}_quality"].to_numpy(float)
        )

    threshold_flips: dict[str, dict[str, object]] = {}
    times = events["window_start_s_legacy"].to_numpy(float)
    boundary_bins = set(legacy_only["window_bin_s"].to_numpy(float))
    event_at_boundary = events["window_bin_s"].isin(boundary_bins).to_numpy()
    for aggregation, score in (
        ("max", "prn_node_rmse_max"),
        ("top3", "prn_node_rmse_top3_mean"),
    ):
        for name, threshold in thresholds.items():
            legacy_flags = events[f"{score}_legacy"].to_numpy(float) > float(threshold)
            quality_flags = events[f"{score}_quality"].to_numpy(float) > float(threshold)
            flips = legacy_flags != quality_flags
            threshold_flips[f"{aggregation}_{name}"] = {
                "threshold": float(threshold),
                "legacy_flags": int(legacy_flags.sum()),
                "quality_flags": int(quality_flags.sum()),
                "flips": int(flips.sum()),
                "pre_onset_flips": int(np.sum(flips & (times < onset_s))),
                "post_onset_flips": int(np.sum(flips & (times >= onset_s))),
                "boundary_bin_flips": int(np.sum(flips & event_at_boundary)),
                "non_boundary_bin_flips": int(np.sum(flips & ~event_at_boundary)),
            }

    rmse_delta = (
        shared["prn_node_rmse_legacy"].to_numpy(float)
        - shared["prn_node_rmse_quality"].to_numpy(float)
    )
    restart_blocks = [block for block in blocks if block.sequence_restart_flag]
    return {
        "schema": "gnss-doppler-lab.receiver-quality-score-audit.v1",
        "parameters": {
            "stride_s": float(stride_s),
            "history_length": int(history_length),
            "onset_s": float(onset_s),
        },
        "inputs": {
            "node_csv": identity(node_csv),
            "legacy_prn_scores_csv": identity(legacy_prn_scores_csv),
            "quality_prn_scores_csv": identity(quality_prn_scores_csv),
            "legacy_event_scores_csv": identity(legacy_event_scores_csv),
            "quality_event_scores_csv": identity(quality_event_scores_csv),
            "quality_summary_json": identity(quality_summary_json),
        },
        "sequence_inventory": {
            "node_rows": int(len(node)),
            "prns": int(node["prn"].nunique()),
            "receiver_segments": int(len(raw_segments)),
            "reacquired_receiver_segments": int(len(reacquired_segments)),
            "continuity_blocks": int(len(blocks)),
            "restart_boundaries": int(len(restart_blocks)),
        },
        "prn_score_comparison": {
            "legacy_rows": int(len(legacy)),
            "quality_rows": int(len(quality)),
            "shared_rows": int(len(shared)),
            "legacy_only_boundary_crossing_rows": int(len(legacy_only)),
            "quality_only_rows": int(len(quality_only)),
            "expected_legacy_rows": int(expected_legacy_rows),
            "expected_quality_rows": int(expected_quality_rows),
            "reacquisition_score_rows": int(quality["reacquisition_flag"].sum()),
            "restart_regime_score_rows": int(quality["sequence_restart_flag"].sum()),
            "history_same_segment_all": bool(
                quality["history_same_segment_flag"].eq(1).all()
            ),
            "shared_rmse_delta": _count_changed(rmse_delta),
        },
        "event_score_comparison": {
            "event_windows": int(len(events)),
            "score_changes": event_changes,
            "threshold_flips": threshold_flips,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-csv", required=True)
    parser.add_argument("--legacy-prn-scores-csv", required=True)
    parser.add_argument("--quality-prn-scores-csv", required=True)
    parser.add_argument("--legacy-event-scores-csv", required=True)
    parser.add_argument("--quality-event-scores-csv", required=True)
    parser.add_argument("--quality-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--stride-s", type=float, default=0.5)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--onset-s", type=float, required=True)
    args = parser.parse_args()

    audit = build_audit(
        Path(args.node_csv),
        Path(args.legacy_prn_scores_csv),
        Path(args.quality_prn_scores_csv),
        Path(args.legacy_event_scores_csv),
        Path(args.quality_event_scores_csv),
        Path(args.quality_summary_json),
        stride_s=args.stride_s,
        history_length=args.history_length,
        onset_s=args.onset_s,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
