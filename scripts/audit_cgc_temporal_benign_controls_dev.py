#!/usr/bin/env python3
"""Audit temporal-CGC false alarms on existing train-normal controls."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import evaluate_cgc_temporal_stabilization_dev as evaluation  # noqa: E402
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table  # noqa: E402


CONFIG = ROOT / "configs/experiments/cgc_normal_detector_freeze_audit_v1.json"
DELAYS = ROOT / "artifacts/cgc_normal_detector_freeze_audit_v1/analysis/normal_delay_estimates.csv"
OUTPUT = ROOT / "artifacts/cgc_temporal_stabilization_dev_v1/benign_controls.json"


def inputs() -> tuple[list[dict[str, Any]], dict[str, dict[str, tuple[float, float, float]]]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = [{**row, "condition": "normal"} for row in evaluation.read_csv(DELAYS)]
    los = {}
    for pair_id, record in config["normal_sources"].items():
        source = ROOT / record["los_log_path"]
        if evaluation.sha256(source) != record["los_log_sha256"]:
            raise ValueError(f"train-normal LOS hash mismatch: {pair_id}")
        los[pair_id] = parse_gps_sdr_sim_los_table(source.read_text(encoding="utf-8"))
    return rows, los


def summarize(rows: list[dict[str, Any]], *, window: int) -> dict[str, Any]:
    eligible = [row for row in rows if int(row["bin_index"]) >= 1]
    pair_ids = sorted({str(row["pair_id"]) for row in eligible})
    return {
        "window_bins": window,
        "eligible_bins": len(eligible),
        "raw_alarm_rate": evaluation.rate(eligible, "raw_spoof_alarm"),
        "persistent_alarm_rate": evaluation.rate(eligible, "persistent_spoof_alarm"),
        "persistent_alarm_pair_count": int(sum(
            any(row["persistent_spoof_alarm"] for row in eligible if row["pair_id"] == pair_id)
            for pair_id in pair_ids
        )),
        "diagnostic_joint_raw_alarm_rate": evaluation.rate(eligible, "diagnostic_joint_alarm"),
        "diagnostic_joint_persistent_alarm_rate": evaluation.rate(
            eligible, "diagnostic_joint_persistent_alarm"
        ),
        "diagnostic_joint_persistent_alarm_pair_count": int(sum(
            any(row["diagnostic_joint_persistent_alarm"] for row in eligible if row["pair_id"] == pair_id)
            for pair_id in pair_ids
        )),
        "centered_delay_rms_median_chips": float(np.median([
            row["centered_delay_rms_chips"] for row in eligible
        ])),
        "centered_delay_rms_95th_percentile_chips": float(np.quantile([
            row["centered_delay_rms_chips"] for row in eligible
        ], 0.95)),
    }


def main() -> None:
    delay_rows, los = inputs()
    summaries = []
    for window in (1, evaluation.SELECTED_WINDOW):
        _, scored = evaluation.score_delays(delay_rows, los, window_bins=window)
        scored = evaluation.add_persistence(
            scored, "raw_spoof_alarm", "persistent_spoof_alarm"
        )
        scored = evaluation.add_persistence(
            scored, "diagnostic_joint_alarm", "diagnostic_joint_persistent_alarm"
        )
        summaries.append(summarize(scored, window=window))
    document = {
        "schema": "gnss-doppler-lab.cgc-temporal-benign-controls-development",
        "schema_version": 1,
        "role": "post-hoc train-normal diagnostic; not an independent validation",
        "rows": summaries,
        "inputs": {
            "config": {"path": str(CONFIG.resolve()), "sha256": evaluation.sha256(CONFIG)},
            "delay_estimates": {"path": str(DELAYS.resolve()), "sha256": evaluation.sha256(DELAYS)},
        },
        "claim_boundary": (
            "These five train-normal streams predate the temporal candidate, but the 0.10-chip diagnostic "
            "gate and five-bin median were inspected against other reused results before this audit."
        ),
    }
    evaluation.write_json(OUTPUT, document)
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
