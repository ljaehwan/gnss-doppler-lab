#!/usr/bin/env python3
"""Re-score GNSS-OpenIF S1/S2 with development-only temporal CGC logic."""
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
import evaluate_gnss_openif_s1_cgc as openif  # noqa: E402
from gnss_doppler_lab.gcmr_geometry import ephemeris_health_selection  # noqa: E402
from gnss_doppler_lab.rinex_nav import parse_rinex2_gps_nav_gz  # noqa: E402
from gnss_doppler_lab.temporal_cgc import causal_prn_median  # noqa: E402


SCENARIOS = {
    "S1": {
        "config": ROOT / "configs/experiments/gnss_openif_s1_real_multipath_v1.json",
        "evaluation": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gnss-openif-s1-real-multipath-v1/evaluation"),
    },
    "S2": {
        "config": ROOT / "configs/experiments/gnss_openif_s2_real_multipath_v1.json",
        "evaluation": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gnss-openif-s2-real-multipath-v1/evaluation"),
    },
}
OUTPUT = ROOT / "artifacts/cgc_temporal_stabilization_dev_v1/openif_controls.json"


def load_context(config: dict[str, Any], delay_rows: list[dict[str, Any]]) -> tuple[dict[int, Any], dict[str, np.ndarray]]:
    dataset, analysis = config["dataset"], config["analysis"]
    nav = openif.verify(
        dataset["broadcast_navigation_path"], dataset["broadcast_navigation_sha256"], "broadcast NAV"
    )
    truth_path = openif.verify(dataset["ground_truth_path"], dataset["ground_truth_sha256"], "ground truth")
    ephemerides = parse_rinex2_gps_nav_gz(
        nav,
        full_gps_week=int(dataset["gps_week"]),
        target_tow_s=float(dataset["recording_start_tow_s"]),
        maximum_toe_age_s=float(analysis["maximum_ephemeris_toe_age_s"]),
    )
    tracked = {int(row["prn"]) for row in delay_rows}
    healthy, _ = ephemeris_health_selection(
        ephemerides, tracked_prns=tracked, min_prns=int(analysis["minimum_prns"])
    )
    return healthy, openif.load_ground_truth(truth_path)


def score_scenario(name: str, config: dict[str, Any], delay_rows: list[dict[str, Any]], *, window: int) -> dict[str, Any]:
    enriched = [
        {**row, "pair_id": name, "condition": "real-multipath"}
        for row in delay_rows
    ]
    filtered = causal_prn_median(enriched, window_bins=window)
    ephemerides, truth = load_context(config, delay_rows)
    for row in filtered:
        row["estimated_delay_chips"] = row["stabilized_delay_chips"]
    geometry_rows = openif.geometry_rows(filtered, ephemerides, truth, config)
    by_bin = {}
    for row in filtered:
        if int(row["prn"]) not in ephemerides:
            continue
        by_bin.setdefault(int(row["bin_index"]), []).append(float(row["estimated_delay_chips"]))
    scored = []
    threshold = float(config["support_normalization"]["partial_f_p_alarm_threshold"])
    for row in geometry_rows:
        values = np.asarray(by_bin[int(row["bin_index"])], dtype=float)
        rms = float(np.sqrt(np.mean((values - np.mean(values)) ** 2)))
        p_value = openif.partial_f_p_value(
            float(row["clock_centered_geometry_residual"]), int(row["prn_count"])
        )
        scored.append(
            {
                **row,
                "pair_id": name,
                "condition": "real-multipath",
                "partial_f_p_value": p_value,
                "centered_delay_rms_chips": rms,
                "raw_spoof_alarm": bool(p_value <= threshold),
                "diagnostic_joint_alarm": bool(
                    p_value <= threshold
                    and rms >= evaluation.DIAGNOSTIC_RMS_THRESHOLD_CHIPS
                ),
            }
        )
    scored = evaluation.add_persistence(scored, "raw_spoof_alarm", "persistent_spoof_alarm")
    scored = evaluation.add_persistence(
        scored, "diagnostic_joint_alarm", "diagnostic_joint_persistent_alarm"
    )
    return {
        "scenario": name,
        "window_bins": window,
        "evaluated_bins": len(scored),
        "raw_alarm_rate": evaluation.rate(scored, "raw_spoof_alarm"),
        "persistent_alarm_rate": evaluation.rate(scored, "persistent_spoof_alarm"),
        "diagnostic_joint_raw_alarm_rate": evaluation.rate(scored, "diagnostic_joint_alarm"),
        "diagnostic_joint_persistent_alarm_rate": evaluation.rate(
            scored, "diagnostic_joint_persistent_alarm"
        ),
        "diagnostic_joint_persistent_alarm_count": int(sum(
            bool(row["diagnostic_joint_persistent_alarm"]) for row in scored
        )),
        "centered_delay_rms_median_chips": float(np.median([
            row["centered_delay_rms_chips"] for row in scored
        ])),
    }


def main() -> None:
    results = []
    input_records = {}
    for name, paths in SCENARIOS.items():
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
        delay_path = paths["evaluation"] / "delay_estimates.csv"
        delay_rows = evaluation.read_csv(delay_path)
        for window in (1, evaluation.SELECTED_WINDOW):
            results.append(score_scenario(name, config, delay_rows, window=window))
        input_records[name] = {
            "config": {"path": str(paths["config"].resolve()), "sha256": evaluation.sha256(paths["config"])},
            "delay_estimates": {"path": str(delay_path.resolve()), "sha256": evaluation.sha256(delay_path)},
        }
    document = {
        "schema": "gnss-doppler-lab.cgc-temporal-openif-controls-development",
        "schema_version": 1,
        "role": "post-hoc real-multipath specificity diagnostic; not a frozen external result",
        "rows": results,
        "inputs": input_records,
        "claim_boundary": (
            "S1 and S2 were previously accessed and the temporal/observability candidate was designed later. "
            "This checks compatibility only; it cannot validate the new detector."
        ),
    }
    evaluation.write_json(OUTPUT, document)
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
