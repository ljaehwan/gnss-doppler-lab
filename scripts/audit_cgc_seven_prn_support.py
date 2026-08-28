#!/usr/bin/env python3
"""Audit the frozen CGC threshold under exactly seven-satellite support."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import f as f_distribution

ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_cgc_real_detection as frozen  # noqa: E402
from run_gcmr_texbat_external import preflight_ds4_alternate  # noqa: E402
from gnss_doppler_lab.clock_centered_geometry import fit_clock_centered_geometry  # noqa: E402
from gnss_doppler_lab.gcmr_geometry import (  # noqa: E402
    ephemeris_health_selection,
    parse_gnss_sdr_gps_ephemeris_xml,
    satellite_observation,
)


DEFAULT_CONFIG = ROOT / "configs/experiments/cgc_real_detection_v1.json"
DEFAULT_SUMMARY = ROOT / "artifacts/cgc_real_detection_v1/summary.json"
DEFAULT_OUTPUT = ROOT / "artifacts/cgc_seven_prn_support_v1"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_subset(entries: list[dict[str, Any]], *, scenario: str, trial: int) -> list[dict[str, Any]]:
    if len(entries) < 7:
        raise ValueError("seven-PRN subset requires at least seven entries")
    def rank(row: dict[str, Any]) -> str:
        text = f"{scenario}:{trial}:{int(row['prn'])}".encode()
        return hashlib.sha256(text).hexdigest()
    return sorted(entries, key=rank)[:7]


def partial_f_p_value(residual: float, prn_count: int) -> float:
    """Return the support-normalized partial-F tail probability."""
    value = float(residual)
    count = int(prn_count)
    if not 0.0 < value <= 1.0 or count <= 4:
        raise ValueError("partial-F requires residual in (0,1] and more than four PRNs")
    statistic = (1.0 - value) * (count - 4) / (3.0 * value)
    return float(f_distribution.sf(statistic, 3, count - 4))



def load_delay_groups(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = {
                "scenario": raw["scenario"],
                "role": raw["role"],
                "bin_index": int(raw["bin_index"]),
                "prn": int(raw["prn"].removeprefix("G")),
                "delay": float(raw["estimated_delay_chips"]),
                "asymmetry": float(raw["median_early_late_asymmetry"]),
            }
            grouped.setdefault((row["scenario"], row["bin_index"]), []).append(row)
    return grouped


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path, summary_path: Path, output: Path, trials: int) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be positive")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    resolved = frozen.validate_config(config)
    frozen_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    delay_record = frozen_summary["artifacts"]["delay_estimates"]
    delay_path = Path(delay_record["path"]).resolve()
    if sha256(delay_path) != delay_record["sha256"]:
        raise ValueError("frozen delay estimate SHA-256 mismatch")
    groups = load_delay_groups(delay_path)
    detection_record = frozen_summary["artifacts"]["detection_scores"]
    detection_path = Path(detection_record["path"]).resolve()
    if sha256(detection_path) != detection_record["sha256"]:
        raise ValueError("frozen detection score SHA-256 mismatch")
    calibration_p_values: list[float] = []
    calibration_start, calibration_end = config["analysis"]["calibration_interval_seconds"]
    with detection_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            start_s = float(raw["bin_start_s"])
            if (
                raw["scenario"] == "cleanStatic"
                and float(calibration_start) <= start_s < float(calibration_end)
            ):
                calibration_p_values.append(partial_f_p_value(
                    float(raw["clock_centered_geometry_residual"]),
                    int(raw["prn_count"]),
                ))
    if len(calibration_p_values) < 80:
        raise ValueError("insufficient cleanStatic bins for partial-F calibration")
    partial_f_p_threshold = float(np.quantile(calibration_p_values, 0.05))
    source_by_name = {source["name"]: source for source in resolved["sources"]}
    geometry: dict[str, dict[str, Any]] = {}
    for name, source in source_by_name.items():
        ephemerides = parse_gnss_sdr_gps_ephemeris_xml(source["paths"]["ephemeris"])
        tracked = {
            int(row["prn"])
            for (scenario, _), rows in groups.items() if scenario == name
            for row in rows
        }
        preflight = preflight_ds4_alternate(
            source["paths"]["nmea"].parent,
            ephemerides,
            configured_tow0_s=float(source["tow0_s"]),
            tracked_prns=tracked,
            min_prns=7,
        )
        healthy, _ = ephemeris_health_selection(
            ephemerides, tracked_prns=tracked, min_prns=7
        )
        geometry[name] = {
            "source": source,
            "healthy": healthy,
            "receiver_ecef": preflight["receiver_position_contract"]["ecef"],
        }

    threshold = float(frozen_summary["thresholds"]["residual_alarm_threshold"])
    distortion_threshold = float(
        frozen_summary["thresholds"]["multipath_enrichment_threshold"]
    )
    los_by_key: dict[tuple[str, int, int], tuple[float, float, float]] = {}
    for (scenario, bin_index), entries in groups.items():
        support = geometry[scenario]
        tow = (float(support["source"]["tow0_s"]) + bin_index + 0.5) % 604800.0
        for row in entries:
            prn = int(row["prn"])
            if prn in support["healthy"]:
                los_by_key[(scenario, bin_index, prn)] = satellite_observation(
                    support["receiver_ecef"], support["healthy"][prn], tow
                ).los_ecef

    trial_rows: list[dict[str, Any]] = []
    for trial in range(trials):
        by_scenario: dict[str, list[dict[str, Any]]] = {}
        for (scenario, bin_index), entries in sorted(groups.items()):
            support = geometry[scenario]
            valid = [row for row in entries if int(row["prn"]) in support["healthy"]]
            if len(valid) < 7:
                continue
            chosen = stable_subset(valid, scenario=scenario, trial=trial)
            los = np.asarray([
                los_by_key[(scenario, bin_index, int(row["prn"]))] for row in chosen
            ])
            delays = np.asarray([float(row["delay"]) for row in chosen])
            fit = fit_clock_centered_geometry(los, delays)
            residual = float(fit.clock_centered_normalized_residual)
            partial_p = partial_f_p_value(residual, 7)
            by_scenario.setdefault(scenario, []).append({
                "bin_index": bin_index,
                "region": frozen.source_region(
                    scenario, str(chosen[0]["role"]), float(bin_index)
                ),
                "residual": residual,
                "raw_alarm": bool(
                    residual <= threshold
                ),
                "partial_f_p_value": partial_p,
                "raw_alarm_support_normalized": partial_p <= partial_f_p_threshold,
                "multipath_enriched": bool(
                    np.quantile([row["asymmetry"] for row in chosen], 0.75)
                    >= distortion_threshold
                ),
            })
        flattened: list[dict[str, Any]] = []
        for scenario, rows in by_scenario.items():
            rows.sort(key=lambda row: int(row["bin_index"]))
            persistent = frozen.persistent_alarm(
                np.asarray([row["raw_alarm"] for row in rows]),
                np.asarray([row["bin_index"] for row in rows]),
                window=5,
                required=3,
            )
            support_persistent = frozen.persistent_alarm(
                np.asarray([row["raw_alarm_support_normalized"] for row in rows]),
                np.asarray([row["bin_index"] for row in rows]),
                window=5,
                required=3,
            )
            for row, alarm, support_alarm in zip(
                rows, persistent, support_persistent
            ):
                row["scenario"] = scenario
                row["persistent_alarm"] = bool(alarm)
                row["persistent_alarm_support_normalized"] = bool(support_alarm)
                flattened.append(row)
        negative = [
            row for row in flattened
            if (row["scenario"] == "cleanDynamic" and row["region"] == "locked_normal")
            or (row["scenario"] == "ds7" and row["region"] == "stable_pre")
        ]
        enriched_negative = [row for row in negative if row["multipath_enriched"]]
        ds7_post = [
            row for row in flattened
            if row["scenario"] == "ds7" and row["region"] == "stable_post"
        ]
        detected = [row for row in ds7_post if row["persistent_alarm"]]
        support_detected = [
            row for row in ds7_post
            if row["persistent_alarm_support_normalized"]
        ]
        delay = (
            float(min(row["bin_index"] + 1 for row in detected) - 100.0)
            if detected else None
        )
        support_delay = (
            float(min(row["bin_index"] + 1 for row in support_detected) - 100.0)
            if support_detected else None
        )
        negative_rate = float(np.mean([row["persistent_alarm"] for row in negative]))
        enriched_rate = (
            float(np.mean([row["persistent_alarm"] for row in enriched_negative]))
            if enriched_negative else 0.0
        )
        support_negative_rate = float(np.mean([
            row["persistent_alarm_support_normalized"] for row in negative
        ]))
        support_enriched_rate = (
            float(np.mean([
                row["persistent_alarm_support_normalized"]
                for row in enriched_negative
            ]))
            if enriched_negative else 0.0
        )
        trial_rows.append({
            "trial": trial,
            "negative_bin_count": len(negative),
            "legacy_negative_persistent_alarm_rate": negative_rate,
            "support_normalized_negative_persistent_alarm_rate": support_negative_rate,
            "enriched_negative_bin_count": len(enriched_negative),
            "legacy_enriched_negative_persistent_alarm_rate": enriched_rate,
            "support_normalized_enriched_negative_persistent_alarm_rate": support_enriched_rate,
            "ds7_post_bin_count": len(ds7_post),
            "legacy_ds7_persistent_detection": bool(detected),
            "legacy_ds7_detection_delay_s": delay,
            "support_normalized_ds7_persistent_detection": bool(support_detected),
            "support_normalized_ds7_detection_delay_s": support_delay,
            "legacy_specificity_gates_passed": (
                negative_rate <= 0.05 and enriched_rate <= 0.10
            ),
            "support_normalized_specificity_gates_passed": (
                support_negative_rate <= 0.05 and support_enriched_rate <= 0.10
            ),
            "legacy_primary_delay_gate_passed": (
                delay is not None and delay <= 60.0
            ),
            "support_normalized_primary_delay_gate_passed": (
                support_delay is not None and support_delay <= 60.0
            ),
        })

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "trial_metrics.csv"
    write_csv(metrics_path, trial_rows)
    legacy_rates = np.asarray([
        row["legacy_negative_persistent_alarm_rate"] for row in trial_rows
    ])
    support_rates = np.asarray([
        row["support_normalized_negative_persistent_alarm_rate"]
        for row in trial_rows
    ])
    legacy_delays = [
        float(row["legacy_ds7_detection_delay_s"])
        for row in trial_rows
        if row["legacy_ds7_detection_delay_s"] is not None
    ]
    support_delays = [
        float(row["support_normalized_ds7_detection_delay_s"])
        for row in trial_rows
        if row["support_normalized_ds7_detection_delay_s"] is not None
    ]
    result = {
        "schema": "gnss-doppler-lab.cgc-seven-prn-support-audit",
        "schema_version": 2,
        "status": (
            "PARTIAL_F_SEVEN_PRN_SPECIFICITY_SUPPORTED"
            if all(
                row["support_normalized_specificity_gates_passed"]
                for row in trial_rows
            )
            else "PARTIAL_F_SEVEN_PRN_SPECIFICITY_NOT_SUPPORTED"
        ),
        "subset_size": 7,
        "deterministic_subset_trials": trials,
        "partial_f": {
            "formula": "F=((SSE_clock-SSE_direction_clock)/3)/(SSE_direction_clock/(N-4)); p=sf_F(F;3,N-4)",
            "calibration_source": "cleanStatic bins [330,420), unchanged 5th-percentile policy",
            "p_value_alarm_threshold": partial_f_p_threshold,
            "external_or_s1_threshold_tuning": False,
            "specificity_trial_pass_rate": float(np.mean([
                row["support_normalized_specificity_gates_passed"]
                for row in trial_rows
            ])),
            "negative_persistent_alarm_rate": {
                "minimum": float(support_rates.min()),
                "median": float(np.median(support_rates)),
                "maximum": float(support_rates.max()),
            },
            "ds7_detection_trial_rate": float(np.mean([
                row["support_normalized_ds7_persistent_detection"]
                for row in trial_rows
            ])),
            "ds7_detection_delay_s": {
                "minimum": min(support_delays) if support_delays else None,
                "median": (
                    float(np.median(support_delays)) if support_delays else None
                ),
                "maximum": max(support_delays) if support_delays else None,
            },
            "legacy_primary_delay_gate_trial_pass_rate": float(np.mean([
                row["support_normalized_primary_delay_gate_passed"]
                for row in trial_rows
            ])),
        },
        "legacy_unadjusted_residual": {
            "threshold_refitting": False,
            "specificity_trial_pass_rate": float(np.mean([
                row["legacy_specificity_gates_passed"] for row in trial_rows
            ])),
            "negative_persistent_alarm_rate": {
                "minimum": float(legacy_rates.min()),
                "median": float(np.median(legacy_rates)),
                "maximum": float(legacy_rates.max()),
            },
            "ds7_detection_trial_rate": float(np.mean([
                row["legacy_ds7_persistent_detection"] for row in trial_rows
            ])),
            "ds7_detection_delay_s": {
                "minimum": min(legacy_delays) if legacy_delays else None,
                "median": float(np.median(legacy_delays)) if legacy_delays else None,
                "maximum": max(legacy_delays) if legacy_delays else None,
            },
        },
        "claim_boundary": (
            "exact-seven-PRN support audit. The partial-F score is calibrated "
            "only on the original cleanStatic calibration interval; S1 and "
            "attack outcomes do not tune its threshold."
        ),
        "inputs": {
            "frozen_summary": {
                "path": str(summary_path.resolve()),
                "sha256": sha256(summary_path),
            },
            "delay_estimates": delay_record,
            "detection_scores": detection_record,
            "config": {
                "path": str(config_path.resolve()),
                "sha256": sha256(config_path),
            },
        },
        "artifacts": {
            "trial_metrics": {
                "path": str(metrics_path),
                "sha256": sha256(metrics_path),
                "row_count": len(trial_rows),
            }
        },
    }
    summary_out = output / "summary.json"
    summary_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frozen-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config, args.frozen_summary, args.output, args.trials)
