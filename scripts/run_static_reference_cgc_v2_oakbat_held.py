#!/usr/bin/env python3
"""Preflight, receive, and score the frozen OAKBAT OS5/OS6 held pair."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_oakbat_9tap_detection_pipeline as receiver  # noqa: E402
import run_static_reference_cgc as v1  # noqa: E402
from gnss_doppler_lab.gcmr_experiment import parse_preonset_nmea_position  # noqa: E402
from gnss_doppler_lab.gcmr_geometry import (  # noqa: E402
    GPS_WEEK_SECONDS,
    ephemeris_health_selection,
    parse_gnss_sdr_gps_ephemeris_xml,
    satellite_observation,
)
from gnss_doppler_lab.static_reference_geometry import score_static_reference_geometry  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/static_reference_cgc_v2_oakbat_held.json"
RELEASE_INPUTS = (
    "configs/experiments/static_reference_cgc_v2_oakbat_held.json",
    "docs/results/static_reference_cgc_v2_oakbat_held_protocol.md",
    "scripts/run_static_reference_cgc_v2_oakbat_held.py",
)


def verify_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative], cwd=REPO_ROOT,
            check=True, capture_output=True,
        )
        if subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative], cwd=REPO_ROOT
        ).returncode:
            raise ValueError(f"held release input is not committed and clean: {relative}")
    return {
        "head_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "input_sha256": {relative: v1.sha256(REPO_ROOT / relative) for relative in RELEASE_INPUTS},
    }


def validate_config(config: dict[str, Any]) -> dict[str, Path]:
    if (
        config.get("schema") != "gnss-doppler-lab.static-reference-cgc-oakbat-held-config"
        or config.get("schema_version") != 1
    ):
        raise ValueError("unsupported held config")
    if config["experiment"]["held_scenarios"] != ["os5", "os6"]:
        raise ValueError("held scenario roster drifted")
    if [row["id"] for row in config["held"]] != ["os5", "os6"]:
        raise ValueError("held input order drifted")
    expected = {
        "baseline_seconds": [40, 110], "stable_post_seconds": [160, 479],
        "minimum_prns": 8, "minimum_observations_per_prn_bin": 30,
        "persistence_window_seconds": 5, "persistence_required_seconds": 3,
    }
    for key, value in expected.items():
        if config["analysis"].get(key) != value:
            raise ValueError(f"held analysis contract drifted: {key}")
    paths = {
        "v1_summary": v1.verify(config["frozen_v1"]["summary"], "v1 summary"),
        "clean_scores": v1.verify(config["frozen_v1"]["clean_scores"], "v1 clean scores"),
        "adapter": v1.verify(config["receiver"]["adapter"], "receiver adapter"),
        "executable": v1.verify(config["receiver"]["executable"], "receiver executable"),
    }
    summary = json.loads(paths["v1_summary"].read_text(encoding="utf-8"))
    frozen = config["frozen_v1"]
    if (
        float(summary["thresholds"]["partial_f_p_alarm_threshold"])
        != float(frozen["partial_f_p_alarm_threshold"])
        or float(summary["thresholds"]["displacement_alarm_threshold_m"])
        != float(frozen["displacement_alarm_threshold_m"])
    ):
        raise ValueError("v1 decision thresholds drifted")
    with paths["clean_scores"].open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if 40 <= int(row["second"]) < 110]
    clean_q = float(np.quantile(
        [float(row["displacement_norm_m"]) for row in rows],
        float(frozen["clean_baseline_eligibility_quantile"]),
    ))
    if not math.isclose(
        clean_q, float(frozen["clean_baseline_eligibility_threshold_m"]),
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError("clean-only reference eligibility threshold drifted")
    for held in config["held"]:
        iq = v1.resolve(held["iq"]["path"])
        if not iq.is_file() or iq.stat().st_size != int(held["iq"]["bytes"]):
            raise ValueError(f"{held['id']} IQ byte contract failed")
        if v1.sha256(iq) != held["iq"]["sha256"]:
            raise ValueError(f"{held['id']} IQ hash mismatch")
        paths[f"iq_{held['id']}"] = iq
    return paths


def _residuals(
    pseudorange: dict[tuple[int, int], float], receiver_ecef: np.ndarray,
    ephemerides: dict[int, Any], tow0: float,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], np.ndarray]]:
    residuals: dict[tuple[int, int], float] = {}
    los: dict[tuple[int, int], np.ndarray] = {}
    for (second, prn), value in pseudorange.items():
        if prn not in ephemerides:
            continue
        observation = satellite_observation(
            receiver_ecef, ephemerides[prn], (tow0 + second + 0.5) % GPS_WEEK_SECONDS
        )
        residuals[second, prn] = value - observation.range_m
        los[second, prn] = np.asarray(observation.los_ecef, dtype=np.float64)
    return residuals, los


def _geometry_rows(
    residuals: dict[tuple[int, int], float],
    los: dict[tuple[int, int], np.ndarray],
    coefficients: dict[int, np.ndarray],
    reference: np.ndarray | None,
    *, start_second: int, end_second: int, minimum_prns: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    first: list[tuple[int, np.ndarray, np.ndarray, Any]] = []
    for second in range(start_second, end_second):
        prns = sorted(prn for prn in coefficients if (second, prn) in residuals)
        if len(prns) < minimum_prns:
            continue
        signed = np.asarray([
            residuals[second, prn] - np.polyval(coefficients[prn], second + 0.5)
            for prn in prns
        ], dtype=np.float64)
        matrix = np.asarray([los[second, prn] for prn in prns], dtype=np.float64)
        score = score_static_reference_geometry(matrix, signed)
        first.append((second, matrix, signed, score))
    if not first:
        raise ValueError("no supported geometry bins")
    frozen_reference = (
        np.median(np.stack([item[3].displacement for item in first]), axis=0)
        if reference is None else np.asarray(reference, dtype=np.float64)
    )
    rows = [
        v1._score_row(second, score_static_reference_geometry(
            matrix, signed, displacement_reference_m=frozen_reference
        ))
        for second, matrix, signed, _ in first
    ]
    return rows, frozen_reference


def score_held(
    scenario: str, run_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    analysis = config["analysis"]
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_iq = next(row["iq"]["sha256"] for row in config["held"] if row["id"] == scenario)
    if manifest["source"]["iq_sha256"] != expected_iq:
        raise ValueError(f"{scenario} receiver source identity drifted")
    position = parse_preonset_nmea_position(
        run_dir / "nmea_pvt.nmea",
        gps_tow_at_time_zero_s=float(analysis["geometry_tow0_s"]),
        onset_s=float(analysis["attack_onset_s"]), position_window_s=(40, 110),
    )
    receiver_ecef = np.asarray(position["ecef"], dtype=np.float64)
    ephemerides = parse_gnss_sdr_gps_ephemeris_xml(run_dir / "gps_ephemeris.xml")
    healthy, health = ephemeris_health_selection(
        ephemerides, tracked_prns=set(ephemerides), min_prns=int(analysis["minimum_prns"])
    )

    # Eligibility is completed before any post-onset pseudorange row is read.
    baseline_pseudorange = v1.load_pseudorange_bins(
        run_dir / "raw/observables.mat", start_second=0, end_second=110,
        sample_period_s=float(analysis["sample_period_s"]),
        minimum_observations=int(analysis["minimum_observations_per_prn_bin"]),
    )
    baseline_residuals, baseline_los = _residuals(
        baseline_pseudorange, receiver_ecef, healthy, float(analysis["geometry_tow0_s"])
    )
    coefficients = v1.baseline_coefficients(
        baseline_residuals, start_second=40, end_second=110,
        minimum_bins=int(analysis["minimum_baseline_bins_per_prn"]),
    )
    if len(coefficients) < int(analysis["minimum_prns"]):
        raise ValueError(f"{scenario} has too few baseline-supported PRNs")
    baseline_rows, reference = _geometry_rows(
        baseline_residuals, baseline_los, coefficients, None,
        start_second=40, end_second=110, minimum_prns=int(analysis["minimum_prns"]),
    )
    eligibility_value = float(np.quantile(
        [row["displacement_norm_m"] for row in baseline_rows],
        float(config["frozen_v1"]["clean_baseline_eligibility_quantile"]),
    ))
    eligible = eligibility_value <= float(
        config["frozen_v1"]["clean_baseline_eligibility_threshold_m"]
    )
    base = {
        "scenario": scenario,
        "reference_eligible": eligible,
        "reference_eligibility_q99_m": eligibility_value,
        "reference_eligibility_threshold_m": float(
            config["frozen_v1"]["clean_baseline_eligibility_threshold_m"]
        ),
        "baseline_bin_count": len(baseline_rows),
        "baseline_prns": sorted(coefficients),
        "post_score_accessed": False,
        "receiver_position_ecef_m": receiver_ecef.tolist(),
        "ephemeris_health": health,
        "receiver_manifest": {"path": str(manifest_path), "sha256": v1.sha256(manifest_path)},
    }
    if not eligible:
        return {**base, "baseline_rows": baseline_rows, "post_rows": []}

    all_pseudorange = v1.load_pseudorange_bins(
        run_dir / "raw/observables.mat", start_second=0, end_second=479,
        sample_period_s=float(analysis["sample_period_s"]),
        minimum_observations=int(analysis["minimum_observations_per_prn_bin"]),
    )
    residuals, los = _residuals(
        all_pseudorange, receiver_ecef, healthy, float(analysis["geometry_tow0_s"])
    )
    rows, _ = _geometry_rows(
        residuals, los, coefficients, reference,
        start_second=40, end_second=479, minimum_prns=int(analysis["minimum_prns"]),
    )
    thresholds = {
        "partial_f_p_alarm_threshold": float(config["frozen_v1"]["partial_f_p_alarm_threshold"]),
        "displacement_alarm_threshold_m": float(config["frozen_v1"]["displacement_alarm_threshold_m"]),
    }
    v1.apply_decision(rows, thresholds, start_second=40, end_second=479, analysis={
        "persistence_window_seconds": analysis["persistence_window_seconds"],
        "persistence_required_seconds": analysis["persistence_required_seconds"],
    })
    return {**base, "post_score_accessed": True, "baseline_rows": baseline_rows, "post_rows": rows}


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = validate_config(config)
    release = verify_release()
    output = v1.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"held result already exists: {summary_path}")
    receiver.preflight_output_space(
        output, scenario_count=2,
        minimum_free_bytes=int(config["receiver"]["minimum_free_gib"] * 1024**3),
    )

    results: dict[str, dict[str, Any]] = {}
    for held in config["held"]:
        scenario = held["id"]
        scenario_root = output / "preprocessed" / scenario
        scenario_root.mkdir(parents=True, exist_ok=True)
        manifest_path = receiver.run_receiver(
            scenario, paths[f"iq_{scenario}"], scenario_root,
            exe=str(paths["executable"]), timeout_s=int(config["receiver"]["timeout_s"]),
            force=False,
        )
        results[scenario] = score_held(scenario, manifest_path.parent, config)
        v1.write_csv(output / f"{scenario}_baseline_scores.csv", results[scenario]["baseline_rows"])
        if results[scenario]["post_rows"]:
            v1.write_csv(output / f"{scenario}_scores.csv", results[scenario]["post_rows"])

    post = {}
    for scenario, result in results.items():
        if not result["post_rows"]:
            post[scenario] = {"status": "ABSTAINED_REFERENCE_INELIGIBLE"}
            continue
        post[scenario] = v1.summarize(result["post_rows"], config["analysis"]["stable_post_seconds"])
    os6_rows = [
        row for row in results["os6"]["post_rows"] if 160 <= int(row["second"]) < 479
    ]
    if os6_rows:
        median_vector = np.median(np.asarray([
            [row["displacement_x_m"], row["displacement_y_m"], row["displacement_z_m"]]
            for row in os6_rows
        ], dtype=np.float64), axis=0)
        direction_cosine = float(median_vector[2] / np.linalg.norm(median_vector))
    else:
        median_vector, direction_cosine = np.full(3, np.nan), None
    os6_first = post["os6"].get("first_persistent_alarm_s")
    gates = {
        "os5_reference_eligible": bool(results["os5"]["reference_eligible"]),
        "os6_reference_eligible": bool(results["os6"]["reference_eligible"]),
        "os5_time_push_rejected": bool(
            results["os5"]["reference_eligible"]
            and post["os5"].get("persistent_alarm_rate", 1.0)
            <= float(config["decision_gates"]["maximum_os5_persistent_alarm_rate"])
        ),
        "os6_position_push_detected": os6_first is not None,
        "os6_detection_delay": bool(
            os6_first is not None
            and os6_first - int(config["analysis"]["attack_onset_s"])
            <= float(config["decision_gates"]["maximum_os6_detection_delay_s"])
        ),
        "os6_ecef_z_direction": bool(
            direction_cosine is not None
            and direction_cosine >= float(
                config["decision_gates"]["minimum_os6_median_ecef_z_direction_cosine"]
            )
        ),
    }
    supported = all(gates.values())
    summary = {
        "schema": "gnss-doppler-lab.static-reference-cgc-oakbat-held-result",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STATIC_REFERENCE_CGC_HELD_SUPPORTED" if supported else "STATIC_REFERENCE_CGC_HELD_NOT_SUPPORTED",
        "all_held_gates_passed": supported,
        "gates": gates,
        "reference_eligibility": {
            scenario: {
                key: result[key] for key in (
                    "reference_eligible", "reference_eligibility_q99_m",
                    "reference_eligibility_threshold_m", "baseline_bin_count",
                    "baseline_prns", "post_score_accessed",
                )
            } for scenario, result in results.items()
        },
        "stable_post": post,
        "os6_median_displacement_ecef_m": median_vector.tolist() if os6_rows else None,
        "os6_median_ecef_z_direction_cosine": direction_cosine,
        "release": release,
        "config": {"path": str(config_path), "sha256": v1.sha256(config_path)},
        "receiver_manifests": {
            scenario: result["receiver_manifest"] for scenario, result in results.items()
        },
        "limitations": [
            "OAKBAT scenario semantics were known, although these receiver outputs and reference-CGC scores were not opened before release.",
            "The eligibility gate can abstain but cannot repair a receiver pseudorange ambiguity failure.",
            "The fixed 89.1-m displacement threshold does not cover the previously observed 70-m FGI case.",
            "This is a static laboratory replay result and not a real-multipath field validation.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": summary["status"], "gates": gates,
        "reference_eligibility": summary["reference_eligibility"],
        "stable_post": post,
        "os6_median_ecef_z_direction_cosine": direction_cosine,
        "output": str(output),
    }, indent=2, allow_nan=False))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
