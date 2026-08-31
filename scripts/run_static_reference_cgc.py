#!/usr/bin/env python3
"""Run the frozen static pre-attack-reference CGC audit."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from gnss_doppler_lab.gcmr_experiment import parse_preonset_nmea_position  # noqa: E402
from gnss_doppler_lab.gcmr_geometry import (  # noqa: E402
    GPS_WEEK_SECONDS,
    ephemeris_health_selection,
    parse_gnss_sdr_gps_ephemeris_xml,
    satellite_observation,
)
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table  # noqa: E402
from gnss_doppler_lab.static_reference_geometry import (  # noqa: E402
    StaticReferenceGeometryScore,
    clean_only_thresholds,
    joint_raw_alarm,
    persistent_alarm_by_second,
    score_static_reference_geometry,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/static_reference_cgc_v1.json"
RELEASE_INPUTS = (
    "configs/experiments/static_reference_cgc_v1.json",
    "docs/results/static_reference_cgc_protocol_v1.md",
    "src/gnss_doppler_lab/static_reference_geometry.py",
    "scripts/run_static_reference_cgc.py",
    "tests/test_static_reference_geometry.py",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def verify(record: dict[str, str], label: str) -> Path:
    path = resolve(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256(path)
    if observed != record["sha256"]:
        raise ValueError(f"{label} hash mismatch: {observed}")
    return path


def verify_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        if subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative], cwd=REPO_ROOT
        ).returncode:
            raise ValueError(f"release input is not committed and clean: {relative}")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    return {
        "head_commit": commit,
        "input_sha256": {relative: sha256(REPO_ROOT / relative) for relative in RELEASE_INPUTS},
    }


def validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != "gnss-doppler-lab.static-reference-cgc-config"
        or config.get("schema_version") != 1
    ):
        raise ValueError("unsupported static reference-CGC config")
    analysis = config["analysis"]
    expected = {
        "minimum_observations_per_prn_bin": 30,
        "minimum_prns": 8,
        "baseline_seconds": [40, 110],
        "clean_calibration_seconds": [134, 344],
        "clean_held_seconds": [361, 479],
        "excluded_transition_seconds": [110, 160],
        "oakbat_stable_post_seconds": [160, 479],
        "persistence_window_seconds": 5,
        "persistence_required_seconds": 3,
    }
    for key, value in expected.items():
        if analysis.get(key) != value:
            raise ValueError(f"frozen analysis contract drifted: {key}")
    if [row["id"] for row in config["oakbat"]["scenarios"]] != [
        "cleanStatic", "os2", "os3", "os4"
    ]:
        raise ValueError("OAKBAT scenario roster drifted")
    if [row["id"] for row in config["matched_static_multipath"]] != [
        "pv1-pair-001", "pv1-pair-002"
    ]:
        raise ValueError("matched multipath roster drifted")


def load_pseudorange_bins(
    path: Path,
    *,
    start_second: int,
    end_second: int,
    sample_period_s: float,
    minimum_observations: int,
) -> dict[tuple[int, int], float]:
    with h5py.File(path, "r") as handle:
        required = ("Flag_valid_pseudorange", "PRN", "Pseudorange_m")
        if any(name not in handle for name in required):
            raise ValueError(f"pseudorange arrays missing from {path}")
        valid = np.asarray(handle["Flag_valid_pseudorange"]) > 0.5
        prn = np.asarray(handle["PRN"]).astype(np.int64)
        pseudorange = np.asarray(handle["Pseudorange_m"], dtype=np.float64)
    if not (valid.shape == prn.shape == pseudorange.shape):
        raise ValueError("pseudorange arrays have inconsistent shapes")
    per_second = int(round(1.0 / float(sample_period_s)))
    result: dict[tuple[int, int], float] = {}
    for second in range(start_second, end_second):
        rows = slice(second * per_second, (second + 1) * per_second)
        for identifier in np.unique(prn[rows]):
            identifier = int(identifier)
            if not 1 <= identifier <= 32:
                continue
            selected = valid[rows] & (prn[rows] == identifier)
            values = pseudorange[rows][selected]
            values = values[np.isfinite(values) & (values > 0.0)]
            if len(values) >= minimum_observations:
                result[second, identifier] = float(np.median(values))
    return result


def baseline_coefficients(
    residuals: dict[tuple[int, int], float],
    *,
    start_second: int,
    end_second: int,
    minimum_bins: int,
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    identifiers = sorted({prn for _, prn in residuals})
    for prn in identifiers:
        seconds = [
            second for second in range(start_second, end_second)
            if (second, prn) in residuals
        ]
        if len(seconds) < minimum_bins:
            continue
        x = np.asarray(seconds, dtype=np.float64) + 0.5
        y = np.asarray([residuals[second, prn] for second in seconds], dtype=np.float64)
        result[prn] = np.polyfit(x, y, 1)
    return result


def _score_row(second: int, score: StaticReferenceGeometryScore) -> dict[str, Any]:
    return {
        "second": int(second),
        "prn_count": score.prn_count,
        "rank": score.rank,
        "displacement_x_m": float(score.displacement[0]),
        "displacement_y_m": float(score.displacement[1]),
        "displacement_z_m": float(score.displacement[2]),
        "displacement_norm_m": score.displacement_norm,
        "clock_bias_m": score.clock_bias,
        "clock_centered_residual": score.clock_centered_residual,
        "directional_coherence": score.directional_coherence,
        "partial_f": score.partial_f,
        "partial_f_p_value": score.partial_f_p_value,
    }


def score_oakbat_scenario(
    scenario: dict[str, Any], analysis: dict[str, Any], source_iq_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = {
        key: verify(scenario[key], f"{scenario['id']}.{key}")
        for key in ("observables", "ephemeris", "nmea", "receiver_manifest")
    }
    if "nmea_manifest" in scenario:
        paths["nmea_manifest"] = verify(scenario["nmea_manifest"], "cleanStatic.nmea_manifest")
    manifest = json.loads(paths["receiver_manifest"].read_text(encoding="utf-8"))
    if manifest["source"]["iq_sha256"] != source_iq_sha256:
        raise ValueError(f"{scenario['id']} source IQ identity drifted")
    if "nmea_manifest" in paths:
        nmea_manifest = json.loads(paths["nmea_manifest"].read_text(encoding="utf-8"))
        if nmea_manifest["source"]["iq_sha256"] != source_iq_sha256:
            raise ValueError("cleanStatic NMEA is not derived from the same source IQ")

    baseline_start, baseline_end = map(int, analysis["baseline_seconds"])
    position = parse_preonset_nmea_position(
        paths["nmea"],
        gps_tow_at_time_zero_s=float(analysis["oakbat_geometry_tow0_s"]),
        onset_s=float(analysis["oakbat_attack_onset_s"]),
        position_window_s=(baseline_start, baseline_end),
    )
    receiver_ecef = np.asarray(position["ecef"], dtype=np.float64)
    ephemerides = parse_gnss_sdr_gps_ephemeris_xml(paths["ephemeris"])
    healthy, health = ephemeris_health_selection(
        ephemerides, tracked_prns=set(ephemerides), min_prns=int(analysis["minimum_prns"])
    )
    pseudorange = load_pseudorange_bins(
        paths["observables"], start_second=0, end_second=479,
        sample_period_s=float(analysis["sample_period_s"]),
        minimum_observations=int(analysis["minimum_observations_per_prn_bin"]),
    )
    residuals: dict[tuple[int, int], float] = {}
    los: dict[tuple[int, int], np.ndarray] = {}
    tow0 = float(analysis["oakbat_geometry_tow0_s"])
    for (second, prn), value in pseudorange.items():
        if prn not in healthy:
            continue
        observation = satellite_observation(
            receiver_ecef, healthy[prn], (tow0 + second + 0.5) % GPS_WEEK_SECONDS
        )
        residuals[second, prn] = value - observation.range_m
        los[second, prn] = np.asarray(observation.los_ecef, dtype=np.float64)
    coefficients = baseline_coefficients(
        residuals, start_second=baseline_start, end_second=baseline_end,
        minimum_bins=int(analysis["minimum_baseline_bins_per_prn"]),
    )
    if len(coefficients) < int(analysis["minimum_prns"]):
        raise ValueError(f"{scenario['id']} has too few baseline-supported PRNs")

    intermediate: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    first_pass: dict[int, StaticReferenceGeometryScore] = {}
    for second in range(baseline_start, 479):
        prns = sorted(prn for prn in coefficients if (second, prn) in residuals)
        if len(prns) < int(analysis["minimum_prns"]):
            continue
        signed = np.asarray([
            residuals[second, prn] - np.polyval(coefficients[prn], second + 0.5)
            for prn in prns
        ], dtype=np.float64)
        los_matrix = np.asarray([los[second, prn] for prn in prns], dtype=np.float64)
        intermediate[second] = (los_matrix, signed)
        first_pass[second] = score_static_reference_geometry(los_matrix, signed)
    baseline_fits = [
        score.displacement for second, score in first_pass.items()
        if baseline_start <= second < baseline_end
    ]
    if not baseline_fits:
        raise ValueError(f"{scenario['id']} has no baseline geometry fit")
    reference = np.median(np.stack(baseline_fits), axis=0)
    rows = [
        _score_row(second, score_static_reference_geometry(
            values[0], values[1], displacement_reference_m=reference
        ))
        for second, values in sorted(intermediate.items())
    ]
    metadata = {
        "scenario": scenario["id"],
        "role": scenario["role"],
        "expected_physics": scenario["expected_physics"],
        "receiver_position": {
            "ecef_m": receiver_ecef.tolist(), "llh": position["llh"],
            "sample_count": position["sample_count"],
        },
        "baseline_prns": sorted(coefficients),
        "baseline_reference_displacement_m": reference.tolist(),
        "geometry_bin_count": len(rows),
        "ephemeris_health": health,
        "inputs": scenario,
    }
    return rows, metadata


def score_matched_multipath(
    pair: dict[str, Any], analysis: dict[str, Any]
) -> list[dict[str, Any]]:
    normal_path = verify(pair["normal_observables"], f"{pair['id']}.normal")
    multipath_path = verify(pair["multipath_observables"], f"{pair['id']}.multipath")
    los_path = verify(pair["los_log"], f"{pair['id']}.los")
    start, end = map(int, analysis["synthetic_analysis_seconds"])
    kwargs = {
        "start_second": start, "end_second": end,
        "sample_period_s": float(analysis["sample_period_s"]),
        "minimum_observations": int(analysis["minimum_observations_per_prn_bin"]),
    }
    normal = load_pseudorange_bins(normal_path, **kwargs)
    multipath = load_pseudorange_bins(multipath_path, **kwargs)
    los = {
        int(name[1:]): np.asarray(vector, dtype=np.float64)
        for name, vector in parse_gps_sdr_sim_los_table(
            los_path.read_text(encoding="utf-8")
        ).items()
    }
    rows: list[dict[str, Any]] = []
    for second in range(start, end):
        prns = sorted(
            prn for prn in los
            if (second, prn) in normal and (second, prn) in multipath
        )
        if len(prns) < int(analysis["minimum_prns"]):
            continue
        signed = np.asarray([
            multipath[second, prn] - normal[second, prn] for prn in prns
        ], dtype=np.float64)
        score = score_static_reference_geometry(
            np.asarray([los[prn] for prn in prns], dtype=np.float64), signed
        )
        rows.append(_score_row(second, score))
    return rows


def apply_decision(
    rows: list[dict[str, Any]], thresholds: dict[str, float | int],
    *, start_second: int, end_second: int, analysis: dict[str, Any],
) -> None:
    raw: dict[int, bool] = {}
    for row in rows:
        proxy = StaticReferenceGeometryScore(
            displacement=np.asarray([
                row["displacement_x_m"], row["displacement_y_m"], row["displacement_z_m"]
            ], dtype=np.float64),
            displacement_norm=float(row["displacement_norm_m"]),
            clock_bias=float(row["clock_bias_m"]),
            clock_centered_residual=float(row["clock_centered_residual"]),
            directional_coherence=float(row["directional_coherence"]),
            partial_f=float(row["partial_f"]),
            partial_f_p_value=float(row["partial_f_p_value"]),
            prn_count=int(row["prn_count"]), rank=int(row["rank"]),
        )
        raw[int(row["second"])] = joint_raw_alarm(proxy, thresholds)
    persistent = persistent_alarm_by_second(
        raw, start_second=start_second, end_second=end_second,
        window_seconds=int(analysis["persistence_window_seconds"]),
        required_seconds=int(analysis["persistence_required_seconds"]),
    )
    for row in rows:
        second = int(row["second"])
        row["raw_alarm"] = bool(raw[second])
        row["persistent_alarm"] = bool(persistent[second])


def summarize(rows: list[dict[str, Any]], interval: Iterable[int]) -> dict[str, Any]:
    start, end = map(int, interval)
    selected = [row for row in rows if start <= int(row["second"]) < end]
    if not selected:
        return {"interval_seconds": [start, end], "available_bin_count": 0}
    persistent = np.asarray([row["persistent_alarm"] for row in selected], dtype=bool)
    raw = np.asarray([row["raw_alarm"] for row in selected], dtype=bool)
    first = next((int(row["second"]) for row in selected if row["persistent_alarm"]), None)
    return {
        "interval_seconds": [start, end],
        "available_bin_count": len(selected),
        "minimum_prn_count": min(int(row["prn_count"]) for row in selected),
        "raw_alarm_count": int(raw.sum()),
        "raw_alarm_rate": float(raw.mean()),
        "persistent_alarm_count": int(persistent.sum()),
        "persistent_alarm_rate": float(persistent.mean()),
        "first_persistent_alarm_s": first,
        "median_displacement_norm_m": float(np.median([
            row["displacement_norm_m"] for row in selected
        ])),
        "median_clock_centered_residual": float(np.median([
            row["clock_centered_residual"] for row in selected
        ])),
        "median_partial_f_p_value": float(np.median([
            row["partial_f_p_value"] for row in selected
        ])),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    release = verify_release()
    analysis = config["analysis"]
    output = resolve(config["output_root"])
    if output.exists():
        raise FileExistsError(f"frozen output already exists: {output}")

    oakbat_rows: dict[str, list[dict[str, Any]]] = {}
    oakbat_meta: dict[str, dict[str, Any]] = {}
    for scenario in config["oakbat"]["scenarios"]:
        rows, metadata = score_oakbat_scenario(
            scenario, analysis, config["oakbat"]["source_iq_sha256"][scenario["id"]]
        )
        oakbat_rows[scenario["id"]] = rows
        oakbat_meta[scenario["id"]] = metadata

    calibration_start, calibration_end = map(int, analysis["clean_calibration_seconds"])
    calibration = [
        row for row in oakbat_rows["cleanStatic"]
        if calibration_start <= int(row["second"]) < calibration_end
    ]
    thresholds = clean_only_thresholds(
        [row["partial_f_p_value"] for row in calibration],
        [row["displacement_norm_m"] for row in calibration],
        p_quantile=float(analysis["clean_partial_f_lower_quantile"]),
        displacement_quantile=float(analysis["clean_displacement_upper_quantile"]),
    )
    for rows in oakbat_rows.values():
        apply_decision(rows, thresholds, start_second=40, end_second=479, analysis=analysis)

    multipath_rows: dict[str, list[dict[str, Any]]] = {}
    for pair in config["matched_static_multipath"]:
        rows = score_matched_multipath(pair, analysis)
        apply_decision(rows, thresholds, start_second=18, end_second=30, analysis=analysis)
        multipath_rows[pair["id"]] = rows

    clean_held = summarize(oakbat_rows["cleanStatic"], analysis["clean_held_seconds"])
    post = {
        name: summarize(oakbat_rows[name], analysis["oakbat_stable_post_seconds"])
        for name in ("os2", "os3", "os4")
    }
    multipath = {
        name: summarize(rows, analysis["synthetic_comparison_seconds"])
        for name, rows in multipath_rows.items()
    }
    gates = config["decision_gates"]
    os4_first = post["os4"].get("first_persistent_alarm_s")
    decisions = {
        "minimum_clean_calibration_bins": len(calibration) >= int(gates["minimum_clean_calibration_bins"]),
        "clean_held_false_alarm": clean_held.get("persistent_alarm_rate", 1.0)
        <= float(gates["maximum_clean_held_persistent_alarm_rate"]),
        "os2_clock_push_rejected": post["os2"].get("persistent_alarm_rate", 1.0)
        <= float(gates["maximum_clock_push_persistent_alarm_rate"]),
        "os3_clock_push_rejected": post["os3"].get("persistent_alarm_rate", 1.0)
        <= float(gates["maximum_clock_push_persistent_alarm_rate"]),
        "os4_position_push_detected": os4_first is not None,
        "os4_detection_delay": os4_first is not None and
        os4_first - int(analysis["oakbat_attack_onset_s"])
        <= float(gates["maximum_os4_detection_delay_s"]),
        "matched_multipath_rejected": all(
            result.get("persistent_alarm_count", 1) == 0 for result in multipath.values()
        ),
    }
    supported = all(decisions.values())
    output.mkdir(parents=True)
    for name, rows in oakbat_rows.items():
        write_csv(output / f"oakbat_{name}_scores.csv", rows)
    for name, rows in multipath_rows.items():
        write_csv(output / f"{name}_multipath_scores.csv", rows)
    summary = {
        "schema": "gnss-doppler-lab.static-reference-cgc-result",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STATIC_REFERENCE_CGC_SUPPORTED" if supported else "STATIC_REFERENCE_CGC_NOT_SUPPORTED",
        "all_preregistered_gates_passed": supported,
        "gates": decisions,
        "thresholds": thresholds,
        "clean_held": clean_held,
        "oakbat_stable_post": post,
        "matched_static_multipath": multipath,
        "oakbat_metadata": oakbat_meta,
        "release": release,
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "limitations": [
            "The OAKBAT recordings had been opened for earlier detectors; this is a preregistered retrospective score, not an untouched external trial.",
            "The multipath controls are synthetic matched receiver-RF realizations, not labeled field multipath.",
            "Per-PRN linear clean trends compensate omitted satellite-clock, atmosphere, transmit-time, and Sagnac terms over this static audit.",
            "The method assumes a trusted static pre-attack interval and does not apply to a moving receiver without a separate motion reference.",
            "OS2 and OS3 are intentional common-time-spoof controls; rejecting them is not a claim of universal spoof detection.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"], "thresholds": thresholds,
        "clean_held": clean_held, "oakbat_stable_post": post,
        "matched_static_multipath": multipath, "gates": decisions,
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
