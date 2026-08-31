#!/usr/bin/env python3
"""Audit loss of prompt-local observability after FGI TGD spoof takeover."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.clock_centered_geometry import fit_clock_centered_geometry  # noqa: E402
from gnss_doppler_lab.gcmr_geometry import (  # noqa: E402
    ephemeris_health_selection,
    parse_gnss_sdr_gps_ephemeris_xml,
    satellite_observation,
)
from gnss_doppler_lab.trajectory import llh_to_ecef  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/experiments/fgi_spoofrepo_tgd_observability_audit_v1.json"
GPS_UTC_LEAP_OFFSET_S = 18.0
GPS_WEEK_SECONDS = 604800.0


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def verify(record: dict[str, str], label: str) -> Path:
    path = resolve(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256(path)
    if observed != record["sha256"]:
        raise ValueError(f"{label} SHA-256 mismatch: {observed}")
    return path


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "gnss-doppler-lab.fgi-spoofrepo-tgd-observability-audit-config":
        raise ValueError("unsupported audit config schema")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported audit config version")
    experiment = config["experiment"]
    if experiment.get("confirmatory_detection_claim") is not False:
        raise ValueError("this audit must remain exploratory")
    if experiment.get("detector_threshold_refitting") is not False:
        raise ValueError("detector refitting is forbidden")
    analysis = config["analysis"]
    expected = {
        "sample_period_s": 0.02,
        "bin_seconds": 1.0,
        "minimum_observations_per_prn_bin": 40,
        "minimum_prns": 8,
        "analysis_seconds": [40, 230],
        "settled_clean_baseline_seconds": [43, 120],
        "excluded_receiver_clock_initialization_seconds": [40, 43],
        "excluded_transition_seconds": [120, 160],
        "stable_post_seconds": [160, 230],
        "nmea_static_position_seconds": [43, 120],
        "chip_rate_hz": 1023000.0,
        "speed_of_light_mps": 299792458.0,
        "baseline_start_sensitivity_seconds": [43, 50, 60, 70, 80],
    }
    for key, value in expected.items():
        if analysis.get(key) != value:
            raise ValueError(f"analysis contract drifted: {key}")
    if sorted(analysis["healthy_prns"]) != [5, 7, 8, 9, 13, 14, 15, 18, 20, 22, 27, 30]:
        raise ValueError("healthy PRN roster drifted")


def _valid_sentence(line: str) -> list[str] | None:
    line = line.strip()
    if not line.startswith("$") or "*" not in line:
        return None
    body, raw = line[1:].rsplit("*", 1)
    try:
        expected = int(raw[:2], 16)
    except ValueError:
        return None
    checksum = 0
    for character in body:
        checksum ^= ord(character)
    return body.split(",") if checksum == expected else None


def _hms(value: str) -> tuple[int, int, float]:
    if len(value) < 6:
        raise ValueError("invalid NMEA time")
    return int(value[:2]), int(value[2:4]), float(value[4:])


def _degree(value: str, hemisphere: str) -> float:
    width = 2 if hemisphere in ("N", "S") else 3
    result = float(value[:width]) + float(value[width:]) / 60.0
    return -result if hemisphere in ("S", "W") else result


def _utc_tow(date: Any, hms: tuple[int, int, float]) -> float:
    hour, minute, second = hms
    return (
        ((date.weekday() + 1) % 7) * 86400.0
        + hour * 3600.0
        + minute * 60.0
        + second
        + GPS_UTC_LEAP_OFFSET_S
    ) % GPS_WEEK_SECONDS


def parse_nmea_ecef_bins(path: Path, tow0_s: float, start_s: int, end_s: int) -> dict[int, np.ndarray]:
    current_date = None
    bins: dict[int, list[np.ndarray]] = {}
    for line in path.read_text(errors="replace").splitlines():
        fields = _valid_sentence(line)
        if not fields:
            continue
        try:
            kind = fields[0][-3:]
            if kind == "RMC" and len(fields) > 9 and fields[2] == "A":
                current_date = datetime.strptime(fields[9], "%d%m%y").date()
            elif (
                kind == "GGA"
                and current_date is not None
                and len(fields) > 11
                and int(fields[6]) > 0
                and fields[9]
                and fields[10] == "M"
            ):
                tow = _utc_tow(current_date, _hms(fields[1]))
                relative = (tow - tow0_s + GPS_WEEK_SECONDS / 2.0) % GPS_WEEK_SECONDS - GPS_WEEK_SECONDS / 2.0
                bin_index = int(np.floor(relative))
                if start_s <= bin_index < end_s:
                    ecef = np.asarray(
                        llh_to_ecef(
                            _degree(fields[2], fields[3]),
                            _degree(fields[4], fields[5]),
                            float(fields[9]),
                        ),
                        dtype=np.float64,
                    )
                    bins.setdefault(bin_index, []).append(ecef)
        except (IndexError, ValueError):
            continue
    return {index: np.median(np.stack(values), axis=0) for index, values in bins.items()}


def median_reference(vectors: Iterable[np.ndarray]) -> np.ndarray:
    values = [np.asarray(vector, dtype=np.float64) for vector in vectors]
    if not values:
        raise ValueError("cannot form reference from no vectors")
    return np.median(np.stack(values), axis=0)


def load_pseudorange_bins(
    path: Path,
    *,
    prns: set[int],
    start_s: int,
    end_s: int,
    sample_period_s: float,
    minimum_observations: int,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    with h5py.File(path, "r") as handle:
        required = ("Flag_valid_pseudorange", "PRN", "Pseudorange_m")
        if any(name not in handle for name in required):
            raise ValueError("observables MAT lacks required pseudorange arrays")
        valid = np.asarray(handle["Flag_valid_pseudorange"]) > 0.5
        identifiers = np.asarray(handle["PRN"]).astype(np.int64)
        pseudorange = np.asarray(handle["Pseudorange_m"], dtype=np.float64)
    if not (valid.shape == identifiers.shape == pseudorange.shape):
        raise ValueError("pseudorange arrays have inconsistent shapes")
    per_bin = int(round(1.0 / sample_period_s))
    medians: dict[tuple[int, int], float] = {}
    counts: dict[tuple[int, int], int] = {}
    for bin_index in range(start_s, end_s):
        rows = slice(bin_index * per_bin, (bin_index + 1) * per_bin)
        for prn in sorted(prns):
            selected = valid[rows] & (identifiers[rows] == prn)
            values = pseudorange[rows][selected]
            values = values[np.isfinite(values) & (values > 0.0)]
            if len(values) >= minimum_observations:
                medians[bin_index, prn] = float(np.median(values))
                counts[bin_index, prn] = int(len(values))
    return medians, counts


def fit_linear_prn_baselines(
    residuals: dict[tuple[int, int], float], prns: Iterable[int], start_s: int, end_s: int
) -> dict[int, np.ndarray]:
    coefficients: dict[int, np.ndarray] = {}
    for prn in sorted(prns):
        bins = [index for index in range(start_s, end_s) if (index, prn) in residuals]
        if len(bins) < 2:
            raise ValueError(f"insufficient clean baseline for PRN {prn}")
        x = np.asarray([index + 0.5 for index in bins], dtype=np.float64)
        y = np.asarray([residuals[index, prn] for index in bins], dtype=np.float64)
        coefficients[prn] = np.polyfit(x, y, 1)
    return coefficients


def fit_pseudorange_geometry(
    residuals: dict[tuple[int, int], float],
    los: dict[tuple[int, int], np.ndarray],
    coefficients: dict[int, np.ndarray],
    *,
    start_s: int,
    end_s: int,
    minimum_prns: int,
) -> dict[int, Any]:
    fitted: dict[int, Any] = {}
    for bin_index in range(start_s, end_s):
        prns = sorted(prn for prn in coefficients if (bin_index, prn) in residuals)
        if len(prns) < minimum_prns:
            continue
        observations = np.asarray(
            [residuals[bin_index, prn] - np.polyval(coefficients[prn], bin_index + 0.5) for prn in prns],
            dtype=np.float64,
        )
        fitted[bin_index] = fit_clock_centered_geometry(
            np.asarray([los[bin_index, prn] for prn in prns], dtype=np.float64), observations
        )
    return fitted


def load_delay_rows(path: Path) -> dict[int, list[dict[str, str]]]:
    by_bin: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_bin.setdefault(int(row["bin_index"]), []).append(row)
    return by_bin


def fit_local_tap_geometry(
    delay_rows: dict[int, list[dict[str, str]]],
    los: dict[tuple[int, int], np.ndarray],
    *,
    metres_per_chip: float,
    minimum_prns: int,
) -> dict[int, Any]:
    fitted: dict[int, Any] = {}
    for bin_index, rows in sorted(delay_rows.items()):
        usable = [row for row in rows if (bin_index, int(row["prn"])) in los]
        if len(usable) < minimum_prns:
            continue
        fitted[bin_index] = fit_clock_centered_geometry(
            np.asarray([los[bin_index, int(row["prn"])] for row in usable], dtype=np.float64),
            np.asarray([float(row["estimated_delay_chips"]) * metres_per_chip for row in usable], dtype=np.float64),
        )
    return fitted


def comparison_row(
    bin_index: int,
    nmea_displacement: np.ndarray,
    pseudorange_fit: Any,
    tap_fit: Any,
    pseudorange_reference: np.ndarray,
    tap_reference: np.ndarray,
) -> dict[str, Any]:
    nmea = np.asarray(nmea_displacement, dtype=np.float64)
    pseudo = np.asarray(pseudorange_fit.theta[:3], dtype=np.float64) - pseudorange_reference
    tap = np.asarray(tap_fit.theta[:3], dtype=np.float64) - tap_reference

    def measures(vector: np.ndarray) -> tuple[float, float, float, float]:
        norm = float(np.linalg.norm(vector))
        target_norm = float(np.linalg.norm(nmea))
        cosine = float(np.dot(vector, nmea) / (norm * target_norm)) if norm > 0.0 and target_norm > 0.0 else float("nan")
        return norm, cosine, abs(norm - target_norm), float(np.linalg.norm(vector - nmea))

    pseudo_norm, pseudo_cosine, pseudo_magnitude_error, pseudo_vector_error = measures(pseudo)
    tap_norm, tap_cosine, tap_magnitude_error, tap_vector_error = measures(tap)
    return {
        "bin_index": bin_index,
        "region": "settled_clean" if 43 <= bin_index < 120 else "stable_post" if 160 <= bin_index < 230 else "excluded",
        "nmea_dx_ecef_m": float(nmea[0]),
        "nmea_dy_ecef_m": float(nmea[1]),
        "nmea_dz_ecef_m": float(nmea[2]),
        "nmea_displacement_norm_m": float(np.linalg.norm(nmea)),
        "pseudorange_dx_ecef_m": float(pseudo[0]),
        "pseudorange_dy_ecef_m": float(pseudo[1]),
        "pseudorange_dz_ecef_m": float(pseudo[2]),
        "pseudorange_displacement_norm_m": pseudo_norm,
        "pseudorange_nmea_direction_cosine": pseudo_cosine,
        "pseudorange_nmea_magnitude_error_m": pseudo_magnitude_error,
        "pseudorange_nmea_vector_error_m": pseudo_vector_error,
        "pseudorange_clock_centered_residual": float(pseudorange_fit.clock_centered_normalized_residual),
        "tap_dx_ecef_m": float(tap[0]),
        "tap_dy_ecef_m": float(tap[1]),
        "tap_dz_ecef_m": float(tap[2]),
        "tap_displacement_norm_m": tap_norm,
        "tap_nmea_direction_cosine": tap_cosine,
        "tap_nmea_magnitude_error_m": tap_magnitude_error,
        "tap_nmea_vector_error_m": tap_vector_error,
        "tap_clock_centered_residual": float(tap_fit.clock_centered_normalized_residual),
    }


def summarize_region(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty comparison region")
    fields = (
        "nmea_displacement_norm_m",
        "pseudorange_displacement_norm_m",
        "pseudorange_nmea_direction_cosine",
        "pseudorange_nmea_magnitude_error_m",
        "pseudorange_nmea_vector_error_m",
        "pseudorange_clock_centered_residual",
        "tap_displacement_norm_m",
        "tap_nmea_direction_cosine",
        "tap_nmea_magnitude_error_m",
        "tap_nmea_vector_error_m",
        "tap_clock_centered_residual",
    )
    return {
        "matched_bin_count": len(rows),
        **{f"median_{field}": float(np.nanmedian([row[field] for row in rows])) for field in fields},
        "fraction_pseudorange_nmea_direction_cosine_at_least_0_8": float(
            np.mean([row["pseudorange_nmea_direction_cosine"] >= 0.8 for row in rows])
        ),
        "fraction_tap_nmea_direction_cosine_at_least_0_8": float(
            np.mean([row["tap_nmea_direction_cosine"] >= 0.8 for row in rows])
        ),
    }


def sensitivity_rows(
    starts: Iterable[int],
    residuals: dict[tuple[int, int], float],
    los: dict[tuple[int, int], np.ndarray],
    prns: set[int],
    nmea_displacements: dict[int, np.ndarray],
    *,
    end_clean_s: int,
    post_start_s: int,
    post_end_s: int,
    minimum_prns: int,
) -> list[dict[str, Any]]:
    result = []
    for start_s in starts:
        coefficients = fit_linear_prn_baselines(residuals, prns, start_s, end_clean_s)
        fitted = fit_pseudorange_geometry(
            residuals, los, coefficients, start_s=start_s, end_s=post_end_s, minimum_prns=minimum_prns
        )
        reference = median_reference(fitted[index].theta[:3] for index in range(start_s, end_clean_s) if index in fitted)
        rows = []
        for index in range(post_start_s, post_end_s):
            if index not in fitted or index not in nmea_displacements:
                continue
            vector = np.asarray(fitted[index].theta[:3]) - reference
            nmea = nmea_displacements[index]
            vector_norm, nmea_norm = np.linalg.norm(vector), np.linalg.norm(nmea)
            rows.append({
                "cosine": float(np.dot(vector, nmea) / (vector_norm * nmea_norm)),
                "vector_error": float(np.linalg.norm(vector - nmea)),
            })
        result.append({
            "clean_baseline_start_s": start_s,
            "clean_baseline_end_s": end_clean_s,
            "stable_post_matched_bin_count": len(rows),
            "median_pseudorange_nmea_direction_cosine": float(np.median([row["cosine"] for row in rows])),
            "tenth_percentile_pseudorange_nmea_direction_cosine": float(np.quantile([row["cosine"] for row in rows], 0.1)),
            "median_pseudorange_nmea_vector_error_m": float(np.median([row["vector_error"] for row in rows])),
        })
    return result


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
    paths = {name: verify(record, name) for name, record in config["inputs"].items()}
    detection = json.loads(paths["frozen_detection_summary"].read_text(encoding="utf-8"))
    if detection["primary"]["status"] != "NOT_SUPPORTED":
        raise ValueError("frozen detector result is not the expected negative result")

    analysis = config["analysis"]
    start_s, end_s = map(int, analysis["analysis_seconds"])
    clean_start_s, clean_end_s = map(int, analysis["settled_clean_baseline_seconds"])
    post_start_s, post_end_s = map(int, analysis["stable_post_seconds"])
    prns = set(map(int, analysis["healthy_prns"]))
    tow0_s = float(config["dataset"]["recording_start_tow_s"])
    nmea_positions = parse_nmea_ecef_bins(paths["nmea"], tow0_s, start_s, end_s)
    receiver_reference = median_reference(
        nmea_positions[index] for index in range(clean_start_s, clean_end_s) if index in nmea_positions
    )
    nmea_displacements = {index: position - receiver_reference for index, position in nmea_positions.items()}

    ephemerides = parse_gnss_sdr_gps_ephemeris_xml(paths["ephemeris"])
    healthy, health = ephemeris_health_selection(ephemerides, tracked_prns=prns, min_prns=analysis["minimum_prns"])
    if set(healthy) != prns:
        raise ValueError("healthy ephemeris roster drifted")
    pseudorange, counts = load_pseudorange_bins(
        paths["observables"],
        prns=prns,
        start_s=start_s,
        end_s=end_s,
        sample_period_s=float(analysis["sample_period_s"]),
        minimum_observations=int(analysis["minimum_observations_per_prn_bin"]),
    )
    residuals: dict[tuple[int, int], float] = {}
    los: dict[tuple[int, int], np.ndarray] = {}
    for (bin_index, prn), value in pseudorange.items():
        observation = satellite_observation(
            receiver_reference, healthy[prn], (tow0_s + bin_index + 0.5) % GPS_WEEK_SECONDS
        )
        residuals[bin_index, prn] = value - observation.range_m
        los[bin_index, prn] = np.asarray(observation.los_ecef, dtype=np.float64)

    coefficients = fit_linear_prn_baselines(residuals, prns, clean_start_s, clean_end_s)
    pseudorange_fits = fit_pseudorange_geometry(
        residuals, los, coefficients, start_s=start_s, end_s=end_s, minimum_prns=int(analysis["minimum_prns"])
    )
    metres_per_chip = float(analysis["speed_of_light_mps"]) / float(analysis["chip_rate_hz"])
    tap_fits = fit_local_tap_geometry(
        load_delay_rows(paths["frozen_delay_estimates"]),
        los,
        metres_per_chip=metres_per_chip,
        minimum_prns=int(analysis["minimum_prns"]),
    )
    pseudo_reference = median_reference(
        pseudorange_fits[index].theta[:3] for index in range(clean_start_s, clean_end_s) if index in pseudorange_fits
    )
    tap_reference = median_reference(
        tap_fits[index].theta[:3] for index in range(clean_start_s, clean_end_s) if index in tap_fits
    )

    comparisons = []
    for index in range(start_s, end_s):
        if index not in nmea_displacements or index not in pseudorange_fits or index not in tap_fits:
            continue
        comparisons.append(comparison_row(
            index,
            nmea_displacements[index],
            pseudorange_fits[index],
            tap_fits[index],
            pseudo_reference,
            tap_reference,
        ))
    regions = {
        name: summarize_region([row for row in comparisons if row["region"] == name])
        for name in ("settled_clean", "stable_post")
    }
    sensitivity = sensitivity_rows(
        analysis["baseline_start_sensitivity_seconds"],
        residuals,
        los,
        prns,
        nmea_displacements,
        end_clean_s=clean_end_s,
        post_start_s=post_start_s,
        post_end_s=post_end_s,
        minimum_prns=int(analysis["minimum_prns"]),
    )

    signature = config["exploratory_mechanistic_signature"]
    clean, post = regions["settled_clean"], regions["stable_post"]
    error_ratio = post["median_tap_nmea_vector_error_m"] / post["median_pseudorange_nmea_vector_error_m"]
    frozen_post_alarms = detection["primary"]["regions"]["stable_post"]["persistent_alarm_count"]
    gates = {
        "minimum_matched_clean_bins": clean["matched_bin_count"] >= signature["minimum_matched_clean_bins"],
        "minimum_matched_stable_post_bins": post["matched_bin_count"] >= signature["minimum_matched_stable_post_bins"],
        "minimum_stable_post_nmea_displacement_m": post["median_nmea_displacement_norm_m"] >= signature["minimum_stable_post_nmea_displacement_m"],
        "minimum_pseudorange_nmea_median_direction_cosine": post["median_pseudorange_nmea_direction_cosine"] >= signature["minimum_pseudorange_nmea_median_direction_cosine"],
        "maximum_pseudorange_nmea_median_vector_error_m": post["median_pseudorange_nmea_vector_error_m"] <= signature["maximum_pseudorange_nmea_median_vector_error_m"],
        "maximum_local_tap_nmea_median_direction_cosine": post["median_tap_nmea_direction_cosine"] <= signature["maximum_local_tap_nmea_median_direction_cosine"],
        "minimum_local_to_pseudorange_median_vector_error_ratio": error_ratio >= signature["minimum_local_to_pseudorange_median_vector_error_ratio"],
        "frozen_detector_zero_persistent_post_alarms": frozen_post_alarms == 0,
    }
    supported = all(gates.values())
    output = resolve(config["output_root"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_csv(output / "vector_comparison.csv", comparisons)
    write_csv(output / "baseline_sensitivity.csv", sensitivity)
    summary = {
        "schema": "gnss-doppler-lab.fgi-spoofrepo-tgd-observability-audit-result",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "post-hoc exploratory mechanism audit; not independent detector validation",
        "status": "OBSERVABILITY_LOSS_SUPPORTED" if supported else "OBSERVABILITY_LOSS_NOT_SUPPORTED",
        "all_exploratory_mechanistic_signatures_passed": supported,
        "gates": gates,
        "regions": regions,
        "stable_post_local_to_pseudorange_median_vector_error_ratio": float(error_ratio),
        "frozen_detector_stable_post_persistent_alarm_count": int(frozen_post_alarms),
        "pseudorange_clean_reference_ecef_m": pseudo_reference.tolist(),
        "tap_clean_reference_ecef_m": tap_reference.tolist(),
        "trusted_static_receiver_ecef_m": receiver_reference.tolist(),
        "metres_per_ca_chip": metres_per_chip,
        "pseudorange_valid_observation_count": int(sum(counts.values())),
        "ephemeris_health": health,
        "baseline_sensitivity": sensitivity,
        "config": {"path": str(config_path.resolve()), "sha256": sha256(config_path)},
        "inputs": config["inputs"],
        "limitations": [
            "NMEA PVT and pseudorange are outputs of the same receiver and are not independent evidence.",
            "The pseudorange orbit model is differential and omits precise transmit-time, Sagnac, clock, atmosphere, and relativistic corrections.",
            "The settled-clean PRN trend is extrapolated only for this short post-hoc mechanism audit.",
            "This result cannot revise the frozen NOT_SUPPORTED detector decision.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(output / "summary.json"),
        "status": summary["status"],
        "post_nmea_displacement_m": post["median_nmea_displacement_norm_m"],
        "post_pseudorange_nmea_cosine": post["median_pseudorange_nmea_direction_cosine"],
        "post_pseudorange_vector_error_m": post["median_pseudorange_nmea_vector_error_m"],
        "post_tap_nmea_cosine": post["median_tap_nmea_direction_cosine"],
        "post_tap_vector_error_m": post["median_tap_nmea_vector_error_m"],
        "error_ratio": error_ratio,
    }, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
