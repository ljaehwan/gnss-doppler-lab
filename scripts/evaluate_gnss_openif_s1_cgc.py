#!/usr/bin/env python3
"""Evaluate frozen CGC spoof-alarm specificity on real GNSS-OpenIF S1."""
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

import run_cgc_real_detection as frozen_real  # noqa: E402
import run_cgc_rf_challenge_pilot as pilot  # noqa: E402
from gnss_doppler_lab.clock_centered_geometry import fit_clock_centered_geometry  # noqa: E402
from gnss_doppler_lab.correlator_geometry import complex_profile_features  # noqa: E402
from gnss_doppler_lab.gcmr_geometry import (  # noqa: E402
    ephemeris_health_selection,
    parse_gnss_sdr_gps_ephemeris_xml,
    satellite_observation,
    validate_ephemeris_time_alignment,
)
from gnss_doppler_lab.rinex_nav import parse_rinex2_gps_nav_gz  # noqa: E402
from gnss_doppler_lab.tracking_peaks import (  # noqa: E402
    available_tracking_prns,
    load_receiver_tracking_peak_series_segments,
)


DEFAULT_CONFIG = ROOT / "configs/experiments/gnss_openif_s1_real_multipath_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def verify(path: str | Path, expected: str, label: str) -> Path:
    target = resolve(path)
    if not target.is_file():
        raise FileNotFoundError(f"{label} missing: {target}")
    observed = sha256(target)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return target


def load_ground_truth(path: Path) -> dict[str, np.ndarray]:
    values = np.loadtxt(path, skiprows=2)
    if values.ndim != 2 or values.shape[1] < 9 or len(values) < 2:
        raise ValueError("unexpected GNSS-OpenIF S1 ground-truth format")
    tow = values[:, 2].astype(np.float64)
    ecef = values[:, 6:9].astype(np.float64)
    if not np.all(np.diff(tow) > 0) or not np.isfinite(ecef).all():
        raise ValueError("ground-truth time/ECEF values are invalid")
    return {"tow_s": tow, "ecef_m": ecef}


def interpolate_ecef(truth: dict[str, np.ndarray], tow_s: float) -> np.ndarray:
    tow = truth["tow_s"]
    if tow_s < tow[0] or tow_s > tow[-1]:
        raise ValueError(f"TOW {tow_s} is outside ground-truth support")
    return np.asarray([
        np.interp(tow_s, tow, truth["ecef_m"][:, axis]) for axis in range(3)
    ], dtype=np.float64)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def consolidate_profile_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicate (bin, PRN) summaries across channels and restart windows."""
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["bin_index"]), int(row["prn"]))
        grouped.setdefault(key, []).append(row)
    combined: list[dict[str, Any]] = []
    for (bin_index, prn), group in sorted(grouped.items()):
        combined.append({
            "bin_index": bin_index,
            "bin_start_s": float(group[0]["bin_start_s"]),
            "prn": prn,
            "prn_name": str(group[0]["prn_name"]),
            "epoch_count": max(int(row["epoch_count"]) for row in group),
            "estimated_delay_chips": float(np.median([row["estimated_delay_chips"] for row in group])),
            "median_template_distance": float(np.median([row["median_template_distance"] for row in group])),
            "median_early_late_asymmetry": float(np.median([row["median_early_late_asymmetry"] for row in group])),
            "median_cn0_db_hz": float(np.median([row["median_cn0_db_hz"] for row in group])),
        })
    return combined


def profile_rows(
    run_dir: Path, estimator: Any, *, bin_seconds: float,
    minimum_epochs: int, start_s: float, end_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prn_name in available_tracking_prns(run_dir):
        prn = int(prn_name[1:])
        segments = load_receiver_tracking_peak_series_segments(
            run_dir, prn_name, tap_count=9, require_complex_taps=True
        )
        for segment in segments:
            times = segment.time_s
            mask = (times >= start_s) & (times < end_s)
            if not np.any(mask):
                continue
            times = times[mask]
            taps = segment.complex_taps[mask]
            features = complex_profile_features(taps, prompt_index=4)
            estimates, distances, _ = estimator.estimate(features)
            asymmetry = frozen_real.early_late_asymmetry(
                np.stack((taps.real, taps.imag), axis=-1)
            )
            cn0 = segment.cn0_db_hz[mask]
            bins = np.floor(times / bin_seconds).astype(np.int64)
            for bin_index in np.unique(bins):
                select = bins == bin_index
                count = int(np.count_nonzero(select))
                if count < minimum_epochs:
                    continue
                rows.append({
                    "bin_index": int(bin_index),
                    "bin_start_s": float(bin_index * bin_seconds),
                    "prn": prn,
                    "prn_name": prn_name,
                    "epoch_count": count,
                    "estimated_delay_chips": float(np.median(estimates[select])),
                    "median_template_distance": float(np.median(distances[select])),
                    "median_early_late_asymmetry": float(np.median(asymmetry[select])),
                    "median_cn0_db_hz": float(np.median(cn0[select])),
                })
    return consolidate_profile_rows(rows)


def geometry_rows(
    delay_rows: list[dict[str, Any]], ephemerides: dict[int, Any],
    truth: dict[str, np.ndarray], config: dict[str, Any],
) -> list[dict[str, Any]]:
    analysis = config["analysis"]
    dataset = config["dataset"]
    named_mp = set(dataset["official_multipath_prns"]["whole_test"])
    named_mp.update(dataset["official_multipath_prns"]["specific_epochs_without_machine_readable_epoch_labels"])
    by_bin: dict[int, list[dict[str, Any]]] = {}
    for row in delay_rows:
        if int(row["prn"]) in ephemerides:
            by_bin.setdefault(int(row["bin_index"]), []).append(row)
    result: list[dict[str, Any]] = []
    for bin_index, entries in sorted(by_bin.items()):
        if len(entries) < int(analysis["minimum_prns"]):
            continue
        tow = float(dataset["recording_start_tow_s"]) + (bin_index + 0.5) * float(analysis["bin_seconds"])
        receiver_ecef = interpolate_ecef(truth, tow)
        los = np.asarray([
            satellite_observation(receiver_ecef, ephemerides[int(row["prn"])], tow).los_ecef
            for row in entries
        ], dtype=np.float64)
        delays = np.asarray([float(row["estimated_delay_chips"]) for row in entries], dtype=np.float64)
        fit = fit_clock_centered_geometry(los, delays)
        prns = [int(row["prn"]) for row in entries]
        prn22_index = prns.index(22) if 22 in prns else None
        loo_residual = None
        prn22_prediction_error = None
        if prn22_index is not None and len(entries) - 1 >= int(analysis["minimum_prns"]):
            keep = np.arange(len(entries)) != prn22_index
            loo_fit = fit_clock_centered_geometry(los[keep], delays[keep])
            design22 = np.r_[-los[prn22_index], 1.0]
            loo_residual = float(loo_fit.clock_centered_normalized_residual)
            prn22_prediction_error = float(abs(delays[prn22_index] - design22 @ loo_fit.theta))
        asymmetry = np.asarray([
            float(row["median_early_late_asymmetry"]) for row in entries
        ], dtype=np.float64)
        result.append({
            "bin_index": bin_index,
            "bin_start_s": float(bin_index * float(analysis["bin_seconds"])),
            "bin_end_s": float((bin_index + 1) * float(analysis["bin_seconds"])),
            "gps_tow_s": tow,
            "prn_count": len(entries),
            "prns": " ".join(f"G{prn:02d}" for prn in prns),
            "official_named_multipath_prn_count": len(named_mp.intersection(prns)),
            "clock_centered_geometry_residual": float(fit.clock_centered_normalized_residual),
            "spoof_score": float(-fit.clock_centered_normalized_residual),
            "directional_geometry_coherence": float(fit.directional_coherence),
            "fit_rank": int(fit.rank),
            "q75_prn_early_late_asymmetry": float(np.quantile(asymmetry, 0.75)),
            "prn22_present": prn22_index is not None,
            "without_prn22_clock_centered_residual": loo_residual,
            "prn22_leave_one_out_prediction_error_chips": prn22_prediction_error,
        })
    return result


def partial_f_p_value(residual: float, prn_count: int) -> float:
    """Normalize nested geometry-fit improvement for finite satellite support."""
    value = float(residual)
    count = int(prn_count)
    if not np.isfinite(value) or value < 0.0 or value > 1.0 + 1e-9:
        raise ValueError("geometry residual must be in [0,1]")
    if count <= 4:
        raise ValueError("partial-F requires more than four PRNs")
    value = min(max(value, np.finfo(float).tiny), 1.0)
    statistic = (1.0 - value) * (count - 4) / (3.0 * value)
    return float(f_distribution.sf(statistic, 3, count - 4))


def apply_frozen_alarm(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    detector = config["frozen_detector"]
    support = config["support_normalization"]
    residual_threshold = float(detector["residual_alarm_threshold"])
    p_threshold = float(support["partial_f_p_alarm_threshold"])
    distortion_threshold = float(detector["multipath_enrichment_threshold"])
    for row in rows:
        residual = float(row["clock_centered_geometry_residual"])
        row["residual_alarm_threshold"] = residual_threshold
        row["partial_f_p_alarm_threshold"] = p_threshold
        row["multipath_enrichment_threshold"] = distortion_threshold
        row["legacy_raw_spoof_alarm"] = residual <= residual_threshold
        row["partial_f_p_value"] = partial_f_p_value(
            residual, int(row["prn_count"])
        )
        row["raw_spoof_alarm"] = row["partial_f_p_value"] <= p_threshold
        row["multipath_enriched"] = bool(
            float(row["q75_prn_early_late_asymmetry"]) >= distortion_threshold
        )
    bins = np.asarray([row["bin_index"] for row in rows], dtype=np.int64)
    persistent = frozen_real.persistent_alarm(
        np.asarray([row["raw_spoof_alarm"] for row in rows], dtype=bool),
        bins,
        window=int(detector["persistence_window_bins"]),
        required=int(detector["persistence_required_bins"]),
    )
    legacy_persistent = frozen_real.persistent_alarm(
        np.asarray([row["legacy_raw_spoof_alarm"] for row in rows], dtype=bool),
        bins,
        window=int(detector["persistence_window_bins"]),
        required=int(detector["persistence_required_bins"]),
    )
    for row, alarm, legacy_alarm in zip(rows, persistent, legacy_persistent):
        row["persistent_spoof_alarm"] = bool(alarm)
        row["legacy_persistent_spoof_alarm"] = bool(legacy_alarm)
        row["detector_classification"] = (
            "spoof_alarm" if alarm else
            "multipath_enriched_negative" if row["multipath_enriched"] else
            "no_spoof_alarm"
        )

def summarize_prns(rows: list[dict[str, Any]], named_mp: set[int]) -> list[dict[str, Any]]:
    result = []
    for prn in sorted({int(row["prn"]) for row in rows}):
        group = [row for row in rows if int(row["prn"]) == prn]
        result.append({
            "prn": prn,
            "prn_name": f"G{prn:02d}",
            "official_named_multipath": prn in named_mp,
            "bin_count": len(group),
            "median_estimated_delay_chips": float(np.median([row["estimated_delay_chips"] for row in group])),
            "median_template_distance": float(np.median([row["median_template_distance"] for row in group])),
            "median_early_late_asymmetry": float(np.median([row["median_early_late_asymmetry"] for row in group])),
            "median_cn0_db_hz": float(np.median([row["median_cn0_db_hz"] for row in group])),
        })
    return result


def evaluate(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "gnss-doppler-lab.gnss-openif-s1-real-multipath-config":
        raise ValueError("unsupported config schema")
    dataset = config["dataset"]
    if dataset["iq_sha256"] == "PENDING_DOWNLOAD":
        raise ValueError("IQ SHA-256 must be frozen before evaluation")
    iq = verify(dataset["iq_path"], dataset["iq_sha256"], "S1 IQ")
    if iq.stat().st_size != int(dataset["iq_bytes"]):
        raise ValueError("S1 IQ byte count mismatch")
    truth_path = verify(dataset["ground_truth_path"], dataset["ground_truth_sha256"], "S1 ground truth")
    verify(dataset["readme_path"], dataset["readme_sha256"], "GNSS-OpenIF README")
    broadcast_nav_path = verify(dataset["broadcast_navigation_path"], dataset["broadcast_navigation_sha256"], "NOAA broadcast NAV")
    detector = config["frozen_detector"]
    verify(detector["threshold_source"], detector["threshold_source_sha256"], "threshold source")
    template_path = verify(detector["template_config"], detector["template_config_sha256"], "template config")
    verify(detector["clock_centered_module"], detector["clock_centered_module_sha256"], "clock-centered module")
    verify(detector["correlator_geometry_module"], detector["correlator_geometry_module_sha256"], "correlator module")
    verify(detector["geometry_module"], detector["geometry_module_sha256"], "geometry module")
    support = config["support_normalization"]
    support_audit_path = verify(
        support["audit_summary"], support["audit_summary_sha256"],
        "partial-F seven-PRN audit",
    )
    support_audit = json.loads(support_audit_path.read_text(encoding="utf-8"))
    if support_audit["status"] != "PARTIAL_F_SEVEN_PRN_SPECIFICITY_SUPPORTED":
        raise ValueError("partial-F seven-PRN audit did not support specificity")
    if float(support_audit["partial_f"]["p_value_alarm_threshold"]) != float(
        support["partial_f_p_alarm_threshold"]
    ):
        raise ValueError("partial-F threshold drifted from the sealed audit")
    if support.get("s1_outcome_accessed_before_freeze") is not False:
        raise ValueError("S1 outcome must remain unseen before support-score freeze")


    receiver = config["receiver"]
    run_dirs = [resolve(path) for path in receiver["run_dirs"]]
    if not run_dirs:
        raise ValueError("at least one receiver run is required")
    manifests: list[dict[str, Any]] = []
    receiver_ephemerides: dict[int, Any] = {}
    tracked_names: list[str] = []
    hash_bound_source_seen = False
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["receiver"]["executable_sha256"] != receiver["executable_sha256"]:
            raise ValueError("receiver executable drifted")
        if manifest["tracking"]["tap_count"] != 9:
            raise ValueError("receiver is not complex nine-tap")
        source_sha = manifest["source"].get("iq_sha256")
        if source_sha not in (None, dataset["iq_sha256"]):
            raise ValueError("receiver source IQ SHA-256 mismatch")
        hash_bound_source_seen |= source_sha == dataset["iq_sha256"]
        if Path(manifest["source"]["iq"]).resolve() != iq:
            raise ValueError("receiver source IQ path drifted")
        if int(manifest["source"]["iq_bytes"]) != int(dataset["iq_bytes"]):
            raise ValueError("receiver source IQ byte count drifted")
        if int(manifest["source"]["intermediate_frequency_hz"]) != int(receiver["observed_complex_spectrum_center_hz"]):
            raise ValueError("receiver IF translation drifted")
        if int(manifest["acquisition"]["channel_count"]) != int(receiver["channel_count"]):
            raise ValueError("receiver channel count drifted")
        receiver_config_path = run_dir / manifest["receiver"]["config"]
        if sha256(receiver_config_path) != manifest["receiver"]["config_sha256"]:
            raise ValueError("receiver config SHA-256 mismatch")
        receiver_text = receiver_config_path.read_text(encoding="utf-8").splitlines()
        expected_lines = [
            f"InputFilter.IF={receiver['observed_complex_spectrum_center_hz']}",
            f"Channels_1C.count={receiver['channel_count']}",
            f"Acquisition_1C.pfa={receiver['acquisition_pfa']}",
            f"Acquisition_1C.max_dwells={receiver['acquisition_max_dwells']}",
            f"Tracking_1C.tap_count={receiver['tap_count']}",
            f"Tracking_1C.tap_spacing_chips={receiver['tap_spacing_chips']}",
        ]
        if "start_offset_s" in manifest["source"]:
            expected_lines.append(
                f"SignalSource.seconds_to_skip={manifest['source']['start_offset_s']}"
            )
        if any(line not in receiver_text for line in expected_lines):
            raise ValueError("receiver config does not match the sealed experiment")
        receiver_ephemeris_path = run_dir / "gps_ephemeris.xml"
        if receiver_ephemeris_path.is_file():
            receiver_ephemerides.update(
                parse_gnss_sdr_gps_ephemeris_xml(receiver_ephemeris_path)
            )
        for prn_name in available_tracking_prns(run_dir):
            if prn_name not in tracked_names:
                tracked_names.append(prn_name)
        manifests.append(manifest)
    if not hash_bound_source_seen:
        raise ValueError("no receiver run is hash-bound to the sealed IQ")
    observed_offsets = sorted(float(item["source"].get("start_offset_s", 0.0)) for item in manifests)
    if observed_offsets != sorted(float(value) for value in receiver["expected_start_offsets_s"]):
        raise ValueError("receiver restart-window offsets drifted")
    ephemerides = parse_rinex2_gps_nav_gz(
        broadcast_nav_path, full_gps_week=int(dataset["gps_week"]),
        target_tow_s=float(dataset["recording_start_tow_s"]),
        maximum_toe_age_s=float(config["analysis"]["maximum_ephemeris_toe_age_s"]),
    )
    tracked = {int(name[1:]) for name in tracked_names}
    healthy, health = ephemeris_health_selection(
        ephemerides, tracked_prns=tracked, min_prns=int(config["analysis"]["minimum_prns"])
    )
    alignment = validate_ephemeris_time_alignment(
        healthy,
        full_gps_week=int(dataset["gps_week"]),
        recording_start_tow_s=float(dataset["recording_start_tow_s"]),
        max_toe_age_s=float(config["analysis"]["maximum_ephemeris_toe_age_s"]),
    )
    truth = load_ground_truth(truth_path)
    template = json.loads(template_path.read_text(encoding="utf-8"))
    estimator = pilot._estimator(template)
    analysis = config["analysis"]
    delays = consolidate_profile_rows([
        row
        for run_dir in run_dirs
        for row in profile_rows(
            run_dir, estimator,
            bin_seconds=float(analysis["bin_seconds"]),
            minimum_epochs=int(analysis["minimum_epochs_per_prn_bin"]),
            start_s=float(analysis["analysis_start_s"]),
            end_s=float(analysis["analysis_end_s"]),
        )
    ])
    geometries = geometry_rows(delays, healthy, truth, config)
    if not geometries:
        raise ValueError("no geometry bins satisfy the frozen support gate")
    apply_frozen_alarm(geometries, config)
    named_mp = set(dataset["official_multipath_prns"]["whole_test"])
    named_mp.update(dataset["official_multipath_prns"]["specific_epochs_without_machine_readable_epoch_labels"])
    prn_rows = summarize_prns(delays, named_mp)
    persistent = np.asarray([row["persistent_spoof_alarm"] for row in geometries], dtype=bool)
    legacy_raw = np.asarray(
        [row["legacy_raw_spoof_alarm"] for row in geometries], dtype=bool
    )
    legacy_persistent = np.asarray(
        [row["legacy_persistent_spoof_alarm"] for row in geometries], dtype=bool
    )
    enriched = np.asarray([row["multipath_enriched"] for row in geometries], dtype=bool)
    prn22_rows = [row for row in geometries if row["prn22_present"]]
    enriched_rate = float((persistent & enriched).sum() / enriched.sum()) if enriched.any() else None
    legacy_enriched_rate = (
        float((legacy_persistent & enriched).sum() / enriched.sum())
        if enriched.any() else None
    )
    gates_cfg = config["specificity_gates"]
    gates = {
        "minimum_evaluated_bins": len(geometries) >= int(gates_cfg["minimum_evaluated_bins"]),
        "require_prn22_tracking": (22 in tracked) if gates_cfg["require_prn22_tracking"] else True,
        "maximum_persistent_spoof_alarm_rate": float(persistent.mean()) <= float(gates_cfg["maximum_persistent_spoof_alarm_rate"]),
        "maximum_multipath_enriched_persistent_spoof_alarm_rate": (
            enriched_rate is not None and enriched_rate <= float(gates_cfg["maximum_multipath_enriched_persistent_spoof_alarm_rate"])
        ),
    }
    receiver_runs = [
        {
            "run_directory": str(run_dir),
            "start_offset_s": float(manifest["source"].get("start_offset_s", 0.0)),
            "requested_duration_s": float(manifest["source"]["requested_duration_s"]),
            "receiver_config_sha256": manifest["receiver"]["config_sha256"],
            "tracked_prns": manifest["acquisition"]["tracked_prns"],
            "valid_epoch_count": int(manifest["tracking"]["valid_epoch_count"]),
        }
        for run_dir, manifest in zip(run_dirs, manifests)
    ]
    summary: dict[str, Any] = {
        "schema": "gnss-doppler-lab.gnss-openif-s1-real-multipath-result",
        "schema_version": 2,
        "status": "REAL_MULTIPATH_SPECIFICITY_SUPPORTED" if all(gates.values()) else "REAL_MULTIPATH_SPECIFICITY_NOT_SUPPORTED",
        "all_specificity_gates_passed": all(gates.values()),
        "gates": gates,
        "claim_boundary": "offline specificity on external real field multipath using a hash-bound broadcast-orbit oracle; spoof sensitivity is assessed only in the separately sealed synthetic DS7 audit and its delay tradeoff remains",
        "detector_score": "support-normalized nested-model partial-F p-value",
        "s1_threshold_tuning": False,
        "support_audit_sha256": support["audit_summary_sha256"],
        "evaluated_bin_count": len(geometries),
        "minimum_prn_count": min(int(row["prn_count"]) for row in geometries),
        "maximum_prn_count": max(int(row["prn_count"]) for row in geometries),
        "raw_spoof_alarm_count": sum(bool(row["raw_spoof_alarm"]) for row in geometries),
        "raw_spoof_alarm_rate": float(np.mean([row["raw_spoof_alarm"] for row in geometries])),
        "persistent_spoof_alarm_count": int(persistent.sum()),
        "persistent_spoof_alarm_rate": float(persistent.mean()),
        "multipath_enriched_bin_count": int(enriched.sum()),
        "multipath_enriched_persistent_spoof_alarm_rate": enriched_rate,
        "median_partial_f_p_value": float(np.median([
            row["partial_f_p_value"] for row in geometries
        ])),
        "median_clock_centered_geometry_residual": float(np.median([row["clock_centered_geometry_residual"] for row in geometries])),
        "legacy_unadjusted_residual": {
            "raw_spoof_alarm_count": int(legacy_raw.sum()),
            "raw_spoof_alarm_rate": float(legacy_raw.mean()),
            "persistent_spoof_alarm_count": int(legacy_persistent.sum()),
            "persistent_spoof_alarm_rate": float(legacy_persistent.mean()),
            "multipath_enriched_persistent_spoof_alarm_rate": legacy_enriched_rate,
        },
        "prn22_tracking_bin_count": sum(int(row["prn"]) == 22 for row in delays),
        "prn22_geometry_bin_count": len(prn22_rows),
        "prn22_median_leave_one_out_prediction_error_chips": (
            float(np.median([row["prn22_leave_one_out_prediction_error_chips"] for row in prn22_rows if row["prn22_leave_one_out_prediction_error_chips"] is not None]))
            if any(row["prn22_leave_one_out_prediction_error_chips"] is not None for row in prn22_rows) else None
        ),
        "tracked_prns": tracked_names,
        "receiver_runs": receiver_runs,
        "healthy_tracked_prns": health["healthy_tracked_prns"],
        "ephemeris_alignment": alignment,
        "ephemeris_provenance": {
            "geometry_source": "NOAA CORS validated daily GPS broadcast navigation",
            "broadcast_navigation_prns": sorted(ephemerides),
            "receiver_decoded_ephemeris_prns": sorted(receiver_ephemerides),
        },
        "frozen_thresholds": {
            "partial_f_p_alarm_threshold": support["partial_f_p_alarm_threshold"],
            "legacy_unadjusted_residual_alarm_threshold": detector["residual_alarm_threshold"],
            "multipath_enrichment_threshold": detector["multipath_enrichment_threshold"],
            "persistence": f"{detector['persistence_required_bins']}-of-{detector['persistence_window_bins']}",
        },
    }
    output = resolve(config["output_root"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "delay_estimates.csv", delays)
    write_csv(output / "geometry_scores.csv", geometries)
    write_csv(output / "prn_summary.csv", prn_rows)
    summary["artifacts"] = {
        name: {"path": str(output / name), "sha256": sha256(output / name)}
        for name in ("delay_estimates.csv", "geometry_scores.csv", "prn_summary.csv")
    }
    summary["config"] = {"path": str(config_path), "sha256": sha256(config_path)}
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args().config.resolve())
