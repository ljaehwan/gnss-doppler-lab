#!/usr/bin/env python3
"""Apply the single-release frozen CGC detector to JammerTest JT23-17.1.6."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import audit_jammertest2023_jt17_support as support_audit  # noqa: E402
import run_cgc_real_detection as real_detector  # noqa: E402
import run_cgc_rf_challenge_pilot as pilot  # noqa: E402
import run_fgi_spoofrepo_tgd_cgc_detection as fgi_detector  # noqa: E402
from gnss_doppler_lab.clock_centered_geometry import fit_clock_centered_geometry  # noqa: E402
from gnss_doppler_lab.gcmr_experiment import parse_preonset_nmea_position  # noqa: E402
from gnss_doppler_lab.gcmr_geometry import (  # noqa: E402
    ephemeris_health_selection,
    parse_gnss_sdr_gps_ephemeris_xml,
    satellite_observation,
)


DEFAULT_CONFIG = ROOT / "configs/experiments/jammertest2023_jt17_cgc_v1.json"
PROTOCOL = ROOT / "docs/results/jammertest2023_jt17_cgc_protocol_v1.md"
RELEASE_TOKEN = "RELEASE-JAMMERTEST2023-JT17-CGC-V1"


def resolve(value: str | Path) -> Path:
    return support_audit.resolve(value)


def sha256(path: str | Path) -> str:
    return support_audit.sha256(path)


def validate_config(config: dict[str, Any]) -> None:
    support_audit.validate_config(config)
    experiment = config["experiment"]
    if experiment.get("threshold_refitting") is not False:
        raise ValueError("JammerTest threshold refitting is forbidden")
    if experiment.get("post_release_detector_tuning") is not False:
        raise ValueError("post-release detector tuning is forbidden")
    analysis = config["analysis"]
    if analysis["analysis_seconds"] != [40.0, 556.0]:
        raise ValueError("JammerTest analysis interval drifted")
    if int(analysis["minimum_prns"]) != 8 or int(analysis["minimum_epochs_per_prn_bin"]) != 200:
        raise ValueError("JammerTest detector support rule drifted")
    detector = config["frozen_detector"]
    if detector["partial_f_p_alarm_threshold"] != 0.06028418845288192:
        raise ValueError("partial-F threshold drifted")
    if detector["persistence_window_bins"] != 5 or detector["persistence_required_bins"] != 3:
        raise ValueError("persistence rule drifted")
    expected = {
        "minimum_clean_geometry_bins": 60,
        "minimum_aligned_spoof_geometry_bins": 60,
        "minimum_carryoff_onset_geometry_bins": 20,
        "maximum_clean_persistent_alarm_rate": 0.05,
        "maximum_motion_onset_latency_s": 30.0,
        "require_motion_onset_median_p_below_clean_median_p": True,
        "aligned_spoof_alarm_rate_is_descriptive_only": True,
        "serial_bin_auc_is_descriptive_only": True,
        "decision_rule": "SUPPORTED only if support, clean specificity, one onset-reset 3-of-5 carry-off alarm by 30 s, and median-direction gates all pass",
    }
    if config["evaluation"] != expected:
        raise ValueError("JammerTest terminal gates drifted")


def verify_record(record: dict[str, str], label: str) -> Path:
    path = resolve(record["path"])
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"{label} identity mismatch: {path}")
    return path


def region(bin_start_s: float, config: dict[str, Any]) -> str:
    value = float(bin_start_s)
    for name, bounds in config["analysis"]["regions_seconds"].items():
        lo, hi = map(float, bounds)
        if lo <= value < hi:
            return name
    return "excluded"


def geometry_rows(
    delay_rows: list[dict[str, Any]], ephemerides: dict[int, Any],
    receiver_ecef: tuple[float, float, float], recording_start_tow_s: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    analysis, detector = config["analysis"], config["frozen_detector"]
    by_bin: dict[int, list[dict[str, Any]]] = {}
    for row in delay_rows:
        by_bin.setdefault(int(row["bin_index"]), []).append(row)
    result: list[dict[str, Any]] = []
    for bin_index, entries in sorted(by_bin.items()):
        if len(entries) < int(analysis["minimum_prns"]):
            continue
        tow = (float(recording_start_tow_s) + bin_index + 0.5) % 604800.0
        los = np.asarray([
            satellite_observation(receiver_ecef, ephemerides[int(row["prn"])], tow).los_ecef
            for row in entries
        ], dtype=np.float64)
        delays = np.asarray([row["estimated_delay_chips"] for row in entries], dtype=np.float64)
        fit = fit_clock_centered_geometry(los, delays)
        p_value = fgi_detector.partial_f_p_value(
            fit.clock_centered_normalized_residual, len(entries)
        )
        result.append({
            "bin_index": bin_index,
            "bin_start_s": float(bin_index),
            "region": region(float(bin_index), config),
            "gps_tow_s": tow,
            "prn_count": len(entries),
            "prns": " ".join(row["prn_name"] for row in entries),
            "clock_centered_geometry_residual": float(fit.clock_centered_normalized_residual),
            "partial_f_p_value": p_value,
            "raw_spoof_alarm": bool(p_value <= detector["partial_f_p_alarm_threshold"]),
            "directional_geometry_coherence": float(fit.directional_coherence),
            "estimated_displacement_x_chips": float(fit.theta[0]),
            "estimated_displacement_y_chips": float(fit.theta[1]),
            "estimated_displacement_z_chips": float(fit.theta[2]),
            "estimated_displacement_norm_chips": float(np.linalg.norm(fit.theta[:3])),
            "clock_bias_chips": float(fit.theta[3]),
            "fit_rank": int(fit.rank),
        })
    bins = np.asarray([row["bin_index"] for row in result], dtype=np.int64)
    raw = np.asarray([row["raw_spoof_alarm"] for row in result], dtype=bool)
    persistent = real_detector.persistent_alarm(
        raw, bins,
        window=int(detector["persistence_window_bins"]),
        required=int(detector["persistence_required_bins"]),
    )
    motion_onset = int(config["dataset"]["planned_carryoff_motion_onset_s"])
    raw_by_bin = {int(row["bin_index"]): bool(row["raw_spoof_alarm"]) for row in result}
    for row, alarm in zip(result, persistent):
        row["persistent_spoof_alarm"] = bool(alarm)
        bin_index = int(row["bin_index"])
        lo = max(motion_onset, bin_index - int(detector["persistence_window_bins"]) + 1)
        post_onset_raw = sum(raw_by_bin.get(candidate, False) for candidate in range(lo, bin_index + 1))
        row["onset_reset_persistent_alarm"] = bool(
            bin_index >= motion_onset
            and post_onset_raw >= int(detector["persistence_required_bins"])
        )
    return result


def _region_metrics(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    group = [row for row in rows if row["region"] == name]
    if not group:
        raise ValueError(f"no geometry rows in {name}")
    raw = np.asarray([row["raw_spoof_alarm"] for row in group], dtype=bool)
    persistent = np.asarray([row["persistent_spoof_alarm"] for row in group], dtype=bool)
    p_values = np.asarray([row["partial_f_p_value"] for row in group], dtype=np.float64)
    return {
        "geometry_bin_count": len(group),
        "minimum_prn_count": min(row["prn_count"] for row in group),
        "maximum_prn_count": max(row["prn_count"] for row in group),
        "raw_alarm_count": int(raw.sum()),
        "raw_alarm_rate": float(raw.mean()),
        "persistent_alarm_count": int(persistent.sum()),
        "persistent_alarm_rate": float(persistent.mean()),
        "median_partial_f_p_value": float(np.median(p_values)),
        "median_geometry_residual": float(np.median([
            row["clock_centered_geometry_residual"] for row in group
        ])),
        "median_estimated_displacement_norm_chips": float(np.median([
            row["estimated_displacement_norm_chips"] for row in group
        ])),
    }


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        name: _region_metrics(rows, name)
        for name in ("clean", "aligned_spoof", "carryoff_onset")
    }
    onset_rows = [
        row for row in rows
        if row["region"] == "carryoff_onset" and row["onset_reset_persistent_alarm"]
    ]
    first = float(onset_rows[0]["bin_start_s"]) if onset_rows else None
    motion_onset = float(config["dataset"]["planned_carryoff_motion_onset_s"])
    latency = None if first is None else first - motion_onset
    evaluation = config["evaluation"]
    gates = {
        "minimum_clean_geometry_bins": metrics["clean"]["geometry_bin_count"] >= evaluation["minimum_clean_geometry_bins"],
        "minimum_aligned_spoof_geometry_bins": metrics["aligned_spoof"]["geometry_bin_count"] >= evaluation["minimum_aligned_spoof_geometry_bins"],
        "minimum_carryoff_onset_geometry_bins": metrics["carryoff_onset"]["geometry_bin_count"] >= evaluation["minimum_carryoff_onset_geometry_bins"],
        "maximum_clean_persistent_alarm_rate": metrics["clean"]["persistent_alarm_rate"] <= evaluation["maximum_clean_persistent_alarm_rate"],
        "motion_onset_persistent_alarm_within_30_s": latency is not None and 0.0 <= latency <= evaluation["maximum_motion_onset_latency_s"],
        "motion_onset_median_p_below_clean_median_p": metrics["carryoff_onset"]["median_partial_f_p_value"] < metrics["clean"]["median_partial_f_p_value"],
    }
    passed = all(gates.values())
    clean = [row for row in rows if row["region"] == "clean"]
    onset = [row for row in rows if row["region"] == "carryoff_onset"]
    labels = np.r_[np.zeros(len(clean), dtype=np.int64), np.ones(len(onset), dtype=np.int64)]
    p_values = np.asarray([
        *[row["partial_f_p_value"] for row in clean],
        *[row["partial_f_p_value"] for row in onset],
    ], dtype=np.float64)
    auc = float(roc_auc_score(
        labels, -np.log10(np.maximum(p_values, np.finfo(float).tiny))
    ))
    return {
        "status": (
            "REAL_CARRYOFF_TRANSFER_SUPPORTED"
            if passed else "REAL_CARRYOFF_TRANSFER_NOT_SUPPORTED"
        ),
        "all_preregistered_gates_passed": passed,
        "gates": gates,
        "regions": metrics,
        "first_onset_reset_persistent_alarm_bin_start_s": first,
        "latency_from_planned_carryoff_motion_onset_s": latency,
        "secondary_serial_bin_auc_clean_vs_carryoff_onset": auc,
        "aligned_spoof_alarm_rate_role": "descriptive_only",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    release = support_audit.committed_release()
    dataset, receiver = config["dataset"], config["receiver"]
    iq = resolve(dataset["iq_path"])
    if not iq.is_file() or iq.stat().st_size != int(dataset["iq_bytes"]):
        raise ValueError("JammerTest IQ identity mismatch")
    run_dir = resolve(receiver["output_root"]) / receiver["run_id"]
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source"].get("iq_sha256") != dataset["iq_sha256"]:
        raise ValueError("receiver did not pin the published full-file SHA-256")
    support_path = resolve(config["support"]["output_root"]) / "summary.json"
    support = json.loads(support_path.read_text(encoding="utf-8"))
    if support.get("decision") != "SUPPORT_ELIGIBLE":
        raise ValueError("score-free support did not pass; scoring is forbidden")
    if support.get("score_accessed") is not False or support.get("tap_values_read") is not False:
        raise ValueError("support audit crossed the score-free boundary")
    if support["release"]["commit"] != release["commit"]:
        raise ValueError("support and detector release commits differ")
    if support["receiver_manifest"]["sha256"] != sha256(manifest_path):
        raise ValueError("receiver manifest changed after support release")
    template_path = verify_record(config["frozen_detector"]["template_config"], "template config")
    verify_record(config["frozen_detector"]["threshold_source"], "threshold source")
    partial_path = verify_record(config["frozen_detector"]["partial_f_audit"], "partial-F audit")
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    if float(partial["partial_f"]["p_value_alarm_threshold"]) != config["frozen_detector"]["partial_f_p_alarm_threshold"]:
        raise ValueError("partial-F source threshold drifted")
    output = resolve(config["evaluation_output_root"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    state = {
        "phase": "released_before_score_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": release,
        "config_sha256": sha256(config_path),
        "protocol_sha256": sha256(PROTOCOL),
        "support_summary_sha256": sha256(support_path),
        "score_accessed": False,
    }
    state_path = output / "release_state.json"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ephemerides = parse_gnss_sdr_gps_ephemeris_xml(run_dir / "gps_ephemeris.xml")
    expected_healthy = {int(value[1:]) for value in support["healthy_ephemeris_prns"]}
    healthy_map, health = ephemeris_health_selection(
        ephemerides, tracked_prns=expected_healthy, min_prns=int(config["analysis"]["minimum_prns"])
    )
    if set(healthy_map) != expected_healthy:
        raise ValueError("healthy PRN roster changed after support preflight")
    tow0 = float(support["recording_timing"]["recording_start_tow_s"])
    position = parse_preonset_nmea_position(
        run_dir / "nmea_pvt.nmea",
        gps_tow_at_time_zero_s=tow0,
        onset_s=float(dataset["official_spoof_rf_onset_s"]),
        position_window_s=tuple(config["analysis"]["receiver_position_seconds"]),
    )
    template = json.loads(template_path.read_text(encoding="utf-8"))
    estimator = pilot._estimator(template)
    delays = fgi_detector.delay_rows(
        run_dir, estimator, set(healthy_map), config["analysis"]
    )
    geometry = geometry_rows(delays, healthy_map, position["ecef"], tow0, config)
    primary = summarize(geometry, config)
    write_csv(output / "delay_estimates.csv", delays)
    write_csv(output / "geometry_scores.csv", geometry)
    state["phase"] = "score_accessed_terminal"
    state["score_accessed"] = True
    state["score_accessed_at_utc"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema": "gnss-doppler-lab.jammertest2023-jt17-cgc-result.v1",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": release,
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "protocol": {"path": str(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "dataset": dataset,
        "receiver_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "support_summary": {"path": str(support_path), "sha256": sha256(support_path)},
        "recording_start_tow_s": tow0,
        "receiver_position": position,
        "ephemeris_health": health,
        "detector": config["frozen_detector"],
        "delay_row_count": len(delays),
        "geometry_row_count": len(geometry),
        "primary": primary,
        "claim_boundary": config["claim_boundary"],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path), "status": primary["status"],
        "clean_persistent_alarm_rate": primary["regions"]["clean"]["persistent_alarm_rate"],
        "aligned_spoof_persistent_alarm_rate_descriptive": primary["regions"]["aligned_spoof"]["persistent_alarm_rate"],
        "carryoff_latency_s": primary["latency_from_planned_carryoff_motion_onset_s"],
        "serial_bin_auc_descriptive": primary["secondary_serial_bin_auc_clean_vs_carryoff_onset"],
    }, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-token", required=True)
    args = parser.parse_args()
    if args.release_token != RELEASE_TOKEN:
        raise ValueError("release token mismatch")
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
