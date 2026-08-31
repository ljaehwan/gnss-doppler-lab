#!/usr/bin/env python3
"""Run the single-release frozen CGC detector on FGI-SpoofRepo TGD."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.stats import f as f_distribution
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_cgc_real_detection as real_detector  # noqa: E402
import run_cgc_rf_challenge_pilot as pilot  # noqa: E402
from gnss_doppler_lab.clock_centered_geometry import fit_clock_centered_geometry  # noqa: E402
from gnss_doppler_lab.correlator_geometry import complex_profile_features  # noqa: E402
from gnss_doppler_lab.gcmr_experiment import (  # noqa: E402
    parse_preonset_nmea_position,
    preflight_oakbat_geometry,
)
from gnss_doppler_lab.gcmr_geometry import (  # noqa: E402
    ephemeris_health_selection,
    parse_gnss_sdr_gps_ephemeris_xml,
    satellite_observation,
)
from gnss_doppler_lab.tracking_peaks import (  # noqa: E402
    available_tracking_prns,
    load_receiver_tracking_peak_series_segments,
)


DEFAULT_CONFIG = ROOT / "configs/experiments/fgi_spoofrepo_tgd_cgc_detection_v1.json"
PROTOCOL = ROOT / "docs/results/fgi_spoofrepo_tgd_cgc_detection_protocol_v1.md"
RELEASE_TOKEN = "RELEASE-FGI-SPOOFREPO-TGD-CGC-DETECTION-V1"
RELEASE_INPUTS = (
    "configs/experiments/fgi_spoofrepo_tgd_cgc_detection_v1.json",
    "docs/results/fgi_spoofrepo_tgd_cgc_detection_protocol_v1.md",
    "scripts/run_fgi_spoofrepo_tgd_cgc_detection.py",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify(record: dict[str, str], label: str) -> Path:
    path = resolve(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256(path)
    if observed != record["sha256"]:
        raise ValueError(f"{label} SHA-256 mismatch: {observed}")
    return path


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "gnss-doppler-lab.fgi-spoofrepo-tgd-cgc-detection-config" or config.get("schema_version") != 1:
        raise ValueError("unsupported FGI detector config")
    if config["experiment"].get("threshold_refitting") is not False or config["experiment"].get("post_release_tuning_or_retest") is not False:
        raise ValueError("post-release tuning is forbidden")
    analysis = config["analysis"]
    expected_analysis = {
        "bin_seconds": 1.0,
        "minimum_epochs_per_prn_bin": 40,
        "minimum_prns": 8,
        "analysis_seconds": [40.0, 230.0],
        "clean_seconds": [40.0, 120.0],
        "excluded_transition_seconds": [120.0, 160.0],
        "stable_post_seconds": [160.0, 230.0],
        "receiver_position_seconds": [40.0, 120.0],
        "maximum_ephemeris_toe_age_s": 7200.0,
        "epoch_chunk_size": 50000,
        "cn0_weighting": False,
    }
    if analysis != expected_analysis:
        raise ValueError("analysis contract drifted")
    detector = config["frozen_detector"]
    if detector["partial_f_p_alarm_threshold"] != 0.06028418845288192:
        raise ValueError("partial-F threshold drifted")
    if detector["persistence_window_bins"] != 5 or detector["persistence_required_bins"] != 3:
        raise ValueError("persistence rule drifted")
    evaluation = config["evaluation"]
    expected_gates = {
        "minimum_clean_geometry_bins": 60,
        "minimum_stable_post_geometry_bins": 60,
        "maximum_clean_persistent_alarm_rate": 0.05,
        "minimum_stable_post_persistent_detection_rate": 0.8,
        "require_post_median_p_below_clean_median_p": True,
        "serial_bin_auc_is_secondary_descriptive_only": True,
        "decision_rule": "SUPPORTED only if all fixed support, clean false-alarm, stable-post detection, and median-direction gates pass",
    }
    if evaluation != expected_gates:
        raise ValueError("evaluation gates drifted")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def committed_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        _git("ls-files", "--error-unmatch", relative)
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT).returncode:
            raise ValueError(f"release input is not committed and clean: {relative}")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "inputs": {relative: {"sha256": sha256(ROOT / relative)} for relative in RELEASE_INPUTS},
    }


def verify_context(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    iq = resolve(dataset["iq_path"])
    if not iq.is_file() or iq.stat().st_size != dataset["iq_bytes"] or sha256(iq) != dataset["iq_sha256"]:
        raise ValueError("FGI RF source mismatch")
    run_dir = resolve(config["receiver"]["run_dir"])
    receiver = config["receiver"]
    fixed_files = {
        "manifest.json": receiver["manifest_sha256"],
        "receiver.conf": receiver["receiver_config_sha256"],
        "gps_ephemeris.xml": receiver["gps_ephemeris_sha256"],
        "raw/observables.mat": receiver["observables_sha256"],
        "nmea_pvt.nmea": receiver["nmea_sha256"],
    }
    for relative, expected in fixed_files.items():
        if sha256(run_dir / relative) != expected:
            raise ValueError(f"receiver source drifted: {relative}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source"]["iq_sha256"] != dataset["iq_sha256"]:
        raise ValueError("receiver manifest RF hash mismatch")
    if manifest["tracking"]["tap_count"] != 9 or manifest["tracking"]["tap_spacing_chips"] != 0.125:
        raise ValueError("receiver is not the frozen complex nine-tap run")
    support_path = verify(config["support_preflight"], "support preflight")
    support = json.loads(support_path.read_text(encoding="utf-8"))
    if support["decision"] != config["support_preflight"]["required_decision"] or support["score_accessed"] is not False:
        raise ValueError("score-free input support did not pass")
    for label, record in config["implementation"].items():
        verify(record, label)
    template_path = verify(config["frozen_detector"]["template_config"], "template config")
    verify(config["frozen_detector"]["threshold_source"], "threshold source")
    partial_path = verify(config["frozen_detector"]["partial_f_audit"], "partial-F audit")
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    if float(partial["partial_f"]["p_value_alarm_threshold"]) != config["frozen_detector"]["partial_f_p_alarm_threshold"]:
        raise ValueError("partial-F audit threshold drifted")
    return {
        "run_dir": run_dir,
        "template": json.loads(template_path.read_text(encoding="utf-8")),
        "support": support,
    }


def _estimate(estimator: Any, taps: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    estimates = np.empty(len(taps), dtype=np.float64)
    distances = np.empty(len(taps), dtype=np.float64)
    for start in range(0, len(taps), chunk_size):
        end = min(len(taps), start + chunk_size)
        features = complex_profile_features(taps[start:end], prompt_index=4)
        estimates[start:end], distances[start:end], _ = estimator.estimate(features)
    return estimates, distances


def delay_rows(run_dir: Path, estimator: Any, healthy: set[int], analysis: dict[str, Any]) -> list[dict[str, Any]]:
    start_s, end_s = map(float, analysis["analysis_seconds"])
    bin_seconds = float(analysis["bin_seconds"])
    minimum_epochs = int(analysis["minimum_epochs_per_prn_bin"])
    chunk_size = int(analysis["epoch_chunk_size"])
    raw_rows: list[dict[str, Any]] = []
    available = set(available_tracking_prns(run_dir))
    for prn in sorted(healthy):
        name = f"G{prn:02d}"
        if name not in available:
            continue
        for segment in load_receiver_tracking_peak_series_segments(
            run_dir, name, tap_count=9, require_complex_taps=True
        ):
            mask = (segment.time_s >= start_s) & (segment.time_s < end_s)
            if not np.any(mask):
                continue
            times = segment.time_s[mask]
            taps = segment.complex_taps[mask]
            estimates, distances = _estimate(estimator, taps, chunk_size)
            bins = np.floor(times / bin_seconds).astype(np.int64)
            for bin_index in np.unique(bins):
                selected = bins == bin_index
                count = int(np.count_nonzero(selected))
                if count < minimum_epochs:
                    continue
                raw_rows.append({
                    "bin_index": int(bin_index),
                    "bin_start_s": float(bin_index * bin_seconds),
                    "prn": prn,
                    "prn_name": name,
                    "epoch_count": count,
                    "estimated_delay_chips": float(np.median(estimates[selected])),
                    "median_template_distance": float(np.median(distances[selected])),
                    "median_cn0_db_hz": float(np.median(segment.cn0_db_hz[mask][selected])),
                })
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault((row["bin_index"], row["prn"]), []).append(row)
    consolidated = []
    for (bin_index, prn), group in sorted(grouped.items()):
        consolidated.append({
            "bin_index": bin_index,
            "bin_start_s": group[0]["bin_start_s"],
            "prn": prn,
            "prn_name": f"G{prn:02d}",
            "epoch_count": max(row["epoch_count"] for row in group),
            "duplicate_channel_segment_count": len(group),
            "estimated_delay_chips": float(np.median([row["estimated_delay_chips"] for row in group])),
            "median_template_distance": float(np.median([row["median_template_distance"] for row in group])),
            "median_cn0_db_hz": float(np.median([row["median_cn0_db_hz"] for row in group])),
        })
    return consolidated


def partial_f_p_value(residual: float, prn_count: int) -> float:
    value = float(residual)
    count = int(prn_count)
    if not np.isfinite(value) or value < 0.0 or value > 1.0 + 1e-9 or count <= 4:
        raise ValueError("invalid partial-F inputs")
    value = min(max(value, np.finfo(float).tiny), 1.0)
    statistic = (1.0 - value) * (count - 4) / (3.0 * value)
    return float(f_distribution.sf(statistic, 3, count - 4))


def region(bin_start_s: float) -> str:
    if 40.0 <= bin_start_s < 120.0:
        return "clean"
    if 160.0 <= bin_start_s < 230.0:
        return "stable_post"
    return "excluded_transition"


def geometry_rows(rows: list[dict[str, Any]], ephemerides: dict[int, Any], receiver_ecef: tuple[float, float, float], config: dict[str, Any]) -> list[dict[str, Any]]:
    analysis = config["analysis"]
    detector = config["frozen_detector"]
    by_bin: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_bin.setdefault(int(row["bin_index"]), []).append(row)
    result = []
    for bin_index, entries in sorted(by_bin.items()):
        if len(entries) < int(analysis["minimum_prns"]):
            continue
        tow = (float(config["dataset"]["recording_start_tow_s"]) + bin_index + 0.5) % 604800.0
        los = np.asarray([
            satellite_observation(receiver_ecef, ephemerides[int(row["prn"])], tow).los_ecef
            for row in entries
        ], dtype=np.float64)
        delays = np.asarray([row["estimated_delay_chips"] for row in entries], dtype=np.float64)
        fit = fit_clock_centered_geometry(los, delays)
        p_value = partial_f_p_value(fit.clock_centered_normalized_residual, len(entries))
        result.append({
            "bin_index": bin_index,
            "bin_start_s": float(bin_index),
            "region": region(float(bin_index)),
            "gps_tow_s": tow,
            "prn_count": len(entries),
            "prns": " ".join(row["prn_name"] for row in entries),
            "clock_centered_geometry_residual": float(fit.clock_centered_normalized_residual),
            "partial_f_p_value": p_value,
            "raw_spoof_alarm": bool(p_value <= detector["partial_f_p_alarm_threshold"]),
            "directional_geometry_coherence": float(fit.directional_coherence),
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
    for row, alarm in zip(result, persistent):
        row["persistent_spoof_alarm"] = bool(alarm)
    return result


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    groups = {
        name: [row for row in rows if row["region"] == name]
        for name in ("clean", "stable_post")
    }
    metrics: dict[str, Any] = {}
    for name, group in groups.items():
        if not group:
            raise ValueError(f"no geometry rows in {name}")
        raw = np.asarray([row["raw_spoof_alarm"] for row in group], dtype=bool)
        persistent = np.asarray([row["persistent_spoof_alarm"] for row in group], dtype=bool)
        p_values = np.asarray([row["partial_f_p_value"] for row in group], dtype=np.float64)
        metrics[name] = {
            "geometry_bin_count": len(group),
            "minimum_prn_count": min(row["prn_count"] for row in group),
            "maximum_prn_count": max(row["prn_count"] for row in group),
            "raw_alarm_count": int(raw.sum()),
            "raw_alarm_rate": float(raw.mean()),
            "persistent_alarm_count": int(persistent.sum()),
            "persistent_alarm_rate": float(persistent.mean()),
            "median_partial_f_p_value": float(np.median(p_values)),
            "median_geometry_residual": float(np.median([row["clock_centered_geometry_residual"] for row in group])),
            "median_estimated_displacement_norm_chips": float(np.median([row["estimated_displacement_norm_chips"] for row in group])),
        }
    labels = np.r_[np.zeros(len(groups["clean"]), dtype=np.int64), np.ones(len(groups["stable_post"]), dtype=np.int64)]
    p_all = np.asarray([
        *[row["partial_f_p_value"] for row in groups["clean"]],
        *[row["partial_f_p_value"] for row in groups["stable_post"]],
    ], dtype=np.float64)
    after_onset = [
        row for row in rows
        if row["bin_start_s"] >= config["dataset"]["documented_attack_onset_s"]
        and row["persistent_spoof_alarm"]
    ]
    first = float(after_onset[0]["bin_start_s"]) if after_onset else None
    evaluation = config["evaluation"]
    gates = {
        "minimum_clean_geometry_bins": metrics["clean"]["geometry_bin_count"] >= evaluation["minimum_clean_geometry_bins"],
        "minimum_stable_post_geometry_bins": metrics["stable_post"]["geometry_bin_count"] >= evaluation["minimum_stable_post_geometry_bins"],
        "maximum_clean_persistent_alarm_rate": metrics["clean"]["persistent_alarm_rate"] <= evaluation["maximum_clean_persistent_alarm_rate"],
        "minimum_stable_post_persistent_detection_rate": metrics["stable_post"]["persistent_alarm_rate"] >= evaluation["minimum_stable_post_persistent_detection_rate"],
        "post_median_p_below_clean_median_p": metrics["stable_post"]["median_partial_f_p_value"] < metrics["clean"]["median_partial_f_p_value"],
    }
    passed = all(gates.values())
    return {
        "status": "SUPPORTED" if passed else "NOT_SUPPORTED",
        "all_preregistered_gates_passed": passed,
        "gates": gates,
        "regions": metrics,
        "secondary_serial_bin_auc": float(roc_auc_score(labels, -np.log10(np.maximum(p_all, np.finfo(float).tiny)))),
        "first_persistent_alarm_bin_start_s_at_or_after_documented_onset": first,
        "descriptive_latency_from_documented_onset_s": None if first is None else first - config["dataset"]["documented_attack_onset_s"],
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
    release = committed_release()
    context = verify_context(config)
    output = resolve(config["output_root"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    state = {
        "phase": "released_before_score_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": release,
        "config_sha256": sha256(config_path),
        "protocol_sha256": sha256(PROTOCOL),
        "score_accessed": False,
    }
    (output / "release_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_dir = context["run_dir"]
    ephemerides = parse_gnss_sdr_gps_ephemeris_xml(run_dir / "gps_ephemeris.xml")
    healthy_map, health = ephemeris_health_selection(
        ephemerides,
        tracked_prns=set(config["receiver"]["expected_healthy_prns"]),
        min_prns=int(config["analysis"]["minimum_prns"]),
    )
    if sorted(healthy_map) != config["receiver"]["expected_healthy_prns"]:
        raise ValueError("healthy PRN roster drifted")
    preflight = preflight_oakbat_geometry(
        run_dir / "raw/observables.mat",
        run_dir / "nmea_pvt.nmea",
        ephemerides,
        configured_tow0_s=float(config["dataset"]["recording_start_tow_s"]),
        max_toe_age_s=float(config["analysis"]["maximum_ephemeris_toe_age_s"]),
        tow_tolerance_s=0.05,
        onset_s=float(config["dataset"]["documented_attack_onset_s"]),
        tracked_prns=set(healthy_map),
        min_prns=int(config["analysis"]["minimum_prns"]),
    )
    position = parse_preonset_nmea_position(
        run_dir / "nmea_pvt.nmea",
        gps_tow_at_time_zero_s=float(config["dataset"]["recording_start_tow_s"]),
        onset_s=float(config["dataset"]["documented_attack_onset_s"]),
        position_window_s=tuple(config["analysis"]["receiver_position_seconds"]),
    )
    estimator = pilot._estimator(context["template"])
    delays = delay_rows(run_dir, estimator, set(healthy_map), config["analysis"])
    geometry = geometry_rows(delays, healthy_map, position["ecef"], config)
    primary = summarize(geometry, config)
    write_csv(output / "delay_estimates.csv", delays)
    write_csv(output / "geometry_scores.csv", geometry)
    state["phase"] = "score_accessed_terminal"
    state["score_accessed"] = True
    state["score_accessed_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "release_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema": "gnss-doppler-lab.fgi-spoofrepo-tgd-cgc-detection-result",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": release,
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "protocol": {"path": str(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "dataset": config["dataset"],
        "detector": config["frozen_detector"],
        "receiver_position": position,
        "geometry_preflight": preflight,
        "ephemeris_health": health,
        "delay_row_count": len(delays),
        "geometry_row_count": len(geometry),
        "primary": primary,
        "claim_boundary": config["claim_boundary"],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "status": primary["status"],
        "clean_persistent_alarm_rate": primary["regions"]["clean"]["persistent_alarm_rate"],
        "stable_post_persistent_detection_rate": primary["regions"]["stable_post"]["persistent_alarm_rate"],
        "secondary_serial_bin_auc": primary["secondary_serial_bin_auc"],
        "descriptive_latency_s": primary["descriptive_latency_from_documented_onset_s"],
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

