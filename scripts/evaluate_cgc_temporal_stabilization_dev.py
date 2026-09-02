#!/usr/bin/env python3
"""Evaluate a causal signed-delay stabilization candidate for CGC.

This is a development-only analysis.  It reuses immutable receiver outputs and
never changes the released CGC detector or its frozen Partial-F threshold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import audit_cgc_locked_phase_root_cause as cause  # noqa: E402
import run_cgc_code_carrier_decoupling_pilot as pilot  # noqa: E402
import run_cgc_rf_challenge_pilot as challenge  # noqa: E402
import run_cgc_rf_geometry_aperture_validation as geometry  # noqa: E402
from gnss_doppler_lab.clock_centered_geometry import fit_clock_centered_geometry  # noqa: E402
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table  # noqa: E402
from gnss_doppler_lab.static_reference_geometry import partial_f_score  # noqa: E402
from gnss_doppler_lab.temporal_cgc import causal_prn_median  # noqa: E402


CAMPAIGN_ROOT = Path("/home/ubuntu/hdd_data/cgc_code_carrier_fresh_static_v1")
PHASE_ROOT = Path("/home/ubuntu/hdd_data/cgc_locked_phase_sweep_dev_v1")
FRESH_CONFIG = ROOT / "configs/experiments/cgc_code_carrier_fresh_static_v1.json"
MULTIPATH_CONFIG = ROOT / "configs/experiments/cgc_rf_geometry_aperture_validation_v1.json"
PHASE_SUMMARY = ROOT / "artifacts/cgc_locked_phase_sweep_dev_v1/summary.json"
DEFAULT_OUTPUT = ROOT / "artifacts/cgc_temporal_stabilization_dev_v1"
P_THRESHOLD = 0.06028418845288192
WINDOWS = (1, 2, 3, 5, 7, 9)
SELECTED_WINDOW = 5
# A development diagnostic, not a frozen detector threshold.  Four 0.025-chip
# dictionary steps are used to ask whether an observability gate is warranted.
DIAGNOSTIC_RMS_THRESHOLD_CHIPS = 0.10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fresh_los() -> dict[str, dict[str, tuple[float, float, float]]]:
    result = {}
    for pair_root in sorted((CAMPAIGN_ROOT / "pairs").iterdir()):
        if pair_root.is_dir():
            result[pair_root.name] = parse_gps_sdr_sim_los_table(
                (pair_root / "components/authentic/simulator.log").read_text(encoding="utf-8")
            )
    return result


def multipath_inputs() -> tuple[list[dict[str, Any]], dict[str, dict[str, tuple[float, float, float]]]]:
    config = json.loads(MULTIPATH_CONFIG.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    los_by_stream = {}
    for item in config["geometries"]:
        pair_id = str(item["geometry_id"])
        source = ROOT / item["source_assets"]["authentic_los_log"]["path"]
        if sha256(source) != item["source_assets"]["authentic_los_log"]["sha256"]:
            raise ValueError(f"multipath LOS hash mismatch: {pair_id}")
        los_by_stream[pair_id] = parse_gps_sdr_sim_los_table(source.read_text(encoding="utf-8"))
        delay_path = (
            ROOT / "artifacts/cgc_rf_ga_v1/geometries" / pair_id
            / "common_multipath/multipath_delay_estimates_9tap.csv"
        )
        for row in read_csv(delay_path):
            rows.append({**row, "pair_id": pair_id, "condition": "multipath"})
    return rows, los_by_stream


def score_delays(
    delay_rows: list[dict[str, Any]],
    los_by_stream: dict[str, dict[str, tuple[float, float, float]]],
    *,
    window_bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stabilized = causal_prn_median(delay_rows, window_bins=window_bins)
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in stabilized:
        key = (str(row["pair_id"]), str(row["condition"]), int(row["bin_index"]))
        grouped.setdefault(key, []).append(row)
    scored: list[dict[str, Any]] = []
    for (pair_id, condition, bin_index), entries in sorted(grouped.items()):
        los = los_by_stream[pair_id]
        selected = [row for row in entries if str(row["prn"]) in los]
        if len(selected) < 8:
            continue
        delays = np.asarray([float(row["stabilized_delay_chips"]) for row in selected])
        fit = fit_clock_centered_geometry(
            np.asarray([los[str(row["prn"])] for row in selected]), delays
        )
        statistic, p_value = partial_f_score(fit.clock_centered_normalized_residual, len(selected))
        centered_rms = float(np.sqrt(np.mean((delays - np.mean(delays)) ** 2)))
        scored.append(
            {
                "pair_id": pair_id,
                "condition": condition,
                "bin_index": bin_index,
                "bin_start_s": float(bin_index),
                "prn_count": len(selected),
                "clock_centered_geometry_residual": fit.clock_centered_normalized_residual,
                "directional_geometry_coherence": fit.directional_coherence,
                "partial_f": statistic,
                "partial_f_p_value": p_value,
                "partial_f_score": float(-math.log10(max(p_value, np.finfo(float).tiny))),
                "centered_delay_rms_chips": centered_rms,
                "estimated_displacement_norm_chips": float(np.linalg.norm(fit.theta[:3])),
                "raw_spoof_alarm": bool(p_value <= P_THRESHOLD),
                "diagnostic_joint_alarm": bool(
                    p_value <= P_THRESHOLD and centered_rms >= DIAGNOSTIC_RMS_THRESHOLD_CHIPS
                ),
            }
        )
    return stabilized, scored


def add_persistence(rows: list[dict[str, Any]], alarm_field: str, output_field: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(str(row["pair_id"]), str(row["condition"])) for row in rows})
    for pair_id, condition in keys:
        selected = sorted(
            (row for row in rows if row["pair_id"] == pair_id and row["condition"] == condition),
            key=lambda row: int(row["bin_index"]),
        )
        alarms = {int(row["bin_index"]): bool(row[alarm_field]) for row in selected}
        for row in selected:
            current = int(row["bin_index"])
            active = sum(alarms.get(candidate, False) for candidate in range(current - 4, current + 1)) >= 3
            output.append({**row, output_field: bool(active)})
    return output


def truth_agreement(
    stabilized: list[dict[str, Any]], *, condition: str = "doppler-locked"
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in stabilized:
        if str(row["condition"]) == condition and int(row["bin_index"]) >= cause.HOLD_START_BIN:
            grouped.setdefault((str(row["pair_id"]), int(row["bin_index"])), []).append(row)
    result = []
    for (pair_id, bin_index), entries in sorted(grouped.items()):
        pair_root = CAMPAIGN_ROOT / "pairs" / pair_id
        authentic = cause.truth_by_time_prn(pair_root / "components/authentic/truth.csv")
        spoof = cause.truth_by_time_prn(pair_root / "components/doppler-locked/truth.csv")
        time_s = round(bin_index + 0.5, 1)
        estimated, expected = [], []
        for row in entries:
            key = (time_s, str(row["prn"]))
            if key not in authentic or key not in spoof:
                continue
            estimated.append(float(row["stabilized_delay_chips"]))
            expected.append(
                (spoof[key]["code_range_m"] - authentic[key]["code_range_m"])
                / cause.CHIP_LENGTH_M
            )
        if len(estimated) >= 3:
            result.append(
                {
                    "pair_id": pair_id,
                    "condition": condition,
                    "bin_index": bin_index,
                    **cause.centered_truth_agreement(np.asarray(estimated), np.asarray(expected)),
                }
            )
    return result


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([bool(row[field]) for row in rows])) if rows else math.nan


def first_alarm(rows: list[dict[str, Any]], field: str, start_bin: int = 5) -> float | None:
    for row in sorted(rows, key=lambda item: int(item["bin_index"])):
        if int(row["bin_index"]) >= start_bin and bool(row[field]):
            return float(row["bin_start_s"])
    return None


def window_summary(
    window: int,
    spoof_delays: list[dict[str, Any]],
    spoof_los: dict[str, dict[str, tuple[float, float, float]]],
    multipath_delays: list[dict[str, Any]],
    multipath_los: dict[str, dict[str, tuple[float, float, float]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stabilized, spoof = score_delays(spoof_delays, spoof_los, window_bins=window)
    _, multipath = score_delays(multipath_delays, multipath_los, window_bins=window)
    spoof = add_persistence(spoof, "raw_spoof_alarm", "persistent_spoof_alarm")
    spoof = add_persistence(spoof, "diagnostic_joint_alarm", "diagnostic_joint_persistent_alarm")
    multipath = add_persistence(multipath, "raw_spoof_alarm", "persistent_spoof_alarm")
    multipath = add_persistence(multipath, "diagnostic_joint_alarm", "diagnostic_joint_persistent_alarm")
    hold = [row for row in spoof if int(row["bin_index"]) >= 12]
    pre = [row for row in spoof if int(row["bin_index"]) < 5]
    labels = np.r_[np.ones(len(hold), dtype=int), np.zeros(len(multipath), dtype=int)]
    score = np.asarray([row["partial_f_score"] for row in hold + multipath])
    gated_score = np.asarray([
        row["partial_f_score"]
        if float(row["centered_delay_rms_chips"]) >= DIAGNOSTIC_RMS_THRESHOLD_CHIPS
        else 0.0
        for row in hold + multipath
    ])
    agreement = truth_agreement(stabilized)
    summary = {
        "window_bins": window,
        "spoof_hold_bins": len(hold),
        "multipath_bins": len(multipath),
        "partial_f_auc": float(roc_auc_score(labels, score)),
        "diagnostic_rms_gated_auc": float(roc_auc_score(labels, gated_score)),
        "spoof_hold_raw_alarm_rate": rate(hold, "raw_spoof_alarm"),
        "spoof_hold_persistent_alarm_rate": rate(hold, "persistent_spoof_alarm"),
        "multipath_raw_alarm_rate": rate(multipath, "raw_spoof_alarm"),
        "multipath_persistent_alarm_rate": rate(multipath, "persistent_spoof_alarm"),
        "pre_attack_persistent_alarm_count": int(sum(bool(row["persistent_spoof_alarm"]) for row in pre)),
        "diagnostic_joint_spoof_hold_raw_alarm_rate": rate(hold, "diagnostic_joint_alarm"),
        "diagnostic_joint_multipath_raw_alarm_rate": rate(multipath, "diagnostic_joint_alarm"),
        "diagnostic_joint_pre_attack_persistent_alarm_count": int(
            sum(bool(row["diagnostic_joint_persistent_alarm"]) for row in pre)
        ),
        "median_truth_direction_r2": float(np.median([row["truth_direction_r2"] for row in agreement])),
    }
    return summary, spoof, multipath, agreement


def pair_summary(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for pair_id in sorted({str(row["pair_id"]) for row in scored}):
        rows = [row for row in scored if row["pair_id"] == pair_id]
        hold = [row for row in rows if int(row["bin_index"]) >= 12]
        pre = [row for row in rows if int(row["bin_index"]) < 5]
        first = first_alarm(rows, "persistent_spoof_alarm")
        joint_first = first_alarm(rows, "diagnostic_joint_persistent_alarm")
        result.append(
            {
                "pair_id": pair_id,
                "hold_raw_alarm_rate": rate(hold, "raw_spoof_alarm"),
                "hold_persistent_alarm_rate": rate(hold, "persistent_spoof_alarm"),
                "median_hold_p_value": float(np.median([row["partial_f_p_value"] for row in hold])),
                "first_persistent_alarm_s": first,
                "latency_from_onset_s": None if first is None else first - 5.0,
                "pre_attack_persistent_alarm_count": int(sum(bool(row["persistent_spoof_alarm"]) for row in pre)),
                "diagnostic_joint_hold_raw_alarm_rate": rate(hold, "diagnostic_joint_alarm"),
                "diagnostic_joint_first_persistent_alarm_s": joint_first,
                "diagnostic_joint_latency_from_onset_s": None if joint_first is None else joint_first - 5.0,
                "diagnostic_joint_pre_attack_persistent_alarm_count": int(
                    sum(bool(row["diagnostic_joint_persistent_alarm"]) for row in pre)
                ),
            }
        )
    return result


def ensure_phase_delays(output: Path) -> list[dict[str, Any]]:
    cache = output / "phase_delay_estimates.csv"
    if cache.is_file():
        return read_csv(cache)
    phase_summary = json.loads(PHASE_SUMMARY.read_text(encoding="utf-8"))
    fresh = json.loads(FRESH_CONFIG.read_text(encoding="utf-8"))
    controlled = json.loads((ROOT / fresh["inputs"]["controlled_template"]["path"]).read_text(encoding="utf-8"))
    estimator = challenge._estimator(controlled)
    tokyo = CAMPAIGN_ROOT / "pairs/ccfs-s1-a-tokyo"
    los = parse_gps_sdr_sim_los_table((tokyo / "components/authentic/simulator.log").read_text(encoding="utf-8"))
    master = read_csv(CAMPAIGN_ROOT / "analysis/delay_estimates.csv")
    rows: list[dict[str, Any]] = []
    for item in phase_summary["rows"]:
        phase = float(item["phase_offset_deg"])
        tag = f"phase-{int(round(phase)):03d}"
        if math.isclose(phase, 0.0):
            delays = [
                row for row in master
                if row["pair_id"] == "ccfs-s1-a-tokyo" and row["condition"] == "doppler-locked"
            ]
        else:
            manifest = Path(item["receiver_manifest"])
            delays, _ = geometry.analyze_stream(tag, manifest, estimator, los, fresh, 9)
        for row in delays:
            rows.append({**row, "pair_id": tag, "condition": "phase-sweep", "phase_offset_deg": phase})
    write_csv(cache, rows)
    return rows


def phase_summary(output: Path, window: int) -> list[dict[str, Any]]:
    delay_rows = ensure_phase_delays(output)
    tokyo_los = parse_gps_sdr_sim_los_table(
        (CAMPAIGN_ROOT / "pairs/ccfs-s1-a-tokyo/components/authentic/simulator.log").read_text(encoding="utf-8")
    )
    los = {str(row["pair_id"]): tokyo_los for row in delay_rows}
    stabilized, scored = score_delays(delay_rows, los, window_bins=window)
    scored = add_persistence(scored, "raw_spoof_alarm", "persistent_spoof_alarm")
    truth_rows = []
    authentic = cause.truth_by_time_prn(CAMPAIGN_ROOT / "pairs/ccfs-s1-a-tokyo/components/authentic/truth.csv")
    spoof = cause.truth_by_time_prn(CAMPAIGN_ROOT / "pairs/ccfs-s1-a-tokyo/components/doppler-locked/truth.csv")
    for tag in sorted({str(row["pair_id"]) for row in stabilized}):
        for bin_index in range(12, 30):
            entries = [row for row in stabilized if row["pair_id"] == tag and int(row["bin_index"]) == bin_index]
            estimated, expected = [], []
            for row in entries:
                key = (round(bin_index + 0.5, 1), str(row["prn"]))
                if key in authentic and key in spoof:
                    estimated.append(float(row["stabilized_delay_chips"]))
                    expected.append((spoof[key]["code_range_m"] - authentic[key]["code_range_m"]) / cause.CHIP_LENGTH_M)
            if len(estimated) >= 3:
                truth_rows.append({"pair_id": tag, "bin_index": bin_index, **cause.centered_truth_agreement(np.asarray(estimated), np.asarray(expected))})
    result = []
    for tag in sorted({str(row["pair_id"]) for row in scored}):
        rows = [row for row in scored if row["pair_id"] == tag]
        hold = [row for row in rows if int(row["bin_index"]) >= 12]
        first = first_alarm(rows, "persistent_spoof_alarm")
        phase = float(next(row["phase_offset_deg"] for row in delay_rows if row["pair_id"] == tag))
        truth = [row for row in truth_rows if row["pair_id"] == tag]
        result.append(
            {
                "phase_offset_deg": phase,
                "window_bins": window,
                "median_truth_direction_r2": float(np.median([row["truth_direction_r2"] for row in truth])),
                "median_hold_p_value": float(np.median([row["partial_f_p_value"] for row in hold])),
                "hold_raw_alarm_rate": rate(hold, "raw_spoof_alarm"),
                "hold_persistent_alarm_rate": rate(hold, "persistent_spoof_alarm"),
                "first_persistent_alarm_s": first,
                "latency_from_onset_s": None if first is None else first - 5.0,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-phase-sweep", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    spoof_delays = [
        row for row in read_csv(CAMPAIGN_ROOT / "analysis/delay_estimates.csv")
        if row["condition"] == "doppler-locked"
    ]
    spoof_los = fresh_los()
    multipath_delays, multipath_los = multipath_inputs()
    windows, selected_payload = [], None
    for window in WINDOWS:
        print(f"[window] {window}", flush=True)
        summary, spoof, multipath, agreement = window_summary(
            window, spoof_delays, spoof_los, multipath_delays, multipath_los
        )
        windows.append(summary)
        if window == SELECTED_WINDOW:
            selected_payload = (spoof, multipath, agreement)
    if selected_payload is None:
        raise RuntimeError("selected window was not evaluated")
    spoof, multipath, agreement = selected_payload
    write_csv(output / "selected_spoof_scores.csv", spoof)
    write_csv(output / "selected_multipath_scores.csv", multipath)
    write_csv(output / "selected_truth_agreement.csv", agreement)
    pairs = pair_summary(spoof)
    phases = [] if args.skip_phase_sweep else phase_summary(output, SELECTED_WINDOW)
    summary = {
        "schema": "gnss-doppler-lab.cgc-temporal-stabilization-development",
        "schema_version": 1,
        "role": "development-only candidate selection; not a fresh validation result",
        "candidate": {
            "name": "causal per-PRN signed-delay median before unchanged CGC",
            "selected_window_bins": SELECTED_WINDOW,
            "selection_reason": "matches the existing five-bin decision horizon; chosen before phase-sweep reanalysis",
            "partial_f_p_alarm_threshold": P_THRESHOLD,
            "persistence": "3 of latest 5 one-second bins",
            "threshold_changed": False,
        },
        "diagnostic_observability_gate": {
            "used_in_selected_detector": False,
            "centered_delay_rms_threshold_chips": DIAGNOSTIC_RMS_THRESHOLD_CHIPS,
            "reason": "post-hoc diagnostic of pre-attack overfit; requires independent calibration before use",
        },
        "window_sweep": windows,
        "selected_pair_results": pairs,
        "selected_phase_sweep": phases,
        "inputs": {
            "fresh_delay_estimates": {
                "path": str((CAMPAIGN_ROOT / "analysis/delay_estimates.csv").resolve()),
                "sha256": sha256(CAMPAIGN_ROOT / "analysis/delay_estimates.csv"),
            },
            "fresh_config": {"path": str(FRESH_CONFIG.resolve()), "sha256": sha256(FRESH_CONFIG)},
            "multipath_config": {"path": str(MULTIPATH_CONFIG.resolve()), "sha256": sha256(MULTIPATH_CONFIG)},
            "phase_summary": {"path": str(PHASE_SUMMARY.resolve()), "sha256": sha256(PHASE_SUMMARY)},
        },
        "claim_boundary": (
            "The same five released exact-lock pairs and five existing synthetic multipath streams were reused for development. "
            "The window sweep is adaptive and the phase sweep reuses one Tokyo geometry. A new untouched receiver-RF campaign "
            "is required before replacing the frozen CGC or making a confirmatory claim."
        ),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
