#!/usr/bin/env python3
"""Development-only time audit for physically coherent carry-off then hold."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.acquisition_surface import (  # noqa: E402
    compute_acquisition_surface,
    read_s8_iq,
)
from gnss_doppler_lab.doppler_observability import (  # noqa: E402
    dominant_doppler_peaks,
    local_probe,
    normalized_doppler_envelope,
)


DEFAULT_CAMPAIGN_ROOT = Path("/home/ubuntu/hdd_data/cgc_code_carrier_fresh_static_v1")
DEFAULT_OUTPUT_ROOT = Path("/home/ubuntu/hdd_data/cgc_doppler_silent_hold_development_v1")
SOURCE_RATE_HZ = 25_000_000
ANALYSIS_RATE_HZ = 5_000_000
COHERENT_MS = 20
DOPPLER_STEP_HZ = 10
SEARCH_HALF_WIDTH_HZ = 250
PROBE_HALF_WIDTH_HZ = 25
DUAL_PEAK_HEIGHT = 0.5
DUAL_PEAK_PROMINENCE = 0.1
DUAL_PEAK_MINIMUM_SEPARATION_HZ = 60.0
TU_MINIMUM_SEPARATION_HZ = 3.0
TU_MINIMUM_PRNS = 5
CGC_MINIMUM_PRNS = 8
CGC_P_THRESHOLD = 0.06028418845288192
CHIP_LENGTH_M = 299_792_458.0 / 1_023_000.0
ANCHOR_TIMES_S = (4.5, 7.5, 8.5, 12.5, 20.5, 28.5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def read_rows(path: Path) -> list[dict[str, str]]:
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


def truth_map(path: Path) -> dict[tuple[float, int], dict[str, float]]:
    output: dict[tuple[float, int], dict[str, float]] = {}
    for row in read_rows(path):
        key = (round(float(row["time_s"]), 1), int(row["prn"]))
        output[key] = {
            name: float(value)
            for name, value in row.items()
            if name not in {"time_s", "prn"}
        }
    return output


def truth_bin_metrics(
    authentic: dict[tuple[float, int], dict[str, float]],
    spoof: dict[tuple[float, int], dict[str, float]],
    bin_index: int,
) -> dict[str, Any]:
    time_s = round(float(bin_index) + 0.5, 1)
    prns = sorted(prn for t, prn in set(authentic).intersection(spoof) if t == time_s)
    if not prns:
        raise ValueError(f"no common truth rows at {time_s:.1f} s")
    doppler = np.asarray([
        abs(spoof[(time_s, prn)]["carrier_doppler_hz"] - authentic[(time_s, prn)]["carrier_doppler_hz"])
        for prn in prns
    ])
    code_m = np.asarray([
        abs(spoof[(time_s, prn)]["code_range_m"] - authentic[(time_s, prn)]["code_range_m"])
        for prn in prns
    ])
    eligible = int(np.count_nonzero(doppler >= TU_MINIMUM_SEPARATION_HZ))
    return {
        "truth_time_s": time_s,
        "truth_prn_count": len(prns),
        "tu_oracle_prn_count": eligible,
        "tu_oracle_available": eligible >= TU_MINIMUM_PRNS,
        "median_abs_doppler_separation_hz": float(np.median(doppler)),
        "maximum_abs_doppler_separation_hz": float(np.max(doppler)),
        "median_abs_code_offset_m": float(np.median(code_m)),
        "maximum_abs_code_offset_m": float(np.max(code_m)),
        "maximum_abs_code_offset_chips": float(np.max(code_m) / CHIP_LENGTH_M),
    }


def tracking_bins(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    values: dict[tuple[int, int], dict[str, list[float]]] = {}
    for row in read_rows(path):
        time_s = float(row["time_s"])
        if not 0.0 <= time_s < 30.0:
            continue
        bin_index = int(np.floor(time_s))
        prn = int(row["prn"][1:])
        slot = values.setdefault((bin_index, prn), {"doppler": [], "lock": [], "cn0": []})
        slot["doppler"].append(float(row["carrier_doppler_hz"]))
        slot["lock"].append(float(row["carrier_lock_test"]))
        slot["cn0"].append(float(row["CN0_SNV_dB_Hz"]))
    output: dict[tuple[int, int], dict[str, Any]] = {}
    for key, slot in values.items():
        output[key] = {
            "epoch_count": len(slot["doppler"]),
            "median_carrier_doppler_hz": float(np.median(slot["doppler"])),
            "median_carrier_lock_test": float(np.median(slot["lock"])),
            "median_cn0_db_hz": float(np.median(slot["cn0"])),
        }
    return output


def geometry_map(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_rows(path):
        if row["condition"] != "carrier-coupled":
            continue
        output[(row["pair_id"], int(row["bin_index"]))] = {
            "cgc_prn_count": int(row["prn_count"]),
            "partial_f_p_value": float(row["partial_f_p_value"]),
            "raw_spoof_alarm": parse_bool(row["raw_spoof_alarm"]),
            "persistent_spoof_alarm": parse_bool(row["persistent_spoof_alarm"]),
        }
    return output


def phase_for_bin(bin_index: int) -> str:
    if bin_index < 5:
        return "baseline"
    if bin_index < 10:
        return "pull-off"
    if bin_index < 12:
        return "guard"
    return "hold"


def longest_consecutive_bins(bins: Iterable[int]) -> int:
    ordered = sorted(set(int(value) for value in bins))
    longest = current = 0
    previous: int | None = None
    for value in ordered:
        current = current + 1 if previous is not None and value == previous + 1 else 1
        longest = max(longest, current)
        previous = value
    return longest


def timeline_for_pair(
    pair_id: str,
    authentic: dict[tuple[float, int], dict[str, float]],
    spoof: dict[tuple[float, int], dict[str, float]],
    tracking: dict[tuple[int, int], dict[str, Any]],
    geometry: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bin_index in range(30):
        truth = truth_bin_metrics(authentic, spoof, bin_index)
        track = [value for (index, _), value in tracking.items() if index == bin_index]
        if not track:
            raise ValueError(f"no tracking support for {pair_id} bin {bin_index}")
        cgc = geometry.get((pair_id, bin_index))
        if cgc is None:
            raise ValueError(f"no CGC row for {pair_id} bin {bin_index}")
        complement = (
            bin_index >= 12
            and not truth["tu_oracle_available"]
            and bool(cgc["persistent_spoof_alarm"])
            and int(cgc["cgc_prn_count"]) >= CGC_MINIMUM_PRNS
        )
        rows.append({
            "pair_id": pair_id,
            "bin_index": bin_index,
            "bin_start_s": float(bin_index),
            "phase": phase_for_bin(bin_index),
            **truth,
            "tracking_prn_count": len(track),
            "median_carrier_lock_test": float(np.median([value["median_carrier_lock_test"] for value in track])),
            "median_cn0_db_hz": float(np.median([value["median_cn0_db_hz"] for value in track])),
            **cgc,
            "doppler_silent_cgc_complement": complement,
        })
    return rows


def closest_truth(
    values: dict[tuple[float, int], dict[str, float]], time_s: float, prn: int
) -> dict[str, float]:
    key = (round(time_s, 1), prn)
    if key not in values:
        raise ValueError(f"missing truth row {key}")
    return values[key]


def iq_anchor_rows(
    pair_id: str,
    iq_path: Path,
    rf_manifest: dict[str, Any],
    authentic: dict[tuple[float, int], dict[str, float]],
    spoof: dict[tuple[float, int], dict[str, float]],
    tracking: dict[tuple[int, int], dict[str, Any]],
    completed: dict[tuple[str, float, str], dict[str, Any]],
    output_csv: Path,
) -> list[dict[str, Any]]:
    rows = list(completed.values())
    requested = rf_manifest["simulation_v4"]["receiver"]["requested"]
    carrier_offset = float(requested["carrier_offset_hz"])
    carrier_drift = float(requested["frequency_drift_hz_per_s"])
    source_samples = int(round(SOURCE_RATE_HZ * COHERENT_MS / 1000.0))
    samples_per_code = int(round(ANALYSIS_RATE_HZ / 1000.0))
    for time_s in ANCHOR_TIMES_S:
        bin_index = int(np.floor(time_s))
        candidates = sorted(prn for index, prn in tracking if index == bin_index)
        pending = [prn for prn in candidates if (pair_id, time_s, f"G{prn:02d}") not in completed]
        if not pending:
            continue
        raw = read_s8_iq(iq_path, source_samples, int(round(time_s * SOURCE_RATE_HZ)))
        iq = resample_poly(raw, 1, SOURCE_RATE_HZ // ANALYSIS_RATE_HZ).astype(np.complex64)
        for prn in pending:
            label = f"G{prn:02d}"
            center = float(tracking[(bin_index, prn)]["median_carrier_doppler_hz"])
            lower = DOPPLER_STEP_HZ * int(np.floor((center - SEARCH_HALF_WIDTH_HZ) / DOPPLER_STEP_HZ))
            upper = DOPPLER_STEP_HZ * int(np.ceil((center + SEARCH_HALF_WIDTH_HZ) / DOPPLER_STEP_HZ))
            surface = compute_acquisition_surface(
                iq,
                prn,
                ANALYSIS_RATE_HZ,
                coherent_ms=COHERENT_MS,
                doppler_min_hz=lower,
                doppler_max_hz=upper,
                doppler_step_hz=DOPPLER_STEP_HZ,
            )
            envelope = normalized_doppler_envelope(surface.magnitude, samples_per_code=samples_per_code)
            peaks = dominant_doppler_peaks(
                surface.doppler_bins_hz,
                envelope,
                minimum_height=DUAL_PEAK_HEIGHT,
                minimum_prominence=DUAL_PEAK_PROMINENCE,
                minimum_separation_hz=DUAL_PEAK_MINIMUM_SEPARATION_HZ,
            )
            auth_truth = closest_truth(authentic, time_s, prn)
            spoof_truth = closest_truth(spoof, time_s, prn)
            frontend_hz = carrier_offset + carrier_drift * time_s
            auth_hz = auth_truth["carrier_doppler_hz"] + frontend_hz
            spoof_hz = spoof_truth["carrier_doppler_hz"] + frontend_hz
            search_contract_valid = True
            try:
                auth_probe_hz, auth_probe = local_probe(
                    surface.doppler_bins_hz, envelope, auth_hz, half_width_hz=PROBE_HALF_WIDTH_HZ
                )
                spoof_probe_hz, spoof_probe = local_probe(
                    surface.doppler_bins_hz, envelope, spoof_hz, half_width_hz=PROBE_HALF_WIDTH_HZ
                )
            except ValueError:
                search_contract_valid = False
                auth_probe_hz = auth_probe = float("nan")
                spoof_probe_hz = spoof_probe = float("nan")
            separation = abs(spoof_hz - auth_hz)
            row = {
                "pair_id": pair_id,
                "time_s": time_s,
                "phase": phase_for_bin(bin_index),
                "prn": label,
                "tracking_center_hz": center,
                "expected_authentic_doppler_hz": auth_hz,
                "expected_spoof_doppler_hz": spoof_hz,
                "truth_abs_doppler_separation_hz": separation,
                "tu_oracle_prn_eligible": separation >= TU_MINIMUM_SEPARATION_HZ,
                "dominant_peak_count": len(peaks.frequencies_hz),
                "dominant_peak_frequencies_hz": ";".join(f"{value:.3f}" for value in peaks.frequencies_hz),
                "dominant_peak_heights": ";".join(f"{value:.6f}" for value in peaks.normalized_heights),
                "search_contract_valid": search_contract_valid,
                "actual_dual_peak_observed": search_contract_valid and len(peaks.frequencies_hz) >= 2,
                "authentic_probe_hz": auth_probe_hz,
                "authentic_probe_height": auth_probe,
                "spoof_probe_hz": spoof_probe_hz,
                "spoof_probe_height": spoof_probe,
                "expected_pair_visible_at_audit_resolution": (
                    search_contract_valid
                    and separation >= DUAL_PEAK_MINIMUM_SEPARATION_HZ
                    and auth_probe >= DUAL_PEAK_HEIGHT
                    and spoof_probe >= DUAL_PEAK_HEIGHT
                    and len(peaks.frequencies_hz) >= 2
                ),
            }
            rows.append(row)
            completed[(pair_id, time_s, label)] = row
        write_csv(output_csv, sorted(rows, key=lambda row: (row["pair_id"], float(row["time_s"]), row["prn"])))
    return sorted(rows, key=lambda row: (row["pair_id"], float(row["time_s"]), row["prn"]))


def anchor_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["pair_id"]), float(row["time_s"])), []).append(row)
    output = []
    for (pair_id, time_s), group in sorted(groups.items()):
        valid = [row for row in group if parse_bool(row.get("search_contract_valid", True))]
        output.append({
            "pair_id": pair_id,
            "time_s": time_s,
            "phase": group[0]["phase"],
            "tracked_prn_count": len(group),
            "iq_evaluable_prn_count": len(valid),
            "tu_oracle_prn_count": sum(parse_bool(row["tu_oracle_prn_eligible"]) for row in group),
            "actual_dual_peak_prn_count": sum(parse_bool(row["actual_dual_peak_observed"]) for row in valid),
            "expected_pair_visible_prn_count": sum(parse_bool(row["expected_pair_visible_at_audit_resolution"]) for row in valid),
            "median_truth_abs_doppler_separation_hz": float(np.median([float(row["truth_abs_doppler_separation_hz"]) for row in group])),
            "median_authentic_probe_height": float(np.median([float(row["authentic_probe_height"]) for row in valid])) if valid else "",
            "median_spoof_probe_height": float(np.median([float(row["spoof_probe_height"]) for row in valid])) if valid else "",
        })
    return output


def pair_diagnostics(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for pair_id in sorted({str(row["pair_id"]) for row in timeline}):
        selected = [row for row in timeline if row["pair_id"] == pair_id]
        pull = [row for row in selected if row["phase"] == "pull-off"]
        hold = [row for row in selected if row["phase"] == "hold"]
        complement_bins = [int(row["bin_index"]) for row in hold if row["doppler_silent_cgc_complement"]]
        output.append({
            "pair_id": pair_id,
            "pull_off_bins_tu_oracle_available": sum(bool(row["tu_oracle_available"]) for row in pull),
            "maximum_hold_doppler_separation_hz": max(float(row["maximum_abs_doppler_separation_hz"]) for row in hold),
            "median_hold_code_offset_m": float(np.median([float(row["median_abs_code_offset_m"]) for row in hold])),
            "maximum_hold_code_offset_chips": max(float(row["maximum_abs_code_offset_chips"]) for row in hold),
            "hold_raw_alarm_rate": float(np.mean([bool(row["raw_spoof_alarm"]) for row in hold])),
            "hold_persistent_alarm_rate": float(np.mean([bool(row["persistent_spoof_alarm"]) for row in hold])),
            "longest_doppler_silent_cgc_complement_seconds": longest_consecutive_bins(complement_bins),
            "minimum_hold_cgc_prns": min(int(row["cgc_prn_count"]) for row in hold),
        })
    return output


def plot_timeline(
    timeline: list[dict[str, Any]], anchors: list[dict[str, Any]], output_root: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    bins = sorted({int(row["bin_index"]) for row in timeline})
    x = np.asarray([value + 0.5 for value in bins], dtype=float)
    oracle = [[int(row["tu_oracle_prn_count"]) for row in timeline if int(row["bin_index"]) == value] for value in bins]
    p_values = [[float(row["partial_f_p_value"]) for row in timeline if int(row["bin_index"]) == value] for value in bins]
    persistent = [[bool(row["persistent_spoof_alarm"]) for row in timeline if int(row["bin_index"]) == value] for value in bins]
    oracle_median = np.asarray([np.median(value) for value in oracle])
    oracle_min = np.asarray([np.min(value) for value in oracle])
    oracle_max = np.asarray([np.max(value) for value in oracle])
    p_median = np.asarray([np.median(value) for value in p_values])
    p_min = np.asarray([np.min(value) for value in p_values])
    p_max = np.asarray([np.max(value) for value in p_values])
    persistent_count = np.asarray([sum(value) for value in persistent])

    fig, axes = plt.subplots(2, 1, figsize=(7.15, 5.25), sharex=True, constrained_layout=True)
    phases = [(0, 5, "0.94"), (5, 10, "#fff0dc"), (10, 12, "0.96"), (12, 30, "#eaf3fb")]
    for ax in axes:
        for lo, hi, color in phases:
            ax.axvspan(lo, hi, color=color, zorder=0)
        for edge in (5, 10, 12):
            ax.axvline(edge, color="0.45", linewidth=0.8)
        ax.grid(True, alpha=0.22)

    ax = axes[0]
    ax.fill_between(x, oracle_min, oracle_max, color="#0072b2", alpha=0.16, linewidth=0)
    ax.plot(x, oracle_median, color="#0072b2", marker="o", markersize=3.0, linewidth=1.4, label="Oracle PRNs with |Delta f| >= 3 Hz")
    anchor_times = sorted({float(row["time_s"]) for row in anchors})
    anchor_medians = [np.median([int(row["actual_dual_peak_prn_count"]) for row in anchors if float(row["time_s"]) == time_s]) for time_s in anchor_times]
    ax.scatter(anchor_times, anchor_medians, color="#d55e00", marker="s", s=28, label="Actual-IQ dual-peak PRNs (median)", zorder=3)
    ax.axhline(TU_MINIMUM_PRNS, color="black", linestyle="--", linewidth=1.0, label="Tu input: 5 PRNs")
    ax.set_ylabel("PRN support")
    ax.set_ylim(-0.5, max(15.5, float(np.max(oracle_max)) + 1.0))
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    ax.set_title("Dual-Doppler support disappears while code-geometry evidence persists")

    ax = axes[1]
    floor = 1e-6
    ax.fill_between(x, np.maximum(p_min, floor), np.maximum(p_max, floor), color="#009e73", alpha=0.15, linewidth=0)
    ax.plot(x, np.maximum(p_median, floor), color="#009e73", marker="o", markersize=3.0, linewidth=1.4, label="Median CGC Partial-F p-value")
    ax.axhline(CGC_P_THRESHOLD, color="#cc3311", linestyle="--", linewidth=1.0, label="CGC threshold")
    ax.set_yscale("log")
    ax.set_ylabel("Partial-F p-value")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, 30)
    ax2 = ax.twinx()
    ax2.step(x, persistent_count, where="mid", color="#332288", linewidth=1.3, label="Persistent CGC pairs")
    ax2.set_ylabel("Persistent pairs / 5")
    ax2.set_ylim(-0.2, 5.4)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="lower right")
    for suffix in ("pdf", "svg"):
        fig.savefig(output_root / f"doppler_silent_hold_timeline.{suffix}")
    plt.close(fig)


def run(campaign_root: Path, output_root: Path, resume: bool) -> Path:
    source_summary_path = campaign_root / "summary.json"
    source = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source.get("schema") != "gnss-doppler-lab.cgc-code-carrier-fresh-static-result":
        raise ValueError("unexpected source campaign schema")
    geometry_path = Path(source["artifacts"]["geometry_scores"]["path"])
    geometry = geometry_map(geometry_path)
    output_root.mkdir(parents=True, exist_ok=True)
    iq_csv = output_root / "iq_anchor_metrics.csv"
    completed: dict[tuple[str, float, str], dict[str, Any]] = {}
    if resume and iq_csv.is_file():
        for row in read_rows(iq_csv):
            row.setdefault("search_contract_valid", "True")
            completed[(row["pair_id"], float(row["time_s"]), row["prn"])] = row
    elif iq_csv.exists():
        raise FileExistsError(f"output exists; use --resume: {iq_csv}")

    timeline: list[dict[str, Any]] = []
    identities = []
    for pair in source["pairs"]:
        pair_id = pair["pair_id"]
        print(f"[development] {pair_id}", flush=True)
        pair_root = campaign_root / "pairs" / pair_id
        authentic_path = pair_root / "components" / "authentic" / "truth.csv"
        spoof_path = pair_root / "components" / "carrier-coupled" / "truth.csv"
        receiver_manifest_path = Path(pair["conditions"]["carrier-coupled"]["receiver_manifest"]["path"])
        receiver_root = receiver_manifest_path.parent
        tracking_path = receiver_root / "tracking.csv"
        rf_manifest_path = pair_root / "rf" / "carrier-coupled" / "manifest.json"
        rf_manifest = json.loads(rf_manifest_path.read_text(encoding="utf-8"))
        iq_path = rf_manifest_path.parent / rf_manifest["iq"]["path"]
        authentic = truth_map(authentic_path)
        spoof = truth_map(spoof_path)
        tracking = tracking_bins(tracking_path)
        timeline.extend(timeline_for_pair(pair_id, authentic, spoof, tracking, geometry))
        iq_anchor_rows(pair_id, iq_path, rf_manifest, authentic, spoof, tracking, completed, iq_csv)
        identities.append({
            "pair_id": pair_id,
            "rf_manifest": str(rf_manifest_path.resolve()),
            "rf_manifest_sha256": sha256(rf_manifest_path),
            "iq": str(iq_path.resolve()),
            "iq_sha256_from_manifest": rf_manifest["iq"]["sha256"],
            "receiver_manifest": str(receiver_manifest_path.resolve()),
            "receiver_manifest_sha256": sha256(receiver_manifest_path),
        })

    timeline = sorted(timeline, key=lambda row: (row["pair_id"], int(row["bin_index"])))
    iq_rows = sorted(completed.values(), key=lambda row: (row["pair_id"], float(row["time_s"]), row["prn"]))
    anchors = anchor_summary(iq_rows)
    diagnostics = pair_diagnostics(timeline)
    write_csv(output_root / "timeline.csv", timeline)
    write_csv(iq_csv, iq_rows)
    write_csv(output_root / "iq_anchor_summary.csv", anchors)
    plot_timeline(timeline, anchors, output_root)

    mechanism_checks = {
        "all_pairs_have_three_tu_available_pull_off_bins": all(row["pull_off_bins_tu_oracle_available"] >= 3 for row in diagnostics),
        "all_pairs_have_hold_max_below_half_hz": all(row["maximum_hold_doppler_separation_hz"] < 0.5 for row in diagnostics),
        "all_pairs_keep_code_offset_above_25m": all(row["median_hold_code_offset_m"] >= 25.0 for row in diagnostics),
        "all_pairs_remain_inside_half_chip_aperture": all(row["maximum_hold_code_offset_chips"] <= 0.5 for row in diagnostics),
        "at_least_four_pairs_have_ten_second_complement": sum(row["longest_doppler_silent_cgc_complement_seconds"] >= 10 for row in diagnostics) >= 4,
        "zero_persistent_pre_attack_cgc_alarms": sum(bool(row["persistent_spoof_alarm"]) for row in timeline if row["phase"] == "baseline") == 0,
    }
    summary = {
        "schema": "gnss-doppler-lab.cgc-doppler-silent-hold-development",
        "schema_version": 1,
        "role": "development mechanism audit on previously exposed carrier-coupled RF; not fresh confirmation and not an exact Tu reproduction",
        "source_campaign": {"path": str(source_summary_path.resolve()), "sha256": sha256(source_summary_path), "decision": source["decision"]},
        "intervals_seconds": {"baseline": [0, 5], "pull_off": [5, 10], "guard": [10, 12], "hold": [12, 30]},
        "tu_style_contract": {"minimum_abs_doppler_separation_hz": TU_MINIMUM_SEPARATION_HZ, "minimum_prns": TU_MINIMUM_PRNS, "basis": "Tu et al. 2020 operating condition; oracle-favourable input availability, not detector reproduction"},
        "actual_iq_audit": {
            "anchor_times_s": ANCHOR_TIMES_S,
            "source_rate_hz": SOURCE_RATE_HZ,
            "analysis_rate_hz": ANALYSIS_RATE_HZ,
            "coherent_ms": COHERENT_MS,
            "doppler_step_hz": DOPPLER_STEP_HZ,
            "search_half_width_hz": SEARCH_HALF_WIDTH_HZ,
            "search_center": "per-PRN receiver tracking median in containing one-second bin",
            "dominant_peak_rule": {"minimum_height": DUAL_PEAK_HEIGHT, "minimum_prominence": DUAL_PEAK_PROMINENCE, "minimum_separation_hz": DUAL_PEAK_MINIMUM_SEPARATION_HZ},
            "row_count": len(iq_rows),
        },
        "pair_diagnostics": diagnostics,
        "mechanism_checks": mechanism_checks,
        "development_mechanism_consistent": all(mechanism_checks.values()),
        "source_identities": identities,
        "artifacts": {
            "timeline_csv": str((output_root / "timeline.csv").resolve()),
            "iq_anchor_metrics_csv": str(iq_csv.resolve()),
            "iq_anchor_summary_csv": str((output_root / "iq_anchor_summary.csv").resolve()),
            "figure_pdf": str((output_root / "doppler_silent_hold_timeline.pdf").resolve()),
            "figure_svg": str((output_root / "doppler_silent_hold_timeline.svg").resolve()),
        },
        "interpretation_boundary": "A consistent development mechanism justifies freezing a new fresh-static release. It does not establish superiority to all Doppler detectors, reproduce Tu et al. exactly, or convert the exposed source pairs into held-out evidence.",
    }
    destination = output_root / "summary.json"
    write_json(destination, summary)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    destination = run(args.campaign_root.resolve(), args.output_root.resolve(), args.resume)
    print(destination.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
