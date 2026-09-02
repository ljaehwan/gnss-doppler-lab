#!/usr/bin/env python3
"""Audit code/carrier observability across coupled, locked, and hold regimes.

This is a read-only, development analysis of the already released fresh-static
campaign. It does not rerun the receiver or alter the frozen CGC decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CAMPAIGN_ROOT = Path("/home/ubuntu/hdd_data/cgc_code_carrier_fresh_static_v1")
DEFAULT_OUTPUT_ROOT = Path("artifacts/cgc_three_regime_complementarity_v1")
L1_TO_CA_CARRIER_RATIO = 1540.0
TU_MINIMUM_SEPARATION_HZ = 3.0
TU_MINIMUM_PRNS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def truth_table(path: Path) -> dict[tuple[float, int], dict[str, float]]:
    table: dict[tuple[float, int], dict[str, float]] = {}
    for row in read_csv(path):
        key = (round(float(row["time_s"]), 1), int(row["prn"]))
        table[key] = {
            name: float(value)
            for name, value in row.items()
            if name not in {"time_s", "prn"}
        }
    if not table:
        raise ValueError(f"empty truth table: {path}")
    return table


def phase_for_bin(bin_index: int) -> str:
    if bin_index < 5:
        return "baseline"
    if bin_index < 10:
        return "pull-off"
    if bin_index < 12:
        return "guard"
    return "hold"


def truth_observability_metrics(
    authentic: dict[tuple[float, int], dict[str, float]],
    spoof: dict[tuple[float, int], dict[str, float]],
    bin_index: int,
) -> dict[str, Any]:
    time_s = round(float(bin_index) + 0.5, 1)
    keys = sorted(
        key for key in set(authentic).intersection(spoof) if key[0] == time_s
    )
    if not keys:
        raise ValueError(f"no common truth rows at {time_s:.1f} s")

    code_offset_m = np.asarray(
        [spoof[key]["code_range_m"] - authentic[key]["code_range_m"] for key in keys]
    )
    code_rate_difference = np.asarray(
        [
            spoof[key]["code_frequency_hz"] - authentic[key]["code_frequency_hz"]
            for key in keys
        ]
    )
    carrier_difference = np.asarray(
        [
            spoof[key]["carrier_doppler_hz"] - authentic[key]["carrier_doppler_hz"]
            for key in keys
        ]
    )
    consistency_mismatch = (
        carrier_difference - L1_TO_CA_CARRIER_RATIO * code_rate_difference
    )
    tu_prns = int(
        np.count_nonzero(np.abs(carrier_difference) >= TU_MINIMUM_SEPARATION_HZ)
    )
    return {
        "truth_time_s": time_s,
        "truth_prn_count": len(keys),
        "tu_oracle_prn_count": tu_prns,
        "tu_oracle_input_available": tu_prns >= TU_MINIMUM_PRNS,
        "median_abs_code_offset_m": float(np.median(np.abs(code_offset_m))),
        "maximum_abs_code_offset_m": float(np.max(np.abs(code_offset_m))),
        "median_abs_code_rate_difference_chips_s": float(
            np.median(np.abs(code_rate_difference))
        ),
        "maximum_abs_code_rate_difference_chips_s": float(
            np.max(np.abs(code_rate_difference))
        ),
        "median_abs_carrier_doppler_difference_hz": float(
            np.median(np.abs(carrier_difference))
        ),
        "maximum_abs_carrier_doppler_difference_hz": float(
            np.max(np.abs(carrier_difference))
        ),
        "median_abs_code_carrier_mismatch_equivalent_hz": float(
            np.median(np.abs(consistency_mismatch))
        ),
        "maximum_abs_code_carrier_mismatch_equivalent_hz": float(
            np.max(np.abs(consistency_mismatch))
        ),
    }


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1"}:
        return True
    if lowered in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def geometry_table(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    table: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in read_csv(path):
        key = (row["pair_id"], row["condition"], int(row["bin_index"]))
        table[key] = {
            "cgc_prn_count": int(row["prn_count"]),
            "partial_f_p_value": float(row["partial_f_p_value"]),
            "raw_spoof_alarm": parse_bool(row["raw_spoof_alarm"]),
            "persistent_spoof_alarm": parse_bool(row["persistent_spoof_alarm"]),
        }
    if not table:
        raise ValueError(f"empty geometry table: {path}")
    return table


def regime_name(condition: str, phase: str) -> str:
    if phase == "pull-off":
        return "consistent_pull_off" if condition == "carrier-coupled" else "locked_pull_off"
    if phase == "hold":
        return "coupled_position_hold" if condition == "carrier-coupled" else "locked_position_hold"
    return f"{condition}_{phase}".replace("-", "_")


def build_timeline(campaign_root: Path) -> list[dict[str, Any]]:
    geometry = geometry_table(campaign_root / "analysis/geometry_scores.csv")
    pair_root = campaign_root / "pairs"
    pair_ids = sorted(path.name for path in pair_root.iterdir() if path.is_dir())
    if not pair_ids:
        raise ValueError(f"no campaign pairs: {pair_root}")

    rows: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        components = pair_root / pair_id / "components"
        authentic = truth_table(components / "authentic/truth.csv")
        for condition in ("carrier-coupled", "doppler-locked"):
            spoof = truth_table(components / condition / "truth.csv")
            for bin_index in range(30):
                key = (pair_id, condition, bin_index)
                if key not in geometry:
                    continue
                phase = phase_for_bin(bin_index)
                rows.append(
                    {
                        "pair_id": pair_id,
                        "condition": condition,
                        "bin_index": bin_index,
                        "bin_start_s": float(bin_index),
                        "phase": phase,
                        "regime": regime_name(condition, phase),
                        **truth_observability_metrics(authentic, spoof, bin_index),
                        **geometry[key],
                    }
                )
    if not rows:
        raise ValueError("timeline is empty")
    return rows


def summarize_regimes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_regimes = (
        "consistent_pull_off",
        "locked_pull_off",
        "coupled_position_hold",
        "locked_position_hold",
    )
    summary: list[dict[str, Any]] = []
    for regime in selected_regimes:
        subset = [row for row in rows if row["regime"] == regime]
        if not subset:
            raise ValueError(f"missing regime: {regime}")
        pair_ids = sorted({str(row["pair_id"]) for row in subset})
        pair_raw_rates = [
            float(np.mean([bool(row["raw_spoof_alarm"]) for row in subset if row["pair_id"] == pair_id]))
            for pair_id in pair_ids
        ]
        pair_persistent = [
            any(bool(row["persistent_spoof_alarm"]) for row in subset if row["pair_id"] == pair_id)
            for pair_id in pair_ids
        ]
        summary.append(
            {
                "regime": regime,
                "pair_count": len(pair_ids),
                "bin_count": len(subset),
                "tu_oracle_available_bin_rate": float(
                    np.mean([bool(row["tu_oracle_input_available"]) for row in subset])
                ),
                "median_abs_code_offset_m": float(
                    np.median([row["median_abs_code_offset_m"] for row in subset])
                ),
                "median_abs_code_rate_difference_chips_s": float(
                    np.median(
                        [row["median_abs_code_rate_difference_chips_s"] for row in subset]
                    )
                ),
                "median_abs_carrier_doppler_difference_hz": float(
                    np.median(
                        [row["median_abs_carrier_doppler_difference_hz"] for row in subset]
                    )
                ),
                "median_abs_code_carrier_mismatch_equivalent_hz": float(
                    np.median(
                        [
                            row["median_abs_code_carrier_mismatch_equivalent_hz"]
                            for row in subset
                        ]
                    )
                ),
                "median_pair_cgc_raw_alarm_rate": float(np.median(pair_raw_rates)),
                "cgc_persistent_detection_pairs": int(sum(pair_persistent)),
            }
        )
    return summary


def plot_timeline(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conditions = ("carrier-coupled", "doppler-locked")
    colors = {"carrier-coupled": "#0072B2", "doppler-locked": "#D55E00"}
    labels = {"carrier-coupled": "Consistent Doppler", "doppler-locked": "Locked Doppler"}
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 6.5), sharex=True, constrained_layout=True)
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        times = sorted({int(row["bin_index"]) for row in selected})
        tu = [
            np.median([row["tu_oracle_prn_count"] for row in selected if row["bin_index"] == time])
            for time in times
        ]
        mismatch = [
            np.median(
                [
                    row["median_abs_code_carrier_mismatch_equivalent_hz"]
                    for row in selected
                    if row["bin_index"] == time
                ]
            )
            for time in times
        ]
        cgc = [
            np.mean(
                [
                    bool(row["raw_spoof_alarm"])
                    for row in selected
                    if row["bin_index"] == time
                ]
            )
            for time in times
        ]
        x = np.asarray(times, dtype=float) + 0.5
        axes[0].plot(x, tu, marker="o", markersize=3, color=colors[condition], label=labels[condition])
        axes[1].plot(x, mismatch, marker="o", markersize=3, color=colors[condition])
        axes[2].plot(x, cgc, marker="o", markersize=3, color=colors[condition])

    axes[0].axhline(TU_MINIMUM_PRNS, color="0.35", linestyle="--", linewidth=1, label="Tu input: 5 PRNs")
    axes[0].set_ylabel("PRNs with\n$|\\Delta f_c|\\geq3$ Hz")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[1].set_ylabel("Code/carrier mismatch\n(carrier-equivalent Hz)")
    axes[1].set_yscale("symlog", linthresh=0.05)
    axes[2].set_ylabel("CGC raw-alarm\nfraction (5 pairs)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(-0.03, 1.03)
    for axis in axes:
        axis.axvspan(5, 10, color="#E69F00", alpha=0.10)
        axis.axvspan(12, 30, color="#009E73", alpha=0.07)
        axis.grid(True, alpha=0.22)
    axes[0].text(7.5, axes[0].get_ylim()[1] * 0.93, "pull-off", ha="center", va="top", fontsize=8)
    axes[0].text(21.0, axes[0].get_ylim()[1] * 0.93, "position hold", ha="center", va="top", fontsize=8)
    fig.savefig(path, format="svg")
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def run(campaign_root: Path, output_root: Path) -> dict[str, Any]:
    source_summary = campaign_root / "summary.json"
    if not source_summary.is_file():
        raise FileNotFoundError(source_summary)
    timeline = build_timeline(campaign_root)
    regimes = summarize_regimes(timeline)
    output_root.mkdir(parents=True, exist_ok=True)
    timeline_path = output_root / "timeline.csv"
    regime_path = output_root / "regime_summary.csv"
    figure_path = output_root / "three_regime_timeline.svg"
    write_csv(timeline_path, timeline)
    write_csv(regime_path, regimes)
    plot_timeline(timeline, figure_path)
    document = {
        "schema": "gnss-doppler-lab.cgc-three-regime-complementarity-audit",
        "schema_version": 1,
        "status": "development_reanalysis_of_exposed_fresh_static_data",
        "source": {
            "campaign_root": str(campaign_root.resolve()),
            "summary": {"path": str(source_summary.resolve()), "sha256": sha256(source_summary)},
        },
        "operating_contract": {
            "tu_minimum_carrier_separation_hz": TU_MINIMUM_SEPARATION_HZ,
            "tu_minimum_prns": TU_MINIMUM_PRNS,
            "code_carrier_relation": "Delta carrier Doppler = 1540 * Delta code rate for GPS L1 C/A consistent-Doppler spoofing",
            "code_carrier_mismatch_role": "truth-domain physical observable only; no receiver threshold or detector claim",
        },
        "regimes": regimes,
        "decision": {
            "supported": [
                "Tu-style carrier dual-peak input is available during the tested consistent-Doppler pull-off",
                "Tu-style carrier dual-peak input is absent during the tested locked-Doppler pull-off and both position holds",
                "the truth code/carrier mismatch is large during locked pull-off and collapses in position hold",
                "CGC is persistent for all five carrier-coupled position holds",
            ],
            "not_supported": [
                "the current CGC signed-delay estimator is robust across locked-Doppler position holds",
                "an operational carrier/code consistency detector result",
                "universal superiority over Tu et al. or other Doppler detectors",
            ],
        },
        "artifacts": {
            "timeline": {"path": str(timeline_path.resolve()), "sha256": sha256(timeline_path)},
            "regime_summary": {"path": str(regime_path.resolve()), "sha256": sha256(regime_path)},
            "figure": {"path": str(figure_path.resolve()), "sha256": sha256(figure_path)},
        },
    }
    write_json(output_root / "summary.json", document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = run(args.campaign_root.resolve(), args.output_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
