#!/usr/bin/env python3
"""Diagnose why frozen CGC weakens in exact Doppler-locked simulations.

This is a read-only development audit of the released five-pair campaign.  It
does not change the frozen detector, thresholds, or released result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.correlator_geometry import complex_profile_features  # noqa: E402
from gnss_doppler_lab.tracking_peaks import (  # noqa: E402
    available_tracking_prns,
    load_receiver_tracking_peak_series_segments,
)


DEFAULT_CAMPAIGN_ROOT = Path("/home/ubuntu/hdd_data/cgc_code_carrier_fresh_static_v1")
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/cgc_locked_phase_root_cause_v1"
SIMULATOR_SOURCE = ROOT / ".tools/gps-sdr-sim-code-carrier-src/gpssim.c"
LIGHT_SPEED_M_S = 299_792_458.0
CA_CODE_RATE_CHIPS_S = 1_023_000.0
CHIP_LENGTH_M = LIGHT_SPEED_M_S / CA_CODE_RATE_CHIPS_S
HOLD_START_BIN = 12


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth_by_time_prn(path: Path) -> dict[tuple[float, str], dict[str, float]]:
    result: dict[tuple[float, str], dict[str, float]] = {}
    for row in read_csv(path):
        key = (round(float(row["time_s"]), 1), f"G{int(row['prn']):02d}")
        result[key] = {
            name: float(value)
            for name, value in row.items()
            if name not in {"time_s", "prn"}
        }
    if not result:
        raise ValueError(f"empty truth file: {path}")
    return result


def centered_truth_agreement(
    estimated_delay_chips: np.ndarray,
    truth_delay_chips: np.ndarray,
) -> dict[str, float]:
    """Measure geometry-shape agreement while ignoring the common clock term."""
    estimated = np.asarray(estimated_delay_chips, dtype=np.float64)
    truth = np.asarray(truth_delay_chips, dtype=np.float64)
    if estimated.shape != truth.shape or estimated.ndim != 1 or len(estimated) < 3:
        raise ValueError("agreement requires equal one-dimensional vectors of length >= 3")
    estimated_centered = estimated - np.mean(estimated)
    truth_centered = truth - np.mean(truth)
    estimated_norm = float(np.linalg.norm(estimated_centered))
    truth_norm = float(np.linalg.norm(truth_centered))
    if estimated_norm <= 0.0 or truth_norm <= 0.0:
        return {
            "absolute_centered_correlation": 0.0,
            "truth_direction_r2": 0.0,
            "signed_affine_slope": 0.0,
            "estimated_to_truth_spread_ratio": 0.0,
        }
    correlation = float(np.dot(estimated_centered, truth_centered) / (estimated_norm * truth_norm))
    slope = float(np.dot(truth_centered, estimated_centered) / np.dot(truth_centered, truth_centered))
    return {
        "absolute_centered_correlation": abs(correlation),
        "truth_direction_r2": correlation * correlation,
        "signed_affine_slope": slope,
        "estimated_to_truth_spread_ratio": float(
            np.std(estimated_centered) / np.std(truth_centered)
        ),
    }


def quadrature_fraction(complex_taps: np.ndarray, *, prompt_index: int = 4) -> np.ndarray:
    """Return prompt-aligned energy outside the real (in-phase) tap axis."""
    features = complex_profile_features(complex_taps, prompt_index=prompt_index)
    matrix = features[None, :] if features.ndim == 1 else features
    tap_count = matrix.shape[1] // 2
    return np.linalg.norm(matrix[:, tap_count:], axis=1)


def receiver_run(pair_root: Path, pair_id: str, condition: str) -> Path:
    expected = pair_root / "receiver" / f"cgc-cc-fresh-{pair_id}-{condition}"
    if not (expected / "manifest.json").is_file():
        raise FileNotFoundError(expected)
    return expected


def geometry_lookup(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    return {
        (row["pair_id"], row["condition"], int(row["bin_index"])): row
        for row in read_csv(path)
    }


def delay_groups(path: Path) -> dict[tuple[str, str, int], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in read_csv(path):
        key = (row["pair_id"], row["condition"], int(row["bin_index"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def phase_rows(campaign_root: Path, *, epoch_step: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs_root = campaign_root / "pairs"
    for pair_root in sorted(path for path in pairs_root.iterdir() if path.is_dir()):
        pair_id = pair_root.name
        for condition in ("carrier-coupled", "doppler-locked"):
            run = receiver_run(pair_root, pair_id, condition)
            for prn in available_tracking_prns(run):
                for segment in load_receiver_tracking_peak_series_segments(
                    run,
                    prn,
                    epoch_step=epoch_step,
                    tap_count=9,
                    require_complex_taps=True,
                ):
                    fractions = quadrature_fraction(segment.complex_taps)
                    bins = np.floor(segment.time_s).astype(np.int64)
                    for bin_index in np.unique(bins):
                        if bin_index < HOLD_START_BIN:
                            continue
                        selected = bins == bin_index
                        rows.append(
                            {
                                "pair_id": pair_id,
                                "condition": condition,
                                "bin_index": int(bin_index),
                                "prn": prn,
                                "sampled_epoch_count": int(np.count_nonzero(selected)),
                                "median_quadrature_fraction": float(np.median(fractions[selected])),
                                "median_cn0_db_hz": float(np.median(segment.cn0_db_hz[selected])),
                            }
                        )
    if not rows:
        raise ValueError("no receiver phase rows produced")
    return rows


def agreement_rows(campaign_root: Path) -> list[dict[str, Any]]:
    delays = delay_groups(campaign_root / "analysis" / "delay_estimates.csv")
    geometry = geometry_lookup(campaign_root / "analysis" / "geometry_scores.csv")
    rows: list[dict[str, Any]] = []
    pairs_root = campaign_root / "pairs"
    for pair_root in sorted(path for path in pairs_root.iterdir() if path.is_dir()):
        pair_id = pair_root.name
        authentic = truth_by_time_prn(pair_root / "components/authentic/truth.csv")
        for condition in ("carrier-coupled", "doppler-locked"):
            spoof = truth_by_time_prn(pair_root / f"components/{condition}/truth.csv")
            for bin_index in range(HOLD_START_BIN, 30):
                key = (pair_id, condition, bin_index)
                if key not in delays or key not in geometry:
                    continue
                truth_time = round(bin_index + 0.5, 1)
                estimated: list[float] = []
                truth: list[float] = []
                distances: list[float] = []
                for item in delays[key]:
                    prn = item["prn"]
                    truth_key = (truth_time, prn)
                    if truth_key not in authentic or truth_key not in spoof:
                        continue
                    estimated.append(float(item["estimated_delay_chips"]))
                    truth.append(
                        (spoof[truth_key]["code_range_m"] - authentic[truth_key]["code_range_m"])
                        / CHIP_LENGTH_M
                    )
                    distances.append(float(item["median_template_distance"]))
                metrics = centered_truth_agreement(np.asarray(estimated), np.asarray(truth))
                score = geometry[key]
                rows.append(
                    {
                        "pair_id": pair_id,
                        "condition": condition,
                        "bin_index": bin_index,
                        "prn_count": len(estimated),
                        **metrics,
                        "median_template_distance": float(np.median(distances)),
                        "partial_f_p_value": float(score["partial_f_p_value"]),
                        "raw_spoof_alarm": score["raw_spoof_alarm"],
                    }
                )
    if not rows:
        raise ValueError("no truth-agreement rows produced")
    return rows


def median_by_condition(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    return {
        condition: float(np.median([float(row[field]) for row in rows if row["condition"] == condition]))
        for condition in ("carrier-coupled", "doppler-locked")
    }


def aggregate_phase_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    keys = sorted({(str(row["pair_id"]), str(row["condition"]), int(row["bin_index"])) for row in rows})
    for pair_id, condition, bin_index in keys:
        selected = [
            row for row in rows
            if row["pair_id"] == pair_id and row["condition"] == condition and row["bin_index"] == bin_index
        ]
        result.append(
            {
                "pair_id": pair_id,
                "condition": condition,
                "bin_index": bin_index,
                "prn_count": len(selected),
                "median_quadrature_fraction": float(np.median([row["median_quadrature_fraction"] for row in selected])),
                "median_cn0_db_hz": float(np.median([row["median_cn0_db_hz"] for row in selected])),
            }
        )
    return result


def simulator_phase_evidence(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    zero_initializers = len(re.findall(r"phase_ini\s*=\s*0\.0\s*;", text))
    independent_update = "computeCodeCarrierPhase(&chan[i], rho, rho_carrier, 0.1);" in text
    if zero_initializers < 1 or not independent_update:
        raise ValueError("expected simulator phase implementation was not found")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "zero_carrier_phase_initializer_count": zero_initializers,
        "independent_code_carrier_update_found": independent_update,
        "interpretation": (
            "Each separately generated component starts with zero carrier phase. "
            "When the spoof carrier trajectory is locked to the authentic trajectory, "
            "their relative carrier phase is therefore fixed at zero in this campaign."
        ),
    }


def plot_summary(
    phase: list[dict[str, Any]], agreement: list[dict[str, Any]], output: Path
) -> None:
    colors = {"carrier-coupled": "#0072B2", "doppler-locked": "#D55E00"}
    labels = {"carrier-coupled": "Carrier-coupled", "doppler-locked": "Exact locked"}
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), constrained_layout=True)
    specifications = (
        (phase, "median_quadrature_fraction", "Quadrature fraction", (0.0, 0.75)),
        (agreement, "truth_direction_r2", r"Delay-geometry agreement $R^2$", (0.0, 1.05)),
        (agreement, "partial_f_p_value", r"Partial-$F$ $p$-value", (0.0, 0.5)),
    )
    rng = np.random.default_rng(20260902)
    for axis, (rows, field, ylabel, ylim) in zip(axes, specifications):
        for position, condition in enumerate(("carrier-coupled", "doppler-locked")):
            values = np.asarray([float(row[field]) for row in rows if row["condition"] == condition])
            box = axis.boxplot(
                [values], positions=[position], widths=0.48, patch_artist=True,
                showfliers=False, medianprops={"color": "black", "linewidth": 1.2},
            )
            box["boxes"][0].set(facecolor=colors[condition], alpha=0.33, edgecolor=colors[condition])
            jitter = rng.normal(0.0, 0.035, len(values))
            axis.scatter(np.full(len(values), position) + jitter, values, s=6, alpha=0.23, color=colors[condition], linewidths=0)
        axis.set_xticks([0, 1], [labels["carrier-coupled"], labels["doppler-locked"]], rotation=16, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
        axis.grid(axis="y", color="0.88", linewidth=0.6)
    axes[2].axhline(0.06028418845288192, color="black", linestyle="--", linewidth=0.9, label="alarm threshold")
    axes[2].legend(loc="upper left", fontsize=7, frameon=False)
    for label, axis in zip(("(a)", "(b)", "(c)"), axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", ha="left", fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg")
    output.write_text("\n".join(line.rstrip() for line in output.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epoch-step", type=int, default=20)
    args = parser.parse_args()
    if args.epoch_step < 1:
        raise ValueError("epoch-step must be positive")

    campaign_root = args.campaign_root.resolve()
    output_root = args.output_root.resolve()
    per_prn_phase = phase_rows(campaign_root, epoch_step=args.epoch_step)
    phase = aggregate_phase_rows(per_prn_phase)
    agreement = agreement_rows(campaign_root)
    evidence = simulator_phase_evidence(SIMULATOR_SOURCE)

    write_csv(output_root / "per_prn_phase_metrics.csv", per_prn_phase)
    write_csv(output_root / "phase_metrics.csv", phase)
    write_csv(output_root / "delay_truth_agreement.csv", agreement)
    plot_summary(phase, agreement, output_root / "locked_phase_root_cause.svg")

    summary = {
        "schema": "gnss-doppler-lab.cgc-locked-phase-root-cause-audit",
        "schema_version": 1,
        "role": "development-only post-release causal diagnosis",
        "campaign_root": str(campaign_root),
        "hold_start_bin": HOLD_START_BIN,
        "epoch_step": args.epoch_step,
        "simulator_phase_evidence": evidence,
        "hold_bin_medians": {
            "quadrature_fraction": median_by_condition(phase, "median_quadrature_fraction"),
            "truth_direction_r2": median_by_condition(agreement, "truth_direction_r2"),
            "absolute_centered_correlation": median_by_condition(agreement, "absolute_centered_correlation"),
            "estimated_to_truth_spread_ratio": median_by_condition(agreement, "estimated_to_truth_spread_ratio"),
            "template_distance": median_by_condition(agreement, "median_template_distance"),
            "partial_f_p_value": median_by_condition(agreement, "partial_f_p_value"),
            "cn0_db_hz": median_by_condition(phase, "median_cn0_db_hz"),
        },
        "interpretation_boundary": (
            "The audit can identify a deterministic zero-relative-phase degeneracy in this simulator campaign "
            "and its association with lost quadrature information and weaker delay-geometry recovery. "
            "It cannot establish performance over arbitrary real-spoofer initial phases without a receiver-in-loop phase sweep."
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["hold_bin_medians"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
