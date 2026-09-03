#!/usr/bin/env python3
"""Audit CGC delay sensing and LOS geometry against retained simulator truth.

This is a descriptive post-hoc audit of the already frozen
``cgc-temporal-final-static-v1`` receiver--RF campaign.  It does not create a
new test partition or refit any detector threshold.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gnss_doppler_lab.clock_centered_geometry import fit_clock_centered_geometry
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/cgc_temporal_final_static_v1.json"
SOURCE_RECORD = ROOT / "docs/results/cgc_temporal_final_static_v1_summary.json"
OUTPUT = ROOT / "docs/results/cgc_formula_accuracy_audit_v1_summary.json"
FIGURE_DIR = ROOT / "docs/results/figures"
FIGURE_PNG = FIGURE_DIR / "cgc_formula_accuracy_audit_v1.png"
FIGURE_PDF = FIGURE_DIR / "cgc_formula_accuracy_audit_v1.pdf"

CONDITIONS = ("carrier-coupled", "doppler-locked")
CHIP_LENGTH_M = 293.0522561094819
HOLD_START_S = 12.0
SIGN_TRUTH_MIN_CHIPS = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}")


def numeric_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    estimate = np.asarray(estimate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if estimate.shape != truth.shape or estimate.ndim != 1 or not len(estimate):
        raise ValueError("metric vectors must be nonempty and aligned")
    if not np.isfinite(estimate).all() or not np.isfinite(truth).all():
        raise ValueError("metric vectors must be finite")
    error = estimate - truth
    eligible = np.abs(truth) >= SIGN_TRUTH_MIN_CHIPS
    correlation = float(np.corrcoef(estimate, truth)[0, 1])
    return {
        "count": int(len(error)),
        "mae_chips": float(np.mean(np.abs(error))),
        "rmse_chips": float(np.sqrt(np.mean(error**2))),
        "bias_chips": float(np.mean(error)),
        "pearson_correlation": correlation,
        "sign_truth_min_abs_chips": SIGN_TRUTH_MIN_CHIPS,
        "sign_eligible_count": int(np.count_nonzero(eligible)),
        "sign_accuracy": float(
            np.mean(np.sign(estimate[eligible]) == np.sign(truth[eligible]))
        ),
    }


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("distribution values must be nonempty and finite")
    return {
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def load_truth_csv(pair_root: Path, condition: str) -> pd.DataFrame:
    manifest_path = pair_root / "components" / condition / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truth_path = pair_root / "components" / condition / "truth.csv"
    verify_file(truth_path, manifest["truth"]["sha256"], f"{condition} truth")
    truth = pd.read_csv(truth_path)
    if len(truth) != int(manifest["truth"]["rows"]):
        raise ValueError(f"{condition} truth row count mismatch")
    return truth


def pair_truth(
    data_root: Path, pair_id: str, condition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_root = data_root / "pairs" / pair_id
    authentic = load_truth_csv(pair_root, "authentic")
    secondary = load_truth_csv(pair_root, condition)
    merged = secondary.merge(
        authentic,
        on=["time_s", "prn"],
        suffixes=("_secondary", "_authentic"),
        validate="one_to_one",
    )
    if len(merged) != len(authentic) or len(merged) != len(secondary):
        raise ValueError(f"truth grids do not match for {pair_id}/{condition}")
    merged["truth_delay_chips"] = (
        merged["code_range_m_secondary"] - merged["code_range_m_authentic"]
    ) / CHIP_LENGTH_M
    merged["bin_index"] = np.floor(merged["time_s"]).astype(int)
    binned = (
        merged.groupby(["bin_index", "prn"], as_index=False)["truth_delay_chips"]
        .median()
    )
    binned["pair_id"] = pair_id
    binned["condition"] = condition
    return merged, binned


def load_all_truth(
    config: dict[str, Any], data_root: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    binned, sampled = [], {}
    for pair in config["pairs"]:
        pair_id = pair["candidate_id"]
        for condition in CONDITIONS:
            sample, bins = pair_truth(data_root, pair_id, condition)
            sampled[(pair_id, condition)] = sample
            binned.append(bins)
    return pd.concat(binned, ignore_index=True), sampled


def merge_estimates(
    path: Path,
    estimate_column: str,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[
        frame["condition"].isin(CONDITIONS)
        & (frame["bin_start_s"] >= HOLD_START_S)
    ].copy()
    frame["prn_int"] = frame["prn"].str.removeprefix("G").astype(int)
    merged = frame.merge(
        truth,
        left_on=["pair_id", "condition", "bin_index", "prn_int"],
        right_on=["pair_id", "condition", "bin_index", "prn"],
        validate="one_to_one",
        suffixes=("", "_truth"),
    )
    if len(merged) != len(frame):
        raise ValueError(f"unmatched estimator rows in {path}")
    keys = ["pair_id", "condition", "bin_index"]
    merged["centered_estimate"] = (
        merged[estimate_column]
        - merged.groupby(keys)[estimate_column].transform("mean")
    )
    merged["centered_truth"] = (
        merged["truth_delay_chips"]
        - merged.groupby(keys)["truth_delay_chips"].transform("mean")
    )
    return merged


def estimator_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "absolute": numeric_metrics(
            frame[column].to_numpy(), frame["truth_delay_chips"].to_numpy()
        ),
        "clock_centered": numeric_metrics(
            frame["centered_estimate"].to_numpy(),
            frame["centered_truth"].to_numpy(),
        ),
        "by_condition": {},
        "by_pair": {},
    }
    for condition, rows in frame.groupby("condition", sort=True):
        result["by_condition"][condition] = numeric_metrics(
            rows[column].to_numpy(), rows["truth_delay_chips"].to_numpy()
        )
    for pair_id, rows in frame.groupby("pair_id", sort=True):
        result["by_pair"][pair_id] = numeric_metrics(
            rows[column].to_numpy(), rows["truth_delay_chips"].to_numpy()
        )
    return result


def los_approximation_summary(
    config: dict[str, Any],
    data_root: Path,
    sampled_truth: dict[tuple[str, str], pd.DataFrame],
) -> dict[str, Any]:
    errors = []
    for pair in config["pairs"]:
        pair_id = pair["candidate_id"]
        displacement = np.asarray(pair["target_offset_enu_m"], dtype=np.float64)
        log_path = data_root / "pairs" / pair_id / "components/authentic/simulator.log"
        manifest = json.loads(
            (data_root / "pairs" / pair_id / "components/authentic/manifest.json")
            .read_text(encoding="utf-8")
        )
        verify_file(log_path, manifest["log"]["sha256"], f"{pair_id} LOS log")
        los = parse_gps_sdr_sim_los_table(log_path.read_text(encoding="utf-8"))
        samples = sampled_truth[(pair_id, "doppler-locked")]
        samples = samples[samples["time_s"] >= HOLD_START_S].copy()
        samples["linear_delay_chips"] = samples["prn"].map(
            lambda prn: (
                -float(np.dot(los[f"G{int(prn):02d}"], displacement))
                / CHIP_LENGTH_M
                if f"G{int(prn):02d}" in los
                else np.nan
            )
        )
        samples = samples.dropna(subset=["linear_delay_chips"])
        errors.extend(
            (
                samples["truth_delay_chips"] - samples["linear_delay_chips"]
            ).tolist()
        )
    error = np.asarray(errors, dtype=np.float64)
    return {
        "count": int(len(error)),
        "mae_chips": float(np.mean(np.abs(error))),
        "rmse_chips": float(np.sqrt(np.mean(error**2))),
        "maximum_absolute_error_chips": float(np.max(np.abs(error))),
        "mae_m": float(np.mean(np.abs(error)) * CHIP_LENGTH_M),
        "maximum_absolute_error_m": float(np.max(np.abs(error)) * CHIP_LENGTH_M),
        "interpretation": (
            "Exact simulator code-range difference minus the first-order prediction "
            "from the fixed startup LOS used by CGC; therefore this conservatively "
            "includes both range linearization and up to 30 s of LOS aging."
        ),
    }


def recovery_summary(
    config: dict[str, Any], data_root: Path, stabilized: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for pair in config["pairs"]:
        pair_id = pair["candidate_id"]
        truth_m = np.asarray(pair["target_offset_enu_m"], dtype=np.float64)
        truth_chips = truth_m / CHIP_LENGTH_M
        truth_norm = float(np.linalg.norm(truth_chips))
        log_path = data_root / "pairs" / pair_id / "components/authentic/simulator.log"
        los = parse_gps_sdr_sim_los_table(log_path.read_text(encoding="utf-8"))
        for condition in CONDITIONS:
            selected = stabilized[
                (stabilized["pair_id"] == pair_id)
                & (stabilized["condition"] == condition)
            ]
            for bin_index, group in selected.groupby("bin_index", sort=True):
                group = group[group["prn"].isin(los)].sort_values("prn")
                fit = fit_clock_centered_geometry(
                    np.asarray([los[prn] for prn in group["prn"]]),
                    group["stabilized_delay_chips"].to_numpy(),
                )
                estimate = fit.theta[:3]
                estimate_norm = float(np.linalg.norm(estimate))
                rows.append({
                    "pair_id": pair_id,
                    "condition": condition,
                    "bin_index": int(bin_index),
                    "direction_cosine": float(
                        np.dot(estimate, truth_chips) / (estimate_norm * truth_norm)
                    ),
                    "norm_ratio": estimate_norm / truth_norm,
                    "absolute_norm_relative_error": abs(estimate_norm / truth_norm - 1.0),
                    "vector_error_m": float(
                        np.linalg.norm(estimate - truth_chips) * CHIP_LENGTH_M
                    ),
                    "clock_centered_geometry_residual": (
                        fit.clock_centered_normalized_residual
                    ),
                })
    frame = pd.DataFrame(rows)
    result: dict[str, Any] = {
        "fit_count": int(len(frame)),
        "direction_cosine": distribution(frame["direction_cosine"].to_numpy()),
        "norm_ratio": distribution(frame["norm_ratio"].to_numpy()),
        "absolute_norm_relative_error": distribution(
            frame["absolute_norm_relative_error"].to_numpy()
        ),
        "vector_error_m": distribution(frame["vector_error_m"].to_numpy()),
        "clock_centered_geometry_residual": distribution(
            frame["clock_centered_geometry_residual"].to_numpy()
        ),
        "by_condition": {},
        "by_pair": {},
    }
    for condition, group in frame.groupby("condition", sort=True):
        result["by_condition"][condition] = {
            "direction_cosine_median": float(np.median(group["direction_cosine"])),
            "absolute_norm_relative_error_median": float(
                np.median(group["absolute_norm_relative_error"])
            ),
            "vector_error_m_median": float(np.median(group["vector_error_m"])),
        }
    for pair_id, group in frame.groupby("pair_id", sort=True):
        result["by_pair"][pair_id] = {
            "direction_cosine_median": float(np.median(group["direction_cosine"])),
            "absolute_norm_relative_error_median": float(
                np.median(group["absolute_norm_relative_error"])
            ),
            "vector_error_m_median": float(np.median(group["vector_error_m"])),
        }
    return result, frame


def render_figure(stabilized: pd.DataFrame, recovery: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"carrier-coupled": "#0072B2", "doppler-locked": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)

    left = axes[0]
    for condition in CONDITIONS:
        rows = stabilized[stabilized["condition"] == condition]
        left.scatter(
            rows["truth_delay_chips"], rows["stabilized_delay_chips"],
            s=9, alpha=0.20, edgecolors="none", color=colors[condition],
            label=condition.replace("-", " "),
        )
    low = float(min(stabilized["truth_delay_chips"].min(), stabilized["stabilized_delay_chips"].min()))
    high = float(max(stabilized["truth_delay_chips"].max(), stabilized["stabilized_delay_chips"].max()))
    pad = 0.04 * (high - low)
    left.plot([low - pad, high + pad], [low - pad, high + pad], "k--", lw=1.0)
    left.set(xlim=(low - pad, high + pad), ylim=(low - pad, high + pad))
    left.set_xlabel("Simulator-truth delay (chips)")
    left.set_ylabel("Estimated signed delay (chips)")
    left.set_title("(a) 9-tap delay truth audit")
    left.grid(alpha=0.22)
    left.legend(frameon=False, fontsize=7, loc="upper left")

    right = axes[1]
    for condition in CONDITIONS:
        rows = recovery[recovery["condition"] == condition]
        right.scatter(
            rows["norm_ratio"], rows["direction_cosine"], s=13, alpha=0.48,
            edgecolors="none", color=colors[condition],
            label=condition.replace("-", " "),
        )
    right.axvline(1.0, color="0.25", linestyle="--", linewidth=1.0)
    right.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0)
    right.set_xlabel(r"Recovered norm / true norm")
    right.set_ylabel("Direction cosine")
    right.set_title("(b) Recovered 3-D displacement")
    right.set_xlim(0.55, 1.45)
    right.set_ylim(0.5, 1.02)
    right.grid(alpha=0.22)

    fig.savefig(FIGURE_PNG, dpi=220)
    fig.savefig(
        FIGURE_PDF,
        metadata={"Creator": "audit_cgc_formula_accuracy_v1.py", "CreationDate": None},
    )
    plt.close(fig)


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_RECORD.read_text(encoding="utf-8"))
    data_root = Path(config["output_root"])
    if data_root != Path("/home/ubuntu/hdd_data/cgc_temporal_final_static_v1"):
        raise ValueError("unexpected final-static data root")
    if source["decision"] != "SUPPORTED" or not source["all_preregistered_gates_passed"]:
        raise ValueError("source final-static release is not the frozen supported record")

    raw_path = data_root / "analysis/delay_estimates.csv"
    stabilized_path = data_root / "analysis/stabilized_delay_estimates.csv"
    verify_file(
        raw_path, source["analysis_artifacts"]["delay_estimates_sha256"],
        "frozen raw delay estimates",
    )
    verify_file(
        stabilized_path,
        source["analysis_artifacts"]["stabilized_delay_estimates_sha256"],
        "frozen stabilized delay estimates",
    )

    truth, sampled_truth = load_all_truth(config, data_root)
    raw = merge_estimates(raw_path, "estimated_delay_chips", truth)
    stabilized = merge_estimates(
        stabilized_path, "stabilized_delay_chips", truth
    )
    los_accuracy = los_approximation_summary(config, data_root, sampled_truth)
    recovery, recovery_frame = recovery_summary(config, data_root, stabilized)
    raw_summary = estimator_summary(raw, "estimated_delay_chips")
    stabilized_summary = estimator_summary(stabilized, "stabilized_delay_chips")

    checks = {
        "fixed_startup_los_max_error_at_most_0p002_chip": (
            los_accuracy["maximum_absolute_error_chips"] <= 0.002
        ),
        "stabilized_delay_mae_at_most_0p075_chip": (
            stabilized_summary["absolute"]["mae_chips"] <= 0.075
        ),
        "stabilized_sign_accuracy_at_least_0p90": (
            stabilized_summary["absolute"]["sign_accuracy"] >= 0.90
        ),
        "median_direction_cosine_at_least_0p95": (
            recovery["direction_cosine"]["median"] >= 0.95
        ),
        "median_absolute_norm_error_at_most_0p15": (
            recovery["absolute_norm_relative_error"]["median"] <= 0.15
        ),
    }
    decision = "SUPPORTIVE_DESCRIPTIVE_AUDIT" if all(checks.values()) else "MIXED"
    render_figure(stabilized, recovery_frame)

    result = {
        "schema": "gnss-doppler-lab.cgc-formula-accuracy-audit",
        "schema_version": 1,
        "decision": decision,
        "source_campaign": {
            "name": "cgc-temporal-final-static-v1",
            "data_root": str(data_root),
            "source_record": str(SOURCE_RECORD.relative_to(ROOT)),
            "source_record_sha256": sha256(SOURCE_RECORD),
            "pair_count": len(config["pairs"]),
            "conditions": list(CONDITIONS),
            "hold_start_s": HOLD_START_S,
            "receiver_rf": "25 MHz simulated I/Q processed by modified GNSS-SDR",
        },
        "operational_los_approximation": los_accuracy,
        "signed_delay_accuracy": {
            "raw_one_second": raw_summary,
            "causal_five_bin_median": stabilized_summary,
        },
        "three_dimensional_recovery": recovery,
        "descriptive_checks": checks,
        "artifacts": {
            "figure_png": str(FIGURE_PNG.relative_to(ROOT)),
            "figure_pdf": str(FIGURE_PDF.relative_to(ROOT)),
        },
        "claim_boundary": [
            "Post-hoc descriptive truth audit of the existing frozen final-static campaign.",
            "It verifies the measurement/model chain but is not a new independent detection test.",
            "The five geometries are simulated static receiver--RF cases, not field measurements.",
            "The fixed-startup-LOS comparison includes both first-order range approximation and LOS aging over the 30 s recording.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
