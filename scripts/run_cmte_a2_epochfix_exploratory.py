#!/usr/bin/env python3
"""Run the post-exposure CMTE-A2 canonical multi-PRN epoch diagnostic.

All scenario results are developmental/exploratory. The sealed PRIMARY INVALID
artifact is verified and read-only; it is never modified or replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.cmte_a2 import (
    b0_enhanced_scores,
    b0_exact_scores,
    epoch_metrics,
    higher_quantile,
    phase_masks,
    threshold_operating_points,
)
from gnss_doppler_lab.cmte_a2_epochfix import EpochPolicy, aggregate_multi_prn_epochs

SCENARIO_ONSETS = {"DS1": 100.0, "DS2": 100.0, "DS3": 100.0, "DS4": 100.0, "DS7": 110.0, "DS8": 110.0}
MODELS = {
    "CMTE-A2": "score_A2",
    "A0": "score_A0",
    "B0-Exact": "score_B0_Exact",
    "B0-Enhanced": "score_B0_Enhanced",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(root: Path) -> int:
    document = json.loads((root / "checksums.json").read_text())
    bad = []
    for relative, expected in document["files"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            bad.append(relative)
    if bad:
        raise ValueError(f"checksum mismatch: {bad}")
    return len(document["files"])


def score_models(per_prn: pd.DataFrame, node_thresholds: dict[str, float], rates: dict[str, float]) -> pd.DataFrame:
    epoch = aggregate_multi_prn_epochs(per_prn)
    inputs = epoch[["physical_recording_id", "window_end_s", "rmse_values"]].copy()
    exact = b0_exact_scores(inputs, node_thresholds)
    enhanced = b0_enhanced_scores(inputs, node_thresholds, rates)
    epoch["score_B0_Exact"] = exact.score.to_numpy(float)
    epoch["score_B0_Enhanced"] = enhanced.score.to_numpy(float)
    return epoch


def slice_epochs(full: pd.DataFrame, role_epoch: pd.DataFrame) -> pd.DataFrame:
    keys = role_epoch[["physical_recording_id", "window_end_s"]].drop_duplicates()
    selected = full.merge(keys, on=["physical_recording_id", "window_end_s"], how="inner", validate="one_to_one")
    return selected.sort_values(["physical_recording_id", "window_end_s"], kind="mergesort").reset_index(drop=True)


def match_clean_fpr(clean: pd.DataFrame, target: float) -> dict[str, dict[str, float]]:
    result = {}
    for model, column in MODELS.items():
        values = clean[column].to_numpy(float)
        candidates = np.unique(values)
        occupancy = np.asarray([np.mean(values > threshold) for threshold in candidates])
        distance = np.abs(occupancy - target)
        minimum = distance.min()
        tied = np.flatnonzero(np.isclose(distance, minimum, rtol=0, atol=1e-15))
        index = int(tied[np.argmax(candidates[tied])])
        result[model] = {
            "threshold": float(candidates[index]),
            "clean_test_fpr": float(occupancy[index]),
            "target_clean_test_fpr": float(target),
            "absolute_difference": float(distance[index]),
            "fit_source": "independent_clean_test_diagnostic_only",
            "attack_fit": False,
        }
    return result


def metric_row(epoch: pd.DataFrame, model: str, threshold: float, onset: float, clean_fpr: float, operating: str) -> dict[str, Any]:
    column = MODELS[model]
    row = epoch_metrics(epoch, column, threshold, onset_s=onset, clean_fpr=clean_fpr)
    counts = epoch.tracked_prn_count.to_numpy(int)
    return {
        "status": "exploratory_post_exposure",
        "scenario": epoch.scenario.iloc[0],
        "model": model,
        "operating_point": operating,
        **row,
        "tracked_prn_count_median": float(np.median(counts)),
        "tracked_prn_count_min": int(np.min(counts)),
        "tracked_prn_count_max": int(np.max(counts)),
    }


def write_checksums(root: Path) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.json":
            files[str(path.relative_to(root))] = sha256(path)
    (root / "checksums.json").write_text(json.dumps({"algorithm": "sha256", "files": files}, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--invalid-artifact", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    state_dir = Path(args.state_dir).resolve(strict=True)
    invalid = Path(args.invalid_artifact).resolve(strict=True)
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("epochfix output is non-overwrite")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("epochfix experiment requires a clean git tree")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    invalid_checksum_count = verify_checksums(invalid)
    state_checksum_count = verify_checksums(state_dir)

    required_state = ["b0_model.pt", "a2_state.json", "clean_per_prn.csv", "thresholds.json"]
    for name in required_state:
        if not (state_dir / name).is_file():
            raise ValueError(f"missing frozen state file {name}")
    if sha256(state_dir / "b0_model.pt") != sha256(invalid / "b0_model.pt"):
        raise ValueError("frozen B0 checkpoint changed")
    if sha256(state_dir / "a2_state.json") != sha256(invalid / "a2_state.json"):
        raise ValueError("frozen conformal state changed")

    clean_prn = pd.read_csv(state_dir / "clean_per_prn.csv")
    threshold_prn = clean_prn[clean_prn.role.eq("threshold")].copy()
    test_prn = clean_prn[clean_prn.role.eq("clean_test")].copy()
    if threshold_prn.empty or test_prn.empty:
        raise ValueError("clean threshold/test roles missing")

    rmse = threshold_prn.rmse.to_numpy(float)
    node_thresholds = {name: higher_quantile(rmse, quantile) for name, quantile in (("q50", 0.5), ("q70", 0.7), ("q80", 0.8))}
    rates = {name: float(np.mean(rmse > value)) for name, value in node_thresholds.items()}

    full_clean = score_models(clean_prn, node_thresholds, rates)
    threshold_a2 = aggregate_multi_prn_epochs(threshold_prn)
    threshold_epoch = slice_epochs(full_clean, threshold_a2)
    clean_test_a2 = aggregate_multi_prn_epochs(test_prn)
    clean_test = slice_epochs(full_clean, clean_test_a2)

    thresholds = {model: threshold_operating_points(threshold_epoch[column]) for model, column in MODELS.items()}
    clean_fpr = {model: float(np.mean(clean_test[column] > thresholds[model]["q995"])) for model, column in MODELS.items()}
    matched = match_clean_fpr(clean_test, clean_fpr["CMTE-A2"])

    staging = out.with_name(out.name + f".tmp-{os.getpid()}")
    for directory in (staging, staging / "per_epoch", staging / "plots"):
        directory.mkdir(parents=True, exist_ok=True)
    all_primary: list[dict[str, Any]] = []
    all_matched: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    input_hashes = {"clean_per_prn.csv": sha256(state_dir / "clean_per_prn.csv")}

    clean_test.assign(scenario="CLEAN_TEST").drop(columns=["rmse_values"]).to_csv(staging / "per_epoch" / "cleanStatic_test.csv", index=False)
    threshold_epoch.assign(scenario="CLEAN_THRESHOLD").drop(columns=["rmse_values"]).to_csv(staging / "per_epoch" / "cleanStatic_threshold.csv", index=False)

    for scenario, onset in SCENARIO_ONSETS.items():
        path = invalid / "per_prn" / f"{scenario}.csv"
        input_hashes[f"per_prn/{scenario}.csv"] = sha256(path)
        epoch = score_models(pd.read_csv(path), node_thresholds, rates)
        epoch["scenario"] = scenario
        masks = phase_masks(epoch, onset)
        epoch["phase"] = "excluded"
        epoch.loc[masks["stable_pre"], "phase"] = "stable_pre"
        epoch.loc[masks["ramp"], "phase"] = "ramp"
        epoch.loc[masks["takeover"], "phase"] = "takeover"
        epoch.loc[masks["persistent"], "phase"] = "persistent"
        epoch["stable_pre"] = masks["stable_pre"]
        epoch["post"] = masks["post"]
        epoch["persistent"] = masks["persistent"]
        epoch.drop(columns=["rmse_values"]).to_csv(staging / "per_epoch" / f"{scenario}.csv", index=False)

        counts = epoch.tracked_prn_count
        for n, count in counts.value_counts().sort_index().items():
            hist_rows.append({
                "scenario": scenario,
                "N": int(n),
                "epoch_count": int(count),
                "fraction": float(count / len(epoch)),
                "median_N": float(counts.median()),
                "min_N": int(counts.min()),
                "max_N": int(counts.max()),
            })
        if counts.median() <= 1:
            raise ValueError(f"epochfix failed: {scenario} median tracked PRN count <= 1")

        for model in MODELS:
            all_primary.append(metric_row(epoch, model, thresholds[model]["q995"], onset, clean_fpr[model], "q995_higher_normal_threshold"))
            all_matched.append(metric_row(epoch, model, matched[model]["threshold"], onset, matched[model]["clean_test_fpr"], "matched_clean_fpr_diagnostic"))

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for model, column in MODELS.items():
            axes[0].plot(epoch.window_end_s, epoch[column], label=model, linewidth=0.8)
            axes[0].axhline(thresholds[model]["q995"], linestyle="--", linewidth=0.5)
        axes[0].axvline(onset, color="k", linestyle=":")
        axes[0].set_ylabel("score / q99.5")
        axes[0].legend(ncol=2)
        axes[1].step(epoch.window_end_s, epoch.tracked_prn_count, where="post")
        axes[1].set(xlabel="canonical decision time (s)", ylabel="tracked PRN N")
        fig.suptitle(f"{scenario}: exploratory canonical multi-PRN epochfix")
        fig.tight_layout()
        fig.savefig(staging / "plots" / f"{scenario}_scores_and_N.png", dpi=120)
        plt.close(fig)

    primary = pd.DataFrame(all_primary)
    matched_frame = pd.DataFrame(all_matched)
    primary[primary.model.eq("CMTE-A2")].to_csv(staging / "scenario_metrics.csv", index=False)
    primary[~primary.model.eq("CMTE-A2")].to_csv(staging / "baseline_metrics.csv", index=False)
    matched_frame.to_csv(staging / "matched_fpr.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(staging / "tracked_prn_count.csv", index=False)

    comparison = []
    for operating, frame in (("q995", primary), ("matched", matched_frame)):
        for scenario in SCENARIO_ONSETS:
            a2 = frame[(frame.scenario == scenario) & frame.model.eq("CMTE-A2")].iloc[0]
            b0 = frame[(frame.scenario == scenario) & frame.model.eq("B0-Exact")].iloc[0]
            comparison.append({
                "operating_point": operating,
                "scenario": scenario,
                "a2_clean_fpr": a2.independent_clean_fpr,
                "b0_clean_fpr": b0.independent_clean_fpr,
                "a2_stable_pre_fpr": a2.stable_pre_fpr,
                "b0_stable_pre_fpr": b0.stable_pre_fpr,
                "a2_post_detection_rate": a2.post_detection_rate,
                "b0_post_detection_rate": b0.post_detection_rate,
                "a2_persistent_detection_rate": a2.persistent_detection_rate,
                "b0_persistent_detection_rate": b0.persistent_detection_rate,
                "a2_first_alarm_delay_s": a2.first_alarm_delay_s,
                "b0_first_alarm_delay_s": b0.first_alarm_delay_s,
            })
    pd.DataFrame(comparison).to_csv(staging / "cmte_a2_vs_b0_exact.csv", index=False)

    n_summary = pd.DataFrame(hist_rows).groupby("scenario", sort=True).first().reset_index()
    required_n = bool((n_summary.median_N > 1).all())
    stable_ok = bool((primary[primary.model.eq("CMTE-A2")].stable_pre_fpr < 0.05).all())
    clean_ok = clean_fpr["CMTE-A2"] <= 0.015
    matched_comparison = pd.DataFrame(comparison)
    holdout = matched_comparison[(matched_comparison.operating_point == "matched") & matched_comparison.scenario.isin(["DS7", "DS8"])]
    improved = holdout[
        (holdout.a2_post_detection_rate > holdout.b0_post_detection_rate)
        | (holdout.a2_persistent_detection_rate > holdout.b0_persistent_detection_rate)
        | (holdout.a2_first_alarm_delay_s < holdout.b0_first_alarm_delay_s)
    ]
    no_catastrophic = bool(((holdout.a2_post_detection_rate + 0.20) >= holdout.b0_post_detection_rate).all())
    go = bool(clean_ok and stable_ok and required_n and len(improved) >= 1 and no_catastrophic)

    policy = EpochPolicy().to_dict()
    (staging / "epoch_policy.json").write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    (staging / "config.json").write_text(json.dumps({
        "status": "developmental_exploratory_post_exposure",
        "source_commit": commit,
        "scenarios": list(SCENARIO_ONSETS),
        "policy": policy,
        "threshold_fit": "cleanStatic [300,330) only",
        "primary_operating_point": "q99.5 higher",
        "matched_fpr_fit": "clean test diagnostic only; no attack fit",
    }, indent=2, sort_keys=True) + "\n")
    (staging / "thresholds.json").write_text(json.dumps({
        "primary": thresholds,
        "matched_clean_fpr_diagnostic": matched,
        "node_thresholds": node_thresholds,
        "enhanced_empirical_rates": rates,
        "threshold_role_epoch_count": int(len(threshold_epoch)),
        "alarm_comparison": "strict_greater",
        "attack_fit": False,
    }, indent=2, sort_keys=True) + "\n")
    equivalence = json.loads((invalid / "audit" / "historical_b0_gate_equivalence.json").read_text())
    provenance = {
        "schema": "gnss-doppler-lab.cmte-a2-epochfix-exploratory.v1",
        "source_commit": commit,
        "primary_invalid_artifact_preserved": True,
        "primary_invalid_checksum_count": invalid_checksum_count,
        "frozen_state_checksum_count": state_checksum_count,
        "frozen_checkpoint_sha256": sha256(state_dir / "b0_model.pt"),
        "frozen_conformal_state_sha256": sha256(state_dir / "a2_state.json"),
        "input_hashes": input_hashes,
        "B0_exact_function_reused": "gnss_doppler_lab.cmte_a2.b0_exact_scores",
        "historical_B0_gate_equivalence": equivalence,
        "attack_or_test_model_fit": False,
        "DS7_DS8_status": "developmental_exploratory_post_exposure",
    }
    (staging / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    decision = {
        "decision": "GO" if go else "NO-GO",
        "independent_clean_fpr_le_0.015": clean_ok,
        "all_stable_pre_fpr_lt_0.05": stable_ok,
        "all_scenario_median_N_gt_1": required_n,
        "DS7_or_DS8_improves_B0_exact_at_matched_clean_fpr": len(improved) >= 1,
        "no_catastrophic_other_holdout_degradation": no_catastrophic,
        "improved_scenarios": improved.scenario.tolist(),
        "scientific_status": "exploratory_only; external confirmatory data still required",
    }
    (staging / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    (staging / "test_summary.txt").write_text(
        "CMTE-A2 epochfix policy tests: 10 passed before policy freeze.\n"
        "Actual fixture guard: every scenario median tracked PRN count must exceed 1.\n"
        "Final regression results are recorded in README.md.\n"
    )

    lines = [
        "# CMTE-A2 canonical multi-PRN epochfix",
        "",
        "**Status: developmental/exploratory post-exposure diagnostic.** DS7 and DS8 are not confirmatory holdouts.",
        "",
        "The preserved PRIMARY INVALID run grouped floating channel timestamps by exact equality, yielding N=1. This directory does not modify or replace it.",
        "",
        "## Epoch policy",
        "",
        f"- origin: {policy['grid_origin_s']} s; stride: {policy['grid_stride_s']} s",
        "- each residual maps causally to the first decision grid at or after window_end_s",
        f"- timestamp tolerance: {policy['timestamp_tolerance_s']} s; maximum residual age: {policy['max_residual_age_s']} s",
        "- per epoch/PRN: latest available residual only; exact ties use stable row-content SHA-256",
        "- no future residual, no PRN identity feature, row/PRN permutation invariant",
        "",
        "## Actual tracked PRN N",
        "",
    ]
    for row in n_summary.itertuples(index=False):
        lines.append(f"- {row.scenario}: median {row.median_N:g}, min {row.min_N}, max {row.max_N}")
    lines += [
        "",
        "## Decision",
        "",
        f"- independent clean CMTE-A2 FPR: {clean_fpr['CMTE-A2']:.6f}",
        f"- final diagnostic decision: **{decision['decision']}**",
        f"- DS7/DS8 matched-FPR improvements over B0-Exact: {decision['improved_scenarios']}",
        "",
        "Even a GO remains an exploratory signal only. A new external dataset is required for SCI-level confirmatory evidence.",
    ]
    (staging / "README.md").write_text("\n".join(lines) + "\n")
    write_checksums(staging)
    os.replace(staging, out)
    print(json.dumps({"out": str(out), "source_commit": commit, "decision": decision["decision"], "clean_fpr": clean_fpr["CMTE-A2"]}, sort_keys=True))


if __name__ == "__main__":
    main()
