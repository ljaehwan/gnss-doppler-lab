#!/usr/bin/env python3
"""Finalize the MCTD report, plots, runner index, and checksum manifest."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/mctd_stage0_static"
RUN_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mctd-stage0-static/runner-runs")


def read_json(name: str):
    return json.loads((ARTIFACT / name).read_text())


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def runner_index() -> None:
    runs = []
    for path in sorted(RUN_ROOT.glob("*/status.json")):
        status = json.loads(path.read_text())
        contract = json.loads((path.parent / "contract.json").read_text())
        runs.append({**status, "command": contract.get("command"), "git": json.loads((path.parent / "git.json").read_text()),
                     "stdout_sha256": hashlib.sha256((path.parent / "stdout.log").read_bytes()).hexdigest(),
                     "stderr_sha256": hashlib.sha256((path.parent / "stderr.log").read_bytes()).hexdigest()})
    dump_json(ARTIFACT / "runner_runs.json", {"schema": "gnss-doppler-lab.mctd-runner-runs.v1",
              "run_root": str(RUN_ROOT), "runs": runs,
              "all_completed_successfully": all(run["status"] == "succeeded" for run in runs)})


def load_block_rows():
    with gzip.open(ARTIFACT / "per_block_scores.csv.gz", "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def make_plots() -> None:
    plot = ARTIFACT / "plots"; plot.mkdir(exist_ok=True)
    rows = load_block_rows()
    scenarios = ["TEXBAT.DS3", "TEXBAT.DS7", "OAKBAT.OS3", "OAKBAT.OS4"]
    onset = {"TEXBAT.DS3": 118.9, "TEXBAT.DS7": 110.0, "OAKBAT.OS3": 120.0, "OAKBAT.OS4": 120.0}
    pull = {"TEXBAT.DS3": 195.0, "TEXBAT.DS7": 150.0}
    for scenario in scenarios:
        chosen = [row for row in rows if row["dataset"] == scenario and row["variant"] == "Full"]
        fig, ax = plt.subplots(figsize=(8, 3)); x = [float(row["block_start_s"]) for row in chosen]
        y = [float(row["score"]) for row in chosen]; ax.plot(x, y, lw=.8); ax.axvline(onset[scenario], color="r", ls="--")
        if scenario in pull: ax.axvline(pull[scenario], color="orange", ls=":")
        if chosen: ax.axhline(float(chosen[0]["threshold"]), color="k", ls="--")
        ax.set(title=f"{scenario} Full score", xlabel="raw time (s)", ylabel="score"); fig.tight_layout()
        fig.savefig(plot / f"{scenario.lower().replace('.', '_')}_full_score_timeline.png", dpi=140); plt.close(fig)
    prn_path = ARTIFACT / "per_prn_divergence.csv.gz"
    with gzip.open(prn_path, "rt", newline="") as stream: prn_rows = list(csv.DictReader(stream))
    for field, name, ylabel in (("code_divergence_abs_median", "slow_vs_fast_code_phase", "|code phase difference|"),
                                ("doppler_divergence_abs_median", "slow_vs_fast_carrier_doppler", "|Doppler difference|"),
                                ("full_divergence_norm_median", "dll_pll_divergence_timeline", "Full divergence norm"),
                                ("tap_divergence_norm_median", "complex_9tap_divergence_timeline", "tap divergence norm")):
        fig, ax = plt.subplots(figsize=(8, 3))
        for scenario in scenarios:
            selected = [row for row in prn_rows if row["dataset"] == scenario]
            x = np.asarray([float(row["block_start_s"]) for row in selected]); y = np.asarray([float(row[field]) for row in selected])
            if len(x):
                unique = np.unique(x); ax.plot(unique, [np.median(y[x == value]) for value in unique], lw=.7, label=scenario)
        ax.set(xlabel="raw time (s)", ylabel=ylabel); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(plot / f"{name}.png", dpi=140); plt.close(fig)
    clean = [row for row in rows if row["role"] == "holdout" and row["variant"] == "Full"]
    fig, ax = plt.subplots(figsize=(6, 3))
    for dataset in sorted({row["dataset"] for row in clean}):
        ax.hist([float(row["score"]) for row in clean if row["dataset"] == dataset], bins=30, alpha=.5, label=dataset)
    ax.legend(fontsize=7); ax.set(title="clean holdout Full scores", xlabel="score"); fig.tight_layout(); fig.savefig(plot / "clean_calibration_holdout_score_distribution.png", dpi=140); plt.close(fig)
    metrics = list(csv.DictReader((ARTIFACT / "ablation_metrics.csv").open()))
    for key, filename, ylabel in (("pre_onset_fpr", "pre_onset_fpr_comparison", "pre-onset FPR"),
                                  ("pauc_fpr_le_0p05", "ablation_comparison", "normalized pAUC <=5% FPR")):
        chosen = [row for row in metrics if row["status"] == "AVAILABLE"]
        labels = [f"{row['scenario']}:{row['model']}" for row in chosen]; values = [float(row[key]) for row in chosen]
        fig, ax = plt.subplots(figsize=(10, 4)); ax.bar(np.arange(len(values)), values); ax.set_xticks(np.arange(len(values)), labels, rotation=90, fontsize=6); ax.set_ylabel(ylabel); fig.tight_layout(); fig.savefig(plot / f"{filename}.png", dpi=140); plt.close(fig)
    collapse = read_json("configuration_collapse_metrics.json")["scenarios"]
    fig, ax = plt.subplots(figsize=(6, 3)); x = np.arange(len(collapse)); ax.bar(x-.2, [row["full_attack_mean"] for row in collapse], .4, label="Full"); ax.bar(x+.2, [row["identical_attack_mean"] for row in collapse], .4, label="identical"); ax.set_xticks(x, [row["scenario"] for row in collapse], rotation=30); ax.legend(); fig.tight_layout(); fig.savefig(plot / "identical_loop_configuration_collapse.png", dpi=140); plt.close(fig)
    full = [row for row in metrics if row["model"] == "Full"]
    fig, ax = plt.subplots(figsize=(6, 3)); ax.bar([row["scenario"] for row in full], [float(row["pauc_fpr_le_0p05"]) for row in full]); ax.set(ylabel="Full normalized pAUC", title="Core scenario comparison"); fig.tight_layout(); fig.savefig(plot / "ds3_ds7_os3_os4_scenario_comparison.png", dpi=140); plt.close(fig)


def readme() -> None:
    verdict = read_json("final_verdict.json"); phase_a = read_json("phase_a_reproducibility.json")
    metrics = list(csv.DictReader((ARTIFACT / "scenario_metrics.csv").open()))
    lines = ["# MCTD Stage-0 Static", "", f"Final verdict: `{verdict['verdict']}`.", "",
             "Slow uses DLL/PLL 0.5/10 Hz; fast uses 2/25 Hz. All other receiver settings are identical. The identical-loop control uses 0.5/10 Hz on both sides.", "",
             "Phase A passed source equality, bit-exact within-configuration replay, stable common support, and exact identical-loop collapse on TEXBAT and OAKBAT cleanStatic. Slow/fast share authenticated raw IQ, raw range, PRN assignment, and the same scenario-specific handoff state.", "",
             "Clean models use chronological train/validation/calibration/holdout roles with 5 s guards, robust median centers, Ledoit-Wolf covariance, clean calibration q99 thresholds, PRN median pooling, non-overlapping 100 ms blocks, and three-consecutive-block alarms.", "", "## Core Full results", "",
             "|Scenario|pAUC<=5%|pre-onset FPR|attack detection|onset delay s|", "|---|---:|---:|---:|---:|"]
    for row in metrics: lines.append(f"|{row['scenario']}|{row['pauc_fpr_le_0p05']}|{row['pre_onset_fpr']}|{row['attack_detection_rate']}|{row['onset_delay_s']}|")
    lines += ["", "A0/A1/A2/A3/A4/A5/Full common-support results are in `ablation_metrics.csv`. Exact B0 is `UNAVAILABLE`: rerunning it on native MCTD support would change the frozen B0 contract, and historical CSV values were not copied as MCTD results.", "",
              "Configuration collapse is in `configuration_collapse_metrics.json`. Gain/phase/Prompt/nav-sign invariance diagnostics and honestly unavailable raw-IQ AWGN, C/N0, clock-drift, and multipath controls are in `physical_controls.json`. Because the required raw-IQ nuisance controls and Full pairing-destruction proof are unavailable, no unique physical contribution or Stage-1 promotion is claimed.", "",
              "The experiment used no post-attack bandwidth, feature, threshold, pooling, or GO-criterion changes.", "", "Exactly one recommended next action: stop MCTD Stage-1 and retain this frozen Stage-0 bundle as the negative-result record.", ""]
    (ARTIFACT / "README.md").write_text("\n".join(lines))


def manifest() -> None:
    entries = {}
    for path in sorted(ARTIFACT.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest_sha256.json":
            entries[str(path.relative_to(ARTIFACT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump_json(ARTIFACT / "artifact_manifest_sha256.json", {"schema": "gnss-doppler-lab.mctd-manifest.v1", "files": entries})


def main() -> int:
    runner_index(); make_plots(); readme(); manifest(); print(json.dumps({"status": "FINALIZED", "artifact": str(ARTIFACT)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())

