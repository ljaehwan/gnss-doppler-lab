#!/usr/bin/env python3
"""Render CRISP result diagnostics from frozen score exports only."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.crisp import binary_metrics
from gnss_doppler_lab.crisp_data import read_records, scenario_files

ART = ROOT / "artifacts/crisp_stage0_static"
DATA = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b")
SPECS = {
    "TEXBAT.cleanStatic": ("TEXBAT", "texbat_cleanstatic/rep1", None, None),
    "TEXBAT.DS3": ("TEXBAT", "texbat_ds3/rep1", 118.9, 195.0),
    "TEXBAT.DS7": ("TEXBAT", "texbat_ds7/rep1", 110.0, 150.0),
    "OAKBAT.cleanStatic": ("OAKBAT", "oakbat_cleanstatic/rep1", None, None),
    "OAKBAT.OS3": ("OAKBAT", "oakbat_os3/rep1", 120.0, None),
    "OAKBAT.OS4": ("OAKBAT", "oakbat_os4/rep1", 120.0, None),
}
ATTACKS = [name for name, spec in SPECS.items() if spec[2] is not None]


def load_rows(path: Path, zipped: bool = False) -> list[dict[str, str]]:
    opener = gzip.open if zipped else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(ART / "plots" / name, dpi=120, metadata={"Software": "CRISP Stage-0"})
    plt.close(fig)


def alarm_state(rows: list[dict[str, str]], threshold: float) -> np.ndarray:
    result = np.zeros(len(rows), dtype=bool)
    run = 0
    previous = None
    for index, row in enumerate(rows):
        block = int(row["block_id"])
        if previous is None or block != previous + 1:
            run = 0
        run = run + 1 if float(row["score"]) > threshold else 0
        result[index] = run >= 3
        previous = block
    return result


def main() -> None:
    blocks = load_rows(ART / "per_block_scores.csv.gz", zipped=True)
    per_prn = load_rows(ART / "per_prn_metrics.csv")
    scenario_metrics = load_rows(ART / "scenario_metrics.csv")
    ablations = load_rows(ART / "ablation_metrics.csv")
    thresholds = json.loads((ART / "thresholds.json").read_text())
    controls = json.loads((ART / "control_metrics.json").read_text())
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in blocks:
        grouped.setdefault((row["scenario"], row["method"]), []).append(row)

    supplemental = {}
    for scenario in ATTACKS:
        dataset, _directory, onset, pull_off = SPECS[scenario]
        rows = grouped[(scenario, "Full")]
        times = np.asarray([float(row["timestamp_s"]) for row in rows])
        scores = np.asarray([float(row["score"]) for row in rows])
        valid_counts = np.asarray([int(row["valid_prn_count"]) for row in rows])
        alarm = alarm_state(rows, thresholds[dataset]["methods"]["Full"]["q99"])
        after = times >= onset
        pull = times >= pull_off if pull_off is not None else np.zeros(len(times), dtype=bool)
        supplemental[scenario] = {
            "persistent_alarm_ratio_after_signal_onset": float(np.mean(alarm[after])),
            "pull_off_first_alarm_delay_s": float(times[pull & alarm][0] - pull_off) if np.any(pull & alarm) else None,
            "pull_off_s": pull_off,
            "reset_gap_count": sum(int(row["reset_count"]) for row in per_prn if row["scenario"] == scenario),
            "signal_onset_s": onset,
            "transition_detection_rate": "UNAVAILABLE_WINDOW_NOT_PREREGISTERED",
            "established_detection_rate": "UNAVAILABLE_WINDOW_NOT_PREREGISTERED",
            "valid_prn_count_min": int(np.min(valid_counts)),
            "valid_prn_count_median": float(np.median(valid_counts)),
            "valid_prn_count_max": int(np.max(valid_counts)),
        }
    (ART / "supplemental_metrics.json").write_text(json.dumps(supplemental, indent=2, sort_keys=True, allow_nan=False) + "\n")

    pooled = {}
    for method in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "Full"):
        labels, scores = [], []
        for scenario in ("OAKBAT.OS3", "OAKBAT.OS4"):
            onset = SPECS[scenario][2]
            rows = grouped[(scenario, method)]
            labels.extend(float(row["timestamp_s"]) >= onset for row in rows)
            scores.extend(float(row["score"]) for row in rows)
        pooled[method] = binary_metrics(np.asarray(labels, dtype=np.int8), np.asarray(scores))
    (ART / "oak_os3_os4_pooled_metrics.json").write_text(json.dumps(pooled, indent=2, sort_keys=True, allow_nan=False) + "\n")

    clean_scores = np.concatenate([
        np.asarray([float(row["score"]) for row in grouped[(name, "Full")] if float(row["timestamp_s"]) >= 320.0])
        for name in ("TEXBAT.cleanStatic", "OAKBAT.cleanStatic")
    ])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(clean_scores, bins=80, density=True)
    ax.set(xlabel="Full score", ylabel="density", title="Clean holdout score distribution")
    save(fig, "clean_score_distribution.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    for ax, scenario in zip(axes.flat, ATTACKS, strict=True):
        rows = grouped[(scenario, "Full")]
        stride = max(1, len(rows) // 2500)
        ax.plot([float(row["timestamp_s"]) for row in rows[::stride]], [float(row["score"]) for row in rows[::stride]], lw=.6)
        ax.axvline(SPECS[scenario][2], color="k", ls="--", lw=.8)
        ax.set_title(scenario)
    save(fig, "scenario_score_timeline.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, scenario in zip(axes.flat, ATTACKS, strict=True):
        dataset = SPECS[scenario][0]
        threshold = thresholds[dataset]["methods"]["Full"]["q99"]
        rows = grouped[(scenario, "Full")]
        times = np.asarray([float(row["timestamp_s"]) for row in rows])
        scores = np.asarray([float(row["score"]) for row in rows])
        alarms = alarm_state(rows, threshold)
        stride = max(1, len(rows) // 2500)
        ax.plot(times[::stride], scores[::stride], lw=.6)
        ax.axhline(threshold, color="tab:red", ls="--")
        ax.scatter(times[alarms], scores[alarms], s=3, color="tab:red")
        ax.set_title(scenario)
    save(fig, "threshold_alarm_timeline.png")

    scenarios = list(SPECS)
    prns = sorted({int(row["prn"]) for row in per_prn if row["prn"] != "-1"})
    heat = np.full((len(scenarios), len(prns)), np.nan)
    for row in per_prn:
        if row["prn"] != "-1" and row["median_full_score"]:
            heat[scenarios.index(row["scenario"]), prns.index(int(row["prn"]))] = float(row["median_full_score"])
    fig, ax = plt.subplots(figsize=(11, 5))
    image = ax.imshow(heat, aspect="auto")
    ax.set(xticks=np.arange(len(prns)), xticklabels=prns, yticks=np.arange(len(scenarios)), yticklabels=scenarios, xlabel="PRN", title="Median Full score by scenario and PRN")
    fig.colorbar(image, ax=ax)
    save(fig, "per_prn_heatmap.png")

    x, y = [], []
    for scenario in ATTACKS:
        full = {row["block_id"]: float(row["score"]) for row in grouped[(scenario, "Full")]}
        mag = {row["block_id"]: float(row["score"]) for row in grouped[(scenario, "A1")]}
        keys = sorted(full.keys() & mag.keys())[::10]
        x.extend(mag[key] for key in keys)
        y.extend(full[key] for key in keys)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(x, y, s=3, alpha=.25)
    ax.set(xlabel="A1 magnitude-change score", ylabel="Full projective score", title="Projective versus magnitude score")
    save(fig, "projective_vs_magnitude.png")

    cn0_by_scenario_prn = {}
    for scenario, (_dataset, directory, _onset, _pull) in SPECS.items():
        for path in scenario_files(DATA / directory):
            _, records = read_records(path)
            physical = records[(records["valid_tracking"] == 1) & (records["valid_lock"] == 1) & (records["cn0_db_hz"] >= 28.0)]
            if len(physical):
                cn0_by_scenario_prn[(scenario, int(np.median(physical["prn"])))] = float(np.median(physical["cn0_db_hz"]))
    x, y = [], []
    for row in per_prn:
        key = (row["scenario"], int(row["prn"]))
        if row["median_full_score"] and key in cn0_by_scenario_prn:
            x.append(cn0_by_scenario_prn[key]); y.append(float(row["median_full_score"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(x, y)
    ax.set(xlabel="Median supported C/N0 (dB-Hz)", ylabel="Median Full PRN score", title="Projective score versus C/N0")
    save(fig, "projective_vs_cn0.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, scenario in zip(axes.flat, ATTACKS, strict=True):
        rows = grouped[(scenario, "Full")]
        stride = max(1, len(rows) // 2500)
        ax.plot([float(row["timestamp_s"]) for row in rows[::stride]], [int(row["valid_prn_count"]) for row in rows[::stride]], lw=.7)
        ax.axhline(4, color="tab:red", ls="--")
        ax.set_title(scenario)
    save(fig, "valid_prn_coverage.png")

    inv = controls["algebraic_invariance"]["tests"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = list(inv)
    ax.bar(np.arange(len(names)), [inv[name]["max_abs_error"] for name in names])
    ax.axhline(1e-10, color="tab:red", ls="--", label="tolerance")
    ax.set_yscale("log"); ax.set_xticks(np.arange(len(names)), names, rotation=30, ha="right"); ax.legend()
    ax.set_title("Algebraic invariance errors")
    save(fig, "invariance_control_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for index, scenario in enumerate(ATTACKS):
        rows = {row["method"]: float(row["pauc_fpr_le_0_05"]) for row in ablations if row["scenario"] == scenario}
        ax.plot(list(range(8)), [rows[name] for name in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "Full")], marker="o", label=scenario)
    ax.set_xticks(range(8), ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "Full")); ax.set_ylabel("normalized pAUC <=5% FPR"); ax.legend()
    save(fig, "ablation_pauc.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for row in scenario_metrics:
        if row["scenario"] in ATTACKS:
            ax.scatter(float(row["q99_fpr"]), float(row["first_alarm_delay_s"]), label=row["scenario"])
    ax.axvline(.05, color="tab:red", ls="--"); ax.axhline(5, color="tab:red", ls="--")
    ax.set(xlabel="pre-onset FPR", ylabel="first-alarm delay (s)", title="Delay-FPR comparison"); ax.legend()
    save(fig, "delay_fpr_comparison.png")


if __name__ == "__main__":
    main()
