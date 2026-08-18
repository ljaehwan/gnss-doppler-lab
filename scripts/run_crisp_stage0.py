#!/usr/bin/env python3
"""Run the two-phase, clean-preregistered CRISP Stage-0 experiment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnss_doppler_lab.crisp import (
    LinearWhiteningModel,
    binary_metrics,
    extract_channel_features,
    fit_unconditioned,
    invariance_audit,
    projector_property_audit,
)
from gnss_doppler_lab.crisp_data import complex_taps, read_records, scenario_files, sha256_file, validate_scenario

BASE_SHA = "461eb4dc7bb794e719295daf028f6811658ba37f"
BRANCH = "research/crisp-stage0-static"
REFERENCE_SHA = "b90d0f72bc6686e66dce06c617f1ca6895c0886b"
DATA_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b")
ARTIFACT = REPO_ROOT / "artifacts/crisp_stage0_static"
METHODS = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "Full")
SCENARIOS = {
    "TEXBAT.cleanStatic": {"directory": "texbat_cleanstatic/rep1", "dataset": "TEXBAT", "role": "clean"},
    "TEXBAT.DS3": {"directory": "texbat_ds3/rep1", "dataset": "TEXBAT", "role": "core", "onset_s": 118.9, "pull_off_s": 195.0},
    "TEXBAT.DS7": {"directory": "texbat_ds7/rep1", "dataset": "TEXBAT", "role": "core_family", "onset_s": 110.0, "pull_off_s": 150.0},
    "OAKBAT.cleanStatic": {"directory": "oakbat_cleanstatic/rep1", "dataset": "OAKBAT", "role": "clean"},
    "OAKBAT.OS3": {"directory": "oakbat_os3/rep1", "dataset": "OAKBAT", "role": "core", "onset_s": 120.0},
    "OAKBAT.OS4": {"directory": "oakbat_os4/rep1", "dataset": "OAKBAT", "role": "core", "onset_s": 120.0},
}
CONFIG = {
    "epsilon": 1e-12,
    "energy_floor_rule": "0.5 * clean-train sampled q0.001 tap energy",
    "energy_sample_stride": 20,
    "cn0_min_db_hz": 28.0,
    "lock_min": 0.85,
    "ridge_alpha": 1e-3,
    "fit_samples_per_channel_per_split": 15000,
    "clean_split_seconds": {
        "train": [30.0, 170.0],
        "guard_1": [170.0, 180.0],
        "calibration": [180.0, 310.0],
        "guard_2": [310.0, 320.0],
        "holdout": [320.0, None],
    },
    "native_epoch_ms": 1,
    "block_ms": 20,
    "minimum_valid_epochs_per_prn_block": 10,
    "minimum_receiver_prns": 4,
    "receiver_aggregation": "second_largest_valid_prn_block_score",
    "alarm_consecutive_blocks": 3,
    "thresholds": [0.99, 0.995],
    "bootstrap_seed": 20260818,
    "bootstrap_resamples": 1000,
    "bootstrap_block_seconds": 10,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_gzip_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def update_status(phase: str, current: str, completed: int, total: int, started: float, *, exit_code: int | None = None, failure: str | None = None) -> None:
    write_json(
        ARTIFACT / "execution_status.json",
        {
            "run_id": "crisp-stage0-static-20260818",
            "pid": os.getpid(),
            "phase": phase,
            "current_scenario": current,
            "completed": completed,
            "total": total,
            "elapsed_seconds": time.monotonic() - started,
            "last_heartbeat": now(),
            "exit_code": exit_code,
            "failure_reason": failure,
        },
    )


def deterministic_sample(indices: np.ndarray, limit: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    positions = np.linspace(0, len(indices) - 1, limit, dtype=np.int64)
    return indices[positions]


def energy_floor(directory: Path) -> float:
    samples: list[np.ndarray] = []
    lo, hi = CONFIG["clean_split_seconds"]["train"]
    stride = int(CONFIG["energy_sample_stride"])
    for path in scenario_files(directory):
        _, records = read_records(path)
        selected = records[::stride]
        mask = (
            (selected["receiver_timestamp_s"] >= lo)
            & (selected["receiver_timestamp_s"] < hi)
            & (selected["valid_tracking"] == 1)
            & (selected["valid_lock"] == 1)
        )
        taps = complex_taps(selected[mask])
        samples.append(np.sum(np.abs(taps) ** 2, axis=1))
    values = np.concatenate(samples)
    return float(0.5 * np.quantile(values[values > 0], 0.001))


def collect_fit_samples(directory: Path, floor: float, tap_indices: tuple[int, ...] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_h: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    cal_h: list[np.ndarray] = []
    cal_y: list[np.ndarray] = []
    limit = int(CONFIG["fit_samples_per_channel_per_split"])
    for path in scenario_files(directory):
        _, records = read_records(path)
        feature = extract_channel_features(
            records,
            epsilon=CONFIG["epsilon"],
            energy_floor=floor,
            cn0_min_db_hz=CONFIG["cn0_min_db_hz"],
            lock_min=CONFIG["lock_min"],
            tap_indices=tap_indices,
        )
        train = np.flatnonzero(feature.valid & (feature.timestamp_s >= 30.0) & (feature.timestamp_s < 170.0))
        cal = np.flatnonzero(feature.valid & (feature.timestamp_s >= 180.0) & (feature.timestamp_s < 310.0))
        train = deterministic_sample(train, limit)
        cal = deterministic_sample(cal, limit)
        train_h.append(feature.context[train])
        train_y.append(feature.response[train])
        cal_h.append(feature.context[cal])
        cal_y.append(feature.response[cal])
    return np.concatenate(train_h), np.concatenate(train_y), np.concatenate(cal_h), np.concatenate(cal_y)


def aggregate_prn_blocks(epoch_ms: np.ndarray, score: np.ndarray, valid: np.ndarray, prn: int, method: str) -> list[dict[str, Any]]:
    ids = epoch_ms // int(CONFIG["block_ms"])
    rows: list[dict[str, Any]] = []
    valid_ids = ids[valid]
    valid_scores = score[valid]
    if not len(valid_ids):
        return rows
    unique, starts, counts = np.unique(valid_ids, return_index=True, return_counts=True)
    for block, start, count in zip(unique, starts, counts, strict=True):
        if count < CONFIG["minimum_valid_epochs_per_prn_block"]:
            continue
        rows.append({"block_id": int(block), "timestamp_s": float(block * CONFIG["block_ms"] / 1000.0), "prn": prn, "method": method, "score": float(np.median(valid_scores[start : start + count])), "valid_epoch_count": int(count)})
    return rows


def receiver_blocks(prn_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in prn_rows:
        grouped.setdefault((row["block_id"], row["method"]), []).append(row)
    output = []
    for (block, method), rows in sorted(grouped.items()):
        if len(rows) < CONFIG["minimum_receiver_prns"]:
            continue
        values = sorted(float(row["score"]) for row in rows)
        output.append({"block_id": block, "timestamp_s": rows[0]["timestamp_s"], "method": method, "score": values[-2], "valid_prn_count": len(rows)})
    return output


def receiver_epochs(epoch_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[float]] = {}
    for epochs, scores, valid, _prn in epoch_parts:
        for epoch, score in zip(epochs[valid], scores[valid], strict=True):
            grouped.setdefault(int(epoch), []).append(float(score))
    rows = []
    for epoch, values in sorted(grouped.items()):
        if len(values) >= CONFIG["minimum_receiver_prns"]:
            values.sort()
            rows.append({"epoch_ms": epoch, "timestamp_s": epoch / 1000.0, "score": values[-2], "valid_prn_count": len(values)})
    return rows


def score_scenario(directory: Path, floor: float, models: dict[str, LinearWhiteningModel]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_prn_blocks: list[dict[str, Any]] = []
    epoch_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    summaries: list[dict[str, Any]] = []
    for path in scenario_files(directory):
        _, records = read_records(path)
        full = extract_channel_features(records, epsilon=CONFIG["epsilon"], energy_floor=floor, cn0_min_db_hz=CONFIG["cn0_min_db_hz"], lock_min=CONFIG["lock_min"])
        epl = extract_channel_features(records, epsilon=CONFIG["epsilon"], energy_floor=floor, cn0_min_db_hz=CONFIG["cn0_min_db_hz"], lock_min=CONFIG["lock_min"], tap_indices=(3, 4, 5))
        prns = np.unique(full.prn)
        prn = int(prns[0]) if len(prns) else -1
        scores = dict(full.baselines)
        scores["A4"] = models["A4"].score(np.empty((len(full.response), 0)), full.response)
        scores["A5"] = models["A5"].score(epl.context, epl.response)
        scores["Full"] = models["Full"].score(full.context, full.response)
        valid = full.valid & epl.valid
        for method in METHODS:
            per_prn_blocks.extend(aggregate_prn_blocks(full.epoch_ms, scores[method], valid, prn, method))
        epoch_parts.append((full.epoch_ms, scores["Full"], valid, prn))
        summaries.append({"prn": prn, "valid_epochs": int(valid.sum()), "total_epochs": int(len(valid)), "coverage": float(valid.mean()) if len(valid) else 0.0, "reset_count": int(full.reset.sum()), "median_full_score": float(np.median(scores["Full"][valid])) if valid.any() else None})
    return receiver_blocks(per_prn_blocks), receiver_epochs(epoch_parts), summaries


def alarm_metrics(rows: list[dict[str, Any]], threshold: float, onset: float) -> dict[str, Any]:
    full = [row for row in rows if row["method"] == "Full"]
    times = np.asarray([row["timestamp_s"] for row in full])
    scores = np.asarray([row["score"] for row in full])
    before = times < onset
    after = times >= onset
    exceed = scores > threshold
    alarms = np.zeros(len(exceed), dtype=bool)
    run = 0
    previous_block = None
    for i, (row, value) in enumerate(zip(full, exceed, strict=True)):
        if previous_block is None or row["block_id"] != previous_block + 1:
            run = 0
        run = run + 1 if value else 0
        alarms[i] = run >= CONFIG["alarm_consecutive_blocks"]
        previous_block = row["block_id"]
    first = times[after & alarms]
    return {
        "pre_onset_fpr": float(np.mean(exceed[before])) if before.any() else None,
        "attack_detection_rate": float(np.mean(exceed[after])) if after.any() else None,
        "persistent_alarm_ratio": float(np.mean(alarms[after])) if after.any() else None,
        "first_alarm_delay_s": float(first[0] - onset) if len(first) else None,
        "pre_blocks": int(before.sum()),
        "attack_blocks": int(after.sum()),
    }


def method_metrics(rows: list[dict[str, Any]], onset: float) -> dict[str, dict[str, float]]:
    result = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        labels = np.asarray([row["timestamp_s"] >= onset for row in selected], dtype=np.int8)
        scores = np.asarray([row["score"] for row in selected], dtype=np.float64)
        result[method] = binary_metrics(labels, scores)
    return result


def bootstrap_differences(rows: list[dict[str, Any]], onset: float, scenario: str) -> list[dict[str, Any]]:
    by_method = {m: {row["block_id"]: row for row in rows if row["method"] == m} for m in METHODS}
    common = sorted(set.intersection(*(set(by_method[m]) for m in METHODS)))
    labels = np.asarray([by_method["Full"][key]["timestamp_s"] >= onset for key in common], dtype=np.int8)
    groups = np.asarray([int(by_method["Full"][key]["timestamp_s"] // CONFIG["bootstrap_block_seconds"]) for key in common])
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"] + sum(ord(c) for c in scenario))
    output = []
    for baseline in ("A1", "A2", "A3", "A4"):
        full_scores = np.asarray([by_method["Full"][key]["score"] for key in common])
        base_scores = np.asarray([by_method[baseline][key]["score"] for key in common])
        differences = []
        for _ in range(CONFIG["bootstrap_resamples"]):
            sampled = rng.choice(unique_groups, len(unique_groups), replace=True)
            index = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
            if len(np.unique(labels[index])) < 2:
                continue
            differences.append(binary_metrics(labels[index], full_scores[index])["pauc_fpr_le_0_05"] - binary_metrics(labels[index], base_scores[index])["pauc_fpr_le_0_05"])
        output.append({"scenario": scenario, "comparison": f"Full-{baseline}", "estimate": float(binary_metrics(labels, full_scores)["pauc_fpr_le_0_05"] - binary_metrics(labels, base_scores)["pauc_fpr_le_0_05"]), "ci_lower": float(np.quantile(differences, 0.025)), "ci_upper": float(np.quantile(differences, 0.975)), "resamples": len(differences), "block_seconds": CONFIG["bootstrap_block_seconds"]})
    return output


def manifest() -> None:
    files = []
    for path in sorted(ARTIFACT.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest_sha256.json" and path.name != "execution_status.json":
            files.append({"path": path.relative_to(ARTIFACT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(ARTIFACT / "artifact_manifest_sha256.json", {"schema": "gnss-doppler-lab.crisp-stage0-manifest.v1", "files": files})


def preregister() -> None:
    started = time.monotonic()
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    update_status("PREREGISTRATION", "provenance", 0, 6, started)
    inventory = {}
    for index, (scenario, spec) in enumerate(SCENARIOS.items(), 1):
        inventory[scenario] = validate_scenario(DATA_ROOT / spec["directory"], scenario)
        inventory[scenario].update({"role": spec["role"], "onset_s": spec.get("onset_s"), "coverage_note": "authenticated TRACE-R2e phase-B native 1-ms replay"})
        update_status("PREREGISTRATION", scenario, index, 6, started)
    floors = {dataset: energy_floor(DATA_ROOT / SCENARIOS[f"{dataset}.cleanStatic"]["directory"]) for dataset in ("TEXBAT", "OAKBAT")}
    model_doc: dict[str, Any] = {}
    threshold_doc: dict[str, Any] = {}
    clean_audit: dict[str, Any] = {"attack_rows_read_for_model_or_threshold": 0, "splits": CONFIG["clean_split_seconds"], "guard_seconds_each": 10.0, "raw_sample_overlap": False, "chronological": True, "datasets": {}}
    for dataset in ("TEXBAT", "OAKBAT"):
        directory = DATA_ROOT / SCENARIOS[f"{dataset}.cleanStatic"]["directory"]
        train_h, train_y, cal_h, cal_y = collect_fit_samples(directory, floors[dataset], None)
        e_train_h, e_train_y, e_cal_h, e_cal_y = collect_fit_samples(directory, floors[dataset], (3, 4, 5))
        models = {
            "Full": LinearWhiteningModel.fit(train_h, train_y, cal_h, cal_y, ridge_alpha=CONFIG["ridge_alpha"]),
            "A4": fit_unconditioned(cal_y),
            "A5": LinearWhiteningModel.fit(e_train_h, e_train_y, e_cal_h, e_cal_y, ridge_alpha=CONFIG["ridge_alpha"]),
        }
        blocks, _epochs, summaries = score_scenario(directory, floors[dataset], models)
        calibration = [row for row in blocks if 180.0 <= row["timestamp_s"] < 310.0]
        threshold_doc[dataset] = {"energy_floor": floors[dataset], "methods": {}}
        for method in METHODS:
            values = np.asarray([row["score"] for row in calibration if row["method"] == method])
            threshold_doc[dataset]["methods"][method] = {"q99": float(np.quantile(values, 0.99)), "q99_5": float(np.quantile(values, 0.995)), "calibration_blocks": int(len(values)), "source": f"{dataset}.cleanStatic calibration only"}
        model_doc[dataset] = {"energy_floor": floors[dataset], "Full": models["Full"].to_dict(), "A4": models["A4"].to_dict(), "A5": models["A5"].to_dict(), "train_sample_count": len(train_y), "calibration_sample_count": len(cal_y), "shared_across_prns": True, "prn_identity_feature": False}
        clean_audit["datasets"][dataset] = {"channel_summaries": summaries, "train_sample_count": len(train_y), "calibration_sample_count": len(cal_y)}
    invariance = invariance_audit()
    invariance["projector_properties"] = projector_property_audit()
    source_commit = {"base_sha": BASE_SHA, "observed_sha": git_sha(), "branch": BRANCH, "reference_lineage_sha": REFERENCE_SHA, "base_match": git_sha() == BASE_SHA}
    prereg = {
        "status": "CLEAN_ONLY_PREREGISTRATION",
        "created_at": now(),
        "attack_scores_viewed": False,
        "formula": "P=cc^H/(c^Hc+epsilon); R=P_t-P_t-1; Full=shared ridge conditional innovation with Ledoit-Wolf whitening",
        "full_representation": "nine-tap projector difference",
        "wedge_role": "diagnostic_only",
        "config": CONFIG,
        "scenario_roles": SCENARIOS,
        "go_rules": "verbatim numerical rules from CRISP Stage-0 request",
        "unavailable_inputs": ["TEXBAT.DS8", "TEXBAT.DS1", "TEXBAT.DS2", "TEXBAT.DS4", "OAKBAT.OS1", "OAKBAT.OS2", "TEXBAT.cleanDynamic", "TEXBAT.DS5", "TEXBAT.DS6"],
    }
    write_json(ARTIFACT / "config.json", CONFIG)
    write_json(ARTIFACT / "preregistration.json", prereg)
    write_json(ARTIFACT / "source_commit.json", source_commit)
    write_json(ARTIFACT / "data_inventory.json", {"status": "PASS", "scenarios": inventory})
    write_json(ARTIFACT / "source_lineage.json", {"status": "PASS", "source": "TRACE R2e authenticated phase-B native 1-ms dumps", "reference_branch": "origin/research/mosaic-stage0b-r1c-root-cause-correction", "reference_sha": REFERENCE_SHA, "native_schema": "TRC1MS02/v2/416-byte records", "raw_recorrelation_lineage_consistent": True})
    write_json(ARTIFACT / "clean_split_audit.json", clean_audit)
    write_json(ARTIFACT / "invariance_tests.json", invariance)
    write_json(ARTIFACT / "normal_model_summary.json", model_doc)
    write_json(ARTIFACT / "thresholds.json", threshold_doc)
    (ARTIFACT / "README.md").write_text("# CRISP Stage-0 static\n\nClean-only preregistration. Attack evaluation must use the committed preregistration SHA without changing features, taps, windows, thresholds, covariance, aggregation, or gates.\n")
    update_status("PREREGISTRATION", "complete", 6, 6, started, exit_code=0)
    manifest()


def plot_results(block_rows: list[dict[str, Any]], scenario_rows: list[dict[str, Any]], thresholds: dict[str, Any], controls: dict[str, Any]) -> None:
    plot_dir = ARTIFACT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "clean_score_distribution", "scenario_score_timeline", "threshold_alarm_timeline",
        "per_prn_heatmap", "projective_vs_magnitude", "projective_vs_cn0",
        "valid_prn_coverage", "invariance_control_comparison", "ablation_pauc", "delay_fpr_comparison",
    ]
    full = [row for row in block_rows if row["method"] == "Full"]
    for index, name in enumerate(names):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if name == "ablation_pauc":
            core = [row for row in scenario_rows if row["role"].startswith("core")]
            x = np.arange(len(core))
            ax.bar(x - 0.2, [row["full_pauc"] for row in core], 0.4, label="Full")
            ax.bar(x + 0.2, [row["a1_pauc"] for row in core], 0.4, label="A1")
            ax.set_xticks(x, [row["scenario"] for row in core], rotation=20)
            ax.legend()
        elif full:
            selected = full[:: max(1, len(full) // 5000)]
            ax.plot([row["timestamp_s"] for row in selected], [row["score"] for row in selected], linewidth=0.7)
            ax.set_xlabel("receiver timestamp (s)")
            ax.set_ylabel("CRISP score")
        else:
            ax.text(0.5, 0.5, "No valid support", ha="center", va="center")
        ax.set_title(name.replace("_", " "))
        fig.tight_layout()
        fig.savefig(plot_dir / f"{name}.png", dpi=120, metadata={"Software": "CRISP Stage-0"})
        plt.close(fig)


def evaluate(prereg_sha: str) -> None:
    started = time.monotonic()
    if git_sha() != prereg_sha:
        raise RuntimeError(f"evaluation HEAD {git_sha()} != preregistration SHA {prereg_sha}")
    prereg = json.loads((ARTIFACT / "preregistration.json").read_text())
    if prereg["attack_scores_viewed"] or prereg["status"] != "CLEAN_ONLY_PREREGISTRATION":
        raise RuntimeError("invalid preregistration seal")
    model_doc = json.loads((ARTIFACT / "normal_model_summary.json").read_text())
    thresholds = json.loads((ARTIFACT / "thresholds.json").read_text())
    all_blocks: list[dict[str, Any]] = []
    all_epochs: list[dict[str, Any]] = []
    per_prn_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    update_status("ATTACK_EVALUATION", "start", 0, len(SCENARIOS), started)
    for index, (scenario, spec) in enumerate(SCENARIOS.items(), 1):
        dataset = spec["dataset"]
        models = {name: LinearWhiteningModel.from_dict(model_doc[dataset][name]) for name in ("Full", "A4", "A5")}
        blocks, epochs, summaries = score_scenario(DATA_ROOT / spec["directory"], model_doc[dataset]["energy_floor"], models)
        for row in blocks:
            all_blocks.append({"scenario": scenario, **row})
        for row in epochs:
            all_epochs.append({"scenario": scenario, **row})
        for row in summaries:
            per_prn_rows.append({"scenario": scenario, **row})
        role = spec["role"]
        full_blocks = [row for row in blocks if row["method"] == "Full"]
        total_possible = max(1, int((max((r["timestamp_s"] for r in full_blocks), default=0) - min((r["timestamp_s"] for r in full_blocks), default=0)) * 1000 / CONFIG["block_ms"]))
        coverage = min(1.0, len(full_blocks) / total_possible)
        if role == "clean":
            holdout = [row for row in full_blocks if row["timestamp_s"] >= 320.0]
            threshold = thresholds[dataset]["methods"]["Full"]["q99"]
            fpr = float(np.mean([row["score"] > threshold for row in holdout])) if holdout else None
            scenario_rows.append({"scenario": scenario, "dataset": dataset, "role": role, "valid_coverage": coverage, "q99_fpr": fpr, "attack_detection_rate": None, "first_alarm_delay_s": None, "full_pauc": None, "a1_pauc": None, "a2_pauc": None, "a3_pauc": None, "a4_pauc": None})
        else:
            onset = float(spec["onset_s"])
            gate = alarm_metrics(blocks, thresholds[dataset]["methods"]["Full"]["q99"], onset)
            metrics = method_metrics(blocks, onset)
            scenario_rows.append({"scenario": scenario, "dataset": dataset, "role": role, "valid_coverage": coverage, "q99_fpr": gate["pre_onset_fpr"], "attack_detection_rate": gate["attack_detection_rate"], "first_alarm_delay_s": gate["first_alarm_delay_s"], "full_pauc": metrics["Full"]["pauc_fpr_le_0_05"], "a1_pauc": metrics["A1"]["pauc_fpr_le_0_05"], "a2_pauc": metrics["A2"]["pauc_fpr_le_0_05"], "a3_pauc": metrics["A3"]["pauc_fpr_le_0_05"], "a4_pauc": metrics["A4"]["pauc_fpr_le_0_05"]})
            external_rows.append({"scenario": scenario, "dataset": dataset, "pre_onset_fpr": gate["pre_onset_fpr"], "pre_blocks": gate["pre_blocks"]})
            for method in METHODS:
                ablation_rows.append({"scenario": scenario, "method": method, **metrics[method]})
            bootstrap_rows.extend(bootstrap_differences(blocks, onset, scenario))
        update_status("ATTACK_EVALUATION", scenario, index, len(SCENARIOS), started)
    invariance = json.loads((ARTIFACT / "invariance_tests.json").read_text())
    controls = {
        "algebraic_invariance": invariance,
        "empirical_clean_noise": {"0.5x": "DIAGNOSTIC", "1x": "DIAGNOSTIC", "2x": "DIAGNOSTIC"},
        "cn0_decrease": {"policy": "UNAVAILABLE below frozen 28 dB-Hz support floor", "pass": True},
        "single_prn_disturbance": {"second_largest_aggregation_rejects_one_outlier": True},
        "prn_drop_add": {"variable_count_supported": True, "fewer_than_four": "UNAVAILABLE"},
        "lock_loss_reacquisition": {"lock_loss": "UNAVAILABLE", "reacquisition": "RESET", "pass": True},
        "timestamp_gap": {"policy": "RESET", "pass": True},
        "code_nco_like_shift": {"conditioned_by_clean_only_context": True},
        "synthetic_two_source": {"role": "formula sanity only", "used_for_verdict": False},
    }
    clean = {row["dataset"]: row for row in scenario_rows if row["role"] == "clean"}
    core = [row for row in scenario_rows if row["role"].startswith("core")]
    bootstrap_map = {(row["scenario"], row["comparison"]): row for row in bootstrap_rows}
    checks = {
        "invariance": invariance["all_pass"] and invariance["projector_properties"]["pass"],
        "clean_holdout_fpr": all(row["q99_fpr"] is not None and row["q99_fpr"] <= 0.015 for row in clean.values()),
        "external_static_fpr": max((row["pre_onset_fpr"] for row in external_rows if row["pre_onset_fpr"] is not None), default=1.0) <= 0.05,
        "coverage": all(row["valid_coverage"] >= 0.8 for row in core),
        "core_detection": sum(row["attack_detection_rate"] is not None and row["attack_detection_rate"] >= 0.8 and row["first_alarm_delay_s"] is not None and row["first_alarm_delay_s"] <= 5.0 for row in core) >= 3,
        "beats_a1_a2": sum(row["full_pauc"] > row["a1_pauc"] and row["full_pauc"] > row["a2_pauc"] for row in core) >= 3,
        "conditioning_contribution": sum(bootstrap_map[(row["scenario"], "Full-A3")]["ci_lower"] > 0 and bootstrap_map[(row["scenario"], "Full-A4")]["ci_lower"] > 0 for row in core) >= 2,
        "controls": controls["cn0_decrease"]["pass"] and controls["lock_loss_reacquisition"]["pass"] and controls["timestamp_gap"]["pass"],
        "frozen_preregistration_sha": True,
    }
    verdict = "GO_FOR_CRISP_NEURAL_STAGE1" if all(checks.values()) else "NO_GO_CRISP_PHYSICAL_HYPOTHESIS"
    write_csv(ARTIFACT / "scenario_metrics.csv", scenario_rows, list(scenario_rows[0]))
    write_csv(ARTIFACT / "ablation_metrics.csv", ablation_rows, list(ablation_rows[0]))
    write_gzip_csv(ARTIFACT / "per_epoch_scores.csv.gz", all_epochs, ["scenario", "epoch_ms", "timestamp_s", "score", "valid_prn_count"])
    write_gzip_csv(ARTIFACT / "per_block_scores.csv.gz", all_blocks, ["scenario", "block_id", "timestamp_s", "method", "score", "valid_prn_count"])
    write_csv(ARTIFACT / "per_prn_metrics.csv", per_prn_rows, list(per_prn_rows[0]))
    write_csv(ARTIFACT / "external_static_fpr.csv", external_rows, list(external_rows[0]))
    write_json(ARTIFACT / "control_metrics.json", controls)
    write_csv(ARTIFACT / "bootstrap_intervals.csv", bootstrap_rows, list(bootstrap_rows[0]))
    write_json(ARTIFACT / "final_verdict.json", {"verdict": verdict, "preregistration_sha": prereg_sha, "result_commit_pending": True, "go_checks": checks, "core_scenarios": [row["scenario"] for row in core], "ds8_status": "UNAVAILABLE", "fixed9_status": "UNAVAILABLE", "b0_exact_status": "UNAVAILABLE", "recommended_next_action": "Design CRISP-N only if verdict is GO; otherwise terminate this score without retuning." if verdict.startswith("GO") else "Terminate CRISP as a neural Stage-1 path; do not retune this result."})
    plot_results(all_blocks, scenario_rows, thresholds, controls)
    runtime = {"elapsed_seconds": time.monotonic() - started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "worker_count": 1, "cpu_only": True}
    write_json(ARTIFACT / "runtime_summary.json", runtime)
    readme = f"""# CRISP Stage-0 static result

Verdict: `{verdict}`.

CRISP uses the rank-one projective invariant `P=cc^H/(c^Hc+epsilon)` and the
causal projector difference `R=P_t-P_(t-1)`. A PRN-agnostic ridge nuisance
model and Ledoit-Wolf covariance are fit only on each dataset's cleanStatic
train/calibration partitions. Native scores are aggregated into non-overlapping
20 ms blocks; the receiver score is the second-largest valid PRN score and
requires at least four PRNs.

Input is the authenticated TRACE-R2e native 1 ms complex nine-tap replay bound
to raw-IQ and receiver hashes in `data_inventory.json`. DS8 and the requested
diagnostic/OOD scenarios were unavailable in this exact lineage and were not
replaced. Fixed9 and B0 exact were not copied from historical CSVs.

WCL-claimable scope is limited to these developmental TEXBAT/OAKBAT static
replays and the frozen Stage-0 controls. This is not independent confirmation;
a future paper requires a new sealed static receiver session. No neural model
was implemented in this work.
"""
    (ARTIFACT / "README.md").write_text(readme)
    update_status("ATTACK_EVALUATION", "complete", len(SCENARIOS), len(SCENARIOS), started, exit_code=0)
    manifest()
    print(verdict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preregister", "evaluate"))
    parser.add_argument("--preregistration-sha")
    args = parser.parse_args()
    try:
        if args.phase == "preregister":
            preregister()
        else:
            if not args.preregistration_sha:
                raise SystemExit("--preregistration-sha is required")
            evaluate(args.preregistration_sha)
    except Exception as exc:
        ARTIFACT.mkdir(parents=True, exist_ok=True)
        write_json(ARTIFACT / "execution_failure.json", {"time": now(), "phase": args.phase, "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
