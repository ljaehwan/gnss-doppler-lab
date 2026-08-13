#!/usr/bin/env python3
"""Run the frozen clean-only A5 independent-input ablation."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.gcspo_a5 import a5_prior_precision, role_a5_terms
from gnss_doppler_lab.gcspo_artifacts import canonical_write_json
from gnss_doppler_lab.gcspo_clean import run_clean_a1
from gnss_doppler_lab.gcspo_core import empirical_threshold
from gnss_doppler_lab.gcspo_full import _score_terms

_ROWS = None


def _objective(smoothness):
    values = []
    for row in _ROWS:
        prior = a5_prior_precision(prn_count=len(row["prns"]), smoothness=float(smoothness))
        values.append(_score_terms(row["terms"], prior)["gcv"])
    return float(smoothness), float(np.mean(values))


def _score_index(index_and_smoothness):
    index, smoothness = index_and_smoothness; row = _ROWS[index]
    score = _score_terms(row["terms"], a5_prior_precision(prn_count=len(row["prns"]), smoothness=float(smoothness)))
    score.pop("state")
    return {key: row[key] for key in ("window_start_s", "availability_s", "prns", "epoch_ids", "epoch_prn_support")} | score


def _pool(processes):
    return mp.get_context("fork").Pool(processes=processes)


def _choose(objectives):
    best = objectives[0]
    for candidate in objectives[1:]:
        scale = max(abs(candidate["mean_gcv"]), abs(best["mean_gcv"]), 1)
        if candidate["mean_gcv"] < best["mean_gcv"] - 1e-12 * scale or (abs(candidate["mean_gcv"] - best["mean_gcv"]) <= 1e-12 * scale and candidate["lambda"] > best["lambda"]): best = candidate
    return best["lambda"]


def score_parallel(rows, smoothness, workers):
    global _ROWS; _ROWS = rows
    with _pool(workers) as pool: return pool.map(_score_index, [(index, smoothness) for index in range(len(rows))], chunksize=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/gcspo_stage0_static_rerun")
    parser.add_argument("--clean-root", type=Path, default=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4: raise ValueError("workers must be in [1,4]")
    config = json.loads((args.artifact_dir / "config.json").read_text())
    clean = run_clean_a1(args.clean_root, ridge_grid=config["h0_predictor"]["ridge_grid"])
    validated = {"code_error_chips", "pll_phase_error_cycles", "carrier_doppler_hz", "code_frequency_offset_chips_s"}
    global _ROWS
    _ROWS = role_a5_terms(clean["data"], clean["model"], clean["whitener"], clean["gamma"], validated, 150, 210)
    if len(_ROWS) < 100: raise ValueError("A5 has fewer than 100 common validation windows")
    validation_windows = len(_ROWS)
    with _pool(args.workers) as pool: raw_objectives = pool.map(_objective, list(map(float, config["lambda_selection"]["grid"])), chunksize=1)
    objectives = [{"lambda": value, "mean_gcv": objective} for value, objective in raw_objectives]
    selected = _choose(objectives)
    print(f"A5_LAMBDA_PASS windows={len(_ROWS)} lambda={selected}", flush=True)
    _ROWS = role_a5_terms(clean["data"], clean["model"], clean["whitener"], clean["gamma"], validated, 220, 340)
    calibration = score_parallel(_ROWS, selected, args.workers); del _ROWS
    print(f"A5_CALIBRATION_PASS windows={len(calibration)}", flush=True)
    _ROWS = role_a5_terms(clean["data"], clean["model"], clean["whitener"], clean["gamma"], validated, 350, 470)
    holdout = score_parallel(_ROWS, selected, args.workers); del _ROWS
    values = np.asarray([row["score"] for row in calibration])
    thresholds = {"q99": empirical_threshold(values, .99), "q995": empirical_threshold(values, .995)}
    report = {"schema": "gnss-doppler-lab.gcspo-stage0.clean-a5-report.v1", "run_status": "CLEAN_A5_PASS",
              "protected_attack_rows_read": False, "attack_access_count": 0, "validated_rows": sorted(validated),
              "lambda": selected, "lambda_objectives": objectives, "lambda_validation_windows": validation_windows,
              "thresholds": thresholds, "calibration": calibration, "holdout": holdout}
    canonical_write_json(args.artifact_dir / "clean_a5_report.json", report)
    threshold_path = args.artifact_dir / "thresholds.json"; payload = json.loads(threshold_path.read_text()); payload["A5"] = thresholds
    canonical_write_json(threshold_path, payload)
    print(f"A5_CLEAN_PASS calibration={len(calibration)} holdout={len(holdout)}", flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
