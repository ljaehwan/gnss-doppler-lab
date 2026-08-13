#!/usr/bin/env python3
"""Run frozen clean-only A2/A3/A4 ablations after the primary clean pass."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.gcspo_ablations import fit_a2_loading, role_a2_terms, score_a2_terms, select_a2_lambda
from gnss_doppler_lab.gcspo_artifacts import canonical_write_json
from gnss_doppler_lab.gcspo_clean import residual_table, run_clean_a1
from gnss_doppler_lab.gcspo_core import empirical_threshold
from gnss_doppler_lab.gcspo_full import GeometryCache, geometry_preflight, role_full_terms, score_full_terms


def _thresholds(rows):
    values = np.asarray([row["score"] for row in rows])
    return {"q99": empirical_threshold(values, .99), "q995": empirical_threshold(values, .995)}


def _strip_state(rows):
    return [{key: value for key, value in row.items() if key != "state"} for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/gcspo_stage0_static_rerun")
    parser.add_argument("--clean-root", type=Path, default=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9"))
    args = parser.parse_args()
    config = json.loads((args.artifact_dir / "config.json").read_text())
    primary = json.loads((args.artifact_dir / "normal_model_summary.json").read_text())
    clean = run_clean_a1(args.clean_root, ridge_grid=config["h0_predictor"]["ridge_grid"])
    train_epochs, _, _, train_z = residual_table(clean["data"], clean["model"], clean["whitener"], 30, 140)
    loading, loading_report = fit_a2_loading([train_z[train_epochs == epoch] for epoch in np.unique(train_epochs)])
    a2_validation = role_a2_terms(clean["data"], clean["model"], clean["whitener"], clean["gamma"], loading, 150, 210)
    a2_lambda, a2_objectives = select_a2_lambda(a2_validation, config["lambda_selection"]["grid"])
    a2_cal = score_a2_terms(role_a2_terms(clean["data"], clean["model"], clean["whitener"], clean["gamma"], loading, 220, 340), a2_lambda)
    a2_hold = score_a2_terms(role_a2_terms(clean["data"], clean["model"], clean["whitener"], clean["gamma"], loading, 350, 470), a2_lambda)
    geometry = geometry_preflight(args.clean_root, tracked_prns=clean["data"].prn)
    methods = {}
    for method, rows in (("A3", {"code_error_chips", "pll_phase_error_cycles"}),
                         ("A4", {"carrier_doppler_hz", "code_frequency_offset_chips_s"})):
        cache = GeometryCache(geometry["ephemerides"], geometry["receiver_ecef"], rows)
        smoothness = primary["lambda_selected"]
        calibration = score_full_terms(role_full_terms(clean["data"], clean["model"], clean["whitener"], clean["gamma"], cache, 220, 340), smoothness)
        holdout = score_full_terms(role_full_terms(clean["data"], clean["model"], clean["whitener"], clean["gamma"], cache, 350, 470), smoothness)
        methods[method] = {"lambda": smoothness, "validated_rows": sorted(rows), "thresholds": _thresholds(calibration),
                           "calibration": _strip_state(calibration), "holdout": _strip_state(holdout)}
        print(f"{method}_CLEAN_PASS calibration={len(calibration)} holdout={len(holdout)}", flush=True)
    methods["A2"] = {"loading": loading.tolist(), "loading_report": loading_report, "lambda": a2_lambda,
                     "lambda_objectives": a2_objectives, "thresholds": _thresholds(a2_cal),
                     "calibration": _strip_state(a2_cal), "holdout": _strip_state(a2_hold)}
    thresholds_path = args.artifact_dir / "thresholds.json"; thresholds = json.loads(thresholds_path.read_text())
    for method in ("A2", "A3", "A4"): thresholds[method] = methods[method]["thresholds"]
    canonical_write_json(thresholds_path, thresholds)
    canonical_write_json(args.artifact_dir / "clean_ablation_report.json", {
        "schema": "gnss-doppler-lab.gcspo-stage0.clean-ablation-report.v1", "run_status": "CLEAN_ABLATIONS_PARTIAL_PASS",
        "protected_attack_rows_read": False, "attack_access_count": 0, "methods": methods,
        "remaining": ["A0_B0_EXACT", "A5_INDEPENDENT_INPUT"],
    })
    print(f"A2_CLEAN_PASS calibration={len(a2_cal)} holdout={len(a2_hold)} lambda={a2_lambda}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
