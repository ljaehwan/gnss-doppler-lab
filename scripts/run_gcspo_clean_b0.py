#!/usr/bin/env python3
"""Recompute frozen B0 from clean receiver rows and align it to Full support."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.gcspo_artifacts import canonical_write_json, sha256_file
from gnss_doppler_lab.gcspo_b0 import build_scheduled_node_table, exact_common_support, role_filter
from gnss_doppler_lab.gcspo_core import empirical_threshold

CHECKPOINT_SHA = "f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b"
GATE_SHA = "45e4df9beed4b689700e928ef8217cbed17c2178a47ff240c40eec3c74e3f414"
GATE_SCRIPT_SHA = "1bb353101d1a98a8bd6a6feaa72757829669f718f3e2e2d497e64422177c5e10"


def _pandas():
    import pandas
    return pandas


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _full_frame(rows, common_mask_rows):
    pd = _pandas()
    support = {float(row["window_start_s"]): row["prns"] for row in common_mask_rows}
    return pd.DataFrame([{"window_start_s": row["window_start_s"], "prns": support[float(row["window_start_s"])], "score": row["score"]} for row in rows if float(row["window_start_s"]) in support])


def _event_scores(gate, prn, node_thresholds):
    prepared = prn.assign(run_id="cleanStatic", window_mid_s=prn.window_start_s + .5)
    return gate.build_event_scores(prepared, node_thresholds, alpha=.75)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/gcspo_stage0_static_rerun")
    parser.add_argument("--clean-root", type=Path, default=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9"))
    args = parser.parse_args()
    pd = _pandas()
    model_dir = ROOT / "artifacts/ai_morph_gru_cleanStatic_q70_frame"
    checkpoint = model_dir / "prn_local_gru_predictor.pt"
    gate_path = ROOT / "configs/detectors/texbat_btail_gate_v1.json"
    gate_script = ROOT / "scripts/eval_btail_support_gate.py"
    for path, expected in ((checkpoint, CHECKPOINT_SHA), (gate_path, GATE_SHA), (gate_script, GATE_SCRIPT_SHA)):
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen B0 byte mismatch: {path}: {actual}")

    work = args.artifact_dir.parent / f".{args.artifact_dir.name}.b0_clean_recomputed"
    work.mkdir(parents=True, exist_ok=True)
    node = work / "scheduled_node_windows.csv"
    build_scheduled_node_table(args.clean_root, {"calibration": (220., 340.), "holdout": (350., 470.)}).to_csv(node, index=False)
    scorer = _module("gcspo_b0_scorer", ROOT / "scripts/score_texbat_prn_node_gru.py")
    scorer.score_node_csv(node, model_dir, work, "cleanStatic", onset_s=None, output_prefix="gcspo_b0", dataset_prefix="cleanStatic")
    prn_path, _, _ = scorer.score_output_paths(work, "cleanStatic", "gcspo_b0")
    prn = pd.read_csv(prn_path)
    report = json.loads((args.artifact_dir / "clean_only_report.json").read_text())
    roles = {}
    all_cal_nodes, all_hold_nodes = role_filter(prn, 220., 340.), role_filter(prn, 350., 470.)
    cal_nodes, cal_full = exact_common_support(all_cal_nodes, _full_frame(report["scores"]["Full_calibration"], report["scores"]["A1_calibration"]))
    hold_nodes, hold_full = exact_common_support(all_hold_nodes, _full_frame(report["scores"]["Full_holdout"], report["scores"]["A1_holdout"]))
    if cal_full.empty or hold_full.empty:
        raise RuntimeError("B0/Full exact clean common support is empty")
    node_thresholds = {name: float(cal_nodes.prn_node_rmse.quantile(q)) for name, q in (("q50", .5), ("q70", .7), ("q80", .8))}
    gate = _module("gcspo_b0_gate", gate_script)
    cal_events = _event_scores(gate, cal_nodes, node_thresholds)
    hold_events = _event_scores(gate, hold_nodes, node_thresholds)
    score_col = gate.FINAL_SCORE
    thresholds = {name: empirical_threshold(cal_events[score_col].to_numpy(float), q) for name, q in (("q99", .99), ("q995", .995))}
    roles["calibration"] = cal_events.to_dict("records")
    roles["holdout"] = hold_events.to_dict("records")
    fpr = {name: float(np.mean(hold_events[score_col].to_numpy(float) > value)) for name, value in thresholds.items()}
    threshold_path = args.artifact_dir / "thresholds.json"
    threshold_doc = json.loads(threshold_path.read_text())
    threshold_doc["A0_B0"] = {**thresholds, "node_thresholds": node_thresholds}
    threshold_doc.setdefault("calibration_windows", {})["A0_B0"] = len(cal_events)
    canonical_write_json(threshold_path, threshold_doc)
    canonical_write_json(args.artifact_dir / "clean_b0_report.json", {
        "schema": "gnss-doppler-lab.gcspo-stage0.clean-b0-report.v1",
        "run_status": "B0_EXACT_CLEAN_PASS", "protected_attack_rows_read": False,
        "checkpoint_sha256": CHECKPOINT_SHA, "gate_config_sha256": GATE_SHA,
        "gate_script_sha256": GATE_SCRIPT_SHA, "feature_source": "recomputed_from_clean_receiver_rows",
        "common_support": {"calibration_windows": len(cal_full), "holdout_windows": len(hold_full)},
        "thresholds": thresholds, "node_thresholds": node_thresholds, "holdout_fpr": fpr,
        "scores": roles,
    })
    print(f"B0_EXACT_CLEAN_PASS calibration={len(cal_events)} holdout={len(hold_events)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
