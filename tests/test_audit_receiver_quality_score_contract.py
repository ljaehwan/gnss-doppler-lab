from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_receiver_quality_score_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location("quality_audit_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_audit_localizes_decision_flips_to_removed_gap_history(tmp_path):
    module = load_module()
    indexes = [0, 1, 2, 3, 6, 7, 8, 9]
    node = pd.DataFrame([
        {
            "run_id": "run-a",
            "prn": "G01",
            "channel": 0,
            "segment_index": 0,
            "window_index": index,
            "epoch_count": 50,
            "window_bin_s": index * 0.5,
            "window_start_s": index * 0.5,
            "window_mid_s": index * 0.5 + 0.5,
            "window_end_s": index * 0.5 + 1.0,
        }
        for index in indexes
    ])
    node_path = tmp_path / "nodes.csv"
    node.to_csv(node_path, index=False)

    legacy_indexes = [2, 3, 6, 7, 8, 9]
    quality_indexes = [2, 3, 8, 9]
    legacy = pd.DataFrame({
        "run_id": ["run-a"] * 6,
        "prn": ["G01"] * 6,
        "window_bin_s": [index * 0.5 for index in legacy_indexes],
        "prn_node_rmse": [0.1, 0.1, 1.0, 1.0, 0.1, 0.1],
    })
    quality_rows = []
    for index, block_index, position in (
        (2, 0, 2), (3, 0, 3), (8, 1, 2), (9, 1, 3)
    ):
        start_index = index - 2
        quality_rows.append({
            "run_id": "run-a",
            "prn": "G01",
            "window_bin_s": index * 0.5,
            "prn_node_rmse": 0.1,
            "channel": 0,
            "segment_index": 0,
            "prn_segment_ordinal": 0,
            "continuity_block_index": block_index,
            "target_window_index": index,
            "target_sequence_position": position,
            "epoch_count": 50,
            "tracking_age_s": index * 0.5,
            "continuity_age_s": position * 0.5,
            "segment_start_s": 0.0,
            "history_start_window_index": start_index,
            "history_end_window_index": index - 1,
            "history_start_s": start_index * 0.5,
            "history_end_s": index * 0.5 + 0.5,
            "history_length": 2,
            "reacquisition_flag": 0,
            "sequence_restart_flag": int(block_index > 0),
            "history_same_segment_flag": 1,
        })
    quality = pd.DataFrame(quality_rows)
    legacy_path = tmp_path / "legacy_prn.csv"
    quality_path = tmp_path / "quality_prn.csv"
    legacy.to_csv(legacy_path, index=False)
    quality.to_csv(quality_path, index=False)

    bins = [index * 0.5 for index in legacy_indexes]
    legacy_values = [0.1, 0.1, 1.0, 1.0, 0.1, 0.1]
    quality_values = [0.1] * 6
    def events(values):
        return pd.DataFrame({
            "window_bin_s": bins,
            "window_start_s": bins,
            "prn_node_rmse_max": values,
            "prn_node_rmse_top3_mean": values,
            "prn_node_rmse_mean": values,
        })
    legacy_event_path = tmp_path / "legacy_event.csv"
    quality_event_path = tmp_path / "quality_event.csv"
    events(legacy_values).to_csv(legacy_event_path, index=False)
    events(quality_values).to_csv(quality_event_path, index=False)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"normal_prn_thresholds": {"q": 0.5}}))

    audit = module.build_audit(
        node_path,
        legacy_path,
        quality_path,
        legacy_event_path,
        quality_event_path,
        summary_path,
        stride_s=0.5,
        history_length=2,
        onset_s=3.75,
    )

    assert audit["sequence_inventory"]["restart_boundaries"] == 1
    comparison = audit["prn_score_comparison"]
    assert comparison["legacy_rows"] == comparison["expected_legacy_rows"] == 6
    assert comparison["quality_rows"] == comparison["expected_quality_rows"] == 4
    assert comparison["legacy_only_boundary_crossing_rows"] == 2
    flip = audit["event_score_comparison"]["threshold_flips"]["max_q"]
    assert flip["flips"] == flip["boundary_bin_flips"] == 2
    assert flip["non_boundary_bin_flips"] == 0
