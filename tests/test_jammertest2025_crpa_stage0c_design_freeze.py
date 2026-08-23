from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/jammertest2025_crpa_stage0c_spatial_discrimination"


def load_split() -> list[dict]:
    with (ARTIFACT / "split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_design_is_label_only_and_frozen() -> None:
    design = json.loads((ARTIFACT / "design_freeze.json").read_text())
    assert design["status"] == "LABEL_ONLY_DESIGN_FREEZE_PRE_FEATURE"
    assert design["iq_feature_bytes_read"] == 0
    assert design["stage0b_class_spatial_results_used"] is False
    assert design["primary"]["config"]["powers_dbm"] == [30, 40]
    assert design["primary"]["config"]["block_size"] == 32
    assert design["primary"]["config"]["folds"] == 3
    assert design["numerical_invariance"]["rtol"] == 1e-10
    assert design["numerical_invariance"]["atol"] == 1e-12


def test_primary_and_sensitivity_feasibility_is_frozen() -> None:
    design = json.loads((ARTIFACT / "design_freeze.json").read_text())
    assert design["primary"]["audit"]["feasible"] is True
    assert design["sensitivity_a"]["audit"]["feasible"] is True
    assert design["sensitivity_b"]["audit"]["feasible"] is False
    assert design["block_size_2048"]["executable"] is False


def test_no_train_test_group_overlap_and_guard_is_adjacent() -> None:
    rows = load_split()
    for evaluation in ("primary", "sensitivity_a"):
        folds = sorted({int(row["fold"]) for row in rows if row["evaluation"] == evaluation})
        for fold in folds:
            selected = [
                row for row in rows
                if row["evaluation"] == evaluation and int(row["fold"]) == fold
            ]
            train = {int(row["group_key"]) for row in selected if row["final_role"] == "train"}
            test = {int(row["group_key"]) for row in selected if row["final_role"] == "test"}
            guard = {int(row["group_key"]) for row in selected if row["final_role"] == "guard"}
            assert train.isdisjoint(test)
            assert train.isdisjoint(guard)
            for block in test:
                assert block - 1 in test or block - 1 in guard or not any(
                    int(row["group_key"]) == block - 1 for row in selected
                )
                assert block + 1 in test or block + 1 in guard or not any(
                    int(row["group_key"]) == block + 1 for row in selected
                )


def test_selected_cells_are_deterministically_balanced() -> None:
    rows = load_split()
    for evaluation in ("primary", "sensitivity_a"):
        folds = sorted({int(row["fold"]) for row in rows if row["evaluation"] == evaluation})
        for fold in folds:
            for role in ("train", "test"):
                counts = Counter(
                    (row["transmit_power_dbm"], row["binary_class"])
                    for row in rows
                    if row["evaluation"] == evaluation
                    and int(row["fold"]) == fold
                    and row["final_role"] == role
                )
                for power in sorted({key[0] for key in counts}):
                    assert counts[(power, "positive")] == counts[(power, "negative")]


def test_primary_and_exclusion_cover_all_released_label_indices() -> None:
    rows = load_split()
    primary = {int(row["sample_index"]) for row in rows if row["evaluation"] == "primary"}
    with (ARTIFACT / "exclusion_manifest.csv").open(newline="", encoding="utf-8") as handle:
        excluded = {int(row["sample_index"]) for row in csv.DictReader(handle)}
    assert primary.isdisjoint(excluded)
    assert len(primary | excluded) == 36_186


def test_execution_is_bound_to_pushed_design_freeze() -> None:
    source = (ROOT / "scripts/run_jammertest2025_crpa_stage0c.py").read_text()
    executor = (ROOT / "scripts/execute_jammertest2025_crpa_stage0c.py").read_text()
    assert "execute_jammertest2025_crpa_stage0c import main" in source
    assert "67495be08486b5479fbf09ee1b03c9faddb2a077" in (ARTIFACT / "design_freeze_commit.json").read_text()
    assert "--raw-npy" in executor
    assert "/home/ubuntu/ssd_data" not in executor
