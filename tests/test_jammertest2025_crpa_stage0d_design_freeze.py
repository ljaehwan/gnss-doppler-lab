from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from gnss_doppler_lab.jammertest_crpa_stage0d import CLASS_BLOCKS, FOLD_TEST_BLOCKS


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/jammertest2025_crpa_stage0d_true_spoof_discrimination"


def split_rows() -> list[dict]:
    with (ARTIFACT / "split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_design_is_label_only_and_exact_class() -> None:
    design = json.loads((ARTIFACT / "design_freeze.json").read_text())
    assert design["status"] == "LABEL_ONLY_DESIGN_FREEZE_PRE_IQ"
    assert design["iq_bytes_read"] == 0
    assert design["stage0c_model_results_used"] is False
    assert design["eligibility"] == {
        "area": 1,
        "negative_exact_class": "Prn",
        "positive_exact_class": "Spoof",
        "transmit_power_dbm": 40,
    }
    assert design["class_counts"] == {"Prn": 164, "Spoof": 124}


def test_fixed_five_fold_assignment_and_class_specific_guard() -> None:
    rows = split_rows()
    for fold, assignment in enumerate(FOLD_TEST_BLOCKS):
        current = [row for row in rows if int(row["fold"]) == fold]
        for class_name, expected_blocks in assignment.items():
            observed = {
                int(row["class_block"])
                for row in current
                if row["class_name"] == class_name and row["role"] == "test"
            }
            assert observed == set(expected_blocks)
            guard = {
                int(row["class_block"])
                for row in current
                if row["class_name"] == class_name and row["role"] == "guard"
            }
            run = set(CLASS_BLOCKS[class_name])
            expected_guard = {
                adjacent
                for block in expected_blocks
                for adjacent in (block - 1, block + 1)
                if adjacent in run and adjacent not in expected_blocks
            }
            assert guard == expected_guard


def test_all_288_samples_are_oof_exactly_once() -> None:
    rows = split_rows()
    tests = [row for row in rows if row["role"] == "test"]
    counts = Counter(int(row["sample_index"]) for row in tests)
    assert len(counts) == 288
    assert set(counts.values()) == {1}
    assert Counter(row["class_name"] for row in tests) == {"Spoof": 124, "Prn": 164}


def test_all_enumerated_11_class_blocks_have_oof_coverage() -> None:
    rows = split_rows()
    observed = {
        row["class_block_key"] for row in rows if row["role"] == "test"
    }
    expected = {
        f"{class_name}:{block}"
        for class_name, blocks in CLASS_BLOCKS.items()
        for block in blocks
    }
    assert len(expected) == 11
    assert observed == expected


def test_train_guard_test_and_train_test_blocks_do_not_overlap() -> None:
    rows = split_rows()
    for fold in range(5):
        current = [row for row in rows if int(row["fold"]) == fold]
        samples = {
            role: {int(row["sample_index"]) for row in current if row["role"] == role}
            for role in ("train", "guard", "test")
        }
        assert samples["train"].isdisjoint(samples["guard"])
        assert samples["train"].isdisjoint(samples["test"])
        assert samples["guard"].isdisjoint(samples["test"])
        blocks = {
            role: {row["class_block_key"] for row in current if row["role"] == role}
            for role in ("train", "test")
        }
        assert blocks["train"].isdisjoint(blocks["test"])
        for role in ("train", "test"):
            assert {row["class_name"] for row in current if row["role"] == role} == {"Spoof", "Prn"}


def test_pipeline_and_matching_contract_are_frozen() -> None:
    design = json.loads((ARTIFACT / "design_freeze.json").read_text())
    assert design["balancing"]["test_sample_removal"] is False
    assert design["balancing"]["train_sample_removal"] is False
    assert design["pipeline"]["classifier"]["class_weight"] == "balanced"
    assert design["matching"]["primary_caliper_db"] == 0.25
    assert design["matching"]["sensitivity_calipers_db"] == [0.1, 0.5, 1.0]
    assert design["matching"]["train_test_cross_pairing"] is False
    assert set(design["models"]) == {"M0", "M1", "M2", "M2R", "M3"}


def test_complete_oof_contract_passes() -> None:
    contract = json.loads((ARTIFACT / "complete_oof_contract.json").read_text())
    assert contract["status"] == "PASS"
    assert contract["errors"] == []
    assert contract["oof_missing_count"] == 0
    assert contract["oof_duplicate_count"] == 0
    assert contract["enumerated_unique_class_block_count"] == 11


def test_execution_is_bound_to_both_pushed_freezes() -> None:
    runner = (ROOT / "scripts/run_jammertest2025_crpa_stage0d.py").read_text()
    executor = (ROOT / "scripts/execute_jammertest2025_crpa_stage0d.py").read_text()
    contract = (ROOT / "src/gnss_doppler_lab/jammertest_crpa_stage0d_execution.py").read_text()
    assert "execute_jammertest2025_crpa_stage0d" in runner
    assert "DESIGN_FREEZE_COMMIT" in executor and "POWER_MATCH_FREEZE_COMMIT" in executor
    assert "33c3c6924f2a7f42ca964e1bd136239cacaea04c" in contract
    assert "aa1859b411b65a2599325642b7c0d4a9abf6558c" in contract
    assert "/home/ubuntu/ssd_data" not in runner
    assert "/home/ubuntu/ssd_data" not in executor
