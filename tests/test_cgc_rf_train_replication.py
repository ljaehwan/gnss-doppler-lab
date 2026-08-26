import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_cgc_rf_train_replication.py"
CONFIG = ROOT / "configs" / "experiments" / "cgc_rf_train_replication_v1.json"


def load_module():
    name = "cgc_rf_train_replication_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_preregistered_config_and_authorized_inputs_are_pinned():
    module = load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    context = module.validate_config(config, verify_pair_inputs=True)

    assert [pair["paired_group_id"] for pair in context["pairs"]] == [
        f"pv1-pair-{index:03d}" for index in range(2, 7)
    ]


def test_comparison_boundary_and_receiver_run_ids_are_pair_specific():
    module = load_module()
    pair = {
        "paired_group_id": "pv1-pair-005",
        "utc": "2022-01-01T09:00:00Z",
        "spoofing": {
            "start_seconds": 10.0,
            "transition_seconds": 8.0,
            "power_ramp_seconds": 12.0,
        },
    }

    assert module.comparison_start_seconds(pair) == 23.0
    assert module.pilot.multipath_run_id(pair) == (
        "cgc-rf-mp-p005_20220101T090000Z"
    )


def test_pair_block_evaluation_applies_frozen_gates():
    module = load_module()
    rows = []
    for index in range(5):
        multipath = 0.80 + 0.02 * index
        spoof = 0.20 + 0.01 * index
        rows.append({
            "clock_centered_multipath_median_residual": multipath,
            "clock_centered_spoof_median_residual": spoof,
            "legacy_separation": 0.10,
            "multipath_comparison_bin_count": 8,
            "spoof_comparison_bin_count": 8,
        })

    result = module.evaluate_pair_summaries(
        rows,
        bootstrap_seed=2026091299,
        bootstrap_repetitions=100,
        gates=module.EXPECTED_GATES,
    )

    assert result["pair_block_auc"] == 1.0
    assert result["positive_separation_pair_count"] == 5
    assert result["clock_centered_improvement_over_legacy_pair_count"] == 5
    assert result["all_support_gates_passed"] is True
    assert result["status"] == (
        "supported_on_unused_train_replication_requires_locked_test"
    )


def test_gate_drift_is_rejected():
    module = load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["evaluation"]["support_gates"]["minimum_pair_block_auc"] = 0.79

    with pytest.raises(ValueError, match="support gates drifted"):
        module.validate_config(config)
