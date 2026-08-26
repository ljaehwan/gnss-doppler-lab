import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_cgc_rf_locked_test.py"
CONFIG = ROOT / "configs" / "experiments" / "cgc_rf_locked_test_v1.json"


def load_module():
    name = "cgc_rf_locked_test_runner_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_sealed_config_verifies_only_frozen_test_metadata_and_pins():
    module = load_module()

    context = module.validate_config(frozen_config())

    assert [pair["paired_group_id"] for pair in context["pairs"]] == [
        "pv1-pair-010",
        "pv1-pair-011",
        "pv1-pair-012",
    ]
    assert all(pair["split"] == "test" for pair in context["pairs"])
    assert context["pinned_path_hash_count"] == 15
    assert context["source_root"].name == (
        "simulation_v4_paired_test_generation_v1"
    )
    assert context["output_root"].name == "cgc_rf_locked_test_v1"


def test_validate_only_does_not_require_or_start_a_release(capsys):
    module = load_module()

    assert module.main(["--validate-only"]) == 0

    output = capsys.readouterr().out
    assert "test signals were not accessed" in output


def test_frozen_gate_or_pair_drift_is_rejected():
    module = load_module()
    gate_drift = copy.deepcopy(frozen_config())
    gate_drift["evaluation"]["support_gates"][
        "minimum_pair_block_auc"
    ] = 0.79
    with pytest.raises(ValueError, match="support gates drifted"):
        module.validate_config(
            gate_drift,
            enforce_config_hash=False,
        )

    pair_drift = copy.deepcopy(frozen_config())
    pair_drift["data_boundary"]["allowed_pair_ids"][-1] = "pv1-pair-009"
    with pytest.raises(ValueError, match="pair roster drifted"):
        module.validate_config(
            pair_drift,
            enforce_config_hash=False,
        )


def test_pair_block_evaluation_applies_all_six_locked_gates():
    module = load_module()
    rows = [
        {
            "clock_centered_multipath_median_residual": 0.80,
            "clock_centered_spoof_median_residual": 0.10,
            "legacy_separation": 0.10,
            "multipath_comparison_bin_count": 9,
            "spoof_comparison_bin_count": 9,
            "startup_los_prn_count": 11,
        },
        {
            "clock_centered_multipath_median_residual": 0.70,
            "clock_centered_spoof_median_residual": 0.20,
            "legacy_separation": 0.15,
            "multipath_comparison_bin_count": 7,
            "spoof_comparison_bin_count": 7,
            "startup_los_prn_count": 10,
        },
        {
            "clock_centered_multipath_median_residual": 0.60,
            "clock_centered_spoof_median_residual": 0.30,
            "legacy_separation": 0.20,
            "multipath_comparison_bin_count": 8,
            "spoof_comparison_bin_count": 8,
            "startup_los_prn_count": 9,
        },
    ]

    result = module.evaluate_pair_summaries(
        rows,
        bootstrap_seed=2026091399,
        bootstrap_repetitions=100,
        gates=module.EXPECTED_GATES,
    )

    assert result["pair_block_auc"] == 1.0
    assert result["positive_separation_pair_count"] == 3
    assert result["all_support_gates_passed"] is True
    assert result["status"] == "SUPPORTED"
    assert set(result["gates"]) == set(module.EXPECTED_GATES)


def test_support_fails_when_startup_los_gate_fails():
    module = load_module()
    rows = [
        {
            "clock_centered_multipath_median_residual": 0.8,
            "clock_centered_spoof_median_residual": 0.1,
            "legacy_separation": 0.1,
            "multipath_comparison_bin_count": 8,
            "spoof_comparison_bin_count": 8,
            "startup_los_prn_count": los,
        }
        for los in (11, 10, 7)
    ]

    result = module.evaluate_pair_summaries(
        rows,
        bootstrap_seed=2026091399,
        bootstrap_repetitions=20,
        gates=module.EXPECTED_GATES,
    )

    gate = result["gates"]["minimum_startup_los_prns_per_pair"]
    assert gate == {"observed": 7, "required": 8, "passed": False}
    assert result["all_support_gates_passed"] is False
    assert result["status"] == "NOT_SUPPORTED"


def test_comparison_boundaries_are_fixed_by_test_scenarios():
    module = load_module()
    context = module.validate_config(frozen_config())

    assert [
        module.comparison_start_seconds(pair)
        for pair in context["pairs"]
    ] == [21.0, 23.0, 22.0]
