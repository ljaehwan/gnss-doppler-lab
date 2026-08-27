import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_cgc_rf_observability_anchors.py"
SPEC = importlib.util.spec_from_file_location("run_cgc_rf_observability_anchors", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_condition_id_and_roster_contract():
    assert MODULE.expected_condition_ids() == [
        "pneg6-d040", "pneg6-d060", "pneg6-d080",
        "ppos3-d040", "ppos3-d060", "ppos3-d080",
    ]


def test_power_regime_evaluation_accepts_predicted_ordering():
    rows = [
        {"separation_m": 40.0, "serial_bin_auc": 0.7, "multipath_bin_count": 9, "spoof_bin_count": 9},
        {"separation_m": 60.0, "serial_bin_auc": 0.85, "multipath_bin_count": 9, "spoof_bin_count": 9},
        {"separation_m": 80.0, "serial_bin_auc": 0.95, "multipath_bin_count": 9, "spoof_bin_count": 9},
    ]
    result = MODULE.evaluate_power_regime(rows, threshold=0.8, minimum_bins=8)
    assert result["ordering_reproduced"] is True
    assert all(result["gates"].values())


def test_power_regime_evaluation_rejects_nonmonotonic_rf_auc():
    rows = [
        {"separation_m": 40.0, "serial_bin_auc": 0.7, "multipath_bin_count": 9, "spoof_bin_count": 9},
        {"separation_m": 60.0, "serial_bin_auc": 0.9, "multipath_bin_count": 9, "spoof_bin_count": 9},
        {"separation_m": 80.0, "serial_bin_auc": 0.85, "multipath_bin_count": 9, "spoof_bin_count": 9},
    ]
    result = MODULE.evaluate_power_regime(rows, threshold=0.8, minimum_bins=8)
    assert result["ordering_reproduced"] is False
    assert result["gates"]["strict_auc_ordering"] is False


def test_comparison_start_is_frozen_after_both_ramps():
    assert MODULE.comparison_start_seconds({
        "start_seconds": 10.0,
        "transition_seconds": 5.0,
        "power_ramp_seconds": 5.0,
    }) == 16.0
