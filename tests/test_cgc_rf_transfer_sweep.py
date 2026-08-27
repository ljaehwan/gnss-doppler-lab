import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_cgc_rf_transfer_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_cgc_rf_transfer_sweep", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_complete_distance_power_roster_is_distance_major():
    rows = MODULE.condition_specs()
    assert len(rows) == 18
    assert [row["condition_id"] for row in rows[:4]] == [
        "pneg6-d020", "ppos3-d020", "pneg6-d040", "ppos3-d040"
    ]
    assert [row["condition_id"] for row in rows[-2:]] == ["pneg6-d240", "ppos3-d240"]


def test_constant_pull_off_rate_and_common_settle_bound():
    for row in MODULE.condition_specs():
        assert row["transition_seconds"] == row["distance_m"] / 20.0
        assert 5.0 + row["transition_seconds"] <= 17.0


def test_contiguous_observability_intervals_are_not_filled_across_gaps():
    assert MODULE.contiguous_intervals(
        [20.0, 40.0, 60.0, 80.0, 100.0],
        [False, True, True, False, True],
    ) == [[40.0, 60.0], [100.0, 100.0]]


def test_curve_reports_finite_window_and_post_peak_decline():
    aucs = [0.55, 0.7, 0.82, 0.9, 0.88, 0.81, 0.75, 0.65, 0.6]
    rows = [
        {
            "distance_m": distance,
            "serial_bin_auc": auc,
            "spoof_bin_count": 12,
            "multipath_bin_count": 12,
            "template_delay_edge_fraction": distance / 1000.0,
        }
        for distance, auc in zip(MODULE.DISTANCES_M, aucs)
    ]
    result = MODULE.evaluate_curve(rows, threshold=0.8, minimum_bins=8)
    assert result["peak_distance_m"] == 80.0
    assert result["first_tested_threshold_crossing_m"] == 60.0
    assert result["above_threshold_grid_intervals_m"] == [[60.0, 120.0]]
    assert result["post_peak_decline_at_final_cell"] is True
    assert result["strictly_increasing_over_full_grid"] is False


def test_curve_rejects_incomplete_grid():
    import pytest

    with pytest.raises(ValueError, match="complete frozen distance grid"):
        MODULE.evaluate_curve([], threshold=0.8, minimum_bins=8)
