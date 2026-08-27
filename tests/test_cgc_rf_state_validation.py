import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_cgc_rf_state_validation.py"
SPEC = importlib.util.spec_from_file_location("run_cgc_rf_state_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


GATES = {
    "unresolved_40m_max_auc": 0.8,
    "onset_60m_max_auc": 0.8,
    "onset_60m_min_absolute_direction_cosine": 0.7,
    "onset_direction_must_improve_over_40m": True,
    "metric_distances_m": [100.0, 160.0],
    "metric_min_auc": 0.8,
    "metric_min_absolute_direction_cosine": 0.85,
    "metric_max_absolute_relative_displacement_error": 0.15,
    "saturation_distance_m": 240.0,
    "saturation_min_auc": 0.8,
    "saturation_min_edge_fraction": 0.1,
    "saturation_max_relative_displacement_error": -0.05,
    "overall_rule": "every gate passes independently in all four geometry-power groups",
}


def row(distance, auc, direction, relative_error, edge=0.0):
    truth = distance / 293.0522561094819
    return {
        "distance_m": distance,
        "distance_chips": truth,
        "serial_bin_auc": auc,
        "median_absolute_direction_cosine": direction,
        "median_estimated_displacement_norm_chips": truth * (1.0 + relative_error),
        "template_delay_edge_fraction": edge,
        "spoof_bin_count": 12,
        "multipath_bin_count": 12,
        "minimum_spoof_prn_count": 10,
        "minimum_multipath_prn_count": 10,
    }


def passing_rows():
    return [
        row(40.0, 0.3, 0.4, 0.5),
        row(60.0, 0.6, 0.8, 0.1),
        row(80.0, 0.7, 0.82, 0.08),
        row(100.0, 0.95, 0.95, 0.03),
        row(160.0, 0.98, 0.96, -0.04),
        row(240.0, 0.99, 0.94, -0.12, 0.25),
    ]


def test_roster_has_two_held_out_geometries_and_24_cells():
    rows = MODULE.condition_specs()
    assert len(rows) == 24
    assert {entry["geometry_id"] for entry in rows} == {"straight", "sweep"}
    assert all(not entry["condition_id"].startswith("static") for entry in rows)


def test_all_cells_use_constant_pull_off_rate():
    for entry in MODULE.condition_specs():
        assert entry["transition_seconds"] == entry["distance_m"] / 20.0
        assert 5.0 + entry["transition_seconds"] <= 17.0


def test_complete_state_group_passes_every_gate():
    result = MODULE.evaluate_state_group(passing_rows(), GATES, minimum_bins=8)
    assert result["state_map_reproduced"] is True
    assert all(result["gates"].values())


def test_state_group_fails_when_metric_bias_is_too_large():
    rows = passing_rows()
    rows[3] = row(100.0, 0.95, 0.95, -0.2)
    result = MODULE.evaluate_state_group(rows, GATES, minimum_bins=8)
    assert result["state_map_reproduced"] is False
    assert result["gates"]["metric_displacement_error"] is False


def test_80m_transition_has_no_pass_gate():
    rows = passing_rows()
    rows[2] = row(80.0, 0.05, 0.1, 2.0, 0.9)
    result = MODULE.evaluate_state_group(rows, GATES, minimum_bins=8)
    assert result["state_map_reproduced"] is True
    assert result["transition_80m"]["auc"] == 0.05


def test_incomplete_state_group_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="complete six-distance grid"):
        MODULE.evaluate_state_group(passing_rows()[:-1], GATES, minimum_bins=8)
