import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_cgc_rf_geometry_aperture_validation.py"
SPEC = importlib.util.spec_from_file_location("run_cgc_rf_geometry_aperture_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


GATES = {
    "metric_min_auc": 0.8,
    "metric_min_absolute_direction_cosine": 0.85,
    "metric_max_absolute_relative_displacement_error": 0.15,
    "saturation_min_auc": 0.8,
    "saturation_min_absolute_direction_cosine": 0.85,
    "saturation_min_edge_fraction": 0.1,
    "saturation_max_relative_displacement_error": -0.05,
}


def row(distance, taps=9, *, auc=0.95, direction=0.95, relative_error=0.02, edge=0.0, geometry="denver-static", power=-6.0):
    truth = distance / 293.0522561094819
    return {
        "geometry_id": geometry, "final_advantage_db": power, "distance_m": distance,
        "distance_chips": truth, "aperture_taps": taps, "serial_bin_auc": auc,
        "median_absolute_direction_cosine": direction,
        "median_estimated_displacement_norm_chips": truth * (1.0 + relative_error),
        "template_delay_edge_fraction": edge, "spoof_bin_count": 12, "multipath_bin_count": 12,
        "minimum_spoof_prn_count": 9, "minimum_multipath_prn_count": 9,
    }


def passing_geometry_rows():
    return [
        row(40.0, auc=0.2, direction=0.3, relative_error=1.0),
        row(60.0, auc=0.6, direction=0.7, relative_error=0.4),
        row(100.0, relative_error=0.04),
        row(240.0, relative_error=-0.2, edge=0.3),
    ]


def test_roster_has_five_geometries_four_distances_two_powers():
    specs = MODULE.condition_specs()
    assert len(specs) == 40
    assert {entry["geometry_id"] for entry in specs} == set(MODULE.GEOMETRY_IDS)
    assert {entry["distance_m"] for entry in specs} == {40.0, 60.0, 100.0, 240.0}
    assert {entry["final_advantage_db"] for entry in specs} == {-6.0, 3.0}


def test_receiver_run_ids_are_unique_and_short():
    ids = {MODULE.receiver_run_id(entry["condition_id"]) for entry in MODULE.condition_specs()}
    assert len(ids) == 40
    assert max(map(len, ids)) < 24


def test_early_boundary_has_no_universal_gate():
    result = MODULE.evaluate_geometry_group(passing_geometry_rows(), GATES, minimum_bins=8)
    assert result["mechanism_reproduced"] is True
    assert "40m_unresolved_auc" not in result["gates"]


def test_metric_and_saturation_gates_are_independent():
    rows = passing_geometry_rows()
    rows[2] = row(100.0, relative_error=0.2)
    result = MODULE.evaluate_geometry_group(rows, GATES, minimum_bins=8)
    assert result["mechanism_reproduced"] is False
    assert result["gates"]["100m_metric_error"] is False
    assert result["gates"]["240m_saturation_bias"] is True


def aperture_rows(*, violate=False):
    result = []
    errors_100 = {3: 0.10, 5: 0.07, 9: 0.04}
    errors_240 = {3: -0.45, 5: -0.30, 9: -0.18}
    edges = {3: 0.7, 5: 0.5, 9: 0.3}
    for geometry in MODULE.GEOMETRY_IDS:
        for power in MODULE.POWERS_DB:
            for taps in MODULE.APERTURE_TAPS:
                result.append(row(100.0, taps, relative_error=errors_100[taps], geometry=geometry, power=power))
                error = -0.55 if violate and taps == 9 else errors_240[taps]
                result.append(row(240.0, taps, relative_error=error, edge=edges[taps], geometry=geometry, power=power))
    return result


def test_aperture_mechanism_passes_paired_monotone_pattern():
    result = MODULE.evaluate_aperture_mechanism(aperture_rows())
    assert result["aperture_mechanism_supported"] is True
    assert all(result["gates"].values())


def test_aperture_mechanism_reports_nonmonotone_failure():
    result = MODULE.evaluate_aperture_mechanism(aperture_rows(violate=True))
    assert result["aperture_mechanism_supported"] is False
    assert result["gates"]["240m_absolute_error_nonincreasing_3_to_5_to_9"] is False
