import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_cgc_observability_boundary.py"
SPEC = importlib.util.spec_from_file_location("run_cgc_observability_boundary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_dominant_secondary_preserves_delay_when_authentic_is_stronger():
    delay, ratio, owner = MODULE.dominant_secondary(np.asarray([-0.2, 0.1]), -6.0)
    assert np.allclose(delay, [-0.2, 0.1])
    assert np.isclose(ratio, 10 ** (-6 / 20))
    assert owner == "authentic_prompt"


def test_dominant_secondary_flips_delay_when_counterfeit_is_stronger():
    delay, ratio, owner = MODULE.dominant_secondary(np.asarray([-0.2, 0.1]), 6.0)
    assert np.allclose(delay, [0.2, -0.1])
    assert np.isclose(ratio, 10 ** (-6 / 20))
    assert owner == "counterfeit_prompt"


def test_dominant_secondary_rejects_equal_power_prompt_ambiguity():
    try:
        MODULE.dominant_secondary(np.asarray([0.1]), 0.0)
    except ValueError as error:
        assert "nonzero" in str(error)
    else:
        raise AssertionError("zero-dB ambiguity must be rejected")


def test_nuisance_design_uses_unit_directions_and_derangements():
    generator = {
        "vertical_direction_abs_max": 0.25,
        "authentic_center_std_chips": 0.02,
        "authentic_center_abs_max_chips": 0.06,
        "relative_phase_min_rad": -np.pi,
        "relative_phase_max_rad": np.pi,
        "complex_noise_std_min": 0.005,
        "complex_noise_std_max": 0.025,
    }
    design = MODULE.nuisance_design(8, 5, generator, 123)
    assert np.allclose(np.linalg.norm(design["directions"], axis=1), 1.0)
    identity = np.arange(8)
    assert all(np.all(row != identity) for row in design["permutations"])


def test_boundary_rows_preserve_nonmonotonic_observability():
    cells = [
        {"counterfeit_advantage_db": -6.0, "spatial_separation_m": 10.0, "spatial_separation_chips": 0.1, "tracking_owner": "authentic_prompt", "secondary_to_primary_amplitude_ratio": 0.5, "observable": False},
        {"counterfeit_advantage_db": -6.0, "spatial_separation_m": 20.0, "spatial_separation_chips": 0.2, "tracking_owner": "authentic_prompt", "secondary_to_primary_amplitude_ratio": 0.5, "observable": True},
        {"counterfeit_advantage_db": -6.0, "spatial_separation_m": 40.0, "spatial_separation_chips": 0.4, "tracking_owner": "authentic_prompt", "secondary_to_primary_amplitude_ratio": 0.5, "observable": False},
    ]
    row = MODULE.boundary_rows(cells, [-6.0])[0]
    assert row["minimum_observable_separation_m"] == 20.0
    assert row["largest_nonobservable_below_boundary_m"] == 10.0
    assert row["monotone_observable_at_and_above_boundary"] is False
