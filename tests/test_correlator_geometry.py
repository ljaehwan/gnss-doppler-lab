import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.correlator_geometry import (
    TemplateDelayEstimator,
    build_complex_template_bank,
    build_template_bank,
    complex_profile_features,
    fit_common_geometry,
    profile_width_variance,
    random_derangement,
    triangular_correlation,
    two_path_complex_profile,
    two_path_magnitude_profile,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "correlator_geometry_identifiability_train_v1.json"
RUNNER = ROOT / "scripts" / "audit_correlator_geometry_identifiability.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("correlator_geometry_audit", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_triangular_correlation_obeys_ideal_ca_support():
    offsets = np.asarray([-1.5, -1.0, -0.25, 0.0, 0.25, 1.0, 1.5])
    assert triangular_correlation(offsets) == pytest.approx([0, 0, 0.75, 1, 0.75, 0, 0])
    with pytest.raises(ValueError, match="finite"):
        triangular_correlation([0.0, np.nan])


def test_complex_profile_features_remove_only_common_carrier_phase():
    taps = np.arange(-0.5, 0.5001, 0.125)
    profile = two_path_complex_profile(
        taps,
        authentic_center_chips=0.0,
        secondary_delay_chips=0.25,
        secondary_amplitude_ratio=0.4,
        relative_phase_rad=0.7,
    )
    rotated = profile * np.exp(1j * 1.234)
    assert complex_profile_features(profile, prompt_index=4) == pytest.approx(
        complex_profile_features(rotated, prompt_index=4), abs=1e-12
    )
    assert two_path_magnitude_profile(
        taps,
        authentic_center_chips=0.0,
        secondary_delay_chips=0.25,
        secondary_amplitude_ratio=0.4,
        relative_phase_rad=0.7,
    ) == pytest.approx(np.abs(profile))


def test_noiseless_on_grid_template_recovers_signed_delay():
    taps = np.arange(-0.5, 0.5001, 0.125)
    axes = {
        "delays_chips": [-0.25, 0.25],
        "centers_chips": [0.0],
        "amplitude_ratios": [0.4],
        "phases_rad": [0.7],
    }
    profile = two_path_complex_profile(
        taps,
        authentic_center_chips=0.0,
        secondary_delay_chips=0.25,
        secondary_amplitude_ratio=0.4,
        relative_phase_rad=0.7,
    )
    complex_bank = build_complex_template_bank(taps, prompt_index=4, **axes)
    complex_estimator = TemplateDelayEstimator(complex_bank)
    delay, distance, _ = complex_estimator.estimate(
        complex_profile_features(profile, prompt_index=4)
    )
    assert delay[0] == pytest.approx(0.25)
    assert distance[0] == pytest.approx(0.0, abs=1e-12)

    magnitude_bank = build_template_bank(taps, **axes)
    magnitude_estimator = TemplateDelayEstimator(magnitude_bank)
    magnitude_delay, magnitude_distance, _ = magnitude_estimator.estimate(np.abs(profile))
    assert magnitude_delay[0] == pytest.approx(0.25)
    assert magnitude_distance[0] == pytest.approx(0.0, abs=1e-12)


def test_common_geometry_is_exact_and_prn_derangement_breaks_it():
    los = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=float)
    los[-1] /= np.linalg.norm(los[-1])
    theta = np.asarray([0.2, -0.1, 0.04, 0.02])
    delays = np.column_stack((-los, np.ones(len(los)))) @ theta
    exact = fit_common_geometry(los, delays)
    permuted = fit_common_geometry(los, delays[[1, 2, 3, 4, 5, 0]])
    assert exact.rank == 4
    assert exact.theta == pytest.approx(theta, abs=1e-12)
    assert exact.normalized_residual == pytest.approx(0.0, abs=1e-20)
    assert exact.coherence == pytest.approx(1.0)
    assert permuted.normalized_residual > 0.05


def test_derangement_and_width_preserve_exact_single_profile_multiset():
    module = load_runner()
    rng = np.random.default_rng(17)
    profiles = np.arange(54, dtype=float).reshape(6, 9) + 1.0
    permutation = random_derangement(len(profiles), rng)
    assert np.all(permutation != np.arange(len(profiles)))
    assert sorted(permutation.tolist()) == list(range(len(profiles)))
    assert module._exact_row_multiset_match(profiles, profiles[permutation])
    widths = profile_width_variance(profiles, np.arange(9, dtype=float))
    assert sorted(widths) == pytest.approx(sorted(widths[permutation]))
    with pytest.raises(ValueError, match="at least two"):
        random_derangement(1, rng)


def test_frozen_config_is_train_only_and_pins_complex_observability_ladder():
    module = load_runner()
    config = frozen_config()
    _, source, _, split = module.validate_config(config, verify_source_artifacts=True)
    train_ids = [pair["paired_group_id"] for pair in split["pairs"] if pair["split"] == "train"]
    assert config["data_boundary"]["allowed_pair_ids"] == train_ids
    assert set(config["los_sources"]) == set(train_ids)
    assert set(source["artifacts"]) == set(train_ids)
    assert config["correlator"]["prompt_index"] == 4
    assert set(config["correlator"]["observation_modes"]) == {"magnitude_9tap", "complex_9tap"}
    assert config["exploratory_support_rule"]["requires_validation_confirmation"] is True
    assert config["claim_boundary"]["actual_receiver_complex_taps_evaluated"] is False


def test_config_rejects_partition_tap_and_claim_boundary_drift():
    module = load_runner()
    config = frozen_config()
    leaked = copy.deepcopy(config)
    leaked["data_boundary"]["validation_pairs_accessed"] = True
    with pytest.raises(ValueError, match="validation access"):
        module.validate_config(leaked)
    three_tap = copy.deepcopy(config)
    three_tap["correlator"]["tap_offsets_chips"] = [-0.125, 0.0, 0.125]
    with pytest.raises(ValueError, match="nine-tap"):
        module.validate_config(three_tap)
    overstated = copy.deepcopy(config)
    overstated["claim_boundary"]["actual_receiver_complex_taps_evaluated"] = True
    with pytest.raises(ValueError, match="claim boundary"):
        module.validate_config(overstated)
