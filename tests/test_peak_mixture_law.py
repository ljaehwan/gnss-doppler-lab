import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.peak_mixture_law import (
    displacement_envelope_proxy,
    enu_line_of_sight,
    first_persistent_crossing,
    linear_amplitude_ratio,
    los_displacement_proxy,
    mixture_variance_excess,
    parse_gps_sdr_sim_los_table,
    robust_center_scale,
    smoothstep_progress,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "simulation_v4_peak_mixture_law_v1.json"
RUNNER = ROOT / "scripts" / "audit_simulation_v4_peak_mixture_law.py"
LOS_CONFIG = ROOT / "configs" / "experiments" / "simulation_v4_los_censoring_audit_v1.json"
LOS_RUNNER = ROOT / "scripts" / "audit_simulation_v4_los_censoring.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("simulation_v4_peak_mixture_audit", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_los_runner():
    spec = importlib.util.spec_from_file_location("simulation_v4_los_censoring", LOS_RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_mixture_variance_identity_is_exact_for_shifted_equal_shape_peaks():
    positions = np.arange(-20.0, 21.0)
    base = np.zeros_like(positions)
    base[17:24] = np.array([1, 2, 4, 6, 4, 2, 1], dtype=float)
    base /= base.sum()
    delay = 3
    rho = 0.7
    shifted = np.roll(base, delay)
    mixture = (base + rho * shifted) / (1 + rho)

    def variance(profile):
        mean = float(np.sum(positions * profile))
        return float(np.sum((positions - mean) ** 2 * profile))

    assert variance(mixture) - variance(base) == pytest.approx(
        mixture_variance_excess(rho, delay)
    )


def test_smoothstep_and_amplitude_envelope_match_simulation_contract():
    assert smoothstep_progress(9.0, 10.0, 10.0) == 0.0
    assert smoothstep_progress(15.0, 10.0, 10.0) == pytest.approx(0.5)
    assert smoothstep_progress(25.0, 10.0, 10.0) == 1.0
    assert linear_amplitude_ratio(
        9.0,
        start_s=10.0,
        ramp_s=8.0,
        initial_advantage_db=-18.0,
        final_advantage_db=2.0,
    ) == 0.0
    midpoint = linear_amplitude_ratio(
        14.0,
        start_s=10.0,
        ramp_s=8.0,
        initial_advantage_db=-18.0,
        final_advantage_db=2.0,
    )
    expected = 0.5 * (10 ** (-18 / 20) + 10 ** (2 / 20))
    assert midpoint == pytest.approx(expected)


def test_displacement_proxy_is_zero_at_onset_and_positive_after_carryoff():
    event = {
        "start_seconds": 10.0,
        "transition_seconds": 10.0,
        "target_offset_enu_m": [100.0, 50.0, 0.0],
        "initial_advantage_db": -18.0,
        "final_advantage_db": 3.0,
        "power_ramp_seconds": 8.0,
    }
    assert displacement_envelope_proxy(event, 10.0, chip_length_m=293.0) == 0.0
    assert displacement_envelope_proxy(event, 15.0, chip_length_m=293.0) > 0.0
    assert displacement_envelope_proxy(event, 20.0, chip_length_m=293.0) > 0.0


def test_enu_line_of_sight_obeys_cardinal_angle_contract():
    assert enu_line_of_sight(0.0, 0.0) == pytest.approx((0.0, 1.0, 0.0))
    assert enu_line_of_sight(90.0, 0.0) == pytest.approx((1.0, 0.0, 0.0))
    assert enu_line_of_sight(123.0, 90.0) == pytest.approx((0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="physical range"):
        enu_line_of_sight(360.0, 10.0)


def test_simulator_los_parser_and_projected_variance():
    log = """header
01  90.0  0.0  20739220.5  4.1
22  0.0  45.0  22317950.8  6.2
Time into run = 0.2
"""
    table = parse_gps_sdr_sim_los_table(log)
    assert set(table) == {"G01", "G22"}
    assert table["G01"] == pytest.approx((1.0, 0.0, 0.0))
    event = {
        "start_seconds": 10.0,
        "transition_seconds": 10.0,
        "target_offset_enu_m": [100.0, 0.0, 0.0],
        "initial_advantage_db": -18.0,
        "final_advantage_db": 2.0,
        "power_ramp_seconds": 8.0,
    }
    projected = los_displacement_proxy(
        event, 20.0, table["G01"], chip_length_m=293.0
    )
    orthogonal = los_displacement_proxy(
        event, 20.0, enu_line_of_sight(0.0, 0.0), chip_length_m=293.0
    )
    assert projected > 0.0
    assert orthogonal == pytest.approx(0.0, abs=1e-20)
    with pytest.raises(ValueError, match="no LOS table"):
        parse_gps_sdr_sim_los_table("Time into run = 0.2")
    with pytest.raises(ValueError, match="unit norm"):
        los_displacement_proxy(event, 20.0, (1.0, 1.0, 1.0), chip_length_m=293.0)


def test_v1_summary_records_use_strict_json_null_for_missing_crossings():
    module = load_runner()
    import pandas as pd

    records = module._json_records(pd.DataFrame([{"value": np.nan, "ok": 1.0}]))
    assert records == [{"value": None, "ok": 1.0}]


def test_robust_center_scale_uses_median_and_resists_outlier():
    center, scale = robust_center_scale([1.0, 1.1, 0.9, 1.0, 100.0])
    assert center == 1.0
    assert 0.0 < scale < 1.0
    with pytest.raises(ValueError, match="nonempty finite"):
        robust_center_scale([])


def test_persistent_crossing_requires_contiguous_causal_windows():
    times = [9.875, 10.0, 10.125, 10.25, 11.0, 11.125, 11.25]
    scores = [9, 2, 2, 2, 3, 3, 3]
    assert first_persistent_crossing(
        times,
        scores,
        threshold=1.0,
        onset_s=10.0,
        persistence=3,
        expected_step_s=0.125,
    ) == 10.0
    assert first_persistent_crossing(
        [10.0, 10.5, 10.625],
        [2, 2, 2],
        threshold=1.0,
        onset_s=10.0,
        persistence=3,
        expected_step_s=0.125,
    ) is None


def test_frozen_audit_config_is_train_only_and_requires_validation_confirmation():
    module = load_runner()
    config = frozen_config()
    _, record, _, split = module.validate_config(config)
    train_ids = [pair["paired_group_id"] for pair in split["pairs"] if pair["split"] == "train"]
    assert train_ids == [f"pv1-pair-{index:03d}" for index in range(1, 7)]
    assert config["data_boundary"]["allowed_pair_ids"] == train_ids
    assert config["exploratory_support_rule"]["requires_confirmation_on_validation"] is True
    assert record["decision"]["test_status"] == "locked"


def test_audit_config_rejects_validation_access_and_three_tap_drift():
    module = load_runner()
    config = frozen_config()
    validation = copy.deepcopy(config)
    validation["data_boundary"]["validation_pairs_accessed"] = True
    with pytest.raises(ValueError, match="validation access"):
        module.validate_config(validation)
    three_tap = copy.deepcopy(config)
    three_tap["tracking_contract"]["tap_count"] = 3
    with pytest.raises(ValueError, match="nine-tap"):
        module.validate_config(three_tap)


def test_los_censoring_config_preserves_failed_v1_and_train_only_boundary():
    module = load_los_runner()
    config = json.loads(LOS_CONFIG.read_text(encoding="utf-8"))
    _, source, _, artifact, _, split = module.validate_config(config, verify_outputs=True)
    train_ids = [pair["paired_group_id"] for pair in split["pairs"] if pair["split"] == "train"]
    assert source["physical_diagnostic"]["exploratory_status"] == "not_supported_on_train"
    assert config["experiment"]["analysis_origin"].startswith("post-hoc")
    assert config["data_boundary"]["allowed_pair_ids"] == train_ids
    assert set(artifact["artifacts"]) == set(train_ids)
    assert config["analysis"]["strict_minimum_prns"] == 4
    assert config["analysis"]["censoring_aware_minimum_prns"] == 1
    assert config["confirmation"]["formula_and_policies_must_freeze_before_validation"] is True


def test_los_censoring_config_rejects_boundary_and_policy_drift():
    module = load_los_runner()
    config = json.loads(LOS_CONFIG.read_text(encoding="utf-8"))
    leaked = copy.deepcopy(config)
    leaked["data_boundary"]["texbat_recordings_accessed"] = ["ds2"]
    with pytest.raises(ValueError, match="TEXBAT"):
        module.validate_config(leaked)
    drifted = copy.deepcopy(config)
    drifted["analysis"]["censoring_aware_minimum_prns"] = 2
    with pytest.raises(ValueError, match="censoring analysis"):
        module.validate_config(drifted)
    relabeled = copy.deepcopy(config)
    relabeled["experiment"]["analysis_origin"] = "confirmatory"
    with pytest.raises(ValueError, match="post-hoc"):
        module.validate_config(relabeled)
