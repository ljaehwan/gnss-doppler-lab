import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_simulation_v4_normal_independent_validation.py"
CONFIG = ROOT / "configs" / "experiments" / "simulation_v4_normal_independent_validation_v1.json"


def load_module():
    spec = importlib.util.spec_from_file_location("simulation_v4_normal_independent_validation", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_frozen_config_has_independent_static_and_dynamic_coverage():
    module = load_module()
    config = frozen_config()
    module._validate_config(config)
    assert [run["domain"] for run in config["runs"]].count("static") == 2
    assert [run["domain"] for run in config["runs"]].count("dynamic") == 3
    assert {run["motion"]["kind"] for run in config["runs"] if run["domain"] == "dynamic"} == {
        "straight", "circle", "parallel-sweep"
    }
    assert len({run["receiver_seed"] for run in config["runs"]}) == 5


def test_validator_rejects_profile_motion_and_seed_drift():
    module = load_module()
    config = frozen_config()

    wrong_taps = copy.deepcopy(config)
    wrong_taps["gnss_sdr"]["tracking_tap_count"] = 3
    with pytest.raises(ValueError, match="9-tap"):
        module._validate_config(wrong_taps)

    missing_motion = copy.deepcopy(config)
    del missing_motion["runs"][2]["motion"]
    with pytest.raises(ValueError, match="must define motion"):
        module._validate_config(missing_motion)

    duplicate_seed = copy.deepcopy(config)
    duplicate_seed["runs"][1]["receiver_seed"] = duplicate_seed["runs"][0]["receiver_seed"]
    with pytest.raises(ValueError, match="unique"):
        module._validate_config(duplicate_seed)


def test_run_impairment_preserves_cn0_and_per_second_phase_noise():
    module = load_module()
    config = frozen_config()
    sample_rate = config["rf_profile"]["rf_sample_rate_hz"]
    reference_rate = config["receiver"]["reference_sample_rate_hz"]
    reference_phase_noise = config["receiver"]["phase_noise_std_rad_per_sqrt_sample"]
    for run in config["runs"]:
        impairment = module._run_impairment(config, run)
        reconstructed_cn0 = impairment.sample_snr_db + 10 * math.log10(sample_rate)
        assert reconstructed_cn0 == pytest.approx(run["target_composite_cn0_db_hz"])
        assert impairment.seed == run["receiver_seed"]
        assert impairment.phase_noise_std_rad_per_sqrt_sample == pytest.approx(
            reference_phase_noise * math.sqrt(reference_rate / sample_rate)
        )


def test_receiver_state_gate_has_pass_conditional_and_stop_levels():
    module = load_module()
    config = frozen_config()

    passed = module.receiver_state_gate({
        "a": {"carrier_lock_above_0_5_fraction": 0.72, "cn0_db_hz_median": 46.0}
    }, config)
    assert passed["gate_status"] == "pass"

    conditional = module.receiver_state_gate({
        "a": {"carrier_lock_above_0_5_fraction": 0.85, "cn0_db_hz_median": 46.0}
    }, config)
    assert conditional["gate_status"] == "conditional"

    stopped = module.receiver_state_gate({
        "a": {"carrier_lock_above_0_5_fraction": 0.95, "cn0_db_hz_median": 46.0}
    }, config)
    assert stopped["gate_status"] == "stop"


def test_dynamic_trajectory_is_reproducible_and_has_exact_contract(tmp_path):
    module = load_module()
    config = frozen_config()
    run = next(run for run in config["runs"] if run["name"] == "iv-d-london-circle")
    trajectory, record = module._trajectory_position(tmp_path, run)
    assert len(trajectory.rows) == 300
    assert record["row_count"] == 300
    assert record["sha256"] == module._sha256(record["path"])
    assert record["metadata_sha256"] == module._sha256(record["metadata_path"])

    first_bytes = Path(record["path"]).read_bytes()
    _, repeated = module._trajectory_position(tmp_path, run)
    assert Path(repeated["path"]).read_bytes() == first_bytes
    assert repeated["sha256"] == record["sha256"]


def test_receiver_run_id_stays_below_gnss_sdr_path_risk_limit():
    module = load_module()
    run = frozen_config()["runs"][-1]
    assert len(module._receiver_run_id(run)) < 64
