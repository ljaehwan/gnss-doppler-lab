import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_simulation_v4_normal_calibration.py"
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "simulation_v4_normal_calibration_v1.json"
CONFIG_V2 = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "simulation_v4_normal_calibration_v2.json"
CONFIG_V3 = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "simulation_v4_normal_calibration_v3.json"
CONFIG_V4 = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "simulation_v4_normal_calibration_v4.json"


def load_module():
    spec = importlib.util.spec_from_file_location("simulation_v4_normal_calibration", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equivalent_cn0_preserves_reference_across_sample_rates():
    module = load_module()
    for sample_rate in (2_600_000, 5_200_000, 10_400_000, 25_000_000):
        sample_snr = module.equivalent_sample_snr_db(57.0, sample_rate)
        assert sample_snr + 10 * math.log10(sample_rate) == pytest.approx(57.0)
    assert module.equivalent_sample_snr_db(57.0, 25_000_000) < module.equivalent_sample_snr_db(57.0, 2_600_000)


def test_candidate_phase_noise_is_scaled_to_constant_per_second_strength():
    module = load_module()
    config = json.loads(CONFIG.read_text())
    low = next(row for row in config["candidates"] if row["rf_sample_rate_hz"] == 2_600_000)
    high = next(row for row in config["candidates"] if row["rf_sample_rate_hz"] == 10_400_000)
    low_cfg = module._candidate_impairment(config, low)
    high_cfg = module._candidate_impairment(config, high)
    assert low_cfg.phase_noise_std_rad_per_sqrt_sample == pytest.approx(2e-5)
    assert high_cfg.phase_noise_std_rad_per_sqrt_sample == pytest.approx(1e-5)
    assert low_cfg.phase_noise_std_rad_per_sqrt_sample**2 * low["rf_sample_rate_hz"] == pytest.approx(
        high_cfg.phase_noise_std_rad_per_sqrt_sample**2 * high["rf_sample_rate_hz"]
    )


def test_frozen_config_validates_candidates_boundary_and_weights():
    module = load_module()
    config = json.loads(CONFIG.read_text())
    module._validate_config(config)
    module._validate_config(json.loads(CONFIG_V2.read_text()))
    module._validate_config(json.loads(CONFIG_V3.read_text()))
    module._validate_config(json.loads(CONFIG_V4.read_text()))

    duplicate = copy.deepcopy(config)
    duplicate["candidates"][1]["name"] = duplicate["candidates"][0]["name"]
    with pytest.raises(ValueError, match="unique"):
        module._validate_config(duplicate)

    bad_weights = copy.deepcopy(config)
    bad_weights["ranking_weights"]["domain_separation"] = 0.4
    with pytest.raises(ValueError, match="sum to one"):
        module._validate_config(bad_weights)


def test_fidelity_loss_uses_domain_metrics_and_receiver_target_ranges():
    module = load_module()
    config = json.loads(CONFIG.read_text())
    result = {
        "domain_classifier": {"pooled_separability_auc": 0.75},
        "distribution": {
            "median_ks_statistic": 0.25,
            "median_robust_median_shift": 0.75,
        },
    }
    in_range = {
        "carrier_lock_above_0_5_fraction": 0.72,
        "cn0_db_hz_median": 46.0,
    }
    loss, terms = module.fidelity_loss(result, in_range, config)
    assert terms == {
        "domain_separation": 0.5,
        "median_ks": 0.25,
        "median_robust_shift": 0.5,
        "lock_range_distance": 0.0,
        "cn0_range_distance": 0.0,
    }
    assert loss == pytest.approx(0.4)

    out_of_range = dict(in_range, carrier_lock_above_0_5_fraction=0.60, cn0_db_hz_median=40.0)
    worse_loss, worse_terms = module.fidelity_loss(result, out_of_range, config)
    assert worse_terms["lock_range_distance"] > 0
    assert worse_terms["cn0_range_distance"] > 0
    assert worse_loss > loss


def test_external_component_is_hash_verified_and_reused(tmp_path):
    module = load_module()
    source = tmp_path / "source.bin"
    source.write_bytes(b"authentic-iq")
    config = json.loads(CONFIG.read_text())
    config["authentic_components"] = {
        "2600000": {"path": str(source), "sha256": module._sha256(source)}
    }
    path, manifest = module._component(
        config,
        "unused-config-hash",
        tmp_path / "output",
        2_600_000,
        None,
        resume=False,
    )
    assert path == source
    assert manifest["reused_external_component"] is True
    assert manifest["iq_sha256"] == module._sha256(source)


def test_candidate_resume_keeps_generated_rf_manifest_immutable(tmp_path):
    module = load_module()
    config = json.loads(CONFIG.read_text())
    candidate = config["candidates"][0]
    timestamp = module._campaign_datetime(config).strftime("%Y%m%dT%H%M%SZ")
    storage_id = f"{config['campaign']['name']}-{candidate['name']}_{timestamp}"
    run_dir = tmp_path / "rf" / storage_id
    run_dir.mkdir(parents=True)
    iq_path = run_dir / "gps_l1ca_s8_iq.bin"
    iq_path.write_bytes(b"frozen-candidate-iq")
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "run_id": "original-run-id",
        "iq": {"path": str(iq_path), "sha256": module._sha256(iq_path)},
        "calibration": {
            "config_sha256": "frozen-config",
            "candidate": candidate,
            "runner_script_sha256": "generation-time-runner-hash",
        },
    }, sort_keys=True))
    frozen_bytes = manifest_path.read_bytes()

    observed = module._candidate_rf(
        config, "frozen-config", tmp_path, candidate, tmp_path / "unused.bin", {}, resume=True
    )

    assert observed == manifest_path
    assert manifest_path.read_bytes() == frozen_bytes


def test_receiver_iq_alias_is_a_no_copy_hard_link(tmp_path):
    module = load_module()
    storage = tmp_path / "storage.bin"
    storage.write_bytes(b"candidate-iq")
    alias = module._receiver_iq_alias(tmp_path / "output", "candidate", storage)
    assert alias.read_bytes() == storage.read_bytes()
    assert alias.samefile(storage)
    assert len(module._receiver_run_id("candidate", "20220101T000000Z")) < 64


def test_executable_path_resolves_path_name_and_repo_relative(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/gnss-sdr" if name == "gnss-sdr" else None)
    assert module._executable_path("gnss-sdr") == "/usr/bin/gnss-sdr"
    expected = SCRIPT.parents[1] / ".tools" / "gnss-sdr-method-a-9tap"
    assert module._executable_path(".tools/gnss-sdr-method-a-9tap") == str(expected.resolve())


def test_receiver_variant_suffix_is_safe():
    module = load_module()
    assert module._variant_suffix("") == ""
    assert module._variant_suffix("spacing0125") == "_spacing0125"
    with pytest.raises(ValueError, match="receiver variant"):
        module._variant_suffix("../escape")
