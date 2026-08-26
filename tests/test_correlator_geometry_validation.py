import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "correlator_geometry_identifiability_validation_v1.json"
RUNNER = ROOT / "scripts" / "validate_correlator_geometry_identifiability.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("correlator_geometry_validation", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_validation_contract_pins_supported_train_candidate_and_only_007_to_009():
    module = load_runner()
    config = frozen_config()
    _, train, _, record, _, artifact, _, split = module.validate_config(
        config, verify_inputs=True
    )
    validation_ids = [
        pair["paired_group_id"]
        for pair in split["pairs"]
        if pair["split"] == "validation"
    ]
    assert validation_ids == ["pv1-pair-007", "pv1-pair-008", "pv1-pair-009"]
    assert config["data_boundary"]["allowed_pair_ids"] == validation_ids
    assert record["status"] == "supported_on_train_requires_validation"
    assert artifact["candidate_status"] == "supported_on_train_requires_validation"
    assert config["frozen_support_rule"] == train["exploratory_support_rule"]
    assert config["data_boundary"]["test_pairs_accessed"] is False
    assert config["data_boundary"]["texbat_recordings_accessed"] == []


def test_analysis_config_changes_only_seed_bootstrap_and_validation_boundary():
    module = load_runner()
    config = frozen_config()
    _, train, *_ = module.validate_config(config)
    analysis = module.build_analysis_config(config, train)
    assert analysis["correlator"] == train["correlator"]
    assert analysis["template_estimator"] == train["template_estimator"]
    assert analysis["exploratory_support_rule"] == train["exploratory_support_rule"]
    train_generator = copy.deepcopy(train["generator"])
    analysis_generator = copy.deepcopy(analysis["generator"])
    assert analysis_generator.pop("seed") == 2026082701
    train_generator.pop("seed")
    assert analysis_generator == train_generator
    assert analysis["evaluation"]["bootstrap_seed"] == 2026082702
    assert analysis["data_boundary"]["allowed_pair_ids"] == [
        "pv1-pair-007", "pv1-pair-008", "pv1-pair-009"
    ]
    assert analysis["data_boundary"]["test_pairs_accessed"] is False


def test_validation_contract_rejects_threshold_seed_and_test_access_drift():
    module = load_runner()
    config = frozen_config()
    threshold = copy.deepcopy(config)
    threshold["frozen_support_rule"]["complex_geometry_auc_min"] = 0.79
    with pytest.raises(ValueError, match="thresholds differ"):
        module.validate_config(threshold)
    seed = copy.deepcopy(config)
    seed["validation_randomness"]["generator_seed"] += 1
    with pytest.raises(ValueError, match="randomness"):
        module.validate_config(seed)
    test_access = copy.deepcopy(config)
    test_access["data_boundary"]["test_pairs_accessed"] = True
    with pytest.raises(ValueError, match="test access"):
        module.validate_config(test_access)


def test_validation_contract_rejects_overstated_receiver_or_rf_claims():
    module = load_runner()
    config = frozen_config()
    for key in (
        "actual_multipath_rf_generated",
        "actual_receiver_tracking_evaluated",
        "actual_receiver_complex_taps_evaluated",
    ):
        drifted = copy.deepcopy(config)
        drifted["claim_boundary"][key] = True
        with pytest.raises(ValueError, match="claim boundary"):
            module.validate_config(drifted)
