import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_simulation_v4_paired_train_generation.py"
CONFIG = ROOT / "configs" / "experiments" / "simulation_v4_paired_train_generation_v1.json"


def load_module():
    spec = importlib.util.spec_from_file_location("simulation_v4_paired_train_generation", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_frozen_generation_config_accesses_only_six_train_pairs():
    module = load_module()
    config = frozen_config()
    _, split, _, manifest, _, normal = module._validate_generation_config(config)
    assert [pair["paired_group_id"] for pair in split["pairs"] if pair["split"] == "train"] == [
        f"pv1-pair-{index:03d}" for index in range(1, 7)
    ]
    assert manifest["test_release"]["status"] == "locked"
    assert config["data_boundary"]["validation_pairs_accessed"] is False
    assert config["data_boundary"]["test_pairs_accessed"] is False
    assert normal["gnss_sdr"]["tracking_tap_count"] == 9
    assert normal["features"]["tap_count"] == 3


def test_generation_config_rejects_non_train_or_test_access():
    module = load_module()
    config = frozen_config()

    validation = copy.deepcopy(config)
    validation["campaign"]["partition"] = "validation"
    with pytest.raises(ValueError, match="hard-limited to the train"):
        module._validate_generation_config(validation)

    test_access = copy.deepcopy(config)
    test_access["data_boundary"]["test_pairs_accessed"] = True
    with pytest.raises(ValueError, match="test access flag"):
        module._validate_generation_config(test_access)


def test_static_and_dynamic_carryoff_trajectories_have_exact_contract(tmp_path):
    module = load_module()
    _, split, _, _, _, _ = module._validate_generation_config(frozen_config())
    for pair in (split["pairs"][0], split["pairs"][4]):
        component_dir = tmp_path / pair["paired_group_id"]
        component_dir.mkdir()
        _, authentic, record = module._trajectory_position(component_dir, pair)
        event = module._spoof_event(pair)
        counterfeit, counterfeit_record = module._counterfeit_position(
            component_dir, authentic, event
        )
        assert len(authentic) == len(counterfeit.rows) == 300
        assert counterfeit_record["row_count"] == 300
        assert module._sha256(counterfeit_record["path"]) == counterfeit_record["sha256"]
        realized = counterfeit_record["realized_final_spoof_offset_from_authentic_enu_m"]
        assert realized == pytest.approx(event.target_offset_enu_m, abs=0.05)
        if pair["domain"] == "static":
            assert record is None
        else:
            assert record["sha256"] == module._sha256(record["path"])


def test_event_labels_preserve_pre_onset_normal_windows():
    module = load_module()
    pair = json.loads(
        (ROOT / "configs" / "experiments" / "simulation_v4_paired_split_v1.json").read_text()
    )["pairs"][0]
    event = module._spoof_event(pair)
    assert module._event_label("normal", event, 25.0) == ("normal", "steady_normal", 0)
    assert module._event_label("spoof", event, 9.5) == ("normal", "pre_event_normal", 0)
    assert module._event_label("spoof", event, 10.0) == ("spoofing", "carryoff_transition", 1)
    assert module._event_label("spoof", event, 20.0) == ("spoofing", "carryoff_final", 1)


def test_receiver_run_ids_are_short_and_pair_member_unique():
    module = load_module()
    _, split, _, _, _, _ = module._validate_generation_config(frozen_config())
    run_ids = {
        module._receiver_run_id(pair, member)
        for pair in split["pairs"]
        if pair["split"] == "train"
        for member in ("normal", "spoof")
    }
    assert len(run_ids) == 12
    assert max(map(len, run_ids)) < 64
