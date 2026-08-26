import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_cgc_texbat_external.py"
SPEC = importlib.util.spec_from_file_location("run_cgc_texbat_external", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def config():
    return json.loads((ROOT / "configs/experiments/cgc_texbat_external_v1.json").read_text())


def test_config_and_real_input_pins_validate():
    context = MOD.validate_config(config())
    assert [row["name"] for row in context["sources"]] == ["ds1", "ds2", "ds3"]


def test_regions_exclude_acquisition_and_transition():
    assert MOD.region(29.0) == "excluded"
    assert MOD.region(30.0) == "stable_pre"
    assert MOD.region(89.0) == "stable_pre"
    assert MOD.region(90.0) == "excluded"
    assert MOD.region(109.0) == "excluded"
    assert MOD.region(110.0) == "stable_post"


def test_scenario_change_uses_frozen_positive_direction():
    rows = [
        {"region": "stable_pre", "clock_centered_geometry_residual": value, "prn_count": 9}
        for value in (0.7, 0.8, 0.9)
    ] + [
        {"region": "stable_post", "clock_centered_geometry_residual": value, "prn_count": 8}
        for value in (0.1, 0.2, 0.3)
    ]
    summary = MOD.summarize_scenario("ds1", rows)
    assert summary["pre_minus_post_residual_change"] == pytest.approx(0.6)
    assert summary["positive_preregistered_direction"] is True
    assert summary["secondary_serial_bin_auc"] == 1.0


def test_threshold_or_interval_drift_fails_closed():
    document = copy.deepcopy(config())
    document["evaluation"]["threshold_fitting"] = True
    with pytest.raises(ValueError, match="evaluation"):
        MOD.validate_config(document)
    document = copy.deepcopy(config())
    document["analysis"]["stable_post_start_seconds"] = 120.0
    with pytest.raises(ValueError, match="analysis"):
        MOD.validate_config(document)
