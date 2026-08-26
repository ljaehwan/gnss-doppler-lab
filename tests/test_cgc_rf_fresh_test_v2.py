import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_cgc_rf_fresh_test_v2.py"
SPEC = importlib.util.spec_from_file_location("run_cgc_rf_fresh_test_v2", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def config():
    return json.loads((ROOT / "configs/experiments/cgc_rf_fresh_test_v2.json").read_text())


def test_fresh_config_and_selected_rows_validate():
    context = MOD.validate_config(config())
    assert [row["paired_group_id"] for row in context["pairs"]] == MOD.EXPECTED_PAIR_IDS
    assert [MOD._motion_kind(row) for row in context["pairs"]] == ["static", "straight", "parallel-sweep"]


def test_selected_pair_cannot_change_after_support_preflight():
    document = copy.deepcopy(config())
    document["pairs"][0]["position"]["altitude_m"] += 1.0
    with pytest.raises(ValueError, match="changed"):
        MOD.validate_config(document)


def test_threshold_and_pair_substitution_fail_closed():
    document = copy.deepcopy(config())
    document["evaluation"]["threshold_fitting"] = True
    with pytest.raises(ValueError, match="evaluation"):
        MOD.validate_config(document)
    document = copy.deepcopy(config())
    document["data_boundary"]["post_release_pair_substitution_forbidden"] = False
    with pytest.raises(ValueError, match="leakage"):
        MOD.validate_config(document)
