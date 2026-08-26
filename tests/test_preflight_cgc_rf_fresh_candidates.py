import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/preflight_cgc_rf_fresh_candidates.py"
SPEC = importlib.util.spec_from_file_location("preflight_cgc_rf_fresh_candidates", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def config():
    return json.loads((ROOT / "configs/experiments/cgc_rf_fresh_candidate_pool_v2.json").read_text())


def test_candidate_pool_contract_is_valid():
    MOD.validate_config(config())


def test_selection_takes_first_support_eligible_per_motion():
    document = config()
    counts = {row["paired_group_id"]: 9 for row in document["candidates"]}
    expected = []
    for kind in document["preflight"]["required_motion_kinds"]:
        rows = [row for row in document["candidates"] if MOD.motion_kind(row) == kind]
        counts[rows[1]["paired_group_id"]] = 10
        counts[rows[2]["paired_group_id"]] = 12
        expected.append(rows[1]["paired_group_id"])
    selected = MOD.select_candidates(document, counts)
    assert [row["paired_group_id"] for row in selected] == expected


def test_selection_fails_closed_when_motion_has_no_supported_candidate():
    document = config()
    counts = {row["paired_group_id"]: 11 for row in document["candidates"]}
    for row in document["candidates"]:
        if MOD.motion_kind(row) == "straight":
            counts[row["paired_group_id"]] = 9
    with pytest.raises(RuntimeError, match="straight"):
        MOD.select_candidates(document, counts)


def test_validation_rejects_score_access():
    document = copy.deepcopy(config())
    document["experiment"]["score_access_during_preflight"] = True
    with pytest.raises(ValueError, match="score access"):
        MOD.validate_config(document)
