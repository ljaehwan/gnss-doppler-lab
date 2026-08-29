from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/recover_tuni_gps_ss29_support_preflight_v1_1.py"
CONFIG = ROOT / "configs/experiments/tuni_gps_ss29_support_preflight_v1_1.json"
SPEC = importlib.util.spec_from_file_location("tuni_gps_ss29_support_v1_1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v1_1_timebase_correction_contract_is_valid() -> None:
    MODULE.validate_config(_config())


def test_v1_1_rejects_support_rule_change() -> None:
    document = copy.deepcopy(_config())
    document["experiment"]["support_rule_change"] = True
    with pytest.raises(ValueError, match="scope"):
        MODULE.validate_config(document)


def test_v1_1_rejects_internal_rate_drift() -> None:
    document = copy.deepcopy(_config())
    document["timebase"]["internal_tracking_sample_rate_hz"] = 50_000_000
    with pytest.raises(ValueError, match="timebase"):
        MODULE.validate_config(document)
