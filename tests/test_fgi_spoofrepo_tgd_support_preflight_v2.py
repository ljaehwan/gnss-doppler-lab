from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_fgi_spoofrepo_tgd_support_v2.py"
SPEC = importlib.util.spec_from_file_location("fgi_tgd_support_preflight_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def config() -> dict:
    return json.loads(MODULE.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_frozen_v2_config_is_valid() -> None:
    MODULE.validate_config(config())


def test_minimum_rows_is_eighty_percent_of_telemetry_cadence() -> None:
    document = config()
    receiver = document["receiver_output"]
    gate = document["support_gate"]
    assert receiver["telemetry_synchronized_dump_rate_hz"] == 50
    assert gate["minimum_epochs_per_prn_bin"] == 40
    assert gate["minimum_telemetry_cadence_occupancy"] == 0.8


def test_v2_forbids_detector_and_receiver_replay() -> None:
    experiment = config()["experiment"]
    assert experiment["detector_score_access"] is False
    assert experiment["delay_template_access"] is False
    assert experiment["threshold_refitting"] is False
    assert experiment["receiver_replay"] is False

