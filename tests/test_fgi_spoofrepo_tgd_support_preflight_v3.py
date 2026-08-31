from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_fgi_spoofrepo_tgd_support_v3.py"
SPEC = importlib.util.spec_from_file_location("fgi_tgd_support_preflight_v3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def config() -> dict:
    return json.loads(MODULE.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_frozen_v3_config_is_valid() -> None:
    MODULE.validate_config(config())


def test_identity_filter_removes_non_ephemeris_prns() -> None:
    audit = {
        "rules": {"analysis_interval_seconds": [0.0, 2.0]},
        "eligible_prns_by_bin": {
            "0": [1, 5, 7, 8, 9, 13, 14, 15, 18, 20],
            "1": [1, 2, 3, 5, 7, 8, 9, 13, 14, 15, 18],
        },
    }
    result = MODULE.filter_interval(
        audit, {5, 7, 8, 9, 13, 14, 15, 18, 20, 22, 27, 30}, 8, 2
    )
    assert result["maximum_identity_valid_prns"] == 9
    assert result["primary_bin_count"] == 2
    assert result["support_eligible"] is True
    assert 1 not in result["identity_valid_prns_by_bin"]["0"]


def test_v3_forbids_detector_access() -> None:
    experiment = config()["experiment"]
    assert experiment["detector_score_access"] is False
    assert experiment["delay_template_access"] is False
    assert experiment["threshold_refitting"] is False

