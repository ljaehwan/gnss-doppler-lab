from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preflight_tuni_gps_ss29_support.py"
CONFIG = ROOT / "configs/experiments/tuni_gps_ss29_support_preflight_v1.json"
SPEC = importlib.util.spec_from_file_location("tuni_gps_ss29_support", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_frozen_ss29_support_contract_is_valid() -> None:
    MODULE.validate_config(_config())


def test_validation_rejects_detector_score_access() -> None:
    document = copy.deepcopy(_config())
    document["experiment"]["detector_score_access"] = True
    with pytest.raises(ValueError, match="forbid"):
        MODULE.validate_config(document)


def test_content_range_requires_exact_bounded_response() -> None:
    headers = "HTTP/2 206\ncontent-range: bytes 0-1999999999/29999832000\n"
    assert MODULE.parse_content_range(headers) == (0, 1_999_999_999, 29_999_832_000)
    with pytest.raises(ValueError, match="Content-Range"):
        MODULE.parse_content_range("HTTP/2 200\ncontent-length: 29999832000\n")


def test_target_bin_counts_are_separate_from_total_support() -> None:
    audit = {
        "eligible_prns_by_bin": {
            "5": [1, 2, 3, 4, 5, 6, 7, 8],
            "6": [1, 2, 3, 4, 5, 6, 21, 32],
            "7": [1, 2, 21, 32],
        }
    }
    assert MODULE.target_bin_counts(audit, [1, 2, 21, 32]) == {
        "G01": 3, "G02": 3, "G21": 2, "G32": 2,
    }
