from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_fgi_spoofrepo_tgd_cgc_detection.py"
SPEC = importlib.util.spec_from_file_location("fgi_tgd_cgc_detection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def config() -> dict:
    return json.loads(MODULE.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_frozen_detector_config_is_valid() -> None:
    MODULE.validate_config(config())


def test_region_boundaries_are_frozen() -> None:
    assert MODULE.region(40.0) == "clean"
    assert MODULE.region(119.0) == "clean"
    assert MODULE.region(120.0) == "excluded_transition"
    assert MODULE.region(159.0) == "excluded_transition"
    assert MODULE.region(160.0) == "stable_post"
    assert MODULE.region(229.0) == "stable_post"
    assert MODULE.region(230.0) == "excluded_transition"


def test_partial_f_tail_decreases_with_better_geometry_fit() -> None:
    weak = MODULE.partial_f_p_value(0.8, 12)
    strong = MODULE.partial_f_p_value(0.1, 12)
    assert np.isfinite([weak, strong]).all()
    assert 0.0 <= strong < weak <= 1.0


def test_threshold_and_persistence_are_unchanged() -> None:
    detector = config()["frozen_detector"]
    assert detector["partial_f_p_alarm_threshold"] == 0.06028418845288192
    assert detector["persistence_window_bins"] == 5
    assert detector["persistence_required_bins"] == 3

