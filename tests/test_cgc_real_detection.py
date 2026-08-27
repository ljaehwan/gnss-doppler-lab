import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_cgc_real_detection.py"
SPEC = importlib.util.spec_from_file_location("run_cgc_real_detection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_early_late_asymmetry_is_zero_for_symmetric_profile():
    magnitude = np.asarray([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=np.float32)
    iq = np.zeros((1, 9, 2), dtype=np.float32)
    iq[0, :, 0] = magnitude
    assert MODULE.early_late_asymmetry(iq).tolist() == [0.0]


def test_early_late_asymmetry_detects_one_sided_distortion():
    iq = np.zeros((1, 9, 2), dtype=np.float32)
    iq[0, :, 0] = np.asarray([2, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    assert np.isclose(MODULE.early_late_asymmetry(iq)[0], 0.1)


def test_persistent_alarm_is_causal_and_resets_across_gaps():
    raw = np.asarray([1, 0, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
    bins = np.asarray([0, 1, 2, 3, 4, 8, 9, 10, 11], dtype=np.int64)
    observed = MODULE.persistent_alarm(raw, bins, window=5, required=3)
    assert observed.tolist() == [False, False, False, False, True, False, False, False, False]


def test_source_regions_match_frozen_intervals():
    assert MODULE.source_region("cleanStatic", "calibration_only", 329) == "development_excluded"
    assert MODULE.source_region("cleanStatic", "calibration_only", 330) == "calibration"
    assert MODULE.source_region("cleanStatic", "calibration_only", 420) == "development_excluded"
    assert MODULE.source_region("cleanDynamic", "locked_normal", 0) == "locked_normal"
    assert MODULE.source_region("ds7", "primary_attack", 30) == "stable_pre"
    assert MODULE.source_region("ds7", "primary_attack", 90) == "excluded_transition"
    assert MODULE.source_region("ds7", "primary_attack", 110) == "stable_post"
