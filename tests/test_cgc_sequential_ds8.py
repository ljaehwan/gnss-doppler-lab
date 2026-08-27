import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_cgc_sequential_ds8.py"
SPEC = importlib.util.spec_from_file_location("run_cgc_sequential_ds8", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_robust_normal_parameters_use_frozen_median_mad_law():
    location, scale = MODULE.robust_normal_parameters(np.asarray([0.0, 1.0, 2.0]))
    assert location == 1.0
    assert scale == 1.4826


def test_page_cusum_accumulates_lower_residual_evidence_causally():
    evidence, statistic = MODULE.page_cusum(
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.asarray([0, 1, 2, 3]),
        location=1.0,
        scale=1.0,
        clip=3.0,
        reference=0.5,
    )
    assert evidence.tolist() == [0.0, 1.0, 1.0, 1.0]
    assert statistic.tolist() == [0.0, 0.5, 1.0, 1.5]


def test_page_cusum_resets_across_missing_bin():
    _, statistic = MODULE.page_cusum(
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0, 1, 4]),
        location=1.0,
        scale=1.0,
        clip=3.0,
        reference=0.5,
    )
    assert statistic.tolist() == [0.5, 1.0, 0.5]


def test_page_cusum_clips_single_bin_evidence():
    evidence, statistic = MODULE.page_cusum(
        np.asarray([-100.0]),
        np.asarray([0]),
        location=1.0,
        scale=1.0,
        clip=3.0,
        reference=0.5,
    )
    assert evidence.tolist() == [3.0]
    assert statistic.tolist() == [2.5]
