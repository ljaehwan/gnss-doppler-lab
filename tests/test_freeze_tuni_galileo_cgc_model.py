from pathlib import Path
import importlib.util

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_tuni_galileo_cgc_model",
    ROOT / "scripts" / "freeze_tuni_galileo_cgc_model.py",
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_normalized_shape_removes_common_complex_gain() -> None:
    base = np.asarray([1 + 2j, 2 + 1j, 3 - 1j, 4 + 2j, 5 + 3j, 2 - 2j, 1 - 3j, -1 + 1j, 3 + 4j])
    taps = np.vstack([base, base * (2 - 3j)])
    feature = MOD.normalized_shape(taps.real, taps.imag)
    assert feature.shape == (2, 16)
    np.testing.assert_allclose(feature[0], feature[1], rtol=1e-12, atol=1e-12)


def test_normalized_shape_rejects_zero_prompt() -> None:
    taps = np.ones((2, 9), dtype=np.complex128)
    taps[1, MOD.PROMPT_INDEX] = 0
    with pytest.raises(ValueError, match="prompt contains a zero"):
        MOD.normalized_shape(taps.real, taps.imag)


def test_pair_cosines_are_time_local() -> None:
    times = np.asarray([1.0, 1.0, 2.0, 2.0])
    residuals = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    np.testing.assert_allclose(MOD.pair_cosines(times, residuals), [1.0, -1.0])


def test_fixed_detector_thresholds_exceed_clean_design_targets() -> None:
    assert MOD.DISTORTION_THRESHOLD == 1.5
    assert MOD.COHERENCE_THRESHOLD == 0.8
    assert MOD.PERSISTENCE_BINS == 4

