from pathlib import Path

import h5py
import numpy as np

from gnss_doppler_lab.trace_native_cadence import (
    CADENCE_1MS,
    CADENCE_20MS,
    CADENCE_GAP,
    ScenarioSpec,
    block_support,
    classify_cadence,
    load_consecutive_pairs,
    transition_mask,
)


def _write_mat(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle["PRN_start_sample_count"] = [1000, 2000, 3000, 23000, 43000, 93000]
        handle["PRN"] = [7, 7, 7, 7, 7, 7]


def test_actual_sample_count_cadence_classification():
    labels = classify_cadence(np.array([.001, .020, .050, .010, -1.0]))
    assert labels.tolist() == [CADENCE_1MS, CADENCE_20MS, CADENCE_GAP, "invalid_or_outlier", "invalid_or_outlier"]


def test_both_sides_of_cadence_transition_are_excluded():
    labels = np.array([CADENCE_1MS, CADENCE_1MS, CADENCE_20MS, CADENCE_20MS])
    assert transition_mask(labels).tolist() == [False, True, True, False]


def test_loader_preserves_actual_dt_and_block_support(tmp_path):
    (tmp_path / "raw").mkdir()
    _write_mat(tmp_path / "raw/epl_tracking_ch_0.mat")
    rows = load_consecutive_pairs(ScenarioSpec("fixture", "clean", tmp_path, 1_000_000))
    assert np.allclose(rows.dt_s, [.001, .001, .020, .020, .050])
    assert block_support(rows) == [
        {"block_start_s": 0.0, "block_end_s": 0.5, "valid_prn_count": 1, "pair_count": 1}
    ]


def test_prompt_referenced_carrier_policy():
    # A common carrier rotation vanishes under Prompt referencing, so applying
    # full Doppler phase afterward would double-apply a removed nuisance.
    taps = np.linspace(.5, 1.0, 9).astype(complex)
    rotated = taps * np.exp(1j * 1.7)
    norm = taps * np.conj(taps[4]) / abs(taps[4]) ** 2
    rotated_norm = rotated * np.conj(rotated[4]) / abs(rotated[4]) ** 2
    assert np.allclose(norm, rotated_norm)
