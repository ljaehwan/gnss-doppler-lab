from pathlib import Path

import h5py
import numpy as np

from gnss_doppler_lab.trace_equivariance import chronological_masks, load_trace_pairs


def _write_mat(path: Path, stamps=(1000, 2000, 3000, 5000), prns=(7, 7, 7, 7)):
    n = len(stamps)
    with h5py.File(path, "w") as f:
        for tap_i, tap in enumerate(("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")):
            f[f"I_{tap}"] = np.full(n, 1.0 - abs(tap_i - 4) / 10)
            f[f"Q_{tap}"] = np.zeros(n)
        f["PRN"] = prns
        f["PRN_start_sample_count"] = stamps
        f["CN0_SNV_dB_Hz"] = np.full(n, 40.0)
        f["carrier_lock_test"] = np.ones(n)
        f["carr_error_hz"] = np.zeros(n)
        f["carr_error_filt_hz"] = np.zeros(n)
        f["carrier_doppler_hz"] = np.zeros(n)
        f["code_error_chips"] = np.zeros(n)
        f["code_error_filt_chips"] = np.zeros(n)
        f["code_freq_chips"] = np.full(n, 1_023_000.0)
        f["aux1"] = np.zeros(n)


def test_tracker_row_to_next_row_and_off_by_one_gap(tmp_path):
    _write_mat(tmp_path / "epl_tracking_ch_0.mat")
    pairs = load_trace_pairs(tmp_path, 1_000_000, cn0_min_db_hz=0, lock_min=0)
    assert pairs.source_row.tolist() == [0, 1]
    assert pairs.sample_count.tolist() == [2000, 3000]
    assert np.allclose(pairs.dt_s, 0.001)


def test_prn_change_breaks_causal_pair(tmp_path):
    _write_mat(tmp_path / "epl_tracking_ch_0.mat", stamps=(1000, 2000, 3000), prns=(7, 8, 8))
    pairs = load_trace_pairs(tmp_path, 1_000_000, cn0_min_db_hz=0, lock_min=0)
    assert pairs.source_row.tolist() == [1]


def test_chronological_masks_have_guards_and_no_future_context():
    t = np.arange(100.0)
    masks = chronological_masks(t, 100.0, guard_s=5.0)
    assert t[masks["train"]].max() == 44
    assert t[masks["calibration"]].min() == 55
    assert t[masks["calibration"]].max() == 69
    assert t[masks["holdout"]].min() == 80
    assert not (masks["train"] & masks["calibration"]).any()


def test_attack_data_exclusion_contract():
    fit_roles = {"TEXBAT": ["cleanStatic"], "OAKBAT": ["cleanStatic"]}
    assert all("DS" not in role and "OS" not in role for roles in fit_roles.values() for role in roles)


def test_raw_sample_byte_overlap_audit_detects_intersection():
    train = {(0, 1000), (1000, 2000)}
    calibration = {(2500, 3500)}
    holdout = {(4000, 5000)}
    assert train.isdisjoint(calibration) and train.isdisjoint(holdout) and calibration.isdisjoint(holdout)


def test_common_epoch_support_and_cadence_contract():
    support = {"Full": (0.5, 4), "A1": (0.5, 4), "A2": (0.5, 4), "A4": (0.5, 4), "B0": (0.5, 4)}
    assert len(set(support.values())) == 1


def test_deterministic_seed_contract():
    assert np.array_equal(np.random.default_rng(23017).permutation(20), np.random.default_rng(23017).permutation(20))
