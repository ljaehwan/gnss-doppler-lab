from pathlib import Path
import json

import numpy as np
import pytest

from gnss_doppler_lab.mctd import (
    chronological_masks, consecutive_alarms, epoch_scores, mahalanobis_score,
    nominal_epoch_ms, nonoverlap_blocks, paired_bootstrap_blocks,
    permutation_invariant_score, prompt_normalize, robust_fit, unwrap_by_prn,
)


def test_prompt_complex_normalization_and_gate():
    taps = np.ones((2, 9), dtype=np.complex128) * (2 + 2j)
    taps[0, 4] = 1 + 1j
    taps[1, 4] = 0
    normalized, valid = prompt_normalize(taps, min_magnitude=1e-3)
    assert normalized[0, 4] == pytest.approx(1 + 0j)
    assert valid.tolist() == [True, False]


def test_phase_unwrap_is_prn_local():
    prn = np.array([1, 2, 1, 2])
    epoch = np.array([0, 0, 1, 1])
    phase = np.array([3.0, -3.0, -3.0, 3.0])
    out = unwrap_by_prn(prn, epoch, phase)
    assert abs(out[2] - out[0]) < 1
    assert abs(out[3] - out[1]) < 1


def test_nominal_epoch_alignment():
    assert nominal_epoch_ms(np.array([25001, 49999]), 25_000_000).tolist() == [1, 2]


def test_robust_model_and_identical_collapse():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(1000, 5))
    model = robust_fit(x)
    scores = mahalanobis_score(np.zeros((20, 5)), robust_fit(np.zeros((100, 5))))
    assert np.max(scores) == pytest.approx(0.0)
    assert np.isfinite(mahalanobis_score(x[:10], model)).all()


def test_prn_permutation_invariance_and_variable_n():
    epoch = np.repeat([1, 2], [4, 5])
    prn = np.r_[np.arange(4), np.arange(5)]
    score = np.arange(9.0)
    expected = epoch_scores(epoch, prn, score)
    order = np.array([3, 1, 0, 2, 8, 5, 7, 4, 6])
    actual = permutation_invariant_score(epoch[order], prn[order], score[order])
    for left, right in zip(expected, actual):
        np.testing.assert_allclose(left, right)


def test_fewer_than_four_prns_rejected():
    epoch, score, count = epoch_scores(np.zeros(3), np.arange(3), np.ones(3))
    assert len(epoch) == len(score) == len(count) == 0


def test_nonoverlap_100ms_blocks():
    block, score, count = nonoverlap_blocks(np.array([0, 99, 100, 199]), np.array([1, 3, 5, 7]))
    assert block.tolist() == [0, 100]
    assert score.tolist() == [2, 6]
    assert count.tolist() == [2, 2]


def test_true_consecutive_alarm_and_gap_reset():
    blocks = np.array([0, 100, 200, 400, 500, 600])
    alarms = consecutive_alarms(blocks, np.ones(6) * 2, 1)
    assert alarms.tolist() == [False, False, True, False, False, True]


def test_chronological_split_and_guard_no_overlap():
    time = np.arange(0.0, 500.0, 0.1)
    masks = chronological_masks(time)
    assert all(mask.any() for mask in masks.values())
    total = sum(mask.astype(int) for mask in masks.values())
    assert np.max(total) == 1
    assert np.all(np.diff([time[mask].min() for mask in masks.values()]) > 0)


def test_bootstrap_block_construction():
    np.testing.assert_array_equal(paired_bootstrap_blocks(np.array([0, 9.999, 10, 20])), [0, 0, 1, 2])


def test_freeze_files_and_scenario_handoffs_are_declared():
    root = Path(__file__).resolve().parents[1]
    artifact = root / "artifacts/mctd_stage0_static"
    if artifact.exists():
        assert (artifact / "preregistration.json").exists()
        configs = list((artifact / "frozen_configs").glob("**/*.conf"))
        assert configs
        assert any("texbat_ds3" in path.name for path in configs)
        assert any("texbat_ds7" in path.name for path in configs)


def _config(path):
    return {line.split("=", 1)[0]: line.split("=", 1)[1] for line in path.read_text().splitlines()
            if line and not line.startswith("#") and "=" in line}


def test_slow_fast_configs_only_change_loop_dynamics_and_label():
    root = Path(__file__).resolve().parents[1] / "artifacts/mctd_stage0_static/frozen_configs/phase_a"
    slow = _config(root / "texbat_cleanstatic__slow.conf")
    fast = _config(root / "texbat_cleanstatic__fast.conf")
    changed = {key for key in slow if slow[key] != fast[key]}
    assert changed == {"Tracking_1C.dll_bw_hz", "Tracking_1C.pll_bw_hz", "Tracking_1C.trace_scenario_id"}
    for key in ("SignalSource.filename", "SignalSource.seconds_to_skip", "SignalSource.samples",
                "Tracking_1C.trace_handoff_filename", "Tracking_1C.tap_spacing_chips"):
        assert slow[key] == fast[key]


def test_identical_loop_configs_are_physically_identical():
    root = Path(__file__).resolve().parents[1] / "artifacts/mctd_stage0_static/frozen_configs/phase_a"
    left = _config(root / "oakbat_cleanstatic__identical_left.conf")
    right = _config(root / "oakbat_cleanstatic__identical_right.conf")
    left.pop("Tracking_1C.trace_scenario_id"); right.pop("Tracking_1C.trace_scenario_id")
    assert left == right


def test_scenario_specific_handoff_not_reused():
    root = Path(__file__).resolve().parents[1] / "artifacts/mctd_stage0_static/frozen_configs/full"
    ds3 = _config(root / "texbat_ds3__slow.conf")["Tracking_1C.trace_handoff_filename"]
    ds7 = _config(root / "texbat_ds7__slow.conf")["Tracking_1C.trace_handoff_filename"]
    os3 = _config(root / "oakbat_os3__slow.conf")["Tracking_1C.trace_handoff_filename"]
    os4 = _config(root / "oakbat_os4__slow.conf")["Tracking_1C.trace_handoff_filename"]
    assert ds3 != ds7 and os3 != os4


def test_preregistration_is_clean_only_and_frozen():
    path = Path(__file__).resolve().parents[1] / "artifacts/mctd_stage0_static/preregistration.json"
    value = json.loads(path.read_text())
    assert value["attack_data_accessed_by_mctd"] is False
    assert value["status"] == "SEALED_BEFORE_ATTACK_EVALUATION"
    assert len(value["freeze_commit"]) == 40


def test_phase_a_native_schema_has_causal_action_and_required_fields():
    from gnss_doppler_lab.trace_native_1ms import RECORD_DTYPE
    names = set(RECORD_DTYPE.names)
    assert {"action_used_source_loop_sequence", "loop_sequence", "raw_interval_start_sample",
            "action_next_code_nco_rate_chips_s", "action_next_carrier_doppler_hz",
            "dll_discriminator_chips", "pll_phase_error_cycles"} <= names


def test_verifier_entrypoint_exists_for_fresh_clone():
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts/verify_mctd_stage0.py").is_file()
