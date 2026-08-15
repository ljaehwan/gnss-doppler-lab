import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnss_doppler_lab.b0_dependence_calibrated import (
    FEATURE_COLUMNS,
    aggregate_receiver_scores,
    artifact_manifest,
    attach_tracked_count,
    build_node_windows_from_npz,
    causal_examples,
    choose_block_seconds,
    cn0_tertile_edges,
    conformal_pvalues,
    consecutive_alarm,
    chronological_role_split,
    file_sha256,
    fit_standardizer,
    fit_stratum_calibrator,
    integrated_autocorrelation_time,
    official_timeline,
    paired_block_bootstrap,
    power_evalues,
    receiver_blocks,
    safe_prompt_normalize,
    scenario_family,
    score_block_evidence,
    score_prn_evidence,
    sequential_e_cusum,
)


def _nodes(epoch_count=120, prns=("G01", "G02", "G03", "G04", "G05", "G06", "G07")):
    rows = []
    for epoch in range(epoch_count):
        time = epoch * .5
        for prn_index, prn in enumerate(prns):
            row = {
                "physical_recording_id": "cleanStatic",
                "run_id": "cleanStatic",
                "role": "train",
                "prn": prn,
                "segment": "0",
                "channel": str(prn_index),
                "history_chunk": 0,
                "window_start_s": time,
                "window_end_s": time + 1,
                "window_mid_s": time + .5,
                "window_bin_s": time + .5,
                "lagged_cn0_db_hz": 35 + prn_index + .01 * epoch,
                "raw_sample_start": int(time * 1000),
                "raw_sample_end_exclusive": int((time + 1) * 1000),
                "raw_byte_start": int(time * 4000),
                "raw_byte_end_exclusive": int((time + 1) * 4000),
            }
            row.update({name: 1 + .001 * epoch + .01 * index for index, name in enumerate(FEATURE_COLUMNS)})
            rows.append(row)
    return pd.DataFrame(rows)


def _residuals(epoch_count=80, prns=("G01", "G02", "G03", "G04", "G05", "G06", "G07")):
    nodes = _nodes(epoch_count, prns)
    return nodes[[
        "physical_recording_id", "prn", "window_start_s", "window_end_s",
        "window_bin_s", "lagged_cn0_db_hz",
    ]].assign(
        b0_residual_rmse=lambda frame: .1 + .001 * frame.window_bin_s
    )


def test_prompt_normalization_is_finite_and_masks_zero_low_power():
    iq = np.zeros((3, 9, 2), dtype=float)
    iq[0, :, 0] = np.arange(1, 10)
    iq[0, 4, 0] = 5
    iq[1, :, 0] = 1e-10
    iq[2, :, 1] = 2
    normalized, valid = safe_prompt_normalize(iq)
    assert valid.tolist() == [True, False, True]
    assert normalized[0, 4] == pytest.approx(1)
    assert np.isfinite(normalized).all()
    assert np.array_equal(normalized[1], np.zeros(9))


def test_npz_node_conversion_uses_strictly_lagged_cn0_and_sample_lineage(tmp_path):
    epochs = 2500
    time = np.arange(epochs) * .001
    iq = np.ones((epochs, 9, 2), dtype=np.float32)
    iq[:, 4, 1] = 0
    path = tmp_path / "clean.npz"
    np.savez(
        path,
        complex_iq=iq,
        prn=np.ones(epochs, dtype=np.int16),
        time_s=time,
        segment_index=np.zeros(epochs, dtype=np.int16),
        channel=np.zeros(epochs, dtype=np.int16),
        sample_count=np.arange(epochs, dtype=np.uint64) * 25000,
        cn0_db_hz=30 + time,
    )
    nodes, audit = build_node_windows_from_npz(path, recording_id="clean")
    assert not nodes.empty and audit["sample_and_byte_lineage_preserved"]
    first = nodes.iloc[0]
    expected = np.median((30 + time)[(time >= first.window_start_s - 1) & (time < first.window_start_s)])
    assert first.lagged_cn0_db_hz == pytest.approx(expected)
    assert first.raw_byte_start == 4 * first.raw_sample_start
    assert first.raw_byte_end_exclusive == 4 * first.raw_sample_end_exclusive


def test_chronological_split_has_guards_no_epoch_or_raw_overlap_and_normal_only():
    roles, audit = chronological_role_split(_nodes(160), guard_seconds=6)
    assert set(roles) == {"train", "validation", "calibration", "holdout"}
    assert audit["no_target_epoch_overlap"]
    assert audit["no_raw_sample_or_byte_interval_overlap"]
    assert audit["normal_only"] and not audit["attack_labels_used"]
    for left, right in zip(("train", "validation", "calibration"), ("validation", "calibration", "holdout")):
        assert roles[right].window_bin_s.min() - roles[left].window_bin_s.max() >= 6.5


def test_causal_windows_reset_by_role_and_never_use_future_or_cross_guard():
    roles, _ = chronological_role_split(_nodes(180), guard_seconds=6)
    mean, stdev = fit_standardizer(roles["train"].loc[:, FEATURE_COLUMNS].to_numpy())
    for role, frame in roles.items():
        x, y, metadata, audit = causal_examples(frame, mean, stdev, seq_len=12)
        assert x.shape[1:] == (12, 9) and y.shape[1] == 9
        assert audit["first_target_index_is_seq_len"] and audit["causal_no_lookahead"]
        assert metadata.role.eq(role).all()
        assert metadata.window_bin_s.min() >= frame.window_bin_s.min() + 6


def test_conformal_ties_minimum_rank_and_fixed_evalue_formula():
    calibration = np.array([1., 2., 2., 4.])
    actual = conformal_pvalues(calibration, [2., 3., 5.])
    np.testing.assert_allclose(actual, [4 / 5, 2 / 5, 1 / 5])
    assert actual.min() == pytest.approx(1 / (len(calibration) + 1))
    np.testing.assert_allclose(power_evalues(actual), .5 * actual ** -.5)


def test_nuisance_merge_order_preserves_count_before_global_and_uses_no_prn_identity():
    frame = attach_tracked_count(_residuals(80))
    edges = cn0_tertile_edges(frame.lagged_cn0_db_hz)
    calibrator = fit_stratum_calibrator(frame, cn0_edges=edges, block_seconds=2, minimum_blocks=5)
    assert set(calibrator.entries) == {(a, b) for a in range(3) for b in range(3)}
    assert all(entry.merge_level in {"exact", "adjacent_cn0", "all_cn0_preserve_count", "global"}
               for entry in calibrator.entries.values())
    assert all("prn" not in entry.merge_level.lower() for entry in calibrator.entries.values())
    # There are only N=7 epochs, so absent count bins must fall through to global.
    assert calibrator.entries[(0, 0)].merge_level == "global"
    assert calibrator.entries[(0, 1)].merge_level != "global"


def test_receiver_aggregation_is_prn_permutation_invariant_and_variable_count_safe():
    calibration = attach_tracked_count(_residuals(80))
    edges = cn0_tertile_edges(calibration.lagged_cn0_db_hz)
    calibrator = fit_stratum_calibrator(calibration, cn0_edges=edges, block_seconds=2, minimum_blocks=5)
    query = _residuals(3)
    scored = score_prn_evidence(query, calibrator)
    first = aggregate_receiver_scores(scored)
    second = aggregate_receiver_scores(scored.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(first, second)
    fewer = query[query.prn.isin(["G01", "G02", "G03", "G04", "G05"])]
    variable = aggregate_receiver_scores(score_prn_evidence(fewer, calibrator))
    assert variable.tracked_prn_count.eq(5).all() and variable.score_valid.all()


def test_n_less_than_four_suppresses_receiver_score():
    calibration = attach_tracked_count(_residuals(80))
    edges = cn0_tertile_edges(calibration.lagged_cn0_db_hz)
    calibrator = fit_stratum_calibrator(calibration, cn0_edges=edges, block_seconds=2, minimum_blocks=5)
    query = _residuals(2, ("G01", "G02", "G03"))
    receiver = aggregate_receiver_scores(score_prn_evidence(query, calibrator))
    assert not receiver.score_valid.any()
    assert receiver.set_score.isna().all()


def test_block_boundaries_max_score_and_sequential_accumulator():
    receiver = pd.DataFrame({
        "physical_recording_id": ["a"] * 8,
        "window_bin_s": np.arange(8) * .5,
        "set_score": [0., 1., 2., 1., 4., 3., 1., 0.],
        "score_valid": [True] * 8,
        "tracked_prn_count": [7] * 8,
    })
    blocks = receiver_blocks(receiver, block_seconds=2)
    assert blocks.block_score.tolist() == [2., 4.]
    scored = score_block_evidence(blocks, calibration_block_scores=[0., 1., 2., 3.])
    expected = sequential_e_cusum(scored.block_evalue, scored.physical_recording_id)
    np.testing.assert_allclose(scored.e_cusum, expected)
    assert sequential_e_cusum([2., .25, 3.]).tolist() == pytest.approx([2., .5, 3.])


def test_iat_block_rule_uses_two_seconds_or_smallest_half_second_strictly_above():
    white = integrated_autocorrelation_time([1., -1.] * 20)
    assert choose_block_seconds(white["iat_seconds"]) == 2
    assert choose_block_seconds(2.0) == 2
    assert choose_block_seconds(2.01) == 2.5


def test_consecutive_threshold_alarm_is_causal():
    assert consecutive_alarm([True, True, False, True, True, True], consecutive_epochs=3).tolist() == [
        False, False, False, False, False, True
    ]


def test_official_timeline_mapping_and_ds7_ds8_family_grouping():
    assert official_timeline("DS3") == {"signal_onset": 118.9, "pull_off": 195.0}
    assert official_timeline("ds4")["pull_off"] == 225.0
    assert scenario_family("DS7") == scenario_family("DS8") == "DS7-DS8"
    assert scenario_family("DS3") == "DS3"


def test_paired_ten_second_bootstrap_is_deterministic_and_paired():
    labels = np.r_[np.zeros(20), np.ones(20)]
    first = np.arange(40, dtype=float)
    second = first - 1
    times = np.arange(40) * .5
    metric = lambda _labels, values: values.mean()
    a = paired_block_bootstrap(labels, first, second, times, metric=metric, repetitions=50, seed=4)
    b = paired_block_bootstrap(labels, first, second, times, metric=metric, repetitions=50, seed=4)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, np.ones(50))


def test_deterministic_split_reproduction():
    first_roles, first_audit = chronological_role_split(_nodes(180).sample(frac=1, random_state=1))
    second_roles, second_audit = chronological_role_split(_nodes(180))
    assert first_audit["roles"] == second_audit["roles"]
    for role in first_roles:
        assert set(first_roles[role].window_bin_s) == set(second_roles[role].window_bin_s)


def test_artifact_checksum_manifest_excludes_itself_and_detects_content(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "artifact_manifest_sha256.json").write_text("old")
    manifest = artifact_manifest(tmp_path)
    assert [item["path"] for item in manifest["files"]] == ["a.txt"]
    assert manifest["files"][0]["sha256"] == file_sha256(tmp_path / "a.txt")
    json.dumps(manifest, allow_nan=False)
