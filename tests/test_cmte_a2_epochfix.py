from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnss_doppler_lab.cmte_a2_epochfix import (
    EpochPolicy,
    aggregate_multi_prn_epochs,
    canonical_decision_rows,
)


POLICY = EpochPolicy(
    grid_origin_s=0.0,
    grid_stride_s=0.5,
    timestamp_tolerance_s=1e-9,
    max_residual_age_s=0.55,
)


def _row(prn: str, availability: float, p: float, *, recording: str = "r", history: str = "h", segment: str = "0", channel: int = 0, rmse: float = 0.1, q: float = 1.0):
    row = {
        "physical_recording_id": recording,
        "history_id": history,
        "prn": prn,
        "segment": segment,
        "channel": channel,
        "window_start_s": availability - 1.0,
        "window_end_s": availability,
        "p": p,
        "q": q,
        "rmse": rmse,
    }
    for i in range(9):
        row[f"residual_{i:03d}"] = float(i + 1) * 0.001 + p
    return row


def _frame(*rows):
    return pd.DataFrame(rows)


def test_jittered_multi_prn_timestamps_join_same_decision_epoch():
    frame = _frame(
        _row("G01", 10.101, 0.5),
        _row("G02", 10.203, 0.25),
        _row("G03", 10.399, 0.125),
    )
    out = aggregate_multi_prn_epochs(frame, POLICY)
    epoch = out.loc[np.isclose(out.window_end_s, 10.5)].iloc[0]
    assert epoch.tracked_prn_count == 3


def test_at_most_one_residual_per_prn_per_epoch_and_latest_wins():
    frame = _frame(
        _row("G01", 10.10, 0.8),
        _row("G01", 10.30, 0.2),
        _row("G02", 10.20, 0.5),
    )
    selected = canonical_decision_rows(frame, POLICY)
    epoch = selected[np.isclose(selected.decision_time_s, 10.5)]
    assert not epoch.duplicated(["physical_recording_id", "decision_time_s", "prn"]).any()
    assert epoch.loc[epoch.prn.eq("G01"), "window_end_s"].item() == pytest.approx(10.30)


def test_future_residual_is_never_used():
    frame = _frame(_row("G01", 0.5001, 0.2), _row("G02", 0.49, 0.4))
    selected = canonical_decision_rows(frame, POLICY)
    at_half = selected[np.isclose(selected.decision_time_s, 0.5)]
    assert set(at_half.prn) == {"G02"}
    assert (selected.window_end_s <= selected.decision_time_s + POLICY.timestamp_tolerance_s).all()


def test_row_permutation_invariance():
    frame = _frame(*[_row(f"G{i:02d}", 1.01 + i * 0.01, 0.1 + i * 0.05) for i in range(1, 7)])
    a = aggregate_multi_prn_epochs(frame, POLICY)
    b = aggregate_multi_prn_epochs(frame.sample(frac=1, random_state=7), POLICY)
    pd.testing.assert_frame_equal(a, b)


def test_prn_label_permutation_invariance():
    frame = _frame(*[_row(f"G{i:02d}", 2.05 + i * 0.01, 0.08 + i * 0.04) for i in range(1, 6)])
    renamed = frame.copy()
    renamed["prn"] = renamed.prn.map(dict(zip(sorted(renamed.prn.unique()), reversed(sorted(renamed.prn.unique())))))
    a = aggregate_multi_prn_epochs(frame, POLICY).drop(columns=["prn_set_hash"])
    b = aggregate_multi_prn_epochs(renamed, POLICY).drop(columns=["prn_set_hash"])
    pd.testing.assert_frame_equal(a, b)


def test_small_floating_jitter_does_not_change_aggregation():
    frame = _frame(*[_row(f"G{i:02d}", 3.10 + i * 0.02, 0.1 + i * 0.03) for i in range(1, 6)])
    jittered = frame.copy()
    jittered["window_end_s"] += np.array([1e-7, -1e-7, 2e-7, -2e-7, 0.0])
    jittered["window_start_s"] += np.array([1e-7, -1e-7, 2e-7, -2e-7, 0.0])
    a = aggregate_multi_prn_epochs(frame, POLICY)
    b = aggregate_multi_prn_epochs(jittered, POLICY)
    pd.testing.assert_series_equal(a.tracked_prn_count, b.tracked_prn_count)
    np.testing.assert_allclose(a.score_A2, b.score_A2, rtol=0, atol=0)


def test_variable_prn_count_and_residual_age_exclusion():
    frame = _frame(
        _row("G01", 0.10, 0.5),
        _row("G02", 0.20, 0.4),
        _row("G01", 0.70, 0.3),
    )
    out = aggregate_multi_prn_epochs(frame, POLICY)
    assert out.set_index("window_end_s").tracked_prn_count.to_dict() == {0.5: 2, 1.0: 1}
    assert out.max_residual_age_s.max() <= POLICY.max_residual_age_s + POLICY.timestamp_tolerance_s


def test_recording_segment_channel_and_cadence_gap_do_not_leak_state():
    frame = _frame(
        _row("G01", 0.10, 0.5, recording="r1", history="h1", segment="0", channel=0),
        _row("G02", 0.20, 0.4, recording="r1", history="h1", segment="0", channel=1),
        _row("G01", 0.90, 0.3, recording="r1", history="h2", segment="1", channel=2),
        _row("G01", 0.10, 0.2, recording="r2", history="h3", segment="0", channel=0),
    )
    selected = canonical_decision_rows(frame, POLICY)
    r1_one = selected[(selected.physical_recording_id == "r1") & np.isclose(selected.decision_time_s, 1.0)]
    assert set(r1_one.prn) == {"G01"}
    assert r1_one.history_id.item() == "h2"
    assert not ((selected.physical_recording_id == "r1") & (selected.decision_time_s > 1.0)).any()
    assert set(selected.physical_recording_id) == {"r1", "r2"}


def test_real_ds7_fixture_has_multi_prn_median():
    path = Path(__file__).resolve().parents[1] / "artifacts/cmte_a2_texbat_ds78/per_prn/DS7.csv"
    if not path.is_file():
        pytest.skip("sealed DS7 fixture not present")
    out = aggregate_multi_prn_epochs(pd.read_csv(path), POLICY)
    assert out.tracked_prn_count.median() > 1
    assert out.tracked_prn_count.max() >= 4


def test_multi_prn_mean_matches_manual_scalar_calculation():
    pvalues = np.array([0.5, 0.25, 0.125])
    frame = _frame(*[_row(f"G{i+1:02d}", 4.10 + i * 0.01, float(p)) for i, p in enumerate(pvalues)])
    out = aggregate_multi_prn_epochs(frame, POLICY)
    epoch = out.loc[np.isclose(out.window_end_s, 4.5)].iloc[0]
    assert epoch.score_A2 == pytest.approx(float(np.mean(-np.log(pvalues))))
    assert epoch.mean_neg_log_p == pytest.approx(epoch.score_A2)
