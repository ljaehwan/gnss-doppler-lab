import math

import pandas as pd
import pytest

from gnss_doppler_lab.quality_conditioned_tail import (
    GLOBAL_SCORE,
    QUALITY_SCORE,
    annotate_quality_state,
    binomial_tail_surprise,
    build_event_scores,
    calibrate_tail_detectors,
    evaluate_attack,
    fit_node_thresholds,
    quality_bin_names,
)


def _scores(rows):
    frame = pd.DataFrame(rows, columns=["run_id", "prn", "window_bin_s", "prn_node_rmse"])
    frame["window_start_s"] = frame["window_bin_s"]
    frame["window_mid_s"] = frame["window_bin_s"] + 0.5
    frame["window_end_s"] = frame["window_bin_s"] + 1.0
    return frame


def _thresholds(early, middle, mature):
    names = quality_bin_names((1.0, 2.0))
    return {
        names[0]: {"q50": early, "q70": early, "q80": early},
        names[1]: {"q50": middle, "q70": middle, "q80": middle},
        names[2]: {"q50": mature, "q70": mature, "q80": mature},
    }


def test_quality_age_is_causal_and_resets_after_a_gap():
    frame = _scores([
        ("r", "G01", 0.0, 0.1),
        ("r", "G01", 0.5, 0.2),
        ("r", "G01", 1.0, 0.3),
        ("r", "G01", 3.0, 0.4),
        ("r", "G01", 3.5, 0.5),
    ])

    got = annotate_quality_state(frame, age_cutoffs_s=(1.0, 2.0), max_gap_s=0.75)

    assert got["quality_age_s"].tolist() == [0.0, 0.5, 1.0, 0.0, 0.5]
    assert got["quality_segment_index"].tolist() == [0, 0, 0, 1, 1]
    assert got["quality_bin"].tolist() == [
        "age_lt_1s", "age_lt_1s", "age_1_to_2s", "age_lt_1s", "age_lt_1s"
    ]


def test_quality_annotation_does_not_depend_on_future_scores():
    prefix = _scores([("r", "G01", 0.0, 0.1), ("r", "G01", 0.5, 0.2)])
    full = pd.concat([prefix, _scores([("r", "G01", 4.0, 100.0)])], ignore_index=True)

    prefix_result = annotate_quality_state(prefix, age_cutoffs_s=(1.0, 2.0))
    full_result = annotate_quality_state(full, age_cutoffs_s=(1.0, 2.0)).iloc[:2]

    assert full_result["quality_age_s"].tolist() == prefix_result["quality_age_s"].tolist()
    assert full_result["quality_bin"].tolist() == prefix_result["quality_bin"].tolist()


def test_fit_thresholds_uses_separate_quality_distributions():
    early = [("r", f"G{i:02d}", 0.0, float(i)) for i in range(1, 5)]
    middle = [("r", f"G{i:02d}", 1.0, float(10 + i)) for i in range(1, 5)]
    mature = [("r", f"G{i:02d}", 2.0, float(20 + i)) for i in range(1, 5)]
    annotated = annotate_quality_state(
        _scores(early + middle + mature), age_cutoffs_s=(0.5, 1.5), max_gap_s=1.5
    )

    global_thresholds, quality_thresholds, counts, fallbacks = fit_node_thresholds(
        annotated, age_cutoffs_s=(0.5, 1.5), min_bin_rows=4
    )

    assert global_thresholds["q50"] == pytest.approx(12.5)
    assert quality_thresholds["age_lt_0.5s"]["q50"] == pytest.approx(2.5)
    assert quality_thresholds["age_0.5_to_1.5s"]["q50"] == pytest.approx(12.5)
    assert quality_thresholds["age_ge_1.5s"]["q50"] == pytest.approx(22.5)
    assert counts == {"age_lt_0.5s": 4, "age_0.5_to_1.5s": 4, "age_ge_1.5s": 4}
    assert not any(fallbacks.values())


def test_sparse_quality_bin_falls_back_to_global_thresholds():
    annotated = annotate_quality_state(
        _scores([("r", "G01", 0.0, 1.0), ("r", "G01", 0.5, 2.0)]),
        age_cutoffs_s=(1.0, 2.0),
    )

    global_thresholds, quality_thresholds, _, fallbacks = fit_node_thresholds(
        annotated, age_cutoffs_s=(1.0, 2.0), min_bin_rows=10
    )

    assert all(fallbacks.values())
    assert all(thresholds == global_thresholds for thresholds in quality_thresholds.values())


def test_quality_threshold_changes_evidence_for_same_local_score():
    frame = _scores([
        ("r", "G01", 0.0, 0.8),
        ("r", "G02", 0.0, 0.8),
        ("r", "G01", 1.0, 0.8),
        ("r", "G02", 1.0, 0.8),
        ("r", "G01", 2.0, 0.8),
        ("r", "G02", 2.0, 0.8),
    ])
    events = build_event_scores(
        frame,
        global_node_thresholds={"q50": 0.5, "q70": 0.5, "q80": 0.5},
        quality_node_thresholds=_thresholds(1.0, 0.7, 0.9),
        age_cutoffs_s=(1.0, 2.0),
        max_gap_s=1.5,
    )

    assert events["global_k_q50"].tolist() == [2, 2, 2]
    assert events["quality_k_q50"].tolist() == [0, 2, 0]
    assert events.loc[1, QUALITY_SCORE] > events.loc[0, QUALITY_SCORE]


def test_event_ewma_resets_for_each_run():
    frame = _scores([
        ("r1", "G01", 0.0, 1.0),
        ("r1", "G01", 0.5, 0.0),
        ("r2", "G01", 0.0, 0.0),
        ("r2", "G01", 0.5, 1.0),
    ])
    thresholds = _thresholds(0.5, 0.5, 0.5)
    events = build_event_scores(
        frame,
        global_node_thresholds={"q50": 0.5, "q70": 0.5, "q80": 0.5},
        quality_node_thresholds=thresholds,
        age_cutoffs_s=(1.0, 2.0),
    )

    r2 = events[events.run_id == "r2"].reset_index(drop=True)
    assert r2.loc[0, GLOBAL_SCORE] == 0.0
    assert r2.loc[1, GLOBAL_SCORE] > 0.0


def test_calibration_produces_matched_global_and_quality_thresholds():
    rows = []
    for prn in ("G01", "G02", "G03", "G04"):
        for index in range(12):
            rows.append(("clean", prn, index * 0.5, index / 10.0))
    calibration, events = calibrate_tail_detectors(
        _scores(rows), age_cutoffs_s=(1.0, 3.0), min_bin_rows=4
    )

    assert calibration.calibration_rows == 48
    assert calibration.calibration_events == 12
    assert math.isfinite(calibration.global_event_threshold)
    assert math.isfinite(calibration.quality_event_threshold)
    assert {GLOBAL_SCORE, QUALITY_SCORE}.issubset(events.columns)


def test_attack_metrics_separate_buffered_rate_from_raw_detection_delay():
    events = pd.DataFrame({
        "run_id": ["r"] * 7,
        "window_start_s": [80.0, 89.0, 100.0, 100.5, 101.0, 110.0, 110.5],
        QUALITY_SCORE: [0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0],
    })

    metrics = evaluate_attack(events, QUALITY_SCORE, 1.0, onset_s=100.0)

    assert metrics["pre_false_positive_rate"] == 0.0
    assert metrics["post_detection_rate"] == pytest.approx(0.5)
    assert metrics["first_detection_delay_s"] == 0.0
    assert metrics["first_detection_available_delay_s"] == 1.0
    assert metrics["first_three_consecutive_delay_s"] == 1.0


def test_binomial_tail_matches_exact_probability():
    assert binomial_tail_surprise(4, 4, 0.5) == pytest.approx(-math.log(0.5**4))


def test_duplicate_prn_event_is_rejected():
    frame = _scores([("r", "G01", 0.0, 0.1), ("r", "G01", 0.0, 0.2)])
    with pytest.raises(ValueError, match="duplicate"):
        annotate_quality_state(frame)
