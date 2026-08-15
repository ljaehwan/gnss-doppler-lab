from __future__ import annotations

import pytest

from gnss_doppler_lab.gcspo_r1_support import (
    exact_b0_full_contrast_r1,
    integrate_protected_b0_r1,
    validate_protected_method_support_r1,
)


def _row(start: float, *, phase: str = "pre_onset", prns=(3, 6, 7, 8), epochs=(100, 101)):
    support = tuple((epoch, tuple(prns)) for epoch in epochs)
    return {
        "phase": phase,
        "window_start_s": start,
        "availability_s": start + 1.0,
        "prns": tuple(prns),
        "epoch_ids": tuple(epochs),
        "epoch_prn_support": support,
        "score": 10.0 + start,
    }


def _methods():
    full = [_row(0.0), _row(6.0)]
    return {name: [dict(row) for row in full] for name in ("A1", "A2", "A3", "A4", "A5", "Full")}


def _b0_prn_rows(start: float, *, prns=(3, 6, 7, 8), epochs=(100, 101), score=2.5):
    return [
        {
            "phase": "pre_onset",
            "window_start_s": start,
            "availability_s": start + 1.0,
            "prn": prn,
            "event_score": score,
            "epoch_ids": tuple(epochs),
            "epoch_prn_support": tuple((epoch, (prn,)) for epoch in epochs),
        }
        for prn in prns
    ]


def test_r1_preserves_full_standalone_and_pairs_b0_only_on_exact_common_support():
    methods = integrate_protected_b0_r1(_methods(), _b0_prn_rows(6.0), score_column="event_score")

    assert [row["window_start_s"] for row in methods["Full"]] == [0.0, 6.0]
    assert [row["window_start_s"] for row in methods["A0"]] == [6.0]
    support = validate_protected_method_support_r1(methods, required_phases=("pre_onset",))
    assert support["b0_common_support_status"] == "AVAILABLE"
    assert support["phase_counts"]["Full"] == {"pre_onset": 2}
    assert support["phase_counts"]["A0"] == {"pre_onset": 1}


def test_r1_rejects_same_time_when_native_prn_support_differs_without_discarding_full():
    methods = integrate_protected_b0_r1(_methods(), _b0_prn_rows(6.0, prns=(3,)), score_column="event_score")

    assert len(methods["Full"]) == 2
    assert methods["A0"] == []
    support = validate_protected_method_support_r1(methods, required_phases=("pre_onset",))
    assert support["b0_common_support_status"] == "UNAVAILABLE_ON_COMMON_SUPPORT"


def test_r1_does_not_use_nearest_neighbor_timestamp_join():
    methods = integrate_protected_b0_r1(_methods(), _b0_prn_rows(6.0000001), score_column="event_score")

    assert methods["A0"] == []
    assert len(methods["Full"]) == 2


def test_r1_b0_full_contrast_uses_intersection_and_reports_full_standalone_count():
    full = [{**_row(0.0), "scenario": "DS3", "method": "Full"},
            {**_row(6.0), "scenario": "DS3", "method": "Full"}]
    a0 = [{**_row(6.0), "scenario": "DS3", "method": "A0", "score": 2.5}]

    result = exact_b0_full_contrast_r1(full + a0, required_scenarios=("DS3",))

    phase = result["scenario_phase_results"]["DS3"]["pre_onset"]
    assert phase["full_standalone_windows"] == 2
    assert phase["common_support_windows"] == 1
    assert phase["status"] == "AVAILABLE"


def test_r1_rejects_aggregated_b0_with_inconsistent_declared_native_support():
    malformed = {
        "phase": "pre_onset",
        "window_start_s": 6.0,
        "availability_s": 7.0,
        "event_score": 2.5,
        "prns": (99,),
        "epoch_ids": (999,),
        "epoch_prn_support": tuple((epoch, (3, 6, 7, 8)) for epoch in (100, 101)),
    }

    with pytest.raises(ValueError, match="native support is malformed"):
        integrate_protected_b0_r1(_methods(), [malformed], score_column="event_score")


def test_r1_adapter_scope_patches_and_restores_all_four_bindings():
    from gnss_doppler_lab import gcspo_evaluate, gcspo_verify_artifacts
    from gnss_doppler_lab.gcspo_r1_support import r1_support_adapter_scope

    originals = (
        gcspo_evaluate.integrate_protected_b0,
        gcspo_evaluate.validate_protected_method_support,
        gcspo_evaluate.exact_b0_full_contrast,
        gcspo_verify_artifacts.exact_b0_full_contrast,
    )
    with r1_support_adapter_scope():
        assert gcspo_evaluate.integrate_protected_b0 is integrate_protected_b0_r1
        assert gcspo_evaluate.validate_protected_method_support is validate_protected_method_support_r1
        assert gcspo_evaluate.exact_b0_full_contrast is exact_b0_full_contrast_r1
        assert gcspo_verify_artifacts.exact_b0_full_contrast is exact_b0_full_contrast_r1
    assert (
        gcspo_evaluate.integrate_protected_b0,
        gcspo_evaluate.validate_protected_method_support,
        gcspo_evaluate.exact_b0_full_contrast,
        gcspo_verify_artifacts.exact_b0_full_contrast,
    ) == originals

    with pytest.raises(RuntimeError, match="forced"):
        with r1_support_adapter_scope():
            raise RuntimeError("forced")
    assert (
        gcspo_evaluate.integrate_protected_b0,
        gcspo_evaluate.validate_protected_method_support,
        gcspo_evaluate.exact_b0_full_contrast,
        gcspo_verify_artifacts.exact_b0_full_contrast,
    ) == originals


def test_r1_rejects_near_availability_in_b0_join():
    shifted = _b0_prn_rows(6.0)
    for row in shifted:
        row["availability_s"] += 1e-9

    methods = integrate_protected_b0_r1(_methods(), shifted, score_column="event_score")
    assert methods["A0"] == []
    assert [row["window_start_s"] for row in methods["Full"]] == [0.0, 6.0]


def test_r1_rejects_noncanonical_prn_local_epoch_order():
    malformed = _b0_prn_rows(6.0)
    for row in malformed:
        row["epoch_ids"] = tuple(reversed(row["epoch_ids"]))
        row["epoch_prn_support"] = tuple(reversed(row["epoch_prn_support"]))

    with pytest.raises(ValueError, match="native support is malformed"):
        integrate_protected_b0_r1(_methods(), malformed, score_column="event_score")


def test_r1_per_prn_event_score_requires_bit_exact_equality_and_preserves_score():
    rows = _b0_prn_rows(6.0, score=2.5)
    methods = integrate_protected_b0_r1(_methods(), rows, score_column="event_score")
    assert methods["A0"][0]["score"] == 2.5

    rows[-1]["event_score"] = 2.5 + 5e-13
    with pytest.raises(ValueError, match="differs across PRN"):
        integrate_protected_b0_r1(_methods(), rows, score_column="event_score")
