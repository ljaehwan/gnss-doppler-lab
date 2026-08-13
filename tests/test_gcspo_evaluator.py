from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _row(scenario, phase, method, end, score, *, prns=(3, 7, 11, 19), label=True,
         phase_start=110., phase_end=150., epoch_ids=None, epoch_prn_support=None):
    epoch_ids = tuple(range(round((end - 1.) / .02), round(end / .02))) if epoch_ids is None else tuple(epoch_ids)
    epoch_prn_support = (tuple((epoch, tuple(prns)) for epoch in epoch_ids)
                         if epoch_prn_support is None else tuple(epoch_prn_support))
    return {"scenario": scenario, "phase": phase, "method": method,
            "window_start_s": end - 1., "availability_s": end, "score": float(score),
            "prns": tuple(prns), "epoch_ids": epoch_ids, "epoch_prn_support": epoch_prn_support,
            "label": bool(label), "phase_start_s": phase_start, "phase_end_s": phase_end}


def test_phase_containment_and_missing_slot_breaks_persistence():
    from gnss_doppler_lab.gcspo_statistics import phase_contained, scheduled_persistence

    assert phase_contained({"window_start_s": 109., "availability_s": 110.}, 0., 110.) is False
    assert phase_contained({"window_start_s": 109., "availability_s": 109.999}, 0., 110.) is True
    rows = [{"availability_s": t, "score": 2.} for t in (1., 1.5, 2., 3., 3.5, 4., 4.5, 5.)]
    flags = scheduled_persistence(rows, threshold=1.)
    assert flags[2] is False                 # warm-up: only three slots, not five
    assert flags[3:6] == [False, False, False]  # missing 2.5 s cannot be bridged
    assert flags[7] is True


def test_exact_contrast_support_includes_event_epoch_prn_identity():
    from gnss_doppler_lab.gcspo_statistics import exact_contrast_support

    rows = [_row("DS3", "transition", method, 120., value)
            for method, value in (("Full", 3), ("A1", 2), ("A2", 1))]
    assert len(exact_contrast_support(rows, ("Full", "A1", "A2"))) == 1
    replacement = (3, 7, 11, 23)
    rows[-1] = {**rows[-1], "prns": replacement,
                "epoch_prn_support": tuple((epoch, replacement) for epoch in rows[-1]["epoch_ids"])}
    assert exact_contrast_support(rows, ("Full", "A1", "A2")) == []


def test_exact_contrast_support_excludes_unequal_actual_epochs_and_rejects_duplicates():
    from gnss_doppler_lab.gcspo_statistics import exact_contrast_support

    full = _row("DS3", "transition", "Full", 120., 3.)
    missing_epoch = {**_row("DS3", "transition", "A1", 120., 2.),
                     "epoch_ids": full["epoch_ids"][:-1],
                     "epoch_prn_support": full["epoch_prn_support"][:-1]}
    assert exact_contrast_support([full, missing_epoch], ("Full", "A1")) == []

    duplicate_prn = {**_row("DS3", "transition", "A1", 120., 2.),
                     "epoch_prn_support": tuple((epoch, prns + (prns[-1],))
                                                for epoch, prns in full["epoch_prn_support"])}
    with pytest.raises(ValueError, match="authenticated actual support"):
        exact_contrast_support([full, duplicate_prn], ("Full", "A1"))

    with pytest.raises(ValueError, match="duplicate method row"):
        exact_contrast_support([full, dict(full)], ("Full",))


def test_method_rows_preserve_native_geometry_and_continuous_prn_support(monkeypatch):
    import gnss_doppler_lab.gcspo_evaluate as evaluate
    from gnss_doppler_lab.gcspo_statistics import exact_contrast_support

    epochs = np.repeat(np.asarray([0, 1]), 5)
    prns = np.tile(np.asarray([3, 7, 11, 19, 23]), 2)
    monkeypatch.setattr(evaluate, "residual_table",
                        lambda *args: (epochs, prns, np.empty((10, 10)), np.empty((10, 10))))

    def row(prn_values):
        support = ((0, tuple(prn_values)), (1, tuple(prn_values)))
        return {"window_start_s": 0., "availability_s": .04, "score": 1.,
                "prns": list(prn_values), "epoch_ids": (0, 1), "epoch_prn_support": support}

    class Geometry:
        def __init__(self, *args):
            self.validated_rows = set(args[-1])

    monkeypatch.setattr(evaluate, "GeometryCache", Geometry)
    monkeypatch.setattr(evaluate, "a1_role_scores", lambda *args: [row((3, 7, 11, 19, 23))])
    monkeypatch.setattr(evaluate, "role_a2_terms", lambda *args: [row((3, 7, 11, 19, 23))])
    monkeypatch.setattr(evaluate, "score_a2_terms", lambda rows, _: rows)
    monkeypatch.setattr(evaluate, "role_a5_terms", lambda *args: [row((3, 7, 11, 19))])
    monkeypatch.setattr(evaluate, "score_a5_terms", lambda rows, _: rows)
    monkeypatch.setattr(evaluate, "role_full_terms", lambda *args: [row((3, 7, 11, 19))])
    monkeypatch.setattr(evaluate, "score_full_terms", lambda rows, _: rows)

    methods = evaluate._method_rows(object(), object(), object(), object(),
                                    {"ephemerides": {}, "receiver_ecef": np.zeros(3)},
                                    np.zeros(10), {"A2": 1., "A5": 1., "Full": 1.},
                                    (("transition", 0., 1.2),),
                                    {"code_error_chips", "pll_phase_error_cycles",
                                     "carrier_doppler_hz", "code_frequency_offset_chips_s"})
    assert methods["A1"][0]["prns"] == [3, 7, 11, 19, 23]
    assert methods["Full"][0]["prns"] == [3, 7, 11, 19]
    scientific = [{**item, "scenario": "DS3", "method": method,
                   "label": True, "phase_start_s": 0., "phase_end_s": 1.2}
                  for method in ("Full", "A1") for item in methods[method]]
    assert exact_contrast_support(scientific, ("Full", "A1")) == []


def test_balanced_pauc_makes_ds7_ds8_one_cell_and_excludes_replay():
    from gnss_doppler_lab.gcspo_statistics import primary_pauc_rows, scenario_phase_balanced_pauc

    rows = []
    for scenario in ("DS7", "DS8"):
        rows += [_row(scenario, "pre_onset_replay", "Full", 100., 99., label=False,
                      phase_start=0., phase_end=110.)]
        rows += [_row(scenario, "transition", "Full", 120., 4., label=True)]
    rows += [_row("DS3", "pre_onset", "Full", 100., 0., label=False,
                  phase_start=0., phase_end=118.9)]
    selected = primary_pauc_rows(rows, method="Full")
    assert len(selected) == 3
    family = [row["logical_cell"] for row in selected if row["scenario"] in {"DS7", "DS8"}]
    assert family == ["positive:DS7_DS8:transition"] * 2
    assert scenario_phase_balanced_pauc(selected) == pytest.approx(1.)


def test_ds7_ds8_hierarchical_weights_balance_members_not_row_counts():
    from gnss_doppler_lab.gcspo_statistics import hierarchical_scenario_phase_weights

    rows = ([{"scenario": "DS7", "phase": "transition", "label": True}] * 2 +
            [{"scenario": "DS8", "phase": "transition", "label": True}] * 4 +
            [{"scenario": "DS3", "phase": "transition", "label": True}] +
            [{"scenario": "cleanStatic", "phase": "holdout", "label": False}] * 2)
    weights = hierarchical_scenario_phase_weights(rows)
    assert weights[:2] == pytest.approx([.125, .125])
    assert weights[2:6] == pytest.approx([.0625] * 4)
    assert weights[6] == pytest.approx(.5)
    assert weights[7:] == pytest.approx([.5, .5])
    labels = np.asarray([row["label"] for row in rows])
    assert weights[labels].sum() == pytest.approx(1.)
    assert weights[~labels].sum() == pytest.approx(1.)


def test_ds7_ds8_one_available_member_duplicate_rows_are_invariant_and_missing_member_fails():
    from gnss_doppler_lab.gcspo_statistics import hierarchical_scenario_phase_weights, scenario_phase_balanced_pauc

    def material(ds8_count):
        return ([{"scenario": "DS7", "phase": "transition", "label": True, "score": 3.}] +
                [{"scenario": "DS8", "phase": "transition", "label": True, "score": 1.}] * ds8_count +
                [{"scenario": "cleanStatic", "phase": "holdout", "label": False, "score": 2.}] * 2)
    assert scenario_phase_balanced_pauc(material(1)) == pytest.approx(scenario_phase_balanced_pauc(material(7)))
    only_ds7 = [{"scenario": "DS7", "phase": "transition", "label": True}] * 3 + [
                {"scenario": "cleanStatic", "phase": "holdout", "label": False}]
    weights = hierarchical_scenario_phase_weights(only_ds7)
    assert weights[:3] == pytest.approx([1 / 3] * 3)
    with pytest.raises(ValueError, match="member"):
        hierarchical_scenario_phase_weights([{"phase": "transition", "label": True}])


def test_bootstrap_is_real_2000_replicates_nextafter_and_family_coupled():
    from gnss_doppler_lab.gcspo_statistics import paired_block_bootstrap

    rows = []
    for method, offset in (("Full", 2.), ("A1", 0.)):
        for end in (10., 20., 30.):
            rows.append(_row("DS3", "pre_onset", method, end, offset - end / 100,
                             label=False, phase_start=0., phase_end=30.))
        for scenario in ("DS7", "DS8"):
            for end in (120., 130., 140.):
                rows.append(_row(scenario, "transition", method, end, offset + end / 100,
                                 label=True, phase_start=110., phase_end=140.))
    result = paired_block_bootstrap(rows, "Full", "A1", replicates=2000, seed=23)
    assert result["replicates"] == 2000 and len(result["values"]) == 2000
    assert result["lcb_95"] == np.sort(result["values"])[99]
    assert result["family_coupled"] is True
    assert all(draw["DS7_DS8:transition:DS7"] == draw["DS7_DS8:transition:DS8"]
               for draw in result["first_replicate_family_draws"])


def test_relation_destructions_are_executed_and_preserve_frozen_marginals():
    from gnss_doppler_lab.gcspo_statistics import execute_relation_destructions

    residual = np.arange(4 * 80 * 3, dtype=float).reshape(4, 80, 3)
    x = 1 / np.sqrt(3)
    los = np.asarray([[x, x, x], [x, -x, -x], [-x, x, -x], [-x, -x, x]])
    result = execute_relation_destructions(residual, los, np.asarray([3, 7, 11, 19]),
                                           scenario="DS3", phase="transition", segment_id="s0")
    assert not np.any(result["los_permutation"] == np.arange(4))
    assert result["preservation"]["LOS_SHUFFLE"] is True
    assert result["preservation"]["PER_PRN_TEMPORAL_SHIFT"] is True
    assert not np.array_equal(result["shifted_residual"], residual)


def test_all_six_gates_are_computed_from_numeric_evidence():
    from gnss_doppler_lab.gcspo_statistics import compute_scientific_gates

    evidence = {
        "clean_holdout_fpr": .01, "external_pre_fpr": {"DS3": .04},
        "incremental_lcb": {"Full-A1": .1, "Full-A2": .01},
        "destruction": {name: {"lcb": .01, "median_relative_loss": .3}
                        for name in ("LOS_SHUFFLE", "PER_PRN_TEMPORAL_SHIFT")},
        "persistence": {"DS3": {"ratio": .5, "delay_s": 10.},
                        "DS7_DS8": {"ratio": .8, "delay_s": 2.}},
        "controls": [{"id": "COMMON_GAIN", "persistent_alarm_ratio": .1,
                      "max_consecutive_alarms": 9},
                     {"id": "CLOCK_DRIFT", "specificity_ratio": .25}],
        "shared": {"full_pauc": .7, "a5_pauc": .71, "full_median_edf": 3., "a5_median_edf": 4.},
    }
    gates = compute_scientific_gates(evidence)
    assert [gate["id"] for gate in gates] == ["G1_FALSE_ALARM", "G2_INCREMENTAL", "G3_GEOMETRY",
                                               "G4_PERSISTENCE", "G5_CONTROLS", "G6_SHARED"]
    assert all(gate["status"] == "PASS" for gate in gates)
    evidence["incremental_lcb"]["Full-A2"] = 0.
    assert compute_scientific_gates(evidence)[1]["status"] == "FAIL"


def test_delivered_evaluator_contains_no_placeholder_bootstrap_or_literal_gate_failures():
    text = (Path(__file__).parents[1] / "src/gnss_doppler_lab/gcspo_evaluate.py").read_text()
    assert '"lcb_95": "NA"' not in text
    assert "relation-destruction improvement gate not established" not in text
    assert '"id": "G2_INCREMENTAL", "status": "FAIL"' not in text
    assert "paired_block_bootstrap" in text and "compute_scientific_gates" in text
