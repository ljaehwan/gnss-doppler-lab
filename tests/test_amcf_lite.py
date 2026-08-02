from __future__ import annotations

import json
import math
from collections.abc import Mapping

import numpy as np
import pytest

from gnss_doppler_lab.amcf_lite import (
    TAP_COORDS,
    MaskedSetModel,
    PromptGate,
    aggregate_epoch_scores,
    assign_clean_role,
    binary_auc,
    causal_decision_rows,
    calibrate_normal_thresholds,
    first_sustained_alarm_delay,
    fit_prompt_gate,
    normalize_prompt,
    random_query_order,
    represent_values,
    robust_top2,
    score_query_path,
    select_next_uncertain,
    student_t_nll,
    tapwise_complex_qa,
)


def _iq(n: int = 8) -> np.ndarray:
    rng = np.random.default_rng(4)
    z = rng.normal(size=(n, 9)) + 1j * rng.normal(size=(n, 9))
    z[:, 4] += 4 + 2j
    return np.stack((z.real, z.imag), axis=-1)


def _fit_rows(n: int = 40) -> np.ndarray:
    rng = np.random.default_rng(8)
    x = np.empty((n, 9, 2))
    for i in range(n):
        a, b = rng.normal(size=2)
        x[i, :, 0] = 0.3 * TAP_COORDS + a + rng.normal(scale=.05, size=9)
        x[i, :, 1] = -0.2 * TAP_COORDS + b + rng.normal(scale=.05, size=9)
    return x


def test_global_phase_invariance():
    iq = _iq()
    gate = PromptGate(min_prompt_magnitude=.1)
    a, va = normalize_prompt(iq, gate)
    z = iq[..., 0] + 1j * iq[..., 1]
    z *= np.exp(1j * 1.234)
    b, vb = normalize_prompt(np.stack((z.real, z.imag), -1), gate)
    np.testing.assert_allclose(a, b, atol=2e-7, rtol=2e-7)
    np.testing.assert_array_equal(va, vb)


def test_navigation_sign_invariance():
    iq = _iq()
    gate = PromptGate(.1)
    a, va = normalize_prompt(iq, gate)
    b, vb = normalize_prompt(-iq, gate)
    np.testing.assert_allclose(a, b, atol=0, rtol=0)
    np.testing.assert_array_equal(va, vb)


def test_low_prompt_quality_gate_is_train_only_and_numerically_stable():
    iq = _iq(20)
    iq[:2, 4] = 0
    gate = fit_prompt_gate(iq, np.arange(20.0), train_end_s=10, quantile=.2)
    assert gate.fit_rows == 10 and gate.fit_interval == [0.0, 10.0]
    normalized, valid = normalize_prompt(iq, gate)
    assert not valid[:2].any()
    assert np.isfinite(normalized[valid]).all()
    assert np.isnan(normalized[~valid]).all()


def test_representation_shapes_and_unit_phase():
    z, valid = normalize_prompt(_iq(), PromptGate(.1))
    assert valid.all()
    assert represent_values(z, "complex").shape == (8, 9, 2)
    assert represent_values(z, "magnitude").shape == (8, 9, 1)
    phase = represent_values(z, "phase")
    np.testing.assert_allclose(np.linalg.norm(phase, axis=-1), 1, atol=1e-12)


def test_masked_model_has_no_hidden_value_leakage_and_is_reproducible():
    x = _fit_rows()
    a = MaskedSetModel(2, hidden=12, seed=19, epochs=4).fit(x)
    b = MaskedSetModel(2, hidden=12, seed=19, epochs=4).fit(x)
    observed = {3: x[0, 3], 4: x[0, 4], 5: x[0, 5]}
    pa = a.predict(observed, 1)
    pb = b.predict(observed, 1)
    np.testing.assert_array_equal(pa.location, pb.location)
    np.testing.assert_array_equal(pa.scale, pb.scale)
    mutated = x[0].copy(); mutated[[0, 1, 2, 6, 7, 8]] = 1e9
    same_observed = {k: mutated[k] for k in (3, 4, 5)}
    pm = a.predict(same_observed, 1)
    np.testing.assert_array_equal(pa.location, pm.location)
    np.testing.assert_array_equal(pa.scale, pm.scale)


class _ObservedOnly(Mapping):
    def __init__(self, source, allowed): self.source, self.allowed = source, set(allowed)
    def __getitem__(self, key):
        if key not in self.allowed: raise AssertionError("selector read hidden value")
        return self.source[key]
    def __iter__(self): return iter(sorted(self.allowed))
    def __len__(self): return len(self.allowed)


def test_selector_never_reads_hidden_values_and_matches_manual_uncertainty():
    x = _fit_rows()
    model = MaskedSetModel(2, hidden=10, seed=3, epochs=3).fit(x)
    observed = _ObservedOnly(x[0], {3, 4, 5})
    candidates = [0, 1, 2, 6, 7, 8]
    got = select_next_uncertain(model, observed, candidates)
    manual = max(candidates, key=lambda tap: (float(np.mean(model.predict(observed, tap).scale)), -tap))
    assert got == manual


def test_causal_first_grid_latest_stable_tie_and_no_future():
    iq = _iq(5)
    out = causal_decision_rows(
        iq, time_s=np.array([.49, .5001, .62, .62, 1.01]),
        prn=np.array([1, 2, 1, 1, 1]), channel=np.array([0, 1, 0, 0, 0]),
        segment_index=np.array([0, 0, 0, 0, 1]), sample_count=np.array([1, 2, 3, 4, 5]),
        recording_id="r",
    )
    at_half = out.decision_time_s == .5
    assert set(out.prn[at_half]) == {1}
    assert np.all(out.time_s <= out.decision_time_s + 1e-9)
    at_one_g01 = (out.decision_time_s == 1.0) & (out.prn == 1)
    assert out.sample_count[at_one_g01].item() == 4
    assert len(set(zip(out.recording_id, out.decision_time_s, out.prn))) == len(out)


def test_prn_and_input_order_do_not_change_epoch_scores_with_variable_n():
    rows = [
        {"recording_id": "r", "decision_time_s": .5, "prn": 1, "score": 1.},
        {"recording_id": "r", "decision_time_s": .5, "prn": 2, "score": 100.},
        {"recording_id": "r", "decision_time_s": .5, "prn": 3, "score": 3.},
        {"recording_id": "r", "decision_time_s": 1., "prn": 1, "score": 7.},
    ]
    a = aggregate_epoch_scores(rows)
    renamed = [{**r, "prn": 10-r["prn"]} for r in reversed(rows)]
    b = aggregate_epoch_scores(renamed)
    assert a == b
    assert [r["tracked_prn_count"] for r in a] == [3, 1]
    assert a[0]["score"] == 3.0  # robust median, not PRN order or max


def test_normal_only_calibration_and_higher_quantiles():
    scores = np.arange(100., dtype=float)
    roles = np.array(["calibration"] * 100, object)
    scenarios = np.array(["cleanStatic"] * 100, object)
    got = calibrate_normal_thresholds(scores, roles, scenarios)
    assert got["q99"] == 99 and got["q995"] == 99 and got["fit_count"] == 100
    bad = scenarios.copy(); bad[-1] = "DS1"
    with pytest.raises(ValueError, match="normal-only"):
        calibrate_normal_thresholds(scores, roles, bad)
    roles[-1] = "test"
    with pytest.raises(ValueError, match="calibration"):
        calibrate_normal_thresholds(scores, roles, scenarios)


def test_student_t_manual_nll_top2_and_query_scoring():
    y = np.array([2., -1.]); mu = np.array([1., -1.]); scale = np.array([2., .5]); df = 4.
    expected = (math.lgamma((df + 1) / 2) - math.lgamma(df / 2))
    expected = -expected + .5 * math.log(df * math.pi) + np.log(scale) + (df + 1) / 2 * np.log1p(((y-mu)/scale)**2/df)
    np.testing.assert_allclose(student_t_nll(y, mu, scale), expected)
    assert robust_top2([1., 9., 3., 7.]) == 8.
    model = MaskedSetModel(2, hidden=8, seed=5, epochs=2).fit(_fit_rows())
    values = _fit_rows(2)[0]
    scored = score_query_path(model, values, [3, 4, 5, 2, 6])
    assert scored["query_order"] == [3, 4, 5, 2, 6]
    assert len(scored["queried_nll"]) == 5
    assert scored["score"] == robust_top2(scored["queried_nll"])


def test_random_selection_uses_no_values_and_is_seeded():
    class Forbidden:
        def __getitem__(self, key): raise AssertionError("random selector read values")
    assert random_query_order(5, seed=12, values=Forbidden()) == random_query_order(5, seed=12, values=Forbidden())
    assert random_query_order(5, seed=12)[:3] == [3, 4, 5]
    assert len(set(random_query_order(7, seed=4))) == 7


def test_exact_chronological_roles():
    assert assign_clean_role(0.) == "train"
    assert assign_clean_role(239.999) == "train"
    assert assign_clean_role(240.) is None
    assert assign_clean_role(250.) == "validation"
    assert assign_clean_role(330.) is None
    assert assign_clean_role(340.) == "calibration"
    assert assign_clean_role(409.999) == "calibration"
    assert assign_clean_role(410.) is None
    assert assign_clean_role(420.) == "clean_test"


def test_epoch_selection_histogram_is_permutation_invariant():
    rows = [
        {"recording_id": "r", "decision_time_s": .5, "prn": 1, "score": 1., "query_order": [3, 4, 5, 2, 6]},
        {"recording_id": "r", "decision_time_s": .5, "prn": 2, "score": 3., "query_order": [3, 4, 5, 1, 7]},
    ]
    a = aggregate_epoch_scores(rows)
    b = aggregate_epoch_scores(list(reversed(rows)))
    assert a == b
    assert json.loads(a[0]["selected_tap_histogram_json"]) == {"1": 1, "2": 1, "3": 2, "4": 2, "5": 2, "6": 1, "7": 1}


def test_auc_and_sustained_alarm_delay_known_values():
    roc, pr = binary_auc(np.array([0., 1.]), np.array([2., 3.]))
    assert roc == pytest.approx(1.0)
    assert pr == pytest.approx(1.0)
    times = np.arange(9.5, 13.0, .5)
    alarms = np.array([True, False, True, True, True, False, True])
    assert first_sustained_alarm_delay(times, alarms, onset_s=10.0, run_length=3) == pytest.approx(0.5)
    assert first_sustained_alarm_delay(times, np.zeros_like(alarms), onset_s=10.0, run_length=3) is None


def test_tapwise_complex_qa_has_fixed_tap_order_and_phase_curvature():
    z, valid = normalize_prompt(_iq(16), PromptGate(.1))
    assert valid.all()
    qa = tapwise_complex_qa(z)
    assert list(qa) == ["E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4"]
    assert set(qa["P"]) == {"magnitude", "phase_rad", "phase_curvature_rad"}
    assert qa["P"]["magnitude"]["count"] == 16
    assert np.isfinite(qa["P"]["phase_curvature_rad"]["median"])
