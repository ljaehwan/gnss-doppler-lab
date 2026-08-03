from __future__ import annotations

import inspect
import numpy as np
import pytest
import torch

from gnss_doppler_lab.amcf_shape_only import (
    SIDE_INDICES, TAP_NAMES, CleanTensorSplit, FeatureWindow, PromptGate,
    ShapeOnlyModel, all9_loo_score, alarm_flags, assert_target_iqr,
    build_history_examples, calibration_loo_evidence, common_timestamp_pairs,
    conformal_evidence, fit_clean_model, fit_prompt_gate, fit_robust_scaler,
    higher_thresholds, make_fixed_validation_bank, normalize_by_prompt,
    paired_block_bootstrap, phase_masks, primary_status, robust_features,
    schema_collapse_audit, sustained_three_delay, transform_robust,
)


def raw(n=80, seed=9):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 9)) + 1j * rng.normal(size=(n, 9))
    z[:, 4] += 8 + 3j
    return z


def split(role="train", recording="cleanStatic", n=10, d=2, seed=4):
    rng = np.random.default_rng(seed)
    return CleanTensorSplit(
        current=rng.normal(size=(n, 8, d)).astype("f4"),
        history=rng.normal(size=(n, 12, 8, d)).astype("f4"),
        role=role,
        recording_id=recording,
    )


def test_prompt_normalization_invariance_and_prompt_never_feature():
    z = raw()
    gate = fit_prompt_gate(z[:, 4], role=np.full(len(z), "train"), recording=np.full(len(z), "cleanStatic"))
    a, valid = normalize_by_prompt(z, gate)
    phase = np.exp(1j * 0.731)
    b, vb = normalize_by_prompt(z * phase, gate)
    c, vc = normalize_by_prompt(-z, gate)
    d, vd = normalize_by_prompt(z * 7.0, PromptGate(gate.minimum * 7.0, gate.eps * 49.0))
    assert a.shape == (len(z), 8) and 4 not in SIDE_INDICES and TAP_NAMES[4] == "P"
    np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(a, c, rtol=0, atol=0)
    np.testing.assert_allclose(a, d, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(valid, vb); np.testing.assert_array_equal(valid, vc); np.testing.assert_array_equal(valid, vd)


def test_prompt_gate_clean_train_only_and_low_prompt_rejected():
    z = raw(12); z[3, 4] = 0
    with pytest.raises(ValueError, match="cleanStatic train"):
        fit_prompt_gate(z[:, 4], role=np.array(["train"] * 12), recording=np.array(["DS1"] * 12))
    gate = fit_prompt_gate(z[:, 4], role=np.array(["train"] * 12), recording=np.array(["cleanStatic"] * 12))
    out, valid = normalize_by_prompt(z, gate)
    assert not valid[3] and np.isfinite(out[valid]).all() and np.isnan(out[~valid]).all()


def test_literal_features_no_padding_duplicates_dimensions_and_variance():
    z, valid = normalize_by_prompt(raw(120), PromptGate(0.01))
    fc = robust_features(z[valid], "complex")
    fm = robust_features(z[valid], "magnitude")
    assert fc.shape == (8, 4) and fm.shape == (8, 2)
    x = z[valid, 0]
    assert fc[0, 0] == pytest.approx(np.median(x.real))
    assert fc[0, 2] == pytest.approx(np.median(np.abs(x.real - np.median(x.real))))
    assert fm[0, 1] == pytest.approx(np.median(np.abs(np.abs(x) - np.median(np.abs(x)))))
    assert np.all(np.var(fc, axis=0) > 0) and np.all(np.var(fm, axis=0) > 0)
    assert len({tuple(fc[:, j]) for j in range(4)}) == 4
    with pytest.raises(ValueError, match="IQR"):
        assert_target_iqr(np.zeros((20, 8, 4)), tolerance=1e-8)


def test_forbidden_field_model_api_and_metadata_poison_has_no_path():
    forbidden = {"cn0", "log_prompt", "prompt_magnitude", "valid_fraction", "valid_count", "rejected_count", "raw_count", "recording_id", "scenario_id", "prn"}
    params = set(inspect.signature(ShapeOnlyModel.forward).parameters)
    assert not forbidden.intersection(params)
    model = ShapeOnlyModel(4, hidden=8).eval()
    h = torch.zeros(2, 12, 8, 4); cur = torch.randn(2, 8, 4); mask = torch.ones(2, 8, dtype=torch.bool); mask[:, 2] = False
    with torch.no_grad(): a = model(h, cur, mask)
    poison = {k: object() for k in forbidden}
    with pytest.raises(TypeError): model(h, cur, mask, **poison)
    with torch.no_grad(): b = model(h, cur, mask)
    torch.testing.assert_close(a[0], b[0]); torch.testing.assert_close(a[1], b[1])


def fw(t, role="train", prn="G01", segment="s", channel="c"):
    return FeatureWindow("r", segment, channel, prn, role, t - 1.0, t, np.full((8, 2), t, dtype=np.float32))


def test_causal_previous_12_exact_cadence_gap_reset_no_padding_split_boundary():
    rows = [fw(i * .5) for i in range(15)]
    ex = build_history_examples(rows)
    assert len(ex) == 3 and ex[0].current.source_end == 6.0
    assert [x.source_end for x in ex[0].history] == list(np.arange(.0, 6.0, .5))
    assert all(x.source_end < ex[0].current.source_end for x in ex[0].history)
    gapped = rows[:13] + [fw(7.0), fw(7.5)]
    assert [e.current.source_end for e in build_history_examples(gapped)] == [6.0]
    changed = rows[:13] + [fw(6.5, role="validation"), fw(7.0, role="validation")]
    assert [e.current.source_end for e in build_history_examples(changed)] == [6.0]
    other = rows[:12] + [fw(6.0, prn="G02"), fw(6.5, segment="x")]
    assert build_history_examples(other) == []


def test_model_tap_order_and_all9_order_and_prn_permutation_invariant():
    torch.manual_seed(2); model = ShapeOnlyModel(4, hidden=8).eval()
    rng = np.random.default_rng(3); h = rng.normal(size=(12, 8, 4)).astype("f4"); cur = rng.normal(size=(8, 4)).astype("f4")
    score = all9_loo_score(model, h, cur)
    perm = np.array([7, 2, 4, 0, 6, 1, 5, 3])
    score2 = all9_loo_score(model, h[:, perm], cur[perm])
    assert score == pytest.approx(score2, rel=1e-6)
    epoch_a = np.median([score, score + 2, score + 1])
    epoch_b = np.median([score + 1, score, score + 2])
    assert epoch_a == epoch_b


def test_representation_fair_parameter_count_and_no_unused_parameters():
    for hidden in (8, 32):
        c = ShapeOnlyModel(4, hidden=hidden); m = ShapeOnlyModel(2, hidden=hidden)
        nc = sum(p.numel() for p in c.parameters()); nm = sum(p.numel() for p in m.parameters())
        assert abs(nc - nm) / max(nc, nm) <= .05
    model = ShapeOnlyModel(4, hidden=8); h = torch.randn(3, 12, 8, 4); cur = torch.randn(3, 8, 4); mask = torch.ones(3, 8, dtype=torch.bool); mask[:, 0] = False
    loc, scale = model(h, cur, mask); loss = (loc.square() + scale).mean(); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_clean_only_scaler_immutable_and_attack_rejected():
    x = np.arange(40 * 8 * 2, dtype=float).reshape(40, 8, 2)
    scaler = fit_robust_scaler(x, role=np.array(["train"] * 40), recording=np.array(["cleanStatic"] * 40))
    med = scaler.median.copy(); iqr = scaler.iqr.copy(); a = transform_robust(x, scaler)
    poison = x.copy(); poison[-1] = 1e99; b = transform_robust(poison, scaler)
    np.testing.assert_array_equal(scaler.median, med); np.testing.assert_array_equal(scaler.iqr, iqr); np.testing.assert_allclose(a[:-1], b[:-1])
    with pytest.raises(ValueError, match="cleanStatic train"):
        fit_robust_scaler(x, role=np.array(["train"] * 40), recording=np.array(["DS1"] * 40))


def test_fixed_validation_bank_hash_minibatches_and_nonconverged_primary_incomplete():
    bank, digest = make_fixed_validation_bank(7, seed=101)
    bank2, digest2 = make_fixed_validation_bank(7, seed=101)
    np.testing.assert_array_equal(bank, bank2); assert digest == digest2 == "c937a48c05ac3edf0b7b265c60fd92836c57d582cc3f317c132365d64de58e08" and len(bank) == 56
    tr = split(n=12, d=2); va = split(role="validation", n=7, d=2, seed=5)
    model, optimizer, audit = fit_clean_model(tr, va, seed=101, hidden=8, batch_size=5, max_epochs=2, patience=20)
    assert audit["validation_bank_hash"] == digest
    assert audit["optimizer_updates"] == 6 and audit["optimizer_updates"] > 1
    assert audit["finite"] and audit["best_checkpoint_restored"] and audit["best_optimizer_restored"]
    assert audit["converged"] is False and audit["stop_reason"] == "cap_nonconverged"
    assert optimizer.state_dict()["state"] and primary_status([audit]) == "INCOMPLETE: nonconverged seed"
    assert primary_status([{"seed": 101, "converged": True}]) == "INCOMPLETE: missing required seed"
    with pytest.raises(ValueError, match="cleanStatic"):
        fit_clean_model(split(recording="DS1"), va, seed=101, hidden=8, max_epochs=1)
    with pytest.raises(ValueError, match="seed"):
        fit_clean_model(tr, va, seed=999, hidden=8, max_epochs=1)


def test_conformal_hand_fixture_higher_strict_and_attack_rejected():
    cal = np.array([1., 2., 3., 4.])
    p, e = conformal_evidence(cal, np.array([2.5, 4.0, 5.0]))
    np.testing.assert_allclose(p, [3/5, 2/5, 1/5]); np.testing.assert_allclose(e, -np.log(p))
    loo = calibration_loo_evidence(cal)
    expected_p = np.array([1., .75, .5, .25])
    np.testing.assert_allclose(loo, -np.log(expected_p))
    th = higher_thresholds(np.vstack([loo, loo + .1, loo + .2]), role=np.array(["calibration"] * 4), recording=np.array(["cleanStatic"] * 4))
    assert th["comparison"] == "strict_greater" and th["q995"] >= th["q99"]
    assert alarm_flags([th["q99"], np.nextafter(th["q99"], np.inf)], th["q99"]).tolist() == [False, True]
    with pytest.raises(ValueError, match="cleanStatic calibration"):
        higher_thresholds(np.vstack([loo]), role=np.array(["calibration"] * 4), recording=np.array(["DS1"] * 4))


def test_common_timestamp_join_paired_bootstrap_and_sign_complex_minus_comparator():
    c = {0.0: 1., .5: 4., 1.0: 6., 1.5: 8.}; m = {.5: 1., 1.0: 2., 2.0: 9.}
    t, a, b = common_timestamp_pairs(c, m)
    np.testing.assert_array_equal(t, [.5, 1.]); np.testing.assert_array_equal(a, [4., 6.]); np.testing.assert_array_equal(b, [1., 2.])
    out = paired_block_bootstrap(t, a, b, lambda x: float(np.mean(x)), reps=20, block_s=10., seed=2)
    assert out["delta_definition"] == "Complex-comparator" and out["estimate"] == pytest.approx(3.5) and out["reps"] == 20


def test_actual_source_phase_wholly_post_delay_and_already_alarming():
    start = np.array([30., 79., 99.5, 100., 100.5, 140.]); end = np.array([31., 80., 100.5, 101., 101.5, 141.])
    masks = phase_masks(start, end, onset_s=100.)
    assert masks["stable_pre"].tolist() == [True, True, False, False, False, False]
    assert masks["post"].tolist() == [False, False, False, True, True, True]
    assert masks["persistent"].tolist() == [False, False, False, False, False, True]
    times = np.array([79., 79.5, 80., 100.5, 101., 101.5]); post = np.array([False, False, False, True, True, True])
    assert sustained_three_delay(times, [False, False, False, True, True, True], post, stable_pre=[False, False, False, False, False, False], onset_s=100.) == .5
    assert sustained_three_delay(times, [True, True, True, True, True, True], post, stable_pre=[True, True, True, False, False, False], onset_s=100.) == "N/A: already alarming in stable-pre"


def test_alarm_recompute_and_schema_collapse_guards():
    scores = np.array([1., 2., 3.]); flags = alarm_flags(scores, 2.)
    np.testing.assert_array_equal(flags, scores > 2.)
    good = schema_collapse_audit(
        complex_features=np.arange(5 * 8 * 4, dtype=float).reshape(5, 8, 4),
        magnitude_features=np.arange(5 * 8 * 2, dtype=float).reshape(5, 8, 2),
        complex_schema=("median_real", "median_imag", "mad_real", "mad_imag"),
        magnitude_schema=("median_abs", "mad_abs"), tracked_prn_counts=[2, 3, 4], stable_pre_alarms=[False, True],
    )
    assert good["pass"] and good["no_cn0_branch"] and good["tracked_prn_median"] > 1
    bad = schema_collapse_audit(np.ones((2, 8, 4)), np.ones((2, 8, 2)), ("x",), ("x",), [1], [True, True])
    assert not bad["pass"]


def test_source_window_open_left_closed_right_and_wholly_same_role():
    from gnss_doppler_lab.amcf_shape_only import make_feature_window
    z = raw(7); times = np.array([4., 4.01, 4.5, 4.99, 5., 5.01, 5.5])
    roles = np.array(["train", "train", "train", "train", "train", "validation", "validation"])
    gate = PromptGate(.01)
    w = make_feature_window(z, times, roles, source_end=5., recording_id="cleanStatic", segment_id="s", channel_id="c", prn="G01", gate=gate, representation="complex")
    assert w.source_start == 4. and w.source_end == 5. and w.role == "train" and w.features.shape == (8, 4)
    with pytest.raises(ValueError, match="wholly"):
        make_feature_window(z, times, roles, source_end=5.5, recording_id="cleanStatic", segment_id="s", channel_id="c", prn="G01", gate=gate, representation="complex")


def test_epl_auxiliary_labels_only_and_epoch_prn_median():
    from gnss_doppler_lab.amcf_shape_only import epl_loo_diagnostic, epoch_prn_score
    torch.manual_seed(12); model = ShapeOnlyModel(2, hidden=8).eval(); rng = np.random.default_rng(8)
    h = rng.normal(size=(12, 8, 2)).astype("f4"); cur = rng.normal(size=(8, 2)).astype("f4")
    out = epl_loo_diagnostic(model, h, cur)
    assert set(out) == {"E", "L", "mean"} and np.isfinite(list(out.values())).all()
    assert epoch_prn_score({"G03": 7., "G01": 1., "G02": 4.}) == 4.
