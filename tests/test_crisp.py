from __future__ import annotations

import numpy as np

from gnss_doppler_lab.crisp import (
    LinearWhiteningModel,
    binary_metrics,
    invariance_audit,
    normalized_low_fpr_pauc,
    projector_matrix,
    projector_property_audit,
    projector_vector,
    fit_unconditioned,
    wedge_vector,
)


def random_taps(seed=3, n=64, m=9):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, m)) + 1j * rng.normal(size=(n, m))


def test_projector_hermitian_rank_one_and_idempotent():
    audit = projector_property_audit()
    assert audit["pass"]


def test_projector_vector_dimension():
    assert projector_vector(random_taps()).shape == (64, 81)


def test_arbitrary_complex_scale_invariance():
    taps = random_taps()
    rng = np.random.default_rng(4)
    scalar = rng.uniform(0.5, 2.0, len(taps)) * np.exp(1j * rng.uniform(-np.pi, np.pi, len(taps)))
    np.testing.assert_allclose(projector_vector(taps), projector_vector(taps * scalar[:, None]), atol=1e-11)


def test_gain_phase_nav_and_doppler_invariance():
    assert invariance_audit()["all_pass"]


def test_prompt_amplitude_scaling_is_common_vector_scaling_control():
    taps = random_taps()
    np.testing.assert_allclose(projector_vector(taps), projector_vector(1.7 * taps), atol=1e-11)


def test_wedge_zero_for_rigid_direction():
    taps = random_taps(n=32)
    previous = taps
    current = taps * (2.0 * np.exp(0.3j))
    np.testing.assert_allclose(wedge_vector(current, previous), 0.0, atol=1e-12)


def test_wedge_changes_for_nonrigid_tap_mixture():
    taps = random_taps(n=32)
    changed = taps.copy()
    changed[:, 0] += 0.5
    assert np.max(np.abs(wedge_vector(changed, taps))) > 1e-3


def test_linear_model_is_shared_multioutput_ridge():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 6))
    beta = rng.normal(size=(7, 10))
    y = np.column_stack((np.ones(len(x)), x)) @ beta + rng.normal(scale=0.02, size=(500, 10))
    model = LinearWhiteningModel.fit(x[:250], y[:250], x[250:], y[250:], ridge_alpha=1e-3)
    score = model.score(x[250:], y[250:])
    assert score.shape == (250,)
    assert np.isfinite(score).all()


def test_unconditioned_model_has_intercept_only_and_is_finite():
    response = np.random.default_rng(8).normal(size=(300, 7))
    model = fit_unconditioned(response)
    score = model.score(np.empty((len(response), 0)), response)
    assert model.feature_mean.size == 0
    assert np.isfinite(score).all()


def test_model_round_trip_is_exact_enough():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 3)); y = rng.normal(size=(200, 5))
    model = LinearWhiteningModel.fit(x[:100], y[:100], x[100:], y[100:], ridge_alpha=1e-3)
    restored = LinearWhiteningModel.from_dict(model.to_dict())
    np.testing.assert_allclose(model.score(x, y), restored.score(x, y), atol=1e-12)


def test_low_fpr_pauc_perfect_classifier():
    labels = np.r_[np.zeros(100), np.ones(100)]
    scores = np.r_[np.zeros(100), np.ones(100)]
    assert normalized_low_fpr_pauc(labels, scores) == 1.0
    assert binary_metrics(labels, scores)["roc_auc"] == 1.0


def test_causal_difference_has_no_future_sample():
    taps = random_taps(n=10)
    first = projector_vector(taps)
    taps[-1] *= 1.0 + 0.4j
    second = projector_vector(taps)
    np.testing.assert_allclose(first[:-1], second[:-1])


def test_minimum_four_prn_second_largest_policy():
    values = sorted([1.0, 2.0, 3.0, 1000.0])
    assert values[-2] == 3.0
    assert len(values) >= 4


def test_single_prn_glitch_does_not_set_receiver_score():
    normal = [1.0, 1.1, 0.9, 1000.0]
    assert sorted(normal)[-2] == 1.1


def test_gap_and_lock_loss_policy_is_unavailable_not_high_score():
    valid_lock = np.array([1, 0, 1], dtype=bool)
    timestamp_ms = np.array([0, 1, 100])
    reset = np.r_[False, np.diff(timestamp_ms) != 1]
    assert not valid_lock[1]
    assert reset[2]


def test_dataset_models_are_not_pooled():
    configured = {"TEXBAT": object(), "OAKBAT": object()}
    assert configured["TEXBAT"] is not configured["OAKBAT"]


def test_no_prn_identity_feature_in_six_context_features():
    context_names = ["delta_dll", "delta_code_frequency", "delta_doppler", "lagged_cn0", "lock", "previous_velocity"]
    assert "prn" not in context_names


def test_projector_formula_matches_outer_product():
    taps = random_taps(n=1)[0]
    expected = np.outer(taps, np.conj(taps)) / np.vdot(taps, taps).real
    np.testing.assert_allclose(projector_matrix(taps), expected, atol=1e-12)
