import numpy as np
import pytest

from gnss_doppler_lab.gcmr_peak_innovation import (
    ConditionalInnovationWhitener,
    EventDiagnostics,
    NormalOnlyCalibrator,
    PairRelationModel,
    SharedLocalPredictor,
    aggregate_event_score,
    common_drive_statistics,
    geometry_features,
    pair_anomaly_score,
    relation_destruction,
    safe_prompt_normalize,
)


def _normal_innovations(seed=7, samples=40, n=5):
    return np.random.default_rng(seed).normal(size=(samples, n, 3))


def test_safe_prompt_normalize_masks_zero_and_nonfinite():
    epl = np.array([[2., 4., 6.], [1., 0., 3.], [np.nan, 2., 3.], [1., np.inf, 3.]])
    normalized, valid = safe_prompt_normalize(epl, eps=1e-9, min_prompt=0.1)
    assert valid.tolist() == [True, False, False, False]
    assert np.allclose(normalized[0], [0.5, 1., 1.5])
    assert np.all(np.isfinite(normalized))
    assert np.all(normalized[~valid] == 0.)


def test_shared_predictor_callback_is_prn_local_and_variable_n():
    seen = []
    def callback(history):
        seen.append(history.shape)
        return history[-1] + np.array([1., 2., 3.])
    predictor = SharedLocalPredictor(callback)
    history = np.zeros((4, 3, 3))
    result = predictor.predict(history)
    assert result.shape == (3, 3)
    assert seen == [(4, 3, 3)]
    assert np.allclose(result, [1., 2., 3.])
    assert predictor.predict(np.zeros((2, 7, 3))).shape == (7, 3)


def test_whitener_finite_3d_and_normal_only_context_fit():
    normal = _normal_innovations()
    context = np.linspace(25, 50, normal.shape[0] * normal.shape[1]).reshape(normal.shape[:2])
    whitener = ConditionalInnovationWhitener(min_bin_samples=8, regularization=1e-4).fit(normal, context=context)
    z = whitener.transform(normal[0], context=context[0])
    assert z.shape == (5, 3)
    assert np.all(np.isfinite(z))
    assert whitener.dimension == 3
    with pytest.raises(ValueError):
        ConditionalInnovationWhitener().fit(np.empty((0, 3)))
    with pytest.raises(ValueError):
        ConditionalInnovationWhitener().fit(np.array([[[np.nan, 0, 0]]]))


def test_geometry_and_pair_relation_are_symmetric():
    los = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    elevation = np.array([30., 50., 40.])
    assert np.allclose(geometry_features(los[0], elevation[0], los[1], elevation[1]),
                       geometry_features(los[1], elevation[1], los[0], elevation[0]))
    z = _normal_innovations(samples=20, n=3)
    model = PairRelationModel(ridge=1e-3).fit(z, los=los, elevation=elevation)
    matrix = model.expected_matrix(los, elevation)
    assert np.allclose(matrix, matrix.T)
    observed = z[0]
    assert pair_anomaly_score(observed, model, los, elevation) == pytest.approx(
        pair_anomaly_score(observed[[2, 0, 1]], model, los[[2, 0, 1]], elevation[[2, 0, 1]])
    )


def test_common_drive_permutation_variable_n_and_not_hard_alarm():
    z = np.array([[1., 0., 0.], [2., 0., 0.], [-3., 0., 0.]])
    first = common_drive_statistics(z, loading_threshold=0.99)
    second = common_drive_statistics(z[[2, 0, 1]], loading_threshold=0.99)
    assert first.n == 3 and first.loading_count < 4 and not first.at_least_four
    assert first.s_common == pytest.approx(second.s_common)
    assert first.n_eff == pytest.approx(second.n_eff)
    assert common_drive_statistics(z[:2]).n == 2


def test_relation_destruction_preserves_norms_and_changes_configuration():
    z = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.], [1., 1., 0.]])
    destroyed = relation_destruction(z, seed=19)
    assert np.allclose(np.linalg.norm(destroyed, axis=1), np.linalg.norm(z, axis=1))
    assert not np.allclose(destroyed @ destroyed.T, z @ z.T)
    assert np.allclose(destroyed, relation_destruction(z, seed=19))


def test_calibrator_quantiles_validation_and_full_excludes_binomial_tail():
    cal = NormalOnlyCalibrator().fit({"s_common": [0., 1., 2., 3.], "n_eff": [1., 2., 3., 4.],
                                      "s_pair": [1., 2., 3., 4.], "energy": [2., 4., 6., 8.],
                                      "scalar_rmse": [0., 1., 2., 3.]})
    assert cal.q99("energy") == pytest.approx(np.quantile([2., 4., 6., 8.], .99))
    assert cal.q995("energy") == pytest.approx(np.quantile([2., 4., 6., 8.], .995))
    assert cal.target_fpr_threshold("energy", .01) == cal.q99("energy")
    with pytest.raises(ValueError): NormalOnlyCalibrator().fit({"energy": []})
    with pytest.raises(ValueError): NormalOnlyCalibrator().fit({"energy": [np.nan]})
    d = EventDiagnostics(n=3, n_eff=2., loading_count=2, at_least_four=False,
                         s_common=2., s_pair=3., energy=5., scalar_rmse=1., binomial_tail=1e-100)
    full_a = aggregate_event_score(d, cal, "Full")
    full_b = aggregate_event_score(EventDiagnostics(**{**d.__dict__, "binomial_tail": 0.9}), cal, "Full")
    assert full_a == pytest.approx(full_b)
    assert aggregate_event_score(d, cal, "A1") != aggregate_event_score(
        EventDiagnostics(**{**d.__dict__, "binomial_tail": 0.9}), cal, "A1")
    assert np.isfinite(aggregate_event_score(d, cal, "A0"))
    assert np.isfinite(aggregate_event_score(d, cal, "A2"))
    assert np.isfinite(aggregate_event_score(d, cal, "A3"))
    assert np.isfinite(aggregate_event_score(d, cal, "A4"))
